"""Polite, cached web fetching for competitor pages.

    fetcher = Fetcher()
    page = fetcher.get("https://www.valleyvet.com/iverhart-plus-for-dogs/")

Rules it follows, so the project is a good citizen:
  - robots.txt is checked before every URL; disallowed URLs are never fetched
  - at least `request_delay_seconds` between requests to the same site
    (longer if robots.txt sets a Crawl-delay for us)
  - every page is cached in cache/pages/, so re-runs don't hit the site again
  - the User-Agent says what this is

Python's built-in urllib.robotparser ignores "*" wildcards, so a rule like
Chewy's "Disallow: *query=*" would be missed. RobotsRules below handles
wildcards and "$" the way Google documents them.
"""
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from src.config import ROOT, settings

USER_AGENT = "SupplierOnboardingAgent/0.1 (portfolio project; low volume; respects robots.txt)"
UA_TOKEN = "supplieronboardingagent"


class Blocked(Exception):
    """robots.txt disallows this URL."""


class FetchError(Exception):
    """The site didn't answer (timeout, connection refused, ...)."""


# ====================================================================== robots.txt

class RobotsRules:
    """Allow/Disallow rules for one site, for our user agent."""

    def __init__(self, text: str, agent: str = UA_TOKEN):
        groups: list[tuple[list[str], list[tuple[str, str]], float | None]] = []
        agents, rules, delay, in_rules = [], [], None, False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            field, value = (p.strip() for p in line.split(":", 1))
            field = field.lower()
            if field == "user-agent":
                if in_rules:                              # a new group starts
                    groups.append((agents, rules, delay))
                    agents, rules, delay, in_rules = [], [], None, False
                agents.append(value.lower())
            elif field in ("allow", "disallow"):
                in_rules = True
                if value:                                 # "Disallow:" with no path allows everything
                    rules.append((field, value))
            elif field == "crawl-delay":
                in_rules = True
                try:
                    delay = float(value)
                except ValueError:
                    pass
        if agents:
            groups.append((agents, rules, delay))

        # Use the group that names us; otherwise the "*" group.
        mine = [g for g in groups if any(a != "*" and a in agent for a in g[0])]
        chosen = mine or [g for g in groups if "*" in g[0]]
        self.rules = [r for g in chosen for r in g[1]]
        self.crawl_delay = next((g[2] for g in chosen if g[2] is not None), None)

    @staticmethod
    def _pattern(path: str) -> re.Pattern:
        regex = re.escape(path).replace(r"\*", ".*")
        if regex.endswith(r"\$"):
            regex = regex[:-2] + "$"
        return re.compile(regex)

    def allowed(self, url: str) -> bool:
        """Longest matching rule wins; on a tie, Allow wins."""
        parsed = urlparse(url)
        target = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        best: tuple[int, bool] | None = None
        for kind, path in self.rules:
            if self._pattern(path).match(target):
                candidate = (len(path), kind == "allow")
                if best is None or candidate > best:
                    best = candidate
        return True if best is None else best[1]


# ====================================================================== fetcher

GetFn = Callable[[str], tuple[int, str]]        # url -> (status code, text)


def http_get(url: str) -> tuple[int, str]:
    import httpx

    timeout = settings().get("enrichment", {}).get("timeout_seconds", 20)
    response = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US"})
    return response.status_code, response.text


@dataclass
class Page:
    url: str
    status: int
    html: str
    from_cache: bool


class Fetcher:
    def __init__(self, get_fn: GetFn = http_get, cache_dir: Path | None = None,
                 delay_seconds: float | None = None, sleep: Callable[[float], None] = time.sleep):
        self.get_fn = get_fn
        self.cache_dir = cache_dir or ROOT / "cache" / "pages"
        self.delay = (settings().get("enrichment", {}).get("request_delay_seconds", 3)
                      if delay_seconds is None else delay_seconds)
        self.sleep = sleep
        self._robots: dict[str, RobotsRules] = {}
        self._unreachable: dict[str, str] = {}         # site -> why we gave up on it
        self._last_request: dict[str, float] = {}

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")

    def robots(self, url: str) -> RobotsRules:
        host = urlparse(url)
        key = f"{host.scheme}://{host.netloc}"
        if key in self._unreachable:
            raise FetchError(self._unreachable[key])
        if key not in self._robots:
            try:
                status, text = self._polite_get(f"{key}/robots.txt", key)
            except FetchError as err:
                # Can't read the rules -> don't crawl. Remembered, so we don't retry for every URL.
                self._unreachable[key] = f"could not read {key}/robots.txt ({err}); not crawling this site"
                raise FetchError(self._unreachable[key]) from err
            # No robots.txt (404) means no rules; a server error means "don't crawl".
            self._robots[key] = RobotsRules(text if status == 200 else "" if status in (404, 410)
                                            else "User-agent: *\nDisallow: /")
        return self._robots[key]

    def _polite_get(self, url: str, site: str) -> tuple[int, str]:
        rules = self._robots.get(site)
        delay = max(self.delay, (rules.crawl_delay or 0) if rules else 0)
        wait = self._last_request.get(site, -1e9) + delay - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        try:
            return self.get_fn(url)
        except Exception as err:                        # timeouts, DNS, refused, TLS...
            raise FetchError(f"{type(err).__name__}") from err
        finally:
            self._last_request[site] = time.monotonic()

    def get(self, url: str) -> Page:
        """Fetch a page (or read it from the cache). Raises Blocked if robots.txt says no."""
        path = self._cache_path(url)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Page(url, data["status"], data["html"], from_cache=True)
        if not self.robots(url).allowed(url):
            raise Blocked(f"robots.txt disallows {url}")
        host = urlparse(url)
        status, html = self._polite_get(url, f"{host.scheme}://{host.netloc}")
        if status == 200:                               # only good pages are cached
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "status": status, "html": html,
                                        "fetched": time.strftime("%Y-%m-%d %H:%M")}), encoding="utf-8")
        return Page(url, status, html, from_cache=False)
