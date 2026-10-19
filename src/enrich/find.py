"""Find candidate product pages for one SKU on one competitor site.

Two ways, set per site in config/competitors.yaml (`find_by`):

  sitemap  Download the site's product sitemap once (cached for a week), then
           search its URLs locally with BM25, the same keyword search used in
           retrieve.py. One polite download instead of a search request per SKU,
           and it works when the site's search page needs JavaScript or is
           disallowed by robots.txt.
  search   Fetch the site's search page and take product links from it.
           Only works when results are in the HTML (no JavaScript needed).
"""
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin, urlparse

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from src.config import ROOT
from src.enrich.fetch import Blocked, Fetcher, FetchError
from src.retrieve import tokenize
from src.schemas import InputRow

SITEMAP_MAX_AGE_DAYS = 7


@dataclass
class Candidate:
    url: str
    score: float
    why: str


def query_for(row: InputRow) -> str:
    return " ".join(p for p in (row.brand, row.name) if p)


# ====================================================================== sitemap

def _locs(xml: str) -> list[str]:
    return [u.strip().replace("&amp;", "&") for u in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.S)]


def sitemap_urls(fetcher: Fetcher, site: dict, max_files: int = 20) -> list[str]:
    """All product URLs listed in a site's sitemap (follows sitemap indexes)."""
    cache = ROOT / "cache" / "sitemaps" / f"{site['name']}.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - saved["saved_at"] < SITEMAP_MAX_AGE_DAYS * 86400:
            return saved["urls"]

    product_filter = site.get("sitemap_filter", "product")
    todo, urls, seen, failed = [site["sitemap_url"]], [], set(), 0
    while todo and len(seen) < max_files:
        sitemap = todo.pop(0)
        if sitemap in seen:
            continue
        seen.add(sitemap)
        try:
            page = fetcher.get(sitemap)
        except FetchError:
            if sitemap == site["sitemap_url"]:
                raise                                     # the site itself is unreachable
            failed += 1
            continue
        if page.status != 200:
            continue
        if "<sitemapindex" in page.html:
            children = _locs(page.html)
            todo += [c for c in children if product_filter in c] or children
        else:
            urls += _locs(page.html)

    if urls and not failed:                               # only cache a complete download
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"saved_at": time.time(), "urls": urls}), encoding="utf-8")
    return urls


def slug_words(url: str) -> str:
    """'https://site.com/tiagard-12-5-liquid-concentrate/' -> 'tiagard 12 5 liquid concentrate'"""
    return " ".join(re.findall(r"[a-z0-9]+", urlparse(url).path.lower()))


def rank_urls(urls: list[str], row: InputRow, k: int = 5) -> list[Candidate]:
    """Rank sitemap URLs by the words in them.

    BM25 is scored over ALL urls (so a word like the brand counts as rare),
    then, if the brand appears in any URL, only those URLs are kept. Fuzzy
    similarity breaks ties, which also helps with small sitemaps where BM25's
    rarity weights are near zero.
    """
    if not urls:
        return []
    query = query_for(row)
    docs = [tokenize(slug_words(u)) or ["_"] for u in urls]
    bm25 = BM25Okapi(docs).get_scores(tokenize(query))
    brand_words = tokenize(row.brand or "")
    with_brand = [i for i, d in enumerate(docs) if brand_words and all(w in d for w in brand_words)]
    scored = []
    for i in with_brand or range(len(urls)):
        fuzzy = fuzz.token_set_ratio(query.lower(), slug_words(urls[i])) / 100
        if with_brand or bm25[i] > 0 or fuzzy >= 0.5:      # brand in the URL is signal enough
            scored.append((urls[i], round(float(bm25[i]) + fuzzy, 2)))
    scored.sort(key=lambda x: -x[1])
    return [Candidate(u, s, "sitemap URL words") for u, s in scored[:k]]


# ====================================================================== search page

def search_links(fetcher: Fetcher, site: dict, row: InputRow, k: int = 5) -> list[Candidate]:
    url = site["search_url"].format(query=quote_plus(query_for(row)))
    page = fetcher.get(url)                       # raises Blocked if robots.txt disallows search
    if page.status != 200:
        return []
    base = urlparse(site["base_url"]).netloc
    hint = site.get("product_path_hint") or "/"
    links = {}
    for href, text in re.findall(r'<a[^>]+href=["\']([^"\'#]+)["\'][^>]*>(.*?)</a>', page.html, re.S | re.I):
        full = urljoin(site["base_url"], href)
        parsed = urlparse(full)
        text = re.sub(r"<[^>]+>|\s+", " ", text).strip()
        if parsed.netloc == base and hint in parsed.path and parsed.path.strip("/") and text:
            links.setdefault(full.split("?")[0], text)
    scored = [(u, fuzz.token_set_ratio(query_for(row).lower(), f"{t} {slug_words(u)}".lower())) for u, t in links.items()]
    scored.sort(key=lambda x: -x[1])
    return [Candidate(u, s, "search results") for u, s in scored[:k] if s >= 50]


# ====================================================================== either

def find_candidates(fetcher: Fetcher, site: dict, row: InputRow, k: int = 5) -> tuple[list[Candidate], str]:
    """Returns (candidates, note). The note explains an empty result."""
    try:
        if site.get("find_by", "search") == "sitemap":
            urls = sitemap_urls(fetcher, site)
            found = rank_urls(urls, row, k)
            return found, f"{len(urls)} sitemap URLs searched" if urls else "sitemap empty or unreachable"
        found = search_links(fetcher, site, row, k)
        return found, "" if found else "no product links on the search page (results may need JavaScript)"
    except (Blocked, FetchError) as err:
        return [], str(err)
