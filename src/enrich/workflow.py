"""Stage 3b, fixed version: look a product up on competitor sites.

    python -m src.map inbox/VPS6.csv                # the draft
    python -m src.enrich.workflow inbox/VPS6.csv    # competitor suggestions

The same steps in the same order for every SKU (no agent deciding yet):

    for each enabled site in config/competitors.yaml:
        find candidate pages      (sitemap or search page)        find.py
        for the top few:
            fetch the page        (robots.txt, delay, cache)      fetch.py
            read it               (JSON-LD, codes, main text)     page.py
            same product?         (GTIN / brand + size, in code)  match.py
            if yes: extract facts (model, with quotes; checked)   extract.py
                    and stop looking on this site
    compare all facts with the draft                              src/evidence.py

Weekend 5 builds the agent version of this loop and compares the two on the
gold set. This fixed version is the baseline it has to beat.
"""
import json
import re
from pathlib import Path

import yaml

from src.config import ROOT
from src.enrich.extract import ModelFn, default_model_fn, extract_facts
from src.enrich.fetch import Blocked, Fetcher, FetchError
from src.enrich.find import find_candidates
from src.enrich.match import USABLE, check_match
from src.enrich.page import read_page
from src.evidence import Evidence, apply_evidence, remove_previous
from src.schemas import Draft, InputRow, Source

TAG = "[competitors]"
PAGES_PER_SITE = 3


REQUIRED = {"sitemap": ["sitemap_url"], "search": ["search_url"]}


def site_problems(site: dict) -> list[str]:
    """What's wrong with one competitors.yaml entry (empty list = fine)."""
    name = site.get("name", "?")
    problems = []
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        problems.append(f"name {name!r}: use lowercase letters, digits, - or _ (it's used in file names)")
    find_by = site.get("find_by", "search")
    if find_by not in REQUIRED:
        problems.append(f"{name}: find_by must be one of {sorted(REQUIRED)}, not {find_by!r}")
    for key in REQUIRED.get(find_by, []):
        if not site.get(key):
            problems.append(f"{name}: find_by {find_by} needs {key}")
    if find_by == "search" and "{query}" not in (site.get("search_url") or "{query}"):
        problems.append(f"{name}: search_url needs a {{query}} placeholder")
    return problems


def enabled_sites() -> list[dict]:
    data = yaml.safe_load((ROOT / "config/competitors.yaml").read_text(encoding="utf-8"))
    sites = []
    for site in data.get("sites", []):
        if not (site.get("enabled") and site.get("robots_checked")):
            continue
        problems = site_problems(site)
        if problems:
            print("Skipping a site in config/competitors.yaml:\n  " + "\n  ".join(problems))
            continue
        sites.append(site)
    return sites


def enrich_row(row: InputRow, draft: Draft, flags: list[str], sites: list[dict], fetcher: Fetcher,
               model_fn: ModelFn = default_model_fn) -> dict:
    """Look one product up on every site. Updates draft sources and flags in place; returns a report."""
    remove_previous(draft, flags, kind="competitor", tag=TAG)
    evidence, new_flags, site_reports = [], [], []

    for site in sites:
        print(f"{row.sku}: looking on {site['name']}...", flush=True)
        report = {"site": site["name"], "candidates": [], "pages": [], "matched_url": None, "note": ""}
        candidates, report["note"] = find_candidates(fetcher, site, row)
        report["candidates"] = [c.__dict__ for c in candidates]

        for cand in candidates[:PAGES_PER_SITE]:
            try:
                page = fetcher.get(cand.url)
            except Blocked as err:
                report["pages"].append({"url": cand.url, "verdict": "blocked", "reason": str(err)})
                continue
            except FetchError as err:
                report["pages"].append({"url": cand.url, "verdict": "error", "reason": f"no answer: {err}"})
                continue
            if page.status != 200:
                report["pages"].append({"url": cand.url, "verdict": "error", "reason": f"HTTP {page.status}"})
                continue
            data = read_page(cand.url, page.html)
            verdict, reason = check_match(data, row)
            report["pages"].append({"url": cand.url, "title": data.title, "verdict": verdict, "reason": reason,
                                    "price": data.price})
            if verdict not in USABLE:
                continue

            facts, problems = extract_facts(row, data, model_fn)
            new_flags += [f"{p} {TAG}" for p in problems]
            report["matched_url"], report["price"] = cand.url, data.price
            report["facts"] = {k: v.model_dump() for k, v in facts.items()}
            for key, v in facts.items():
                confidence = v.confidence if verdict == "match" else "medium"   # "likely" match: never high
                evidence.append(Evidence(key=key, value=v.value, quote=v.quote, confidence=confidence,
                                         source=Source(kind="competitor", ref=site["name"], url=cand.url, quote=v.quote)))
            break                                             # one matched page per site is enough
        site_reports.append(report)

    suggestions, compare_flags = apply_evidence(draft, evidence, TAG)
    new_flags += compare_flags
    flags.extend(new_flags)
    return {"sku": row.sku, "sites": site_reports, "suggestions": [s.model_dump() for s in suggestions],
            "flags": new_flags}


def run_file(csv_path: Path, fetcher: Fetcher | None = None, model_fn: ModelFn = default_model_fn) -> list[dict]:
    from src.ingest import ingest_file

    sites = enabled_sites()
    if not sites:
        print("No competitor sites are enabled (enabled: true and robots_checked: true) in config/competitors.yaml")
        return []
    fetcher = fetcher or Fetcher()
    rows, _ = ingest_file(csv_path)
    reports = []
    for row in rows:
        draft_path = ROOT / "output" / "drafts" / f"{row.sku}.json"
        if not draft_path.exists():
            print(f"{row.sku}: no draft yet. Run  python -m src.map {csv_path}  first.")
            continue
        saved = json.loads(draft_path.read_text(encoding="utf-8"))
        draft, flags = Draft.model_validate(saved["draft"]), list(saved.get("flags", []))
        report = enrich_row(row, draft, flags, sites, fetcher, model_fn)
        draft_path.write_text(json.dumps({"draft": draft.model_dump(), "flags": flags}, indent=2,
                                         ensure_ascii=False), encoding="utf-8")
        out = ROOT / "output" / "competitors" / f"{row.sku}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        reports.append(report)
    return reports


def print_report(report: dict) -> None:
    print(f"\n{report['sku']}")
    for site in report["sites"]:
        print(f"  {site['site']}: {len(site['candidates'])} candidate(s)" + (f"  ({site['note']})" if site["note"] else ""))
        for c in site["candidates"]:
            print(f"      {c['score']:>6}  {c['url']}")
        for p in site["pages"]:
            print(f"    {p['verdict']:<9} {p['url']}\n              {p['reason']}")
        for key, v in (site.get("facts") or {}).items():
            print(f"    fact      {key:<18} {str(v['value'])[:40]:<42} quote: \"{(v['quote'] or '')[:50]}\"")
        if site.get("price") is not None:
            print(f"    price     ${site['price']}")
    for s in report["suggestions"]:
        was = f"  (draft: {s['current']})" if s["current"] else ""
        print(f"  suggestion  {s['field']} = {s['proposed']['value']}{was}  (sites agreeing: {s['agreement']})")
    for f in report["flags"]:
        print(f"  flag        {f.removesuffix(TAG).strip()}")


if __name__ == "__main__":
    # python -m src.enrich.workflow inbox/VPS6.csv
    import sys

    for rep in run_file(Path(sys.argv[1])):
        print_report(rep)
