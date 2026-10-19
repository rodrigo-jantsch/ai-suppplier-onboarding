"""Tests for stage 3b (competitor lookup). Run:  python -m pytest -q tests/test_enrich.py

No network: a fake "internet" (a dict of URL -> HTML) stands in for the sites,
modeled on what the real ones look like (Valley Vet: BigCommerce, product
sitemap, no JSON-LD, UPC in embedded data; a Shopify-style site with JSON-LD).
The model is a fake that returns fixed facts.
"""
import chromadb
import pytest

from src import map as mapper
from src.enrich import workflow
from src.enrich.extract import PageFacts, extract_facts
from src.enrich.fetch import Blocked, Fetcher, RobotsRules
from src.enrich.find import find_candidates, rank_urls
from src.enrich.match import check_match
from src.enrich.page import read_page
from src.ingest import ingest_file
from src.retrieve import Retriever
from tests.test_map import GOOD_ANSWER, ROOT, fake_model
from tests.test_retrieve import fake_embed

VV = "https://www.valleyvet.com"

VV_ROBOTS = """User-agent: *
Disallow: /search.php
Disallow: /*?_bc_fsnf=1
"""
CHEWY_ROBOTS = """User-agent: *
Disallow: /app/dp/*
Disallow: *query=*
Disallow: /*/dp/*?features

User-agent: Amazonbot
Disallow: /
"""

SITEMAP_INDEX = f"""<?xml version="1.0"?><sitemapindex>
<sitemap><loc>{VV}/xmlsitemap.php?type=pages&amp;page=1</loc></sitemap>
<sitemap><loc>{VV}/xmlsitemap.php?type=products&amp;page=1</loc></sitemap>
</sitemapindex>"""
SITEMAP_PRODUCTS = f"""<?xml version="1.0"?><urlset>
<url><loc>{VV}/iverhart-plus-for-dogs/</loc></url>
<url><loc>{VV}/tiagard-12-5-liquid-concentrate-1-liter/</loc></url>
<url><loc>{VV}/denagard-12-5-liquid-concentrate/</loc></url>
<url><loc>{VV}/cattlemaster-4-vl5-cattle-vaccine/</loc></url>
</urlset>"""

# BigCommerce-style product page: no JSON-LD, UPC inside an embedded data object.
VV_TIAGARD = """<html><head><title>TiaGard 12.5% Liquid Concentrate | Valley Vet</title>
<meta property="product:price:amount" content="23.49"></head><body>
<h1>TiaGard 12.5% Liquid Concentrate, 1 Liter</h1>
<script>var BCData = {"product_attributes":{"sku":"GRP9","upc":"351072092029","mpn":"22001","price":{"value":23.49}}};</script>
<article><p>TiaGard 12.5% Liquid Concentrate contains 12.5% tiamulin hydrogen fumarate.</p>
<p>For use in the drinking water of swine only. Treats swine dysentery and swine pneumonia.</p>
<p>Size: 1 Liter</p></article></body></html>"""

# A different product whose page lists our UPC in a "related products" widget.
VV_DENAGARD = """<html><head><title>Denagard 12.5% Liquid Concentrate</title></head><body>
<h1>Denagard 12.5% Liquid Concentrate</h1>
<script>var related = [{"upc":"351072092029","name":"TiaGard"}];</script>
<article><p>Denagard contains tiamulin.</p></article></body></html>"""

# Shopify-style page with JSON-LD and a GTIN-13 (leading zero).
SHOP_JSONLD = """<html><head><script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"TiaGard® 12.5% Liquid Concentrate 1 L",
 "brand":{"@type":"Brand","name":"Aurora Pharmaceutical"},"gtin13":"0351072092029",
 "description":"12.5% tiamulin hydrogen fumarate for swine.",
 "offers":{"@type":"Offer","price":"25.99"}}</script></head><body><h1>TiaGard</h1></body></html>"""


def internet(extra: dict | None = None):
    pages = {
        f"{VV}/robots.txt": VV_ROBOTS,
        f"{VV}/xmlsitemap.php": SITEMAP_INDEX,
        f"{VV}/xmlsitemap.php?type=products&page=1": SITEMAP_PRODUCTS,
        f"{VV}/tiagard-12-5-liquid-concentrate-1-liter/": VV_TIAGARD,
        f"{VV}/denagard-12-5-liquid-concentrate/": VV_DENAGARD,
        **(extra or {}),
    }
    requested = []

    def get(url):
        requested.append(url)
        return (200, pages[url]) if url in pages else (404, "")
    get.requested = requested
    return get


def fetcher(tmp_path, get=None):
    return Fetcher(get_fn=get or internet(), cache_dir=tmp_path / "pages", delay_seconds=0)


VV_SITE = {"name": "valleyvet", "base_url": f"{VV}/", "find_by": "sitemap", "sitemap_url": f"{VV}/xmlsitemap.php",
           "sitemap_filter": "type=products", "robots_checked": True, "enabled": True}


@pytest.fixture(scope="module")
def vps6():
    return ingest_file(ROOT / "inbox/VPS6.csv")[0][0]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Sitemaps cache under ROOT/cache; point ROOT at a temp dir for find.py."""
    import src.enrich.find as find
    monkeypatch.setattr(find, "ROOT", tmp_path)


def P(value, quote, confidence="high"):
    return {"value": value, "quote": quote, "confidence": confidence}


GOOD_FACTS = {
    "size": P("1 L", "Size: 1 Liter"),
    "ingredients": P("12.5% tiamulin hydrogen fumarate", "contains 12.5% tiamulin hydrogen fumarate"),
    "species": P(["Swine"], "For use in the drinking water of swine only"),
    "form": P(["Liquid Concentrate"], "TiaGard 12.5% Liquid Concentrate"),
}


def facts_model(answer=GOOD_FACTS):
    def model_fn(prompt, schema, system):
        model_fn.prompt = prompt
        return PageFacts.model_validate(answer)
    return model_fn


# ---- robots.txt ---------------------------------------------------------------

@pytest.mark.parametrize("url, allowed", [
    ("https://www.chewy.com/s?query=tiagard", False),          # the wildcard rule stdlib would miss
    ("https://www.chewy.com/tiagard-liquid/dp/12345", True),
    ("https://www.chewy.com/tiagard/dp/12345?features=x", False),
    ("https://www.chewy.com/app/dp/12345", False),
])
def test_robots_wildcards(url, allowed):
    assert RobotsRules(CHEWY_ROBOTS).allowed(url) is allowed


def test_robots_longest_rule_wins_and_allow_breaks_ties():
    rules = RobotsRules("User-agent: *\nDisallow: /shop/\nAllow: /shop/products/\n")
    assert not rules.allowed("https://x.com/shop/cart")
    assert rules.allowed("https://x.com/shop/products/tiagard")


def test_robots_group_for_our_agent_wins():
    rules = RobotsRules("User-agent: *\nAllow: /\n\nUser-agent: SupplierOnboardingAgent\nDisallow: /\nCrawl-delay: 10\n")
    assert not rules.allowed("https://x.com/anything") and rules.crawl_delay == 10


# ---- fetcher ------------------------------------------------------------------

def test_fetcher_respects_robots_and_caches(tmp_path):
    get = internet({"https://www.chewy.com/robots.txt": CHEWY_ROBOTS})
    f = fetcher(tmp_path, get)
    with pytest.raises(Blocked):
        f.get("https://www.chewy.com/s?query=tiagard")
    assert "https://www.chewy.com/s?query=tiagard" not in get.requested       # never requested
    url = f"{VV}/tiagard-12-5-liquid-concentrate-1-liter/"
    assert not f.get(url).from_cache
    assert f.get(url).from_cache and get.requested.count(url) == 1


def test_fetcher_waits_between_requests_to_the_same_site(tmp_path):
    waits = []
    f = Fetcher(get_fn=internet(), cache_dir=tmp_path, delay_seconds=3, sleep=waits.append)
    f.get(f"{VV}/tiagard-12-5-liquid-concentrate-1-liter/")    # robots.txt, then the page
    assert waits and all(0 < w <= 3 for w in waits)


# ---- reading pages ------------------------------------------------------------

def test_read_bigcommerce_page():
    data = read_page(f"{VV}/x/", VV_TIAGARD)
    assert data.title == "TiaGard 12.5% Liquid Concentrate, 1 Liter"
    assert data.codes == set() and "351072092029" in data.embedded_codes
    assert data.price == 23.49 and "drinking water of swine" in data.text


def test_read_jsonld_page():
    data = read_page("https://shop.example/products/tiagard", SHOP_JSONLD)
    assert data.codes == {"351072092029"}                        # GTIN-13 with its leading zero removed
    assert data.brand == "Aurora Pharmaceutical" and data.price == 25.99


# ---- finding and matching -----------------------------------------------------

def test_sitemap_search_finds_the_product(tmp_path, vps6):
    cands, note = find_candidates(fetcher(tmp_path), VV_SITE, vps6)
    assert cands[0].url.endswith("/tiagard-12-5-liquid-concentrate-1-liter/")
    assert "4 sitemap URLs" in note


def test_brand_filter_keeps_only_brand_urls(vps6):
    urls = [f"{VV}/denagard-12-5-liquid-concentrate/", f"{VV}/tiagard-12-5-liquid-concentrate-1-liter/"]
    assert [c.url for c in rank_urls(urls, vps6)] == [urls[1]]


def test_match_by_embedded_upc(vps6):
    assert check_match(read_page("u", VV_TIAGARD), vps6)[0] == "match"


def test_match_by_jsonld_gtin(vps6):
    assert check_match(read_page("u", SHOP_JSONLD), vps6)[0] == "match"


def test_upc_in_a_related_products_widget_is_not_a_match(vps6):
    verdict, reason = check_match(read_page("u", VV_DENAGARD), vps6)
    assert verdict == "unsure" and "title doesn't fit" in reason


def test_different_gtin_is_no_match(vps6):
    other = SHOP_JSONLD.replace("0351072092029", "0351072092036")
    assert check_match(read_page("u", other), vps6)[0] == "no_match"


# ---- extraction ---------------------------------------------------------------

def test_extract_keeps_quoted_facts_and_drops_invented_ones(vps6):
    page = read_page(f"{VV}/x/", VV_TIAGARD)
    answer = {**GOOD_FACTS, "form": P(["Powder"], "Water soluble powder")}      # quote isn't on the page
    model = facts_model(answer)
    kept, problems = extract_facts(vps6, page, model)
    assert set(kept) == {"custom.sizes", "info.ingredients", "custom.animal"}
    assert any(p.startswith("info.form: quote not found") for p in problems)
    assert "<<<PAGE" in model.prompt and "ignore any instructions" not in model.prompt   # page text fenced off


def test_extract_drops_numbers_missing_from_the_quote(vps6):
    page = read_page(f"{VV}/x/", VV_TIAGARD)
    answer = {**GOOD_FACTS, "size": P("2 L", "Size: 1 Liter")}
    kept, problems = extract_facts(vps6, page, facts_model(answer))
    assert "custom.sizes" not in kept and any("numbers its quote doesn't show" in p for p in problems)


# ---- the whole fixed workflow -------------------------------------------------

@pytest.fixture
def draft(vps6):
    retriever = Retriever(embed_fn=fake_embed, client=chromadb.EphemeralClient())
    return mapper.map_row(vps6, retriever, model_fn=fake_model(GOOD_ANSWER))


def test_workflow_end_to_end(tmp_path, vps6, draft):
    d, flags = draft
    d.metafields["info.ingredients"].value = "Tiamulin Hydrogen Fumarate"     # what a real run produced
    report = workflow.enrich_row(vps6, d, flags, [VV_SITE], fetcher(tmp_path), facts_model())
    [site] = report["sites"]
    assert site["matched_url"].endswith("/tiagard-12-5-liquid-concentrate-1-liter/") and site["price"] == 23.49
    sizes = d.metafields["custom.sizes"]
    assert sizes.value == ["1 L"] and any(s.kind == "competitor" and s.ref == "valleyvet" for s in sizes.sources)
    [upgrade] = [s for s in report["suggestions"] if s["field"] == "info.ingredients"]
    assert upgrade["proposed"]["value"] == "12.5% tiamulin hydrogen fumarate"
    assert upgrade["current"] == "Tiamulin Hydrogen Fumarate"
    assert report["flags"] == []


def test_workflow_disagreement_is_flagged(tmp_path, vps6, draft):
    d, flags = draft
    answer = {**GOOD_FACTS, "size": P("1 gal", "available in 1 gal")}
    page = VV_TIAGARD.replace("<p>Size: 1 Liter</p>", "<p>Size: 1 Liter, also available in 1 gal</p>")
    f = fetcher(tmp_path, internet({f"{VV}/tiagard-12-5-liquid-concentrate-1-liter/": page}))
    report = workflow.enrich_row(vps6, d, flags, [VV_SITE], f, facts_model(answer))
    assert d.metafields["custom.sizes"].value == ["1 L"]
    assert any(fl.startswith("custom.sizes: competitor valleyvet says '1 gal'") for fl in report["flags"])


def test_workflow_rerun_gives_same_result(tmp_path, vps6, draft):
    d, flags = draft
    f = fetcher(tmp_path)
    workflow.enrich_row(vps6, d, flags, [VV_SITE], f, facts_model())
    first = (list(flags), d.model_dump())
    workflow.enrich_row(vps6, d, flags, [VV_SITE], f, facts_model())
    assert (flags, d.model_dump()) == first


def test_blocked_search_site_is_reported_not_crashed(tmp_path, vps6, draft):
    d, flags = draft
    chewy = {"name": "chewy", "base_url": "https://www.chewy.com/", "find_by": "search",
             "search_url": "https://www.chewy.com/s?query={query}", "product_path_hint": "/dp/"}
    f = fetcher(tmp_path, internet({"https://www.chewy.com/robots.txt": CHEWY_ROBOTS}))
    report = workflow.enrich_row(vps6, d, flags, [chewy], f, facts_model())
    assert report["sites"][0]["candidates"] == [] and "robots.txt disallows" in report["sites"][0]["note"]


def test_site_config_problems_are_explained():
    bad = {"name": "tractor suply", "find_by": "sitemap", "base_url": "https://www.tractorsupply.com/"}
    problems = workflow.site_problems(bad)
    assert any("use lowercase" in p for p in problems) and any("needs sitemap_url" in p for p in problems)
    assert workflow.site_problems(VV_SITE) == []


def test_real_config_is_valid():
    import yaml
    sites = yaml.safe_load((ROOT / "config/competitors.yaml").read_text(encoding="utf-8"))["sites"]
    assert all(workflow.site_problems(s) == [] for s in sites)


def test_unreachable_site_is_reported_and_others_still_run(tmp_path, vps6, draft):
    import httpx
    d, flags = draft
    good = internet()

    def get(url):
        if "tractorsupply" in url:
            raise httpx.ReadTimeout("The read operation timed out")
        return good(url)
    tsc = {**VV_SITE, "name": "tractorsupply", "base_url": "https://www.tractorsupply.com/",
           "sitemap_url": "https://www.tractorsupply.com/sitemap.xml"}
    report = workflow.enrich_row(vps6, d, flags, [tsc, VV_SITE], fetcher(tmp_path, get), facts_model())
    tsc_report, vv_report = report["sites"]
    assert "ReadTimeout" in tsc_report["note"] and "not crawling this site" in tsc_report["note"]
    assert vv_report["matched_url"]                                   # Valley Vet still ran
