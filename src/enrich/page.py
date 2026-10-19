"""Read product facts out of a competitor page's HTML.

Three layers, most reliable first:
  1. schema.org Product JSON-LD: name, brand, gtin, sku, price, description.
     Most Shopify stores and many others include it.
  2. Product codes embedded anywhere in the page, e.g. BigCommerce's
     "upc":"814514021204" (Valley Vet has no JSON-LD but has this).
  3. The readable main text (trafilatura), for the model to extract from.
"""
import html as html_lib
import json
import re
from dataclasses import dataclass, field


@dataclass
class PageData:
    url: str
    title: str = ""
    brand: str = ""
    codes: set[str] = field(default_factory=set)     # GTIN/UPC from JSON-LD (leading zeros removed)
    embedded_codes: set[str] = field(default_factory=set)   # codes found elsewhere in the page source
    price: float | None = None
    description: str = ""
    text: str = ""                                    # main readable text


def clean_code(code) -> str:
    digits = re.sub(r"\D", "", str(code or ""))
    return digits.lstrip("0") if 8 <= len(digits) <= 14 else ""


def _jsonld_blocks(html: str) -> list:
    blocks = []
    for raw in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            blocks.append(json.loads(raw.strip()))
        except json.JSONDecodeError:
            continue                                   # broken JSON-LD is common; skip it
    return blocks


def _products(node) -> list[dict]:
    """Every object with @type Product, wherever it sits (lists, @graph, nesting)."""
    found = []
    if isinstance(node, list):
        for item in node:
            found += _products(item)
    elif isinstance(node, dict):
        kind = node.get("@type")
        kinds = kind if isinstance(kind, list) else [kind]
        if "Product" in kinds or "ProductGroup" in kinds:
            found.append(node)
        for key in ("@graph", "hasVariant", "isVariantOf", "itemListElement", "item"):
            if key in node:
                found += _products(node[key])
    return found


def _name(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    if isinstance(value, list):
        return _name(value[0]) if value else ""
    return str(value or "")


def _meta(html: str, prop: str) -> str:
    m = re.search(rf'<meta[^>]+(?:property|name)=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']*)', html, re.I)
    if not m:
        m = re.search(rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]*(?:property|name)=["\']{re.escape(prop)}["\']', html, re.I)
    return html_lib.unescape(m.group(1)).strip() if m else ""


EMBEDDED_CODE = re.compile(r'["\'](?:gtin(?:8|12|13|14)?|upc|ean|barcode)["\']\s*:\s*["\']?(\d{8,14})', re.I)


def main_text(html: str) -> str:
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    except ImportError:
        text = ""
    if not text:                                       # fallback: strip tags
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def read_page(url: str, html: str) -> PageData:
    data = PageData(url=url)
    for product in (p for block in _jsonld_blocks(html) for p in _products(block)):
        data.title = data.title or _name(product.get("name"))
        data.brand = data.brand or _name(product.get("brand"))
        data.description = data.description or html_lib.unescape(str(product.get("description") or ""))
        for key in ("gtin", "gtin8", "gtin12", "gtin13", "gtin14", "upc", "ean"):
            if clean_code(product.get(key)):
                data.codes.add(clean_code(product.get(key)))
        offers = product.get("offers") or {}
        for offer in offers if isinstance(offers, list) else [offers]:
            if isinstance(offer, dict):
                for key in ("gtin", "gtin12", "gtin13", "gtin14"):
                    if clean_code(offer.get(key)):
                        data.codes.add(clean_code(offer.get(key)))
                if data.price is None and offer.get("price") not in (None, ""):
                    try:
                        data.price = float(offer["price"])
                    except (TypeError, ValueError):
                        pass

    data.embedded_codes = {clean_code(c) for c in EMBEDDED_CODE.findall(html) if clean_code(c)} - data.codes
    data.title = data.title or _meta(html, "og:title") or html_lib.unescape(
        (re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S | re.I) or re.search(r"<title>(.*?)</title>", html, re.S | re.I)
         or [None, ""])[1]).strip()
    data.title = re.sub(r"<[^>]+>|\s+", " ", data.title).strip()
    data.brand = data.brand or _meta(html, "product:brand") or _meta(html, "og:brand")
    if data.price is None:
        try:
            data.price = float(_meta(html, "product:price:amount") or _meta(html, "og:price:amount"))
        except ValueError:
            pass
    data.description = data.description or _meta(html, "description") or _meta(html, "og:description")
    data.text = main_text(html)
    return data
