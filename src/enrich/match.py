"""Is this competitor page the SAME product? Decided in code, not by the model.

A wrong match is the worst error this stage can make: every fact taken from
the page would be about another product. So:

    match      the page's structured GTIN equals our UPC, or our UPC appears in
               the page source AND the page title shows our brand and size
               (a UPC alone could sit in a "related products" widget)
    likely     no usable code, but brand and size both fit; its facts are kept
               at medium confidence at most
    unsure     can't tell: dropped
    no_match   the page's GTINs are different: dropped
"""
from rapidfuzz import fuzz

from src.enrich.page import PageData, clean_code
from src.evidence import norm, size_variants
from src.map import normalize_size
from src.schemas import InputRow

USABLE = {"match", "likely"}


def brand_fits(row: InputRow, page: PageData) -> bool:
    if not row.brand:
        return False
    where = norm(f"{page.title} {page.brand}")
    return norm(row.brand) in where or fuzz.partial_ratio(norm(row.brand), where) >= 90


def size_fits(row: InputRow, page: PageData) -> bool:
    size = normalize_size(row.size)
    return bool(size) and size in size_variants(f"{page.title}\n{page.text[:4000]}")


def check_match(page: PageData, row: InputRow) -> tuple[str, str]:
    upc = clean_code(row.upc)

    if upc and page.codes:
        if upc in page.codes:
            return "match", f"GTIN {row.upc} in the page's product data"
        return "no_match", f"page GTINs {sorted(page.codes)[:3]} don't include {row.upc}"
    if upc and upc in page.embedded_codes:
        if brand_fits(row, page) and size_fits(row, page):
            return "match", f"UPC {row.upc} in the page source; brand and size on the page"
        return "unsure", f"UPC {row.upc} appears in the page source, but the title doesn't fit ({page.title!r})"
    if brand_fits(row, page) and size_fits(row, page):
        return "likely", f"no matching code; brand and size {normalize_size(row.size)!r} both on the page"
    if not brand_fits(row, page):
        return "unsure", f"brand {row.brand!r} not in the page title {page.title!r}"
    return "unsure", f"brand fits but size {normalize_size(row.size)!r} not found on the page"
