"""Stage 3a: read the supplier's own product images and check them against the draft.

    python -m src.map inbox/VPS6.csv       # makes output/drafts/VPS6.json
    python -m src.images inbox/VPS6.csv    # reads VPS6's images, updates that draft

For each product image (the "Picture URLs" column):

  1. Download it once into cache/images/<sku>/ (later runs reuse the file).
  2. Ask the vision model two things. Does this image show THIS product? And
     what does the label say for size, active ingredient, species and form?
     Every value must come with the exact words printed on the label.
  3. Compare in code with the draft:
        image agrees with the draft  -> add the image as a second source
        image disagrees              -> flag for review; the CSV value stays
        draft field is empty         -> a Suggestion for the review page
        image shows another product  -> flag the image; ignore what it says

What an image says is never written over the supplier's data: vision models
misread small print, and on a medicine label 12.5% vs 1.25% matters.
"""
import json
import mimetypes
import re
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src import llm
from src.config import ROOT, settings
from src.evidence import (Evidence, adds_detail, agrees, apply_evidence, as_list, norm,  # noqa: F401
                          numbers, numbers_in_quote, remove_previous, size_variants)
from src.schemas import Draft, InputRow, Source

TAG = "[image check]"          # marks flags this step adds, so a re-run replaces them


# ====================================================================== what the model returns

class LabelValue(BaseModel):
    value: str | list[str] | None = Field(description="What the label says, or null if it isn't visible")
    label_text: str | None = Field(description="The exact words printed on the label that show this value; null if none")
    confidence: Literal["high", "medium", "low"]


class ImageReading(BaseModel):
    shows_this_product: Literal["yes", "no", "unsure"] = Field(
        description="Does the image show the EXPECTED PRODUCT? 'no' if it shows a different product, brand, "
                    "strength or size; 'unsure' if you can't tell (for example, no label is visible)")
    reason: str = Field(description="One short sentence: what in the image supports that answer")
    brand: LabelValue = Field(description="Brand name printed on the package")
    size: LabelValue = Field(description="Net contents, e.g. '1 L', '16 oz', '60 count'")
    ingredients: LabelValue = Field(
        description="Active ingredient(s) WITH the strength or concentration printed with them, e.g. "
                    "'12.5% tiamulin hydrogen fumarate'. Look across the whole label (product name line, "
                    "'Active Ingredient' section) for the percentage or mg amount; leave it out only if "
                    "none is printed.")
    species: LabelValue = Field(description="LIST of animals the label says the product is for, e.g. ['Swine']")
    form: LabelValue = Field(description="LIST: product form as printed, e.g. ['Liquid Concentrate']")


# Label field -> (draft metafield, how it's compared; see src/evidence.py)
FIELDS = {
    "size": ("custom.sizes", "compare"),
    "ingredients": ("info.ingredients", "compare"),
    "species": ("custom.animal", "compare"),
    "form": ("info.form", "suggest"),
}

SYSTEM = """You read product packaging photos for an animal health retailer.
Report only what is printed and visible in the image. Never use outside knowledge and never guess.
For every value, copy the exact words printed on the label into label_text.
If a value isn't visible, set value to null and confidence to low."""


def build_prompt(row: InputRow) -> str:
    # Only what's needed to recognize the product. The draft's size and
    # ingredients are NOT shown, so the model can't just repeat them back.
    expected = {"brand": row.brand, "name": row.name, "manufacturer": row.manufacturer}
    return ("EXPECTED PRODUCT (from the supplier file):\n"
            + json.dumps({k: v for k, v in expected.items() if v}, ensure_ascii=False, indent=2)
            + "\n\nFirst decide whether the image shows this exact product. Then read the label fields.")


# ====================================================================== downloading

FetchFn = Callable[[str, Path], tuple[bytes, str]]


def fetch_image(url: str, folder: Path) -> tuple[bytes, str]:
    """Return (bytes, mime type) for an image URL, downloading it only once."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / image_name(url)
    if not path.exists():
        import httpx

        timeout = settings().get("images", {}).get("download_timeout_seconds", 20)
        response = httpx.get(url, timeout=timeout, follow_redirects=True,
                             headers={"User-Agent": "supplier-onboarding-agent (portfolio project)"})
        response.raise_for_status()
        path.write_bytes(response.content)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return path.read_bytes(), mime


def image_name(url: str) -> str:
    """'https://cdn.../hres_imgs/VPS6-MAIN.jpg' -> 'VPS6-MAIN.jpg'"""
    name = Path(urlparse(url).path).name or "image"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


# ====================================================================== checks in code

def grounded(value: LabelValue) -> bool:
    """Every number in the value must be printed in the label text the model quoted."""
    return numbers_in_quote(value.value, value.label_text)


def same_brand(row: InputRow, brand: LabelValue) -> bool:
    """Code-side identity check, on top of the model's own yes/no."""
    if not brand.value or not (row.brand or row.name):
        return True                                # nothing to compare: trust the model's answer
    expected = norm(" ".join(p for p in (row.brand, row.manufacturer, row.name) if p))
    return fuzz.partial_ratio(norm(" ".join(as_list(brand.value))), expected) >= 80


# ====================================================================== put together

ReadFn = Callable[[str, list, str], ImageReading]


def default_read_fn(prompt: str, images: list, system: str) -> ImageReading:
    return llm.chat_json(prompt, ImageReading, system=system, images=images)


def check_images(row: InputRow, draft: Draft, flags: list[str],
                 read_fn: ReadFn = default_read_fn, fetch_fn: FetchFn = fetch_image) -> dict:
    """Check one product's images against its draft.

    Updates `draft` (image sources) and `flags` in place, and returns a report:
    {"sku", "images": [...], "suggestions": [Suggestion...], "flags": [...]}.
    Running it twice gives the same result: earlier image sources and flags are
    removed first.
    """
    remove_previous(draft, flags, kind="image", tag=TAG)

    max_images = settings().get("images", {}).get("max_per_product", 3)
    folder = ROOT / "cache" / "images" / row.sku
    new_flags, readings, evidence = [], [], []

    for url in row.images[:max_images]:
        name = image_name(url)
        try:
            data, mime = fetch_fn(url, folder)
        except Exception as err:                    # a broken link is a finding, not a crash
            new_flags.append(f"image: could not download {name} ({type(err).__name__}) {TAG}")
            readings.append({"url": url, "file": name, "error": str(err)})
            continue

        reading = read_fn(build_prompt(row), [(data, mime)], SYSTEM)
        verdict = reading.shows_this_product
        if verdict == "yes" and not same_brand(row, reading.brand):
            verdict = "unsure"
            new_flags.append(f"image: {name} says brand '{reading.brand.value}', expected '{row.brand}' {TAG}")
        readings.append({"url": url, "file": name, "verdict": verdict, "reading": reading.model_dump()})

        if verdict == "no":
            new_flags.append(f"image: {name} does not show this product ({reading.reason}) {TAG}")
            continue
        if verdict == "unsure":
            continue                                # e.g. a lifestyle photo with no label: not used

        for field, (key, _) in FIELDS.items():
            value: LabelValue = getattr(reading, field)
            if value.value in (None, "", []) or value.confidence == "low":
                continue
            if not grounded(value):
                new_flags.append(f"{key}: image {name} gave '{value.value}' but the label text it quoted "
                                 f"doesn't show those numbers; ignored {TAG}")
                continue
            evidence.append(Evidence(
                key=key, value=value.value, quote=value.label_text,
                confidence=value.confidence if value.label_text else "medium",
                source=Source(kind="image", ref=name, url=url, quote=value.label_text)))

    suggestions, compare_flags = apply_evidence(draft, evidence, TAG)
    new_flags += compare_flags
    flags.extend(new_flags)
    return {"sku": row.sku, "images": readings, "suggestions": [s.model_dump() for s in suggestions],
            "flags": new_flags}


def run_file(csv_path: Path, read_fn: ReadFn = default_read_fn, fetch_fn: FetchFn = fetch_image) -> list[dict]:
    """Check every product in a supplier file whose draft already exists."""
    from src.ingest import ingest_file

    rows, _ = ingest_file(csv_path)
    reports = []
    for row in rows:
        draft_path = ROOT / "output" / "drafts" / f"{row.sku}.json"
        if not draft_path.exists():
            print(f"{row.sku}: no draft yet. Run  python -m src.map {csv_path}  first.")
            continue
        if not row.images:
            print(f"{row.sku}: no image URLs in the supplier file; skipped.")
            continue
        saved = json.loads(draft_path.read_text(encoding="utf-8"))
        draft, flags = Draft.model_validate(saved["draft"]), list(saved.get("flags", []))
        report = check_images(row, draft, flags, read_fn, fetch_fn)

        draft_path.write_text(json.dumps({"draft": draft.model_dump(), "flags": flags}, indent=2,
                                         ensure_ascii=False), encoding="utf-8")
        out = ROOT / "output" / "images" / f"{row.sku}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        reports.append(report)
    return reports


def print_report(report: dict) -> None:
    print(f"\n{report['sku']}")
    for img in report["images"]:
        if "error" in img:
            print(f"  {img['file']:<28} download failed: {img['error'][:80]}")
            continue
        r = img["reading"]
        print(f"  {img['file']:<28} shows this product: {img['verdict']:<6} {r['reason'][:70]}")
        for field in FIELDS:
            v = r[field]
            if v["value"] not in (None, "", []):
                print(f"      {field:<12} {str(v['value'])[:40]:<42} [{v['confidence']}]  label: \"{(v['label_text'] or '')[:50]}\"")
    for s in report["suggestions"]:
        was = f"  (draft: {s['current']})" if s["current"] else ""
        print(f"  suggestion  {s['field']} = {s['proposed']['value']}{was}  (images agreeing: {s['agreement']})")
    for f in report["flags"]:
        print(f"  flag        {f.removesuffix(TAG).strip()}")
    if not report["suggestions"] and not report["flags"]:
        print("  no suggestions, no flags: the images agree with the draft")


if __name__ == "__main__":
    # python -m src.images inbox/VPS6.csv
    import sys

    for rep in run_file(Path(sys.argv[1])):
        print_report(rep)
