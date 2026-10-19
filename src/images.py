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
from src.map import normalize_size
from src.schemas import Draft, FieldValue, InputRow, Source, Suggestion

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
    ingredients: LabelValue = Field(description="Active ingredient(s) WITH strength, e.g. '12.5% tiamulin hydrogen fumarate'")
    species: LabelValue = Field(description="LIST of animals the label says the product is for, e.g. ['Swine']")
    form: LabelValue = Field(description="LIST: product form as printed, e.g. ['Liquid Concentrate']")


# Label field -> draft metafield, and how the two are compared.
#   compare: a value on both sides is checked; disagreement is flagged
#   suggest: only used to fill an empty field (the label's wording, like
#            "Liquid Concentrate", rarely matches the store's filter terms)
FIELDS = {
    "size": ("custom.sizes", "compare"),
    "ingredients": ("info.ingredients", "compare"),
    "species": ("custom.animal", "compare"),
    "form": ("info.form", "suggest"),
}
LIST_FIELDS = {"custom.sizes", "custom.animal", "info.form"}

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

def numbers(text) -> set[str]:
    """Every number in a value: '12.5% tiamulin' -> {'12.5'}; ['1 L'] -> {'1'}."""
    if isinstance(text, list):
        text = " ".join(text)
    return {f"{float(n):g}" for n in re.findall(r"\d+(?:\.\d+)?", str(text or ""))}


def as_list(value) -> list[str]:
    if value in (None, "", []):
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("®", "").replace("™", "").casefold()).strip()


def size_variants(text: str) -> set[str]:
    """'1 Liter (33.8 fl oz)' -> {'1 L', '33.8 oz', ...}: every amount printed, normalized."""
    found = {normalize_size(text)}
    for amount in re.findall(r"\d+(?:\.\d+)?\s*[a-zA-Z][a-zA-Z. ]*?(?=[\s,()/]|$)", text):
        found.add(normalize_size(amount))
    return found


def agrees(key: str, draft_value, image_value) -> bool:
    if key == "custom.sizes":
        printed = set().union(*(size_variants(v) for v in as_list(image_value))) if as_list(image_value) else set()
        return {normalize_size(v) for v in as_list(draft_value)} <= printed
    if key == "info.ingredients":
        same_numbers = numbers(draft_value) == numbers(image_value)
        return same_numbers and fuzz.token_set_ratio(norm(draft_value), norm(image_value)) >= 80
    # lists of names (species): agree if they share any entry
    return bool({norm(v) for v in as_list(draft_value)} & {norm(v) for v in as_list(image_value)})


def adds_detail(key: str, draft_value, image_value) -> bool:
    """The label says the same thing plus a strength the draft is missing:
    draft 'Tiamulin Hydrogen Fumarate', label '12.5% tiamulin hydrogen fumarate'."""
    if key != "info.ingredients" or numbers(draft_value) or not numbers(image_value):
        return False
    return fuzz.token_set_ratio(norm(draft_value), norm(image_value)) >= 80


def grounded(value: LabelValue) -> bool:
    """Every number in the value must be printed in the label text the model quoted."""
    return numbers(value.value) <= numbers(value.label_text)


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
    # Start clean, so re-running doesn't duplicate anything.
    flags[:] = [f for f in flags if not f.endswith(TAG)]
    for fv in draft.metafields.values():
        fv.sources = [s for s in fv.sources if s.kind != "image"]

    max_images = settings().get("images", {}).get("max_per_product", 3)
    folder = ROOT / "cache" / "images" / row.sku
    new_flags, readings, found = [], [], {}        # found: key -> {normalized value: [(value, Source, conf)]}
    current_of = {}                                # key -> draft value the suggestion would replace

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

        for field, (key, mode) in FIELDS.items():
            value: LabelValue = getattr(reading, field)
            if value.value in (None, "", []) or value.confidence == "low":
                continue
            if not grounded(value):
                new_flags.append(f"{key}: image {name} gave '{value.value}' but the label text it quoted "
                                 f"doesn't show those numbers; ignored {TAG}")
                continue
            confidence = value.confidence if value.label_text else "medium"
            source = Source(kind="image", ref=name, url=url, quote=value.label_text)
            current = draft.metafields.get(key)

            if current is not None and current.value not in (None, "", []):
                if mode != "compare":
                    continue
                if agrees(key, current.value, value.value):
                    current.sources.append(source)
                    if current.confidence == "medium":
                        current.confidence = "high"     # two independent sources agree
                    elif current.confidence == "low":
                        current.confidence = "medium"
                elif adds_detail(key, current.value, value.value):
                    current_of[key] = current.value     # same thing, more complete: suggest an upgrade
                    normalized = tuple(sorted(norm(v) for v in as_list(value.value)))
                    found.setdefault(key, {}).setdefault(normalized, []).append((value.value, source, confidence))
                else:
                    new_flags.append(f"{key}: image {name} says '{value.value}', draft says "
                                     f"'{current.value}' {TAG}")
            else:
                normalized = tuple(sorted(norm(v) for v in as_list(value.value)))
                found.setdefault(key, {}).setdefault(normalized, []).append((value.value, source, confidence))

    # Empty draft fields: one suggestion per distinct value; several images agreeing count up.
    suggestions = []
    for key, by_value in found.items():
        if len(by_value) > 1:
            new_flags.append(f"{key}: images disagree on this value; see suggestions {TAG}")
        for entries in by_value.values():
            value, _, confidence = entries[0]
            if key in LIST_FIELDS and isinstance(value, str):
                value = [value]
            elif key not in LIST_FIELDS and isinstance(value, list):
                value = ", ".join(value)
            files = ", ".join(s.ref for _, s, _ in entries)
            suggestions.append(Suggestion(
                field=key,
                proposed=FieldValue(value=value, sources=[s for _, s, _ in entries], confidence=confidence),
                current=current_of.get(key),
                agreement=len(entries),
                reason=(f"The label ({files}) adds detail the draft is missing" if key in current_of
                        else f"Read from the supplier's image ({files}); the draft has no value"),
            ))

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
