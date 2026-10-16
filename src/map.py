"""Stage 2b: build a Shopify listing Draft from one InputRow.

Two kinds of work, kept deliberately separate:

  1. FIXED RULES, in plain code (deterministic): handle, tags, Prop 65, Rx,
     hazardous, ORMD, dropship, biological, sizes, plain field copies.
     Same input -> same output, every time. The model never touches these.

  2. GUIDANCE, by the model (Gemini), with retrieved context: title, Shopify
     category, description, and the store's filter fields. The model only sees
     the product, the category candidates, the guidance rules and examples.

Then code checks the model's answer: the category must be a real Shopify
category, and every quote the model gives as evidence must exist in the
product data. Problems become flags for the review step.

Try it:  python -m src.map inbox/VPS6.csv
"""
import json
import re
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src import llm
from src.config import ROOT, settings
from src.retrieve import Retriever, load_rule_docs, retrieve_for_row
from src.schemas import Draft, FieldValue, InputRow, Source

PROP65_DISCLOSURE = "gid://shopify/Metaobject/238268514493"


# ====================================================================== helpers

def yes(value: str | None) -> bool:
    """Supplier yes/no flags: 'Y', 'Yes', '1', 'true' -> True; anything else False."""
    return (value or "").strip().upper() in {"Y", "YES", "1", "TRUE"}


def fv(value, *sources: Source, confidence: str = "high") -> FieldValue:
    return FieldValue(value=value, sources=list(sources), confidence=confidence)


def csv_src(ref: str, quote=None) -> Source:
    return Source(kind="csv", ref=ref, quote=None if quote is None else str(quote))


def rule_src(rule_id: str) -> Source:
    return Source(kind="rule", ref=rule_id)


def slugify(title: str) -> str:
    """HANDLE-01: lowercase; any run of non-letters/digits -> one hyphen; trim."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def normalize_size(raw: str | None) -> str | None:
    """UNIT-01.  '16OZ'->'16 oz', '1 liter'->'1 L', '13.6KG'->'30 lb', '60CT'->'60 count'."""
    if not raw:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z. ]+?)\s*", raw)
    if not m:
        return raw.strip()
    number, unit = float(m.group(1)), m.group(2).lower().replace(".", "").strip()
    fmt = lambda n: f"{n:g}"  # noqa: E731  16.0 -> "16"
    if unit in {"oz", "fl oz", "floz", "ounce", "ounces"}:
        return f"{fmt(number)} oz"
    if unit in {"ml", "milliliter", "milliliters"}:
        return f"{fmt(number)} ml"
    if unit in {"l", "liter", "liters", "litre", "litres"}:
        return f"{fmt(number)} L"
    if unit in {"lb", "lbs", "pound", "pounds"}:
        return f"{fmt(number)} lb"
    if unit in {"kg", "kgs", "kilogram", "kilograms"}:
        return f"{fmt(round(number * 2.2046, 1))} lb"
    if unit in {"ct", "count", "pk", "pack"}:
        return f"{fmt(number)} count"
    return raw.strip()


# ====================================================================== 1. fixed rules

def apply_fixed_rules(row: InputRow) -> tuple[dict, list[str]]:
    """Everything that code can decide from the supplier data alone.

    Returns (fields, flags). `fields` uses Draft field names, plus
    "metafields" and "variant_metafields" dicts. Handle and title-dependent
    values are added later, once the title exists.
    """
    a = row.attributes
    flags: list[str] = []
    mf: dict[str, FieldValue] = {}
    vmf: dict[str, FieldValue] = {}
    tags: list[str] = []
    tag_sources: list[Source] = []

    # ---- plain copies from the supplier columns
    fields = {
        "vendor": fv(row.brand, csv_src("brand", row.brand)),
        "price": fv(row.msrp, csv_src("msrp", row.msrp)),
        "barcode": fv(row.upc, csv_src("upc", row.upc)),
    }
    if row.weight_lb is not None:
        fields["weight_lb"] = fv(row.weight_lb, csv_src("weight_lb", row.weight_lb), rule_src("UNIT-01"))
    if row.manufacturer:
        mf["custom.vendor"] = fv(row.manufacturer, csv_src("manufacturer", row.manufacturer))
    if row.country_of_origin:
        mf["custom.country_of_origin"] = fv(row.country_of_origin,
                                            csv_src("Country/Region of Manufacturer", row.country_of_origin))
    if row.animal_type:
        animals = [x.strip() for x in re.split(r"[,;/]", row.animal_type) if x.strip()]
        mf["custom.animal"] = fv(animals, csv_src("Animal Type", row.animal_type))
    if a.get("Jeffers Product Group Code"):
        mf["custom.pgc"] = fv(a["Jeffers Product Group Code"],
                              csv_src("Jeffers Product Group Code", a["Jeffers Product Group Code"]))
    # MERCK-01: Merck items are identified by SKU prefix
    mf["custom.is_merck"] = fv(row.sku.strip().startswith("012"), csv_src("sku", row.sku), rule_src("MERCK-01"))
    mf["compliance.california_signature_required"] = fv(
        yes(a.get("California Sig Required")), csv_src("California Sig Required", a.get("California Sig Required")))
    vmf["custom.fob"] = fv(yes(a.get("Jeffers Charge FOB")), csv_src("Jeffers Charge FOB", a.get("Jeffers Charge FOB")))

    # ---- UNIT-01: sizes
    size = normalize_size(row.size)
    if size:
        mf["custom.sizes"] = fv([size], csv_src("size", row.size), rule_src("UNIT-01"))

    # ---- TAG-01: parent reference + New Arrival
    if row.parent_sku:
        tags.append(f"CatalogRef: {row.parent_sku}")
        tag_sources.append(csv_src("Variation Parent SKU", row.parent_sku))
    tags.append("New Arrival")
    tag_sources.append(rule_src("TAG-01"))

    # ---- PROP-01
    validated = yes(a.get("Jeffers Prop 65 Validated"))
    prop65 = yes(a.get("Jeffers Is Prop 65")) if validated else True
    mf["compliance.prop_65"] = fv(prop65, csv_src("Jeffers Is Prop 65", a.get("Jeffers Is Prop 65")),
                                  csv_src("Jeffers Prop 65 Validated", a.get("Jeffers Prop 65 Validated")),
                                  rule_src("PROP-01"))
    if prop65:
        mf["shopify.disclosure"] = fv([PROP65_DISCLOSURE], rule_src("PROP-01"))

    # ---- RX-01 (attribute first, legend text as backup)
    legend = "federal law restricts this drug to use by or on the order of a licensed veterinarian"
    text = " ".join(filter(None, [row.description, row.short_description, *row.bullets])).lower()
    rx = yes(a.get("Is Prescription Required")) or legend in text
    mf["custom.prescription_rx_"] = fv(rx, csv_src("Is Prescription Required", a.get("Is Prescription Required")),
                                       rule_src("RX-01"))
    if rx:
        tags.append("GMC: RX")
        tag_sources.append(rule_src("RX-01"))
        flags.append("RX-01: prescription item, needs review")

    # ---- HAZ-02, BIO-01, DRO-01 (variant level)
    vmf["custom.hazardous"] = fv(yes(a.get("Jeffers Is Hazardous")),
                                 csv_src("Jeffers Is Hazardous", a.get("Jeffers Is Hazardous")), rule_src("HAZ-02"))
    vmf["custom.ormd"] = fv(yes(a.get("Is DOT/ORMD")), csv_src("Is DOT/ORMD", a.get("Is DOT/ORMD")), rule_src("HAZ-02"))
    bio = yes(a.get("Jeffers Is Biological"))
    vmf["custom.biological"] = fv(bio, csv_src("Jeffers Is Biological", a.get("Jeffers Is Biological")),
                                  rule_src("BIO-01"))
    if bio:
        tags.append("GMC: Vaccine Shipping")
        tag_sources.append(rule_src("BIO-01"))
        flags.append("BIO-01: biological, needs review")
    drop = yes(a.get("Is Dropship"))
    vmf["custom.dropship"] = fv(drop, csv_src("Is Dropship", a.get("Is Dropship")), rule_src("DRO-01"))
    if drop:
        tags.append("GMC: Dropship")
        tag_sources.append(rule_src("DRO-01"))

    fields["tags"] = fv(tags, *tag_sources)
    fields["metafields"] = mf
    fields["variant_metafields"] = vmf
    return fields, flags


# ====================================================================== 2. guidance (model)

class Proposed(BaseModel):
    """One value the model proposes, with its evidence."""
    value: str | list[str] | None = Field(description="The value, or null if the product data doesn't support one")
    quote: str | None = Field(description="Exact text copied from the PRODUCT DATA that supports the value; null if none")
    confidence: Literal["high", "medium", "low"]


class ModelListing(BaseModel):
    """What the model is asked to fill. Everything else comes from code."""
    title: Proposed
    product_type: Proposed = Field(description="Exactly one category name, copied from CATEGORY CANDIDATES")
    body_html: Proposed = Field(description="Short HTML description: one <p> and a <ul> of key facts")
    ingredients: Proposed = Field(description="Active ingredient(s) and strength, e.g. '12.5% tiamulin hydrogen fumarate'")
    form: Proposed = Field(description="List: product form and how it's given, e.g. ['Solution', 'Drinking Water']")
    department: Proposed
    category: Proposed
    product_types: Proposed = Field(description="List of short filter terms: uses, conditions treated, product format")


SYSTEM = """You turn supplier product data into a Shopify listing for an animal health retailer.
Use ONLY the PRODUCT DATA below as facts. Never invent facts, numbers or claims.
For every field, copy an exact supporting phrase from PRODUCT DATA into `quote`.
If the data doesn't support a field, set value to null and confidence to low.
Fix obvious typos from the supplier (e.g. a stray "NA" before a word) instead of copying them.
Follow the GUIDANCE RULES exactly."""


def build_prompt(row: InputRow, candidates: list[dict], guidance_rules: list[str], examples: list[str]) -> str:
    product = row.model_dump(exclude={"raw", "images", "attributes", "supplier"}, exclude_none=True)
    product["attributes"] = {k: v for k, v in row.attributes.items() if v and not k.startswith("Bullet Point")}
    lines = [
        "PRODUCT DATA:", json.dumps(product, ensure_ascii=False, indent=2), "",
        "CATEGORY CANDIDATES (choose exactly one, copy the name exactly):",
        *[f"- {c['name']}" for c in candidates], "",
        "GUIDANCE RULES:", *guidance_rules, "",
        "APPROVED EXAMPLES (style reference only, not facts about this product):", *examples,
    ]
    return "\n".join(lines)


def guidance_rule_texts() -> list[str]:
    """All guidance rules, always: there are only a few (see the retrieval notes)."""
    return [d.text for d in load_rule_docs() if d.metadata.get("kind") == "guidance"]


# ====================================================================== 3. checks

def product_text(row: InputRow) -> str:
    """All supplier text the model was allowed to quote from."""
    parts = [row.name, row.brand, row.manufacturer, row.size, row.short_description, row.description,
             row.animal_type, *row.bullets, *row.attributes.values()]
    return " ".join(p for p in parts if p)


def quote_found(quote: str | None, source_text: str, threshold: int) -> bool:
    """True if the quote (approximately) appears in the source text."""
    if not quote:
        return False
    return fuzz.partial_ratio(quote.lower(), source_text.lower()) >= threshold


# ====================================================================== put together

ModelFn = Callable[[str, type[BaseModel], str], BaseModel]


def default_model_fn(prompt: str, schema: type[BaseModel], system: str) -> BaseModel:
    return llm.chat_json(prompt, schema, system=system)


def map_row(row: InputRow, retriever: Retriever, model_fn: ModelFn = default_model_fn) -> tuple[Draft, list[str]]:
    """Build one Draft for one product. Returns (draft, flags)."""
    threshold = settings()["validation"]["quote_match_threshold"]
    fields, flags = apply_fixed_rules(row)

    retrieved = retrieve_for_row(retriever, row)
    candidates = retrieved["categories"]
    examples = [h.metadata["json"] for h in retrieved["examples"]]
    prompt = build_prompt(row, candidates, guidance_rule_texts(), examples)
    proposal: ModelListing = model_fn(prompt, ModelListing, SYSTEM)

    source_text = product_text(row)

    def from_model(p: Proposed, rule_id: str | None = None, key: str = "") -> FieldValue:
        """Turn a model proposal into a FieldValue, checking its evidence.

        - quote given but not in the supplier data -> confidence low + flag
        - no quote at all                           -> confidence at most medium
        - model itself says low                     -> flag for review
        """
        confidence = p.confidence
        if p.value not in (None, "", []):
            if p.quote and not quote_found(p.quote, source_text, threshold):
                confidence = "low"
                flags.append(f"{key}: quote not found in supplier data")
            elif not p.quote and confidence == "high":
                confidence = "medium"
            elif confidence == "low":
                flags.append(f"{key}: low confidence")
        sources = [Source(kind="csv", ref="product data", quote=p.quote)]
        if rule_id:
            sources.append(rule_src(rule_id))
        return FieldValue(value=p.value, sources=sources, confidence=confidence)

    # Category must be one of the candidates (and therefore a real Shopify category).
    names = {c["name"] for c in candidates}
    category = from_model(proposal.product_type, "TAX-01", "product_type")
    if category.value not in names:
        flags.append(f"product_type: '{category.value}' is not a candidate; using the top level instead")
        category = fv("Animals & Pet Supplies", rule_src("TAX-01"), confidence="low")

    title = from_model(proposal.title, "TITLE-01", "title")
    handle_value = slugify(title.value) if isinstance(title.value, str) else None

    mf = fields["metafields"]
    for key, proposed in (("info.ingredients", proposal.ingredients), ("info.form", proposal.form),
                          ("custom.department", proposal.department), ("custom.category", proposal.category),
                          ("custom.product_types", proposal.product_types)):
        if proposed.value not in (None, "", []):
            mf[key] = from_model(proposed, key=key)

    draft = Draft(
        sku=row.sku,
        title=title,
        product_type=category,
        handle=fv(handle_value, rule_src("HANDLE-01")),
        body_html=from_model(proposal.body_html, key="body_html"),
        **fields,
    )
    return draft, flags


def save_draft(draft: Draft, flags: list[str]) -> Path:
    out_dir = ROOT / "output" / "drafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{draft.sku}.json"
    path.write_text(json.dumps({"draft": draft.model_dump(), "flags": flags}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


if __name__ == "__main__":
    # python -m src.map inbox/VPS6.csv
    import sys

    from src.ingest import ingest_file

    rows, _ = ingest_file(Path(sys.argv[1]))
    retriever = Retriever()
    for row in rows:
        draft, flags = map_row(row, retriever)
        path = save_draft(draft, flags)
        print(f"\n=== {draft.sku}")
        print(f"  title:        {draft.title.value}  ({draft.title.confidence})")
        print(f"  category:     {draft.product_type.value}  ({draft.product_type.confidence})")
        print(f"  handle:       {draft.handle.value}")
        print(f"  tags:         {draft.tags.value}")
        for k, v in draft.metafields.items():
            print(f"  {k:<42} {v.value}  ({v.confidence})")
        for k, v in draft.variant_metafields.items():
            print(f"  variant {k:<34} {v.value}")
        print(f"  flags: {flags or 'none'}")
        print(f"  saved: {path.relative_to(ROOT)}")
