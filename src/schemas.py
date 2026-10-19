"""Data shapes passed between stages.

Every value an LLM proposes carries its source, so validation and the review
page can check it. These models double as the JSON schemas sent to the LLM
for structured output.
"""
from typing import Literal

from pydantic import BaseModel, Field

Value = str | float | bool | list[str] | None


class InputRow(BaseModel):
    """Stage 1 output: one supplier row mapped to the standard input schema.

    Fixed rules (BIO-01, DRO-01, HAZ-02, PROP-01, RX-01, TAG-01) read `attributes`
    and `parent_sku` directly, so nothing the supplier sent is lost.
    """
    sku: str
    supplier: str = Field(description="Source file name, e.g. VPS6.csv")
    name: str
    upc: str | None = None
    brand: str | None = None
    manufacturer: str | None = None
    short_description: str | None = None
    description: str | None = None
    size: str | None = None
    cost: float | None = None
    msrp: float | None = None
    weight_lb: float | None = None
    height: float | None = None
    length: float | None = None
    width: float | None = None
    parent_sku: str | None = None
    country_of_origin: str | None = None
    animal_type: str | None = None
    bullets: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict, description="All AttributeN name/value pairs")
    raw: dict[str, str] = Field(default_factory=dict, description="Original row, untouched")


class Source(BaseModel):
    kind: Literal["csv", "rule", "example", "competitor", "image"]
    ref: str = Field(description="CSV column name, rule id, example sku, competitor name, or image file name")
    url: str | None = None
    quote: str | None = Field(default=None, description="Exact text copied from the CSV cell, page, or product label")


class FieldValue(BaseModel):
    value: Value
    sources: list[Source]
    confidence: Literal["high", "medium", "low"]


class Draft(BaseModel):
    """Stage 2 output: built ONLY from supplier data + store rules."""
    sku: str
    title: FieldValue
    product_type: FieldValue
    vendor: FieldValue
    tags: FieldValue
    price: FieldValue
    barcode: FieldValue
    handle: FieldValue
    metafields: dict[str, FieldValue] = Field(description='Product metafields, keys like "custom.animal"')
    variant_metafields: dict[str, FieldValue] = Field(default_factory=dict, description='Variant metafields, keys like "custom.hazardous"')
    weight_lb: FieldValue | None = None
    body_html: FieldValue


class Suggestion(BaseModel):
    """Stage 3 output: from product images or competitor pages, needs human approval."""
    field: str
    proposed: FieldValue
    current: Value = None
    agreement: int = Field(description="How many sources (images or competitor pages) agree on this value")
    reason: str


class ReviewItem(BaseModel):
    """Stage 4 output: what the review page shows for one SKU."""
    draft: Draft
    suggestions: list[Suggestion] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
