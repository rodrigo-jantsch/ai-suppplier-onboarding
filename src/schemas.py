"""Data shapes passed between stages.

Every value an LLM proposes carries its source, so validation and the review
page can check it. These models double as the JSON schemas sent to the LLM
for structured output.
"""
from typing import Literal

from pydantic import BaseModel, Field

Value = str | float | bool | list[str] | None


class InputRow(BaseModel):
    """Stage 1 output: one supplier row mapped to the standard input schema."""
    sku: str
    upc: str | None = None
    brand: str | None = None
    name: str
    description: str | None = None
    size: str | None = None
    cost: float | None = None
    msrp: float | None = None
    supplier: str
    raw: dict[str, str] = Field(default_factory=dict, description="Original row, untouched")


class Source(BaseModel):
    kind: Literal["csv", "rule", "example", "competitor"]
    ref: str = Field(description="CSV column name, rule id, example sku, or competitor name")
    url: str | None = None
    quote: str | None = Field(default=None, description="Exact text copied from the CSV cell or page")


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
    metafields: dict[str, FieldValue] = Field(description='Keys like "custom.species"')
    body_html: FieldValue


class Suggestion(BaseModel):
    """Stage 3 output: from competitor pages, needs human approval."""
    field: str
    proposed: FieldValue
    current: Value = None
    agreement: int = Field(description="How many competitors agree on this value")
    reason: str


class ReviewItem(BaseModel):
    """Stage 4 output: what the review page shows for one SKU."""
    draft: Draft
    suggestions: list[Suggestion] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
