"""Stage 1: read a supplier CSV and turn each row into an InputRow.

No LLM here: this is plain, testable code. Every supplier sends different
column names and layouts; this stage's only job is to translate whatever they
sent into one standard shape (InputRow) without losing anything.

    python -m pytest -q tests/test_ingest.py     # run the tests
    python -m src.ingest inbox/VPS6.csv           # see the result for a file
"""
import csv
import re
from pathlib import Path

from src.config import load
from src.schemas import InputRow

# Fields that should become numbers rather than text.
NUMBER_FIELDS = ("cost", "msrp", "weight_lb", "height", "length", "width")

# Matches "Attribute12Name" and captures "12".
ATTRIBUTE_NAME = re.compile(r"Attribute(\d+)Name")
# Matches either half of an attribute pair, so map_columns can skip them.
ATTRIBUTE_COLUMN = re.compile(r"Attribute\d+(Name|Value)")


def normalize_header(header: str) -> str:
    """Make a header comparable to the aliases in config/input_schema.yaml.

        "\\ufeffAuction Title" -> "auction title"
        "  Item_#  "          -> "item #"
    """
    h = header.replace("﻿", "")      # byte-order mark, if any
    h = h.strip().lower()
    return re.sub(r"[\s_]+", " ", h)      # any run of spaces/underscores -> one space


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of {column: value} dicts.

    "utf-8-sig" removes the invisible byte-order mark some exports (like VPS6)
    put at the very start of the file. newline="" lets the csv module handle
    line breaks inside quoted cells, such as the multi-paragraph Description.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def pivot_attributes(row: dict[str, str]) -> dict[str, str]:
    """Turn AttributeNName / AttributeNValue column pairs into one dict.

        {"Attribute1Name": "Animal Type", "Attribute1Value": "Swine"}
        -> {"Animal Type": "Swine"}

    Pairs with an empty name are skipped. Empty values are kept on purpose:
    an empty "Jeffers Prop 65 Validated" means "not confirmed" to PROP-01.
    """
    attributes = {}
    for column, name in row.items():
        match = ATTRIBUTE_NAME.fullmatch((column or "").strip())
        if not match or not (name or "").strip():
            continue
        number = match.group(1)
        value = row.get(f"Attribute{number}Value") or ""
        attributes[name.strip()] = value.strip()
    return attributes


def map_columns(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Match CSV headers to standard fields using input_schema.yaml aliases.

    Returns (mapping, unknown):
      mapping: {standard_field: original_header}, e.g. {"sku": "Inventory Number"}
      unknown: headers that matched no alias, so you can add them to the config
    """
    columns = load("input_schema")["columns"]

    # Build a lookup from every normalized alias to its standard field:
    # {"inventory number": "sku", "retail price": "msrp", ...}
    alias_to_field = {}
    for field, spec in columns.items():
        for alias in spec["aliases"]:
            alias_to_field[normalize_header(alias)] = field

    mapping, unknown = {}, []
    for header in headers:
        if not header or ATTRIBUTE_COLUMN.fullmatch(header.strip()):
            continue                                  # pivot_attributes handles these
        field = alias_to_field.get(normalize_header(header))
        if field is None:
            unknown.append(header)
        elif field not in mapping:                    # first matching header wins
            mapping[field] = header
    return mapping, unknown


def _to_number(text: str | None) -> float | None:
    """'$1,024.50' -> 1024.5; empty or non-numeric -> None."""
    if not text:
        return None
    try:
        return float(text.replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def _first_attribute(attributes: dict[str, str], names: list[str]) -> str | None:
    """Return the first non-empty value among the listed attribute names."""
    for name in names:
        if attributes.get(name):
            return attributes[name]
    return None


def _bullets(attributes: dict[str, str], prefix: str) -> list[str]:
    """Collect 'Bullet Point 1..N' values in number order, skipping empties."""
    numbered = []
    for name, value in attributes.items():
        if name.startswith(prefix) and value:
            number = int(re.sub(r"\D", "", name) or 0)   # "Bullet Point 10" -> 10
            numbered.append((number, value))
    return [value for _, value in sorted(numbered)]


def to_input_row(row: dict[str, str], mapping: dict[str, str], supplier: str) -> InputRow:
    """Build one InputRow from a CSV row, using the column mapping."""
    attribute_spec = load("input_schema")["attributes"]
    attributes = pivot_attributes(row)

    # Read every mapped column: stripped text, empty string -> None.
    values: dict = {}
    for field, header in mapping.items():
        text = (row.get(header) or "").strip()
        values[field] = text or None

    for field in NUMBER_FIELDS:
        if field in values:
            values[field] = _to_number(values[field])

    # Some facts live in attributes rather than columns (as in VPS6).
    if not values.get("size"):
        values["size"] = _first_attribute(attributes, attribute_spec["size"])
    values["country_of_origin"] = _first_attribute(attributes, attribute_spec["country_of_origin"])
    values["animal_type"] = _first_attribute(attributes, attribute_spec["animal_type"])
    values["bullets"] = _bullets(attributes, attribute_spec["bullets_prefix"])

    # "ITEMIMAGEURL1=https://...jpg,ITEMIMAGEURL2=https://...jpg" -> list of URLs
    values["images"] = re.findall(r"https?://[^,\s]+", values.get("images") or "")

    return InputRow(
        **values,
        supplier=supplier,
        attributes=attributes,
        raw={k: v for k, v in row.items() if k is not None},   # csv uses None for extra cells
    )


def ingest_file(path: Path) -> tuple[list[InputRow], list[str]]:
    """Read one supplier file and return (input_rows, unknown_headers)."""
    rows = read_csv(path)
    if not rows:
        return [], []
    mapping, unknown = map_columns(list(rows[0].keys()))   # headers are the same for every row
    return [to_input_row(row, mapping, path.name) for row in rows], unknown


if __name__ == "__main__":
    # Quick manual check:  python -m src.ingest inbox/VPS6.csv
    import sys

    rows, unknown = ingest_file(Path(sys.argv[1]))
    for r in rows:
        print(r.model_dump_json(indent=2, exclude={"raw"}))
    if unknown:
        print("\nUnknown headers (add aliases to config/input_schema.yaml):", unknown)
