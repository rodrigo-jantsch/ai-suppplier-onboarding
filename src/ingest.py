"""Stage 1: read a supplier CSV and turn each row into an InputRow.

No LLM here: this is plain, testable code. The tests in tests/test_ingest.py
describe exactly what each function must do. Work top to bottom:

    python -m pytest -q tests/test_ingest.py

Hints are in each docstring. Useful standard-library pieces: csv.DictReader,
re.fullmatch / re.sub / re.findall.
"""
import csv
import re
from pathlib import Path

from src.config import load
from src.schemas import InputRow


def normalize_header(header: str) -> str:
    """Make a header comparable to the aliases in config/input_schema.yaml.

    Remove a byte-order mark ("\\ufeff") if present, trim, lowercase, and turn
    any run of spaces/underscores into a single space.
        "\\ufeffAuction Title" -> "auction title"
        "  Item_#  "          -> "item #"
    """
    raise NotImplementedError


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of {column: value} dicts.

    Hint: open with encoding="utf-8-sig" and newline="". The "-sig" variant
    removes the invisible byte-order mark at the start of the VPS6 export, so
    the first header is "Auction Title" instead of "\\ufeffAuction Title".
    """
    raise NotImplementedError


def pivot_attributes(row: dict[str, str]) -> dict[str, str]:
    """Turn AttributeNName / AttributeNValue column pairs into one dict.

        {"Attribute1Name": "Animal Type", "Attribute1Value": "Swine"}
        -> {"Animal Type": "Swine"}

    Skip pairs whose Name is empty. Keep empty values (an empty
    "Jeffers Prop 65 Validated" means something to PROP-01).
    Hint: re.fullmatch(r"Attribute(\\d+)Name", key) gives you N.
    """
    raise NotImplementedError


def map_columns(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Match CSV headers to standard fields using the aliases in input_schema.yaml.

    Returns (mapping, unknown):
      mapping: {standard_field: original_header}, e.g. {"sku": "Inventory Number"}
      unknown: headers that matched no alias (report these, don't crash)

    Ignore the AttributeNName / AttributeNValue columns; pivot_attributes
    handles those. Compare normalize_header(header) against normalized aliases.
    If two headers match the same field, keep the first.
    Hint: load("input_schema")["columns"] gives {field: {"aliases": [...]}}.
    """
    raise NotImplementedError


def to_input_row(row: dict[str, str], mapping: dict[str, str], supplier: str) -> InputRow:
    """Build one InputRow from a CSV row.

    - Text fields: row[mapping[field]], stripped; empty string -> None.
    - Numbers (cost, msrp, weight_lb, height, length, width): float, or None
      if empty or not a number. Strip "$" and "," first.
    - attributes: pivot_attributes(row).
    - From attributes, using load("input_schema")["attributes"]:
        size (only if no size column was mapped): first non-empty of the listed names
        country_of_origin, animal_type: first non-empty of the listed names
        bullets: values of "Bullet Point 1", "Bullet Point 2", ... in number order, skipping empties
    - images: every https://... URL inside the images column.
      Hint: re.findall(r"https?://[^,\\s]+", text)
    - raw: the original row (drop the None key csv adds for extra commas).
    """
    raise NotImplementedError


def ingest_file(path: Path) -> tuple[list[InputRow], list[str]]:
    """Read one supplier file and return (input_rows, unknown_headers).

    supplier = the file name (path.name). Map headers once, then convert each row.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Quick manual check:  python -m src.ingest inbox/VPS6.csv
    import sys

    rows, unknown = ingest_file(Path(sys.argv[1]))
    for r in rows:
        print(r.model_dump_json(indent=2, exclude={"raw"}))
    if unknown:
        print("\nUnknown headers (add aliases to config/input_schema.yaml):", unknown)
