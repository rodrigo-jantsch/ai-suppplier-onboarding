"""Tests for stage 1 (ingest). Run from the repo root:  python -m pytest -q

These describe what ingest.py must do with the real VPS6 export and the two
sample supplier files. Make them pass one at a time, top to bottom.
"""
from pathlib import Path

import pytest

from src import ingest

INBOX = Path(__file__).resolve().parent.parent / "inbox"


# ---- helpers ---------------------------------------------------------------

def test_normalize_header_strips_bom_case_and_spacing():
    assert ingest.normalize_header("﻿Auction Title") == "auction title"
    assert ingest.normalize_header("  Item_#  ") == "item #"
    assert ingest.normalize_header("Net   Wt") == "net wt"


def test_read_csv_removes_bom_from_first_header():
    rows = ingest.read_csv(INBOX / "VPS6.csv")
    assert len(rows) == 1
    first_header = list(rows[0].keys())[0]
    assert first_header == "Auction Title"          # not "﻿Auction Title"


def test_pivot_attributes_turns_pairs_into_a_dict():
    row = {"Attribute1Name": "Animal Type", "Attribute1Value": "Swine",
           "Attribute2Name": "Is Dropship", "Attribute2Value": "N",
           "Attribute3Name": "", "Attribute3Value": ""}
    assert ingest.pivot_attributes(row) == {"Animal Type": "Swine", "Is Dropship": "N"}


def test_map_columns_keeps_brand_and_manufacturer_separate():
    mapping, unknown = ingest.map_columns(["Inventory Number", "Brand", "Manufacturer", "Retail Price", "Foo Bar"])
    assert mapping["sku"] == "Inventory Number"
    assert mapping["brand"] == "Brand"
    assert mapping["manufacturer"] == "Manufacturer"
    assert mapping["msrp"] == "Retail Price"
    assert unknown == ["Foo Bar"]


def test_map_columns_ignores_attribute_columns():
    mapping, unknown = ingest.map_columns(["Inventory Number", "Attribute1Name", "Attribute1Value"])
    assert unknown == []


# ---- the real export: VPS6 -------------------------------------------------

@pytest.fixture(scope="module")
def vps6():
    rows, _unknown = ingest.ingest_file(INBOX / "VPS6.csv")
    assert len(rows) == 1
    return rows[0]


def test_vps6_core_fields(vps6):
    assert vps6.sku == "VPS6"
    assert vps6.upc == "351072092029"
    assert vps6.name == "TiaGard 1-Liter F-Style w/View-Strip"
    assert vps6.brand == "TiaGard"
    assert vps6.manufacturer == "Aurora Pharmaceutical"
    assert vps6.msrp == 24.99
    assert vps6.supplier == "VPS6.csv"


def test_vps6_numbers_and_parent(vps6):
    assert vps6.weight_lb == 2.4
    assert (vps6.height, vps6.length, vps6.width) == (9.0, 4.5, 2.5)
    assert vps6.parent_sku == "VPS5"


def test_vps6_attributes_for_fixed_rules(vps6):
    a = vps6.attributes
    assert a["Animal Type"] == "Swine"
    assert a["Jeffers Is Biological"] == "N"
    assert a["Is Dropship"] == "N"
    assert a["Is Prescription Required"] == "N"
    assert a["Jeffers Is Prop 65"] == "0"
    assert a["Jeffers Prop 65 Validated"] == ""


def test_vps6_promoted_attributes(vps6):
    assert vps6.size == "1 liter"
    assert vps6.country_of_origin == "United States"
    assert vps6.animal_type == "Swine"
    assert len(vps6.bullets) == 5
    assert vps6.bullets[0].startswith("Five-day dosing")


def test_vps6_images_are_urls(vps6):
    assert len(vps6.images) == 2
    assert all(u.startswith("https://") for u in vps6.images)
    assert vps6.images[0].endswith("VPS6-MAIN.jpg")


# ---- other supplier formats ------------------------------------------------

def test_supplier_b_unknown_headers_are_reported():
    rows, unknown = ingest.ingest_file(INBOX / "supplier_b.csv")
    assert len(rows) == 3
    assert rows[0].sku == "B-9001"
    assert rows[0].manufacturer == "Northfield Labs"
    assert rows[0].size == "13.6KG"            # normalizing units is a later step
    assert rows[0].msrp == 64.99
    assert unknown == []                        # every supplier_b header has an alias
