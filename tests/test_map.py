"""Tests for stage 2b (mapping). Run:  python -m pytest -q tests/test_map.py

The fixed rules are checked against gold/VPS6.json. The model is replaced by
a fake that returns a fixed answer, so we can test the guardrails around it
(category must be a candidate, quotes must exist) without calling Gemini.
"""
import json
from pathlib import Path

import chromadb
import pytest

from src import map as mapper
from src.ingest import ingest_file
from src.retrieve import Retriever
from tests.test_retrieve import fake_embed

ROOT = Path(__file__).resolve().parent.parent
GOLD = json.loads((ROOT / "gold/VPS6.json").read_text(encoding="utf-8"))["expected_draft"]


@pytest.fixture(scope="module")
def vps6():
    return ingest_file(ROOT / "inbox/VPS6.csv")[0][0]


@pytest.fixture(scope="module")
def retriever():
    return Retriever(embed_fn=fake_embed, client=chromadb.EphemeralClient())


def fake_model(answer: dict):
    """A stand-in for Gemini that always returns `answer`."""
    def model_fn(prompt, schema, system):
        model_fn.prompt = prompt
        return schema.model_validate(answer)
    return model_fn


def P(value, quote=None, confidence="high"):
    return {"value": value, "quote": quote, "confidence": confidence}


GOOD_ANSWER = {
    "title": P(GOLD["title"], "TiaGard® 12.5%"),
    "product_type": P("Animals & Pet Supplies", "Swine"),
    "body_html": P("<p>Fast-acting swine treatment.</p>", "fast-acting solution"),
    "ingredients": P("12.5% tiamulin hydrogen fumarate", "Tiamulin Hydrogen Fumarate"),
    "form": P(["Solution", "Drinking Water"], "administration in drinking water"),
    "department": P("Animal Health & Wellness", None, "medium"),
    "category": P("Medicine", None, "medium"),
    "product_types": P(["Pneumonia", "Diarrhea"], "swine pneumonia"),
}


# ---- helpers ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("16OZ", "16 oz"), ("1 liter", "1 L"), ("13.6KG", "30 lb"), ("60CT", "60 count"),
    ("6 CT", "6 count"), ("7LB", "7 lb"), ("500 ml", "500 ml"), (None, None),
])
def test_normalize_size(raw, expected):
    assert mapper.normalize_size(raw) == expected


def test_slugify_matches_gold_handle():
    assert mapper.slugify(GOLD["title"]) == GOLD["handle"]


# ---- fixed rules vs the gold label -----------------------------------------

def test_fixed_rules_match_gold(vps6):
    fields, flags = mapper.apply_fixed_rules(vps6)
    mf, vmf = fields["metafields"], fields["variant_metafields"]
    assert fields["vendor"].value == GOLD["vendor"]
    assert fields["price"].value == GOLD["price"]
    assert fields["barcode"].value == GOLD["barcode"]
    assert fields["weight_lb"].value == GOLD["variant.weight_lb"]
    assert fields["tags"].value == GOLD["tags"]
    for key in ("custom.animal", "custom.sizes", "custom.prescription_rx_", "custom.vendor", "custom.is_merck",
                "custom.pgc", "compliance.prop_65", "custom.country_of_origin",
                "compliance.california_signature_required", "shopify.disclosure"):
        assert mf[key].value == GOLD["metafields"][key], key
    for key, value in GOLD["variant_metafields"].items():
        assert vmf[key].value == value, key
    assert flags == []                                   # VPS6 triggers no review rule


def test_every_fixed_value_has_a_source(vps6):
    fields, _ = mapper.apply_fixed_rules(vps6)
    for key, v in {**fields["metafields"], **fields["variant_metafields"]}.items():
        assert v.sources, key


@pytest.mark.parametrize("sku, expected", [("012345", True), ("0123-A", True), ("VPS6", False), ("A012", False)])
def test_merck_rule_uses_sku_prefix(vps6, sku, expected):
    fields, _ = mapper.apply_fixed_rules(vps6.model_copy(update={"sku": sku}))
    assert fields["metafields"]["custom.is_merck"].value is expected


def test_rx_rule_adds_tag_and_flag(vps6):
    rx_row = vps6.model_copy(update={"attributes": {**vps6.attributes, "Is Prescription Required": "Y"}})
    fields, flags = mapper.apply_fixed_rules(rx_row)
    assert fields["metafields"]["custom.prescription_rx_"].value is True
    assert "GMC: RX" in fields["tags"].value
    assert any(f.startswith("RX-01") for f in flags)


# ---- the whole map step, with a fake model ---------------------------------

def test_map_row_builds_a_draft(vps6, retriever):
    model = fake_model(GOOD_ANSWER)
    draft, flags = mapper.map_row(vps6, retriever, model_fn=model)
    assert draft.title.value == GOLD["title"]
    assert draft.handle.value == GOLD["handle"]
    assert draft.product_type.value == "Animals & Pet Supplies"
    assert draft.metafields["info.form"].value == ["Solution", "Drinking Water"]
    assert flags == []
    # the prompt carries the candidates and only the guidance rules
    assert "Animals & Pet Supplies" in model.prompt
    assert "TITLE-01" in model.prompt and "TAX-01" in model.prompt
    assert "PROP-01" not in model.prompt


def test_category_outside_candidates_is_replaced_and_flagged(vps6, retriever):
    answer = {**GOOD_ANSWER, "product_type": P("Animals & Pet Supplies > Livestock", "Swine")}
    draft, flags = mapper.map_row(vps6, retriever, model_fn=fake_model(answer))
    assert draft.product_type.value == "Animals & Pet Supplies"
    assert draft.product_type.confidence == "low"
    assert any(f.startswith("product_type") for f in flags)


def test_invented_quote_lowers_confidence(vps6, retriever):
    answer = {**GOOD_ANSWER, "ingredients": P("50% amoxicillin", "Contains 50% amoxicillin")}
    draft, flags = mapper.map_row(vps6, retriever, model_fn=fake_model(answer))
    assert draft.metafields["info.ingredients"].confidence == "low"
    assert any(f.startswith("info.ingredients") for f in flags)


def test_model_cannot_change_fixed_rules(vps6, retriever):
    draft, _ = mapper.map_row(vps6, retriever, model_fn=fake_model(GOOD_ANSWER))
    assert draft.metafields["compliance.prop_65"].value is True
    assert draft.tags.value == GOLD["tags"]
