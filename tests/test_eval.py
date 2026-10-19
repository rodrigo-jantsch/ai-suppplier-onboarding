"""Tests for the evaluation script. Run:  python -m pytest -q tests/test_eval.py"""
import pytest

from src import eval as ev


@pytest.mark.parametrize("field, expected, got, outcome", [
    ("compliance.prop_65", True, True, "correct"),
    ("compliance.prop_65", True, False, "wrong"),
    ("price", 24.99, 24.99, "correct"),
    ("vendor", "TiaGard", "tiagard ", "correct"),               # case and spaces ignored
    ("custom.sizes", ["1 L"], ["1 L"], "correct"),
    ("custom.product_types", ["Scours", "Oral"], ["Oral", "Scours"], "correct"),   # order ignored
    ("custom.product_types", ["Scours", "Oral"], ["Oral", "Pneumonia"], "partial"),
    ("custom.product_types", ["Scours"], ["Oral Medication"], "wrong"),
    ("product_type", "Animals & Pet Supplies", "Animals & Pet Supplies", "correct"),
    ("product_type", "A > B > C", "A > B", "partial"),             # correct but too general
    ("product_type", "A > B", "A > B > C", "wrong"),               # too specific
    ("title", "TiaGard® 12.5% Liquid", "TiaGard 12.5% Liquid", "correct"),
])
def test_compare(field, expected, got, outcome):
    assert ev.compare(field, expected, got)[0] == outcome


def test_too_general_category_scores_by_depth():
    assert ev.compare("product_type", "A > B > C > D", "A > B")[1] == 0.5


GOLD = {
    "sku": "X1",
    "expected_draft": {
        "title": "Acme Spray", "vendor": "Acme", "product_type": "A > B",
        "metafields": {"custom.department": "Health", "custom.category": "Medicine", "compliance.prop_65": True},
        "variant_metafields": {"custom.hazardous": False},
    },
    "expected_flags": ["RX-01"],
}


def saved(values: dict, flags: list[str]):
    def fv(v):
        return {"value": v}
    return {"draft": {
        "title": fv(values.get("title")), "vendor": fv(values.get("vendor")),
        "product_type": fv(values.get("product_type")),
        "metafields": {k: fv(v) for k, v in values.get("mf", {}).items()},
        "variant_metafields": {k: fv(v) for k, v in values.get("vmf", {}).items()},
    }, "flags": flags}


def test_blank_flagged_vs_silent_missing_vs_silent_error():
    draft = saved({"title": "Acme Spray", "vendor": "Acme", "product_type": "A > B",
                   "mf": {"compliance.prop_65": False}, "vmf": {"custom.hazardous": False}},
                  flags=["custom.department: left blank for review (model not sure)", "RX-01: prescription"])
    results, flag_check = ev.score_sku(GOLD, draft)
    by_field = {r.field: r for r in results}
    assert by_field["custom.department"].outcome == "blank"          # empty + flagged
    assert by_field["custom.category"].outcome == "missing"          # empty, no flag
    assert by_field["compliance.prop_65"].outcome == "wrong"         # wrong, not flagged
    assert by_field["variant:custom.hazardous"].outcome == "correct"
    assert flag_check == {"expected": 1, "fired": 1}

    s = ev.summarize(results)
    assert s["silent_errors"] == 1 and s["blank_flagged"] == 1 and s["silent_missing"] == 1
    assert by_field["custom.department"].kind == "model" and by_field["compliance.prop_65"].kind == "fixed"


def test_handle_is_judged_against_the_drafts_own_title():
    gold = {"sku": "X2", "expected_draft": {"title": "Old Title", "handle": "old-title"}}
    good = {"draft": {"title": {"value": "New Title"}, "handle": {"value": "new-title"}}, "flags": []}
    bad = {"draft": {"title": {"value": "New Title"}, "handle": {"value": "something-else"}}, "flags": []}
    assert {r.field: r.outcome for r in ev.score_sku(gold, good)[0]}["handle"] == "correct"
    assert {r.field: r.outcome for r in ev.score_sku(gold, bad)[0]}["handle"] == "wrong"


def test_image_scores():
    gold = {"sku": "X1", "expected_draft": {"metafields": {"custom.sizes": ["1 L"]}},
            "findable_facts": {"info.ingredients": "12.5% tiamulin hydrogen fumarate"},
            "expected_images": {"MAIN.jpg": "yes", "WRONG.jpg": "no", "BACK.jpg": "yes"}}

    def img(file, verdict, size, ingredients=None):
        lv = lambda v: {"value": v, "label_text": v, "confidence": "high"}   # noqa: E731
        return {"file": file, "verdict": verdict, "reading": {
            "size": lv(size), "ingredients": lv(ingredients), "species": lv(None), "form": lv("Liquid")}}

    report = {"images": [img("MAIN.jpg", "yes", "1 Liter (33.8 fl oz)", "Tiamulin hydrogen fumarate 12.5%"),
                         img("WRONG.jpg", "no", "16 oz"),
                         img("BACK.jpg", "no", "1 L")]}          # a correct image wrongly rejected
    s = ev.score_images(gold, report)
    assert (s["wrong_planted"], s["wrong_caught"], s["false_alarms"]) == (1, 1, 1)
    assert (s["label_values"], s["label_correct"]) == (2, 2)    # form is not scored; rejected images skipped


def test_append_result_adds_new_columns(tmp_path):
    out = tmp_path / "results.csv"
    ev.append_result(out, {"timestamp": "t1", "overall_score": 80, "note": "old"})
    ev.append_result(out, {"timestamp": "t2", "overall_score": 85, "label_precision": 90.0, "note": "new"})
    lines = out.read_text().splitlines()
    assert lines[0] == "timestamp,overall_score,note,label_precision"
    assert lines[1] == "t1,80,old," and lines[2] == "t2,85,new,90.0"
