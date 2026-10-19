"""Score saved drafts against the gold labels.

    python -m src.map inbox/VPS6.csv          # make drafts (calls Gemini)
    python -m src.images inbox/VPS6.csv       # optional: check product images
    python -m src.eval --note "baseline"      # score them (no API calls)

For every gold label in gold/, finds output/drafts/<sku>.json and compares
field by field. Each field gets one outcome:

    correct   matches the gold value (after normalizing case, spaces, ®, order)
    partial   close: category too general, list partly right, similar title
    blank     left empty and flagged for review (a safe miss)
    missing   left empty with NO flag (a silent miss)
    wrong     a value that doesn't match the gold value

The number that matters most for trust is "silent errors": wrong values that
were not flagged. Those are what would reach the store without anyone looking.

If the image check has run (output/images/<sku>.json), two more numbers:

    wrong images caught   images gold marks "no" that the check flagged
    label-read precision  values read from labels that agree with gold

Each run appends one line to results.csv, so you can compare runs over time.
"""
import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

from src.config import ROOT, settings
from src.map import slugify

# Fields the model decides (everything else comes from fixed rules in code).
MODEL_FIELDS = {"title", "product_type", "body_html", "info.ingredients", "info.form",
                "custom.department", "custom.category", "custom.subcategory", "custom.product_types"}
TITLE_CORRECT_AT = 90        # token similarity (0-100) that counts a title as correct


@dataclass
class FieldResult:
    sku: str
    field: str
    kind: str          # "fixed" or "model"
    outcome: str       # correct / partial / blank / missing / wrong
    score: float       # 1.0 correct, 0..1 partial, 0 otherwise
    expected: object
    got: object
    flagged: bool


# ------------------------------------------------------------------ normalizing

def norm_text(value) -> str:
    text = str(value).replace("®", "").replace("™", "").casefold()
    return re.sub(r"\s+", " ", text).strip()


def is_empty(value) -> bool:
    return value is None or value == "" or value == []


def compare(field: str, expected, got) -> tuple[str, float]:
    """Return (outcome, score) for one non-empty value."""
    if isinstance(expected, bool) or isinstance(got, bool):
        return ("correct", 1.0) if expected == got else ("wrong", 0.0)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return ("correct", 1.0) if abs(float(got) - float(expected)) < 1e-6 else ("wrong", 0.0)
        except (TypeError, ValueError):
            return "wrong", 0.0
    if field == "product_type":
        e, g = norm_text(expected), norm_text(got)
        if e == g:
            return "correct", 1.0
        if e.startswith(g + " > "):                     # a correct but more general level
            depth_e, depth_g = e.count(" > ") + 1, g.count(" > ") + 1
            return "partial", round(depth_g / depth_e, 2)
        return "wrong", 0.0
    if field == "title":
        sim = fuzz.token_set_ratio(norm_text(expected), norm_text(got))
        return ("correct", 1.0) if sim >= TITLE_CORRECT_AT else ("partial", round(sim / 100, 2))
    if isinstance(expected, list) or isinstance(got, list):
        e = {norm_text(x) for x in (expected if isinstance(expected, list) else [expected])}
        g = {norm_text(x) for x in (got if isinstance(got, list) else [got])}
        if e == g:
            return "correct", 1.0
        overlap = len(e & g) / len(e | g)
        return ("partial", round(overlap, 2)) if overlap else ("wrong", 0.0)
    return ("correct", 1.0) if norm_text(expected) == norm_text(got) else ("wrong", 0.0)


# ------------------------------------------------------------------ scoring

def draft_values(saved: dict) -> dict:
    """Flatten a saved draft into {field: value} using the gold label's names."""
    d = saved["draft"]
    out = {}
    for key in ("title", "product_type", "vendor", "tags", "price", "barcode", "handle", "body_html"):
        if d.get(key):
            out[key] = d[key]["value"]
    if d.get("weight_lb"):
        out["variant.weight_lb"] = d["weight_lb"]["value"]
    for key, fv in (d.get("metafields") or {}).items():
        out[key] = fv["value"]
    for key, fv in (d.get("variant_metafields") or {}).items():
        out[f"variant:{key}"] = fv["value"]
    return out


def gold_values(gold: dict) -> dict:
    g = gold["expected_draft"]
    out = {k: v for k, v in g.items() if k not in ("metafields", "variant_metafields")}
    out.update(g.get("metafields", {}))
    out.update({f"variant:{k}": v for k, v in g.get("variant_metafields", {}).items()})
    return out


def score_sku(gold: dict, saved: dict) -> tuple[list[FieldResult], dict]:
    sku = gold["sku"]
    expected_all, got_all = gold_values(gold), draft_values(saved)
    flags = saved.get("flags", [])
    results = []
    for field, expected in expected_all.items():
        if is_empty(expected):
            continue
        name = field.removeprefix("variant:")
        kind = "model" if name in MODEL_FIELDS else "fixed"
        flagged = any(f.startswith(name + ":") for f in flags)
        got = got_all.get(field)
        if field == "handle" and isinstance(got_all.get("title"), str):
            # The handle is computed from the draft's own title (HANDLE-01), so judge
            # it against that title; title differences are already scored on "title".
            expected = slugify(got_all["title"])
        if is_empty(got):
            outcome, score = ("blank", 0.0) if flagged else ("missing", 0.0)
        else:
            outcome, score = compare(name, expected, got)
        results.append(FieldResult(sku, field, kind, outcome, score, expected, got, flagged))

    # Did every expected review flag fire?
    expected_flags = gold.get("expected_flags", [])
    fired = [rule for rule in expected_flags if any(f.startswith(rule) for f in flags)]
    flag_check = {"expected": len(expected_flags), "fired": len(fired)}
    return results, flag_check


def summarize(results: list[FieldResult]) -> dict:
    def pct(rows, outcome=None):
        if not rows:
            return None
        if outcome:
            return round(100 * sum(r.outcome == outcome for r in rows) / len(rows), 1)
        return round(100 * sum(r.score for r in rows) / len(rows), 1)

    fixed = [r for r in results if r.kind == "fixed"]
    model = [r for r in results if r.kind == "model"]
    return {
        "fields": len(results),
        "fixed_score": pct(fixed),
        "model_score": pct(model),
        "overall_score": pct(results),
        "blank_flagged": sum(r.outcome == "blank" for r in results),
        "silent_missing": sum(r.outcome == "missing" for r in results),
        "silent_errors": sum(r.outcome == "wrong" and not r.flagged for r in results),
        "flagged_errors": sum(r.outcome == "wrong" and r.flagged for r in results),
    }


# ------------------------------------------------------------------ image check

def score_images(gold: dict, report: dict) -> dict:
    """Compare an image-check report (output/images/<sku>.json) with the gold label.

    gold["expected_images"] = {"VPS6-MAIN.jpg": "yes", "OTHER.jpg": "no"} says
    which images really show the product. Label values are judged the same
    way the image check compares them, on the fields it compares.
    """
    from src.images import FIELDS, agrees

    expected = gold.get("expected_images", {})
    truth = {**gold["expected_draft"].get("metafields", {}), **gold.get("findable_facts", {})}
    out = {"images": 0, "wrong_planted": 0, "wrong_caught": 0, "false_alarms": 0, "label_values": 0, "label_correct": 0}
    for img in report.get("images", []):
        if "error" in img:
            continue
        out["images"] += 1
        verdict, should_be = img["verdict"], expected.get(img["file"])
        if should_be == "no":
            out["wrong_planted"] += 1
            out["wrong_caught"] += verdict == "no"
        elif should_be == "yes" and verdict == "no":
            out["false_alarms"] += 1
        if verdict != "yes":
            continue
        for field, (key, mode) in FIELDS.items():
            v = img["reading"][field]
            if mode != "compare" or v["value"] in (None, "", []) or v["confidence"] == "low" or key not in truth:
                continue
            out["label_values"] += 1
            out["label_correct"] += agrees(key, truth[key], v["value"])
    return out


def append_result(path: Path, row: dict) -> None:
    """Append one run to results.csv, adding any new columns to the header."""
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    columns = list(rows[0].keys()) if rows else []
    columns += [c for c in row if c not in columns]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows + [row])


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description="Score drafts against gold labels")
    parser.add_argument("--note", default="", help="what changed in this run, saved to results.csv")
    parser.add_argument("--all", action="store_true", help="print every field, not just the imperfect ones")
    args = parser.parse_args()

    all_results, flag_checks, scored, skipped, image_scores = [], [], [], [], []
    for gold_path in sorted((ROOT / "gold").glob("*.json")):
        if gold_path.name.startswith("_"):
            continue
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        draft_path = ROOT / "output" / "drafts" / f"{gold['sku']}.json"
        if not draft_path.exists():
            skipped.append(gold["sku"])
            continue
        results, flag_check = score_sku(gold, json.loads(draft_path.read_text(encoding="utf-8")))
        image_path = ROOT / "output" / "images" / f"{gold['sku']}.json"
        if image_path.exists():
            image_scores.append(score_images(gold, json.loads(image_path.read_text(encoding="utf-8"))))
        all_results += results
        flag_checks.append(flag_check)
        scored.append(gold["sku"])

    if not scored:
        print("No drafts to score. Run:  python -m src.map inbox/<file>.csv")
        return

    # Field table: imperfect fields first, they're the interesting ones
    marks = {"correct": "✓", "partial": "~", "blank": "○", "missing": "!", "wrong": "✗"}
    print(f"\nScored {len(scored)} SKU(s): {', '.join(scored)}"
          + (f"   (no draft yet: {', '.join(skipped)})" if skipped else ""))
    for r in sorted(all_results, key=lambda r: (r.sku, r.outcome == "correct", r.kind, r.field)):
        if r.outcome == "correct" and not args.all:
            continue
        flag = " [flagged]" if r.flagged else ""
        print(f"  {marks[r.outcome]} {r.sku:<8} {r.kind:<5} {r.field:<38} {r.outcome:<7} {r.score:<4}{flag}")
        print(f"        expected: {str(r.expected)[:90]}")
        print(f"        got:      {str(r.got)[:90]}")

    s = summarize(all_results)
    exp = sum(c["expected"] for c in flag_checks)
    fired = sum(c["fired"] for c in flag_checks)
    print(f"""
Summary
  fixed-rule fields   {s['fixed_score']}%
  model fields        {s['model_score']}%
  overall             {s['overall_score']}%   ({s['fields']} fields)
  blank, flagged      {s['blank_flagged']}   (safe: a reviewer fills these)
  missing, unflagged  {s['silent_missing']}
  wrong, flagged      {s['flagged_errors']}
  wrong, NOT flagged  {s['silent_errors']}   <- the number to drive to zero
  review flags fired  {fired}/{exp} expected""")

    row = {"timestamp": datetime.now().isoformat(timespec="minutes"), "model": settings()["llm"]["chat_model"],
           "skus": len(scored), "fields": s["fields"], "fixed_score": s["fixed_score"],
           "model_score": s["model_score"], "overall_score": s["overall_score"],
           "blank_flagged": s["blank_flagged"], "silent_missing": s["silent_missing"],
           "flagged_errors": s["flagged_errors"], "silent_errors": s["silent_errors"]}
    if image_scores:
        t = {k: sum(x[k] for x in image_scores) for k in image_scores[0]}
        precision = round(100 * t["label_correct"] / t["label_values"], 1) if t["label_values"] else None
        caught = f"{t['wrong_caught']}/{t['wrong_planted']}"
        print(f"""
Image check ({len(image_scores)} SKU(s), {t['images']} images)
  wrong images caught   {caught}   (planted wrong images flagged)
  false alarms          {t['false_alarms']}   (correct images called wrong)
  label-read precision  {precision}%   ({t['label_correct']}/{t['label_values']} values agree with gold)""")
        row.update({"images": t["images"], "wrong_images_caught": caught,
                    "image_false_alarms": t["false_alarms"], "label_precision": precision})
    row["note"] = args.note

    out = ROOT / "results.csv"
    append_result(out, row)
    print(f"\nAppended to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
