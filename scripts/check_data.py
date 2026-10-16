"""Check that the knowledge base and gold labels agree with each other.

Run from the repo root:  python -m scripts.check_data

Errors mean something will break or be scored wrong later; warnings are worth
a look. Exits with code 1 when there are errors, so it can run before commits.
"""
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RULE_FIELDS_REQUIRED = ("id", "title", "kind", "trigger", "action")
RULE_KINDS = {"deterministic", "guidance"}
SPECIAL_DRAFT_KEYS = {"variant.weight_lb"}   # not metafields; handled by name
GOLD_DRAFT_REQUIRED = ("title", "product_type", "vendor", "tags", "price", "barcode", "handle", "metafields")

errors: list[str] = []
warnings: list[str] = []


def err(where: str, msg: str) -> None:
    errors.append(f"{where}: {msg}")


def warn(where: str, msg: str) -> None:
    warnings.append(f"{where}: {msg}")


def slugify(title: str) -> str:
    """HANDLE-01: lowercase, runs of non-letters/digits -> one hyphen, trim."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


# ------------------------------------------------------------------ taxonomy

def check_taxonomy() -> set[str]:
    path = ROOT / "config/taxonomy.yaml"
    types = yaml.safe_load(path.read_text(encoding="utf-8"))["types"]
    names = [t.get("name", "") for t in types]
    ids = [t.get("id", "") for t in types]
    for label, values in (("id", ids), ("name", names)):
        dupes = {v for v in values if values.count(v) > 1}
        for d in sorted(dupes):
            err("taxonomy", f"duplicate {label}: {d}")
    for t in types:
        where = f"taxonomy[{t.get('id')}]"
        for field in ("id", "name", "description", "keywords"):
            if not t.get(field):
                err(where, f"missing {field}")
        if "  " in t.get("name", ""):
            err(where, "double space in name")
    name_set = set(names)
    for n in names:
        parts = n.split(" > ")
        for i in range(1, len(parts)):
            parent = " > ".join(parts[:i])
            if parent not in name_set:
                err("taxonomy", f"parent level missing as an entry: {parent}")
    return name_set


# ------------------------------------------------------------------ rules

def parse_rule(path: Path) -> dict[str, str]:
    """Rule files are simple 'key: value' lines."""
    rule = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if m:
            rule[m.group(1)] = m.group(2).strip()
    return rule


def check_rules() -> set[str]:
    rule_ids = set()
    for path in sorted((ROOT / "rules").glob("*.md")):
        rule = parse_rule(path)
        where = f"rules/{path.name}"
        for field in RULE_FIELDS_REQUIRED:
            if not rule.get(field):
                err(where, f"missing '{field}:'")
        if rule.get("id") and rule["id"] != path.stem:
            err(where, f"id '{rule['id']}' doesn't match file name")
        if rule.get("kind") and rule["kind"] not in RULE_KINDS:
            err(where, f"kind must be one of {sorted(RULE_KINDS)}, got '{rule['kind']}'")
        if rule.get("kind") == "deterministic" and "every product" not in rule.get("trigger", "").lower():
            if '"' not in rule.get("trigger", ""):
                warn(where, "deterministic rule doesn't name an exact column or attribute in quotes")
        for field, value in rule.items():
            if value.count('"') % 2:
                warn(where, f"unbalanced quote in '{field}'")
        rule_ids.add(rule.get("id", path.stem))
    return rule_ids


# ------------------------------------------------------------------ metafields

def check_metafields() -> tuple[dict[str, dict], dict[str, dict]]:
    data = yaml.safe_load((ROOT / "config/metafields.yaml").read_text(encoding="utf-8"))
    product, variant = {}, {}

    def add(entries, target, where):
        for e in entries or []:
            key = f"{e.get('namespace')}.{e.get('key')}"
            if not e.get("type"):
                err(where, f"{key} has no type")
            if key in target:
                err(where, f"{key} defined twice")
            target[key] = e

    shared = data.get("shared", {})
    add(shared.get("product"), product, "metafields.shared.product")
    add(shared.get("variant"), variant, "metafields.shared.variant")
    for type_id, entries in (data.get("by_type") or {}).items():
        add(entries, product, f"metafields.by_type.{type_id}")
    return product, variant


def value_matches_type(value, mtype: str) -> bool:
    if mtype.startswith("list."):
        return isinstance(value, list)
    if mtype == "boolean":
        return isinstance(value, bool)
    if mtype.startswith("number_integer"):
        return isinstance(value, int) and not isinstance(value, bool)
    if mtype.startswith("number_decimal"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


# ------------------------------------------------------------------ inbox

def load_inbox_skus() -> dict[str, dict[str, str]]:
    """Map every cell value that looks like a SKU to its row, across all CSVs."""
    rows = {}
    for path in sorted((ROOT / "inbox").glob("*.csv")):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for col in ("Inventory Number", "supplier_sku", "Item #", "sku", "SKU"):
                    if row.get(col):
                        rows[row[col].strip()] = {"_file": path.name, **row}
    return rows


def attributes(row: dict[str, str]) -> dict[str, str]:
    """Turn AttributeNName / AttributeNValue pairs into {name: value}."""
    out = {}
    for k, v in row.items():
        m = re.match(r"Attribute(\d+)Name$", k or "")
        if m and v:
            out[v.strip()] = (row.get(f"Attribute{m.group(1)}Value") or "").strip()
    return out


# ------------------------------------------------------------------ gold

def check_gold(tax_names, rule_ids, product_mf, variant_mf, inbox) -> int:
    count = 0
    for path in sorted((ROOT / "gold").glob("*.json")):
        if path.name.startswith("_"):
            continue
        count += 1
        where = f"gold/{path.name}"
        try:
            gold = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            err(where, f"invalid JSON: {e}")
            continue
        sku = gold.get("sku", "")
        if sku != path.stem:
            err(where, f"sku '{sku}' doesn't match file name")
        draft = gold.get("expected_draft", {})
        for field in GOLD_DRAFT_REQUIRED:
            if field not in draft:
                err(where, f"expected_draft missing '{field}'")

        # category must be a real Shopify entry
        pt = draft.get("product_type")
        if pt is not None and pt not in tax_names:
            err(where, f"product_type not in taxonomy: '{pt}'")

        # handle follows HANDLE-01
        if draft.get("title") and draft.get("handle"):
            expected = slugify(draft["title"])
            if draft["handle"] != expected:
                err(where, f"handle doesn't follow HANDLE-01; expected '{expected}'")

        # tags follow TAG-01
        tags = draft.get("tags", [])
        if "New Arrival" not in tags:
            err(where, "tags missing 'New Arrival' (TAG-01)")
        cat_tags = [t for t in tags if t.lower().replace("-", "").startswith("catalogref")]
        for t in cat_tags:
            if not re.fullmatch(r"CatalogRef: \S+", t):
                err(where, f"tag '{t}' should look like 'CatalogRef: <Parent Item Number>'")

        # metafields defined, with values of the right type
        for section, defs, label in (("metafields", product_mf, "product"),
                                     ("variant_metafields", variant_mf, "variant")):
            for key, value in (draft.get(section) or {}).items():
                if key not in defs:
                    err(where, f"{section}: '{key}' not defined as a {label} metafield in metafields.yaml")
                    continue
                d = defs[key]
                if not value_matches_type(value, d["type"]):
                    err(where, f"{section}: '{key}' value {value!r} doesn't fit type {d['type']}")
                allowed = d.get("allowed")
                if allowed:
                    vals = value if isinstance(value, list) else [value]
                    bad = [v for v in vals if v not in allowed]
                    if bad:
                        err(where, f"{section}: '{key}' values {bad} not in allowed list")

        # PROP-01 consistency inside the gold label
        mf = draft.get("metafields", {})
        if mf.get("compliance.prop_65") is True and not mf.get("shopify.disclosure"):
            err(where, "compliance.prop_65 is true but shopify.disclosure is missing (PROP-01)")

        # findable facts must be defined and not already in the draft
        for key in gold.get("findable_facts", {}):
            if key not in product_mf and key not in variant_mf and key not in SPECIAL_DRAFT_KEYS:
                err(where, f"findable_facts: '{key}' not defined in metafields.yaml")
            if key in mf or key in draft:
                warn(where, f"findable_facts: '{key}' is also in expected_draft")

        # flags point at real rules
        for flag in gold.get("expected_flags", []):
            if flag not in rule_ids:
                err(where, f"expected_flags: no rule file for '{flag}'")

        # the SKU exists in an inbox CSV, and fixed-rule fields agree with it
        row = inbox.get(sku)
        if not row:
            err(where, f"SKU '{sku}' not found in any inbox CSV")
            continue
        if gold.get("upc") and row.get("UPC") and gold["upc"] != row["UPC"].strip():
            err(where, f"upc {gold['upc']} != CSV UPC {row['UPC']}")
        attrs = attributes(row)
        parent = (row.get("Variation Parent SKU") or "").strip()
        if parent and f"CatalogRef: {parent}" not in tags:
            err(where, f"tags should include 'CatalogRef: {parent}' from Variation Parent SKU (TAG-01)")
        yes = lambda name: attrs.get(name, "").upper() in ("Y", "1", "TRUE")  # noqa: E731
        vmf = draft.get("variant_metafields", {})
        for attr, key, rule in (("Jeffers Is Biological", "custom.biological", "BIO-01"),
                                ("Is Dropship", "custom.dropship", "DRO-01"),
                                ("Jeffers Is Hazardous", "custom.hazardous", "HAZ-02")):
            if attr in attrs and key in vmf and vmf[key] != yes(attr):
                err(where, f"variant {key}={vmf[key]} but CSV '{attr}'='{attrs[attr]}' ({rule})")
        if "Is Prescription Required" in attrs and "custom.prescription_rx_" in mf:
            if mf["custom.prescription_rx_"] != yes("Is Prescription Required"):
                err(where, f"custom.prescription_rx_ disagrees with CSV 'Is Prescription Required' (RX-01)")
        if "compliance.prop_65" in mf:
            expected = yes("Jeffers Is Prop 65") if yes("Jeffers Prop 65 Validated") else True
            if mf["compliance.prop_65"] != expected:
                err(where, f"compliance.prop_65 should be {expected} by PROP-01")
    return count


# ------------------------------------------------------------------ main

def main() -> None:
    tax_names = check_taxonomy()
    rule_ids = check_rules()
    product_mf, variant_mf = check_metafields()
    inbox = load_inbox_skus()
    n_gold = check_gold(tax_names, rule_ids, product_mf, variant_mf, inbox)

    print(f"Checked: {len(tax_names)} taxonomy entries, {len(rule_ids)} rules, "
          f"{len(product_mf)} product + {len(variant_mf)} variant metafields, "
          f"{n_gold} gold labels, {len(inbox)} inbox SKUs\n")
    for w in warnings:
        print(f"  WARN   {w}")
    for e in errors:
        print(f"  ERROR  {e}")
    if not errors and not warnings:
        print("  All consistent.")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
