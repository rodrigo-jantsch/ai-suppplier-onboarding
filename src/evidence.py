"""Compare outside evidence (product images, competitor pages) with a draft.

Shared by src/images.py and src/enrich/, so both follow the same rules:

    evidence agrees with the draft     -> added as another source (confidence up)
    evidence adds missing detail       -> suggestion to upgrade the draft value
    evidence disagrees                 -> flag for review; the draft value stays
    draft field is empty               -> suggestion for the review page

Evidence never overwrites the draft. Only a person accepting a suggestion does.
"""
import re
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from src.map import normalize_size
from src.schemas import Draft, FieldValue, Source, Suggestion

# Draft metafields filled from outside evidence, and how they're compared.
#   compare: a value on both sides is checked; disagreement is flagged
#   suggest: only used to fill an empty field (outside wording, like
#            "Liquid Concentrate", rarely matches the store's filter terms)
MODES = {
    "custom.sizes": "compare",
    "info.ingredients": "compare",
    "custom.animal": "compare",
    "info.form": "suggest",
}
LIST_FIELDS = {"custom.sizes", "custom.animal", "info.form"}


@dataclass
class Evidence:
    key: str                     # draft metafield, e.g. "custom.sizes"
    value: str | list[str]
    quote: str | None            # label text or page text it was read from
    confidence: Literal["high", "medium", "low"]
    source: Source               # kind "image" or "competitor"

    @property
    def where(self) -> str:      # "image VPS6-MAIN.jpg", "competitor valleyvet"
        return f"{self.source.kind} {self.source.ref}"


# ====================================================================== text helpers

def as_list(value) -> list[str]:
    if value in (None, "", []):
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def norm(text) -> str:
    return re.sub(r"\s+", " ", " ".join(as_list(text)).replace("®", "").replace("™", "").casefold()).strip()


def numbers(text) -> set[str]:
    """Every number in a value: '12.5% tiamulin' -> {'12.5'}; ['1.0 L'] -> {'1'}."""
    return {f"{float(n):g}" for n in re.findall(r"\d+(?:\.\d+)?", " ".join(as_list(text)))}


WORD_NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "eight": "8", "ten": "10", "twelve": "12", "half": "0.5"}
AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|oz|ml|l|liters?|litres?|gal(?:lons?)?|lbs?|kg|ct|count)\b", re.I)


def size_variants(text) -> set[str]:
    """Every amount printed, normalized: 'One Liter (33.8 fl. oz; 1000 mL)' -> {'1 L', '33.8 oz', '1000 ml'}.
    1000 ml and up also counts as liters, so '1000 mL' matches a draft that says '1 L'."""
    text = " ".join(as_list(text))
    for word, digit in WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.I)
    found = {normalize_size(text.strip())} if text.strip() else set()
    for number, unit in AMOUNT.findall(text):
        size = normalize_size(f"{number} {unit}")
        found.add(size)
        if size and size.endswith(" ml") and float(number) >= 1000:
            found.add(f"{float(number) / 1000:g} L")
    return found


# ====================================================================== comparisons

def agrees(key: str, draft_value, value, quote: str | None = None) -> bool:
    if key == "custom.sizes":
        printed = size_variants(value) | size_variants(quote)
        return {normalize_size(v) for v in as_list(draft_value)} <= printed
    if key == "info.ingredients":
        same_numbers = numbers(draft_value) == numbers(value)
        return same_numbers and fuzz.token_set_ratio(norm(draft_value), norm(value)) >= 80
    # lists of names (species): agree if they share any entry
    return bool({norm(v) for v in as_list(draft_value)} & {norm(v) for v in as_list(value)})


def adds_detail(key: str, draft_value, value) -> bool:
    """Same thing plus a strength the draft is missing:
    draft 'Tiamulin Hydrogen Fumarate', evidence '12.5% tiamulin hydrogen fumarate'."""
    if key != "info.ingredients" or numbers(draft_value) or not numbers(value):
        return False
    return fuzz.token_set_ratio(norm(draft_value), norm(value)) >= 80


def numbers_in_quote(value, quote) -> bool:
    """Every number in the value must appear in the text it was quoted from."""
    return numbers(value) <= numbers(quote)


# ====================================================================== apply to a draft

REASONS = {
    "image": ("Read from the supplier's image ({refs})", "The label ({refs}) adds detail the draft is missing"),
    "competitor": ("Found on competitor pages ({refs})", "Competitor pages ({refs}) add detail the draft is missing"),
}


def apply_evidence(draft: Draft, evidence: list[Evidence], tag: str) -> tuple[list[Suggestion], list[str]]:
    """Compare evidence with the draft. Updates draft sources in place.

    Returns (suggestions, flags). Flags start with the field name (so eval
    sees the field as flagged) and end with `tag` (so a re-run can replace them).
    """
    flags: list[str] = []
    found: dict[str, dict[tuple, list[Evidence]]] = {}   # key -> normalized value -> evidence
    current_of: dict[str, object] = {}                    # key -> draft value an upgrade would replace

    for ev in evidence:
        mode = MODES.get(ev.key, "suggest")
        current = draft.metafields.get(ev.key)
        if current is not None and current.value not in (None, "", []):
            if mode != "compare":
                continue
            if agrees(ev.key, current.value, ev.value, ev.quote):
                current.sources.append(ev.source)
                if current.confidence == "medium":
                    current.confidence = "high"            # two independent sources agree
                elif current.confidence == "low":
                    current.confidence = "medium"
                continue
            if not adds_detail(ev.key, current.value, ev.value):
                flags.append(f"{ev.key}: {ev.where} says '{ev.value}', draft says '{current.value}' {tag}")
                continue
            current_of[ev.key] = current.value
        normalized = tuple(sorted(norm(v) for v in as_list(ev.value)))
        found.setdefault(ev.key, {}).setdefault(normalized, []).append(ev)

    suggestions = []
    for key, by_value in found.items():
        if len(by_value) > 1:
            flags.append(f"{key}: sources disagree on this value; see suggestions {tag}")
        for group in by_value.values():
            first = group[0]
            value = first.value
            if key in LIST_FIELDS and isinstance(value, str):
                value = [value]
            elif key not in LIST_FIELDS and isinstance(value, list):
                value = ", ".join(value)
            refs = ", ".join(dict.fromkeys(e.source.ref for e in group))
            fill, upgrade = REASONS.get(first.source.kind, REASONS["competitor"])
            reason = (upgrade if key in current_of else fill + "; the draft has no value").format(refs=refs)
            suggestions.append(Suggestion(
                field=key,
                proposed=FieldValue(value=value, sources=[e.source for e in group],
                                    confidence=max((e.confidence for e in group), key=["low", "medium", "high"].index)),
                current=current_of.get(key),
                agreement=len({e.source.ref for e in group}),
                reason=reason,
            ))
    return suggestions, flags


def remove_previous(draft: Draft, flags: list[str], kind: str, tag: str) -> None:
    """Undo an earlier run of a step, so re-running it doesn't duplicate anything."""
    flags[:] = [f for f in flags if not f.endswith(tag)]
    for fv in draft.metafields.values():
        fv.sources = [s for s in fv.sources if s.kind != kind]
