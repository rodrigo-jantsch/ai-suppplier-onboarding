"""Tests for stage 3a (image check). Run:  python -m pytest -q tests/test_images.py

No network and no Gemini: images are generated in memory and the vision
model is replaced by a fake that returns a fixed reading, so we test the
checks around the model.
"""
import chromadb
import pytest

from src import images as im
from src import map as mapper
from src.ingest import ingest_file
from src.llm import gemini_input
from src.retrieve import Retriever
from tests.test_map import GOOD_ANSWER, ROOT, fake_model
from tests.test_retrieve import fake_embed


def jpeg() -> bytes:
    return b"\xff\xd8\xff\xe0 not a real picture; the fake model never looks"


def fake_fetch(url, folder):
    return jpeg(), "image/jpeg"


def L(value, label_text=None, confidence="high"):
    return {"value": value, "label_text": label_text, "confidence": confidence}


GOOD_READING = {
    "shows_this_product": "yes",
    "reason": "Label reads TiaGard 12.5% Liquid Concentrate, 1 Liter",
    "brand": L("TiaGard", "TiaGard®"),
    "size": L("1 L", "1 Liter (33.8 fl oz)"),
    "ingredients": L("12.5% tiamulin hydrogen fumarate", "Tiamulin hydrogen fumarate 12.5%"),
    "strength": L(None, None, "low"),
    "species": L(["Swine"], "For use in swine drinking water"),
    "form": L(["Liquid Concentrate"], "Liquid Concentrate"),
}


def fake_reader(*readings: dict):
    """Returns the given readings in order, one per image (the last one repeats)."""
    calls = []

    def read_fn(prompt, images, system):
        calls.append((prompt, images))
        return im.ImageReading.model_validate(readings[min(len(calls), len(readings)) - 1])
    read_fn.calls = calls
    return read_fn


@pytest.fixture(scope="module")
def vps6():
    return ingest_file(ROOT / "inbox/VPS6.csv")[0][0]


@pytest.fixture
def draft(vps6):
    retriever = Retriever(embed_fn=fake_embed, client=chromadb.EphemeralClient())
    d, flags = mapper.map_row(vps6, retriever, model_fn=fake_model(GOOD_ANSWER))
    return d, flags


def run(vps6, draft, *readings):
    d, flags = draft
    reader = fake_reader(*readings)
    report = im.check_images(vps6, d, flags, read_fn=reader, fetch_fn=fake_fetch)
    return d, flags, report, reader


# ---- helpers ----------------------------------------------------------------

def test_image_name():
    assert im.image_name("https://cdn.example.com/hres_imgs/VPS6-MAIN.jpg?v=2") == "VPS6-MAIN.jpg"


@pytest.mark.parametrize("draft_value, image_value, expected", [
    (["1 L"], "1 Liter (33.8 fl oz)", True),
    (["1 L"], "1 gal", False),
    (["1 L"], "One Liter", True),                                  # the real VPS6 label
    (["1 L"], "1000 mL", True),
    (["16 oz"], "16 fl. oz", True),
    ("12.5% tiamulin hydrogen fumarate", "Tiamulin Hydrogen Fumarate 12.5%", True),
    ("12.5% tiamulin hydrogen fumarate", "1.25% tiamulin hydrogen fumarate", False),   # misread strength
])
def test_agrees(draft_value, image_value, expected):
    key = "custom.sizes" if isinstance(draft_value, list) else "info.ingredients"
    assert im.agrees(key, draft_value, image_value) is expected


def test_gemini_input_adds_image_blocks():
    assert gemini_input("hi", None) == "hi"
    blocks = gemini_input("hi", [(b"\xff\xd8", "image/jpeg")])
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1]["type"] == "image" and blocks[1]["mime_type"] == "image/jpeg"


def test_fetch_downloads_once(tmp_path, monkeypatch):
    import httpx
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, content=jpeg(), request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake_get)
    url = "https://cdn.example.com/x/VPS6-MAIN.jpg"
    first = im.fetch_image(url, tmp_path)
    second = im.fetch_image(url, tmp_path)
    assert first == second and first[1] == "image/jpeg"
    assert len(calls) == 1                                   # second call read the cached file


# ---- the check --------------------------------------------------------------

def test_agreeing_image_adds_a_second_source(vps6, draft):
    d, flags, report, reader = run(vps6, draft, GOOD_READING)
    assert len(reader.calls) == 2                            # MAIN and BACK images
    sizes = d.metafields["custom.sizes"]
    assert sizes.value == ["1 L"]                            # value unchanged
    assert [s.ref for s in sizes.sources if s.kind == "image"] == ["VPS6-MAIN.jpg", "VPS6-BACK.jpg"]
    assert report["flags"] == [] and report["suggestions"] == []


def test_prompt_does_not_reveal_the_draft_values(vps6, draft):
    _, _, _, reader = run(vps6, draft, GOOD_READING)
    prompt = reader.calls[0][0]
    assert "TiaGard" in prompt
    assert "tiamulin" not in prompt.lower() and "12.5" not in prompt


def test_disagreement_is_flagged_and_csv_value_kept(vps6, draft):
    reading = {**GOOD_READING, "size": L("1 gal", "1 gal")}
    d, flags, report, _ = run(vps6, draft, reading)
    assert d.metafields["custom.sizes"].value == ["1 L"]
    assert any(f.startswith("custom.sizes: image VPS6-MAIN.jpg says '1 gal'") for f in flags)


def test_misread_strength_is_flagged(vps6, draft):
    reading = {**GOOD_READING, "ingredients": L("1.25% tiamulin hydrogen fumarate", "tiamulin hydrogen fumarate 1.25%")}
    d, flags, _, _ = run(vps6, draft, reading)
    assert d.metafields["info.ingredients"].value == "12.5% tiamulin hydrogen fumarate"
    assert any(f.startswith("info.ingredients: image") for f in flags)


def test_empty_field_becomes_a_suggestion(vps6, draft):
    d, flags = draft
    del d.metafields["info.ingredients"]
    _, _, report, _ = run(vps6, (d, flags), GOOD_READING)
    [s] = [s for s in report["suggestions"] if s["field"] == "info.ingredients"]
    assert s["proposed"]["value"] == "12.5% tiamulin hydrogen fumarate"
    assert s["agreement"] == 2                               # both images read the same value
    assert s["proposed"]["sources"][0]["kind"] == "image"
    assert "info.ingredients" not in d.metafields            # a suggestion, not written into the draft


def test_wrong_product_image_is_flagged_and_ignored(vps6, draft):
    wrong = {**GOOD_READING, "shows_this_product": "no", "reason": "Shows a dog shampoo bottle",
             "size": L("16 oz", "16 fl oz")}
    d, flags, report, _ = run(vps6, draft, wrong)
    assert any(f.startswith("image: VPS6-MAIN.jpg does not show this product") for f in flags)
    assert not any(f.startswith("custom.sizes") for f in flags)     # its values were ignored
    assert report["suggestions"] == []


def test_brand_mismatch_overrides_a_yes(vps6, draft):
    reading = {**GOOD_READING, "brand": L("Durvet", "Durvet")}
    d, flags, _, _ = run(vps6, draft, reading)
    assert any("says brand 'Durvet'" in f for f in flags)
    assert not any(s.kind == "image" for s in d.metafields["custom.sizes"].sources)   # not used


def test_number_not_on_the_quoted_label_is_ignored(vps6, draft):
    reading = {**GOOD_READING, "size": L("2 L", "1 Liter")}
    _, flags, _, _ = run(vps6, draft, reading)
    assert any("doesn't show those numbers" in f for f in flags)
    assert not any("custom.sizes: image VPS6-MAIN.jpg says" in f for f in flags)


def test_download_failure_is_a_flag_not_a_crash(vps6, draft):
    d, flags = draft

    def broken(url, folder):
        raise ConnectionError("blocked")
    report = im.check_images(vps6, d, flags, read_fn=fake_reader(GOOD_READING), fetch_fn=broken)
    assert any(f.startswith("image: could not download VPS6-MAIN.jpg") for f in report["flags"])


def test_running_twice_gives_the_same_result(vps6, draft):
    reading = {**GOOD_READING, "size": L("1 gal", "1 gal")}
    d, flags, _, _ = run(vps6, draft, reading)
    first = (list(flags), d.model_dump())
    im.check_images(vps6, d, flags, read_fn=fake_reader(reading), fetch_fn=fake_fetch)
    assert (flags, d.model_dump()) == first


def test_label_that_adds_the_strength_is_an_upgrade_suggestion(vps6, draft):
    d, flags = draft
    d.metafields["info.ingredients"].value = "Tiamulin Hydrogen Fumarate"     # what a real run produced
    _, flags, report, _ = run(vps6, (d, flags), GOOD_READING)
    [s] = report["suggestions"]
    assert s["field"] == "info.ingredients" and s["current"] == "Tiamulin Hydrogen Fumarate"
    assert s["proposed"]["value"] == "12.5% tiamulin hydrogen fumarate" and s["agreement"] == 2
    assert not any(f.startswith("info.ingredients") for f in flags)          # not a conflict


def test_size_read_from_the_real_label_agrees():
    # What Gemini actually returned for VPS6-MAIN.jpg
    assert im.agrees("custom.sizes", ["1 L"], "One Liter", "One Liter (1.057 Quarts) (33.8 fl. oz; 1000 mL)")
    assert not im.agrees("custom.sizes", ["2 L"], "One Liter", "One Liter (1.057 Quarts) (33.8 fl. oz; 1000 mL)")


def test_strength_read_separately_is_joined_to_the_ingredient(vps6, draft):
    # What Gemini actually did on the real labels: name without strength. Now strength has its own field.
    reading = {**GOOD_READING, "ingredients": L("Tiamulin hydrogen fumarate", "Tiamulin hydrogen fumarate"),
               "strength": L("12.5%", "TiaGard 12.5%")}
    d, flags, report, _ = run(vps6, draft, reading)
    got = report["images"][0]["reading"]["ingredients"]
    assert got["value"] == "12.5% Tiamulin hydrogen fumarate"
    assert got["label_text"] == "TiaGard 12.5% … Tiamulin hydrogen fumarate"
    assert not any(f.startswith("info.ingredients") for f in flags)              # agrees with the draft


def test_strength_not_on_its_quoted_label_text_is_not_joined():
    r = im.ImageReading.model_validate({**GOOD_READING, "ingredients": L("tiamulin", "tiamulin"),
                                        "strength": L("45%", "TiaGard 12.5%")})
    assert im.with_strength(r).ingredients.value == "tiamulin"
