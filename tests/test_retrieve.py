"""Tests for stage 2a (retrieval). Run:  python -m pytest -q tests/test_retrieve.py

These use a small fake embedder (hashed word counts) and an in-memory Chroma,
so they run in seconds without Ollama. They test the plumbing: indexing,
hybrid search, fusion and query building. Real embedding quality is checked
by running  python -m src.retrieve inbox/VPS6.csv  on your machine.
"""
import hashlib
import math
from pathlib import Path

import chromadb
import pytest

from src import retrieve
from src.ingest import ingest_file

INBOX = Path(__file__).resolve().parent.parent / "inbox"
DIM = 256


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words hashed into 256 buckets, normalized. Similar words -> similar vectors."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for word in retrieve.tokenize(t.replace("search_document:", "").replace("search_query:", "")):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture(scope="module")
def retriever():
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    r = retrieve.Retriever(embed_fn=counting_embed, client=chromadb.EphemeralClient())
    r.embed_calls = calls
    return r


@pytest.fixture(scope="module")
def vps6():
    rows, _ = ingest_file(INBOX / "VPS6.csv")
    return rows[0]


# ---- pieces ----------------------------------------------------------------

def test_tokenize():
    assert retrieve.tokenize("Prop 65 for the UN1950!") == ["prop", "65", "un1950"]


def test_rrf_rewards_agreement():
    scores = retrieve.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    # a and b are near the top of both lists; c and d appear only once
    assert set(sorted(scores, key=scores.get, reverse=True)[:2]) == {"a", "b"}
    assert scores["c"] < scores["a"] and scores["d"] < scores["b"]


def test_rrf_keeps_items_found_by_one_method_only():
    scores = retrieve.reciprocal_rank_fusion([["x"], []], k=60)
    assert "x" in scores


def test_fingerprint_changes_with_content():
    a = [retrieve.Doc("1", "hello")]
    b = [retrieve.Doc("1", "hello world")]
    assert retrieve.fingerprint(a) != retrieve.fingerprint(b)


# ---- indexing --------------------------------------------------------------

def test_all_collections_indexed(retriever):
    assert retriever._collections["taxonomy"].count() == len(retrieve.load_taxonomy_docs())
    assert retriever._collections["rules"].count() == len(retrieve.load_rule_docs())
    assert retriever._collections["examples"].count() == len(retrieve.load_example_docs())


def test_unchanged_index_is_not_re_embedded(retriever):
    before = len(retriever.embed_calls)
    retrieve.Retriever(embed_fn=retriever.embed_fn, client=retriever.client)  # same client, same docs
    assert len(retriever.embed_calls) == before   # loaded from Chroma, no new embedding calls


# ---- search ----------------------------------------------------------------

def test_hits_report_both_ranks(retriever):
    hits = retriever.search("rules", "Prop 65 California warning disclosure", k=3)
    assert hits[0].id == "PROP-01"
    assert hits[0].vector_rank is not None and hits[0].keyword_rank is not None


def test_keyword_search_finds_exact_terms(retriever):
    hits = retriever.search("rules", "Variation Parent SKU CatalogRef", k=3)
    assert "TAG-01" in [h.id for h in hits]


def test_query_text_uses_product_facts(vps6):
    q = retrieve.query_text(vps6)
    assert "TiaGard" in q and "Swine" in q
    assert "24.99" not in q and "https://" not in q


def test_vps6_candidates_always_include_the_top_level(retriever, vps6):
    results = retrieve.retrieve_for_row(retriever, vps6)
    names = [c["name"] for c in results["categories"]]
    assert "Animals & Pet Supplies" in names       # gold product_type for VPS6
    assert len(results["rules"]) == 5
    assert len(results["examples"]) >= 1


def test_candidates_add_parent_levels_after_retrieved(retriever):
    hit = retriever.search("taxonomy", "aquarium water treatment conditioner", k=1)
    cands = retrieve.category_candidates(retriever, hit)
    assert cands[0]["source"] == "retrieved"
    parents = [c["name"] for c in cands if c["source"] == "parent level"]
    assert parents[0] == "Animals & Pet Supplies"                 # most general first
    leaf = cands[0]["name"]
    assert all(leaf.startswith(p) for p in parents)               # every parent is on the path
    assert len(parents) == leaf.count(" > ")                      # one per level above the leaf
