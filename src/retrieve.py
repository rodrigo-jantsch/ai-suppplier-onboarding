"""Stage 2a: the retriever (the "R" in RAG).

Given one product (an InputRow), find the most relevant pieces of the
knowledge base so the mapping step can put only those into the prompt:

    taxonomy  - candidate Shopify categories          (config/taxonomy.yaml)
    rules     - listing and compliance rules           (rules/*.md)
    examples  - similar approved listings, for style   (examples/approved.jsonl)

It uses HYBRID SEARCH, as in Module 2 of the RAG course:

    1. Semantic search  - embeddings (nomic-embed-text) stored in Chroma.
                          Finds meaning: "pig" ~ "swine" ~ "livestock".
    2. Keyword search   - BM25 over the same texts.
                          Finds exact words: "Prop 65", "UN1950", a brand name.
    3. Reciprocal Rank Fusion (RRF) merges the two ranked lists.

Try it:  python -m src.retrieve inbox/VPS6.csv
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml
from rank_bm25 import BM25Okapi

from src import llm
from src.config import ROOT, settings
from src.schemas import InputRow

EmbedFn = Callable[[list[str]], list[list[float]]]

# nomic-embed-text is trained with these task prefixes: documents and queries
# are embedded slightly differently, which improves retrieval quality.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# How many results each search method contributes before fusion.
CANDIDATES_PER_METHOD = 20


# ------------------------------------------------------------------ documents

@dataclass
class Doc:
    """One searchable piece of the knowledge base."""
    id: str
    text: str                       # what gets embedded and keyword-indexed
    metadata: dict = field(default_factory=dict)


def load_taxonomy_docs() -> list[Doc]:
    types = yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf-8"))["types"]
    docs = []
    for t in types:
        text = f"{t['name']}. {t.get('description', '')} Keywords: {', '.join(t.get('keywords', []))}"
        docs.append(Doc(id=t["id"], text=text,
                        metadata={"name": t["name"], "fallback": bool(t.get("fallback", False))}))
    return docs


def load_rule_docs() -> list[Doc]:
    docs = []
    for path in sorted((ROOT / "rules").glob("*.md")):
        body = path.read_text(encoding="utf-8")
        kind = re.search(r"^kind:\s*(\S+)", body, re.M)
        docs.append(Doc(id=path.stem, text=body.strip(),
                        metadata={"kind": kind.group(1) if kind else "unknown"}))
    return docs


def load_example_docs() -> list[Doc]:
    path = ROOT / "examples/approved.jsonl"
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        # Embed what describes the product, not the whole JSON.
        text = f"{ex.get('title', '')}. {ex.get('product_type', '')}. Tags: {', '.join(ex.get('tags', []))}"
        docs.append(Doc(id=str(ex["sku"]), text=text, metadata={"json": line}))
    return docs


COLLECTIONS: dict[str, Callable[[], list[Doc]]] = {
    "taxonomy": load_taxonomy_docs,
    "rules": load_rule_docs,
    "examples": load_example_docs,
}


# ------------------------------------------------------------------ helpers

# Words too common to help keyword search find anything specific.
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or
that the this to was were will with without your you our we can no not all any
each per than then into over under up out also only more most""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, minus filler words, for BM25.
    'Prop 65 for the dogs!' -> ['prop', '65', 'dogs']"""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def fingerprint(docs: list[Doc]) -> str:
    """Short hash of the documents, to know when the index is out of date."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.id.encode())
        h.update(d.text.encode())
    return h.hexdigest()[:16]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merge several ranked lists of ids into one score per id.

    Each list gives a document 1 / (k + rank) points (rank starts at 1).
    A document ranked well by BOTH methods beats one ranked first by only one.
    k (60 is the usual value) softens the gap between rank 1 and rank 2.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    """One retrieved document, with how each method ranked it (for debugging)."""
    id: str
    text: str
    metadata: dict
    score: float
    vector_rank: int | None
    keyword_rank: int | None


# ------------------------------------------------------------------ retriever

class Retriever:
    """Hybrid retriever over the three knowledge-base collections.

    embed_fn and client are injectable so tests can run without Ollama or a disk index.
    """

    def __init__(self, embed_fn: EmbedFn | None = None, client=None, persist_dir: Path | None = None):
        import chromadb

        self.embed_fn = embed_fn or llm.embed
        self.client = client or chromadb.PersistentClient(path=str(persist_dir or ROOT / "chroma"))
        self.rrf_k = settings()["retrieval"].get("rrf_k", 60)
        self._docs: dict[str, dict[str, Doc]] = {}
        self._bm25: dict[str, tuple[BM25Okapi, list[str]]] = {}
        self._collections = {}
        for name, loader in COLLECTIONS.items():
            self._build(name, loader())

    def _build(self, name: str, docs: list[Doc]) -> None:
        """Index one collection: Chroma for vectors (only re-embedded if the
        source changed) and BM25 in memory (cheap, rebuilt every time)."""
        self._docs[name] = {d.id: d for d in docs}
        self._bm25[name] = (BM25Okapi([tokenize(d.text) for d in docs]), [d.id for d in docs])

        fp = fingerprint(docs)
        existing = {c.name for c in self.client.list_collections()}
        if name in existing:
            col = self.client.get_collection(name)
            if (col.metadata or {}).get("fingerprint") == fp and col.count() == len(docs):
                self._collections[name] = col          # up to date: no embedding needed
                return
            self.client.delete_collection(name)

        col = self.client.create_collection(name, metadata={"hnsw:space": "cosine", "fingerprint": fp})
        if docs:
            vectors = self.embed_fn([DOC_PREFIX + d.text for d in docs])
            col.add(ids=[d.id for d in docs], documents=[d.text for d in docs], embeddings=vectors)
        self._collections[name] = col

    def search(self, collection: str, query: str, k: int = 5) -> list[Hit]:
        """Hybrid search one collection and return the top k hits."""
        docs = self._docs[collection]
        n = min(CANDIDATES_PER_METHOD, len(docs))
        if n == 0:
            return []

        # 1. Semantic: nearest vectors in Chroma (approximate nearest neighbours, HNSW).
        query_vec = self.embed_fn([QUERY_PREFIX + query])[0]
        result = self._collections[collection].query(query_embeddings=[query_vec], n_results=n)
        vector_ranking = result["ids"][0]

        # 2. Keyword: BM25 scores for every doc; keep those that match at all.
        bm25, ids = self._bm25[collection]
        scores = bm25.get_scores(tokenize(query))
        keyword_ranking = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i]) if scores[i] > 0][:n]

        # 3. Fuse.
        fused = reciprocal_rank_fusion([vector_ranking, keyword_ranking], k=self.rrf_k)
        top = sorted(fused, key=lambda doc_id: -fused[doc_id])[:k]
        return [
            Hit(id=doc_id, text=docs[doc_id].text, metadata=docs[doc_id].metadata, score=fused[doc_id],
                vector_rank=vector_ranking.index(doc_id) + 1 if doc_id in vector_ranking else None,
                keyword_rank=keyword_ranking.index(doc_id) + 1 if doc_id in keyword_ranking else None)
            for doc_id in top
        ]


# ------------------------------------------------------------------ per product

def query_text(row: InputRow) -> str:
    """Turn a product into a search query: the facts that say what it IS.

    Leaves out prices, dimensions, URLs and yes/no flags; they carry no meaning
    for finding a category or style example.
    """
    parts = [row.name, row.brand, row.manufacturer, row.animal_type, row.size,
             row.short_description, *row.bullets[:3], (row.description or "")[:400]]
    return " ".join(p for p in parts if p)


def category_candidates(retriever: Retriever, hits: list[Hit]) -> list[dict]:
    """Expand retrieved categories with all their parent levels.

    Retrieval ranks by similarity, so a general category ("Animals & Pet
    Supplies") almost never wins on its own: its text is broad by design. Code
    therefore adds every parent of each retrieved category, plus the top level,
    so the model can always choose a more general level (rules/TAX-01.md).

    Returns specific matches first, in retrieval order, then the added parents
    from most general to most specific. Each item says whether retrieval found
    it or code added it, which is useful when reviewing mistakes.
    """
    by_name = {d.metadata["name"]: d for d in retriever._docs["taxonomy"].values()}
    seen, retrieved, parents = set(), [], []
    for h in hits:
        name = h.metadata["name"]
        if name not in seen:
            seen.add(name)
            retrieved.append({"name": name, "source": "retrieved",
                              "vector_rank": h.vector_rank, "keyword_rank": h.keyword_rank})
    for h in hits:
        parts = h.metadata["name"].split(" > ")
        for i in range(1, len(parts)):
            name = " > ".join(parts[:i])
            if name in by_name and name not in seen:
                seen.add(name)
                parents.append({"name": name, "source": "parent level",
                                "vector_rank": None, "keyword_rank": None})
    parents.sort(key=lambda c: c["name"].count(" > "))
    return retrieved + parents


def retrieve_for_row(retriever: Retriever, row: InputRow) -> dict:
    """Everything the mapping step needs for one product."""
    cfg = settings()["retrieval"]
    q = query_text(row)
    taxonomy_hits = retriever.search("taxonomy", q, k=cfg["taxonomy_top_k"])
    return {
        "taxonomy": taxonomy_hits,
        "categories": category_candidates(retriever, taxonomy_hits),
        "rules": retriever.search("rules", q, k=cfg["rules_top_k"]),
        "examples": retriever.search("examples", q, k=cfg["examples_top_k"]),
    }


if __name__ == "__main__":
    # python -m src.retrieve inbox/VPS6.csv
    import sys

    from src.ingest import ingest_file

    rows, _ = ingest_file(Path(sys.argv[1]))
    retriever = Retriever()
    for row in rows:
        print(f"\n=== {row.sku}: {row.name}")
        print(f"query: {query_text(row)[:150]}...")
        results = retrieve_for_row(retriever, row)
        for collection in ("taxonomy", "rules", "examples"):
            print(f"\n  {collection}:")
            for h in results[collection]:
                label = h.metadata.get("name", h.id)
                print(f"    {h.score:.4f}  vector#{h.vector_rank or '-':<3} keyword#{h.keyword_rank or '-':<3} {label}")
        print("\n  category candidates sent to the model:")
        for c in results["categories"]:
            print(f"    [{c['source']:<12}] {c['name']}")
