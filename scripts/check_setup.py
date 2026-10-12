"""Weekend 1 smoke test. Run from the repo root:  python -m scripts.check_setup

Checks, in order:
  1. Config and API key are in place; Ollama is running with the embedding model
  2. Plain chat works with the configured provider
  3. Structured output returns valid JSON for a Pydantic schema
  4. Embeddings work, and similar texts really are closer (Module 2 in action)
"""
import os
import sys
import time

import numpy as np
from pydantic import BaseModel

from src import llm
from src.config import settings


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def main() -> None:
    cfg = settings()
    provider = cfg["llm"]["provider"]
    chat_model = cfg["llm"]["chat_model"] if provider == "gemini" else cfg["llm"]["ollama_chat_model"]
    embed_model = cfg["embeddings"]["model"]

    step(1, "Config, API key and local embedding model")
    print(f"  chat: {provider} / {chat_model}")
    print(f"  embeddings: {cfg['embeddings']['provider']} / {embed_model}")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("  GEMINI_API_KEY not found. Copy .env.example to .env and paste your key.")
    try:
        import ollama
        names = [m.model for m in ollama.list().models]
    except Exception as err:  # noqa: BLE001
        sys.exit(f"  Can't reach Ollama. Is the app running?  ({err})")
    if not any(n.split(":")[0] == embed_model.split(":")[0] for n in names):
        sys.exit(f"  Missing '{embed_model}'. Run: ollama pull {embed_model}")
    print("  ok")

    step(2, "Plain chat")
    t = time.time()
    try:
        reply = llm.chat_text("In one sentence, what is a flea and tick spray for dogs?")
    except Exception as err:  # noqa: BLE001
        sys.exit(f"  Chat call failed: {err}\n  Check the key, the model name in settings.yaml, and free-tier limits.")
    print(f"  {reply.strip()}\n  ({time.time() - t:.1f}s)")

    step(3, "Structured output")

    class Parsed(BaseModel):
        brand: str
        species: list[str]
        size: str

    t = time.time()
    parsed = llm.chat_json(
        "Extract brand, species and size from this supplier text: "
        "'ACME FLEA/TICK SPRAY DOG, 16OZ'. Normalize size like '16 oz'.",
        Parsed,
    )
    print(f"  {parsed.model_dump()}  ({time.time() - t:.1f}s)")

    step(4, "Embeddings")
    texts = ["flea spray for dogs", "canine flea and tick treatment", "clumping cat litter"]
    vecs = np.array(llm.embed(texts))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim_close = float(vecs[0] @ vecs[1])
    sim_far = float(vecs[0] @ vecs[2])
    print(f"  dimensions: {vecs.shape[1]}")
    print(f"  cosine('{texts[0]}', '{texts[1]}') = {sim_close:.3f}")
    print(f"  cosine('{texts[0]}', '{texts[2]}') = {sim_far:.3f}")
    print("  ok" if sim_close > sim_far else "  unexpected: unrelated text scored closer")

    print("\nAll checks passed. Setup is ready for Weekend 2.")


if __name__ == "__main__":
    main()
