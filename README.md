# Supplier Onboarding Agent

Turns a messy supplier CSV into review-ready Shopify listings:

1. **Ingest** the CSV and map supplier columns to a standard schema
2. **Map** each row to Shopify fields and metafields using retrieved store rules (RAG)
3. **Enrich** missing facts from competitor product pages (agent with tools)
4. **Validate** with deterministic checks (schema, quotes, product match, compliance)
5. **Review and export** to a Matrixify CSV, with a person approving suggestions

Runs locally on free models through Ollama. Full plan: see the "Supplier Onboarding Agent — Build Plan" doc.

## Status

- [x] Weekend 1 scaffold: configs, starter rules, sample CSVs, schemas, LLM wrapper, setup check
- [ ] Weekend 1 data: real products, full taxonomy, rules, gold set
- [ ] Weekend 2: ingest + map

## Setup (macOS)

The chat model runs on Google's Gemini API (free tier); embeddings run locally
with a small Ollama model, so this works on a machine with 4 GB of RAM.

```bash
# 1. Install Ollama (https://ollama.com/download), open it once, then:
ollama pull nomic-embed-text

# 2. Python environment (Python 3.11+)
cd ~/Downloads/"AI Learning"/supplier-onboarding-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. API key: create one in Google AI Studio, then
cp .env.example .env
open -e .env        # paste the key after GEMINI_API_KEY=, save

# 4. Check everything works
python -m scripts.check_setup
```

Model names live in `config/settings.yaml`. If the free tier rejects
`gemini-3.8-flash`, check https://aistudio.google.com/rate-limit and switch to
a model that's free for your account (e.g. `gemini-3.5-flash-lite`).
Free-tier prompts may be used by Google to improve its products, so only
public product data goes through it.

## Repo layout

```text
config/       settings, input schema, taxonomy, metafields, competitor sites
rules/        listing and compliance rules, one file each
examples/     approved listings used as style examples
gold/         hand-labeled answers for evaluation
inbox/        supplier CSVs
src/          pipeline code (schemas.py, llm.py, ...)
scripts/      helper scripts (check_setup.py)
```

## Notes

- The two CSVs in `inbox/` are **samples with fictional products and placeholder UPCs**. Replace them with real products (public UPCs) that 2-3 retailers sell, so enrichment has pages to find.
- `supplier_b.csv` deliberately uses different headers (`Item #`, `Net Wt`, `Product Name`). Some won't match the aliases in `input_schema.yaml`; that's the case the LLM fallback in `ingest.py` handles in Weekend 2.
- Rules in `rules/` are illustrative for this project, not legal guidance.
- No employer data in this repo.
# ai-suppplier-onboarding
