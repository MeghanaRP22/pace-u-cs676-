# CS676 Project 1 — Submission notes

**Student:** Harish · **Report:** `report/CS676_Project1_Technique_Report.pdf` (and `.docx`)

## Results (rules only — no API calls)

Course evaluation set (24 URLs), `uv run python evaluate.py`:

| Scorer | MAE | Band accuracy | Worst | Held-out MAE |
|---|---|---|---|---|
| Course baseline (`--baseline`) | 0.142 | 66.7% | 0.410 | 0.243 |
| Course baseline + LLM (README, Opus 5) | 0.086 | 83.3% | 0.230 | — |
| **Submitted URL model** | **0.064** | **87.5%** | **0.240** | **0.080** |
| URL model without published ratings (`--variant structural`) | 0.072 | 91.7% | 0.210 | 0.086 |

All 48 URLs including my 24 extra held-out URLs (`--extended`): MAE 0.199 → 0.083,
band accuracy 47.9% → 83.3%, 90% interval coverage 89.6%.

## What I built (novelty beyond a lookup table)

1. **Learned URL model** — 30 structural features (DOI, journal-article path, `.int`,
   gov equivalents, Q&A/UGC paths, self-hosted blog subdomains, clickbait slug,
   throwaway domain names, look-alike hosts, personal academic pages...) fed to a
   fractional-logit regression whose weights are fitted by `train.py` on 230
   labelled URLs in `training_data.py` (leakage-checked against the evaluation set).
2. **Published ratings as a feature** — Lin et al. (2023, *PNAS Nexus*) domain
   quality scores for 11,520 domains (`data/domain_pc1.csv`), instead of more hand-typed numbers.
3. **Uncertainty** — 200 bootstrap refits + out-of-bag residuals → 90% interval on every score.
4. **Evidence layer** — reads the page (citation meta tags, schema.org type, author,
   date, outbound citations, corrections/ethics links, sponsorship) and queries
   OpenAlex/Crossref for retractions, published versions of preprints and citations.
   Dead domains, 404s, timeouts, blocked fetches and TLS errors are handled.
5. **Precision-weighted LLM blend** replacing the fixed 0.6/0.4 constant.
6. **Reader-facing explanations** — verdict, source type, the reasons that moved the
   score most, and what to check.

## Files

| File | Purpose |
|---|---|
| `credibility.py` | The scorer (contract unchanged: `score_url()` → `{"score", "explanation"}`) |
| `train.py`, `training_data.py`, `model_weights.json` | Fitting the model and its weights |
| `evaluate.py` | Extended harness: blocks, extended set, baseline, ablation, `--net`, `--llm`, `--tune-blend` |
| `test_credibility.py` | 21 original + 56 added tests (77 total, no network or key needed) |
| `main.py` | App: parallel cached scoring, chips with range + source type, evidence list, batch scorer |
| `baseline/credibility_baseline.py` | The untouched course scorer, for before/after |
| `results/` | CSVs and JSON behind every number in the report |
| `Dockerfile`, `.dockerignore`, `deploy/SPACE_README.md` | Hugging Face Space (Docker SDK) |

## Commands

```bash
uv sync
uv run python test_credibility.py        # 77 passed, 0 failed
uv run python evaluate.py                # course 24 URLs
uv run python evaluate.py --extended     # all 48
uv run python evaluate.py --baseline     # original scorer, for comparison
uv run python train.py                   # refit weights (optional; weights are included)
uv run streamlit run main.py
```

## Still to do before submitting (needs your key / internet)

- `uv run python evaluate.py --llm` and `--llm --extended` → fill the "+ LLM" row in
  Table 1 of the report (`report/build_report.js`, then `node report/build_report.js`,
  or edit the .docx directly). Optionally `--llm --tune-blend`.
- `uv run python evaluate.py --net --extended` on a machine with internet, for the
  full evidence-layer numbers (my development machine only had DNS).
- Hugging Face bonus: create a Docker Space, copy `deploy/SPACE_README.md` to the
  Space's `README.md`, push everything except `.env`/`.venv`, and add
  `ANTHROPIC_API_KEY` as a Space secret.
