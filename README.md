# AI-Assisted Vulnerability Risk Prioritization

Predictive probability model for vulnerability triage — replaces static CVSS severity heuristics with a calibrated `P(Exploit | Features)` estimate.

See [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for the problem statement, architecture, and roadmap, and [`DATA_SOURCES.md`](./DATA_SOURCES.md) for the three data sources (NVD, CISA KEV, FIRST EPSS) and how they join.

## Current result

Read [`FINDINGS.md`](./FINDINGS.md) before quoting any number from this repo.

Evaluated point-in-time (EPSS read from dated archive snapshots, not a current
pull), on CVEs published 2023-07 → 2024-12, base rate 0.459%:

| ranker | PR-AUC | P@100 |
|---|---|---|
| CVSS base score | 0.0125 | 0.050 |
| EPSS (point-in-time) | **0.2529** | 0.480 |
| XGB (no EPSS) | 0.0813 | 0.200 |
| XGB + EPSS (point-in-time) | 0.2304 | **0.510** |

The model **does not beat EPSS on PR-AUC** — it is 8.9% worse — but improves
precision at the top-100 cutoff where triage actually happens. An earlier run
reported 0.4176 PR-AUC; that number came from EPSS scores pulled in 2026 for
CVEs published in 2023-24, after the exploitation they were predicting, and is
not reproducible in deployment. `src/schema.py` now rejects feature frames that
would repeat it.

## Architecture

![Three data sources join on CVE ID into a feature table that trains a calibrated XGBoost classifier; the model serves probability and SHAP explanations via FastAPI and a Svelte UI, with a drift monitor triggering retraining.](./docs/architecture.svg)

## Layout

```
src/
  schema.py      # data contract; marks each column point-in-time-safe or snapshot-only
  ingestion/     # NVD, CISA KEV, FIRST EPSS clients
    epss_history.py  # dated EPSS archive snapshots (the leakage fix)
  features/      # joins the three sources into a training table
  model/
    evaluate.py      # original temporal evaluation
    evaluate_pit.py  # point-in-time re-evaluation
    verify.py        # split / seed / calibration checks
  api/           # FastAPI inference endpoint with SHAP explanations
data/
  raw/           # pulled API responses + cached EPSS snapshots, untouched
  processed/     # joined feature tables (parquet)
notebooks/       # exploratory analysis
tests/
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.ingestion.nvd_client   # sanity-check NVD access
python -m src.ingestion.kev_client   # sanity-check CISA KEV feed
python -m src.ingestion.epss_client  # sanity-check FIRST EPSS API
```
