# AI-Assisted Vulnerability Risk Prioritization

Predictive probability model for vulnerability triage — replaces static CVSS severity heuristics with a calibrated `P(Exploit | Features)` estimate.

See [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for the problem statement, architecture, and roadmap, and [`DATA_SOURCES.md`](./DATA_SOURCES.md) for the three data sources (NVD, CISA KEV, FIRST EPSS) and how they join.

## Current result

Read [`FINDINGS.md`](./FINDINGS.md) before quoting any number from this repo.

Evaluated point-in-time (EPSS read from dated archive snapshots, not a current
pull) against a bounded target (added to CISA KEV within 365 days of
publication), on CVEs published 2023-07 → 2024-12, base rate 0.422%:

| ranker | PR-AUC | P@100 |
|---|---|---|
| CVSS base score | 0.0120 | 0.050 |
| **EPSS (point-in-time)** | **0.2701** | **0.480** |
| XGB (no EPSS) | 0.0813 | 0.140 |
| XGB + EPSS (point-in-time) | 0.2066 | 0.410 |

**The model does not currently beat EPSS** — worse on PR-AUC and worse at P@100,
the cutoff that matters for triage, in **0 of 5 seeds** (five-seed mean 0.1866 ±
0.0019 against EPSS at 0.2701). The content-only model is a real but weak
standalone signal (19.3x lift over base rate) that does not survive combination
with a stronger one — most likely a capacity problem, 299 training positives
against ~3,000 features.

An earlier run reported 0.4176 PR-AUC. That came from EPSS scores pulled in 2026
for CVEs published in 2023-24 — after the exploitation they were predicting —
scored against an open-ended "in KEV ever" label. `src/schema.py` now rejects
feature frames that would repeat either mistake.

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
