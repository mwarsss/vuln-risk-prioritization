# AI-Assisted Vulnerability Risk Prioritization

Predictive probability model for vulnerability triage — replaces static CVSS severity heuristics with a calibrated `P(Exploit | Features)` estimate.

See [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for the problem statement, architecture, and roadmap, and [`DATA_SOURCES.md`](./DATA_SOURCES.md) for the three data sources (NVD, CISA KEV, FIRST EPSS) and how they join.

## Current result

Read [`FINDINGS.md`](./FINDINGS.md) before quoting any number from this repo.

Evaluated point-in-time — EPSS read from the dated archive snapshot for
publication + 1 day, never a current pull — against a bounded target (added to
CISA KEV within 365 days of publication). CVEs published 2023-07 → 2024-12,
54,290 rows, 209 positives, base rate 0.385%.

| ranker | PR-AUC | P@100 |
|---|---|---|
| CVSS base score | 0.0120 | 0.050 |
| **EPSS (point-in-time)** | **0.1753** | **0.410** |
| XGB (no EPSS) | 0.0457 | — |
| rank blend, weight chosen on validation | 0.0457 | 0.112 |

**The model does not beat EPSS as a ranker.** Feeding EPSS in as a feature fails,
and rank-blending fails too — though not for the reason previously recorded here.
An honest sweep *does* have an interior optimum (w=0.05 scores 0.2194, +25% over
EPSS alone), but validation picks w=1.0 in all five seeds, landing on the worst
point of the curve. A useful blend weight exists and this protocol cannot find
it; 63 validation positives are not enough. See §7.

**One thing does work: gate, don't mix.** Read one day after publication, EPSS is
a small, sharp instrument — only 37 of 54,290 test CVEs score above 0.01, and
80% of the exploited ones sit in the pile it scores below 0.01. It barely orders
*within* its bands (0.89–3.32x lift); its power is the separation *between* them.
The content model is the mirror image, at 12.8x lift inside the bottom band.
Letting EPSS rank the head of the queue and a content model order its tail:

| budget | EPSS | EPSS + content tail | |
|---|---|---|---|
| 100 | 41 | 13.0 ± 1.8 | far worse, 0/5 seeds |
| 1,000 | 72 | 59.4 ± 2.9 | worse, 0/5 seeds |
| 5,000 | 90 | **130.0 ± 1.7** | **+40, 5/5 seeds** |
| 10,000 | 107 | **158.4 ± 3.8** | **+51, 5/5 seeds** |

Past a ~1,500-CVE budget this recovers roughly **40 more exploited CVEs per
5,000 patched** — recall 43.1% → 62.2% — using only NVD metadata. Below that
crossover it is actively harmful. This is a backlog-sweep result, not a
weekly-patch-cycle one. See [`FINDINGS.md`](./FINDINGS.md) §7–9.

### Two leaks, both measured

An early run reported 0.4176 PR-AUC, from EPSS scores pulled in 2026 for CVEs
published in 2023-24 — after the exploitation they were predicting — against an
open-ended "in KEV ever" label (§1–4).

The fix for that introduced a second, smaller one: aligning EPSS to the start of
the month *after* publication bought coverage but put the snapshot a median 16
days past publication, against a median 15 days to KEV listing. **59% of
positives were scored with EPSS values read after CISA had already catalogued
them**, inflating the EPSS baseline from 0.1753 to 0.2701 (§6). Daily alignment
brings contamination to the floor the catalogue itself sets.

`src/schema.py:validate()` now runs on every point-in-time load and refuses both
mistakes — it previously had no call sites at all.

## Architecture

![Three data sources join on CVE ID into a feature table that trains a calibrated XGBoost classifier; the model serves probability and SHAP explanations via FastAPI and a Svelte UI, with a drift monitor triggering retraining.](./docs/architecture.svg)

## Layout

```
src/
  schema.py      # data contract; marks each column point-in-time-safe or snapshot-only
  ingestion/     # NVD, CISA KEV, FIRST EPSS clients
    epss_history.py  # dated EPSS archive snapshots (the leakage fix)
    prefetch_daily.py   # warms the ~1,300-file daily snapshot cache
  features/      # joins the three sources into a training table
  model/
    dataset.py       # THE validated loader — attaches dated EPSS, runs schema.validate
    evaluate.py      # original temporal evaluation (deliberately leaked baseline)
    evaluate_pit.py  # point-in-time re-evaluation
    verify.py        # split / seed / calibration checks
    verify_pit_lag.py   # measures snapshot-vs-KEV-listing contamination
    blend.py         # rank-blend vs EPSS; weight chosen on validation
    complementarity.py  # per-EPSS-band: where does content add signal?
    two_stage.py     # gated policy — EPSS ranks the head, content the tail
  model/bundle.py    # persisted content model + gated ranking policy (what gets served)
  model/train.py     # temporal fit -> model.joblib
  api/           # FastAPI: /score (one CVE) and /rank (a backlog, gated order)
data/
  raw/           # pulled API responses + cached EPSS snapshots, untouched
  processed/     # joined feature tables (parquet)
notebooks/       # exploratory analysis
tests/           # python -m tests.test_ranking / tests.test_schema_gate
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.ingestion.nvd_client   # sanity-check NVD access
python -m src.ingestion.kev_client   # sanity-check CISA KEV feed
python -m src.ingestion.epss_client  # sanity-check FIRST EPSS API

python -m src.ingestion.prefetch_daily   # ~1,300 daily EPSS snapshots (~1.6 GB, once)
python -m src.model.two_stage            # the gated-policy result
python -m pytest tests               # ranking, schema gate, serving bundle

python -m src.model.train            # fit the content model -> model.joblib
uvicorn src.api.main:app             # serve /score and /rank
```

`/rank` falls back to EPSS-only below a ~1,500-CVE budget, where FINDINGS §9 shows the gate
is worse than EPSS alone. EPSS passed to the API must be the value as of publication + 1 day.
