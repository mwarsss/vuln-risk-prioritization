# Findings

## 1. The first headline number was produced by target leakage

`data/processed/eval_results.json` reports:

| ranker | PR-AUC | P@100 |
|---|---|---|
| CVSS base score | 0.0125 | 0.04 |
| EPSS score | 0.4086 | 0.81 |
| XGBoost (no EPSS) | 0.0509 | 0.12 |
| XGBoost + EPSS | **0.4176** | 0.70 |

The evaluation harness is sound — temporal split, PR-AUC over accuracy,
encoders fit on train only, NaNs ranked last. The split is not the problem.
**The feature timing is.**

`epss_score` came from a bulk Kaggle mirror pulled **2026-08-09**, a single
undated column. The test window is CVEs published **2023–2024**. EPSS is
recomputed daily by FIRST using live exploitation telemetry, so by the time of
that pull it had already observed the exploitation that `is_kev` records. The
feature was measured *after* the outcome it is supposed to predict.

The separation is not subtle:

| | KEV-positive | KEV-negative |
|---|---|---|
| median EPSS | 0.508 | 0.005 |
| share EPSS > 0.90 | **32.7%** | 0.10% |

For scale: in the genuine 2023-01-02 snapshot only **0.16% of all CVEs** score
above 0.9. A third of exploited CVEs sitting there in a 2026 pull is EPSS
recording history, not forecasting it.

## 2. Corrected result

`src/ingestion/epss_history.py` reads EPSS from FIRST's dated archive
(`epss.cyentia.com/epss_scores-YYYY-MM-DD.csv.gz`) instead of a current pull, so
each CVE is scored with the EPSS value available shortly after its own
publication. `src/model/evaluate_pit.py` re-runs the identical comparison on
that basis. Window is bounded by the archive start (2021-04-14); 98.5% EPSS
coverage; train ≤2023-06-30 (53,983 rows / 360 KEV), test 2023-07-01→2024-12-31
(54,310 rows / 249 KEV, base rate 0.459%).

| ranker | PR-AUC | ROC-AUC | P@100 | P@500 | R@1000 |
|---|---|---|---|---|---|
| CVSS base score | 0.0125 | 0.745 | 0.050 | 0.022 | 0.064 |
| EPSS (leaked, 2026 pull) | 0.4174 | 0.974 | 0.740 | 0.244 | 0.622 |
| **EPSS (point-in-time)** | **0.2529** | 0.800 | 0.480 | 0.206 | 0.474 |
| XGB (no EPSS) | 0.0813 | 0.882 | 0.200 | 0.100 | 0.309 |
| **XGB + EPSS (point-in-time)** | **0.2304** | 0.903 | **0.510** | 0.158 | 0.406 |
| XGB + EPSS (leaked) | 0.4514 | 0.985 | 0.580 | 0.340 | 0.843 |

**EPSS PR-AUC falls 39.4% (0.417 → 0.253) once read as of publication.** That
gap is the leakage, measured.

## 3. What the model is honestly worth

Three things follow, and the second one is not flattering:

1. **CVSS is close to useless for this task.** PR-AUC 0.0125 against a 0.459%
   base rate. It measures severity, not exploitation. The premise of the
   project holds.
2. **The model does not beat EPSS on PR-AUC.** 0.2304 vs 0.2529 — the ensemble
   is *8.9% worse* than simply using the free public score. Any claim that this
   pipeline outperforms EPSS is unsupported.
3. **It does win at the top of the ranking**, which is where triage happens:
   P@100 0.510 vs 0.480, and ROC-AUC 0.903 vs 0.800. A team patching 100 CVEs a
   cycle finds ~3 more true exploited vulns per cycle. Real, but modest.

The defensible claim is: *a content-only model reaches 17.7x lift over base
rate using no proprietary feed, and adding it to EPSS improves the top-100
cutoff while slightly degrading global ranking.* Not "beats EPSS".

## 4. Known remaining gap: the label is still open-ended

`is_kev` means "appears in CISA KEV **as of the 2026-08-09 pull**", not "was
exploited within N days of publication". A CVE published in 2023 and added to
KEV in 2026 counts as a positive the model was expected to catch at publication
time. This inflates apparent achievability and drifts every time CISA backfills.

The fix needs `kev_date_added`, which the Kaggle mirror drops — it is present in
the primary CISA feed that `src/ingestion/kev_client.py` already targets.
`src.schema.build_horizon_label()` implements the bounded target
(`exploited_within_365d`, immature rows returned as `<NA>` rather than silently
counted as negatives) and is ready to use once that column is ingested.
**Numbers in §2 should be treated as provisional until this lands.**

## 5. Guardrail

The failure was invisible in the data — the leaked and the honest column are
both just floats in `[0,1]`. So the constraint is enforced structurally rather
than left to review: `src/schema.py` marks every column `PIT_SAFE` or
`POINT_IN_TIME`, and `validate()` **rejects** a training frame carrying EPSS
without an `epss_as_of` snapshot date, or with a snapshot predating publication.

Run against the original feature table, it fails, which is the correct outcome:

```
ERROR   epss_score, epss_perc present without epss_as_of. These are revised
        daily from exploitation telemetry, so a current pull leaks the label
        for historical CVEs. Load a dated snapshot or drop them.
```

### A second bug the same check catches

The first point-in-time run returned EPSS PR-AUC 0.0046 at ROC-AUC 0.4952 —
exactly random. Cause: CVEs were aligned to the snapshot at the start of their
own publication month, but EPSS only scores a CVE *after* it is published. Of
CVEs published in January 2023, **1.2%** appear in the 2023-01-02 snapshot and
**100%** in the 2023-02-01 one. Coverage was 1.1% and the feature was silently
all-NaN. Alignment now targets the period *following* publication (98.5%
coverage), and `validate()` flags any `epss_as_of` earlier than
`published_date`.

## Reproducing

```bash
python -m src.schema                  # print the data contract
python -m src.model.evaluate          # original run (leaked EPSS)
python -m src.model.evaluate_pit      # corrected run (fetches ~45 snapshots, cached)
python -m src.model.verify            # split/seed/calibration checks
```
