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

## 4. Bounded label (resolved)

`is_kev` originally meant "appears in CISA KEV **as of the pull date**" — a CVE
published in 2023 and added to KEV in 2026 counted as something the model should
have caught on day one. `kev_client.py` now pulls `dateAdded` from the primary
CISA feed (1,665 entries, zero missing dates) and `prepare_kaggle.py` joins it,
so `schema.build_horizon_label()` can bound the target to *exploited within N
days of publication*. Immature rows come back `<NA>` and are dropped rather than
silently counted as negatives.

Two structural facts about the catalogue surfaced doing this, both of which
affect how the label should be read:

- **287 CVEs share `dateAdded` = 2021-11-03**, the day the catalogue launched.
  That is a backfill batch, not detection latency.
- **213 of 918 in-window KEV CVEs (23%) have negative days-to-KEV** — CISA
  flagged them *before* NVD published them. Median days-to-KEV is 15, p25 is 0.
  For roughly a quarter of positives, exploitation is already public knowledge
  at the moment the model is asked to predict it.

Re-run at a 365-day horizon (774 in-KEV CVEs fall outside the horizon and become
negatives; test base rate drops 0.459% → 0.422%):

| ranker | PR-AUC | ROC-AUC | P@100 | P@500 | R@1000 |
|---|---|---|---|---|---|
| CVSS base score | 0.0120 | 0.747 | 0.050 | 0.022 | 0.070 |
| EPSS (leaked) | 0.4136 | 0.974 | 0.730 | 0.228 | 0.624 |
| **EPSS (point-in-time)** | **0.2701** | 0.800 | **0.480** | **0.204** | **0.507** |
| XGB (no EPSS) | 0.0813 | 0.884 | 0.140 | 0.096 | 0.345 |
| XGB + EPSS (point-in-time) | 0.2066 | 0.912 | 0.410 | 0.166 | 0.480 |
| XGB + EPSS (leaked) | 0.4157 | 0.989 | 0.510 | 0.328 | 0.869 |

**This reverses the one win the model had.** Under the open-ended label the
ensemble beat EPSS at P@100 (0.510 vs 0.480). Under the bounded label it loses
there too: 0.410 vs 0.480, and 0.2066 vs 0.2701 on PR-AUC — 23.5% worse. The
earlier P@100 edge came from CVEs exploited *long* after publication, which the
open-ended target rewarded and a 1-year triage horizon does not.

Honest standing: **adding this model to EPSS makes prioritization worse on every
metric except ROC-AUC**, and ROC-AUC is the wrong metric at a 0.42% base rate.
The content-only model is a real but weak signal (0.0813 PR-AUC, 19.3x lift over
base rate) that does not survive combination with a stronger one.

This is not seed noise. Over five seeds on the same split:

| ranker | PR-AUC | P@100 |
|---|---|---|
| EPSS (point-in-time) | 0.2701 | 0.480 |
| XGB (no EPSS) | 0.0752 ± 0.0042 | 0.144 ± 0.020 |
| XGB + EPSS | 0.1866 ± 0.0019 | 0.368 ± 0.007 |

**XGB + EPSS beats EPSS in 0 of 5 seeds.** The 0.0835 shortfall is roughly 44
standard deviations of seed variance, so the ranking is not close. Note also
that seed 42 — the one used in the table above and in `evaluate.py` — returns
0.2066, well above the 0.1866 five-seed mean. The headline single-seed figure
flatters the model; the mean is the number to quote.

The most likely cause is capacity, not concept: 299 positives in train against
~3,000 TF-IDF features plus one-hots, with `scale_pos_weight` ≈ 180. That is a
setup that overfits and dilutes a good input rather than building on it. Worth
trying before abandoning: drop TF-IDF, use EPSS as a monotone constraint or as
an offset/prior instead of one feature among thousands, or rank-blend
`XGB (no EPSS)` with EPSS rather than training on top of it.

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
