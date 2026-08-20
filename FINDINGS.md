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

> Still true as stated — as a *ranker*, and as a *global* combination. Section 9
> finds one combination shape where it does survive, past a budget crossover.
> Note also that the 0.2701 EPSS baseline in this section is itself inflated by
> the alignment leak section 6 documents; the honest figure is 0.1753.

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

> **This guess was not tested here.** The rank-blend it proposes was built and
> run — see section 7. Kept as the record of what was believed at this stage.

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

## 6. The guardrail never ran, and a second leak got in behind it

Section 5 claims the contract is "enforced structurally rather than left to
review". It was not. `validate()` had **zero call sites** anywhere in the
repository outside its own module — `grep -rn "validate" src/ tests/` returned
nothing else. `prepare_kaggle.py` wrote `features.parquet` with a current-pull
`epss_score` and no `epss_as_of`, and `python -m src.model.evaluate` — listed in
this document's own Reproducing block — trained on it and emitted the leaked
0.4176 without complaint. The gate existed as a function nobody called.

That is how the following got through.

### The fix for the first leak introduced a smaller one

Section 5 describes correcting a coverage bug: CVEs are largely absent from the
EPSS snapshot published inside their own month, so alignment moved to the start
of the month *after* publication, lifting coverage from 1.1% to 98.5%.

That works, and it shifts the feature a median **16 days** past publication.
Section 4 measured median days-to-KEV at 15.

So for the majority of positives, the "point-in-time" EPSS was read *after* CISA
had already catalogued the CVE as exploited:

| | clean | contaminated |
|---|---|---|
| in-horizon positives | 91 | **136 (59.4%)** |
| median EPSS | 0.0005 | **0.0088** |
| share EPSS > 0.1 | 8.8% | **27.9%** |

Contaminated rows carry 17.6x the median EPSS of clean ones. That is the same
signature section 1 documents for the 2026 pull (0.508 vs 0.005), one order of
magnitude smaller and correspondingly harder to notice.

### Daily alignment, and what remains

`attach_pit_epss(quantize="D")` targets publication + 1 day instead. Contamination
falls to the floor set by the catalogue itself:

| alignment | contaminated positives | median lag | snapshots |
|---|---|---|---|
| monthly (previous) | 59.8% | 17d | 46 |
| weekly | 49.3% | 5d | 195 |
| **daily** | **37.1%** | **<1d** | 1,314 |
| floor — CISA listed on/before publication | 37.1% | — | — |

Daily reaches the floor exactly; the residual 37.1% is the section-4 fact that
CISA frequently flags a CVE before NVD publishes it, which no feature alignment
can address. Coverage costs 3.6 points (98.5% → 96.4% on the sampled window).

Twenty-three rows still tripped the check after the switch. All 23 are positives,
`kev_date_added` equals `epss_as_of` exactly, and none had a tighter snapshot
available — CISA catalogued them within a day of publication. `load_pit_frame`
drops them: exploitation was public knowledge at prediction time, so there is no
forecast for a prioritizer to make, and scoring EPSS on them measures recall of a
known fact. It costs ~9% of test positives and makes the benchmark marginally
easier, which is a real choice rather than a neutral cleanup.

### The guardrail now runs

`validate()` is called from `src/model/dataset.py` on every point-in-time load
and from `prepare_kaggle.py` at build time. Two bugs in the gate itself had to be
fixed first:

- The checks were a chained `if/elif`, so **any** frame with a coverage gap
  skipped the leak check entirely — the honest frames were the ones that escaped
  scrutiny.
- It flagged rows whose `epss_as_of` was null alongside a null score, which is
  ordinary left-join behaviour, and rejected correctly-built frames for it.

It now separates *avoidable* contamination (snapshot more than one archive day
past publication → error) from *irreducible* (already next-day → warning). Run
against the monthly frame it refuses to load it:

```
SchemaError: 1 schema violation(s):
  - epss_as_of postdates kev_date_added on 114 row(s) whose KEV listing came
    after publication, with a snapshot more than a day past publication.
```

`tests/test_schema_gate.py` pins each case, including
`test_coverage_gap_does_not_mask_the_leak_check`.

### Everything below is re-measured

Sections 7-9 were first written against the monthly-aligned frame. Every number
in them has been recomputed on daily alignment. The corrected EPSS baseline is
**0.1753 PR-AUC, not 0.2701** — the previous figure was inflated 54% by the
contamination above. Test window is now 54,290 rows / 209 positives / 0.385%
base rate.

## 7. Rank-blending: the honest weight is unfindable

Section 4 guessed the ensemble underperformed for capacity reasons — 299
positives against ~3,000 features — and proposed rank-blending as the fix.
`src/model/blend.py` implements it: the content model never sees EPSS, and the
two are combined afterwards as percentile ranks. The mixing weight `w` is chosen
on a validation slice carved off the end of train, never on test.

Split: fit ≤2022-12-31 (39,930 rows / 233 positive), validate 2023-01→2023-06
(14,050 / 63), test 2023-07→2024-12 (54,290 / 209).

The test sweep, as a shape:

| w | 0.00 | **0.05** | 0.10 | 0.20 | 0.40 | 0.70 | 1.00 |
|---|---|---|---|---|---|---|---|
| PR-AUC | 0.1753 | **0.2194** | 0.2058 | 0.1935 | 0.1752 | 0.1574 | 0.0457 |

**There is an interior optimum.** A 5% admixture of the content model beats pure
EPSS by 25% (0.2194 vs 0.1753). Under the previous, contaminated alignment this
curve fell monotonically and the conclusion recorded here was that no mixture
could help. That conclusion was an artifact of the leak: an inflated EPSS is
harder to improve on.

And it does not survive contact with an honest protocol.

**Validation chose `w = 1.0` in all five seeds** — pure content model — which
scores **0.0457** on test, the single worst point on the curve. Not a near-miss
at 0.05: the opposite end.

| ranker | PR-AUC | P@100 |
|---|---|---|
| EPSS (point-in-time) | **0.1753** | 0.410 |
| XGB content only | 0.0457 | — |
| rank blend, `w` from validation | 0.0457 | 0.112 |
| rank blend at the test optimum `w=0.05` | *0.2194* | — |

Blend beats EPSS in **0 of 5 seeds**, by −73.9%.

The honest reading is narrow and worth stating precisely: **a useful blend weight
exists, and this protocol cannot find it.** Validation carries 63 positives, and
whatever it measures at that size does not transfer six months forward. Quoting
0.2194 would be selection on test — the same error as the original leak wearing a
different hat. The number a deployment would actually get is 0.0457.

## 8. Where the content model knows something EPSS does not

`src/model/complementarity.py` asks where the content model adds signal, rather
than assuming the answer. Beating EPSS outright was never realistic: FIRST trains
it on ~120M vulnerability-days with >1.6M observed exploitation events and ~2,850
features, including exploit-code availability and cross-platform chatter. This
project has ~230 training positives and NVD metadata.

Read as of publication + 1 day, EPSS is a much smaller thing than the monthly
alignment suggested — **only 37 of 54,290 test CVEs score above 0.01**, against
256 before. At that age it has had almost nothing to observe.

| EPSS band | n | exploited | rate | EPSS AP | lift | content AP | lift |
|---|---|---|---|---|---|---|---|
| [0, 0.001) | 49,399 | 129 | 0.261% | 0.0045 | 1.73x | **0.0335** | **12.8x** |
| [0.001, 0.01) | 383 | 39 | 10.18% | 0.3381 | 3.32x | **0.7144** | **7.02x** |
| [0.01, 0.1) | 33 | 25 | 75.76% | 0.6752 | 0.89x | **0.9150** | 1.21x |
| [0.1, 1.01) | 4 | 2 | 50.0% | too few | | too few | |

The lift columns — AP over each band's own base rate — carry the mechanism.
**EPSS barely orders within any band** (1.73x, 3.32x, 0.89x). Its power is the
separation *between* them: base rate climbs 0.261% → 10.2% → 75.8%, a 290-fold
spread across 37 CVEs. An extremely sharp, extremely small head.

The content model is the mirror image: 12.8x lift inside the bottom band, fading
to 1.21x at the top where EPSS is strongest.

Two cuts matter operationally:

- **The discard pile.** 49,782 CVEs (91.7% of test) carry EPSS < 0.01, and
  **168 of the window's 209 exploited CVEs — 80% — are in there.** The content
  model reaches AP 0.0700 against a 0.337% base rate: **20.7x lift**, P@100
  0.140.
- **Cold start is now real.** 4,471 CVEs (8.2%) have no EPSS score at all one day
  after publication, against 1.2% under monthly alignment, and 14 of them were
  exploited. The content model scores them at 10.1x lift. Under the old alignment
  this population was too small to measure and the hypothesis was recorded as
  refuted; at honest timing it exists.

So the two are complementary in a precise sense — **EPSS says which bucket, the
content model says where in the bucket** — and the bucket that matters is the
bottom one, because it holds 80% of the positives and EPSS is nearly blind
inside it.

## 9. The one thing that works: gate, don't mix

`src/model/two_stage.py` tests the shape section 8 implies. EPSS ranks everything
at or above a cutoff; the content model orders the rest. Cutoff chosen on
validation, per budget, at the same corpus fraction — never on test.

| budget | gate c | EPSS hits | gated (5 seeds) | beats EPSS | EPSS recall | gated recall |
|---|---|---|---|---|---|---|
| 100 | 0.1 | 41 | 13.0 ± 1.8 | **0/5** | 19.6% | 6.2% |
| 200 | 0.1 | 54 | 22.4 ± 1.7 | **0/5** | 25.8% | 10.7% |
| 500 | 0.1 | 66 | 45.0 ± 0.6 | **0/5** | 31.6% | 21.5% |
| 1,000 | 0.1 | 72 | 59.4 ± 2.9 | **0/5** | 34.4% | 28.4% |
| 2,000 | 0.1 | 84 | 93.0 ± 0.9 | **5/5** | 40.2% | 44.5% |
| 5,000 | 0.05 † | 90 | **130.0 ± 1.7** | **5/5** | 43.1% | **62.2%** |
| 10,000 | 0.05 † | 107 | **158.4 ± 3.8** | **5/5** | 51.2% | **75.8%** |

† 2 of 8 cutoffs tie on validation, so the pick is arbitrary between them.

**Below ~1,500 the gate is destructive; above it, decisive.** At a 5,000-deep
queue it finds **40 more exploited CVEs** than EPSS alone — recall 43.1% → 62.2%,
5 of 5 seeds, spread ±1.7. At 10,000, +51 and recall 51.2% → 75.8%.

The crossover is the mechanism from section 8 acting rather than observed. Below
it the budget never leaves EPSS's 37-CVE head, so handing ordering to the content
model destroys the between-band sort that is EPSS's entire contribution — hence
−28 hits at budget 100. Above it the budget is deep in the 49,399-CVE tail where
EPSS runs at 1.73x lift and the content model at 12.8x.

### Tie-breaking, and a null that stopped being null

Preserving EPSS's ordering exactly and letting the content model break only exact
ties does nothing at shallow budgets, for a measurable reason: the tied group
straddling each budget boundary holds no positives. At depth that changes.

| budget | tied group | straddling the cut | positives in group | tie-break vs EPSS | vs random tie-break |
|---|---|---|---|---|---|
| 500 | 55 | 18 | 0 | +0 | +0.0 |
| 2,000 | 498 | 489 | 1 | +0 | +0.0 |
| 10,000 | 9,763 | 8,185 | 16 | **+9** | **+8.2** |

At budget 10,000 a tenth of the corpus is tied at EPSS 0.00045, that group holds
16 exploited CVEs, and ordering it by content score recovers 9 more than EPSS's
arbitrary order and 8.2 more than breaking ties at random over 20 draws. Under
the contaminated alignment this comparison was vacuous at every budget; it is not
a large effect, but it is now a real one.

### What this is worth, stated carefully

1. **This is backlog triage, not the weekly patch cycle.** 5,000 CVEs is 9.2% of
   everything published in the 18-month window. A team patching the top 100 is
   actively harmed: 41 hits becomes 13.
2. **Cutoff selection is weak.** Validation holds 63 positives, ties across
   cutoffs are common, and section 7 is a direct demonstration that validation at
   this size can point exactly the wrong way. The gate should be fixed by policy,
   not re-selected per run.
3. **It still does not beat EPSS as a ranker.** PR-AUC is unchanged. The model
   wins one operating regime, not the task.

The defensible claim: **EPSS should decide the top of the queue and a content
model should order its tail; past a ~1,500-CVE budget that recovers roughly 40
additional exploited CVEs per 5,000 patched — 19 recall points — using only NVD
metadata.** Not "beats EPSS", and not nothing.

## Reproducing

```bash
python -m src.schema                  # print the data contract
python -m src.model.evaluate          # original run (leaked EPSS)
python -m src.model.evaluate_pit      # corrected run (fetches ~45 snapshots, cached)
python -m src.model.verify            # split/seed/calibration checks
python -m src.ingestion.prefetch_daily  # ~1,300 daily EPSS snapshots (~1.6 GB, once)
python -m src.model.blend             # rank-blend against EPSS (w chosen on validation)
python -m src.model.complementarity   # per-subpopulation: where does content help?
python -m src.model.two_stage         # gated policy: EPSS first, content orders the rest
python -m src.model.verify_pit_lag    # snapshot-vs-KEV-listing contamination
python -m tests.test_ranking          # ranking primitives
python -m tests.test_schema_gate      # every leak the contract must refuse
```

Sections 7-9 default to `--quantize D` (publication + 1 day). Pass
`--quantize MS` to reproduce the monthly alignment section 6 rejects — the
schema gate will refuse to load the frame, which is the point.
