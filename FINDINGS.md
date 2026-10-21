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

> Superseded by section 10, which removes all three limits above.

## 10. Banded: keep EPSS's bands, let content order inside them

Section 9's gate keeps EPSS's ordering only above one cutoff — validation chose
0.1 at every budget up to 2,000, which protects **4 CVEs** — and hands the rest to
the content model. That discards the between-band separation section 8 measured
as EPSS's entire contribution (0.26% → 10.2% → 75.8% base rate across bands). The
shallow-budget loss is that discard, not a weakness of the content model.

`banded_score()` keeps every band boundary and drops only the within-band order,
where section 8 shows EPSS at 0.89–3.32x lift and the content model at
1.21–12.8x:

    1. Assign each CVE to an EPSS decade band: [0.1, 1], [0.01, 0.1),
       [0.001, 0.01), [0, 0.001). Unscored CVEs join the bottom band.
    2. Higher band always outranks lower.
    3. Within a band, order by the content model.

**Nothing is selected.** No cutoff, no weight, no validation pick — the decade
edges are the ones `complementarity.py` already reported. Sections 7 and 9 both
failed at the selection step, so this policy is built to have no selection step.

| budget | EPSS | gate (§9) | **banded** | per seed | ≥ EPSS | EPSS recall | banded recall |
|---|---|---|---|---|---|---|---|
| 100 | 41 | 13.0 | **54** | 53.4 ± 1.0 | 5/5 | 19.6% | **25.8%** |
| 200 | 54 | 22.4 | **66** | 65.8 ± 0.4 | 5/5 | 25.8% | **31.6%** |
| 500 | 66 | 45.0 | **69** | 68.8 ± 1.3 | 5/5 | 31.6% | **33.0%** |
| 1,000 | 72 | 59.4 | **87** | 86.0 ± 1.8 | 5/5 | 34.4% | **41.6%** |
| 2,000 | 84 | 93.0 | **107** | 107.2 ± 1.9 | 5/5 | 40.2% | **51.2%** |
| 5,000 | 90 | 130.0 | **143** | 140.2 ± 1.0 | 5/5 | 43.1% | **68.4%** |
| 10,000 | 107 | 158.4 | **167** | 162.6 ± 2.9 | 5/5 | 51.2% | **79.9%** |

(Gate column is the five-seed mean, as in section 9; banded is the
seed-averaged model, with per-seed spread alongside.)

**It never loses to EPSS, at any budget, in any seed** — the crossover is gone.
At the weekly-patch depth of 100, a team catches 53 exploited CVEs instead of 41.
And it beats the gate everywhere, including where the gate was strongest.

**It beats EPSS as a ranker.** PR-AUC **0.2421 ± 0.0008 vs 0.1753 (+38%)**, 5 of
5 seeds — the metric sections 3, 7 and 9 could not move. It also exceeds the
blend's test-selected optimum (0.2194, section 7) without looking at test. On
the validation window, where nothing was tuned either, it wins 5 of 5 seeds
(0.0193 vs 0.0161).

### Checks

- **Band edges.** Half-decade edges give PR-AUC 0.2224 and beat EPSS at every
  budget too (48.0 at 100). Collapsing to two bands at 0.01 gives 0.1759 and
  loses at shallow budgets (31.0 at 100) — the mechanism predicts exactly this:
  the gain needs the head's band structure preserved. Every variant is reported
  by `two_stage.py`; the policy stays on decades.
- **Unscored CVEs.** 4,471 test CVEs (14 positive) have no EPSS one day after
  publication. Ranking them last instead of in the bottom band (seed 0) leaves
  budgets ≤1,000 unchanged and costs 7 hits at 5,000 and 11 at 10,000. Most of
  the gain is the banding; cold-start placement adds to it at depth.
- **Baseline unchanged.** EPSS and gate figures reproduce section 9 exactly.

### What this is worth, stated carefully

1. **The decade edges were not chosen blind.** They come from section 8's band
   table, computed on this test window. They are the obvious edges, half-decades
   also work, and validation agrees — but a fresh window is the proper test.
2. **It is still a content model over NVD metadata.** EPSS decides which band;
   this decides order inside it. It is a complement to EPSS, not a replacement,
   and it inherits EPSS's band boundaries wholesale.
3. **209 positives.** The 5,000-budget gain of +53 is 25 recall points, but the
   whole test window holds 209 exploited CVEs.

The defensible claim: **keeping EPSS's coarse bands and letting a content-only
model order inside them beats EPSS at every patch budget from 100 to 10,000 and
as a global ranker (PR-AUC +38%), in all five seeds, with no tuned parameter.**

> Section 11 re-tests this: one input was itself a leak, the headline model now
> drops it, and the result holds on a window nothing here had touched.

## 11. Verification: trying to break section 10

Section 10 rests on one test window, the window its band edges were read from,
and five seeds — which measure model noise, not sampling noise over 209
positives. `src/model/verify_banded.py` runs six checks. Nothing in it selects
anything; every variant is printed.

### A third leak: NVD enrichment

The content features come from a 2026 NVD mirror. `vendor`, `n_vendors` and
`n_cpe` are parsed from the CPE list, which NVD adds *during analysis* — days to
months after publication — and through its 2024 backlog NVD analysed KEV CVEs
first. Unanalysed CVEs appear as `vendor = UNKNOWN`, `n_cpe = 0`:

| published | not exploited, no CPE | exploited, no CPE |
|---|---|---|
| 2021–2023 | 0.0% | 0.0% |
| 2024 | **18.9%** | 0.0% |
| 2025 | **32.2%** | 0.0% |

"Never analysed" means "not in KEV". Even where analysis happened, the CPE list
is usually absent one day after publication. The schema called these columns
`PIT_SAFE`; they are now `Timing.ENRICHED` and excluded from
`PIT_SAFE_FEATURES`, with a test pinning it. CWE and CVSS show no comparable
missingness gap (unknown-CWE 1.9–20% for exploited vs 4.4–14% for not).

### Ablation — does the result need the leak?

Original section 9-10 test set (54,290 CVEs, 209 exploited), 5 seeds:

| ranker | 100 | 500 | 1,000 | 5,000 | 10,000 | PR-AUC |
|---|---|---|---|---|---|---|
| EPSS alone | 41 | 66 | 72 | 90 | 107 | 0.1753 |
| banded, **CVSS** within band (no model) | 29 | 68 | 75 | 123 | 146 | 0.1414 |
| banded, model: all features (§10) | 53.4 | 68.8 | 86.0 | 140.2 | 162.6 | 0.2421 |
| **banded, model: no CPE-derived** | **52.0** | **69.6** | **92.8** | **140.2** | **162.8** | **0.2484** |
| banded, model: description only | 52.4 | 71.0 | 88.8 | 124.6 | 138.8 | 0.2461 |

**Removing the leaked inputs does not hurt; it helps slightly.** The CPE
features were never carrying the result. The point-in-time-safe model is the
headline from here on. The all-features row reproduces section 10 exactly.

**The model is necessary.** Ordering within bands by CVSS — a free rule —
loses to plain EPSS at shallow budgets (29 vs 41) and on PR-AUC. Description
text alone gets most of the way at shallow budgets; structured fields add the
deep-budget gain.

### Paired bootstrap — sampling noise, not seed noise

2,000 resamples of test CVEs, gain over EPSS:

| budget | 100 | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| gain | +11.3 | +12.4 | +3.6 | +21.3 | +27.1 | +51.1 | +56.0 | +0.074 |
| 95% CI | [+2, +21] | [+3, +24] | [+0, +9] | [+12, +32] | [+14, +41] | [+34, +68] | [+38, +74] | [+0.047, +0.102] |

Every interval excludes a loss. Budget 500 is the weak point: the interval
touches zero and 5.1% of resamples show no gain.

### Permutation null — is it the model, or the bands?

Shuffle the content score inside each band (200 draws), keeping the bands:

| budget | 100 | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| real | 51 | 66 | 70 | 93 | 111 | 141 | 164 |
| shuffled (max of 200) | 33.7 (41) | 43.6 (50) | 66.2 (68) | 67.5 (72) | 70.2 (77) | 78.1 (86) | 91.4 (103) |
| EPSS | 41 | 54 | 66 | 72 | 84 | 90 | 107 |

Bands with a random within-band order do *worse* than EPSS. The real model beats
every one of 200 shuffles at every budget. The gain is the model's.

### Rolling origin and a fresh holdout

Six consecutive windows, each trained only on CVEs published before it, 3 seeds
each. Cells are hits, EPSS → banded:

| test window | CVEs | expl. | 100 | 500 | 2,000 | PR-AUC |
|---|---|---|---|---|---|---|
| 2022 H2 | 12,708 | 50 | 3 → 13.7 | 11 → 20.3 | 22 → 29.0 | 0.034 → 0.055 |
| 2023 H1 | 14,105 | 63 | 4 → 5.3 | 7 → 10.0 | 13 → 28.0 | 0.016 → 0.019 |
| 2023 H2 | 14,701 | 72 | 17 → 30.0 | 31 → 43.3 | 40 → 57.7 | 0.185 → 0.302 |
| 2024 H1 | 20,339 | 60 | 15 → 17.7 | 21 → 31.0 | 23 → 44.7 | 0.163 → 0.203 |
| 2024 H2 | 19,282 | 77 | 21 → 22.0 | 26 → 33.0 | 27 → 49.3 | 0.208 → 0.282 |
| **2025-01 → 08-06** | **26,862** | **116** | **12 → 17.3** | **28 → 39.7** | **58 → 69.7** | **0.073 → 0.113** |

**18 of 18 window-seeds are at least as good as EPSS at every budget**, and
PR-AUC improves in all six windows. The one tie is 2023 H2 at budget 250
(30 vs 30).

The last window is the real out-of-sample test. A 365-day label is mature
through 2025-08-06 against the 2026-08-07 KEV pull. Band edges and
hyperparameters were fixed long before 2025 was loaded. The decision to drop CPE
features was made from the enrichment audit (which includes 2025's missingness
rates, and shows the same gap in 2024) — never from any 2025 *ranking* result.
No ranking on this window was computed before the policy was frozen. Its
bootstrap:

| budget | 100 | 250 | 500 | 1,000 | 2,000 | PR-AUC |
|---|---|---|---|---|---|---|
| gain | +5.6 | +6.6 | +11.0 | +5.9 | +11.6 | +0.041 |
| 95% CI | [+2, +10] | [+0, +13] | [+2, +20] | **[−1, +14]** | [+3, +20] | [+0.016, +0.071] |

PR-AUC +55% (0.073 → 0.113), interval clear of zero. At budget 1,000 the
interval crosses zero (7.5% of resamples show no gain); that budget is not
established on fresh data.

### What remains unverified

1. **Other NVD fields have no dated history here.** Descriptions, CVSS and CWE
   are read from the same 2026 mirror. They show no missingness gap like CPE,
   and description-only reaches the same PR-AUC, but post-publication edits
   cannot be ruled out without NVD's change history. That is the next leak to
   check.
2. **EPSS changed under the fresh window.** FIRST released EPSS v4 in March
   2025; the 2025 baseline is a different model from 2023-24. It is still the
   score a team would actually have had, so the comparison is fair, but the
   windows are not like-for-like with each other.
3. **Small counts.** The fresh window holds 116 exploited CVEs. Gains of 5–12
   hits at shallow budgets are real by the bootstrap, and small.

The defensible claim, now: **ordering CVEs within EPSS bands by a content model
built only from publication-time NVD data beats EPSS on PR-AUC in every one of
six rolling windows, including a 2025 holdout untouched by any design decision
(+55%), and at almost every budget with bootstrap intervals clear of zero.**

## 12. TriageBand: review changes, compared on operational metrics

A review of this project asked for: metrics a patching team feels instead of an
area under a curve; fewer features for a better signal-to-noise ratio; a learned
two-layer gradient boost over EPSS; an EPSS-like model built from scratch with
EPSS-style feature engineering; and per-CVE explanations. `src/model/compare.py`
tests each as a variant of TriageBand (the banded method of sections 10-11).

### Metrics

`src/model/metrics.py` follows the EPSS papers. At an *effort* level — the share
of all CVEs a team remediates — **coverage** is the share of exploited CVEs
caught and **efficiency** the share of patched CVEs that were exploited. Also
reported: the effort needed to reach 50% coverage, and for probability outputs,
Brier score and expected calibration error. PR-AUC remains in the JSON as a
secondary figure.

### Exploit availability, dated

`src/features/exploit_features.py` adds what EPSS leans on hardest — whether
exploit code exists — from two sources that date every item: Exploit-DB
(`date_added`) and PoC-in-GitHub (`created_at`; star counts are current values
and never read). Counts are taken as of publication + 1 day, the same instant as
the EPSS snapshot. The leak this avoids is large:

| exploit present on… | exploited CVEs | not exploited |
|---|---|---|
| counted as of publication + 1 day | 14.0% | 1.50% |
| counted from today's data (a naive pull) | **53.1%** | 3.61% |

A naive count would have been a fourth leak, bigger than the third. Day one,
the signal is almost all GitHub proofs of concept; Exploit-DB rarely lists an
exploit that early and no Metasploit module appeared within a day.

### Decision rule, fixed before results

Highest mean coverage at 1%, 2% and 5% effort over the five development windows
(2022 H2 – 2024 H2). The 2025 window confirms; it does not choose.

| window | V0 EPSS | V1 TriageBand | V2 lean | **V3 +exploit** | V4 lean+exploit | V5 EPSS-lite | V6 stack A | V7 stack B |
|---|---|---|---|---|---|---|---|---|
| 2022 H2 | 0.140 | 0.360 | 0.267 | 0.380 | 0.287 | 0.320 | 0.360 | 0.313 |
| 2023 H1 | 0.095 | 0.143 | 0.122 | 0.148 | 0.127 | 0.212 | 0.344 | 0.259 |
| 2023 H2 | 0.403 | 0.532 | 0.472 | 0.532 | 0.477 | 0.241 | 0.426 | 0.398 |
| 2024 H1 | 0.333 | 0.517 | 0.417 | 0.539 | 0.417 | 0.211 | 0.361 | 0.294 |
| 2024 H2 | 0.316 | 0.437 | 0.364 | 0.416 | 0.333 | 0.195 | 0.325 | 0.281 |
| **dev mean** | 0.257 | 0.398 | 0.328 | **0.403** | 0.328 | 0.236 | 0.363 | 0.309 |
| **2025 holdout** | 0.282 | 0.356 | 0.342 | **0.365** | 0.342 | 0.141 | 0.322 | 0.299 |

One seed per variant (seed spread in earlier sections was ±0.002 PR-AUC); the
chosen variant is trained with five for deployment.

**V3 is chosen and the holdout confirms it.** On 2025 (26,862 CVEs, 116
exploited), against EPSS:

| effort | CVEs patched | EPSS caught | TriageBand caught | efficiency |
|---|---|---|---|---|
| 1% | 269 | 23 (20%) | **34 (29%)** | 8.6% → 12.6% |
| 2% | 537 | 29 (25%) | **43 (37%)** | 5.4% → 8.0% |
| 5% | 1,343 | 46 (40%) | **50 (43%)** | 3.4% → 3.7% |
| 10% | 2,686 | 62 (53%) | **87 (75%)** | 2.3% → 3.2% |

To reach 50% coverage: **1,564 patches instead of 1,907 (18% less work)**.

### What each review suggestion did

- **Exploit features (V3 vs V1): a small gain, not a large one.** Better in 3 of 5
  development windows, equal in one, worse in one (2024 H2), and +0.009 on 2025
  — about one more exploited CVE per budget level. Kept because the rule picked
  it; the honest description is "modest".
- **Feature reduction (V2, V4): consistently worse.** Cutting to the top 50
  features by gain lost 5–10 points of coverage in every window. The signal is
  spread thinly across many description terms, and gain from one model on a few
  hundred positives is a noisy ranking (its top 50 includes terms like
  `16 ipados`). Fewer features did not mean less noise here. K = 50 was fixed in
  advance; other reduction methods were not tried and remain open.
- **Two-layer boost (V6, V7): beats EPSS, loses to the bands.** Both readings of
  "layer-1 score + EPSS" were built — with five extra features (A) and without
  (B) — as shallow boosts constrained to be monotone in every input. A is the
  better of the two. Neither matches the fixed banded rule on average, but A wins
  decisively in 2023 H1 (0.344 vs 0.148), the window where EPSS itself was
  weakest: a learned combination can lean away from EPSS when EPSS is having a
  bad period, and a fixed rule cannot. Worth revisiting with more training data.
- **EPSS reimplemented from public data (V5): close in development, short in
  2025.** With no EPSS input, content plus dated exploit features reaches 0.236
  against real EPSS's 0.257 on development windows, and wins 2 of 5 — but falls
  to 0.141 against 0.282 in 2025. FIRST's private telemetry is worth a great
  deal; public day-one data recovers part of it.
- **Calibration.** On 2025, EPSS averages a 0.08% prediction against a 0.43%
  realised rate — it forecasts 30-day exploitation, this label is one year. The
  stacks are closer (0.15%, Brier 0.0041 vs 0.0042). TriageBand itself is a
  ranking, not a probability, and makes no calibration claim.
- **Explanations.** `src/model/explain.py` turns exact per-CVE SHAP values from
  the trees into the three strongest present reasons, shown with every ranked
  CVE, e.g. *description mentions "unauthenticated"; CVSS base score 10.0*.

### Train / serve difference

The model is trained on exploit counts as of publication + 1 day. Live ranking
counts everything known today, because a defender scoring a month-old CVE can
see its exploits. Exploit counts only grow with age, so live scores for older
CVEs lean higher than training ever saw — the right direction for a
prioritiser, but the evaluation above measures the day-one setting only.

## Reproducing

```bash
python -m src.schema                  # print the data contract
python -m src.model.evaluate          # original run (leaked EPSS)
python -m src.model.evaluate_pit      # corrected run (fetches ~45 snapshots, cached)
python -m src.model.verify            # split/seed/calibration checks
python -m src.ingestion.prefetch_daily --end 2025-08-06  # ~1,530 daily EPSS snapshots (once)
python -m src.model.blend             # rank-blend against EPSS (w chosen on validation)
python -m src.model.complementarity   # per-subpopulation: where does content help?
python -m src.model.two_stage         # gated + banded policies vs EPSS (sections 9-10)
python -m src.model.verify_banded     # section 11: leak audit, ablation, bootstrap, null, rolling, 2025 holdout
python -m src.model.compare           # section 12: TriageBand variants on coverage / efficiency
python -m src.model.verify_pit_lag    # snapshot-vs-KEV-listing contamination
python -m tests.test_ranking          # ranking primitives
python -m tests.test_schema_gate      # every leak the contract must refuse
```

Sections 7-9 default to `--quantize D` (publication + 1 day). Pass
`--quantize MS` to reproduce the monthly alignment section 6 rejects — the
schema gate will refuse to load the frame, which is the point.
