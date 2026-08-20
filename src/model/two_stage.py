"""Does the content model's niche convert into a policy that beats EPSS at a fixed budget?

`complementarity.py` established the niche: 65% of exploited CVEs land in the
population EPSS scores below 0.01, and inside that population the content model
reaches AP 0.0714 against a 0.279% base rate — 25.6x lift. The effect is
sharpest one band further down, among CVEs scored below 0.001, where EPSS runs
at 1.31x lift over that band's base rate and the content model at 19.6x.

That is necessary but not sufficient. A triage team does not rank a
sub-population, it spends a fixed patch budget over everything. `blend.py`
already showed the naive way to use this fails: a single global weight averages
the benefit away and PR-AUC falls monotonically. The right shape is a *gate*, not
a mixture — EPSS decides first, and the content model only orders what EPSS
declined to distinguish.

Policy under test, at budget B:

    1. Take CVEs with EPSS >= c, ordered by EPSS, up to B.
    2. If budget remains, fill it from CVEs with EPSS < c, ordered by the
       content model.

Pure EPSS is the same policy with step 2 ordered by EPSS instead. So the
comparison isolates exactly one thing: who orders the discard pile.

`c` is chosen on a validation slice carved off the end of train, never on test —
same discipline as blend.py. Reporting the best cutoff found on test would be the
same selection error as the original EPSS leak, just wearing a different hat.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.model.dataset import load_pit_frame, temporal_split
from src.model.evaluate import build_matrices, fit_xgb

CUTOFFS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
BUDGETS = [100, 200, 500, 1000, 2000, 5000, 10000]
TIE_TRIALS = 20


def tiebreak_score(epss: np.ndarray, content: np.ndarray) -> np.ndarray:
    """EPSS ordering preserved exactly; the content model only breaks exact ties.

    This is the surgical version of the section-7 finding. The band analysis
    showed the content model orders CVEs *within* a region where EPSS is
    constant — and 'where EPSS is constant' means, precisely, tied scores.
    Gating discards EPSS's ordering across a whole range; this discards none of
    it.
    """
    epss = np.nan_to_num(epss, nan=-1.0)
    order = np.lexsort((-content, -epss))  # primary EPSS desc, secondary content desc
    score = np.empty(len(epss), dtype=float)
    score[order] = np.arange(len(epss), 0, -1)
    return score


def random_tiebreak_score(epss: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Control: same EPSS ordering, ties broken arbitrarily.

    Plain `argsort` breaks ties by row order, which here is publication order —
    not neutral. Comparing against a random tie-break isolates what the content
    model contributes from what any tie-break contributes.
    """
    epss = np.nan_to_num(epss, nan=-1.0)
    return tiebreak_score(epss, rng.random(len(epss)))


def gated_score(epss: np.ndarray, content: np.ndarray, cutoff: float) -> np.ndarray:
    """One composite score encoding 'EPSS first, then content model'.

    Everything at or above the cutoff outranks everything below it. Within each
    tier, ordering comes from that tier's own score. Ranks are mapped into [0,1)
    so the +1 offset cleanly separates the tiers.
    """
    epss = np.nan_to_num(epss, nan=-1.0)
    above = epss >= cutoff
    out = np.empty(len(epss), dtype=float)

    def unit_rank(vals: np.ndarray) -> np.ndarray:
        if len(vals) == 0:
            return vals
        order = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
        return order / max(len(vals), 1)

    out[above] = 1.0 + unit_rank(epss[above])
    out[~above] = unit_rank(content[~above])
    return out


def recall_at(y: np.ndarray, scores: np.ndarray, k: int) -> tuple[int, float]:
    k = min(k, len(scores))
    top = np.argsort(-scores, kind="stable")[:k]
    hits = int(y[top].sum())
    return hits, hits / max(int(y.sum()), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", default="2022-12-31")
    ap.add_argument("--val-end", default="2023-06-30")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quantize", default="D",
                    help="EPSS snapshot alignment: 'D' = publication + 1 day (default, "
                         "least leakage); 'MS' = start of the following month (legacy, "
                         "median 16-day lag)")
    ap.add_argument("--out", default="data/processed/two_stage.json")
    args = ap.parse_args()

    df = load_pit_frame(args.horizon_days, args.test_end, quantize=args.quantize)
    fit, val, test = temporal_split(df, args.fit_end, args.val_end)
    yfit, yval, ytest = fit["is_kev"].values, val["is_kev"].values, test["is_kev"].values
    epss_val, epss_test = val["epss_score"].values, test["epss_score"].values

    # Both calls fit the encoders on the same `fit` frame, so Xfit is identical
    # between them — train once per seed and score both matrices rather than
    # fitting the same model twice.
    Xfit, Xval, _ = build_matrices(fit, val, use_epss=False)
    _, Xtest, _ = build_matrices(fit, test, use_epss=False)
    # Keep the per-seed predictions, not just their mean: a headline delta that
    # only holds for the seed-averaged model is not a result this project quotes.
    models = [fit_xgb(Xfit, yfit, seed=s) for s in range(args.seeds)]
    preds_val = [m.predict_proba(Xval)[:, 1] for m in models]
    preds_test = [m.predict_proba(Xtest)[:, 1] for m in models]
    pred_val = np.mean(preds_val, axis=0)
    pred_test = np.mean(preds_test, axis=0)

    epss_only = np.nan_to_num(epss_test, nan=-1.0)

    # --- is there any tie structure for a tie-break to exploit? ---
    # Reported inline because it is the only thing that makes the tie-break
    # column interpretable: if the group straddling a budget holds no positives,
    # every tie-break scores identically and the comparison is vacuous.
    n_unique = len(np.unique(epss_only))
    print(f"EPSS takes {n_unique:,} distinct values across {len(test):,} test CVEs.")
    print("tie structure at each budget boundary:")
    es = epss_only[np.argsort(-epss_only, kind="stable")]
    tie_diag = {}
    for b in BUDGETS:
        if b > len(es):
            continue
        boundary = float(es[b - 1])
        grp = epss_only == boundary
        straddle = int(grp.sum()) - int((es[:b] == boundary).sum())
        pos_in_grp = int(ytest[grp].sum())
        tie_diag[str(b)] = {"boundary_epss": boundary, "tied_group": int(grp.sum()),
                            "straddling": straddle, "positives_in_group": pos_in_grp}
        print(f"  budget {b:>6,}: EPSS={boundary:.5f}  tied group={int(grp.sum()):>5,} "
              f"({straddle:>5,} beyond the cut)  positives in group={pos_in_grp}")
    print()

    # --- policy C: preserve EPSS ordering, break only exact ties ---
    tied = tiebreak_score(epss_test, pred_test)
    rng = np.random.default_rng(0)
    rand_runs = [random_tiebreak_score(epss_test, rng) for _ in range(TIE_TRIALS)]

    # --- gate cutoff chosen on validation, separately for each budget ---
    # The best cutoff depends on how deep the queue goes, so a single cutoff
    # picked at one reference budget does not transfer. Validation uses the same
    # *fraction* of its corpus, not the same absolute count, since it is smaller.
    scale = len(val) / len(test)
    chosen, val_curves = {}, {}
    print("gate cutoff chosen on validation (same corpus fraction), per budget:")
    for b in BUDGETS:
        vb = max(int(round(b * scale)), 1)
        scores = [recall_at(yval, gated_score(epss_val, pred_val, c), vb)[1] for c in CUTOFFS]
        chosen[b] = CUTOFFS[int(np.argmax(scores))]
        val_curves[str(b)] = dict(zip(map(str, CUTOFFS), map(float, scores)))
        # How many cutoffs tie at the top matters: argmax silently returns the
        # lowest, so a "chosen" cutoff can be an arbitrary pick among equals
        # rather than a real preference. Say so rather than let the table imply
        # more selection than happened.
        n_tied = sum(abs(s - max(scores)) < 1e-12 for s in scores)
        note = f"  ({n_tied} of {len(CUTOFFS)} cutoffs tie here — pick is arbitrary)" \
            if n_tied > 1 else ""
        print(f"  budget {b:>6,} (val {vb:>5,}): c = {chosen[b]}{note}")
    print()

    print("=" * 116)
    print(f"{'budget':>8s} {'gate c':>8s} {'EPSS':>7s} {'rand-tie':>9s} {'tie-brk':>8s} "
          f"{'gated':>7s} {'delta':>7s} {'per-seed gated':>20s} {'wins':>6s} "
          f"{'EPSS rec':>9s} {'gated rec':>10s}")
    print("-" * 116)
    rows = []
    for b in BUDGETS:
        c_b = chosen[b]
        h_e, r_e = recall_at(ytest, epss_only, b)
        h_g, r_g = recall_at(ytest, gated_score(epss_test, pred_test, c_b), b)
        h_t, r_t = recall_at(ytest, tied, b)
        rand_hits = [recall_at(ytest, s, b)[0] for s in rand_runs]
        h_r = float(np.mean(rand_hits))

        seed_hits = [recall_at(ytest, gated_score(epss_test, p, c_b), b)[0]
                     for p in preds_test]
        wins = sum(h > h_e for h in seed_hits)

        rows.append({"budget": b, "cutoff": c_b, "epss_hits": h_e, "gated_hits": h_g,
                     "tiebreak_hits": h_t, "random_tiebreak_hits_mean": h_r,
                     "random_tiebreak_hits_sd": float(np.std(rand_hits)),
                     "gated_hits_per_seed": seed_hits,
                     "gated_hits_seed_mean": float(np.mean(seed_hits)),
                     "gated_hits_seed_sd": float(np.std(seed_hits)),
                     "seeds_beating_epss": wins,
                     "epss_recall": r_e, "gated_recall": r_g, "tiebreak_recall": r_t})
        print(f"{b:>8,} {c_b:>8.4f} {h_e:>7,} {h_r:>9.1f} {h_t:>8,} "
              f"{h_g:>7,} {h_g - h_e:>+7,} "
              f"{np.mean(seed_hits):>13.1f} ±{np.std(seed_hits):>4.1f} "
              f"{wins:>4d}/{args.seeds} {r_e:>8.1%} {r_g:>9.1%}")
    print("=" * 116)

    total_pos = int(ytest.sum())
    print(f"\n{total_pos} exploited CVEs in the test window. Columns are how many a team "
          f"patching\nthat many CVEs would have caught. 'rand-tie' is EPSS with ties "
          f"broken at random\n(mean of {TIE_TRIALS} draws) — the fair control for "
          f"'tie-break'.")

    with open(args.out, "w") as f:
        json.dump({
            "horizon_days": args.horizon_days,
            "cutoff_by_budget": {str(b): c for b, c in chosen.items()},
            "cutoff_grid": CUTOFFS,
            "val_recall_by_cutoff": val_curves,
            "n_test": len(test), "test_positives": total_pos,
            "test_base_rate": float(ytest.mean()),
            "n_distinct_epss": int(n_unique),
            "tie_structure": tie_diag,
            "seeds": args.seeds, "tie_trials": TIE_TRIALS,
            "results": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
