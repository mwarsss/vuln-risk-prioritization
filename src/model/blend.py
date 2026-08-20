"""Rank-blend the content model with EPSS instead of feeding EPSS as a feature.

Feeding `epss_score` in as one column among ~3,000 TF-IDF features made things
worse: XGB + EPSS scored 0.1866 +- 0.0019 PR-AUC against EPSS alone at 0.2701,
losing in 0/5 seeds. The hypothesis is capacity, not concept — roughly 300
positive examples against 3,018 encoded columns is ~10 features per positive,
so the tree ensemble has plenty of room to fit noise and dilute a strong input.

A blend sidesteps that entirely. The content model never sees EPSS, so it cannot
drown it; the two scores are combined afterwards, at the ranking level.

Ranks rather than raw scores because the two are not on comparable scales:
`scale_pos_weight` deliberately inflates XGB's probabilities (measured ~180x in
verify.py) while EPSS is a calibrated probability. Averaging those directly lets
XGB dominate for reasons that have nothing to do with signal. Percentile ranks
put both on [0,1] and make the weight mean what it looks like it means.

Weight selection is the part that is easy to get wrong. Sweeping w on the test
set and reporting the best value is selection on test — it would inflate the
result the same way the EPSS leak did, just more subtly. So w is chosen on a
validation slice carved off the end of train, and the test number is reported at
that w only. The full test sweep is printed too, but as diagnosis, not as a
result to quote.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from src.model.dataset import load_pit_frame, temporal_split
from src.model.evaluate import build_matrices, fit_xgb, precision_recall_at_k

WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1]. NaN ranks last — absent evidence, not low risk."""
    x = np.asarray(x, dtype=float)
    filled = np.where(np.isnan(x), -np.inf, x)
    return rankdata(filled, method="average") / len(filled)


def evaluate_blend(y: np.ndarray, r_model: np.ndarray, r_epss: np.ndarray,
                   w: float) -> tuple[float, float]:
    """w=0 is pure EPSS, w=1 is pure content model."""
    blended = w * r_model + (1.0 - w) * r_epss
    return (average_precision_score(y, blended),
            precision_recall_at_k(y, blended, 100)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", default="2022-12-31", help="content model trains on/before this")
    ap.add_argument("--val-end", default="2023-06-30", help="w is chosen on (fit-end, val-end]")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quantize", default="D",
                    help="EPSS snapshot alignment; 'D' = publication + 1 day")
    ap.add_argument("--out", default="data/processed/blend_results.json")
    args = ap.parse_args()

    df = load_pit_frame(args.horizon_days, args.test_end, quantize=args.quantize)
    fit, val, test = temporal_split(df, args.fit_end, args.val_end)
    yfit, yval, ytest = fit["is_kev"].values, val["is_kev"].values, test["is_kev"].values

    r_epss_val = pct_rank(val["epss_score"].values)
    r_epss_test = pct_rank(test["epss_score"].values)

    epss_test_ap = average_precision_score(ytest, np.nan_to_num(test["epss_score"].values, nan=-1))
    epss_test_p100 = precision_recall_at_k(ytest, np.nan_to_num(test["epss_score"].values, nan=-1), 100)[0]

    # Content model never sees EPSS — that is the whole point of the design.
    # Encoders are fit on `fit` in both calls, so Xfit is the same matrix and one
    # model per seed can score both slices.
    Xfit, Xval, _ = build_matrices(fit, val, use_epss=False)
    _, Xtest, _ = build_matrices(fit, test, use_epss=False)

    chosen_ws, test_aps, test_p100s, solo_aps = [], [], [], []
    sweep_by_seed = []

    for seed in range(args.seeds):
        model = fit_xgb(Xfit, yfit, seed=seed)

        r_model_val = pct_rank(model.predict_proba(Xval)[:, 1])
        r_model_test = pct_rank(model.predict_proba(Xtest)[:, 1])

        # Pick w on validation only.
        val_aps = [evaluate_blend(yval, r_model_val, r_epss_val, w)[0] for w in WEIGHTS]
        w_star = float(WEIGHTS[int(np.argmax(val_aps))])

        ap, p100 = evaluate_blend(ytest, r_model_test, r_epss_test, w_star)
        chosen_ws.append(w_star)
        test_aps.append(ap)
        test_p100s.append(p100)
        solo_aps.append(average_precision_score(ytest, r_model_test))
        sweep_by_seed.append([evaluate_blend(ytest, r_model_test, r_epss_test, w)[0]
                              for w in WEIGHTS])

    sweep = np.array(sweep_by_seed).mean(axis=0)

    print("test PR-AUC across the weight sweep (mean over seeds) — diagnosis only:")
    print(f"  {'w':>5s}  {'PR-AUC':>8s}   (w=0 pure EPSS, w=1 pure content model)")
    for w, ap in zip(WEIGHTS, sweep):
        mark = "  <- best on test" if ap == sweep.max() else ""
        print(f"  {w:>5.2f}  {ap:>8.4f}{mark}")

    print(f"\nw chosen on validation: {chosen_ws} (mean {np.mean(chosen_ws):.2f})")
    print("\n" + "=" * 66)
    print(f"{'ranker':<34s} {'PR-AUC':>10s} {'P@100':>8s}")
    print("-" * 66)
    print(f"{'EPSS (point-in-time)':<34s} {epss_test_ap:>10.4f} {epss_test_p100:>8.3f}")
    print(f"{'XGB content only':<34s} "
          f"{np.mean(solo_aps):>10.4f} {'':>8s}")
    print(f"{'XGB + EPSS as feature (prior run)':<34s} {0.1866:>10.4f} {0.368:>8.3f}")
    print(f"{'rank blend (w from validation)':<34s} "
          f"{np.mean(test_aps):>10.4f} {np.mean(test_p100s):>8.3f}")
    print("=" * 66)

    delta = np.mean(test_aps) - epss_test_ap
    beats = sum(a > epss_test_ap for a in test_aps)
    print(f"\nblend vs EPSS alone: {delta:+.4f} PR-AUC "
          f"({delta / epss_test_ap * 100:+.1f}%), beats it in {beats}/{args.seeds} seeds")
    print(f"blend sd across seeds: {np.std(test_aps):.4f}")

    with open(args.out, "w") as f:
        json.dump({
            "horizon_days": args.horizon_days,
            "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
            "test_base_rate": float(ytest.mean()),
            "epss_pr_auc": float(epss_test_ap), "epss_p_at_100": float(epss_test_p100),
            "xgb_content_only_pr_auc": float(np.mean(solo_aps)),
            "chosen_w": chosen_ws,
            "blend_pr_auc_mean": float(np.mean(test_aps)),
            "blend_pr_auc_sd": float(np.std(test_aps)),
            "blend_p_at_100_mean": float(np.mean(test_p100s)),
            "beats_epss_in_seeds": int(beats),
            "test_sweep": {str(w): float(a) for w, a in zip(WEIGHTS, sweep)},
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
