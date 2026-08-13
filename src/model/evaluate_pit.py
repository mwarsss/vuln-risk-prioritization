"""Point-in-time re-evaluation: what the ranker is actually worth in deployment.

`evaluate.py` splits temporally but reads `epss_score` from a single 2026-08-09
bulk pull, so every historical row carries an EPSS value computed after the
exploitation it is meant to predict. This script re-runs the same comparison
with EPSS read from the archive snapshot nearest each CVE's own publication
month, which is the information a triage team would have held at decision time.

Everything else is held constant against `evaluate.py` — same features, same
model, same metrics — so the delta between the two outputs is attributable to
the EPSS timing alone.

Window is bounded by the EPSS archive (2021-04-14 onward), so both train and
test are much smaller here than in the leaked run. Compare this script's
`EPSS (leaked)` row against its `EPSS (point-in-time)` row for the like-for-like
number; the headline in `eval_results.json` is not comparable, it is scored on a
different and larger window.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.ingestion.epss_history import ARCHIVE_START, attach_pit_epss
from src.model.evaluate import IN_PATH, build_matrices, fit_xgb, score_ranker
from src.schema import build_horizon_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2023-06-30")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--quantize", default="MS", help="MS=monthly snapshots, D=daily")
    ap.add_argument("--horizon-days", type=int, default=365,
                    help="label positive if added to KEV within N days of publication; "
                         "0 keeps the open-ended 'in KEV as of pull date' target")
    ap.add_argument("--out", default="data/processed/eval_results_pit.json")
    args = ap.parse_args()

    df = pd.read_parquet(IN_PATH).dropna(subset=["published_date"])
    df["is_kev"] = df["is_kev"].astype(bool)
    df["published_date"] = pd.to_datetime(df["published_date"])

    if args.horizon_days > 0:
        if "kev_date_added" not in df.columns:
            raise SystemExit(
                "kev_date_added missing — run `python -m src.features.prepare_kaggle` "
                "to rejoin the CISA feed, or pass --horizon-days 0"
            )
        as_of = pd.to_datetime(df["kev_date_added"]).max()
        label = build_horizon_label(df, args.horizon_days, as_of=as_of)
        immature = int(label.isna().sum())
        flipped = int((df["is_kev"] & label.fillna(False).eq(False)).sum())
        print(f"label: exploited within {args.horizon_days}d of publication "
              f"(catalog as-of {as_of.date()})")
        print(f"  {flipped:,} CVEs are in KEV but fell outside the horizon -> negative")
        print(f"  {immature:,} rows dropped as label-immature\n")
        df = df.assign(is_kev=label)
        df = df[df["is_kev"].notna()]
        df["is_kev"] = df["is_kev"].astype(bool)
    else:
        print("label: open-ended 'in KEV as of pull date'\n")

    # Keep the leaked EPSS under a different name so both can be scored side by side.
    df = df.rename(columns={"epss_score": "epss_leaked"})
    df = df.drop(columns=[c for c in ("epss_perc", "epss_as_of") if c in df.columns])

    # The archive bounds the usable window on both sides.
    df = df[df["published_date"] >= ARCHIVE_START]
    df = df[df["published_date"] <= args.test_end]
    print(f"{len(df):,} CVEs published {ARCHIVE_START.date()} .. {args.test_end}")

    print("attaching point-in-time EPSS (one snapshot per publication month)...")
    df = attach_pit_epss(df, quantize=args.quantize)
    cov = df["epss_score"].notna().mean()
    print(f"point-in-time EPSS coverage: {cov:.1%}\n")

    train = df[df["published_date"] <= args.train_end]
    test = df[df["published_date"] > args.train_end]
    ytr, yte = train["is_kev"].values, test["is_kev"].values
    print(f"train: {len(train):,} rows, {ytr.sum():,} KEV ({ytr.mean():.3%})")
    print(f"test : {len(test):,} rows, {yte.sum():,} KEV ({yte.mean():.3%})\n")

    if yte.sum() < 10:
        raise SystemExit("too few positives in the test window to score meaningfully")

    results = [
        score_ranker("CVSS base score", yte, test["base_score"].values),
        score_ranker("EPSS (leaked)", yte, test["epss_leaked"].values),
        score_ranker("EPSS (point-in-time)", yte, test["epss_score"].values),
    ]

    for use_epss, tag in ((False, "XGB (no EPSS)"),
                          (True, "XGB + EPSS (point-in-time)")):
        Xtr, Xte, _ = build_matrices(train, test, use_epss)
        model = fit_xgb(Xtr, ytr)
        results.append(score_ranker(tag, yte, model.predict_proba(Xte)[:, 1]))

    # And the leaked stack, so the gap is visible in one table.
    tr_l = train.assign(epss_score=train["epss_leaked"])
    te_l = test.assign(epss_score=test["epss_leaked"])
    Xtr, Xte, _ = build_matrices(tr_l, te_l, use_epss=True)
    results.append(score_ranker("XGB + EPSS (leaked)", yte,
                                fit_xgb(Xtr, ytr).predict_proba(Xte)[:, 1]))

    base = yte.mean()
    print(f"test base rate: {base:.4%}\n")
    hdr = f"{'ranker':<28s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'P@100':>7s} {'P@500':>7s} {'R@1000':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<28s} {r.pr_auc:>8.4f} {r.roc_auc:>8.4f} "
              f"{r.prec_at_k[100]:>7.3f} {r.prec_at_k[500]:>7.3f} {r.recall_at_k[1000]:>7.3f}")

    lookup = {r.name: r.pr_auc for r in results}
    leaked, pit = lookup["EPSS (leaked)"], lookup["EPSS (point-in-time)"]
    print(f"\nEPSS PR-AUC collapses {leaked:.4f} -> {pit:.4f} "
          f"({(pit / max(leaked, 1e-9) - 1) * 100:+.1f}%) once the score is read "
          "as of publication instead of 2026.")

    with open(args.out, "w") as f:
        json.dump({"base_rate": float(base), "train_end": args.train_end,
                   "test_end": args.test_end, "n_train": len(train),
                   "n_test": len(test), "pit_epss_coverage": float(cov),
                   "quantize": args.quantize, "horizon_days": args.horizon_days,
                   "results": [r.__dict__ for r in results]},
                  f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
