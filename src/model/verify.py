"""Verification checks that decide whether the headline numbers can be believed.

Four independent checks, each targeting a way this specific result could be fake:

  1. Random-split vs temporal-split gap  — quantifies the optimism you get by
     shuffling. If random CV is much higher, the shuffled number was fiction.
  2. Rolling-origin backtest             — the result must hold across several
     train/test boundaries, not just one lucky cut.
  3. Seed stability                      — variance across seeds must be small
     relative to the gap between models, or the ranking is noise.
  4. Calibration                         — a "probability" that is not calibrated
     cannot be used for expected-value triage decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.model.evaluate import (
    IN_PATH, build_matrices, fit_xgb, precision_recall_at_k, score_ranker,
)

WINDOWS = [("2020-12-31", "2022-12-31"),
           ("2021-12-31", "2023-12-31"),
           ("2022-12-31", "2024-12-31")]


def load() -> pd.DataFrame:
    df = pd.read_parquet(IN_PATH)
    return df[(df["published_year"] >= 2005)].dropna(subset=["published_date"])


def check_random_vs_temporal(df: pd.DataFrame) -> None:
    """The headline claim of this project is that random CV overstates performance."""
    print("=" * 72)
    print("CHECK 1: random split vs temporal split (leakage quantification)")
    print("=" * 72)

    tr = df[df["published_date"] <= "2022-12-31"]
    te = df[(df["published_date"] > "2022-12-31") & (df["published_date"] <= "2024-12-31")]
    Xtr, Xte, _ = build_matrices(tr, te, use_epss=False)
    m = fit_xgb(Xtr, tr["is_kev"].values)
    temporal = average_precision_score(te["is_kev"].values, m.predict_proba(Xte)[:, 1])

    # Same data pooled, then shuffled — the mistake the original train.py makes.
    pooled = pd.concat([tr, te], ignore_index=True)
    y = pooled["is_kev"].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_i, te_i in skf.split(pooled, y):
        a, b = pooled.iloc[tr_i], pooled.iloc[te_i]
        Xa, Xb, _ = build_matrices(a, b, use_epss=False)
        mm = fit_xgb(Xa, y[tr_i])
        scores.append(average_precision_score(y[te_i], mm.predict_proba(Xb)[:, 1]))
    random_cv = float(np.mean(scores))

    print(f"  random 3-fold CV PR-AUC : {random_cv:.4f}")
    print(f"  temporal split PR-AUC   : {temporal:.4f}")
    print(f"  optimism from shuffling : {random_cv - temporal:+.4f} "
          f"({(random_cv / max(temporal, 1e-9) - 1) * 100:+.1f}%)\n")


def check_rolling_origin(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("CHECK 2: rolling-origin backtest (is one split just lucky?)")
    print("=" * 72)
    print(f"{'train<=':<12s} {'test<=':<12s} {'n_test':>8s} {'pos':>6s} "
          f"{'EPSS':>8s} {'XGB':>8s} {'XGB+EPSS':>9s}")
    for tr_end, te_end in WINDOWS:
        tr = df[df["published_date"] <= tr_end]
        te = df[(df["published_date"] > tr_end) & (df["published_date"] <= te_end)]
        if te["is_kev"].sum() < 10:
            print(f"{tr_end:<12s} {te_end:<12s}  too few positives, skipped")
            continue
        yte = te["is_kev"].values
        epss = average_precision_score(yte, np.nan_to_num(te["epss_score"].values, nan=-1))
        row = [epss]
        for use_epss in (False, True):
            Xtr, Xte, _ = build_matrices(tr, te, use_epss)
            m = fit_xgb(Xtr, tr["is_kev"].values)
            row.append(average_precision_score(yte, m.predict_proba(Xte)[:, 1]))
        print(f"{tr_end:<12s} {te_end:<12s} {len(te):>8,} {int(yte.sum()):>6,} "
              f"{row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>9.4f}")
    print()


def check_seed_stability(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("CHECK 3: seed stability (is the model gap bigger than the noise?)")
    print("=" * 72)
    tr = df[df["published_date"] <= "2022-12-31"]
    te = df[(df["published_date"] > "2022-12-31") & (df["published_date"] <= "2024-12-31")]
    yte = te["is_kev"].values
    Xtr, Xte, _ = build_matrices(tr, te, use_epss=False)
    vals = []
    for seed in (0, 1, 2, 3, 4):
        m = fit_xgb(Xtr, tr["is_kev"].values, seed=seed)
        vals.append(average_precision_score(yte, m.predict_proba(Xte)[:, 1]))
    print(f"  PR-AUC over 5 seeds: mean={np.mean(vals):.4f} sd={np.std(vals):.4f} "
          f"min={min(vals):.4f} max={max(vals):.4f}\n")


def check_calibration(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("CHECK 4: calibration (can these scores be used as probabilities?)")
    print("=" * 72)
    tr = df[df["published_date"] <= "2022-12-31"]
    te = df[(df["published_date"] > "2022-12-31") & (df["published_date"] <= "2024-12-31")]
    yte = te["is_kev"].values
    Xtr, Xte, _ = build_matrices(tr, te, use_epss=True)
    m = fit_xgb(Xtr, tr["is_kev"].values)
    p = m.predict_proba(Xte)[:, 1]

    # scale_pos_weight deliberately distorts absolute probabilities to help ranking,
    # so raw output is expected to be inflated. This measures by how much.
    print(f"  mean predicted p : {p.mean():.4f}")
    print(f"  actual positive  : {yte.mean():.4f}")
    print(f"  ratio            : {p.mean() / max(yte.mean(), 1e-9):.1f}x inflated\n")
    print("  decile   n      mean_pred   actual_rate")
    order = np.argsort(-p)
    for d, chunk in enumerate(np.array_split(order, 10)):
        print(f"  {d + 1:>4d}  {len(chunk):>7,}  {p[chunk].mean():>10.4f}  {yte[chunk].mean():>11.4f}")
    print()


def main() -> None:
    df = load()
    check_random_vs_temporal(df)
    check_rolling_origin(df)
    check_seed_stability(df)
    check_calibration(df)


if __name__ == "__main__":
    main()
