"""Translate ranking quality into remediation workload.

Security teams do not buy PR-AUC. They buy "how many tickets must my team work
to cover the vulnerabilities that actually get exploited". This script converts
each ranking strategy into that number, on the held-out 2023-2024 window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.evaluate import build_matrices, fit_xgb, IN_PATH

TRAIN_END, TEST_END = "2022-12-31", "2024-12-31"
TARGET_RECALLS = (0.50, 0.80, 0.90, 0.95)


def workload_for_recall(y: np.ndarray, scores: np.ndarray, target: float) -> tuple[int, float]:
    """How deep into the ranked list must you go to catch `target` of all KEV CVEs?"""
    order = np.argsort(-np.nan_to_num(scores, nan=-1.0), kind="stable")
    cum = np.cumsum(y[order])
    need = target * y.sum()
    idx = int(np.searchsorted(cum, need) + 1)
    return idx, idx / len(y)


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    df = df[df["published_year"] >= 2005].dropna(subset=["published_date"])
    train = df[df["published_date"] <= TRAIN_END]
    test = df[(df["published_date"] > TRAIN_END) & (df["published_date"] <= TEST_END)]
    y = test["is_kev"].values
    n, pos = len(test), int(y.sum())

    strategies: dict[str, np.ndarray] = {
        "CVSS base score": test["base_score"].values,
        "EPSS score": test["epss_score"].values,
    }
    for use_epss in (False, True):
        Xtr, Xte, _ = build_matrices(train, test, use_epss)
        m = fit_xgb(Xtr, train["is_kev"].values)
        strategies["XGBoost + EPSS" if use_epss else "XGBoost (no EPSS)"] = \
            m.predict_proba(Xte)[:, 1]

    print(f"Test window {TRAIN_END} -> {TEST_END}")
    print(f"{n:,} CVEs published, {pos} later confirmed exploited (KEV)\n")

    print("--- Severity-tier triage (what most teams do today) ---")
    for tiers in (["CRITICAL"], ["CRITICAL", "HIGH"]):
        sub = test[test["base_severity"].isin(tiers)]
        label = "+".join(tiers)
        print(f"  patch all {label:<15s}: {len(sub):>7,} CVEs "
              f"({len(sub)/n:>5.1%} of backlog) -> catches {sub['is_kev'].sum():>3}/{pos} "
              f"KEV ({sub['is_kev'].sum()/pos:>5.1%})")

    print("\n--- Ranked triage: CVEs to review to reach a target KEV coverage ---")
    hdr = f"{'strategy':<20s}" + "".join(f"{int(t*100):>12d}%" for t in TARGET_RECALLS)
    print(hdr)
    print("-" * len(hdr))
    for name, s in strategies.items():
        row = f"{name:<20s}"
        for t in TARGET_RECALLS:
            k, frac = workload_for_recall(y, s, t)
            row += f"{k:>8,} ({frac*100:>2.0f}%)"
        print(row)

    print("\n--- Top-decile concentration ---")
    for name, s in strategies.items():
        order = np.argsort(-np.nan_to_num(s, nan=-1.0), kind="stable")
        top = order[: n // 10]
        print(f"  {name:<20s} top 10% ({len(top):,} CVEs) contains "
              f"{y[top].sum():>3}/{pos} KEV ({y[top].sum()/pos:>5.1%})")


if __name__ == "__main__":
    main()
