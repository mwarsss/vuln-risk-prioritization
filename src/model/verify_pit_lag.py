"""Does the point-in-time EPSS snapshot postdate the KEV listing it predicts?

Section 5 fixed a coverage bug by aligning each CVE to the EPSS snapshot at the
start of the month *after* publication — CVEs are largely absent from the
snapshot published in their own month. That fix buys 98.5% coverage, but it also
shifts the feature 1-31 days forward of publication, and section 4 measured
median days-to-KEV at 15 with p25 at 0.

If a CVE reaches the CISA catalogue before the snapshot used to score it, EPSS
had already observed the exploitation it is being asked to predict. That is the
same failure this project exists to eliminate, at a one-month scale instead of a
three-year one, and it would contaminate the EPSS baseline that sections 6-8 are
all measured against.

This measures it directly rather than reasoning about it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingestion.epss_history import ARCHIVE_START, attach_pit_epss
from src.model.evaluate import IN_PATH
from src.schema import build_horizon_label

TRAIN_END = "2023-06-30"
TEST_END = "2024-12-31"


def main() -> None:
    df = pd.read_parquet(IN_PATH).dropna(subset=["published_date"])
    df["published_date"] = pd.to_datetime(df["published_date"])
    as_of = pd.to_datetime(df["kev_date_added"]).max()
    df = df.assign(is_kev=build_horizon_label(df, 365, as_of=as_of))
    df = df[df["is_kev"].notna()]
    df["is_kev"] = df["is_kev"].astype(bool)
    df = df[(df["published_date"] >= ARCHIVE_START) & (df["published_date"] <= TEST_END)]
    df = df.rename(columns={"epss_score": "epss_leaked"}).drop(
        columns=[c for c in ("epss_perc", "epss_as_of") if c in df.columns])
    df = attach_pit_epss(df, quantize="MS")

    test = df[df["published_date"] > TRAIN_END].copy()
    test["kev_date_added"] = pd.to_datetime(test["kev_date_added"])
    test["epss_as_of"] = pd.to_datetime(test["epss_as_of"])

    lag = (test["epss_as_of"] - test["published_date"]).dt.days
    print(f"test rows: {len(test):,}   positives: {int(test['is_kev'].sum())}")
    print("\nsnapshot lag behind publication (days):")
    print(f"  min {lag.min():.0f}  p25 {lag.quantile(.25):.0f}  median {lag.median():.0f}  "
          f"p75 {lag.quantile(.75):.0f}  max {lag.max():.0f}  mean {lag.mean():.1f}")

    kev = test[test["kev_date_added"].notna()].copy()
    print(f"\nrows carrying a kev_date_added: {len(kev):,}")

    contaminated = kev["kev_date_added"] <= kev["epss_as_of"]
    n_c = int(contaminated.sum())
    print(f"  KEV listing on or before the EPSS snapshot used to score it: "
          f"{n_c:,} ({n_c / max(len(kev), 1):.1%})")
    print("  -> for these, EPSS had already seen the exploitation it 'predicts'")

    # Restrict to positives inside the evaluated horizon: those are the rows that
    # actually drive PR-AUC, so contamination there is what matters.
    pos = kev[kev["is_kev"]]
    c_pos = int((pos["kev_date_added"] <= pos["epss_as_of"]).sum())
    print(f"\nin-horizon positives: {len(pos):,}, contaminated: {c_pos:,} "
          f"({c_pos / max(len(pos), 1):.1%})")

    if len(pos):
        clean = pos[pos["kev_date_added"] > pos["epss_as_of"]]
        dirty = pos[pos["kev_date_added"] <= pos["epss_as_of"]]
        for name, grp in (("clean", clean), ("contaminated", dirty)):
            if len(grp):
                e = grp["epss_score"].dropna()
                print(f"  {name:<13} n={len(grp):>4}  median EPSS={e.median():.4f}  "
                      f"share >0.1={np.mean(e > 0.1):.1%}")
        print("\nIf the contaminated group carries visibly higher EPSS, the baseline")
        print("is reading the answer rather than forecasting it.")


if __name__ == "__main__":
    main()
