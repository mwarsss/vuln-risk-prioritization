"""Label-maturity and class-balance diagnostics.

`cisa_kev` is a snapshot taken the day the mirror was pulled: it says "this CVE
has appeared in KEV by now", not "within N days of publication". CVEs published
recently have had less time to be catalogued, so their observed positive rate is
censored downward. Any temporal split has to account for that or the model gets
graded against labels that have not finished arriving.
"""
from __future__ import annotations

import pandas as pd

IN_PATH = "data/processed/features.parquet"


def main() -> None:
    df = pd.read_parquet(IN_PATH)

    print(f"rows           : {len(df):,}")
    print(f"positives (KEV): {df['is_kev'].sum():,}")
    print(f"base rate      : {df['is_kev'].mean():.4%}")
    print(f"date range     : {df['published_date'].min()} -> {df['published_date'].max()}")
    print(f"missing desc   : {df['description'].isna().sum():,}")
    print(f"missing epss   : {df['epss_score'].isna().sum():,}")
    print(f"missing cvss   : {df['base_score'].isna().sum():,}")

    print("\n--- KEV rate by publication year (label maturity) ---")
    by_year = df.groupby("published_year").agg(
        n=("is_kev", "size"),
        kev=("is_kev", "sum"),
    )
    by_year["kev_rate"] = by_year["kev"] / by_year["n"]
    print(by_year.loc[by_year.index >= 2010].to_string(
        formatters={"kev_rate": "{:.3%}".format}
    ))

    print("\n--- EPSS vs KEV ---")
    has_epss = df.dropna(subset=["epss_score"])
    print(f"mean EPSS, KEV=1: {has_epss.loc[has_epss.is_kev == 1, 'epss_score'].mean():.4f}")
    print(f"mean EPSS, KEV=0: {has_epss.loc[has_epss.is_kev == 0, 'epss_score'].mean():.4f}")

    print("\n--- CVSS vs KEV ---")
    print(f"mean CVSS, KEV=1: {df.loc[df.is_kev == 1, 'base_score'].mean():.3f}")
    print(f"mean CVSS, KEV=0: {df.loc[df.is_kev == 0, 'base_score'].mean():.3f}")

    print("\n--- how much of the catalogue is 'critical'? (the triage problem) ---")
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        sub = df[df["base_severity"] == tier]
        if len(sub):
            print(f"{tier:9s} n={len(sub):>7,}  kev={sub['is_kev'].sum():>5,}  rate={sub['is_kev'].mean():.3%}")

    print("\n--- top CWEs among KEV ---")
    print(df[df.is_kev == 1]["primary_cwe"].value_counts().head(10).to_string())

    print("\n--- top vendors among KEV ---")
    print(df[df.is_kev == 1]["vendor"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
