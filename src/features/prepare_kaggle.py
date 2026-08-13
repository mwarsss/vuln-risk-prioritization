"""Build the training table from the Kaggle CVE/KEV/EPSS bulk mirror.

The mirror ships two CSVs that join on `cve_id`:
  - cve_cisa_epss_enriched_dataset.csv : CVSS fields, EPSS score, KEV label, published_date
  - cve_corpus.csv                     : description text, CWE list, CPE list

Writes data/processed/features.parquet.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/kaggle_cve_kev_epss")
OUT_PATH = Path("data/processed/features.parquet")

ENRICHED = RAW_DIR / "cve_cisa_epss_enriched_dataset.csv"
CORPUS = RAW_DIR / "cve_corpus.csv"


def _parse_list(value: object) -> list[str]:
    """The corpus stores Python list literals as strings; recover them safely."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _primary_cwe(cwes: list[str]) -> str:
    """First real CWE id; NVD placeholders (NVD-CWE-Other/noinfo) collapse to UNKNOWN."""
    for cwe in cwes:
        if cwe.startswith("CWE-"):
            return cwe
    return "UNKNOWN"


_CPE_VENDOR = re.compile(r"^cpe:2\.3:[aoh]:([^:]+):")


def _vendor_product(cpes: list[str]) -> tuple[str, int]:
    """Vendor of the first CPE, plus how many distinct vendors the CVE touches.

    Breadth matters: a CVE spanning many vendors is usually a library/protocol
    flaw, which behaves differently from a single-product bug.
    """
    vendors = []
    for cpe in cpes:
        m = _CPE_VENDOR.match(cpe)
        if m:
            vendors.append(m.group(1))
    if not vendors:
        return "UNKNOWN", 0
    return vendors[0], len(set(vendors))


def build() -> pd.DataFrame:
    enriched = pd.read_csv(ENRICHED)
    corpus = pd.read_csv(CORPUS)

    corpus["description"] = corpus["description_data"].map(
        lambda v: " ".join(_parse_list(v))
    )
    cwes = corpus["cwe_data"].map(_parse_list)
    corpus["primary_cwe"] = cwes.map(_primary_cwe)
    corpus["n_cwe"] = cwes.map(len)

    cpes = corpus["cpe_data"].map(_parse_list)
    vp = cpes.map(_vendor_product)
    corpus["vendor"] = vp.map(lambda t: t[0])
    corpus["n_vendors"] = vp.map(lambda t: t[1])
    corpus["n_cpe"] = cpes.map(len)

    df = enriched.merge(
        corpus[["cve_id", "description", "primary_cwe", "n_cwe", "vendor", "n_vendors", "n_cpe"]],
        on="cve_id",
        how="left",
    )

    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", format="mixed")
    df["published_year"] = df["published_date"].dt.year
    df["is_kev"] = df["cisa_kev"].astype(str).str.lower().eq("true").astype(int)
    df["desc_len"] = df["description"].fillna("").str.len()

    return df


if __name__ == "__main__":
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"rows={len(df):,}  positives={df['is_kev'].sum():,}  -> {OUT_PATH}")
