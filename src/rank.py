"""Rank a list of CVEs for patching: EPSS picks the band, the model orders inside it.

    python -m src.rank CVE-2024-3400 CVE-2024-21887 CVE-2023-4966
    python -m src.rank --file scan_cves.txt --top 50 --csv ranked.csv

Pulls today's EPSS (FIRST API) and each CVE's NVD record, encodes the record
with the vocabulary the model was trained on, and orders with the same
`banded_score` FINDINGS sections 10-11 evaluate. Set NVD_API_KEY to lift NVD's
public rate limit (5 requests / 30 s) — without it, expect ~6 s per CVE.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ingestion.epss_client import fetch_epss_scores
from src.ingestion.nvd_client import cve_features, fetch_cve
from src.model.train import MODEL_PATH
from src.model.two_stage import banded_score

BAND_LABELS = ("EPSS < 0.001", "EPSS 0.001–0.01", "EPSS 0.01–0.1", "EPSS ≥ 0.1")


def load_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python -m src.model.train` first")
    return joblib.load(path)


def score_rows(rows: pd.DataFrame, epss: dict[str, float], bundle: dict) -> pd.DataFrame:
    """Rank feature rows already fetched. Split from rank_cves so it runs offline."""
    out = rows.copy()
    out["epss"] = out["cve_id"].map(epss)
    known = out["description"].fillna("").str.len() > 0
    content = np.zeros(len(out))
    if known.any():
        X = bundle["featurizer"].transform(out[known])
        content[known.values] = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]],
                                        axis=0)
    # No NVD record means no content signal: last within its band, and flagged.
    content[~known.values] = -1.0
    edges = bundle["band_edges"]
    out["model_score"] = np.where(known, content, np.nan)
    out["band"] = [BAND_LABELS[i] for i in
                   np.searchsorted(np.asarray(edges), out["epss"].fillna(0.0), side="right")]
    out["note"] = np.select([~known, out["epss"].isna()],
                            ["not in NVD — ordered by EPSS band only", "no EPSS yet — bottom band"],
                            default="")
    out["_s"] = banded_score(out["epss"].values.astype(float), content, edges)
    out = out.sort_values("_s", ascending=False, kind="stable").drop(columns="_s")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def rank_cves(cve_ids: list[str], bundle: dict | None = None,
              api_key: str | None = None, verbose: bool = True) -> pd.DataFrame:
    ids = list(dict.fromkeys(c.strip().upper() for c in cve_ids if c.strip()))
    bundle = bundle or load_bundle()
    api_key = api_key or os.environ.get("NVD_API_KEY")
    epss = fetch_epss_scores(ids)
    rows = []
    for i, cid in enumerate(ids):
        rec = fetch_cve(cid, api_key)
        rows.append(cve_features(rec) if rec else {"cve_id": cid, "description": ""})
        if verbose and len(ids) > 5:
            print(f"  NVD {i + 1}/{len(ids)}", end="\r", flush=True)
        if i < len(ids) - 1:
            time.sleep(0.7 if api_key else 6.0)
    if verbose and len(ids) > 5:
        print()
    frame = pd.DataFrame(rows).reindex(columns=["cve_id", "published_date", "description",
                                                *bundle["features"]["numeric"],
                                                *bundle["features"]["categorical"]])
    return score_rows(frame, epss, bundle)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cves", nargs="*", help="CVE IDs")
    ap.add_argument("--file", type=Path, help="text file, one CVE ID per line")
    ap.add_argument("--top", type=int, default=None, help="show only the first N")
    ap.add_argument("--csv", type=Path, help="write the full ranking here")
    args = ap.parse_args()

    ids = list(args.cves)
    if args.file:
        ids += args.file.read_text().split()
    if not ids:
        ap.error("give CVE IDs or --file")

    bundle = load_bundle()
    print(f"model trained through {bundle['trained_through']} on {bundle['n_train']:,} CVEs")
    ranked = rank_cves(ids, bundle)
    if args.csv:
        ranked.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    show = ranked.head(args.top) if args.top else ranked
    with pd.option_context("display.max_colwidth", 40, "display.width", 140):
        print(show[["rank", "cve_id", "band", "epss", "model_score", "base_score", "note"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
