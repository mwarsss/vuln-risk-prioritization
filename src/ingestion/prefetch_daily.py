"""Warm the EPSS snapshot cache for daily point-in-time alignment.

Monthly quantization needs ~46 snapshots; daily needs ~1,300. Fetching them
inside an evaluation run makes the run look hung for an hour and re-fetches
nothing usefully, so pull them once here and let every downstream script hit a
warm cache.

Sequential and unhurried on purpose — this is a public archive.
"""
from __future__ import annotations

import argparse

import pandas as pd
import requests

from src.ingestion.epss_history import ARCHIVE_START, _nearest_available, fetch_snapshot
from src.model.evaluate import IN_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    pub = pd.to_datetime(
        pd.read_parquet(IN_PATH, columns=["published_date"])["published_date"]).dropna()
    pub = pub[(pub >= ARCHIVE_START) & (pub <= args.end)]
    # attach_pit_epss(quantize="D") targets publication + 1 day.
    wanted = sorted((pub.dt.normalize() + pd.Timedelta(days=1)).unique())

    print(f"{len(wanted):,} distinct daily snapshots needed")
    fetched = missing = 0
    for i, snap in enumerate(wanted, 1):
        snap = pd.Timestamp(snap)
        try:
            fetch_snapshot(snap)
            fetched += 1
        except (requests.HTTPError, ValueError):
            try:
                fetch_snapshot(_nearest_available(snap, range(1, 8)))
                fetched += 1
            except FileNotFoundError:
                missing += 1
        if i % 50 == 0:
            print(f"  {i:>5,}/{len(wanted):,}  ok={fetched:,} missing={missing:,}", flush=True)

    print(f"done: {fetched:,} cached, {missing:,} unavailable")


if __name__ == "__main__":
    main()
