"""Point-in-time EPSS snapshots.

`epss_client.py` fetches today's score, which is the right thing for scoring a
CVE that was published today and the wrong thing for building a training set.
FIRST recomputes EPSS daily from live exploitation telemetry, so today's score
for a 2023 CVE already reflects whether that CVE was exploited — the exact fact
the model is meant to predict.

Measured on this project's own data (test window: CVEs published 2023-2024,
EPSS pulled 2026-08-09):

    EPSS > 0.90    33% of KEV-positive CVEs     0.1% of KEV-negative CVEs
    median EPSS    0.508 positive               0.005 negative

A 100x median separation is not forecasting skill, it is the score having
observed the outcome. Training against it produced PR-AUC 0.418 that cannot be
reproduced in deployment.

This module reads the dated archive instead, so a CVE published 2023-03-15 is
trained against the EPSS score as it stood on 2023-03-15 — the information a
triage team would actually have had.

Archive: https://epss.cyentia.com/epss_scores-YYYY-MM-DD.csv.gz (public, no auth,
roughly 2021-04-14 onward; earlier CVEs have no EPSS and must train without it).
"""
from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://epss.cyentia.com/epss_scores-{date}.csv.gz"
CACHE_DIR = Path("data/raw/epss_snapshots")
# FIRST's published archive start. Requests before this return 404.
ARCHIVE_START = pd.Timestamp("2021-04-14")


def fetch_snapshot(date: str | pd.Timestamp, *, cache_dir: Path = CACHE_DIR,
                   timeout: int = 60) -> pd.DataFrame:
    """Return the full EPSS table as published on `date`.

    Columns: cve_id, epss_score, epss_perc, epss_as_of. Cached to disk — these
    files are immutable once published, so a snapshot is only ever fetched once.
    """
    date = pd.Timestamp(date).normalize()
    if date < ARCHIVE_START:
        raise ValueError(
            f"EPSS archive starts {ARCHIVE_START.date()}; {date.date()} predates it. "
            "CVEs published before then must be trained without EPSS features."
        )

    stamp = date.strftime("%Y-%m-%d")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"epss-{stamp}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    url = ARCHIVE_URL.format(date=stamp)
    log.info("fetching EPSS snapshot %s", stamp)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    with gzip.open(io.BytesIO(resp.content), "rt") as fh:
        # v2+ files open with a '#model_version...' comment; v1 files do not.
        df = pd.read_csv(fh, comment="#")

    df = df.rename(columns={"cve": "cve_id", "epss": "epss_score",
                            "percentile": "epss_perc"})
    # EPSS v1 (pre-2022) published no percentile column.
    if "epss_perc" not in df.columns:
        df["epss_perc"] = pd.NA
    df = df[["cve_id", "epss_score", "epss_perc"]]
    df["epss_as_of"] = date
    df.to_parquet(cached, index=False)
    return df


def _nearest_available(date: pd.Timestamp, offsets: range) -> pd.Timestamp:
    """EPSS occasionally skips a publication day; walk forward to the next one."""
    for delta in offsets:
        candidate = date + pd.Timedelta(days=delta)
        try:
            fetch_snapshot(candidate)
            return candidate
        except (requests.HTTPError, ValueError):
            continue
    raise FileNotFoundError(f"no EPSS snapshot within {offsets.stop} days of {date.date()}")


def attach_pit_epss(
    df: pd.DataFrame,
    *,
    lag_days: int = 0,
    date_col: str = "published_date",
    quantize: str = "MS",
) -> pd.DataFrame:
    """Attach each CVE's EPSS score as it stood near that CVE's own publication date.

    One snapshot per distinct quantized date rather than one per CVE — the full
    archive is ~250k rows per day, so quantizing to month starts (`MS`) turns
    thousands of downloads into a few dozen. Pass `quantize="D"` for exact daily
    alignment when the extra fidelity is worth the fetch cost.

    `lag_days` shifts the snapshot forward from publication. EPSS has little to
    say about a CVE on its first day; `lag_days=30` asks "what did EPSS think a
    month in", which is a fairer feature but narrows the prediction horizon.
    Whatever is chosen here must also be excluded from the label horizon.

    Rows whose target date predates the EPSS archive keep NaN EPSS and are still
    returned — `evaluate.score_ranker` already ranks NaN last, treating a missing
    score as absence of evidence rather than low risk.
    """
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in frame")

    out = df.copy()
    target = (pd.to_datetime(out[date_col]) + pd.Timedelta(days=lag_days))
    if quantize == "D":
        # A CVE is absent from the snapshot published the same day it appears,
        # so daily alignment still needs to look at least one day forward.
        out["_snap"] = target.dt.normalize() + pd.Timedelta(days=1)
    else:
        # to_period wants anchor-free codes ('M'), not offset aliases ('MS').
        period = {"MS": "M", "QS": "Q", "YS": "Y", "AS": "Y"}.get(quantize, quantize)
        # Snap to the start of the period *after* publication. EPSS only scores a
        # CVE once it has been published and analysed: of CVEs published in
        # January 2023, 1.2% appear in the 2023-01-02 snapshot and 100% appear in
        # the 2023-02-01 one. Aligning to the containing period instead of the
        # following one silently yields ~1% coverage and an all-NaN feature.
        out["_snap"] = (target.dt.to_period(period) + 1).dt.start_time

    # Anything before the archive exists cannot be scored point-in-time.
    out.loc[out["_snap"] < ARCHIVE_START, "_snap"] = pd.NaT

    # Keep only the rows each snapshot is actually needed for. Concatenating whole
    # snapshots works at monthly resolution (~46 files, ~10M rows) but daily
    # alignment needs ~1,300, and 1,300 x 210k rows is a ~276M-row lookup that
    # exhausts memory before the merge runs. Filtering first bounds the lookup by
    # the size of the input frame instead of by the number of snapshots.
    wanted = out.dropna(subset=["_snap"]).groupby("_snap")["cve_id"].agg(set)

    frames = []
    snaps = sorted(out["_snap"].dropna().unique())
    for i, snap in enumerate(snaps, 1):
        snap = pd.Timestamp(snap)
        try:
            table = fetch_snapshot(snap)
        except (requests.HTTPError, ValueError):
            try:
                actual = _nearest_available(snap, range(1, 8))
                table = fetch_snapshot(actual)
                log.warning("no snapshot for %s, used %s", snap.date(), actual.date())
            except FileNotFoundError:
                log.warning("no snapshot near %s; those rows get NaN EPSS", snap.date())
                continue
        table = table[table["cve_id"].isin(wanted.loc[snap])]
        frames.append(table.assign(_snap=snap))
        if len(snaps) > 100 and i % 200 == 0:
            log.info("attached %d/%d snapshots", i, len(snaps))

    if not frames:
        out[["epss_score", "epss_perc", "epss_as_of"]] = pd.NA
        return out.drop(columns="_snap")

    lookup = pd.concat(frames, ignore_index=True)
    # Drop any current-pull EPSS columns so the merge cannot silently keep them.
    out = out.drop(columns=[c for c in ("epss_score", "epss_perc", "epss_as_of")
                            if c in out.columns])
    merged = out.merge(lookup, on=["cve_id", "_snap"], how="left")
    return merged.drop(columns="_snap")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    snap = fetch_snapshot("2023-01-02")
    print(f"{len(snap):,} CVEs in the 2023-01-02 snapshot")
    print(snap.head())
    print(f"\nscores >0.9: {(snap.epss_score > 0.9).sum():,} "
          f"({(snap.epss_score > 0.9).mean():.3%}) — compare against the 33% of "
          "KEV-positive CVEs that carry >0.9 in a 2026 pull")
