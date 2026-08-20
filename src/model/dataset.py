"""One validated loader for every point-in-time evaluation.

`blend.py`, `complementarity.py` and `two_stage.py` each rebuilt the same
sequence by hand: read the parquet, bound the label to a horizon, clip to the
EPSS archive window, rename the leaked column away, attach dated EPSS. Three
copies drifted in exactly the ways that matter — two of them lacked the
`kev_date_added` guard `evaluate_pit.py` has, and all three inherited the
monthly snapshot alignment that turned out to leak.

Centralising it also gives the schema contract somewhere to actually run. Before
this, `src/schema.py:validate()` had no call sites anywhere in the repo, so the
"structural guardrail" FINDINGS section 5 describes existed only as a function
nobody invoked.

Default alignment is `quantize="D"` — publication + 1 day. Monthly alignment
snaps to the start of the *following* month, a median 16-day lag, by which point
CISA had already catalogued 59% of the positives it is asked to predict.
"""
from __future__ import annotations

import pandas as pd

from src.ingestion.epss_history import ARCHIVE_START, attach_pit_epss
from src.model.evaluate import IN_PATH
from src.schema import build_horizon_label, contaminated_mask, validate


def load_pit_frame(horizon_days: int = 365, end: str = "2024-12-31",
                   quantize: str = "D", verbose: bool = True) -> pd.DataFrame:
    """Feature frame with a horizon-bounded label and point-in-time EPSS attached.

    Raises rather than returns a frame that violates the data contract — a
    silently leaked frame is the failure mode this whole project exists to
    document, so it should not be reachable by forgetting a keyword argument.
    """
    df = pd.read_parquet(IN_PATH).dropna(subset=["published_date"])
    df["published_date"] = pd.to_datetime(df["published_date"])

    if "kev_date_added" not in df.columns:
        raise SystemExit(
            "kev_date_added missing — run `python -m src.features.prepare_kaggle` "
            "to rejoin the CISA feed"
        )

    as_of = pd.to_datetime(df["kev_date_added"]).max()
    label = build_horizon_label(df, horizon_days, as_of=as_of)
    df = df.assign(is_kev=label)
    df = df[df["is_kev"].notna()]
    df["is_kev"] = df["is_kev"].astype(bool)

    df = df[(df["published_date"] >= ARCHIVE_START) & (df["published_date"] <= end)]

    # Keep the current-pull column under a name that cannot be mistaken for the
    # honest one, and drop any stale as-of so attach_pit_epss sets it fresh.
    df = df.rename(columns={"epss_score": "epss_leaked"}).drop(
        columns=[c for c in ("epss_perc", "epss_as_of") if c in df.columns])

    if verbose:
        print(f"attaching point-in-time EPSS (quantize={quantize!r}) to {len(df):,} CVEs...")
    df = attach_pit_epss(df, quantize=quantize)

    # Rows CISA catalogued before the earliest snapshot that could describe them.
    # No alignment reaches them, so the choice is to score EPSS on a value that
    # already encodes the answer, or to drop them. Dropping is the honest option
    # and costs little in meaning: these were publicly known to be exploited at
    # publication, so there is no forecast for a prioritizer to make.
    contam = contaminated_mask(df) & df["epss_as_of"].notna()
    n_contam = int(contam.sum())
    if n_contam:
        pos = int(df.loc[contam, "is_kev"].sum())
        if verbose:
            print(f"dropping {n_contam:,} row(s) ({pos:,} positive) catalogued by CISA "
                  f"within one archive day of publication —\n  their EPSS snapshot "
                  f"cannot predate the listing, so the feature already encodes the label")
        df = df[~contam]

    report = validate(df, require_pit=True)
    if verbose and report.warnings:
        print(report)
    report.raise_if_failed()

    return df


def temporal_split(df: pd.DataFrame, fit_end: str, val_end: str | None = None,
                   verbose: bool = True):
    """Split by publication date. Returns (fit, val, test) — val is empty if unused."""
    fit = df[df["published_date"] <= fit_end]
    if val_end is None:
        val = df.iloc[:0]
        test = df[df["published_date"] > fit_end]
    else:
        val = df[(df["published_date"] > fit_end) & (df["published_date"] <= val_end)]
        test = df[df["published_date"] > val_end]

    if verbose:
        for name, part in (("fit", fit), ("val", val), ("test", test)):
            if len(part):
                y = part["is_kev"].values
                print(f"{name:<5}: {len(part):,} rows, {y.sum():,} positive ({y.mean():.3%})")
        print()
    return fit, val, test
