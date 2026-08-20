"""Data contract for the vulnerability risk prioritization pipeline.

Every column carries an `as_of` marker recording *when its value is knowable*.
This is not documentation — it is the check that stops the pipeline's main
failure mode.

Background: the first evaluation of this project reported PR-AUC 0.418 for
`XGBoost + EPSS` against a 0.409 EPSS baseline. Both numbers are unreachable in
production. The `epss_score` column came from a bulk dataset pulled 2026-08-09,
but the test window is CVEs published 2023-2024. EPSS ingests live exploitation
telemetry, so by pull time it had already observed the exploitation that the
label (`is_kev`) records. 33% of KEV-positive CVEs in that window carry
EPSS > 0.90, against 0.1% of non-KEV CVEs. The feature was measured after the
outcome it predicts.

A feature is PIT_SAFE only if its value is fixed at CVE publication and never
revised. Anything recomputed later (EPSS, KEV membership, CVSS rescoring) is
POINT_IN_TIME and must be read from a dated snapshot, never a current pull.
`validate()` refuses a training frame that mixes the two without a snapshot date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Timing(str, Enum):
    """When a column's value becomes knowable."""

    # Fixed at CVE publication, never revised. Safe to read from any pull.
    PIT_SAFE = "pit_safe"
    # Recomputed continuously upstream. Only valid from a dated snapshot;
    # reading the current value into a historical row leaks the outcome.
    POINT_IN_TIME = "point_in_time"
    # The prediction target.
    LABEL = "label"
    # Identifier / bookkeeping, not fed to the model.
    KEY = "key"


class Role(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IDENTIFIER = "identifier"
    TARGET = "target"


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    role: Role
    timing: Timing
    source: str
    description: str
    nullable: bool = True
    # Inclusive bounds for numeric columns; validate() enforces them.
    bounds: tuple[float, float] | None = None


SCHEMA: tuple[Column, ...] = (
    Column("cve_id", "string", Role.IDENTIFIER, Timing.KEY, "NVD",
           "CVE-YYYY-NNNNN. Join key across all three sources.", nullable=False),
    Column("published_date", "datetime64[ns]", Role.IDENTIFIER, Timing.KEY, "NVD",
           "CVE publication timestamp. Defines the temporal split boundary.",
           nullable=False),
    Column("published_year", "int16", Role.NUMERIC, Timing.KEY, "NVD",
           "Derived from published_date; used to drop pre-2005 sparse years."),

    # --- CVSS: assigned at publication. Note the caveat on rescoring below. ---
    Column("base_score", "float32", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "CVSS base score. Static severity, not exploitation evidence.",
           bounds=(0.0, 10.0)),
    Column("exploitability_score", "float32", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "CVSS exploitability sub-score.", bounds=(0.0, 10.0)),
    Column("impact_score", "float32", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "CVSS impact sub-score.", bounds=(0.0, 10.0)),
    Column("base_severity", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "LOW / MEDIUM / HIGH / CRITICAL."),
    Column("attack_vector", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "NETWORK / ADJACENT_NETWORK / LOCAL / PHYSICAL."),
    Column("attack_complexity", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "LOW / HIGH."),
    Column("privileges_required", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "NONE / LOW / HIGH."),
    Column("user_interaction", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "NONE / REQUIRED."),
    Column("scope", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "UNCHANGED / CHANGED."),
    Column("confidentiality_impact", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "NONE / LOW / HIGH."),
    Column("integrity_impact", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "NONE / LOW / HIGH."),
    Column("availability_impact", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "NONE / LOW / HIGH."),

    # --- CVE content: the honest signal. No leakage path. ---
    Column("description", "string", Role.TEXT, Timing.PIT_SAFE, "NVD",
           "English vulnerability description. TF-IDF source."),
    Column("desc_len", "int32", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "Character length of description. Proxy for disclosure detail."),
    Column("primary_cwe", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "First CWE identifier assigned."),
    Column("n_cwe", "int16", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "Count of CWEs assigned.", bounds=(0, 100)),
    Column("vendor", "category", Role.CATEGORICAL, Timing.PIT_SAFE, "NVD",
           "Primary affected vendor, parsed from CPE."),
    Column("n_vendors", "int16", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "Distinct vendors affected. Proxy for blast radius.", bounds=(0, 10_000)),
    Column("n_cpe", "int32", Role.NUMERIC, Timing.PIT_SAFE, "NVD",
           "Count of affected product configurations.", bounds=(0, 100_000)),

    # --- Recomputed upstream. Snapshot-only. ---
    Column("epss_score", "float32", Role.NUMERIC, Timing.POINT_IN_TIME, "FIRST EPSS",
           "Exploitation probability. FIRST revises this daily using live "
           "exploitation telemetry, so a current pull encodes the outcome for "
           "any historical CVE. Load from a dated snapshot only.",
           bounds=(0.0, 1.0)),
    Column("epss_perc", "float32", Role.NUMERIC, Timing.POINT_IN_TIME, "FIRST EPSS",
           "EPSS percentile on the snapshot date. Same constraint as epss_score.",
           bounds=(0.0, 1.0)),
    Column("epss_as_of", "datetime64[ns]", Role.IDENTIFIER, Timing.KEY, "FIRST EPSS",
           "Snapshot date the EPSS values were read from. Required whenever "
           "epss_score is present; its absence is what made the first "
           "evaluation unreproducible."),

    # --- Label. ---
    Column("is_kev", "bool", Role.TARGET, Timing.LABEL, "CISA KEV",
           "Whether the CVE appears in the CISA KEV catalogue. y.", nullable=False),
    Column("kev_date_added", "datetime64[ns]", Role.IDENTIFIER, Timing.LABEL, "CISA KEV",
           "Date CISA added the CVE. Required to build a horizon-bounded target "
           "(exploited within N days of publication) instead of the open-ended "
           "'exploited ever, as of the pull date'."),
)

BY_NAME: dict[str, Column] = {c.name: c for c in SCHEMA}

PIT_SAFE_FEATURES: tuple[str, ...] = tuple(
    c.name for c in SCHEMA
    if c.timing is Timing.PIT_SAFE and c.role is not Role.IDENTIFIER
)
POINT_IN_TIME_FEATURES: tuple[str, ...] = tuple(
    c.name for c in SCHEMA if c.timing is Timing.POINT_IN_TIME
)
TARGET = "is_kev"


class SchemaError(ValueError):
    """Raised when a frame violates the contract in a way that would corrupt results."""


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> Report:
        if self.errors:
            raise SchemaError(
                f"{len(self.errors)} schema violation(s):\n  - "
                + "\n  - ".join(self.errors)
            )
        return self

    def __str__(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        return "\n".join(lines) or "  clean"


SNAPSHOT_RESOLUTION_DAYS = 1  # the EPSS archive publishes at most one file per day


def contaminated_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose EPSS snapshot was taken on or after their CISA KEV listing.

    Excludes CVEs catalogued on or before publication: for those, exploitation
    predates the prediction task entirely and no feature alignment changes that.
    Section 4 measures them at ~37% of positives.
    """
    as_of = pd.to_datetime(df["epss_as_of"])
    published = pd.to_datetime(df["published_date"])
    added = pd.to_datetime(df["kev_date_added"])
    return (added > published) & (added <= as_of)


def validate(df: pd.DataFrame, *, require_pit: bool = True) -> Report:
    """Check a feature frame against the contract.

    `require_pit=True` (the default, and what training should use) rejects a
    frame carrying EPSS columns with no `epss_as_of` date. That combination is
    exactly the one that produced the unreproducible 0.418 PR-AUC: it cannot be
    distinguished from a leaked frame by inspecting the values, so it is
    refused rather than warned about.

    Pass `require_pit=False` only for exploratory analysis of a current-pull
    snapshot, where every row shares one as-of date and no temporal split is
    being made.
    """
    rep = Report()
    present = set(df.columns)

    for col in SCHEMA:
        if col.name not in present:
            # Only the identity/target spine is mandatory; features may be absent.
            if not col.nullable and col.role in (Role.IDENTIFIER, Role.TARGET):
                rep.errors.append(f"{col.name}: required column missing")
            continue

        s = df[col.name]
        if not col.nullable and s.isna().any():
            rep.errors.append(f"{col.name}: {int(s.isna().sum())} null(s) in a non-nullable column")

        if col.bounds is not None and pd.api.types.is_numeric_dtype(s):
            lo, hi = col.bounds
            bad = int(((s < lo) | (s > hi)).sum())
            if bad:
                rep.errors.append(
                    f"{col.name}: {bad} value(s) outside {lo}–{hi} "
                    f"(observed {s.min():.4g}–{s.max():.4g})"
                )

    # The leakage gate. These checks are deliberately independent rather than
    # chained: an earlier one firing must not stop a later one from running, or
    # a frame with an ordinary coverage gap silently skips the check that
    # actually matters.
    epss_present = [c for c in POINT_IN_TIME_FEATURES if c in present]
    if epss_present:
        if "epss_as_of" not in present or df.get("epss_as_of") is None:
            msg = (
                f"{', '.join(epss_present)} present without epss_as_of. These are "
                "revised daily from exploitation telemetry, so a current pull "
                "leaks the label for historical CVEs. Load a dated snapshot "
                "(src.ingestion.epss_history) or drop them."
            )
            (rep.errors if require_pit else rep.warnings).append(msg)
        else:
            as_of = pd.to_datetime(df["epss_as_of"])

            # A row may legitimately have no snapshot — attach_pit_epss left-joins,
            # so uncovered CVEs get NaN for the scores AND the date together. Only
            # a *score without a date* is unexplained.
            scored = pd.concat([df[c].notna() for c in epss_present], axis=1).any(axis=1)
            orphan = int((scored & as_of.isna()).sum())
            if orphan:
                (rep.errors if require_pit else rep.warnings).append(
                    f"epss_as_of: {orphan} row(s) carry an EPSS value with no snapshot date"
                )

            if "published_date" in present:
                published = pd.to_datetime(df["published_date"])
                # A snapshot taken before publication cannot describe the CVE.
                ahead = int((as_of < published).sum())
                if ahead:
                    rep.errors.append(
                        f"epss_as_of precedes published_date on {ahead} row(s)"
                    )

                # The forward leak, and the one that is easy to reintroduce by
                # coarsening the snapshot alignment: if a CVE reached the CISA
                # catalogue before the snapshot used to score it, EPSS had already
                # observed the exploitation it is meant to forecast.
                #
                # Two kinds, and they need different verdicts. If the snapshot sits
                # more than one archive day past publication, a tighter alignment
                # would have avoided the leak — that is a bug, and an error. If it
                # is already next-day, the EPSS archive has no finer resolution to
                # offer and the contamination is irreducible: warn, and expect the
                # caller to drop those rows rather than silently score them.
                if "kev_date_added" in present:
                    contam = contaminated_mask(df)
                    tighter_possible = as_of > published + pd.Timedelta(days=SNAPSHOT_RESOLUTION_DAYS)
                    avoidable = int((contam & tighter_possible).sum())
                    irreducible = int((contam & ~tighter_possible).sum())
                    if avoidable:
                        (rep.errors if require_pit else rep.warnings).append(
                            f"epss_as_of postdates kev_date_added on {avoidable} row(s) "
                            "whose KEV listing came after publication, with a snapshot "
                            "more than a day past publication. The snapshot observed "
                            "the exploitation it is supposed to predict, and a tighter "
                            "alignment would avoid it — use quantize='D'."
                        )
                    if irreducible:
                        rep.warnings.append(
                            f"{irreducible} row(s) were catalogued by CISA within one "
                            "archive day of publication, so no snapshot can predate the "
                            "listing. Exploitation was public knowledge at publication; "
                            "drop them rather than scoring them."
                        )

                lag = (as_of - published).dt.days.dropna()
                if len(lag) and lag.median() > 7:
                    rep.warnings.append(
                        f"epss_as_of sits a median {lag.median():.0f} days after "
                        "publication. Every day of lag is a day EPSS may have "
                        "observed the outcome; prefer quantize='D'."
                    )

    if TARGET in present and "kev_date_added" not in present:
        rep.warnings.append(
            "is_kev present without kev_date_added — the target is 'exploited ever, "
            "as of pull date', which drifts as CISA backfills. Prefer a "
            "horizon-bounded label (see build_horizon_label)."
        )

    return rep


def build_horizon_label(
    df: pd.DataFrame, horizon_days: int = 365, *, as_of: pd.Timestamp | str | None = None
) -> pd.Series:
    """Label a CVE positive if CISA added it to KEV within `horizon_days` of publication.

    The open-ended `is_kev` flag answers "has this ever been exploited, as of
    whenever the data was pulled". That target moves every time CISA backfills
    the catalogue, and it rewards a model for CVEs exploited years after the
    prediction point. A bounded horizon asks the question a triage team actually
    has: given this CVE today, will it be weaponised inside the next year?

    Rows whose horizon extends past `as_of` are returned as <NA> — their labels
    are not yet mature and including them as negatives understates the positive
    rate.
    """
    for required in ("published_date", "kev_date_added"):
        if required not in df.columns:
            raise SchemaError(f"build_horizon_label needs {required}")

    published = pd.to_datetime(df["published_date"])
    added = pd.to_datetime(df["kev_date_added"])
    as_of = pd.Timestamp(as_of) if as_of is not None else added.max()

    within = (added - published).dt.days.le(horizon_days) & added.notna()
    label = within.astype("boolean")

    immature = published + pd.Timedelta(days=horizon_days) > as_of
    label[immature & ~within.fillna(False)] = pd.NA
    return label.rename(f"exploited_within_{horizon_days}d")


# Serving-side storage for scored output. Kept deliberately narrow: the API
# returns a ranking plus the evidence behind it, and every row records which
# model and which EPSS snapshot produced it so a score can be reconstructed.
SCORED_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS vuln_score (
    cve_id              TEXT        NOT NULL,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version       TEXT        NOT NULL,
    epss_as_of          DATE        NOT NULL,

    risk_score          REAL        NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_percentile     REAL        CHECK (risk_percentile BETWEEN 0 AND 1),
    rank_in_batch       INTEGER,

    -- Baselines retained alongside the model score. The model has to be shown
    -- beating these on every batch, not just once at training time.
    epss_score          REAL        CHECK (epss_score BETWEEN 0 AND 1),
    cvss_base_score     REAL        CHECK (cvss_base_score BETWEEN 0 AND 10),

    -- Top SHAP contributions as [{"feature": ..., "value": ..., "shap": ...}].
    explanation         JSONB,

    PRIMARY KEY (cve_id, scored_at, model_version)
);

CREATE INDEX IF NOT EXISTS vuln_score_rank_idx
    ON vuln_score (scored_at DESC, risk_score DESC);

-- Outcome log, written when a scored CVE later lands in KEV. This is what makes
-- deployed precision@k measurable instead of assumed.
CREATE TABLE IF NOT EXISTS vuln_outcome (
    cve_id          TEXT        NOT NULL PRIMARY KEY,
    kev_date_added  DATE        NOT NULL,
    first_scored_at TIMESTAMPTZ,
    days_to_exploit INTEGER
);
"""


def describe() -> str:
    """Render the contract as a table. Used by docs and CI."""
    w = max(len(c.name) for c in SCHEMA) + 2
    lines = [f"{'column':<{w}} {'timing':<15} {'role':<13} source",
             "-" * (w + 45)]
    for c in SCHEMA:
        lines.append(f"{c.name:<{w}} {c.timing.value:<15} {c.role.value:<13} {c.source}")
    lines.append("")
    lines.append(f"{len(PIT_SAFE_FEATURES)} point-in-time-safe features, "
                 f"{len(POINT_IN_TIME_FEATURES)} snapshot-only, target={TARGET!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
