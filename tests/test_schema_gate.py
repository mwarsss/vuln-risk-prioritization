"""Tests for the leakage gate in src/schema.py.

These exist because the gate spent this project's whole life uncalled: FINDINGS
section 5 described it as a structural guarantee while `validate()` had zero
call sites outside its own module, and the monthly-alignment leak it was
supposed to make impossible went undetected for exactly that reason.

Each test below is a leak the gate must refuse, plus the honest frames it must
accept. Run standalone (`python -m tests.test_schema_gate`) or under pytest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import validate


def frame(**over) -> pd.DataFrame:
    """A minimal, honest point-in-time frame."""
    n = over.pop("n", 6)
    base = {
        "cve_id": [f"CVE-2023-{i:04d}" for i in range(n)],
        "published_date": pd.to_datetime(["2023-03-01"] * n),
        "epss_score": np.linspace(0.001, 0.5, n),
        "epss_perc": np.linspace(0.1, 0.9, n),
        "epss_as_of": pd.to_datetime(["2023-03-02"] * n),
        "kev_date_added": pd.to_datetime(["2023-06-01"] * n),
        "is_kev": [True] * n,
    }
    base.update(over)
    return pd.DataFrame(base)


def test_accepts_an_honest_point_in_time_frame():
    assert validate(frame(), require_pit=True).ok


def test_rejects_epss_without_an_as_of_date():
    df = frame().drop(columns=["epss_as_of"])
    rep = validate(df, require_pit=True)
    assert not rep.ok
    assert any("without epss_as_of" in e for e in rep.errors)


def test_rejects_a_snapshot_taken_before_publication():
    df = frame(epss_as_of=pd.to_datetime(["2023-02-01"] * 6))
    rep = validate(df, require_pit=True)
    assert not rep.ok
    assert any("precedes published_date" in e for e in rep.errors)


def test_rejects_a_snapshot_taken_after_the_kev_listing():
    """The monthly-alignment leak: EPSS read after CISA already catalogued the CVE."""
    df = frame(
        published_date=pd.to_datetime(["2023-03-01"] * 6),
        kev_date_added=pd.to_datetime(["2023-03-20"] * 6),
        epss_as_of=pd.to_datetime(["2023-04-01"] * 6),
    )
    rep = validate(df, require_pit=True)
    assert not rep.ok
    assert any("postdates kev_date_added" in e for e in rep.errors)


def test_ignores_kev_listings_that_predate_publication():
    """No alignment can fix these, so counting them would make the gate unpassable."""
    df = frame(
        published_date=pd.to_datetime(["2023-03-10"] * 6),
        kev_date_added=pd.to_datetime(["2023-03-01"] * 6),  # listed before publication
        epss_as_of=pd.to_datetime(["2023-03-11"] * 6),
    )
    assert validate(df, require_pit=True).ok


def test_tolerates_rows_with_no_snapshot_coverage():
    """attach_pit_epss left-joins: uncovered rows lose score AND date together."""
    df = frame()
    df.loc[:1, ["epss_score", "epss_perc", "epss_as_of"]] = np.nan
    assert validate(df, require_pit=True).ok


def test_rejects_a_score_carrying_no_snapshot_date():
    """A value without a date is unexplained, unlike a gap in coverage."""
    df = frame()
    df.loc[:1, "epss_as_of"] = pd.NaT  # score left in place
    rep = validate(df, require_pit=True)
    assert not rep.ok
    assert any("no snapshot date" in e for e in rep.errors)


def test_coverage_gap_does_not_mask_the_leak_check():
    """The bug that made the gate useless: chained elif skipped the real check."""
    df = frame(
        published_date=pd.to_datetime(["2023-03-01"] * 6),
        kev_date_added=pd.to_datetime(["2023-03-20"] * 6),
        epss_as_of=pd.to_datetime(["2023-04-01"] * 6),
    )
    df.loc[:0, ["epss_score", "epss_perc", "epss_as_of"]] = np.nan  # add a coverage gap
    rep = validate(df, require_pit=True)
    assert any("postdates kev_date_added" in e for e in rep.errors), \
        "a coverage gap must not stop the leak check from running"


def test_warns_on_a_large_median_snapshot_lag():
    df = frame(epss_as_of=pd.to_datetime(["2023-04-01"] * 6),
               kev_date_added=[pd.NaT] * 6, is_kev=[False] * 6)
    rep = validate(df, require_pit=True)
    assert any("days after" in w for w in rep.warnings)


def test_require_pit_false_downgrades_errors_to_warnings():
    df = frame().drop(columns=["epss_as_of"])
    rep = validate(df, require_pit=False)
    assert rep.ok and rep.warnings


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
