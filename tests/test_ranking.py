"""Tests for the ranking primitives the policy comparisons are built on.

These are pure functions over numpy arrays — no parquet, no model, no network —
so they run in milliseconds and are the right place to pin down ordering
semantics. The numbers in FINDINGS.md sections 6-8 depend entirely on these
behaving as described, and an ordering bug here would be invisible in the
output: a wrong ranking still produces a plausible-looking recall table.

Run standalone (`python -m tests.test_ranking`) or under pytest.
"""
from __future__ import annotations

import numpy as np

from src.model.blend import pct_rank
from src.model.two_stage import gated_score, random_tiebreak_score, recall_at, tiebreak_score


def order_of(scores: np.ndarray) -> list[int]:
    """Indices best-first, matching how recall_at consumes a score array."""
    return list(np.argsort(-scores, kind="stable"))


# --- gated_score: EPSS decides the tier, each tier orders itself -------------

def test_gate_puts_every_above_cutoff_ahead_of_every_below():
    epss = np.array([0.20, 0.001, 0.06, 0.0])
    # content strongly prefers the two below-gate CVEs; the gate must override it
    content = np.array([0.0, 0.99, 0.0, 0.98])
    s = gated_score(epss, content, cutoff=0.05)
    assert set(order_of(s)[:2]) == {0, 2}, "above-gate CVEs must occupy the top slots"


def test_above_gate_tier_is_ordered_by_epss_not_content():
    epss = np.array([0.10, 0.90, 0.30])
    content = np.array([0.99, 0.01, 0.50])
    s = gated_score(epss, content, cutoff=0.05)
    assert order_of(s) == [1, 2, 0]


def test_below_gate_tier_is_ordered_by_content_not_epss():
    epss = np.array([0.004, 0.001, 0.003])
    content = np.array([0.10, 0.90, 0.50])
    s = gated_score(epss, content, cutoff=0.05)
    assert order_of(s) == [1, 2, 0]


def test_missing_epss_falls_below_the_gate():
    epss = np.array([np.nan, 0.90])
    content = np.array([0.99, 0.01])
    s = gated_score(epss, content, cutoff=0.05)
    assert order_of(s) == [1, 0], "absent EPSS is no evidence, not high risk"


# --- tiebreak_score: EPSS ordering preserved exactly -------------------------

def test_tiebreak_never_reorders_distinct_epss():
    epss = np.array([0.5, 0.4, 0.3])
    content = np.array([0.0, 0.0, 1.0])  # content wants index 2 first
    s = tiebreak_score(epss, content)
    assert order_of(s) == [0, 1, 2], "a higher EPSS must always outrank a lower one"


def test_tiebreak_orders_ties_by_content():
    """The property the whole section-8 comparison rests on."""
    epss = np.array([0.001, 0.001, 0.001])
    content = np.array([0.2, 0.9, 0.5])
    s = tiebreak_score(epss, content)
    assert order_of(s) == [1, 2, 0]


def test_tiebreak_is_not_inert_when_ties_exist():
    """Guards against a silent no-op — the failure mode that looks like a result.

    If tie-breaking were inert, this would return the stable index order and the
    policy comparison would report +0 while appearing to work.
    """
    epss = np.array([0.01] * 6)
    content = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert order_of(tiebreak_score(epss, content)) != order_of(np.zeros(6))
    assert order_of(tiebreak_score(epss, content)) == [5, 4, 3, 2, 1, 0]


def test_random_tiebreak_varies_across_draws_but_respects_epss():
    epss = np.array([0.9] + [0.001] * 40)
    seen = set()
    for seed in range(8):
        s = random_tiebreak_score(epss, np.random.default_rng(seed))
        o = order_of(s)
        assert o[0] == 0, "the distinct high-EPSS CVE stays on top in every draw"
        seen.add(tuple(o))
    assert len(seen) > 1, "random tie-breaking must actually vary between seeds"


# --- pct_rank and recall_at --------------------------------------------------

def test_pct_rank_sends_nan_to_the_bottom():
    r = pct_rank(np.array([0.5, np.nan, 0.9]))
    assert r[1] == min(r), "NaN is absent evidence and must rank last"
    assert r[2] > r[0]


def test_recall_at_counts_hits_in_the_top_k():
    y = np.array([1, 0, 1, 0, 1], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    hits, recall = recall_at(y, scores, 3)
    assert hits == 2
    assert recall == 2 / 3


def test_recall_at_clamps_budget_to_population():
    y = np.array([1, 0], dtype=bool)
    hits, recall = recall_at(y, np.array([0.1, 0.2]), 100)
    assert (hits, recall) == (1, 1.0)


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
