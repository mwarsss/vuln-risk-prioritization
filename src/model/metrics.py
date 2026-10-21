"""Operational metrics: what a patching team actually experiences.

The headline is no longer an area under a curve. Following the EPSS papers
(Jacobs et al.), a prioritiser is judged at a level of *effort* — the share of
all CVEs a team remediates:

  coverage    of everything that got exploited, how much did we patch?   (recall)
  efficiency  of everything we patched, how much was actually exploited? (precision)

plus the inverse question a manager asks — how much work to reach a coverage
target — and, for models that emit probabilities, whether those probabilities
mean what they say (Brier score, expected calibration error).
"""
from __future__ import annotations

import numpy as np

EFFORTS = (0.005, 0.01, 0.02, 0.05, 0.10)   # share of all CVEs remediated
COVERAGE_TARGETS = (0.25, 0.50)


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")


def coverage_efficiency(y: np.ndarray, scores: np.ndarray, efforts=EFFORTS) -> dict:
    """Coverage and efficiency when remediating the top `effort` share of CVEs."""
    cum = np.cumsum(y[_order(scores)])
    n, pos = len(y), max(int(y.sum()), 1)
    out = {}
    for e in efforts:
        k = max(int(round(e * n)), 1)
        hits = int(cum[k - 1])
        out[e] = {"patched": k, "caught": hits, "coverage": hits / pos, "efficiency": hits / k}
    return out


def effort_for_coverage(y: np.ndarray, scores: np.ndarray, targets=COVERAGE_TARGETS) -> dict:
    """Fewest CVEs (and share of all CVEs) to patch before catching `target` of exploited."""
    cum = np.cumsum(y[_order(scores)])
    pos = int(y.sum())
    out = {}
    for t in targets:
        need = int(np.ceil(t * pos))
        k = int(np.searchsorted(cum, need) + 1) if need else 0
        out[t] = {"patched": k, "effort": k / len(y)}
    return out


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Brier score and expected calibration error for probability outputs.

    Bins are quantile-based: at a 0.4% base rate, equal-width bins would put
    almost every CVE in the first one and say nothing.
    """
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
            "mean_predicted": float(p.mean()), "base_rate": float(y.mean())}
