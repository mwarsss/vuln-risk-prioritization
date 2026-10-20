"""Tests for the live path: NVD record -> features -> banded ranking.

No network and no trained model — a hand-built NVD record and a stub model — so
the ordering and encoding rules are pinned independently of either.

Run standalone (`python -m tests.test_rank`) or under pytest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingestion.nvd_client import cve_features
from src.rank import score_rows

NVD_RECORD = {
    "id": "CVE-2099-0001",
    "published": "2099-01-01T00:00:00.000",
    "descriptions": [{"lang": "en", "value": "Remote code execution in Foo."},
                     {"lang": "es", "value": "Ejecución remota en Foo."}],
    "weaknesses": [{"description": [{"lang": "en", "value": "NVD-CWE-noinfo"},
                                    {"lang": "en", "value": "CWE-94"}]}],
    "metrics": {"cvssMetricV31": [
        {"type": "Secondary", "exploitabilityScore": 1.0, "impactScore": 1.0,
         "cvssData": {"baseScore": 5.0, "baseSeverity": "MEDIUM", "attackVector": "LOCAL"}},
        {"type": "Primary", "exploitabilityScore": 3.9, "impactScore": 5.9,
         "cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL", "attackVector": "NETWORK",
                      "attackComplexity": "LOW", "privilegesRequired": "NONE",
                      "userInteraction": "NONE", "scope": "UNCHANGED",
                      "confidentialityImpact": "HIGH", "integrityImpact": "HIGH",
                      "availabilityImpact": "HIGH"}}]},
    "configurations": [{"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:foo:foo:1:*:*:*:*:*:*:*"}]}]}],
}


def test_features_prefer_nvd_primary_cvss_and_english_text():
    f = cve_features(NVD_RECORD)
    assert f["base_score"] == 9.8 and f["attack_vector"] == "NETWORK"
    assert f["description"] == "Remote code execution in Foo."
    assert f["desc_len"] == len(f["description"])


def test_features_match_training_cwe_rules():
    f = cve_features(NVD_RECORD)
    assert f["primary_cwe"] == "CWE-94", "placeholders are skipped, as in prepare_kaggle"
    assert f["n_cwe"] == 2, "n_cwe counts every listed weakness, as in training"


def test_features_never_read_cpe():
    """CPE is enrichment and was a label proxy — FINDINGS section 11."""
    f = cve_features(NVD_RECORD)
    assert not {"vendor", "n_vendors", "n_cpe"} & set(f)


def test_record_without_cvss_still_yields_a_row():
    f = cve_features({"id": "CVE-2099-0002", "descriptions": [{"lang": "en", "value": "x"}]})
    assert f["base_score"] is None and f["primary_cwe"] == "UNKNOWN"


class _Feat:
    def transform(self, df):
        return df[["base_score"]].astype(float).values


class _Model:
    def predict_proba(self, X):
        p = X[:, 0] / 10.0
        return np.column_stack([1 - p, p])


BUNDLE = {"featurizer": _Feat(), "models": [_Model()], "band_edges": (0.001, 0.01, 0.1)}


def _rows(**cols):
    n = len(cols["cve_id"])
    cols.setdefault("description", ["d"] * n)
    return pd.DataFrame(cols)


def test_higher_band_outranks_higher_model_score():
    rows = _rows(cve_id=["A", "B"], base_score=[9.9, 1.0])
    out = score_rows(rows, {"A": 0.0005, "B": 0.05}, BUNDLE)
    assert list(out["cve_id"]) == ["B", "A"]


def test_model_orders_within_a_band():
    rows = _rows(cve_id=["A", "B", "C"], base_score=[2.0, 9.0, 5.0])
    out = score_rows(rows, {"A": 0.0002, "B": 0.0001, "C": 0.0009}, BUNDLE)
    assert list(out["cve_id"]) == ["B", "C", "A"]
    assert list(out["rank"]) == [1, 2, 3]


def test_missing_epss_joins_bottom_band_and_is_flagged():
    rows = _rows(cve_id=["A", "B"], base_score=[9.0, 1.0])
    out = score_rows(rows, {"B": 0.0005}, BUNDLE).set_index("cve_id")
    assert list(out.index) == ["A", "B"], "unscored competes in the bottom band, not below it"
    assert out.loc["A", "band"] == "EPSS < 0.001" and "no EPSS" in out.loc["A", "note"]


def test_cve_missing_from_nvd_ranks_last_in_its_band_and_is_flagged():
    rows = _rows(cve_id=["A", "B"], base_score=[None, 1.0], description=["", "d"])
    out = score_rows(rows, {"A": 0.0005, "B": 0.0005}, BUNDLE).set_index("cve_id")
    assert list(out.index) == ["B", "A"]
    assert np.isnan(out.loc["A", "model_score"]) and "not in NVD" in out.loc["A", "note"]


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
