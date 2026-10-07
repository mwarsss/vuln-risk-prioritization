"""Serving bundle: parity with the evaluated feature path, no EPSS input, API round trip.

Synthetic data only — the real frame needs the Kaggle mirror and ~1.6 GB of EPSS
snapshots. Run: python -m pytest tests/test_bundle.py
"""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import main as api
from src.model.bundle import TEXT_GROUP, fit_bundle
from src.model.evaluate import CAT_COLS, NUM_COLS, build_matrices


def synthetic(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    risky = rng.random(n) < 0.08
    words = np.array("buffer overflow remote code execution deserialization sql injection "
                     "authentication bypass cross site scripting path traversal denial".split())
    desc = [" ".join(rng.choice(words, 6)) + (" remote code execution unauthenticated" if r else "")
            for r in risky]
    df = pd.DataFrame({c: rng.choice(["A", "B", "C"], n) for c in CAT_COLS})
    for c in NUM_COLS:
        df[c] = rng.random(n) * 10
    df["description"] = desc
    df["desc_len"] = df["description"].str.len()
    df["epss_score"] = rng.random(n)
    df["is_kev"] = risky
    df["cve_id"] = [f"CVE-2020-{i:05d}" for i in range(n)]
    return df


def test_matrix_matches_evaluated_path():
    df = synthetic()
    bundle = fit_bundle(df)
    Xe, _, names = build_matrices(df, df, use_epss=False)
    assert names == bundle.feature_names
    assert abs(bundle._matrix(df) - Xe).max() < 1e-12


def test_epss_is_never_a_model_input():
    df = synthetic()
    bundle = fit_bundle(df)
    shuffled = df.assign(epss_score=np.random.default_rng(1).random(len(df)))
    assert np.array_equal(bundle.score(df), bundle.score(shuffled))
    assert not any("epss" in n for n in bundle.feature_names)


def test_percentile_is_monotone_and_bounded():
    bundle = fit_bundle(synthetic())
    s = np.linspace(0, 1, 50)
    p = bundle.percentile(s)
    assert (np.diff(p) >= 0).all() and p.min() >= 0 and p.max() <= 1


def test_explanations_sum_to_margin():
    import xgboost as xgb
    df = synthetic()
    bundle = fit_bundle(df)
    row = df.iloc[:1]
    contrib = bundle.explain(row, top=10_000)[0]
    raw = bundle.model.get_booster().predict(
        xgb.DMatrix(bundle._matrix(row), feature_names=bundle.feature_names), pred_contribs=True)[0]
    p = bundle.score(row)[0]
    assert abs(sum(contrib.values()) + raw[-1] - np.log(p / (1 - p))) < 1e-3
    assert TEXT_GROUP in contrib


def _client(tmp_path, monkeypatch):
    path = tmp_path / "m.joblib"
    fit_bundle(synthetic()).save(str(path))
    monkeypatch.setenv("VRP_MODEL", str(path))
    api.get_bundle.cache_clear()
    return TestClient(api.app)


def test_api_score_and_rank(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post("/score", json={"cve_id": "CVE-2024-1", "epss_score": 0.7,
                                    "description": "unauthenticated remote code execution"})
    assert r.status_code == 200
    body = r.json()
    assert body["epss_tier"] == "head" and 0 <= body["content_percentile"] <= 1

    findings = [{"cve_id": f"CVE-2024-{i}", "epss_score": None if i % 4 == 0 else i / 100,
                 "description": "buffer overflow"} for i in range(30)]
    shallow = client.post("/rank", json={"findings": findings, "budget": 5}).json()
    assert shallow["gate_helps"] is False and shallow["selected"][0] == "CVE-2024-29"
    deep = client.post("/rank", json={"findings": findings, "budget": 2000}).json()
    assert deep["gate_helps"] is True and sorted(deep["order"]) == sorted(f["cve_id"] for f in findings)


def test_missing_model_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("VRP_MODEL", str(tmp_path / "nope.joblib"))
    api.get_bundle.cache_clear()
    assert TestClient(api.app).post("/score", json={"cve_id": "x"}).status_code == 503
