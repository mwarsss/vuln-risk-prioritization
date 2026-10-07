"""Persisted content model + the gated ranking policy, for serving.

FINDINGS section 9 is the only result in this repo that survives point-in-time
evaluation: EPSS orders the head of the queue, a content model (NVD metadata +
description text, no EPSS input) orders the tail. The scaffold's `train.py` and
API still served a two-feature `cvss + epss` model that sections 2-8 show is
the wrong shape, so this module packages the thing the findings actually
support.

Two deliberate choices:

* **No calibrated probability.** The content model trains with
  `scale_pos_weight` for a ranking objective, so its raw output is not a
  probability, and isotonic calibration on ~200 positives would be noise. The
  bundle reports a *percentile against the training-score distribution* instead,
  which is honest about what the number means.
* **Fixed gate cutoff.** Section 9 shows per-run validation cutoff selection is
  weak (63 positives, ties). The cutoff is a policy constant, not re-selected.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.model.evaluate import CAT_COLS, NUM_COLS, fit_xgb
from src.model.two_stage import gated_score

GATE_CUTOFF = 0.05
# Section 9: below roughly this many CVEs the gate is worse than EPSS alone.
MIN_USEFUL_BUDGET = 1500
TEXT_GROUP = "description_text"


@dataclass
class ContentBundle:
    encoder: OneHotEncoder
    tfidf: TfidfVectorizer
    model: xgb.XGBClassifier
    feature_names: list[str]
    reference_scores: np.ndarray  # sorted training scores, for percentiles
    gate_cutoff: float = GATE_CUTOFF

    # -- matrices ---------------------------------------------------------
    def _matrix(self, df: pd.DataFrame) -> csr_matrix:
        num = csr_matrix(df[NUM_COLS].astype(float).fillna(-1).values)
        cat = self.encoder.transform(df[CAT_COLS].fillna("NA").astype(str))
        txt = self.tfidf.transform(df["description"].fillna(""))
        return hstack([num, cat, txt]).tocsr()

    # -- scoring ----------------------------------------------------------
    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._matrix(df))[:, 1]

    def percentile(self, scores: np.ndarray) -> np.ndarray:
        """Fraction of training CVEs this score meets or beats, in [0, 1]."""
        return np.searchsorted(self.reference_scores, scores, side="right") / len(self.reference_scores)

    def rank_key(self, df: pd.DataFrame, epss: np.ndarray) -> np.ndarray:
        """Gated queue key: higher = patch sooner. EPSS head, content tail."""
        return gated_score(np.asarray(epss, dtype=float), self.score(df), self.gate_cutoff)

    # -- explanation ------------------------------------------------------
    def explain(self, df: pd.DataFrame, top: int = 8) -> list[dict[str, float]]:
        """Per-row additive contributions (log-odds), TF-IDF terms pooled into one group.

        Uses XGBoost's native TreeSHAP (`pred_contribs`), so the `shap` package
        is not needed at serving time.
        """
        contribs = self.model.get_booster().predict(
            xgb.DMatrix(self._matrix(df), feature_names=self.feature_names), pred_contribs=True
        )
        out = []
        for row in contribs:
            grouped: dict[str, float] = {}
            for name, value in zip(self.feature_names, row[:-1]):  # last col = bias
                key = TEXT_GROUP if name.startswith("tfidf::") else name
                grouped[key] = grouped.get(key, 0.0) + float(value)
            ranked = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:top]
            out.append(dict(ranked))
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ContentBundle":
        return joblib.load(path)


def fit_bundle(fit: pd.DataFrame, seed: int = 42) -> ContentBundle:
    """Fit encoders and the content model on `fit` only. `epss_score` is never read."""
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=True)
    encoder.fit(fit[CAT_COLS].fillna("NA").astype(str))
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=5,
                            stop_words="english", sublinear_tf=True)
    tfidf.fit(fit["description"].fillna(""))
    names = list(NUM_COLS) + list(encoder.get_feature_names_out(CAT_COLS)) + \
        [f"tfidf::{t}" for t in tfidf.get_feature_names_out()]

    shell = ContentBundle(encoder, tfidf, None, names, np.empty(0))  # type: ignore[arg-type]
    X = shell._matrix(fit)
    shell.model = fit_xgb(X, fit["is_kev"].astype(int).values, seed=seed)
    shell.reference_scores = np.sort(shell.model.predict_proba(X)[:, 1])
    return shell
