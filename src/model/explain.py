"""Reason codes: why did the model score this CVE where it did?

Exact SHAP values from XGBoost's own tree path (`pred_contribs=True`), averaged
over the seed ensemble, then translated into sentences a patching team can
act on. Only features that are *present* on the CVE become reasons — a word in
its description, a CVSS value it has — because "the description does not say
X" is not something anyone can check or act on.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

_CAT_LABELS = {
    "attack_vector": "attack vector", "attack_complexity": "attack complexity",
    "privileges_required": "privileges required", "user_interaction": "user interaction",
    "scope": "scope", "confidentiality_impact": "confidentiality impact",
    "integrity_impact": "integrity impact", "availability_impact": "availability impact",
    "base_severity": "severity", "primary_cwe": "weakness",
}
_NUM_LABELS = {
    "base_score": "CVSS base score {v:.1f}", "exploitability_score": "CVSS exploitability {v:.1f}",
    "impact_score": "CVSS impact {v:.1f}", "n_cwe": "{v:.0f} weakness type(s) listed",
    "desc_len": "description length {v:.0f} chars",
}
# Exploit availability: a reason only when present.
_EXPLOIT_LABELS = {
    "github_pocs": "{v:.0f} public proof-of-concept repo(s) on GitHub",
    "edb_exploits": "{v:.0f} Exploit-DB entr(ies)", "msf_module": "Metasploit module available",
    "any_exploit": "public exploit code exists",
}


def contributions(models, X) -> np.ndarray:
    """Mean SHAP contribution per feature (log-odds), shape (n, n_features)."""
    d = xgb.DMatrix(X)
    c = np.mean([m.get_booster().predict(d, pred_contribs=True) for m in models], axis=0)
    return c[:, :-1]  # drop the bias column


def describe(name: str, value: float) -> str | None:
    """Human sentence for a present feature, or None if it is absent."""
    if name.startswith("tfidf::"):
        return f'description mentions "{name[7:]}"' if value > 0 else None
    if name in _EXPLOIT_LABELS:
        return _EXPLOIT_LABELS[name].format(v=value) if value > 0 else None
    if name in _NUM_LABELS:
        return _NUM_LABELS[name].format(v=value) if value >= 0 else None
    for col, label in _CAT_LABELS.items():
        if name.startswith(col + "_"):
            level = name[len(col) + 1:]
            if value < 1 or "infrequent" in level or level == "NA":
                return None
            return f"{label}: {level.replace('_', ' ').lower()}" if col != "primary_cwe" \
                else f"{label}: {level}"
    return None


def reasons(models, X, names: list[str], top: int = 3) -> list[list[str]]:
    """Top `top` present features pushing each CVE's risk *up*."""
    contrib = contributions(models, X)
    X = X.tocsr()
    out = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel()
        picked = []
        for j in np.argsort(-contrib[i]):
            if contrib[i, j] <= 0 or len(picked) == top:
                break
            text = describe(names[j], row[j])
            if text:
                picked.append(text)
        out.append(picked or ["no strong risk signal in the NVD record"])
    return out
