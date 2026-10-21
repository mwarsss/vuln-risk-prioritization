"""TriageBand variants, compared on operational metrics across time.

Reviewer notes asked for: fewer, stronger features; EPSS-style exploit
features; a learned two-layer boost on top of EPSS; metrics a patching team
feels rather than an AUC. This runs every candidate through the same
rolling-origin windows as verify_banded.py.

Decision rule, fixed before any result is seen:
  Choose the variant with the highest mean coverage at 1%, 2% and 5% effort,
  averaged over the five development windows (2022 H2 – 2024 H2). The 2025
  window is then reported for the chosen variant as confirmation only — it is
  not used to choose. All variants are printed on 2025 for transparency; a
  variant that only wins there does not get picked.

Variants
  V0 EPSS            FIRST's score alone
  V1 TriageBand      EPSS band, content model (all PIT-safe features) within
  V2 lean            V1 with the content model cut to its top-K features
  V3 +exploit        V1 plus dated exploit-availability features
  V4 lean+exploit    V3 cut to top-K
  V5 EPSS-lite       V4's content model alone, no EPSS input at all
  V6 stack A         layer 2 boost on [layer-1 score, EPSS, ~5 key features]
  V7 stack B         layer 2 boost on [layer-1 score, EPSS] only
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.features.exploit_features import EXPLOIT_COLS, attach_exploit_features, load_events
from src.model.dataset import load_pit_frame
from src.model.evaluate import Featurizer, fit_xgb
from src.model.metrics import calibration, coverage_efficiency, effort_for_coverage
from src.model.two_stage import banded_score
from src.model.verify_banded import FEATURE_SETS, FRESH_END, PRIMARY, WINDOWS

TOP_K = 50                       # fixed a priori; not tuned
DECISION_EFFORTS = (0.01, 0.02, 0.05)
STACK_A_EXTRA = ["github_pocs", "any_exploit", "base_score", "exploitability_score",
                 "privileges_required_NONE"]
DEV, FRESH = WINDOWS[:-1], WINDOWS[-1]


def top_k_columns(model, k: int) -> np.ndarray:
    """Indices of a fitted model's k highest-gain features (it saw training data only)."""
    gain = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(gain, key=gain.get, reverse=True)
    return np.array(sorted(int(f[1:]) for f in ranked[:k]))


class Content:
    """Layer-1 content model: featurize, optionally cut to top-K, seed ensemble."""

    def __init__(self, num, cat, k: int | None, seeds: int, select_from: "Content" = None,
                 keep_names: list[str] | None = None):
        self.feat, self.k, self.seeds = Featurizer(num, cat), k, seeds
        self.select_from, self.keep_names = select_from, keep_names

    def fit(self, df: pd.DataFrame) -> "Content":
        X, y = self.feat.fit_transform(df), df["is_kev"].values
        if self.keep_names is not None:
            # Same features by name: a refit TF-IDF vocabulary reorders columns.
            idx = {n: i for i, n in enumerate(self.feat.names())}
            self.cols = np.array(sorted(idx[n] for n in self.keep_names if n in idx))
        elif self.k:
            # Ranked by the full model's first seed — fitted on the same training
            # rows, so selection never sees test data.
            self.cols = top_k_columns(self.select_from.models[0], self.k)
        else:
            self.cols = np.arange(X.shape[1])
        X = X[:, self.cols]
        self.models = [fit_xgb(X, y, seed=s) for s in range(self.seeds)]
        return self

    def matrix(self, df):
        return self.feat.transform(df)[:, self.cols]

    def names(self) -> list[str]:
        return [self.feat.names()[i] for i in self.cols]

    def predict(self, df) -> np.ndarray:
        X = self.matrix(df)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)


def oof_scores(train: pd.DataFrame, make, folds: int = 5) -> np.ndarray:
    """Out-of-fold layer-1 scores on train, in contiguous time blocks."""
    order = np.argsort(train["published_date"].values, kind="stable")
    out = np.empty(len(train))
    for block in np.array_split(order, folds):
        mask = np.ones(len(train), bool); mask[block] = False
        out[block] = make().fit(train.iloc[mask]).predict(train.iloc[block])
    return out


def stack_frame(df, layer1, cols, content: Content) -> pd.DataFrame:
    f = pd.DataFrame({"layer1": layer1, "epss": df["epss_score"].values})
    if cols:
        X = content.feat.transform(df)
        names = content.feat.names()
        for c in cols:
            if c in df.columns:
                f[c] = df[c].astype(float).values
            elif c in names:
                f[c] = X[:, names.index(c)].toarray().ravel()
    return f


def fit_stack(f: pd.DataFrame, y) -> XGBClassifier:
    # Shallow, monotone non-decreasing in every input: more risk signal can
    # never lower the score. No class reweighting, so outputs stay probabilities.
    m = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                      min_child_weight=5, monotone_constraints=tuple([1] * f.shape[1]),
                      eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=0)
    return m.fit(f, y)


def run_window(df, start, end, seeds) -> dict:
    stop = pd.Timestamp(end) + pd.Timedelta(days=1)
    train = df[df["published_date"] < start]
    test = df[(df["published_date"] >= start) & (df["published_date"] < stop)]
    y, e = test["is_kev"].values, test["epss_score"].values
    num, cat = FEATURE_SETS[PRIMARY]
    num_x = list(num) + list(EXPLOIT_COLS)

    s = {"V0 EPSS": np.nan_to_num(e, nan=-1.0)}
    c1 = Content(num, cat, None, seeds).fit(train)
    s["V1 TriageBand"] = banded_score(e, c1.predict(test))
    c2 = Content(num, cat, TOP_K, seeds, select_from=c1).fit(train)
    s["V2 lean"] = banded_score(e, c2.predict(test))
    c3 = Content(num_x, cat, None, seeds).fit(train)
    s["V3 +exploit"] = banded_score(e, c3.predict(test))
    c4 = Content(num_x, cat, TOP_K, seeds, select_from=c3).fit(train)
    p4 = c4.predict(test)
    s["V4 lean+exploit"] = banded_score(e, p4)
    s["V5 EPSS-lite"] = p4

    lean = c4.names()
    oof = oof_scores(train, lambda: Content(num_x, cat, None, 1, keep_names=lean))
    probs = {}
    for name, extra in (("V6 stack A", STACK_A_EXTRA), ("V7 stack B", [])):
        m = fit_stack(stack_frame(train, oof, extra, c4), train["is_kev"].values)
        probs[name] = m.predict_proba(stack_frame(test, p4, extra, c4))[:, 1]
        s[name] = probs[name]

    res = {"start": start, "end": end, "n": len(test), "positives": int(y.sum()),
           "lean_features": c4.names(), "variants": {}}
    for name, sc in s.items():
        res["variants"][name] = {
            "coverage_efficiency": {str(k): v for k, v in coverage_efficiency(y, sc).items()},
            "effort_for_coverage": {str(k): v for k, v in effort_for_coverage(y, sc).items()},
            "pr_auc": float(average_precision_score(y, sc)),
        }
    res["calibration"] = {"V0 EPSS": calibration(y, np.nan_to_num(e, nan=0.0)),
                          **{k: calibration(y, v) for k, v in probs.items()}}
    return res


def decision_score(v: dict) -> float:
    return float(np.mean([v["coverage_efficiency"][str(e)]["coverage"] for e in DECISION_EFFORTS]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="data/processed/compare.json")
    args = ap.parse_args()

    df = attach_exploit_features(load_pit_frame(365, FRESH_END, verbose=False), load_events())
    results = []
    for start, end in WINDOWS:
        r = run_window(df, start, end, args.seeds)
        results.append(r)
        cells = "  ".join(f"{k.split()[0]}:{decision_score(v):.3f}" for k, v in r["variants"].items())
        print(f"{start}  n={r['n']:,} pos={r['positives']}  {cells}", flush=True)

    names = list(results[0]["variants"])
    dev = {k: float(np.mean([decision_score(r["variants"][k]) for r in results[:-1]])) for k in names}
    chosen = max((k for k in names if k != "V0 EPSS"), key=dev.get)
    print("\nmean coverage at 1/2/5% effort — development windows (decides):")
    for k in names:
        print(f"  {k:<18s} {dev[k]:.3f}" + ("   ← chosen" if k == chosen else ""))
    fr = results[-1]["variants"]
    print("\n2025 holdout (confirms; not used to choose):")
    for k in names:
        print(f"  {k:<18s} {decision_score(fr[k]):.3f}")
    with open(args.out, "w") as f:
        json.dump({"decision_rule": __doc__.split("Decision rule")[1].split("Variants")[0].strip(),
                   "top_k": TOP_K, "dev_scores": dev, "chosen": chosen, "windows": results}, f,
                  indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
