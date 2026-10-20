"""Temporal evaluation of exploitation-probability models.

Compares four rankers on a held-out *future* window:
  1. CVSS base score      — the incumbent heuristic this project exists to replace
  2. EPSS score           — FIRST's published probability, the external baseline
  3. XGBoost, no EPSS     — what the model learns from CVE content alone
  4. XGBoost + EPSS       — the full stack

Split is temporal, never random. A random split lets the model see CVEs published
after the ones it is scored on, and KEV membership is heavily correlated with
publication era, so random CV reports a number the deployed system can never hit.

Reported metric is PR-AUC plus precision/recall@k. @k is the operational metric:
a team patches a bounded number of CVEs per cycle, so what matters is how many
true exploited vulns land in the top k, not global accuracy.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

IN_PATH = "data/processed/features.parquet"

CAT_COLS = [
    "base_severity", "attack_vector", "attack_complexity", "privileges_required",
    "user_interaction", "scope", "confidentiality_impact", "integrity_impact",
    "availability_impact", "primary_cwe", "vendor",
]
NUM_COLS = ["base_score", "exploitability_score", "impact_score",
            "n_cwe", "n_vendors", "n_cpe", "desc_len"]

K_VALUES = (100, 500, 1000)


@dataclass
class Result:
    name: str
    pr_auc: float
    roc_auc: float
    prec_at_k: dict
    recall_at_k: dict
    lift_at_100: float


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Rank by score, take the top k, measure how many true positives are in there."""
    k = min(k, len(scores))
    top = np.argsort(-scores, kind="stable")[:k]
    hits = y_true[top].sum()
    return hits / k, hits / max(y_true.sum(), 1)


def score_ranker(name: str, y_true: np.ndarray, scores: np.ndarray) -> Result:
    # NaNs rank last rather than poisoning the sort — a missing EPSS score is
    # "no evidence", not "low risk".
    scores = np.nan_to_num(scores, nan=-1.0)
    base_rate = y_true.mean()
    prec, rec = {}, {}
    for k in K_VALUES:
        p, r = precision_recall_at_k(y_true, scores, k)
        prec[k], rec[k] = p, r
    return Result(
        name=name,
        pr_auc=average_precision_score(y_true, scores),
        roc_auc=roc_auc_score(y_true, scores),
        prec_at_k=prec,
        recall_at_k=rec,
        lift_at_100=prec[100] / base_rate if base_rate > 0 else float("nan"),
    )


class Featurizer:
    """Numeric + one-hot + TF-IDF encoding, fitted once and reusable.

    Split out of build_matrices so the trained encoders can be saved alongside
    the model (train.py) and applied to live CVEs (rank.py) exactly as in
    evaluation.
    """

    def __init__(self, num_cols, cat_cols):
        self.num, self.cats = list(num_cols), list(cat_cols)

    def fit(self, df: pd.DataFrame) -> "Featurizer":
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame):
        # fit_transform, not fit-then-transform: TF-IDF rounds the two paths
        # differently in the last bit, and published numbers come from this one.
        blocks = []
        if self.num:
            blocks.append(csr_matrix(df[self.num].astype(float).fillna(-1).values))
        if self.cats:
            self.enc = OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=True)
            blocks.append(self.enc.fit_transform(df[self.cats].fillna("NA").astype(str)))
        self.tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                     min_df=5, stop_words="english", sublinear_tf=True)
        blocks.append(self.tfidf.fit_transform(df["description"].fillna("")))
        return hstack(blocks).tocsr()

    def transform(self, df: pd.DataFrame):
        blocks = []
        if self.num:
            blocks.append(csr_matrix(df[self.num].astype(float).fillna(-1).values))
        if self.cats:
            blocks.append(self.enc.transform(df[self.cats].fillna("NA").astype(str)))
        blocks.append(self.tfidf.transform(df["description"].fillna("")))
        return hstack(blocks).tocsr()

    def names(self) -> list[str]:
        out = list(self.num)
        if self.cats:
            out += list(self.enc.get_feature_names_out(self.cats))
        return out + [f"tfidf::{t}" for t in self.tfidf.get_feature_names_out()]


def build_matrices(train: pd.DataFrame, test: pd.DataFrame, use_epss: bool,
                   num_cols=None, cat_cols=None):
    """Fit encoders on train only, then apply to test (no test-set fitting).

    `num_cols` / `cat_cols` default to NUM_COLS / CAT_COLS; verify_banded.py
    passes narrower sets for its feature ablations.
    """
    num = list(NUM_COLS if num_cols is None else num_cols) + (["epss_score"] if use_epss else [])
    cats = list(CAT_COLS if cat_cols is None else cat_cols)
    f = Featurizer(num, cats)
    Xtr = f.fit_transform(train)
    return Xtr, f.transform(test), f.names()


def fit_xgb(Xtr, ytr, seed: int = 42) -> XGBClassifier:
    pos = max(int(ytr.sum()), 1)
    neg = len(ytr) - pos
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(Xtr, ytr)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2022-12-31",
                    help="train on CVEs published on/before this date")
    ap.add_argument("--test-end", default="2024-12-31",
                    help="test window ends here; later CVEs are dropped as label-immature")
    ap.add_argument("--min-year", type=int, default=2005)
    ap.add_argument("--out", default="data/processed/eval_results.json")
    args = ap.parse_args()

    df = pd.read_parquet(IN_PATH)
    df = df[df["published_year"] >= args.min_year]
    df = df.dropna(subset=["published_date"])

    train = df[df["published_date"] <= args.train_end]
    test = df[(df["published_date"] > args.train_end) &
              (df["published_date"] <= args.test_end)]

    print(f"train: {len(train):,} rows, {train.is_kev.sum():,} KEV ({train.is_kev.mean():.3%})")
    print(f"test : {len(test):,} rows, {test.is_kev.sum():,} KEV ({test.is_kev.mean():.3%})")
    print(f"(dropped {len(df) - len(train) - len(test):,} rows published after {args.test_end} "
          "— labels not yet mature)\n")

    ytr = train["is_kev"].values
    yte = test["is_kev"].values

    results = [
        score_ranker("CVSS base score", yte, test["base_score"].values),
        score_ranker("EPSS score", yte, test["epss_score"].values),
    ]

    for use_epss in (False, True):
        tag = "XGBoost + EPSS" if use_epss else "XGBoost (no EPSS)"
        Xtr, Xte, names = build_matrices(train, test, use_epss)
        model = fit_xgb(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        results.append(score_ranker(tag, yte, p))

        if not use_epss:
            imp = model.feature_importances_
            top = np.argsort(-imp)[:20]
            print("--- top 20 features (XGBoost, no EPSS) ---")
            for i in top:
                print(f"  {names[i]:<45s} {imp[i]:.4f}")
            print()

    base_rate = yte.mean()
    print(f"test base rate: {base_rate:.4%}\n")
    header = f"{'model':<20s} {'PR-AUC':>8s} {'ROC-AUC':>8s} " + \
             " ".join(f"{'P@'+str(k):>8s}" for k in K_VALUES) + \
             " " + " ".join(f"{'R@'+str(k):>8s}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r.name:<20s} {r.pr_auc:>8.4f} {r.roc_auc:>8.4f} "
        row += " ".join(f"{r.prec_at_k[k]:>8.3f}" for k in K_VALUES)
        row += " " + " ".join(f"{r.recall_at_k[k]:>8.3f}" for k in K_VALUES)
        print(row)

    print(f"\nlift @100 vs random: " +
          ", ".join(f"{r.name}={r.lift_at_100:.1f}x" for r in results))

    with open(args.out, "w") as f:
        json.dump({"base_rate": float(base_rate),
                   "train_end": args.train_end, "test_end": args.test_end,
                   "n_train": len(train), "n_test": len(test),
                   "results": [asdict(r) for r in results]}, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
