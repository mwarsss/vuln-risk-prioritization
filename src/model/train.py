"""Fit the deployable content model behind the banded ranker.

Exactly the configuration FINDINGS section 11 verified: point-in-time-safe NVD
features only (no CPE-derived columns), the same XGBoost settings, and the mean
of five seeds. Trained on every CVE whose 365-day label is mature, through the
validated loader — so the schema gate runs and rows CISA listed within a day of
publication are dropped, as in evaluation.

The bundle holds the fitted encoders with the models: a live CVE must be encoded
by the vocabulary the model was trained on, not one refitted at scoring time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.model.dataset import load_pit_frame
from src.model.evaluate import Featurizer, fit_xgb
from src.model.two_stage import BAND_EDGES
from src.model.verify_banded import FEATURE_SETS, FRESH_END, PRIMARY

MODEL_PATH = Path("models/banded.joblib")


def train(end: str = FRESH_END, seeds: int = 5, out: Path = MODEL_PATH) -> dict:
    df = load_pit_frame(365, end)
    num, cat = FEATURE_SETS[PRIMARY]
    feat = Featurizer(num, cat)
    X = feat.fit_transform(df)
    y = df["is_kev"].values
    print(f"training on {len(df):,} CVEs ({int(y.sum())} exploited within 365 days), "
          f"published {df['published_date'].min().date()} → {df['published_date'].max().date()}")
    models = [fit_xgb(X, y, seed=s) for s in range(seeds)]

    bundle = {
        "featurizer": feat,
        "models": models,
        "band_edges": BAND_EDGES,
        "features": {"numeric": list(num), "categorical": list(cat), "text": "description"},
        "trained_through": end,
        "n_train": len(df),
        "n_positive": int(y.sum()),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    in_sample = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    print(f"wrote {out} — mean in-sample score {in_sample[y].mean():.3f} exploited vs "
          f"{in_sample[~y].mean():.3f} not (sanity only; see FINDINGS §11 for real performance)")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=FRESH_END, help="last publication date with a mature label")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    train(args.end, args.seeds, args.out)


if __name__ == "__main__":
    main()
