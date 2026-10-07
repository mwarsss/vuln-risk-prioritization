"""Fit the serving bundle on the point-in-time training window.

Replaces the scaffold trainer (random 5-fold CV, `cvss + epss` features,
isotonic calibration over `scale_pos_weight`). Random folds mix publication
dates and reward leakage; the split here is temporal, matching every evaluation
in FINDINGS. Evaluation lives in `two_stage.py`; this only produces the artifact.

    python -m src.model.train [--out model.joblib]
"""
from __future__ import annotations

import argparse

from src.model.bundle import fit_bundle
from src.model.dataset import load_pit_frame, temporal_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", default="2022-12-31")
    ap.add_argument("--val-end", default="2023-06-30")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    df = load_pit_frame(args.horizon_days, args.test_end)
    fit, _, _ = temporal_split(df, args.fit_end, args.val_end)
    bundle = fit_bundle(fit)
    bundle.save(args.out)
    print(f"fit on {len(fit):,} CVEs ({int(fit['is_kev'].sum())} positives) -> {args.out}")


if __name__ == "__main__":
    main()
