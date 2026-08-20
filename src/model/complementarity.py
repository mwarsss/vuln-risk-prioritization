"""Where, if anywhere, does the content model know something EPSS does not?

Beating EPSS outright is the wrong target. FIRST trains EPSS on ~120M
vulnerability-days with >1.6M observed exploitation events and ~2,850 features,
including two families this project has no access to: public exploit-code
availability (Exploit-DB, Metasploit, Nuclei, Shodan adoption) and "chatter"
across vendor advisories, social platforms and disclosure programs. This project
has ~300 positive labels and NVD metadata. Matching that head-on is not a
realistic goal.

FIRST's own documentation names the gap worth aiming at instead:

    "Sparse signal for newly disclosed vulnerabilities"
    "Cold start: all practitioners face sparse early information"

Exploit code and chatter both accumulate *after* disclosure. A CVE description
exists at minute zero. So the plausible niche is the early window, before EPSS's
strongest features have anything to read.

This script tests that directly rather than assuming it, in two cuts:

  1. By EPSS band — does the content model rank well inside the low-EPSS
     population, i.e. among CVEs that EPSS has effectively dismissed? Those are
     the misses that actually hurt a triage team.
  2. By EPSS availability — CVEs with no EPSS score at all are the extreme cold
     start. If the content model carries them, that is a deployable role
     regardless of what the blend does.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score

from src.model.dataset import load_pit_frame, temporal_split
from src.model.evaluate import build_matrices, fit_xgb, precision_recall_at_k

BANDS = [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.01)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2023-06-30")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quantize", default="D",
                    help="EPSS snapshot alignment; 'D' = publication + 1 day")
    ap.add_argument("--out", default="data/processed/complementarity.json")
    args = ap.parse_args()

    df = load_pit_frame(args.horizon_days, args.test_end, quantize=args.quantize)
    train, _, test = temporal_split(df, args.train_end)
    test = test.reset_index(drop=True)
    ytr, yte = train["is_kev"].values, test["is_kev"].values

    Xtr, Xte, _ = build_matrices(train, test, use_epss=False)
    preds = np.mean([fit_xgb(Xtr, ytr, seed=s).predict_proba(Xte)[:, 1]
                     for s in range(args.seeds)], axis=0)
    epss = test["epss_score"].values

    out: dict = {"horizon_days": args.horizon_days, "n_test": len(test),
                 "test_base_rate": float(yte.mean())}

    print(f"test: {len(test):,} CVEs, {yte.sum():,} positive ({yte.mean():.3%})")
    print(f"content-model predictions averaged over {args.seeds} seeds\n")

    # --- Cut 1: within EPSS bands ---
    print("=" * 78)
    print("CUT 1: does the content model rank within a band EPSS has already scored?")
    print("=" * 78)
    print(f"{'EPSS band':<18s} {'n':>8s} {'pos':>6s} {'rate':>8s} "
          f"{'EPSS AP':>9s} {'content AP':>11s}")
    print("-" * 78)
    bands_out = []
    for lo, hi in BANDS:
        m = (epss >= lo) & (epss < hi) & ~np.isnan(epss)
        n, pos = int(m.sum()), int(yte[m].sum())
        if n == 0:
            continue
        row = {"lo": lo, "hi": hi, "n": n, "positives": pos,
               "rate": float(yte[m].mean())}
        if pos >= 5:
            # Within a band EPSS is near-constant, so its AP here is close to the
            # band base rate; the content model having a higher AP means it is
            # ordering CVEs that EPSS cannot separate.
            row["epss_ap"] = float(average_precision_score(yte[m], epss[m]))
            row["content_ap"] = float(average_precision_score(yte[m], preds[m]))
            print(f"[{lo:<6.3f},{hi:>5.2f})  {n:>8,} {pos:>6,} {yte[m].mean():>7.3%} "
                  f"{row['epss_ap']:>9.4f} {row['content_ap']:>11.4f}")
        else:
            print(f"[{lo:<6.3f},{hi:>5.2f})  {n:>8,} {pos:>6,} {yte[m].mean():>7.3%} "
                  f"{'too few':>9s} {'too few':>11s}")
        bands_out.append(row)
    out["bands"] = bands_out

    # --- Cut 2: the low-EPSS population, where triage misses live ---
    print("\n" + "=" * 78)
    print("CUT 2: the population EPSS has dismissed (score < 0.01)")
    print("=" * 78)
    dismissed = (epss < 0.01) & ~np.isnan(epss)
    n_d, pos_d = int(dismissed.sum()), int(yte[dismissed].sum())
    print(f"{n_d:,} CVEs ({n_d / len(test):.1%} of test) carry EPSS < 0.01")
    print(f"  of those, {pos_d:,} were exploited within {args.horizon_days}d "
          f"({yte[dismissed].mean():.3%}) — EPSS's misses")
    if pos_d >= 5:
        ap_c = average_precision_score(yte[dismissed], preds[dismissed])
        p100 = precision_recall_at_k(yte[dismissed], preds[dismissed], 100)[0]
        lift = ap_c / max(yte[dismissed].mean(), 1e-9)
        print(f"  content model on this subset: AP={ap_c:.4f}  P@100={p100:.3f}  "
              f"lift={lift:.1f}x over the subset base rate")
        out["dismissed"] = {"n": n_d, "positives": pos_d,
                            "base_rate": float(yte[dismissed].mean()),
                            "content_ap": float(ap_c), "content_p_at_100": float(p100),
                            "lift": float(lift)}

    # --- Cut 3: no EPSS at all — the true cold start ---
    print("\n" + "=" * 78)
    print("CUT 3: CVEs with no EPSS score at the snapshot (true cold start)")
    print("=" * 78)
    missing = np.isnan(epss)
    n_m, pos_m = int(missing.sum()), int(yte[missing].sum())
    print(f"{n_m:,} CVEs ({n_m / len(test):.1%} of test), {pos_m:,} exploited")
    if pos_m >= 5:
        ap_m = average_precision_score(yte[missing], preds[missing])
        print(f"  content model: AP={ap_m:.4f} against base rate "
              f"{yte[missing].mean():.3%} "
              f"({ap_m / max(yte[missing].mean(), 1e-9):.1f}x lift)")
        out["cold_start"] = {"n": n_m, "positives": pos_m,
                             "base_rate": float(yte[missing].mean()),
                             "content_ap": float(ap_m)}
    else:
        print("  too few positives to score — EPSS coverage is near-total here")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
