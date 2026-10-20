"""Try to break the banded result (FINDINGS section 10) before anyone else does.

Section 10 reports one test window, 5 seeds, edges read off that same window.
Seeds measure model noise, not sampling noise over 209 positives, and the
window was the one every earlier decision was made on. Six checks:

  1. Enrichment audit — do NVD fields filled in *after* publication track the
     label? The features come from a 2026 NVD mirror; NVD prioritised KEV CVEs
     through its 2024 analysis backlog.
  2. Feature ablation — banded with all features, with the CPE-derived ones
     removed, with description text only, and with CVSS (no model at all).
  3. Paired bootstrap over CVEs — confidence intervals for the gain itself.
  4. Within-band permutation null — shuffle the content score inside each
     band. If the gain survives, it was never the model.
  5. Rolling-origin backtest — six consecutive 6-month windows, each trained
     only on what was published before it.
  6. The last of those windows, 2025-01 → 2025-08-06, is a fresh holdout: no
     analysis in this repo had loaded it before this script.

Nothing here selects anything. Every variant is printed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.model.dataset import load_pit_frame
from src.model.evaluate import CAT_COLS, NUM_COLS, build_matrices, fit_xgb
from src.model.two_stage import banded_score

# vendor / n_vendors / n_cpe are parsed from NVD's CPE list, which NVD adds
# during analysis — days to months after publication, and in 2024-25 only for
# CVEs it prioritised, KEV among them. Not available at publication + 1 day.
CPE_DERIVED = ("vendor", "n_vendors", "n_cpe")
FEATURE_SETS = {
    "all features": (NUM_COLS, CAT_COLS),
    "no CPE-derived": ([c for c in NUM_COLS if c not in CPE_DERIVED],
                       [c for c in CAT_COLS if c not in CPE_DERIVED]),
    "description only": ([], []),
}
PRIMARY = "no CPE-derived"

BUDGETS = [100, 200, 500, 1000, 2000, 5000, 10000]
ROLL_BUDGETS = [100, 250, 500, 1000, 2000]
# Original section 9-10 split, so ablations compare against reported numbers.
ORIG_FIT_END, ORIG_VAL_END, ORIG_TEST_END = "2022-12-31", "2023-06-30", "2024-12-31"
# KEV pull is 2026-08-07; a 365-day label is mature through 2025-08-06.
FRESH_END = "2025-08-06"
WINDOWS = [("2022-07-01", "2022-12-31"), ("2023-01-01", "2023-06-30"),
           ("2023-07-01", "2023-12-31"), ("2024-01-01", "2024-06-30"),
           ("2024-07-01", "2024-12-31"), ("2025-01-01", FRESH_END)]


def hits_at(y: np.ndarray, scores: np.ndarray, budgets) -> list[int]:
    cum = np.cumsum(y[np.argsort(-scores, kind="stable")])
    return [int(cum[min(b, len(cum)) - 1]) for b in budgets]


def epss_rank(epss: np.ndarray) -> np.ndarray:
    return np.nan_to_num(epss, nan=-1.0)


def content_preds(train, test, feats, seeds):
    num, cat = feats
    Xtr, Xte, _ = build_matrices(train, test, use_epss=False, num_cols=num, cat_cols=cat)
    y = train["is_kev"].values
    return [fit_xgb(Xtr, y, seed=s).predict_proba(Xte)[:, 1] for s in range(seeds)]


def audit(df: pd.DataFrame) -> dict:
    print("1. ENRICHMENT AUDIT — share of CVEs with no CPE data (vendor UNKNOWN, n_cpe 0)")
    out = {}
    for yr, g in df.groupby(df["published_date"].dt.year):
        m = g["n_cpe"].fillna(0) == 0
        neg, pos = float(m[~g["is_kev"]].mean()), float(m[g["is_kev"]].mean())
        out[int(yr)] = {"no_cpe_negatives": neg, "no_cpe_positives": pos,
                        "positives": int(g["is_kev"].sum())}
        print(f"   {yr}: not exploited {neg:6.1%}   exploited {pos:6.1%}   "
              f"({int(g['is_kev'].sum())} exploited)")
    print("   -> a missing CPE list marks 'NVD never analysed this', which in 2024-25 "
          "means 'not prioritised', which means 'not in KEV'.\n")
    return out


def ablation(df: pd.DataFrame, seeds: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    print(f"2. FEATURE ABLATION — original split, test {ORIG_VAL_END} → {ORIG_TEST_END}, "
          f"{seeds} seeds")
    fit = df[df["published_date"] <= ORIG_FIT_END]
    # Same comparisons temporal_split and load_pit_frame make, so this frame is
    # row-for-row the section 9-10 test set.
    test = df[(df["published_date"] > ORIG_VAL_END) & (df["published_date"] <= ORIG_TEST_END)]
    y, e = test["is_kev"].values, test["epss_score"].values
    print(f"   test: {len(test):,} CVEs, {int(y.sum())} exploited")

    rows = {"EPSS alone": (hits_at(y, epss_rank(e), BUDGETS),
                           [average_precision_score(y, epss_rank(e))])}
    cvss = test["base_score"].astype(float).fillna(0).values
    rows["banded, CVSS within band"] = (hits_at(y, banded_score(e, cvss), BUDGETS),
                                        [average_precision_score(y, banded_score(e, cvss))])
    primary_preds = None
    for name, feats in FEATURE_SETS.items():
        preds = content_preds(fit, test, feats, seeds)
        if name == PRIMARY:
            primary_preds = preds
        per_seed = [hits_at(y, banded_score(e, p), BUDGETS) for p in preds]
        rows[f"banded, model: {name}"] = (
            list(np.mean(per_seed, axis=0)),
            [average_precision_score(y, banded_score(e, p)) for p in preds])

    head = "".join(f"{b:>8,}" for b in BUDGETS)
    print(f"   {'ranker':<38s}{head}   PR-AUC")
    out = {}
    for name, (h, aps) in rows.items():
        print(f"   {name:<38s}" + "".join(f"{v:>8.1f}" for v in h) +
              f"   {np.mean(aps):.4f}" + (f" ± {np.std(aps):.4f}" if len(aps) > 1 else ""))
        out[name] = {"hits": dict(zip(map(str, BUDGETS), map(float, h))),
                     "pr_auc_mean": float(np.mean(aps)), "pr_auc_sd": float(np.std(aps))}
    print()
    return out, y, e, np.mean(primary_preds, axis=0)


def bootstrap(y, e, content, reps: int, label: str = "3. PAIRED BOOTSTRAP",
              budgets=BUDGETS) -> dict:
    print(f"{label} — {reps:,} resamples of test CVEs, model: {PRIMARY}")
    rng = np.random.default_rng(0)
    base, band = epss_rank(e), banded_score(e, content)
    d_hits, d_ap = [], []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        if yi.sum() == 0:
            continue
        d_hits.append(np.subtract(hits_at(yi, band[i], budgets), hits_at(yi, base[i], budgets)))
        d_ap.append(average_precision_score(yi, band[i]) - average_precision_score(yi, base[i]))
    d_hits, d_ap = np.array(d_hits), np.array(d_ap)
    out = {}
    for j, b in enumerate(budgets):
        lo, hi = np.percentile(d_hits[:, j], [2.5, 97.5])
        p_le0 = float((d_hits[:, j] <= 0).mean())
        out[str(b)] = {"mean": float(d_hits[:, j].mean()), "ci95": [float(lo), float(hi)],
                       "share_not_better": p_le0}
        print(f"   budget {b:>6,}: gain {d_hits[:, j].mean():+6.1f}  95% CI [{lo:+.0f}, {hi:+.0f}]"
              f"  resamples with no gain {p_le0:.1%}")
    lo, hi = np.percentile(d_ap, [2.5, 97.5])
    out["pr_auc"] = {"mean": float(d_ap.mean()), "ci95": [float(lo), float(hi)],
                     "share_not_better": float((d_ap <= 0).mean())}
    print(f"   PR-AUC     : gain {d_ap.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  resamples with no gain {(d_ap <= 0).mean():.1%}\n")
    return out


def permutation_null(y, e, content, draws: int) -> dict:
    print(f"4. WITHIN-BAND PERMUTATION NULL — content score shuffled inside each band, "
          f"{draws} draws")
    rng = np.random.default_rng(1)
    band_id = np.floor(banded_score(e, content)).astype(int)
    null = []
    for _ in range(draws):
        shuffled = content.copy()
        for b in np.unique(band_id):
            m = band_id == b
            shuffled[m] = rng.permutation(content[m])
        null.append(hits_at(y, banded_score(e, shuffled), BUDGETS))
    null = np.array(null)
    real = hits_at(y, banded_score(e, content), BUDGETS)
    epss = hits_at(y, epss_rank(e), BUDGETS)
    out = {}
    for j, b in enumerate(BUDGETS):
        out[str(b)] = {"real": real[j], "null_mean": float(null[:, j].mean()),
                       "null_max": int(null[:, j].max()), "epss": epss[j]}
        print(f"   budget {b:>6,}: real {real[j]:>4}   shuffled {null[:, j].mean():6.1f} "
              f"(max {null[:, j].max():>3})   EPSS {epss[j]:>4}")
    print("   -> if 'shuffled' sat near 'real', the bands alone would be doing the work.\n")
    return out


def rolling(df: pd.DataFrame, seeds: int) -> list[dict]:
    print(f"5-6. ROLLING-ORIGIN BACKTEST — model: {PRIMARY}, {seeds} seeds per window; "
          f"last window is the fresh holdout")
    head = "".join(f"{b:>13,}" for b in ROLL_BUDGETS)
    print(f"   {'window':<24s}{'CVEs':>7s}{'expl':>5s}{head}{'PR-AUC EPSS→banded':>24s}")
    out = []
    for start, end in WINDOWS:
        train = df[df["published_date"] < start]
        # Half-open on whole days: published_date carries a time of day, so
        # `<= end` would drop every CVE published during the window's last day.
        stop = pd.Timestamp(end) + pd.Timedelta(days=1)
        test = df[(df["published_date"] >= start) & (df["published_date"] < stop)]
        y, e = test["is_kev"].values, test["epss_score"].values
        preds = content_preds(train, test, FEATURE_SETS[PRIMARY], seeds)
        h_e = hits_at(y, epss_rank(e), ROLL_BUDGETS)
        per_seed = np.array([hits_at(y, banded_score(e, p), ROLL_BUDGETS) for p in preds])
        h_b = per_seed.mean(axis=0)
        wins = (per_seed >= np.array(h_e)).all(axis=1).sum()
        ap_e = average_precision_score(y, epss_rank(e))
        ap_b = [average_precision_score(y, banded_score(e, p)) for p in preds]
        cells = "".join(f"{he:>5}→{hb:>6.1f} " for he, hb in zip(h_e, h_b))
        tag = "  ← fresh" if end == FRESH_END else ""
        print(f"   {start}→{end}{len(test):>7,}{int(y.sum()):>5} {cells}"
              f"{ap_e:>9.4f} → {np.mean(ap_b):.4f}{tag}")
        out.append({"start": start, "end": end, "n": len(test), "positives": int(y.sum()),
                    "train_positives": int(train["is_kev"].sum()),
                    "epss_hits": dict(zip(map(str, ROLL_BUDGETS), h_e)),
                    "banded_hits_seed_mean": dict(zip(map(str, ROLL_BUDGETS), map(float, h_b))),
                    "seeds_not_worse_at_every_budget": int(wins),
                    "pr_auc_epss": float(ap_e), "pr_auc_banded_per_seed": list(map(float, ap_b))})
    print("   cells are EPSS → banded hits at each budget\n")
    return out, (y, e, np.mean(preds, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--roll-seeds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--out", default="data/processed/verify_banded.json")
    args = ap.parse_args()

    df = load_pit_frame(365, FRESH_END)
    print()
    res = {"audit": audit(df)}
    res["ablation"], y, e, content = ablation(df, args.seeds)
    res["bootstrap"] = bootstrap(y, e, content, args.boot)
    res["permutation_null"] = permutation_null(y, e, content, args.perm)
    res["rolling"], fresh = rolling(df, args.roll_seeds)
    res["fresh_bootstrap"] = bootstrap(*fresh, args.boot,
                                       label="6b. FRESH-HOLDOUT BOOTSTRAP (2025)",
                                       budgets=ROLL_BUDGETS)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
