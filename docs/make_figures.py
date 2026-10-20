"""Render the write-up figures into docs/.

    python docs/make_figures.py

Numbers come from data/processed/verify_banded.json and FINDINGS.md; terminal
screenshots are rendered from real captured output in docs/terminal/*.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).parent
SURF, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _title(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=13, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=9.5, color=INK2, va="top")


def method_diagram(out: Path) -> None:
    """Same bugs, same bands; only the order inside each band changes."""
    bands = [("EPSS ≥ 0.1", "50%"), ("0.01 – 0.1", "76%"),
             ("0.001 – 0.01", "10%"), ("< 0.001", "0.26%")]
    n = 18
    # Illustration: exploited positions per band, EPSS order vs model order.
    epss_pos = [{2, 9, 13, 16}, {0, 3, 4, 6, 8, 10, 11, 14, 15, 17}, {7, 15}, {5, 12, 16}]
    model_pos = [{0, 1, 3, 6}, {0, 1, 2, 3, 4, 5, 6, 8, 10, 13}, {0, 2}, {0, 1, 4}]

    fig = plt.figure(figsize=(10, 5.2), dpi=200, facecolor=SURF)
    # One full-figure axes in figure units: scatter markers stay round, unlike
    # patches drawn in figure coordinates on a non-square figure.
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _title(fig, "EPSS picks the band. The model picks the order inside it.",
           "Illustration, not to scale · each dot is a bug · patch order: top band first, "
           "then left to right")
    for col, (label, pos, x0) in enumerate((("EPSS alone", epss_pos, 0.17),
                                            ("EPSS band + model", model_pos, 0.59))):
        fig.text(x0, 0.79, label, fontsize=11, weight="bold", color=BLUE if col else ORANGE)
        for r, ((name, rate), hits) in enumerate(zip(bands, pos)):
            y = 0.68 - r * 0.135
            xs = [x0 + i * 0.0205 for i in range(n)]
            ax.scatter(xs, [y] * n, s=70, linewidths=1.2,
                       c=[INK if i in hits else SURF for i in range(n)],
                       edgecolors=[INK if i in hits else MUTED for i in range(n)])
            if col == 0:
                fig.text(0.04, y + 0.012, name, fontsize=9.5, color=INK, va="center")
                fig.text(0.04, y - 0.03, f"{rate} exploited", fontsize=8.5, color=INK2,
                         va="center")
    fig.text(0.17, 0.08, "●  exploited      ○  not exploited", fontsize=9.5, color=INK2)
    fig.text(0.59, 0.08, "Exploited bugs move to the front of every band,\n"
             "so a short patch list catches more of them.", fontsize=9.5, color=INK2)
    fig.add_artist(FancyArrowPatch((0.536, 0.48), (0.572, 0.48), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=22, lw=2, color=INK2))
    fig.text(0.554, 0.455, "model\nreorders", ha="center", va="top", fontsize=8.5, color=INK2)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


def leak_timeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    _title(fig, "Leak #2: the “honest” score was read after the answer was public",
           "Median days after a bug is published · for 59% of exploited bugs the "
           "month-after score came after CISA had listed them")
    ax.axvspan(15, 21, color=ORANGE, alpha=0.08, lw=0)
    ax.text(18, 0.62, "attack already public", ha="center", fontsize=9, color=INK2)
    ax.axhline(0, color=MUTED, lw=1.5, zorder=1)
    events = [(0, "bug published", INK, "above"),
              (1, "next-day EPSS score\n(the fix)", BLUE, "below"),
              (15, "CISA lists it\nas exploited", INK, "above"),
              (16, "month-after EPSS score\n(leak #2)", ORANGE, "below")]
    for d, label, c, side in events:
        ax.plot(d, 0, "o", ms=10, color=c, mec=SURF, mew=2, zorder=3)
        dy = 0.32 if side == "above" else -0.32
        ax.text(d, dy, label, ha="center", va="bottom" if dy > 0 else "top",
                fontsize=9.5, color=c if c != INK else INK, weight="bold" if c != INK else None)
    ax.set_xlim(-1.5, 21); ax.set_ylim(-1, 1)
    ax.set_xticks([0, 1, 5, 10, 15, 16, 20]); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    ax.set_xlabel("days after publication", color=INK2)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.76, bottom=0.18)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


def rolling_chart(out: Path, verify: dict) -> None:
    rows = verify["rolling"]
    names = ["2022 Jul–Dec", "2023 Jan–Jun", "2023 Jul–Dec", "2024 Jan–Jun",
             "2024 Jul–Dec", "2025 Jan–Aug (fresh)"]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    _title(fig, "Better than EPSS in every period, including data never used for any decision",
           "PR-AUC by test period · each model trained only on bugs published before its "
           "period · mean of 3 seeds")
    for i, (r, name) in enumerate(zip(rows, names)):
        y = len(rows) - 1 - i
        e = r["pr_auc_epss"]
        b = sum(r["pr_auc_banded_per_seed"]) / len(r["pr_auc_banded_per_seed"])
        ax.plot([e, b], [y, y], color=MUTED, lw=2, zorder=1)
        ax.plot(e, y, "o", ms=9, color=ORANGE, mec=SURF, mew=1.5, zorder=2)
        ax.plot(b, y, "o", ms=9, color=BLUE, mec=SURF, mew=1.5, zorder=2)
        ax.text(b + 0.008, y, f"+{(b / e - 1):.0%}", va="center", fontsize=9, color=INK2)
        fresh = "fresh" in name
        ax.text(-0.006, y, name, ha="right", va="center", fontsize=9.5, color=INK,
                weight="bold" if fresh else None)
    top = len(rows) - 1
    # Label the widest pair (2023 H2) so the two labels cannot collide.
    r2, y2 = rows[2], top - 2
    ax.annotate("EPSS alone", (r2["pr_auc_epss"], y2), xytext=(0, 11),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.annotate("EPSS band + model", (sum(r2["pr_auc_banded_per_seed"]) / 3, y2),
                xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xlim(0, 0.34); ax.set_ylim(-0.6, top + 0.8); ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    ax.set_xlabel("PR-AUC (higher is better)", color=INK2)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.13)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


def architecture(out: Path) -> None:
    fig = plt.figure(figsize=(11, 5.4), dpi=200, facecolor=SURF)
    _title(fig, "How it fits together",
           "Offline: learn from history without leaking the future · "
           "Live: rank today's bugs with the same rules")

    def box(x, y, w, h, head, body, edge=MUTED):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor=SURF,
                                          edgecolor=edge, lw=1.4))
        fig.text(x + 0.012, y + h - 0.025, head, fontsize=9.5, weight="bold", color=INK, va="top")
        fig.text(x + 0.012, y + h - 0.075, body, fontsize=8.3, color=INK2, va="top",
                 linespacing=1.35)

    def arrow(a, b):
        fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=12, color=INK2, lw=1.2))

    fig.text(0.04, 0.80, "OFFLINE — TRAINING", fontsize=8.5, weight="bold", color=INK2)
    box(0.04, 0.53, 0.17, 0.23, "Sources",
        "NVD records\nEPSS daily archive\nCISA KEV list (labels)")
    box(0.25, 0.53, 0.22, 0.23, "dataset.py",
        "EPSS as of day after publication\nlabel: exploited within 1 year\nschema gate refuses leaky data")
    box(0.51, 0.53, 0.2, 0.23, "train.py",
        "XGBoost × 5 seeds\npublication-time features only\n(no CPE)")
    box(0.75, 0.53, 0.2, 0.23, "models/banded.joblib",
        "model + its encoders\n135,232 CVEs, 621 exploited", edge=BLUE)
    arrow((0.215, 0.645), (0.245, 0.645)); arrow((0.475, 0.645), (0.505, 0.645))
    arrow((0.715, 0.645), (0.745, 0.645))

    fig.text(0.04, 0.44, "LIVE — RANKING", fontsize=8.5, weight="bold", color=INK2)
    box(0.04, 0.17, 0.17, 0.23, "Your scan", "list of CVE IDs")
    box(0.25, 0.17, 0.22, 0.23, "rank.py — fetch",
        "today's EPSS (FIRST API)\nNVD record: description,\nCVSS, CWE")
    box(0.51, 0.17, 0.2, 0.23, "rank.py — score",
        "1. EPSS picks the band\n2. model orders inside it\n3. flags missing data", edge=BLUE)
    box(0.75, 0.17, 0.2, 0.23, "Output",
        "CLI table / CSV\nAPI: POST /rank")
    arrow((0.215, 0.285), (0.245, 0.285)); arrow((0.475, 0.285), (0.505, 0.285))
    arrow((0.715, 0.285), (0.745, 0.285))
    arrow((0.85, 0.525), (0.66, 0.405))
    fig.text(0.04, 0.06, "verify_banded.py re-tests the whole method: leak audit, ablation, "
             "bootstrap, permutation null, six time periods, 2025 holdout.",
             fontsize=8.5, color=INK2)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


def terminal(src: Path, out: Path) -> None:
    lines = src.read_text(encoding="utf-8").rstrip("\n").split("\n")
    BG, FG, PROMPT, DIM = "#1a1a19", "#f0efec", "#86b6ef", "#a9a89f"
    w = max(len(l) for l in lines)
    fig = plt.figure(figsize=(w * 0.088 + 0.7, len(lines) * 0.22 + 0.6), dpi=200, facecolor=BG)
    cont = False
    for i, l in enumerate(lines):
        is_cmd = l.startswith("$") or cont
        cont = is_cmd and l.endswith("\\")
        color = PROMPT if is_cmd else (DIM if l.strip() in ("…",) else FG)
        fig.text(0.3 / fig.get_figwidth(), 1 - (0.3 + i * 0.22) / fig.get_figheight(), l,
                 family="DejaVu Sans Mono", fontsize=9.2, color=color, va="top")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main() -> None:
    verify = json.loads((DOCS.parent / "data/processed/verify_banded.json").read_text())
    method_diagram(DOCS / "method-banded.png")
    leak_timeline(DOCS / "leak2-timeline.png")
    rolling_chart(DOCS / "rolling-windows.png", verify)
    architecture(DOCS / "architecture.png")
    for name in ("verify", "api"):
        terminal(DOCS / "terminal" / f"{name}.txt", DOCS / f"terminal-{name}.png")
    print("wrote", ", ".join(p.name for p in sorted(DOCS.glob("*.png"))))


if __name__ == "__main__":
    main()
