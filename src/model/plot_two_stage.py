"""Render docs/banded-vs-epss.png from data/processed/two_stage.json.

Run after `python -m src.model.two_stage`; needs matplotlib.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("data/processed/two_stage.json"))
B = [x["budget"] for x in r["results"]]
epss = [x["epss_hits"] for x in r["results"]]
gate = [x["gated_hits_seed_mean"] for x in r["results"]]
band = [x["banded_hits_seed_mean"] for x in r["banded"]["results"]]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)
for y, c, ls, lab, dy in ((band, "#2a78d6", "-", "EPSS band + model (new)", 7),
                          (epss, "#eb6834", "-", "EPSS alone", 0),
                          (gate, "#1baf7a", (0, (4, 3)), "Old gate", -7)):
    ax.plot(B, y, color=c, lw=2, ls=ls, marker="o", ms=5, mec=SURF, mew=1.5, zorder=3)
    ax.annotate(lab, (B[-1], y[-1]), xytext=(8, dy), textcoords="offset points",
                va="center", color=INK, fontsize=10)
for b, e, n in zip(B, epss, band):
    if b in (100, 5000):
        right = b == 100
        ax.annotate(f"{n:.0f} vs {e}", (b, n), xytext=(0, 9) if right else (-8, 6), textcoords="offset points",
                    ha="center" if right else "right", va="bottom", color=INK2, fontsize=9)
ax.set_xscale("log"); ax.set_xticks(B); ax.set_xticklabels([f"{b:,}" for b in B])
ax.minorticks_off(); ax.set_ylim(0, 185); ax.set_xlim(80, 12500)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
ax.set_xlabel("Bugs patched (log scale)", color=INK2)
ax.set_ylabel("Exploited bugs caught (of 209)", color=INK2)
fig.text(0.07, 0.95, "Ranking inside EPSS bands catches more at every budget",
         color=INK, fontsize=13, weight="bold")
fig.text(0.07, 0.905, "CVEs published Jul 2023 – Dec 2024 · mean of 5 training seeds",
         color=INK2, fontsize=9.5)
fig.subplots_adjust(left=0.09, right=0.76, top=0.86, bottom=0.12)
fig.savefig("docs/banded-vs-epss.png", facecolor=SURF)
