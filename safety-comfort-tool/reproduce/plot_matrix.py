# -*- coding: utf-8 -*-
"""
Plot the safety-comfort matrix (paper Fig. 8) from the pipeline output.

Reads:  output/station/stations_summary.csv   (station_slug, I_phy, I_per, quadrant)
        data/line2_stations.json              (for English station names + centrality tier)
Writes: reproduce/figures/matrix.png
"""
import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

# centrality tier -> colour (City / Regional / District Centre)
TIER_COLOR = {"City Centre": "#DC2626", "Regional Centre": "#F59E0B", "District Centre": "#1D4ED8"}
# map context_ko (in line2_stations.json) -> tier; edit if your station list differs
TIER_BY_SLUG = {
    "city_hall": "City Centre", "euljiro_ipgu": "City Centre",
    "gangnam": "City Centre", "samseong": "City Centre",
    "sinchon": "Regional Centre", "sadang": "Regional Centre",
    "hongdae_ipgu": "District Centre", "konkuk_ipgu": "District Centre",
    "sindorim": "District Centre", "mullae": "District Centre",
}


def main():
    summary = ROOT / "output" / "station" / "stations_summary.csv"
    if not summary.exists():
        raise SystemExit(f"Run the pipeline first; not found: {summary}")
    df = pd.read_csv(summary)

    meta = json.loads((ROOT / "data" / "line2_stations.json").read_text(encoding="utf-8"))["stations"]
    name_en = {s["slug"]: s.get("name_en", s["slug"]) for s in meta}

    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=300)
    lim = max(2.0, df[["I_phy", "I_per"]].abs().max().max() + 0.4)
    for (x0, y0, c) in [(0, 0, "#86EFAC"), (-lim, 0, "#FDE68A"),
                        (-lim, -lim, "#FCA5A5"), (0, -lim, "#93C5FD")]:
        ax.add_patch(Rectangle((x0, y0), lim, lim, facecolor=c, alpha=0.16))
    ax.axhline(0, color="black", lw=1.1)
    ax.axvline(0, color="black", lw=1.1)

    for _, r in df.iterrows():
        slug = r["station_slug"]
        col = TIER_COLOR.get(TIER_BY_SLUG.get(slug, ""), "#666666")
        ax.scatter(r["I_phy"], r["I_per"], s=210, color=col, edgecolor="white",
                   linewidth=1.8, zorder=5)
        ax.annotate(name_en.get(slug, slug), (r["I_phy"], r["I_per"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=6)

    for x, y, t, ha, va in [(lim*0.95, lim*0.95, "Q1  Stable", "right", "top"),
                            (-lim*0.95, lim*0.95, "Q2  Facility-Deficient", "left", "top"),
                            (-lim*0.95, -lim*0.95, "Q3  Compound-Vulnerable", "left", "bottom"),
                            (lim*0.95, -lim*0.95, "Q4  Enclosure-Dominant", "right", "bottom")]:
        ax.text(x, y, t, ha=ha, va=va, fontsize=10.5, fontweight="bold", color="#333", alpha=0.8)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Physical Safety Index", fontsize=12, fontweight="bold", color="#1E2761")
    ax.set_ylabel("Visual Comfort Index", fontsize=12, fontweight="bold", color="#1E2761")
    ax.grid(True, linestyle=":", alpha=0.3)
    handles = [Line2D([], [], color=c, marker="o", linestyle="None", markersize=10,
                      markeredgecolor="white", label=l) for l, c in TIER_COLOR.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.95,
              title="Centrality (2040 Seoul Plan)", title_fontsize=10)

    out = ROOT / "reproduce" / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "matrix.png", dpi=300, bbox_inches="tight")
    print("saved", out / "matrix.png")


if __name__ == "__main__":
    main()
