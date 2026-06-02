# -*- coding: utf-8 -*-
"""
Study-area map (paper Fig. 6): ten Line-2 stations classified by their centrality
tier in the 2040 Seoul Plan, over the Seoul administrative-boundary basemap.

Requires data/seoul_dong.geojson (see data/README.md) and geopandas.
Writes: reproduce/figures/studyarea_map.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = ["Times New Roman", "Malgun Gothic", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

RED, ORANGE, BLUE = "#DC2626", "#F59E0B", "#1D4ED8"
TIER = {  # slug -> colour
    "city_hall": RED, "euljiro_ipgu": RED, "gangnam": RED, "samseong": RED,
    "sinchon": ORANGE, "sadang": ORANGE,
    "hongdae_ipgu": BLUE, "konkuk_ipgu": BLUE, "sindorim": BLUE, "mullae": BLUE,
}
RING = ["city_hall", "euljiro_ipgu", "konkuk_ipgu", "samseong", "gangnam",
        "sadang", "sindorim", "mullae", "hongdae_ipgu", "sinchon", "city_hall"]


def main():
    stations = json.loads((ROOT / "data" / "line2_stations.json").read_text(encoding="utf-8"))["stations"]
    st = {s["slug"]: s for s in stations}
    geojson = ROOT / "data" / "seoul_dong.geojson"
    if not geojson.exists():
        raise SystemExit(f"Provide the boundary GeoJSON (see data/README.md): {geojson}")

    seoul = gpd.read_file(geojson)
    seoul = seoul[seoul["adm_nm"].str.startswith("서울", na=False)]
    gu = seoul.dissolve(by="sggnm").reset_index()
    outer = seoul.dissolve()

    fig, ax = plt.subplots(figsize=(11, 9))
    gu.plot(ax=ax, color="#F5F5F5", edgecolor="#CCCCCC", linewidth=0.6)
    outer.plot(ax=ax, color="none", edgecolor="#777", linewidth=1.4)

    for s in stations:
        ax.add_patch(Circle((s["lng"], s["lat"]), 0.0027, facecolor=TIER[s["slug"]],
                            alpha=0.20, edgecolor=TIER[s["slug"]], linewidth=1.5, linestyle="--"))
    ax.plot([st[k]["lng"] for k in RING], [st[k]["lat"] for k in RING],
            color="#10B981", linewidth=2.5, alpha=0.5, label="Subway Line 2 (approx.)")
    for s in stations:
        ax.scatter(s["lng"], s["lat"], s=140, color=TIER[s["slug"]],
                   edgecolor="white", linewidth=2, zorder=5)
        ax.annotate(f"{s['name_ko']}\n({s.get('name_en','')})", (s["lng"], s["lat"]),
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=TIER[s["slug"]], linewidth=0.8, alpha=0.92))

    handles = [Line2D([], [], color="#10B981", lw=2.5, alpha=0.5, label="Subway Line 2 (approx.)")]
    for l, c in [("City Centre", RED), ("Regional Centre", ORANGE), ("District Centre", BLUE)]:
        handles.append(Line2D([], [], color=c, marker="o", linestyle="None", markersize=10,
                              markeredgecolor="white", label=l))
    ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.95,
              title="Centrality (2040 Seoul Plan)", title_fontsize=10)

    lngs = [s["lng"] for s in stations]; lats = [s["lat"] for s in stations]
    ax.set_xlim(min(lngs)-0.025, max(lngs)+0.025)
    ax.set_ylim(min(lats)-0.025, max(lats)+0.025)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.35)

    out = ROOT / "reproduce" / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "studyarea_map.png", dpi=300, bbox_inches="tight")
    print("saved", out / "studyarea_map.png")


if __name__ == "__main__":
    main()
