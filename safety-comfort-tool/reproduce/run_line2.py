"""
modes/mode_c_station_pipeline.py — Mode C entry point

Station-radius Agentic CPTED pipeline. For each major Line 2 station (defined
in `data/line2_stations.json`), analyze the area within a circular buffer
(default 500 m) using the same 4-agent stack as Modes A and B, then combine
into a single SCDM scatter across all stations.

Output: `output/stations/{stations_summary.csv, stations_scdm.png, ...}`

Flags:
    --skip-svi          Skip Street View collect + segment (no Google SV cost)
    --no-browser        Do not auto-open the result
    --radius <m>        Override radius (default 500m)
    --stations <path>   Use a different station list JSON
    --only <slugs>      Comma-separated subset, e.g. --only gangnam,sinchon
"""
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import config
from agents.orchestrator import Orchestrator


def main():
    p = argparse.ArgumentParser(description="Line 2 station-radius safety & comfort evaluation")
    p.add_argument("--skip-svi",   action="store_true", help="Skip SV collect+segment")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the result")
    p.add_argument("--radius",     type=int, default=300, help="Buffer radius in meters (default 300)")
    p.add_argument("--stations",   default=os.path.join(ROOT, "data", "line2_stations.json"),
                   help="Path to station list JSON")
    p.add_argument("--only",       default=None,
                   help="Comma-separated slug subset, e.g. 'gangnam,sinchon'")
    args = p.parse_args()

    # Load station list
    with open(args.stations, "r", encoding="utf-8") as f:
        data = json.load(f)
    stations = data["stations"]

    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        stations = [s for s in stations if s["slug"] in wanted]
        if not stations:
            print(f"[ERROR] No station matched --only={args.only}")
            sys.exit(1)

    # Banner
    print("═" * 60)
    print(" Street-environment safety & comfort — station-radius pipeline")
    print("═" * 60)
    print(f"  Station list:  {args.stations}")
    print(f"  Stations (n):  {len(stations)}")
    print(f"  Radius:        {args.radius} m")
    print(f"  Skip SVI:      {args.skip_svi}")
    print("─" * 60)
    for s in stations:
        print(f"  • {s['slug']:>16s}  ({s['name_ko']:<10s}) "
              f"context={s.get('context_ko', '')}  "
              f"@ ({s['lat']:.4f}, {s['lng']:.4f})")
    print("═" * 60)

    orch = Orchestrator(mode="station")
    artifacts = orch.run_station_pipeline(
        stations,
        radius_m=args.radius,
        skip_svi=args.skip_svi,
        open_browser=not args.no_browser,
    )

    if not artifacts.get("summary"):
        sys.exit(1)


if __name__ == "__main__":
    main()
