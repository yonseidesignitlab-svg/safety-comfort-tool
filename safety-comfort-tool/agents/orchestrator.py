"""
agents/orchestrator.py — Orchestrator (evaluation pipeline)

Coordinates the multi-agent workflow for the station-radius evaluation:
decomposes the request, assigns tasks to the specialized agents, and
synchronizes their outputs into the final per-station evaluation table.

Stages:
  [1/4] Physical Auditor    → per-station CCTV/Bell/Light + V_phy (weighted density)
  [2/4] Perceptual Observer → street-view sample plan per station
  [3/4] Perceptual Observer → collect + segment + V_per per station
  [4/4] Evaluator           → z-score I_phy / I_per + matrix quadrant → summary CSV

The figures in the paper (study-area map, matrix scatter) are produced from the
summary CSV by the scripts under ./reproduce.
"""
from __future__ import annotations
import os
import time
from contextlib import contextmanager

import config
from core.runlog import RunLogger


class Orchestrator:
    """Workflow coordinator for the multi-agent evaluation system."""

    def __init__(self, mode: str = "station", cfg=config):
        assert mode in ("station",), f"unknown mode: {mode}"
        self.mode = mode
        self.cfg = cfg

    @contextmanager
    def step(self, name: str):
        bar = "-" * 60
        print(f"\n{bar}\n {name}\n{bar}")
        t = time.time()
        try:
            yield
        finally:
            print(f"...done in {time.time()-t:.2f}s")

    # -----------------------------------------------------------------
    # Station-radius evaluation pipeline
    # -----------------------------------------------------------------
    def run_station_pipeline(
        self,
        stations: list[dict],
        *,
        radius_m: int = 300,
        skip_svi: bool = False,
        open_browser: bool = False,
        log: bool = True,
    ) -> dict:
        """
        For each station, evaluate physical safety (facility density) and
        visual comfort (street-view openness) within a circular buffer, then
        combine into a standardized per-station table.

        `stations`: list of dicts with at least {slug, name_ko, lat, lng}.
        """
        import pandas as pd

        from agents.physical_auditor    import Physical_Auditor
        from agents.perceptual_observer import Perceptual_Observer
        from agents.evaluator           import Evaluator

        out_dir = self.cfg.get_station_out_dir()
        artifacts: dict = {"out_dir": out_dir, "stations": [s["slug"] for s in stations]}

        runlog = RunLogger(
            mode="Station",
            out_dir=out_dir,
            title=f"Station-radius evaluation - {len(stations)} stations",
            params={
                "n_stations":  len(stations),
                "stations":    [s["slug"] for s in stations],
                "radius_m":    radius_m,
                "skip_svi":    skip_svi,
                "spacing_m":   self.cfg.SAMPLE_SPACING_M,
                "W_CCTV":      self.cfg.W_CCTV,
                "W_BELL":      self.cfg.W_BELL,
                "W_LIGHT":     self.cfg.W_LIGHT,
                "W_POS":       self.cfg.W_POS,
                "W_NEG":       self.cfg.W_NEG,
            },
        ) if log else None

        pa = Physical_Auditor()
        po = Perceptual_Observer()
        ev = Evaluator()

        # =====================================================
        # [1/4] Physical Auditor across all stations
        # =====================================================
        with self.step(f"[1/4] Physical Auditor - {len(stations)} stations"):
            infra = pa.load_infra_gdfs()    # all of Seoul (stations span multiple districts)
            phy_rows = []
            for s in stations:
                rec = pa.analyze_station(s["slug"], s["lat"], s["lng"], infra, radius_m=radius_m)
                rec["name_ko"] = s.get("name_ko", s["slug"])
                rec["context"] = s.get("context_ko", "")
                phy_rows.append(rec)
                print(f"  {s['slug']:>16s}  CCTV={rec['n_CCTV']:>3d}  Bell={rec['n_Bell']:>3d}  "
                      f"Light={rec['n_Light']:>3d}  D_phy={rec['V_phy']:.1f}/km^2")
            phy_df = pd.DataFrame(phy_rows)
            phy_df.to_csv(os.path.join(out_dir, "physical_audit.csv"), index=False, encoding="utf-8-sig")
            artifacts["physical_audit"] = os.path.join(out_dir, "physical_audit.csv")
            if runlog:
                runlog.add_artifact("physical_audit", artifacts["physical_audit"],
                                    "Per-station CCTV/Bell/Light + S_phy/D_phy")

        # =====================================================
        # [2/4] Perceptual Observer — sample plans
        # =====================================================
        with self.step("[2/4] Perceptual Observer - sample plans"):
            all_plans = []
            for s in stations:
                station_dir = os.path.join(out_dir, s["slug"])
                os.makedirs(station_dir, exist_ok=True)
                plan = po.plan_station_samples(
                    s["slug"], s["lat"], s["lng"],
                    radius_m=radius_m, bidirectional=self.cfg.BIDIRECTIONAL, verbose=True,
                )
                if not plan.empty:
                    plan.to_csv(os.path.join(station_dir, "svi_plan.csv"),
                                index=False, encoding="utf-8-sig")
                all_plans.append(plan)
            full_plan = pd.concat(all_plans, ignore_index=True) if all_plans else pd.DataFrame()
            full_plan.to_csv(os.path.join(out_dir, "svi_plan_all.csv"),
                             index=False, encoding="utf-8-sig")
            artifacts["svi_plan"] = os.path.join(out_dir, "svi_plan_all.csv")
            print(f"   Total planned images: {len(full_plan)}")
            if runlog:
                runlog.add_artifact("svi_plan_all", artifacts["svi_plan"],
                                    f"Combined SVI plan ({len(full_plan)} images)")

        # =====================================================
        # [3/4] Perceptual Observer — collect + segment + V_per per station
        # =====================================================
        if skip_svi:
            print("\n[3/4] Perceptual Observer - SKIPPED (--skip-svi)")
            artifacts["per_per_station"] = None
        else:
            per_rows = []
            with self.step("[3/4] Perceptual Observer - SV collect + segment + V_per"):
                for s in stations:
                    station_dir = os.path.join(out_dir, s["slug"])
                    img_dir = os.path.join(station_dir, "svi_images")
                    plan_path = os.path.join(station_dir, "svi_plan.csv")
                    if not os.path.exists(plan_path):
                        continue
                    plan = pd.read_csv(plan_path)
                    if plan.empty:
                        continue
                    sample_pts = plan.rename(columns={"lon": "lng"}).to_dict("records")
                    print(f"\n  -> {s['slug']}: {len(sample_pts)} images planned")
                    stats = po.collect_images(sample_pts, img_dir, check_metadata=True, delay_s=0.0)
                    print(f"    collect ok={stats['ok']} skip={stats['skipped']} fail={stats['failed']}")

                    overlay_dir = os.path.join(station_dir, "svi_overlays")
                    pixel_df = po.segment_directory(img_dir, save_overlays=True, overlay_dir=overlay_dir)
                    if pixel_df.empty:
                        continue
                    pixel_df = po.compute_V_per_per_image(pixel_df)
                    pixel_df.to_csv(os.path.join(station_dir, "pixel_per_image.csv"),
                                    index=False, encoding="utf-8-sig")
                    agg = po.aggregate_to_station(pixel_df)
                    per_rows.append({"station_slug": s["slug"], **agg})

                if per_rows:
                    per_df = pd.DataFrame(per_rows)
                    per_df.to_csv(os.path.join(out_dir, "per_per_station.csv"),
                                  index=False, encoding="utf-8-sig")
                    artifacts["per_per_station"] = os.path.join(out_dir, "per_per_station.csv")
                    if runlog:
                        runlog.add_artifact("per_per_station", artifacts["per_per_station"],
                                            "Per-station V_per")
                else:
                    artifacts["per_per_station"] = None

        # =====================================================
        # [4/4] Evaluator — z-score + matrix quadrant
        # =====================================================
        with self.step("[4/4] Evaluator - standardize indices + matrix quadrant"):
            phy_df = ev.compute_I_phy(phy_df, value_col="V_phy")

            if artifacts.get("per_per_station"):
                per_df = pd.read_csv(artifacts["per_per_station"])
                combined = ev.combine(
                    phy_df, per_df, key="station_slug",
                    phy_value_col="V_phy", per_value_col="V_per",
                )
            else:
                combined = phy_df.copy()
                combined["V_per"] = 0.0
                combined["I_per"] = 0.0
                combined["quadrant"] = "NA"
                combined["quadrant_label"] = "NA (per missing)"

            preferred = ["station_slug", "name_ko", "context",
                         "n_CCTV", "n_Bell", "n_Light", "S_phy",
                         "V_phy", "I_phy", "V_per", "I_per",
                         "quadrant", "quadrant_label"]
            cols = [c for c in preferred if c in combined.columns] + \
                   [c for c in combined.columns if c not in preferred]
            combined = combined[cols]

            summary_csv = os.path.join(out_dir, "stations_summary.csv")
            combined.to_csv(summary_csv, index=False, encoding="utf-8-sig")
            artifacts["summary"] = summary_csv
            if runlog:
                runlog.add_artifact("summary_csv", summary_csv, "Final per-station table")
                runlog.add_summary("scdm_stats", Evaluator.summary_stats(combined))

        if runlog:
            runlog.status = "ok"
            runlog.write()
            artifacts["run_log_json"] = runlog.json_path
            artifacts["run_log_md"]   = runlog.md_path

        print("\n" + "=" * 60)
        print(f" Pipeline complete - {len(stations)} stations")
        print("=" * 60)
        for k, v in artifacts.items():
            if isinstance(v, (str, int, float)) or v is None:
                print(f"  {k:18s}: {v}")
        return artifacts
