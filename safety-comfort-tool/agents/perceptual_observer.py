"""
agents/perceptual_observer.py — Perceptual Observer

Thesis Section 3.3 + 4.3:
  "The Perceptual Observer collects and preprocesses visual environment data
   (Street View imagery + semantic segmentation), producing the Perceptual
   Comfort Value V_per per image and aggregating to route/cell granularity."

Lifecycle:
  plan_*_samples()       → produce (lat, lng, heading) collection plan
  collect_images()       → fetch SV images from Google API
  load_model()           → load DeepLabV3+ ResNet101 (Cityscapes pretrained)
  segment_image/dir()    → pixel-class counts per image
  compute_V_per_per_image()  → EQ1 + EQ2 (W_per, V_per) per image
  aggregate_to_route()   → mean V_per per route (Mode A)
  aggregate_to_cells()   → mean V_per per cell (Mode B)

Sources merged:
  PREVIOUS/route_planning_agent.py  — plan_route_samples + collect_images
  PREVIOUS/visual_perception_agent.py — load_model + segment_image + V_per math
  UPDATE/svi_sampling_plan.py        — plan_dong_samples (osmnx-based)
  UPDATE/_archive/svi_collector.py   — batch collect (resumable)
  UPDATE/_archive/svi_segment.py     — batch segmentation
  UPDATE/_archive/svi_per_index.py   — cell aggregation
"""
from __future__ import annotations
import os
import sys
import csv
import glob
import time
import json
from datetime import datetime
from typing import Optional, Callable, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from PIL import Image

import config
from core.geo import resample_polyline, calculate_bearing
from core.streetview import streetview_metadata_ok, fetch_streetview_image, save_image_bytes


class Perceptual_Observer:
    """Street View collection + DeepLabV3+ segmentation + V_per computation."""

    def __init__(self, cfg=config):
        self.cfg = cfg
        self._model = None
        self._device = None

    # =================================================================
    # Model lifecycle (lazy)
    # =================================================================
    def load_model(self):
        """Load DeepLabV3+ ResNet101, Cityscapes pretrained, eval mode (cached)."""
        if self._model is not None:
            return self._model, self._device

        import torch
        if not os.path.isdir(self.cfg.REPO_DIR):
            raise FileNotFoundError(f"DeepLabV3Plus repo not found at {self.cfg.REPO_DIR}")
        if not os.path.exists(self.cfg.WEIGHT_PATH):
            raise FileNotFoundError(f"Weight file not found at {self.cfg.WEIGHT_PATH}")

        if self.cfg.REPO_DIR not in sys.path:
            sys.path.insert(0, self.cfg.REPO_DIR)
        from network import modeling   # type: ignore  (provided by external repo)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = modeling.deeplabv3plus_resnet101(num_classes=19, output_stride=8)
        ckpt = torch.load(self.cfg.WEIGHT_PATH, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        # Strip 'module.' prefix from DataParallel checkpoints
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval().to(device)
        self._model, self._device = model, device
        return model, device

    # =================================================================
    # Mode A: route-level sample plan
    # =================================================================
    def plan_route_samples(
        self, raw_coords: list[tuple[float, float]], spacing_m: int
    ) -> list[dict]:
        """
        Resample a user-drawn polyline at `spacing_m` meters and assign a
        heading to each point (direction toward the next point).
        Returns [{lat, lng, heading}, ...].
        """
        if not raw_coords or len(raw_coords) < 2:
            return []
        resampled = resample_polyline(raw_coords, spacing_m)
        out, last_h = [], 0.0
        for i, p in enumerate(resampled):
            if i < len(resampled) - 1:
                h = calculate_bearing(p[0], p[1], resampled[i + 1][0], resampled[i + 1][1])
                last_h = h
            else:
                h = last_h
            out.append({"lat": p[0], "lng": p[1], "heading": float(h)})
        return out

    def process_new_route(
        self, raw_coords: list[tuple[float, float]], name: str, spacing_m: int
    ) -> dict:
        """Route record used by Mode A's session_state.all_routes."""
        return {
            "id":         datetime.now().strftime("%f"),
            "name":       name,
            "safe_name":  name.replace(" ", ""),
            "raw_coords": raw_coords,
            "final_points": self.plan_route_samples(raw_coords, spacing_m),
        }

    # =================================================================
    # Mode B: dong-level sample plan (replaces UPDATE/svi_sampling_plan.py)
    # =================================================================
    def plan_dong_samples(
        self,
        target_dong: str,
        dong_short: str,
        *,
        grid_size_m: Optional[int] = None,
        spacing_m: Optional[int] = None,
        bidirectional: Optional[bool] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate per-cell SVI sampling plan for a dong.
        Writes `output/{dong_short}/svi_plan.csv` and returns the same DataFrame.
        """
        import osmnx as ox

        grid_size_m  = grid_size_m  or self.cfg.GRID_SIZE_M
        spacing_m    = spacing_m    or self.cfg.SAMPLE_SPACING_M
        bidirectional = self.cfg.BIDIRECTIONAL if bidirectional is None else bidirectional

        if verbose: print(f"[PO] plan_dong_samples({target_dong}, grid={grid_size_m}m, spacing={spacing_m}m)")
        dongs = gpd.read_file(self.cfg.DATA_GEOJSON)
        dong = dongs[dongs["adm_nm"] == target_dong]
        if dong.empty:
            raise ValueError(f"Dong '{target_dong}' not found in geojson")
        dong_wgs = dong.to_crs(self.cfg.CRS_WGS)
        dong_tm  = dong.to_crs(self.cfg.CRS_KTM)
        dong_geom_tm  = dong_tm.geometry.iloc[0]
        dong_geom_wgs = dong_wgs.geometry.iloc[0]

        # Grid + intersection
        minx, miny, maxx, maxy = dong_geom_tm.bounds
        cells, cid = [], 0
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cb = box(x, y, x + grid_size_m, y + grid_size_m)
                if cb.intersects(dong_geom_tm):
                    clipped = cb.intersection(dong_geom_tm)
                    if not clipped.is_empty and clipped.area > 0:
                        cells.append({"cell_id": cid, "geometry": clipped})
                        cid += 1
                y += grid_size_m
            x += grid_size_m
        grid_tm = gpd.GeoDataFrame(cells, crs=self.cfg.CRS_KTM)

        # OSM walk network → cell filter
        G = ox.graph_from_polygon(dong_geom_wgs, network_type="walk")
        edges_wgs = ox.graph_to_gdfs(G, nodes=False).reset_index(drop=True)
        edges_tm  = edges_wgs.to_crs(self.cfg.CRS_KTM)
        with_road = gpd.sjoin(grid_tm[["cell_id", "geometry"]], edges_tm[["geometry"]],
                              predicate="intersects", how="inner")
        kept = set(with_road["cell_id"].unique())
        grid_tm["has_road"] = grid_tm["cell_id"].isin(kept)
        analysis = grid_tm[grid_tm["has_road"]].reset_index(drop=True)
        if verbose: print(f"[PO]    analysis cells: {len(analysis)}")

        # Per-edge resampling + heading
        records, point_id = [], 0
        for _, row in edges_tm.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            segs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for line in segs:
                if line.length < spacing_m:
                    continue
                n = max(2, int(line.length // spacing_m) + 1)
                ds = np.linspace(0, line.length, n)
                pts = [line.interpolate(d) for d in ds]
                for i, pt in enumerate(pts):
                    if i < len(pts) - 1:
                        nxt = pts[i + 1]
                    else:
                        nxt = pt; pt = pts[i - 1]
                    pt_wgs  = gpd.GeoSeries([pt],  crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]
                    nxt_wgs = gpd.GeoSeries([nxt], crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]
                    h_fwd = calculate_bearing(pt_wgs.y, pt_wgs.x, nxt_wgs.y, nxt_wgs.x)
                    if i == len(pts) - 1:
                        h_fwd = (h_fwd + 180) % 360
                        pt = pts[i]
                        pt_wgs = gpd.GeoSeries([pt], crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]
                    records.append({
                        "point_id":    point_id,
                        "lat":         pt_wgs.y,
                        "lon":         pt_wgs.x,
                        "heading_fwd": round(h_fwd, 2),
                        "heading_rev": round((h_fwd + 180) % 360, 2),
                        "geometry_tm": pt,
                    })
                    point_id += 1

        sample_gdf = gpd.GeoDataFrame(records, geometry="geometry_tm", crs=self.cfg.CRS_KTM)

        # Coordinate-based dedup (1m precision)
        sample_gdf["_lat_r"] = sample_gdf["lat"].round(5)
        sample_gdf["_lon_r"] = sample_gdf["lon"].round(5)
        sample_gdf = (sample_gdf.drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
                      .reset_index(drop=True)
                      .drop(columns=["_lat_r", "_lon_r"]))
        sample_gdf["point_id"] = range(len(sample_gdf))

        # Attach cell_id
        sample_gdf = gpd.sjoin(sample_gdf, analysis[["cell_id", "geometry"]],
                               predicate="within", how="left").drop(columns=["index_right"])
        sample_gdf = sample_gdf[sample_gdf["cell_id"].notna()].copy()
        sample_gdf["cell_id"] = sample_gdf["cell_id"].astype(int)

        # Materialize plan rows (one per image)
        if bidirectional:
            fwd = sample_gdf[["point_id", "cell_id", "lat", "lon"]].copy()
            fwd["heading"] = sample_gdf["heading_fwd"]; fwd["direction"] = "fwd"
            rev = sample_gdf[["point_id", "cell_id", "lat", "lon"]].copy()
            rev["heading"] = sample_gdf["heading_rev"]; rev["direction"] = "rev"
            plan = pd.concat([fwd, rev], ignore_index=True)
        else:
            plan = sample_gdf[["point_id", "cell_id", "lat", "lon", "heading_fwd"]].copy()
            plan = plan.rename(columns={"heading_fwd": "heading"})
            plan["direction"] = "fwd"

        plan["image_id"] = plan.apply(
            lambda r: f"c{int(r['cell_id']):04d}_p{int(r['point_id']):05d}_{r['direction']}", axis=1
        )
        plan = (plan[["image_id", "cell_id", "point_id", "direction", "lat", "lon", "heading"]]
                .sort_values(["cell_id", "point_id", "direction"]).reset_index(drop=True))

        out_dir = self.cfg.get_dong_out_dir(dong_short)
        csv_path = os.path.join(out_dir, "svi_plan.csv")
        plan.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if verbose: print(f"[PO]    plan rows: {len(plan)} → {csv_path}")
        return plan

    # =================================================================
    # Mode C: station-radius sample plan
    # =================================================================
    def plan_station_samples(
        self,
        station_name: str,
        lat: float,
        lng: float,
        *,
        radius_m: int = 500,
        spacing_m: Optional[int] = None,
        bidirectional: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Mode C sample plan — circular buffer around a single station coordinate.

        Args:
            station_name: slug used in image_id (e.g. 'gangnam')
            lat, lng: station center (WGS84)
            radius_m: circular buffer radius in meters
            spacing_m: SV sampling interval along roads (default config.SAMPLE_SPACING_M)
            bidirectional: fwd-only by default (consistent with route-level analysis)

        Returns DataFrame with columns:
            image_id, station_slug, point_id, direction, lat, lon, heading
        """
        import osmnx as ox
        from shapely.geometry import Point

        spacing_m = spacing_m or self.cfg.SAMPLE_SPACING_M
        if verbose: print(f"[PO] plan_station_samples({station_name}, r={radius_m}m, spacing={spacing_m}m)")

        # Center point in both CRS
        center_wgs = Point(lng, lat)
        center_tm  = gpd.GeoSeries([center_wgs], crs=self.cfg.CRS_WGS).to_crs(self.cfg.CRS_KTM).iloc[0]
        buffer_tm  = center_tm.buffer(radius_m)
        buffer_wgs = gpd.GeoSeries([buffer_tm], crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]

        # OSMnx walk network within the buffer
        G = ox.graph_from_polygon(buffer_wgs, network_type="walk")
        edges_wgs = ox.graph_to_gdfs(G, nodes=False).reset_index(drop=True)
        edges_tm  = edges_wgs.to_crs(self.cfg.CRS_KTM)

        # Clip edges to circular buffer
        edges_tm = edges_tm[edges_tm.geometry.intersects(buffer_tm)].copy()
        edges_tm["geometry"] = edges_tm.geometry.intersection(buffer_tm)
        edges_tm = edges_tm[~edges_tm.geometry.is_empty]
        if verbose: print(f"[PO]    edges in buffer: {len(edges_tm)}")

        # Per-edge resampling + heading (same logic as plan_dong_samples)
        records, point_id = [], 0
        for _, row in edges_tm.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            segs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for line in segs:
                if line.length < spacing_m:
                    continue
                n = max(2, int(line.length // spacing_m) + 1)
                ds = np.linspace(0, line.length, n)
                pts = [line.interpolate(d) for d in ds]
                for i, pt in enumerate(pts):
                    if i < len(pts) - 1:
                        nxt = pts[i + 1]
                    else:
                        nxt = pt; pt = pts[i - 1]
                    pt_wgs  = gpd.GeoSeries([pt],  crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]
                    nxt_wgs = gpd.GeoSeries([nxt], crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]
                    h_fwd = calculate_bearing(pt_wgs.y, pt_wgs.x, nxt_wgs.y, nxt_wgs.x)
                    if i == len(pts) - 1:
                        h_fwd = (h_fwd + 180) % 360
                        pt = pts[i]
                        pt_wgs = gpd.GeoSeries([pt], crs=self.cfg.CRS_KTM).to_crs(self.cfg.CRS_WGS).iloc[0]
                    records.append({
                        "point_id":    point_id,
                        "lat":         pt_wgs.y,
                        "lon":         pt_wgs.x,
                        "heading_fwd": round(h_fwd, 2),
                        "heading_rev": round((h_fwd + 180) % 360, 2),
                        "geometry_tm": pt,
                    })
                    point_id += 1

        if not records:
            if verbose: print(f"[PO]    no sample points generated")
            return pd.DataFrame(columns=["image_id","station_slug","point_id","direction","lat","lon","heading"])

        sample_gdf = gpd.GeoDataFrame(records, geometry="geometry_tm", crs=self.cfg.CRS_KTM)

        # Final clip — only points strictly inside the buffer
        sample_gdf = sample_gdf[sample_gdf.geometry.within(buffer_tm)].copy()

        # Coordinate-based dedup (1m precision)
        sample_gdf["_lat_r"] = sample_gdf["lat"].round(5)
        sample_gdf["_lon_r"] = sample_gdf["lon"].round(5)
        sample_gdf = (sample_gdf.drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
                      .reset_index(drop=True)
                      .drop(columns=["_lat_r", "_lon_r"]))
        sample_gdf["point_id"] = range(len(sample_gdf))

        # Materialize plan rows
        if bidirectional:
            fwd = sample_gdf[["point_id", "lat", "lon"]].copy()
            fwd["heading"] = sample_gdf["heading_fwd"]; fwd["direction"] = "fwd"
            rev = sample_gdf[["point_id", "lat", "lon"]].copy()
            rev["heading"] = sample_gdf["heading_rev"]; rev["direction"] = "rev"
            plan = pd.concat([fwd, rev], ignore_index=True)
        else:
            plan = sample_gdf[["point_id", "lat", "lon", "heading_fwd"]].copy()
            plan = plan.rename(columns={"heading_fwd": "heading"})
            plan["direction"] = "fwd"

        plan["station_slug"] = station_name
        plan["image_id"] = plan.apply(
            lambda r: f"{station_name}_p{int(r['point_id']):04d}_{r['direction']}", axis=1
        )
        plan = (plan[["image_id", "station_slug", "point_id", "direction", "lat", "lon", "heading"]]
                .sort_values(["point_id", "direction"]).reset_index(drop=True))

        if verbose: print(f"[PO]    plan rows: {len(plan)}")
        return plan

    # =================================================================
    # SV image collection (works for both modes)
    # =================================================================
    def collect_images(
        self,
        sample_points: list[dict],     # [{lat, lng or lon, heading, [image_id]}, ...]
        output_dir: str,
        *,
        api_key: Optional[str] = None,
        size_w: int = 640, size_h: int = 480,
        fov: int = 90, pitch: int = 0,
        use_outdoor: bool = True,
        check_metadata: bool = True,
        delay_s: float = 0.05,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> dict:
        """
        Fetch Google Street View images for each sample point.
        Resumes if files already exist (no re-download).

        progress_cb(fraction, status_message) — optional UI hook for Streamlit.
        Returns: {ok: int, skipped: int, failed: int, dir: str}
        """
        api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY not set")
        os.makedirs(output_dir, exist_ok=True)

        n = max(1, len(sample_points))
        ok = skipped = failed = 0
        for i, pt in enumerate(sample_points):
            lat = pt.get("lat")
            lng = pt.get("lng") if "lng" in pt else pt.get("lon")
            head = float(pt.get("heading", 0))
            img_id = pt.get("image_id") or f"{i+1:03d}_{lat:.6f}_{lng:.6f}_{int(head)}deg"
            dest = os.path.join(output_dir, f"{img_id}.jpg")

            if os.path.exists(dest):
                skipped += 1
            else:
                if check_metadata and not streetview_metadata_ok(lat, lng, api_key, use_outdoor=use_outdoor):
                    failed += 1
                else:
                    content = fetch_streetview_image(
                        lat, lng, head, api_key,
                        size_w=size_w, size_h=size_h, fov=fov, pitch=pitch,
                        use_outdoor=use_outdoor,
                    )
                    if content is None:
                        failed += 1
                    else:
                        save_image_bytes(content, dest)
                        ok += 1
            if progress_cb:
                progress_cb((i + 1) / n, f"{i+1}/{n}  ok={ok} skip={skipped} fail={failed}")
            if delay_s:
                time.sleep(delay_s)

        return {"ok": ok, "skipped": skipped, "failed": failed, "dir": output_dir}

    # =================================================================
    # Segmentation — per-image
    # =================================================================
    def segment_image(self, img_path: str, save_overlay_to: Optional[str] = None) -> Optional[dict]:
        """
        Run DeepLabV3+ on one image, return Cityscapes pixel counts.
        Optionally writes a colour-coded blended overlay PNG.

        Returns dict with keys:
            Filename, Total_Pixels,
            Sidewalk, Vegetation, Terrain, Sky,         # Positive
            Building, Wall, Fence, Pole,                # Negative
            Sum_Pos, Sum_Neg
        """
        try:
            import torch
            import torch.nn.functional as F
            from torchvision import transforms

            model, device = self.load_model()
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            total = W * H

            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = logits.max(1)[1].squeeze().cpu().numpy()
            counts = np.bincount(pred.flatten(), minlength=19)

            rec = {
                "Filename":    os.path.basename(img_path),
                "Total_Pixels": int(total),
                # Positive
                "Sidewalk":   int(counts[1]),
                "Vegetation": int(counts[8]),
                "Terrain":    int(counts[9]),
                "Sky":        int(counts[10]),
                # Negative
                "Building":   int(counts[2]),
                "Wall":       int(counts[3]),
                "Fence":      int(counts[4]),
                "Pole":       int(counts[5]),
            }
            rec["Sum_Pos"] = sum(rec[c] for c in ("Sidewalk", "Vegetation", "Terrain", "Sky"))
            rec["Sum_Neg"] = sum(rec[c] for c in ("Building", "Wall", "Fence", "Pole"))

            if save_overlay_to:
                colorized = self.cfg.CITYSCAPES_COLORS[pred].astype(np.uint8)
                blended = Image.blend(img, Image.fromarray(colorized), alpha=0.5)
                os.makedirs(os.path.dirname(save_overlay_to), exist_ok=True)
                blended.save(save_overlay_to)
            return rec
        except Exception as e:
            print(f"[PO][segment_image] {img_path}: {e}")
            return None

    # =================================================================
    # Segmentation — directory (Mode B batch)
    # =================================================================
    def segment_directory(
        self,
        image_dir: str,
        *,
        save_overlays: bool = False,
        overlay_dir: Optional[str] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        glob_pattern: str = "*.jpg",
    ) -> pd.DataFrame:
        """
        Batch-segment every image in `image_dir`. Returns a per-image DataFrame.
        """
        files = sorted(glob.glob(os.path.join(image_dir, glob_pattern)))
        rows = []
        n = max(1, len(files))
        for i, f in enumerate(files):
            overlay_path = (os.path.join(overlay_dir or image_dir, f"seg_{os.path.basename(f)}")
                            if save_overlays else None)
            rec = self.segment_image(f, save_overlay_to=overlay_path)
            if rec:
                rows.append(rec)
            if progress_cb:
                progress_cb((i + 1) / n, f"{i+1}/{n}  {os.path.basename(f)}")
        return pd.DataFrame(rows)

    # =================================================================
    # V_per computation (EQ1 + EQ2)
    # =================================================================
    def compute_V_per_per_image(
        self, pixel_df: pd.DataFrame,
        *,
        w_pos: Optional[float] = None, w_neg: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        EQ1: W_per = w_pos * Sum_Pos − w_neg * Sum_Neg
        EQ2: V_per = (W_per / Total_Pixels) * 100
        Adds W_per, V_per columns. Does not modify input.
        """
        w_pos = self.cfg.W_POS if w_pos is None else float(w_pos)
        w_neg = self.cfg.W_NEG if w_neg is None else float(w_neg)
        df = pixel_df.copy()
        df["W_per"] = w_pos * df["Sum_Pos"] - w_neg * df["Sum_Neg"]
        df["V_per"] = (df["W_per"] / df["Total_Pixels"].replace(0, np.nan)) * 100.0
        df["V_per"] = df["V_per"].fillna(0)
        return df

    # =================================================================
    # Aggregation
    # =================================================================
    @staticmethod
    def aggregate_to_route(per_image_df: pd.DataFrame) -> float:
        """Mode A: mean V_per across all images of a single route."""
        if per_image_df is None or per_image_df.empty:
            return 0.0
        return float(per_image_df["V_per"].mean())

    @staticmethod
    def aggregate_to_station(per_image_df: pd.DataFrame) -> dict:
        """
        Mode C: mean V_per across all images collected within a station's radius.
        Returns dict with: n_images, V_per, V_per_std.
        """
        if per_image_df is None or per_image_df.empty:
            return {"n_images": 0, "V_per": 0.0, "V_per_std": 0.0}
        return {
            "n_images":   int(len(per_image_df)),
            "V_per":      float(per_image_df["V_per"].mean()),
            "V_per_std":  float(per_image_df["V_per"].std()) if len(per_image_df) > 1 else 0.0,
        }

    def aggregate_to_cells(
        self, pixel_df: pd.DataFrame, plan_df: pd.DataFrame,
        *,
        w_pos: Optional[float] = None, w_neg: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Mode B: per-cell V_per. `pixel_df` Filename must match plan_df.image_id+ext.
        Returns DataFrame with [cell_id, n_images, V_per_mean].
        """
        df = self.compute_V_per_per_image(pixel_df, w_pos=w_pos, w_neg=w_neg)
        df["image_id"] = df["Filename"].apply(lambda s: os.path.splitext(s)[0])
        joined = df.merge(plan_df[["image_id", "cell_id"]], on="image_id", how="inner")
        out = (joined.groupby("cell_id")
                     .agg(n_images=("V_per", "size"), V_per=("V_per", "mean"))
                     .reset_index())
        return out
