"""
agents/physical_auditor.py — Physical Auditor

Thesis Section 3.3:
  "The Physical Auditor collects and preprocesses physical infrastructure data
   (CCTV / Emergency Bell / Security Light), computing the Weighted Safety Sum
   S_phy and the Physical Safety Density D_phy or per-length value V_phy."

This agent is geometry-agnostic: the same .analyze() method works on a route
buffer (LineString → buffer(100m) → Polygon) for Mode A AND on a grid cell
Polygon for Mode B. The only difference is whether `length_km` is provided.

EQ1 (S_phy):  weighted sum of facility counts
EQ2 (V/D):    V_phy = S_phy / length_km   (route)
              D_phy = S_phy / area_km²    (cell)
EQ3 (I_phy):  Z-score across the analysis set  (handled by Evaluator)

Sources merged:
  PREVIOUS/infrastructure_analysis_agent.py — route-buffer counting + viz prep
  UPDATE/dong_grid_analysis.py             — grid generation, osmnx road filter,
                                              cell sjoin, area density
"""
from __future__ import annotations
import os
from typing import Optional, Iterable

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, box

import config


class Physical_Auditor:
    """Spatial aggregation of CCTV / Emergency Bell / Security Light."""

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    def __init__(self, cfg=config):
        self.cfg = cfg
        self._raw_cache: dict[str, pd.DataFrame] = {}
        self._gdf_cache: dict[str, gpd.GeoDataFrame] = {}

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------
    def load_raw(self, target_gu: Optional[Iterable[str]] = None) -> dict[str, pd.DataFrame]:
        """
        Load the three Seoul facility CSVs as raw DataFrames.
        Filters by 자치구 if provided (Mode A often restricts to 마포구·서대문구).
        Cached per-instance.
        """
        key = "all" if target_gu is None else ",".join(sorted(target_gu))
        if key in self._raw_cache:
            return self._raw_cache[key]

        paths = {"CCTV": self.cfg.DATA_CCTV, "Bell": self.cfg.DATA_BELL, "Light": self.cfg.DATA_LIGHT}
        out: dict[str, pd.DataFrame] = {}
        for kind, path in paths.items():
            df = pd.read_csv(path)
            df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
            df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
            df = df.dropna(subset=["위도", "경도"])
            if target_gu and "자치구" in df.columns:
                df = df[df["자치구"].isin(list(target_gu))]
            out[kind] = df.reset_index(drop=True)
        self._raw_cache[key] = out
        return out

    def load_infra_gdfs(self, target_gu: Optional[Iterable[str]] = None) -> dict[str, gpd.GeoDataFrame]:
        """
        Return CCTV/Bell/Light as projected GeoDataFrames (EPSG:5179).
        Use this for any spatial operation; the projected CRS yields correct
        meter-based buffer/distance/area math.
        """
        key = "all" if target_gu is None else ",".join(sorted(target_gu))
        if key in self._gdf_cache:
            return self._gdf_cache[key]

        raw = self.load_raw(target_gu)
        out: dict[str, gpd.GeoDataFrame] = {}
        for kind, df in raw.items():
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["경도"], df["위도"]),
                crs=self.cfg.CRS_WGS,
            ).to_crs(self.cfg.CRS_KTM)
            out[kind] = gdf
        self._gdf_cache[key] = out
        return out

    # -----------------------------------------------------------------
    # Geometry-agnostic core
    # -----------------------------------------------------------------
    def analyze(
        self,
        geometry,
        infra_gdfs: dict[str, gpd.GeoDataFrame],
        *,
        length_km: Optional[float] = None,
        area_km2: Optional[float] = None,
    ) -> dict:
        """
        Count facilities within `geometry` (projected, EPSG:5179) and compute
        weighted aggregates per thesis EQ1+EQ2.

        Args:
            geometry: shapely Polygon (cell / buffered route / dong).
                      MUST already be in EPSG:5179.
            infra_gdfs: dict from .load_infra_gdfs() — also in EPSG:5179.
            length_km: if provided → V_phy = S_phy / length_km  (route mode)
            area_km2:  if not provided → computed from geometry.area (m²)

        Returns: dict with n_CCTV, n_Bell, n_Light, S_phy, area_km2,
                 V_phy (always set: V uses length if provided else area density).
        """
        counts = {"CCTV": 0, "Bell": 0, "Light": 0}
        for kind, gdf in infra_gdfs.items():
            if gdf is not None and not gdf.empty:
                counts[kind] = int(gdf.geometry.within(geometry).sum())

        # EQ1: Weighted Safety Sum
        s_phy = (
            counts["CCTV"]  * self.cfg.W_CCTV  +
            counts["Bell"]  * self.cfg.W_BELL  +
            counts["Light"] * self.cfg.W_LIGHT
        )

        if area_km2 is None:
            area_km2 = geometry.area / 1e6

        # EQ2: Scale adjustment
        if length_km is not None and length_km > 0:
            v_phy = s_phy / length_km          # route-mode: linear density
            mode = "linear"
        elif area_km2 > 0:
            v_phy = s_phy / area_km2           # cell/dong-mode: area density
            mode = "area"
        else:
            v_phy = 0.0
            mode = "degenerate"

        return {
            "n_CCTV":   counts["CCTV"],
            "n_Bell":   counts["Bell"],
            "n_Light":  counts["Light"],
            "S_phy":    float(s_phy),
            "area_km2": float(area_km2),
            "length_km": float(length_km) if length_km is not None else None,
            "V_phy":    float(v_phy),
            "scale_mode": mode,
        }

    # -----------------------------------------------------------------
    # Mode A convenience — route → buffered polygon
    # -----------------------------------------------------------------
    def analyze_route(
        self,
        raw_coords: list[tuple[float, float]],
        buffer_m: float,
        infra_gdfs: dict[str, gpd.GeoDataFrame],
    ) -> Optional[dict]:
        """
        raw_coords: [(lat, lng), ...] in EPSG:4326.
        Returns analyze() output augmented with length_km, or None if input invalid.
        """
        if not raw_coords or len(raw_coords) < 2:
            return None
        # Shapely expects (x=lon, y=lat)
        line_wgs = LineString([(c[1], c[0]) for c in raw_coords])
        gline_tm = gpd.GeoSeries([line_wgs], crs=self.cfg.CRS_WGS).to_crs(self.cfg.CRS_KTM)
        length_m = float(gline_tm.length.iloc[0])
        length_km = length_m / 1000.0
        buf = gline_tm.buffer(buffer_m).iloc[0]
        return self.analyze(buf, infra_gdfs, length_km=length_km)

    # -----------------------------------------------------------------
    # Mode B convenience — single cell polygon
    # -----------------------------------------------------------------
    def analyze_cell(self, cell_geom, infra_gdfs: dict[str, gpd.GeoDataFrame]) -> dict:
        """cell_geom must be EPSG:5179 Polygon (matches grid_tm rows from run_dong_grid)."""
        return self.analyze(cell_geom, infra_gdfs)

    # -----------------------------------------------------------------
    # Mode C convenience — single station, circular buffer
    # -----------------------------------------------------------------
    def analyze_station(
        self,
        station_name: str,
        lat: float,
        lng: float,
        infra_gdfs: dict[str, gpd.GeoDataFrame],
        *,
        radius_m: int = 500,
    ) -> dict:
        """
        Count facilities within a circular buffer around a station coordinate.
        Mirrors analyze_route() but for a point-radius geometry.

        Returns dict with station_slug + standard analyze() fields.
        """
        from shapely.geometry import Point
        center_wgs = Point(lng, lat)
        center_tm  = gpd.GeoSeries([center_wgs], crs=self.cfg.CRS_WGS).to_crs(self.cfg.CRS_KTM).iloc[0]
        buf_tm     = center_tm.buffer(radius_m)
        rec = self.analyze(buf_tm, infra_gdfs)  # length_km=None → area density
        rec["station_slug"] = station_name
        rec["radius_m"]     = radius_m
        return rec

    # -----------------------------------------------------------------
    # Mode B: full dong-grid pipeline (replaces UPDATE/dong_grid_analysis.py)
    # -----------------------------------------------------------------
    def run_dong_grid(
        self,
        target_dong: str,
        dong_short: str,
        *,
        grid_size_m: Optional[int] = None,
        verbose: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Full Mode B step 1: dong polygon → grid → OSMnx walk filter → infra sjoin
        → S_phy + D_phy. (I_phy z-score is applied by Evaluator afterward.)

        Returns: GeoDataFrame in EPSG:5179 with columns
            cell_id, geometry, area_km2, has_road,
            n_CCTV, n_Bell, n_Light, S_phy, D_phy
            (I_phy is added later by Evaluator)
        """
        import osmnx as ox  # heavy — defer import

        grid_size_m = grid_size_m or self.cfg.GRID_SIZE_M

        # --- 1. Dong boundary ---
        if verbose: print("[PA] Loading dong boundary…")
        dongs = gpd.read_file(self.cfg.DATA_GEOJSON)
        dong = dongs[dongs["adm_nm"] == target_dong].copy()
        if dong.empty:
            raise ValueError(f"Dong '{target_dong}' not found in {self.cfg.DATA_GEOJSON}")
        dong_wgs = dong.to_crs(self.cfg.CRS_WGS)
        dong_tm  = dong.to_crs(self.cfg.CRS_KTM)
        dong_geom_tm  = dong_tm.geometry.iloc[0]
        dong_geom_wgs = dong_wgs.geometry.iloc[0]
        if verbose: print(f"      Area: {dong_geom_tm.area/1e6:.3f} km²")

        # --- 2. Grid generation ---
        if verbose: print("[PA] Generating grid cells…")
        minx, miny, maxx, maxy = dong_geom_tm.bounds
        cells = []
        cell_id = 0
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cell_box = box(x, y, x + grid_size_m, y + grid_size_m)
                if cell_box.intersects(dong_geom_tm):
                    clipped = cell_box.intersection(dong_geom_tm)
                    if not clipped.is_empty and clipped.area > 0:
                        cells.append({"cell_id": cell_id, "geometry": clipped})
                        cell_id += 1
                y += grid_size_m
            x += grid_size_m
        grid_tm = gpd.GeoDataFrame(cells, crs=self.cfg.CRS_KTM)
        grid_tm["area_km2"] = grid_tm.geometry.area / 1e6
        if verbose: print(f"      Cells before road filter: {len(grid_tm)}")

        # --- 3. OSMnx walk network ---
        if verbose: print("[PA] Downloading OSM walk network…")
        G = ox.graph_from_polygon(dong_geom_wgs, network_type="walk")
        edges_wgs = ox.graph_to_gdfs(G, nodes=False).reset_index(drop=True)
        edges_tm = edges_wgs.to_crs(self.cfg.CRS_KTM)

        # --- 4. Pedestrian-cell filter ---
        if verbose: print("[PA] Filtering cells by road intersection…")
        with_road = gpd.sjoin(grid_tm[["cell_id", "geometry"]],
                              edges_tm[["geometry"]],
                              predicate="intersects", how="inner")
        kept = set(with_road["cell_id"].unique())
        grid_tm["has_road"] = grid_tm["cell_id"].isin(kept)
        if verbose:
            print(f"      Kept: {grid_tm['has_road'].sum()}, "
                  f"Excluded: {(~grid_tm['has_road']).sum()}")

        # --- 5. Infra sjoin + S_phy + D_phy ---
        if verbose: print("[PA] Aggregating facilities…")
        infra = self.load_infra_gdfs()   # all gu, no filter
        analysis = grid_tm[grid_tm["has_road"]].copy().reset_index(drop=True)
        for kind, gdf in infra.items():
            j = gpd.sjoin(gdf, analysis[["cell_id", "geometry"]],
                          predicate="within", how="inner")
            counts = j.groupby("cell_id").size()
            analysis[f"n_{kind}"] = analysis["cell_id"].map(counts).fillna(0).astype(int)

        # EQ1
        analysis["S_phy"] = (
            analysis["n_CCTV"]  * self.cfg.W_CCTV  +
            analysis["n_Bell"]  * self.cfg.W_BELL  +
            analysis["n_Light"] * self.cfg.W_LIGHT
        )
        # EQ2 (area density)
        analysis["D_phy"] = analysis["S_phy"] / analysis["area_km2"]
        # Convenience: also expose V_phy as area-mode value (Evaluator will Z-score this)
        analysis["V_phy"] = analysis["D_phy"]

        if verbose:
            print(f"      Analysis cells: {len(analysis)}, "
                  f"D_phy μ={analysis['D_phy'].mean():.1f}, "
                  f"σ={analysis['D_phy'].std():.1f}")

        # Attach the full grid + edges for downstream viz consumers
        analysis.attrs["full_grid"]    = grid_tm
        analysis.attrs["edges_wgs"]    = edges_wgs
        analysis.attrs["dong_wgs"]     = dong_wgs
        analysis.attrs["infra_gdfs"]   = infra
        return analysis

    # -----------------------------------------------------------------
    # Mode A helper — bounding-box pre-filter for Streamlit map perf
    # -----------------------------------------------------------------
    @staticmethod
    def filter_by_bbox(raw_data: dict[str, pd.DataFrame],
                       routes: list[dict], buffer_m: int = 100) -> dict[str, pd.DataFrame]:
        """
        Quick lat/lng bbox filter (no GeoPandas) — used by Streamlit to keep
        only points near the user's drawn routes for fast map rendering.
        """
        if not routes:
            return {k: (df.iloc[0:0] if df is not None else df) for k, df in raw_data.items()}
        lat_buf = buffer_m / 111000.0
        lng_buf = buffer_m / 88000.0
        bboxes = []
        for r in routes:
            coords = r.get("raw_coords") or []
            if not coords:
                continue
            lats = [p[0] for p in coords]; lngs = [p[1] for p in coords]
            bboxes.append((min(lats)-lat_buf, max(lats)+lat_buf,
                           min(lngs)-lng_buf, max(lngs)+lng_buf))
        if not bboxes:
            return {k: (df.iloc[0:0] if df is not None else df) for k, df in raw_data.items()}

        out: dict[str, pd.DataFrame] = {}
        for kind, df in raw_data.items():
            if df is None or df.empty:
                out[kind] = df; continue
            mask = pd.Series(False, index=df.index)
            for (a, b, c, d) in bboxes:
                mask = mask | ((df["위도"] >= a) & (df["위도"] <= b) &
                               (df["경도"] >= c) & (df["경도"] <= d))
            out[kind] = df[mask].copy()
        return out
