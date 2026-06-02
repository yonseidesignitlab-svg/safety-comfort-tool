"""
File I/O helpers — encoding-safe CSV loader + dong polygon extractor.

Source: PREVIOUS/utils.py read_csv_safe, UPDATE/dong_grid_analysis.py polygon load logic.
"""
import os
import pandas as pd
import geopandas as gpd


def read_csv_safe(filepath):
    """
    Korean-CSV-tolerant loader. Tries common encodings in order.
    Returns DataFrame or None on failure.
    """
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(filepath, encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            return None
        except Exception:
            return None
    return None


def load_dong_polygon(geojson_path: str, target_dong_name: str):
    """
    Load the hangjeongdong GeoJSON and return a single-row GeoDataFrame
    matching the requested dong's full administrative name.

    Args:
        geojson_path: path to the dong-boundary GeoJSON (e.g. seoul_dong.geojson)
        target_dong_name: full name e.g. "서울특별시 마포구 합정동"
    Returns:
        GeoDataFrame (EPSG:4326), single row. Raises if not found.
    """
    gdf = gpd.read_file(geojson_path, encoding="utf-8")
    # The GeoJSON typically has "sgg_nm" / "adm_nm" or "adm_nm2" fields.
    # Try the most likely name columns in order:
    for name_col in ("adm_nm", "ADM_NM", "adm_nm2", "ADM_NM2", "name"):
        if name_col in gdf.columns:
            hit = gdf[gdf[name_col] == target_dong_name]
            if not hit.empty:
                return hit.reset_index(drop=True)
    raise ValueError(
        f"Dong not found: '{target_dong_name}'. Available columns: {list(gdf.columns)}"
    )
