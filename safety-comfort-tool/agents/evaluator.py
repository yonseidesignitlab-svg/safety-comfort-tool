"""
agents/evaluator.py — Evaluator

Thesis Section 3.3 / 4.1–4.4:
  "The Evaluator transforms the processed data from the preceding stage into
   comparable quantitative indices and synthesizes them to classify each
   spatial unit's position in the Safety-Comfort Decision Matrix (SCDM)."

Responsibilities:
  • EQ3 (standardization) — Z-score I_phy and I_per across the analysis set
  • SCDM quadrant classification (Q1 Optimal / Q2 Surveillance-Deficient /
    Q3 Vulnerable / Q4 Fortified)
  • Combine the two indices into a single dataframe ready for the
    Diagnostic_Reporter to visualize.

Sources merged: z-score logic previously scattered across
  PREVIOUS/infrastructure_analysis_agent.py:136-146 (route I_phy)
  PREVIOUS/visual_perception_agent.py:142-187     (route I_psy → I_per)
  PREVIOUS/app(route_analysis).py:128-130,182-183 (μ/σ aggregation)
  UPDATE/dong_grid_analysis.py:151-153            (cell I_phy)
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

import config


class Evaluator:
    """Z-score standardization + SCDM quadrant classification."""

    def __init__(self, cfg=config):
        self.cfg = cfg

    # -----------------------------------------------------------------
    # Core: Z-score (EQ3)
    # -----------------------------------------------------------------
    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """
        Standard Z-score. Returns 0 everywhere if σ == 0 (degenerate single-sample
        case) — this matches the thesis's "relative comparison within the set"
        interpretation: a single point is, by definition, the mean.
        """
        s = pd.to_numeric(series, errors="coerce")
        mu, sigma = float(s.mean()), float(s.std())
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=s.index, name=series.name)
        return (s - mu) / sigma

    # -----------------------------------------------------------------
    # Index computation
    # -----------------------------------------------------------------
    def compute_I_phy(self, df: pd.DataFrame, value_col: str = "V_phy") -> pd.DataFrame:
        """Add 'I_phy' column = z_score(df[value_col])."""
        df = df.copy()
        df["I_phy"] = self.z_score(df[value_col])
        return df

    def compute_I_per(self, df: pd.DataFrame, value_col: str = "V_per") -> pd.DataFrame:
        """Add 'I_per' column = z_score(df[value_col])."""
        df = df.copy()
        df["I_per"] = self.z_score(df[value_col])
        return df

    # -----------------------------------------------------------------
    # SCDM quadrant assignment
    # -----------------------------------------------------------------
    @staticmethod
    def quadrant_of(i_phy: float, i_per: float) -> str:
        """Return Q1..Q4 per thesis Section 4.4."""
        if i_phy is None or i_per is None or pd.isna(i_phy) or pd.isna(i_per):
            return "NA"
        if   i_phy >= 0 and i_per >= 0: return "Q1"
        elif i_phy <  0 and i_per >= 0: return "Q2"
        elif i_phy <  0 and i_per <  0: return "Q3"
        else:                           return "Q4"

    def classify_quadrant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'quadrant' (Q1..Q4) and 'quadrant_label' (Optimal/...) columns."""
        df = df.copy()
        df["quadrant"] = [self.quadrant_of(p, c) for p, c in zip(df["I_phy"], df["I_per"])]
        labels = {k: v[0] for k, v in self.cfg.QUADRANTS.items()}
        df["quadrant_label"] = df["quadrant"].map(labels).fillna("NA")
        return df

    # -----------------------------------------------------------------
    # Combine two indices into one diagnostic frame
    # -----------------------------------------------------------------
    def combine(
        self,
        phy_df: pd.DataFrame,
        per_df: pd.DataFrame,
        *,
        key: str,
        phy_value_col: str = "V_phy",
        per_value_col: str = "V_per",
    ) -> pd.DataFrame:
        """
        Merge the two value frames on `key` (e.g. 'Route_Name' or 'cell_id'),
        compute I_phy + I_per (Z-score within this combined set), and assign
        SCDM quadrants.

        Use this as the single canonical "diagnostic frame" produced by the
        Evaluator that the Diagnostic_Reporter then visualizes.
        """
        # Outer-merge so we retain partial rows (one index missing → quadrant=NA)
        merged = pd.merge(
            phy_df, per_df, on=key, how="outer", suffixes=("", "_per_dup")
        )
        # Drop duplicate columns that came in via the merge (e.g. University columns)
        merged = merged.loc[:, ~merged.columns.duplicated()]

        merged = self.compute_I_phy(merged, value_col=phy_value_col)
        merged = self.compute_I_per(merged, value_col=per_value_col)
        merged = self.classify_quadrant(merged)
        return merged

    # -----------------------------------------------------------------
    # Summary stats helper (for Diagnostic_Reporter to print in report)
    # -----------------------------------------------------------------
    @staticmethod
    def summary_stats(df: pd.DataFrame) -> dict:
        """Quadrant counts + per-quadrant mean indices + dataset μ/σ for both indices."""
        out = {
            "n":          int(len(df)),
            "I_phy_mean": float(df["I_phy"].mean())  if "I_phy" in df else None,
            "I_phy_std":  float(df["I_phy"].std())   if "I_phy" in df else None,
            "I_per_mean": float(df["I_per"].mean())  if "I_per" in df else None,
            "I_per_std":  float(df["I_per"].std())   if "I_per" in df else None,
        }
        if "quadrant" in df:
            counts = df["quadrant"].value_counts().to_dict()
            for q in ("Q1", "Q2", "Q3", "Q4", "NA"):
                out[f"count_{q}"] = int(counts.get(q, 0))
        return out
