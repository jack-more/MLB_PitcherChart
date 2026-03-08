#!/usr/bin/env python3
"""
Step 3b: Feature engineering for MiLB pitchers.

Reuses the same feature extraction logic as 03_feature_engineering.py but:
  - Reads from milb_statcast_{year}.parquet files
  - Uses a lower pitch threshold (200 vs 300 for MLB)
  - Tags all output as source="milb"
  - Saves to milb_pitcher_seasons.parquet

Usage:
    python pipeline/03b_milb_features.py
"""

import os
import sys
import gc
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    PITCH_TYPES, MIN_PITCH_TYPE_PCT,
    SHOULDER_HEIGHT_APPROX, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
    CURRENT_YEAR,
)

# Import feature computation functions from shared module
from s03_feature_engineering_compat import (
    compute_pitch_usage, compute_spin_rates, compute_arm_angle,
    compute_whiff_rate, compute_handedness, compute_velo_and_extras,
    compute_movement, compute_pitcher_names,
)

# Lower threshold for MiLB (shorter seasons, less data)
MILB_MIN_PITCHES = 200
MILB_SEASONS = list(range(2023, CURRENT_YEAR + 1))


def process_milb_season(year: int) -> pd.DataFrame:
    """Process a single MiLB season into pitcher-season features."""
    path = os.path.join(RAW_DATA_DIR, f"milb_statcast_{year}.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] No MiLB data file for {year}")
        return pd.DataFrame()

    print(f"  Loading MiLB {year}...")
    df = pd.read_parquet(path)

    if "game_year" not in df.columns:
        df["game_year"] = year

    # Filter to pitches with valid pitch_type
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()
    print(f"  {len(df):,} MiLB pitches loaded")

    if len(df) == 0:
        return pd.DataFrame()

    # Reclassify SV → ST (simple default for MiLB since we don't have career mapping)
    if "SV" in df["pitch_type"].values:
        n_sv = (df["pitch_type"] == "SV").sum()
        df.loc[df["pitch_type"] == "SV", "pitch_type"] = "ST"
        print(f"    Reclassified {n_sv} SV pitches → ST")

    # Compute all feature groups (same as MLB)
    usage = compute_pitch_usage(df)
    spin = compute_spin_rates(df)
    arm = compute_arm_angle(df)
    whiff = compute_whiff_rate(df)
    hand = compute_handedness(df)
    extras = compute_velo_and_extras(df)
    movement = compute_movement(df)
    names = compute_pitcher_names(df)

    # Merge everything
    features = usage
    for feat_df in [spin, arm, whiff, hand, extras, movement]:
        features = features.merge(feat_df, on=["pitcher", "game_year"], how="left")
    features = features.merge(names, on="pitcher", how="left")

    # Filter by minimum pitches
    qualified = features[features["total_pitches"] >= MILB_MIN_PITCHES].copy()
    print(f"  {len(qualified)} qualified MiLB pitcher-seasons for {year}")

    return qualified


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    all_features = []
    for year in MILB_SEASONS:
        print(f"\n{'='*50}")
        print(f"Processing MiLB {year}")
        print(f"{'='*50}")

        result = process_milb_season(year)
        if len(result) > 0:
            all_features.append(result)
        gc.collect()

    if not all_features:
        print("No MiLB data processed!")
        return

    milb_pitchers = pd.concat(all_features, ignore_index=True)

    # Tag as MiLB source
    milb_pitchers["source"] = "milb"

    # Default is_sp to 0 (MiLB roles are less reliable)
    if "is_sp" not in milb_pitchers.columns:
        milb_pitchers["is_sp"] = 0

    # Fill NaNs in feature columns
    fill_cols = [c for c in milb_pitchers.columns if c.startswith(("pct_", "spin_", "avg_"))]
    fill_cols += ["arm_angle", "whiff_rate", "zone_rate", "groundball_rate",
                  "avg_extension", "pfx_x_avg", "pfx_z_avg"]
    for col in fill_cols:
        if col in milb_pitchers.columns:
            milb_pitchers[col] = milb_pitchers[col].fillna(0)

    # Remove position players
    if "avg_velo_FF" in milb_pitchers.columns:
        pos_mask = (milb_pitchers["avg_velo_FF"] > 0) & (milb_pitchers["avg_velo_FF"] < 80)
        removed = pos_mask.sum()
        milb_pitchers = milb_pitchers[~pos_mask].copy()
        if removed > 0:
            print(f"\n  Removed {removed} position-player MiLB entries")

    out_path = os.path.join(PROCESSED_DATA_DIR, "milb_pitcher_seasons.parquet")
    milb_pitchers.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\nSaved: {out_path}")
    print(f"Total MiLB pitcher-seasons: {len(milb_pitchers):,}")
    print(f"Columns: {list(milb_pitchers.columns)}")


if __name__ == "__main__":
    main()
