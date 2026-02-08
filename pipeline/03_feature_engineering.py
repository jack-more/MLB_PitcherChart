"""
Step 3: Aggregate pitch-level Statcast data into pitcher-season feature vectors.

Produces the feature matrix used for K-Means clustering:
  - Pitch mix usage rates (pct_FF, pct_SI, ...)
  - Spin rates (spin_overall, spin_FF, ...)
  - Arm angle (derived from release point)
  - Whiff rate
  - Handedness (is_rhp)
  - Role (is_sp)
  - Velocity, extension, zone rate, groundball rate
"""

import os
import sys
import gc
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEASONS, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    PITCH_TYPES, MIN_PITCHES, MIN_PITCH_TYPE_PCT,
    SHOULDER_HEIGHT_APPROX, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
)


def compute_pitch_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pitch type usage rates per pitcher-season."""
    total = df.groupby(["pitcher", "game_year"]).size().reset_index(name="total_pitches")
    by_type = df.groupby(["pitcher", "game_year", "pitch_type"]).size().reset_index(name="count")
    by_type = by_type.merge(total, on=["pitcher", "game_year"])
    by_type["pct"] = by_type["count"] / by_type["total_pitches"]

    # Pivot to wide format
    usage = by_type.pivot_table(
        index=["pitcher", "game_year"],
        columns="pitch_type",
        values="pct",
        fill_value=0.0,
    ).reset_index()

    # Ensure all pitch type columns exist
    for pt in PITCH_TYPES:
        col = pt  # column name from pivot
        if col not in usage.columns:
            usage[col] = 0.0

    # Rename to pct_ prefix and keep only tracked types
    rename_map = {pt: f"pct_{pt}" for pt in PITCH_TYPES if pt in usage.columns}
    usage = usage.rename(columns=rename_map)

    keep_cols = ["pitcher", "game_year"] + [f"pct_{pt}" for pt in PITCH_TYPES]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in usage.columns]
    usage = usage[keep_cols].copy()

    # Merge total_pitches for later filtering
    usage = usage.merge(total, on=["pitcher", "game_year"])

    return usage


def compute_spin_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean spin rate overall and per key pitch type."""
    overall = df.groupby(["pitcher", "game_year"])["release_spin_rate"].mean().reset_index()
    overall.columns = ["pitcher", "game_year", "spin_overall"]

    # Spin per pitch type (FF, SL, CU are most informative for clustering)
    spin_types = ["FF", "SL", "CU"]
    spin_dfs = [overall]

    for pt in spin_types:
        pt_data = df[df["pitch_type"] == pt].groupby(
            ["pitcher", "game_year"]
        )["release_spin_rate"].mean().reset_index()
        pt_data.columns = ["pitcher", "game_year", f"spin_{pt}"]
        spin_dfs.append(pt_data)

    result = spin_dfs[0]
    for sdf in spin_dfs[1:]:
        result = result.merge(sdf, on=["pitcher", "game_year"], how="left")

    return result


def compute_arm_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Derive arm angle from release_pos_x and release_pos_z."""
    subset = df[["pitcher", "game_year", "release_pos_x", "release_pos_z", "p_throws"]].dropna(
        subset=["release_pos_x", "release_pos_z"]
    ).copy()

    # Adjust x for handedness: RHP positive = arm side, LHP flip
    subset["adj_x"] = np.where(
        subset["p_throws"] == "R",
        subset["release_pos_x"],
        -subset["release_pos_x"],
    )

    subset["arm_angle_deg"] = np.degrees(
        np.arctan2(subset["adj_x"], subset["release_pos_z"] - SHOULDER_HEIGHT_APPROX)
    )

    result = subset.groupby(["pitcher", "game_year"])["arm_angle_deg"].mean().reset_index()
    result.columns = ["pitcher", "game_year", "arm_angle"]
    return result


def compute_whiff_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute whiff rate (swinging_strike / total_swings)."""
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)

    result = df.groupby(["pitcher", "game_year"]).agg(
        total_swings=("is_swing", "sum"),
        total_whiffs=("is_whiff", "sum"),
    ).reset_index()

    result["whiff_rate"] = result["total_whiffs"] / result["total_swings"].clip(lower=1)
    return result[["pitcher", "game_year", "whiff_rate"]]


def compute_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """Extract pitcher handedness as binary feature."""
    hand = df.groupby(["pitcher", "game_year"])["p_throws"].first().reset_index()
    hand["is_rhp"] = (hand["p_throws"] == "R").astype(int)
    return hand[["pitcher", "game_year", "is_rhp"]]


def compute_velo_and_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average fastball velo, extension, zone rate, groundball rate."""
    # Fastball velocity (FF only)
    ff = df[df["pitch_type"] == "FF"]
    velo = ff.groupby(["pitcher", "game_year"])["release_speed"].mean().reset_index()
    velo.columns = ["pitcher", "game_year", "avg_velo_FF"]

    # Release extension
    ext = df.groupby(["pitcher", "game_year"])["release_extension"].mean().reset_index()
    ext.columns = ["pitcher", "game_year", "avg_extension"]

    # Zone rate (zone 1-9 = in the strike zone)
    df_zone = df.dropna(subset=["zone"]).copy()
    df_zone["in_zone"] = df_zone["zone"].between(1, 9).astype(int)
    zone_rate = df_zone.groupby(["pitcher", "game_year"]).agg(
        total_pitches_z=("zone", "count"),
        zone_pitches=("in_zone", "sum"),
    ).reset_index()
    zone_rate["zone_rate"] = zone_rate["zone_pitches"] / zone_rate["total_pitches_z"].clip(lower=1)
    zone_rate = zone_rate[["pitcher", "game_year", "zone_rate"]]

    # Ground ball rate (among batted balls: type == 'X')
    batted = df[df["type"] == "X"].dropna(subset=["bb_type"]).copy()
    if len(batted) > 0:
        batted["is_gb"] = (batted["bb_type"] == "ground_ball").astype(int)
        gb = batted.groupby(["pitcher", "game_year"]).agg(
            batted_balls=("bb_type", "count"),
            ground_balls=("is_gb", "sum"),
        ).reset_index()
        gb["groundball_rate"] = gb["ground_balls"] / gb["batted_balls"].clip(lower=1)
        gb = gb[["pitcher", "game_year", "groundball_rate"]]
    else:
        gb = pd.DataFrame(columns=["pitcher", "game_year", "groundball_rate"])

    # Merge all extras
    result = velo
    for extra_df in [ext, zone_rate, gb]:
        result = result.merge(extra_df, on=["pitcher", "game_year"], how="outer")

    return result


def compute_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average horizontal and vertical pitch movement per pitcher-season."""
    subset = df.dropna(subset=["pfx_x", "pfx_z"]).copy()
    result = subset.groupby(["pitcher", "game_year"]).agg(
        pfx_x_avg=("pfx_x", "mean"),
        pfx_z_avg=("pfx_z", "mean"),
    ).reset_index()
    return result


def compute_pitcher_names(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most common player_name per pitcher ID."""
    names = df.groupby("pitcher")["player_name"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
    ).reset_index()
    names.columns = ["pitcher", "player_name"]
    return names


def process_season(year: int) -> pd.DataFrame:
    """Process a single season into pitcher-season features."""
    path = os.path.join(RAW_DATA_DIR, f"statcast_{year}.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] No data file for {year}")
        return pd.DataFrame()

    print(f"  Loading {year}...")
    df = pd.read_parquet(path)

    # Ensure game_year exists
    if "game_year" not in df.columns:
        df["game_year"] = year

    # Filter to pitches with valid pitch_type
    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")].copy()

    print(f"  {len(df):,} pitches loaded")

    # Compute all feature groups
    usage = compute_pitch_usage(df)
    spin = compute_spin_rates(df)
    arm = compute_arm_angle(df)
    whiff = compute_whiff_rate(df)
    hand = compute_handedness(df)
    extras = compute_velo_and_extras(df)
    movement = compute_movement(df)
    names = compute_pitcher_names(df)

    # Merge everything on pitcher + game_year
    features = usage
    for feat_df in [spin, arm, whiff, hand, extras, movement]:
        features = features.merge(feat_df, on=["pitcher", "game_year"], how="left")

    # Add names (pitcher-level, no game_year)
    features = features.merge(names, on="pitcher", how="left")

    # Split by minimum pitches: qualified for clustering vs sub-threshold
    qualified = features[features["total_pitches"] >= MIN_PITCHES].copy()
    sub_threshold = features[features["total_pitches"] < MIN_PITCHES].copy()

    print(f"  {len(qualified)} qualified pitcher-seasons for {year}")
    if len(sub_threshold) > 0:
        print(f"  {len(sub_threshold)} sub-threshold pitcher-seasons (< {MIN_PITCHES} pitches)")

    return qualified, sub_threshold


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    all_features = []
    all_sub_threshold = []
    for year in SEASONS:
        print(f"\n{'='*50}")
        print(f"Processing {year}")
        print(f"{'='*50}")

        result = process_season(year)
        if isinstance(result, tuple):
            qualified, sub_thresh = result
        else:
            qualified, sub_thresh = result, pd.DataFrame()

        if len(qualified) > 0:
            all_features.append(qualified)
        if len(sub_thresh) > 0:
            all_sub_threshold.append(sub_thresh)

        gc.collect()

    if not all_features:
        print("No data processed!")
        return

    pitcher_seasons = pd.concat(all_features, ignore_index=True)

    # Load and merge SP/RP roles
    roles_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_roles.parquet")
    if os.path.exists(roles_path):
        roles = pd.read_parquet(roles_path)
        pitcher_seasons = pitcher_seasons.merge(
            roles[["pitcher", "game_year", "role"]],
            on=["pitcher", "game_year"],
            how="left",
        )
        pitcher_seasons["is_sp"] = (pitcher_seasons["role"] == "SP").astype(int)
        pitcher_seasons.drop(columns=["role"], inplace=True)
    else:
        print("WARNING: pitcher_roles.parquet not found — defaulting is_sp to 0")
        pitcher_seasons["is_sp"] = 0

    # Fill remaining NaNs with 0 for clustering features
    fill_cols = [c for c in pitcher_seasons.columns if c.startswith(("pct_", "spin_", "avg_"))]
    fill_cols += ["arm_angle", "whiff_rate", "zone_rate", "groundball_rate", "avg_extension", "pfx_x_avg", "pfx_z_avg"]
    for col in fill_cols:
        if col in pitcher_seasons.columns:
            pitcher_seasons[col] = pitcher_seasons[col].fillna(0)

    out_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    pitcher_seasons.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\nSaved: {out_path}")
    print(f"Total pitcher-seasons: {len(pitcher_seasons):,}")
    print(f"Columns: {list(pitcher_seasons.columns)}")

    # Save sub-threshold pitchers for nearest-cluster assignment in step 4
    if all_sub_threshold:
        sub_df = pd.concat(all_sub_threshold, ignore_index=True)

        # Merge SP/RP roles for sub-threshold too
        if os.path.exists(roles_path):
            sub_df = sub_df.merge(
                roles[["pitcher", "game_year", "role"]],
                on=["pitcher", "game_year"],
                how="left",
            )
            sub_df["is_sp"] = (sub_df["role"] == "SP").astype(int)
            sub_df.drop(columns=["role"], inplace=True)
        else:
            sub_df["is_sp"] = 0

        for col in fill_cols:
            if col in sub_df.columns:
                sub_df[col] = sub_df[col].fillna(0)

        sub_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons_sub_threshold.parquet")
        sub_df.to_parquet(sub_path, engine="pyarrow", compression="snappy")
        print(f"\nSaved sub-threshold: {sub_path}")
        print(f"Sub-threshold pitcher-seasons: {len(sub_df):,}")
    else:
        print("\nNo sub-threshold pitcher-seasons found.")


if __name__ == "__main__":
    main()
