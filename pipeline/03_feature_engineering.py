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
from scipy.stats import entropy as sp_entropy

from config import (
    SEASONS, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    PITCH_TYPES, MIN_PITCHES, MIN_PITCH_TYPE_PCT,
    SHOULDER_HEIGHT_APPROX, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
    MIN_PITCHES_PER_SIDE,
)


def build_sv_mapping(raw_data_dir: str, seasons: list) -> dict:
    """Build per-pitcher SV reclassification mapping.

    For each pitcher who throws SV, compute their career-average SV speed and
    vertical break (pfx_z). Classify their SV as CU, SL, or ST based on:
      - pfx_z < -0.50  → CU (curveball drop)
      - speed > 84 mph → SL (slider velocity)
      - else           → ST (sweeper)

    Returns dict: {pitcher_id: "CU"/"SL"/"ST"}
    """
    all_sv = []
    for year in seasons:
        path = os.path.join(raw_data_dir, f"statcast_{year}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=["pitcher", "pitch_type", "release_speed", "pfx_z"])
        sv = df[df["pitch_type"] == "SV"].dropna(subset=["release_speed", "pfx_z"])
        if len(sv) > 0:
            all_sv.append(sv)

    if not all_sv:
        return {}

    sv_all = pd.concat(all_sv, ignore_index=True)
    agg = sv_all.groupby("pitcher").agg(
        avg_speed=("release_speed", "mean"),
        avg_pfx_z=("pfx_z", "mean"),
        count=("pitcher", "count"),
    ).reset_index()

    sv_map = {}
    for _, row in agg.iterrows():
        pid = int(row["pitcher"])
        if row["avg_pfx_z"] < -0.50:
            sv_map[pid] = "CU"
        elif row["avg_speed"] > 84:
            sv_map[pid] = "SL"
        else:
            sv_map[pid] = "ST"

    # Summary
    from collections import Counter
    counts = Counter(sv_map.values())
    print(f"  SV reclassification mapping: {len(sv_map)} pitchers")
    print(f"    → CU: {counts.get('CU', 0)}, SL: {counts.get('SL', 0)}, ST: {counts.get('ST', 0)}")

    return sv_map


def reclassify_sv(df: pd.DataFrame, sv_map: dict) -> pd.DataFrame:
    """Reclassify SV pitches based on per-pitcher mapping."""
    sv_mask = df["pitch_type"] == "SV"
    n_sv = sv_mask.sum()
    if n_sv == 0:
        return df

    df = df.copy()
    reclassified = 0
    for pid, new_type in sv_map.items():
        mask = sv_mask & (df["pitcher"] == pid)
        count = mask.sum()
        if count > 0:
            df.loc[mask, "pitch_type"] = new_type
            reclassified += count

    # Any remaining SV pitches from pitchers not in the map → default to ST
    remaining = (df["pitch_type"] == "SV").sum()
    if remaining > 0:
        df.loc[df["pitch_type"] == "SV", "pitch_type"] = "ST"
        reclassified += remaining

    print(f"    Reclassified {reclassified:,} SV pitches (of {n_sv:,})")
    return df


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
    # Fastball velocity (FF preferred, fallback to fastest pitch)
    ff = df[df["pitch_type"] == "FF"]
    velo_ff = ff.groupby(["pitcher", "game_year"])["release_speed"].mean().reset_index()
    velo_ff.columns = ["pitcher", "game_year", "avg_velo_FF"]

    # Fallback for pitchers with no FF: use their fastest pitch type's avg velo
    all_velo = (
        df.dropna(subset=["release_speed"])
        .groupby(["pitcher", "game_year", "pitch_type"])["release_speed"]
        .mean()
        .reset_index()
    )
    fastest = (
        all_velo.sort_values("release_speed", ascending=False)
        .drop_duplicates(subset=["pitcher", "game_year"], keep="first")
        [["pitcher", "game_year", "release_speed"]]
        .rename(columns={"release_speed": "avg_velo_FF"})
    )
    # Merge: use FF velo where available, fallback to fastest pitch
    velo = velo_ff.merge(
        fastest, on=["pitcher", "game_year"], how="outer", suffixes=("", "_fallback")
    )
    velo["avg_velo_FF"] = velo["avg_velo_FF"].fillna(velo["avg_velo_FF_fallback"])
    velo = velo[["pitcher", "game_year", "avg_velo_FF"]]

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


def compute_zone_location(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pitch location zone features per pitcher-season.

    Three layers:
      Layer 1: WHERE they locate (split by same-side vs opposite-side batters)
      Layer 2: Platoon shifts (how much they change by batter side)
      Layer 3: Location entropy (how predictable their pattern is)

    Returns DataFrame keyed on [pitcher, game_year] with 13 zone columns.
    """
    # Filter to pitches with valid location data
    loc = df.dropna(subset=["plate_x", "plate_z", "stand", "sz_top", "sz_bot"]).copy()
    if len(loc) == 0:
        return pd.DataFrame(columns=["pitcher", "game_year"])

    # Determine same-side vs opposite-side
    loc["is_same_side"] = (loc["stand"] == loc["p_throws"])

    # Normalize vertical position to batter's zone (0 = bottom, 1 = top)
    zone_span = (loc["sz_top"] - loc["sz_bot"]).clip(lower=0.1)
    loc["zone_height"] = (loc["plate_z"] - loc["sz_bot"]) / zone_span

    # Arm-side x: positive = arm side for both RHP and LHP
    loc["arm_side_x"] = np.where(
        loc["p_throws"] == "R",
        loc["plate_x"],
        -loc["plate_x"],
    )

    # Boolean classification flags (vectorized)
    loc["is_upper"] = (loc["zone_height"] > 0.5).astype(np.int8)
    loc["is_arm_side"] = (loc["arm_side_x"] > 0).astype(np.int8)
    loc["is_heart"] = (
        (loc["plate_x"].abs() < 0.3) &
        (loc["zone_height"].between(0.3, 0.7))
    ).astype(np.int8)
    loc["is_edge"] = (
        ~loc["is_heart"].astype(bool) &
        (loc["plate_x"].abs() < 1.2) &
        (loc["zone_height"].between(-0.2, 1.2))
    ).astype(np.int8)

    # 9-quadrant grid for entropy (3 lateral x 3 vertical, vectorized)
    loc["lat_bin"] = np.select(
        [loc["arm_side_x"] > 0.28, loc["arm_side_x"] < -0.28],
        [0, 2], default=1,
    )
    loc["vert_bin"] = np.select(
        [loc["zone_height"] > 0.67, loc["zone_height"] < 0.33],
        [0, 2], default=1,
    )
    loc["quadrant"] = loc["lat_bin"] * 3 + loc["vert_bin"]  # 0-8

    # --- Aggregate per pitcher-season-side ---
    def _side_features(group):
        n = len(group)
        if n == 0:
            return pd.Series({
                "up_rate": np.nan, "arm_side": np.nan,
                "heart_rate": np.nan, "edge_rate": np.nan,
                "location_entropy": np.nan, "n_loc": 0,
            })

        up_rate = group["is_upper"].mean()
        arm_side = group["is_arm_side"].mean()
        heart_rate = group["is_heart"].mean()
        edge_rate = group["is_edge"].mean()

        # Shannon entropy of 9-quadrant distribution
        quad_counts = group["quadrant"].value_counts()
        quad_probs = np.zeros(9)
        for q, cnt in quad_counts.items():
            quad_probs[int(q)] = cnt
        total = quad_probs.sum()
        if total > 0:
            quad_probs = quad_probs / total
        loc_entropy = float(sp_entropy(quad_probs, base=2))

        return pd.Series({
            "up_rate": up_rate,
            "arm_side": arm_side,
            "heart_rate": heart_rate,
            "edge_rate": edge_rate,
            "location_entropy": loc_entropy,
            "n_loc": n,
        })

    print("    Computing zone location features...")
    side_stats = loc.groupby(
        ["pitcher", "game_year", "is_same_side"]
    ).apply(_side_features).reset_index()

    # Split into same-side and opposite-side
    ss = side_stats[side_stats["is_same_side"] == True].copy()
    os_ = side_stats[side_stats["is_same_side"] == False].copy()

    ss = ss.rename(columns={
        "up_rate": "ss_up_rate", "arm_side": "ss_arm_side",
        "heart_rate": "ss_heart_rate", "edge_rate": "ss_edge_rate",
        "location_entropy": "ss_location_entropy", "n_loc": "ss_n",
    }).drop(columns=["is_same_side"])

    os_ = os_.rename(columns={
        "up_rate": "os_up_rate", "arm_side": "os_arm_side",
        "heart_rate": "os_heart_rate", "edge_rate": "os_edge_rate",
        "location_entropy": "os_location_entropy", "n_loc": "os_n",
    }).drop(columns=["is_same_side"])

    # Merge same-side and opposite-side
    result = ss.merge(os_, on=["pitcher", "game_year"], how="outer")

    # Compute overall stats as fallback for thin splits
    overall = loc.groupby(["pitcher", "game_year"]).apply(_side_features).reset_index()
    overall = overall.rename(columns={
        "up_rate": "_ov_up", "arm_side": "_ov_arm",
        "heart_rate": "_ov_heart", "edge_rate": "_ov_edge",
        "location_entropy": "_ov_entropy",
    })

    result = result.merge(
        overall[["pitcher", "game_year", "_ov_up", "_ov_arm", "_ov_heart", "_ov_edge", "_ov_entropy"]],
        on=["pitcher", "game_year"], how="left",
    )

    # Apply fallback for same-side (< MIN_PITCHES_PER_SIDE)
    ss_mask = result["ss_n"].fillna(0) < MIN_PITCHES_PER_SIDE
    for feat, ov in [("ss_up_rate", "_ov_up"), ("ss_arm_side", "_ov_arm"),
                     ("ss_heart_rate", "_ov_heart"), ("ss_edge_rate", "_ov_edge"),
                     ("ss_location_entropy", "_ov_entropy")]:
        result.loc[ss_mask, feat] = result.loc[ss_mask, ov]

    # Apply fallback for opposite-side
    os_mask = result["os_n"].fillna(0) < MIN_PITCHES_PER_SIDE
    for feat, ov in [("os_up_rate", "_ov_up"), ("os_arm_side", "_ov_arm"),
                     ("os_heart_rate", "_ov_heart"), ("os_edge_rate", "_ov_edge"),
                     ("os_location_entropy", "_ov_entropy")]:
        result.loc[os_mask, feat] = result.loc[os_mask, ov]

    # Layer 2: Platoon shifts
    result["platoon_lateral_shift"] = (result["ss_arm_side"] - result["os_arm_side"]).abs()
    result["platoon_height_shift"] = (result["ss_up_rate"] - result["os_up_rate"]).abs()

    # Layer 3: Entropy shift (positive = LESS predictable vs opposite-side)
    result["entropy_shift"] = result["os_location_entropy"] - result["ss_location_entropy"]

    # Keep only the 13 output columns
    keep = ["pitcher", "game_year",
            "ss_up_rate", "ss_arm_side", "ss_heart_rate", "ss_edge_rate",
            "os_up_rate", "os_arm_side", "os_heart_rate", "os_edge_rate",
            "platoon_lateral_shift", "platoon_height_shift",
            "ss_location_entropy", "os_location_entropy", "entropy_shift"]

    return result[keep]


def compute_pitcher_names(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most common player_name per pitcher ID."""
    names = df.groupby("pitcher")["player_name"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
    ).reset_index()
    names.columns = ["pitcher", "player_name"]
    return names


def process_season(year: int, sv_map: dict = None) -> pd.DataFrame:
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

    # Reclassify SV pitches per pitcher mapping
    if sv_map:
        df = reclassify_sv(df, sv_map)

    # Compute all feature groups
    usage = compute_pitch_usage(df)
    spin = compute_spin_rates(df)
    arm = compute_arm_angle(df)
    whiff = compute_whiff_rate(df)
    hand = compute_handedness(df)
    extras = compute_velo_and_extras(df)
    movement = compute_movement(df)
    zone_loc = compute_zone_location(df)
    names = compute_pitcher_names(df)

    # Merge everything on pitcher + game_year
    features = usage
    for feat_df in [spin, arm, whiff, hand, extras, movement, zone_loc]:
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

    # Build per-pitcher SV reclassification mapping (scan all years once)
    print("Building SV reclassification mapping...")
    sv_map = build_sv_mapping(RAW_DATA_DIR, SEASONS)

    # Save mapping for reproducibility
    import json
    sv_map_path = os.path.join(PROCESSED_DATA_DIR, "sv_reclassification.json")
    with open(sv_map_path, "w") as f:
        json.dump({str(k): v for k, v in sv_map.items()}, f, indent=2)
    print(f"  Saved mapping to {sv_map_path}")

    all_features = []
    all_sub_threshold = []
    for year in SEASONS:
        print(f"\n{'='*50}")
        print(f"Processing {year}")
        print(f"{'='*50}")

        result = process_season(year, sv_map=sv_map)
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
    # Zone location features
    fill_cols += [
        "ss_up_rate", "ss_arm_side", "ss_heart_rate", "ss_edge_rate",
        "os_up_rate", "os_arm_side", "os_heart_rate", "os_edge_rate",
        "platoon_lateral_shift", "platoon_height_shift",
        "ss_location_entropy", "os_location_entropy", "entropy_shift",
    ]
    for col in fill_cols:
        if col in pitcher_seasons.columns:
            pitcher_seasons[col] = pitcher_seasons[col].fillna(0)

    # Remove position players (non-pitchers who occasionally pitch)
    pre_filter = len(pitcher_seasons)
    pos_mask = (pitcher_seasons["avg_velo_FF"] > 0) & (pitcher_seasons["avg_velo_FF"] < 80)
    pitcher_seasons = pitcher_seasons[~pos_mask].copy()
    removed = pre_filter - len(pitcher_seasons)
    if removed > 0:
        print(f"\n  Removed {removed} position-player pitcher-seasons (avg_velo_FF < 80 mph)")

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

        # Remove position players from sub-threshold too
        sub_pos_mask = (sub_df["avg_velo_FF"] > 0) & (sub_df["avg_velo_FF"] < 80)
        sub_removed = sub_pos_mask.sum()
        sub_df = sub_df[~sub_pos_mask].copy()
        if sub_removed > 0:
            print(f"  Removed {sub_removed} position-player sub-threshold entries")

        sub_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons_sub_threshold.parquet")
        sub_df.to_parquet(sub_path, engine="pyarrow", compression="snappy")
        print(f"\nSaved sub-threshold: {sub_path}")
        print(f"Sub-threshold pitcher-seasons: {len(sub_df):,}")
    else:
        print("\nNo sub-threshold pitcher-seasons found.")


if __name__ == "__main__":
    main()
