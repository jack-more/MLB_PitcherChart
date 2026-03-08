"""
Shared feature engineering functions for both MLB and MiLB pipelines.

Extracted from 03_feature_engineering.py so they can be reused by 03b_milb_features.py
without duplicating 400+ lines of code.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PITCH_TYPES, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
    SHOULDER_HEIGHT_APPROX,
)


def compute_pitch_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pitch type usage rates per pitcher-season."""
    total = df.groupby(["pitcher", "game_year"]).size().reset_index(name="total_pitches")
    by_type = df.groupby(["pitcher", "game_year", "pitch_type"]).size().reset_index(name="count")
    by_type = by_type.merge(total, on=["pitcher", "game_year"])
    by_type["pct"] = by_type["count"] / by_type["total_pitches"]

    usage = by_type.pivot_table(
        index=["pitcher", "game_year"],
        columns="pitch_type",
        values="pct",
        fill_value=0.0,
    ).reset_index()

    for pt in PITCH_TYPES:
        if pt not in usage.columns:
            usage[pt] = 0.0

    rename_map = {pt: f"pct_{pt}" for pt in PITCH_TYPES if pt in usage.columns}
    usage = usage.rename(columns=rename_map)

    keep_cols = ["pitcher", "game_year"] + [f"pct_{pt}" for pt in PITCH_TYPES]
    keep_cols = [c for c in keep_cols if c in usage.columns]
    usage = usage[keep_cols].copy()
    usage = usage.merge(total, on=["pitcher", "game_year"])

    return usage


def compute_spin_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean spin rate overall and per key pitch type."""
    overall = df.groupby(["pitcher", "game_year"])["release_spin_rate"].mean().reset_index()
    overall.columns = ["pitcher", "game_year", "spin_overall"]

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
    ff = df[df["pitch_type"] == "FF"]
    velo_ff = ff.groupby(["pitcher", "game_year"])["release_speed"].mean().reset_index()
    velo_ff.columns = ["pitcher", "game_year", "avg_velo_FF"]

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
    velo = velo_ff.merge(
        fastest, on=["pitcher", "game_year"], how="outer", suffixes=("", "_fallback")
    )
    velo["avg_velo_FF"] = velo["avg_velo_FF"].fillna(velo["avg_velo_FF_fallback"])
    velo = velo[["pitcher", "game_year", "avg_velo_FF"]]

    ext = df.groupby(["pitcher", "game_year"])["release_extension"].mean().reset_index()
    ext.columns = ["pitcher", "game_year", "avg_extension"]

    df_zone = df.dropna(subset=["zone"]).copy()
    df_zone["in_zone"] = df_zone["zone"].between(1, 9).astype(int)
    zone_rate = df_zone.groupby(["pitcher", "game_year"]).agg(
        total_pitches_z=("zone", "count"),
        zone_pitches=("in_zone", "sum"),
    ).reset_index()
    zone_rate["zone_rate"] = zone_rate["zone_pitches"] / zone_rate["total_pitches_z"].clip(lower=1)
    zone_rate = zone_rate[["pitcher", "game_year", "zone_rate"]]

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
