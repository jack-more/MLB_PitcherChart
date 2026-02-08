"""
Step 6: Compute hitter stats against each pitcher cluster.

Tags every pitch with its pitcher's cluster, then aggregates PA-level
and pitch-level outcomes per batter × cluster × year.
"""

import os
import sys
import gc
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEASONS, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
)


def compute_pa_stats(pa_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate plate-appearance-level stats per batter × cluster × year."""
    pa_data = pa_data.copy()

    pa_data["is_hit"] = pa_data["events"].isin(
        ["single", "double", "triple", "home_run"]
    ).astype(int)
    pa_data["is_single"] = (pa_data["events"] == "single").astype(int)
    pa_data["is_double"] = (pa_data["events"] == "double").astype(int)
    pa_data["is_triple"] = (pa_data["events"] == "triple").astype(int)
    pa_data["is_hr"] = (pa_data["events"] == "home_run").astype(int)
    pa_data["is_bb"] = pa_data["events"].isin(["walk", "intent_walk"]).astype(int)
    pa_data["is_k"] = pa_data["events"].isin(
        ["strikeout", "strikeout_double_play"]
    ).astype(int)
    pa_data["is_hbp"] = (pa_data["events"] == "hit_by_pitch").astype(int)
    pa_data["is_ab"] = (~pa_data["events"].isin([
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "sac_fly_double_play", "catcher_interf", "intent_walk",
    ])).astype(int)

    agg = pa_data.groupby(["batter", "game_year", "cluster"]).agg(
        PA=("events", "count"),
        AB=("is_ab", "sum"),
        H=("is_hit", "sum"),
        singles=("is_single", "sum"),
        doubles=("is_double", "sum"),
        triples=("is_triple", "sum"),
        HR=("is_hr", "sum"),
        BB=("is_bb", "sum"),
        K=("is_k", "sum"),
        HBP=("is_hbp", "sum"),
        woba_sum=("woba_value", "sum"),
        woba_denom_sum=("woba_denom", "sum"),
    ).reset_index()

    # Derived stats
    agg["BA"] = agg["H"] / agg["AB"].clip(lower=1)
    agg["OBP"] = (agg["H"] + agg["BB"] + agg["HBP"]) / agg["PA"].clip(lower=1)
    agg["SLG"] = (
        agg["singles"] + 2 * agg["doubles"] + 3 * agg["triples"] + 4 * agg["HR"]
    ) / agg["AB"].clip(lower=1)
    agg["K_pct"] = agg["K"] / agg["PA"].clip(lower=1)
    agg["BB_pct"] = agg["BB"] / agg["PA"].clip(lower=1)
    agg["wOBA"] = agg["woba_sum"] / agg["woba_denom_sum"].clip(lower=1)

    return agg


def compute_pitch_level_stats(pitch_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level whiff data per batter × cluster × year."""
    pitch_data = pitch_data.copy()
    pitch_data["is_swing"] = pitch_data["description"].isin(SWING_DESCRIPTIONS).astype(int)
    pitch_data["is_whiff"] = pitch_data["description"].isin(WHIFF_DESCRIPTIONS).astype(int)

    agg = pitch_data.groupby(["batter", "game_year", "cluster"]).agg(
        pitches_seen=("pitch_type", "count"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
    ).reset_index()

    agg["whiff_rate_vs"] = agg["whiffs"] / agg["swings"].clip(lower=1)
    return agg


def get_batter_names(seasons: list) -> pd.DataFrame:
    """Extract batter names from raw data. Uses the most common name per batter ID."""
    # Statcast has batter IDs but batter names are not always present
    # We'll use the player_name field from pitches where the batter is batting
    # Actually, Statcast rows have pitcher name in player_name. Batter names need reverse lookup.
    # For now, collect from pybaseball reverse lookup.
    names = {}
    for year in seasons:
        path = os.path.join(RAW_DATA_DIR, f"statcast_{year}.parquet")
        if not os.path.exists(path):
            continue
        # Statcast doesn't have batter name directly in a clean way.
        # We'll do a reverse lookup in bulk at the end.
    return pd.DataFrame()  # Handled below via pybaseball


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Load cluster assignments
    pitcher_clusters = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet"),
        columns=["pitcher", "game_year", "cluster"],
    )
    print(f"Pitcher cluster assignments: {len(pitcher_clusters):,}")

    all_pa_stats = []
    all_pitch_stats = []

    for year in SEASONS:
        path = os.path.join(RAW_DATA_DIR, f"statcast_{year}.parquet")
        if not os.path.exists(path):
            print(f"  [{year}] No data — skipping")
            continue

        print(f"\n  Processing {year}...")
        raw = pd.read_parquet(path)

        if "game_year" not in raw.columns:
            raw["game_year"] = year

        # Tag pitches with cluster
        raw = raw.merge(pitcher_clusters, on=["pitcher", "game_year"], how="inner")
        print(f"    {len(raw):,} pitches tagged with clusters")

        if len(raw) == 0:
            continue

        # PA-level stats (final pitch of each PA = where events is not null)
        pa_data = raw[raw["events"].notna()].copy()
        if len(pa_data) > 0:
            pa_stats = compute_pa_stats(pa_data)
            all_pa_stats.append(pa_stats)
            print(f"    PA stats: {len(pa_stats):,} batter-cluster-year rows")

        # Pitch-level stats
        pitch_stats = compute_pitch_level_stats(raw)
        all_pitch_stats.append(pitch_stats)
        print(f"    Pitch stats: {len(pitch_stats):,} batter-cluster-year rows")

        del raw
        gc.collect()

    if not all_pa_stats:
        print("No PA data processed!")
        return

    hitter_pa = pd.concat(all_pa_stats, ignore_index=True)
    hitter_pitch = pd.concat(all_pitch_stats, ignore_index=True)

    # Merge pitch-level whiff data into PA stats
    hitter_final = hitter_pa.merge(
        hitter_pitch[["batter", "game_year", "cluster", "pitches_seen", "whiff_rate_vs"]],
        on=["batter", "game_year", "cluster"],
        how="left",
    )

    # Get batter names via reverse lookup
    print("\nLooking up batter names...")
    try:
        from pybaseball import playerid_reverse_lookup
        unique_batters = hitter_final["batter"].unique().tolist()

        # Batch lookup
        batch_size = 500
        name_dfs = []
        for i in range(0, len(unique_batters), batch_size):
            batch = unique_batters[i:i + batch_size]
            try:
                result = playerid_reverse_lookup(batch, key_type="mlbam")
                name_dfs.append(result)
            except Exception as e:
                print(f"  Name lookup batch {i} failed: {e}")

        if name_dfs:
            names = pd.concat(name_dfs, ignore_index=True)
            names["batter_name"] = names["name_first"] + " " + names["name_last"]
            hitter_final = hitter_final.merge(
                names[["key_mlbam", "batter_name"]],
                left_on="batter",
                right_on="key_mlbam",
                how="left",
            )
            hitter_final.drop(columns=["key_mlbam"], errors="ignore", inplace=True)
        else:
            hitter_final["batter_name"] = "Unknown"
    except Exception as e:
        print(f"  Reverse lookup failed: {e}")
        hitter_final["batter_name"] = "Unknown"

    # Fill missing names
    hitter_final["batter_name"] = hitter_final["batter_name"].fillna("Unknown")

    # Save main output
    out_path = os.path.join(PROCESSED_DATA_DIR, "hitter_vs_cluster.parquet")
    hitter_final.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\nSaved: {out_path} ({len(hitter_final):,} rows)")

    # --- Export JSON for frontend ---
    frontend_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "src", "data",
    )
    os.makedirs(frontend_dir, exist_ok=True)

    # Batter list for autocomplete
    batters_list = (
        hitter_final.groupby(["batter", "batter_name"])
        .agg(total_PA=("PA", "sum"))
        .reset_index()
        .sort_values("total_PA", ascending=False)
    )
    batters_json = batters_list[["batter", "batter_name", "total_PA"]].to_dict(orient="records")
    with open(os.path.join(frontend_dir, "batters.json"), "w") as f:
        json.dump(batters_json, f)
    print(f"Frontend batters list: {len(batters_json):,} batters")

    # Hitter vs cluster data (round floats for smaller JSON)
    export_cols = [
        "batter", "batter_name", "game_year", "cluster",
        "PA", "AB", "H", "HR", "BB", "K", "HBP",
        "BA", "OBP", "SLG", "K_pct", "BB_pct", "wOBA",
        "pitches_seen", "whiff_rate_vs",
        "singles", "doubles", "triples",
    ]
    export_cols = [c for c in export_cols if c in hitter_final.columns]
    export_df = hitter_final[export_cols].copy()

    # Round float columns
    float_cols = export_df.select_dtypes(include=[np.floating]).columns
    export_df[float_cols] = export_df[float_cols].round(4)

    # Save as JSON (split into chunks if too large, but usually ~10-20MB is fine)
    hvc_json = export_df.to_dict(orient="records")
    with open(os.path.join(frontend_dir, "hitter_vs_cluster.json"), "w") as f:
        json.dump(hvc_json, f)
    print(f"Frontend hitter_vs_cluster: {len(hvc_json):,} rows")

    # Pitcher seasons (for pitcher dots on the graph)
    pitcher_seasons_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    if os.path.exists(pitcher_seasons_path):
        ps = pd.read_parquet(pitcher_seasons_path)
        ps_export_cols = [
            "pitcher", "player_name", "game_year", "cluster",
            "pca_x", "pca_y", "pca_z",
            "whiff_rate", "avg_velo_FF", "arm_angle", "is_rhp", "is_sp",
        ]
        ps_export_cols = [c for c in ps_export_cols if c in ps.columns]
        ps_export = ps[ps_export_cols].copy()
        float_cols_ps = ps_export.select_dtypes(include=[np.floating]).columns
        ps_export[float_cols_ps] = ps_export[float_cols_ps].round(4)
        ps_json = ps_export.to_dict(orient="records")
        with open(os.path.join(frontend_dir, "pitcher_seasons.json"), "w") as f:
            json.dump(ps_json, f)
        print(f"Frontend pitcher_seasons: {len(ps_json):,} rows")

    print("\nHitter vs cluster processing complete!")


if __name__ == "__main__":
    main()
