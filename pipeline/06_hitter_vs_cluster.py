"""
Step 6: Compute hitter stats against each pitcher cluster (GMM-weighted).

Uses soft cluster probabilities from GMM: each PA is fractionally distributed
across all archetypes based on the pitcher's probability vector.

Example: If a pitcher is 82% Barnburner / 14% Heavy Duty / 4% CutCraft,
a single against that pitcher counts as 0.82 singles vs Barnburner,
0.14 singles vs Heavy Duty, 0.04 singles vs CutCraft.
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

# Only distribute PA weight to clusters with at least this much probability
PROB_THRESHOLD = 0.01


def _tag_pa_outcomes(pa_data: pd.DataFrame) -> pd.DataFrame:
    """Add binary outcome columns to PA-level data (modifies in place for speed)."""
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
    return pa_data


def _tag_pitch_outcomes(pitch_data: pd.DataFrame) -> pd.DataFrame:
    """Add swing/whiff columns to pitch-level data (modifies in place for speed)."""
    pitch_data["is_swing"] = pitch_data["description"].isin(SWING_DESCRIPTIONS).astype(int)
    pitch_data["is_whiff"] = pitch_data["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    return pitch_data


def expand_weighted(
    data: pd.DataFrame,
    proba_lookup: dict,
    cluster_cols: list,
) -> pd.DataFrame:
    """Expand each row across clusters using pitcher probability weights.

    For each row, looks up the pitcher's probability vector and creates one row
    per cluster (above PROB_THRESHOLD) with a 'weight' column.

    proba_lookup: dict mapping (pitcher, game_year) -> dict of {cluster_id: probability}
    cluster_cols: list of all cluster IDs
    """
    pitcher_ids = data["pitcher"].astype(int).values
    game_years = data["game_year"].astype(int).values

    # Pre-compute unique pitcher-year combos in this data for speed
    unique_keys = set(zip(pitcher_ids, game_years))
    # Filter lookup to only relevant keys
    relevant_lookup = {k: v for k, v in proba_lookup.items() if k in unique_keys}

    # Build expansion arrays
    row_indices = []
    clusters = []
    weights = []

    for i in range(len(data)):
        key = (pitcher_ids[i], game_years[i])
        probs = relevant_lookup.get(key)
        if probs is None:
            continue
        for cid, prob in probs.items():
            row_indices.append(i)
            clusters.append(cid)
            weights.append(prob)

    if not row_indices:
        return pd.DataFrame()

    # Expand data using fancy indexing
    expanded = data.iloc[row_indices].copy()
    expanded = expanded.reset_index(drop=True)
    expanded["cluster"] = clusters
    expanded["weight"] = np.array(weights, dtype=np.float32)

    return expanded


def compute_weighted_pa_stats(pa_expanded: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted PA stats per batter × cluster × year."""
    # Multiply all outcome columns by weight
    outcome_cols = [
        "is_hit", "is_single", "is_double", "is_triple", "is_hr",
        "is_bb", "is_k", "is_hbp", "is_ab",
    ]
    for col in outcome_cols:
        pa_expanded[f"w_{col}"] = pa_expanded[col] * pa_expanded["weight"]

    pa_expanded["w_woba_value"] = pa_expanded["woba_value"].fillna(0) * pa_expanded["weight"]
    pa_expanded["w_woba_denom"] = pa_expanded["woba_denom"].fillna(0) * pa_expanded["weight"]

    agg = pa_expanded.groupby(["batter", "game_year", "cluster", "stand"]).agg(
        PA=("weight", "sum"),
        AB=("w_is_ab", "sum"),
        H=("w_is_hit", "sum"),
        singles=("w_is_single", "sum"),
        doubles=("w_is_double", "sum"),
        triples=("w_is_triple", "sum"),
        HR=("w_is_hr", "sum"),
        BB=("w_is_bb", "sum"),
        K=("w_is_k", "sum"),
        HBP=("w_is_hbp", "sum"),
        woba_sum=("w_woba_value", "sum"),
        woba_denom_sum=("w_woba_denom", "sum"),
    ).reset_index()

    # Derived stats
    agg["BA"] = agg["H"] / agg["AB"].clip(lower=0.01)
    agg["OBP"] = (agg["H"] + agg["BB"] + agg["HBP"]) / agg["PA"].clip(lower=0.01)
    agg["SLG"] = (
        agg["singles"] + 2 * agg["doubles"] + 3 * agg["triples"] + 4 * agg["HR"]
    ) / agg["AB"].clip(lower=0.01)
    agg["K_pct"] = agg["K"] / agg["PA"].clip(lower=0.01)
    agg["BB_pct"] = agg["BB"] / agg["PA"].clip(lower=0.01)
    agg["wOBA"] = agg["woba_sum"] / agg["woba_denom_sum"].clip(lower=0.01)

    return agg


def compute_weighted_pitch_stats(pitch_expanded: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted pitch-level whiff data per batter × cluster × year."""
    pitch_expanded["w_swing"] = pitch_expanded["is_swing"] * pitch_expanded["weight"]
    pitch_expanded["w_whiff"] = pitch_expanded["is_whiff"] * pitch_expanded["weight"]

    agg = pitch_expanded.groupby(["batter", "game_year", "cluster", "stand"]).agg(
        pitches_seen=("weight", "sum"),
        swings=("w_swing", "sum"),
        whiffs=("w_whiff", "sum"),
    ).reset_index()

    agg["whiff_rate_vs"] = agg["whiffs"] / agg["swings"].clip(lower=0.01)
    return agg


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Load cluster probability vectors
    proba_path = os.path.join(PROCESSED_DATA_DIR, "cluster_probabilities.parquet")
    if not os.path.exists(proba_path):
        print("ERROR: cluster_probabilities.parquet not found!")
        print("  Run 04_clustering.py first (now uses GMM with soft probabilities).")
        return

    proba_df = pd.read_parquet(proba_path)
    print(f"Loaded cluster probabilities: {len(proba_df):,} pitcher-seasons")

    # Identify cluster columns (everything except pitcher and game_year)
    cluster_cols = [c for c in proba_df.columns if c not in ("pitcher", "game_year")]
    print(f"  Cluster columns: {len(cluster_cols)} ({', '.join(cluster_cols[:5])}...)")

    # Build fast lookup: (pitcher, game_year) -> {cluster_id: probability}
    # Only keep probabilities above threshold
    proba_lookup = {}
    for _, row in proba_df.iterrows():
        key = (int(row["pitcher"]), int(row["game_year"]))
        probs = {c: row[c] for c in cluster_cols if row[c] >= PROB_THRESHOLD}
        if probs:
            proba_lookup[key] = probs
    print(f"  Probability lookup: {len(proba_lookup):,} pitcher-seasons")

    # Also load hard assignments for pitcher matching (so we only process
    # pitches from pitchers that have cluster assignments)
    pitcher_ids_set = set(proba_lookup.keys())

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

        # Filter to pitchers that have cluster assignments (vectorized)
        raw["_pitcher_year"] = list(zip(
            raw["pitcher"].astype(int).values,
            raw["game_year"].astype(int).values,
        ))
        raw = raw[raw["_pitcher_year"].isin(pitcher_ids_set)].copy()
        raw.drop(columns=["_pitcher_year"], inplace=True)
        print(f"    {len(raw):,} pitches from clustered pitchers")

        if len(raw) == 0:
            continue

        # ----- PA-level stats (weighted) -----
        pa_data = raw[raw["events"].notna()].copy()
        if len(pa_data) > 0:
            _tag_pa_outcomes(pa_data)
            pa_expanded = expand_weighted(pa_data, proba_lookup, cluster_cols)
            if len(pa_expanded) > 0:
                pa_stats = compute_weighted_pa_stats(pa_expanded)
                all_pa_stats.append(pa_stats)
                print(f"    PA stats: {len(pa_stats):,} weighted batter-cluster-year rows")
                del pa_expanded
            del pa_data

        # ----- Pitch-level stats (weighted) -----
        _tag_pitch_outcomes(raw)
        pitch_expanded = expand_weighted(raw, proba_lookup, cluster_cols)
        if len(pitch_expanded) > 0:
            pitch_stats = compute_weighted_pitch_stats(pitch_expanded)
            all_pitch_stats.append(pitch_stats)
            print(f"    Pitch stats: {len(pitch_stats):,} weighted batter-cluster-year rows")
            del pitch_expanded

        del raw
        gc.collect()

    if not all_pa_stats:
        print("No PA data processed!")
        return

    hitter_pa = pd.concat(all_pa_stats, ignore_index=True)
    hitter_pitch = pd.concat(all_pitch_stats, ignore_index=True)

    # Re-aggregate across years (in case same batter-cluster-year came from multiple chunks)
    hitter_pa = hitter_pa.groupby(["batter", "game_year", "cluster", "stand"]).agg({
        "PA": "sum", "AB": "sum", "H": "sum", "singles": "sum", "doubles": "sum",
        "triples": "sum", "HR": "sum", "BB": "sum", "K": "sum", "HBP": "sum",
        "woba_sum": "sum", "woba_denom_sum": "sum",
    }).reset_index()
    # Recalculate derived stats after re-aggregation
    hitter_pa["BA"] = hitter_pa["H"] / hitter_pa["AB"].clip(lower=0.01)
    hitter_pa["OBP"] = (hitter_pa["H"] + hitter_pa["BB"] + hitter_pa["HBP"]) / hitter_pa["PA"].clip(lower=0.01)
    hitter_pa["SLG"] = (
        hitter_pa["singles"] + 2 * hitter_pa["doubles"] + 3 * hitter_pa["triples"] + 4 * hitter_pa["HR"]
    ) / hitter_pa["AB"].clip(lower=0.01)
    hitter_pa["K_pct"] = hitter_pa["K"] / hitter_pa["PA"].clip(lower=0.01)
    hitter_pa["BB_pct"] = hitter_pa["BB"] / hitter_pa["PA"].clip(lower=0.01)
    hitter_pa["wOBA"] = hitter_pa["woba_sum"] / hitter_pa["woba_denom_sum"].clip(lower=0.01)

    hitter_pitch = hitter_pitch.groupby(["batter", "game_year", "cluster", "stand"]).agg({
        "pitches_seen": "sum", "swings": "sum", "whiffs": "sum",
    }).reset_index()
    hitter_pitch["whiff_rate_vs"] = hitter_pitch["whiffs"] / hitter_pitch["swings"].clip(lower=0.01)

    # Merge pitch-level whiff data into PA stats
    hitter_final = hitter_pa.merge(
        hitter_pitch[["batter", "game_year", "cluster", "stand", "pitches_seen", "whiff_rate_vs"]],
        on=["batter", "game_year", "cluster", "stand"],
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
    batters_list["total_PA"] = batters_list["total_PA"].round(1)
    batters_json = batters_list[["batter", "batter_name", "total_PA"]].to_dict(orient="records")
    with open(os.path.join(frontend_dir, "batters.json"), "w") as f:
        json.dump(batters_json, f)
    print(f"Frontend batters list: {len(batters_json):,} batters")

    # Hitter vs cluster data (round floats for smaller JSON)
    export_cols = [
        "batter", "batter_name", "game_year", "cluster", "stand",
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

    # NOTE: pitcher_seasons.json is now exported by export_frontend.py (with GMM probas)
    # Only copy batters.json and hitter_vs_cluster.json here

    # Also export to frontend/public/ for static serving
    public_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "public",
    )
    if os.path.isdir(public_dir):
        import shutil
        for fname in ["batters.json", "hitter_vs_cluster.json"]:
            src = os.path.join(frontend_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(public_dir, fname))
        print(f"Copied JSON exports to {public_dir}")

    print("\nHitter vs cluster processing complete!")


if __name__ == "__main__":
    main()
