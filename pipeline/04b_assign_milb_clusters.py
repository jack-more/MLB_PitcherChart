#!/usr/bin/env python3
"""
Step 4b: Assign MiLB pitchers to MLB archetypes using saved GMM models.

Loads the trained MLB scaler + GMM + PCA models, runs MiLB pitcher features
through them to assign cluster IDs (e.g., R_3 = "Yakker"). Does NOT retrain
any models — MiLB data never contaminates the MLB clustering.

After assignment, merges MiLB pitcher-seasons into the main pitcher_seasons.parquet
with source="milb" so the daily matchup system can look them up.

Usage:
    python pipeline/04b_assign_milb_clusters.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, PROCESSED_DATA_DIR, CLUSTER_FEATURES, RANDOM_STATE,
)

# Features used for per-hand clustering (same as 04_clustering.py)
HAND_CLUSTER_FEATURES = [f for f in CLUSTER_FEATURES if f != "is_rhp"]

# X-offset for PCA visualization (separate RHP and LHP in galaxy view)
X_OFFSET = 6.0


def main():
    milb_path = os.path.join(PROCESSED_DATA_DIR, "milb_pitcher_seasons.parquet")
    if not os.path.exists(milb_path):
        print("ERROR: milb_pitcher_seasons.parquet not found. Run 03b_milb_features.py first.")
        return

    milb = pd.read_parquet(milb_path)
    print(f"Loaded {len(milb):,} MiLB pitcher-seasons")

    if len(milb) == 0:
        print("No MiLB pitchers to assign.")
        return

    # Ensure all cluster features exist
    for f in HAND_CLUSTER_FEATURES:
        if f not in milb.columns:
            milb[f] = 0.0

    # Load cluster meta for post-hoc rules
    meta_path = os.path.join(MODELS_DIR, "cluster_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    posthoc_rules = meta.get("posthoc_rules", {})

    # Split by hand
    milb_rhp = milb[milb["is_rhp"] == 1].copy() if "is_rhp" in milb.columns else pd.DataFrame()
    milb_lhp = milb[milb["is_rhp"] == 0].copy() if "is_rhp" in milb.columns else pd.DataFrame()

    parts = []
    for prefix, sub_hand, label in [
        ("R", milb_rhp, "RHP"),
        ("L", milb_lhp, "LHP"),
    ]:
        if len(sub_hand) == 0:
            continue

        # Load MLB models
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{prefix}.joblib")
        gmm_path = os.path.join(MODELS_DIR, f"gmm_{prefix}.joblib")
        pca_path = os.path.join(MODELS_DIR, f"pca_{prefix}.joblib")

        if not all(os.path.exists(p) for p in [scaler_path, gmm_path, pca_path]):
            print(f"  WARNING: Missing models for {label} — skipping")
            continue

        scaler = joblib.load(scaler_path)
        gmm = joblib.load(gmm_path)
        pca = joblib.load(pca_path)

        print(f"\n  Assigning {len(sub_hand):,} MiLB {label} pitchers...")

        # Prepare feature matrix
        X = sub_hand[HAND_CLUSTER_FEATURES].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale using MLB scaler (same normalization as MLB pitchers)
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict cluster (hard assignment)
        local_labels = gmm.predict(X_scaled)
        sub_hand["cluster"] = [f"{prefix}_{c}" for c in local_labels]

        # PCA coordinates for galaxy view
        X_3d = pca.transform(X_scaled)
        sub_hand["pca_y"] = X_3d[:, 1]
        sub_hand["pca_z"] = X_3d[:, 2]
        if prefix == "R":
            sub_hand["pca_x"] = X_3d[:, 0] + X_OFFSET
        else:
            sub_hand["pca_x"] = -X_3d[:, 0] - X_OFFSET

        # Apply post-hoc rules (e.g., Undertow extraction)
        for rule_suffix, rule in posthoc_rules.items():
            conditions = rule.get("conditions", {})
            new_cid = f"{prefix}_{rule_suffix}"

            # Check if this post-hoc cluster exists for this hand
            posthoc_ids = meta.get(f"{prefix.lower()}hp_cluster_ids", [])
            if new_cid not in posthoc_ids:
                continue

            mask = sub_hand["cluster"].str.startswith(prefix + "_")
            for feat, (op, thresh) in conditions.items():
                if feat in sub_hand.columns:
                    if op == "<":
                        mask = mask & (sub_hand[feat] < thresh)
                    elif op == ">":
                        mask = mask & (sub_hand[feat] > thresh)
                    elif op == ">=":
                        mask = mask & (sub_hand[feat] >= thresh)
                    elif op == "<=":
                        mask = mask & (sub_hand[feat] <= thresh)
            n_moved = mask.sum()
            if n_moved > 0:
                sub_hand.loc[mask, "cluster"] = new_cid
                print(f"    {new_cid}: extracted {n_moved} MiLB pitchers (post-hoc)")

        parts.append(sub_hand)

        # Print assignment summary
        cluster_counts = sub_hand["cluster"].value_counts()
        print(f"    Cluster distribution:")
        for cid, cnt in cluster_counts.head(10).items():
            print(f"      {cid}: {cnt}")
        if len(cluster_counts) > 10:
            print(f"      ... and {len(cluster_counts) - 10} more clusters")

    if not parts:
        print("No MiLB pitchers assigned.")
        return

    milb_assigned = pd.concat(parts, ignore_index=True)
    milb_assigned["source"] = "milb"

    # Apply cluster merges (same as MLB pipeline)
    from config import CLUSTER_MERGES
    if CLUSTER_MERGES:
        for old_cid, new_cid in CLUSTER_MERGES.items():
            n_moved = (milb_assigned["cluster"] == old_cid).sum()
            if n_moved > 0:
                milb_assigned.loc[milb_assigned["cluster"] == old_cid, "cluster"] = new_cid
                print(f"  Merge: {old_cid} -> {new_cid} ({n_moved} MiLB pitchers)")

    # Save standalone MiLB file
    milb_out = os.path.join(PROCESSED_DATA_DIR, "milb_pitcher_seasons.parquet")
    milb_assigned.to_parquet(milb_out, engine="pyarrow", compression="snappy")
    print(f"\nSaved MiLB assignments: {milb_out} ({len(milb_assigned):,} pitcher-seasons)")

    # Now merge into the main pitcher_seasons.parquet
    main_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    if os.path.exists(main_path):
        main_df = pd.read_parquet(main_path)

        # Add source column to MLB pitchers if not present
        if "source" not in main_df.columns:
            main_df["source"] = "mlb"

        # Remove any previously merged MiLB entries
        main_df = main_df[main_df["source"] != "milb"].copy()

        # Ensure matching columns
        common_cols = list(set(main_df.columns) & set(milb_assigned.columns))
        milb_to_merge = milb_assigned[common_cols].copy()

        # Append MiLB pitchers
        combined = pd.concat([main_df, milb_to_merge], ignore_index=True)
        combined.to_parquet(main_path, engine="pyarrow", compression="snappy")

        n_mlb = (combined["source"] == "mlb").sum()
        n_milb = (combined["source"] == "milb").sum()
        print(f"\nUpdated pitcher_seasons.parquet:")
        print(f"  MLB pitchers: {n_mlb:,}")
        print(f"  MiLB pitchers: {n_milb:,}")
        print(f"  Total: {len(combined):,}")
    else:
        print(f"\nWARNING: {main_path} not found — saving MiLB data standalone only")

    print("\nMiLB cluster assignment complete!")


if __name__ == "__main__":
    main()
