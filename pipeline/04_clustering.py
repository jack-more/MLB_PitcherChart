"""
Step 4: K-Means clustering on pitcher-season feature vectors.

Runs SEPARATE clustering for RHP and LHP pitchers.
- StandardScaler preprocessing (per-hand)
- Silhouette score for optimal K selection (min K=8)
- 3D PCA projection for galaxy view
- RHP clusters offset to the right (+X), LHP to the left (-X)
- Cluster IDs: R_0, R_1, ... for RHP; L_0, L_1, ... for LHP
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, CLUSTER_FEATURES,
    K_RANGE, RANDOM_STATE,
)

# Features to use for clustering — remove is_rhp since we split by hand
HAND_CLUSTER_FEATURES = [f for f in CLUSTER_FEATURES if f != "is_rhp"]

MIN_K = 8   # Minimum clusters per hand
X_OFFSET = 5.0  # How far to shift RHP right / LHP left in PCA space


def find_optimal_k(X_scaled, k_range, min_k=MIN_K):
    """Run KMeans for each K, return optimal K (enforcing minimum)."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"    K={k}  sil={sil:.4f}  inertia={km.inertia_:.0f}")

    df = pd.DataFrame(results)
    practical = df[df["k"] >= min_k]
    if len(practical) > 0:
        best = practical.loc[practical["silhouette"].idxmax()]
    else:
        best = df.loc[df["silhouette"].idxmax()]
    optimal_k = int(best["k"])
    if optimal_k < min_k:
        optimal_k = min_k
    print(f"    -> Optimal K: {optimal_k} (sil={best['silhouette']:.4f})")
    return optimal_k


def cluster_hand(pitcher_seasons, hand_label, prefix, features):
    """Cluster one hand group. Returns (updated df, models dict)."""
    print(f"\n{'='*50}")
    print(f"  Clustering {hand_label} pitchers ({len(pitcher_seasons):,} pitcher-seasons)")
    print(f"{'='*50}")

    X = pitcher_seasons[features].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Find optimal K
    print(f"  Searching for optimal K...")
    optimal_k = find_optimal_k(X_scaled, K_RANGE, MIN_K)

    # Fit final KMeans
    print(f"  Fitting KMeans with K={optimal_k}...")
    km = KMeans(n_clusters=optimal_k, n_init=20, max_iter=500, random_state=RANDOM_STATE)
    local_labels = km.fit_predict(X_scaled)

    # Assign prefixed cluster IDs: R_0, R_1, ... or L_0, L_1, ...
    pitcher_seasons = pitcher_seasons.copy()
    pitcher_seasons["cluster"] = [f"{prefix}_{c}" for c in local_labels]

    # Print sizes
    for cid, count in pitcher_seasons["cluster"].value_counts().sort_index().items():
        print(f"    {cid}: {count:,}")

    # 3D PCA
    print(f"  Computing 3D PCA...")
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_3d = pca.fit_transform(X_scaled)
    print(f"    Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}")

    pitcher_seasons["pca_x_raw"] = X_3d[:, 0]
    pitcher_seasons["pca_y"] = X_3d[:, 1]
    pitcher_seasons["pca_z"] = X_3d[:, 2]

    # Offset X: RHP to the right (+), LHP to the left (-)
    if prefix == "R":
        pitcher_seasons["pca_x"] = pitcher_seasons["pca_x_raw"] + X_OFFSET
    else:
        pitcher_seasons["pca_x"] = -pitcher_seasons["pca_x_raw"] - X_OFFSET

    models = {
        "scaler": scaler,
        "kmeans": km,
        "pca": pca,
        "optimal_k": optimal_k,
    }

    return pitcher_seasons, models


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load pitcher-season features
    data_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    pitcher_seasons = pd.read_parquet(data_path)
    print(f"Loaded {len(pitcher_seasons):,} pitcher-seasons")

    # Fill missing clustering features
    for f in HAND_CLUSTER_FEATURES:
        if f not in pitcher_seasons.columns:
            pitcher_seasons[f] = 0.0

    # Split by handedness
    rhp = pitcher_seasons[pitcher_seasons["is_rhp"] == 1].copy()
    lhp = pitcher_seasons[pitcher_seasons["is_rhp"] == 0].copy()
    print(f"\nRHP: {len(rhp):,}  |  LHP: {len(lhp):,}")

    # Cluster each hand independently
    rhp_out, rhp_models = cluster_hand(
        rhp, "RHP", "R", HAND_CLUSTER_FEATURES
    )
    lhp_out, lhp_models = cluster_hand(
        lhp, "LHP", "L", HAND_CLUSTER_FEATURES
    )

    # Merge back
    all_pitchers = pd.concat([rhp_out, lhp_out], ignore_index=True)

    # Save models
    for prefix, models in [("R", rhp_models), ("L", lhp_models)]:
        joblib.dump(models["scaler"], os.path.join(MODELS_DIR, f"scaler_{prefix}.joblib"))
        joblib.dump(models["kmeans"], os.path.join(MODELS_DIR, f"kmeans_{prefix}.joblib"))
        joblib.dump(models["pca"], os.path.join(MODELS_DIR, f"pca_{prefix}.joblib"))
    print("\nModels saved.")

    # Drop temp column and save
    all_pitchers.drop(columns=["pca_x_raw"], inplace=True, errors="ignore")
    all_pitchers.to_parquet(data_path, engine="pyarrow", compression="snappy")
    print(f"\nUpdated pitcher_seasons saved: {data_path}")
    print(f"  Total: {len(all_pitchers):,} pitcher-seasons")
    print(f"  RHP clusters: {rhp_models['optimal_k']}")
    print(f"  LHP clusters: {lhp_models['optimal_k']}")

    # Save meta
    meta = {
        "rhp_k": rhp_models["optimal_k"],
        "lhp_k": lhp_models["optimal_k"],
        "total_clusters": rhp_models["optimal_k"] + lhp_models["optimal_k"],
        "features": HAND_CLUSTER_FEATURES,
        "x_offset": X_OFFSET,
    }
    with open(os.path.join(MODELS_DIR, "kmeans_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # Assign sub-threshold pitchers to nearest cluster
    # ═══════════════════════════════════════════════════════════════
    sub_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons_sub_threshold.parquet")
    if os.path.exists(sub_path):
        print(f"\n{'='*50}")
        print("Assigning sub-threshold pitchers to nearest clusters...")
        print(f"{'='*50}")

        sub = pd.read_parquet(sub_path)

        # Fill missing features
        for f in HAND_CLUSTER_FEATURES:
            if f not in sub.columns:
                sub[f] = 0.0

        # Split by hand
        sub_rhp = sub[sub["is_rhp"] == 1].copy() if "is_rhp" in sub.columns else pd.DataFrame()
        sub_lhp = sub[sub["is_rhp"] == 0].copy() if "is_rhp" in sub.columns else pd.DataFrame()

        sub_parts = []
        for prefix, models, sub_hand, label in [
            ("R", rhp_models, sub_rhp, "RHP"),
            ("L", lhp_models, sub_lhp, "LHP"),
        ]:
            if len(sub_hand) == 0:
                continue

            X_sub = sub_hand[HAND_CLUSTER_FEATURES].fillna(0).values
            X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale using the fitted scaler
            X_sub_scaled = models["scaler"].transform(X_sub)
            X_sub_scaled = np.clip(X_sub_scaled, -10, 10)
            X_sub_scaled = np.nan_to_num(X_sub_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict nearest cluster
            local_labels = models["kmeans"].predict(X_sub_scaled)
            sub_hand["cluster"] = [f"{prefix}_{c}" for c in local_labels]

            # PCA coordinates
            X_3d = models["pca"].transform(X_sub_scaled)
            sub_hand["pca_y"] = X_3d[:, 1]
            sub_hand["pca_z"] = X_3d[:, 2]
            if prefix == "R":
                sub_hand["pca_x"] = X_3d[:, 0] + X_OFFSET
            else:
                sub_hand["pca_x"] = -X_3d[:, 0] - X_OFFSET

            sub_parts.append(sub_hand)
            print(f"  Assigned {len(sub_hand):,} sub-threshold {label} pitchers")

        if sub_parts:
            sub_assigned = pd.concat(sub_parts, ignore_index=True)
            # Combine with qualified pitchers and save
            all_pitchers = pd.concat([all_pitchers, sub_assigned], ignore_index=True)
            all_pitchers.to_parquet(data_path, engine="pyarrow", compression="snappy")
            print(f"\n  Updated pitcher_seasons with sub-threshold assignments")
            print(f"  New total: {len(all_pitchers):,} pitcher-seasons")
    else:
        print("\nNo sub-threshold pitcher file found — skipping nearest-cluster assignment.")

    print("\nClustering complete!")


if __name__ == "__main__":
    main()
