"""
Step 4: GMM (Gaussian Mixture Model) clustering on pitcher-season feature vectors.

Runs SEPARATE clustering for RHP and LHP pitchers.
- StandardScaler preprocessing (per-hand)
- BIC for optimal K selection (min K=8)
- Soft probability vectors per pitcher-season (stored separately)
- Hard cluster assignment via argmax for backward compatibility
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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, CLUSTER_FEATURES,
    K_RANGE, RANDOM_STATE,
)

# Features to use for clustering — remove is_rhp since we split by hand
HAND_CLUSTER_FEATURES = [f for f in CLUSTER_FEATURES if f != "is_rhp"]

MIN_K = 10  # Minimum clusters per hand
X_OFFSET = 5.0  # How far to shift RHP right / LHP left in PCA space

# Force specific K per hand (set to None to auto-select via BIC)
FORCE_K = {"R": None, "L": None}

# Feature weights applied AFTER StandardScaler (1.0 = no boost).
FEATURE_WEIGHTS = {}

# Post-hoc archetype extraction rules.
# After GMM clustering, pitcher-seasons matching these rules are split
# into dedicated archetypes regardless of their GMM assignment.
# Format: { new_cluster_suffix: { feature: (op, threshold), ... }, ... }
# A pitcher-season must satisfy ALL conditions to be extracted.
# min_size: minimum pitcher-seasons required to form the cluster (else stays in GMM cluster).
POSTHOC_RULES = {
    "UT": {  # Undertow — sinker-dominant with <10% four-seam, SI>40%
        "conditions": {"pct_FF": ("<", 0.10), "pct_SI": (">", 0.40)},
        "min_size": 8,
    },
}


def find_optimal_k(X_scaled, k_range, min_k=MIN_K):
    """Run GMM for each K, return optimal K using BIC (lower is better), enforcing minimum."""
    results = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, n_init=5, max_iter=300,
            covariance_type="full", random_state=RANDOM_STATE,
        )
        labels = gmm.fit_predict(X_scaled)
        bic = gmm.bic(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else -1
        results.append({"k": k, "bic": bic, "silhouette": sil})
        print(f"    K={k}  BIC={bic:.0f}  sil={sil:.4f}")

    df = pd.DataFrame(results)
    practical = df[df["k"] >= min_k]
    if len(practical) > 0:
        # BIC: lower is better
        best = practical.loc[practical["bic"].idxmin()]
    else:
        best = df.loc[df["bic"].idxmin()]
    optimal_k = int(best["k"])
    if optimal_k < min_k:
        optimal_k = min_k
    print(f"    -> Optimal K: {optimal_k} (BIC={best['bic']:.0f}, sil={best['silhouette']:.4f})")
    return optimal_k


def cluster_hand(pitcher_seasons, hand_label, prefix, features):
    """Cluster one hand group using GMM. Returns (updated df, proba_df, models dict).

    proba_df has columns: pitcher, game_year, and one column per cluster ID
    (e.g. R_0, R_1, ...) with the soft probability for that cluster.
    """
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

    # Apply feature weights (boost key pitch-type columns so GMM separates them)
    for feat_name, weight in FEATURE_WEIGHTS.items():
        if feat_name in features:
            idx = features.index(feat_name)
            X_scaled[:, idx] *= weight
            print(f"    Feature weight: {feat_name} x{weight}")

    # Find optimal K (or use forced K if set)
    forced = FORCE_K.get(prefix)
    if forced:
        print(f"  Using forced K={forced} for {hand_label}")
        optimal_k = forced
    else:
        print(f"  Searching for optimal K...")
        optimal_k = find_optimal_k(X_scaled, K_RANGE, MIN_K)

    # Fit final GMM (high n_init for stable convergence to global minimum)
    print(f"  Fitting GMM with K={optimal_k}, n_init=50...")
    gmm = GaussianMixture(
        n_components=optimal_k, n_init=50, max_iter=1000,
        covariance_type="full", random_state=RANDOM_STATE,
    )
    local_labels = gmm.fit_predict(X_scaled)

    # Get soft probability vectors
    proba = gmm.predict_proba(X_scaled)  # shape: (n_samples, optimal_k)
    cluster_ids = [f"{prefix}_{c}" for c in range(optimal_k)]

    # Build probability DataFrame
    pitcher_seasons = pitcher_seasons.copy()
    proba_df = pd.DataFrame(proba, columns=cluster_ids, index=pitcher_seasons.index)
    proba_df["pitcher"] = pitcher_seasons["pitcher"].values
    proba_df["game_year"] = pitcher_seasons["game_year"].values

    # Hard assignment via argmax (backward-compatible)
    pitcher_seasons["cluster"] = [f"{prefix}_{c}" for c in local_labels]

    # Print sizes and avg max-probability (how "crisp" assignments are)
    avg_max_prob = proba.max(axis=1).mean()
    print(f"  Avg max-cluster probability: {avg_max_prob:.3f} (1.0 = perfectly crisp)")
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
        "gmm": gmm,
        "pca": pca,
        "optimal_k": optimal_k,
        "cluster_ids": cluster_ids,
    }

    return pitcher_seasons, proba_df, models


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load pitcher-season features
    data_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    pitcher_seasons = pd.read_parquet(data_path)

    # Deduplicate in case previous runs appended sub-threshold pitchers
    before = len(pitcher_seasons)
    pitcher_seasons = pitcher_seasons.drop_duplicates(subset=["pitcher", "game_year"], keep="first")
    if len(pitcher_seasons) < before:
        print(f"  Deduped: {before:,} -> {len(pitcher_seasons):,}")

    # Only cluster qualified pitcher-seasons (MIN_PITCHES threshold)
    from config import MIN_PITCHES
    pitcher_seasons = pitcher_seasons[pitcher_seasons["total_pitches"] >= MIN_PITCHES].copy()
    print(f"Loaded {len(pitcher_seasons):,} qualified pitcher-seasons (>={MIN_PITCHES} pitches)")

    # Clean any stale clustering columns from previous runs
    for col in ["cluster", "pca_x", "pca_y", "pca_z", "pca_x_raw"]:
        if col in pitcher_seasons.columns:
            pitcher_seasons.drop(columns=[col], inplace=True)

    # Fill missing clustering features
    for f in HAND_CLUSTER_FEATURES:
        if f not in pitcher_seasons.columns:
            pitcher_seasons[f] = 0.0

    # Split by handedness
    rhp = pitcher_seasons[pitcher_seasons["is_rhp"] == 1].copy()
    lhp = pitcher_seasons[pitcher_seasons["is_rhp"] == 0].copy()
    print(f"\nRHP: {len(rhp):,}  |  LHP: {len(lhp):,}")

    # Cluster each hand independently
    rhp_out, rhp_proba, rhp_models = cluster_hand(
        rhp, "RHP", "R", HAND_CLUSTER_FEATURES
    )
    lhp_out, lhp_proba, lhp_models = cluster_hand(
        lhp, "LHP", "L", HAND_CLUSTER_FEATURES
    )

    # Merge back
    all_pitchers = pd.concat([rhp_out, lhp_out], ignore_index=True)

    # Merge probability vectors (each hand has its own cluster columns)
    all_proba = pd.concat([rhp_proba, lhp_proba], ignore_index=True)
    # Fill NaN for cross-hand clusters (RHP pitchers have 0 probability for LHP clusters)
    all_proba = all_proba.fillna(0.0)

    # ═══════════════════════════════════════════════════════════════
    # Post-hoc archetype extraction (rule-based splits)
    # ═══════════════════════════════════════════════════════════════
    posthoc_ids = {"R": list(rhp_models["cluster_ids"]), "L": list(lhp_models["cluster_ids"])}
    if POSTHOC_RULES:
        print(f"\n{'='*50}")
        print("  Post-hoc archetype extraction")
        print(f"{'='*50}")
        for rule_suffix, rule in POSTHOC_RULES.items():
            conditions = rule["conditions"]
            min_size = rule["min_size"]
            for prefix, label in [("R", "RHP"), ("L", "LHP")]:
                mask = all_pitchers["cluster"].str.startswith(prefix + "_")
                for feat, (op, thresh) in conditions.items():
                    if feat in all_pitchers.columns:
                        if op == "<":
                            mask = mask & (all_pitchers[feat] < thresh)
                        elif op == ">":
                            mask = mask & (all_pitchers[feat] > thresh)
                        elif op == ">=":
                            mask = mask & (all_pitchers[feat] >= thresh)
                        elif op == "<=":
                            mask = mask & (all_pitchers[feat] <= thresh)

                n_match = mask.sum()
                new_cid = f"{prefix}_{rule_suffix}"
                if n_match >= min_size:
                    all_pitchers.loc[mask, "cluster"] = new_cid
                    posthoc_ids[prefix].append(new_cid)
                    print(f"  {new_cid}: extracted {n_match} pitcher-seasons from {label} ({rule_suffix} rule)")
                else:
                    print(f"  {new_cid}: only {n_match} matches (need {min_size}) — skipped")

    # ═══════════════════════════════════════════════════════════════
    # Merge tiny clusters (<MIN_CLUSTER_SIZE qualified) into nearest neighbor
    # ═══════════════════════════════════════════════════════════════
    MIN_CLUSTER_SIZE = 8
    for prefix in ["R", "L"]:
        hand_mask = all_pitchers["cluster"].str.startswith(prefix + "_")
        hand_pitchers = all_pitchers[hand_mask]
        cluster_sizes = hand_pitchers["cluster"].value_counts()
        small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index.tolist()

        if small_clusters:
            # Compute centroids of all non-small clusters
            big_clusters = [c for c in cluster_sizes.index if c not in small_clusters]
            centroids = {}
            for cid in big_clusters:
                c_rows = all_pitchers[all_pitchers["cluster"] == cid]
                centroids[cid] = c_rows[HAND_CLUSTER_FEATURES].fillna(0).mean().values

            for small_cid in small_clusters:
                small_rows = all_pitchers[all_pitchers["cluster"] == small_cid]
                small_centroid = small_rows[HAND_CLUSTER_FEATURES].fillna(0).mean().values
                # Find nearest big cluster
                best_cid, best_dist = None, float("inf")
                for big_cid, big_cent in centroids.items():
                    dist = np.linalg.norm(small_centroid - big_cent)
                    if dist < best_dist:
                        best_cid, best_dist = big_cid, dist
                all_pitchers.loc[all_pitchers["cluster"] == small_cid, "cluster"] = best_cid
                if small_cid in posthoc_ids[prefix]:
                    posthoc_ids[prefix].remove(small_cid)
                print(f"  Merged {small_cid} ({len(small_rows)} members) -> {best_cid}")

    # Save probability vectors
    proba_path = os.path.join(PROCESSED_DATA_DIR, "cluster_probabilities.parquet")
    all_proba.to_parquet(proba_path, engine="pyarrow", compression="snappy")
    print(f"\nSaved cluster probabilities: {proba_path} ({len(all_proba):,} rows)")

    # Save models
    for prefix, models in [("R", rhp_models), ("L", lhp_models)]:
        joblib.dump(models["scaler"], os.path.join(MODELS_DIR, f"scaler_{prefix}.joblib"))
        joblib.dump(models["gmm"], os.path.join(MODELS_DIR, f"gmm_{prefix}.joblib"))
        joblib.dump(models["pca"], os.path.join(MODELS_DIR, f"pca_{prefix}.joblib"))
    print("\nModels saved.")

    # Drop temp column and save
    all_pitchers.drop(columns=["pca_x_raw"], inplace=True, errors="ignore")
    all_pitchers.to_parquet(data_path, engine="pyarrow", compression="snappy")
    print(f"\nUpdated pitcher_seasons saved: {data_path}")
    rhp_total = len([c for c in posthoc_ids["R"]])
    lhp_total = len([c for c in posthoc_ids["L"]])
    print(f"  Total: {len(all_pitchers):,} pitcher-seasons")
    print(f"  RHP clusters: {rhp_total}")
    print(f"  LHP clusters: {lhp_total}")

    # Save meta
    meta = {
        "rhp_k": rhp_total,
        "lhp_k": lhp_total,
        "total_clusters": rhp_total + lhp_total,
        "features": HAND_CLUSTER_FEATURES,
        "x_offset": X_OFFSET,
        "method": "gmm+posthoc",
        "rhp_cluster_ids": posthoc_ids["R"],
        "lhp_cluster_ids": posthoc_ids["L"],
        "posthoc_rules": {k: {"conditions": {f: list(v) for f, v in r["conditions"].items()}, "min_size": r["min_size"]} for k, r in POSTHOC_RULES.items()},
    }
    with open(os.path.join(MODELS_DIR, "cluster_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # Backward-compatible alias
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
        sub_proba_parts = []
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

            # Predict nearest cluster (hard) and probabilities (soft)
            local_labels = models["gmm"].predict(X_sub_scaled)
            sub_proba = models["gmm"].predict_proba(X_sub_scaled)
            sub_hand["cluster"] = [f"{prefix}_{c}" for c in local_labels]

            # Build probability DataFrame for sub-threshold pitchers
            sub_proba_df = pd.DataFrame(
                sub_proba, columns=models["cluster_ids"], index=sub_hand.index
            )
            sub_proba_df["pitcher"] = sub_hand["pitcher"].values
            sub_proba_df["game_year"] = sub_hand["game_year"].values
            sub_proba_parts.append(sub_proba_df)

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

            # Apply post-hoc rules to sub-threshold pitchers too
            for rule_suffix, rule in POSTHOC_RULES.items():
                conditions = rule["conditions"]
                for prefix in ["R", "L"]:
                    new_cid = f"{prefix}_{rule_suffix}"
                    if new_cid not in posthoc_ids[prefix]:
                        continue  # Rule wasn't applied for this hand
                    mask = sub_assigned["cluster"].str.startswith(prefix + "_")
                    for feat, (op, thresh) in conditions.items():
                        if feat in sub_assigned.columns:
                            if op == "<":
                                mask = mask & (sub_assigned[feat] < thresh)
                            elif op == ">":
                                mask = mask & (sub_assigned[feat] > thresh)
                            elif op == ">=":
                                mask = mask & (sub_assigned[feat] >= thresh)
                            elif op == "<=":
                                mask = mask & (sub_assigned[feat] <= thresh)
                    n_sub = mask.sum()
                    if n_sub > 0:
                        sub_assigned.loc[mask, "cluster"] = new_cid
                        print(f"  {new_cid}: extracted {n_sub} sub-threshold pitchers")

            # Combine with qualified pitchers and save
            all_pitchers = pd.concat([all_pitchers, sub_assigned], ignore_index=True)
            all_pitchers.to_parquet(data_path, engine="pyarrow", compression="snappy")
            print(f"\n  Updated pitcher_seasons with sub-threshold assignments")
            print(f"  New total: {len(all_pitchers):,} pitcher-seasons")

        # Append sub-threshold probabilities
        if sub_proba_parts:
            sub_all_proba = pd.concat(sub_proba_parts, ignore_index=True).fillna(0.0)
            all_proba = pd.concat([all_proba, sub_all_proba], ignore_index=True).fillna(0.0)
            all_proba.to_parquet(proba_path, engine="pyarrow", compression="snappy")
            print(f"  Updated cluster_probabilities with sub-threshold pitchers")
    else:
        print("\nNo sub-threshold pitcher file found — skipping nearest-cluster assignment.")

    print("\nClustering complete!")


if __name__ == "__main__":
    main()
