"""
v22: Per-Year Clustering Analysis

Runs a THOROUGH analysis for a single year:
  1) Feature diagnostics: correlation matrix, VIF, feature distributions
  2) Drop-one silhouette analysis: which features help/hurt clustering?
  3) KMeans clustering (RHP/LHP separately)
  4) Cluster profiling with our naming convention
  5) Comparison summary for user review

Usage:
    python3 pipeline/v22_year_analysis.py --year 2025
"""

import os
import sys
import json
import argparse
import warnings
import pandas as pd
import numpy as np
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, CLUSTER_FEATURES, K_RANGE, RANDOM_STATE

# Features to use for clustering -- remove is_rhp since we split by hand
HAND_CLUSTER_FEATURES = [f for f in CLUSTER_FEATURES if f != "is_rhp"]

MIN_K = 4  # Lower floor for single-year (fewer pitchers)
MAX_K = 15
X_OFFSET = 5.0

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════
# 1. FEATURE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def feature_diagnostics(df, features, label="ALL"):
    """Compute correlation matrix, VIF, and basic stats for the feature set."""
    print(f"\n{'='*60}")
    print(f"  FEATURE DIAGNOSTICS — {label}")
    print(f"{'='*60}")

    X = df[features].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Basic stats
    print(f"\n  Pitcher-seasons: {len(df):,}")
    print(f"  Features: {len(features)}")

    stats = df[features].describe().T[["mean", "std", "min", "max"]]
    print(f"\n  Feature Distributions:")
    print(f"  {'Feature':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Zero%':>8}")
    print(f"  {'-'*73}")
    for feat in features:
        col = df[feat].fillna(0)
        zero_pct = (col == 0).mean() * 100
        print(f"  {feat:<25} {col.mean():>8.3f} {col.std():>8.3f} {col.min():>8.3f} {col.max():>8.3f} {zero_pct:>7.1f}%")

    # Correlation matrix — find high correlations
    corr = df[features].fillna(0).corr()
    print(f"\n  High Correlations (|r| > 0.5):")
    print(f"  {'Feature 1':<25} {'Feature 2':<25} {'r':>8}")
    print(f"  {'-'*60}")
    pairs_found = 0
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                print(f"  {features[i]:<25} {features[j]:<25} {r:>8.3f}")
                pairs_found += 1
    if pairs_found == 0:
        print(f"  (none)")

    # VIF
    print(f"\n  Variance Inflation Factors (VIF):")
    print(f"  {'Feature':<25} {'VIF':>10}")
    print(f"  {'-'*37}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Add small noise to prevent singular matrix
    X_vif = X_scaled + np.random.normal(0, 1e-6, X_scaled.shape)

    vif_results = []
    for i, feat in enumerate(features):
        try:
            vif = variance_inflation_factor(X_vif, i)
            vif_results.append((feat, vif))
        except Exception as e:
            vif_results.append((feat, float('nan')))

    vif_results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)
    for feat, vif in vif_results:
        flag = " *** HIGH" if vif > 10 else (" ** MODERATE" if vif > 5 else "")
        print(f"  {feat:<25} {vif:>10.2f}{flag}")

    return corr, vif_results


# ═══════════════════════════════════════════════════════════════
# 2. DROP-ONE SILHOUETTE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def drop_one_analysis(df, features, label="ALL", k=None):
    """For each feature, drop it and compare silhouette score."""
    print(f"\n{'='*60}")
    print(f"  DROP-ONE SILHOUETTE — {label} (n={len(df):,})")
    print(f"{'='*60}")

    X = df[features].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Baseline: all features
    if k is None:
        # Quick search for optimal K
        best_k, best_sil = MIN_K, 0
        for test_k in range(MIN_K, MAX_K + 1):
            km = KMeans(n_clusters=test_k, n_init=10, max_iter=300, random_state=RANDOM_STATE)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            if sil > best_sil:
                best_k, best_sil = test_k, sil
        k = best_k
        print(f"  Optimal K for baseline: {k} (sil={best_sil:.4f})")

    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    baseline_sil = silhouette_score(X_scaled, labels)
    print(f"  Baseline silhouette (all {len(features)} features, K={k}): {baseline_sil:.4f}")

    # Drop each feature
    results = []
    for i, feat in enumerate(features):
        remaining = [f for f in features if f != feat]
        X_drop = df[remaining].fillna(0).values
        X_drop = np.nan_to_num(X_drop, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_drop = StandardScaler()
        X_drop_scaled = scaler_drop.fit_transform(X_drop)
        X_drop_scaled = np.clip(X_drop_scaled, -10, 10)
        X_drop_scaled = np.nan_to_num(X_drop_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        km_drop = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=RANDOM_STATE)
        labels_drop = km_drop.fit_predict(X_drop_scaled)
        sil_drop = silhouette_score(X_drop_scaled, labels_drop)

        delta = sil_drop - baseline_sil
        results.append((feat, sil_drop, delta))

    # Sort by delta (positive = dropping IMPROVES clustering)
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  {'Feature':<25} {'Sil w/o':>10} {'Delta':>10} {'Impact':>20}")
    print(f"  {'-'*67}")
    for feat, sil, delta in results:
        if delta > 0.01:
            impact = "DROP CANDIDATE"
        elif delta > 0.005:
            impact = "weak/noisy"
        elif delta < -0.01:
            impact = "IMPORTANT (keep)"
        elif delta < -0.005:
            impact = "useful"
        else:
            impact = "neutral"
        print(f"  {feat:<25} {sil:>10.4f} {delta:>+10.4f} {impact:>20}")

    return baseline_sil, k, results


# ═══════════════════════════════════════════════════════════════
# 3. KMEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════

def cluster_single_year(df, features, year, prefix, label):
    """Full KMeans clustering for one hand in one year."""
    print(f"\n{'='*60}")
    print(f"  CLUSTERING {label} — {year} (n={len(df):,})")
    print(f"{'='*60}")

    X = df[features].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Search for optimal K
    print(f"  K search (K={MIN_K}..{MAX_K}):")
    results = []
    for k in range(MIN_K, MAX_K + 1):
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
        print(f"    K={k:>2}  sil={sil:.4f}  inertia={km.inertia_:>10.0f}")

    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df["silhouette"].idxmax()]
    optimal_k = int(best_row["k"])
    print(f"  -> Optimal K: {optimal_k} (sil={best_row['silhouette']:.4f})")

    # Fit final model
    km = KMeans(n_clusters=optimal_k, n_init=20, max_iter=500, random_state=RANDOM_STATE)
    local_labels = km.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = [f"{prefix}_{c}" for c in local_labels]

    # Cluster sizes
    print(f"\n  Cluster sizes:")
    for cid, count in df["cluster"].value_counts().sort_index().items():
        print(f"    {cid}: {count:,}")

    # 3D PCA
    pca = PCA(n_components=min(3, len(features)), random_state=RANDOM_STATE)
    X_3d = pca.fit_transform(X_scaled)
    print(f"\n  PCA variance explained: "
          f"PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
          f"PC3={pca.explained_variance_ratio_[2]:.1%}")

    df["pca_y"] = X_3d[:, 1]
    df["pca_z"] = X_3d[:, 2]
    if prefix == "R":
        df["pca_x"] = X_3d[:, 0] + X_OFFSET
    else:
        df["pca_x"] = -X_3d[:, 0] - X_OFFSET

    # Per-cluster silhouette
    sample_sils = silhouette_samples(X_scaled, local_labels)
    print(f"\n  Per-cluster silhouette:")
    for i in range(optimal_k):
        mask = local_labels == i
        cid = f"{prefix}_{i}"
        mean_sil = sample_sils[mask].mean()
        print(f"    {cid}: {mean_sil:.4f} (n={mask.sum():,})")

    models = {
        "scaler": scaler,
        "kmeans": km,
        "pca": pca,
        "optimal_k": optimal_k,
        "silhouette": float(best_row["silhouette"]),
    }

    return df, models, res_df


# ═══════════════════════════════════════════════════════════════
# 4. CLUSTER PROFILING & NAMING
# ═══════════════════════════════════════════════════════════════

def _pitcher_traits(row, hand):
    """Build traits dict from an individual pitcher row."""
    return {
        "ff": row.get("pct_FF", 0),
        "si": row.get("pct_SI", 0),
        "sl": row.get("pct_SL", 0),
        "cu": row.get("pct_CU", 0),
        "ch": row.get("pct_CH", 0),
        "fs": row.get("pct_FS", 0),
        "fc": row.get("pct_FC", 0),
        "st": row.get("pct_ST", 0),
        "kc": row.get("pct_KC", 0),
        "kn": row.get("pct_KN", 0),
        "whiff": row.get("whiff_rate", 0),
        "gb": row.get("groundball_rate", 0),
        "velo": row.get("avg_velo_FF", 0),
        "is_sp": row.get("is_sp", 0),
        "spin": row.get("spin_overall", 0),
    }


def profile_clusters(df, features, year):
    """Profile pitchers: name each K-means cluster from its medoid,
    then assign that name to all members.

    Sub-archetype and MUTT detection still use individual pitcher traits.
    Output keyed by hand_archetype (e.g. RHP_Barnburner).
    """
    from scipy.spatial.distance import cdist

    print(f"\n{'='*60}")
    print(f"  CLUSTER PROFILES (DNA SYSTEM) — {year}")
    print(f"{'='*60}")

    emoji_map = {
        "Snake": "\U0001F40D", "Triple Threat": "3\uFE0F\u20E3",
        "Split Demon": "\U0001F479", "Yakker": "\U0001F9AC",
        "Boomerang": "\U0001FA83",
        "Knuckleball Wizard": "\U0001F9D9", "Barnburner": "\u26FD",
        "Uncle Charlie": "\U0001F37A", "Ghost": "\U0001F47B",
        "Cutman": "\U0001F5E1", "Earthworm": "\U0001FAB1",
        "Gardener": "\U0001F9D1\u200D\U0001F33E", "Undertow": "\U0001F30A",
        "Eephus Lobber": "\U0001FAA6", "Kitchen Sink": "\U0001F6B0",
        "Heavy Duty": "\U0001F3CB\uFE0F", "Swordfighter": "\u2694\uFE0F",
        "CutCraft": "\u2702\uFE0F",
    }

    # ── Step 1: Name each K-means cluster from its geometric medoid ──
    cluster_archetypes = {}   # cluster_id -> archetype name
    cluster_medoid_rows = {}  # cluster_id -> medoid pitcher row

    for cid in df["cluster"].unique():
        members = df[df["cluster"] == cid]
        if len(members) == 0:
            continue

        # Position-player junk
        if members["total_pitches"].mean() < 100:
            cluster_archetypes[cid] = "Eephus Lobber"
            cluster_medoid_rows[cid] = members.iloc[0]
            continue

        # Find geometric medoid in clustering feature space
        feature_cols = [f for f in features if f in members.columns]
        coords = members[feature_cols].fillna(0).values
        if len(coords) < 2:
            medoid_row = members.iloc[0]
        else:
            dist_matrix = cdist(coords, coords, metric='euclidean')
            total_dists = dist_matrix.sum(axis=1)
            medoid_row = members.iloc[int(np.argmin(total_dists))]

        cluster_medoid_rows[cid] = medoid_row
        hand = "RHP" if cid.startswith("R") else "LHP"
        t = _pitcher_traits(medoid_row, hand)
        cluster_archetypes[cid] = _archetype_name(t, hand)

    print(f"\n  K-means clusters → archetype names (via medoid):")
    for cid in sorted(cluster_archetypes.keys()):
        med = cluster_medoid_rows.get(cid)
        med_name = med.get("player_name", "?") if med is not None else "?"
        print(f"    {cid} → {cluster_archetypes[cid]} (medoid: {med_name})")

    # ── Step 2: Assign each pitcher archetype from their cluster ──
    arch_list, sub_list, dna_list = [], [], []
    for idx, prow in df.iterrows():
        cid = prow["cluster"]
        hand = "RHP" if prow.get("is_rhp", 1) == 1 else "LHP"

        # Position-player junk
        if prow.get("total_pitches", 0) < 100:
            arch_list.append("Eephus Lobber")
            sub_list.append("Pure")
            dna_list.append("Eephus Lobber")
            continue

        # Primary archetype comes from the CLUSTER (medoid-named)
        primary = cluster_archetypes.get(cid, "Kitchen Sink")

        # Sub-archetype: what other rules does THIS pitcher trigger?
        t = _pitcher_traits(prow, hand)
        triggered = []
        for name, test_fn, defining_fn in ARCHETYPE_RULES:
            if name == primary or name == "Kitchen Sink":
                continue
            if test_fn(t, hand):
                triggered.append((name, defining_fn(t)))

        if triggered:
            triggered.sort(key=lambda x: x[1], reverse=True)
            sub = triggered[0][0]
        else:
            sub = "Pure"

        # MUTT check: does this pitcher's individual rule assignment
        # differ from their cluster's archetype?
        individual_arch = _archetype_name(t, hand)
        is_mutt = (individual_arch != primary)

        # Build display string
        if sub == "Pure" and not is_mutt:
            dna_str = primary
        elif is_mutt:
            dna_str = f"\U0001F9EC {primary} / {sub}"
        else:
            dna_str = f"{primary} / {sub}"

        arch_list.append(primary)
        sub_list.append(sub)
        dna_list.append(dna_str)

    df["archetype"] = arch_list
    df["sub_archetype"] = sub_list
    df["archetype_dna"] = dna_list

    # ── Step 3: Group by (hand, archetype) and build profiles ──
    profiles = {}
    for (hand_label, arch_name), group in df.groupby(
        [df["is_rhp"].map({1: "RHP", 0: "LHP"}), "archetype"]
    ):
        count = len(group)
        if count == 0:
            continue

        key = f"{hand_label}_{arch_name}"

        # Find group's geometric medoid for representative stats
        pca_cols = [c for c in ["pca_x", "pca_y", "pca_z"] if c in group.columns]
        if len(pca_cols) >= 2 and len(group) >= 2:
            pca_coords = group[pca_cols].fillna(0).values
            dist_matrix = cdist(pca_coords, pca_coords, metric='euclidean')
            total_dists = dist_matrix.sum(axis=1)
            medoid_row = group.iloc[int(np.argmin(total_dists))]
        else:
            medoid_row = group.iloc[0]

        avg_pitches = group["total_pitches"].mean()  # group metadata, not trait averaging

        # Role from medoid
        medoid_sp = float(medoid_row.get("is_sp", 0))
        if medoid_sp > 0.55:
            role = "SP"
        elif medoid_sp < 0.35:
            role = "RP"
        else:
            role = "SW"

        # Medoid pitcher's actual traits (NO averaging)
        medoid_traits = {}
        for col_key, col_name in [
            ("ff", "pct_FF"), ("si", "pct_SI"), ("sl", "pct_SL"), ("cu", "pct_CU"),
            ("ch", "pct_CH"), ("fs", "pct_FS"), ("fc", "pct_FC"), ("st", "pct_ST"),
            ("kc", "pct_KC"), ("kn", "pct_KN"), ("whiff", "whiff_rate"),
            ("gb", "groundball_rate"), ("velo", "avg_velo_FF"), ("spin", "spin_overall"),
        ]:
            medoid_traits[col_key] = round(float(medoid_row.get(col_name, 0)), 3) if col_name in group.columns else 0
        medoid_traits["is_sp"] = round(float(medoid_sp), 3)

        # Top pitches from medoid
        pitch_keys = [("ff", "FF"), ("si", "SI"), ("sl", "SL"), ("cu", "CU"),
                      ("ch", "CH"), ("fs", "FS"), ("fc", "FC"), ("st", "ST"),
                      ("kc", "KC"), ("kn", "KN")]
        top_pitches = sorted([(label, medoid_traits[k]) for k, label in pitch_keys if medoid_traits[k] > 0.03],
                             key=lambda x: x[1], reverse=True)[:4]
        top_str = " | ".join([f"{p[0]}: {p[1]:.0%}" for p in top_pitches])

        # Example pitchers (top 5 by pitches thrown)
        examples = []
        top_idx = group["total_pitches"].nlargest(min(5, count)).index
        for idx in top_idx[:5]:
            examples.append(f"{df.loc[idx].get('player_name', '?')}")

        # Sub-archetype breakdown
        sub_counts = Counter(group["sub_archetype"])
        mutt_count = sum(1 for d in group["archetype_dna"] if "\U0001F9EC" in str(d))

        emoji = emoji_map.get(arch_name, "\u2753")

        profiles[key] = {
            "name": arch_name,
            "emoji": emoji,
            "role": role,
            "hand": hand_label,
            "count": count,
            "avg_pitches": round(avg_pitches),
            "top_pitches": top_str,
            "examples": examples,
            "traits": medoid_traits,
            "sub_archetypes": {k: v for k, v in sub_counts.most_common()},
            "mutt_count": mutt_count,
            "medoid": medoid_row.get("player_name", "?"),
        }

        # Print summary
        pure_count = sub_counts.get("Pure", 0)
        print(f"\n  {key} {emoji} {arch_name} ({role}) — {count} pitchers")
        print(f"    Medoid: {medoid_row.get('player_name', '?')}")
        print(f"    Pitches: {top_str}")
        print(f"    Whiff: {medoid_traits['whiff']:.1%} | GB: {medoid_traits['gb']:.1%} | Velo: {medoid_traits['velo']:.1f}")
        print(f"    Purebred: {pure_count} | Mutts: {mutt_count}")
        print(f"    Examples: {', '.join(examples)}")
        if len(sub_counts) > 1:
            sub_str = ", ".join(f"{n}: {c}" for n, c in sub_counts.most_common() if n != "Pure")
            print(f"    Sub-breeds: {sub_str}")

    return profiles


def _archetype_name(t, hand):
    """Apply our naming convention to a trait dict. 16 archetypes, priority order."""
    if t["kn"] > 0.10:
        return "Knuckleball Wizard"
    if t["fs"] > 0.15:
        return "Split Demon"
    if t["kc"] > 0.15:
        return "Uncle Charlie"
    if t["si"] > 0.35 and t["ff"] < 0.05:
        return "Undertow"
    if t["st"] > 0.20:
        return "Boomerang"
    if t["ch"] > 0.20:
        return "Ghost"
    if t["si"] > 0.35 and t["fc"] > 0.15:
        return "Snake"
    if t["si"] > 0.35 and t["sl"] > 0.18:
        return "Gardener"
    if t["si"] > 0.50:
        return "Earthworm"
    if t["ff"] > 0.40 and t["sl"] > 0.30:
        return "Barnburner"
    if t["ff"] > 0.40 and t["sl"] > 0.15 and t["cu"] > 0.10:
        return "Triple Threat"
    if t["cu"] > 0.15 and (t["fc"] > 0.12 or t["sl"] > 0.15):
        return "CutCraft"
    if t["cu"] > 0.12:
        return "Yakker"
    if t["fc"] > 0.15:
        return "Cutman"
    if t["sl"] > 0.25:
        return "Swordfighter"
    if t["ff"] + t["si"] > 0.50:
        return "Heavy Duty"
    return "Kitchen Sink"


# ── Archetype rule definitions for DNA system ──
# Each rule: (name, test_fn, defining_pitches_fn)
#   test_fn(t, hand) -> bool
#   defining_pitches_fn(t) -> float (sum of the pitches that define this archetype)
ARCHETYPE_RULES = [
    ("Knuckleball Wizard", lambda t, h: t["kn"] > 0.10,           lambda t: t["kn"]),
    ("Split Demon",        lambda t, h: t["fs"] > 0.15,           lambda t: t["fs"]),
    ("Uncle Charlie",      lambda t, h: t["kc"] > 0.15,           lambda t: t["kc"]),
    ("Undertow",           lambda t, h: t["si"] > 0.35 and t["ff"] < 0.05, lambda t: t["si"]),
    ("Boomerang",          lambda t, h: t["st"] > 0.20,           lambda t: t["st"]),
    ("Ghost",              lambda t, h: t["ch"] > 0.20,           lambda t: t["ch"]),
    ("Snake",              lambda t, h: t["si"] > 0.35 and t["fc"] > 0.15, lambda t: t["si"] + t["fc"]),
    ("Gardener",           lambda t, h: t["si"] > 0.35 and t["sl"] > 0.18, lambda t: t["si"] + t["sl"]),
    ("Earthworm",          lambda t, h: t["si"] > 0.50,           lambda t: t["si"]),
    ("Barnburner",         lambda t, h: t["ff"] > 0.40 and t["sl"] > 0.30, lambda t: t["ff"] + t["sl"]),
    ("Triple Threat",      lambda t, h: t["ff"] > 0.40 and t["sl"] > 0.15 and t["cu"] > 0.10, lambda t: t["ff"] + t["sl"] + t["cu"]),
    ("CutCraft",           lambda t, h: t["cu"] > 0.15 and (t["fc"] > 0.12 or t["sl"] > 0.15), lambda t: t["cu"] + t["fc"]),
    ("Yakker",             lambda t, h: t["cu"] > 0.12,           lambda t: t["cu"]),
    ("Cutman",             lambda t, h: t["fc"] > 0.15,           lambda t: t["fc"]),
    ("Swordfighter",       lambda t, h: t["sl"] > 0.25,           lambda t: t["sl"]),
    ("Heavy Duty",         lambda t, h: t["ff"] + t["si"] > 0.50, lambda t: t["ff"] + t["si"]),
    ("Kitchen Sink",       lambda t, h: True,                     lambda t: 0.0),
]


def _archetype_dna(t, hand):
    """Build pitcher DNA: primary + sub-archetype + MUTT tag.

    Returns (primary, sub_archetype, archetype_dna_string)
    """
    primary = _archetype_name(t, hand)

    # Find all OTHER rules this pitcher also triggers (skip primary & Kitchen Sink)
    triggered = []
    for name, test_fn, defining_fn in ARCHETYPE_RULES:
        if name == primary or name == "Kitchen Sink":
            continue
        if test_fn(t, hand):
            triggered.append((name, defining_fn(t)))

    # Sub = triggered rule with highest defining pitch contribution
    if triggered:
        triggered.sort(key=lambda x: x[1], reverse=True)
        sub = triggered[0][0]
    else:
        sub = "Pure"

    # Kitchen Sink = always MUTT — these guys didn't trigger ANY rule
    # Find their closest identity by highest defining pitch contribution
    if primary == "Kitchen Sink":
        best_name, best_pct = None, 0
        for name, test_fn, defining_fn in ARCHETYPE_RULES:
            if name == "Kitchen Sink":
                continue
            pct = defining_fn(t)
            if pct > best_pct:
                best_pct = pct
                best_name = name
        sub = best_name if best_name else "Pure"
        dna_str = f"\U0001F9EC Kitchen Sink / {sub}"
        return primary, sub, dna_str

    # MUTT check: primary's defining pitches < 45% of mix
    is_mutt = False
    for name, test_fn, defining_fn in ARCHETYPE_RULES:
        if name == primary:
            primary_pct = defining_fn(t)
            if primary_pct < 0.45:
                is_mutt = True
            break

    # Build display string
    if sub == "Pure":
        dna_str = primary
    elif is_mutt:
        dna_str = f"\U0001F9EC {primary} / {sub}"
    else:
        dna_str = f"{primary} / {sub}"

    return primary, sub, dna_str


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v22: Per-year clustering analysis")
    parser.add_argument("--year", type=int, default=2025, help="Year to analyze")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip feature diagnostics")
    parser.add_argument("--skip-dropone", action="store_true", help="Skip drop-one analysis")
    args = parser.parse_args()

    year = args.year

    print(f"{'#'*60}")
    print(f"  v22 PER-YEAR ANALYSIS: {year}")
    print(f"{'#'*60}")

    # Load full dataset and filter to year
    data_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    all_data = pd.read_parquet(data_path)
    df = all_data[all_data["game_year"] == year].copy()

    print(f"\n  Total pitcher-seasons in {year}: {len(df):,}")
    rhp = df[df["is_rhp"] == 1].copy()
    lhp = df[df["is_rhp"] == 0].copy()
    print(f"  RHP: {len(rhp):,}  |  LHP: {len(lhp):,}")

    features = HAND_CLUSTER_FEATURES

    # Fill missing features
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
        if f not in rhp.columns:
            rhp[f] = 0.0
        if f not in lhp.columns:
            lhp[f] = 0.0

    # ═════════════ DIAGNOSTICS ═════════════
    if not args.skip_diagnostics:
        print(f"\n\n{'#'*60}")
        print(f"  SECTION 1: FEATURE DIAGNOSTICS")
        print(f"{'#'*60}")

        # All pitchers
        corr_all, vif_all = feature_diagnostics(df, features, f"ALL {year}")

        # RHP only
        corr_rhp, vif_rhp = feature_diagnostics(rhp, features, f"RHP {year}")

        # LHP only
        corr_lhp, vif_lhp = feature_diagnostics(lhp, features, f"LHP {year}")

    # ═════════════ DROP-ONE ═════════════
    if not args.skip_dropone:
        print(f"\n\n{'#'*60}")
        print(f"  SECTION 2: DROP-ONE SILHOUETTE")
        print(f"{'#'*60}")

        rhp_baseline, rhp_k, rhp_dropone = drop_one_analysis(rhp, features, f"RHP {year}")
        lhp_baseline, lhp_k, lhp_dropone = drop_one_analysis(lhp, features, f"LHP {year}")

    # ═════════════ CLUSTERING ═════════════
    print(f"\n\n{'#'*60}")
    print(f"  SECTION 3: KMEANS CLUSTERING")
    print(f"{'#'*60}")

    rhp_out, rhp_models, rhp_k_results = cluster_single_year(
        rhp, features, year, "R", "RHP"
    )
    lhp_out, lhp_models, lhp_k_results = cluster_single_year(
        lhp, features, year, "L", "LHP"
    )

    # Merge
    all_clustered = pd.concat([rhp_out, lhp_out], ignore_index=True)

    # ═════════════ PROFILES ═════════════
    print(f"\n\n{'#'*60}")
    print(f"  SECTION 4: CLUSTER PROFILES")
    print(f"{'#'*60}")

    profiles = profile_clusters(all_clustered, features, year)

    # ═════════════ SUMMARY ═════════════
    print(f"\n\n{'#'*60}")
    print(f"  SUMMARY: {year}")
    print(f"{'#'*60}")

    total_clusters = rhp_models["optimal_k"] + lhp_models["optimal_k"]
    total_pitchers = len(all_clustered)
    total_archetypes = len(profiles)
    junk = sum(1 for p in profiles.values() if p["name"] == "Eephus Lobber")

    print(f"\n  KMeans clusters: {total_clusters} ({rhp_models['optimal_k']} RHP + {lhp_models['optimal_k']} LHP)")
    print(f"  DNA archetypes: {total_archetypes} (grouped by hand + archetype)")
    print(f"  Total pitchers: {total_pitchers:,}")
    print(f"  Junk (Eephus Lobber): {junk}")
    print(f"  RHP silhouette: {rhp_models['silhouette']:.4f}")
    print(f"  LHP silhouette: {lhp_models['silhouette']:.4f}")

    # Archetype distribution
    print(f"\n  Archetype Distribution:")
    for key in sorted(profiles.keys()):
        p = profiles[key]
        emoji = p.get("emoji", "")
        mutt_pct = p["mutt_count"] / p["count"] * 100 if p["count"] > 0 else 0
        pure_ct = p["sub_archetypes"].get("Pure", 0)
        print(f"    {emoji} {key}: {p['count']} pitchers ({pure_ct} pure, {p['mutt_count']} mutts [{mutt_pct:.0f}%])")

    # Save results
    out_dir = os.path.join(MODELS_DIR, str(year))
    os.makedirs(out_dir, exist_ok=True)

    # Save models
    for prefix, models in [("R", rhp_models), ("L", lhp_models)]:
        joblib.dump(models["scaler"], os.path.join(out_dir, f"scaler_{prefix}.joblib"))
        joblib.dump(models["kmeans"], os.path.join(out_dir, f"kmeans_{prefix}.joblib"))
        joblib.dump(models["pca"], os.path.join(out_dir, f"pca_{prefix}.joblib"))

    # Save profiles
    with open(os.path.join(out_dir, "cluster_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)

    # Save clustered data
    all_clustered.to_parquet(os.path.join(out_dir, "pitcher_seasons.parquet"),
                             engine="pyarrow", compression="snappy")

    # Save meta
    meta = {
        "year": year,
        "rhp_k": rhp_models["optimal_k"],
        "lhp_k": lhp_models["optimal_k"],
        "rhp_silhouette": rhp_models["silhouette"],
        "lhp_silhouette": lhp_models["silhouette"],
        "features": features,
        "total_pitcher_seasons": len(all_clustered),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Results saved to: {out_dir}/")
    print(f"\n  DONE! Review the output above and discuss findings.")


if __name__ == "__main__":
    main()
