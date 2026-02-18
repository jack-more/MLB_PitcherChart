"""
Export GLOBAL GMM pipeline data to frontend JSON format.

Reads from the single global model output (NOT per-year directories):
  - data/processed/pitcher_seasons.parquet      (all pitcher-seasons with cluster assignments)
  - data/processed/cluster_probabilities.parquet (GMM soft probabilities)
  - models/cluster_profiles.json                 (archetype names/colors from 05_cluster_naming.py)

Generates frontend-ready JSON files:
  - clusters.json:         Archetype profiles keyed by cluster ID (R_0, L_4, etc.)
  - pitcher_seasons.json:  All pitcher-seasons with GMM soft probabilities

Usage:
    python3 pipeline/export_frontend.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_SRC_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")
FRONTEND_PUBLIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "public")

# Emoji map for archetype short names (matched by base archetype word)
EMOJI_MAP = {
    "Snake":               "\U0001F40D",
    "Triple Threat":       "3\uFE0F\u20E3",
    "Split Demon":         "\U0001F479",
    "Yakker":              "\U0001F9AC",
    "Boomerang":           "\U0001FA83",
    "Knuckleball Wizard":  "\U0001F9D9",
    "Barnburner":          "\u26FD",
    "Uncle Charlie":       "\U0001F37A",
    "Ghost":               "\U0001F47B",
    "Cutman":              "\U0001F5E1",
    "Earthworm":           "\U0001FAB1",
    "Gardener":            "\U0001F9D1\u200D\U0001F33E",
    "Undertow":            "\U0001F30A",
    "Eephus Lobber":       "\U0001FAA6",
    "Kitchen Sink":        "\U0001F6B0",
    "Heavy Duty":          "\U0001F3CB\uFE0F",
    "Swordfighter":        "\u2694\uFE0F",
    "CutCraft":            "\u2702\uFE0F",
    "Wormburner":          "\U0001FAB1",
    "Heater-Heavy":        "\U0001F525",
    "Sinker-Cutter":       "\u2694\uFE0F",
    "Cutter-Curve Craftsman": "\u2702\uFE0F",
}


def _extract_archetype_name(short_name: str) -> str:
    """Extract the archetype base name from short_name like 'Cutman RP' -> 'Cutman'."""
    parts = short_name.strip().split()
    # Last part is role (SP/RP/SW), everything before is the name
    if parts and parts[-1] in ("SP", "RP", "SW"):
        return " ".join(parts[:-1])
    return short_name


def _find_emoji(archetype_name: str) -> str:
    """Find emoji for an archetype name, trying exact match then partial."""
    if archetype_name in EMOJI_MAP:
        return EMOJI_MAP[archetype_name]
    # Try partial match (first word)
    first_word = archetype_name.split()[0] if archetype_name else ""
    for key, emoji in EMOJI_MAP.items():
        if key.startswith(first_word):
            return emoji
    return ""


def export_all():
    print("=" * 60)
    print("  EXPORTING FRONTEND DATA (GLOBAL GMM)")
    print("=" * 60)

    # ── Load global model outputs ──
    ps_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    proba_path = os.path.join(PROCESSED_DATA_DIR, "cluster_probabilities.parquet")
    profiles_path = os.path.join(MODELS_DIR, "cluster_profiles.json")

    if not os.path.exists(ps_path):
        print(f"  ERROR: {ps_path} not found! Run 04_clustering.py first.")
        return

    df = pd.read_parquet(ps_path)
    print(f"  Loaded {len(df)} pitcher-seasons from global parquet")

    # Filter to qualified pitcher-seasons (>= 300 pitches)
    qualified = df[df["total_pitches"] >= 300].copy()
    print(f"  Qualified (>=300 pitches): {len(qualified)} pitcher-seasons")

    # Load soft probabilities
    proba_df = None
    if os.path.exists(proba_path):
        proba_df = pd.read_parquet(proba_path)
        print(f"  Loaded {len(proba_df)} rows of GMM soft probabilities")
    else:
        print(f"  WARNING: {proba_path} not found — no soft probs will be exported")

    # Load cluster profiles (from 05_cluster_naming.py)
    if not os.path.exists(profiles_path):
        print(f"  ERROR: {profiles_path} not found! Run 05_cluster_naming.py first.")
        return

    profiles = json.load(open(profiles_path))
    print(f"  Loaded {len(profiles)} cluster profiles")

    # ── Enhance cluster profiles with emojis for frontend ──
    clusters_out = {}
    for cid, prof in profiles.items():
        arch_name = _extract_archetype_name(prof["short_name"])
        emoji = _find_emoji(arch_name)
        role = prof["short_name"].split()[-1] if prof["short_name"].split()[-1] in ("SP", "RP", "SW") else ""

        # Build enhanced short_name with emoji: "🐍 Snake RP"
        enhanced_short = f"{emoji} {arch_name} {role}".strip()

        # Get representative stats for the cluster panel
        rep = prof.get("representative", {})

        clusters_out[cid] = {
            "full_name": prof["full_name"],
            "short_name": enhanced_short,
            "color": prof["color"],
            "hand": prof["hand"],
            "pitcher_count": prof["pitcher_count"],
            "example_pitchers": prof.get("example_pitchers", []),
            "pca_x": prof.get("pca_x", 0),
            "pca_y": prof.get("pca_y", 0),
            "pca_z": prof.get("pca_z", 0),
            # Stats from medoid pitcher (representative)
            "is_sp": round(rep.get("is_sp", 0), 3),
            "avg_velo_FF": round(rep.get("avg_velo_FF", 0), 1),
            "whiff_rate": round(rep.get("whiff_rate", 0), 4),
            "groundball_rate": round(rep.get("groundball_rate", 0), 4),
            "spin_overall": round(rep.get("spin_overall", 0), 1),
        }

    print(f"\n  Built {len(clusters_out)} cluster entries for frontend")

    # ── Merge soft probabilities into qualified DataFrame ──
    if proba_df is not None:
        # Proba columns are cluster IDs like R_0, R_1, ..., L_0, L_1, ...
        proba_cols = [c for c in proba_df.columns if c not in ("pitcher", "game_year")]

        # Merge on pitcher + game_year
        qualified = qualified.merge(
            proba_df, on=["pitcher", "game_year"], how="left", suffixes=("", "_proba")
        )
        print(f"  Merged soft probabilities: {len(proba_cols)} clusters")
    else:
        proba_cols = []

    # ── Build pitcher-season records ──
    all_pitcher_seasons = []

    for _, r in qualified.iterrows():
        cluster_id = str(r["cluster"])

        # Get archetype name from profiles
        prof = profiles.get(cluster_id, {})
        arch_name = _extract_archetype_name(prof.get("short_name", cluster_id))

        rec = {
            "pitcher": int(r.pitcher) if pd.notna(r.pitcher) else 0,
            "player_name": str(r.player_name),
            "game_year": int(r.game_year),
            "is_rhp": int(r.is_rhp),
            "is_sp": round(float(r.is_sp), 3),
            "cluster": cluster_id,  # e.g., "R_3", "L_4" — matches clusters.json keys
            "archetype": arch_name,  # e.g., "Gardener", "Ghost"
            "pca_x": round(float(r.pca_x), 4),
            "pca_y": round(float(r.pca_y), 4),
            "pca_z": round(float(r.pca_z), 4) if pd.notna(r.get("pca_z", None)) else 0,
            "avg_velo_FF": round(float(r.avg_velo_FF), 1) if pd.notna(r.get("avg_velo_FF", None)) else 0,
            "whiff_rate": round(float(r.whiff_rate), 4) if pd.notna(r.get("whiff_rate", None)) else 0,
            "arm_angle": round(float(r.arm_angle), 1) if pd.notna(r.get("arm_angle", None)) else 0,
            "pfx_x_avg": round(float(r.pfx_x_avg), 4) if pd.notna(r.get("pfx_x_avg", None)) else 0,
            "pfx_z_avg": round(float(r.pfx_z_avg), 4) if pd.notna(r.get("pfx_z_avg", None)) else 0,
        }

        # Add GMM soft probabilities keyed by cluster ID
        if proba_cols:
            gmm_proba = {}
            for col in proba_cols:
                val = r.get(col)
                if pd.notna(val):
                    prob = float(val)
                    if prob >= 0.005:  # Only include >= 0.5% to save JSON space
                        gmm_proba[col] = round(prob, 3)
            rec["gmm_proba"] = gmm_proba

        all_pitcher_seasons.append(rec)

    print(f"\n  Built {len(all_pitcher_seasons)} pitcher-season records")

    # ── Save to frontend/src/data/ ──
    os.makedirs(FRONTEND_SRC_DIR, exist_ok=True)

    clusters_path = os.path.join(FRONTEND_SRC_DIR, "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clusters_out, f, indent=2, ensure_ascii=False)
    print(f"\n  clusters.json: {len(clusters_out)} archetypes -> {clusters_path}")

    ps_out_path = os.path.join(FRONTEND_SRC_DIR, "pitcher_seasons.json")
    with open(ps_out_path, "w") as f:
        json.dump(all_pitcher_seasons, f, ensure_ascii=False)
    print(f"  pitcher_seasons.json: {len(all_pitcher_seasons)} records -> {ps_out_path}")

    # ── Also copy to frontend/public/ ──
    if os.path.isdir(FRONTEND_PUBLIC_DIR):
        pub_clusters = os.path.join(FRONTEND_PUBLIC_DIR, "clusters.json")
        with open(pub_clusters, "w") as f:
            json.dump(clusters_out, f, indent=2, ensure_ascii=False)
        print(f"  clusters.json -> {pub_clusters}")

        pub_ps = os.path.join(FRONTEND_PUBLIC_DIR, "pitcher_seasons.json")
        with open(pub_ps, "w") as f:
            json.dump(all_pitcher_seasons, f, ensure_ascii=False)
        print(f"  pitcher_seasons.json -> {pub_ps}")

    print(f"\n  DONE! Frontend data exported with global GMM.")


if __name__ == "__main__":
    export_all()
