"""
Export pipeline data to frontend JSON format.

Reads per-year models/ output and generates frontend-ready JSON files:
  - clusters.json: Archetype profiles keyed by {hand}_{archetype}
  - pitcher_seasons.json: All pitcher-seasons across all years

NO CENTROIDS. Uses pipeline traits (individual pitcher stats) and medoid PCA positions.

Usage:
    python3 pipeline/export_frontend.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR

FRONTEND_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "src", "data"
)

YEARS = range(2015, 2026)

# Consistent colors per archetype name
ARCHETYPE_COLORS = {
    "Knuckleball Wizard": "#9b59b6",
    "Split Demon": "#e74c3c",
    "Uncle Charlie": "#f39c12",
    "Undertow": "#3498db",
    "Boomerang": "#1abc9c",
    "Ghost": "#95a5a6",
    "Snake": "#2ecc71",
    "Gardener": "#27ae60",
    "Earthworm": "#8b4513",
    "Barnburner": "#e63946",
    "Triple Threat": "#f4a261",
    "CutCraft": "#2a9d8f",
    "Yakker": "#e9c46a",
    "Cutman": "#457b9d",
    "Swordfighter": "#a8dadc",
    "Heavy Duty": "#264653",
    "Kitchen Sink": "#6c757d",
    "Eephus Lobber": "#343a40",
}

EMOJI_MAP = {
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


def _find_medoid(group):
    """Find the real pitcher closest to the group's geometric center in PCA space."""
    coords = group[["pca_x", "pca_y", "pca_z"]].values
    center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    medoid_idx = np.argmin(dists)
    row = group.iloc[medoid_idx]
    return float(row.pca_x), float(row.pca_y), float(row.pca_z)


def export_all():
    print("=" * 60)
    print("  EXPORTING FRONTEND DATA (NO CENTROIDS)")
    print("=" * 60)

    all_pitcher_seasons = []
    all_dfs = []
    all_profiles = {}  # merged from per-year cluster_profiles.json

    for year in YEARS:
        year_dir = os.path.join(MODELS_DIR, str(year))
        parquet_path = os.path.join(year_dir, "pitcher_seasons.parquet")

        if not os.path.exists(parquet_path):
            print(f"  SKIP {year}: no parquet found")
            continue

        df = pd.read_parquet(parquet_path)
        print(f"  {year}: {len(df)} pitcher-seasons")

        # Load pipeline profiles (traits from individual pitcher stats)
        profiles_path = os.path.join(year_dir, "cluster_profiles.json")
        if os.path.exists(profiles_path):
            year_profiles = json.load(open(profiles_path))
            for key, prof in year_profiles.items():
                if key not in all_profiles:
                    all_profiles[key] = prof
                else:
                    # Keep most recent year's profile (last year wins)
                    all_profiles[key] = prof

        # Build archetype key
        df["archetype_key"] = df.apply(
            lambda r: f"{'RHP' if r.is_rhp == 1 else 'LHP'}_{r.archetype}", axis=1
        )
        all_dfs.append(df)

        # Export pitcher-season records
        for _, r in df.iterrows():
            rec = {
                "pitcher": int(r.pitcher) if pd.notna(r.pitcher) else 0,
                "player_name": str(r.player_name),
                "game_year": int(r.game_year),
                "is_rhp": int(r.is_rhp),
                "is_sp": round(float(r.is_sp), 3),
                "cluster": r.archetype_key,
                "archetype": str(r.archetype),
                "sub_archetype": str(r.get("sub_archetype", "Pure")),
                "archetype_dna": str(r.get("archetype_dna", r.archetype)),
                "pca_x": round(float(r.pca_x), 4),
                "pca_y": round(float(r.pca_y), 4),
                "pca_z": round(float(r.pca_z), 4),
                "avg_velo_FF": round(float(r.avg_velo_FF), 1) if pd.notna(r.avg_velo_FF) else 0,
                "whiff_rate": round(float(r.whiff_rate), 4) if pd.notna(r.whiff_rate) else 0,
                "arm_angle": round(float(r.arm_angle), 1) if pd.notna(r.get("arm_angle", None)) else 0,
                "pfx_x_avg": round(float(r.pfx_x_avg), 4) if pd.notna(r.get("pfx_x_avg", None)) else 0,
                "pfx_z_avg": round(float(r.pfx_z_avg), 4) if pd.notna(r.get("pfx_z_avg", None)) else 0,
            }
            all_pitcher_seasons.append(rec)

    if not all_dfs:
        print("  ERROR: No data found!")
        return

    # ── Combine all years into one DataFrame ──
    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Build cluster profiles (NO CENTROIDS) ──
    clusters_out = {}

    for key, group in combined.groupby("archetype_key"):
        parts = key.split("_", 1)
        hand = parts[0]
        archetype_name = parts[1]

        # PCA position: medoid (real pitcher closest to center)
        med_x, med_y, med_z = _find_medoid(group)

        # Get traits from pipeline profiles (individual pitcher stats, NOT centroid averages)
        prof = all_profiles.get(key, {})
        traits = prof.get("traits", {})

        # Role from pipeline traits
        sp_ratio = traits.get("is_sp", float(group["is_sp"].mean()))
        if sp_ratio > 0.55:
            role = "SP"
        elif sp_ratio < 0.35:
            role = "RP"
        else:
            role = "SW"

        # Example pitchers from most recent year
        max_year = int(group["game_year"].max())
        recent = group[group["game_year"] == max_year]
        examples = recent.nlargest(min(3, len(recent)), "total_pitches")["player_name"].tolist()
        example_strs = [f"{name} ({max_year})" for name in examples]

        emoji = EMOJI_MAP.get(archetype_name, "")
        color = ARCHETYPE_COLORS.get(archetype_name, "#888888")
        short_name = f"{emoji} {archetype_name} {role}"

        clusters_out[key] = {
            "full_name": f"{hand} {archetype_name}",
            "short_name": short_name,
            "color": color,
            "hand": hand,
            "pitcher_count": len(group),
            "example_pitchers": example_strs,
            "top_pitches": prof.get("top_pitches", ""),
            # Stats from pipeline traits (individual pitcher data)
            "is_sp": round(sp_ratio, 3),
            "avg_velo_FF": round(traits.get("velo", 0), 1),
            "whiff_rate": round(traits.get("whiff", 0), 4),
            "groundball_rate": round(traits.get("gb", 0), 4),
            # PCA medoid position (real pitcher, not phantom average)
            "pca_x": round(med_x, 4),
            "pca_y": round(med_y, 4),
            "pca_z": round(med_z, 4),
        }

    # ── Save ──
    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)

    clusters_path = os.path.join(FRONTEND_DATA_DIR, "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clusters_out, f, indent=2, ensure_ascii=False)
    print(f"\n  clusters.json: {len(clusters_out)} archetypes -> {clusters_path}")

    ps_path = os.path.join(FRONTEND_DATA_DIR, "pitcher_seasons.json")
    with open(ps_path, "w") as f:
        json.dump(all_pitcher_seasons, f, ensure_ascii=False)
    print(f"  pitcher_seasons.json: {len(all_pitcher_seasons)} records -> {ps_path}")

    print(f"\n  NOTE: hitter_vs_cluster.json needs re-generation with new archetype keys.")
    print(f"  DONE! Frontend data exported.")


if __name__ == "__main__":
    export_all()
