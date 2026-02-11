"""
Step 5: Generate quirky archetype names for each K-Means cluster.

Reads the centroids (in original scale), applies rule-based naming
examining dominant traits, and saves cluster_profiles.json.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, CLUSTER_FEATURES


# Baseball-friendly color palette for the frontend
CLUSTER_COLORS = [
    "#e63946",  # crimson
    "#457b9d",  # steel blue
    "#2a9d8f",  # teal
    "#e9c46a",  # gold
    "#f4a261",  # burnt orange
    "#264653",  # dark navy
    "#6a4c93",  # purple
    "#1d3557",  # midnight
    "#a8dadc",  # light teal
    "#f77f00",  # tangerine
    "#d62828",  # fire red
    "#588157",  # forest
    "#bc6c25",  # sienna
    "#606c38",  # olive
    "#023047",  # deep navy
]


def _get_top_pitches(row: pd.Series, n: int = 3) -> list:
    """Return the top n pitch types by usage from the centroid."""
    pitch_cols = [c for c in row.index if c.startswith("pct_")]
    pitches = [(c.replace("pct_", ""), float(row[c])) for c in pitch_cols if row[c] > 0.03]
    return sorted(pitches, key=lambda x: x[1], reverse=True)[:n]


def _role_str(is_sp: float) -> str:
    if is_sp > 0.55:
        return "Starter"
    elif is_sp < 0.35:
        return "Reliever"
    return "Swingman"


def _role_short(is_sp: float) -> str:
    if is_sp > 0.55:
        return "SP"
    elif is_sp < 0.35:
        return "RP"
    return "SW"


def _score_traits(row: pd.Series) -> dict:
    """Score how distinctive each trait is for this cluster centroid."""
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
        "sv": row.get("pct_SV", 0),
        "kn": row.get("pct_KN", 0),
        "whiff": row.get("whiff_rate", 0),
        "gb": row.get("groundball_rate", 0),
        "velo": row.get("avg_velo_FF", 0),
        "is_rhp": row.get("is_rhp", 0),
        "is_sp": row.get("is_sp", 0),
        "spin": row.get("spin_overall", 0),
        "ext": row.get("avg_extension", 0),
        # Zone location traits
        "ss_up": row.get("ss_up_rate", 0.5),
        "ss_arm": row.get("ss_arm_side", 0.5),
        "ss_heart": row.get("ss_heart_rate", 0.08),
        "ss_edge": row.get("ss_edge_rate", 0.5),
        "os_up": row.get("os_up_rate", 0.5),
        "os_arm": row.get("os_arm_side", 0.5),
        "os_heart": row.get("os_heart_rate", 0.08),
        "os_edge": row.get("os_edge_rate", 0.5),
        "platoon_lat": row.get("platoon_lateral_shift", 0),
        "platoon_ht": row.get("platoon_height_shift", 0),
        "ss_entropy": row.get("ss_location_entropy", 3.0),
        "os_entropy": row.get("os_location_entropy", 3.0),
        "entropy_shift": row.get("entropy_shift", 0),
    }


def generate_full_name(row: pd.Series) -> str:
    """Generate a descriptive archetype name from centroid feature values.

    Uses a layered approach: primary identity (most distinctive pitch trait),
    then secondary modifiers (outcome, velocity, groundball tendency).
    """
    t = _score_traits(row)
    role = _role_str(t["is_sp"])
    parts = ["The"]

    # --- Primary identity: what makes this cluster's PITCH MIX unique ---
    if t["sv"] > 0.10:
        parts.append("Screwball Unicorn")
    elif t["kn"] > 0.10:
        parts.append("Knuckleball Wizard")
    elif t["fs"] > 0.15:
        parts.append("Splitter Assassin")
    elif t["kc"] > 0.15:
        parts.append("Knuckle-Curve Sorcerer")
    elif t["st"] > 0.20:
        parts.append("Sweeper Merchant")
    elif t["ch"] > 0.20:
        parts.append("Circle-Change Phantom")
    elif t["si"] > 0.40 and t["ff"] < 0.05:
        parts.append("Pure Sinker Ghost")
    elif t["si"] > 0.35 and t["fc"] > 0.15:
        parts.append("Sinker-Cutter Hybrid")
    elif t["si"] > 0.35 and t["sl"] > 0.18:
        parts.append("Sinker-Slider Earthworm")
    elif t["si"] > 0.35:
        parts.append("Sinker Savant")
    elif t["ff"] > 0.45 and t["sl"] > 0.30:
        parts.append("Gas-and-Snap Two-Pitch Demon")
    elif t["ff"] > 0.40 and t["sl"] > 0.15 and t["cu"] > 0.10:
        parts.append("Fastball-Curve-Slider Triple Threat")
    elif t["cu"] > 0.15 and t["fc"] > 0.12:
        parts.append("Cutter-Curve Craftsman")
    elif t["cu"] > 0.12 and t["is_sp"] > 0.55:
        parts.append("Uncle Charlie")
    elif t["cu"] > 0.12:
        parts.append("Yakker Specialist")
    elif t["fc"] > 0.15:
        parts.append("Cutter Carver")
    elif t["ff"] + t["si"] > 0.55 and t["whiff"] > 0.25:
        parts.append("Heater-Heavy Flamethrower")
    elif t["ff"] + t["si"] > 0.55:
        parts.append("Heater-Heavy")
    else:
        parts.append("Kitchen Sink Illusionist")

    # --- Zone location modifier (only when distinctive) ---
    avg_up = (t["ss_up"] + t["os_up"]) / 2
    avg_arm = (t["ss_arm"] + t["os_arm"]) / 2
    avg_heart = (t["ss_heart"] + t["os_heart"]) / 2
    avg_entropy = (t["ss_entropy"] + t["os_entropy"]) / 2

    if avg_up > 0.55 and avg_arm > 0.55:
        parts.append("Up-and-In")
    elif avg_up > 0.55 and avg_arm < 0.45:
        parts.append("Up-and-Away")
    elif avg_up > 0.55:
        parts.append("High-Zone")
    elif avg_up < 0.35:
        parts.append("Low-Zone")

    if avg_heart > 0.12:
        parts.append("Heart-Attacker")
    elif avg_heart < 0.06:
        parts.append("Nibbler")

    if t["platoon_lat"] > 0.25:
        parts.append("Platoon-Shifter")

    if avg_entropy < 2.6:
        parts.append("One-Track")

    # --- Secondary modifier: outcome profile ---
    if t["whiff"] > 0.26:
        parts.append("Swing-and-Miss Machine")
    elif t["whiff"] < 0.215:
        parts.append("Contact Trap")

    if t["gb"] > 0.50:
        parts.append("Groundball Harvester")
    elif t["gb"] > 0.45:
        parts.append("Worm Burner")
    elif t["gb"] < 0.41:
        parts.append("Flyball Daredevil")

    parts.append(role)
    return " ".join(parts)


def _zone_prefix(t: dict) -> str:
    """Return a short zone location prefix for the short name, or empty string."""
    avg_up = (t["ss_up"] + t["os_up"]) / 2
    avg_arm = (t["ss_arm"] + t["os_arm"]) / 2
    avg_heart = (t["ss_heart"] + t["os_heart"]) / 2

    if avg_up > 0.55 and avg_arm > 0.55:
        return "Up-In "
    elif avg_up > 0.55 and avg_arm < 0.45:
        return "Up-Away "
    elif avg_up > 0.55:
        return "High "
    elif avg_up < 0.35:
        return "Low "
    elif avg_heart > 0.12:
        return "Heart "
    elif avg_heart < 0.06:
        return "Nibbler "
    return ""


def generate_short_name(row: pd.Series, full_name: str) -> str:
    """Create a punchy 2-4 word label. Every cluster MUST get a unique name."""
    t = _score_traits(row)
    role = _role_short(t["is_sp"])
    zp = _zone_prefix(t)

    # Ordered by most exotic/distinctive pitch trait first
    if t["sv"] > 0.10:
        return f"{zp}Screwball Unicorn {role}"
    if t["kn"] > 0.10:
        return f"{zp}Knuckleball Wizard {role}"
    if t["fs"] > 0.15:
        return f"{zp}Splitter Assassin {role}"
    if t["kc"] > 0.15:
        return f"{zp}Knuckle-Curve Sorcerer {role}"
    if t["st"] > 0.20:
        return f"{zp}Sweeper Merchant {role}"
    if t["ch"] > 0.20:
        return f"{zp}Circle-Change Phantom {role}"
    if t["si"] > 0.40 and t["ff"] < 0.05:
        return f"{zp}Sinker Ghost {role}"
    if t["si"] > 0.35 and t["fc"] > 0.15:
        return f"{zp}Sinker-Cutter {role}"
    if t["si"] > 0.35 and t["sl"] > 0.18:
        return f"{zp}Earthworm {role}"
    if t["si"] > 0.35:
        return f"{zp}Sinker Savant {role}"
    if t["ff"] > 0.45 and t["sl"] > 0.30:
        return f"{zp}Gas & Snap {role}"
    if t["ff"] > 0.40 and t["sl"] > 0.15 and t["cu"] > 0.10:
        return f"{zp}Triple Threat {role}"
    if t["cu"] > 0.15 and t["fc"] > 0.12:
        return f"{zp}Cutter-Curve Craftsman {role}"
    if t["cu"] > 0.12 and t["is_sp"] > 0.55:
        return f"{zp}Uncle Charlie {role}"
    if t["cu"] > 0.12:
        return f"{zp}Yakker {role}"
    if t["fc"] > 0.15:
        return f"{zp}Cutter Carver {role}"
    if t["ff"] + t["si"] > 0.55 and t["whiff"] > 0.25:
        return f"{zp}Flamethrower {role}"
    if t["ff"] + t["si"] > 0.55:
        return f"{zp}Heater-Heavy {role}"
    if t["ch"] + t["fs"] > 0.15:
        return f"{zp}Offspeed Wizard {role}"

    return f"{zp}Kitchen Sink {role}"


def find_nearest_pitchers(
    pitcher_seasons: pd.DataFrame,
    cluster_id_str: str,
    n: int = 3,
) -> list:
    """Find the n pitchers closest to the cluster centroid using PCA distance."""
    mask = pitcher_seasons["cluster"] == cluster_id_str
    if mask.sum() == 0:
        return []

    subset = pitcher_seasons[mask]
    # Use PCA coords as proxy for centroid distance
    cx = subset["pca_x"].mean()
    cy = subset["pca_y"].mean()
    cz = subset["pca_z"].mean() if "pca_z" in subset.columns else 0
    dists = np.sqrt((subset["pca_x"] - cx)**2 + (subset["pca_y"] - cy)**2 +
                    ((subset["pca_z"] - cz)**2 if "pca_z" in subset.columns else 0))
    nearest = dists.nsmallest(n).index

    examples = []
    for idx in nearest:
        row = pitcher_seasons.loc[idx]
        name = row.get("player_name", "Unknown")
        year = int(row.get("game_year", 0))
        examples.append(f"{name} ({year})")
    return examples


# Separate color palettes for RHP (warm) and LHP (cool)
RHP_COLORS = [
    "#e63946", "#f4a261", "#e9c46a", "#d62828", "#f77f00",
    "#bc6c25", "#ff6b6b", "#ffa07a", "#ff4500", "#ff8c00",
    "#c1292e", "#d4a053", "#e07840", "#b8860b", "#cd5c5c",
]
LHP_COLORS = [
    "#457b9d", "#2a9d8f", "#264653", "#6a4c93", "#1d3557",
    "#a8dadc", "#588157", "#606c38", "#023047", "#48cae4",
    "#5f9ea0", "#4682b4", "#6495ed", "#7b68ee", "#3cb371",
]


def main():
    # Load data
    data_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    pitcher_seasons = pd.read_parquet(data_path)

    centroids_csv = os.path.join(MODELS_DIR, "centroids.csv")
    centroid_df = pd.read_csv(centroids_csv, index_col=0)

    centroid_pca_path = os.path.join(MODELS_DIR, "centroid_pca.csv")
    centroid_pca = pd.read_csv(centroid_pca_path, index_col=0) if os.path.exists(centroid_pca_path) else None

    # Load meta
    meta_path = os.path.join(MODELS_DIR, "kmeans_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    features_used = meta.get("features", CLUSTER_FEATURES)

    cluster_ids = list(centroid_df.index)
    print(f"Naming {len(cluster_ids)} clusters ({sum(1 for c in cluster_ids if c.startswith('R'))} RHP, "
          f"{sum(1 for c in cluster_ids if c.startswith('L'))} LHP)...\n")

    profiles = {}
    rhp_idx = 0
    lhp_idx = 0

    for cid in cluster_ids:
        row = centroid_df.loc[cid]
        is_rhp = cid.startswith("R")
        hand = "RHP" if is_rhp else "LHP"

        # Pick color from the appropriate palette
        if is_rhp:
            color = RHP_COLORS[rhp_idx % len(RHP_COLORS)]
            rhp_idx += 1
        else:
            color = LHP_COLORS[lhp_idx % len(LHP_COLORS)]
            lhp_idx += 1

        full_name = generate_full_name(row)
        short_name = generate_short_name(row, full_name)
        examples = find_nearest_pitchers(pitcher_seasons, cid, n=3)
        count = int((pitcher_seasons["cluster"] == cid).sum())

        # PCA centroid position
        pca_pos = {}
        if centroid_pca is not None and cid in centroid_pca.index:
            pca_row = centroid_pca.loc[cid]
            pca_pos = {
                "pca_x": round(float(pca_row["centroid_pca_x"]), 4),
                "pca_y": round(float(pca_row["centroid_pca_y"]), 4),
                "pca_z": round(float(pca_row.get("centroid_pca_z", 0)), 4),
            }

        # Build centroid dict from clustering features
        centroid_dict = {col: round(float(row[col]), 4) for col in features_used if col in row.index}

        # Add movement data (not a clustering feature, but needed for display axes)
        cluster_mask = pitcher_seasons["cluster"] == cid
        for extra_col in ["pfx_x_avg", "pfx_z_avg"]:
            if extra_col in pitcher_seasons.columns and cluster_mask.sum() > 0:
                centroid_dict[extra_col] = round(float(pitcher_seasons.loc[cluster_mask, extra_col].mean()), 4)

        profiles[cid] = {
            "full_name": full_name,
            "short_name": short_name,
            "color": color,
            "hand": hand,
            "pitcher_count": count,
            "example_pitchers": examples,
            "centroid": centroid_dict,
            **pca_pos,
        }

        print(f"  {cid} [{hand}]: {short_name}")
        print(f"    Full: {full_name}")
        print(f"    Examples: {', '.join(examples)}")
        print(f"    Count: {count}")
        print()

    # Save profiles
    out_path = os.path.join(MODELS_DIR, "cluster_profiles.json")
    with open(out_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved: {out_path}")

    # Also export for frontend (src/data and public)
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "src", "data", "clusters.json",
    )
    os.makedirs(os.path.dirname(frontend_path), exist_ok=True)
    with open(frontend_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Frontend export: {frontend_path}")

    public_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "public", "clusters.json",
    )
    if os.path.isdir(os.path.dirname(public_path)):
        with open(public_path, "w") as f:
            json.dump(profiles, f, indent=2)
        print(f"Public export: {public_path}")


if __name__ == "__main__":
    main()
