"""
Step 5: Generate quirky archetype names for each K-Means cluster.

Uses the geometric medoid (real pitcher closest to center) for each cluster,
applies rule-based naming examining dominant traits, and saves cluster_profiles.json.
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
    """Return the top n pitch types by usage from the medoid pitcher."""
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
    """Score how distinctive each trait is for this cluster's medoid pitcher."""
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
        "is_rhp": row.get("is_rhp", 0),
        "is_sp": row.get("is_sp", 0),
        "spin": row.get("spin_overall", 0),
        "ext": row.get("avg_extension", 0),
    }


def generate_full_name(row: pd.Series, hand: str = "RHP") -> str:
    """Generate a descriptive archetype name from medoid pitcher feature values.

    Uses a layered approach: primary identity (most distinctive pitch trait),
    then secondary modifiers (outcome, velocity, groundball tendency).
    """
    t = _score_traits(row)
    role = _role_str(t["is_sp"])
    parts = ["The"]

    # --- Primary identity: what makes this cluster's PITCH MIX unique ---
    if t["kn"] > 0.10:
        parts.append("Knuckleball Wizard")
    elif t["fs"] > 0.15:
        parts.append("Split Demon")
    elif t["kc"] > 0.15:
        parts.append("Uncle Charlie Knuckle-Curve")
    # Undertow BEFORE Boomerang — no-FF sinker identity takes priority
    elif t["si"] > 0.35 and t["ff"] < 0.05:
        parts.append("Undertow Pure Sinker")
    elif t["st"] > 0.20:
        parts.append("Boomerang Sweeper")
    elif t["ch"] > 0.20:
        parts.append("Ghost Changeup")
    elif t["si"] > 0.35 and t["fc"] > 0.15:
        parts.append("Sinker-Cutter Hybrid")
    elif t["si"] > 0.35 and t["sl"] > 0.18:
        if hand == "LHP":
            parts.append("Gardener Sinker-Slider")
        else:
            parts.append("Snake Sinker-Slider")
    elif t["si"] > 0.35:
        parts.append("Wormburner Sinker")
    elif t["ff"] > 0.45 and t["sl"] > 0.30:
        parts.append("Barnburner Two-Pitch Demon")
    elif t["ff"] > 0.40 and t["sl"] > 0.15 and t["cu"] > 0.10:
        parts.append("Triple Threat")
    elif t["cu"] > 0.15 and t["fc"] > 0.12:
        parts.append("Cutter-Curve Craftsman")
    elif t["cu"] > 0.12 and t["is_sp"] > 0.55:
        parts.append("Uncle Charlie")
    elif t["cu"] > 0.12:
        parts.append("Yakker Specialist")
    elif t["fc"] > 0.15:
        parts.append("Cutman Specialist")
    elif t["ff"] + t["si"] > 0.55 and t["whiff"] > 0.25:
        parts.append("Barnburner Flamethrower")
    elif t["ff"] + t["si"] > 0.55:
        parts.append("Heater-Heavy")
    else:
        parts.append("Kitchen Sink Illusionist")

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


def generate_short_name(row: pd.Series, full_name: str, hand: str = "RHP") -> str:
    """Create a punchy 2-4 word label. Every cluster MUST get a unique name."""
    t = _score_traits(row)
    role = _role_short(t["is_sp"])

    # Ordered by most exotic/distinctive pitch trait first
    if t["kn"] > 0.10:
        return f"Knuckleball Wizard {role}"
    if t["fs"] > 0.15:
        return f"Split Demon {role}"
    if t["kc"] > 0.15:
        return f"Uncle Charlie {role}"
    # Undertow BEFORE Boomerang — no-FF sinker identity takes priority
    if t["si"] > 0.35 and t["ff"] < 0.05:
        return f"Undertow {role}"
    if t["st"] > 0.20:
        return f"Boomerang {role}"
    if t["ch"] > 0.20:
        return f"Ghost {role}"
    if t["si"] > 0.35 and t["fc"] > 0.15:
        return f"Sinker-Cutter {role}"
    if t["si"] > 0.35 and t["sl"] > 0.18:
        if hand == "LHP":
            return f"Gardener {role}"
        return f"Snake {role}"
    if t["si"] > 0.35:
        return f"Wormburner {role}"
    if t["ff"] > 0.45 and t["sl"] > 0.30:
        return f"Barnburner {role}"
    if t["ff"] > 0.40 and t["sl"] > 0.15 and t["cu"] > 0.10:
        return f"Triple Threat {role}"
    if t["cu"] > 0.15 and t["fc"] > 0.12:
        return f"Cutter-Curve Craftsman {role}"
    if t["cu"] > 0.12 and t["is_sp"] > 0.55:
        return f"Uncle Charlie {role}"
    if t["cu"] > 0.12:
        return f"Yakker {role}"
    if t["fc"] > 0.15:
        return f"Cutman {role}"
    if t["ff"] + t["si"] > 0.55 and t["whiff"] > 0.25:
        return f"Barnburner {role}"
    if t["ff"] + t["si"] > 0.55:
        return f"Heater-Heavy {role}"
    if t["ch"] + t["fs"] > 0.15:
        return f"Ghost {role}"

    return f"Kitchen Sink {role}"


def find_nearest_pitchers(
    pitcher_seasons: pd.DataFrame,
    cluster_id_str: str,
    n: int = 3,
) -> list:
    """Find the n pitchers closest to the cluster's geometric medoid in PCA space."""
    from scipy.spatial.distance import cdist

    mask = pitcher_seasons["cluster"] == cluster_id_str
    if mask.sum() == 0:
        return []

    subset = pitcher_seasons[mask]
    pca_cols = ["pca_x", "pca_y"] + (["pca_z"] if "pca_z" in subset.columns else [])
    coords = subset[pca_cols].fillna(0).values

    if len(coords) < 2:
        row = subset.iloc[0]
        return [f"{row.get('player_name', 'Unknown')} ({int(row.get('game_year', 0))})"]

    # True geometric medoid
    dist_matrix = cdist(coords, coords, metric='euclidean')
    total_dists = dist_matrix.sum(axis=1)
    medoid_local_idx = int(np.argmin(total_dists))

    # Distances from medoid
    dists_from_medoid = dist_matrix[medoid_local_idx]
    nearest_local = np.argsort(dists_from_medoid)[:n]

    examples = []
    for local_idx in nearest_local:
        row = subset.iloc[local_idx]
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
    from scipy.spatial.distance import cdist

    # Load data
    data_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_seasons.parquet")
    pitcher_seasons = pd.read_parquet(data_path)

    # Load meta
    meta_path = os.path.join(MODELS_DIR, "kmeans_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    features_used = meta.get("features", CLUSTER_FEATURES)

    # Build medoid rows: for each cluster, find the actual pitcher closest
    # to geometric medoid (minimizes total distance to all other members)
    cluster_ids = sorted(pitcher_seasons["cluster"].unique())
    medoid_rows = {}
    for cid in cluster_ids:
        members = pitcher_seasons[pitcher_seasons["cluster"] == cid]
        if len(members) == 0:
            continue
        feature_cols = [f for f in features_used if f != "is_rhp" and f in members.columns]
        coords = members[feature_cols].fillna(0).values
        if len(coords) < 2:
            medoid_rows[cid] = members.iloc[0]
        else:
            dist_matrix = cdist(coords, coords, metric='euclidean')
            total_dists = dist_matrix.sum(axis=1)
            medoid_rows[cid] = members.iloc[int(np.argmin(total_dists))]

    print(f"Naming {len(cluster_ids)} clusters ({sum(1 for c in cluster_ids if c.startswith('R'))} RHP, "
          f"{sum(1 for c in cluster_ids if c.startswith('L'))} LHP)...\n")

    profiles = {}
    rhp_idx = 0
    lhp_idx = 0

    for cid in cluster_ids:
        if cid not in medoid_rows:
            continue
        row = medoid_rows[cid]
        is_rhp = cid.startswith("R")
        hand = "RHP" if is_rhp else "LHP"

        # Pick color from the appropriate palette
        if is_rhp:
            color = RHP_COLORS[rhp_idx % len(RHP_COLORS)]
            rhp_idx += 1
        else:
            color = LHP_COLORS[lhp_idx % len(LHP_COLORS)]
            lhp_idx += 1

        full_name = generate_full_name(row, hand)
        short_name = generate_short_name(row, full_name, hand)
        examples = find_nearest_pitchers(pitcher_seasons, cid, n=3)
        count = int((pitcher_seasons["cluster"] == cid).sum())

        # Detect position-player junk clusters (avg < 100 pitches per season)
        cluster_members = pitcher_seasons[pitcher_seasons["cluster"] == cid]
        avg_pitches = cluster_members["total_pitches"].mean() if len(cluster_members) > 0 else 0
        if avg_pitches < 100:
            role = _role_short(row.get("is_sp", 0))
            short_name = f"Eephus Lobber {role}"
            full_name = f"The Eephus Lobber (Position Player) {_role_str(row.get('is_sp', 0))}"

        # PCA position from medoid pitcher (real pitcher, not phantom average)
        pca_pos = {
            "pca_x": round(float(row.get("pca_x", 0)), 4),
            "pca_y": round(float(row.get("pca_y", 0)), 4),
            "pca_z": round(float(row.get("pca_z", 0)), 4),
        }

        # Build representative dict from medoid pitcher's actual features
        rep_dict = {col: round(float(row.get(col, 0)), 4) for col in features_used if col in row.index}

        # Movement data from medoid pitcher (not averaged)
        for extra_col in ["pfx_x_avg", "pfx_z_avg"]:
            if extra_col in row.index:
                rep_dict[extra_col] = round(float(row.get(extra_col, 0)), 4)

        profiles[cid] = {
            "full_name": full_name,
            "short_name": short_name,
            "color": color,
            "hand": hand,
            "pitcher_count": count,
            "example_pitchers": examples,
            "representative": rep_dict,
            **pca_pos,
        }

        print(f"  {cid} [{hand}]: {short_name}")
        print(f"    Full: {full_name}")
        print(f"    Medoid: {row.get('player_name', '?')}")
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
