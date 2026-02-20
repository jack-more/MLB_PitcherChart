"""
Step 09: Compute hitter archetypes via rule-based profile classification.

For each batter-year, examines their wOBA performance across the 26 pitcher
archetypes (grouped into families: fastball, breaking, offspeed, movement)
and assigns a descriptive label like "Heater Hunter" or "Archetype-Proof".

No clustering needed — labels are derived from the shape of each batter's
matchup profile, considering same-side vs opposite-side splits.

Outputs:
  - frontend/public/hitter_archetypes.json
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, RANDOM_STATE, SEASONS,
    HITTER_MIN_PA_FOR_ARCHETYPE, HITTER_MIN_CLUSTER_COVERAGE,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PUBLIC = os.path.join(PROJECT_ROOT, "frontend", "public")


# ── Archetype definitions ──────────────────────────────────────────────
ARCHETYPE_DEFS = {
    "archetype_proof": {
        "label": "Archetype-Proof",
        "color": "#FFD700", "emoji": "\U0001f6e1\ufe0f",
        "description": "Elite against every pitcher type",
    },
    "heater_hunter": {
        "label": "Heater Hunter",
        "color": "#FF4500", "emoji": "\U0001f525",
        "description": "Destroys fastball-heavy pitchers",
    },
    "hook_smasher": {
        "label": "Hook Smasher",
        "color": "#8B5CF6", "emoji": "\U0001fa9d",
        "description": "Feasts on curveball and breaking ball pitchers",
    },
    "offspeed_assassin": {
        "label": "Offspeed Assassin",
        "color": "#06B6D4", "emoji": "\U0001f52b",
        "description": "Punishes changeup and splitter pitchers",
    },
    "tunnel_buster": {
        "label": "Tunnel Buster",
        "color": "#3B82F6", "emoji": "\U0001f513",
        "description": "Handles deceptive mixed arsenals",
    },
    "worm_burners_nightmare": {
        "label": "Worm Burner's Nightmare",
        "color": "#10B981", "emoji": "\U0001fab1",
        "description": "Destroys sinker and groundball pitchers",
    },
    "lefty_masher": {
        "label": "Lefty Masher",
        "color": "#22D3EE", "emoji": "\U0001f535",
        "description": "Dominates left-handed pitching",
    },
    "righty_crusher": {
        "label": "Righty Crusher",
        "color": "#F97316", "emoji": "\U0001f7e0",
        "description": "Dominates right-handed pitching",
    },
    "power_surge": {
        "label": "Power Surge",
        "color": "#DC2626", "emoji": "\u26a1",
        "description": "Explosive HR-driven damage",
    },
    "swiss_army_knife": {
        "label": "Swiss Army Knife",
        "color": "#14B8A6", "emoji": "\U0001fa93",
        "description": "Versatile hitter, above average across the board",
    },
    "matchup_roulette": {
        "label": "Matchup Roulette",
        "color": "#EC4899", "emoji": "\U0001f3b0",
        "description": "Wildly volatile across pitcher types",
    },
    "spin_dizzy": {
        "label": "Spin Dizzy",
        "color": "#EF4444", "emoji": "\U0001f635",
        "description": "Struggles against breaking stuff",
    },
    "gas_can": {
        "label": "Gas Can",
        "color": "#9CA3AF", "emoji": "\u26fd",
        "description": "Overpowered by fastball-dominant pitchers",
    },
    "chase_artist": {
        "label": "Chase Artist",
        "color": "#F59E0B", "emoji": "\U0001f3a3",
        "description": "Expands the zone against offspeed and movement",
    },
    "contact_king": {
        "label": "Contact King",
        "color": "#84CC16", "emoji": "\U0001f451",
        "description": "Consistent contact hitter, rarely overmatched",
    },
    "same_side_slayer": {
        "label": "Same-Side Slayer",
        "color": "#A855F7", "emoji": "\U0001f93a",
        "description": "Defies the platoon advantage — crushes same-side pitching",
    },
    "platoon_bat": {
        "label": "Platoon Bat",
        "color": "#FB923C", "emoji": "\U0001f3cf",
        "description": "Strong platoon profile — much better vs opposite hand",
    },
    "flatliner": {
        "label": "Flatliner",
        "color": "#6B7280", "emoji": "\U0001f4c9",
        "description": "Below average against everything",
    },
    "streaky": {
        "label": "Streaky",
        "color": "#78716C", "emoji": "\U0001f4ca",
        "description": "Inconsistent across pitcher types",
    },
    "breaking_ball_bandit": {
        "label": "Breaking Ball Bandit",
        "color": "#A855F7", "emoji": "\U0001f9b9",
        "description": "Exploits curveball and sweeper-heavy arsenals",
    },
}

# Add badge URLs
for key, d in ARCHETYPE_DEFS.items():
    label_enc = d["label"].replace(" ", "%20").replace("-", "--").replace("'", "%27")
    d["badge_url"] = (
        f"https://img.shields.io/badge/"
        f"{d['emoji']}-{label_enc}-{d['color'][1:]}?style=flat-square"
    )


def _load_cluster_families():
    """Group the 26 pitcher clusters into families by archetype name."""
    cpath = os.path.join(FRONTEND_PUBLIC, "clusters.json")
    with open(cpath) as f:
        clusters = json.load(f)

    id_to_name = {}
    for cid, info in clusters.items():
        sn = info.get("short_name", "")
        parts = sn.split()
        if len(parts) >= 3:
            name = " ".join(parts[1:-1])
        elif len(parts) == 2:
            name = parts[1]
        else:
            name = sn
        id_to_name[cid] = name

    families = {
        "fastball": set(), "breaking": set(),
        "offspeed": set(), "movement": set(),
        "lhp": set(), "rhp": set(),
    }
    fb_names = {"Heater-Heavy", "Barnburner", "Undertow", "Wormburner"}
    brk_names = {"Uncle Charlie", "Yakker", "Boomerang"}
    off_names = {"Ghost", "Split Demon"}

    for cid, name in id_to_name.items():
        if name in fb_names:
            families["fastball"].add(cid)
        elif name in brk_names:
            families["breaking"].add(cid)
        elif name in off_names:
            families["offspeed"].add(cid)
        else:
            families["movement"].add(cid)
        if cid.startswith("L_"):
            families["lhp"].add(cid)
        else:
            families["rhp"].add(cid)

    return families, id_to_name


def classify_batter(profile, stand):
    """Classify a batter-year into an archetype based on their profile.

    Args:
        profile: dict with keys:
            rel_fastball, rel_breaking, rel_offspeed, rel_movement,
            rel_lhp, rel_rhp, overall_woba, rel_variance, k_rate, hr_rate
        stand: 'L', 'R', or 'S' (switch)

    Returns:
        archetype key string
    """
    fb = profile["rel_fastball"]
    brk = profile["rel_breaking"]
    off = profile["rel_offspeed"]
    mov = profile["rel_movement"]
    lhp = profile["rel_lhp"]
    rhp = profile["rel_rhp"]
    woba = profile["overall_woba"]
    var = profile["rel_variance"]
    k_rate = profile["k_rate"]
    hr_rate = profile["hr_rate"]

    # Compute same-side vs opposite-side relative performance
    if stand == "L":
        same_side = lhp   # L vs LHP = same side
        opp_side = rhp     # L vs RHP = opposite
    elif stand == "R":
        same_side = rhp    # R vs RHP = same side
        opp_side = lhp     # R vs LHP = opposite
    else:
        same_side = min(lhp, rhp)  # switch hitters: less relevant
        opp_side = max(lhp, rhp)

    # Best/worst family
    fam_scores = {"fastball": fb, "breaking": brk, "offspeed": off, "movement": mov}
    best_fam = max(fam_scores, key=fam_scores.get)
    worst_fam = min(fam_scores, key=fam_scores.get)
    best_val = fam_scores[best_fam]
    worst_val = fam_scores[worst_fam]
    spread = best_val - worst_val

    # Priority-ordered classification rules

    # 1. Archetype-Proof: elite wOBA AND good everywhere (no family < 0.85)
    if woba >= 0.370 and worst_val >= 0.85 and var < 0.06:
        return "archetype_proof"

    # 2. Same-Side Slayer: crushes same-side pitching (defies platoon)
    if stand in ("L", "R") and same_side >= 1.1 and same_side > opp_side + 0.1:
        return "same_side_slayer"

    # 3. Platoon Bat: much better vs opposite hand
    if stand in ("L", "R") and opp_side >= 1.1 and opp_side > same_side + 0.15:
        return "platoon_bat"

    # 4. Heater Hunter: best vs fastball, clearly above others
    if best_fam == "fastball" and fb >= 1.1 and fb > brk + 0.1 and fb > off + 0.08:
        return "heater_hunter"

    # 5. Hook Smasher: best vs breaking balls
    if best_fam == "breaking" and brk >= 1.1 and brk > fb + 0.1:
        return "hook_smasher"

    # 6. Offspeed Assassin: best vs offspeed
    if best_fam == "offspeed" and off >= 1.1 and off > fb + 0.08 and off > brk + 0.08:
        return "offspeed_assassin"

    # 7. Tunnel Buster: best vs movement/mixed
    if best_fam == "movement" and mov >= 1.1 and mov > fb + 0.08:
        return "tunnel_buster"

    # 8. Breaking Ball Bandit: good vs breaking, decent overall
    if brk >= 1.05 and woba >= 0.310:
        return "breaking_ball_bandit"

    # 9. Worm Burner's Nightmare: good vs fastball AND movement (sinker guys)
    if fb >= 1.05 and mov >= 1.02 and woba >= 0.310:
        return "worm_burners_nightmare"

    # 10. Power Surge: high HR rate with good wOBA
    if hr_rate >= 0.045 and woba >= 0.340:
        return "power_surge"

    # 11. Swiss Army Knife: above average everywhere, balanced
    if woba >= 0.330 and worst_val >= 0.90 and spread < 0.2:
        return "swiss_army_knife"

    # 12. Lefty Masher (for switch/right hitters)
    if stand != "L" and lhp >= 1.12 and lhp > rhp + 0.12:
        return "lefty_masher"

    # 13. Righty Crusher (for switch/left hitters)
    if stand != "R" and rhp >= 1.12 and rhp > lhp + 0.12:
        return "righty_crusher"

    # 14. Matchup Roulette: high variance
    if var >= 0.06 and 0.28 <= woba <= 0.36:
        return "matchup_roulette"

    # 15. Contact King: decent wOBA, low K rate, consistent
    if woba >= 0.300 and k_rate <= 0.18 and var < 0.05:
        return "contact_king"

    # 16. Spin Dizzy: weak vs breaking
    if worst_fam == "breaking" and brk < 0.85 and woba < 0.340:
        return "spin_dizzy"

    # 17. Gas Can: weak vs velocity
    if worst_fam == "fastball" and fb < 0.85 and woba < 0.340:
        return "gas_can"

    # 18. Chase Artist: weak vs offspeed/movement
    if off < 0.90 and mov < 0.92 and woba < 0.330:
        return "chase_artist"

    # 19. Streaky: high variance, moderate level
    if var >= 0.04:
        return "streaky"

    # 20. Flatliner: below average
    if woba < 0.290:
        return "flatliner"

    # Fallback: Contact King (average, consistent)
    return "contact_king"


def main():
    print("Step 09: Hitter Archetypes (rule-based profile classification)\n")

    # ── Load data ──────────────────────────────────────────────────────
    parquet_path = os.path.join(PROCESSED_DATA_DIR, "hitter_vs_cluster.parquet")
    if not os.path.exists(parquet_path):
        print("ERROR: hitter_vs_cluster.parquet not found!")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows, {df['batter'].nunique():,} batters")

    families, id_to_name = _load_cluster_families()
    cluster_cols = sorted(df["cluster"].unique())
    print(f"Pitcher clusters: {len(cluster_cols)}")

    # ── Aggregate per batter-year-cluster ──────────────────────────────
    agg = df.groupby(["batter", "batter_name", "game_year", "cluster"]).agg(
        PA=("PA", "sum"),
        woba_sum=("woba_sum", "sum"),
        woba_denom_sum=("woba_denom_sum", "sum"),
        K=("K", "sum"),
        HR=("HR", "sum"),
    ).reset_index()
    agg["wOBA"] = agg["woba_sum"] / agg["woba_denom_sum"].clip(lower=0.01)

    # Determine batter stand (most common in the data)
    stand_mode = df.groupby("batter")["stand"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "R")
    # Check if batter bats both sides -> switch
    stand_counts = df.groupby("batter")["stand"].nunique()
    is_switch = stand_counts >= 2
    stand_map = {}
    for bid in stand_mode.index:
        if is_switch.get(bid, False):
            stand_map[bid] = "S"
        else:
            stand_map[bid] = stand_mode[bid]

    # Overall wOBA per batter-year
    overall = agg.groupby(["batter", "batter_name", "game_year"]).apply(
        lambda g: np.average(g["wOBA"], weights=g["PA"]) if g["PA"].sum() > 0 else 0.32
    ).reset_index(name="overall_woba")
    agg = agg.merge(overall, on=["batter", "batter_name", "game_year"])

    # Relative wOBA
    agg["rel_woba"] = agg["wOBA"] / agg["overall_woba"].clip(lower=0.1)
    agg["rel_woba"] = agg["rel_woba"].clip(0.2, 2.5)

    # Pivot
    rel_pivot = agg.pivot_table(
        index=["batter", "batter_name", "game_year"],
        columns="cluster", values="rel_woba", aggfunc="first",
    )
    pa_pivot = agg.pivot_table(
        index=["batter", "batter_name", "game_year"],
        columns="cluster", values="PA", aggfunc="sum", fill_value=0,
    )
    k_pivot = agg.pivot_table(
        index=["batter", "batter_name", "game_year"],
        columns="cluster", values="K", aggfunc="sum", fill_value=0,
    )
    hr_pivot = agg.pivot_table(
        index=["batter", "batter_name", "game_year"],
        columns="cluster", values="HR", aggfunc="sum", fill_value=0,
    )

    for c in cluster_cols:
        if c not in rel_pivot.columns:
            rel_pivot[c] = 1.0
        if c not in pa_pivot.columns:
            pa_pivot[c] = 0
    rel_pivot = rel_pivot[cluster_cols].fillna(1.0)
    pa_pivot = pa_pivot[cluster_cols]

    # Filter
    total_pa = pa_pivot.sum(axis=1)
    coverage = (pa_pivot >= 3).sum(axis=1)
    mask = (total_pa >= HITTER_MIN_PA_FOR_ARCHETYPE) & (coverage >= HITTER_MIN_CLUSTER_COVERAGE)
    print(f"Qualified batter-years: {mask.sum():,} / {len(rel_pivot):,}")

    rel_pivot = rel_pivot[mask]
    pa_pivot = pa_pivot[mask]
    k_pivot = k_pivot.reindex(rel_pivot.index).fillna(0)
    hr_pivot = hr_pivot.reindex(rel_pivot.index).fillna(0)
    overall_woba = overall.set_index(["batter", "batter_name", "game_year"])["overall_woba"]
    overall_woba = overall_woba.reindex(rel_pivot.index).fillna(0.32)
    total_pa = total_pa[mask]

    # ── Compute family profiles and classify each batter-year ──────────
    print("\nClassifying batter-years...")
    assignments = {}  # batter_id -> {year: archetype_key}
    vectors = {}      # batter_id -> {year: [rel_woba per cluster]}
    label_counts = {}

    rel_all = rel_pivot.values
    pa_all = pa_pivot.values

    for row_idx, (batter_id, batter_name, year) in enumerate(rel_pivot.index):
        bid = str(int(batter_id))
        yr = str(int(year))
        stand = stand_map.get(int(batter_id), "R")

        # PA-weighted family means
        profile = {}
        for fam_name, fam_cids in families.items():
            col_idxs = [cluster_cols.index(c) for c in cluster_cols if c in fam_cids]
            if col_idxs:
                fam_rel = rel_all[row_idx, col_idxs]
                fam_pa = pa_all[row_idx, col_idxs]
                fam_pa_sum = fam_pa.sum()
                if fam_pa_sum > 0:
                    profile[f"rel_{fam_name}"] = float((fam_rel * fam_pa).sum() / fam_pa_sum)
                else:
                    profile[f"rel_{fam_name}"] = 1.0
            else:
                profile[f"rel_{fam_name}"] = 1.0

        profile["overall_woba"] = float(overall_woba.iloc[row_idx])
        # Variance of relative profile
        pa_row = pa_all[row_idx]
        pa_sum = pa_row.sum()
        if pa_sum > 0:
            wt_mean = (rel_all[row_idx] * pa_row).sum() / pa_sum
            profile["rel_variance"] = float(((rel_all[row_idx] - wt_mean) ** 2 * pa_row).sum() / pa_sum)
        else:
            profile["rel_variance"] = 0.0
        profile["k_rate"] = float(k_pivot.iloc[row_idx].sum() / max(total_pa.iloc[row_idx], 1))
        profile["hr_rate"] = float(hr_pivot.iloc[row_idx].sum() / max(total_pa.iloc[row_idx], 1))

        arch_key = classify_batter(profile, stand)

        if bid not in assignments:
            assignments[bid] = {}
        assignments[bid][yr] = arch_key

        if bid not in vectors:
            vectors[bid] = {}
        vectors[bid][yr] = [round(float(v), 3) for v in rel_all[row_idx]]

        label_counts[arch_key] = label_counts.get(arch_key, 0) + 1

    # ── Print results ──────────────────────────────────────────────────
    print(f"\nAssigned {sum(label_counts.values()):,} batter-years across {len(label_counts)} archetypes:")
    for key, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        label = ARCHETYPE_DEFS[key]["label"]
        emoji = ARCHETYPE_DEFS[key]["emoji"]
        pct = count / sum(label_counts.values()) * 100
        print(f"  {emoji} {label:25s} {count:5,} ({pct:5.1f}%)")

    # ── Export JSON ────────────────────────────────────────────────────
    # Only include archetypes that were actually used
    used_archetypes = {k: v for k, v in ARCHETYPE_DEFS.items() if k in label_counts}

    output = {
        "archetypes": used_archetypes,
        "cluster_cols": cluster_cols,
        "assignments": assignments,
        "vectors": vectors,
    }

    out_path = os.path.join(FRONTEND_PUBLIC, "hitter_archetypes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    fsize = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSaved: {out_path} ({fsize:.1f} MB)")
    print(f"Batters: {len(assignments):,}")

    print("\nHitter archetype computation complete!")


if __name__ == "__main__":
    main()
