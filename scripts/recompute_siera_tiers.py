#!/usr/bin/env python3
"""
Recompute pitcher_tiers.json z_scores from SIERA instead of whiff_rate.

SIERA is inverted: lower is better, so z_score = (mean - pitcher) / stdev
This means a pitcher with SIERA well below cluster mean gets a positive z_score.
"""

import json
import statistics
from pathlib import Path

TIERS_PATH = Path(__file__).resolve().parent.parent / "frontend" / "public" / "pitcher_tiers.json"

TIER_THRESHOLDS = [
    ("T1_Apex",     1.0,  0.87),
    ("T2_Core",     0.3,  0.95),
    ("T3_Standard", -0.3, 1.00),
    ("T4_Fringe",   None, 1.11),
]

def assign_tier(z):
    for tier_name, threshold, mult in TIER_THRESHOLDS:
        if threshold is None or z >= threshold:
            return tier_name, mult
    # fallback
    return "T4_Fringe", 1.11

def main():
    with open(TIERS_PATH) as f:
        data = json.load(f)

    # Group by cluster
    clusters = {}
    for key, entry in data.items():
        c = entry["cluster"]
        clusters.setdefault(c, []).append(key)

    # Recompute z_scores per cluster
    for cluster, keys in clusters.items():
        siera_vals = [data[k]["siera"] for k in keys]
        mean_s = statistics.mean(siera_vals)
        stdev_s = statistics.pstdev(siera_vals)  # population stdev

        for k in keys:
            entry = data[k]
            if stdev_s > 0:
                # INVERTED: lower SIERA = higher (positive) z_score
                z = (mean_s - entry["siera"]) / stdev_s
            else:
                z = 0.0

            z = round(z, 4)
            tier_name, base_mult = assign_tier(z)

            entry["z_score"] = z
            entry["tier"] = tier_name
            entry["base_multiplier"] = base_mult

            # Recalculate effective_multiplier (base * archetype_dominance adjustment)
            ad = entry.get("archetype_dominance", 1.0)
            # effective = base_mult adjusted by archetype_dominance
            # From the original data pattern: effective ≈ base * ad  (roughly)
            # But looking at examples: 0.87 * 1.01 = 0.8787 vs actual 0.8687
            # Actually: effective = base_mult - (1 - ad) * (1 - base_mult) ... let me check
            # 0.87, ad=1.01 -> eff=0.8687  => 0.87 - 0.01*(1-0.87) = 0.87 - 0.0013 = 0.8687 ✓
            # 1.11, ad=1.01 -> eff=1.1111  => 1.11 - 0.01*(1-1.11) = 1.11 + 0.0011 = 1.1111 ✓
            # 0.95, ad=0.9908 -> eff=0.9505 => 0.95 - (-0.0092)*(1-0.95) = 0.95 + 0.00046 = 0.9505 ✓
            # 0.87, ad=1.1079 -> eff=0.856  => 0.87 - 0.1079*(1-0.87) = 0.87 - 0.01403 = 0.856 ✓
            # Formula: effective = base - (ad - 1) * (1 - base)
            effective = base_mult - (ad - 1.0) * (1.0 - base_mult)
            entry["effective_multiplier"] = round(effective, 4)

    with open(TIERS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Print verification
    print("=== Verification ===")
    for key_suffix, label in [("694973_2024", "Skenes 2024"), ("694973_2025", "Skenes 2025")]:
        if key_suffix in data:
            e = data[key_suffix]
            print(f"{label}: SIERA={e['siera']}, z={e['z_score']}, tier={e['tier']}, eff_mult={e['effective_multiplier']}")

    # Webb entries
    for k, e in data.items():
        if "Webb, Logan" in e.get("player_name", "") and e["game_year"] >= 2023:
            print(f"Webb {e['game_year']}: SIERA={e['siera']}, z={e['z_score']}, tier={e['tier']}, eff_mult={e['effective_multiplier']}")

    # Tier distribution
    from collections import Counter
    tier_counts = Counter(e["tier"] for e in data.values())
    print(f"\nTier distribution: {dict(tier_counts)}")

if __name__ == "__main__":
    main()
