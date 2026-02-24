#!/usr/bin/env python3
"""
Fetch per-batter stolen base rates and export for BaseRuns integration.

Pulls seasonal batting stats from FanGraphs via pybaseball, cross-references
to MLBAM IDs using the existing batters.json name index, and exports
batter_baserunning.json with SB/CS per-game rates.

Usage:
    python scripts/fetch_baserunning.py [--seasons 2023 2024 2025]
"""

import json
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEASONS as ALL_SEASONS

FRONTEND_PUBLIC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "public",
)


def normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    # Remove accents, periods, suffixes
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().strip()
    # Remove suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv", " v"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    # Remove periods
    name = name.replace(".", "")
    return name


def build_mlbam_name_index(batters_path: str) -> dict:
    """Build normalized name → MLBAM ID index from batters.json."""
    with open(batters_path) as f:
        batters = json.load(f)

    index = {}
    for b in batters:
        mlbam_id = b.get("batter")
        name = b.get("batter_name", "")
        if mlbam_id and name:
            norm = normalize_name(name)
            index[norm] = mlbam_id

            # Also index last_name, first_name format
            parts = name.strip().split()
            if len(parts) >= 2:
                # "first last" → also try "last, first"
                alt = normalize_name(f"{parts[-1]}, {' '.join(parts[:-1])}")
                index[alt] = mlbam_id

    return index


def fetch_batting_stats(seasons: list) -> list:
    """Fetch SB/CS data from FanGraphs for given seasons."""
    from pybaseball import batting_stats

    all_rows = []
    for year in seasons:
        print(f"  Fetching {year} batting stats from FanGraphs...")
        try:
            df = batting_stats(year, qual=0)  # qual=0 to get everyone
            if df is None or len(df) == 0:
                print(f"    No data for {year}")
                continue

            for _, row in df.iterrows():
                name = row.get("Name", "")
                g = row.get("G", 0) or 0
                pa = row.get("PA", 0) or 0
                sb = row.get("SB", 0) or 0
                cs = row.get("CS", 0) or 0

                if g < 10 or pa < 30:
                    continue  # Skip tiny samples

                all_rows.append({
                    "name": name,
                    "fangraphs_id": row.get("IDfg"),
                    "season": year,
                    "g": int(g),
                    "pa": int(pa),
                    "sb": int(sb),
                    "cs": int(cs),
                    "sb_per_game": round(sb / max(g, 1), 4),
                    "cs_per_game": round(cs / max(g, 1), 4),
                    "sb_per_pa": round(sb / max(pa, 1), 6),
                    "cs_per_pa": round(cs / max(pa, 1), 6),
                    "sb_rate": round(sb / max(sb + cs, 1), 4) if (sb + cs) > 0 else 0,
                })

            print(f"    {year}: {len([r for r in all_rows if r['season'] == year])} batters")
        except Exception as e:
            print(f"    Error fetching {year}: {e}")

    return all_rows


def match_to_mlbam(rows: list, name_index: dict) -> dict:
    """Match FanGraphs rows to MLBAM IDs and aggregate with recency weighting.

    Returns dict: mlbam_id → {sb_per_pa, cs_per_pa, sb_per_game, cs_per_game, ...}
    """
    RECENCY_WEIGHT = 2.0

    # Group by normalized name
    by_name = defaultdict(list)
    for row in rows:
        norm = normalize_name(row["name"])
        by_name[norm].append(row)

    matched = 0
    unmatched = 0
    result = {}

    for norm_name, player_rows in by_name.items():
        mlbam_id = name_index.get(norm_name)
        if mlbam_id is None:
            unmatched += 1
            continue

        matched += 1

        # Recency-weighted aggregate
        max_year = max(r["season"] for r in player_rows)
        total_weight = 0
        weighted_sb_pg = 0
        weighted_cs_pg = 0
        weighted_sb_ppa = 0
        weighted_cs_ppa = 0
        total_sb = 0
        total_cs = 0
        total_g = 0

        for r in player_rows:
            w = RECENCY_WEIGHT if r["season"] == max_year else 1.0
            total_weight += w
            weighted_sb_pg += r["sb_per_game"] * w
            weighted_cs_pg += r["cs_per_game"] * w
            weighted_sb_ppa += r["sb_per_pa"] * w
            weighted_cs_ppa += r["cs_per_pa"] * w
            total_sb += r["sb"]
            total_cs += r["cs"]
            total_g += r["g"]

        if total_weight > 0:
            result[mlbam_id] = {
                "sb_per_game": round(weighted_sb_pg / total_weight, 4),
                "cs_per_game": round(weighted_cs_pg / total_weight, 4),
                "sb_per_pa": round(weighted_sb_ppa / total_weight, 6),
                "cs_per_pa": round(weighted_cs_ppa / total_weight, 6),
                "career_sb": total_sb,
                "career_cs": total_cs,
                "career_g": total_g,
                "sb_pct": round(total_sb / max(total_sb + total_cs, 1), 3),
            }

    print(f"\n  Matched {matched} batters to MLBAM IDs, {unmatched} unmatched")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch batter baserunning data")
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=[s for s in ALL_SEASONS if s >= 2021],
        help="Seasons to fetch (default: 2021+)",
    )
    args = parser.parse_args()

    print("=== Fetching batter baserunning data ===")
    print(f"  Seasons: {args.seasons}")

    # Build MLBAM name index
    batters_path = os.path.join(FRONTEND_PUBLIC, "batters.json")
    if not os.path.exists(batters_path):
        print(f"ERROR: {batters_path} not found. Run pipeline first.")
        return

    name_index = build_mlbam_name_index(batters_path)
    print(f"  Name index: {len(name_index)} entries")

    # Fetch from FanGraphs
    rows = fetch_batting_stats(args.seasons)
    print(f"\n  Total rows fetched: {len(rows)}")

    # Match to MLBAM IDs
    baserunning = match_to_mlbam(rows, name_index)

    # Export
    out_path = os.path.join(FRONTEND_PUBLIC, "batter_baserunning.json")
    with open(out_path, "w") as f:
        # Convert int keys to strings for JSON
        json.dump({str(k): v for k, v in baserunning.items()}, f)
    print(f"\n  Saved: {out_path} ({len(baserunning)} batters)")

    # Stats
    sb_batters = [v for v in baserunning.values() if v["sb_per_game"] > 0.1]
    print(f"  Speed threats (>0.1 SB/game): {len(sb_batters)}")

    # Show top 10
    top = sorted(baserunning.items(), key=lambda x: x[1]["sb_per_game"], reverse=True)[:10]
    print("\n  Top 10 base stealers (SB/game):")
    for mlbam_id, data in top:
        print(f"    {mlbam_id}: {data['sb_per_game']:.3f} SB/g, "
              f"{data['cs_per_game']:.3f} CS/g, "
              f"{data['sb_pct']:.0%} success")

    print("\nDone!")


if __name__ == "__main__":
    main()
