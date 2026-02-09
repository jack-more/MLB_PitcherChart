"""
Step 08 — Hitter Timing Archetypes

Computes per-batter-cluster timing patterns from raw Statcast data:
  - Slow Starter:   BA improves from 1st AB to 2nd/3rd+ AB (TTO splits)
  - Ambush Hitter:  Higher BA in first AB vs cluster
  - Streak Hitter:  High variance in per-game hit rates (feast or famine)
  - Steady Eddie:   Low variance in per-game hit rates (consistent)

Outputs: frontend/public/hitter_timing.json
"""

import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "frontend", "public")


def main():
    print("Loading raw Statcast data...")
    raw_path = os.path.join(BASE, "data", "statcast_raw.parquet")
    if not os.path.exists(raw_path):
        print("  statcast_raw.parquet not found — skipping timing analysis")
        # Write empty placeholder so frontend doesn't break
        with open(os.path.join(OUT, "hitter_timing.json"), "w") as f:
            json.dump({}, f)
        return

    df = pd.read_parquet(raw_path, columns=[
        "batter", "pitcher", "game_pk", "at_bat_number", "events",
        "game_year",
    ])

    # Load pitcher cluster assignments
    ps_path = os.path.join(OUT, "pitcher_seasons.json")
    with open(ps_path) as f:
        pitcher_seasons = json.load(f)

    # Build pitcher → cluster mapping (latest season)
    pitcher_cluster = {}
    for p in pitcher_seasons:
        pid = p["pitcher"]
        yr = p["game_year"]
        if pid not in pitcher_cluster or yr > pitcher_cluster[pid][1]:
            pitcher_cluster[pid] = (p["cluster"], yr)
    pitcher_cluster = {k: v[0] for k, v in pitcher_cluster.items()}

    # Filter to PA-level rows (events not null)
    print("Filtering to plate appearances...")
    df = df.dropna(subset=["events"])
    df = df[df["events"] != ""]

    # Map pitcher → cluster
    df["cluster"] = df["pitcher"].map(pitcher_cluster)
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(str)

    # Determine if hit
    hit_events = {"single", "double", "triple", "home_run"}
    df["is_hit"] = df["events"].isin(hit_events).astype(int)

    # ── TTO (Times Through Order) ──
    # Within each (batter, game_pk), rank ABs to get TTO number
    print("Computing TTO splits...")
    df = df.sort_values(["batter", "game_pk", "at_bat_number"])
    df["tto"] = df.groupby(["batter", "game_pk"]).cumcount()  # 0-indexed
    df["tto_bin"] = df["tto"].clip(upper=2)  # 0, 1, 2+ (cap at 3rd TTO)

    # ── Per-batter-cluster TTO BA ──
    tto_stats = (
        df.groupby(["batter", "cluster", "tto_bin"])
        .agg(hits=("is_hit", "sum"), pa=("is_hit", "count"))
        .reset_index()
    )
    tto_stats["ba"] = tto_stats["hits"] / tto_stats["pa"].clip(lower=1)

    # ── Per-game hit rates (for variance/consistency) ──
    print("Computing per-game consistency...")
    game_rates = (
        df.groupby(["batter", "cluster", "game_pk"])
        .agg(hits=("is_hit", "sum"), pa=("is_hit", "count"))
        .reset_index()
    )
    game_rates["game_ba"] = game_rates["hits"] / game_rates["pa"].clip(lower=1)

    # Need at least 5 games vs a cluster to compute meaningful variance
    game_var = (
        game_rates.groupby(["batter", "cluster"])
        .agg(
            n_games=("game_ba", "count"),
            ba_var=("game_ba", "var"),
            ba_mean=("game_ba", "mean"),
        )
        .reset_index()
    )
    game_var = game_var[game_var["n_games"] >= 5]

    # ── Assign timing labels ──
    print("Assigning timing labels...")
    results = {}

    # Pivot TTO stats: batter-cluster → {tto0_ba, tto1_ba, tto2_ba}
    tto_pivot = tto_stats.pivot_table(
        index=["batter", "cluster"], columns="tto_bin", values="ba"
    ).reset_index()
    tto_pivot.columns = ["batter", "cluster"] + [
        f"tto{int(c)}_ba" if isinstance(c, (int, float)) else c
        for c in tto_pivot.columns[2:]
    ]

    # Merge TTO with variance
    merged = pd.merge(tto_pivot, game_var, on=["batter", "cluster"], how="outer")

    for _, row in merged.iterrows():
        batter = int(row["batter"]) if pd.notna(row["batter"]) else None
        cluster = row["cluster"] if pd.notna(row["cluster"]) else None
        if batter is None or cluster is None:
            continue

        key = f"{batter}_{cluster}"
        labels = []

        tto0 = row.get("tto0_ba", np.nan)
        tto1 = row.get("tto1_ba", np.nan)
        tto2 = row.get("tto2_ba", np.nan)

        # Ambush Hitter: 1st TTO BA > later TTOs by 30+ points
        if pd.notna(tto0) and pd.notna(tto1):
            if tto0 - tto1 > 0.030:
                labels.append("Ambush Hitter")

        # Slow Starter: later TTOs better than 1st by 30+ points
        later_avg = np.nanmean([tto1, tto2]) if pd.notna(tto1) else np.nan
        if pd.notna(tto0) and pd.notna(later_avg):
            if later_avg - tto0 > 0.030:
                labels.append("Slow Starter")

        # Streak/Steady: based on variance
        ba_var = row.get("ba_var", np.nan)
        if pd.notna(ba_var):
            if ba_var > 0.12:
                labels.append("Streak Hitter")
            elif ba_var < 0.06:
                labels.append("Steady Eddie")

        if labels:
            results[key] = {
                "labels": labels,
                "tto": [
                    round(tto0, 3) if pd.notna(tto0) else None,
                    round(tto1, 3) if pd.notna(tto1) else None,
                    round(tto2, 3) if pd.notna(tto2) else None,
                ],
                "var": round(ba_var, 4) if pd.notna(ba_var) else None,
                "games": int(row["n_games"]) if pd.notna(row.get("n_games")) else 0,
            }

    out_path = os.path.join(OUT, "hitter_timing.json")
    with open(out_path, "w") as f:
        json.dump(results, f, separators=(",", ":"))

    print(f"  Wrote {len(results)} batter-cluster timing profiles → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
