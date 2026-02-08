"""
Step 2: Classify each pitcher-season as SP (starter) or RP (reliever).

Primary source: FanGraphs pitching stats via pybaseball (GS/G ratio).
Fallback: derive from Statcast appearance data (first inning == 1).
"""

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEASONS, RAW_DATA_DIR, PROCESSED_DATA_DIR


def fetch_fangraphs_roles(seasons: list) -> pd.DataFrame:
    """Pull FanGraphs pitching stats and classify SP/RP by GS/G ratio."""
    from pybaseball import pitching_stats

    all_roles = []
    for year in seasons:
        print(f"  FanGraphs pitching stats for {year}...")
        try:
            fg = pitching_stats(year, year, qual=0)
            time.sleep(2)
        except Exception as e:
            print(f"    Failed for {year}: {e}")
            continue

        if fg is None or len(fg) == 0:
            continue

        fg = fg[["IDfg", "Name", "G", "GS"]].copy()
        fg["game_year"] = year
        fg["gs_ratio"] = fg["GS"] / fg["G"].clip(lower=1)
        fg["role"] = np.where(fg["gs_ratio"] >= 0.5, "SP", "RP")
        all_roles.append(fg)
        print(f"    -> {len(fg)} pitchers")

    if not all_roles:
        return pd.DataFrame()

    return pd.concat(all_roles, ignore_index=True)


def map_fangraphs_to_mlbam(fg_roles: pd.DataFrame, mlbam_ids: list) -> pd.DataFrame:
    """Map FanGraphs IDfg to Statcast MLBAM pitcher IDs."""
    from pybaseball import playerid_reverse_lookup

    print("  Mapping MLBAM IDs to FanGraphs IDs...")
    # Process in batches to avoid timeouts
    batch_size = 500
    id_maps = []
    for i in range(0, len(mlbam_ids), batch_size):
        batch = mlbam_ids[i:i + batch_size]
        try:
            result = playerid_reverse_lookup(batch, key_type="mlbam")
            id_maps.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"    Batch {i}-{i+batch_size} failed: {e}")

    if not id_maps:
        return pd.DataFrame()

    id_map = pd.concat(id_maps, ignore_index=True)

    # Join: id_map has key_mlbam and key_fangraphs
    merged = id_map[["key_mlbam", "key_fangraphs"]].merge(
        fg_roles[["IDfg", "game_year", "role", "G", "GS"]],
        left_on="key_fangraphs",
        right_on="IDfg",
        how="inner",
    )
    merged = merged.rename(columns={"key_mlbam": "pitcher"})
    merged = merged[["pitcher", "game_year", "role", "G", "GS"]].copy()

    return merged


def derive_roles_from_statcast(seasons: list) -> pd.DataFrame:
    """Fallback: derive SP/RP from Statcast pitch data (first inning appearances)."""
    all_derived = []
    for year in seasons:
        path = os.path.join(RAW_DATA_DIR, f"statcast_{year}.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path, columns=["pitcher", "game_pk", "inning"])
        appearances = df.groupby(["pitcher", "game_pk"]).agg(
            min_inning=("inning", "min")
        ).reset_index()
        appearances["is_start"] = (appearances["min_inning"] == 1).astype(int)

        role_df = appearances.groupby("pitcher").agg(
            games=("game_pk", "nunique"),
            starts=("is_start", "sum"),
        ).reset_index()
        role_df["game_year"] = year
        role_df["role"] = np.where(
            role_df["starts"] / role_df["games"].clip(lower=1) >= 0.5, "SP", "RP"
        )
        all_derived.append(role_df[["pitcher", "game_year", "role", "games", "starts"]])

    if not all_derived:
        return pd.DataFrame()

    return pd.concat(all_derived, ignore_index=True)


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Collect all unique MLBAM pitcher IDs from raw Statcast data
    all_pitcher_ids = set()
    for year in SEASONS:
        path = os.path.join(RAW_DATA_DIR, f"statcast_{year}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=["pitcher"])
            all_pitcher_ids.update(df["pitcher"].unique().tolist())

    print(f"Total unique pitcher MLBAM IDs: {len(all_pitcher_ids):,}")

    # Primary: FanGraphs roles
    print("\n--- FanGraphs role classification ---")
    fg_roles = fetch_fangraphs_roles(SEASONS)

    if len(fg_roles) > 0:
        roles = map_fangraphs_to_mlbam(fg_roles, list(all_pitcher_ids))
        print(f"FanGraphs-mapped roles: {len(roles):,} pitcher-seasons")
    else:
        roles = pd.DataFrame(columns=["pitcher", "game_year", "role"])

    # Fallback: Statcast-derived for any missing pitcher-seasons
    print("\n--- Statcast-derived roles (fallback) ---")
    derived = derive_roles_from_statcast(SEASONS)

    if len(derived) > 0:
        # Find pitcher-seasons NOT in FanGraphs roles
        if len(roles) > 0:
            roles_key = set(zip(roles["pitcher"], roles["game_year"]))
            missing_mask = ~derived.apply(
                lambda r: (r["pitcher"], r["game_year"]) in roles_key, axis=1
            )
            fallback = derived[missing_mask].copy()
        else:
            fallback = derived.copy()

        if len(fallback) > 0:
            fallback = fallback.rename(columns={"games": "G", "starts": "GS"})
            roles = pd.concat([roles, fallback[["pitcher", "game_year", "role", "G", "GS"]]], ignore_index=True)
            print(f"Added {len(fallback):,} fallback pitcher-seasons")

    # Ensure integer types
    roles["pitcher"] = roles["pitcher"].astype(int)
    roles["game_year"] = roles["game_year"].astype(int)

    out_path = os.path.join(PROCESSED_DATA_DIR, "pitcher_roles.parquet")
    roles.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\nSaved: {out_path} ({len(roles):,} rows)")
    print(f"  SP: {(roles['role']=='SP').sum():,}  |  RP: {(roles['role']=='RP').sum():,}")


if __name__ == "__main__":
    main()
