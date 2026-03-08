#!/usr/bin/env python3
"""
Step 1b: Fetch MiLB (Minor League) Statcast pitch-level data from Baseball Savant.

Scrapes the Minor League Statcast Search CSV endpoint for AAA and Single-A data.
Saves to data/raw/milb_statcast_{year}.parquet with the same column schema as MLB data.

Usage:
    python pipeline/01b_fetch_milb_statcast.py
"""

import os
import sys
import time
import io
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, KEEP_COLUMNS, CURRENT_YEAR

# MiLB Statcast available from 2021 (Single-A FSL), full AAA from 2023
MILB_SEASONS = list(range(2023, CURRENT_YEAR + 1))

# Baseball Savant CSV endpoint for minor league data
SAVANT_MILB_URL = "https://baseballsavant.mlb.com/statcast-search-minors/csv"

# Month windows for MiLB seasons (April through September)
MILB_MONTH_RANGES = [
    ("04-01", "04-30"),
    ("05-01", "05-31"),
    ("06-01", "06-30"),
    ("07-01", "07-31"),
    ("08-01", "08-31"),
    ("09-01", "09-30"),
]

# Also fetch Spring Training data (March)
SPRING_MONTH_RANGES = [
    ("02-20", "03-31"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/csv,application/csv,*/*",
}


def fetch_milb_chunk(start_dt: str, end_dt: str, level: str = "aaa") -> pd.DataFrame:
    """Fetch a single chunk of MiLB Statcast data from Baseball Savant."""
    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",  # Regular season
        "hfPR": "",
        "hfZ": "",
        "hfStadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "hfSit": "",
        "player_type": "pitcher",
        "hfOuts": "",
        "hfOpponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_dt,
        "game_date_lt": end_dt,
        "hfMo": "",
        "hfTeam": "",
        "home_road": "",
        "hfRO": "",
        "position": "",
        "hfInfield": "",
        "hfOutfield": "",
        "hfInn": "",
        "hfBBT": "",
        "hfFlag": "",
        "metric_1": "",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "min_pas": "0",
        "chk_stats_sweetspot_speed": "on",
        "chk_stats_velocity": "on",
        "chk_stats_spin_rate": "on",
        "type": "details",
    }

    # Set level filter
    if level == "aaa":
        params["hfLevel"] = "AAA|"
    elif level == "a":
        params["hfLevel"] = "A|"
    else:
        params["hfLevel"] = ""  # All levels

    try:
        resp = requests.get(SAVANT_MILB_URL, params=params, headers=HEADERS, timeout=120)
        resp.raise_for_status()

        if len(resp.content) < 100:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


def fetch_milb_season(year: int) -> pd.DataFrame:
    """Fetch all MiLB Statcast data for a single year."""
    chunks = []

    # Regular season months
    for start_md, end_md in MILB_MONTH_RANGES:
        start_dt = f"{year}-{start_md}"
        end_dt = f"{year}-{end_md}"

        for level in ["aaa", "a"]:
            print(f"  Fetching MiLB {level.upper()} {start_dt} to {end_dt} ...")

            for attempt in range(3):
                df = fetch_milb_chunk(start_dt, end_dt, level=level)
                if len(df) > 0 or attempt == 2:
                    break
                wait = 15 * (attempt + 1)
                print(f"    Retry {attempt+1}/3, waiting {wait}s...")
                time.sleep(wait)

            if len(df) > 0:
                chunks.append(df)
                print(f"    -> {len(df):,} pitches")
            else:
                print(f"    -> 0 pitches")

            time.sleep(3)  # polite pause

    # Spring Training data (current year only)
    if year == CURRENT_YEAR:
        for start_md, end_md in SPRING_MONTH_RANGES:
            start_dt = f"{year}-{start_md}"
            end_dt = f"{year}-{end_md}"

            for level in ["aaa", "a"]:
                print(f"  Fetching MiLB {level.upper()} Spring {start_dt} to {end_dt} ...")
                df = fetch_milb_chunk(start_dt, end_dt, level=level)
                if len(df) > 0:
                    chunks.append(df)
                    print(f"    -> {len(df):,} pitches")
                else:
                    print(f"    -> 0 pitches")
                time.sleep(3)

    if not chunks:
        print(f"  No MiLB data found for {year}")
        return pd.DataFrame()

    season_df = pd.concat(chunks, ignore_index=True)

    # Drop duplicates (overlapping date ranges or level coverage)
    if "game_pk" in season_df.columns and "at_bat_number" in season_df.columns:
        season_df = season_df.drop_duplicates(
            subset=["game_pk", "at_bat_number", "pitch_number"], keep="first"
        )

    # Keep only columns we need (same as MLB pipeline)
    available_cols = [c for c in KEEP_COLUMNS if c in season_df.columns]
    season_df = season_df[available_cols].copy()

    # Ensure game_year column exists
    if "game_year" not in season_df.columns:
        if "game_date" in season_df.columns:
            season_df["game_year"] = pd.to_datetime(season_df["game_date"]).dt.year
        else:
            season_df["game_year"] = year

    print(f"  {year} MiLB total: {len(season_df):,} pitches")
    return season_df


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    for year in MILB_SEASONS:
        out_path = os.path.join(RAW_DATA_DIR, f"milb_statcast_{year}.parquet")

        # Skip complete past seasons
        if os.path.exists(out_path) and year < CURRENT_YEAR:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"[MiLB {year}] Already exists ({size_mb:.1f} MB) — skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Fetching MiLB {year}...")
        print(f"{'='*50}")

        season_df = fetch_milb_season(year)

        if len(season_df) == 0:
            print(f"[MiLB {year}] No data — skipping save")
            continue

        season_df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[MiLB {year}] Saved: {out_path} ({size_mb:.1f} MB, {len(season_df):,} rows)")

    print("\nMiLB Statcast fetch complete.")


if __name__ == "__main__":
    main()
