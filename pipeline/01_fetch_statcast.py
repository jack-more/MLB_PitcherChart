"""
Step 1: Fetch Statcast pitch-level data from Baseball Savant via pybaseball.

Pulls one month at a time per season to avoid rate limits, then concatenates
and saves as Parquet. Skips already-fetched complete seasons.
"""

import os
import sys
import time
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEASONS, MONTH_RANGES, RAW_DATA_DIR, CACHE_DIR,
    KEEP_COLUMNS, CURRENT_YEAR,
)


def fetch_season(year: int) -> pd.DataFrame:
    """Fetch all regular-season Statcast data for a single year."""
    from pybaseball import statcast, cache
    cache.enable()

    chunks = []
    for start_md, end_md in MONTH_RANGES:
        start_dt = f"{year}-{start_md}"
        end_dt = f"{year}-{end_md}"
        print(f"  Fetching {start_dt} to {end_dt} ...")

        retries = 3
        for attempt in range(retries):
            try:
                df = statcast(start_dt=start_dt, end_dt=end_dt)
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"  Retry {attempt+1}/{retries} after error: {e}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  FAILED after {retries} attempts for {start_dt}-{end_dt}: {e}")
                    df = pd.DataFrame()

        if len(df) > 0:
            chunks.append(df)
            print(f"    -> {len(df):,} pitches")
        else:
            print(f"    -> 0 pitches (off-season or error)")

        time.sleep(2)  # polite pause between requests

    if not chunks:
        print(f"  No data found for {year}")
        return pd.DataFrame()

    season_df = pd.concat(chunks, ignore_index=True)

    # Filter to regular season only
    if "game_type" in season_df.columns:
        season_df = season_df[season_df["game_type"] == "R"].copy()

    # Keep only the columns we need (skip any that don't exist in older data)
    available_cols = [c for c in KEEP_COLUMNS if c in season_df.columns]
    season_df = season_df[available_cols].copy()

    print(f"  {year} total: {len(season_df):,} regular-season pitches")
    return season_df


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    for year in SEASONS:
        out_path = os.path.join(RAW_DATA_DIR, f"statcast_{year}.parquet")

        # Skip complete seasons that are already downloaded
        if os.path.exists(out_path) and year < CURRENT_YEAR:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"[{year}] Already exists ({size_mb:.1f} MB) — skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Fetching {year}...")
        print(f"{'='*50}")

        season_df = fetch_season(year)

        if len(season_df) == 0:
            print(f"[{year}] No data — skipping save")
            continue

        season_df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[{year}] Saved: {out_path} ({size_mb:.1f} MB, {len(season_df):,} rows)")

    print("\nStatcast fetch complete.")


if __name__ == "__main__":
    main()
