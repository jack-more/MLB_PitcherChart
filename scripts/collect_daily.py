#!/usr/bin/env python3
"""
Daily data collector for MLB SIM pipeline.

Three data sources merged into unified daily_lineups.json:
  1. RotoWire  — confirmed lineups + batting orders
  2. MLB Stats API — schedule, game_pk, MLBAM IDs, probable pitchers
  3. The Odds API — consensus spreads, totals, moneylines

Usage:
    python scripts/collect_daily.py [--date 2026-02-23]
"""

import json
import os
import sys
import time
import unicodedata
import argparse
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import (
    MLB_API_BASE,
    ROTOWIRE_URL, USER_AGENT, REQUEST_TIMEOUT,
    FRONTEND_PUBLIC, DAILY_DATA_DIR,
    DAILY_LINEUPS_FILE, NAME_ID_CACHE_FILE,
)
from scripts.mlb_teams import (
    TEAM_ID_TO_ABBR, ABBR_TO_TEAM_ID, normalize_abbr,
)


# ═══════════════════════════════════════════════════════════
# NAME → ID RESOLUTION
# ═══════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Normalize a player name for matching: lowercase, strip accents, remove suffixes."""
    # Strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def _build_name_index(pitcher_seasons, batters):
    """Build a name → MLBAM ID lookup index from ATLAS data.

    pitcher_seasons.json format: {"pitcher": 543037, "player_name": "Verlander, Justin", ...}
    batters.json format: {"batter": 592518, "batter_name": "manny machado", ...}

    Returns dict: normalized_name → MLBAM ID (int)
    """
    index = {}       # normalized_name → MLBAM ID
    year_track = {}  # normalized_name → most recent game_year

    # Pitchers: name format is "Last, First"
    for ps in pitcher_seasons:
        pid = ps.get("pitcher")
        raw = ps.get("player_name", "")
        if not pid or not raw:
            continue
        # "Verlander, Justin" → "justin verlander"
        parts = raw.split(", ", 1)
        if len(parts) == 2:
            first_last = f"{parts[1]} {parts[0]}"
        else:
            first_last = raw
        norm = _normalize_name(first_last)
        year = ps.get("game_year", 0)
        # Keep the entry with the most recent year
        if norm not in index or year > year_track.get(norm, 0):
            index[norm] = pid
            year_track[norm] = year
        # Also index as "last, first" normalized
        norm_lf = _normalize_name(raw)
        if norm_lf not in index:
            index[norm_lf] = pid

    # Batters: name format is "first last" (already)
    for b in batters:
        bid = b.get("batter")
        raw = b.get("batter_name", "")
        if not bid or not raw:
            continue
        norm = _normalize_name(raw)
        if norm not in index:
            index[norm] = bid
    return index


def _load_name_cache():
    """Load the cached name→ID mappings."""
    if os.path.exists(NAME_ID_CACHE_FILE):
        try:
            with open(NAME_ID_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_name_cache(cache):
    """Save name→ID cache to disk."""
    with open(NAME_ID_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def resolve_player_id(name: str, name_index: dict, cache: dict) -> int | None:
    """Resolve a player display name to MLBAM ID.

    Priority:
      1. Name cache (previous resolutions)
      2. ATLAS name index (pitcher_seasons + batters)
      3. MLB Stats API player search (fallback)
    """
    norm = _normalize_name(name)
    if not norm:
        return None

    # 1. Check cache
    if norm in cache:
        return cache[norm]

    # 2. Check ATLAS index
    if norm in name_index:
        cache[norm] = name_index[norm]
        return name_index[norm]

    # 3. Try partial match (last name only)
    last_name = norm.split()[-1] if norm.split() else norm
    candidates = [(k, v) for k, v in name_index.items() if k.endswith(last_name)]
    if len(candidates) == 1:
        cache[norm] = candidates[0][1]
        return candidates[0][1]

    # 4. MLB Stats API search
    try:
        url = f"{MLB_API_BASE}/people/search?names={name}&sportIds=1"
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=5)
        if resp.ok:
            data = resp.json()
            people = data.get("people", [])
            if people:
                pid = people[0].get("id")
                if pid:
                    cache[norm] = pid
                    return pid
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════
# SOURCE 1: ROTOWIRE SCRAPER
# ═══════════════════════════════════════════════════════════

def _parse_rotowire_odds(lineup_div, home_abbr: str, away_abbr: str) -> dict:
    """Extract odds from a RotoWire lineup card.

    RotoWire embeds composite lines in .lineup__odds-item spans:
      LINE: "BAL -140"  (favorite team + moneyline)
      O/U:  "9.5 Runs"  (total)

    Returns dict with spread, total, home_ml, away_ml (matching Odds API format).
    """
    import re

    odds = {}
    odds_div = lineup_div.select_one(".lineup__odds")
    if not odds_div:
        return odds

    for item in odds_div.select(".lineup__odds-item"):
        label_el = item.select_one("b")
        if not label_el:
            continue
        label = label_el.get_text(strip=True).upper()

        # Prefer composite (consensus) line, fall back to fanduel/betmgm
        value_text = ""
        for src in ["composite", "fanduel", "betmgm", "draftkings", "pointsbet"]:
            span = item.select_one(f"span.{src}")
            if span:
                t = span.get_text(strip=True)
                if t and t != "–" and t != "-":
                    value_text = t
                    break

        if not value_text:
            continue

        if label == "LINE":
            # Format: "BAL -140" or "NYY +120"
            m = re.match(r"([A-Z]{2,3})\s*([+-]?\d+)", value_text)
            if m:
                fav_team = normalize_abbr(m.group(1))
                fav_ml = int(m.group(2))
                # Derive the other side's moneyline (approximate)
                if fav_ml < 0:
                    dog_ml = int(abs(fav_ml) * 0.8) if abs(fav_ml) <= 150 else int(abs(fav_ml) * 0.7)
                else:
                    dog_ml = -int(fav_ml * 1.25) if fav_ml <= 150 else -int(fav_ml * 1.4)

                if fav_team == home_abbr:
                    odds["home_ml"] = fav_ml
                    odds["away_ml"] = dog_ml
                else:
                    odds["away_ml"] = fav_ml
                    odds["home_ml"] = dog_ml

                # Derive run-line spread from moneyline
                # Standard MLB run line is -1.5 for favorite
                if fav_team == home_abbr:
                    odds["spread"] = -1.5
                else:
                    odds["spread"] = 1.5

        elif label == "O/U":
            # Format: "9.5 Runs" or "8.5"
            m = re.search(r"([\d.]+)", value_text)
            if m:
                odds["total"] = float(m.group(1))

    return odds


def scrape_rotowire():
    """Scrape RotoWire daily lineups page.

    Returns list of game dicts with team names, batting orders, pitchers, and odds.
    """
    print("[RotoWire] Fetching daily lineups...")
    try:
        resp = requests.get(
            ROTOWIRE_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[RotoWire] Failed to fetch: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    games = []

    for lineup_div in soup.select("div.lineup.is-mlb"):
        game = {}

        # Teams
        visit_el = lineup_div.select_one(".lineup__team.is-visit .lineup__abbr")
        home_el = lineup_div.select_one(".lineup__team.is-home .lineup__abbr")
        if not visit_el or not home_el:
            continue
        game["away_team"] = normalize_abbr(visit_el.get_text(strip=True))
        game["home_team"] = normalize_abbr(home_el.get_text(strip=True))

        # Starting pitchers (each team side has a pitcher highlight)
        for side_class, key in [("is-visit", "away_pitcher"), ("is-home", "home_pitcher")]:
            # Try within the team div first
            team_div = lineup_div.select_one(f".lineup__team.{side_class}")
            pitcher_name = "TBD"
            pitcher_throws = "R"
            if team_div:
                p_el = team_div.select_one(".lineup__player-highlight-name a")
                if p_el:
                    pitcher_name = p_el.get("title", p_el.get_text(strip=True))
                t_el = team_div.select_one(".lineup__throws")
                if t_el:
                    pitcher_throws = t_el.get_text(strip=True).upper()
            game[key] = {"name": pitcher_name, "throws": pitcher_throws}

        # Batting orders (9 per team)
        for side_class, key in [("is-visit", "away_lineup"), ("is-home", "home_lineup")]:
            batters = []
            list_el = lineup_div.select_one(f".lineup__list.{side_class}")
            if list_el:
                for player_el in list_el.select(".lineup__player"):
                    pos_el = player_el.select_one(".lineup__pos")
                    name_el = player_el.select_one("a")
                    bats_el = player_el.select_one(".lineup__bats")
                    batters.append({
                        "name": name_el.get("title", name_el.get_text(strip=True)) if name_el else "",
                        "pos": pos_el.get_text(strip=True) if pos_el else "",
                        "bats": bats_el.get_text(strip=True) if bats_el else "",
                    })
            game[key] = batters[:9]

        # Lineup status
        status_text = ""
        for cls in ["lineup__status", "lineup__state"]:
            status_el = lineup_div.select_one(f".{cls}")
            if status_el:
                status_text = status_el.get_text(strip=True).lower()
                break
        if "confirm" in status_text:
            game["lineup_status"] = "confirmed"
        elif "expect" in status_text:
            game["lineup_status"] = "expected"
        else:
            game["lineup_status"] = "expected"

        # Game metadata
        time_el = lineup_div.select_one(".lineup__time")
        weather_el = lineup_div.select_one(".lineup__weather-text")
        umpire_el = lineup_div.select_one(".lineup__umpire a")
        game["game_time"] = time_el.get_text(strip=True) if time_el else ""
        game["weather"] = weather_el.get_text(strip=True) if weather_el else ""
        game["umpire"] = umpire_el.get_text(strip=True) if umpire_el else ""

        # ── Odds (from RotoWire composite line) ──
        game["odds"] = _parse_rotowire_odds(lineup_div, game["home_team"], game["away_team"])

        games.append(game)

    print(f"[RotoWire] Scraped {len(games)} games")
    return games


# ═══════════════════════════════════════════════════════════
# SOURCE 2: MLB STATS API
# ═══════════════════════════════════════════════════════════

def fetch_mlb_schedule(date_str: str):
    """Fetch MLB schedule with probable pitchers and lineups from Stats API.

    Args:
        date_str: "YYYY-MM-DD"

    Returns list of game dicts with IDs, game types, pitchers, lineups.
    """
    print(f"[MLB API] Fetching schedule for {date_str}...")

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "startDate": date_str,
        "endDate": date_str,
        "hydrate": "probablePitcher,lineups",
    }

    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[MLB API] Failed to fetch schedule: {e}")
        return []

    games = []
    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            away_team = game.get("teams", {}).get("away", {}).get("team", {})
            home_team = game.get("teams", {}).get("home", {}).get("team", {})

            # Resolve team abbreviations
            away_abbr = away_team.get("abbreviation", "")
            home_abbr = home_team.get("abbreviation", "")
            if not away_abbr:
                away_abbr = TEAM_ID_TO_ABBR.get(away_team.get("id"), "UNK")
            if not home_abbr:
                home_abbr = TEAM_ID_TO_ABBR.get(home_team.get("id"), "UNK")

            # Probable pitchers
            away_pp = game.get("teams", {}).get("away", {}).get("probablePitcher", {})
            home_pp = game.get("teams", {}).get("home", {}).get("probablePitcher", {})

            # Lineups (array of player objects when hydrated)
            away_lineup_ids = []
            home_lineup_ids = []
            lineups = game.get("lineups", {})
            if lineups:
                for p in lineups.get("awayPlayers", []):
                    pid = p.get("id") if isinstance(p, dict) else p
                    if pid:
                        away_lineup_ids.append(pid)
                for p in lineups.get("homePlayers", []):
                    pid = p.get("id") if isinstance(p, dict) else p
                    if pid:
                        home_lineup_ids.append(pid)

            parsed = {
                "game_pk": game.get("gamePk"),
                "game_type": game.get("gameType", "R"),  # S=Spring, R=Regular, W=WBC, etc.
                "away_team": normalize_abbr(away_abbr),
                "home_team": normalize_abbr(home_abbr),
                "game_time_utc": game.get("gameDate", ""),
                "venue": game.get("venue", {}).get("name", ""),
                "status": game.get("status", {}).get("detailedState", ""),
                "series_description": game.get("seriesDescription", ""),
                "away_pitcher": {
                    "id": away_pp.get("id"),
                    "name": away_pp.get("fullName", "TBD"),
                    "throws": away_pp.get("pitchHand", {}).get("code", "R"),
                } if away_pp else {"id": None, "name": "TBD", "throws": "R"},
                "home_pitcher": {
                    "id": home_pp.get("id"),
                    "name": home_pp.get("fullName", "TBD"),
                    "throws": home_pp.get("pitchHand", {}).get("code", "R"),
                } if home_pp else {"id": None, "name": "TBD", "throws": "R"},
                "away_lineup_ids": away_lineup_ids,
                "home_lineup_ids": home_lineup_ids,
            }
            games.append(parsed)

    print(f"[MLB API] Found {len(games)} games ({sum(1 for g in games if g['game_type'] == 'S')} spring training)")
    return games


# ═══════════════════════════════════════════════════════════
# MERGE ALL SOURCES
# ═══════════════════════════════════════════════════════════

def merge_sources(rotowire_games, mlb_games, name_index, name_cache):
    """Merge RotoWire and MLB API data into unified game objects.

    MLB API is the backbone (has IDs, game_pk, game_type).
    RotoWire supplements with confirmed batting orders and odds.
    """
    # Index RotoWire games by (away, home) for matching
    rw_index = {}
    for rw in rotowire_games:
        key = (rw["away_team"], rw["home_team"])
        rw_index[key] = rw

    merged = []
    for mlb_game in mlb_games:
        key = (mlb_game["away_team"], mlb_game["home_team"])
        rw = rw_index.get(key, {})

        game = {
            "game_pk": mlb_game["game_pk"],
            "game_type": mlb_game["game_type"],
            "is_spring_training": mlb_game["game_type"] == "S",
            "away_team": mlb_game["away_team"],
            "home_team": mlb_game["home_team"],
            "game_time_utc": mlb_game["game_time_utc"],
            "venue": mlb_game["venue"],
            "status": mlb_game["status"],
            "series_description": mlb_game.get("series_description", ""),
        }

        # ── Pitchers ──
        # MLB API has MLBAM IDs; RotoWire has display names
        for side in ["away", "home"]:
            mlb_p = mlb_game[f"{side}_pitcher"]
            rw_p = rw.get(f"{side}_pitcher", {})

            if mlb_p.get("id"):
                # MLB API has the ID — use it
                game[f"{side}_pitcher"] = mlb_p
            elif rw_p.get("name") and rw_p["name"] != "TBD":
                # Resolve RotoWire name to ID
                pid = resolve_player_id(rw_p["name"], name_index, name_cache)
                game[f"{side}_pitcher"] = {
                    "id": pid,
                    "name": rw_p["name"],
                    "throws": rw_p.get("throws", "R"),
                }
            else:
                game[f"{side}_pitcher"] = {"id": None, "name": "TBD", "throws": "R"}

        # ── Lineups ──
        # Prefer RotoWire (has batting order + positions) with ID resolution
        # Fall back to MLB API lineup IDs
        for side in ["away", "home"]:
            rw_lineup = rw.get(f"{side}_lineup", [])
            mlb_lineup_ids = mlb_game.get(f"{side}_lineup_ids", [])

            if rw_lineup and any(b.get("name") for b in rw_lineup):
                # RotoWire has lineup — resolve names to IDs
                lineup = []
                for batter in rw_lineup:
                    if not batter.get("name"):
                        continue
                    bid = resolve_player_id(batter["name"], name_index, name_cache)
                    lineup.append({
                        "id": bid,
                        "name": batter["name"],
                        "pos": batter.get("pos", ""),
                        "bats": batter.get("bats", ""),
                    })
                game[f"{side}_lineup"] = lineup
                game["lineup_status"] = rw.get("lineup_status", "expected")
            elif mlb_lineup_ids:
                # MLB API has lineup IDs (no batting order positions)
                lineup = []
                for bid in mlb_lineup_ids:
                    lineup.append({"id": bid, "name": "", "pos": "", "bats": ""})
                game[f"{side}_lineup"] = lineup
                game["lineup_status"] = "expected"
            else:
                game[f"{side}_lineup"] = []
                game["lineup_status"] = "tbd"

        # ── Odds (scraped from RotoWire) ──
        game["odds"] = rw.get("odds", {})

        # ── Metadata from RotoWire ──
        game["weather"] = rw.get("weather", "")
        game["umpire"] = rw.get("umpire", "")

        merged.append(game)

    # Add any RotoWire games not in MLB API (unlikely but handle it)
    mlb_keys = {(g["away_team"], g["home_team"]) for g in mlb_games}
    for rw in rotowire_games:
        key = (rw["away_team"], rw["home_team"])
        if key not in mlb_keys:
            print(f"[Merge] RotoWire game not in MLB API: {rw['away_team']} @ {rw['home_team']}")

    return merged


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Collect daily MLB data")
    parser.add_argument("--date", default=None, help="Date to collect (YYYY-MM-DD), default today")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  MLB DAILY COLLECTOR — {date_str}")
    print(f"{'='*60}\n")

    # Load ATLAS data for name resolution
    print("[Setup] Loading ATLAS data for name resolution...")
    ps_path = os.path.join(FRONTEND_PUBLIC, "pitcher_seasons.json")
    bat_path = os.path.join(FRONTEND_PUBLIC, "batters.json")

    try:
        with open(ps_path) as f:
            pitcher_seasons = json.load(f)
        with open(bat_path) as f:
            batters = json.load(f)
        print(f"[Setup] Loaded {len(pitcher_seasons)} pitcher-seasons, {len(batters)} batters")
    except Exception as e:
        print(f"[Setup] Failed to load ATLAS data: {e}")
        pitcher_seasons, batters = [], []

    name_index = _build_name_index(pitcher_seasons, batters)
    name_cache = _load_name_cache()
    print(f"[Setup] Name index: {len(name_index)} entries, cache: {len(name_cache)} entries")

    # ── Collect from all sources ──
    rotowire_games = scrape_rotowire()
    time.sleep(0.5)  # Rate limit
    mlb_games = fetch_mlb_schedule(date_str)

    # If MLB API found no games for today, try next few days (spring training gaps)
    if not mlb_games:
        print(f"[MLB API] No games on {date_str}, scanning next 7 days...")
        for delta in range(1, 8):
            check_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=delta)).strftime("%Y-%m-%d")
            mlb_games = fetch_mlb_schedule(check_date)
            if mlb_games:
                date_str = check_date
                print(f"[MLB API] Found games on {date_str}")
                break
            time.sleep(0.3)

    # ── Merge ──
    odds_count = sum(1 for g in rotowire_games if g.get("odds"))
    print(f"\n[Merge] Combining {len(rotowire_games)} RotoWire (w/ {odds_count} odds) + {len(mlb_games)} MLB API...")
    merged = merge_sources(rotowire_games, mlb_games, name_index, name_cache)

    # ── Save ──
    output = {
        "slate_date": date_str,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "game_count": len(merged),
        "spring_training_count": sum(1 for g in merged if g.get("is_spring_training")),
        "regular_season_count": sum(1 for g in merged if not g.get("is_spring_training")),
        "games": merged,
    }

    with open(DAILY_LINEUPS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] Wrote {len(merged)} games to {DAILY_LINEUPS_FILE}")

    # Save updated name cache
    _save_name_cache(name_cache)
    print(f"[Output] Name cache: {len(name_cache)} entries saved")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {date_str}")
    print(f"{'='*60}")
    for g in merged:
        st_tag = " [ST]" if g.get("is_spring_training") else ""
        lineup_tag = f"({g.get('lineup_status', '?')})"
        odds_tag = ""
        if g.get("odds"):
            sp = g["odds"].get("spread", "?")
            tot = g["odds"].get("total", "?")
            odds_tag = f" | spread {sp} / O/U {tot}"
        away_p = g.get("away_pitcher", {}).get("name", "TBD")
        home_p = g.get("home_pitcher", {}).get("name", "TBD")
        print(f"  {g['away_team']} @ {g['home_team']}{st_tag} — {away_p} vs {home_p} {lineup_tag}{odds_tag}")

    return output


if __name__ == "__main__":
    main()
