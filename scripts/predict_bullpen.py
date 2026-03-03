#!/usr/bin/env python3
"""
Bullpen usage prediction for MLB SIM.

Three layers:
  1. Availability — who CAN pitch (rest, workload, consecutive days)
  2. Usage prediction — who WILL pitch and in what order
  3. Archetype matchup integration — reliever MS vs upcoming batters

Usage:
    python scripts/predict_bullpen.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import (
    MLB_API_BASE, USER_AGENT, FRONTEND_PUBLIC, DAILY_GAMES_FILE,
    SP_EXPECTED_IP, RP_EXPECTED_IP, ST_STARTER_CAP_IP,
    BULLPEN_HISTORY_FILE,
)
from scripts.mlb_teams import ABBR_TO_TEAM_ID
from scripts.compute_matchups import AtlasData, get_pitcher_cluster, compute_ms


# ═══════════════════════════════════════════════════════════
# BULLPEN HISTORY MANAGEMENT
# ═══════════════════════════════════════════════════════════

def load_bullpen_history() -> dict:
    """Load accumulated bullpen usage history.

    Format: { pitcher_id: [{"date": "2026-02-20", "pitches": 25, "ip": 1.0, "er": 0}] }
    """
    if os.path.exists(BULLPEN_HISTORY_FILE):
        try:
            with open(BULLPEN_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_bullpen_history(history: dict):
    """Save bullpen history to disk."""
    with open(BULLPEN_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def fetch_recent_pitcher_logs(pitcher_id: int, season: int = 2025) -> list:
    """Fetch recent game logs for a pitcher from MLB Stats API.

    Returns list of {date, pitches, ip, er, game_pk}
    """
    url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": season,
    }
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
        if not resp.ok:
            return []
        data = resp.json()

        logs = []
        for split_group in data.get("stats", []):
            for split in split_group.get("splits", []):
                stat = split.get("stat", {})
                logs.append({
                    "date": split.get("date", ""),
                    "pitches": stat.get("numberOfPitches", 0),
                    "ip": _parse_ip(stat.get("inningsPitched", "0")),
                    "er": stat.get("earnedRuns", 0),
                    "game_pk": split.get("game", {}).get("gamePk"),
                })
        return logs
    except Exception:
        return []


def _parse_ip(ip_str: str) -> float:
    """Parse innings pitched string like '6.2' → 6.667."""
    try:
        parts = str(ip_str).split(".")
        innings = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return innings + thirds / 3.0
    except (ValueError, IndexError):
        return 0.0


# ═══════════════════════════════════════════════════════════
# LAYER 1: AVAILABILITY
# ═══════════════════════════════════════════════════════════

def compute_availability(pitcher_id: int, date_str: str, history: dict) -> float:
    """Compute availability score (0.0-1.0) for a reliever.

    Factors:
      - Days since last appearance
      - Pitch count in last 3 appearances
      - Consecutive-day usage penalty
    """
    logs = history.get(str(pitcher_id), [])
    if not logs:
        return 1.0  # Fully rested (no history = available)

    today = datetime.strptime(date_str, "%Y-%m-%d")

    # Filter to last 7 days
    recent = []
    for log in logs:
        try:
            log_date = datetime.strptime(log["date"], "%Y-%m-%d")
            days_ago = (today - log_date).days
            if 0 < days_ago <= 7:
                recent.append({**log, "_days_ago": days_ago})
        except (ValueError, KeyError):
            continue

    if not recent:
        return 1.0

    # Days since last appearance
    min_days = min(r["_days_ago"] for r in recent)
    if min_days == 0:
        return 0.0  # Already pitched today

    rest_score = min(1.0, min_days / 2.0)

    # Workload: total pitches in last 3 appearances
    recent_sorted = sorted(recent, key=lambda r: r["_days_ago"])[:3]
    total_pitches = sum(r.get("pitches", 0) for r in recent_sorted)
    workload_penalty = max(0, (total_pitches - 60) / 100)  # Penalty above 60 pitches

    # Consecutive-day usage
    consec = sum(1 for r in recent if r["_days_ago"] <= 2)
    consec_penalty = 0.2 * max(0, consec - 1)

    return max(0.0, min(1.0, rest_score - workload_penalty - consec_penalty))


# ═══════════════════════════════════════════════════════════
# LAYER 2: USAGE PREDICTION
# ═══════════════════════════════════════════════════════════

def get_team_bullpen(team_abbr: str, atlas: AtlasData) -> list:
    """Get all relievers for a team with their archetype data.

    Uses teams_2026.json rosters + pitcher_seasons.json to identify relievers.
    """
    teams_path = os.path.join(FRONTEND_PUBLIC, "teams_2026.json")
    try:
        with open(teams_path) as f:
            teams_data = json.load(f)
    except Exception:
        return []

    roster_ids = teams_data.get("rosters", {}).get(team_abbr, [])
    relievers = []

    for pid in roster_ids:
        ps = atlas.ps_index.get(pid)
        if ps is None:
            continue
        # is_sp < 0.5 means predominantly reliever
        is_sp = ps.get("is_sp", 0)
        if is_sp >= 0.5:
            continue  # Skip starters

        cluster_id, archetype = get_pitcher_cluster(pid, atlas)
        relievers.append({
            "id": pid,
            "name": ps.get("player_name", "Unknown"),
            "cluster": cluster_id,
            "archetype": archetype,
            "is_sp": is_sp,
            "whiff_rate": ps.get("whiff_rate", 0),
            "groundball_rate": ps.get("groundball_rate", 0),
            "avg_velo_FF": ps.get("avg_velo_FF", 0),
            # Role heuristic: higher velo + higher whiff = later-inning reliever
            "role_score": (ps.get("avg_velo_FF", 90) - 90) * 0.3 + (ps.get("whiff_rate", 0.2) - 0.2) * 100,
        })

    # Sort by role_score descending (closer candidates first)
    relievers.sort(key=lambda r: r["role_score"], reverse=True)

    # Assign role labels
    for i, r in enumerate(relievers):
        if i == 0:
            r["role"] = "closer"
        elif i <= 2:
            r["role"] = "setup"
        elif i <= 4:
            r["role"] = "middle"
        else:
            r["role"] = "long"

    return relievers


def predict_game_bullpen(game: dict, atlas: AtlasData, history: dict) -> dict:
    """Predict bullpen deployment for both teams in a game.

    Returns dict with away_bullpen and home_bullpen predictions.
    """
    date_str = game.get("_slate_date", datetime.now().strftime("%Y-%m-%d"))
    is_st = game.get("is_spring_training", False)
    result = {}

    for side in ["away", "home"]:
        team_abbr = game[f"{side}_team"]
        sp = game.get(f"{side}_pitcher", {})
        sp_id = sp.get("id")

        # Estimate starter innings
        if sp_id:
            ps = atlas.ps_index.get(sp_id, {})
            sp_is_sp = ps.get("is_sp", 0.5)
            if sp_is_sp >= 0.7:
                expected_ip = ST_STARTER_CAP_IP if is_st else SP_EXPECTED_IP
            else:
                expected_ip = RP_EXPECTED_IP
        else:
            expected_ip = SP_EXPECTED_IP if not is_st else ST_STARTER_CAP_IP

        bullpen_ip_needed = max(0, 9 - expected_ip)

        # Get available relievers
        bp = get_team_bullpen(team_abbr, atlas)

        # Score availability
        for r in bp:
            r["availability"] = compute_availability(r["id"], date_str, history)

        available = [r for r in bp if r["availability"] > 0.3]

        # Predict deployment order
        # In a normal game: setup guys → closer
        # In a blowout: middle/long relievers
        predicted = []
        ip_covered = 0
        inning = int(expected_ip) + 1  # First bullpen inning

        for reliever in available:
            if ip_covered >= bullpen_ip_needed:
                break

            # Estimate innings for this reliever
            if reliever["role"] == "closer":
                est_ip = 1.0
            elif reliever["role"] == "setup":
                est_ip = 1.0
            elif reliever["role"] == "middle":
                est_ip = 1.5
            else:
                est_ip = 2.0

            predicted.append({
                "id": reliever["id"],
                "name": reliever["name"],
                "role": reliever["role"],
                "cluster": reliever["cluster"],
                "archetype": reliever["archetype"],
                "availability": round(reliever["availability"], 2),
                "expected_inning": inning,
                "expected_ip": round(est_ip, 1),
            })

            ip_covered += est_ip
            inning += int(est_ip)

        # Reverse order — closer comes last
        if len(predicted) > 1:
            # Swap closer to end
            closer_idx = next((i for i, p in enumerate(predicted) if p["role"] == "closer"), None)
            if closer_idx is not None and closer_idx < len(predicted) - 1:
                closer = predicted.pop(closer_idx)
                predicted.append(closer)

        result[f"{side}_bullpen"] = {
            "starter_expected_ip": round(expected_ip, 1),
            "bullpen_ip_needed": round(bullpen_ip_needed, 1),
            "available_count": len(available),
            "total_relievers": len(bp),
            "predicted": predicted,
            "confidence": "low" if is_st else ("medium" if len(available) < 4 else "high"),
        }

    return result


# ═══════════════════════════════════════════════════════════
# LAYER 3: ARCHETYPE MATCHUP INTEGRATION
# ═══════════════════════════════════════════════════════════

def score_bullpen_matchups(game: dict, atlas: AtlasData):
    """For each predicted reliever, compute MS against the batting order
    they're likely to face.
    """
    for side in ["away", "home"]:
        bp_data = game.get(f"{side}_bullpen", {})
        opp_side = "home" if side == "away" else "away"
        opp_lineup = game.get(f"{opp_side}_lineup", [])

        if not opp_lineup:
            continue

        for reliever in bp_data.get("predicted", []):
            cluster_id = reliever.get("cluster")
            if not cluster_id:
                continue

            # Score against each batter in opposing lineup
            ms_scores = []
            for batter in opp_lineup:
                bid = batter.get("id")
                if bid:
                    ms = compute_ms(bid, cluster_id, atlas)
                    ms_scores.append({"batter_id": bid, "ms": ms["ms"], "woba": ms["woba"]})

            if ms_scores:
                scored = [m for m in ms_scores if m["ms"] > 0]
                reliever["matchup_scores"] = ms_scores
                reliever["avg_ms_faced"] = round(
                    sum(m["ms"] for m in scored) / len(scored), 1
                ) if scored else 0


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  MLB BULLPEN PREDICTOR")
    print(f"{'='*60}\n")

    # Load ATLAS
    atlas = AtlasData()
    atlas.load()

    # Load daily games
    if not os.path.exists(DAILY_GAMES_FILE):
        print(f"[ERROR] {DAILY_GAMES_FILE} not found — run compute_matchups.py first")
        sys.exit(1)

    with open(DAILY_GAMES_FILE) as f:
        daily = json.load(f)

    games = daily.get("games", [])
    date_str = daily.get("slate_date", datetime.now().strftime("%Y-%m-%d"))
    print(f"[Bullpen] Processing {len(games)} games for {date_str}\n")

    # Load history
    history = load_bullpen_history()
    print(f"[Bullpen] History: {len(history)} pitchers tracked")

    # Predict for each game
    for i, game in enumerate(games):
        game["_slate_date"] = date_str
        st_tag = " [ST]" if game.get("is_spring_training") else ""
        print(f"  [{i+1}/{len(games)}] {game['away_team']} @ {game['home_team']}{st_tag}")

        bp_result = predict_game_bullpen(game, atlas, history)
        game.update(bp_result)

        # Score matchups
        score_bullpen_matchups(game, atlas)

        # Summary
        for side in ["away", "home"]:
            bp = game.get(f"{side}_bullpen", {})
            predicted = bp.get("predicted", [])
            avail = bp.get("available_count", 0)
            conf = bp.get("confidence", "?")
            print(f"    {game[f'{side}_team']}: SP ~{bp.get('starter_expected_ip', '?')} IP, "
                  f"{len(predicted)} relievers predicted ({avail} available) [{conf}]")

    # Clean up temp keys
    for game in games:
        game.pop("_slate_date", None)

    # Save updated daily_games.json
    with open(DAILY_GAMES_FILE, "w") as f:
        json.dump(daily, f, indent=2)
    print(f"\n[Output] Updated {DAILY_GAMES_FILE} with bullpen predictions")

    # Save history
    save_bullpen_history(history)


if __name__ == "__main__":
    main()
