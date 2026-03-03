#!/usr/bin/env python3
"""
MLB pick grading — fetches completed game scores and grades spread/total picks.

Reads daily_games.json picks, fetches actual scores from The Odds API,
computes W/L/P results and profit, outputs settlement_results.json.

Usage:
    python scripts/grade_picks.py [--date YYYY-MM-DD]
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, DAILY_GAMES_FILE,
    SETTLEMENT_FILE, DAILY_DATA_DIR, STANDARD_JUICE, STARTING_BANKROLL,
    USER_AGENT,
)
from scripts.mlb_teams import ODDS_TEAM_MAP

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


# ═══════════════════════════════════════════════════════════
# SCORE FETCHING
# ═══════════════════════════════════════════════════════════

def fetch_completed_scores(days_from=3):
    """Fetch completed MLB game scores from The Odds API.

    Returns dict: {(away_abbr, home_abbr): {"home_score": int, "away_score": int, "completed": bool}}
    """
    api_key = ODDS_API_KEY or os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("[Grade] No ODDS_API_KEY — cannot fetch scores")
        return {}

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/scores"
    params = {
        "apiKey": api_key,
        "daysFrom": days_from,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if not resp.ok:
            print(f"[Grade] Odds API scores returned {resp.status_code}")
            return {}

        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"[Grade] Fetched scores — {remaining} API requests remaining")

        data = resp.json()
    except Exception as e:
        print(f"[Grade] Failed to fetch scores: {e}")
        return {}

    scores = {}
    for game in data:
        if not game.get("completed"):
            continue

        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_abbr = ODDS_TEAM_MAP.get(home_full, "")
        away_abbr = ODDS_TEAM_MAP.get(away_full, "")

        if not home_abbr or not away_abbr:
            continue

        game_scores = game.get("scores", [])
        home_score = None
        away_score = None
        for s in game_scores:
            team_name = s.get("name", "")
            score_val = s.get("score")
            if score_val is None:
                continue
            if team_name == home_full:
                home_score = int(score_val)
            elif team_name == away_full:
                away_score = int(score_val)

        if home_score is not None and away_score is not None:
            scores[(away_abbr, home_abbr)] = {
                "home_score": home_score,
                "away_score": away_score,
                "completed": True,
            }

    print(f"[Grade] Found {len(scores)} completed games")
    return scores


def fetch_scores_mlb_api(date_str):
    """Fallback: fetch scores from MLB Stats API.

    Returns dict: {(away_abbr, home_abbr): {"home_score": int, "away_score": int, "completed": bool}}
    """
    from scripts.mlb_teams import TEAM_ID_TO_ABBR

    url = f"https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "startDate": date_str,
        "endDate": date_str,
        "hydrate": "linescore",
    }

    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if not resp.ok:
            return {}
        data = resp.json()
    except Exception:
        return {}

    scores = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue

            teams = game.get("teams", {})
            away_team = teams.get("away", {})
            home_team = teams.get("home", {})

            away_id = away_team.get("team", {}).get("id")
            home_id = home_team.get("team", {}).get("id")
            away_abbr = TEAM_ID_TO_ABBR.get(away_id, "")
            home_abbr = TEAM_ID_TO_ABBR.get(home_id, "")

            if not away_abbr or not home_abbr:
                continue

            away_score = away_team.get("score")
            home_score = home_team.get("score")

            if away_score is not None and home_score is not None:
                scores[(away_abbr, home_abbr)] = {
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "completed": True,
                }

    print(f"[Grade] MLB API fallback: {len(scores)} completed games")
    return scores


# ═══════════════════════════════════════════════════════════
# GRADING LOGIC
# ═══════════════════════════════════════════════════════════

def compute_profit(result, risk_amount):
    """Compute profit for a graded pick at standard juice (-110)."""
    if result == "W":
        return round(risk_amount * (100 / abs(STANDARD_JUICE)), 2)
    elif result == "L":
        return -risk_amount
    else:  # Push
        return 0


def grade_spread_pick(pick, score_data):
    """Grade a spread pick. Returns (result, actual_margin, home_score, away_score)."""
    home_score = score_data["home_score"]
    away_score = score_data["away_score"]
    actual_margin = home_score - away_score  # Positive = home won by this many

    line = pick.get("line", 0)  # Home spread (e.g., -1.5 means home favored by 1.5)
    side = pick.get("side", "")  # "home" or "away"

    # Adjusted margin = actual_margin + spread (from home perspective)
    # If home spread is -1.5, home needs to win by 2+ to cover
    adjusted = actual_margin + line  # line is already from home perspective

    if side == "home":
        if adjusted > 0:
            result = "W"
        elif adjusted < 0:
            result = "L"
        else:
            result = "P"
    elif side == "away":
        if adjusted < 0:
            result = "W"
        elif adjusted > 0:
            result = "L"
        else:
            result = "P"
    else:
        result = "P"

    return result, actual_margin, home_score, away_score


def grade_total_pick(pick, score_data):
    """Grade a total (over/under) pick. Returns (result, actual_total, home_score, away_score)."""
    home_score = score_data["home_score"]
    away_score = score_data["away_score"]
    actual_total = home_score + away_score

    line = pick.get("line", 0)
    side = pick.get("side", "")  # "over" or "under"

    if side == "over":
        if actual_total > line:
            result = "W"
        elif actual_total < line:
            result = "L"
        else:
            result = "P"
    elif side == "under":
        if actual_total < line:
            result = "W"
        elif actual_total > line:
            result = "L"
        else:
            result = "P"
    else:
        result = "P"

    return result, actual_total, home_score, away_score


def grade_all_picks(games, scores):
    """Grade all picks across all games.

    Returns list of graded pick dicts.
    """
    graded = []
    pending = 0

    for game in games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")
        matchup_key = (away, home)
        matchup_str = f"{away} @ {home}"

        picks = game.get("picks", [])
        score_data = scores.get(matchup_key)

        for pick in picks:
            pick_type = pick.get("type", "")
            label = pick.get("label", "")
            edge = pick.get("edge", 0)
            risk = pick.get("risk", 50)
            strength = pick.get("strength", "lean")

            if not score_data:
                pending += 1
                graded.append({
                    "matchup": matchup_str,
                    "label": label,
                    "pick_type": pick_type,
                    "side": pick.get("side", ""),
                    "line": pick.get("line", 0),
                    "edge": edge,
                    "strength": strength,
                    "result": None,
                    "profit": 0,
                    "actual_value": None,
                    "home_score": None,
                    "away_score": None,
                    "risk_amount": risk,
                })
                continue

            if pick_type == "spread":
                result, actual_val, hs, aws = grade_spread_pick(pick, score_data)
            elif pick_type == "total":
                result, actual_val, hs, aws = grade_total_pick(pick, score_data)
            else:
                pending += 1
                continue

            profit = compute_profit(result, risk)

            graded.append({
                "matchup": matchup_str,
                "label": label,
                "pick_type": pick_type,
                "side": pick.get("side", ""),
                "line": pick.get("line", 0),
                "edge": edge,
                "strength": strength,
                "result": result,
                "profit": profit,
                "actual_value": actual_val,
                "home_score": hs,
                "away_score": aws,
                "risk_amount": risk,
            })

    return graded, pending


def build_settlement_results(graded_picks, pending_count):
    """Build the settlement results JSON structure."""
    record = {"W": 0, "L": 0, "P": 0}
    total_profit = 0

    for pick in graded_picks:
        r = pick.get("result")
        if r in record:
            record[r] += 1
            total_profit += pick.get("profit", 0)

    total_graded = record["W"] + record["L"] + record["P"]
    new_bankroll = STARTING_BANKROLL + total_profit

    if pending_count == 0 and total_graded > 0:
        status = "SETTLED"
    elif total_graded > 0:
        status = "PARTIAL"
    else:
        status = "PENDING"

    return {
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "picks": graded_picks,
        "pending_count": pending_count,
        "record": record,
        "total_profit": round(total_profit, 2),
        "starting_bankroll": STARTING_BANKROLL,
        "new_bankroll": round(new_bankroll, 2),
        "status": status,
    }


# ═══════════════════════════════════════════════════════════
# LOAD PREVIOUS SETTLEMENT (for cumulative tracking)
# ═══════════════════════════════════════════════════════════

def load_previous_settlement():
    """Load previous settlement results if they exist."""
    if os.path.exists(SETTLEMENT_FILE):
        try:
            with open(SETTLEMENT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grade MLB picks")
    parser.add_argument("--date", type=str, default=None, help="Date to grade (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MLB PICK GRADER")
    print(f"{'='*60}\n")

    # Load daily games
    if not os.path.exists(DAILY_GAMES_FILE):
        print(f"[ERROR] {DAILY_GAMES_FILE} not found")
        sys.exit(1)

    with open(DAILY_GAMES_FILE) as f:
        daily_data = json.load(f)

    games = daily_data.get("games", [])
    slate_date = args.date or daily_data.get("slate_date", "")
    print(f"[Grade] Grading {len(games)} games from {slate_date}")

    # Count total picks
    total_picks = sum(len(g.get("picks", [])) for g in games)
    if total_picks == 0:
        print("[Grade] No picks to grade")
        return

    print(f"[Grade] {total_picks} total picks to grade")

    # Fetch scores (try Odds API first, then MLB API fallback)
    scores = fetch_completed_scores(days_from=3)
    if not scores and slate_date:
        print("[Grade] Trying MLB API fallback...")
        scores = fetch_scores_mlb_api(slate_date)

    # Grade picks
    graded, pending = grade_all_picks(games, scores)

    # Build results
    results = build_settlement_results(graded, pending)

    # Display summary
    record = results["record"]
    print(f"\n  Record: {record['W']}-{record['L']}-{record['P']}")
    print(f"  Profit: {results['total_profit']:+.2f} $PP")
    print(f"  Bankroll: {results['new_bankroll']:.2f} $PP")
    print(f"  Status: {results['status']}")
    print(f"  Pending: {pending}")

    # Save
    with open(SETTLEMENT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Output] Wrote {SETTLEMENT_FILE}")


if __name__ == "__main__":
    main()
