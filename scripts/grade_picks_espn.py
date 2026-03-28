#!/usr/bin/env python3
"""
MLB pick grading via ESPN Scoreboard API (free, no key needed).

Reads daily_games.json ML picks (edges.ml.is_pick == True),
fetches final scores from ESPN, grades W/L, computes P&L with actual odds,
and saves results to mlb_settlement.json + mlb_pick_history.json.

Usage:
    python scripts/grade_picks_espn.py [--date YYYY-MM-DD]
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import DAILY_GAMES_FILE, DAILY_DATA_DIR, STARTING_BANKROLL

# ═══════════════════════════════════════════════════════════
# ESPN TEAM ABBREVIATION MAP
# ESPN uses slightly different abbrs than our pipeline
# ═══════════════════════════════════════════════════════════

ESPN_ABBR_MAP = {
    "CHW": "CWS",   # ESPN: CHW, ours: CWS
    "WSN": "WSH",   # Sometimes ESPN uses WSN
}


def normalize_espn_abbr(abbr):
    """Convert ESPN abbreviation to our standard abbreviation."""
    return ESPN_ABBR_MAP.get(abbr, abbr)


# ═══════════════════════════════════════════════════════════
# ESPN SCORE FETCHING
# ═══════════════════════════════════════════════════════════

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"


def fetch_espn_scores(date_str):
    """Fetch final scores from ESPN Scoreboard API.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        dict: {(away_abbr, home_abbr): {"away_score": int, "home_score": int, "away_abbr": str, "home_abbr": str}}
    """
    espn_date = date_str.replace("-", "")  # YYYYMMDD
    params = {"dates": espn_date}

    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=15)
        if not resp.ok:
            print(f"[GradeESPN] ESPN API returned {resp.status_code}")
            return {}
        data = resp.json()
    except Exception as e:
        print(f"[GradeESPN] Failed to fetch ESPN scores: {e}")
        return {}

    scores = {}
    events = data.get("events", [])

    for event in events:
        comp = event.get("competitions", [{}])[0]
        status_name = comp.get("status", {}).get("type", {}).get("name", "")

        if status_name != "STATUS_FINAL":
            continue

        teams = {}
        for competitor in comp.get("competitors", []):
            ha = competitor.get("homeAway", "")
            abbr = normalize_espn_abbr(competitor.get("team", {}).get("abbreviation", ""))
            score = competitor.get("score", "0")
            teams[ha] = {"abbr": abbr, "score": int(score)}

        away = teams.get("away", {})
        home = teams.get("home", {})

        if away.get("abbr") and home.get("abbr"):
            key = (away["abbr"], home["abbr"])
            scores[key] = {
                "away_abbr": away["abbr"],
                "home_abbr": home["abbr"],
                "away_score": away["score"],
                "home_score": home["score"],
            }

    print(f"[GradeESPN] Found {len(scores)} completed games for {date_str}")
    return scores


# ═══════════════════════════════════════════════════════════
# PICK EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_ml_picks(daily_data):
    """Extract ML picks where edges.ml.is_pick == True.

    Returns list of pick dicts with all needed fields.
    """
    picks = []
    games = daily_data.get("games", [])

    for game in games:
        edges = game.get("edges", {})
        ml = edges.get("ml", {})

        if not ml.get("is_pick"):
            continue

        away = game.get("away_team", "")
        home = game.get("home_team", "")
        side = ml.get("side", "")
        odds = ml.get("odds", 0)
        ev = ml.get("ev", 0)
        model_wp = ml.get("model_wp", 0)
        market_wp = ml.get("market_wp", 0)
        is_strong = ml.get("is_strong", False)
        confidence = game.get("confidence", {})
        conf_val = confidence.get("value_rating", 5)

        # Unit sizing: 30 $PP base, 50 for strong/high-conf
        unit = 50 if (is_strong and conf_val >= 8) else 30

        picks.append({
            "away_team": away,
            "home_team": home,
            "matchup": f"{away} @ {home}",
            "side": side,
            "odds": odds,
            "ev": round(ev, 1),
            "model_wp": round(model_wp, 1),
            "market_wp": round(market_wp, 1),
            "is_strong": is_strong,
            "confidence": conf_val,
            "unit": unit,
        })

    return picks


# ═══════════════════════════════════════════════════════════
# GRADING LOGIC
# ═══════════════════════════════════════════════════════════

def compute_ml_profit(odds, unit, result):
    """Compute profit for a moneyline bet using actual odds.

    Favorite (negative odds): profit = 100/abs(odds) * unit
    Dog (positive odds): profit = odds/100 * unit
    """
    if result == "W":
        if odds < 0:
            return round((100 / abs(odds)) * unit, 2)
        else:
            return round((odds / 100) * unit, 2)
    elif result == "L":
        return -unit
    else:
        return 0  # push


def grade_picks(picks, scores):
    """Grade ML picks against final scores.

    Returns list of graded pick dicts.
    """
    graded = []

    for pick in picks:
        away = pick["away_team"]
        home = pick["home_team"]
        side = pick["side"]
        odds = pick["odds"]
        unit = pick["unit"]

        score_key = (away, home)
        score_data = scores.get(score_key)

        if not score_data:
            print(f"[GradeESPN] No score found for {away} @ {home} — skipping")
            continue

        away_score = score_data["away_score"]
        home_score = score_data["home_score"]

        # Determine winner
        if away_score > home_score:
            winner = away
        elif home_score > away_score:
            winner = home
        else:
            winner = None  # shouldn't happen in MLB (extras)

        if winner is None:
            result = "P"
        elif side == winner:
            result = "W"
        else:
            result = "L"

        pnl = compute_ml_profit(odds, unit, result)

        # Build final score string
        if side == away:
            final_score = f"{away} {away_score} — {home} {home_score}"
        else:
            final_score = f"{home} {home_score} — {away} {away_score}"

        graded.append({
            "matchup": pick["matchup"],
            "side": side,
            "odds": odds,
            "ev": pick["ev"],
            "model_wp": pick["model_wp"],
            "market_wp": pick["market_wp"],
            "confidence": pick["confidence"],
            "unit": unit,
            "result": result,
            "pnl": pnl,
            "final_score": f"{away} {away_score}-{home} {home_score}",
            "final_score_display": final_score,
            "away_score": away_score,
            "home_score": home_score,
            "is_strong": pick["is_strong"],
        })

    return graded


# ═══════════════════════════════════════════════════════════
# CUMULATIVE TRACKING
# ═══════════════════════════════════════════════════════════

HISTORY_FILE = os.path.join(DAILY_DATA_DIR, "mlb_pick_history.json")
SETTLEMENT_FILE = os.path.join(DAILY_DATA_DIR, "mlb_settlement.json")


def load_history():
    """Load cumulative pick history."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def compute_cumulative(history):
    """Compute cumulative stats from all history entries."""
    total_wins = 0
    total_losses = 0
    total_pushes = 0
    total_pnl = 0
    total_risked = 0

    for day in history:
        rec = day.get("record", {})
        total_wins += rec.get("wins", 0)
        total_losses += rec.get("losses", 0)
        total_pushes += rec.get("pushes", 0)
        total_pnl += day.get("daily_pnl", 0)
        # Sum up units risked
        for p in day.get("picks", []):
            total_risked += p.get("unit", 30)

    bankroll = STARTING_BANKROLL + total_pnl
    roi = round((total_pnl / total_risked * 100), 1) if total_risked > 0 else 0

    return {
        "record": f"{total_wins}-{total_losses}",
        "pnl": round(total_pnl, 2),
        "bankroll": round(bankroll, 2),
        "total_risked": total_risked,
        "roi": roi,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grade MLB ML picks via ESPN")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to grade (YYYY-MM-DD). Defaults to yesterday."
    )
    args = parser.parse_args()

    # Determine grade date
    if args.date:
        grade_date = args.date
    else:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        grade_date = yesterday.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  MLB PICK GRADER (ESPN)")
    print(f"  Grading: {grade_date}")
    print(f"{'='*60}\n")

    # Load daily games — prefer snapshot if available (daily_games.json gets overwritten)
    snapshot_path = os.path.join(DAILY_DATA_DIR, "snapshots", f"{grade_date}.json")
    if os.path.exists(snapshot_path):
        source = snapshot_path
        print(f"[GradeESPN] Using snapshot: {snapshot_path}")
    elif os.path.exists(DAILY_GAMES_FILE):
        source = DAILY_GAMES_FILE
        print(f"[GradeESPN] No snapshot found, using live daily_games.json")
    else:
        print(f"[ERROR] No data found for {grade_date}")
        sys.exit(1)

    with open(source) as f:
        daily_data = json.load(f)

    slate_date = daily_data.get("slate_date", "")
    if slate_date != grade_date:
        print(f"[WARN] Source slate_date ({slate_date}) != grade_date ({grade_date})")
        print(f"[WARN] Proceeding anyway — will grade picks from {slate_date}")

    # Extract ML picks
    picks = extract_ml_picks(daily_data)
    if not picks:
        print("[GradeESPN] No ML picks found in daily_games.json")
        return

    print(f"[GradeESPN] {len(picks)} ML picks to grade:")
    for p in picks:
        odds_str = f"{p['odds']:+d}" if p['odds'] > 0 else str(p['odds'])
        print(f"  {p['matchup']} — {p['side']} ML ({odds_str}) | EV: {p['ev']}% | {p['unit']} $PP")

    # Fetch scores from ESPN
    scores = fetch_espn_scores(slate_date)
    if not scores:
        print("[GradeESPN] No completed games found — try again later")
        sys.exit(1)

    # Grade picks
    graded = grade_picks(picks, scores)

    if not graded:
        print("[GradeESPN] No picks could be graded")
        sys.exit(1)

    # Calculate daily stats
    wins = sum(1 for p in graded if p["result"] == "W")
    losses = sum(1 for p in graded if p["result"] == "L")
    pushes = sum(1 for p in graded if p["result"] == "P")
    daily_pnl = round(sum(p["pnl"] for p in graded), 2)

    # Build settlement
    settlement = {
        "date": slate_date,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "picks": graded,
        "record": {"wins": wins, "losses": losses, "pushes": pushes},
        "daily_pnl": daily_pnl,
    }

    # Load history and check for duplicates
    history = load_history()
    existing_dates = {d.get("date") for d in history}

    if slate_date in existing_dates:
        print(f"[GradeESPN] {slate_date} already in history — replacing")
        history = [d for d in history if d.get("date") != slate_date]

    history.append(settlement)
    history.sort(key=lambda d: d.get("date", ""))

    # Compute cumulative stats
    cumulative = compute_cumulative(history)
    settlement["cumulative"] = cumulative

    # Display results
    print(f"\n  {'─'*50}")
    for p in graded:
        result_icon = {"W": "+", "L": "x", "P": "="}[p["result"]]
        odds_str = f"{p['odds']:+d}" if p['odds'] > 0 else str(p['odds'])
        print(f"  [{result_icon}] {p['matchup']} — {p['side']} ML ({odds_str}) → {p['result']} ({p['pnl']:+.2f} $PP) | {p['final_score']}")

    print(f"\n  Daily: {wins}-{losses} | {daily_pnl:+.2f} $PP")
    print(f"  Cumulative: {cumulative['record']} | {cumulative['pnl']:+.2f} $PP | ROI: {cumulative['roi']}%")
    print(f"  Bankroll: {cumulative['bankroll']:.0f} $PP (started: {STARTING_BANKROLL})")

    # Save settlement
    with open(SETTLEMENT_FILE, "w") as f:
        json.dump(settlement, f, indent=2)
    print(f"\n[Output] Wrote {SETTLEMENT_FILE}")

    # Save history
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Output] Wrote {HISTORY_FILE}")


if __name__ == "__main__":
    main()
