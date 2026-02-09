#!/usr/bin/env python3
"""
Fetch current 40-man rosters for all 30 MLB teams from the MLB Stats API.
Outputs teams_2026.json for the ARCHETYPE//ATLAS.MLB team filter.
"""
import json
import urllib.request
import time
import sys

API_BASE = "https://statsapi.mlb.com/api/v1"

# All 30 MLB teams with their IDs, abbreviations, and divisions
MLB_TEAMS = [
    # AL East
    (110, "BAL", "Baltimore Orioles", "AL", "East"),
    (111, "BOS", "Boston Red Sox", "AL", "East"),
    (147, "NYY", "New York Yankees", "AL", "East"),
    (139, "TB",  "Tampa Bay Rays", "AL", "East"),
    (141, "TOR", "Toronto Blue Jays", "AL", "East"),
    # AL Central
    (114, "CLE", "Cleveland Guardians", "AL", "Central"),
    (145, "CWS", "Chicago White Sox", "AL", "Central"),
    (116, "DET", "Detroit Tigers", "AL", "Central"),
    (118, "KC",  "Kansas City Royals", "AL", "Central"),
    (142, "MIN", "Minnesota Twins", "AL", "Central"),
    # AL West
    (117, "HOU", "Houston Astros", "AL", "West"),
    (108, "LAA", "Los Angeles Angels", "AL", "West"),
    (133, "OAK", "Oakland Athletics", "AL", "West"),
    (136, "SEA", "Seattle Mariners", "AL", "West"),
    (140, "TEX", "Texas Rangers", "AL", "West"),
    # NL East
    (144, "ATL", "Atlanta Braves", "NL", "East"),
    (146, "MIA", "Miami Marlins", "NL", "East"),
    (121, "NYM", "New York Mets", "NL", "East"),
    (143, "PHI", "Philadelphia Phillies", "NL", "East"),
    (120, "WSH", "Washington Nationals", "NL", "East"),
    # NL Central
    (112, "CHC", "Chicago Cubs", "NL", "Central"),
    (113, "CIN", "Cincinnati Reds", "NL", "Central"),
    (158, "MIL", "Milwaukee Brewers", "NL", "Central"),
    (134, "PIT", "Pittsburgh Pirates", "NL", "Central"),
    (138, "STL", "St. Louis Cardinals", "NL", "Central"),
    # NL West
    (109, "ARI", "Arizona Diamondbacks", "NL", "West"),
    (115, "COL", "Colorado Rockies", "NL", "West"),
    (119, "LAD", "Los Angeles Dodgers", "NL", "West"),
    (135, "SD",  "San Diego Padres", "NL", "West"),
    (137, "SF",  "San Francisco Giants", "NL", "West"),
]

def fetch_json(url):
    """Fetch JSON from URL with retry."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MLB-PitcherChart/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise e

def fetch_roster(team_id):
    """Fetch 40-man roster for a team, return list of pitcher MLBAM IDs."""
    url = f"{API_BASE}/teams/{team_id}/roster?rosterType=40Man"
    data = fetch_json(url)
    pitchers = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        position = entry.get("position", {})
        # Include pitchers (P) and two-way players
        if position.get("abbreviation") == "P" or position.get("type") == "Pitcher":
            pitchers.append(person["id"])
    return pitchers

def main():
    # Load existing pitcher IDs from pitcher_seasons.json to compute overlap
    ps_path = "/Users/jackmorello/Desktop/MLB_PitcherChart/frontend/public/pitcher_seasons.json"
    with open(ps_path) as f:
        ps_data = json.load(f)
    known_ids = set(r["pitcher"] for r in ps_data)
    print(f"Known pitcher IDs in data: {len(known_ids)}")

    teams_meta = {}
    rosters = {}
    total_pitchers = 0
    total_in_data = 0

    for team_id, abbr, name, league, division in MLB_TEAMS:
        try:
            pitcher_ids = fetch_roster(team_id)
            in_data = [pid for pid in pitcher_ids if pid in known_ids]
            print(f"  {abbr:4s} {name:30s} → {len(pitcher_ids):2d} pitchers, {len(in_data):2d} in data")
            teams_meta[abbr] = {"name": name, "lg": league, "div": division}
            rosters[abbr] = pitcher_ids
            total_pitchers += len(pitcher_ids)
            total_in_data += len(in_data)
            time.sleep(0.2)  # Rate limit
        except Exception as e:
            print(f"  ERROR fetching {abbr}: {e}", file=sys.stderr)
            teams_meta[abbr] = {"name": name, "lg": league, "div": division}
            rosters[abbr] = []

    print(f"\nTotal: {total_pitchers} pitchers across 30 teams, {total_in_data} found in pitcher_seasons.json")

    # Build output
    output = {
        "updated": "2026-02-09",
        "teams": teams_meta,
        "rosters": rosters
    }

    out_path = "/Users/jackmorello/Desktop/MLB_PitcherChart/frontend/public/teams_2026.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
