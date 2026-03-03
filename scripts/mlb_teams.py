"""
Shared MLB team constants for the daily pipeline.
Combines data from fetch_rosters.py (MLB API IDs) and mlbsim.html (colors, UI data).
"""

# ── MLB Teams: (API_ID, ABBR, FULL_NAME, LEAGUE, DIVISION) ──
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

# ── Quick lookups ──
TEAM_ID_TO_ABBR = {t[0]: t[1] for t in MLB_TEAMS}
ABBR_TO_TEAM_ID = {t[1]: t[0] for t in MLB_TEAMS}
ABBR_TO_FULL_NAME = {t[1]: t[2] for t in MLB_TEAMS}

# ── The Odds API full team names → our abbreviations ──
ODDS_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# Reverse: abbreviation → Odds API name
ABBR_TO_ODDS_NAME = {v: k for k, v in ODDS_TEAM_MAP.items()}

# ── Team primary colors (from mlbsim.html TEAM_COLORS) ──
TEAM_COLORS = {
    "BAL": "#DF4601", "BOS": "#BD3039", "NYY": "#003087", "TB": "#092C5C", "TOR": "#134A8E",
    "CLE": "#00385D", "CWS": "#27251F", "DET": "#0C2340", "KC": "#004687", "MIN": "#002B5C",
    "HOU": "#002D62", "LAA": "#BA0021", "OAK": "#003831", "SEA": "#0C2C56", "TEX": "#003278",
    "ATL": "#CE1141", "MIA": "#00A3E0", "NYM": "#002D72", "PHI": "#E81828", "WSH": "#AB0003",
    "CHC": "#0E3386", "CIN": "#C6011F", "MIL": "#FFC52F", "PIT": "#27251F", "STL": "#C41E3A",
    "ARI": "#A71930", "COL": "#333366", "LAD": "#005A9C", "SD": "#2F241D", "SF": "#FD5A1E",
}

# ── RotoWire abbreviation normalization ──
# RotoWire sometimes uses different abbreviations than MLB Stats API
ROTOWIRE_ABBR_MAP = {
    "AZ": "ARI",
    "CHW": "CWS",
    "CWH": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
    "WAS": "WSH",
}


def normalize_abbr(abbr: str) -> str:
    """Normalize a team abbreviation to our standard format."""
    abbr = abbr.strip().upper()
    return ROTOWIRE_ABBR_MAP.get(abbr, abbr)


def team_logo_url(abbr: str) -> str:
    """Get MLB Static team logo SVG URL."""
    team_id = ABBR_TO_TEAM_ID.get(abbr)
    if team_id:
        return f"https://www.mlbstatic.com/team-logos/{team_id}.svg"
    return ""
