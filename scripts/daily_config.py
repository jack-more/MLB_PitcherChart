"""
Daily pipeline configuration for MLB SIM.
Loads environment variables, defines API endpoints, paths, and thresholds.
"""

import os
import sys
from dotenv import load_dotenv

# ── Load .env from project root ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Paths ──
FRONTEND_PUBLIC = os.path.join(PROJECT_ROOT, "frontend", "public")
DAILY_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "daily")
os.makedirs(DAILY_DATA_DIR, exist_ok=True)

# ── API Keys ──
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── API Endpoints ──
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "baseball_mlb"
ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"

# ── Scraping ──
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 15  # seconds

# ── Matchup Thresholds ──
MIN_PA_FOR_MS = 5           # Minimum PA for archetype matchup score
MIN_PA_FOR_H2H = 10         # Minimum PA for head-to-head matchup bonus
MIN_PA_DISPLAY = 10          # Minimum PA to show in UI
RECENCY_WEIGHT = 2.0         # Weight multiplier for most recent season
DEFAULT_PA_PER_BATTER = 4    # Assumed plate appearances per batter per game

# ── Projection ──
EDGE_THRESHOLD_SPREAD = 1.0  # Minimum run edge to flag a spread pick
EDGE_THRESHOLD_TOTAL = 1.5   # Minimum run edge to flag a total pick
STRONG_EDGE_MULTIPLIER = 2.0 # Strong edge = threshold × this

# ── Bullpen ──
SP_EXPECTED_IP = 5.5          # Expected innings for a starter
RP_EXPECTED_IP = 2.0          # Expected innings for an opener/reliever start
ST_STARTER_CAP_IP = 3.5       # Spring training starter IP cap
BULLPEN_HISTORY_FILE = os.path.join(DAILY_DATA_DIR, "bullpen_history.json")

# ── Caching ──
NAME_ID_CACHE_FILE = os.path.join(DAILY_DATA_DIR, "name_id_cache.json")
ODDS_CACHE_FILE = os.path.join(DAILY_DATA_DIR, "odds_cache.json")

# ── Betting ──
STANDARD_JUICE = -110         # Standard odds for profit calculation
STARTING_BANKROLL = 1000      # Starting $PP bankroll

# ── Output ──
DAILY_LINEUPS_FILE = os.path.join(DAILY_DATA_DIR, "daily_lineups.json")
DAILY_GAMES_FILE = os.path.join(DAILY_DATA_DIR, "daily_games.json")
SETTLEMENT_FILE = os.path.join(DAILY_DATA_DIR, "settlement_results.json")
