"""Central configuration for the MLB Pitcher Archetype Clustering pipeline."""

import os
from datetime import datetime

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")

# ------------------------------------------------------------------
# Seasons
# ------------------------------------------------------------------
FIRST_STATCAST_SEASON = 2015
CURRENT_YEAR = datetime.now().year
SEASONS = list(range(FIRST_STATCAST_SEASON, CURRENT_YEAR + 1))

# MLB regular-season month windows (used to chunk pybaseball calls)
MONTH_RANGES = [
    ("03-20", "04-30"),
    ("05-01", "05-31"),
    ("06-01", "06-30"),
    ("07-01", "07-31"),
    ("08-01", "08-31"),
    ("09-01", "09-30"),
    ("10-01", "11-10"),
]

# ------------------------------------------------------------------
# Pitch types tracked as clustering features
# ------------------------------------------------------------------
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CH", "CU", "FS", "KC", "ST", "KN"]

# ------------------------------------------------------------------
# Statcast columns to keep from raw data
# ------------------------------------------------------------------
KEEP_COLUMNS = [
    "pitch_type", "pitch_name", "game_date", "game_year", "game_pk", "game_type",
    "release_speed", "release_spin_rate",
    "release_pos_x", "release_pos_z", "release_extension",
    "pfx_x", "pfx_z",
    "player_name", "pitcher", "batter",
    "p_throws", "stand",
    "description", "events", "type",
    "zone", "plate_x", "plate_z",
    "balls", "strikes",
    "bb_type", "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "babip_value", "iso_value",
    "at_bat_number", "pitch_number",
    "inning", "inning_topbot",
    "effective_speed",
    "sz_top", "sz_bot",
]

# ------------------------------------------------------------------
# Pitch outcome descriptions for swing/whiff classification
# ------------------------------------------------------------------
SWING_DESCRIPTIONS = [
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
]
WHIFF_DESCRIPTIONS = ["swinging_strike", "swinging_strike_blocked"]

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
MIN_PITCHES = 300           # Minimum pitches for a pitcher-season to qualify
MIN_PITCH_TYPE_PCT = 0.01   # Minimum usage % to count a pitch type as "has"
SHOULDER_HEIGHT_APPROX = 5.0  # Feet, for arm-angle derivation

# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------
K_RANGE = range(2, 16)
OPTIMAL_K = None  # Set after elbow/silhouette analysis (04_clustering.py writes this)
RANDOM_STATE = 42

CLUSTER_FEATURES = [
    # Pitch mix (10)
    "pct_FF", "pct_SI", "pct_FC", "pct_SL", "pct_CH", "pct_CU", "pct_FS",
    "pct_KC", "pct_ST", "pct_KN",
    # Non-pitch (4) — pruned spin_SL, spin_CU, avg_extension, arm_angle, zone_rate
    "avg_velo_FF", "spin_overall", "groundball_rate", "whiff_rate",
    # Hand flag (filtered out for per-hand clustering)
    "is_rhp",
]

MIN_PITCHES_PER_SIDE = 50  # Min pitches vs a batter side for split zone features

# ------------------------------------------------------------------
# Dashboard / display
# ------------------------------------------------------------------
MIN_PA_DISPLAY = 10  # Minimum PAs for a hitter-vs-cluster row in the UI
