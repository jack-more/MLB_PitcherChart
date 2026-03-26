#!/usr/bin/env python3
"""
Matchup computation engine for MLB SIM.

Loads ATLAS data (clusters, pitcher seasons, hitter vs cluster) and enriches
daily_lineups.json with per-batter Matchup Scores, team run projections,
and spread/total edge calculations.

This is a direct port of the client-side JavaScript (computeMS, projectTeam,
baseRuns) from mlbsim.html to Python, with enhancements:
  - Recency weighting (most recent year weighted 2x)
  - NRI/unknown player flagging
  - Confidence-based projection ranges

Usage:
    python scripts/compute_matchups.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import (
    FRONTEND_PUBLIC, DAILY_LINEUPS_FILE, DAILY_GAMES_FILE,
    MIN_PA_FOR_MS, MIN_PA_FOR_H2H, RECENCY_WEIGHT, DEFAULT_PA_PER_BATTER,
    EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL, STRONG_EDGE_MULTIPLIER,
    CONF_ST_CAP,
)


# ═══════════════════════════════════════════════════════════
# ATLAS DATA LOADER
# ═══════════════════════════════════════════════════════════

class AtlasData:
    """Loads and indexes all ATLAS data for fast matchup lookups."""

    def __init__(self):
        self.clusters = {}          # cluster_id → profile
        self.ps_index = {}          # pitcher_id → most recent season record
        self.ps_all = {}            # pitcher_id → [all season records]
        self.hvc_index = {}         # (batter_id, cluster_id) → [rows]
        self.batter_index = {}      # batter_id → name
        self.hitter_archetypes = {} # batter_id → archetype info
        self.baserunning = {}       # batter_id → {sb_per_pa, cs_per_pa, ...}
        # Adjustment data (from backtest pipeline)
        self.pitcher_tiers = {}     # pitcher_id → {tier, effective_multiplier, ...}
        self.re24_data = {}         # "batterId_year" → {RE24_above_avg, ...}
        self.team_oaa = {}          # team_abbr → {year → {infield_oaa, outfield_oaa}}
        self.park_factors = {}      # team_abbr → park_mult
        self.team_bullpen = {}      # team_abbr → {runs_per_ip, ...}
        self.sp_innings = {}        # pitcher_id → avg_ip_per_gs

    def load(self):
        """Load all data files from frontend/public/."""
        print("[ATLAS] Loading data files...")

        with open(os.path.join(FRONTEND_PUBLIC, "clusters.json")) as f:
            self.clusters = json.load(f)
        print(f"  clusters.json: {len(self.clusters)} archetypes")

        with open(os.path.join(FRONTEND_PUBLIC, "pitcher_seasons.json")) as f:
            pitcher_seasons = json.load(f)
        print(f"  pitcher_seasons.json: {len(pitcher_seasons)} records")

        with open(os.path.join(FRONTEND_PUBLIC, "batters.json")) as f:
            batters = json.load(f)
        print(f"  batters.json: {len(batters)} batters")

        with open(os.path.join(FRONTEND_PUBLIC, "hitter_vs_cluster.json")) as f:
            hvc = json.load(f)
        print(f"  hitter_vs_cluster.json: {len(hvc)} matchup rows")

        # Optional
        hitter_arch_path = os.path.join(FRONTEND_PUBLIC, "hitter_archetypes.json")
        if os.path.exists(hitter_arch_path):
            with open(hitter_arch_path) as f:
                self.hitter_archetypes = json.load(f)
            print(f"  hitter_archetypes.json: loaded")

        # Baserunning (SB/CS rates)
        baserunning_path = os.path.join(FRONTEND_PUBLIC, "batter_baserunning.json")
        if os.path.exists(baserunning_path):
            with open(baserunning_path) as f:
                raw = json.load(f)
            # Keys are string MLBAM IDs in JSON, convert to int
            self.baserunning = {int(k): v for k, v in raw.items()}
            print(f"  batter_baserunning.json: {len(self.baserunning)} batters")
        else:
            print("  batter_baserunning.json: NOT FOUND (SB/CS will use league avg)")

        # ── Load adjustment data (backtest-proven factors) ──
        def _load_optional(filename):
            path = os.path.join(FRONTEND_PUBLIC, filename)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                print(f"  {filename}: {len(data)} records")
                return data
            print(f"  {filename}: NOT FOUND (skipping)")
            return {}

        # Pitcher tiers: keyed "pitcherId_year" → {tier, effective_multiplier, ...}
        # Index by pitcher ID (most recent year)
        raw_tiers = _load_optional("pitcher_tiers.json")
        if isinstance(raw_tiers, dict):
            for key, t in raw_tiers.items():
                pid = t.get("pitcher")
                if pid:
                    if pid not in self.pitcher_tiers or t.get("game_year", 0) > self.pitcher_tiers[pid].get("game_year", 0):
                        self.pitcher_tiers[pid] = t

        # RE24: keyed "batterId_year" → {RE24_above_avg, ...}
        raw_re24 = _load_optional("hitter_re24.json")
        if isinstance(raw_re24, dict):
            self.re24_data = raw_re24

        # Team OAA: keyed "TeamName_year" → {regressed_infield, regressed_outfield}
        # OAA uses short names (Giants, Yankees) not full (San Francisco Giants)
        raw_oaa = _load_optional("team_oaa.json")
        if isinstance(raw_oaa, dict):
            from scripts.mlb_teams import ABBR_TO_FULL_NAME
            # Build short name → abbr (e.g., "Giants" → "SF")
            short_to_abbr = {}
            for abbr, full in ABBR_TO_FULL_NAME.items():
                # Extract last word(s) as short name
                parts = full.split()
                short = parts[-1] if len(parts) <= 3 else " ".join(parts[-2:])
                short_to_abbr[short] = abbr
                # Also try just last word
                short_to_abbr[parts[-1]] = abbr
            # Manual overrides for tricky names
            short_to_abbr["Blue Jays"] = "TOR"
            short_to_abbr["White Sox"] = "CWS"
            short_to_abbr["Red Sox"] = "BOS"
            short_to_abbr["D-backs"] = "ARI"

            for key, o in raw_oaa.items():
                team_name = o.get("team", "")
                year_val = o.get("year", 0)
                abbr = short_to_abbr.get(team_name, "")
                if abbr and year_val >= 2024:
                    if abbr not in self.team_oaa or year_val > self.team_oaa[abbr].get("year", 0):
                        self.team_oaa[abbr] = {
                            "infield_oaa": o.get("regressed_infield", 0),
                            "outfield_oaa": o.get("regressed_outfield", 0),
                            "year": year_val,
                        }

        # Park factors: keyed by team ABBR → {per_team_multiplier, ...}
        raw_park = _load_optional("park_factors.json")
        if isinstance(raw_park, dict):
            for abbr, p in raw_park.items():
                if isinstance(p, dict):
                    self.park_factors[abbr] = p.get("per_team_multiplier", 1.0)
                else:
                    self.park_factors[abbr] = float(p)

        # Team bullpen: keyed "TEAM_year" → {runs_per_ip, above_avg_runs_per_ip}
        raw_bp = _load_optional("team_bullpen.json")
        if isinstance(raw_bp, dict):
            for key, b in raw_bp.items():
                team = b.get("team", "")
                year_val = b.get("year", 0)
                if team and year_val >= 2024:
                    if team not in self.team_bullpen or year_val > self.team_bullpen[team].get("year", 0):
                        self.team_bullpen[team] = b

        # SP innings: keyed "pitcherId_year" → {ip_per_gs, ...}
        raw_sp = _load_optional("sp_innings.json")
        if isinstance(raw_sp, dict):
            for key, s in raw_sp.items():
                # Extract pitcher ID from key (e.g., "519242_2024")
                parts = key.split("_")
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                        if pid not in self.sp_innings or s.get("year", 0) > self.sp_innings.get(pid, {}).get("year", 0):
                            self.sp_innings[pid] = s
                    except ValueError:
                        pass

        # ── Build indexes ──
        # Pitcher: most recent season
        for ps in pitcher_seasons:
            pid = ps.get("pitcher")
            if not pid:
                continue
            if pid not in self.ps_all:
                self.ps_all[pid] = []
            self.ps_all[pid].append(ps)
            if pid not in self.ps_index or ps.get("game_year", 0) > self.ps_index[pid].get("game_year", 0):
                self.ps_index[pid] = ps

        # Hitter vs cluster: (batter_id, cluster_id) → [rows across years]
        for row in hvc:
            key = (row.get("batter"), str(row.get("cluster", "")))
            if key not in self.hvc_index:
                self.hvc_index[key] = []
            self.hvc_index[key].append(row)

        # Batter name index
        for b in batters:
            self.batter_index[b.get("batter")] = b.get("batter_name", "Unknown")

        # Batter overall: batter_id → [all rows across ALL clusters/years]
        # Used as fallback when opposing pitcher has no cluster assignment
        self.batter_overall = {}
        for row in hvc:
            bid = row.get("batter")
            if bid is not None:
                if bid not in self.batter_overall:
                    self.batter_overall[bid] = []
                self.batter_overall[bid].append(row)

        print(f"[ATLAS] Indexed: {len(self.ps_index)} pitchers, "
              f"{len(self.hvc_index)} batter-cluster combos, {len(self.batter_index)} batters, "
              f"{len(self.batter_overall)} batters (overall fallback)")


# ═══════════════════════════════════════════════════════════
# MATCHUP SCORE (port of JS computeMS)
# ═══════════════════════════════════════════════════════════

def get_pitcher_cluster(pitcher_id: int, atlas: AtlasData):
    """Look up pitcher's archetype cluster from most recent season.

    Returns (cluster_id: str, archetype_name: str) or (None, None).
    """
    ps = atlas.ps_index.get(pitcher_id)
    if ps is None:
        return None, None
    return str(ps.get("cluster", "")), ps.get("archetype", "")


def _woba_to_ms(woba: float) -> int:
    """Convert wOBA to Matchup Score (40-99 scale)."""
    if woba >= 0.400:
        ms = 90 + min(9, round((woba - 0.400) * 100))
    elif woba >= 0.370:
        ms = 80 + round((woba - 0.370) / 0.030 * 9)
    elif woba >= 0.340:
        ms = 70 + round((woba - 0.340) / 0.030 * 9)
    elif woba >= 0.310:
        ms = 60 + round((woba - 0.310) / 0.030 * 9)
    elif woba >= 0.270:
        ms = 50 + round((woba - 0.270) / 0.040 * 9)
    else:
        ms = 40 + round(max(0, woba - 0.200) / 0.070 * 9)
    return max(40, min(99, ms))


def _weighted_stats(rows: list, recency_weight: float = RECENCY_WEIGHT):
    """Compute PA-weighted stats across rows with recency boost.

    Returns (woba, k_pct, bb_pct, raw_pa, weighted_pa) or None if insufficient data.
    """
    if not rows:
        return None

    max_year = max(r.get("game_year", 0) for r in rows)
    tot_pa_w = 0
    woba_sum = 0
    k_sum = 0
    bb_sum = 0

    for r in rows:
        weight = recency_weight if r.get("game_year", 0) == max_year else 1.0
        pa = (r.get("PA") or 0) * weight
        tot_pa_w += pa
        woba_sum += (r.get("wOBA") or 0) * pa
        k_sum += (r.get("K_pct") or 0) * pa
        bb_sum += (r.get("BB_pct") or 0) * pa

    raw_pa = sum(r.get("PA", 0) for r in rows)

    if tot_pa_w < MIN_PA_FOR_MS:
        return None

    return (
        woba_sum / tot_pa_w,
        k_sum / tot_pa_w,
        bb_sum / tot_pa_w,
        raw_pa,
        tot_pa_w,
    )


def compute_ms(batter_id: int, cluster_id: str, atlas: AtlasData) -> dict:
    """Compute Matchup Score (40-99) for a batter vs a pitcher archetype.

    Enhanced over JS version with recency weighting.
    Falls back to batter's overall (cross-archetype) stats when cluster_id
    is None (opposing pitcher not in ATLAS).

    Returns dict with: ms, range, woba, k_pct, bb_pct, pa, source
    """
    _zero = {"ms": 0, "range": [0, 0], "woba": 0, "k_pct": 0, "bb_pct": 0, "pa": 0, "source": "none"}

    if batter_id is None:
        return _zero

    # ── Primary path: archetype-specific lookup ──
    if cluster_id is not None:
        key = (batter_id, cluster_id)
        arch_rows = atlas.hvc_index.get(key, [])

        if arch_rows:
            stats = _weighted_stats(arch_rows)
            if stats:
                woba, k_pct, bb_pct, raw_pa, _ = stats
                ms = _woba_to_ms(woba)
                spread = 4 if raw_pa >= 50 else (8 if raw_pa >= 20 else 14)
                return {
                    "ms": ms,
                    "range": [max(40, ms - spread), min(99, ms + spread)],
                    "woba": round(woba, 3),
                    "k_pct": round(k_pct, 3),
                    "bb_pct": round(bb_pct, 3),
                    "pa": raw_pa,
                    "source": "archetype",
                }

    # ── Fallback: batter's overall stats across ALL archetypes ──
    # Used when pitcher has no cluster (uncharted) or batter has no data
    # for this specific cluster but does have data vs other clusters.
    overall_rows = atlas.batter_overall.get(batter_id, [])
    if overall_rows:
        stats = _weighted_stats(overall_rows)
        if stats:
            woba, k_pct, bb_pct, raw_pa, _ = stats
            ms = _woba_to_ms(woba)
            # Wider confidence range for fallback (less precise without archetype)
            spread = 10 if raw_pa >= 100 else (14 if raw_pa >= 40 else 18)
            return {
                "ms": ms,
                "range": [max(40, ms - spread), min(99, ms + spread)],
                "woba": round(woba, 3),
                "k_pct": round(k_pct, 3),
                "bb_pct": round(bb_pct, 3),
                "pa": raw_pa,
                "source": "overall",  # Flagged so UI can distinguish
            }

    return _zero


def ms_tier(ms: int) -> str:
    """Convert MS to tier label."""
    if ms >= 85:
        return "elite"
    if ms >= 70:
        return "favorable"
    if ms >= 55:
        return "neutral"
    return "tough"


# ═══════════════════════════════════════════════════════════
# BASERUNS (port of JS baseRuns)
# ═══════════════════════════════════════════════════════════

def base_runs(team_h, team_hr, team_bb, team_tb, team_pa, team_sb=0, team_cs=0):
    """BaseRuns run estimator with stolen base integration.

    SB/CS affect run production:
      A (baserunners): SB adds runners in scoring position, CS removes them
      B (advancement): SB contributes to base advancement power
      C (outs):        CS are outs that kill rallies
    """
    A = team_h + team_bb - team_hr + team_sb - team_cs
    B = 1.02 * (1.4 * team_tb - 0.6 * team_h + 0.1 * team_bb + 0.3 * team_sb)
    C = max(team_pa, 36) - team_h - team_bb + team_cs
    D = team_hr
    if B + C == 0:
        return D
    return max(0, A * B / (B + C) + D)


# ═══════════════════════════════════════════════════════════
# PROJECT TEAM (port of JS projectTeam)
# ═══════════════════════════════════════════════════════════

def project_team(lineup, opposing_pitcher_id: int, atlas: AtlasData,
                  pa_override: int = None,
                  batting_team: str = "", opposing_team: str = "",
                  venue_team: str = "", year: int = 2026):
    """Project team offense against opposing pitcher using ATLAS matchups.

    Applies full model adjustments (backtest-proven):
      1. Pitcher tier multiplier (hit suppression/inflation)
      2. RE24 bonus (baserunning/context value)
      3. OAA defense adjustment (opposing team's fielding)
      4. Park factor (venue run environment)
      5. Bullpen adjustment (opposing bullpen quality)

    Returns dict with: runs, coverage, total, batter_details, team_stats, adjustments
    """
    cluster_id, archetype = get_pitcher_cluster(opposing_pitcher_id, atlas) if opposing_pitcher_id else (None, None)
    assumed_pa = pa_override or DEFAULT_PA_PER_BATTER

    team_h, team_hr, team_bb, team_k, team_tb, team_pa = 0, 0, 0, 0, 0, 0
    team_sb, team_cs = 0, 0
    team_re24 = 0
    coverage = 0
    batter_details = []

    for batter in lineup:
        batter_id = batter.get("id")
        batter_name = batter.get("name") or atlas.batter_index.get(batter_id, "Unknown")

        proj = {"h": 0, "hr": 0, "k": 0, "bb": 0, "tb": 0, "woba": 0, "sb": 0, "cs": 0}
        source = "none"

        # Determine which rows to use: archetype-specific or overall fallback
        data_rows = None
        data_source = "none"

        if batter_id and cluster_id:
            key = (batter_id, cluster_id)
            arch_rows = atlas.hvc_index.get(key, [])
            if arch_rows:
                data_rows = arch_rows
                data_source = "archetype"

        # Fallback: use batter's overall stats across ALL clusters
        if data_rows is None and batter_id:
            overall_rows = atlas.batter_overall.get(batter_id, [])
            if overall_rows:
                data_rows = overall_rows
                data_source = "overall"

        if data_rows:
                tot = {"PA": 0, "H": 0, "HR": 0, "BB": 0, "K": 0,
                       "singles": 0, "doubles": 0, "triples": 0}
                woba_weighted = 0

                for r in data_rows:
                    for field in tot:
                        tot[field] += r.get(field, 0)
                    woba_weighted += (r.get("wOBA", 0)) * (r.get("PA", 0))

                if tot["PA"] >= MIN_PA_FOR_MS:
                    pa_total = tot["PA"]
                    ba_rate = tot["H"] / max(pa_total - tot["BB"], 0.01)
                    bb_rate = tot["BB"] / max(pa_total, 0.01)
                    k_rate = tot["K"] / max(pa_total, 0.01)
                    hr_rate = tot["HR"] / max(pa_total, 0.01)
                    tb_total = (tot["singles"] + tot["doubles"] * 2 +
                                tot["triples"] * 3 + tot["HR"] * 4)
                    tb_rate = tb_total / max(pa_total, 0.01)

                    est_ab = assumed_pa * (1 - bb_rate)
                    proj["h"] = round(ba_rate * est_ab, 3)
                    proj["hr"] = round(hr_rate * assumed_pa, 3)
                    proj["k"] = round(k_rate * assumed_pa, 3)
                    proj["bb"] = round(bb_rate * assumed_pa, 3)
                    proj["tb"] = round(tb_rate * assumed_pa, 3)
                    proj["woba"] = round(woba_weighted / pa_total, 3) if pa_total > 0 else 0
                    source = data_source
                    coverage += 1

        # Compute MS for this batter
        ms_data = compute_ms(batter_id, cluster_id, atlas) if batter_id else {
            "ms": 0, "range": [0, 0], "woba": 0, "pa": 0, "source": "none"
        }

        # ── Stolen base projection (per-batter rate × assumed PA) ──
        if batter_id and batter_id in atlas.baserunning:
            br = atlas.baserunning[batter_id]
            proj["sb"] = round(br["sb_per_pa"] * assumed_pa, 4)
            proj["cs"] = round(br["cs_per_pa"] * assumed_pa, 4)

        # ── Tier adjustment: opposing pitcher quality scales hits ──
        tier_mult = 1.0
        if opposing_pitcher_id and opposing_pitcher_id in atlas.pitcher_tiers:
            pt = atlas.pitcher_tiers[opposing_pitcher_id]
            tier_mult = pt.get("effective_multiplier", 1.0)
        if tier_mult != 1.0:
            proj["h"] *= tier_mult
            proj["hr"] *= tier_mult
            proj["tb"] *= tier_mult

        # ── RE24 bonus: per-batter baserunning/context value ──
        re24_bonus = 0
        if batter_id:
            re24_rec = atlas.re24_data.get(f"{batter_id}_{year}")
            if not re24_rec:
                re24_rec = atlas.re24_data.get(f"{batter_id}_{year - 1}")
            if re24_rec:
                re24_bonus = re24_rec.get("RE24_above_avg", 0) * assumed_pa

        team_h += proj["h"]
        team_hr += proj["hr"]
        team_bb += proj["bb"]
        team_k += proj["k"]
        team_tb += proj["tb"]
        team_sb += proj["sb"]
        team_cs += proj["cs"]
        team_pa += assumed_pa
        team_re24 += re24_bonus

        # Compute batter's overall (career) wOBA as baseline
        overall_woba = 0
        overall_rows = atlas.batter_overall.get(batter_id, [])
        if overall_rows:
            ow_pa = sum(r.get("PA", 0) for r in overall_rows)
            ow_woba = sum((r.get("wOBA", 0)) * (r.get("PA", 0)) for r in overall_rows)
            if ow_pa > 0:
                overall_woba = round(ow_woba / ow_pa, 3)

        batter_details.append({
            "id": batter_id,
            "name": batter_name,
            "pos": batter.get("pos", ""),
            "bats": batter.get("bats", ""),
            "proj": proj,
            "source": source,
            "ms": ms_data,
            "overall_woba": overall_woba,
            "tier": ms_tier(ms_data["ms"]) if ms_data["ms"] > 0 else "unknown",
        })

    # ── OAA defense adjustment: opposing team's fielding reduces hits ──
    oaa_adj = 0
    oaa_data = atlas.team_oaa.get(opposing_team, {})
    if oaa_data:
        infield_oaa = oaa_data.get("infield_oaa", 0)
        outfield_oaa = oaa_data.get("outfield_oaa", 0)
        # GB-weighted defense factor (use opposing pitcher's GB rate if available)
        opp_ps = atlas.ps_index.get(opposing_pitcher_id, {})
        gb_rate = opp_ps.get("groundball_rate", 0.45)
        defense_factor = gb_rate * infield_oaa + (1.0 - gb_rate) * outfield_oaa
        if defense_factor != 0:
            hit_mult = max(0.8, min(1.2, 1.0 - defense_factor * 0.005))
            oaa_adj = team_h * (hit_mult - 1.0)
            team_h *= hit_mult

    runs = round(base_runs(team_h, team_hr, team_bb, team_tb, team_pa, team_sb, team_cs), 2)

    # ── RE24 bonus: lineup baserunning/context value ──
    re24_adj = round(team_re24, 3)
    runs += team_re24

    # ── Park factor: venue run environment ──
    park_mult = 1.0
    park_data = atlas.park_factors.get(venue_team)
    if isinstance(park_data, (int, float)):
        park_mult = park_data
    elif isinstance(park_data, dict):
        park_mult = park_data.get("park_mult", park_data.get("factor", 1.0))
    park_adj = runs * (park_mult - 1.0)
    runs *= park_mult

    # ── Bullpen adjustment: opposing bullpen quality ──
    bp_adj = 0
    bp_data = atlas.team_bullpen.get(opposing_team, {})
    if bp_data:
        bp_runs_per_ip = bp_data.get("runs_per_ip", 0)
        bp_above_avg = bp_data.get("above_avg_runs_per_ip", 0)
        # SP innings: how many innings does opposing SP cover?
        sp_ip = 5.85
        sp_data = atlas.sp_innings.get(opposing_pitcher_id)
        if isinstance(sp_data, dict):
            sp_ip = sp_data.get("ip_per_gs", 5.85)
        bp_ip = max(0, 9.0 - sp_ip)
        if bp_above_avg != 0:
            bp_adj = bp_ip * bp_above_avg
            runs += bp_adj

    runs = round(max(0, runs), 2)

    # Get tier info for display
    tier_info = atlas.pitcher_tiers.get(opposing_pitcher_id, {})
    tier_name = tier_info.get("tier", "—")
    tier_mult_val = tier_info.get("effective_multiplier", 1.0)

    return {
        "runs": runs,
        "coverage": coverage,
        "total": len(lineup),
        "lineup_pct": round(coverage / max(len(lineup), 1) * 100, 1),
        "batter_details": batter_details,
        "team_stats": {
            "h": round(team_h, 2),
            "hr": round(team_hr, 2),
            "bb": round(team_bb, 2),
            "k": round(team_k, 2),
            "tb": round(team_tb, 2),
            "sb": round(team_sb, 2),
            "cs": round(team_cs, 2),
            "pa": team_pa,
        },
        "opposing_pitcher": {
            "id": opposing_pitcher_id,
            "cluster": cluster_id,
            "archetype": archetype,
        },
        "adjustments": {
            "tier": tier_name,
            "tier_mult": round(tier_mult_val, 3),
            "re24": round(re24_adj, 3),
            "oaa": round(oaa_adj, 3),
            "park_mult": round(park_mult, 3),
            "park_adj": round(park_adj, 3),
            "bp_adj": round(bp_adj, 3),
        },
    }


# ═══════════════════════════════════════════════════════════
# GAME-LEVEL PROJECTIONS + EDGES
# ═══════════════════════════════════════════════════════════

def compute_game(game: dict, atlas: AtlasData) -> dict:
    """Compute full matchup projections for a single game.

    Adds: away_proj, home_proj, proj_total, proj_spread, edges
    """
    away_pitcher_id = game.get("away_pitcher", {}).get("id")
    home_pitcher_id = game.get("home_pitcher", {}).get("id")
    away_lineup = game.get("away_lineup", [])
    home_lineup = game.get("home_lineup", [])

    # Get pitcher archetype info
    for side, pid in [("away", away_pitcher_id), ("home", home_pitcher_id)]:
        if pid:
            cluster_id, archetype = get_pitcher_cluster(pid, atlas)
            ps = atlas.ps_index.get(pid, {})
            game[f"{side}_pitcher"]["cluster"] = cluster_id
            game[f"{side}_pitcher"]["archetype"] = archetype
            game[f"{side}_pitcher"]["is_sp"] = ps.get("is_sp", 0)
            game[f"{side}_pitcher"]["whiff_rate"] = ps.get("whiff_rate", 0)
            game[f"{side}_pitcher"]["groundball_rate"] = ps.get("groundball_rate", 0)
            game[f"{side}_pitcher"]["avg_velo_FF"] = ps.get("avg_velo_FF", 0)

            # Correct handedness from atlas (MLB API sometimes missing pitchHand)
            if "is_rhp" in ps:
                game[f"{side}_pitcher"]["throws"] = "R" if ps["is_rhp"] else "L"

            # Cluster profile
            cluster_profile = atlas.clusters.get(cluster_id, {})
            game[f"{side}_pitcher"]["archetype_short"] = cluster_profile.get("short_name", "")
            game[f"{side}_pitcher"]["archetype_color"] = cluster_profile.get("color", "#666")
        else:
            game[f"{side}_pitcher"]["cluster"] = None
            game[f"{side}_pitcher"]["archetype"] = None

    # Project teams (home lineup vs away pitcher, away lineup vs home pitcher)
    away_team = game.get("away_team", "")
    home_team = game.get("home_team", "")
    has_lineups = bool(away_lineup) and bool(home_lineup)

    empty_proj = {"runs": 0, "coverage": 0, "total": 0, "lineup_pct": 0,
                  "batter_details": [], "team_stats": {}, "opposing_pitcher": {},
                  "adjustments": {}}

    if has_lineups and home_pitcher_id:
        away_proj = project_team(away_lineup, home_pitcher_id, atlas,
                                 batting_team=away_team, opposing_team=home_team,
                                 venue_team=home_team)
    else:
        away_proj = dict(empty_proj)

    if has_lineups and away_pitcher_id:
        home_proj = project_team(home_lineup, away_pitcher_id, atlas,
                                 batting_team=home_team, opposing_team=away_team,
                                 venue_team=home_team)
    else:
        home_proj = dict(empty_proj)

    game["away_proj"] = away_proj
    game["home_proj"] = home_proj
    game["has_lineups"] = has_lineups

    # Game-level projections
    away_runs = away_proj["runs"]
    home_runs = home_proj["runs"]
    game["proj_total"] = round(away_runs + home_runs, 1)
    game["proj_spread"] = round(home_runs - away_runs, 1)  # Negative = home favored

    # ═══ ML-FIRST EDGE ENGINE ═══
    odds = game.get("odds", {})
    game["edges"] = {}

    away_team = game.get("away_team", "")
    home_team = game.get("home_team", "")

    def ml_to_implied_prob(ml):
        """Convert American odds to implied probability (no-vig)."""
        if ml is None:
            return None
        ml = float(ml)
        if ml < 0:
            return abs(ml) / (abs(ml) + 100)
        else:
            return 100 / (ml + 100)

    def ml_to_decimal(ml):
        """Convert American odds to decimal odds."""
        if ml is None:
            return None
        ml = float(ml)
        if ml < 0:
            return 1 + 100 / abs(ml)
        else:
            return 1 + ml / 100

    def run_diff_to_win_prob(run_diff):
        """Convert projected run differential to win probability.

        Based on MLB historical data: ~50% of games decided by 1 run,
        Pythagorean expectation with exponent 1.83.
        run_diff is away - home (positive = away favored).
        """
        import math
        if run_diff == 0:
            return 0.5
        # Average MLB game: ~4.5 runs per team
        avg_runs = 4.5
        away_runs = avg_runs + run_diff / 2
        home_runs = avg_runs - run_diff / 2
        away_runs = max(away_runs, 1.0)
        home_runs = max(home_runs, 1.0)
        # Pythagorean win% (exponent 1.83 for MLB)
        return away_runs ** 1.83 / (away_runs ** 1.83 + home_runs ** 1.83)

    def win_prob_to_rl_cover_prob(win_prob):
        """Estimate run line (-1.5) cover probability from win probability.

        In MLB, ~58% of wins are by 2+ runs historically.
        So P(cover -1.5) ≈ win_prob × 0.58
        P(cover +1.5) ≈ (1 - win_prob) + win_prob × 0.42  [lose + win by 1]
        """
        WIN_BY_2_PLUS_RATE = 0.58
        return win_prob * WIN_BY_2_PLUS_RATE

    if has_lineups:
        away_ml = odds.get("away_ml")
        home_ml = odds.get("home_ml")
        market_spread = odds.get("spread")
        market_total = odds.get("total")

        # Model win probability from projected run differential
        run_diff = away_runs - home_runs  # positive = away favored
        model_away_wp = run_diff_to_win_prob(run_diff)
        model_home_wp = 1 - model_away_wp

        # Market implied probabilities (remove vig by normalizing)
        if away_ml is not None and home_ml is not None:
            raw_away = ml_to_implied_prob(away_ml)
            raw_home = ml_to_implied_prob(home_ml)
            vig = raw_away + raw_home  # e.g., 1.05
            market_away_wp = raw_away / vig
            market_home_wp = raw_home / vig

            # ── MONEYLINE EDGE ──
            # EV = (model_prob × decimal_payout) - 1
            away_decimal = ml_to_decimal(away_ml)
            home_decimal = ml_to_decimal(home_ml)
            away_ml_ev = round((model_away_wp * away_decimal) - 1, 4)
            home_ml_ev = round((model_home_wp * home_decimal) - 1, 4)

            # ML edge = model probability - market probability
            away_ml_edge = round(model_away_wp - market_away_wp, 4)
            home_ml_edge = round(model_home_wp - market_home_wp, 4)

            # Pick the side with positive EV
            if away_ml_ev > home_ml_ev and away_ml_ev > 0:
                ml_side = away_team
                ml_edge = away_ml_edge
                ml_ev = away_ml_ev
                ml_odds = away_ml
                ml_model_wp = model_away_wp
                ml_market_wp = market_away_wp
            elif home_ml_ev > 0:
                ml_side = home_team
                ml_edge = home_ml_edge
                ml_ev = home_ml_ev
                ml_odds = home_ml
                ml_model_wp = model_home_wp
                ml_market_wp = market_home_wp
            else:
                ml_side = away_team if away_ml_ev > home_ml_ev else home_team
                ml_edge = max(away_ml_edge, home_ml_edge)
                ml_ev = max(away_ml_ev, home_ml_ev)
                ml_odds = away_ml if away_ml_ev > home_ml_ev else home_ml
                ml_model_wp = model_away_wp if away_ml_ev > home_ml_ev else model_home_wp
                ml_market_wp = market_away_wp if away_ml_ev > home_ml_ev else market_home_wp

            game["edges"]["ml"] = {
                "side": ml_side,
                "odds": ml_odds,
                "edge": round(ml_edge * 100, 1),  # as percentage points
                "ev": round(ml_ev * 100, 1),  # EV as percentage
                "model_wp": round(ml_model_wp * 100, 1),
                "market_wp": round(ml_market_wp * 100, 1),
                "is_pick": ml_ev > 0.02,  # Need 2%+ EV to flag
                "is_strong": ml_ev > 0.05,  # 5%+ EV is strong
            }

            # ── RUN LINE EDGE ──
            # Only compute if real RL odds are available from odds source
            rl_odds_data = odds.get("spread_odds", {})
            if rl_odds_data.get("away_price") and rl_odds_data.get("home_price"):
                away_rl_price = rl_odds_data["away_price"]
                home_rl_price = rl_odds_data["home_price"]
                away_rl_point = rl_odds_data.get("away_point", 1.5)
                home_rl_point = rl_odds_data.get("home_point", -1.5)

                away_rl_cover = win_prob_to_rl_cover_prob(model_away_wp) if away_rl_point < 0 else (1 - win_prob_to_rl_cover_prob(model_home_wp))
                home_rl_cover = 1 - away_rl_cover

                away_rl_decimal = ml_to_decimal(away_rl_price)
                home_rl_decimal = ml_to_decimal(home_rl_price)
                away_rl_ev = round((away_rl_cover * away_rl_decimal) - 1, 4)
                home_rl_ev = round((home_rl_cover * home_rl_decimal) - 1, 4)

                if away_rl_ev > home_rl_ev:
                    rl_side = away_team
                    rl_ev = away_rl_ev
                    rl_label = f"{away_team} {away_rl_point:+.1f}"
                    rl_odds_val = away_rl_price
                else:
                    rl_side = home_team
                    rl_ev = home_rl_ev
                    rl_label = f"{home_team} {home_rl_point:+.1f}"
                    rl_odds_val = home_rl_price

                game["edges"]["rl"] = {
                    "side": rl_side,
                    "label": rl_label,
                    "odds": rl_odds_val,
                    "ev": round(rl_ev * 100, 1),
                    "is_pick": rl_ev > 0.02,
                    "is_strong": rl_ev > 0.05,
                }

        # ── TOTAL (O/U) EDGE ──
        if market_total is not None:
            proj_total = game["proj_total"]
            total_edge = round(proj_total - market_total, 1)
            game["edges"]["total"] = {
                "edge": total_edge,
                "market": market_total,
                "projected": proj_total,
                "is_pick": abs(total_edge) >= EDGE_THRESHOLD_TOTAL,
                "is_strong": abs(total_edge) >= EDGE_THRESHOLD_TOTAL * STRONG_EDGE_MULTIPLIER,
                "direction": "over" if total_edge > 0 else "under",
            }

        # Store model probabilities on game for display
        game["model_wp"] = {
            "away": round(model_away_wp * 100, 1),
            "home": round(model_home_wp * 100, 1),
        }

    # Average MS for each lineup
    for side in ["away", "home"]:
        details = game[f"{side}_proj"]["batter_details"]
        scored = [d for d in details if d["ms"]["ms"] > 0]
        if scored:
            game[f"{side}_avg_ms"] = round(sum(d["ms"]["ms"] for d in scored) / len(scored), 1)
        else:
            game[f"{side}_avg_ms"] = 0

    # Compute confidence scores
    game["confidence"] = compute_confidence(game)

    return game


# ═══════════════════════════════════════════════════════════
# CONFIDENCE SCORING (1-10 scale, matches NBA SIM pattern)
# ═══════════════════════════════════════════════════════════

def _conf_color(conf: int) -> str:
    """Map 1-10 confidence to color (matches NBA SIM palette)."""
    if conf >= 8:
        return "#00FF55"   # Green
    elif conf >= 6:
        return "#7FFF00"   # Lime
    elif conf >= 4:
        return "#FFD600"   # Yellow
    elif conf >= 2:
        return "#FF8C00"   # Orange
    else:
        return "#FF3333"   # Red


def _conf_label(conf: int, side: str) -> str:
    """Generate confidence label for spread pick."""
    if conf >= 8:
        return f"TAKE {side}"
    elif conf >= 5:
        return f"LEAN {side}"
    else:
        return "TOSS-UP"


def _conf_label_ou(conf: int, direction: str) -> str:
    """Generate confidence label for O/U pick."""
    dir_str = direction.upper()
    if conf >= 8:
        return f"TAKE {dir_str}"
    elif conf >= 5:
        return f"LEAN {dir_str}"
    else:
        return "TOSS-UP"


def compute_confidence(game: dict) -> dict:
    """Compute confidence scores (1-10) for spread and O/U picks.

    MLB confidence factors:
      1. Edge magnitude (primary) — bigger edge = higher confidence
      2. Coverage quality — % of lineup batters with archetype data
      3. Lineup status — confirmed vs expected
      4. Spring training discount — ST caps confidence at CONF_ST_CAP

    MLB edges are smaller than NBA (1-3 runs vs 3-10 pts), so calibration
    differs: edge 1.0 run -> ~5, edge 2.0 -> ~8, edge 3.0 -> 10.
    """
    edges = game.get("edges", {})
    is_st = game.get("is_spring_training", False)

    # Average coverage across both lineups
    away_cov = game.get("away_proj", {}).get("lineup_pct", 0)
    home_cov = game.get("home_proj", {}).get("lineup_pct", 0)
    avg_coverage = (away_cov + home_cov) / 2  # 0-100

    # Coverage factor: scales from 0.6 (0% coverage) to 1.0 (100% coverage)
    coverage_factor = 0.6 + 0.4 * (avg_coverage / 100)

    # Lineup status bonus
    lineup_status = game.get("lineup_status", "expected")
    status_bonus = 1 if lineup_status == "confirmed" else 0

    result = {}

    # ── ML confidence (primary) ──
    if "ml" in edges:
        ml_ev = edges["ml"]["ev"]  # EV as percentage (e.g., 5.0 = 5%)
        # MLB calibration: 2% EV = 3, 5% = 5, 8% = 7, 12%+ = 10
        base_conf = min(10, max(1, round(ml_ev * 0.7 + 1.5)))
        adj_conf = base_conf * coverage_factor + status_bonus
        spread_conf = max(1, min(10, round(adj_conf)))
        if is_st:
            spread_conf = min(spread_conf, CONF_ST_CAP)

        result["spread_conf"] = spread_conf
        result["spread_conf_color"] = _conf_color(spread_conf)
        result["spread_conf_label"] = _conf_label(spread_conf, edges["ml"]["side"])

    # ── Total (O/U) confidence ──
    if "total" in edges:
        total_edge = abs(edges["total"]["edge"])
        # MLB totals: edge 1.5 -> ~3, edge 3.0 -> ~4, edge 6.0 -> ~6
        base_conf = min(10, max(1, round(total_edge / 1.5 + 2)))
        adj_conf = base_conf * coverage_factor + status_bonus
        total_conf = max(1, min(10, round(adj_conf)))
        if is_st:
            total_conf = min(total_conf, CONF_ST_CAP)

        result["total_conf"] = total_conf
        result["total_conf_color"] = _conf_color(total_conf)
        result["total_conf_label"] = _conf_label_ou(total_conf, edges["total"]["direction"])

    return result


# ═══════════════════════════════════════════════════════════
# TOP PICKS SELECTION
# ═══════════════════════════════════════════════════════════

def select_top_picks(games: list, max_picks: int = 5) -> list:
    """Select the best daily picks based on confidence and edge strength."""
    candidates = []

    for game in games:
        if not game.get("has_lineups"):
            continue

        edges = game.get("edges", {})
        conf = game.get("confidence", {})

        # Spread picks
        if edges.get("spread", {}).get("is_pick"):
            se = edges["spread"]
            spread_conf = conf.get("spread_conf", 5)
            candidates.append({
                "type": "spread",
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "game_pk": game.get("game_pk"),
                "side": se["side"],
                "line": se["market"],
                "projected": se["projected"],
                "edge": abs(se["edge"]),
                "confidence": spread_conf,
                "conf_color": conf.get("spread_conf_color", "#FFD600"),
                "conf_label": conf.get("spread_conf_label", ""),
                "is_strong": se.get("is_strong", False),
                "is_spring_training": game.get("is_spring_training", False),
            })

        # Total picks
        if edges.get("total", {}).get("is_pick"):
            te = edges["total"]
            total_conf = conf.get("total_conf", 5)
            candidates.append({
                "type": "total",
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "game_pk": game.get("game_pk"),
                "direction": te["direction"],
                "line": te["market"],
                "projected": te["projected"],
                "edge": abs(te["edge"]),
                "confidence": total_conf,
                "conf_color": conf.get("total_conf_color", "#FFD600"),
                "conf_label": conf.get("total_conf_label", ""),
                "is_strong": te.get("is_strong", False),
                "is_spring_training": game.get("is_spring_training", False),
            })

    # Sort by confidence first, then edge as tiebreaker
    candidates.sort(key=lambda x: (x.get("confidence", 0), x["edge"]), reverse=True)
    return candidates[:max_picks]


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  MLB MATCHUP ENGINE")
    print(f"{'='*60}\n")

    # Load ATLAS
    atlas = AtlasData()
    atlas.load()

    # Load daily lineups
    if not os.path.exists(DAILY_LINEUPS_FILE):
        print(f"[ERROR] {DAILY_LINEUPS_FILE} not found — run collect_daily.py first")
        sys.exit(1)

    with open(DAILY_LINEUPS_FILE) as f:
        daily = json.load(f)

    games = daily.get("games", [])
    print(f"\n[Compute] Processing {len(games)} games from {daily.get('slate_date', '?')}...\n")

    # Compute matchups for each game
    for i, game in enumerate(games):
        st_tag = " [ST]" if game.get("is_spring_training") else ""
        print(f"  [{i+1}/{len(games)}] {game['away_team']} @ {game['home_team']}{st_tag}", end="")

        game = compute_game(game, atlas)
        games[i] = game

        # Summary
        ar = game["away_proj"]["runs"]
        hr = game["home_proj"]["runs"]
        ac = game["away_proj"]["coverage"]
        hc = game["home_proj"]["coverage"]
        print(f" — {ar:.1f} vs {hr:.1f} (coverage: {ac}/{game['away_proj']['total']}, {hc}/{game['home_proj']['total']})")

    # Select top picks
    top_picks = select_top_picks(games)

    # Build output
    output = {
        "slate_date": daily.get("slate_date"),
        "computed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "game_count": len(games),
        "spring_training_count": sum(1 for g in games if g.get("is_spring_training")),
        "games": games,
        "top_picks": top_picks,
    }

    with open(DAILY_GAMES_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] Wrote {len(games)} enriched games to {DAILY_GAMES_FILE}")

    # Top picks summary
    if top_picks:
        print(f"\n{'='*60}")
        print(f"  TOP PICKS ({len(top_picks)})")
        print(f"{'='*60}")
        for p in top_picks:
            strong = " 🔥" if p["is_strong"] else ""
            st = " [ST]" if p.get("is_spring_training") else ""
            if p["type"] == "spread":
                print(f"  {p['matchup']}{st} — {p['side']} {p['line']:+.1f} (edge {p['edge']:.1f}){strong}")
            else:
                print(f"  {p['matchup']}{st} — {p['direction'].upper()} {p['line']} (edge {p['edge']:.1f}){strong}")
    else:
        print("\n  No picks meet edge thresholds today.")


if __name__ == "__main__":
    main()
