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

def project_team(lineup, opposing_pitcher_id: int, atlas: AtlasData, pa_override: int = None):
    """Project team offense against opposing pitcher using ATLAS matchups.

    Args:
        lineup: list of dicts with "id" key (MLBAM batter IDs)
        opposing_pitcher_id: MLBAM ID of opposing pitcher
        atlas: loaded AtlasData
        pa_override: assumed PA per batter (default from config)

    Returns dict with: runs, coverage, total, batter_details, team_stats
    """
    cluster_id, archetype = get_pitcher_cluster(opposing_pitcher_id, atlas) if opposing_pitcher_id else (None, None)
    assumed_pa = pa_override or DEFAULT_PA_PER_BATTER

    team_h, team_hr, team_bb, team_k, team_tb, team_pa = 0, 0, 0, 0, 0, 0
    team_sb, team_cs = 0, 0
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

        team_h += proj["h"]
        team_hr += proj["hr"]
        team_bb += proj["bb"]
        team_k += proj["k"]
        team_tb += proj["tb"]
        team_sb += proj["sb"]
        team_cs += proj["cs"]
        team_pa += assumed_pa

        batter_details.append({
            "id": batter_id,
            "name": batter_name,
            "pos": batter.get("pos", ""),
            "bats": batter.get("bats", ""),
            "proj": proj,
            "source": source,
            "ms": ms_data,
            "tier": ms_tier(ms_data["ms"]) if ms_data["ms"] > 0 else "unknown",
        })

    runs = round(base_runs(team_h, team_hr, team_bb, team_tb, team_pa, team_sb, team_cs), 2)

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

            # Cluster profile
            cluster_profile = atlas.clusters.get(cluster_id, {})
            game[f"{side}_pitcher"]["archetype_short"] = cluster_profile.get("short_name", "")
            game[f"{side}_pitcher"]["archetype_color"] = cluster_profile.get("color", "#666")
        else:
            game[f"{side}_pitcher"]["cluster"] = None
            game[f"{side}_pitcher"]["archetype"] = None

    # Project teams (home lineup vs away pitcher, away lineup vs home pitcher)
    has_lineups = bool(away_lineup) and bool(home_lineup)

    if has_lineups and home_pitcher_id:
        away_proj = project_team(away_lineup, home_pitcher_id, atlas)
    else:
        away_proj = {"runs": 0, "coverage": 0, "total": 0, "lineup_pct": 0,
                      "batter_details": [], "team_stats": {}, "opposing_pitcher": {}}

    if has_lineups and away_pitcher_id:
        home_proj = project_team(home_lineup, away_pitcher_id, atlas)
    else:
        home_proj = {"runs": 0, "coverage": 0, "total": 0, "lineup_pct": 0,
                      "batter_details": [], "team_stats": {}, "opposing_pitcher": {}}

    game["away_proj"] = away_proj
    game["home_proj"] = home_proj
    game["has_lineups"] = has_lineups

    # Game-level projections
    away_runs = away_proj["runs"]
    home_runs = home_proj["runs"]
    game["proj_total"] = round(away_runs + home_runs, 1)
    game["proj_spread"] = round(home_runs - away_runs, 1)  # Negative = home favored

    # Edge vs market
    odds = game.get("odds", {})
    game["edges"] = {}

    if "spread" in odds and has_lineups:
        market_spread = odds["spread"]  # Home team spread
        proj_spread = game["proj_spread"]
        # Edge: positive means our model thinks home team is better than market
        spread_edge = round(proj_spread - market_spread, 1)
        game["edges"]["spread"] = {
            "edge": spread_edge,
            "market": market_spread,
            "projected": proj_spread,
            "is_pick": abs(spread_edge) >= EDGE_THRESHOLD_SPREAD,
            "is_strong": abs(spread_edge) >= EDGE_THRESHOLD_SPREAD * STRONG_EDGE_MULTIPLIER,
            "side": game["home_team"] if spread_edge < 0 else game["away_team"],
        }

    if "total" in odds and has_lineups:
        market_total = odds["total"]
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

    # Average MS for each lineup
    for side in ["away", "home"]:
        details = game[f"{side}_proj"]["batter_details"]
        scored = [d for d in details if d["ms"]["ms"] > 0]
        if scored:
            game[f"{side}_avg_ms"] = round(sum(d["ms"]["ms"] for d in scored) / len(scored), 1)
        else:
            game[f"{side}_avg_ms"] = 0

    return game


# ═══════════════════════════════════════════════════════════
# TOP PICKS SELECTION
# ═══════════════════════════════════════════════════════════

def select_top_picks(games: list, max_picks: int = 5) -> list:
    """Select the best daily picks based on edge strength.

    Returns list of pick dicts with game reference, edge details, confidence.
    """
    candidates = []

    for game in games:
        if not game.get("has_lineups"):
            continue

        edges = game.get("edges", {})

        # Spread picks
        if edges.get("spread", {}).get("is_pick"):
            se = edges["spread"]
            candidates.append({
                "type": "spread",
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "game_pk": game.get("game_pk"),
                "side": se["side"],
                "line": se["market"],
                "projected": se["projected"],
                "edge": abs(se["edge"]),
                "is_strong": se.get("is_strong", False),
                "is_spring_training": game.get("is_spring_training", False),
            })

        # Total picks
        if edges.get("total", {}).get("is_pick"):
            te = edges["total"]
            candidates.append({
                "type": "total",
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "game_pk": game.get("game_pk"),
                "direction": te["direction"],
                "line": te["market"],
                "projected": te["projected"],
                "edge": abs(te["edge"]),
                "is_strong": te.get("is_strong", False),
                "is_spring_training": game.get("is_spring_training", False),
            })

    # Sort by edge size (descending)
    candidates.sort(key=lambda x: x["edge"], reverse=True)
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
