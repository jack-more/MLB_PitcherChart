#!/usr/bin/env python3
"""
Static HTML generator for MLB SIM.

Reads pre-computed daily_games.json and produces a self-contained
mlbsim.html with all data embedded inline (~100-200KB vs 130MB+ client-side).

Follows the same generate_frontend.py pattern from NBAsim:
  - Python f-string templating
  - Data embedded in HTML data-attributes and inline markup
  - Full CSS + JS in a single .html file

Usage:
    python scripts/generate_mlbsim.py
"""

import json
import os
import sys
import html as html_lib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import (
    FRONTEND_PUBLIC, DAILY_GAMES_FILE, DAILY_DATA_DIR,
    EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL, STRONG_EDGE_MULTIPLIER,
)
from scripts.mlb_teams import TEAM_COLORS, ABBR_TO_FULL_NAME, team_logo_url


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _esc(s):
    """HTML-escape a string."""
    return html_lib.escape(str(s)) if s else ""


def ms_class(ms):
    """Return CSS class for Matchup Score tier."""
    if ms >= 85:
        return "ms-elite"
    elif ms >= 70:
        return "ms-favorable"
    elif ms >= 55:
        return "ms-neutral"
    else:
        return "ms-tough"


def ms_signal(ms):
    """Return signal label for Matchup Score."""
    if ms >= 85:
        return "CRUSH"
    elif ms >= 70:
        return "FAVORABLE"
    elif ms >= 55:
        return "NEUTRAL"
    else:
        return "FADE"


def format_spread(val):
    """Format a spread value for display."""
    if val is None:
        return "—"
    v = float(val)
    if v > 0:
        return f"+{v:.1f}"
    return f"{v:.1f}"


def format_total(val):
    """Format over/under total."""
    if val is None:
        return "—"
    return f"{float(val):.1f}"


def format_ml(val):
    """Format moneyline value."""
    if val is None:
        return "—"
    v = int(val)
    return f"+{v}" if v > 0 else str(v)


def _pct(val):
    """Format decimal as percentage string."""
    if val is None:
        return "—"
    return f"{float(val)*100:.0f}%"


# ═══════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def render_run_bar(game):
    """Render the colored run bar at top of game card."""
    away = game.get("away_team", "???")
    home = game.get("home_team", "???")
    away_runs = game.get("away_projected_runs", 0)
    home_runs = game.get("home_projected_runs", 0)
    total = away_runs + home_runs
    if total == 0:
        total = 1

    away_pct = max(20, (away_runs / total) * 100)
    home_pct = max(20, 100 - away_pct)

    away_color = TEAM_COLORS.get(away, "#666")
    home_color = TEAM_COLORS.get(home, "#666")

    return f'''<div class="run-bar">
  <div class="run-bar-seg" style="width:{away_pct:.0f}%;background:{away_color}">{away_runs:.1f}</div>
  <div class="run-bar-seg" style="width:{home_pct:.0f}%;background:{home_color}">{home_runs:.1f}</div>
</div>'''


def render_card_header(game):
    """Render the team logos + spread center block."""
    away = game.get("away_team", "???")
    home = game.get("home_team", "???")
    odds = game.get("odds", {})

    spread_val = odds.get("spread")
    total_val = odds.get("total")
    spread_display = format_spread(spread_val) if spread_val is not None else "—"
    total_display = f"O/U {format_total(total_val)}" if total_val is not None else ""

    # Pick badge
    pick_html = ""
    picks = game.get("picks", [])
    for pick in picks:
        pick_type = pick.get("type", "")
        pick_label = _esc(pick.get("label", ""))
        strength = pick.get("strength", "lean")
        badge_class = "sim-pick" if strength == "strong" else "sim-pick"
        pick_html += f'<div class="{badge_class}">{pick_label}</div>'

    away_logo = team_logo_url(away)
    home_logo = team_logo_url(home)

    return f'''<div class="card-header">
  <div class="team-block">
    <div class="team-logo"><img src="{away_logo}" alt="{_esc(away)}" style="width:100%;height:100%;object-fit:contain"></div>
    <div class="team-abbr">{_esc(away)}</div>
  </div>
  <div class="card-center">
    <div class="proj-label">SPREAD</div>
    <div class="spread">{spread_display}</div>
    <div class="ou-line">{total_display}</div>
    {pick_html}
  </div>
  <div class="team-block">
    <div class="team-logo"><img src="{home_logo}" alt="{_esc(home)}" style="width:100%;height:100%;object-fit:contain"></div>
    <div class="team-abbr">{_esc(home)}</div>
  </div>
</div>'''


def render_sp_block(game):
    """Render starting pitcher matchup section."""
    away_sp = game.get("away_pitcher", {})
    home_sp = game.get("home_pitcher", {})

    def sp_side(sp):
        name = sp.get("name", "TBD")
        arch = sp.get("archetype", "")
        throws = sp.get("throws", "?")
        cluster = sp.get("cluster", "")
        is_nri = sp.get("source") == "nri"

        if is_nri:
            arch_badge = '<span class="arch-badge" style="background:#666;color:#ccc">UNCHARTED</span>'
        elif arch:
            arch_badge = f'<span class="arch-badge">{_esc(arch)}</span>'
        else:
            arch_badge = ""

        stats_parts = []
        if throws:
            stats_parts.append(f"{throws}HP")
        ps = sp.get("pitcher_stats", {})
        if ps.get("whiff_rate"):
            stats_parts.append(f"Whiff {_pct(ps['whiff_rate'])}")
        if ps.get("groundball_rate"):
            stats_parts.append(f"GB {_pct(ps['groundball_rate'])}")

        stats_str = " · ".join(stats_parts) if stats_parts else ""

        return f'''<div class="sp-side">
      <div class="sp-name">{_esc(name)}</div>
      {arch_badge}
      <div class="sp-stats">{stats_str}</div>
    </div>'''

    return f'''<div class="sp-block">
  {sp_side(away_sp)}
  <div class="sp-vs">VS</div>
  {sp_side(home_sp)}
</div>'''


def render_game_type_badge(game):
    """Render spring training or other game type badge."""
    if game.get("is_spring_training"):
        return '<div class="game-type-badge st-badge">🌴 SPRING TRAINING</div>'
    game_type = game.get("game_type", "R")
    if game_type == "E":
        return '<div class="game-type-badge wbc-badge">EXHIBITION</div>'
    return ""


def render_batter_row(batter, order_num):
    """Render a single batter row in the lineup grid."""
    name = batter.get("name", "Unknown")
    ms = batter.get("ms", 0)
    woba = batter.get("woba", 0)
    pa = batter.get("pa", 0)
    k_pct = batter.get("k_pct")
    pos = batter.get("pos", "")
    bats = batter.get("bats", "")
    is_nri = batter.get("source") == "nri"

    if is_nri:
        ms_display = '<span style="color:#999;font-size:12px">NRI</span>'
        ms_cls = ""
    else:
        ms_display = str(int(ms)) if ms else "—"
        ms_cls = ms_class(ms) if ms else ""

    # Build stats line
    stats_parts = []
    if pos:
        stats_parts.append(pos)
    if bats:
        stats_parts.append(f"{bats}")
    if woba and not is_nri:
        stats_parts.append(f".{int(woba*1000):03d}")
    if pa and not is_nri:
        stats_parts.append(f"{pa}PA")

    stats_str = " · ".join(stats_parts)

    # Range
    range_str = ""
    if batter.get("ms_range") and not is_nri:
        r = batter["ms_range"]
        range_str = f'{r.get("low", "")}-{r.get("high", "")}'

    return f'''<div class="batter-row">
  <div class="batter-top">
    <span class="batter-order">{order_num}</span>
    <span class="batter-name">{_esc(name)}</span>
    <span class="batter-ms {ms_cls}">{ms_display}</span>
  </div>
  <div class="batter-bottom">
    <span class="batter-stats">{stats_str}</span>
    <span class="batter-range">{range_str}</span>
  </div>
</div>'''


def render_lineup_grid(game, game_idx):
    """Render the collapsible lineup grid for both teams."""
    away = game.get("away_team", "???")
    home = game.get("home_team", "???")
    away_lineup = game.get("away_lineup", [])
    home_lineup = game.get("home_lineup", [])

    if not away_lineup and not home_lineup:
        status = game.get("lineup_status", "TBD")
        return f'''<div class="tbd-block">
  LINEUPS {status.upper()} — CHECK BACK CLOSER TO GAME TIME
</div>'''

    # Away batters
    away_rows = ""
    for i, b in enumerate(away_lineup):
        away_rows += render_batter_row(b, i + 1)

    # Home batters
    home_rows = ""
    for i, b in enumerate(home_lineup):
        home_rows += render_batter_row(b, i + 1)

    # Coverage stats
    away_cov = game.get("away_coverage", {})
    home_cov = game.get("home_coverage", {})
    away_cov_pct = away_cov.get("pct", 0)
    home_cov_pct = home_cov.get("pct", 0)
    away_cov_cls = "cov-good" if away_cov_pct >= 70 else "cov-warn"
    home_cov_cls = "cov-good" if home_cov_pct >= 70 else "cov-warn"

    return f'''<div class="lineup-toggle" onclick="toggleLineup({game_idx})">
  <span>LINEUP MATCHUPS</span>
  <span class="arrow" id="arrow-{game_idx}">▼</span>
</div>
<div class="lineup-grid" id="lineup-{game_idx}">
  <div class="lineup-col">
    <div class="lineup-col-hdr">{_esc(away)} LINEUP</div>
    {away_rows}
    <div class="coverage {away_cov_cls}">ATLAS: {away_cov_pct:.0f}% COVERED ({away_cov.get('known', 0)}/{away_cov.get('total', 0)})</div>
  </div>
  <div class="lineup-col">
    <div class="lineup-col-hdr">{_esc(home)} LINEUP</div>
    {home_rows}
    <div class="coverage {home_cov_cls}">ATLAS: {home_cov_pct:.0f}% COVERED ({home_cov.get('known', 0)}/{home_cov.get('total', 0)})</div>
  </div>
</div>'''


def render_bullpen_section(game, side, game_idx):
    """Render bullpen prediction for one team in a game."""
    bp_data = game.get(f"{side}_bullpen", {})
    if not bp_data:
        return ""

    team = game.get(f"{side}_team", "???")
    predicted = bp_data.get("predicted", [])
    sp_ip = bp_data.get("starter_expected_ip", "?")
    bp_needed = bp_data.get("bullpen_ip_needed", "?")
    avail = bp_data.get("available_count", 0)
    total = bp_data.get("total_relievers", 0)
    conf = bp_data.get("confidence", "?")

    if not predicted:
        return ""

    rows = ""
    for rp in predicted:
        name = rp.get("name", "Unknown")
        role = rp.get("role", "")
        arch = rp.get("archetype", "")
        avail_score = rp.get("availability", 0)
        exp_ip = rp.get("expected_ip", 0)
        exp_inn = rp.get("expected_inning", 0)
        avg_ms = rp.get("avg_ms_faced", 0)

        role_badge = role.upper()[:3] if role else ""
        avail_bar_width = int(avail_score * 100)
        ms_text = f"MS {avg_ms:.0f}" if avg_ms else ""

        rows += f'''<div class="bp-row">
  <div class="bp-role">{role_badge}</div>
  <div class="bp-info">
    <div class="bp-name">{_esc(name)}</div>
    <div class="bp-meta">{_esc(arch)} · ~{exp_ip:.0f}IP · Inn {exp_inn} {ms_text}</div>
  </div>
  <div class="bp-avail">
    <div class="bp-avail-bar"><div class="bp-avail-fill" style="width:{avail_bar_width}%"></div></div>
  </div>
</div>'''

    conf_cls = f"bp-conf-{conf}" if conf in ("high", "medium", "low") else ""

    return f'''<div class="bp-section">
  <div class="bp-header">
    <span class="bp-team">{_esc(team)} PEN</span>
    <span class="bp-meta-hdr">SP ~{sp_ip} IP · {bp_needed} IP needed · {avail}/{total} avail</span>
    <span class="bp-confidence {conf_cls}">{conf.upper()}</span>
  </div>
  {rows}
</div>'''


def render_game_card(game, game_idx):
    """Render a complete game card."""
    badge = render_game_type_badge(game)
    run_bar = render_run_bar(game)
    header = render_card_header(game)
    sp_block = render_sp_block(game)
    lineup = render_lineup_grid(game, game_idx)

    # Bullpen sections
    bp_away = render_bullpen_section(game, "away", game_idx)
    bp_home = render_bullpen_section(game, "home", game_idx)
    bp_html = ""
    if bp_away or bp_home:
        bp_html = f'''<div class="bp-toggle" onclick="toggleBullpen({game_idx})">
  <span>BULLPEN PREDICTIONS</span>
  <span class="arrow" id="bp-arrow-{game_idx}">▼</span>
</div>
<div class="bp-container" id="bullpen-{game_idx}">
  {bp_away}
  {bp_home}
</div>'''

    # Edge info
    edge_html = ""
    edge_data = game.get("edge", {})
    if edge_data:
        spread_edge = edge_data.get("spread_edge")
        total_edge = edge_data.get("total_edge")
        parts = []
        if spread_edge and abs(spread_edge) >= EDGE_THRESHOLD_SPREAD:
            parts.append(f"Spread Edge: {spread_edge:+.1f}")
        if total_edge and abs(total_edge) >= EDGE_THRESHOLD_TOTAL:
            direction = "OVER" if total_edge > 0 else "UNDER"
            parts.append(f"Total Edge: {direction} {abs(total_edge):.1f}")
        if parts:
            edge_html = f'<div class="edge-bar">{" · ".join(parts)}</div>'

    game_time = game.get("game_time_et", "")
    venue = game.get("venue", "")
    meta_parts = []
    if game_time:
        meta_parts.append(game_time)
    if venue:
        meta_parts.append(venue)
    meta_str = " · ".join(meta_parts)
    meta_html = f'<div class="game-meta">{_esc(meta_str)}</div>' if meta_str else ""

    return f'''<div class="game-card">
  {badge}
  {run_bar}
  {header}
  {edge_html}
  {sp_block}
  {meta_html}
  {lineup}
  {bp_html}
</div>'''


def render_top_picks(games):
    """Render the top picks sidebar section."""
    picks = []
    for game in games:
        game_picks = game.get("picks", [])
        for pick in game_picks:
            picks.append({
                **pick,
                "away": game.get("away_team", ""),
                "home": game.get("home_team", ""),
                "matchup": f"{game.get('away_team', '')} @ {game.get('home_team', '')}",
            })

    if not picks:
        return '<div class="empty-state">NO STRONG EDGES TODAY</div>'

    # Sort by edge magnitude
    picks.sort(key=lambda p: abs(p.get("edge", 0)), reverse=True)

    rows = ""
    for i, pick in enumerate(picks):
        label = _esc(pick.get("label", ""))
        matchup = _esc(pick.get("matchup", ""))
        edge = pick.get("edge", 0)
        strength = pick.get("strength", "lean")
        pick_cls = "gp-pick-strong" if strength == "strong" else "gp-pick-lean"

        rows += f'''<div class="pick-row">
  <div class="pick-rank">{i+1}</div>
  <div class="pick-info">
    <div class="pick-label {pick_cls}">{label}</div>
    <div class="pick-matchup">{matchup}</div>
  </div>
  <div class="pick-edge">{edge:+.1f}</div>
</div>'''

    return f'''<div class="section-title">TODAY'S EDGES</div>
<div class="section-sub">Games with projected spread/total edge</div>
<div class="picks-container">{rows}</div>'''


def render_trends(games):
    """Render top Matchup Scores across all games."""
    all_batters = []
    for game in games:
        for side in ("away", "home"):
            lineup = game.get(f"{side}_lineup", [])
            opp = game.get("home_team" if side == "away" else "away_team", "")
            sp_name = game.get(f"{'home' if side == 'away' else 'away'}_pitcher", {}).get("name", "TBD")
            team = game.get(f"{side}_team", "")
            for b in lineup:
                if b.get("source") == "nri":
                    continue
                ms = b.get("ms", 0)
                if ms > 0:
                    all_batters.append({
                        **b,
                        "team": team,
                        "opp": opp,
                        "vs_pitcher": sp_name,
                    })

    # Sort by MS descending
    all_batters.sort(key=lambda x: x.get("ms", 0), reverse=True)
    top = all_batters[:20]

    if not top:
        return '<div class="empty-state">NO MATCHUP DATA AVAILABLE</div>'

    rows = ""
    for i, b in enumerate(top):
        name = _esc(b.get("name", ""))
        team = _esc(b.get("team", ""))
        ms = int(b.get("ms", 0))
        ms_cls = ms_class(ms)
        woba = b.get("woba", 0)
        opp = _esc(b.get("opp", ""))
        vs = _esc(b.get("vs_pitcher", ""))

        rows += f'''<div class="trend-row">
  <div class="trend-rank">{i+1}</div>
  <div class="trend-info">
    <div class="trend-name">{name}</div>
    <div class="trend-meta">{team} vs {vs} ({opp}) · .{int(woba*1000):03d} wOBA</div>
  </div>
  <div class="trend-right">
    <div class="trend-ms {ms_cls}">{ms}</div>
  </div>
</div>'''

    return rows


# ═══════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════

def generate_css():
    """Return the complete CSS string."""
    return '''/* ═══ RESET ═══ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%;font-size:16px}
body{font-family:'Inter',sans-serif;background:var(--color-bg);color:var(--color-text);min-height:100vh;overflow-x:hidden;padding-bottom:80px}

/* ═══ DESIGN TOKENS ═══ */
:root{
    --color-bg:#9E9E9E;
    --color-card:#FFFFFF;
    --color-accent:#FFEA00;
    --color-black:#080808;
    --color-text:#000000;
    --color-meta:#4A4A4A;
    --color-elite:#00A334;
    --color-favorable:#2D8B4E;
    --color-neutral:#D4C000;
    --color-tough:#FF3333;
    --font-display:'Anton',sans-serif;
    --font-body:'Inter',sans-serif;
    --font-mono:'JetBrains Mono',monospace;
    --shadow:4px 4px 0px var(--color-black);
    --border:2px solid var(--color-black);
}

/* ═══ BACKGROUND ═══ */
body::before{
    content:'';position:fixed;top:0;left:0;width:100%;height:100%;
    background-image:radial-gradient(circle,rgba(0,0,0,0.12) 1px,transparent 1px);
    background-size:20px 20px;
    pointer-events:none;z-index:0;
}
body::after{
    content:'';position:fixed;top:0;left:0;width:100%;height:100%;
    background:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E");
    pointer-events:none;z-index:0;opacity:0.5;
}

/* ═══ HEADER ═══ */
.header{position:sticky;top:0;z-index:100;background:var(--color-bg)}
.brand-row{display:flex;align-items:center;justify-content:space-between;padding:10px 16px}
.logo{font-family:var(--font-display);font-size:28px;color:var(--color-accent);background:var(--color-black);padding:4px 14px;transform:skewX(-5deg);display:inline-block;letter-spacing:2px;line-height:1.1}
.byline{font-family:var(--font-body);font-size:11px;font-weight:400;color:#555;letter-spacing:0.5px;font-style:italic}
.atlas-btn{font-family:var(--font-mono);font-size:9px;font-weight:700;color:var(--color-meta);text-decoration:none;letter-spacing:1px;padding:4px 10px;border:1.5px solid var(--color-meta);border-radius:12px;transition:all 0.15s}
.atlas-btn:active{background:var(--color-black);color:#fff}
.status-dot{width:12px;height:12px;border-radius:50%;background:var(--color-accent);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(255,234,0,0.4)}50%{opacity:0.7;box-shadow:0 0 0 8px rgba(255,234,0,0)}}


/* ═══ MAIN CONTAINER ═══ */
.container{position:relative;z-index:1;max-width:480px;margin:0 auto;padding:0 12px}

/* ═══ TAB CONTENT ═══ */
.tab-content{display:none}
.tab-content.active{display:block}

/* ═══ FILTER CHIPS ═══ */
.chips{display:flex;gap:8px;overflow-x:auto;padding:12px 0;-webkit-overflow-scrolling:touch;scrollbar-width:none}
.chips::-webkit-scrollbar{display:none}
.chip{font-family:var(--font-mono);font-size:10px;font-weight:700;text-transform:uppercase;padding:6px 14px;background:var(--color-card);border:var(--border);border-radius:20px;box-shadow:2px 2px 0 var(--color-black);white-space:nowrap;cursor:pointer;transition:all 0.15s}
.chip.active{background:var(--color-accent);color:var(--color-black)}
.chip:active{transform:translate(2px,2px);box-shadow:none}

/* ═══ SLATE INFO ═══ */
.slate-info{display:flex;justify-content:space-between;align-items:center;padding:4px 0 10px;font-family:var(--font-mono);font-size:10px;text-transform:uppercase;color:var(--color-meta);letter-spacing:1px}

/* ═══ GAME CARD ═══ */
.game-card{background:var(--color-card);border:var(--border);box-shadow:var(--shadow);margin-bottom:16px;overflow:hidden}

/* Run bar */
.run-bar{display:flex;height:24px;width:100%}
.run-bar-seg{display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:10px;font-weight:700;color:#fff;min-width:20%}

/* Card header */
.card-header{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:12px;gap:8px}
.team-block{text-align:center}
.team-logo{width:48px;height:48px;margin:0 auto 4px;overflow:hidden}
.team-abbr{font-family:var(--font-display);font-size:24px;letter-spacing:2px;text-transform:uppercase}
.card-center{text-align:center}
.card-center .proj-label{font-family:var(--font-display);font-size:10px;color:var(--color-meta);letter-spacing:2px;text-transform:uppercase}
.card-center .spread{font-family:var(--font-mono);font-size:18px;font-weight:700}
.card-center .ou-line{font-family:var(--font-mono);font-size:10px;color:var(--color-meta);margin-top:2px}
.sim-pick{display:inline-block;background:var(--color-accent);color:var(--color-black);border:var(--border);font-family:var(--font-mono);font-size:10px;font-weight:700;padding:3px 10px;margin-top:4px;text-transform:uppercase}

/* SP matchup */
.sp-block{display:grid;grid-template-columns:1fr auto 1fr;padding:12px;gap:8px;border-top:1px dashed #ddd}
.sp-side{text-align:center}
.sp-name{font-family:var(--font-body);font-size:13px;font-weight:700}
.arch-badge{display:inline-block;background:var(--color-black);color:var(--color-accent);font-family:var(--font-mono);font-size:9px;font-weight:700;padding:2px 8px;border-radius:12px;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px}
.sp-stats{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);margin-top:3px}
.sp-vs{font-family:var(--font-display);font-size:14px;color:#ccc;display:flex;align-items:center;justify-content:center}

/* Game type badges */
.game-type-badge{font-family:var(--font-mono);font-size:9px;font-weight:700;text-transform:uppercase;text-align:center;padding:3px 0;letter-spacing:1px}
.st-badge{background:#1a5e2a;color:#90EE90}

/* Game metadata */
.game-meta{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);text-align:center;padding:4px 12px;letter-spacing:0.5px}

/* Edge bar */
.edge-bar{background:var(--color-accent);color:var(--color-black);font-family:var(--font-mono);font-size:10px;font-weight:700;text-align:center;padding:4px;letter-spacing:0.5px;text-transform:uppercase}

/* Lineup toggle */
.lineup-toggle{display:flex;align-items:center;justify-content:center;gap:6px;padding:10px;border-top:1px dashed #ddd;font-family:var(--font-mono);font-size:10px;font-weight:700;text-transform:uppercase;color:var(--color-meta);cursor:pointer;letter-spacing:1px;user-select:none}
.lineup-toggle .arrow{transition:transform 0.2s}
.lineup-toggle.open .arrow{transform:rotate(180deg)}

/* Lineup grid */
.lineup-grid{display:none;grid-template-columns:1fr 1fr;background:#f5f5f5;border-top:1px solid #e0e0e0}
.lineup-grid.open{display:grid}
.lineup-col{padding:8px}
.lineup-col:first-child{border-right:1px solid #ddd}
.lineup-col-hdr{font-family:var(--font-mono);font-size:9px;font-weight:700;text-transform:uppercase;color:var(--color-meta);padding:4px 0 6px;letter-spacing:1px;border-bottom:1px solid #ddd;margin-bottom:4px}

/* Batter row */
.batter-row{min-height:44px;padding:4px 0;border-bottom:1px solid #eee}
.batter-top{display:flex;align-items:baseline;justify-content:space-between}
.batter-order{font-family:var(--font-mono);font-size:10px;color:var(--color-meta);margin-right:4px;min-width:14px}
.batter-name{font-family:var(--font-body);font-size:12px;font-weight:700;flex:1}
.batter-ms{font-family:var(--font-display);font-size:18px;letter-spacing:1px}
.batter-bottom{display:flex;align-items:baseline;justify-content:space-between;margin-top:1px}
.batter-stats{font-family:var(--font-mono);font-size:9px;color:var(--color-meta)}
.batter-range{font-family:var(--font-mono);font-size:9px;color:#bbb}

/* MS colors */
.ms-elite{color:var(--color-elite);text-shadow:0 0 8px rgba(0,163,52,0.3)}
.ms-favorable{color:var(--color-favorable)}
.ms-neutral{color:var(--color-neutral)}
.ms-tough{color:var(--color-tough)}

/* Coverage */
.coverage{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);text-align:center;padding:6px;border-top:1px solid #eee;letter-spacing:0.5px}
.cov-good{color:var(--color-elite)}
.cov-warn{color:var(--color-tough)}

/* No lineups */
.tbd-block{text-align:center;padding:16px;font-family:var(--font-mono);font-size:10px;color:var(--color-meta);text-transform:uppercase;letter-spacing:1px}

/* ═══ BULLPEN SECTION ═══ */
.bp-toggle{display:flex;align-items:center;justify-content:center;gap:6px;padding:10px;border-top:1px dashed #ddd;font-family:var(--font-mono);font-size:10px;font-weight:700;text-transform:uppercase;color:var(--color-meta);cursor:pointer;letter-spacing:1px;user-select:none}
.bp-toggle .arrow{transition:transform 0.2s}
.bp-toggle.open .arrow{transform:rotate(180deg)}
.bp-container{display:none;background:#f5f5f5;border-top:1px solid #e0e0e0}
.bp-container.open{display:block}

.bp-section{padding:8px;border-bottom:1px solid #ddd}
.bp-section:last-child{border-bottom:none}
.bp-header{display:flex;align-items:center;gap:8px;padding:4px 0 6px;flex-wrap:wrap}
.bp-team{font-family:var(--font-display);font-size:14px;letter-spacing:1px}
.bp-meta-hdr{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);flex:1}
.bp-confidence{font-family:var(--font-mono);font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;padding:2px 8px;border-radius:10px}
.bp-conf-high{background:var(--color-elite);color:#fff}
.bp-conf-medium{background:var(--color-neutral);color:var(--color-black)}
.bp-conf-low{background:#999;color:#fff}

.bp-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #eee}
.bp-row:last-child{border-bottom:none}
.bp-role{font-family:var(--font-mono);font-size:9px;font-weight:700;text-transform:uppercase;min-width:28px;text-align:center;background:var(--color-black);color:var(--color-accent);padding:2px 4px;border-radius:6px}
.bp-info{flex:1;min-width:0}
.bp-name{font-family:var(--font-body);font-size:12px;font-weight:700}
.bp-meta{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bp-avail{width:48px;flex-shrink:0}
.bp-avail-bar{height:4px;background:#ddd;border-radius:2px;overflow:hidden}
.bp-avail-fill{height:100%;background:var(--color-elite);border-radius:2px}

/* ═══ PICKS SECTION ═══ */
.picks-container{background:var(--color-card);border:var(--border);box-shadow:var(--shadow);overflow:hidden;margin-bottom:16px}
.pick-row{display:flex;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid #eee}
.pick-row:last-child{border-bottom:none}
.pick-rank{font-family:var(--font-mono);font-size:12px;color:var(--color-meta);min-width:24px;text-align:center}
.pick-info{flex:1;min-width:0}
.pick-label{display:inline-block;font-family:var(--font-mono);font-size:10px;font-weight:700;padding:3px 10px;text-transform:uppercase;letter-spacing:0.5px}
.gp-pick-strong{background:var(--color-accent);color:var(--color-black);border:2px solid var(--color-black)}
.gp-pick-lean{background:var(--color-black);color:var(--color-accent);border:2px solid var(--color-accent)}
.pick-matchup{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);margin-top:2px}
.pick-edge{font-family:var(--font-mono);font-size:14px;font-weight:700;color:var(--color-elite);min-width:48px;text-align:right}

/* ═══ TRENDS ═══ */
.trend-row{display:flex;align-items:center;gap:10px;padding:10px 12px;min-height:56px;border-bottom:1px solid #eee}
.trend-row:last-child{border-bottom:none}
.trend-rank{font-family:var(--font-mono);font-size:12px;color:var(--color-meta);min-width:24px;text-align:center}
.trend-info{flex:1;min-width:0}
.trend-name{font-family:var(--font-body);font-size:13px;font-weight:700}
.trend-meta{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.trend-right{text-align:right;flex-shrink:0}
.trend-ms{font-family:var(--font-display);font-size:28px;letter-spacing:1px}

/* ═══ INFO TAB ═══ */
.info-card{background:var(--color-card);border:var(--border);box-shadow:var(--shadow);padding:16px;margin-bottom:16px}
.info-card h2{font-family:var(--font-display);font-size:20px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}
.info-card p{font-family:var(--font-body);font-size:13px;line-height:1.7;color:#333;margin-bottom:8px}
.tier-table{width:100%;border-collapse:collapse;margin:10px 0}
.tier-table td{padding:6px 10px;font-family:var(--font-mono);font-size:11px;border:1px solid #ddd}
.tier-table .tier-label{font-weight:700}
.formula-block{background:var(--color-black);color:var(--color-accent);font-family:var(--font-mono);font-size:10px;padding:12px;line-height:1.8;overflow-x:auto;white-space:pre;margin:10px 0}

/* ═══ SECTION HEADERS ═══ */
.section-title{font-family:var(--font-display);font-size:24px;letter-spacing:2px;text-transform:uppercase;margin-bottom:2px}
.section-sub{font-family:var(--font-mono);font-size:10px;color:var(--color-meta);margin-bottom:12px}

/* ═══ EMPTY STATE ═══ */
.empty-state{text-align:center;padding:40px 20px;font-family:var(--font-mono);font-size:11px;color:var(--color-meta);text-transform:uppercase;letter-spacing:1px}

/* ═══ FLOATING NAV ═══ */
.nav{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);z-index:200;background:var(--color-black);border:2px solid var(--color-accent);border-radius:28px;display:flex;padding:4px;gap:2px;max-width:400px;width:90%}
.nav-tab{flex:1;text-align:center;font-family:var(--font-mono);font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;padding:10px 4px;color:#fff;border-radius:24px;cursor:pointer;transition:all 0.15s;user-select:none;-webkit-tap-highlight-color:transparent}
.nav-tab.active{background:var(--color-accent);color:var(--color-black)}
.nav-tab:active{opacity:0.8}

/* ═══ GENERATED BADGE ═══ */
.gen-badge{font-family:var(--font-mono);font-size:9px;color:var(--color-meta);text-align:center;padding:8px;letter-spacing:0.5px;opacity:0.6}
'''


# ═══════════════════════════════════════════════════════════
# JAVASCRIPT
# ═══════════════════════════════════════════════════════════

def generate_js():
    """Return the complete JavaScript for interactivity."""
    return '''
// ═══ TAB SWITCHING ═══
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.getAttribute('data-tab');
    const el = document.getElementById('tab-' + target);
    if (el) el.classList.add('active');
    window.scrollTo(0, 0);
  });
});

// ═══ FILTER CHIPS ═══
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    const filter = chip.getAttribute('data-filter');
    document.querySelectorAll('.game-card').forEach(card => {
      if (filter === 'all') {
        card.style.display = '';
      } else if (filter === 'value') {
        card.style.display = card.querySelector('.edge-bar') ? '' : 'none';
      } else if (filter === 'st') {
        card.style.display = card.querySelector('.st-badge') ? '' : 'none';
      } else if (filter === 'regular') {
        card.style.display = card.querySelector('.st-badge') ? 'none' : '';
      }
    });
  });
});

// ═══ LINEUP TOGGLE ═══
function toggleLineup(idx) {
  const grid = document.getElementById('lineup-' + idx);
  const arrow = document.getElementById('arrow-' + idx);
  const toggle = arrow?.closest('.lineup-toggle');
  if (grid) {
    grid.classList.toggle('open');
    if (toggle) toggle.classList.toggle('open');
  }
}

// ═══ BULLPEN TOGGLE ═══
function toggleBullpen(idx) {
  const container = document.getElementById('bullpen-' + idx);
  const arrow = document.getElementById('bp-arrow-' + idx);
  const toggle = arrow?.closest('.bp-toggle');
  if (container) {
    container.classList.toggle('open');
    if (toggle) toggle.classList.toggle('open');
  }
}
'''


# ═══════════════════════════════════════════════════════════
# MAIN HTML GENERATION
# ═══════════════════════════════════════════════════════════

def generate_html(daily_data):
    """Generate the complete self-contained HTML page."""
    games = daily_data.get("games", [])
    slate_date = daily_data.get("slate_date", "")
    is_spring = any(g.get("is_spring_training") for g in games)

    # Format date for display
    if slate_date:
        try:
            dt = datetime.strptime(slate_date, "%Y-%m-%d")
            months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            display_date = f"{months[dt.month-1]} {dt.day}"
        except ValueError:
            display_date = slate_date
    else:
        display_date = "TODAY"

    # Count stats
    num_games = len(games)
    num_st = sum(1 for g in games if g.get("is_spring_training"))
    num_reg = num_games - num_st

    # Build ticker text
    ticker_parts = [f"{display_date} SLATE"]
    ticker_parts.append(f"{num_games} GAMES")
    if num_st > 0:
        ticker_parts.append(f"{num_st} SPRING TRAINING")
    if num_reg > 0:
        ticker_parts.append(f"{num_reg} REGULAR SEASON")

    # Count picks
    total_picks = sum(len(g.get("picks", [])) for g in games)
    if total_picks > 0:
        ticker_parts.append(f"{total_picks} EDGES FOUND")

    ticker_text = " ★ ".join(ticker_parts)

    # Render game cards
    game_cards_html = ""
    for i, game in enumerate(games):
        game_cards_html += render_game_card(game, i)

    # Render picks
    picks_html = render_top_picks(games)

    # Render trends
    trends_html = render_trends(games)

    # Determine filter chips
    filter_chips = '<div class="chip active" data-filter="all">All Games</div>'
    if num_st > 0 and num_reg > 0:
        filter_chips += '<div class="chip" data-filter="st">Spring Training</div>'
        filter_chips += '<div class="chip" data-filter="regular">Regular Season</div>'
    filter_chips += '<div class="chip" data-filter="value">Best Value</div>'

    # Spring training disclaimer
    st_disclaimer = ""
    if is_spring:
        st_disclaimer = '''<div style="background:#1a5e2a;color:#90EE90;font-family:var(--font-mono);font-size:9px;text-align:center;padding:6px;margin-bottom:10px;letter-spacing:0.5px;border-radius:4px">
🌴 SPRING TRAINING — Archetype matchups use regular-season data. Bullpen predictions are experimental. NRI players shown as UNCHARTED.
</div>'''

    # Generation timestamp
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M ET")

    css = generate_css()
    js = generate_js()

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>MLB SIM — {display_date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
{css}
</style>
<link rel="stylesheet" href="https://morellosims.com/morello-auth.css">
</head>
<body>

<!-- Header -->
<header class="header">
    <div class="brand-row">
        <div style="display:flex;align-items:baseline;gap:10px">
            <div class="logo">MLB SIM</div>
            <span class="byline">by Jack Morello</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
            <a href="cosmos.html" class="atlas-btn">ATLAS</a>
            <div class="status-dot"></div>
        </div>
    </div>
</header>

<!-- Main -->
<div class="container">

    <!-- ═══ LINES TAB ═══ -->
    <div class="tab-content active" id="tab-lines">
        <div class="chips">
            {filter_chips}
        </div>
        <div class="slate-info">
            <span>{display_date} SLATE</span>
            <span>{num_games} GAMES</span>
        </div>
        {st_disclaimer}
        {game_cards_html}
        <div class="gen-badge">Generated {gen_time} · Powered by ATLAS Pitcher DNA</div>
    </div>

    <!-- ═══ DAILY PROJECTIONS TAB ═══ -->
    <div class="tab-content" id="tab-daily">
        <div style="padding-top:12px">
            <div class="section-title">DAILY PROJECTIONS</div>
            <div class="section-sub">Today's per-batter matchup scores + team run projections</div>
        </div>
        <div class="slate-info">
            <span>{display_date}</span>
            <span>{num_games} GAMES</span>
        </div>
        {picks_html}
        <div class="picks-container">
            {trends_html}
        </div>
    </div>

    <!-- ═══ TRENDS TAB ═══ -->
    <div class="tab-content" id="tab-trends">
        <div style="padding-top:12px">
            <div class="section-title">TOP MATCHUP SCORES</div>
            <div class="section-sub">Today's top batter-vs-archetype matchups ranked by MS</div>
        </div>
        <div class="picks-container">
            {trends_html}
        </div>
    </div>

    <!-- ═══ FANTASY TAB ═══ -->
    <div class="tab-content" id="tab-fantasy">
        <div style="padding-top:12px">
            <div class="section-title">WEEKLY FANTASY</div>
            <div class="section-sub">7-day projected matchup scores — built for weekly fantasy lineups</div>
        </div>
        <div class="empty-state">WEEKLY PROJECTIONS COMING SOON — AVAILABLE WHEN REGULAR SEASON STARTS</div>
    </div>

    <!-- ═══ INFO TAB ═══ -->
    <div class="tab-content" id="tab-info">
        <div style="padding-top:12px">
            <div class="info-card">
                <h2>HOW MLB SIM WORKS</h2>
                <p>MLB SIM uses the <strong>Pitcher DNA</strong> system (Gaussian Mixture Model clustering) to classify every pitcher into one of 26 archetypes (15 RHP + 11 LHP) based on their pitch mix, velocity, movement, and approach. Every batter has historical performance data against each archetype — because baseball is fundamentally a 1v1 sport, the specific pitcher a batter faces defines their expected performance.</p>
                <p>When lineups are released, MLB SIM projects every batter's performance based on how they've historically hit against the opposing pitcher's archetype. This produces per-game <strong>Matchup Scores</strong> that change daily depending on the starter.</p>
            </div>
            <div class="info-card">
                <h2>MATCHUP SCORE (MS) — 40 TO 99</h2>
                <p>MS is a context-dependent metric that changes each game based on the specific pitcher archetype a batter faces. A batter can be a 92 against a fastball-heavy pitcher but a 48 against a curveball specialist.</p>
                <table class="tier-table">
                    <tr><td class="tier-label" style="color:var(--color-elite)">85-99</td><td>Elite Matchup — historically dominant vs this archetype</td></tr>
                    <tr><td class="tier-label" style="color:var(--color-favorable)">70-84</td><td>Favorable — above-average performance expected</td></tr>
                    <tr><td class="tier-label" style="color:var(--color-neutral)">55-69</td><td>Neutral — roughly league-average</td></tr>
                    <tr><td class="tier-label" style="color:var(--color-tough)">40-54</td><td>Tough Matchup — historically struggles vs this archetype</td></tr>
                </table>
                <div class="formula-block">MS FORMULA (wOBA-based):
wOBA >= .400  →  MS 90-99
wOBA .370-.399 →  MS 80-89
wOBA .340-.369 →  MS 70-79
wOBA .310-.339 →  MS 60-69
wOBA .270-.309 →  MS 50-59
wOBA < .270    →  MS 40-49

H2H BONUS: +0-5 pts when PA >= 10
vs specific pitcher (not just archetype)</div>
            </div>
            <div class="info-card">
                <h2>PROJECTION METHODOLOGY</h2>
                <p>Team runs are projected using the <strong>BaseRuns</strong> formula, a context-neutral run estimator:</p>
                <div class="formula-block">A = H + BB - HR
B = 1.02 × (1.4×TB - 0.6×H + 0.1×BB)
C = PA - H - BB
D = HR
Runs = A×B / (B+C) + D

Spread = Home Runs − Away Runs
O/U Total = Home Runs + Away Runs</div>
                <p>Batter projections use archetype-specific wOBA with head-to-head adjustments when sufficient plate appearance history exists.</p>
            </div>
            <div class="info-card">
                <h2>BULLPEN PREDICTIONS</h2>
                <p>MLB SIM predicts bullpen deployment using three layers:</p>
                <p><strong>Layer 1 — Availability:</strong> Tracks reliever workload history (days rest, recent pitch count, consecutive-day usage) to determine who CAN pitch.</p>
                <p><strong>Layer 2 — Usage Order:</strong> Estimates starter expected innings, then ranks available relievers by role hierarchy (closer → setup → middle → long) and archetype matchup advantage.</p>
                <p><strong>Layer 3 — Matchup Integration:</strong> For each predicted reliever, computes MS against the lineup slots they'll likely face. Higher MS = tougher matchup for the opposing lineup.</p>
            </div>
        </div>
    </div>

</div>

<!-- Floating nav -->
<nav class="nav">
    <div class="nav-tab active" data-tab="lines">Lines</div>
    <div class="nav-tab" data-tab="daily">Daily</div>
    <div class="nav-tab" data-tab="trends">Trends</div>
    <div class="nav-tab" data-tab="fantasy">Fantasy</div>
    <div class="nav-tab" data-tab="info">Info</div>
</nav>

<!-- Auth -->
<script src="https://morellosims.com/morello-auth.js" data-ma-theme="mlb"></script>

<script>
{js}
</script>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  MLB SIM — STATIC HTML GENERATOR")
    print(f"{'='*60}\n")

    # Load daily games data
    if not os.path.exists(DAILY_GAMES_FILE):
        print(f"[ERROR] {DAILY_GAMES_FILE} not found — run compute_matchups.py first")
        sys.exit(1)

    with open(DAILY_GAMES_FILE) as f:
        daily_data = json.load(f)

    games = daily_data.get("games", [])
    slate_date = daily_data.get("slate_date", "unknown")
    print(f"[Generate] Processing {len(games)} games for {slate_date}")

    # Generate HTML
    html = generate_html(daily_data)

    # Write output
    output_path = os.path.join(FRONTEND_PUBLIC, "mlbsim.html")
    with open(output_path, "w") as f:
        f.write(html)
    print(f"[Output] Wrote {output_path} ({len(html):,} bytes)")

    # Summary
    st_count = sum(1 for g in games if g.get("is_spring_training"))
    reg_count = len(games) - st_count
    pick_count = sum(len(g.get("picks", [])) for g in games)
    print(f"[Summary] {len(games)} games ({st_count} ST, {reg_count} REG), {pick_count} edges")


if __name__ == "__main__":
    main()
