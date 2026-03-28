#!/usr/bin/env python3
"""
Update MORELLOSIMS MLB blog section with graded picks + pending picks.

Reads:
  - data/daily/mlb_pick_history.json  (all graded days)
  - data/daily/daily_games.json       (today's pending picks)

Patches the MLB blog section (class="post-mlb-picks") in MORELLOSIMS index.html:
  - Header stats (record, ROI, bankroll, picks count)
  - Bankroll tracker box
  - Daily slate entries with graded + pending picks

Does NOT touch the NBA section.

Usage:
    python scripts/update_mlb_blog.py <morellosims_index_path>
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.daily_config import DAILY_GAMES_FILE, DAILY_DATA_DIR, STARTING_BANKROLL

HISTORY_FILE = os.path.join(DAILY_DATA_DIR, "mlb_pick_history.json")

# ═══════════════════════════════════════════════════════════
# DATE FORMATTING
# ═══════════════════════════════════════════════════════════

MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}
DAY_ABBR = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}


def format_slate_date(date_str):
    """Format YYYY-MM-DD into 'MAR 26 &middot; THU'."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    month = MONTH_ABBR[dt.month]
    day_name = DAY_ABBR[dt.weekday()]
    return f"{month} {dt.day} &middot; {day_name}"


def format_odds(odds):
    """Format odds for display: +157 or -186."""
    if odds > 0:
        return f"+{odds}"
    return str(odds)


def format_pnl(pnl):
    """Format P&L: +16 or -30."""
    if pnl > 0:
        return f"+{pnl:.0f}"
    elif pnl < 0:
        return f"{pnl:.0f}"
    return "0"


# ═══════════════════════════════════════════════════════════
# PICK CARD HTML GENERATION
# ═══════════════════════════════════════════════════════════

def build_settled_pick_card(pick):
    """Build HTML for a settled (W/L) pick card."""
    matchup = pick["matchup"]
    side = pick["side"]
    odds = pick["odds"]
    result = pick["result"]
    pnl = pick["pnl"]
    unit = pick.get("unit", 30)
    confidence = pick.get("confidence", 5)
    model_wp = pick.get("model_wp", 0)
    market_wp = pick.get("market_wp", 0)
    ev = pick.get("ev", 0)
    away_score = pick.get("away_score", 0)
    home_score = pick.get("home_score", 0)

    parts = matchup.split(" @ ")
    away, home = parts[0], parts[1]

    # Colors based on result
    if result == "W":
        side_color = "#2a9d5f"
        result_color = "#00FF55"
        bg_color = "rgba(42,157,95,0.12)"
        border_color = "#2a9d5f"
        badge_bg = "#2a9d5f"
    else:
        side_color = "#ff4444"
        result_color = "#ff4444"
        bg_color = "rgba(255,68,68,0.08)"
        border_color = "#ff4444"
        badge_bg = "#ff4444"

    odds_str = format_odds(odds)
    pnl_str = format_pnl(pnl)

    # Final score display: winning team first
    if side == away:
        score_display = f"{away} {away_score} &mdash; {home} {home_score}"
    else:
        score_display = f"{home} {home_score} &mdash; {away} {away_score}"

    return f"""                            <div class="pick-card" data-status="settled" data-matchup="{matchup}" style="margin:8px 0 6px; padding:10px 12px; background:{bg_color}; border-left:3px solid {border_color}; border-radius:0 4px 4px 0;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                    <p class="mono pick-side-text" style="font-size:10px; color:{side_color}; font-weight:700; letter-spacing:1px; margin:0;">{matchup} &mdash; {side} ML ({odds_str})</p>
                                    <div style="display:flex; gap:6px; align-items:center;">
                                        <span class="mono" style="font-size:9px; background:#f4a261; color:#000; padding:2px 6px; border-radius:2px; font-weight:700;">{unit} $PP</span>
                                        <span class="mono" style="font-size:9px; background:{badge_bg}; color:#000; padding:2px 6px; border-radius:2px; font-weight:700;">ML C:{confidence}</span>
                                    </div>
                                </div>
                                <p class="mono" style="font-size:8px; color:{result_color}; margin:0 0 4px; font-weight:700; letter-spacing:0.5px;">FINAL: {score_display} | {side} ML {result} ({pnl_str} $PP)</p>
                                <p class="blog-body-text pick-rationale" style="font-size:10px; color:#999; margin:0;">Model WP: {model_wp}% vs Market: {market_wp}%. EV: {ev:+.1f}%</p>
                            </div>"""


def build_pending_pick_card(pick):
    """Build HTML for a pending pick card (today's picks, not yet graded)."""
    matchup = pick["matchup"]
    side = pick["side"]
    odds = pick["odds"]
    unit = pick.get("unit", 30)
    confidence = pick.get("confidence", 5)
    model_wp = pick.get("model_wp", 0)
    market_wp = pick.get("market_wp", 0)
    ev = pick.get("ev", 0)

    odds_str = format_odds(odds)

    return f"""                            <div class="pick-card" data-status="pending" data-matchup="{matchup}" style="margin:8px 0 6px; padding:10px 12px; background:rgba(255,234,0,0.06); border-left:3px solid #FFEA00; border-radius:0 4px 4px 0;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                    <p class="mono pick-side-text" style="font-size:10px; color:#FFEA00; font-weight:700; letter-spacing:1px; margin:0;">{matchup} &mdash; {side} ML ({odds_str})</p>
                                    <div style="display:flex; gap:6px; align-items:center;">
                                        <span class="mono" style="font-size:9px; background:#f4a261; color:#000; padding:2px 6px; border-radius:2px; font-weight:700;">{unit} $PP</span>
                                        <span class="mono" style="font-size:9px; background:#FFEA00; color:#000; padding:2px 6px; border-radius:2px; font-weight:700;">ML C:{confidence}</span>
                                    </div>
                                </div>
                                <p class="mono" style="font-size:8px; color:#FFEA00; margin:0 0 4px; font-weight:700; letter-spacing:0.5px;">PENDING</p>
                                <p class="blog-body-text pick-rationale" style="font-size:10px; color:#999; margin:0;">Model WP: {model_wp}% vs Market: {market_wp}%. EV: {ev:+.1f}%</p>
                            </div>"""


# ═══════════════════════════════════════════════════════════
# SLATE DAY HTML GENERATION
# ═══════════════════════════════════════════════════════════

def build_settled_slate(day_data):
    """Build a complete settled slate-day HTML block."""
    date_str = day_data.get("date", "")
    picks = day_data.get("picks", [])
    record = day_data.get("record", {})
    daily_pnl = day_data.get("daily_pnl", 0)

    wins = record.get("wins", 0)
    losses = record.get("losses", 0)
    total_risked = sum(p.get("unit", 30) for p in picks)

    date_label = format_slate_date(date_str)
    meta_text = f"{len(picks)} ML PICK{'S' if len(picks) != 1 else ''} &middot; {total_risked} $PP"

    # Record color
    if daily_pnl > 0:
        record_color = "#00FF55"
        pnl_str = f"+{daily_pnl:.0f}"
    elif daily_pnl < 0:
        record_color = "#ff4444"
        pnl_str = f"{daily_pnl:.0f}"
    else:
        record_color = "#FFEA00"
        pnl_str = "0"

    record_str = f"{wins}-{losses} &middot; {pnl_str} $PP"

    # Build pick cards
    cards_html = "\n".join(build_settled_pick_card(p) for p in picks)

    return f"""
                    <details class="slate-day" open>
                        <summary>
                            <div>
                                <span class="slate-day-label">{date_label}</span>
                                <span class="slate-day-meta">{meta_text}</span>
                            </div>
                            <span class="slate-day-record" style="color:{record_color};">{record_str}</span>
                        </summary>
                        <div class="slate-day-body">
                            <p class="mono" style="font-size:10px; color:#ccbb00; letter-spacing:2px; margin:12px 0 8px; font-weight:700;">&#9612; GAME LINES &mdash; {len(picks)} ML PICK{'S' if len(picks) != 1 else ''}</p>
{cards_html}
                        </div>
                    </details>"""


def build_pending_slate(slate_date, picks):
    """Build a pending slate-day HTML block for today's picks."""
    date_label = format_slate_date(slate_date)
    total_risked = sum(p.get("unit", 30) for p in picks)
    meta_text = f"{len(picks)} ML PICK{'S' if len(picks) != 1 else ''} &middot; {total_risked} $PP"

    cards_html = "\n".join(build_pending_pick_card(p) for p in picks)

    return f"""
                    <details class="slate-day" open>
                        <summary>
                            <div>
                                <span class="slate-day-label">{date_label}</span>
                                <span class="slate-day-meta">{meta_text}</span>
                            </div>
                            <span class="slate-day-record" style="color:#FFEA00;">PENDING</span>
                        </summary>
                        <div class="slate-day-body">
                            <p class="mono" style="font-size:10px; color:#FFEA00; letter-spacing:2px; margin:12px 0 8px; font-weight:700;">&#9612; GAME LINES &mdash; {len(picks)} ML PICK{'S' if len(picks) != 1 else ''}</p>
{cards_html}
                        </div>
                    </details>"""


# ═══════════════════════════════════════════════════════════
# CUMULATIVE STATS
# ═══════════════════════════════════════════════════════════

def compute_cumulative_from_history(history):
    """Compute overall stats from history."""
    total_wins = 0
    total_losses = 0
    total_pnl = 0
    total_risked = 0
    total_picks = 0

    for day in history:
        rec = day.get("record", {})
        total_wins += rec.get("wins", 0)
        total_losses += rec.get("losses", 0)
        total_pnl += day.get("daily_pnl", 0)
        for p in day.get("picks", []):
            total_risked += p.get("unit", 30)
            total_picks += 1

    bankroll = STARTING_BANKROLL + total_pnl
    roi = round((total_pnl / total_risked * 100), 1) if total_risked > 0 else 0

    return {
        "record_str": f"{total_wins}-{total_losses}",
        "wins": total_wins,
        "losses": total_losses,
        "roi": roi,
        "bankroll": round(bankroll, 2),
        "total_risked": total_risked,
        "total_pnl": round(total_pnl, 2),
        "total_picks": total_picks,
    }


# ═══════════════════════════════════════════════════════════
# EXTRACT TODAY'S PENDING PICKS
# ═══════════════════════════════════════════════════════════

def extract_pending_picks(daily_data):
    """Extract ML picks from today's daily_games.json."""
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
        unit = 50 if (is_strong and conf_val >= 8) else 30

        picks.append({
            "matchup": f"{away} @ {home}",
            "side": side,
            "odds": odds,
            "ev": round(ev, 1),
            "model_wp": round(model_wp, 1),
            "market_wp": round(market_wp, 1),
            "confidence": conf_val,
            "unit": unit,
        })

    return picks


# ═══════════════════════════════════════════════════════════
# BLOG PATCHING
# ═══════════════════════════════════════════════════════════

def patch_mlb_blog(blog_html, history, pending_picks, pending_date, stats):
    """Patch the MLB blog section in the full HTML.

    Replaces content inside <details class="blog-card post-mlb-picks">
    while leaving NBA and other sections untouched.
    """

    # ── 1. Find the MLB picks section boundaries ──
    mlb_start_pattern = re.compile(
        r'(<details\s+class="blog-card post-mlb-picks"[^>]*>)',
        re.IGNORECASE
    )
    mlb_match = mlb_start_pattern.search(blog_html)
    if not mlb_match:
        print("[UpdateBlog] Could not find MLB picks section (post-mlb-picks)")
        return blog_html, 0

    mlb_start = mlb_match.start()

    # Find the closing </details> for this section
    # Need to handle nested <details> tags
    depth = 0
    pos = mlb_match.end()
    mlb_end = None
    while pos < len(blog_html):
        # Look for next <details or </details>
        next_open = blog_html.find("<details", pos)
        next_close = blog_html.find("</details>", pos)

        if next_close == -1:
            break

        if next_open != -1 and next_open < next_close:
            depth += 1
            pos = next_open + 8
        else:
            if depth == 0:
                mlb_end = next_close + len("</details>")
                break
            depth -= 1
            pos = next_close + len("</details>")

    if mlb_end is None:
        print("[UpdateBlog] Could not find closing </details> for MLB section")
        return blog_html, 0

    # ── 2. Build header stats ──
    record_str = stats["record_str"]
    roi = stats["roi"]
    bankroll = stats["bankroll"]
    total_picks = stats["total_picks"]
    total_risked = stats["total_risked"]
    total_pnl = stats["total_pnl"]

    roi_str = f"+{roi}%" if roi > 0 else (f"{roi}%" if roi < 0 else "&mdash;")
    bankroll_display = f"{bankroll:,.0f}"

    # Determine date range for header
    if history:
        first_date = history[0].get("date", "")
        last_date = history[-1].get("date", "")
        try:
            fd = datetime.strptime(first_date, "%Y-%m-%d")
            ld = datetime.strptime(last_date, "%Y-%m-%d")
            date_range = f"{MONTH_ABBR[fd.month]} {fd.day} &mdash; {MONTH_ABBR[ld.month]} {ld.day}, {ld.year}"
        except Exception:
            date_range = "2026 MLB SEASON"
    else:
        date_range = "2026 MLB SEASON"

    # ── 3. Build title ──
    if total_picks > 0:
        title_text = f"MLB SIM: {record_str} RECORD ({roi_str} ROI)"
    else:
        title_text = "MLB SIM: OPENING DAY TRACKER"

    # ── 4. Build all slate HTML (newest first) ──
    slates_html = ""

    # Today's pending picks (shown first, at top)
    if pending_picks:
        slates_html += build_pending_slate(pending_date, pending_picks)

    # Graded days (newest first)
    for day in reversed(history):
        slates_html += build_settled_slate(day)

    # If no slates at all, show placeholder
    if not slates_html.strip():
        slates_html = """
                    <details class="slate-day" open>
                        <summary>
                            <div>
                                <span class="slate-day-label">SEASON START</span>
                                <span class="slate-day-meta">WAITING FOR PICKS</span>
                            </div>
                            <span class="slate-day-record" style="color:#FFEA00;">PENDING</span>
                        </summary>
                        <div class="slate-day-body">
                            <div style="text-align:center; padding:24px 16px; color:#555;">
                                <p class="mono" style="font-size:11px; letter-spacing:2px; margin:0;">LINEUPS CONFIRMED &rarr; SIMS RUNNING &rarr; ML PICKS POST</p>
                            </div>
                        </div>
                    </details>"""

    # ── 5. Build the bankroll box ──
    if total_pnl > 0:
        bankroll_color = "#00FF55"
    elif total_pnl < 0:
        bankroll_color = "#ff4444"
    else:
        bankroll_color = "#FFEA00"

    # ── 6. Assemble the full MLB section ──
    new_mlb_section = f"""<details class="blog-card post-mlb-picks" open>
                <summary>
                    <div class="blog-meta mono">
                        <span class="blog-card-type type-picks">PICKS LOG</span>
                        <div class="blog-system-dots">
                            <span class="blog-date">{date_range}</span>
                        </div>
                    </div>
                    <h3 class="blog-title bebas">{title_text}</h3>
                    <div class="blog-hero-stats">
                        <div class="blog-hero-stat stat-record">
                            <span class="stat-value">{record_str}</span>
                            <span class="stat-label">RECORD</span>
                        </div>
                        <div class="blog-hero-stat stat-roi">
                            <span class="stat-value">{roi_str}</span>
                            <span class="stat-label">ROI</span>
                        </div>
                        <div class="blog-hero-stat stat-bankroll">
                            <span class="stat-value">{bankroll_display}</span>
                            <span class="stat-label">BANKROLL</span>
                        </div>
                        <div class="blog-hero-stat stat-picks">
                            <span class="stat-value">{total_picks}</span>
                            <span class="stat-label">PICKS</span>
                        </div>
                    </div>
                    <p class="blog-preview">
                        GMM pitcher archetypes &rarr; batter-vs-cluster matchups &rarr; BaseRuns simulation with park factors, bullpen, and defensive adjustments. 57% winner accuracy on 2,437 backtested games.
                    </p>
                </summary>
                <div class="blog-card-body">

                    <!-- ── BANKROLL ── -->
                    <div style="margin:8px 0 12px; padding:12px 14px; background:rgba(255,234,0,0.06); border:1px solid #FFEA00; border-radius:4px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                            <div>
                                <div class="mono" style="font-size:9px; color:#FFEA00; letter-spacing:1.5px; font-weight:700;">BANKROLL</div>
                                <div class="mono" style="font-size:22px; color:{bankroll_color}; margin-top:2px;">{bankroll_display} $PP</div>
                            </div>
                            <div style="text-align:center;">
                                <div class="mono" style="font-size:9px; color:#555; letter-spacing:1.5px;">TARGET</div>
                                <div class="mono" style="font-size:22px; color:#ccbb00; margin-top:2px;">25,000 $PP</div>
                            </div>
                            <div style="text-align:right;">
                                <div class="mono" style="font-size:9px; color:#555; letter-spacing:1.5px;">TOTAL RISKED</div>
                                <div class="mono" style="font-size:22px; color:#e0e0e0; margin-top:2px;">{total_risked} $PP</div>
                            </div>
                        </div>
                    </div>

                    <!-- ── UNIT SIZING KEY ── -->
                    <div style="margin:0 0 12px; padding:8px 12px; background:rgba(255,255,255,0.02); border:1px solid #1a1a1a; border-radius:4px;">
                        <p class="mono" style="font-size:9px; color:#888; letter-spacing:1.5px; margin:0;">UNIT KEY: <span style="color:#ccbb00;">8-10 CONF = 50 $PP</span> &nbsp;&middot;&nbsp; <span style="color:#ccbb00;">5-7 CONF = 30 $PP</span> &nbsp;&middot;&nbsp; <span style="color:#555;">1-10 SCALE &mdash; 5+ MINIMUM</span></p>
                    </div>

                    <!-- ── SEASON HEADER ── -->
                    <p class="mono" style="font-size:10px; color:#FFEA00; letter-spacing:2px; margin:12px 0 8px; font-weight:700;">&#9612; 2026 MLB SEASON</p>
{slates_html}

                </div>
            </details>"""

    # ── 7. Replace the old MLB section ──
    patched = blog_html[:mlb_start] + new_mlb_section + blog_html[mlb_end:]

    return patched, 1


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_mlb_blog.py <morellosims_index_path>")
        sys.exit(1)

    blog_path = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"  MLB BLOG UPDATER")
    print(f"{'='*60}\n")

    if not os.path.exists(blog_path):
        print(f"[ERROR] Blog file not found: {blog_path}")
        sys.exit(1)

    # ── Load pick history ──
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)
        print(f"[UpdateBlog] Loaded {len(history)} graded days from history")
    else:
        print(f"[UpdateBlog] No history file found — starting fresh")

    # ── Load today's pending picks ──
    pending_picks = []
    pending_date = ""
    if os.path.exists(DAILY_GAMES_FILE):
        with open(DAILY_GAMES_FILE) as f:
            daily_data = json.load(f)
        pending_date = daily_data.get("slate_date", "")
        # Only show as pending if not already graded
        graded_dates = {d.get("date") for d in history}
        if pending_date not in graded_dates:
            pending_picks = extract_pending_picks(daily_data)
            print(f"[UpdateBlog] {len(pending_picks)} pending picks for {pending_date}")
        else:
            print(f"[UpdateBlog] {pending_date} already graded — skipping pending")
    else:
        print(f"[UpdateBlog] No daily_games.json — no pending picks")

    # ── Compute cumulative stats ──
    stats = compute_cumulative_from_history(history)
    # Include pending picks in total count for display
    stats["total_picks"] += len(pending_picks)
    # Include pending risk in total risked
    pending_risk = sum(p.get("unit", 30) for p in pending_picks)
    stats["total_risked"] += pending_risk

    print(f"[UpdateBlog] Cumulative: {stats['record_str']} | ROI: {stats['roi']}% | Bankroll: {stats['bankroll']:.0f}")

    # ── Read blog HTML ──
    with open(blog_path) as f:
        blog_html = f.read()

    # ── Patch ──
    patched, changes = patch_mlb_blog(blog_html, history, pending_picks, pending_date, stats)

    if changes > 0:
        with open(blog_path, "w") as f:
            f.write(patched)
        print(f"[UpdateBlog] Patched MLB section in {blog_path}")
    else:
        print(f"[UpdateBlog] No changes made")


if __name__ == "__main__":
    main()
