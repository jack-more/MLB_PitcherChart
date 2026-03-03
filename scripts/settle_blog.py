#!/usr/bin/env python3
"""
Settle MLB picks on MORELLOSIMS blog.

Reads settlement_results.json and patches the MORELLOSIMS blog with:
  - W/L/P result badges (green/red/yellow)
  - FINAL score lines on pick cards
  - Updated bankroll, record, and status

Usage:
    python scripts/settle_blog.py <settlement_json_path> <morellosims_index_path>

Example:
    python scripts/settle_blog.py data/daily/settlement_results.json /tmp/morellosims/index.html
"""

import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# RESULT STYLING
# ═══════════════════════════════════════════════════════════

RESULT_COLORS = {
    "W": "#00FF55",   # Green (matches NBA SIM)
    "L": "#FF4444",   # Red
    "P": "#FFD600",   # Yellow (push)
}


def format_profit(profit):
    """Format profit for display."""
    if profit > 0:
        return f"+{profit:.0f}"
    elif profit < 0:
        return f"{profit:.0f}"
    else:
        return "0"


# ═══════════════════════════════════════════════════════════
# BLOG PATCHING
# ═══════════════════════════════════════════════════════════

def patch_results_table(blog_html, graded_picks):
    """Patch the results table with W/L/P result badges.

    Looks for table cells with &mdash; (pending) and replaces with colored results.
    """
    changes = 0

    for pick in graded_picks:
        result = pick.get("result")
        if not result:
            continue

        matchup = pick.get("matchup", "")
        label = pick.get("label", "")
        color = RESULT_COLORS.get(result, "#999")

        # Find the matchup row in the results table and replace the pending dash
        # Pattern: matchup text ... <td ...>&mdash;</td> or <td ...>—</td>
        parts = matchup.split(" @ ")
        if len(parts) != 2:
            continue

        away, home = parts

        # Look for the row containing this matchup and replace the result cell
        # This is a targeted regex that finds the result cell near the matchup text
        row_pattern = re.compile(
            rf'({re.escape(away)}\s*@\s*{re.escape(home)}.*?<td[^>]*>)\s*[—&][^<]*(<\/td>)',
            re.DOTALL | re.IGNORECASE
        )

        replacement = rf'\1<span style="color:{color};font-weight:700">{result}</span>\2'
        new_html, n = row_pattern.subn(replacement, blog_html, count=1)
        if n > 0:
            blog_html = new_html
            changes += 1

    return blog_html, changes


def patch_final_scores(blog_html, graded_picks):
    """Add FINAL score lines to pick cards.

    For each graded pick, adds a line like:
    FINAL: NYY 5 — BOS 3 | NYY -1.5 W (+45 $PP)
    """
    changes = 0

    for pick in graded_picks:
        result = pick.get("result")
        if not result:
            continue

        matchup = pick.get("matchup", "")
        home_score = pick.get("home_score")
        away_score = pick.get("away_score")
        label = pick.get("label", "")
        profit = pick.get("profit", 0)

        if home_score is None or away_score is None:
            continue

        parts = matchup.split(" @ ")
        if len(parts) != 2:
            continue

        away, home = parts
        color = RESULT_COLORS.get(result, "#999")
        profit_str = format_profit(profit)

        final_line = (
            f'<div class="mono" style="font-size:11px;margin-top:4px;color:{color}">'
            f'FINAL: {away} {away_score} &mdash; {home} {home_score} | '
            f'{label} <strong>{result}</strong> ({profit_str} $PP)'
            f'</div>'
        )

        # Check if FINAL line already exists for this matchup
        if f"FINAL: {away}" in blog_html and f"{home} {home_score}" in blog_html:
            continue  # Already settled

        # Find the pick card for this matchup and append the FINAL line
        # Look for the matchup text and insert after the pick details
        card_pattern = re.compile(
            rf'({re.escape(away)}\s*@\s*{re.escape(home)}[^<]*<)',
            re.IGNORECASE
        )

        match = card_pattern.search(blog_html)
        if match:
            # Insert FINAL line after the next closing tag
            insert_pos = match.end()
            # Find the end of the current element
            depth = 0
            search_start = insert_pos
            for i in range(search_start, min(search_start + 500, len(blog_html))):
                if blog_html[i:i+4] == '</p>' or blog_html[i:i+6] == '</div>' or blog_html[i:i+5] == '</td>':
                    # Insert after this closing tag
                    end_tag = blog_html[i:].split('>')[0] + '>'
                    insert_at = i + len(end_tag)
                    blog_html = blog_html[:insert_at] + "\n" + final_line + blog_html[insert_at:]
                    changes += 1
                    break

    return blog_html, changes


def patch_bankroll_and_record(blog_html, settlement):
    """Update bankroll display, record, and status in the blog."""
    record = settlement.get("record", {})
    new_bankroll = settlement.get("new_bankroll", 1000)
    total_profit = settlement.get("total_profit", 0)
    status = settlement.get("status", "PENDING")

    record_str = f"{record.get('W', 0)}-{record.get('L', 0)}-{record.get('P', 0)}"

    # Update bankroll value
    bankroll_pattern = re.compile(r'(bankroll[^<]*?)\$?\d[\d,]*\.?\d*', re.IGNORECASE)
    blog_html = bankroll_pattern.sub(
        rf'\g<1>{new_bankroll:.0f}',
        blog_html,
        count=1
    )

    # Update record
    record_pattern = re.compile(r'(record[^<]*?)\d+-\d+-\d+', re.IGNORECASE)
    blog_html = record_pattern.sub(
        rf'\g<1>{record_str}',
        blog_html,
        count=1
    )

    return blog_html


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/settle_blog.py <settlement_json> <morellosims_index>")
        sys.exit(1)

    settlement_path = sys.argv[1]
    blog_path = sys.argv[2]

    print(f"\n{'='*60}")
    print(f"  MLB BLOG SETTLER")
    print(f"{'='*60}\n")

    # Load settlement results
    if not os.path.exists(settlement_path):
        print(f"[ERROR] Settlement file not found: {settlement_path}")
        sys.exit(1)

    with open(settlement_path) as f:
        settlement = json.load(f)

    graded_picks = [p for p in settlement.get("picks", []) if p.get("result")]
    pending = settlement.get("pending_count", 0)
    status = settlement.get("status", "PENDING")

    print(f"[Settle] {len(graded_picks)} graded picks, {pending} pending, status: {status}")

    if not graded_picks:
        print("[Settle] No graded picks to settle")
        return

    # Load blog
    if not os.path.exists(blog_path):
        print(f"[ERROR] Blog file not found: {blog_path}")
        sys.exit(1)

    with open(blog_path) as f:
        blog_html = f.read()

    original_len = len(blog_html)

    # Patch results table
    blog_html, result_changes = patch_results_table(blog_html, graded_picks)
    print(f"[Settle] Patched {result_changes} result cells")

    # Patch FINAL score lines
    blog_html, final_changes = patch_final_scores(blog_html, graded_picks)
    print(f"[Settle] Added {final_changes} FINAL score lines")

    # Patch bankroll and record
    blog_html = patch_bankroll_and_record(blog_html, settlement)

    total_changes = result_changes + final_changes
    if total_changes > 0 or len(blog_html) != original_len:
        with open(blog_path, "w") as f:
            f.write(blog_html)
        print(f"[Settle] Updated {blog_path} with {total_changes} changes")
    else:
        print("[Settle] No changes needed")

    # Summary
    record = settlement.get("record", {})
    print(f"\n  Record: {record.get('W', 0)}-{record.get('L', 0)}-{record.get('P', 0)}")
    print(f"  Profit: {settlement.get('total_profit', 0):+.2f} $PP")
    print(f"  Bankroll: {settlement.get('new_bankroll', 1000):.0f} $PP")


if __name__ == "__main__":
    main()
