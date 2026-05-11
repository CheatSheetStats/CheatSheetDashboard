import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client, Client
import os
import io
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Data source — Supabase storage (unchanged from your original)
# ──────────────────────────────────────────────────────────────────────────────
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)
bucket_name = os.environ.get("BUCKET_NAME")
file_path   = os.environ.get("FILE_PATH")

response = supabase_client.storage.from_(bucket_name).download(file_path)


# ──────────────────────────────────────────────────────────────────────────────
# Page config + styling (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main { padding: 0rem 1rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
[data-testid="stDataFrame"] th { text-align: center !important; }
[data-testid="stDataFrame"] td { text-align: center !important; }

/* Match card styling */
.match-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.match-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; font-size:12px; opacity:0.9; }
.match-teams  { display:flex; justify-content:space-between; align-items:center; margin:16px 0; font-size:16px; font-weight:bold; }
.team-name    { flex:1; text-align:center; }
.vs-divider   { padding:0 12px; font-size:14px; opacity:0.7; }
.match-odds   { display:flex; justify-content:space-around; margin:12px 0; padding:12px; background:rgba(255,255,255,0.1); border-radius:8px; }
.odd-box      { text-align:center; }
.odd-label    { font-size:11px; opacity:0.8; margin-bottom:4px; }
.odd-value    { font-size:16px; font-weight:bold; }
.match-predictions { display:flex; justify-content:space-between; margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.2); font-size:13px; }
.prediction-item   { text-align:center; }
.prediction-label  { font-size:10px; opacity:0.8; }
.prediction-value  { font-weight:bold; margin-top:4px; }
.strong-badge      { background:#ffd700; color:#333; padding:2px 8px; border-radius:12px; font-size:10px; font-weight:bold; }

@media (max-width: 768px) {
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
}
[data-testid="stDataFrame"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

/* Sticky filter strip — keeps controls visible while scrolling the table.
   The 3.5rem offset accounts for Streamlit's outer header bar so the
   title doesn't clip when scrolling. */
.filter-strip {
    position: sticky;
    top: 3.5rem;
    background: #0e1117;
    z-index: 100;
    padding: 8px 0;
    border-bottom: 1px solid #333;
    margin-bottom: 12px;
}

/* Vertical scrollable league panel — replaces the wide horizontal pill list.
   Sits in the right column of the filter strip. */
.league-panel {
    max-height: 240px;
    overflow-y: auto;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 6px 8px;
    background: #1a1c22;
    font-size: 0.82rem;
}
.league-panel label {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 1px 0;
    cursor: pointer;
}
.league-panel label:hover { background: #2c3038; }

/* Trim Streamlit's default top padding so the title sits closer to the top */
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
[data-testid="stExpander"] { margin-bottom: 4px; }

/* ── Column section separators ──────────────────────────────────────────────
   Streamlit renders st.dataframe as a glide-data-grid canvas (so we can't
   target columns directly via CSS), but it exposes a row container we can
   style. Instead we use the standard fallback: HTML table rendering, where
   each cell can be targeted by nth-child. We force the dataframe to use
   the HTML renderer below by tagging the parent with .bordered-table.
   The thick borders mark transitions between the six logical sections of
   the table (prediction → rank → season → form → venue → win/lose%).
*/
.bordered-table table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
}
.bordered-table thead th {
    background: #2a2d36;
    color: #fff;
    text-align: center;
    padding: 6px 8px;
    border-bottom: 2px solid #555;
    position: sticky;
    top: 0;
    z-index: 2;
    white-space: nowrap;
}
.bordered-table tbody td {
    text-align: center;
    padding: 4px 6px;
    border-bottom: 1px solid #333;
    white-space: nowrap;
}
.bordered-table tbody tr:nth-child(even) td { background: #1a1c22; }
.bordered-table tbody tr:hover td           { background: #2c3038; }

/* Thick separators between the six sections.
   Column indices are 1-based and reflect the display_columns order.
   Sections (after rename):
     1.  Date | League | Home | Away
     2.  H% | D% | A% | Pick | Strong | Conf | Draw?
     3.  H Rank | A Rank
     4.  H PPG | A PPG | H GPG | A GPG | H GCPG | A GCPG
     5.  H Form | A Form | H Δ | A Δ
     6.  H @Home | A @Away
     7.  H Win% | A Win% | H Lose% | A Lose%
     8.  BTTS%, BTTS, BTTS!, Lg BTTS, O2.5%, O2.5, O2.5!, Lg O2.5
*/
.bordered-table th:nth-child(4),    /* end of section 1 (Away)        */
.bordered-table td:nth-child(4),
.bordered-table th:nth-child(11),   /* end of section 2 (Draw?)       */
.bordered-table td:nth-child(11),
.bordered-table th:nth-child(13),   /* end of section 3 (A Rank)      */
.bordered-table td:nth-child(13),
.bordered-table th:nth-child(19),   /* end of section 4 (A GCPG)      */
.bordered-table td:nth-child(19),
.bordered-table th:nth-child(23),   /* end of section 5 (A Δ)         */
.bordered-table td:nth-child(23),
.bordered-table th:nth-child(25),   /* end of section 6 (A @Away)     */
.bordered-table td:nth-child(25),
.bordered-table th:nth-child(29),   /* end of section 7 (A Lose%)     */
.bordered-table td:nth-child(29) {
    border-right: 3px solid #6c7280 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Mobile cards builder
# ──────────────────────────────────────────────────────────────────────────────
def _build_mobile_cards_html(df: pd.DataFrame) -> str:
    """Render the filtered fixtures as a stack of expandable cards for mobile."""

    # Card-only stylesheet, scoped to the iframe body
    css = """<style>
    body {
        margin: 0; padding: 0;
        background: transparent;
        color: #e6e6e6;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        font-size: 14px;
    }
    .card {
        background: #1a1c22;
        border: 1px solid #2a2d36;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .card.strong-pick { border-left: 3px solid #ffd700; }
    .card-header {
        display: flex; justify-content: space-between;
        font-size: 0.72rem; color: #888; margin-bottom: 10px;
    }
    .team-row {
        display: grid;
        grid-template-columns: 36px 1fr auto;
        align-items: center; gap: 8px;
        padding: 4px 0;
        font-size: 0.95rem;
    }
    .team-row .rank { color: #888; font-size: 0.78rem; text-align: center; }
    .team-row .name { font-weight: 500; }
    .team-row .pct  {
        font-weight: 600; font-variant-numeric: tabular-nums;
        min-width: 52px; text-align: right;
    }
    .team-row.favourite .pct { color: #4ade80; }
    .draw-row {
        display: grid;
        grid-template-columns: 36px 1fr auto;
        align-items: center; gap: 8px;
        padding: 2px 0 8px;
        font-size: 0.85rem; color: #888;
    }
    .draw-row .name { font-style: italic; }
    .pick-row {
        display: flex; justify-content: space-between; align-items: center;
        border-top: 1px solid #2a2d36;
        padding: 8px 0 6px; font-size: 0.85rem;
    }
    .pick-row .pick-label  { color: #888; margin-right: 4px; }
    .pick-row .pick-value  { font-weight: 600; }
    .pick-row .strong-flag { color: #ffd700; margin-left: 6px; font-size: 0.78rem; }
    .pick-row .stars       { color: #fbbf24; font-size: 0.95rem; letter-spacing: 1px; }
    .markets-row {
        display: flex; gap: 18px; font-size: 0.85rem; padding-bottom: 6px;
    }
    .markets-row .label { color: #888; margin-right: 4px; }
    .markets-row .value { font-weight: 600; }
    .markets-row .market.confident .value { color: #4ade80; }
    .expand-toggle {
        display: flex; justify-content: center; align-items: center; gap: 6px;
        width: 100%;
        background: transparent; border: none;
        border-top: 1px solid #2a2d36;
        color: #888; padding: 8px 0 0; font-size: 0.78rem; cursor: pointer;
        margin-top: 4px;
    }
    .expand-toggle .arrow { transition: transform 0.2s; }
    .card.expanded .expand-toggle .arrow { transform: rotate(180deg); }
    .detail { display: none; margin-top: 12px; padding-top: 10px;
              border-top: 1px solid #2a2d36; font-size: 0.82rem; }
    .card.expanded .detail { display: block; }
    .detail-section { margin-bottom: 10px; }
    .section-title {
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.5px; color: #888; margin-bottom: 4px;
    }
    .stat-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px;
    }
    .stat-pair {
        display: flex; justify-content: space-between;
        font-variant-numeric: tabular-nums;
    }
    .stat-pair .stat-label { color: #888; }
    .stat-pair .stat-val   { font-weight: 500; }
    .stat-pair .stat-val.pos { color: #4ade80; }
    .stat-pair .stat-val.neg { color: #f87171; }
    .winlose-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px;
        margin-top: 4px;
    }
    .winlose-cell {
        text-align: center; background: #0e1117;
        border-radius: 4px; padding: 6px 2px;
    }
    .wl-label { font-size: 0.62rem; color: #888; display: block; }
    .wl-value { font-size: 0.85rem; font-weight: 600; margin-top: 2px; display: block; }
    .wl-value.high { color: #4ade80; }
    .wl-value.low  { color: #f87171; }
    .empty {
        text-align: center; color: #888; font-size: 0.9rem; padding: 40px 0;
    }
    .sort-bar {
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 10px; padding: 0 2px;
    }
    .sort-bar label {
        font-size: 0.75rem; color: #888;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .sort-bar select {
        flex: 1;
        background: #1a1c22;
        color: #e6e6e6;
        border: 1px solid #2a2d36;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.85rem;
        font-family: inherit;
        appearance: none;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='%23888' d='M0 0l5 6 5-6z'/></svg>");
        background-repeat: no-repeat;
        background-position: right 10px center;
        padding-right: 28px;
    }
    .sort-bar select:focus {
        outline: none;
        border-color: #6cb6ff;
    }
    </style>"""

    if len(df) == 0:
        return css + '<div class="empty">No fixtures match the selected filters.</div>'

    # Sort toolbar — shown above the card list. The dropdown triggers a
    # client-side reorder of the existing card DOM (no Streamlit re-run).
    sort_bar = (
        '<div class="sort-bar">'
        '<label for="sort">Sort:</label>'
        '<select id="sort">'
        '<option value="time">Time (earliest first)</option>'
        '<option value="league">League (A→Z)</option>'
        '<option value="confidence">Confidence (highest first)</option>'
        '</select>'
        '</div>'
        '<div id="card-list">'
    )

    def stars(margin):
        if pd.isna(margin): return ""
        if margin < 5:  return "★"
        if margin < 10: return "★★"
        if margin < 20: return "★★★"
        if margin < 35: return "★★★★"
        return "★★★★★"

    def fmt2(v):
        return f"{v:.2f}" if pd.notna(v) else "—"

    def fmt_pct(v):
        return f"{v:.1f}%" if pd.notna(v) else "—"

    def drift_class(v, deadband=0.10):
        if pd.isna(v) or abs(v) < deadband: return ""
        return "pos" if v > 0 else "neg"

    def drift_str(v, deadband=0.10):
        if pd.isna(v): return "—"
        if abs(v) < deadband: return f"{v:+.2f}"
        return f"{v:+.2f}"

    def wl_class(v, high=55, low=30):
        if pd.isna(v): return ""
        if v >= high: return "high"
        if v <  low:  return "low"
        return ""

    cards_html = []
    for _, row in df.iterrows():
        h_pct = row.get("Home Win %") or 0
        d_pct = row.get("Draw %") or 0
        a_pct = row.get("Away Win %") or 0
        pick  = str(row.get("Model Prediction") or "")
        is_h_fav = h_pct >= max(d_pct, a_pct)
        is_a_fav = a_pct >  max(h_pct, d_pct)

        h_team = str(row.get("Home Team") or "")
        a_team = str(row.get("Away Team") or "")
        h_rank = row.get("Home Team Rank")
        a_rank = row.get("Away Team Rank")

        match_dt = row.get("Match Date")
        try:
            if pd.notna(match_dt):
                if hasattr(match_dt, "tzinfo") and match_dt.tzinfo is not None:
                    match_dt = match_dt.tz_convert("Europe/London")
                date_str = match_dt.strftime("%a %d %b · %H:%M")
            else:
                date_str = "—"
        except Exception:
            date_str = str(match_dt)

        league = str(row.get("Excel Document") or "")

        strong_val = row.get("Strong Prediction")
        is_strong = pd.notna(strong_val) and str(strong_val).strip() not in ("", "nan", "None")

        margin = row.get("Confidence Score")
        star_str = stars(margin)

        # BTTS / O2.5 with confident colour cue
        btts_pct = row.get("BTTS %") or 0
        o25_pct  = row.get("Over 2.5 Goals %") or 0

        # Detail panel data (with safe getters)
        h_ppg, a_ppg = row.get("Home PPG (Season)"), row.get("Away PPG (Season)")
        h_gpg, a_gpg = row.get("Home Team GPG"), row.get("Away Team GPG")
        h_gcpg, a_gcpg = row.get("Home Team GCPG"), row.get("Away Team GCPG")
        h_l5, a_l5 = row.get("Home PPG (Last 5)"), row.get("Away PPG (Last 5)")
        h_fd, a_fd = row.get("Home Form Drift"), row.get("Away Form Drift")
        h_v, a_v   = row.get("Home PPG (At Home)"), row.get("Away PPG (Away)")
        h_vd, a_vd = row.get("Home Venue Drift"), row.get("Away Venue Drift")
        h_w, a_w   = row.get("Home Win % (Season)"), row.get("Away Win % (Season)")
        h_l, a_l   = row.get("Home Lose % (Season)"), row.get("Away Lose % (Season)")

        # Sort keys — used by the client-side dropdown to reorder cards
        # without re-rendering the whole iframe.
        try:
            ts_for_sort = match_dt.timestamp() if pd.notna(match_dt) and hasattr(match_dt, 'timestamp') else 0
        except Exception:
            ts_for_sort = 0
        conf_for_sort = float(margin) if pd.notna(margin) else 0.0

        card = []
        card.append(
            f'<div class="card{" strong-pick" if is_strong else ""}" '
            f'data-league="{league}" '
            f'data-time="{ts_for_sort}" '
            f'data-conf="{conf_for_sort}">'
        )
        card.append(f'<div class="card-header"><span>{date_str}</span><span>{league}</span></div>')

        card.append(
            f'<div class="team-row{" favourite" if is_h_fav else ""}">'
            f'<span class="rank">#{int(h_rank)}</span>' if pd.notna(h_rank) else '<div class="team-row"><span class="rank">—</span>'
        )
        # rebuild more cleanly
        card[-1] = (
            f'<div class="team-row{" favourite" if is_h_fav else ""}">'
            f'<span class="rank">{"#" + str(int(h_rank)) if pd.notna(h_rank) else "—"}</span>'
            f'<span class="name">{h_team}</span>'
            f'<span class="pct">{h_pct:.1f}%</span>'
            f'</div>'
        )
        card.append(
            f'<div class="team-row{" favourite" if is_a_fav else ""}">'
            f'<span class="rank">{"#" + str(int(a_rank)) if pd.notna(a_rank) else "—"}</span>'
            f'<span class="name">{a_team}</span>'
            f'<span class="pct">{a_pct:.1f}%</span>'
            f'</div>'
        )
        card.append(
            f'<div class="draw-row">'
            f'<span class="rank"></span>'
            f'<span class="name">Draw</span>'
            f'<span class="pct">{d_pct:.1f}%</span>'
            f'</div>'
        )

        # Pick row
        strong_html = '<span class="strong-flag">⭐ Strong</span>' if is_strong else ''
        card.append(
            f'<div class="pick-row">'
            f'<div><span class="pick-label">Pick:</span>'
            f'<span class="pick-value">{pick}</span>{strong_html}</div>'
            f'<div class="stars">{star_str}</div>'
            f'</div>'
        )

        # Markets row
        btts_class = "market confident" if btts_pct >= 60 else "market"
        o25_class  = "market confident" if o25_pct  >= 60 else "market"
        card.append(
            f'<div class="markets-row">'
            f'<div class="{btts_class}"><span class="label">BTTS</span>'
            f'<span class="value">{btts_pct:.0f}%</span></div>'
            f'<div class="{o25_class}"><span class="label">O2.5</span>'
            f'<span class="value">{o25_pct:.0f}%</span></div>'
            f'</div>'
        )

        # Expand toggle
        card.append(
            '<button class="expand-toggle" onclick="toggleCard(this)">'
            '<span class="toggle-text">More stats</span><span class="arrow">▾</span>'
            '</button>'
        )

        # Detail panel
        detail = []
        detail.append('<div class="detail">')
        detail.append('<div class="detail-section"><div class="section-title">Season form</div>')
        detail.append('<div class="stat-grid">')
        detail.append(f'<div class="stat-pair"><span class="stat-label">PPG</span>'
                      f'<span class="stat-val">{fmt2(h_ppg)} / {fmt2(a_ppg)}</span></div>')
        detail.append(f'<div class="stat-pair"><span class="stat-label">GPG</span>'
                      f'<span class="stat-val">{fmt2(h_gpg)} / {fmt2(a_gpg)}</span></div>')
        detail.append(f'<div class="stat-pair"><span class="stat-label">GCPG</span>'
                      f'<span class="stat-val">{fmt2(h_gcpg)} / {fmt2(a_gcpg)}</span></div>')
        detail.append('</div></div>')

        detail.append('<div class="detail-section"><div class="section-title">Form &amp; venue</div>')
        detail.append('<div class="stat-grid">')
        detail.append(f'<div class="stat-pair"><span class="stat-label">Last 5</span>'
                      f'<span class="stat-val">{fmt2(h_l5)} / {fmt2(a_l5)}</span></div>')
        detail.append(f'<div class="stat-pair"><span class="stat-label">Form Δ</span>'
                      f'<span class="stat-val"><span class="stat-val {drift_class(h_fd)}">{drift_str(h_fd)}</span> / '
                      f'<span class="stat-val {drift_class(a_fd)}">{drift_str(a_fd)}</span></span></div>')
        detail.append(f'<div class="stat-pair"><span class="stat-label">Venue</span>'
                      f'<span class="stat-val">{fmt2(h_v)} @H / {fmt2(a_v)} @A</span></div>')
        detail.append(f'<div class="stat-pair"><span class="stat-label">Venue Δ</span>'
                      f'<span class="stat-val"><span class="stat-val {drift_class(h_vd)}">{drift_str(h_vd)}</span> / '
                      f'<span class="stat-val {drift_class(a_vd)}">{drift_str(a_vd)}</span></span></div>')
        detail.append('</div></div>')

        detail.append('<div class="detail-section"><div class="section-title">Win / lose %</div>')
        detail.append('<div class="winlose-grid">')
        detail.append(f'<div class="winlose-cell"><span class="wl-label">H Win</span>'
                      f'<span class="wl-value {wl_class(h_w)}">{fmt_pct(h_w)}</span></div>')
        detail.append(f'<div class="winlose-cell"><span class="wl-label">A Lose</span>'
                      f'<span class="wl-value {wl_class(a_l)}">{fmt_pct(a_l)}</span></div>')
        detail.append(f'<div class="winlose-cell"><span class="wl-label">H Lose</span>'
                      f'<span class="wl-value {wl_class(h_l)}">{fmt_pct(h_l)}</span></div>')
        detail.append(f'<div class="winlose-cell"><span class="wl-label">A Win</span>'
                      f'<span class="wl-value {wl_class(a_w)}">{fmt_pct(a_w)}</span></div>')
        detail.append('</div></div>')
        detail.append('</div>')

        card.append("\n".join(detail))
        card.append('</div>')
        cards_html.append("\n".join(card))

    # Toggle script. After expanding/collapsing a card we tell Streamlit's
    # parent iframe to resize so the table page can grow with the content.
    # Also handles client-side sorting from the dropdown.
    script = """<script>
    function syncHeight() {
        const h = document.documentElement.scrollHeight;
        if (window.parent && window.parent.postMessage) {
            window.parent.postMessage(
                { type: 'streamlit:setFrameHeight', height: h },
                '*'
            );
        }
    }
    function toggleCard(btn) {
        const card = btn.closest('.card');
        card.classList.toggle('expanded');
        const txt = btn.querySelector('.toggle-text');
        txt.textContent = card.classList.contains('expanded') ? 'Less stats' : 'More stats';
        setTimeout(syncHeight, 50);
    }

    function sortCards(mode) {
        const list = document.getElementById('card-list');
        if (!list) return;
        const cards = Array.from(list.querySelectorAll('.card'));
        cards.sort((a, b) => {
            if (mode === 'league') {
                return a.dataset.league.localeCompare(b.dataset.league);
            }
            if (mode === 'confidence') {
                // highest confidence first
                return parseFloat(b.dataset.conf) - parseFloat(a.dataset.conf);
            }
            // default: time, earliest first
            return parseFloat(a.dataset.time) - parseFloat(b.dataset.time);
        });
        cards.forEach(c => list.appendChild(c));
    }

    // Initial sort: time (earliest first). Then attach the dropdown listener.
    window.addEventListener('load', () => {
        sortCards('time');
        const sel = document.getElementById('sort');
        if (sel) {
            sel.addEventListener('change', e => sortCards(e.target.value));
        }
        setTimeout(syncHeight, 50);
    });
    </script>"""

    return css + sort_bar + "\n".join(cards_html) + "</div>" + script



title_col, toggle_col = st.columns([4, 1])
with title_col:
    st.markdown(
        '<div style="display: flex; align-items: baseline; gap: 12px; margin: 0 0 4px 0;">'
        '<h1 style="margin: 0; font-size: 1.6rem;">⚽ Football Prediction Model</h1>'
        '<span style="color: #888; font-size: 0.85rem;">Model v5 predictions</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with toggle_col:
    mobile_view = st.checkbox(
        "📱 Mobile view",
        value=False,
        help="Switch to a card-based layout designed for phones",
    )

# ── Key / Legend ──────────────────────────────────────────────────────────────
with st.expander("📖 Key — what the columns and filters mean"):
    st.markdown("""
**Column reference** *(click any column header in the table to sort by it)*

**📊 Match identity**
- **Home / Away** — the two teams
- **H Rank / A Rank** — current league rank for each team

**🎯 Probabilities**
- **H% / D% / A%** — model's probability for Home win, Draw, Away win

**🔮 Prediction**
- **Pick** — the model's headline prediction (highest probability, or "Draw" if Draw Gate is on)
- **Strong** — name in this column means the Strong Prediction gate fired (high-conviction pick that confirms across stats)
- **Conf** — confidence as stars: ★ coin flip · ★★ slight lean · ★★★ clear favourite · ★★★★ strong · ★★★★★ very strong

**📈 Season structure** *(how the team has performed all season)*
- **H PPG / A PPG** — Points per game
- **H GPG / A GPG** — Goals scored per game
- **H GCPG / A GCPG** — Goals conceded per game

**🔥 Recent form** *(last 5 matches)*
- **H Form / A Form** — Last-5 PPG
- **H Δ / A Δ** — Form drift: last-5 PPG minus season PPG. Positive = team is hot, negative = cold

**🏟️ Venue**
- **H @Home** — Home team's PPG when playing at home
- **A @Away** — Away team's PPG when playing away
- **H @H Δ / A @A Δ** — Venue drift: venue PPG minus season PPG. Positive = team performs better than usual at this venue, negative = worse

**🎲 Win / Lose %** *(use these to filter accumulator picks. The four columns are paired by what supports each side: H Win% + A Lose% support backing the home team; H Lose% + A Win% support backing the away team)*
- **H Win% / A Win%** — % of season matches won
- **H Lose% / A Lose%** — % of season matches lost

**⚽ Goals markets**
- **BTTS%** — model probability both teams score
- **O2.5%** — model probability of 3+ total goals
*(The Y/N flags are now in the smart filters — see below)*

---

**Smart Filters** *(left sidebar — tick one or more, they stack with AND)*

- **⭐ Strong Predictions Only** — only fixtures where Strong Prediction gate fired
- **🎯 High Confidence** — probability margin (top minus second) ≥ 20pp
- **🟢 xG Dominance** — match xG gap ≥ 0.6 (one team much stronger)
- **🔵 Rank Gap (Top vs Bottom)** — league rank gap ≥ 10
- **🎲 Acca Quality** — favourite has won ≥ 50% of season matches AND underdog has lost ≥ 40% of theirs
- **🥅 Confident BTTS Y** — BTTS% ≥ 60% absolute (high-conviction BTTS picks)
- **⚽ Confident Over 2.5** — Over 2.5% ≥ 60% absolute (high-conviction goal-fest picks)
- **💪 All-In (Strong + High Conf)** — Strong Prediction + win margin ≥ 25pp — most conservative

**Custom thresholds** — same idea but with your own slider values. Tick "Apply custom thresholds" to activate.
""")

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Load + prep data
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(io.BytesIO(response))
df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')

# Detect schema (v4 vs v5) so dashboard works with either upload
HAS_V5_COLS = 'Confidence Score' in df.columns and 'Home xG Diff PM' in df.columns

# Backfill expected columns if missing (graceful degradation)
if 'Excel Document' not in df.columns:
    df['Excel Document'] = 'All Leagues'
if 'Over25YN' not in df.columns and 'Over 2.5 Goals %' in df.columns:
    df['Over25YN'] = df['Over 2.5 Goals %'].apply(lambda x: 'Y' if x >= 50 else 'N')

# Treat empty strings in Strong Prediction as missing
if 'Strong Prediction' in df.columns:
    df['Strong Prediction'] = df['Strong Prediction'].astype(str).str.strip()
    df.loc[df['Strong Prediction'].isin(['', 'nan', 'None']), 'Strong Prediction'] = pd.NA

# Favoured-team perspective columns (used by smart filters)
home_fav = df['Home Win %'] >= df['Away Win %']
df['_fav_win_pct'] = np.where(home_fav, df['Home Win %'],  df['Away Win %'])
df['_fav_team']    = np.where(home_fav, df['Home Team'],   df['Away Team'])

# Season Win% / Lose% from the favourite's and underdog's perspective
# (used by the accumulator filter — "favourite that genuinely wins, underdog that genuinely loses")
if 'Home Win % (Season)' in df.columns and 'Away Win % (Season)' in df.columns:
    df['_fav_season_win']    = np.where(home_fav, df['Home Win % (Season)'],  df['Away Win % (Season)'])
    df['_dog_season_lose']   = np.where(home_fav, df['Away Lose % (Season)'], df['Home Lose % (Season)'])
else:
    df['_fav_season_win']  = np.nan
    df['_dog_season_lose'] = np.nan

# v5-specific perspective columns
if HAS_V5_COLS:
    df['_fav_xg_diff_pm'] = np.where(home_fav, df['Home xG Diff PM'], df['Away xG Diff PM'])
    df['_xg_match_gap']   = (df['Home xG'] - df['Away xG']).abs()
    df['_win_margin']     = df[['Home Win %', 'Draw %', 'Away Win %']].max(axis=1) - \
                            df[['Home Win %', 'Draw %', 'Away Win %']].apply(
                                lambda r: sorted(r, reverse=True)[1], axis=1
                            )
else:
    # Fallback for v4 data
    df['_fav_xg_diff_pm'] = np.nan
    df['_xg_match_gap']   = (df.get('Home xG', 0) - df.get('Away xG', 0)).abs()
    df['_win_margin']     = df.get('Confidence Score', np.nan)

# Rank gap
if 'Home Team Rank' in df.columns and 'Away Team Rank' in df.columns:
    df['_rank_gap'] = (df['Home Team Rank'] - df['Away Team Rank']).abs()
else:
    df['_rank_gap'] = np.nan

# Favourite-side derived fields used by the redesigned smart filters.
#   _fav_gpg_advantage  — favourite's GPG minus opponent's GPG (Hot GPG filter)
#   _fav_match_xg_adv   — favourite's match xG minus opponent's match xG (Hidden Gem)
# These compute relative to whichever side the model has picked. We use
# the model's actual pick (Home/Away/Draw) — for Draw picks the value stays
# unsigned via the absolute fallback (since neither side is "the favourite").
if {'Home Team GPG', 'Away Team GPG'}.issubset(df.columns):
    pick_is_home = df['Model Prediction'] == df['Home Team']
    pick_is_away = df['Model Prediction'] == df['Away Team']
    fav_gpg = np.where(pick_is_home, df['Home Team GPG'],
                np.where(pick_is_away, df['Away Team GPG'], np.nan))
    opp_gpg = np.where(pick_is_home, df['Away Team GPG'],
                np.where(pick_is_away, df['Home Team GPG'], np.nan))
    df['_fav_gpg_advantage'] = fav_gpg - opp_gpg
else:
    df['_fav_gpg_advantage'] = np.nan

if {'Home xG', 'Away xG'}.issubset(df.columns):
    pick_is_home = df['Model Prediction'] == df['Home Team']
    pick_is_away = df['Model Prediction'] == df['Away Team']
    fav_xg = np.where(pick_is_home, df['Home xG'],
                np.where(pick_is_away, df['Away xG'], np.nan))
    opp_xg = np.where(pick_is_home, df['Away xG'],
                np.where(pick_is_away, df['Home xG'], np.nan))
    df['_fav_match_xg_adv'] = fav_xg - opp_xg
else:
    df['_fav_match_xg_adv'] = np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: league + date filters
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="filter-strip">', unsafe_allow_html=True)

leagues = sorted(df['Excel Document'].dropna().unique().tolist())
if "leagues_selected" not in st.session_state:
    st.session_state.leagues_selected = leagues

# Two-column layout: filters on the left, leagues+dates compact on the right.
left_col, right_col = st.columns([3, 1], gap="medium")

# ── RIGHT: leagues + date pickers (compact panel) ──
with right_col:
    # League quick-select buttons — small and inline
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("All leagues", use_container_width=True, help="Select all leagues"):
            st.session_state.leagues_selected = leagues
            st.rerun()
    with bcol2:
        if st.button("Clear", use_container_width=True, help="Clear all leagues"):
            st.session_state.leagues_selected = []
            st.rerun()

    selected_leagues = st.multiselect(
        f"Leagues ({len(st.session_state.leagues_selected)}/{len(leagues)})",
        leagues,
        default=st.session_state.leagues_selected,
        key="leagues_widget",
        label_visibility="visible",
    )
    st.session_state.leagues_selected = selected_leagues

    # Date pickers under the league box
    if df['Match Date'].notna().any():
        min_date = df['Match Date'].min().date()
        max_date = df['Match Date'].max().date()
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
        with dcol2:
            end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
    else:
        start_date = end_date = None


# ──────────────────────────────────────────────────────────────────────────────
# Smart filters — REBUILT for v5 signals
# ──────────────────────────────────────────────────────────────────────────────
# ── LEFT: smart filter checkboxes ──
with left_col:
    st.markdown(
        '<div style="margin: 0 0 6px 0; font-size: 0.85rem; color: #aaa;">'
        '<strong style="color: #ddd;">🎯 Smart Filters</strong> · '
        'tick one or more — they stack with AND logic'
        '</div>',
        unsafe_allow_html=True,
    )

FILTER_DEFS = [
    # ── Pick conviction filters (mutually compatible — stack with AND) ──
    {
        "key":   "strong_only",
        "label": "⭐ Strong",
        "desc":  "Model's Strong Prediction gate fired. Highest conviction tier. "
                 "Validated at 78% precision across 41 picks (3-day weekend sample).",
        "numeric": [],
        "strong":  True,
    },
    {
        "key":   "premium_4star",
        "label": "🎯 4★+ Premium",
        "desc":  "Confidence margin ≥ 20pp (4-star or 5-star). Broader high-conviction "
                 "net than Strong. Validated at 70.5% precision across 61 picks.",
        "numeric": [("_win_margin", ">=", 20)],
        "strong":  False,
    },
    {
        "key":   "hot_gpg",
        "label": "💎 Hot GPG Pick",
        "desc":  "Favourite scores ≥ 0.6 more goals per match than opponent. "
                 "Best single filter from held-out testing — 81.5% precision on "
                 "27 picks across Fri+Sat. Goal-volume favourites only.",
        "numeric": [("_fav_gpg_advantage", ">=", 0.6)],
        "strong":  False,
    },
    {
        "key":   "hidden_gem",
        "label": "🔍 Hidden Gem",
        "desc":  "2-3★ picks with match xG advantage ≥ 0.4 AND confidence ≥ 15pp. "
                 "Possible value picks in the mid-tier — 78% on 9 fixtures so far. "
                 "Small sample, treat as experimental.",
        "numeric": [("_fav_match_xg_adv", ">=", 0.4),
                    ("_win_margin", ">=", 15),
                    ("_win_margin", "<", 20)],
        "strong":  False,
    },

    # ── Market filters (independent of pick conviction) ──
    {
        "key":   "confident_btts",
        "label": "🥅 Confident BTTS Y",
        "desc":  "BTTS% ≥ 60% (absolute, league-independent). "
                 "Validated at 61% precision across 38 picks vs 54% base rate.",
        "numeric": [("BTTS %", ">=", 60)],
        "strong":  False,
    },
    {
        "key":   "confident_o25",
        "label": "⚽ Confident Over 2.5",
        "desc":  "Over 2.5% ≥ 60% (absolute, league-independent). "
                 "Validated at 58% precision across 31 picks vs 52% base rate.",
        "numeric": [("Over 2.5 Goals %", ">=", 60)],
        "strong":  False,
    },

    # ── Quality / avoidance filter ──
    {
        "key":   "hide_draws",
        "label": "🚫 Hide Draw Picks",
        "desc":  "Hide fixtures where the model predicts a Draw — your 'avoid this match' "
                 "flag. Draw picks hit at 36% across the weekend and shouldn't be bet on.",
        "numeric": [],
        "hide_draws":  True,
    },
]

# Render the checkbox grid + custom-thresholds expander inside the left column
with left_col:
    # Visual structure: 4 pick filters on top row, 3 market/quality on bottom.
    # The first 4 entries in FILTER_DEFS are pick filters; the rest are
    # markets + the hide-draws quality filter.
    active_filters = []

    # Row 1 — pick conviction filters
    pick_row = FILTER_DEFS[:4]
    cols = st.columns(4)
    for col, f in zip(cols, pick_row):
        with col:
            if st.checkbox(f["label"], value=False, key=f"chk_{f['key']}", help=f["desc"]):
                active_filters.append(f)

    # Row 2 — markets + quality (3 filters, give them a bit more breathing room)
    market_row = FILTER_DEFS[4:]
    cols = st.columns(4)  # 4 columns; leave the last empty for spacing
    for col, f in zip(cols, market_row):
        with col:
            if st.checkbox(f["label"], value=False, key=f"chk_{f['key']}", help=f["desc"]):
                active_filters.append(f)


    # Custom thresholds tucked into an expander
    with st.expander("⚙️ Custom thresholds (advanced)", expanded=False):
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            custom_win_pct = st.slider("Min win probability (%)", 0, 100, 50, 5)
            custom_fav_win = st.slider("Min favourite season Win%", 0, 100, 0, 5)
        with cc2:
            custom_draw_pct = st.slider("Max draw probability (%)", 0, 50, 30, 1)
            custom_dog_lose = st.slider("Min underdog season Lose%", 0, 100, 0, 5)
        with cc3:
            custom_margin   = st.slider("Min confidence margin (pp)", 0, 50, 0, 1)
            custom_xg_gap   = st.slider("Min match xG gap", 0.0, 3.0, 0.0, 0.1)
        with cc4:
            custom_rank_gap = st.slider("Min league rank gap", 0, 24, 0, 1)
            use_custom      = st.checkbox("Apply custom thresholds", value=False)

    # Custom Checklist — manual pick verification with mix-and-match conditions.
    # Each condition can be Off / Required / Bonus. Bonus checks contribute
    # toward a minimum-bonus-count threshold but aren't individually mandatory.
    # Rendered inline (not behind an expander) so it's always visible.
    st.markdown(
        '<div style="margin: 10px 0 4px 0; font-size: 0.85rem; color: #aaa;">'
        '<strong style="color: #ddd;">📋 Custom Checklist</strong> · '
        'set each condition to <span style="color:#f87171;">Required</span> '
        '(must pass) or <span style="color:#fbbf24;">Bonus</span> '
        '(contributes to a minimum threshold). Auto-hides Draw picks.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Picked-side toggle (separate from the Off/Required/Bonus conditions —
    # it's always applied, you just choose which side(s) you're considering).
    side_col, _, _, _ = st.columns(4)
    with side_col:
        picked_side = st.radio(
            "🏟️ Picked side",
            options=["Home only", "Both", "Away only"],
            index=1,
            horizontal=True,
            help="Which side of the model's pick to consider. 'Both' = home or away.",
        )

    CHECKLIST_CONDITIONS = [
        {"key": "ppg_higher",     "label": "📈 Higher PPG",
         "desc": "Pick's season PPG > opponent's season PPG"},
        {"key": "gpg_higher",     "label": "⚽ Higher GPG (scores more)",
         "desc": "Pick scores more goals per match than opponent"},
        {"key": "gcpg_lower",     "label": "🛡️ Lower GCPG (concedes less)",
         "desc": "Pick concedes fewer goals per match than opponent"},
        {"key": "better_form",    "label": "🔥 Better recent form",
         "desc": "Pick's last-5 PPG > opponent's last-5 PPG"},
        {"key": "venue_stronger", "label": "🏟️ Better venue record",
         "desc": "Pick's PPG at this venue (home or away) > opponent's PPG at their venue"},
        {"key": "win_lose_gap",   "label": "🎯 Wins more, loses less",
         "desc": "Pick has higher season Win% AND lower season Lose% than opponent"},
    ]

    # Render in 3 columns × 2 rows so all 6 conditions are visible at once
    checklist_state = {}
    for row_start in range(0, len(CHECKLIST_CONDITIONS), 3):
        cols = st.columns(3)
        for col, cond in zip(cols, CHECKLIST_CONDITIONS[row_start:row_start + 3]):
            with col:
                state = st.radio(
                    cond["label"],
                    options=["Off", "Required", "Bonus"],
                    index=0,
                    key=f"chk_{cond['key']}_state",
                    help=cond["desc"],
                    horizontal=True,
                )
                checklist_state[cond["key"]] = state

    # Bottom row — slider + apply button
    bonus_count = sum(1 for s in checklist_state.values() if s == "Bonus")
    any_active = (
        any(s != "Off" for s in checklist_state.values())
        or picked_side != "Both"
    )
    slider_col, apply_col = st.columns([3, 1])
    with slider_col:
        if bonus_count > 0:
            min_bonus = st.slider(
                f"Minimum bonus checks to pass (out of {bonus_count})",
                min_value=0, max_value=bonus_count,
                value=min(bonus_count, max(1, bonus_count - 1)),
            )
        else:
            min_bonus = 0
            st.markdown(
                '<div style="color:#666;font-size:0.8rem;padding:8px 0;">'
                'No Bonus conditions set — minimum threshold not applicable.</div>',
                unsafe_allow_html=True,
            )
    with apply_col:
        st.markdown('<div style="height: 26px;"></div>', unsafe_allow_html=True)
        use_checklist = st.checkbox("Apply checklist", value=False,
                                     disabled=not any_active)

    # Placeholder for the metrics dashboard.
    # Filled below, after the filter logic computes filtered_df.
    metrics_placeholder = st.empty()

st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = df[df['Excel Document'].isin(selected_leagues)] if selected_leagues else df.copy()
if start_date and end_date and df['Match Date'].notna().any():
    filtered_df = filtered_df[
        (filtered_df['Match Date'].dt.date >= start_date) &
        (filtered_df['Match Date'].dt.date <= end_date)
    ]

pre_filter_count    = len(filtered_df)
smart_filter_active = False
active_descs        = []

combined_mask = pd.Series(True, index=filtered_df.index)

for f in active_filters:
    smart_filter_active = True
    active_descs.append(f["desc"])
    if f.get("strong"):
        combined_mask &= filtered_df['Strong Prediction'].notna()
    if f.get("hide_draws"):
        combined_mask &= (filtered_df['Model Prediction'] != 'Draw')
    for col, op, val in f["numeric"]:
        if col not in filtered_df.columns:
            continue
        if op == ">=":
            combined_mask &= filtered_df[col] >= val
        elif op == "<=":
            combined_mask &= filtered_df[col] <= val
        elif op == ">":
            combined_mask &= filtered_df[col] > val
        elif op == "<":
            combined_mask &= filtered_df[col] < val

if use_custom:
    smart_filter_active = True
    active_descs.append(
        f"Win% ≥ {custom_win_pct} · Draw% ≤ {custom_draw_pct} · "
        f"Margin ≥ {custom_margin}pp · xG gap ≥ {custom_xg_gap} · "
        f"Rank gap ≥ {custom_rank_gap} · "
        f"Fav Win% ≥ {custom_fav_win} · Dog Lose% ≥ {custom_dog_lose}"
    )
    combined_mask &= (
        (filtered_df['_fav_win_pct'] >= custom_win_pct) &
        (filtered_df['Draw %']       <= custom_draw_pct) &
        (filtered_df['_win_margin']  >= custom_margin) &
        (filtered_df['_xg_match_gap']>= custom_xg_gap) &
        (filtered_df['_rank_gap'].fillna(0)         >= custom_rank_gap) &
        (filtered_df['_fav_season_win'].fillna(0)   >= custom_fav_win) &
        (filtered_df['_dog_season_lose'].fillna(0)  >= custom_dog_lose)
    )

# Custom checklist evaluation. Each condition produces a boolean Series.
# Required conditions become AND-mask additions. Bonus conditions are summed
# per row; the row passes if its bonus count >= min_bonus.
# The picked_side toggle is always applied — it restricts which side(s) qualify.
if use_checklist and any_active:
    smart_filter_active = True

    pick = filtered_df['Model Prediction'].astype(str).str.strip()
    home_t = filtered_df['Home Team'].astype(str).str.strip()
    away_t = filtered_df['Away Team'].astype(str).str.strip()
    pick_is_home = pick == home_t
    pick_is_away = pick == away_t
    # Auto-hide Draw picks since the checklist tests favourite-side stats
    not_a_draw = pick != "Draw"

    # Picked-side toggle
    if picked_side == "Home only":
        side_mask = pick_is_home
    elif picked_side == "Away only":
        side_mask = pick_is_away
    else:  # "Both"
        side_mask = pick_is_home | pick_is_away

    def _pick_side(home_col, away_col):
        return pd.Series(
            np.where(pick_is_home, filtered_df[home_col],
                np.where(pick_is_away, filtered_df[away_col], np.nan)),
            index=filtered_df.index
        )
    def _opp_side(home_col, away_col):
        return pd.Series(
            np.where(pick_is_home, filtered_df[away_col],
                np.where(pick_is_away, filtered_df[home_col], np.nan)),
            index=filtered_df.index
        )

    # Compute each condition's pass/fail per row
    conditions = {
        "ppg_higher":      _pick_side("Home PPG (Season)", "Away PPG (Season)") >
                           _opp_side("Home PPG (Season)", "Away PPG (Season)"),
        "gpg_higher":      _pick_side("Home Team GPG", "Away Team GPG") >
                           _opp_side("Home Team GPG", "Away Team GPG"),
        "gcpg_lower":      _pick_side("Home Team GCPG", "Away Team GCPG") <
                           _opp_side("Home Team GCPG", "Away Team GCPG"),
        "better_form":     _pick_side("Home PPG (Last 5)", "Away PPG (Last 5)") >
                           _opp_side("Home PPG (Last 5)", "Away PPG (Last 5)"),
        # Venue: pick's PPG at THEIR venue vs opp's PPG at THEIR venue
        # When pick is home → pick uses Home PPG (At Home), opp uses Away PPG (Away)
        # When pick is away → pick uses Away PPG (Away), opp uses Home PPG (At Home)
        # The _pick_side / _opp_side helpers handle the swap based on pick_is_home.
        "venue_stronger":  _pick_side("Home PPG (At Home)", "Away PPG (Away)") >
                           _opp_side("Home PPG (At Home)", "Away PPG (Away)"),
        "win_lose_gap":    (_pick_side("Home Win % (Season)", "Away Win % (Season)") >
                            _opp_side("Home Win % (Season)", "Away Win % (Season)")) &
                           (_pick_side("Home Lose % (Season)", "Away Lose % (Season)") <
                            _opp_side("Home Lose % (Season)", "Away Lose % (Season)")),
    }

    required_mask = pd.Series(True, index=filtered_df.index)
    bonus_score = pd.Series(0, index=filtered_df.index)
    required_descs = []
    bonus_descs = []
    for key, state in checklist_state.items():
        if state == "Required":
            required_mask &= conditions[key].fillna(False)
            required_descs.append(key)
        elif state == "Bonus":
            bonus_score += conditions[key].fillna(False).astype(int)
            bonus_descs.append(key)

    bonus_pass = bonus_score >= min_bonus if bonus_descs else pd.Series(True, index=filtered_df.index)
    combined_mask &= not_a_draw & side_mask & required_mask & bonus_pass

    summary_parts = []
    if picked_side != "Both":
        summary_parts.append(f"Side: {picked_side}")
    if required_descs:
        summary_parts.append(f"Required: {', '.join(required_descs)}")
    if bonus_descs:
        summary_parts.append(f"Bonus ≥ {min_bonus}/{len(bonus_descs)}: {', '.join(bonus_descs)}")
    active_descs.append("Checklist · " + " · ".join(summary_parts))

if smart_filter_active:
    filtered_df = filtered_df[combined_mask]

smart_filter_desc = " · ".join(active_descs)


# ──────────────────────────────────────────────────────────────────────────────
# Render the metrics dashboard into the placeholder we created in left_col.
# ──────────────────────────────────────────────────────────────────────────────
def _count(condition_series) -> int:
    """Safe count returning 0 if the column doesn't exist."""
    try:
        return int(condition_series.fillna(False).sum())
    except Exception:
        return 0

def _fmt(n: int, total: int) -> str:
    """'48 (21.9%)' style formatting."""
    if total == 0:
        return "0"
    return f"{n} <span style='color:#888;font-weight:normal;'>({n / total * 100:.0f}%)</span>"

# Compute metrics off the filtered set
n_total          = len(filtered_df)
n_strong         = _count(filtered_df['Strong Prediction'].notna())
n_draws          = _count(filtered_df['Model Prediction'] == 'Draw')
n_conf_btts      = _count(filtered_df['BTTS %'] >= 60) if 'BTTS %' in filtered_df.columns else 0
n_conf_o25       = _count(filtered_df['Over 2.5 Goals %'] >= 60) if 'Over 2.5 Goals %' in filtered_df.columns else 0
n_acca_quality   = _count(
    (filtered_df['_fav_season_win'].fillna(0) >= 50) &
    (filtered_df['_dog_season_lose'].fillna(0) >= 40)
)
avg_h = filtered_df['Home Win %'].mean() if n_total else 0
avg_a = filtered_df['Away Win %'].mean() if n_total else 0

delta_text = ""
if smart_filter_active:
    delta = n_total - pre_filter_count
    delta_text = (f' <span style="color:#888;font-weight:normal;font-size:0.75rem;">'
                  f'({delta:+d} after filters)</span>')

def _metric_html(label: str, value_html: str) -> str:
    return (
        '<div style="background:#1a1c22;border:1px solid #333;border-radius:6px;'
        'padding:10px 14px;text-align:left;">'
        f'<div style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
        f'<div style="font-size:1.4rem;font-weight:600;color:#fff;margin-top:2px;">{value_html}</div>'
        '</div>'
    )

metrics_html = (
    '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px;">'
    + _metric_html("Fixtures shown", f"{n_total}{delta_text}")
    + _metric_html("Strong picks",   _fmt(n_strong, n_total))
    + _metric_html("Draws picked",   _fmt(n_draws, n_total))
    + _metric_html("Confident BTTS", _fmt(n_conf_btts, n_total))
    + _metric_html("Acca Quality",   _fmt(n_acca_quality, n_total))
    + _metric_html("Confident O2.5", _fmt(n_conf_o25, n_total))
    + _metric_html("Avg H%",         f"{avg_h:.1f}%")
    + _metric_html("Avg A%",         f"{avg_a:.1f}%")
    + '</div>'
)

metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Main view
# ──────────────────────────────────────────────────────────────────────────────
if len(filtered_df) == 0:
    st.warning("No fixtures match the selected filters.")
else:
    if smart_filter_active and smart_filter_desc:
        st.info(f"🎯 **Smart filter active** — {smart_filter_desc}")

    # Smart filter active banner moved here so it sits at the top of the table
    # area (the metrics dashboard is now in the filter strip up top).

    # ── Star confidence mapping ────────────────────────────────────────────────
    # Confidence Score is the probability margin (top minus second).
    # Stars rendered in gold so they stand out from the surrounding numbers.
    GOLD = "#fbbf24"  # tailwind amber-400 — readable against dark theme
    def stars_from_margin(m):
        if pd.isna(m):       return "-"
        if m < 5:            stars = "★"
        elif m < 10:         stars = "★★"
        elif m < 20:         stars = "★★★"
        elif m < 35:         stars = "★★★★"
        else:                stars = "★★★★★"
        return f'<span style="color:{GOLD};">{stars}</span>'

    # ── Build display table ────────────────────────────────────────────────────
    # Logical column order matches the six-section thinking flow:
    #   1. Who wins?     → H% / D% / A% / Pred / Strong / ★Conf / Draw?
    #   2. Rank          → H R / A R
    #   3. Season form   → PPG / GPG / GCPG (home and away teams)
    #   4. Recent form   → Last 5 PPG + drift vs season
    #   5. Venue         → Home team's home PPG, Away team's away PPG
    #   6. Win / Lose %  → for filtering against weak/strong sides

    display_columns = [
        'Match Date', 'Excel Document',
        # 1. Match identity (now includes ranks)
        'Home Team', 'Away Team',
        'Home Team Rank', 'Away Team Rank',
        # 2. Probabilities
        'Home Win %', 'Draw %', 'Away Win %',
        # 3. Prediction (border before this section)
        'Model Prediction', 'Strong Prediction', 'Confidence Score',
        # 4. Season structure
        'Home PPG (Season)',  'Away PPG (Season)',
        'Home Team GPG',      'Away Team GPG',
        'Home Team GCPG',     'Away Team GCPG',
        # 5. Recent form
        'Home PPG (Last 5)',  'Away PPG (Last 5)',
        'Home Form Drift',    'Away Form Drift',
        # 6. Venue
        'Home PPG (At Home)', 'Away PPG (Away)',
        'Home Venue Drift',   'Away Venue Drift',
        # 7. Win / Lose % — paired by team, then border, then opposite pairing
        'Home Win % (Season)',  'Away Lose % (Season)',
        'Home Lose % (Season)', 'Away Win % (Season)',
        # 8. BTTS / Over 2.5 (just the percentages — Y/N flags moved to filters)
        'BTTS %', 'Over 2.5 Goals %',
    ]

    available_columns = [c for c in display_columns if c in filtered_df.columns]
    table_df = filtered_df[available_columns].copy()

    # Convert Confidence Score → stars (do this before rename)
    if 'Confidence Score' in table_df.columns:
        table_df['Confidence Score'] = table_df['Confidence Score'].apply(stars_from_margin)

    table_df.rename(columns={
        'Match Date':             'Date',
        'Excel Document':         'League',
        'Home Team':              'Home',
        'Away Team':              'Away',
        'Home Team Rank':         'H Rank',
        'Away Team Rank':         'A Rank',
        'Home Win %':             'H%',
        'Draw %':                 'D%',
        'Away Win %':             'A%',
        'Model Prediction':       'Pick',
        'Strong Prediction':      'Strong',
        'Confidence Score':       'Conf',
        'Home PPG (Season)':      'H PPG',
        'Away PPG (Season)':      'A PPG',
        'Home Team GPG':          'H GPG',
        'Away Team GPG':          'A GPG',
        'Home Team GCPG':         'H GCPG',
        'Away Team GCPG':         'A GCPG',
        'Home PPG (Last 5)':      'H Form',
        'Away PPG (Last 5)':      'A Form',
        'Home Form Drift':        'H Δ',
        'Away Form Drift':        'A Δ',
        'Home PPG (At Home)':     'H @Home',
        'Away PPG (Away)':        'A @Away',
        'Home Venue Drift':       'H @H Δ',
        'Away Venue Drift':       'A @A Δ',
        'Home Win % (Season)':    'H Win%',
        'Away Win % (Season)':    'A Win%',
        'Home Lose % (Season)':   'H Lose%',
        'Away Lose % (Season)':   'A Lose%',
        'BTTS %':                 'BTTS%',
        'Over 2.5 Goals %':       'O2.5%',
    }, inplace=True)

    # ── Formatting ─────────────────────────────────────────────────────────────
    # Plain percentages (no colour banding)
    plain_pct_cols = ['H%', 'D%', 'A%', 'BTTS%', 'O2.5%']
    for c in plain_pct_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

    # Win% / Lose% get colour bands so accumulator-relevant cells pop:
    #   ≥ 55% → green   (this team wins/loses a lot — strong signal)
    #   < 30% → red     (this team rarely wins/loses — weak signal)
    #   in-between → plain (mixed)
    GREEN = "#4ade80"
    RED   = "#f87171"
    def _band_pct(x):
        if pd.isna(x):
            return "-"
        s = f"{x:.1f}%"
        if x >= 55:
            return f'<span style="color:{GREEN};">{s}</span>'
        if x < 30:
            return f'<span style="color:{RED};">{s}</span>'
        return s
    for c in ['H Win%', 'A Win%', 'H Lose%', 'A Lose%']:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(_band_pct)

    two_dp_cols = ['H GPG', 'A GPG', 'H GCPG', 'A GCPG',
                   'H PPG', 'A PPG', 'H Form', 'A Form',
                   'H @Home', 'A @Away']
    for c in two_dp_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    # Form & venue drift get a +/- sign AND colour-coding so direction is
    # obvious at a glance. A ±0.10 dead-band keeps small (probably-noise)
    # drifts uncoloured. Green = team performing better than usual, red = worse.
    DRIFT_DEAD_BAND = 0.10
    def _drift_html(x):
        if pd.isna(x):
            return "-"
        if x >= DRIFT_DEAD_BAND:
            return f'<span style="color:#4ade80;">+{x:.2f}</span>'
        if x <= -DRIFT_DEAD_BAND:
            return f'<span style="color:#f87171;">{x:.2f}</span>'
        # Within dead-band — show value plainly so the reader still sees it.
        return f"{x:+.2f}"
    for c in ['H Δ', 'A Δ', 'H @H Δ', 'A @A Δ']:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(_drift_html)

    rank_cols = ['H Rank', 'A Rank']
    for c in rank_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")

    if 'Date' in table_df.columns:
        # Times in source data are UTC. Convert to UK local time (handles
        # BST/GMT automatically) so kickoffs display as the user expects.
        def _fmt_kickoff(x):
            if pd.isna(x):
                return "-"
            try:
                if x.tzinfo is not None:
                    x = x.tz_convert('Europe/London')
            except Exception:
                pass
            return x.strftime('%Y-%m-%d %H:%M')
        table_df['Date'] = table_df['Date'].apply(_fmt_kickoff)

    # Render via st.components.v1.html so the JS sort handler actually executes.
    # Streamlit's st.markdown strips <script> tags as a security measure, so the
    # table goes into a real iframe with its own CSS scope — that's why all the
    # styling is bundled inline below rather than relying on the page CSS.
    html_table = table_df.to_html(index=False, escape=False, classes="bordered-table-inner")
    html_table = html_table.replace(
        '<table border="1" class="dataframe bordered-table-inner">',
        '<table id="bordered-sortable" class="dataframe bordered-table-inner">',
        1,
    )

    # Inline CSS — these styles need to live inside the iframe.
    iframe_styles = """
    <style>
        body {
            margin: 0;
            background: transparent;
            color: #e6e6e6;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        .table-wrap {
            border: 1px solid #333;
            border-radius: 4px;
        }
        table#bordered-sortable {
            border-collapse: collapse;
            width: 100%;
            font-size: 12px;
        }
        table#bordered-sortable thead th {
            background: #2a2d36;
            color: #fff;
            text-align: center;
            padding: 6px 8px;
            border-bottom: 2px solid #555;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
        }
        table#bordered-sortable thead th:hover { background: #3a3d46; }
        table#bordered-sortable thead th.sort-asc::after  { content: ' ▲'; color: #6cb6ff; }
        table#bordered-sortable thead th.sort-desc::after { content: ' ▼'; color: #6cb6ff; }
        table#bordered-sortable thead th:not(.sort-asc):not(.sort-desc)::after {
            content: ' ↕'; color: #555; font-size: 10px;
        }
        table#bordered-sortable tbody td {
            text-align: center;
            padding: 4px 6px;
            border-bottom: 1px solid #333;
            white-space: nowrap;
        }
        table#bordered-sortable tbody tr:nth-child(even) td { background: #1a1c22; }
        table#bordered-sortable tbody tr:hover td           { background: #2c3038; }

        /* Section separators — match the column groups.
           Indices reflect the display order:
             1.  Date | League | Home | Away              ← border after 4
             2.  H Rank | A Rank                          ← border after 6
             3.  H% | D% | A%                             ← border after 9
             4.  Pick | Strong | Conf                     ← border after 12
             5.  H/A PPG | H/A GPG | H/A GCPG             ← border after 18
             6.  H Form | A Form | H Δ | A Δ              ← border after 22
             7.  H @Home | A @Away | H @H Δ | A @A Δ      ← border after 26
             8.  H Win% | A Lose%                         ← border after 28
             9.  H Lose% | A Win%                         ← border after 30
            10.  BTTS% | O2.5%                            (no trailing border)
        */
        table#bordered-sortable th:nth-child(4),
        table#bordered-sortable td:nth-child(4),
        table#bordered-sortable th:nth-child(6),
        table#bordered-sortable td:nth-child(6),
        table#bordered-sortable th:nth-child(9),
        table#bordered-sortable td:nth-child(9),
        table#bordered-sortable th:nth-child(12),
        table#bordered-sortable td:nth-child(12),
        table#bordered-sortable th:nth-child(18),
        table#bordered-sortable td:nth-child(18),
        table#bordered-sortable th:nth-child(22),
        table#bordered-sortable td:nth-child(22),
        table#bordered-sortable th:nth-child(26),
        table#bordered-sortable td:nth-child(26),
        table#bordered-sortable th:nth-child(28),
        table#bordered-sortable td:nth-child(28),
        table#bordered-sortable th:nth-child(30),
        table#bordered-sortable td:nth-child(30) {
            border-right: 3px solid #6c7280 !important;
        }
    </style>
    """

    sort_script = r"""
    <script>
    (function() {
        const table = document.getElementById('bordered-sortable');
        if (!table) return;

        const headers = table.querySelectorAll('thead th');
        let sortState = { col: null, dir: 1 };

        function cellSortValue(cell) {
            const raw = cell.textContent.trim();
            if (raw === '-' || raw === '' || raw === 'nan' || raw === 'None') {
                return { num: null, str: '' };
            }
            // Datetime: "YYYY-MM-DD HH:MM" or "YYYY-MM-DD" — sort as epoch ms
            if (/^\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?$/.test(raw)) {
                const ts = Date.parse(raw.replace(' ', 'T'));
                if (!isNaN(ts)) return { num: ts, str: raw };
            }
            if (raw.includes('★')) {
                return { num: (raw.match(/★/g) || []).length, str: raw };
            }
            const cleaned = raw.replace(/[%,+]/g, '');
            const num = parseFloat(cleaned);
            if (!isNaN(num)) {
                return { num: num, str: raw };
            }
            return { num: null, str: raw.toLowerCase() };
        }

        headers.forEach((th, idx) => {
            th.addEventListener('click', () => {
                const dir = (sortState.col === idx) ? -sortState.dir : 1;
                sortState = { col: idx, dir: dir };

                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort((a, b) => {
                    const av = cellSortValue(a.cells[idx]);
                    const bv = cellSortValue(b.cells[idx]);
                    if (av.num === null && av.str === '' && (bv.num !== null || bv.str !== '')) return 1;
                    if (bv.num === null && bv.str === '' && (av.num !== null || av.str !== '')) return -1;
                    if (av.num !== null && bv.num !== null) return (av.num - bv.num) * dir;
                    return av.str.localeCompare(bv.str) * dir;
                });
                rows.forEach(r => tbody.appendChild(r));

                headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                th.classList.add(dir > 0 ? 'sort-asc' : 'sort-desc');
            });
        });
    })();
    </script>
    """

    full_html = (
        iframe_styles
        + '<div class="table-wrap">'
        + html_table
        + '</div>'
        + sort_script
    )

    # Size the iframe to the table's natural height so the page scrolls
    # rather than the table. Header row ≈ 38px, body rows ≈ 28px each, plus a
    # small buffer for the border + sort handler attaching to live elements.
    row_count = len(table_df)
    iframe_height = 38 + (row_count * 28) + 20

    # Branch on the toggle — only one view is rendered. Server-side selection
    # is far more reliable than browser-side iframe hiding through Streamlit's
    # nested wrapper divs.
    if mobile_view:
        mobile_html = _build_mobile_cards_html(filtered_df)
        # Initial size assumes collapsed cards (~175px each). When the user
        # taps a card to expand it, the JS inside the iframe posts a new
        # frame-height to Streamlit's parent so the iframe grows accordingly.
        mobile_height = 60 + (len(filtered_df) * 175) + 40
        components.html(mobile_html, height=mobile_height, scrolling=False)
    else:
        components.html(full_html, height=iframe_height, scrolling=False)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.subheader("💾 Export Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Results as CSV",
        data=csv,
        file_name=f'filtered_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
