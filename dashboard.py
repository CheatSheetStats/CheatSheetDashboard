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


st.markdown(
    '<div style="display: flex; align-items: baseline; gap: 12px; margin: 0 0 4px 0;">'
    '<h1 style="margin: 0; font-size: 1.6rem;">⚽ Football Prediction Model</h1>'
    '<span style="color: #888; font-size: 0.85rem;">Model v5 predictions</span>'
    '</div>',
    unsafe_allow_html=True,
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
    {
        "key":   "strong_only",
        "label": "⭐ Strong Predictions",
        "desc":  "Only fixtures where the Strong Prediction gate fired",
        "numeric": [],
        "strong":  True,
    },
    {
        "key":   "high_conf",
        "label": "🎯 High Confidence",
        "desc":  "Win-margin (top minus second prob) ≥ 20pp",
        "numeric": [("_win_margin", ">=", 20)],
        "strong":  False,
    },
    {
        "key":   "xg_dominance",
        "label": "🟢 xG Dominance",
        "desc":  "Match xG gap ≥ 0.6 (one team much stronger)",
        "numeric": [("_xg_match_gap", ">=", 0.6)],
        "strong":  False,
    },
    {
        "key":   "rank_gap",
        "label": "🔵 Rank Gap",
        "desc":  "League rank gap ≥ 10",
        "numeric": [("_rank_gap", ">=", 10)],
        "strong":  False,
    },
    {
        "key":   "acca_quality",
        "label": "🎲 Acca Quality",
        "desc":  "Favourite wins ≥ 50% of season AND underdog loses ≥ 40% of season",
        "numeric": [("_fav_season_win", ">=", 50), ("_dog_season_lose", ">=", 40)],
        "strong":  False,
    },
    {
        "key":   "confident_btts",
        "label": "🥅 Confident BTTS Y",
        "desc":  "BTTS% ≥ 60% (absolute, league-independent)",
        "numeric": [("BTTS %", ">=", 60)],
        "strong":  False,
    },
    {
        "key":   "confident_o25",
        "label": "⚽ Confident Over 2.5",
        "desc":  "Over 2.5% ≥ 60% (absolute, league-independent)",
        "numeric": [("Over 2.5 Goals %", ">=", 60)],
        "strong":  False,
    },
    {
        "key":   "all_in",
        "label": "💪 All-In",
        "desc":  "Strong Prediction AND win-margin ≥ 25pp — most conservative",
        "numeric": [("_win_margin", ">=", 25)],
        "strong":  True,
    },
]

# Render the checkbox grid + custom-thresholds expander inside the left column
with left_col:
    # 2 rows of 4 checkboxes — fits well in the wider left column
    active_filters = []
    n_per_row = 4
    for row_start in range(0, len(FILTER_DEFS), n_per_row):
        row_filters = FILTER_DEFS[row_start:row_start + n_per_row]
        cols = st.columns(n_per_row)
        for col, f in zip(cols, row_filters):
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
    if f["strong"]:
        combined_mask &= filtered_df['Strong Prediction'].notna()
    for col, op, val in f["numeric"]:
        if col not in filtered_df.columns:
            continue
        if op == ">=":
            combined_mask &= filtered_df[col] >= val
        elif op == "<=":
            combined_mask &= filtered_df[col] <= val

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

if smart_filter_active:
    filtered_df = filtered_df[combined_mask]

smart_filter_desc = " · ".join(active_descs)


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Main view
# ──────────────────────────────────────────────────────────────────────────────
if len(filtered_df) == 0:
    st.warning("No fixtures match the selected filters.")
else:
    if smart_filter_active and smart_filter_desc:
        st.info(f"🎯 **Smart filter active** — {smart_filter_desc}")

    # Compact metrics strip — five figures on one tight line instead of five
    # large st.metric cards.
    strong_count = int(filtered_df['Strong Prediction'].notna().sum())
    delta_text = ""
    if smart_filter_active:
        delta = len(filtered_df) - pre_filter_count
        delta_text = f' <span style="color:#888;">({delta:+d} after smart filter)</span>'

    st.markdown(
        '<div style="display: flex; gap: 28px; padding: 4px 0 6px 0; font-size: 0.9rem; '
        'border-bottom: 1px solid #2a2d36; margin-bottom: 8px;">'
        f'<span><span style="color:#888;">Fixtures</span> '
        f'<strong>{len(filtered_df)}</strong>{delta_text}</span>'
        f'<span><span style="color:#888;">Avg H%</span> '
        f'<strong>{filtered_df["Home Win %"].mean():.1f}%</strong></span>'
        f'<span><span style="color:#888;">Avg D%</span> '
        f'<strong>{filtered_df["Draw %"].mean():.1f}%</strong></span>'
        f'<span><span style="color:#888;">Avg A%</span> '
        f'<strong>{filtered_df["Away Win %"].mean():.1f}%</strong></span>'
        f'<span><span style="color:#888;">Strong</span> '
        f'<strong>{strong_count}</strong> '
        f'<span style="color:#888;">({strong_count / len(filtered_df) * 100:.1f}%)</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Star confidence mapping ────────────────────────────────────────────────
    # Confidence Score is the probability margin (top minus second)
    def stars_from_margin(m):
        if pd.isna(m):       return "-"
        if m < 5:            return "★"
        if m < 10:           return "★★"
        if m < 20:           return "★★★"
        if m < 35:           return "★★★★"
        return "★★★★★"

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
    pct_cols = ['H%', 'D%', 'A%', 'H Win%', 'A Win%', 'H Lose%', 'A Lose%',
                'BTTS%', 'O2.5%']
    for c in pct_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

    two_dp_cols = ['H GPG', 'A GPG', 'H GCPG', 'A GCPG',
                   'H PPG', 'A PPG', 'H Form', 'A Form',
                   'H @Home', 'A @Away']
    for c in two_dp_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    # Form drift gets a +/- sign so direction is obvious at a glance
    for c in ['H Δ', 'A Δ', 'H @H Δ', 'A @A Δ']:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")

    rank_cols = ['H Rank', 'A Rank']
    for c in rank_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")

    if 'Date' in table_df.columns:
        table_df['Date'] = table_df['Date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "-"
        )

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

    sort_script = """
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
