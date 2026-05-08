import streamlit as st
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
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)


st.title("⚽ Football Prediction Model Dashboard")
st.caption("Showing Model v5 predictions")
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
st.sidebar.header("🔍 Filters")

leagues = sorted(df['Excel Document'].dropna().unique().tolist())
selected_leagues = st.sidebar.multiselect("Select Leagues", leagues, default=leagues)

if df['Match Date'].notna().any():
    min_date = df['Match Date'].min().date()
    max_date = df['Match Date'].max().date()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
else:
    start_date = end_date = None


# ──────────────────────────────────────────────────────────────────────────────
# Smart filters — REBUILT for v5 signals
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Smart Filters")

st.sidebar.markdown(
    "<small>Tick one or more — they stack with AND logic. "
    "These filters use v5's predictive signals (xG, confidence margin, rank gap).</small>",
    unsafe_allow_html=True
)

FILTER_DEFS = [
    {
        "key":   "strong_only",
        "label": "⭐ Strong Predictions Only",
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
        "label": "🔵 Rank Gap (Top vs Bottom)",
        "desc":  "League rank gap ≥ 10",
        "numeric": [("_rank_gap", ">=", 10)],
        "strong":  False,
    },
    {
        "key":   "all_in",
        "label": "💪 All-In (Strong + High Conf)",
        "desc":  "Strong Prediction AND win-margin ≥ 25pp — most conservative",
        "numeric": [("_win_margin", ">=", 25)],
        "strong":  True,
    },
]

active_filters = []
for f in FILTER_DEFS:
    if st.sidebar.checkbox(f["label"], value=False, key=f"chk_{f['key']}"):
        active_filters.append(f)


# Custom sliders
st.sidebar.markdown("**Custom thresholds:**")
with st.sidebar.expander("Set your own thresholds", expanded=False):
    custom_win_pct    = st.slider("Min win probability (%)", 0, 100, 50, 5)
    custom_draw_pct   = st.slider("Max draw probability (%)", 0,  50, 30, 1)
    custom_margin     = st.slider("Min confidence margin (pp)", 0, 50, 0, 1)
    custom_xg_gap     = st.slider("Min match xG gap", 0.0, 3.0, 0.0, 0.1)
    custom_rank_gap   = st.slider("Min league rank gap", 0, 24, 0, 1)
    use_custom        = st.checkbox("Apply custom thresholds", value=False)


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
        f"Margin ≥ {custom_margin}pp · xG gap ≥ {custom_xg_gap} · Rank gap ≥ {custom_rank_gap}"
    )
    combined_mask &= (
        (filtered_df['_fav_win_pct'] >= custom_win_pct) &
        (filtered_df['Draw %']       <= custom_draw_pct) &
        (filtered_df['_win_margin']  >= custom_margin) &
        (filtered_df['_xg_match_gap']>= custom_xg_gap) &
        (filtered_df['_rank_gap'].fillna(0) >= custom_rank_gap)
    )

if smart_filter_active:
    filtered_df = filtered_df[combined_mask]

smart_filter_desc = " · ".join(active_descs)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar metrics
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metrics")
st.sidebar.metric(
    "Fixtures shown", len(filtered_df),
    delta=f"{len(filtered_df) - pre_filter_count} from smart filter" if smart_filter_active else None
)
st.sidebar.metric("Strong Predictions", int(filtered_df['Strong Prediction'].notna().sum()))


# ──────────────────────────────────────────────────────────────────────────────
# Main view
# ──────────────────────────────────────────────────────────────────────────────
if len(filtered_df) == 0:
    st.warning("No fixtures match the selected filters.")
else:
    if smart_filter_active and smart_filter_desc:
        st.info(f"🎯 **Smart filter active** — {smart_filter_desc}")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg H%", f"{filtered_df['Home Win %'].mean():.1f}%")
    col2.metric("Avg D%", f"{filtered_df['Draw %'].mean():.1f}%")
    col3.metric("Avg A%", f"{filtered_df['Away Win %'].mean():.1f}%")
    strong_count = int(filtered_df['Strong Prediction'].notna().sum())
    col4.metric("Strong", f"{strong_count} ({strong_count / len(filtered_df) * 100:.1f}%)")
    st.markdown("---")

    # Desktop table view (mobile card view path preserved if you re-enable it)
    display_columns = [
        'Match Date', 'Excel Document',
        'Home Team Rank', 'Home Team', 'Away Team', 'Away Team Rank',
        'Home Win %', 'Draw %', 'Away Win %',
        'Model Prediction', 'Strong Prediction', 'Confidence Score',
        'PredictionBTTS', 'Over25YN',
        'Home xG', 'Away xG',
        'Home xG Diff PM', 'Away xG Diff PM',
        'Home Defence XG Over', 'Away Defence XG Over',
        'Home Team GPG', 'Away Team GPG',
        'Home Team GCPG', 'Away Team GCPG',
        'BTTS %', 'Over 2.5 Goals %',
        'League Home Adv (PPG)', 'Style Edge',
    ]

    available_columns = [c for c in display_columns if c in filtered_df.columns]
    table_df = filtered_df[available_columns].copy()

    table_df.rename(columns={
        'Match Date':            'Date',
        'Excel Document':        'League',
        'Home Team Rank':        'H R',
        'Home Team':             'Home',
        'Away Team':             'Away',
        'Away Team Rank':        'A R',
        'Home Win %':            'H%',
        'Draw %':                'D%',
        'Away Win %':            'A%',
        'Home xG':               'H xG',
        'Away xG':               'A xG',
        'Home xG Diff PM':       'H xGΔ',
        'Away xG Diff PM':       'A xGΔ',
        'Home Defence XG Over':  'H DefOver',
        'Away Defence XG Over':  'A DefOver',
        'Home Team GPG':         'H GPG',
        'Away Team GPG':         'A GPG',
        'Home Team GCPG':        'H GCPG',
        'Away Team GCPG':        'A GCPG',
        'PredictionBTTS':        'BTTS',
        'BTTS %':                'BTTS%',
        'Over 2.5 Goals %':      'O2.5%',
        'Over25YN':              'O2.5',
        'Model Prediction':      'Model',
        'Confidence Score':      'Conf',
        'Strong Prediction':     'Strong',
        'League Home Adv (PPG)': 'Home Adv',
        'Style Edge':            'Style',
    }, inplace=True)

    # Formatting
    pct_cols = ['H%', 'D%', 'A%', 'BTTS%', 'O2.5%']
    for c in pct_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

    one_dp_cols = ['H xG', 'A xG', 'H xGΔ', 'A xGΔ', 'H DefOver', 'A DefOver',
                   'H GPG', 'A GPG', 'H GCPG', 'A GCPG', 'Conf', 'Home Adv']
    for c in one_dp_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    if 'Style' in table_df.columns:
        table_df['Style'] = table_df['Style'].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")

    rank_cols = ['H R', 'A R']
    for c in rank_cols:
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")

    if 'Date' in table_df.columns:
        table_df['Date'] = table_df['Date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "-"
        )

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=1200)

    # Export
    st.subheader("💾 Export Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Results as CSV",
        data=csv,
        file_name=f'filtered_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
