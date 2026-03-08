import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client, Client
import os 
import io
from dotenv import load_dotenv
load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)
bucket_name = os.environ.get("BUCKET_NAME")
file_path = os.environ.get("FILE_PATH")

response = supabase_client.storage.from_(bucket_name).download(file_path)


# Page config
st.set_page_config(
    page_title="Football Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Mobile-Responsive CSS
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

.match-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 12px;
    opacity: 0.9;
}

.match-teams {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 16px 0;
    font-size: 16px;
    font-weight: bold;
}

.team-name {
    flex: 1;
    text-align: center;
}

.vs-divider {
    padding: 0 12px;
    font-size: 14px;
    opacity: 0.7;
}

.match-odds {
    display: flex;
    justify-content: space-around;
    margin: 12px 0;
    padding: 12px;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}

.odd-box {
    text-align: center;
}

.odd-label {
    font-size: 11px;
    opacity: 0.8;
    margin-bottom: 4px;
}

.odd-value {
    font-size: 16px;
    font-weight: bold;
}

.match-predictions {
    display: flex;
    justify-content: space-between;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.2);
    font-size: 13px;
}

.prediction-item {
    text-align: center;
}

.prediction-label {
    font-size: 10px;
    opacity: 0.8;
}

.prediction-value {
    font-weight: bold;
    margin-top: 4px;
}

.strong-badge {
    background: #ffd700;
    color: #333;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: bold;
}

/* Mobile responsive improvements */
@media (max-width: 768px) {
    .main { padding: 0rem 0.5rem; }
    
    /* Smaller font for table on mobile */
    [data-testid="stDataFrame"] {
        font-size: 11px !important;
    }
    
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {
        padding: 3px 5px !important;
        white-space: nowrap;
    }
    
    /* Make metrics stack better on mobile */
    [data-testid="stMetric"] {
        font-size: 0.9rem;
    }
    
    /* Reduce title size on mobile */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
}

/* Ensure table scrolls horizontally on small screens */
[data-testid="stDataFrame"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}
</style>
""", unsafe_allow_html=True)


st.title("⚽ Football Prediction Model Dashboard")
st.markdown("---")

df = pd.read_csv(io.BytesIO(response))

df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')

# Add Over 2.5 Goals Y/N
if 'Over 2.5 Goals %' in df.columns:
    df['Over25YN'] = df['Over 2.5 Goals %'].apply(lambda x: 'Y' if x >= 50 else 'N')
# Add PPG and Form difference columns
if 'Home PPG' in df.columns and 'Away PPG' in df.columns:
    df['PPG Δ'] = df['Home PPG'] - df['Away PPG']
if 'Home form PPG' in df.columns and 'Away form PPG' in df.columns:
    df['Form Δ'] = df['Home form PPG'] - df['Away form PPG']

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Mobile View Toggle
is_mobile = False  # mobile card view toggle removed

leagues = sorted(df['Excel Document'].unique())
selected_leagues = st.sidebar.multiselect("Select Leagues", leagues, default=leagues)

if 'Match Date' in df.columns:
    min_date = df['Match Date'].min().date()
    max_date = df['Match Date'].max().date()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
else:
    start_date = end_date = None

# Apply league and date filters
filtered_df = df[df['Excel Document'].isin(selected_leagues)] if selected_leagues else df.copy()
if start_date and end_date and 'Match Date' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['Match Date'].dt.date >= start_date) &
        (filtered_df['Match Date'].dt.date <= end_date)
    ]

# ── Advanced Filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Basic Filters")

show_strong_only = st.sidebar.checkbox("Show Strong Predictions Only")
show_matching_only = st.sidebar.checkbox("Show Model & Confidence Match Only")

show_btts_only = st.sidebar.checkbox("Show BTTS = Y Only")
show_over25_only = st.sidebar.checkbox("Show Over 2.5 = Y Only")
show_acca_win = st.sidebar.checkbox("Acca Win")
show_acca_btts = st.sidebar.checkbox("Acca BTTS")
show_acca_bigch = st.sidebar.checkbox("Acca Win: Big Chances edge (≥0.8)")
show_acca_touches = st.sidebar.checkbox("Acca Win: Touches edge (≥8)")
show_acca_cs = st.sidebar.checkbox("Acca Win: Clean sheet edge (≥18%)")
show_acca_win_strong56 = st.sidebar.checkbox("Acca Win STRONG (5–6 legs preset)")
show_home_team_filter = st.sidebar.checkbox("Home team (Win%>=40, Form>=0, A_GCPG - H_GPG >= 0)")
show_away_team_filter = st.sidebar.checkbox("Away team (Win%>=40, Form<=0, H_GCPG - A_GPG >= 0)")
show_btts_simple_filter = st.sidebar.checkbox("BTTS (BTTS%>=40 and GPG/GCPG>=1.2 both teams)")

# Apply advanced filters
if show_strong_only and 'Strong Prediction' in filtered_df.columns:
    # Keep rows where Strong Prediction has a team/pick value
    filtered_df = filtered_df[filtered_df['Strong Prediction'].astype(str).str.strip().ne('') & filtered_df['Strong Prediction'].notna()]

if show_matching_only and 'Model Prediction' in filtered_df.columns and 'Confidence Pick' in filtered_df.columns:
    def _norm_pick(v):
        if pd.isna(v):
            return ""
        s = str(v).strip()
        # Confidence picks sometimes have "(L) " prefix
        if s.startswith("(L)"):
            s = s.replace("(L)", "", 1).strip()
        if s.startswith("(L) "):
            s = s[4:].strip()
        return s

    model_norm = filtered_df['Model Prediction'].apply(_norm_pick)
    conf_norm = filtered_df['Confidence Pick'].apply(_norm_pick)
    filtered_df = filtered_df[model_norm.eq(conf_norm) & model_norm.ne("")]

# BTTS filter (PredictionBTTS == 'Y')
if show_btts_only and 'PredictionBTTS' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['PredictionBTTS'].astype(str).str.strip().eq('Y')]

# Over 2.5 filter (Over25YN == 'Y')
if show_over25_only and 'Over25YN' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Over25YN'].astype(str).str.strip().eq('Y')]


# Acca filters (high-confidence shortlist)
if show_acca_win:
    # Acca Win: banker-style winner picks
    required_cols = [
        'Model Prediction', 'Confidence Pick', 'Strong Prediction',
        'Home Win %', 'Draw %', 'Away Win %', 'Home xG', 'Away xG'
    ]
    if all(c in filtered_df.columns for c in required_cols):
        def _norm_pick_acca(v):
            if pd.isna(v):
                return ""
            s = str(v).strip()
            if s.startswith("(L)"):
                s = s.replace("(L)", "", 1).strip()
            if s.startswith("(L) "):
                s = s[4:].strip()
            return s

        model_pick = filtered_df['Model Prediction'].astype(str).str.strip()
        conf_pick = filtered_df['Confidence Pick'].apply(_norm_pick_acca)
        strong_pick = filtered_df['Strong Prediction'].astype(str).str.strip()

        # Must be a team pick (not Draw) and model/confidence must match
        is_team_pick = model_pick.ne("Draw") & model_pick.ne("") & filtered_df['Model Prediction'].notna()
        match_ok = model_pick.eq(conf_pick) & conf_pick.ne("")

        # Strong must be present (any non-empty)
        strong_ok = strong_pick.ne("") & filtered_df['Strong Prediction'].notna()

        # Determine whether the pick is home or away (if team columns are present)
        need_matchup_cols = ('Home Team' in filtered_df.columns and 'Away Team' in filtered_df.columns)
        if need_matchup_cols:
            is_home_pick = model_pick.eq(filtered_df['Home Team'].astype(str).str.strip())
            is_away_pick = model_pick.eq(filtered_df['Away Team'].astype(str).str.strip())
            pred_win_pct = filtered_df['Home Win %'].where(
                is_home_pick,
                filtered_df['Away Win %'].where(is_away_pick, filtered_df[['Home Win %','Away Win %']].max(axis=1))
            )
            opp_xg = filtered_df['Away xG'].where(
                is_home_pick,
                filtered_df['Home xG'].where(is_away_pick, filtered_df[['Home xG','Away xG']].min(axis=1))
            )
        else:
            pred_win_pct = filtered_df[['Home Win %','Away Win %']].max(axis=1)
            opp_xg = filtered_df[['Home xG','Away xG']].min(axis=1)
            is_home_pick = pd.Series(False, index=filtered_df.index)
            is_away_pick = pd.Series(False, index=filtered_df.index)

        # Win margin: predicted win % minus second-highest among H/D/A
        def _second_highest(row):
            vals = sorted([row['Home Win %'], row['Draw %'], row['Away Win %']], reverse=True)
            return vals[1] if len(vals) > 1 else 0.0

        second_best = filtered_df.apply(_second_highest, axis=1)
        win_margin = pred_win_pct - second_best

        # Preset: stronger settings for 5–6 leg accas
        pred_win_floor = 62.0
        win_margin_floor = 18.0
        opp_xg_cap = 1.05
        form_floor = 1.0
        if show_acca_win_strong56:
            pred_win_floor = 64.0
            win_margin_floor = 22.0
            opp_xg_cap = 1.00
            form_floor = 1.2

        # --- Existing add-ons (Big Chances adv + form sanity) from v5 (kept) ---
        # Big Chances advantage (predicted team minus opponent), if columns exist
        big_home_cols = ['Home Big Chances', 'Home Team Big Chances', 'Home Big Chances per match']
        big_away_cols = ['Away Big Chances', 'Away Team Big Chances', 'Away Big Chances per match']
        home_big_col = next((c for c in big_home_cols if c in filtered_df.columns), None)
        away_big_col = next((c for c in big_away_cols if c in filtered_df.columns), None)

        big_adv_ok = pd.Series(True, index=filtered_df.index)
        if home_big_col and away_big_col and need_matchup_cols:
            pred_big = filtered_df[home_big_col].where(is_home_pick, filtered_df[away_big_col].where(is_away_pick, pd.NA))
            opp_big  = filtered_df[away_big_col].where(is_home_pick, filtered_df[home_big_col].where(is_away_pick, pd.NA))
            big_adv = (pd.to_numeric(pred_big, errors='coerce') - pd.to_numeric(opp_big, errors='coerce'))
            big_adv_ok = big_adv.ge(0.4)

        # Optional confirmers (tick in sidebar)
        bigch_ok = pd.Series(True, index=filtered_df.index)
        if show_acca_bigch:
            if need_matchup_cols and 'Home Team Big Chances' in filtered_df.columns and 'Away Team Big Chances' in filtered_df.columns:
                pred_big = filtered_df['Home Team Big Chances'].where(is_home_pick, filtered_df['Away Team Big Chances'].where(is_away_pick, pd.NA))
                opp_big  = filtered_df['Away Team Big Chances'].where(is_home_pick, filtered_df['Home Team Big Chances'].where(is_away_pick, pd.NA))
                bigch_ok = (pd.to_numeric(pred_big, errors='coerce') - pd.to_numeric(opp_big, errors='coerce')).ge(0.8)
            else:
                bigch_ok = pd.Series(False, index=filtered_df.index)

        touches_ok = pd.Series(True, index=filtered_df.index)
        if show_acca_touches:
            if need_matchup_cols and 'Home Team Touches' in filtered_df.columns and 'Away Team Touches' in filtered_df.columns:
                pred_t = filtered_df['Home Team Touches'].where(is_home_pick, filtered_df['Away Team Touches'].where(is_away_pick, pd.NA))
                opp_t  = filtered_df['Away Team Touches'].where(is_home_pick, filtered_df['Home Team Touches'].where(is_away_pick, pd.NA))
                touches_ok = (pd.to_numeric(pred_t, errors='coerce') - pd.to_numeric(opp_t, errors='coerce')).ge(8.0)
            else:
                touches_ok = pd.Series(False, index=filtered_df.index)

        cs_ok = pd.Series(True, index=filtered_df.index)
        if show_acca_cs:
            if need_matchup_cols and 'Home Clean Sheet %' in filtered_df.columns and 'Away Clean Sheet %' in filtered_df.columns:
                pred_cs = filtered_df['Home Clean Sheet %'].where(is_home_pick, filtered_df['Away Clean Sheet %'].where(is_away_pick, pd.NA))
                opp_cs  = filtered_df['Away Clean Sheet %'].where(is_home_pick, filtered_df['Home Clean Sheet %'].where(is_away_pick, pd.NA))
                cs_ok = (pd.to_numeric(pred_cs, errors='coerce') - pd.to_numeric(opp_cs, errors='coerce')).ge(18.0)
            else:
                cs_ok = pd.Series(False, index=filtered_df.index)

        # Form gates (only applied if we can compute them)
        form_adv_ok = pd.Series(True, index=filtered_df.index)
        form_floor_ok = pd.Series(True, index=filtered_df.index)
        if need_matchup_cols and 'Form Δ' in filtered_df.columns:
            form_delta = pd.to_numeric(filtered_df['Form Δ'], errors='coerce')
            pred_form_adv = form_delta.where(is_home_pick, (-form_delta).where(is_away_pick, pd.NA))
            form_adv_ok = pred_form_adv.astype(float).ge(0.0)
        if need_matchup_cols and 'Home form PPG' in filtered_df.columns and 'Away form PPG' in filtered_df.columns:
            pred_form_ppg = pd.to_numeric(filtered_df['Home form PPG'], errors='coerce').where(
                is_home_pick,
                pd.to_numeric(filtered_df['Away form PPG'], errors='coerce').where(is_away_pick, pd.NA)
            )
            form_floor_ok = pred_form_ppg.astype(float).ge(form_floor)

        # If using the STRONG 5–6 leg preset, require at least 2 of the 3 dominance confirmers.
        dom2of3_ok = pd.Series(True, index=filtered_df.index)
        if show_acca_win_strong56:
            dom2of3_ok = (bigch_ok.astype(int) + touches_ok.astype(int) + cs_ok.astype(int)).ge(2)

        filt = (
            is_team_pick & match_ok & strong_ok
            & (pred_win_pct >= pred_win_floor)
            & (win_margin >= win_margin_floor)
            & (opp_xg <= opp_xg_cap)
            & big_adv_ok
            & bigch_ok
            & touches_ok
            & cs_ok
            & form_adv_ok
            & form_floor_ok
            & dom2of3_ok
        )
        filtered_df = filtered_df[filt]
    else:
        filtered_df = filtered_df.iloc[0:0]

if show_acca_btts:
    # Acca BTTS: high-precision BTTS legs
    required_cols = [
        'PredictionBTTS', 'Home xG', 'Away xG',
        'Home Team GPG', 'Away Team GPG', 'Home Team GCPG', 'Away Team GCPG',
        'Home Clean Sheet %', 'Away Clean Sheet %'
    ]
    if all(c in filtered_df.columns for c in required_cols):
        min_xg = filtered_df[['Home xG','Away xG']].min(axis=1)
        max_cs = pd.concat([filtered_df['Home Clean Sheet %'], filtered_df['Away Clean Sheet %']], axis=1).max(axis=1)

        filt = (
            filtered_df['PredictionBTTS'].astype(str).str.upper().str.strip().eq('Y') &
            (min_xg >= 1.20) &
            (filtered_df['Home Team GPG'] >= 1.20) & (filtered_df['Away Team GPG'] >= 1.20) &
            (filtered_df['Home Team GCPG'] >= 1.05) & (filtered_df['Away Team GCPG'] >= 1.05) &
            (max_cs <= 42.0)
        )
        filtered_df = filtered_df[filt]
    else:
        filtered_df = filtered_df.iloc[0:0]


# ── Simple team/BTTS filters ───────────────────────────────────────────────────
if show_home_team_filter:
    required_cols = ['Home Win %', 'Away Team GCPG', 'Home Team GPG']
    if all(c in filtered_df.columns for c in required_cols):
        # Form condition: prefer Form Δ (Home form PPG - Away form PPG) if present, else derive from form PPG columns.
        if 'Form Δ' in filtered_df.columns:
            home_form_ok = pd.to_numeric(filtered_df['Form Δ'], errors='coerce').ge(0.0)
        elif 'Home form PPG' in filtered_df.columns and 'Away form PPG' in filtered_df.columns:
            home_form_ok = (pd.to_numeric(filtered_df['Home form PPG'], errors='coerce') - pd.to_numeric(filtered_df['Away form PPG'], errors='coerce')).ge(0.0)
        else:
            home_form_ok = pd.Series(True, index=filtered_df.index)

        gpg_gate = (pd.to_numeric(filtered_df['Away Team GCPG'], errors='coerce') - pd.to_numeric(filtered_df['Home Team GPG'], errors='coerce')).ge(0.0)

        filt = (
            pd.to_numeric(filtered_df['Home Win %'], errors='coerce').ge(40.0) &
            home_form_ok &
            gpg_gate
        )
        filtered_df = filtered_df[filt]
    else:
        filtered_df = filtered_df.iloc[0:0]

if show_away_team_filter:
    required_cols = ['Away Win %', 'Home Team GCPG', 'Away Team GPG']
    if all(c in filtered_df.columns for c in required_cols):
        # Form condition: away should be >= home. Using Form Δ (home-away), that means Form Δ <= 0
        if 'Form Δ' in filtered_df.columns:
            away_form_ok = pd.to_numeric(filtered_df['Form Δ'], errors='coerce').le(0.0)
        elif 'Home form PPG' in filtered_df.columns and 'Away form PPG' in filtered_df.columns:
            away_form_ok = (pd.to_numeric(filtered_df['Away form PPG'], errors='coerce') - pd.to_numeric(filtered_df['Home form PPG'], errors='coerce')).ge(0.0)
        else:
            away_form_ok = pd.Series(True, index=filtered_df.index)

        gpg_gate = (pd.to_numeric(filtered_df['Home Team GCPG'], errors='coerce') - pd.to_numeric(filtered_df['Away Team GPG'], errors='coerce')).ge(0.0)

        filt = (
            pd.to_numeric(filtered_df['Away Win %'], errors='coerce').ge(40.0) &
            away_form_ok &
            gpg_gate
        )
        filtered_df = filtered_df[filt]
    else:
        filtered_df = filtered_df.iloc[0:0]

if show_btts_simple_filter:
    required_cols = ['BTTS %', 'Home Team GPG', 'Away Team GPG', 'Home Team GCPG', 'Away Team GCPG']
    if all(c in filtered_df.columns for c in required_cols):
        filt = (
            pd.to_numeric(filtered_df['BTTS %'], errors='coerce').ge(40.0) &
            pd.to_numeric(filtered_df['Home Team GPG'], errors='coerce').ge(1.2) &
            pd.to_numeric(filtered_df['Away Team GPG'], errors='coerce').ge(1.2) &
            pd.to_numeric(filtered_df['Home Team GCPG'], errors='coerce').ge(1.2) &
            pd.to_numeric(filtered_df['Away Team GCPG'], errors='coerce').ge(1.2)
        )
        filtered_df = filtered_df[filt]
    else:
        filtered_df = filtered_df.iloc[0:0]


# ── Sidebar metrics ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metrics")
st.sidebar.metric("Total Fixtures", len(filtered_df))
st.sidebar.metric("Strong Predictions", filtered_df['Strong Prediction'].notna().sum())

if len(filtered_df) == 0:
    st.warning("No fixtures match the selected filters.")
else:
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg H%", f"{filtered_df['Home Win %'].mean():.1f}%")
    col2.metric("Avg D%", f"{filtered_df['Draw %'].mean():.1f}%")
    col3.metric("Avg A%", f"{filtered_df['Away Win %'].mean():.1f}%")
    strong_count = filtered_df['Strong Prediction'].notna().sum()
    col4.metric("Strong", f"{strong_count} ({strong_count / len(filtered_df) * 100:.1f}%)")
    st.markdown("---")
    
    # Display based on view mode
    if is_mobile:
        # Mobile Card View
        st.info("📱 Card view active - optimized for mobile screens")
        
        for idx, row in filtered_df.iterrows():
            # Extract values with safe defaults
            date = row['Match Date'].strftime('%Y-%m-%d') if pd.notna(row['Match Date']) else 'TBD'
            league = row['Excel Document'] if pd.notna(row['Excel Document']) else ''
            home = row['Home Team'] if pd.notna(row['Home Team']) else 'Home'
            away = row['Away Team'] if pd.notna(row['Away Team']) else 'Away'
            h_pct = f"{row['Home Win %']:.1f}%" if pd.notna(row['Home Win %']) else '-'
            d_pct = f"{row['Draw %']:.1f}%" if pd.notna(row['Draw %']) else '-'
            a_pct = f"{row['Away Win %']:.1f}%" if pd.notna(row['Away Win %']) else '-'
            model = row['Model Prediction'] if pd.notna(row['Model Prediction']) else '-'
            btts = row['PredictionBTTS'] if pd.notna(row['PredictionBTTS']) else '-'
            o25 = row['Over25YN'] if pd.notna(row['Over25YN']) else '-'
            strong = '⭐' if pd.notna(row.get('Strong Prediction')) else ''
            
            # Create card HTML
            card_html = f"""
            <div class="match-card">
                <div class="match-header">
                    <span>{date}</span>
                    <span>{league}</span>
                </div>
                
                <div class="match-teams">
                    <div class="team-name">{home}</div>
                    <div class="vs-divider">vs</div>
                    <div class="team-name">{away}</div>
                </div>
                
                <div class="match-odds">
                    <div class="odd-box">
                        <div class="odd-label">HOME</div>
                        <div class="odd-value">{h_pct}</div>
                    </div>
                    <div class="odd-box">
                        <div class="odd-label">DRAW</div>
                        <div class="odd-value">{d_pct}</div>
                    </div>
                    <div class="odd-box">
                        <div class="odd-label">AWAY</div>
                        <div class="odd-value">{a_pct}</div>
                    </div>
                </div>
                
                <div class="match-predictions">
                    <div class="prediction-item">
                        <div class="prediction-label">MODEL</div>
                        <div class="prediction-value">{model} {strong}</div>
                    </div>
                    <div class="prediction-item">
                        <div class="prediction-label">BTTS</div>
                        <div class="prediction-value">{btts}</div>
                    </div>
                    <div class="prediction-item">
                        <div class="prediction-label">O2.5</div>
                        <div class="prediction-value">{o25}</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    
    else:
        # Desktop Table View
        display_columns = [
            'Match Date', 'Excel Document',
            'Home Team Rank', 'Home Team', 'Away Team', 'Away Team Rank',
            'Home Win %', 'Draw %', 'Away Win %',
            'Model Prediction', 'Confidence Pick', 'Strong Prediction',
            'PredictionBTTS', 'Over25YN', 
            'Home Clean Sheet %', 'Away Clean Sheet %',
            'PPG Δ', 'Form Δ',
            'Home Team GPG',
            'Away Team GPG',
            'Home Team GCPG',
            'Away Team GCPG',
            'Home form PPG',
            'Away form PPG',
            'Home xG', 'Away xG',
            'BTTS %', 'Over 2.5 Goals %'
        ]
        
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        table_df = filtered_df[available_columns].copy()
        
        table_df.rename(columns={
            'Match Date': 'Date',
            'Excel Document': 'League',
            'Home Team Rank': 'H R',
            'Home Team': 'Home',
            'Away Team': 'Away',
            'Away Team Rank': 'A R',
            'Home Win %': 'H%',
            'Draw %': 'D%',
            'Away Win %': 'A%',
            'PPG Δ': 'PPG Δ',
            'Form Δ': 'Form Δ',
            'Home form PPG': 'HF PPG',
            'Away form PPG': 'AF PPG',
            'Home Team GPG': 'H GPG',
            'Away Team GPG': 'A GPG',
            'Home Team GCPG': 'H GCPG',
            'Away Team GCPG': 'A GCPG',
            'Home xG': 'H xG',
            'Away xG': 'A xG',
            'Home Clean Sheet %': 'H CS%',
            'Away Clean Sheet %': 'A CS%',
            'PredictionBTTS': 'BTTS',
            'BTTS %': 'BTTS%',
            'Over 2.5 Goals %': 'O2.5%',
            'Over25YN': 'O2.5',
            'Model Prediction': 'Model',
            'Confidence Pick': 'Confidence',
            'Strong Prediction': 'Strong'
        }, inplace=True)

        # Formatting
        for pct_col in ['H%', 'D%', 'A%', 'BTTS%', 'O2.5%', 'H CS%', 'A CS%']:
            if pct_col in table_df.columns:
                table_df[pct_col] = table_df[pct_col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        for num_col in ['PPG Δ', 'Form Δ', 'H xG', 'A xG']:
            if num_col in table_df.columns:
                table_df[num_col] = table_df[num_col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        for rank_col in ['H R', 'A R']:
            if rank_col in table_df.columns:
                table_df[rank_col] = table_df[rank_col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        if 'Date' in table_df.columns:
            table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=1200)
    
    # Download
    st.subheader("💾 Export Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Results as CSV",
        data=csv,
        file_name=f'filtered_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
