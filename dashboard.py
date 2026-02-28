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
is_mobile = st.sidebar.checkbox("📱 Mobile Card View", value=False, help="Display matches as cards optimized for mobile screens")

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
filtered_df = df[df['Excel Document'].isin(selected_leagues)] if selected_leagues else df.iloc[0:0]
if start_date and end_date and 'Match Date' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['Match Date'].dt.date >= start_date) &
        (filtered_df['Match Date'].dt.date <= end_date)
    ]

# ── Advanced Filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Filters")

show_strong_only = st.sidebar.checkbox("Show Strong Predictions Only")
show_matching_only = st.sidebar.checkbox("Show Model & Confidence Match Only")

show_btts_yes_only = st.sidebar.checkbox("Show BTTS Advanced Filter")
if show_btts_yes_only:
    st.sidebar.caption("BTTS% ≥ 65% | Total xG ≥ 3.2 | Both GPG ≥ 1.3 | Both GCPG ≥ 1.2 | O2.5% ≥ 70%")

show_btts_advanced_lean = st.sidebar.checkbox("Show BTTS Advanced Filter (Lean)")
if show_btts_advanced_lean:
    st.sidebar.caption("Matches 4 of 5: BTTS% ≥ 65% | Total xG ≥ 3.2 | Both GPG ≥ 1.3 | Both GCPG ≥ 1.2 | O2.5% ≥ 70%")

show_btts_strict = st.sidebar.checkbox("🔒 BTTS Strict Filter")
if show_btts_strict:
    st.sidebar.caption("Model BTTS = Y | BTTS% ≥ 70% | Both Form GCPG ≥ 1.0 — 88.9% hit rate across 18 games")

# ── Home Team Filters ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🏠 Home Team Filters")

show_home_base = st.sidebar.checkbox("🏠 Home Base Filter")
if show_home_base:
    st.sidebar.caption("H% ≥ 58% | H CS% ≥ 25% | PPG Δ ≥ 0.5 | Form Δ ≥ 0.4 | H GPG ≥ 1.4 | H GCPG ≤ 1.4")

show_home_strict = st.sidebar.checkbox("🏠 Home Strict Filter")
if show_home_strict:
    st.sidebar.caption("H% ≥ 65% | H CS% ≥ 35% | PPG Δ ≥ 1.0 | Form Δ ≥ 1.0 | H GPG ≥ 1.6 | H GCPG ≤ 1.0")

# ── Away Team Filters ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("✈️ Away Team Filters")

show_away_base = st.sidebar.checkbox("✈️ Away Base Filter")
if show_away_base:
    st.sidebar.caption("A% ≥ 50% | A CS% ≥ 28% | PPG Δ ≤ -0.5 | Form Δ ≤ -0.5 | A GPG ≥ 1.5 | A GCPG ≤ 1.3")

show_away_strict = st.sidebar.checkbox("✈️ Away Strict Filter")
if show_away_strict:
    st.sidebar.caption("A% ≥ 58% | A CS% ≥ 28% | PPG Δ ≤ -0.5 | Form Δ ≤ -0.5 | A GPG ≥ 1.6 | A GCPG ≤ 1.2")

# ── Value Filters ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("💎 Value Filters")

show_home_value = st.sidebar.checkbox("🏠 Home Value Filter")
if show_home_value:
    st.sidebar.caption("Form Δ ≥ 1.0 | Away form PPG ≤ 1.2 — Excludes Base/Strict | 63.6% hit rate across 22 games")

show_away_value = st.sidebar.checkbox("✈️ Away Value Filter")
if show_away_value:
    st.sidebar.caption("Away Form Δ ≥ 1.0 | H form PPG ≤ 1.0 | A form GPG ≥ 1.4 — Excludes Base/Strict | 41.7% hit rate across 24 games")

# ── Apply all filters ─────────────────────────────────────────────────────────

if show_strong_only:
    filtered_df = filtered_df[filtered_df['Strong Prediction'].notna()]

if show_matching_only:
    def predictions_match(row):
        model_pred = row['Model Prediction']
        conf_pick = row['Confidence Pick']
        if pd.isna(model_pred) or pd.isna(conf_pick):
            return False
        return str(model_pred).strip() == str(conf_pick).replace('(L) ', '').strip()
    filtered_df = filtered_df[filtered_df.apply(predictions_match, axis=1)]

if show_btts_yes_only:
    btts_filter = pd.Series([True] * len(filtered_df), index=filtered_df.index)
    if 'BTTS %' in filtered_df.columns:
        btts_filter = btts_filter & (filtered_df['BTTS %'] >= 65)
    if 'Home xG' in filtered_df.columns and 'Away xG' in filtered_df.columns:
        btts_filter = btts_filter & ((filtered_df['Home xG'] + filtered_df['Away xG']) >= 3.2)
    if 'Home Team GPG' in filtered_df.columns and 'Away Team GPG' in filtered_df.columns:
        btts_filter = btts_filter & (filtered_df['Home Team GPG'] >= 1.3) & (filtered_df['Away Team GPG'] >= 1.3)
    if 'Home Team GCPG' in filtered_df.columns and 'Away Team GCPG' in filtered_df.columns:
        btts_filter = btts_filter & (filtered_df['Home Team GCPG'] >= 1.2) & (filtered_df['Away Team GCPG'] >= 1.2)
    if 'Over 2.5 Goals %' in filtered_df.columns:
        btts_filter = btts_filter & (filtered_df['Over 2.5 Goals %'] >= 70)
    filtered_df = filtered_df[btts_filter]

if show_btts_advanced_lean:
    def btts_advanced_lean_criteria(row):
        criteria_met = 0
        if 'BTTS %' in row.index and pd.notna(row['BTTS %']) and row['BTTS %'] >= 65:
            criteria_met += 1
        if 'Home xG' in row.index and 'Away xG' in row.index and pd.notna(row['Home xG']) and pd.notna(row['Away xG']) and (row['Home xG'] + row['Away xG']) >= 3.2:
            criteria_met += 1
        if 'Home Team GPG' in row.index and 'Away Team GPG' in row.index and pd.notna(row['Home Team GPG']) and pd.notna(row['Away Team GPG']) and row['Home Team GPG'] >= 1.3 and row['Away Team GPG'] >= 1.3:
            criteria_met += 1
        if 'Home Team GCPG' in row.index and 'Away Team GCPG' in row.index and pd.notna(row['Home Team GCPG']) and pd.notna(row['Away Team GCPG']) and row['Home Team GCPG'] >= 1.2 and row['Away Team GCPG'] >= 1.2:
            criteria_met += 1
        if 'Over 2.5 Goals %' in row.index and pd.notna(row['Over 2.5 Goals %']) and row['Over 2.5 Goals %'] >= 70:
            criteria_met += 1
        return criteria_met >= 4
    filtered_df = filtered_df[filtered_df.apply(btts_advanced_lean_criteria, axis=1)]


if show_btts_strict:
    required_cols = ['PredictionBTTS', 'BTTS %', 'Home form GCPG', 'Away form GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        filtered_df = filtered_df[
            (filtered_df['PredictionBTTS'] == 'Y') &
            (filtered_df['BTTS %'] >= 70) &
            (filtered_df['Home form GCPG'] >= 1.0) &
            (filtered_df['Away form GCPG'] >= 1.0)
        ]

if show_home_base:
    required_cols = ['Home Win %', 'Home Clean Sheet %', 'PPG Δ', 'Form Δ', 'Home Team GPG', 'Home Team GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        filtered_df = filtered_df[
            (filtered_df['Home Win %'] >= 58) &
            (filtered_df['Home Clean Sheet %'] >= 25) &
            (filtered_df['PPG Δ'] >= 0.5) &
            (filtered_df['Form Δ'] >= 0.4) &
            (filtered_df['Home Team GPG'] >= 1.4) &
            (filtered_df['Home Team GCPG'] <= 1.4)
        ]

if show_home_strict:
    required_cols = ['Home Win %', 'Home Clean Sheet %', 'PPG Δ', 'Form Δ', 'Home Team GPG', 'Home Team GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        filtered_df = filtered_df[
            (filtered_df['Home Win %'] >= 65) &
            (filtered_df['Home Clean Sheet %'] >= 35) &
            (filtered_df['PPG Δ'] >= 1.0) &
            (filtered_df['Form Δ'] >= 1.0) &
            (filtered_df['Home Team GPG'] >= 1.6) &
            (filtered_df['Home Team GCPG'] <= 1.0)
        ]

if show_away_base:
    required_cols = ['Away Win %', 'Away Clean Sheet %', 'PPG Δ', 'Form Δ', 'Away Team GPG', 'Away Team GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        filtered_df = filtered_df[
            (filtered_df['Away Win %'] >= 50) &
            (filtered_df['Away Clean Sheet %'] >= 28) &
            (filtered_df['PPG Δ'] <= -0.5) &
            (filtered_df['Form Δ'] <= -0.5) &
            (filtered_df['Away Team GPG'] >= 1.5) &
            (filtered_df['Away Team GCPG'] <= 1.3)
        ]

if show_away_strict:
    required_cols = ['Away Win %', 'Away Clean Sheet %', 'PPG Δ', 'Form Δ', 'Away Team GPG', 'Away Team GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        filtered_df = filtered_df[
            (filtered_df['Away Win %'] >= 58) &
            (filtered_df['Away Clean Sheet %'] >= 28) &
            (filtered_df['PPG Δ'] <= -0.5) &
            (filtered_df['Form Δ'] <= -0.5) &
            (filtered_df['Away Team GPG'] >= 1.6) &
            (filtered_df['Away Team GCPG'] <= 1.2)
        ]

if show_home_value:
    required_cols = ['Home form PPG', 'Away form PPG', 'Home Win %', 'Home Clean Sheet %',
                     'PPG Δ', 'Form Δ', 'Home Team GPG', 'Home Team GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        home_base_excl = (
            (filtered_df['Home Win %'] >= 58) &
            (filtered_df['Home Clean Sheet %'] >= 25) &
            (filtered_df['PPG Δ'] >= 0.5) &
            (filtered_df['Form Δ'] >= 0.4) &
            (filtered_df['Home Team GPG'] >= 1.4) &
            (filtered_df['Home Team GCPG'] <= 1.4)
        )
        home_strict_excl = (
            (filtered_df['Home Win %'] >= 65) &
            (filtered_df['Home Clean Sheet %'] >= 35) &
            (filtered_df['PPG Δ'] >= 1.0) &
            (filtered_df['Form Δ'] >= 1.0) &
            (filtered_df['Home Team GPG'] >= 1.6) &
            (filtered_df['Home Team GCPG'] <= 1.0)
        )
        filtered_df = filtered_df[
            (filtered_df['Form Δ'] >= 1.0) &
            (filtered_df['Away form PPG'] <= 1.2) &
            ~home_base_excl &
            ~home_strict_excl
        ]

if show_away_value:
    required_cols = ['Home form PPG', 'Away form PPG', 'Away form GPG', 'Away Win %',
                     'Away Clean Sheet %', 'PPG Δ', 'Form Δ', 'Away Team GPG', 'Away Team GCPG']
    if all(col in filtered_df.columns for col in required_cols):
        away_base_excl = (
            (filtered_df['Away Win %'] >= 50) &
            (filtered_df['Away Clean Sheet %'] >= 28) &
            (filtered_df['PPG Δ'] <= -0.5) &
            (filtered_df['Form Δ'] <= -0.5) &
            (filtered_df['Away Team GPG'] >= 1.5) &
            (filtered_df['Away Team GCPG'] <= 1.3)
        )
        away_strict_excl = (
            (filtered_df['Away Win %'] >= 58) &
            (filtered_df['Away Clean Sheet %'] >= 28) &
            (filtered_df['PPG Δ'] <= -0.5) &
            (filtered_df['Form Δ'] <= -0.5) &
            (filtered_df['Away Team GPG'] >= 1.6) &
            (filtered_df['Away Team GCPG'] <= 1.2)
        )
        filtered_df = filtered_df[
            ((filtered_df['Away form PPG'] - filtered_df['Home form PPG']) >= 1.0) &
            (filtered_df['Home form PPG'] <= 1.0) &
            (filtered_df['Away form GPG'] >= 1.4) &
            ~away_base_excl &
            ~away_strict_excl
        ]

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