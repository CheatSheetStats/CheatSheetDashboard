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

# Add a default 'Excel Document' column if it doesn't exist in the CSV
if 'Excel Document' not in df.columns:
    df['Excel Document'] = 'All Leagues'

# Add Over 2.5 Goals Y/N
if 'Over 2.5 Goals %' in df.columns:
    df['Over25YN'] = df['Over 2.5 Goals %'].apply(lambda x: 'Y' if x >= 50 else 'N')
# Add PPG and Form difference columns
if 'Home Team Overall Form PPG' in df.columns and 'Away Team Overall Form PPG' in df.columns:
    df['PPG Δ'] = df['Home Team Overall Form PPG'] - df['Away Team Overall Form PPG']
if 'Home Team Last 5 Form PPG' in df.columns and 'Away Team Last 5 Form PPG' in df.columns:
    df['Form Δ'] = df['Home Team Last 5 Form PPG'] - df['Away Team Last 5 Form PPG']

# ── Pre-compute favoured-team perspective columns ─────────────────────────────
# These are used by the smart filter and reflect whichever team the model favours
import numpy as np

home_fav = df['Home Win %'] >= df['Away Win %']
df['_fav_win_pct']    = np.where(home_fav, df['Home Win %'],                 df['Away Win %'])
df['_fav_season_ppg'] = np.where(home_fav, df['Home Team Overall Form PPG'], df['Away Team Overall Form PPG'])
df['_fav_last5_ppg']  = np.where(home_fav, df['Home Team Last 5 Form PPG'],  df['Away Team Last 5 Form PPG'])
df['_fav_goal_ratio'] = np.where(home_fav, df.get('Home Goal Ratio', 1.0),   df.get('Away Goal Ratio', 1.0))
df['_fav_cs']         = np.where(home_fav, df['Home Team Clean Sheet %'],     df['Away Team Clean Sheet %'])
df['_ppg_diff_abs']   = df['PPG Diff'].abs() if 'PPG Diff' in df.columns else 0

# ── Sidebar filters ───────────────────────────────────────────────────────────
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

# ── Smart filter ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Smart Filters")

st.sidebar.markdown(
    "<small>Select one or more filters — they stack together (AND logic). "
    "Based on analysis of correct vs wrong predictions across 161 real fixtures.</small>",
    unsafe_allow_html=True
)

# Each filter is a dict with:
#   label       — checkbox label shown in sidebar
#   desc        — shown in the info banner when active
#   numeric     — list of (col, op, val) tuples to apply
#   strong      — bool, filter to Strong Prediction rows only
FILTER_DEFS = [
    {
        "key":    "clean_sheet_form",
        "label":  "🟢 Clean Sheet + Form",
        "desc":   "Fav. clean sheet ≥ 30% · Last-5 PPG ≥ 1.5 · Draw% ≤ 22 · PPG gap ≥ 0.4",
        "numeric": [
            ("_fav_cs",        ">=", 30),
            ("_fav_last5_ppg", ">=", 1.5),
            ("Draw %",         "<=", 22),
            ("_ppg_diff_abs",  ">=", 0.4),
        ],
        "strong": False,
    },
    {
        "key":    "ppg_gap",
        "label":  "🔵 PPG Gap + Win% + Low Draw",
        "desc":   "PPG gap ≥ 0.6 · Win% ≥ 60 · Draw% ≤ 20",
        "numeric": [
            ("_ppg_diff_abs", ">=", 0.6),
            ("_fav_win_pct",  ">=", 60),
            ("Draw %",        "<=", 20),
        ],
        "strong": False,
    },
    {
        "key":    "in_form_quality",
        "label":  "🟡 In-Form + Quality Season",
        "desc":   "Season PPG ≥ 1.5 · Last-5 PPG ≥ 1.5 · Clean sheet ≥ 28% · Draw% ≤ 22 · PPG gap ≥ 0.3",
        "numeric": [
            ("_fav_season_ppg", ">=", 1.5),
            ("_fav_last5_ppg",  ">=", 1.5),
            ("_fav_cs",         ">=", 28),
            ("Draw %",          "<=", 22),
            ("_ppg_diff_abs",   ">=", 0.3),
        ],
        "strong": False,
    },
    {
        "key":    "strong_only",
        "label":  "⭐ Strong Predictions Only",
        "desc":   "Only fixtures where the Strong Prediction gate fired",
        "numeric": [],
        "strong": True,
    },
]

# Render as checkboxes — user can tick any combination
active_filters = []
for f in FILTER_DEFS:
    if st.sidebar.checkbox(f["label"], value=False, key=f"chk_{f['key']}"):
        active_filters.append(f)

# ── Custom sliders ─────────────────────────────────────────────────────────────
st.sidebar.markdown("**Custom thresholds:**")
with st.sidebar.expander("Set your own thresholds", expanded=False):
    custom_win_pct    = st.slider("Min win probability (%)",          0,   100, 50,  5)
    custom_draw_pct   = st.slider("Max draw probability (%)",         0,   50,  30,  1)
    custom_ppg_diff   = st.slider("Min PPG quality gap",              0.0, 2.0, 0.0, 0.1)
    custom_season_ppg = st.slider("Min favoured team season PPG",     0.0, 3.0, 0.0, 0.1)
    custom_last5_ppg  = st.slider("Min favoured team last-5 PPG",     0.0, 3.0, 0.0, 0.1)
    custom_cs         = st.slider("Min favoured team clean sheet %",  0,   70,  0,   5)
    use_custom        = st.checkbox("Apply custom thresholds", value=False)

# ── Apply league and date filters first ───────────────────────────────────────
filtered_df = df[df['Excel Document'].isin(selected_leagues)] if selected_leagues else df.copy()
if start_date and end_date and 'Match Date' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['Match Date'].dt.date >= start_date) &
        (filtered_df['Match Date'].dt.date <= end_date)
    ]

# ── Apply smart filters (stack all active ones) ───────────────────────────────
pre_filter_count    = len(filtered_df)
smart_filter_active = False
active_descs        = []

# Build a combined mask across all ticked checkboxes (AND logic)
combined_mask = pd.Series(True, index=filtered_df.index)

for f in active_filters:
    smart_filter_active = True
    active_descs.append(f["desc"])

    # Strong prediction gate
    if f["strong"]:
        combined_mask &= (
            filtered_df['Strong Prediction'].notna() &
            (filtered_df['Strong Prediction'].astype(str).str.strip() != "") &
            (filtered_df['Strong Prediction'].astype(str).str.strip() != "nan")
        )

    # Numeric criteria
    for col, op, val in f["numeric"]:
        if col not in filtered_df.columns:
            continue
        if op == ">=":
            combined_mask &= filtered_df[col] >= val
        elif op == "<=":
            combined_mask &= filtered_df[col] <= val

# Custom threshold mask
if use_custom:
    smart_filter_active = True
    active_descs.append(
        f"Win% ≥ {custom_win_pct} · Draw% ≤ {custom_draw_pct} · "
        f"PPG gap ≥ {custom_ppg_diff} · Season PPG ≥ {custom_season_ppg} · "
        f"Last-5 PPG ≥ {custom_last5_ppg} · Clean sheet ≥ {custom_cs}%"
    )
    combined_mask &= (
        (filtered_df['_fav_win_pct']    >= custom_win_pct)    &
        (filtered_df['Draw %']          <= custom_draw_pct)   &
        (filtered_df['_ppg_diff_abs']   >= custom_ppg_diff)   &
        (filtered_df['_fav_season_ppg'] >= custom_season_ppg) &
        (filtered_df['_fav_last5_ppg']  >= custom_last5_ppg)  &
        (filtered_df['_fav_cs']         >= custom_cs)
    )

if smart_filter_active:
    filtered_df = filtered_df[combined_mask]

smart_filter_desc = " · ".join(active_descs)

# ── Sidebar metrics ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metrics")
st.sidebar.metric("Fixtures shown", len(filtered_df), delta=f"{len(filtered_df) - pre_filter_count} from smart filter" if smart_filter_active else None)
st.sidebar.metric("Strong Predictions", filtered_df['Strong Prediction'].notna().sum())

if len(filtered_df) == 0:
    st.warning("No fixtures match the selected filters.")
else:

    # ── Smart filter info banner ───────────────────────────────────────────────
    if smart_filter_active and smart_filter_desc:
        st.info(f"🎯 **Smart filter active** — {smart_filter_desc}", icon=None)

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
            'Home Team Clean Sheet %', 'Away Team Clean Sheet %',
            'PPG Δ', 'Form Δ',
            'Home Team GPG',
            'Away Team GPG',
            'Home Team GCPG',
            'Away Team GCPG',
            'Home Team Overall Form PPG',
            'Away Team Overall Form PPG',
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
            'Home Team Overall Form PPG': 'HF PPG',
            'Away Team Overall Form PPG': 'AF PPG',
            'Home Team GPG': 'H GPG',
            'Away Team GPG': 'A GPG',
            'Home Team GCPG': 'H GCPG',
            'Away Team GCPG': 'A GCPG',
            'Home xG': 'H xG',
            'Away xG': 'A xG',
            'Home Team Clean Sheet %': 'H CS%',
            'Away Team Clean Sheet %': 'A CS%',
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
        for num_col in ['PPG Δ', 'Form Δ', 'H xG', 'A xG', 'H GPG', 'A GPG', 'H GCPG', 'A GCPG', 'HF PPG', 'AF PPG']:
            if num_col in table_df.columns:
                table_df[num_col] = table_df[num_col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
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
