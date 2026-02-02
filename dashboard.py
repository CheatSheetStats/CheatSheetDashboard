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

# Custom CSS
st.markdown("""
<style>
.main { padding: 0rem 1rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
[data-testid="stDataFrame"] th { text-align: center !important; }
[data-testid="stDataFrame"] td { text-align: center !important; }
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

# Apply filters
filtered_df = df[df['Excel Document'].isin(selected_leagues)] if selected_leagues else df.iloc[0:0]
if start_date and end_date and 'Match Date' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['Match Date'].dt.date >= start_date) &
        (filtered_df['Match Date'].dt.date <= end_date)
    ]

# Advanced filters
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Filters")
show_strong_only = st.sidebar.checkbox("Show Strong Predictions Only")
show_matching_only = st.sidebar.checkbox("Show Model & Confidence Match Only")
show_btts_yes_only = st.sidebar.checkbox("Show BTTS Yes Only")
show_over25_yes_only = st.sidebar.checkbox("Show Over 2.5 Goals Yes Only")
show_strong_home_form = st.sidebar.checkbox("Show Strong Home Form (Form Δ +0.4 or better)")
show_strong_away_form = st.sidebar.checkbox("Show Strong Away Form (Form Δ -0.4 or better)")
show_strong_home_ppg = st.sidebar.checkbox("Show Strong Home PPG (PPG Δ +0.4 or better)")
show_strong_away_ppg = st.sidebar.checkbox("Show Strong Away PPG (PPG Δ -0.4 or better)")

# Apply advanced filters
if show_strong_only:
    filtered_df = filtered_df[filtered_df['Strong Prediction'].notna()]
if show_btts_yes_only:
    filtered_df = filtered_df[filtered_df['PredictionBTTS'] == 'Y']
if show_over25_yes_only:
    filtered_df = filtered_df[filtered_df['Over25YN'] == 'Y']
if show_strong_home_form:
    if 'Form Δ' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Form Δ'] >= 0.4]
if show_strong_away_form:
    if 'Form Δ' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Form Δ'] <= -0.4]
if show_strong_home_ppg:
    if 'PPG Δ' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['PPG Δ'] >= 0.4]
if show_strong_away_ppg:
    if 'PPG Δ' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['PPG Δ'] <= -0.4]
if show_matching_only:
    def predictions_match(row):
        model_pred = row['Model Prediction']
        conf_pick = row['Confidence Pick']
        if pd.isna(model_pred) or pd.isna(conf_pick):
            return False
        return str(model_pred).strip() == str(conf_pick).replace('(L) ', '').strip()
    filtered_df = filtered_df[filtered_df.apply(predictions_match, axis=1)]

# Sidebar metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metrics")
st.sidebar.metric("Total Fixtures", len(filtered_df))
st.sidebar.metric("Strong Predictions", filtered_df['Strong Prediction'].notna().sum())
st.sidebar.metric("BTTS Yes", (filtered_df['PredictionBTTS'] == 'Y').sum())
st.sidebar.metric("O2.5 Yes", (filtered_df['Over25YN'] == 'Y').sum())
if 'Form Δ' in filtered_df.columns:
    strong_home_form_count = (filtered_df['Form Δ'] >= 0.4).sum()
    st.sidebar.metric("Strong Home Form", strong_home_form_count)
    strong_away_form_count = (filtered_df['Form Δ'] <= -0.4).sum()
    st.sidebar.metric("Strong Away Form", strong_away_form_count)
if 'PPG Δ' in filtered_df.columns:
    strong_home_ppg_count = (filtered_df['PPG Δ'] >= 0.4).sum()
    st.sidebar.metric("Strong Home PPG", strong_home_ppg_count)
    strong_away_ppg_count = (filtered_df['PPG Δ'] <= -0.4).sum()
    st.sidebar.metric("Strong Away PPG", strong_away_ppg_count)

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
    
    # Table - compact with differences only, original PPG/Form removed
    display_columns = [
        'Match Date', 'Excel Document',
        'Home Team Rank', 'Home Team', 'Away Team', 'Away Team Rank',
        'Home Win %', 'Draw %', 'Away Win %',
        'Model Prediction', 'Confidence Pick', 'Strong Prediction',
        'PredictionBTTS', 'Over25YN', 
        'Home Clean Sheet %', 'Away Clean Sheet %',
        'PPG Δ', 'Form Δ',
        'Home xG', 'Away xG',
        'BTTS %', 'Over 2.5 Goals %'
    ]
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    table_df = filtered_df[available_columns].copy()
    
    # Rename columns
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