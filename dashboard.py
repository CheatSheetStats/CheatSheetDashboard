import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client
import os
import io
from dotenv import load_dotenv

load_dotenv()

# ===============================
# Supabase setup
# ===============================
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
bucket_name = os.environ.get("BUCKET_NAME")
file_path = os.environ.get("FILE_PATH")

supabase_client = create_client(supabase_url, supabase_key)
response = supabase_client.storage.from_(bucket_name).download(file_path)

# ===============================
# Page config & styling
# ===============================
st.set_page_config(
    page_title="Football Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { padding: 0rem 1rem; }
[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Load data
# ===============================
df = pd.read_csv(io.BytesIO(response))
df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')

# ===============================
# Derived columns
# ===============================
if {'Home PPG', 'Away PPG'}.issubset(df.columns):
    df['PPG Δ'] = df['Home PPG'] - df['Away PPG']

if {'Home form PPG', 'Away form PPG'}.issubset(df.columns):
    df['Form Δ'] = df['Home form PPG'] - df['Away form PPG']

if 'Over 2.5 Goals %'.issubset(df.columns):
    df['Over25YN'] = df['Over 2.5 Goals %'].apply(lambda x: 'Y' if x >= 50 else 'N')

# ===============================
# BTTS signal count (core logic)
# ===============================
df['BTTS_signal_count'] = (
    (df['Home xG'] >= 1.3).astype(int) +
    (df['Away xG'] >= 1.1).astype(int) +
    (df['Home Clean Sheet %'] <= 25).astype(int) +
    (df['Away Clean Sheet %'] <= 25).astype(int)
)

df['BTTS Tier'] = df['BTTS_signal_count'].map({
    4: 'STRONG',
    3: 'MEDIUM'
}).fillna('')

# ===============================
# Sidebar – base filters
# ===============================
st.sidebar.header("🔍 Filters")

leagues = sorted(df['Excel Document'].unique())
selected_leagues = st.sidebar.multiselect("Select Leagues", leagues, default=leagues)

min_date = df['Match Date'].min().date()
max_date = df['Match Date'].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("From", min_date)
with col2:
    end_date = st.date_input("To", max_date)

filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['Excel Document'].isin(selected_leagues)]
filtered_df = filtered_df[
    (filtered_df['Match Date'].dt.date >= start_date) &
    (filtered_df['Match Date'].dt.date <= end_date)
]

# ===============================
# Sidebar – advanced filters
# ===============================
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Match Winner Strength")

home_edge = st.sidebar.checkbox("Strong Home Edge (Δ ≥ +0.8)")
away_edge = st.sidebar.checkbox("Strong Away Edge (Δ ≤ -0.8)")

st.sidebar.subheader("⚽ BTTS Confidence")
btts_strong = st.sidebar.checkbox("BTTS Strong (4/4 signals)")
btts_medium = st.sidebar.checkbox("BTTS Medium (3/4 signals)")

st.sidebar.subheader("🔥 Goals")
over25_strong = st.sidebar.checkbox("Strong Over 2.5 (≥ 58%)")

st.sidebar.subheader("🚫 Avoid Games")
avoid_xg_imbalance = st.sidebar.checkbox("Avoid xG Imbalance (≥ 0.8)")

# ===============================
# Apply winner filters
# ===============================
if home_edge:
    filtered_df = filtered_df[
        (filtered_df['PPG Δ'] >= 0.8) | (filtered_df['Form Δ'] >= 0.8)
    ]

if away_edge:
    filtered_df = filtered_df[
        (filtered_df['PPG Δ'] <= -0.8) | (filtered_df['Form Δ'] <= -0.8)
    ]

# ===============================
# Apply BTTS filters
# ===============================
if btts_strong:
    filtered_df = filtered_df[filtered_df['BTTS_signal_count'] == 4]

if btts_medium:
    filtered_df = filtered_df[
        (filtered_df['BTTS_signal_count'] == 3) &
        (filtered_df['Home Clean Sheet %'] < 35) &
        (filtered_df['Away Clean Sheet %'] < 35)
    ]

# ===============================
# Apply goals / trap filters
# ===============================
if over25_strong:
    filtered_df = filtered_df[filtered_df['Over 2.5 Goals %'] >= 58]

if avoid_xg_imbalance:
    filtered_df = filtered_df[
        (filtered_df['Home xG'] - filtered_df['Away xG']).abs() < 0.8
    ]

# ===============================
# Sidebar metrics
# ===============================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Metrics")
st.sidebar.metric("Total Fixtures", len(filtered_df))
st.sidebar.metric("BTTS Strong", (filtered_df['BTTS Tier'] == 'STRONG').sum())
st.sidebar.metric("BTTS Medium", (filtered_df['BTTS Tier'] == 'MEDIUM').sum())

# ===============================
# Main table
# ===============================
st.title("⚽ Football Prediction Model Dashboard")
st.markdown("---")

if filtered_df.empty:
    st.warning("No fixtures match the selected filters.")
else:
    display_columns = [
        'Match Date', 'Excel Document',
        'Home Team', 'Away Team',
        'Model Prediction', 'Confidence Pick',
        'PPG Δ', 'Form Δ',
        'Home xG', 'Away xG',
        'Home Clean Sheet %', 'Away Clean Sheet %',
        'BTTS Tier', 'Over 2.5 Goals %'
    ]

    table_df = filtered_df[display_columns].copy()
    table_df.rename(columns={
        'Match Date': 'Date',
        'Excel Document': 'League',
        'Home Team': 'Home',
        'Away Team': 'Away',
        'Model Prediction': 'Model',
        'Confidence Pick': 'Confidence',
        'Home Clean Sheet %': 'H CS%',
        'Away Clean Sheet %': 'A CS%',
        'Over 2.5 Goals %': 'O2.5%'
    }, inplace=True)

    table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')

    for col in ['PPG Δ', 'Form Δ', 'Home xG', 'Away xG']:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")

    for col in ['H CS%', 'A CS%', 'O2.5%']:
        table_df[col] = table_df[col].map(lambda x: f"{x:.1f}%")

    st.dataframe(table_df, use_container_width=True, height=1200)

    # ===============================
    # Export
    # ===============================
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Filtered Results",
        data=csv,
        file_name=f"filtered_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
