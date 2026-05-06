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

st.title("⚽ Football Prediction Model Dashboard")
st.markdown("---")

df = pd.read_csv(io.BytesIO(response))

df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')

# Add Over 2.5 Goals Y/N
if 'Over 2.5 Goals %' in df.columns:
    df['Over25YN'] = df['Over 2.5 Goals %'].apply(lambda x: 'Y' if x >= 50 else 'N')

# Add PPG and Form difference columns
if 'Home Team Overall Form PPG' in df.columns and 'Away Team Overall Form PPG' in df.columns:
    df['PPG Δ'] = df['Home Team Overall Form PPG'] - df['Away Team Overall Form PPG']

if 'Home Team Last 5 Form PPG' in df.columns and 'Away Team Last 5 Form PPG' in df.columns:
    df['Form Δ'] = df['Home Team Last 5 Form PPG'] - df['Away Team Last 5 Form PPG']

# Add Form Drift (REVERSED: Overall - Last 5)
if 'Home Team Last 5 Form PPG' in df.columns and 'Home Team Overall Form PPG' in df.columns:
    df['Home Form Drift'] = df['Home Team Overall Form PPG'] - df['Home Team Last 5 Form PPG']

if 'Away Team Last 5 Form PPG' in df.columns and 'Away Team Overall Form PPG' in df.columns:
    df['Away Form Drift'] = df['Away Team Overall Form PPG'] - df['Away Team Last 5 Form PPG']

# Drift difference (momentum edge)
if 'Home Form Drift' in df.columns and 'Away Form Drift' in df.columns:
    df['Drift Δ'] = df['Home Form Drift'] - df['Away Form Drift']

# Simple table view
st.subheader("Matches")

display_columns = [
    'Match Date',
    'Home Team',
    'Away Team',
    'Home Win %',
    'Draw %',
    'Away Win %',
    'PPG Δ',
    'Form Δ',
    'Home Form Drift',
    'Away Form Drift',
    'Drift Δ'
]

available_columns = [col for col in display_columns if col in df.columns]

table_df = df[available_columns].copy()

# Rename for UI
table_df.rename(columns={
    'Match Date': 'Date',
    'Home Team': 'Home',
    'Away Team': 'Away',
    'Home Win %': 'H%',
    'Draw %': 'D%',
    'Away Win %': 'A%',
    'PPG Δ': 'PPG Δ',
    'Form Δ': 'Form Δ',
    'Home Form Drift': 'H Drift',
    'Away Form Drift': 'A Drift',
    'Drift Δ': 'Drift Δ'
}, inplace=True)

# Formatting
for col in ['PPG Δ', 'Form Δ', 'H Drift', 'A Drift', 'Drift Δ']:
    if col in table_df.columns:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

if 'Date' in table_df.columns:
    table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')

st.dataframe(table_df, use_container_width=True)

# Download
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download CSV",
    data=csv,
    file_name=f'data_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv'
)
