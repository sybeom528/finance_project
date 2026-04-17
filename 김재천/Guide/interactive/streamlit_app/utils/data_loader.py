"""Final Project 데이터 로더 (Streamlit 캐싱 포함)."""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import streamlit as st


GUIDE_DIR = Path(__file__).resolve().parents[3]  # Guide/
DATA_DIR = GUIDE_DIR / 'data'
IMG_DIR = GUIDE_DIR / 'images'
DOCS_DIR = GUIDE_DIR / 'docs'
QR_DIR = GUIDE_DIR / 'quick_reference'


# ============================================================
# 데이터 파일 로더
# ============================================================

@st.cache_data
def load_prices():
    return pd.read_csv(DATA_DIR / 'portfolio_prices.csv',
                        parse_dates=['Date'], index_col='Date')


@st.cache_data
def load_alerts():
    return pd.read_csv(DATA_DIR / 'alert_signals.csv',
                        parse_dates=['Date'], index_col='Date')


@st.cache_data
def load_profiles():
    return pd.read_csv(DATA_DIR / 'profiles.csv')


@st.cache_data
def load_regime():
    return pd.read_csv(DATA_DIR / 'regime_history.csv',
                        parse_dates=['Date'], index_col='Date')


@st.cache_data
def load_metrics():
    return pd.read_csv(DATA_DIR / 'step9_metrics.csv', index_col=0)


@st.cache_data
def load_final_recommendation():
    return pd.read_csv(DATA_DIR / 'step10_final_recommendation.csv', index_col=0)


@st.cache_resource
def load_step9_results():
    with open(DATA_DIR / 'step9_backtest_results.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_step11_weights():
    with open(DATA_DIR / 'step11_top10_weights.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_regime_cov():
    with open(DATA_DIR / 'regime_covariance_by_window.pkl', 'rb') as f:
        return pickle.load(f)


# ============================================================
# 상수
# ============================================================

PORT_TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
                 'XLK', 'XLF', 'XLE', 'XLV', 'VOX',
                 'XLY', 'XLP', 'XLI', 'XLU', 'XLRE', 'XLB',
                 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'PG', 'XOM',
                 'TLT', 'AGG', 'SHY', 'TIP', 'GLD', 'DBC']

EQUITY = PORT_TICKERS[:24]
BOND = PORT_TICKERS[24:28]
GOLD = PORT_TICKERS[28:30]

EQUITY_IDX = list(range(24))
BOND_IDX = list(range(24, 28))
GOLD_IDX = list(range(28, 30))

EQUITY_CUT = {
    '보수형': {0: 0.00, 1: 0.15, 2: 0.35, 3: 0.60},
    '중립형': {0: 0.00, 1: 0.10, 2: 0.25, 3: 0.50},
    '적극형': {0: 0.00, 1: 0.05, 2: 0.15, 3: 0.35},
    '공격형': {0: 0.00, 1: 0.00, 2: 0.10, 3: 0.25},
}

ALERT_CONFIG_MAP = {
    'ALERT_A': 'alert_a',
    'ALERT_B': 'alert_b',
    'ALERT_C': 'alert_c',
    'ALERT_D': 'alert_d',
}

PROFILE_COLORS = {
    '보수형': '#2ca02c', '중립형': '#1f77b4',
    '적극형': '#ff7f0e', '공격형': '#d62728'
}

MODE_COLORS = {'M0': '#808080', 'M1': '#1f77b4', 'M2': '#ff7f0e', 'M3': '#d62728'}

ALERT_COLORS = {0: '#2ca02c', 1: '#ffcc00', 2: '#ff7f0e', 3: '#d62728'}

HISTORICAL_CRISES = {
    '2018 Volmageddon': ('2018-02-05', '2018-02-16'),
    '2020 COVID': ('2020-02-20', '2020-04-30'),
    '2022 Q1 긴축': ('2022-01-01', '2022-03-31'),
    '2022 Q2 인플레 쇼크': ('2022-04-01', '2022-06-30'),
    '2023 SVB': ('2023-03-09', '2023-03-31'),
    '2024 엔캐리 청산': ('2024-08-01', '2024-08-15'),
}


def parse_key(key):
    """M1_보수형_ALERT_B → (M1, 보수형, ALERT_B)"""
    mode, rest = key.split('_', 1)
    profile, config = rest.split('_', 1)
    return mode, profile, config
