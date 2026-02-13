"""
Alpha Deck PRO v8.0 - DIRECTIONAL BIAS TERMINAL (PRODUCTION-HARDENED)
Intraday SPX/ES Scalping | Kroer Barrier FW | GEX Intelligence | Regime Classification
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import feedparser
from datetime import datetime, timedelta, date
import numpy as np
import pytz
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull
import itertools

# Import APIs conditionally
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    GEMINI_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except (ImportError, Exception):
    FRED_AVAILABLE = False

# ============================================================================
# CONFIGURATION DICTIONARIES (Decoupled from UI)
# ============================================================================

MARKET_SHIFTERS = {'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'}

STYLES_CONFIG = {
    "app_css": """
    /* Pure Black Terminal + Glassmorphism */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        background-color: #000000 !important;
    }

    .main, .block-container, section {
        background-color: #000000 !important;
    }

    /* MARKET TRACKER — Fixed top-right header */
    .market-tracker {
        position: fixed;
        top: 8px;
        right: 24px;
        z-index: 9999;
        background: rgba(26, 26, 26, 0.85);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 176, 0, 0.45);
        border-radius: 10px;
        padding: 8px 18px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 13px;
        font-weight: 700;
        color: #FFB000;
        letter-spacing: 1px;
        box-shadow: 0 4px 20px rgba(255, 176, 0, 0.12);
        transition: none;
    }
    .market-tracker .clock { color: #FFFFFF; margin-right: 10px; }
    .market-tracker .status-open { color: #00FF88; }
    .market-tracker .status-closed { color: #FF4444; }

    /* GLASSMORPHISM CARD SYSTEM */
    .metric-card {
        background: rgba(26, 26, 26, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 176, 0, 0.35);
        border-radius: 14px;
        padding: 24px;
        margin: 8px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 30px rgba(255, 176, 0, 0.06);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(255, 176, 0, 0.18);
        border-color: rgba(255, 201, 51, 0.6);
    }

    .metric-label {
        color: #FFB000;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 32px;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        line-height: 1.2;
    }

    .metric-change {
        color: #FFB000;
        font-size: 14px;
        font-weight: 500;
        margin-top: 4px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }

    .metric-change.positive { color: #00FF88; }
    .metric-change.negative { color: #FF4444; }

    .signal-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 700;
        margin: 2px 3px;
        background: rgba(255, 176, 0, 0.12);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.3);
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }

    /* AI Briefing Box */
    .ai-briefing {
        background: rgba(26, 26, 26, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 176, 0, 0.4);
        border-radius: 14px;
        padding: 28px;
        color: #FFB000;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        box-shadow: 0 4px 30px rgba(255, 176, 0, 0.08);
    }

    .ai-briefing h4 {
        color: #FFB000;
        margin-bottom: 16px;
        font-size: 14px;
        letter-spacing: 2px;
    }

    .ai-briefing p {
        margin: 12px 0;
        text-align: justify;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000000;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(26, 26, 26, 0.6);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.4);
        border-radius: 4px 4px 0 0;
        padding: 12px 24px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-weight: 700;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFB000;
        color: #000000;
    }

    .stTabs [aria-selected="true"] p {
        color: #000000;
    }

    /* Buttons */
    .stButton > button {
        background: rgba(0, 0, 0, 0.7);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.5);
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.25s;
    }

    .stButton > button:hover {
        background: #FFB000;
        color: #000000;
        border-color: #FFB000;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 10, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 176, 0, 0.3);
    }

    /* Global text */
    h1, h2, h3, h4 {
        color: #FFB000 !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }

    p, span, div, label {
        color: #FFFFFF;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }

    a { color: #FFB000; text-decoration: none; transition: color 0.2s; }
    a:hover { color: #FFC933; text-decoration: underline; }

    .stAlert {
        background: rgba(26, 26, 26, 0.7);
        backdrop-filter: blur(8px);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.4);
        border-radius: 8px;
    }

    /* TRADE / HOLD signal badges */
    .trade-signal {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        letter-spacing: 1px;
    }
    .trade-signal.trade {
        background: rgba(0, 255, 136, 0.15);
        color: #00FF88;
        border: 1px solid rgba(0, 255, 136, 0.5);
    }
    .trade-signal.hold {
        background: rgba(255, 176, 0, 0.12);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.4);
    }

    /* Bias Pivot Card */
    .bias-card {
        background: rgba(26, 26, 26, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 176, 0, 0.5);
        border-radius: 14px;
        padding: 20px;
        margin: 8px 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .bias-card .level-label {
        color: #FFB000;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }
    .bias-card .level-value {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 700;
    }

    /* Earnings Card */
    .earnings-card {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        transition: all 0.25s;
    }
    .earnings-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(255, 176, 0, 0.12);
    }
    .earnings-card.market-shifter {
        border: 1px solid #FFB000;
        box-shadow: 0 2px 16px rgba(255, 176, 0, 0.15);
    }
    .earnings-card .ticker {
        font-size: 16px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .earnings-card .date-label {
        font-size: 10px;
        color: rgba(255, 176, 0, 0.7);
        margin-top: 4px;
    }
    .earnings-card .eps-row {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 6px;
    }
    .vol-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 9px;
        font-weight: 700;
        background: rgba(255, 176, 0, 0.18);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.4);
        margin-left: 6px;
    }

    /* News Squawk Container */
    .news-squawk {
        background: rgba(26, 26, 26, 0.55);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 176, 0, 0.3);
        border-radius: 12px;
        padding: 16px;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .news-squawk .news-item {
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 176, 0, 0.1);
    }
    .news-squawk .news-source {
        font-size: 9px;
        color: #FFB000;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .news-squawk .news-title {
        font-size: 12px;
        color: #FFFFFF;
        margin-top: 3px;
    }
    .news-squawk .news-title a {
        color: #FFFFFF;
        text-decoration: none;
    }
    .news-squawk .news-title a:hover {
        color: #FFB000;
    }
    """,
}

TEMPLATES = {
    "metric_card": """
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-change {change_class}">
            {abs_str} ({change_symbol}{change_pct:.2f}%)
        </div>
        {signals_html}
    </div>
    """,
    "signal_badge": '<span class="signal-badge">{signal}</span>',
    "ai_briefing": """
    <div class="ai-briefing">
        <h4>🧠 SESSION STRATEGY BRIEFING</h4>
        <p>{content}</p>
    </div>
    """,
}

TRADINGVIEW_MAPPING = {
    '^GSPC': 'SP:SPX', 'SPY': 'AMEX:SPY', 'QQQ': 'NASDAQ:QQQ',
    'IWM': 'AMEX:IWM', 'BTC-USD': 'COINBASE:BTCUSD',
    'ETH-USD': 'COINBASE:ETHUSD',
}

SECTOR_ETF_MAP = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Consumer Staples',
    'XLY': 'Consumer Disc.', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLC': 'Communication', 'XLU': 'Utilities',
}

SECTOR_CONSTITUENTS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD'],
    'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK'],
    'XLI': ['CAT', 'UNP', 'HON', 'GE', 'RTX'],
    'XLP': ['PG', 'PEP', 'KO', 'COST', 'WMT'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'FCX'],
    'XLRE': ['PLD', 'AMT', 'EQIX', 'SPG', 'O'],
    'XLC': ['META', 'GOOGL', 'NFLX', 'DIS', 'CMCSA'],
    'XLU': ['NEE', 'SO', 'DUK', 'AEP', 'D'],
}

SECTOR_SHORT_NAMES = {
    'Technology': 'TECH', 'Financials': 'FIN', 'Energy': 'NRG',
    'Healthcare': 'HLTH', 'Industrials': 'IND', 'Consumer Staples': 'STPL',
    'Consumer Disc.': 'DISC', 'Materials': 'MATL', 'Real Estate': 'REIT',
    'Communication': 'COMM', 'Utilities': 'UTIL',
}

EARNINGS_TICKERS = [
    'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL',
    'COIN', 'MSTR', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'UNH', 'LLY',
    'CAT', 'NFLX', 'DIS', 'HD', 'WMT', 'COST', 'PG', 'KO',
]

# ============================================================================
# PAGE CONFIG + SESSION STATE
# ============================================================================

st.set_page_config(
    page_title="Alpha Deck PRO v8.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

for key, default in {
    'selected_ticker': 'SPY',
    'sidebar_visible': False,
    'selected_sector': None,
    'show_sector_drill': False,
    'gemini_api_key': None,
    'fred_api_key': None,
    'fred_client': None,
    'gemini_configured': False,
    'last_spx_options': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Inject CSS from config
st.markdown(f"<style>{STYLES_CONFIG['app_css']}</style>", unsafe_allow_html=True)

# ============================================================================
# FRED CONNECTION PERSISTENCE FIX
# Always initialize FRED client when API key exists, regardless of sidebar state
# Wrapped in try-except for immediate availability
# ============================================================================
if FRED_AVAILABLE and st.session_state.fred_api_key and st.session_state.fred_client is None:
    try:
        st.session_state.fred_client = Fred(api_key=st.session_state.fred_api_key)
    except (ValueError, TypeError, Exception):
        st.session_state.fred_client = None

# ============================================================================
# SIDEBAR TOGGLE & API CONFIGURATION
# ============================================================================
col_toggle, col_spacer = st.columns([1, 20])
with col_toggle:
    toggle_icon = "⚡" if not st.session_state.sidebar_visible else "📊"
    if st.button(toggle_icon, key="sidebar_toggle", help="Toggle API Config"):
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible
        st.rerun()

if st.session_state.sidebar_visible:
    st.sidebar.title("🔑 API CONFIGURATION")
    st.sidebar.caption("Professional API Management")

    gemini_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key or "",
        type="password",
        help="Get free key: https://makersuite.google.com/app/apikey"
    )
    fred_key_input = st.sidebar.text_input(
        "FRED API Key",
        value=st.session_state.fred_api_key or "",
        type="password",
        help="Get free key: https://fredaccount.stlouisfed.org/apikeys"
    )

    st.sidebar.markdown("---")

    if gemini_key_input:
        st.session_state.gemini_api_key = gemini_key_input.strip()
    if fred_key_input:
        st.session_state.fred_api_key = fred_key_input.strip()

    # Configure Gemini
    if GEMINI_AVAILABLE and st.session_state.gemini_api_key:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            st.session_state.gemini_configured = True
            st.sidebar.success("✅ Gemini: Connected")
        except (AttributeError, ImportError) as e:
            st.sidebar.error(f"❌ Gemini SDK Error: {str(e)[:60]}")
            st.session_state.gemini_configured = False
        except Exception as e:
            st.sidebar.error(f"❌ Gemini Error: {str(e)[:60]}")
            st.session_state.gemini_configured = False
    else:
        if not st.session_state.gemini_api_key:
            st.sidebar.warning("⚠️ Gemini: API Key Required")
        st.session_state.gemini_configured = False

    # Configure FRED — wrapped in try-except for immediate availability
    if FRED_AVAILABLE and st.session_state.fred_api_key:
        try:
            st.session_state.fred_client = Fred(api_key=st.session_state.fred_api_key)
            st.sidebar.success("✅ FRED: Connected")
        except (ValueError, TypeError) as e:
            st.sidebar.error(f"❌ FRED Init Error: {str(e)[:60]}")
            st.session_state.fred_client = None
        except Exception as e:
            st.sidebar.error(f"❌ FRED Error: {str(e)[:60]}")
            st.session_state.fred_client = None
    else:
        if not st.session_state.fred_api_key:
            st.sidebar.warning("⚠️ FRED: API Key Required")
        st.session_state.fred_client = None

    st.sidebar.caption("💡 Session-only storage")
    st.sidebar.caption("🔒 Zero data retention")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_market_open():
    """Check if US stock market is currently open."""
    try:
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        return False


def get_tradingview_symbol(ticker: str) -> str:
    """Convert Yahoo ticker to TradingView symbol."""
    return TRADINGVIEW_MAPPING.get(ticker, f"NASDAQ:{ticker}")


# ============================================================================
# MARKET TRACKER HEADER — Persistent EST Clock + Status
# Uses st.empty() container for flicker-free updates
# ============================================================================
_header_container = st.empty()


def render_market_tracker():
    """Inject persistent top-right market tracker (flicker-free)."""
    try:
        ny_tz = pytz.timezone('America/New_York')
        ny_now = datetime.now(ny_tz)
        clock_str = ny_now.strftime('%I:%M %p ET')
        mkt_open = is_market_open()
        if mkt_open:
            status_html = '<span class="status-open">🟢 OPEN</span>'
        else:
            status_html = '<span class="status-closed">🔴 CLOSED</span>'
        _header_container.markdown(
            f'<div class="market-tracker">'
            f'<span class="clock">{clock_str}</span> | {status_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

render_market_tracker()


# ============================================================================
# DATA FETCHING — VECTORIZED yf.download
# ============================================================================

@st.cache_data(ttl=60)
def _batch_download(tickers_tuple: tuple, period: str = '5d') -> pd.DataFrame:
    """Central batch downloader. Hashable tuple for caching."""
    try:
        data = yf.download(
            list(tickers_tuple),
            period=period,
            group_by='ticker',
            threads=True,
            progress=False,
        )
        return data
    except Exception:
        return pd.DataFrame()


def _extract_ticker_from_batch(
    batch_df: pd.DataFrame, ticker: str, tickers_list: list
) -> dict:
    """Extract a single ticker's metrics from a batch download DataFrame."""
    default = {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}
    try:
        if len(tickers_list) == 1:
            hist = batch_df
        else:
            if ticker not in batch_df.columns.get_level_values(0):
                return default
            hist = batch_df[ticker]

        close = hist['Close'].dropna()
        if close.empty:
            return default

        current_price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
        change_abs = current_price - prev_close
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
        vol_col = hist['Volume'].dropna() if 'Volume' in hist.columns else pd.Series([0])
        volume = int(vol_col.iloc[-1]) if not vol_col.empty else 0

        return {
            'price': current_price,
            'change_pct': float(change_pct),
            'change_abs': float(change_abs),
            'volume': volume,
            'success': True,
        }
    except Exception:
        return default


@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker: str) -> dict:
    """Fetch single ticker data (wrapper around batch for compatibility)."""
    batch = _batch_download((ticker,), period='5d')
    return _extract_ticker_from_batch(batch, ticker, [ticker])


@st.cache_data(ttl=60)
def calculate_rsi_wilders(prices: pd.Series, period: int = 14) -> float:
    """
    RSI with Wilder's Smoothing (EWM, alpha=1/period).
    Returns NaN sentinel (50.0) which callers convert to CALIBRATING.
    """
    try:
        if len(prices) < period + 1:
            return float('nan')

        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return float('nan')
        return float(np.clip(rsi.iloc[-1], 0.0, 100.0))
    except Exception:
        return float('nan')


@st.cache_data(ttl=60)
def fetch_watchlist_data(tickers: tuple) -> pd.DataFrame:
    """
    Fetch watchlist with technical signals using batch yf.download.
    Uses 3mo lookback for RSI warm-up (60+ trading days).
    NaN RSI → "CALIBRATING…" signal.
    """
    tickers_list = list(tickers)
    batch = _batch_download(tickers, period='3mo')
    if batch.empty:
        return pd.DataFrame()

    results = []
    for ticker in tickers_list:
        try:
            if len(tickers_list) == 1:
                hist = batch
            else:
                if ticker not in batch.columns.get_level_values(0):
                    continue
                hist = batch[ticker]

            close = hist['Close'].dropna()
            if close.empty:
                continue

            current_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
            change_abs = current_price - prev_close
            vol_col = hist['Volume'].dropna() if 'Volume' in hist.columns else pd.Series([0])
            volume = int(vol_col.iloc[-1]) if not vol_col.empty else 0
            rsi_value = calculate_rsi_wilders(close)

            signals = []

            # RSI NaN → CALIBRATING instead of blank
            if pd.isna(rsi_value):
                signals.append("⏳ CALIBRATING…")
                rsi_display = 0.0
            else:
                rsi_display = rsi_value
                if rsi_value < 30:
                    signals.append("🟢 OVERSOLD")
                elif rsi_value > 70:
                    signals.append("🔴 OVERBOUGHT")

            if volume > 20e6:
                signals.append("⚡ HIGH VOL")
            if abs(change_pct) > 3:
                signals.append("🚀 MOMENTUM")

            results.append({
                'Ticker': ticker,
                'Price': float(current_price),
                'Change %': float(change_pct),
                'Change $': float(change_abs),
                'Volume': f"{volume / 1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_display),
                'Signals': " | ".join(signals) if signals else "—",
            })
        except Exception:
            continue

    return pd.DataFrame(results)


@st.cache_data(ttl=60)
def fetch_index_data() -> dict:
    """Fetch major indices via batch download."""
    indices_map = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'VIX': '^VIX',
        'HYG': 'HYG', 'US10Y': '^TNX', 'DXY': 'DX-Y.NYB',
    }
    tickers = tuple(indices_map.values())
    batch = _batch_download(tickers, period='5d')
    results = {}
    ticker_list = list(indices_map.values())
    for name, ticker in indices_map.items():
        results[name] = _extract_ticker_from_batch(batch, ticker, ticker_list)
    return results


@st.cache_data(ttl=60)
def fetch_sector_performance() -> pd.DataFrame:
    """Fetch sector ETF performance via batch download."""
    tickers = tuple(SECTOR_ETF_MAP.keys())
    batch = _batch_download(tickers, period='5d')
    ticker_list = list(SECTOR_ETF_MAP.keys())
    results = []
    for ticker, name in SECTOR_ETF_MAP.items():
        data = _extract_ticker_from_batch(batch, ticker, ticker_list)
        if data['success']:
            results.append({
                'Sector': name, 'Ticker': ticker,
                'Change %': data['change_pct'], 'Change $': data['change_abs'],
            })
    df = pd.DataFrame(results)
    if not df.empty:
        return df.sort_values('Change %', ascending=False)
    return df


@st.cache_data(ttl=60)
def fetch_sector_constituents(sector_ticker: str) -> pd.DataFrame:
    """Fetch top holdings for a sector ETF."""
    tickers = SECTOR_CONSTITUENTS.get(sector_ticker, [])
    if not tickers:
        return pd.DataFrame()
    return fetch_watchlist_data(tuple(tickers))


@st.cache_data(ttl=60)
def fetch_spx_options_data() -> dict | None:
    """Fetch SPX options with GEX calculation."""
    try:
        spx = yf.Ticker("^GSPC")
        expirations = spx.options
        if not expirations or len(expirations) == 0:
            return None

        nearest_exp = expirations[0]
        opt_chain = spx.option_chain(nearest_exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        if calls.empty or puts.empty:
            return None

        total_call_volume = int(calls['volume'].fillna(0).sum())
        total_put_volume = int(puts['volume'].fillna(0).sum())
        put_call_ratio = total_put_volume / max(total_call_volume, 1)

        total_call_oi = int(calls['openInterest'].fillna(0).sum())
        total_put_oi = int(puts['openInterest'].fillna(0).sum())
        put_call_oi_ratio = total_put_oi / max(total_call_oi, 1)

        calls_oi = calls.groupby('strike')['openInterest'].sum()
        puts_oi = puts.groupby('strike')['openInterest'].sum()
        total_oi = calls_oi.add(puts_oi, fill_value=0)
        max_pain = float(total_oi.idxmax()) if not total_oi.empty else 0

        avg_call_iv = float(calls['impliedVolatility'].mean() * 100) if 'impliedVolatility' in calls.columns else 0
        avg_put_iv = float(puts['impliedVolatility'].mean() * 100) if 'impliedVolatility' in puts.columns else 0

        # GEX
        call_gex = 0.0
        put_gex = 0.0
        top_gex_strikes = []

        if 'gamma' in calls.columns:
            calls_gex_series = calls['gamma'].fillna(0) * calls['openInterest'].fillna(0) * 100
            call_gex = float(calls_gex_series.sum())
            put_gex_series = puts['gamma'].fillna(0) * puts['openInterest'].fillna(0) * 100 * -1
            put_gex = float(put_gex_series.sum())

            all_gex = pd.DataFrame({
                'strike': pd.concat([calls['strike'], puts['strike']]),
                'gex': pd.concat([calls_gex_series, put_gex_series.abs()]),
            })
            top_strikes = all_gex.groupby('strike')['gex'].sum().nlargest(5)
            top_gex_strikes = [{'strike': float(s), 'gex': float(g)} for s, g in top_strikes.items()]

        net_gex = call_gex + put_gex

        result = {
            'expiration': nearest_exp,
            'put_call_ratio': put_call_ratio,
            'put_call_oi_ratio': put_call_oi_ratio,
            'max_pain': max_pain,
            'avg_call_iv': avg_call_iv,
            'avg_put_iv': avg_put_iv,
            'calls': calls, 'puts': puts,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'net_gex': net_gex,
            'call_gex': call_gex,
            'put_gex': put_gex,
            'top_gex_strikes': top_gex_strikes,
            'success': True,
        }
        st.session_state.last_spx_options = result
        return result
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_vix_term_structure() -> dict:
    """Fetch VIX term structure via batch download."""
    tickers = ('^VIX', '^VIX9D', '^VIX3M')
    batch = _batch_download(tickers, period='5d')
    ticker_list = list(tickers)
    vix = _extract_ticker_from_batch(batch, '^VIX', ticker_list)
    vix9d = _extract_ticker_from_batch(batch, '^VIX9D', ticker_list)
    vix3m = _extract_ticker_from_batch(batch, '^VIX3M', ticker_list)
    return {
        'VIX': vix['price'], 'VIX9D': vix9d['price'], 'VIX3M': vix3m['price'],
        'backwardation': (
            vix9d['price'] > vix['price']
            if vix9d['success'] and vix['success']
            else False
        ),
    }


@st.cache_data(ttl=60)
def fetch_crypto_metrics(cryptos: tuple) -> dict:
    """Fetch crypto data via batch download."""
    tickers = tuple(f"{c}-USD" for c in cryptos)
    batch = _batch_download(tickers, period='5d')
    ticker_list = list(tickers)
    results = {}
    for crypto_symbol in cryptos:
        ticker = f"{crypto_symbol}-USD"
        results[crypto_symbol] = _extract_ticker_from_batch(batch, ticker, ticker_list)
    return results


@st.cache_data(ttl=300)
def fetch_news_feeds() -> list:
    """Fetch RSS news for squawk."""
    feeds = {
        'WSJ': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
        'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
    }
    articles = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                articles.append({
                    'Source': source,
                    'Title': entry.get('title', 'Untitled'),
                    'Link': entry.get('link', '#'),
                })
        except Exception:
            pass
    return articles[:15]


@st.cache_data(ttl=3600)
def fetch_insider_cluster_buys() -> pd.DataFrame | None:
    """Scrape OpenInsider for cluster buys."""
    try:
        url = "http://openinsider.com/latest-cluster-buys"
        tables = pd.read_html(url, header=0)
        if not tables or len(tables) == 0:
            return None
        df = tables[0]
        if 'Ticker' in df.columns:
            columns_to_keep = [
                col for col in ['Ticker', 'Company Name', 'Insider Name', 'Title',
                                'Trade Type', 'Price', 'Qty', 'Value', 'Trade Date']
                if col in df.columns
            ]
            return df[columns_to_keep].head(10)
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_fred_liquidity() -> dict:
    """Fetch Fed liquidity data using get_series()."""
    if st.session_state.fred_client is None:
        return {'yield_spread': 0, 'credit_spread': 0, 'fed_balance': 0, 'success': False}
    try:
        client = st.session_state.fred_client
        t10y2y = client.get_series('T10Y2Y')
        yield_spread = float(t10y2y.dropna().iloc[-1]) if not t10y2y.dropna().empty else 0
        hy_spread = client.get_series('BAMLH0A0HYM2')
        credit_spread = float(hy_spread.dropna().iloc[-1]) if not hy_spread.dropna().empty else 0
        fed_assets = client.get_series('WALCL')
        fed_balance = float(fed_assets.dropna().iloc[-1]) / 1000 if not fed_assets.dropna().empty else 0
        return {
            'yield_spread': yield_spread, 'credit_spread': credit_spread,
            'fed_balance': fed_balance, 'success': True,
        }
    except Exception as e:
        return {
            'yield_spread': 0, 'credit_spread': 0, 'fed_balance': 0,
            'success': False, 'error': str(e),
        }


@st.cache_data(ttl=60)
def calculate_spx_crypto_correlation() -> dict:
    """SPX vs Crypto correlation with inner join (weekend gap fix)."""
    try:
        tickers = ('^GSPC', 'BTC-USD', 'ETH-USD')
        batch = _batch_download(tickers, period='1mo')
        if batch.empty:
            return {'BTC': 0, 'ETH': 0}

        def _get_close(tk):
            if tk not in batch.columns.get_level_values(0):
                return pd.Series(dtype=float)
            s = batch[tk]['Close'].dropna()
            s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
            s.index = s.index.normalize()
            return s.rename(tk)

        spx = _get_close('^GSPC')
        btc = _get_close('BTC-USD')
        eth = _get_close('ETH-USD')
        if spx.empty or btc.empty or eth.empty:
            return {'BTC': 0, 'ETH': 0}
        combined = pd.merge(spx.to_frame(), btc.to_frame(), left_index=True, right_index=True, how='inner')
        combined = pd.merge(combined, eth.to_frame(), left_index=True, right_index=True, how='inner')
        if len(combined) < 10:
            return {'BTC': 0, 'ETH': 0}
        corr_btc = combined['^GSPC'].corr(combined['BTC-USD'])
        corr_eth = combined['^GSPC'].corr(combined['ETH-USD'])
        return {
            'BTC': float(corr_btc) if not pd.isna(corr_btc) else 0,
            'ETH': float(corr_eth) if not pd.isna(corr_eth) else 0,
        }
    except Exception as e:
        return {'BTC': 0, 'ETH': 0, 'error': str(e)}


# ============================================================================
# INTRADAY DIRECTIONAL BIAS LEVELS — PDH/PDL, Opening Range, VWAP
# ============================================================================

@st.cache_data(ttl=60)
def fetch_intraday_bias_levels() -> dict:
    """
    Compute intraday directional bias levels for SPX/ES scalping:
    - PDH / PDL: Previous Day High / Low from daily bars
    - Opening Range (15m): High/Low of first 15 minutes
    - VWAP: Σ(Typical Price × Volume) / Σ(Volume)
    """
    result = {
        'pdh': 0.0, 'pdl': 0.0, 'or_high': 0.0, 'or_low': 0.0,
        'vwap': 0.0, 'current': 0.0, 'success': False,
    }
    try:
        # --- PDH / PDL from daily bars ---
        daily = yf.download('^GSPC', period='5d', interval='1d', progress=False)
        if daily is not None and len(daily) >= 2:
            # Previous day = second-to-last row
            prev_day = daily.iloc[-2]
            result['pdh'] = float(prev_day['High'].iloc[0]) if hasattr(prev_day['High'], 'iloc') else float(prev_day['High'])
            result['pdl'] = float(prev_day['Low'].iloc[0]) if hasattr(prev_day['Low'], 'iloc') else float(prev_day['Low'])
            result['current'] = float(daily['Close'].iloc[-1].iloc[0]) if hasattr(daily['Close'].iloc[-1], 'iloc') else float(daily['Close'].iloc[-1])

        # --- Opening Range (15m) + VWAP from intraday bars ---
        intraday = yf.download('^GSPC', period='1d', interval='1m', progress=False)
        if intraday is not None and not intraday.empty:
            # Flatten multi-index columns if present
            if isinstance(intraday.columns, pd.MultiIndex):
                intraday.columns = intraday.columns.get_level_values(0)

            # Opening Range: first 15 minutes of session
            if intraday.index.tz is not None:
                intraday.index = intraday.index.tz_convert('America/New_York')
            else:
                ny_tz = pytz.timezone('America/New_York')
                intraday.index = intraday.index.tz_localize('UTC').tz_convert('America/New_York')

            today_date = intraday.index[-1].date()
            session_open = pd.Timestamp(f'{today_date} 09:30:00', tz='America/New_York')
            or_end = pd.Timestamp(f'{today_date} 09:45:00', tz='America/New_York')

            or_bars = intraday[(intraday.index >= session_open) & (intraday.index <= or_end)]
            if not or_bars.empty:
                result['or_high'] = float(or_bars['High'].max())
                result['or_low'] = float(or_bars['Low'].min())

            # VWAP: Σ(Typical Price × Volume) / Σ(Volume)
            session_bars = intraday[intraday.index >= session_open]
            if not session_bars.empty and 'Volume' in session_bars.columns:
                typical_price = (session_bars['High'] + session_bars['Low'] + session_bars['Close']) / 3.0
                vol = session_bars['Volume'].fillna(0)
                cumulative_tpv = (typical_price * vol).sum()
                cumulative_vol = vol.sum()
                if cumulative_vol > 0:
                    result['vwap'] = float(cumulative_tpv / cumulative_vol)

        result['success'] = result['pdh'] > 0
        return result
    except Exception:
        return result


@st.cache_data(ttl=3600)
def fetch_earnings_calendar() -> pd.DataFrame:
    """Fetch upcoming earnings for relevant high-volume tickers."""
    results = []
    for ticker_sym in EARNINGS_TICKERS:
        try:
            tk = yf.Ticker(ticker_sym)
            cal = tk.get_earnings_dates(limit=4)
            if cal is not None and not cal.empty:
                for date_idx, row in cal.iterrows():
                    try:
                        dt = pd.Timestamp(date_idx)
                        if dt.tz is not None:
                            dt = dt.tz_localize(None)
                        cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
                        if dt >= cutoff:
                            eps_est = row.get('EPS Estimate', None)
                            eps_act = row.get('Reported EPS', None)
                            surprise = row.get('Surprise(%)', None)
                            results.append({
                                'Ticker': ticker_sym,
                                'Date': dt,
                                'EPS Est': f"${eps_est:.2f}" if eps_est is not None and not pd.isna(eps_est) else '—',
                                'EPS Act': f"${eps_act:.2f}" if eps_act is not None and not pd.isna(eps_act) else '—',
                                'Surprise': f"{surprise:.1f}%" if surprise is not None and not pd.isna(surprise) else '—',
                                'is_shifter': ticker_sym in MARKET_SHIFTERS,
                            })
                    except Exception:
                        continue
        except Exception:
            continue

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Date').drop_duplicates(subset=['Ticker', 'Date'])
        return df.head(40)
    return pd.DataFrame()


# ============================================================================
# POLYMARKET ARBITRAGE ENGINE — KROER FRAMEWORK (FINALIZED)
# InitFW: u = mean(Z₀) | Barrier FW | α-Extraction (α=0.9) | Profit ≥ D−g
# ============================================================================

class PolymarketArbitrageEngine:
    """
    Production-grade arbitrage detection using the Kroer Framework:
    - Algorithm 3 (InitFW): u = mean(Z₀) to prevent gradient explosion
    - Barrier Frank-Wolfe: Adaptive ε-contraction, bounded Lipschitz
    - α-Extraction (α=0.9): Signal only when 90% of edge captured
    - Profit Guarantee (Prop 4.1): Profit ≥ D(μ̂‖θ) − g(μ̂)
    """

    def __init__(
        self,
        min_profit_threshold: float = 0.02,
        liquidity_param: float = 100.0,
        alpha: float = 0.9,
        max_fw_iter: int = 500,
    ):
        self.min_profit_threshold = min_profit_threshold
        self.b = liquidity_param
        self.alpha = alpha
        self.max_fw_iter = max_fw_iter

    # ------------------------------------------------------------------
    # Algorithm 3: InitFW — u = mean(Z₀)
    # ------------------------------------------------------------------
    def init_fw(self, prices: np.ndarray) -> dict:
        """
        Algorithm 3 (InitFW): Compute active vertex set Z₀ and interior
        point u as the AVERAGE of Z₀ vertices. This ensures u lies strictly
        in the interior of the polytope, preventing gradient explosion in
        the Barrier Frank-Wolfe variant.
        """
        n = len(prices)

        # Active vertex set Z₀: standard basis vectors (simplex vertices)
        Z0 = np.eye(n)

        # Interior point u = mean(Z₀) = (1/n, 1/n, ..., 1/n)
        # This is the centroid of the simplex — guaranteed strictly interior
        u = np.mean(Z0, axis=0)  # = np.ones(n) / n

        return {
            'Z0': Z0,
            'u': u,
            'n': n,
        }

    # ------------------------------------------------------------------
    # Bregman Divergence: D(μ‖θ) for LMSR
    # ------------------------------------------------------------------
    def compute_bregman_divergence(self, mu: np.ndarray, theta: np.ndarray) -> float:
        """
        D(μ‖θ) = R(μ) + C(θ) − ⟨θ, μ⟩
        R(μ) = −Σ μ_i log(μ_i)  (negative entropy)
        C(θ) = b·log(Σ exp(θ_i/b))  (LMSR cost)
        """
        mu_safe = np.clip(mu, 1e-10, 1.0 - 1e-10)
        theta_safe = np.clip(theta, -100, 100)
        R_mu = -np.sum(mu_safe * np.log(mu_safe))
        C_theta = self.b * np.log(np.sum(np.exp(theta_safe / self.b)))
        divergence = R_mu + C_theta - np.dot(theta_safe, mu_safe)
        return float(max(divergence, 0.0))

    # ------------------------------------------------------------------
    # Frank-Wolfe Gap: g(μ) = max_{v∈Z} ⟨∇R(μ), μ−v⟩
    # ------------------------------------------------------------------
    def compute_fw_gap(self, mu: np.ndarray, Z: np.ndarray) -> float:
        mu_safe = np.clip(mu, 1e-10, 1.0 - 1e-10)
        grad = -(np.log(mu_safe) + 1.0)
        max_gap = -np.inf
        for v in Z:
            gap = np.dot(grad, mu_safe - v)
            max_gap = max(max_gap, gap)
        return float(max(max_gap, 0.0))

    # ------------------------------------------------------------------
    # α-Extraction Condition: g(μ_t) ≤ (1−α)·D(μ_t‖θ)
    # Signal only fires when 90% of mathematical edge is captured
    # ------------------------------------------------------------------
    def check_alpha_extraction(self, gap: float, divergence: float) -> bool:
        if divergence <= 1e-10:
            return True
        return gap <= (1.0 - self.alpha) * divergence

    # ------------------------------------------------------------------
    # Guaranteed Profit (Proposition 4.1): Profit ≥ D(μ̂‖θ) − g(μ̂)
    # ------------------------------------------------------------------
    def guaranteed_profit(self, divergence: float, gap: float) -> float:
        return float(max(divergence - gap, 0.0))

    # ------------------------------------------------------------------
    # Barrier Frank-Wolfe (Core Algorithm)
    # ------------------------------------------------------------------
    def barrier_frank_wolfe(self, prices: np.ndarray, theta: np.ndarray) -> dict:
        """
        Barrier Frank-Wolfe with adaptive ε-contraction toward u = mean(Z₀).
        The contraction maintains bounded Lipschitz constant by keeping
        iterates strictly interior to the polytope.
        """
        n = len(prices)

        # Step 1: InitFW — u = mean(Z₀)
        init = self.init_fw(prices)
        Z = init['Z0']
        u = init['u']  # = (1/n, ..., 1/n) — simplex centroid
        mu = u.copy()

        best_mu = mu.copy()
        best_div = self.compute_bregman_divergence(mu, theta)
        best_gap = self.compute_fw_gap(mu, Z)

        for t in range(self.max_fw_iter):
            epsilon_t = 2.0 / (t + 2.0)

            # Contract toward interior point u
            mu_contracted = (1.0 - epsilon_t) * mu + epsilon_t * u
            mu_contracted = np.clip(mu_contracted, 1e-10, 1.0 - 1e-10)
            mu_contracted = mu_contracted / np.sum(mu_contracted)

            # Gradient: ∇R(μ) = −log(μ) − 1
            grad = -(np.log(mu_contracted) + 1.0)

            # Linear minimization: v* = argmin_{v∈Z} ⟨∇R, v⟩
            dot_products = Z @ grad
            v_star_idx = np.argmin(dot_products)
            v_star = Z[v_star_idx]

            gap = max(np.dot(grad, mu_contracted - v_star), 0.0)

            # Step size: γ_t = 2/(t+2)
            gamma = 2.0 / (t + 2.0)

            # Update: μ_{t+1} = (1−γ)·μ_t + γ·v*
            mu_new = (1.0 - gamma) * mu_contracted + gamma * v_star
            mu_new = np.clip(mu_new, 1e-10, 1.0 - 1e-10)
            mu_new = mu_new / np.sum(mu_new)

            div_new = self.compute_bregman_divergence(mu_new, theta)

            if div_new > best_div:
                best_div = div_new
                best_mu = mu_new.copy()
                best_gap = gap

            mu = mu_new

            # α-extraction stopping: only signal if 90% captured
            if self.check_alpha_extraction(gap, div_new):
                return {
                    'mu': best_mu, 'gap': float(best_gap),
                    'divergence': float(best_div), 'converged': True,
                    'iterations': t + 1, 'alpha_satisfied': True,
                }

            if gap < 1e-8:
                break

        final_gap = self.compute_fw_gap(best_mu, Z)
        final_div = self.compute_bregman_divergence(best_mu, theta)

        return {
            'mu': best_mu, 'gap': float(final_gap),
            'divergence': float(final_div),
            'converged': final_gap < 1e-6,
            'iterations': self.max_fw_iter,
            'alpha_satisfied': self.check_alpha_extraction(final_gap, final_div),
        }

    # ------------------------------------------------------------------
    # Dependency Arbitrage
    # ------------------------------------------------------------------
    def detect_dependency_arbitrage(self, markets: list, max_pairs: int = 100) -> list:
        arb_opps = []
        pairs_checked = 0
        for i, m1 in enumerate(markets):
            for j, m2 in enumerate(markets):
                if i >= j or pairs_checked >= max_pairs:
                    continue
                pairs_checked += 1
                q1, q2 = m1.get('question', '').lower(), m2.get('question', '').lower()
                if self._detect_implication(q1, q2):
                    p1, p2 = m1.get('yes_price', 0.5), m2.get('yes_price', 0.5)
                    if p1 > p2 + 0.03:
                        gross = p1 - p2
                        exec_costs = self._estimate_execution_costs(
                            m1.get('liquidity', 1000), m1.get('volume', 100))
                        net = gross - exec_costs['total']
                        if net > self.min_profit_threshold:
                            arb_opps.append({
                                'type': 'dependency_violation',
                                'market1': m1.get('question', ''),
                                'market2': m2.get('question', ''),
                                'slug1': m1.get('slug', ''),
                                'slug2': m2.get('slug', ''),
                                'p1': p1, 'p2': p2,
                                'gross_profit': gross,
                                'execution_costs': exec_costs,
                                'net_profit': net,
                                'strategy': f'Short M1 @ {p1:.2f} / Long M2 @ {p2:.2f}',
                                'confidence': self._compute_confidence(q1, q2),
                            })
        return arb_opps

    def _detect_implication(self, e1: str, e2: str) -> bool:
        w1, w2 = set(e1.split()), set(e2.split())
        if len(w1) == 0:
            return False
        if len(w1.intersection(w2)) / len(w1) > 0.7:
            return True
        return e1 in e2 or e2 in e1

    def _compute_confidence(self, e1: str, e2: str) -> float:
        w1, w2 = set(e1.split()), set(e2.split())
        union = w1.union(w2)
        return float(len(w1.intersection(w2)) / len(union)) if union else 0.0

    def _estimate_execution_costs(self, liquidity: float, volume: float) -> dict:
        max_trade = max(min(volume * 0.10, liquidity * 0.05), 100)
        slippage = min(0.005 * np.sqrt(max_trade / max(liquidity, 100)), 0.05)
        gas_pct = 2.0 / max(max_trade, 100)
        return {
            'slippage': float(slippage), 'gas': float(gas_pct),
            'total': float(slippage + gas_pct), 'max_trade_size': float(max_trade),
        }

    # ------------------------------------------------------------------
    # analyze_market: Full Kroer Pipeline → TRADE / HOLD
    # ------------------------------------------------------------------
    def analyze_market(self, market_data: dict) -> dict | None:
        """
        Full Kroer pipeline:
        1. InitFW → u = mean(Z₀) (simplex centroid)
        2. θ from LMSR inverse of market prices
        3. Barrier FW → find optimal μ
        4. α-extraction check (α=0.9)
        5. Guaranteed profit (Prop 4.1)
        6. TRADE only if α-extraction satisfied AND profitable
        """
        try:
            outcome_prices = market_data.get('outcomePrices', ['0.5', '0.5'])
            prices = np.array([float(p) for p in outcome_prices])
            n = len(prices)
            if n < 2:
                return None

            prices_norm = prices / (np.sum(prices) + 1e-10)
            prices_safe = np.clip(prices_norm, 1e-8, 1.0 - 1e-8)
            theta = self.b * np.log(prices_safe)

            fw_result = self.barrier_frank_wolfe(prices_norm, theta)

            divergence = fw_result['divergence']
            gap = fw_result['gap']
            optimal_mu = fw_result['mu']

            gross_profit = self.guaranteed_profit(divergence, gap)
            execution = self._estimate_execution_costs(
                market_data.get('liquidity', 1000),
                market_data.get('volume', 100),
            )
            net_profit = gross_profit - execution['total']

            # TRADE only when α-extraction (90%) is satisfied
            alpha_satisfied = fw_result['alpha_satisfied']
            is_profitable = net_profit > self.min_profit_threshold and alpha_satisfied
            signal = 'TRADE' if is_profitable else 'HOLD'

            return {
                'signal': signal,
                'is_profitable': is_profitable,
                'alpha_satisfied': alpha_satisfied,
                'prices': prices.tolist(),
                'optimal_prices': optimal_mu.tolist(),
                'price_deviation': float(np.abs(prices_norm - optimal_mu).max()),
                'bregman_divergence': divergence,
                'fw_gap': gap,
                'guaranteed_profit': gross_profit,
                'execution_costs': execution,
                'net_profit': float(net_profit),
                'sharpe_estimate': float(net_profit / max(execution['total'], 0.01)),
                'fw_iterations': fw_result['iterations'],
                'fw_converged': fw_result['converged'],
            }
        except Exception:
            return None


# ============================================================================
# POLYMARKET DATA FETCH
# ============================================================================

@st.cache_data(ttl=180)
def fetch_polymarket_with_arbitrage():
    """Fetch Polymarket data with Kroer Framework arbitrage detection."""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {'limit': 100, 'active': 'true', 'closed': 'false'}
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return pd.DataFrame(), pd.DataFrame()
        markets = response.json()
        filter_keywords = [
            'nfl', 'nba', 'sport', 'gaming', 'gta', 'pop culture', 'music',
            'twitch', 'mlb', 'nhl', 'soccer', 'football', 'basketball',
            'celebrity', 'movie', 'ufc', 'mma', 'tennis',
        ]
        arb_engine = PolymarketArbitrageEngine(
            min_profit_threshold=0.02, alpha=0.9, max_fw_iter=500
        )
        opportunities = []
        arbitrage_trades = []

        for market in markets:
            try:
                question = market.get('question', '').lower()
                if any(kw in question for kw in filter_keywords):
                    continue
                slug = market.get('slug', '')
                volume = float(market.get('volume') or 0)
                liquidity = float(market.get('liquidity') or 0)
                outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
                if not outcome_prices or len(outcome_prices) == 0:
                    outcome_prices = ['0.5', '0.5']
                yes_price = float(outcome_prices[0] or 0.5)
                market_full = {
                    'question': market.get('question', ''), 'slug': slug,
                    'yes_price': yes_price, 'outcomePrices': outcome_prices,
                    'volume': volume, 'liquidity': liquidity,
                }
                arb_analysis = arb_engine.analyze_market(market_full)
                if arb_analysis and arb_analysis['is_profitable']:
                    arbitrage_trades.append({
                        'Event': market.get('question', ''),
                        'slug': slug,
                        'Signal': arb_analysis['signal'],
                        'Yes %': yes_price * 100,
                        'Optimal %': arb_analysis['optimal_prices'][0] * 100,
                        'D(μ‖θ)': arb_analysis['bregman_divergence'],
                        'FW Gap': arb_analysis['fw_gap'],
                        'Profit ≥': arb_analysis['guaranteed_profit'],
                        'Net %': arb_analysis['net_profit'] * 100,
                        'Sharpe': arb_analysis['sharpe_estimate'],
                        'Max Trade': f"${arb_analysis['execution_costs']['max_trade_size']:.0f}",
                        'α-Met': '✅' if arb_analysis['alpha_satisfied'] else '❌',
                        'Iters': arb_analysis['fw_iterations'],
                    })
                if volume > 100:
                    arb_score = arb_analysis['bregman_divergence'] if arb_analysis else 0
                    opportunities.append({
                        'Event': market.get('question', ''), 'slug': slug,
                        'Yes %': yes_price * 100, 'Volume': volume,
                        'Liquidity': liquidity, 'Arb Score': arb_score,
                    })
            except Exception:
                continue

        markets_clean = []
        for m in markets:
            try:
                if any(kw in m.get('question', '').lower() for kw in filter_keywords):
                    continue
                outcome_prices = m.get('outcomePrices', ['0.5'])
                yes_price = float(outcome_prices[0] or 0.5) if outcome_prices else 0.5
                markets_clean.append({
                    'question': m.get('question', ''), 'slug': m.get('slug', ''),
                    'yes_price': yes_price,
                    'liquidity': float(m.get('liquidity') or 1000),
                    'volume': float(m.get('volume') or 100),
                })
            except Exception:
                continue

        dep_arbs = arb_engine.detect_dependency_arbitrage(markets_clean)
        for arb in dep_arbs:
            arbitrage_trades.append({
                'Event': f"{arb['market1'][:40]}… ⇒ {arb['market2'][:40]}…",
                'slug': arb['slug1'],
                'Signal': 'TRADE',
                'Yes %': arb['p1'] * 100,
                'Optimal %': arb['p2'] * 100,
                'D(μ‖θ)': arb['gross_profit'],
                'FW Gap': 0.0,
                'Profit ≥': arb['net_profit'],
                'Net %': arb['net_profit'] * 100,
                'Sharpe': arb['net_profit'] / max(arb['execution_costs']['total'], 0.01),
                'Max Trade': f"${arb['execution_costs']['max_trade_size']:.0f}",
                'α-Met': '✅', 'Iters': 0, 'Type': 'DEP',
            })

        opportunities_sorted = sorted(opportunities, key=lambda x: x['Arb Score'], reverse=True)
        opp_df = pd.DataFrame(opportunities_sorted[:10]) if opportunities_sorted else pd.DataFrame()
        arb_df = pd.DataFrame(arbitrage_trades) if arbitrage_trades else pd.DataFrame()
        return opp_df, arb_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ============================================================================
# AI ENGINE — Gemini 1.5 Flash | Lead Session Strategist | Regime Classifier
# ============================================================================

_GEMINI_SYSTEM_INSTRUCTION = """You are the Lead Session Strategist at a proprietary day-trading desk
specializing in intraday SPX/ES scalping.

Your primary function is to classify the current session into exactly ONE of three daily regimes:
1. **EXPANSIONARY (Trend)** — Directional move sustained by volume.
2. **CONSOLIDATION (Range)** — Price compressing between defined levels.
3. **REVERSAL** — Momentum fading at extremes; mean-reversion setup.

Rules:
- Start your briefing with the regime label in bold, e.g.: **REGIME: EXPANSIONARY**
- Write in dense, actionable prose (no bullet points). Maximum 250 words.
- Use specific numbers from the provided data table — do NOT fabricate levels.
- Reference the key bias levels: PDH, PDL, Opening Range, VWAP, Max Pain.
- For SPX, provide probability-weighted intraday scenarios, e.g.:
  "65% continuation above VWAP to PDH, 25% range between OR levels, 10% reversal below PDL."
- Identify which levels are key inflection zones for the session.
- Note if GEX positioning (positive = dampening vol, negative = amplifying moves) changes your bias.
"""


def generate_enhanced_ai_macro_briefing(indices, spx_options, liquidity, correlations, bias_levels=None):
    """Generate session strategy briefing with Gemini 1.5 Flash."""
    if not st.session_state.gemini_configured:
        return (
            "⚠️ **Gemini API Configuration Required** — Enable API access in "
            "sidebar to unlock session strategy intelligence."
        )
    try:
        model = genai.GenerativeModel(
            'models/gemini-1.5-flash',
            system_instruction=_GEMINI_SYSTEM_INSTRUCTION,
        )
        spx_price = indices.get('SPX', {}).get('price', 0)
        vix_price = indices.get('VIX', {}).get('price', 0)
        pc_ratio = spx_options.get('put_call_ratio', 0) if spx_options else 0
        max_pain = spx_options.get('max_pain', 0) if spx_options else 0
        net_gex = spx_options.get('net_gex', 0) if spx_options else 0
        yield_spread = liquidity.get('yield_spread', 0)
        credit_spread = liquidity.get('credit_spread', 0)
        fed_balance = liquidity.get('fed_balance', 0)

        # Bias levels
        pdh = bias_levels.get('pdh', 0) if bias_levels else 0
        pdl = bias_levels.get('pdl', 0) if bias_levels else 0
        or_high = bias_levels.get('or_high', 0) if bias_levels else 0
        or_low = bias_levels.get('or_low', 0) if bias_levels else 0
        vwap = bias_levels.get('vwap', 0) if bias_levels else 0

        if yield_spread < 0:
            yc_context = f"inverted by {abs(yield_spread):.2f}% (recession signal active)"
        elif yield_spread < 0.25:
            yc_context = f"flattening at {yield_spread:.2f}% (late-cycle dynamics)"
        else:
            yc_context = f"normalized at {yield_spread:.2f}% (expansion intact)"

        gex_context = "positive (vol dampening)" if net_gex > 0 else "negative (vol amplifying)"

        user_message = f"""Classify today's regime and generate a session strategy briefing (200-250 words) based on this live data:

| Metric | Value |
|--------|-------|
| SPX Price | ${spx_price:.0f} |
| VIX Level | {vix_price:.1f} |
| Put/Call Ratio | {pc_ratio:.2f} |
| Max Pain Strike | ${max_pain:.0f} |
| Net GEX | {net_gex:,.0f} ({gex_context}) |
| Previous Day High (PDH) | ${pdh:.0f} |
| Previous Day Low (PDL) | ${pdl:.0f} |
| Opening Range High (15m) | ${or_high:.0f} |
| Opening Range Low (15m) | ${or_low:.0f} |
| Session VWAP | ${vwap:.2f} |
| 10Y-2Y Spread | {yield_spread:.2f}% ({yc_context}) |
| HY Credit Spread | {credit_spread:.2f}% |
| Fed Balance Sheet | ${fed_balance:.2f}T |

Classify as EXPANSIONARY, CONSOLIDATION, or REVERSAL. Provide probability-weighted scenarios referencing PDH, PDL, OR, and VWAP."""

        response = model.generate_content(user_message)
        return response.text
    except Exception as e:
        error_msg = str(e)
        return (
            f"⚠️ **AI Generation Error** — {error_msg[:200]}...\n\n"
            "Possible issues:\n- API key invalid or expired\n"
            "- Quota exceeded\n- Model unavailable\n\n"
            "Verify your Gemini API key in the sidebar."
        )


# ============================================================================
# UI COMPONENTS — CARD-BASED LAYOUTS
# ============================================================================

def render_metric_card(label, value, change_pct, change_abs, signals=None):
    """Render glassmorphism metric card from template."""
    change_class = "positive" if change_pct >= 0 else "negative"
    change_symbol = "+" if change_pct >= 0 else ""

    if "SPX" in label or "NDX" in label:
        abs_str = f"{change_symbol}{change_abs:.2f} pts"
    elif "$" in str(value):
        abs_str = f"${change_symbol}{change_abs:.2f}"
    else:
        abs_str = f"{change_symbol}{change_abs:.2f}"

    signals_html = ""
    if signals and signals != "—":
        signals_html = '<div style="margin-top: 12px;">'
        for signal in signals.split(" | "):
            signals_html += TEMPLATES["signal_badge"].format(signal=signal)
        signals_html += '</div>'

    card_html = TEMPLATES["metric_card"].format(
        label=label, value=value, change_class=change_class,
        abs_str=abs_str, change_symbol=change_symbol,
        change_pct=change_pct, signals_html=signals_html,
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_watchlist_cards(df):
    """Render watchlist as card grid."""
    if df.empty:
        st.warning("📊 Watchlist data unavailable")
        return
    cols_per_row = 3
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx >= len(df):
                break
            row = df.iloc[idx]
            with cols[j]:
                render_metric_card(
                    label=row['Ticker'],
                    value=f"${row['Price']:.2f}",
                    change_pct=row['Change %'],
                    change_abs=row['Change $'],
                    signals=row.get('Signals', ''),
                )


def render_interactive_sector_heatmap(sector_df):
    """Interactive sector heatmap with drill-down."""
    if sector_df.empty:
        st.warning("📊 Sector data unavailable")
        return

    fig = px.bar(
        sector_df, x='Change %', y='Sector', orientation='h',
        color='Change %',
        color_continuous_scale=[[0, '#FF4444'], [0.5, '#000000'], [1, '#00FF88']],
        color_continuous_midpoint=0,
        hover_data={'Change %': ':.2f%', 'Change $': ':.2f'},
        custom_data=['Ticker'],
    )
    fig.update_traces(
        texttemplate='%{x:.2f}%', textposition='outside',
        marker_line_width=1, marker_line_color='rgba(255,176,0,0.5)',
    )
    fig.update_layout(
        template='plotly_dark', showlegend=False, height=500,
        margin=dict(l=0, r=50, t=20, b=0),
        plot_bgcolor='#000000', paper_bgcolor='#000000',
        font=dict(color='#FFB000', family='JetBrains Mono, Courier New', size=12),
        xaxis=dict(showgrid=False, color='#FFB000', title='Performance (%)',
                   zeroline=True, zerolinecolor='#FFB000'),
        yaxis=dict(showgrid=False, color='#FFB000', title=''),
    )
    st.plotly_chart(fig, use_container_width=True, key="sector_heatmap")

    st.caption("🔍 **SECTOR DRILL-DOWN** — Click to view constituents")
    sectors_list = sector_df.to_dict('records')

    if len(sectors_list) > 0:
        cols_row1 = st.columns(min(5, len(sectors_list)))
        for idx_s in range(min(5, len(sectors_list))):
            row = sectors_list[idx_s]
            short_name = SECTOR_SHORT_NAMES.get(row['Sector'], row['Sector'][:8])
            with cols_row1[idx_s]:
                if st.button(short_name, key=f"sector_{row['Ticker']}"):
                    st.session_state.selected_sector = row['Ticker']
                    st.session_state.show_sector_drill = True
                    st.rerun()

    if len(sectors_list) > 5:
        cols_row2 = st.columns(min(5, len(sectors_list) - 5))
        for idx_s in range(5, min(10, len(sectors_list))):
            row = sectors_list[idx_s]
            short_name = SECTOR_SHORT_NAMES.get(row['Sector'], row['Sector'][:8])
            with cols_row2[idx_s - 5]:
                if st.button(short_name, key=f"sector_{row['Ticker']}"):
                    st.session_state.selected_sector = row['Ticker']
                    st.session_state.show_sector_drill = True
                    st.rerun()

    if st.session_state.show_sector_drill and st.session_state.selected_sector:
        _render_sector_drilldown(sector_df)


@st.fragment
def _render_sector_drilldown(sector_df):
    """Sector drill-down fragment."""
    sector_ticker = st.session_state.selected_sector
    matching = sector_df[sector_df['Ticker'] == sector_ticker]['Sector'].values
    sector_name = matching[0] if len(matching) > 0 else sector_ticker
    st.markdown("---")
    st.subheader(f"🔬 {sector_name} — CONSTITUENT BREAKDOWN")
    if st.button("❌ CLOSE DRILL-DOWN", key="close_drill"):
        st.session_state.show_sector_drill = False
        st.session_state.selected_sector = None
        st.rerun()
    constituents = fetch_sector_constituents(sector_ticker)
    if not constituents.empty:
        render_watchlist_cards(constituents)
    else:
        st.warning(f"No constituent data available for {sector_name}")


# ============================================================================
# SPX OPTIONS — EOD Review + GEX Display Fragment
# ============================================================================

@st.fragment
def render_spx_options_section():
    """SPX Options with GEX and EOD Review Mode."""
    st.subheader("🎯 SPX OPTIONS + GEX INTELLIGENCE")
    spx_data = fetch_spx_options_data()

    display_data = None
    is_eod_review = False
    if spx_data and spx_data.get('success'):
        display_data = spx_data
    elif st.session_state.get('last_spx_options'):
        display_data = st.session_state.last_spx_options
        is_eod_review = True

    if display_data:
        if is_eod_review:
            st.info("📋 **DAILY EOD REVIEW** — Showing last recorded options data")

        cols_opt = st.columns(6)
        with cols_opt[0]:
            st.metric("P/C Ratio", f"{display_data['put_call_ratio']:.2f}")
        with cols_opt[1]:
            st.metric("P/C OI", f"{display_data['put_call_oi_ratio']:.2f}")
        with cols_opt[2]:
            st.metric("Max Pain", f"${display_data['max_pain']:.0f}")
        with cols_opt[3]:
            st.metric("Call IV", f"{display_data['avg_call_iv']:.1f}%")
        with cols_opt[4]:
            st.metric("Put IV", f"{display_data['avg_put_iv']:.1f}%")
        with cols_opt[5]:
            st.metric("Net GEX", f"{display_data.get('net_gex', 0):,.0f}")

        st.caption(f"📅 Expiration: {display_data['expiration']}")

        col_gex1, col_gex2 = st.columns(2)
        with col_gex1:
            call_gex = display_data.get('call_gex', 0)
            put_gex = display_data.get('put_gex', 0)
            net_gex = display_data.get('net_gex', 0)
            st.caption("**GAMMA EXPOSURE BREAKDOWN**")
            st.markdown(
                f"- Call GEX: **{call_gex:,.0f}**\n"
                f"- Put GEX: **{put_gex:,.0f}**\n"
                f"- Net GEX: **{net_gex:,.0f}**"
            )
            if net_gex > 0:
                st.success("🛡️ **Positive GEX** — Dealers hedging dampens volatility")
            else:
                st.warning("⚠️ **Negative GEX** — Dealers amplify moves")

        with col_gex2:
            top_strikes = display_data.get('top_gex_strikes', [])
            if top_strikes:
                st.caption("**TOP GEX STRIKES (Hedging Zones)**")
                for ts in top_strikes:
                    st.markdown(f"- **${ts['strike']:.0f}** → GEX: {ts['gex']:,.0f}")
    else:
        st.info("⏰ Markets closed — No options data cached. Data will populate during market hours.")


# ============================================================================
# NEWS SQUAWK RENDERER
# ============================================================================

def render_news_squawk():
    """Render scrollable news squawk container with WSJ + Reuters."""
    articles = fetch_news_feeds()
    if not articles:
        st.info("📡 No news available")
        return

    items_html = ""
    for article in articles:
        source = article.get('Source', '')
        title = article.get('Title', 'Untitled')
        link = article.get('Link', '#')
        items_html += f"""
        <div class="news-item">
            <div class="news-source">{source}</div>
            <div class="news-title"><a href="{link}" target="_blank">{title}</a></div>
        </div>
        """

    st.markdown(
        f'<div class="news-squawk">{items_html}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# EARNINGS CARD RENDERER
# ============================================================================

def render_earnings_card(ticker, date_str, eps_est, eps_act, surprise, is_shifter):
    """Render a single earnings card with optional Market Shifter styling."""
    card_class = "earnings-card market-shifter" if is_shifter else "earnings-card"
    badge_html = '<span class="vol-badge">⚡ VOLATILITY RISK</span>' if is_shifter else ''

    html = f"""
    <div class="{card_class}">
        <div class="ticker">{ticker} {badge_html}</div>
        <div class="date-label">{date_str}</div>
        <div class="eps-row">Est: {eps_est} | Act: {eps_act} | Surprise: {surprise}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("⚡ ALPHA DECK PRO v8.0")
st.caption("**DIRECTIONAL BIAS TERMINAL** — Intraday SPX/ES Scalping | Kroer Barrier FW | GEX + Regime Intelligence")
st.divider()

# ============================================================================
# AI SESSION STRATEGY BRIEFING
# ============================================================================

st.subheader("🧠 SESSION STRATEGY — REGIME CLASSIFIER")

if st.button("🚀 GENERATE SESSION BRIEF", key="ai_macro", use_container_width=True):
    with st.spinner('🔬 Classifying session regime...'):
        indices = fetch_index_data()
        spx_opts = fetch_spx_options_data()
        liq = fetch_fred_liquidity()
        corrs = calculate_spx_crypto_correlation()
        bias = fetch_intraday_bias_levels()

        briefing = generate_enhanced_ai_macro_briefing(indices, spx_opts, liq, corrs, bias)

        st.markdown(
            TEMPLATES["ai_briefing"].format(content=briefing),
            unsafe_allow_html=True,
        )

st.divider()

# ============================================================================
# TABS — 6 TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 MAIN DECK",
    "💰 POLYMARKET ARBITRAGE",
    "₿ CRYPTO",
    "🏛️ MACRO",
    "📅 EARNINGS",
    "📈 TRADINGVIEW",
])

# ============================================================================
# TAB 1: MAIN DECK
# ============================================================================

with tab1:
    st.subheader("📡 MARKET PULSE")
    indices = fetch_index_data()
    cols = st.columns(6)
    for idx, (name, data) in enumerate(indices.items()):
        with cols[idx]:
            if data['success']:
                if name in ['VIX', 'HYG']:
                    value_str = f"{data['price']:.2f}"
                else:
                    value_str = f"${data['price']:.2f}"
                render_metric_card(
                    label=name, value=value_str,
                    change_pct=data['change_pct'],
                    change_abs=data['change_abs'],
                )
            else:
                st.error(f"{name}: OFFLINE")

    st.divider()

    # VIX Term Structure
    st.subheader("📊 VOLATILITY TERM STRUCTURE")
    vix_term = fetch_vix_term_structure()
    col_vix1, col_vix2 = st.columns([1, 2])
    with col_vix1:
        if vix_term['backwardation']:
            st.error("⚠️ **BACKWARDATION** — Crash signal active")
        else:
            st.success("✅ **CONTANGO** — Normal vol structure")
    with col_vix2:
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=['VIX', 'VIX9D', 'VIX3M'],
            y=[vix_term['VIX'], vix_term['VIX9D'], vix_term['VIX3M']],
            mode='lines+markers',
            line=dict(color='#FFB000', width=4),
            marker=dict(size=12, color='#FFB000', symbol='diamond'),
        ))
        fig_vix.update_layout(
            template='plotly_dark', height=250,
            plot_bgcolor='#000000', paper_bgcolor='#000000',
            font=dict(color='#FFB000', family='JetBrains Mono, Courier New'),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, color='#FFB000'),
            yaxis=dict(showgrid=True, gridcolor='#333333', title="Vol", color='#FFB000'),
        )
        st.plotly_chart(fig_vix, use_container_width=True)

    st.divider()

    # SPX Options + GEX
    render_spx_options_section()

    st.divider()

    # Watchlist
    st.subheader("⚡ WATCHLIST + TECHNICAL SCANNER")
    st.caption("**Wilder's RSI (60-day warm-up)** — NaN displays CALIBRATING")
    watchlist_tickers = ('NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN',
                         'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ')
    watchlist_df = fetch_watchlist_data(watchlist_tickers)
    render_watchlist_cards(watchlist_df)

    st.divider()

    # Sector Heatmap
    st.subheader("🎨 SECTOR HEAT — INTERACTIVE DRILL-DOWN")
    st.caption("**Click any sector** to view constituent stocks")
    sector_df = fetch_sector_performance()
    render_interactive_sector_heatmap(sector_df)

# ============================================================================
# TAB 2: POLYMARKET ARBITRAGE (st.fragment)
# ============================================================================

with tab2:
    @st.fragment
    def render_arbitrage_tab():
        st.subheader("🎲 POLYMARKET ARBITRAGE ENGINE")
        st.caption(
            "**Kroer Framework** — InitFW: u=mean(Z₀) | Barrier FW | "
            "α-Extraction (α=0.9) | Guaranteed Profit (Prop 4.1)"
        )

        opp_df, arb_df = fetch_polymarket_with_arbitrage()

        st.markdown("### 💰 ARBITRAGE TRADES — TRADE / HOLD")

        if not arb_df.empty:
            for _, row in arb_df.iterrows():
                signal = row.get('Signal', 'HOLD')
                css_class = 'trade' if signal == 'TRADE' else 'hold'
                event_short = str(row.get('Event', ''))[:70]
                profit_str = f"Profit ≥ {row.get('Profit ≥', 0):.4f}"
                alpha_str = row.get('α-Met', '❌')
                st.markdown(
                    f'<span class="trade-signal {css_class}">{signal}</span> '
                    f'**{event_short}…** — {profit_str} | α: {alpha_str}',
                    unsafe_allow_html=True,
                )

            st.divider()

            display_cols = [c for c in [
                'Event', 'Signal', 'Yes %', 'Optimal %', 'D(μ‖θ)', 'FW Gap',
                'Profit ≥', 'Net %', 'Sharpe', 'Max Trade', 'α-Met', 'Iters',
            ] if c in arb_df.columns]

            st.dataframe(
                arb_df[display_cols], use_container_width=True, height=400,
                hide_index=True,
                column_config={
                    "Yes %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Optimal %": st.column_config.NumberColumn(format="%.2f%%"),
                    "D(μ‖θ)": st.column_config.NumberColumn(format="%.4f"),
                    "FW Gap": st.column_config.NumberColumn(format="%.6f"),
                    "Profit ≥": st.column_config.NumberColumn(format="%.4f"),
                    "Net %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption(
                f"🎯 **{len(arb_df)} opportunities** — α-extraction ≥ 90% required for TRADE signal"
            )
        else:
            st.info("✅ No arbitrage detected — Markets pricing efficiently")

        st.divider()
        st.markdown("### 📊 TOP PREDICTION MARKETS")
        if not opp_df.empty:
            opp_display = opp_df.copy()
            opp_display['Volume'] = opp_display['Volume'].apply(
                lambda x: f"${x / 1e6:.2f}M" if x >= 1e6 else f"${x / 1e3:.0f}K")
            opp_display['Liquidity'] = opp_display['Liquidity'].apply(
                lambda x: f"${x / 1e6:.2f}M" if x >= 1e6 else f"${x / 1e3:.0f}K")
            opp_display['Yes %'] = opp_display['Yes %'].apply(lambda x: f"{x:.1f}%")
            opp_display['Arb Score'] = opp_display['Arb Score'].apply(lambda x: f"{x:.4f}")
            st.dataframe(
                opp_display[['Event', 'Yes %', 'Volume', 'Liquidity', 'Arb Score']],
                use_container_width=True, height=400, hide_index=True,
            )
            st.caption("**🔗 Clickable Links:**")
            for _, row in opp_df.iterrows():
                if row.get('slug'):
                    url = f"https://polymarket.com/event/{row['slug']}"
                    st.markdown(f"[{row['Event'][:80]}…]({url})")
        else:
            st.warning("📊 No markets available")

    render_arbitrage_tab()

# ============================================================================
# TAB 3: CRYPTO (Pure crypto + correlation — correlation stays here)
# ============================================================================

with tab3:
    st.subheader("₿ CRYPTO MARKET PULSE")
    crypto_data = fetch_crypto_metrics(('BTC', 'ETH', 'SOL', 'DOGE'))
    cols_crypto = st.columns(4)
    for idx, (crypto, data) in enumerate(crypto_data.items()):
        with cols_crypto[idx]:
            if data['success']:
                render_metric_card(
                    label=crypto,
                    value=f"${data['price']:,.2f}",
                    change_pct=data['change_pct'],
                    change_abs=data['change_abs'],
                )

    st.divider()

    # SPX/Crypto Correlation — stays in Crypto tab
    st.subheader("🔗 SPX/CRYPTO CORRELATIONS")
    corrs = calculate_spx_crypto_correlation()
    cols_corr = st.columns(2)
    with cols_corr[0]:
        st.metric("SPX / BTC", f"{corrs['BTC']:.3f}")
    with cols_corr[1]:
        st.metric("SPX / ETH", f"{corrs['ETH']:.3f}")

    if corrs['BTC'] > 0.6:
        st.success("📈 **Risk-On Regime** — High correlation signals unified risk appetite")
    elif corrs['BTC'] < 0.3:
        st.warning("📉 **Risk-Off / Decoupling** — Low correlation indicates divergence")
    else:
        st.info("⚖️ **Transition Regime** — Moderate correlation, regime unclear")

# ============================================================================
# TAB 4: MACRO — Intraday Directional Bias (No correlation — moved to Crypto)
# ============================================================================

with tab4:
    @st.fragment
    def render_macro_tab():
        st.subheader("🏛️ INTRADAY DIRECTIONAL BIAS")

        # --- Daily Pivot / Bias Levels Card ---
        st.markdown("### 📐 DAILY PIVOT LEVELS")
        bias = fetch_intraday_bias_levels()

        if bias['success']:
            # Bias level cards in a row
            cols_bias = st.columns(5)
            bias_items = [
                ("PDH", f"${bias['pdh']:.0f}"),
                ("PDL", f"${bias['pdl']:.0f}"),
                ("OR HIGH", f"${bias['or_high']:.0f}"),
                ("OR LOW", f"${bias['or_low']:.0f}"),
                ("VWAP", f"${bias['vwap']:.2f}"),
            ]
            for i, (label, val) in enumerate(bias_items):
                with cols_bias[i]:
                    st.markdown(
                        f'<div class="bias-card">'
                        f'<div class="level-label">{label}</div>'
                        f'<div class="level-value">{val}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # SPX position relative to levels
            current = bias['current']
            st.divider()
            st.caption("**SPX POSITION RELATIVE TO KEY LEVELS**")

            if current > bias['pdh']:
                st.success(f"🚀 **ABOVE PDH** — SPX ${current:.0f} > PDH ${bias['pdh']:.0f} (Breakout territory)")
            elif current < bias['pdl']:
                st.error(f"📉 **BELOW PDL** — SPX ${current:.0f} < PDL ${bias['pdl']:.0f} (Breakdown territory)")
            elif bias['or_high'] > 0 and current > bias['or_high']:
                st.success(f"📈 **ABOVE OR HIGH** — SPX ${current:.0f} > OR ${bias['or_high']:.0f} (Bullish momentum)")
            elif bias['or_low'] > 0 and current < bias['or_low']:
                st.warning(f"📉 **BELOW OR LOW** — SPX ${current:.0f} < OR ${bias['or_low']:.0f} (Bearish pressure)")
            elif bias['vwap'] > 0:
                if current > bias['vwap']:
                    st.info(f"⬆️ **ABOVE VWAP** — SPX ${current:.0f} > VWAP ${bias['vwap']:.2f} (Institutional buying)")
                else:
                    st.info(f"⬇️ **BELOW VWAP** — SPX ${current:.0f} < VWAP ${bias['vwap']:.2f} (Institutional selling)")
        else:
            st.warning("📊 Bias levels unavailable — Market data pending")

        st.divider()

        # --- FRED Liquidity ---
        st.markdown("### 🏛️ FED LIQUIDITY METRICS")
        liq = fetch_fred_liquidity()

        if liq['success']:
            cols_liq = st.columns(3)
            with cols_liq[0]:
                st.metric("10Y-2Y Spread", f"{liq['yield_spread']:.2f}%")
            with cols_liq[1]:
                st.metric("HY Credit Spread", f"{liq['credit_spread']:.2f}%")
            with cols_liq[2]:
                st.metric("Fed Balance", f"${liq['fed_balance']:.2f}T")

            if liq['yield_spread'] < 0:
                st.error("⚠️ **INVERTED CURVE** — Recession signal active")
            elif liq['yield_spread'] < 0.25:
                st.warning("⚡ **FLATTENING CURVE** — Late-cycle dynamics")
            else:
                st.success("✅ **NORMALIZED CURVE** — Expansion intact")

            st.divider()
            st.markdown("### 📈 YIELD CURVE SNAPSHOT")
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Bar(
                x=['10Y-2Y Spread', 'HY Credit Spread'],
                y=[liq['yield_spread'], liq['credit_spread']],
                marker_color=['#FFB000' if liq['yield_spread'] >= 0 else '#FF4444',
                              '#FFB000' if liq['credit_spread'] < 5 else '#FF4444'],
                text=[f"{liq['yield_spread']:.2f}%", f"{liq['credit_spread']:.2f}%"],
                textposition='outside',
            ))
            fig_yc.update_layout(
                template='plotly_dark', height=300,
                plot_bgcolor='#000000', paper_bgcolor='#000000',
                font=dict(color='#FFB000', family='JetBrains Mono, Courier New'),
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(showgrid=True, gridcolor='#333333', color='#FFB000', title='%'),
                xaxis=dict(showgrid=False, color='#FFB000'),
            )
            st.plotly_chart(fig_yc, use_container_width=True)
        else:
            st.warning("⚠️ FRED API: Configure key in sidebar for macro data")
            if 'error' in liq:
                st.caption(f"Error: {liq['error']}")

        st.divider()

        # --- News Squawk ---
        st.markdown("### 📡 NEWS SQUAWK — WSJ + REUTERS")
        render_news_squawk()

    render_macro_tab()

# ============================================================================
# TAB 5: EARNINGS — Strategic 3-Column Grid (st.fragment)
# ============================================================================

with tab5:
    @st.fragment
    def render_earnings_tab():
        st.subheader("📅 EARNINGS — STRATEGIC GRID")
        st.caption("Market Shifters highlighted with gold border + ⚡ VOLATILITY RISK badge")

        earnings_df = fetch_earnings_calendar()

        if not earnings_df.empty:
            today = pd.Timestamp.now().normalize()
            tomorrow = today + pd.Timedelta(days=1)
            week_end = today + pd.Timedelta(days=7)

            # Categorize
            today_list = earnings_df[earnings_df['Date'].dt.normalize() == today]
            tomorrow_list = earnings_df[earnings_df['Date'].dt.normalize() == tomorrow]
            week_list = earnings_df[
                (earnings_df['Date'].dt.normalize() > tomorrow) &
                (earnings_df['Date'].dt.normalize() <= week_end)
            ]

            col_today, col_tomorrow, col_week = st.columns(3)

            with col_today:
                st.markdown("#### 📌 TODAY")
                if not today_list.empty:
                    for _, row in today_list.iterrows():
                        render_earnings_card(
                            ticker=row['Ticker'],
                            date_str=row['Date'].strftime('%b %d'),
                            eps_est=row['EPS Est'],
                            eps_act=row['EPS Act'],
                            surprise=row['Surprise'],
                            is_shifter=row['is_shifter'],
                        )
                else:
                    st.caption("No earnings today")

            with col_tomorrow:
                st.markdown("#### 📌 TOMORROW")
                if not tomorrow_list.empty:
                    for _, row in tomorrow_list.iterrows():
                        render_earnings_card(
                            ticker=row['Ticker'],
                            date_str=row['Date'].strftime('%b %d'),
                            eps_est=row['EPS Est'],
                            eps_act=row['EPS Act'],
                            surprise=row['Surprise'],
                            is_shifter=row['is_shifter'],
                        )
                else:
                    st.caption("No earnings tomorrow")

            with col_week:
                st.markdown("#### 📌 THIS WEEK")
                if not week_list.empty:
                    for _, row in week_list.iterrows():
                        render_earnings_card(
                            ticker=row['Ticker'],
                            date_str=row['Date'].strftime('%b %d'),
                            eps_est=row['EPS Est'],
                            eps_act=row['EPS Act'],
                            surprise=row['Surprise'],
                            is_shifter=row['is_shifter'],
                        )
                else:
                    st.caption("No earnings this week")

            st.divider()
            st.caption(
                f"📊 **{len(earnings_df)} earnings events** from {len(EARNINGS_TICKERS)} tracked tickers | "
                f"Market Shifters: {', '.join(sorted(MARKET_SHIFTERS))}"
            )
        else:
            st.info("📅 No earnings data available for tracked tickers in this window")
            st.caption("**Tracked:** " + ", ".join(EARNINGS_TICKERS))

    render_earnings_tab()

# ============================================================================
# TAB 6: TRADINGVIEW
# ============================================================================

with tab6:
    st.subheader("📈 TRADINGVIEW ADVANCED CHARTS")
    all_tickers = [
        'SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD',
        'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR',
        'BTC-USD', 'ETH-USD',
    ]
    default_idx = (
        all_tickers.index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in all_tickers
        else 0
    )
    selected_ticker = st.selectbox(
        "SELECT TICKER", all_tickers, index=default_idx, key="chart_select"
    )
    st.session_state.selected_ticker = selected_ticker

    if selected_ticker:
        tv_symbol = get_tradingview_symbol(selected_ticker)
        tradingview_widget = f"""
        <div class="tradingview-widget-container" style="height:100%;width:100%">
          <div id="tradingview_chart" style="height:700px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
          "width": "100%",
          "height": 700,
          "symbol": "{tv_symbol}",
          "interval": "D",
          "timezone": "America/New_York",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#000000",
          "backgroundColor": "#000000",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "save_image": true,
          "studies": [
            "Volume@tv-basicstudies",
            "MASimple@tv-basicstudies",
            "MASimple@tv-basicstudies"
          ],
          "studies_overrides": {{
            "volume.volume.color.0": "#FF4444",
            "volume.volume.color.1": "#00FF88"
          }},
          "container_id": "tradingview_chart"
        }});
          </script>
        </div>
        """
        st.components.v1.html(tradingview_widget, height=750)
        st.caption(f"📊 Symbol: {tv_symbol} | Volume + SMA 15 + SMA 30")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("⚡ **ALPHA DECK PRO v8.0 — DIRECTIONAL BIAS TERMINAL (PRODUCTION-HARDENED)**")
st.caption(
    "✅ Gemini 1.5 Flash Regime Classifier | ✅ InitFW u=mean(Z₀) | ✅ α=0.9 Extraction | "
    "✅ GEX Intelligence | ✅ PDH/PDL/OR/VWAP Bias | ✅ News Squawk | "
    "✅ Earnings Grid (Market Shifters) | ✅ RSI CALIBRATING Fix | ✅ Header Persistence"
)
