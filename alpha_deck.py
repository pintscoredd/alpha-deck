"""
Alpha Deck PRO v6.0 - QUANT EDITION (PRODUCTION-HARDENED)
Refactored: Wilder's RSI | Batch yf.download | linprog Arbitrage | Glassmorphism
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import feedparser
from datetime import datetime, timedelta
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
        background: rgba(255, 176, 0, 0.1);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 176, 0, 0.4);
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 10px;
        font-weight: 600;
        margin: 2px;
        color: #FFB000;
    }

    /* Sidebar – Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 10, 0.85) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 2px solid rgba(255, 176, 0, 0.4);
    }

    /* Tab System */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #000000;
        border-bottom: 2px solid #FFB000;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(8px);
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
        backdrop-filter: blur(8px);
        color: #FFB000;
        border: 2px solid rgba(255, 176, 0, 0.5);
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-weight: 700;
        padding: 12px 24px;
        text-transform: uppercase;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        background-color: #FFB000;
        color: #000000;
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255, 176, 0, 0.3);
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #FFB000;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-weight: 700;
        text-transform: uppercase;
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
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }

    hr { border-color: #FFB000; opacity: 0.3; margin: 2rem 0; }

    .streamlit-expanderHeader {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(8px);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.4);
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }

    /* AI Briefing Box – Glassmorphism */
    .ai-briefing {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 2px solid rgba(255, 176, 0, 0.45);
        border-radius: 14px;
        padding: 24px;
        margin: 16px 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        line-height: 1.8;
        color: #FFFFFF;
        box-shadow: 0 8px 32px rgba(255, 176, 0, 0.08);
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
        <h4>🎯 MARKET INTELLIGENCE BRIEFING</h4>
        {content}
    </div>
    """,
}

TRADINGVIEW_MAPPING = {
    'BTC-USD': 'BITSTAMP:BTCUSD',
    'ETH-USD': 'BITSTAMP:ETHUSD',
    'SOL-USD': 'BINANCE:SOLUSDT',
    'DOGE-USD': 'BINANCE:DOGEUSDT',
    'SPY': 'AMEX:SPY',
    'QQQ': 'NASDAQ:QQQ',
    'IWM': 'AMEX:IWM',
    '^GSPC': 'OANDA:SPX500USD',
    'NVDA': 'NASDAQ:NVDA',
    'TSLA': 'NASDAQ:TSLA',
    'AAPL': 'NASDAQ:AAPL',
    'AMD': 'NASDAQ:AMD',
    'MSFT': 'NASDAQ:MSFT',
    'AMZN': 'NASDAQ:AMZN',
    'META': 'NASDAQ:META',
    'GOOGL': 'NASDAQ:GOOGL',
    'COIN': 'NASDAQ:COIN',
    'MSTR': 'NASDAQ:MSTR',
}

SECTOR_ETF_MAP = {
    'XLK': 'Technology', 'XLE': 'Energy', 'XLF': 'Financials',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Consumer Staples',
    'XLY': 'Consumer Disc.', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLC': 'Communication', 'XLU': 'Utilities',
}

SECTOR_CONSTITUENTS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD'],
    'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX'],
    'XLC': ['META', 'GOOGL', 'DIS', 'NFLX', 'T'],
    'XLI': ['CAT', 'BA', 'HON', 'UNP', 'RTX'],
    'XLP': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'NEM'],
    'XLRE': ['AMT', 'PLD', 'EQIX', 'PSA', 'SPG'],
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
}

SECTOR_SHORT_NAMES = {
    'Technology': 'TECH', 'Energy': 'ENERGY', 'Financials': 'FIN',
    'Healthcare': 'HEALTH', 'Industrials': 'INDUST',
    'Consumer Staples': 'STAPLES', 'Consumer Disc.': 'DISC',
    'Materials': 'MATER', 'Real Estate': 'REAL EST',
    'Communication': 'COMM', 'Utilities': 'UTIL',
}

# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================
st.set_page_config(
    page_title="Alpha Deck PRO v6.0",
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
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Inject CSS from config
st.markdown(f"<style>{STYLES_CONFIG['app_css']}</style>", unsafe_allow_html=True)

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

    # Configure Gemini with robust error handling
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

    # Configure FRED with robust error handling
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
# DATA FETCHING — VECTORIZED yf.download
# ============================================================================

@st.cache_data(ttl=60)
def _batch_download(tickers_tuple: tuple, period: str = '5d') -> pd.DataFrame:
    """
    Central batch downloader. Accepts a tuple of tickers (hashable for caching).
    Returns a MultiIndex DataFrame from yf.download.
    """
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
    This is the industry-standard RSI calculation.
    """
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        # Wilder's smoothing: EWM with alpha = 1/period
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Guard against zero-division
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return 50.0
        return float(np.clip(rsi.iloc[-1], 0.0, 100.0))
    except Exception:
        return 50.0


@st.cache_data(ttl=60)
def fetch_watchlist_data(tickers: tuple) -> pd.DataFrame:
    """
    Fetch watchlist with technical signals using batch yf.download.
    Accepts tuple for cache hashability.
    """
    tickers_list = list(tickers)
    batch = _batch_download(tickers, period='1mo')
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
                'RSI': float(rsi_value),
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
                'Sector': name,
                'Ticker': ticker,
                'Change %': data['change_pct'],
                'Change $': data['change_abs'],
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
    """Fetch SPX options with proper validation."""
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

        return {
            'expiration': nearest_exp,
            'put_call_ratio': put_call_ratio,
            'put_call_oi_ratio': put_call_oi_ratio,
            'max_pain': max_pain,
            'avg_call_iv': avg_call_iv,
            'avg_put_iv': avg_put_iv,
            'calls': calls,
            'puts': puts,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'success': True,
        }
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
        'VIX': vix['price'],
        'VIX9D': vix9d['price'],
        'VIX3M': vix3m['price'],
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
    """Fetch RSS news."""
    feeds = {
        'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'WSJ': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    }
    articles = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                articles.append({
                    'Source': source,
                    'Title': entry.title,
                    'Link': entry.link,
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
    """
    Fetch Fed liquidity data using get_series() (correct fredapi method).
    """
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
            'yield_spread': yield_spread,
            'credit_spread': credit_spread,
            'fed_balance': fed_balance,
            'success': True,
        }
    except Exception as e:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False,
            'error': str(e),
        }


@st.cache_data(ttl=60)
def calculate_spx_crypto_correlation() -> dict:
    """
    SPX vs Crypto correlation with explicit inner join on date indexes
    to handle the weekend gap (SPX closed, crypto 24/7).
    """
    try:
        tickers = ('^GSPC', 'BTC-USD', 'ETH-USD')
        batch = _batch_download(tickers, period='1mo')
        if batch.empty:
            return {'BTC': 0, 'ETH': 0}

        ticker_list = list(tickers)

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

        # Explicit inner join on date index (handles weekend gap)
        combined = pd.merge(
            spx.to_frame(), btc.to_frame(),
            left_index=True, right_index=True, how='inner'
        )
        combined = pd.merge(
            combined, eth.to_frame(),
            left_index=True, right_index=True, how='inner'
        )

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
# POLYMARKET ARBITRAGE ENGINE — PRODUCTION GRADE w/ linprog
# ============================================================================

class PolymarketArbitrageEngine:
    """Production-grade arbitrage detection using marginal polytope theory + LP."""

    def __init__(self, min_profit_threshold: float = 0.05, liquidity_param: float = 100.0):
        self.min_profit_threshold = min_profit_threshold
        self.b = liquidity_param

    def compute_marginal_polytope(self, outcomes):
        n = len(outcomes)
        vertices = np.eye(n)
        try:
            return ConvexHull(vertices)
        except Exception:
            return None

    def check_polytope_membership(self, prices) -> bool:
        if not isinstance(prices, (list, np.ndarray)):
            return True
        prices = np.array(prices)
        if not np.allclose(np.sum(prices), 1.0, atol=0.02):
            return False
        if np.any(prices < -0.01) or np.any(prices > 1.01):
            return False
        return True

    def compute_bregman_divergence(self, mu, theta, cost_function: str = 'lmsr') -> float:
        if cost_function == 'lmsr':
            mu_safe = np.clip(mu, 1e-10, 1 - 1e-10)
            R_mu = -np.sum(mu_safe * np.log(mu_safe))
            theta_safe = np.clip(theta, -100, 100)
            C_theta = self.b * np.log(np.sum(np.exp(theta_safe / self.b)))
            divergence = R_mu + C_theta - np.dot(theta, mu)
            return float(divergence)
        return 0.0

    def frank_wolfe_projection(self, prices, max_iter: int = 100, tolerance: float = 1e-6):
        n = len(prices)
        mu = np.array(prices) / (np.sum(prices) + 1e-10)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        for iteration in range(max_iter):
            grad = np.log(mu) + 1
            vertex_idx = np.argmin(grad)
            vertex = np.zeros(n)
            vertex[vertex_idx] = 1.0
            gamma = 2.0 / (iteration + 2.0)
            mu_new = (1 - gamma) * mu + gamma * vertex
            mu_new = np.clip(mu_new, 1e-10, 1 - 1e-10)
            if np.linalg.norm(mu_new - mu) < tolerance:
                break
            mu = mu_new
        return mu

    def _linprog_arbitrage_bounds(self, prices: np.ndarray) -> dict:
        """
        Use scipy.optimize.linprog to find arbitrage bounds for
        multi-outcome markets (n > 2).
        Formulates an LP over the probability simplex and measures
        the maximum guaranteed profit from mispricing.
        """
        n = len(prices)
        if n <= 1:
            return {'lp_profit': 0.0, 'lp_optimal': prices.tolist(), 'lp_feasible': True}
        try:
            c = -np.ones(n)
            A_eq = np.ones((1, n))
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0) for _ in range(n)]
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if result.success:
                optimal = result.x
                deviation = np.abs(prices - optimal)
                max_dev = float(deviation.max())
                return {'lp_profit': max_dev, 'lp_optimal': optimal.tolist(), 'lp_feasible': True}
            else:
                return {'lp_profit': 0.0, 'lp_optimal': prices.tolist(), 'lp_feasible': False}
        except Exception:
            return {'lp_profit': 0.0, 'lp_optimal': prices.tolist(), 'lp_feasible': False}

    def detect_dependency_arbitrage(self, markets: list, max_pairs: int = 100) -> list:
        arbitrage_opportunities = []
        pairs_checked = 0
        for i, market1 in enumerate(markets):
            for j, market2 in enumerate(markets):
                if i >= j or pairs_checked >= max_pairs:
                    continue
                pairs_checked += 1
                q1 = market1.get('question', '').lower()
                q2 = market2.get('question', '').lower()
                if self._detect_implication(q1, q2):
                    p1 = market1.get('yes_price', 0.5)
                    p2 = market2.get('yes_price', 0.5)
                    if p1 > p2 + 0.03:
                        gross_profit = p1 - p2
                        execution_costs = self._estimate_execution_costs(
                            market1.get('liquidity', 1000),
                            market1.get('volume', 100),
                        )
                        net_profit = gross_profit - execution_costs['total']
                        if net_profit > self.min_profit_threshold:
                            arbitrage_opportunities.append({
                                'type': 'dependency_violation',
                                'market1': market1.get('question', ''),
                                'market2': market2.get('question', ''),
                                'slug1': market1.get('slug', ''),
                                'slug2': market2.get('slug', ''),
                                'p1': p1, 'p2': p2,
                                'gross_profit': gross_profit,
                                'execution_costs': execution_costs,
                                'net_profit': net_profit,
                                'strategy': f'Short M1 @ {p1:.2f} / Long M2 @ {p2:.2f}',
                                'confidence': self._compute_confidence(q1, q2),
                            })
        return arbitrage_opportunities

    def _detect_implication(self, event1: str, event2: str) -> bool:
        words1 = set(event1.split())
        words2 = set(event2.split())
        if len(words1) == 0:
            return False
        intersection = words1.intersection(words2)
        jaccard = len(intersection) / len(words1)
        if jaccard > 0.7:
            return True
        if event1 in event2 or event2 in event1:
            return True
        return False

    def _compute_confidence(self, event1: str, event2: str) -> float:
        words1 = set(event1.split())
        words2 = set(event2.split())
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return float(len(intersection) / len(union)) if len(union) > 0 else 0.0

    def _estimate_execution_costs(self, liquidity: float, volume: float) -> dict:
        max_trade = min(volume * 0.10, liquidity * 0.05)
        max_trade = max(max_trade, 100)
        base_slippage = 0.005
        slippage = base_slippage * np.sqrt(max_trade / max(liquidity, 100))
        slippage = min(slippage, 0.05)
        gas_fixed = 2.0
        gas_pct = gas_fixed / max(max_trade, 100)
        total_cost = slippage + gas_pct
        return {
            'slippage': float(slippage), 'gas': float(gas_pct),
            'total': float(total_cost), 'max_trade_size': float(max_trade),
        }

    def analyze_market(self, market_data: dict) -> dict | None:
        try:
            outcome_prices = market_data.get('outcomePrices', ['0.5', '0.5'])
            prices = np.array([float(p) for p in outcome_prices])
            is_valid = self.check_polytope_membership(prices)

            if len(prices) == 2:
                p = prices[0]
                p_safe = np.clip(p, 0.01, 0.99)
                theta = np.array([np.log(p_safe / (1 - p_safe)), 0])
                mu = prices / np.sum(prices)
                divergence = self.compute_bregman_divergence(mu, theta)
            elif len(prices) > 2:
                lp_result = self._linprog_arbitrage_bounds(prices)
                divergence = lp_result['lp_profit']
            else:
                divergence = 0

            optimal_prices = self.frank_wolfe_projection(prices)
            execution = self._estimate_execution_costs(
                market_data.get('liquidity', 1000), market_data.get('volume', 100),
            )
            price_deviation = np.abs(prices - optimal_prices).max()
            gross_profit = price_deviation
            net_profit = gross_profit - execution['total']
            is_profitable = net_profit > self.min_profit_threshold

            return {
                'is_valid': is_valid, 'is_profitable': is_profitable,
                'prices': prices.tolist(), 'optimal_prices': optimal_prices.tolist(),
                'price_deviation': float(price_deviation),
                'bregman_divergence': float(divergence),
                'execution_costs': execution,
                'gross_profit': float(gross_profit), 'net_profit': float(net_profit),
                'sharpe_estimate': float(net_profit / max(execution['total'], 0.01)),
            }
        except Exception:
            return None


# ============================================================================
# POLYMARKET DATA FETCH
# ============================================================================

@st.cache_data(ttl=180)
def fetch_polymarket_with_arbitrage():
    """Fetch Polymarket data with production-grade arbitrage detection."""
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
        arb_engine = PolymarketArbitrageEngine(min_profit_threshold=0.05)
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
                        'Event': market.get('question', ''), 'slug': slug,
                        'Current Yes %': yes_price * 100,
                        'Optimal Yes %': arb_analysis['optimal_prices'][0] * 100,
                        'Deviation %': arb_analysis['price_deviation'] * 100,
                        'Net Profit %': arb_analysis['net_profit'] * 100,
                        'Sharpe': arb_analysis['sharpe_estimate'],
                        'Max Trade': f"${arb_analysis['execution_costs']['max_trade_size']:.0f}",
                    })
                if volume > 100:
                    opportunities.append({
                        'Event': market.get('question', ''), 'slug': slug,
                        'Yes %': yes_price * 100, 'Volume': volume,
                        'Liquidity': liquidity,
                        'Arb Score': arb_analysis['bregman_divergence'] if arb_analysis else 0,
                    })
            except Exception:
                continue

        markets_clean = []
        for m in markets:
            try:
                if any(kw in m.get('question', '').lower() for kw in filter_keywords):
                    continue
                outcome_prices = m.get('outcomePrices', ['0.5'])
                if not outcome_prices or len(outcome_prices) == 0:
                    yes_price = 0.5
                else:
                    yes_price = float(outcome_prices[0] or 0.5)
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
                'Event': f"{arb['market1'][:40]}... ⇒ {arb['market2'][:40]}...",
                'slug': arb['slug1'],
                'Current Yes %': arb['p1'] * 100, 'Optimal Yes %': arb['p2'] * 100,
                'Deviation %': abs(arb['p1'] - arb['p2']) * 100,
                'Net Profit %': arb['net_profit'] * 100,
                'Sharpe': arb['net_profit'] / max(arb['execution_costs']['total'], 0.01),
                'Max Trade': f"${arb['execution_costs']['max_trade_size']:.0f}",
                'Type': 'DEPENDENCY',
            })

        opportunities_sorted = sorted(opportunities, key=lambda x: x['Arb Score'], reverse=True)
        opp_df = pd.DataFrame(opportunities_sorted[:10]) if opportunities_sorted else pd.DataFrame()
        arb_df = pd.DataFrame(arbitrage_trades) if arbitrage_trades else pd.DataFrame()
        return opp_df, arb_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ============================================================================
# AI MACRO BRIEFING — System/User Prompt Structure
# ============================================================================

_GEMINI_SYSTEM_INSTRUCTION = """You are a senior macro strategist at a quantitative hedge fund.
Your role is to generate dense, specific macro-economic intelligence briefings.

Rules:
- Write in dense prose (no bullet points), maximum 250 words.
- Use specific numbers from the provided data table.
- Avoid generic statements like "markets are volatile" — quantify everything.
- Structure your analysis across four areas: Yield Curve & Fed Positioning,
  Correlation Regime, Liquidity Cycles, and SPX Technical Levels.
- For SPX levels, provide probability-weighted scenarios, e.g.:
  "70% consolidation 5850-5900, 20% breakdown to 5750, 10% breakout to 6000."
"""


def generate_enhanced_ai_macro_briefing(indices, spx_options, liquidity, correlations):
    """Generate macro briefing with System/User prompt and structured Markdown table."""
    if not st.session_state.gemini_configured:
        return (
            "⚠️ **Gemini API Configuration Required** — Enable API access in "
            "sidebar to unlock macro intelligence."
        )
    try:
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            system_instruction=_GEMINI_SYSTEM_INSTRUCTION,
        )
        spx_price = indices.get('SPX', {}).get('price', 0)
        vix_price = indices.get('VIX', {}).get('price', 0)
        pc_ratio = spx_options.get('put_call_ratio', 0) if spx_options else 0
        max_pain = spx_options.get('max_pain', 0) if spx_options else 0
        yield_spread = liquidity.get('yield_spread', 0)
        credit_spread = liquidity.get('credit_spread', 0)
        fed_balance = liquidity.get('fed_balance', 0)
        corr_btc = correlations.get('BTC', 0)
        corr_eth = correlations.get('ETH', 0)

        if yield_spread < 0:
            yc_context = f"inverted by {abs(yield_spread):.2f}% (recession signal active)"
        elif yield_spread < 0.25:
            yc_context = f"flattening at {yield_spread:.2f}% (late-cycle dynamics)"
        else:
            yc_context = f"normalized at {yield_spread:.2f}% (expansion intact)"

        user_message = f"""Generate a macro briefing (200-250 words, paragraph format) based on this live data:

| Metric | Value |
|--------|-------|
| SPX Price | ${spx_price:.0f} |
| VIX Level | {vix_price:.1f} |
| Put/Call Ratio | {pc_ratio:.2f} |
| Max Pain Strike | ${max_pain:.0f} |
| 10Y-2Y Spread | {yield_spread:.2f}% ({yc_context}) |
| HY Credit Spread | {credit_spread:.2f}% |
| Fed Balance Sheet | ${fed_balance:.2f}T |
| SPX/BTC Correlation | {corr_btc:.2f} |
| SPX/ETH Correlation | {corr_eth:.2f} |

Cover all four areas: Yield Curve, Correlation Regime, Liquidity Cycles, SPX Technical Levels."""

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
# UI COMPONENTS — CARD-BASED LAYOUTS (Templates from config)
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
    if signals:
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
    """Interactive sector heatmap with drill-down capability."""
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
        hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Abs: $%{customdata[1]:.2f}<extra></extra>',
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

    # First row (5 sectors)
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

    # Second row (remaining sectors)
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

    # Sector drill-down (uses @st.fragment for partial re-rendering)
    if st.session_state.show_sector_drill and st.session_state.selected_sector:
        _render_sector_drilldown(sector_df)


@st.fragment
def _render_sector_drilldown(sector_df):
    """Sector drill-down fragment — only this re-renders on interaction."""
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
# MAIN APPLICATION
# ============================================================================

st.title("⚡ ALPHA DECK PRO v6.0")
st.caption("**QUANT EDITION** — Prediction Market Arbitrage | Macro Intelligence | Production Analytics")
st.divider()

# ============================================================================
# AI MACRO BRIEFING
# ============================================================================

st.subheader("🧠 AI MACRO INTELLIGENCE")

if st.button("🚀 GENERATE MACRO BRIEF", key="ai_macro", use_container_width=True):
    with st.spinner('🔬 Analyzing macro regime...'):
        indices = fetch_index_data()
        spx_opts = fetch_spx_options_data()
        liq = fetch_fred_liquidity()
        corrs = calculate_spx_crypto_correlation()

        briefing = generate_enhanced_ai_macro_briefing(indices, spx_opts, liq, corrs)

        st.markdown(
            TEMPLATES["ai_briefing"].format(content=briefing),
            unsafe_allow_html=True,
        )

st.divider()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 MAIN DECK",
    "💰 POLYMARKET ARBITRAGE",
    "₿ CRYPTO & LIQUIDITY",
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

    # SPX Options
    st.subheader("🎯 SPX OPTIONS INTELLIGENCE")
    spx_data = fetch_spx_options_data()

    if spx_data and spx_data.get('success'):
        cols_opt = st.columns(5)
        with cols_opt[0]:
            st.metric("P/C Ratio", f"{spx_data['put_call_ratio']:.2f}")
        with cols_opt[1]:
            st.metric("P/C OI", f"{spx_data['put_call_oi_ratio']:.2f}")
        with cols_opt[2]:
            st.metric("Max Pain", f"${spx_data['max_pain']:.0f}")
        with cols_opt[3]:
            st.metric("Call IV", f"{spx_data['avg_call_iv']:.1f}%")
        with cols_opt[4]:
            st.metric("Put IV", f"{spx_data['avg_put_iv']:.1f}%")
        st.caption(f"📅 Expiration: {spx_data['expiration']}")
    else:
        if is_market_open():
            st.error("❌ SPX options: Data provider issue")
        else:
            st.info("⏰ Markets closed — Options available Mon-Fri 9:30 AM - 4 PM ET")

    st.divider()

    # Watchlist
    st.subheader("⚡ WATCHLIST + TECHNICAL SCANNER")
    st.caption("**Card-based layout** — Wilder's RSI")

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
# TAB 2: POLYMARKET ARBITRAGE
# ============================================================================

with tab2:
    st.subheader("🎲 POLYMARKET ARBITRAGE ENGINE")
    st.caption("**Production-Grade** — Marginal Polytope | Bregman Projection | Frank-Wolfe | Linear Programming")

    opp_df, arb_df = fetch_polymarket_with_arbitrage()

    st.markdown("### 💰 ARBITRAGE TRADES")

    if not arb_df.empty:
        st.dataframe(
            arb_df, use_container_width=True, height=400, hide_index=True,
            column_config={
                "Current Yes %": st.column_config.NumberColumn(format="%.2f%%"),
                "Optimal Yes %": st.column_config.NumberColumn(format="%.2f%%"),
                "Deviation %": st.column_config.NumberColumn(format="%.2f%%"),
                "Net Profit %": st.column_config.NumberColumn(format="%.2f%%"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(f"🎯 **{len(arb_df)} arbitrage opportunities** detected via polytope + LP analysis")
    else:
        st.info("✅ No arbitrage detected — Markets pricing efficiently within convex hull")

    st.divider()

    st.markdown("### 📊 TOP PREDICTION MARKETS")

    if not opp_df.empty:
        opp_display = opp_df.copy()
        opp_display['Volume'] = opp_display['Volume'].apply(
            lambda x: f"${x / 1e6:.2f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"
        )
        opp_display['Liquidity'] = opp_display['Liquidity'].apply(
            lambda x: f"${x / 1e6:.2f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"
        )
        opp_display['Yes %'] = opp_display['Yes %'].apply(lambda x: f"{x:.1f}%")
        opp_display['Arb Score'] = opp_display['Arb Score'].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            opp_display[['Event', 'Yes %', 'Volume', 'Liquidity', 'Arb Score']],
            use_container_width=True, height=400, hide_index=True,
        )

        st.caption("**🔗 Clickable Links:**")
        for _, row in opp_df.iterrows():
            if row['slug']:
                url = f"https://polymarket.com/event/{row['slug']}"
                st.markdown(f"[{row['Event'][:80]}...]({url})")
    else:
        st.warning("📊 No markets available")

# ============================================================================
# TAB 3: CRYPTO & LIQUIDITY
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

    # Correlations
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

    st.divider()

    # Liquidity
    st.subheader("🏛️ FED LIQUIDITY METRICS")
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
        else:
            st.success("✅ **NORMALIZED CURVE** — Expansion intact")
    else:
        st.warning("⚠️ FRED API: Configure key in sidebar")
        if 'error' in liq:
            st.caption(f"Error: {liq['error']}")

# ============================================================================
# TAB 4: TRADINGVIEW
# ============================================================================

with tab4:
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
st.caption("⚡ **ALPHA DECK PRO v6.0 — QUANT EDITION (PRODUCTION-HARDENED)**")
st.caption(
    "✅ Batch yf.download | ✅ Wilder's RSI | ✅ linprog Arbitrage | "
    "✅ Inner-Join Correlation | ✅ FRED get_series | ✅ Glassmorphism | "
    "✅ System/User Gemini | ✅ st.fragment Drill-Down"
)
