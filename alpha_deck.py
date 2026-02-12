"""
Alpha Deck PRO v4.0 - GOLD MASTER (ENHANCED)
Bloomberg-Style Trading Terminal - All Issues Resolved, institutional upgrades added
"""

import os
import time
import pickle
import logging
import asyncio

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

# Optional async HTTP client
try:
    import httpx
    HTTPX_AVAILABLE = True
except Exception:
    HTTPX_AVAILABLE = False

# Optional Redis
try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

# Optional fred api
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

# Optional google generative ai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Optional market calendars for holiday-aware open hours
try:
    import pandas_market_calendars as mcal
    MARKET_CAL_AVAILABLE = True
except Exception:
    MARKET_CAL_AVAILABLE = False

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AlphaDeckPRO")

# ============================================================
# Redis client, configured from REDIS_URL environment variable
# ============================================================
redis_client = None
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_AVAILABLE and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL)
        logger.info("Redis connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None
else:
    if REDIS_AVAILABLE:
        logger.info("REDIS_URL not set, skipping Redis")
    else:
        logger.info("redis package not available, skipping Redis")

def redis_cache_get(key):
    try:
        if not redis_client:
            return None
        raw = redis_client.get(key)
        if not raw:
            return None
        return pickle.loads(raw)
    except Exception as e:
        logger.error(f"redis_cache_get error for {key}: {e}")
        return None

def redis_cache_set(key, value, ttl):
    try:
        if not redis_client:
            return None
        redis_client.setex(key, ttl, pickle.dumps(value))
    except Exception as e:
        logger.error(f"redis_cache_set error for {key}: {e}")

# ============================================================
# PAGE CONFIG & BACKGROUND REFRESH
# ============================================================
st.set_page_config(
    page_title="Alpha Deck PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lightweight background refresh every 60 seconds
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# If you prefer st.experimental_rerun, you can replace logic, here we re-run when > 60s
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    try:
        st.experimental_rerun()
    except Exception:
        # older/newer Streamlit, fallback to rerun
        try:
            st.rerun()
        except Exception:
            pass

# Initialize session state
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'

# ============================================================
# SIDEBAR - API CONFIGURATION, environment variable fallback only
# ============================================================
st.sidebar.title("🔑 API CONFIGURATION")
st.sidebar.caption("Enter your API keys, they are used only in session, or set via environment variables")

# Do not hardcode keys, prefer user input or environment variables
gemini_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value="",
    type="password",
    help="You can also set the GEMINI_API_KEY environment variable"
)

fred_key_input = st.sidebar.text_input(
    "FRED API Key",
    value="",
    type="password",
    help="You can also set the FRED_API_KEY environment variable"
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Keys are stored in session only")
st.sidebar.caption("🔒 Never shared or saved")

# Resolve keys, prefer typed key, otherwise environment
GEMINI_API_KEY = gemini_key_input.strip() if gemini_key_input else os.getenv("GEMINI_API_KEY")
FRED_API_KEY = fred_key_input.strip() if fred_key_input else os.getenv("FRED_API_KEY")

# Configure Gemini safely
gemini_configured = False
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        st.sidebar.success("✅ Gemini Connected")
        logger.info("Gemini configured")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini Error: {str(e)[:80]}")
        logger.error(f"Gemini configuration error: {e}")
else:
    st.sidebar.warning("⚠️ Gemini: No API Key or library missing")
    if not GEMINI_AVAILABLE:
        logger.info("google.generativeai not installed")

# Configure FRED safely
fred = None
if FRED_AVAILABLE and FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
        st.sidebar.success("✅ FRED Connected")
        logger.info("FRED configured")
    except Exception as e:
        st.sidebar.error(f"❌ FRED Error: {str(e)[:80]}")
        logger.error(f"FRED configuration error: {e}")
else:
    st.sidebar.warning("⚠️ FRED: No API Key or library missing")
    if not FRED_AVAILABLE:
        logger.info("fredapi not installed")

# ============================================================
# REFINED AMBER TERMINAL THEME CSS, sidebar toggle fixes
# ============================================================
st.markdown("""
    <style>
    /* Pure Black Background */
    .stApp {
        background-color: #000000 !important;
    }
    
    .main, .block-container, section {
        background-color: #000000 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 2px solid #FFB000 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FFB000 !important;
    }

    /* Force sidebar toggle to be visible and colored */
    [data-testid="collapsedControl"], button[title="Toggle sidebar"], button[aria-label="Toggle sidebar"], button[title="Toggle navigation"], button[aria-label="Toggle navigation"] {
        display: block !important;
        position: fixed !important;
        left: 8px !important;
        top: 8px !important;
        z-index: 9999 !important;
        background: transparent !important;
        color: #FFB000 !important;
        border: none !important;
        padding: 6px !important;
        border-radius: 4px !important;
    }
    
    /* Ensure SVG icons inside the button are visible */
    [data-testid="collapsedControl"] svg path,
    button[title="Toggle sidebar"] svg path,
    button[aria-label="Toggle sidebar"] svg path,
    button[title="Toggle navigation"] svg path,
    button[aria-label="Toggle navigation"] svg path {
        fill: #FFB000 !important;
        stroke: #FFB000 !important;
    }
    
    /* Terminal Font for Tables */
    .stDataFrame, .dataframe, table {
        font-family: 'Courier New', Courier, monospace !important;
        background-color: #000000 !important;
        color: #FFB000 !important;
    }
    
    /* Metrics - Amber Theme */
    .stMetric {
        background-color: #000000 !important;
        border: 1px solid #FFB000 !important;
        padding: 15px;
        border-radius: 0px;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    .stMetric label {
        color: #FFB000 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        font-family: 'Courier New', Courier, monospace !important;
        text-transform: uppercase;
    }
    
    .stMetric .metric-value {
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* CRITICAL TAB STYLING FIX */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000000 !important;
        border-bottom: 2px solid #FFB000 !important;
    }
    
    /* Unselected tabs: Amber text on black */
    .stTabs [data-baseweb="tab"] {
        background-color: #000000 !important;
        color: #FFB000 !important;
        border: 1px solid #FFB000 !important;
        border-radius: 0px;
        padding: 10px 20px;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
    }
    
    /* SELECTED TAB: Amber background with BLACK text */
    .stTabs [aria-selected="true"] {
        background-color: #FFB000 !important;
        color: #000000 !important;
    }
    
    .stTabs [aria-selected="true"] p {
        color: #000000 !important;
    }
    
    /* Headers - Amber */
    h1, h2, h3, h4 {
        color: #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    
    /* Text - White/Amber */
    p, span, div, label {
        color: #FFFFFF !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Links - Amber */
    a {
        color: #FFB000 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #FFC933 !important;
        text-decoration: underline !important;
    }
    
    /* Dividers */
    hr {
        border-color: #FFB000 !important;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #FFB000 !important;
        border: 2px solid #FFB000 !important;
        border-radius: 0px !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        padding: 10px 30px;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background-color: #FFB000 !important;
        color: #000000 !important;
    }
    
    /* Captions */
    .stCaption {
        color: #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background-color: #1a1a1a !important;
        color: #FFB000 !important;
        border: 1px solid #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #000000 !important;
        color: #FFB000 !important;
        border: 1px solid #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def is_market_open():
    """Check if US stock market is currently open, with optional holiday-aware check"""
    try:
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)
        # If pandas_market_calendars available, use NYSE schedule for holiday awareness
        if MARKET_CAL_AVAILABLE:
            try:
                nyse = mcal.get_calendar('NYSE')
                schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
                if schedule.empty:
                    return False
                # Now check local time range
                open_time = schedule.iloc[0]['market_open'].tz_convert(ny_tz)
                close_time = schedule.iloc[0]['market_close'].tz_convert(ny_tz)
                return open_time <= now <= close_time
            except Exception as e:
                logger.warning(f"Market calendar check failed, fallback to simple time window: {e}")
        # Fallback simple check: weekdays 9:30-16:00 ET
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception as e:
        logger.error(f"is_market_open error: {e}")
        return False

def get_tradingview_symbol(ticker):
    """Convert Yahoo ticker to TradingView symbol"""
    mapping = {
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
        'MSTR': 'NASDAQ:MSTR'
    }
    return mapping.get(ticker, f"NASDAQ:{ticker}")

def check_technical_signals(row):
    """Generate technical signals for watchlist"""
    signals = []
    try:
        rsi = float(row.get('RSI', 50))
        change_pct = float(row.get('Change %', 0))
        volume_str = str(row.get('Volume', '0M'))
        volume = float(volume_str.replace('M', '')) if 'M' in volume_str else 0
        if rsi < 30:
            signals.append("🟢 OVERSOLD")
        elif rsi > 70:
            signals.append("🔴 OVERBOUGHT")
        if volume > 20:
            signals.append("⚡ HIGH VOL")
        if abs(change_pct) > 3:
            signals.append("🚀 MOMENTUM")
        return " | ".join(signals) if signals else "—"
    except Exception as e:
        logger.error(f"check_technical_signals error: {e}")
        return "—"

# ============================================================
# DATA FETCHING & ANALYTICS - with Redis caching wrapper
# ============================================================
def redis_cache_wrapper(key, ttl, fetch_func, force_local=False):
    """Wrapper that uses Redis if available, otherwise calls fetch_func"""
    try:
        if redis_client and not force_local:
            cached = redis_cache_get(key)
            if cached is not None:
                return cached
            data = fetch_func()
            # Only cache serializable results
            try:
                redis_cache_set(key, data, ttl)
            except Exception as e:
                logger.error(f"redis_cache_wrapper set failed for {key}: {e}")
            return data
        else:
            return fetch_func()
    except Exception as e:
        logger.error(f"redis_cache_wrapper error for {key}: {e}")
        try:
            return fetch_func()
        except Exception as ex:
            logger.error(f"fallback fetch failed for {key}: {ex}")
            return None

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data WITH absolute change"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        if hist.empty:
            return {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_abs = current_price - prev_close
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
        return {
            'price': float(current_price),
            'change_pct': float(change_pct),
            'change_abs': float(change_abs),
            'volume': int(volume),
            'success': True
        }
    except Exception as e:
        logger.error(f"fetch_ticker_data_reliable error for {ticker}: {e}")
        return {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}

@st.cache_data(ttl=60)
def calculate_rsi(prices, period=14):
    """Calculate RSI, safe against division by zero"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
    except Exception as e:
        logger.error(f"calculate_rsi error: {e}")
        return 50.0

@st.cache_data(ttl=60)
def fetch_watchlist_data(tickers):
    """Fetch multiple tickers with technical signals"""
    def _fetch():
        results = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1mo')
                if hist.empty:
                    continue
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                rsi_value = calculate_rsi(hist['Close'])
                row_data = {
                    'Ticker': ticker,
                    'Price': float(current_price),
                    'Change %': float(change_pct),
                    'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                    'RSI': float(rsi_value)
                }
                row_data['Signals'] = check_technical_signals(row_data)
                results.append(row_data)
            except Exception as e:
                logger.warning(f"fetch_watchlist_data error for {ticker}: {e}")
                continue
        return pd.DataFrame(results)
    return redis_cache_wrapper("watchlist_data", 60, _fetch)

@st.cache_data(ttl=60)
def fetch_vix_term_structure():
    """Fetch VIX term structure for volatility analysis"""
    try:
        vix = fetch_ticker_data_reliable('^VIX')
        vix9d = fetch_ticker_data_reliable('^VIX9D')
        vix3m = fetch_ticker_data_reliable('^VIX3M')
        return {
            'VIX': vix['price'],
            'VIX9D': vix9d['price'],
            'VIX3M': vix3m['price'],
            'backwardation': (vix9d['price'] > vix['price']) if (vix9d.get('success') and vix.get('success')) else False
        }
    except Exception as e:
        logger.error(f"fetch_vix_term_structure error: {e}")
        return {'VIX': 0, 'VIX9D': 0, 'VIX3M': 0, 'backwardation': False}

@st.cache_data(ttl=60)
def fetch_crypto_metrics(cryptos):
    """Fetch crypto data as metrics"""
    def _fetch():
        results = {}
        for crypto_symbol in cryptos:
            ticker = f"{crypto_symbol}-USD"
            data = fetch_ticker_data_reliable(ticker)
            results[crypto_symbol] = data
        return results
    return redis_cache_wrapper("crypto_metrics_" + "_".join(cryptos), 60, _fetch)

@st.cache_data(ttl=60)
def fetch_index_data():
    """Fetch major indices with absolute change"""
    def _fetch():
        indices = {
            'SPX': '^GSPC',
            'NDX': '^NDX',
            'VIX': '^VIX',
            'HYG': 'HYG',
            'US10Y': '^TNX',
            'DXY': 'DX-Y.NYB'
        }
        results = {}
        for name, ticker in indices.items():
            try:
                data = fetch_ticker_data_reliable(ticker)
                results[name] = data
            except Exception as e:
                logger.error(f"fetch_index_data error for {ticker}: {e}")
                results[name] = {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}
        return results
    return redis_cache_wrapper("index_data", 60, _fetch)

@st.cache_data(ttl=60)
def fetch_sector_performance():
    """Fetch sector ETF performance"""
    def _fetch():
        sectors = {
            'XLK': 'Technology',
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Disc.',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication',
            'XLU': 'Utilities'
        }
        results = []
        for ticker, name in sectors.items():
            try:
                data = fetch_ticker_data_reliable(ticker)
                if data.get('success'):
                    results.append({
                        'Sector': name,
                        'Change %': data['change_pct']
                    })
            except Exception as e:
                logger.warning(f"fetch_sector_performance error for {ticker}: {e}")
                continue
        df = pd.DataFrame(results)
        return df.sort_values('Change %', ascending=False) if not df.empty else df
    return redis_cache_wrapper("sector_performance", 60, _fetch)

# ============================================================
# ASYNC RSS and POLYMARKET fetchers with caching
# ============================================================
async def _fetch_url_async(client, url, params=None, timeout=15):
    try:
        r = await client.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error(f"_fetch_url_async failed for {url}: {e}")
        return None

def fetch_news_feeds():
    """Fetch RSS news feeds asynchronously if httpx is available, fallback to feedparser.parse on URLs"""
    async def _fetch_all():
        feeds = {
            'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
            'WSJ': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'
        }
        results = []
        if HTTPX_AVAILABLE:
            async with httpx.AsyncClient() as client:
                tasks = [ _fetch_url_async(client, url) for url in feeds.values() ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for (source, url), content in zip(feeds.items(), responses):
                    try:
                        if isinstance(content, Exception) or content is None:
                            continue
                        feed = feedparser.parse(content)
                        for entry in feed.entries[:5]:
                            results.append({
                                'Source': source,
                                'Title': entry.get('title', '')[:200],
                                'Link': entry.get('link', '')
                            })
                    except Exception as e:
                        logger.error(f"fetch_news_feeds parse error for {source}: {e}")
                        continue
        else:
            # synchronous fallback
            for source, url in {
                'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
                'WSJ': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'
            }.items():
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:5]:
                        results.append({
                            'Source': source,
                            'Title': entry.get('title', '')[:200],
                            'Link': entry.get('link', '')
                        })
                except Exception as e:
                    logger.error(f"fetch_news_feeds sync parse error for {source}: {e}")
                    continue
        return results[:15]

    try:
        key = "news_feeds_v1"
        cached = redis_cache_get(key)
        if cached is not None:
            return cached
        if HTTPX_AVAILABLE:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(_fetch_all())
            loop.close()
        else:
            res = asyncio.run(_fetch_all())
        redis_cache_set(key, res, 60)
        return res
    except Exception as e:
        logger.error(f"fetch_news_feeds error: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_insider_cluster_buys():
    """Scrape OpenInsider with robust handling"""
    try:
        url = "http://openinsider.com/latest-cluster-buys"
        tables = pd.read_html(url, header=0)
        if not tables:
            return None
        df = tables[0]
        if 'Ticker' in df.columns:
            columns_to_keep = []
            for col in ['Ticker', 'Company Name', 'Insider Name', 'Title',
                       'Trade Type', 'Price', 'Qty', 'Value', 'Trade Date']:
                if col in df.columns:
                    columns_to_keep.append(col)
            df = df[columns_to_keep].head(10)
            return df
        return None
    except Exception as e:
        logger.error(f"fetch_insider_cluster_buys error: {e}")
        return None

# ============================================================
# FRED LIQUIDITY
# ============================================================
@st.cache_data(ttl=3600)
def fetch_fred_liquidity():
    if not fred:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }
    try:
        t10y2y = fred.get_series_latest_release('T10Y2Y')
        yield_spread = float(t10y2y.iloc[-1]) if not t10y2y.empty else 0
        hy_spread = fred.get_series_latest_release('BAMLH0A0HYM2')
        credit_spread = float(hy_spread.iloc[-1]) if not hy_spread.empty else 0
        fed_assets = fred.get_series_latest_release('WALCL')
        fed_balance = float(fed_assets.iloc[-1]) if not fed_assets.empty else 0
        return {
            'yield_spread': yield_spread,
            'credit_spread': credit_spread,
            'fed_balance': fed_balance / 1000,
            'success': True
        }
    except Exception as e:
        logger.error(f"fetch_fred_liquidity error: {e}")
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }

# ============================================================
# SPX options, hardened and multi-ticker fallback, with skew and gamma
# ============================================================
@st.cache_data(ttl=60)
def fetch_spx_options_data():
    """Fetch SPX/SPY options with robust fallbacks, returns structured dict"""
    try:
        candidates = ["^GSPC", "^SPX", "SPY"]
        last_err = None
        for t in candidates:
            try:
                opt_ticker = yf.Ticker(t)
                expirations = getattr(opt_ticker, "options", []) or []
                if not expirations:
                    logger.info(f"No expirations for {t}, trying next candidate")
                    continue
                nearest_exp = expirations[0]
                opt_chain = opt_ticker.option_chain(nearest_exp)
                calls = getattr(opt_chain, "calls", None) or opt_chain[0]
                puts = getattr(opt_chain, "puts", None) or opt_chain[1]
                # if both empty, skip
                if (calls is None or calls.empty) and (puts is None or puts.empty):
                    logger.info(f"Empty option chain for {t} at {nearest_exp}")
                    continue
                calls = calls.fillna(0)
                puts = puts.fillna(0)
                # safe metrics
                total_call_volume = int(calls['volume'].fillna(0).sum()) if 'volume' in calls.columns else 0
                total_put_volume = int(puts['volume'].fillna(0).sum()) if 'volume' in puts.columns else 0
                put_call_ratio = (total_put_volume / total_call_volume) if total_call_volume > 0 else 0
                total_call_oi = int(calls['openInterest'].fillna(0).sum()) if 'openInterest' in calls.columns else 0
                total_put_oi = int(puts['openInterest'].fillna(0).sum()) if 'openInterest' in puts.columns else 0
                put_call_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0
                # max pain calculation
                try:
                    calls_oi = calls.groupby('strike')['openInterest'].sum() if 'openInterest' in calls.columns else pd.Series(dtype=float)
                    puts_oi = puts.groupby('strike')['openInterest'].sum() if 'openInterest' in puts.columns else pd.Series(dtype=float)
                    total_oi = calls_oi.add(puts_oi, fill_value=0)
                    max_pain = float(total_oi.idxmax()) if not total_oi.empty else 0
                except Exception:
                    max_pain = 0
                # avg IV
                try:
                    avg_call_iv = float(calls['impliedVolatility'].replace([np.inf, -np.inf], np.nan).dropna().mean() * 100) if 'impliedVolatility' in calls.columns else 0
                    avg_put_iv = float(puts['impliedVolatility'].replace([np.inf, -np.inf], np.nan).dropna().mean() * 100) if 'impliedVolatility' in puts.columns else 0
                except Exception:
                    avg_call_iv = 0
                    avg_put_iv = 0
                res = {
                    'success': True,
                    'ticker': t,
                    'expiration': nearest_exp,
                    'put_call_ratio': put_call_ratio,
                    'put_call_oi_ratio': put_call_oi_ratio,
                    'max_pain': max_pain,
                    'avg_call_iv': avg_call_iv,
                    'avg_put_iv': avg_put_iv,
                    'calls': calls,
                    'puts': puts,
                    'total_call_volume': total_call_volume,
                    'total_put_volume': total_put_volume
                }
                return res
            except Exception as e:
                last_err = e
                logger.warning(f"fetch_spx_options_data candidate {t} failed: {e}")
                continue
        logger.error(f"fetch_spx_options_data all candidates failed, last error: {last_err}")
        return {'success': False, 'error': str(last_err), 'checked': candidates}
    except Exception as e:
        logger.error(f"fetch_spx_options_data unexpected error: {e}")
        return {'success': False, 'error': str(e)}

def calculate_skew(calls, puts):
    """Calculate simple skew proxy, difference between average OTM put IV and OTM call IV"""
    try:
        if calls is None or puts is None or calls.empty or puts.empty:
            return 0.0
        # get atm strike approx as median strike
        strikes = pd.concat([calls['strike'], puts['strike']])
        atm = float(strikes.median())
        otm_puts = puts[puts['strike'] < atm]
        otm_calls = calls[calls['strike'] > atm]
        # calculate mean implied volatility in decimal, fallback to 0
        put_iv = float(otm_puts['impliedVolatility'].replace([np.inf, -np.inf], np.nan).dropna().mean()) if 'impliedVolatility' in otm_puts.columns and not otm_puts.empty else 0.0
        call_iv = float(otm_calls['impliedVolatility'].replace([np.inf, -np.inf], np.nan).dropna().mean()) if 'impliedVolatility' in otm_calls.columns and not otm_calls.empty else 0.0
        skew = (put_iv - call_iv) * 100
        return float(skew)
    except Exception as e:
        logger.error(f"calculate_skew error: {e}")
        return 0.0

def calculate_gamma_exposure(options_df, spot_price):
    """Approximate gamma exposure proxy"""
    try:
        if options_df is None or options_df.empty:
            return 0.0
        df = options_df.copy()
        # proxy: openInterest weighted by inverse distance to spot, avoids division by zero
        df['distance'] = (df['strike'] - spot_price).abs() + 1.0
        df['gex_proxy'] = df['openInterest'].fillna(0) / df['distance']
        return float(df['gex_proxy'].sum())
    except Exception as e:
        logger.error(f"calculate_gamma_exposure error: {e}")
        return 0.0

# ============================================================
# POLYMARKET - async fetch, cached
# ============================================================
@st.cache_data(ttl=180)
def fetch_polymarket_advanced_analytics():
    """Fetch Polymarket with slug for linking, async if httpx available"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            'limit': 100,
            'active': 'true',
            'closed': 'false'
        }
        if HTTPX_AVAILABLE:
            async def _get():
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(url, params=params)
                    r.raise_for_status()
                    return r.json()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                markets = loop.run_until_complete(_get())
            finally:
                loop.close()
        else:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                return None
            markets = r.json()
        filter_keywords = ['nfl', 'nba', 'sport', 'gaming', 'gta', 'pop culture',
                          'music', 'twitch', 'mlb', 'nhl', 'soccer', 'football',
                          'basketball', 'celebrity', 'movie', 'ufc', 'mma', 'tennis']
        opportunities = []
        for market in markets:
            try:
                question = market.get('question', '').lower()
                if any(keyword in question for keyword in filter_keywords):
                    continue
                market_slug = market.get('slug', '')
                volume = float(market.get('volume', 0))
                volume_24h = float(market.get('volume24hr', 0))
                liquidity = float(market.get('liquidity', 0))
                outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
                try:
                    yes_price = float(outcome_prices[0])
                except:
                    yes_price = 0.5
                try:
                    no_price = float(outcome_prices[1])
                except:
                    no_price = 1 - yes_price
                total_prob = yes_price + no_price
                prob_deviation = abs(1.0 - total_prob)
                volume_velocity = (volume_24h / volume * 100) if volume > 0 else 0
                liquidity_score = liquidity / 1000
                edge_score = prob_deviation * 100
                activity_score = volume_velocity if volume_24h > 1000 else 0
                opportunity_score = (
                    (edge_score * 3) +
                    (activity_score * 2) +
                    (liquidity_score * 1)
                )
                if volume > 100:
                    opportunities.append({
                        'Event': market.get('question', ''),
                        'slug': market_slug,
                        'Yes %': yes_price * 100,
                        'Vol': volume,
                        'Score': opportunity_score
                    })
            except Exception as e:
                logger.warning(f"fetch_polymarket parsing error: {e}")
                continue
        if not opportunities:
            return None
        opportunities_sorted = sorted(opportunities, key=lambda x: x['Score'], reverse=True)
        return pd.DataFrame(opportunities_sorted[:10])
    except Exception as e:
        logger.error(f"fetch_polymarket_advanced_analytics error: {e}")
        return None

# ============================================================
# AI BRIEFING - Gemini use if available, deterministic fallback otherwise
# ============================================================
def _simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    try:
        risk = "Elevated" if vix_price > 25 or put_call_ratio > 1.2 else "Normal"
        opportunity = "Look for sector momentum plays around tech if SPX near all time highs" if spx_price > 4000 else "Scan for mean reversion setups"
        algos = "Monitor IV spikes and skew for directional bias"
        top_news = news_headlines[:3] if news_headlines else []
        return f"Risk: {risk}\nOpportunity: {opportunity}\nAlgos: {algos}\nNews: {' | '.join(top_news)}"
    except Exception as e:
        logger.error(f"_simple_briefing error: {e}")
        return "Risk: Normal\nOpportunity: Monitor market\nAlgos: Monitor IV and volume"

def generate_ai_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    """Generate AI briefing, try Gemini when available, fallback to simple briefing"""
    if not gemini_configured:
        return _simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        top_headlines = news_headlines[:3]
        prompt = f"""Act as a hedge fund manager. Analyze in 3 bullets (max 150 words):

Market: SPX ${spx_price:.0f}, VIX {vix_price:.1f}, P/C {put_call_ratio:.2f}
News: {' | '.join(top_headlines)}

Output:
1. Risk: [one sentence]
2. Opportunity: [one sentence]
3. Algos: [one sentence]"""
        response = model.generate_content(prompt)
        if not response or not getattr(response, "text", None):
            return _simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines)
        return response.text
    except Exception as e:
        logger.error(f"generate_ai_briefing error: {e}")
        return f"⚠️ AI Error, returning fallback briefing.\n\n{_simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines)}"

# ============================================================
# Service classes to separate logic
# ============================================================
class MarketDataService:
    @staticmethod
    def indices():
        return fetch_index_data()

    @staticmethod
    def ticker(ticker):
        return fetch_ticker_data_reliable(ticker)

    @staticmethod
    def watchlist(tickers):
        return fetch_watchlist_data(tickers)

class OptionsService:
    @staticmethod
    def spx_options():
        return fetch_spx_options_data()

class NewsService:
    @staticmethod
    def headlines():
        return fetch_news_feeds()

class PolymarketService:
    @staticmethod
    def opportunities():
        return fetch_polymarket_advanced_analytics()

class FredService:
    @staticmethod
    def liquidity():
        return fetch_fred_liquidity()

# ============================================================
# MAIN APPLICATION UI
# ============================================================
st.title("⚡ ALPHA DECK PRO v4.0")

# AI BRIEFING
st.subheader("🤖 AI MARKET BRIEFING")

if st.button("⚡ GENERATE MORNING BRIEF", key="ai_brief"):
    with st.spinner('Consulting AI...'):
        indices = MarketDataService.indices()
        spx_data = OptionsService.spx_options()
        news = NewsService.headlines()
        spx_price = indices.get('SPX', {}).get('price', 0)
        vix_price = indices.get('VIX', {}).get('price', 0)
        pc_ratio = spx_data.get('put_call_ratio', 0) if spx_data and spx_data.get('success') else 0
        headlines = [article['Title'] for article in news] if news else []
        briefing = generate_ai_briefing(spx_price, vix_price, pc_ratio, headlines)
        st.markdown(f"```\n{briefing}\n```")

st.divider()

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MAIN DECK", "📊 LIQUIDITY & INSIDER", "₿ CRYPTO & POLY", "📈 TRADINGVIEW"])

# TAB 1: MAIN DECK
with tab1:
    st.subheader("📡 MARKET PULSE")

    # Loading skeletons using placeholders
    placeholder_cols = st.columns(6)
    indices = MarketDataService.indices()

    # Ensure the indices dict has all keys
    for idx, name in enumerate(['SPX', 'NDX', 'VIX', 'HYG', 'US10Y', 'DXY']):
        data = indices.get(name, {'success': False})
        with placeholder_cols[idx]:
            if data.get('success'):
                if name in ['VIX', 'HYG']:
                    abs_str = f"{data['change_abs']:+.2f} pts"
                    value_str = f"{data['price']:.2f}"
                else:
                    abs_str = f"${data['change_abs']:+.2f}"
                    value_str = f"${data['price']:.2f}"
                st.metric(label=name, value=value_str, delta=f"{abs_str} ({data['change_pct']:+.2f}%)")
            else:
                st.metric(label=name, value="LOADING", delta="—")

    st.divider()

    # VOLATILITY TERM STRUCTURE
    st.subheader("📊 VOLATILITY TERM STRUCTURE")
    vix_term = fetch_vix_term_structure()
    col_vix1, col_vix2 = st.columns([3, 7])
    with col_vix1:
        if vix_term.get('backwardation'):
            st.error("⚠️ BACKWARDATION (CRASH SIGNAL)")
        else:
            st.success("✅ CONTANGO (NORMAL)")
        # Show volatility regime
        vix_val = vix_term.get('VIX', 0)
        if vix_val < 15:
            regime = "LOW VOL REGIME"
        elif 15 <= vix_val < 25:
            regime = "NORMAL VOL"
        elif 25 <= vix_val < 35:
            regime = "HIGH VOL"
        else:
            regime = "CRISIS REGIME"
        st.caption(f"Volatility Regime: {regime}")

    with col_vix2:
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=['VIX', 'VIX9D', 'VIX3M'],
            y=[vix_term.get('VIX', 0), vix_term.get('VIX9D', 0), vix_term.get('VIX3M', 0)],
            mode='lines+markers',
            line=dict(color='#FFB000', width=3),
            marker=dict(size=10, color='#FFB000')
        ))
        fig_vix.update_layout(
            title="VIX Term Structure",
            template='plotly_dark',
            height=200,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFB000', family='Courier New'),
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, title="Volatility")
        )
        st.plotly_chart(fig_vix, use_container_width=True)

    st.divider()

    # QUICK CHART SELECT
    st.subheader("🔗 QUICK CHART SELECT")
    st.caption("Click to load in TradingView tab")
    quick_cols = st.columns(4)
    quick_tickers = ['SPY', 'QQQ', 'NVDA', 'BTC-USD']
    for idx, ticker in enumerate(quick_tickers):
        with quick_cols[idx]:
            if st.button(ticker, key=f"quick_{ticker}"):
                st.session_state.selected_ticker = ticker
                st.success(f"✅ {ticker} selected")

    st.divider()

    # SPX OPTIONS - FIXED
    st.subheader("🎯 SPX OPTIONS INTELLIGENCE")
    market_is_open = is_market_open()
    ny_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_tz)
    st.caption(f"Current ET Time: {current_time.strftime('%I:%M %p')} | Day: {current_time.strftime('%A')}")
    if not market_is_open:
        st.warning("⏰ Markets Closed, options data available Mon, Tue, Wed, Thu, Fri 9:30 AM - 4:00 PM ET")

    spx_data = OptionsService.spx_options()

    if spx_data and spx_data.get('success'):
        opt_cols = st.columns(6)
        with opt_cols[0]:
            st.metric("Put/Call Ratio", f"{spx_data['put_call_ratio']:.2f}")
        with opt_cols[1]:
            st.metric("P/C OI Ratio", f"{spx_data['put_call_oi_ratio']:.2f}")
        with opt_cols[2]:
            st.metric("Max Pain", f"${spx_data['max_pain']:.0f}")
        with opt_cols[3]:
            st.metric("Avg Call IV", f"{spx_data['avg_call_iv']:.1f}%")
        with opt_cols[4]:
            st.metric("Avg Put IV", f"{spx_data['avg_put_iv']:.1f}%")
        # Compute skew and gamma
        try:
            calls_df = spx_data.get('calls')
            puts_df = spx_data.get('puts')
            skew_val = calculate_skew(calls_df, puts_df)
            spot_price = MarketDataService.indices().get('SPX', {}).get('price', 0)
            gex_calls = calculate_gamma_exposure(calls_df, spot_price)
            gex_puts = calculate_gamma_exposure(puts_df, spot_price)
            gex_net = gex_calls - gex_puts
            with opt_cols[5]:
                st.metric("Vol Skew", f"{skew_val:.2f}%")
                st.caption("Skew proxy, positive means put IV > call IV")
        except Exception as e:
            logger.error(f"SPX options analytics error: {e}")

        st.caption(f"📅 Expiration: {spx_data['expiration']} | Source: {spx_data.get('ticker')}")
        col_vol1, col_vol2 = st.columns(2)
        with col_vol1:
            fig_volume = go.Figure(data=[
                go.Bar(name='Calls', x=['Volume'], y=[spx_data['total_call_volume'] or 0], marker_color='#00FF00'),
                go.Bar(name='Puts', x=['Volume'], y=[spx_data['total_put_volume'] or 0], marker_color='#FF0000')
            ])
            fig_volume.update_layout(
                title="Call vs Put Volume",
                template='plotly_dark',
                height=250,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#FFB000', family='Courier New'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        with col_vol2:
            try:
                calls_top = spx_data['calls'].nlargest(10, 'volume') if 'calls' in spx_data and not spx_data['calls'].empty else pd.DataFrame()
                puts_top = spx_data['puts'].nlargest(10, 'volume') if 'puts' in spx_data and not spx_data['puts'].empty else pd.DataFrame()
                fig_iv = go.Figure()
                if not calls_top.empty and 'impliedVolatility' in calls_top.columns:
                    fig_iv.add_trace(go.Scatter(
                        x=calls_top['strike'],
                        y=calls_top['impliedVolatility'] * 100,
                        mode='markers',
                        name='Calls',
                        marker=dict(size=10, color='#00FF00')
                    ))
                if not puts_top.empty and 'impliedVolatility' in puts_top.columns:
                    fig_iv.add_trace(go.Scatter(
                        x=puts_top['strike'],
                        y=puts_top['impliedVolatility'] * 100,
                        mode='markers',
                        name='Puts',
                        marker=dict(size=10, color='#FF0000')
                    ))
                fig_iv.update_layout(
                    title="IV by Strike",
                    template='plotly_dark',
                    height=250,
                    xaxis_title="Strike",
                    yaxis_title="IV %",
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#FFB000', family='Courier New'),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_iv, use_container_width=True)
            except Exception as e:
                logger.error(f"SPX IV chart error: {e}")
    else:
        # structured informative error
        if spx_data and spx_data.get('error'):
            st.error("❌ SPX options data unavailable.")
            st.caption(f"Error: {spx_data.get('error')}")
            st.caption(f"Tickers checked: {spx_data.get('checked')}")
        else:
            st.error("❌ SPX options data unavailable, possible reasons:")
            st.caption("• Markets closed, weekend or holiday")
            st.caption("• Yahoo options chain blocked for index")
            st.caption("• Network connectivity")

    st.divider()

    # Watchlist
    col1, col2 = st.columns([6, 4])
    with col1:
        st.subheader("⚡ WATCHLIST + TECHNICAL SCANNER")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        with st.spinner("Loading watchlist..."):
            df = MarketDataService.watchlist(watchlist_tickers)
        if df is not None and not df.empty:
            st.dataframe(
                df,
                use_container_width=True,
                height=500,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                    "RSI": st.column_config.NumberColumn(format="%.1f")
                }
            )
        else:
            st.error("DATA UNAVAILABLE")
    with col2:
        st.subheader("🎨 SECTOR HEAT")
        with st.spinner("Loading sector performance..."):
            sector_df = fetch_sector_performance()
        if sector_df is not None and not sector_df.empty:
            fig = px.bar(
                sector_df,
                x='Change %',
                y='Sector',
                orientation='h',
                color='Change %',
                color_continuous_scale=[[0, '#FF0000'], [0.5, '#000000'], [1, '#00FF00']],
                color_continuous_midpoint=0,
                hover_data={'Change %': ':.2f'}
            )
            fig.update_traces(
                texttemplate='%{x:.2f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>'
            )
            fig.update_layout(
                template='plotly_dark',
                showlegend=False,
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#FFB000', family='Courier New'),
                xaxis=dict(showgrid=False, color='#FFB000'),
                yaxis=dict(showgrid=False, color='#FFB000')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector performance unavailable")

# TAB 2: LIQUIDITY & INSIDER
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏛️ FED LIQUIDITY METRICS")
        liquidity = FredService.liquidity()
        if liquidity and liquidity.get('success'):
            met_cols = st.columns(3)
            with met_cols[0]:
                st.metric("10Y-2Y SPREAD", f"{liquidity['yield_spread']:.2f}%")
            with met_cols[1]:
                st.metric("HY CREDIT SPREAD", f"{liquidity['credit_spread']:.2f}%")
            with met_cols[2]:
                st.metric("FED BALANCE", f"${liquidity['fed_balance']:.2f}T")
            st.markdown("---")
            if liquidity['yield_spread'] < 0:
                st.error("⚠️ INVERTED YIELD CURVE")
            else:
                st.success("✅ NORMAL YIELD CURVE")
            if liquidity['credit_spread'] > 5:
                st.error("⚠️ ELEVATED CREDIT SPREADS")
            else:
                st.success("✅ CREDIT MARKETS STABLE")
        else:
            st.error("FRED DATA UNAVAILABLE, check API key or network")

    with col2:
        st.subheader("🕵️ INSIDER CLUSTER BUYS")
        with st.spinner("Loading insider cluster buys..."):
            insider_df = fetch_insider_cluster_buys()
        if insider_df is not None and not insider_df.empty:
            st.dataframe(insider_df, use_container_width=True, height=400, hide_index=True)
        else:
            st.warning("⚠️ Insider Data: Source Blocking or unavailable")

    st.divider()
    st.subheader("📰 NEWS WIRE")
    with st.spinner("Fetching news..."):
        news = NewsService.headlines()
    if news:
        for article in news:
            st.markdown(f"**[{article['Source']}]** [{article['Title']}]({article['Link']})")
    else:
        st.info("NO NEWS AVAILABLE")

# TAB 3: CRYPTO & POLYMARKET
with tab3:
    st.subheader("₿ CRYPTO MARKET PULSE")
    with st.spinner("Loading crypto metrics..."):
        crypto_data = fetch_crypto_metrics(['BTC', 'ETH', 'SOL', 'DOGE'])
    row1 = st.columns(2)
    row2 = st.columns(2)
    crypto_order = ['BTC', 'ETH', 'SOL', 'DOGE']
    for idx, crypto in enumerate(crypto_order):
        if crypto in crypto_data and crypto_data[crypto].get('success'):
            data = crypto_data[crypto]
            if idx < 2:
                with row1[idx]:
                    st.metric(label=crypto, value=f"${data['price']:,.2f}", delta=f"{data['change_pct']:+.2f}%")
            else:
                with row2[idx - 2]:
                    st.metric(label=crypto, value=f"${data['price']:,.2f}", delta=f"{data['change_pct']:+.2f}%")

    st.divider()
    st.subheader("🎲 POLYMARKET ALPHA")
    with st.spinner("Loading Polymarket opportunities..."):
        poly_df = PolymarketService.opportunities()
    if poly_df is not None and not poly_df.empty:
        poly_df = poly_df.copy()
        poly_df['Vol'] = poly_df['Vol'].apply(lambda x: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
        poly_df['Yes %'] = poly_df['Yes %'].apply(lambda x: f"{x:.1f}%")
        poly_df['Score'] = poly_df['Score'].apply(lambda x: f"{x:.1f}")
        st.dataframe(poly_df[['Event', 'Yes %', 'Vol', 'Score']], use_container_width=True, height=400, hide_index=True)
        st.caption("**Click events to trade:**")
        for idx, row in poly_df.iterrows():
            if row['slug']:
                url = f"https://polymarket.com/event/{row['slug']}"
                st.markdown(f"[{row['Event']}]({url})")
    else:
        st.error("POLYMARKET DATA UNAVAILABLE")

# TAB 4: TRADINGVIEW
with tab4:
    st.subheader("📈 TRADINGVIEW ADVANCED CHARTS")
    all_tickers = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'BTC-USD', 'ETH-USD']
    default_idx = all_tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in all_tickers else 0
    selected_ticker = st.selectbox("SELECT TICKER", all_tickers, index=default_idx, key="chart_select")
    st.session_state.selected_ticker = selected_ticker
    if selected_ticker:
        tv_symbol = get_tradingview_symbol(selected_ticker)
        tradingview_widget = f"""
        <div class="tradingview-widget-container" style="height:100%;width:100%">
          <div id="tradingview_chart" style="height:600px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
          "width": "100%",
          "height": 600,
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
            "volume.volume.color.0": "#FF0000",
            "volume.volume.color.1": "#00FF00"
          }},
          "container_id": "tradingview_chart"
        }}
          );
          </script>
        </div>
        """
        st.components.v1.html(tradingview_widget, height=650)
        st.caption(f"📊 Symbol: {tv_symbol} | Volume + SMAs")

# FOOTER
st.divider()
st.caption("⚡ ALPHA DECK PRO v4.0 - GOLD MASTER ENHANCED")
