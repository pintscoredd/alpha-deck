"""
Alpha Deck PRO v4.0 - GOLD MASTER (ENHANCED + INTEGRATED)
All features: UI/UX, Polymarket arbitrage (Frank-Wolfe + Bregman + IP oracle),
Redis cache, async Polymarket/RSS, AI briefing, options analytics, watchlist cards,
sector drill-down treemap, TradingView embed
"""

# Core / Standard library
import os
import time
import pickle
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 3rd party
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import feedparser
import pytz

# Optional async HTTP
try:
    import httpx
    HTTPX_AVAILABLE = True
except Exception:
    HTTPX_AVAILABLE = False

# Optional aiohttp for other async fetch
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except Exception:
    AIOHTTP_AVAILABLE = False

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

# Optional integer programming solver pulp
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

# Optional scipy for linear programming fallback
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AlphaDeckPRO")

# ============================================================
# Redis client (optional)
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

def redis_cache_get(key: str):
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

def redis_cache_set(key: str, value: Any, ttl: int):
    try:
        if not redis_client:
            return None
        redis_client.setex(key, ttl, pickle.dumps(value))
    except Exception as e:
        logger.error(f"redis_cache_set error for {key}: {e}")

# ============================================================
# Streamlit page config and background refresh
# ============================================================
st.set_page_config(
    page_title="Alpha Deck PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background refresh every 60 seconds
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

# session defaults
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ============================================================
# Sidebar config (emoji toggle)
# ============================================================
# Emoji toggle to open/close sidebar - replaces broken arrow icon
if st.button("⚡", help="Toggle sidebar"):
    st.session_state.sidebar_open = not st.session_state.sidebar_open

if st.session_state.sidebar_open:
    st.sidebar.title("🔑 API CONFIGURATION")
    st.sidebar.caption("Enter your API keys, they are used only in session, or set via environment variables")

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
else:
    # keep values in session if previously entered
    gemini_key_input = ""
    fred_key_input = ""

# Resolve keys
GEMINI_API_KEY = gemini_key_input.strip() if gemini_key_input else os.getenv("GEMINI_API_KEY")
FRED_API_KEY = fred_key_input.strip() if fred_key_input else os.getenv("FRED_API_KEY")

# Configure Gemini safely
gemini_configured = False
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        if st.session_state.sidebar_open:
            st.sidebar.success("✅ Gemini Connected")
        logger.info("Gemini configured")
    except Exception as e:
        if st.session_state.sidebar_open:
            st.sidebar.error(f"❌ Gemini Error: {str(e)[:80]}")
        logger.error(f"Gemini configuration error: {e}")
else:
    if st.session_state.sidebar_open:
        st.sidebar.warning("⚠️ Gemini: No API Key or library missing")
    if not GEMINI_AVAILABLE:
        logger.info("google.generativeai not installed")

# Configure FRED safely
fred = None
if FRED_AVAILABLE and FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
        if st.session_state.sidebar_open:
            st.sidebar.success("✅ FRED Connected")
        logger.info("FRED configured")
    except Exception as e:
        if st.session_state.sidebar_open:
            st.sidebar.error(f"❌ FRED Error: {str(e)[:80]}")
        logger.error(f"FRED configuration error: {e}")
else:
    if st.session_state.sidebar_open:
        st.sidebar.warning("⚠️ FRED: No API Key or library missing")
    if not FRED_AVAILABLE:
        logger.info("fredapi not installed")

# ============================================================
# Theme CSS (amber) with sidebar toggle fixes
# ============================================================
st.markdown("""
    <style>
    /* Pure Black Background */
    .stApp { background-color: #000000 !important; }
    .main, .block-container, section { background-color: #000000 !important; }

    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 2px solid #FFB000 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #FFB000 !important; }

    /* Show and color toggle */
    [data-testid="collapsedControl"], button[title="Toggle sidebar"], button[aria-label="Toggle sidebar"],
    button[title="Toggle navigation"], button[aria-label="Toggle navigation"] {
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
    /* path fill */
    [data-testid="collapsedControl"] svg path, button svg path { fill: #FFB000 !important; stroke: #FFB000 !important; }

    /* Terminal Font for Tables */
    .stDataFrame, .dataframe, table { font-family: 'Courier New', Courier, monospace !important; background-color: #000000 !important; color: #FFB000 !important; }

    /* Metrics - Amber Theme */
    .stMetric { background-color: #000000 !important; border: 1px solid #FFB000 !important; padding: 15px; border-radius: 0px; font-family: 'Courier New', Courier, monospace !important; }
    .stMetric label { color: #FFB000 !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase; }
    .stMetric .metric-value { color: #FFFFFF !important; font-size: 24px !important; font-weight: 700 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #000000 !important; border-bottom: 2px solid #FFB000 !important; }
    .stTabs [data-baseweb="tab"] { background-color: #000000 !important; color: #FFB000 !important; border: 1px solid #FFB000 !important; padding: 10px 20px; font-family: 'Courier New', Courier, monospace !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { background-color: #FFB000 !important; color: #000000 !important; }

    /* Headers */
    h1, h2, h3, h4 { color: #FFB000 !important; font-family: 'Courier New', Courier, monospace !important; font-weight: 700 !important; text-transform: uppercase; }

    /* Buttons */
    .stButton > button { background-color: #000000 !important; color: #FFB000 !important; border: 2px solid #FFB000 !important; padding: 10px 30px; text-transform: uppercase; }
    .stButton > button:hover { background-color: #FFB000 !important; color: #000000 !important; }

    /* Inputs */
    .stTextInput > div > div > input { background-color: #000000 !important; color: #FFB000 !important; border: 1px solid #FFB000 !important; font-family: 'Courier New', Courier, monospace !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Utility functions - avoid ambiguous DataFrame truth checks
# ============================================================
def safe_df(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def format_abs_pct(price: float, prev: float) -> str:
    """Return string like +1.20% (+$4.50)"""
    try:
        change = price - prev
        pct = (change / prev * 100) if prev else 0.0
        return f"{pct:+.2f}% ({change:+.2f})"
    except Exception as e:
        logger.error(f"format_abs_pct error: {e}")
        return "—"

# ============================================================
# DATA FETCHING & ANALYTICS - Redis wrapper + caching
# ============================================================
def redis_cache_wrapper(key: str, ttl: int, fetch_func, force_local: bool=False):
    try:
        if redis_client and not force_local:
            cached = redis_cache_get(key)
            if cached is not None:
                return cached
            data = fetch_func()
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
def fetch_ticker_data_reliable(ticker: str):
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
def calculate_rsi(prices: pd.Series, period: int=14):
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
def fetch_watchlist_data(tickers: List[str]):
    def _fetch():
        results = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1mo')
                if hist.empty or len(hist) < 2:
                    continue
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                rsi_value = calculate_rsi(hist['Close'])
                results.append({
                    'Ticker': ticker,
                    'Price': float(current_price),
                    'Prev': float(prev_close),
                    'Change %': float(change_pct),
                    'Volume': volume,
                    'RSI': float(rsi_value)
                })
            except Exception as e:
                logger.warning(f"fetch_watchlist_data error for {ticker}: {e}")
                continue
        return pd.DataFrame(results)
    return redis_cache_wrapper("watchlist_data", 60, _fetch)

@st.cache_data(ttl=60)
def fetch_vix_term_structure():
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
def fetch_crypto_metrics(cryptos: List[str]):
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
                    results.append({'Sector': name, 'Change %': data['change_pct']})
            except Exception as e:
                logger.warning(f"fetch_sector_performance error for {ticker}: {e}")
                continue
        df = pd.DataFrame(results)
        return df.sort_values('Change %', ascending=False) if not df.empty else df
    return redis_cache_wrapper("sector_performance", 60, _fetch)

# ============================================================
# Async RSS and Polymarket fetchers with caching
# ============================================================
async def _fetch_url_httpx(client, url: str, params: dict=None, timeout: int=15):
    try:
        r = await client.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error(f"_fetch_url_httpx failed for {url}: {e}")
        return None

def fetch_news_feeds():
    async def _fetch_all_httpx():
        feeds = {
            'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
            'WSJ': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'
        }
        results = []
        if HTTPX_AVAILABLE:
            async with httpx.AsyncClient() as client:
                tasks = [_fetch_url_httpx(client, url) for url in feeds.values()]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for (source, url), content in zip(feeds.items(), responses):
                    try:
                        if isinstance(content, Exception) or content is None:
                            continue
                        feed = feedparser.parse(content)
                        for entry in feed.entries[:5]:
                            results.append({'Source': source, 'Title': entry.get('title', '')[:200], 'Link': entry.get('link', '')})
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
                        results.append({'Source': source, 'Title': entry.get('title', '')[:200], 'Link': entry.get('link', '')})
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
            res = loop.run_until_complete(_fetch_all_httpx())
            loop.close()
        else:
            res = asyncio.run(_fetch_all_httpx())
        redis_cache_set(key, res, 60)
        return res
    except Exception as e:
        logger.error(f"fetch_news_feeds error: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_insider_cluster_buys():
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
        return {'yield_spread': 0, 'credit_spread': 0, 'fed_balance': 0, 'success': False}
    try:
        t10y2y = fred.get_series_latest_release('T10Y2Y')
        yield_spread = float(t10y2y.iloc[-1]) if not t10y2y.empty else 0
        hy_spread = fred.get_series_latest_release('BAMLH0A0HYM2')
        credit_spread = float(hy_spread.iloc[-1]) if not hy_spread.empty else 0
        fed_assets = fred.get_series_latest_release('WALCL')
        fed_balance = float(fed_assets.iloc[-1]) if not fed_assets.empty else 0
        return {'yield_spread': yield_spread, 'credit_spread': credit_spread, 'fed_balance': fed_balance / 1000, 'success': True}
    except Exception as e:
        logger.error(f"fetch_fred_liquidity error: {e}")
        return {'yield_spread': 0, 'credit_spread': 0, 'fed_balance': 0, 'success': False}

# ============================================================
# SPX options and analytics (already robust)
# ============================================================

@st.cache_data(ttl=60)
def fetch_spx_options_data():
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
                if (calls is None or calls.empty) and (puts is None or puts.empty):
                    logger.info(f"Empty option chain for {t} at {nearest_exp}")
                    continue
                calls = calls.fillna(0)
                puts = puts.fillna(0)
                total_call_volume = int(calls['volume'].fillna(0).sum()) if 'volume' in calls.columns else 0
                total_put_volume = int(puts['volume'].fillna(0).sum()) if 'volume' in puts.columns else 0
                put_call_ratio = (total_put_volume / total_call_volume) if total_call_volume > 0 else 0
                total_call_oi = int(calls['openInterest'].fillna(0).sum()) if 'openInterest' in calls.columns else 0
                total_put_oi = int(puts['openInterest'].fillna(0).sum()) if 'openInterest' in puts.columns else 0
                put_call_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0
                try:
                    calls_oi = calls.groupby('strike')['openInterest'].sum() if 'openInterest' in calls.columns else pd.Series(dtype=float)
                    puts_oi = puts.groupby('strike')['openInterest'].sum() if 'openInterest' in puts.columns else pd.Series(dtype=float)
                    total_oi = calls_oi.add(puts_oi, fill_value=0)
                    max_pain = float(total_oi.idxmax()) if not total_oi.empty else 0
                except Exception:
                    max_pain = 0
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

def calculate_skew(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    try:
        if calls is None or puts is None or calls.empty or puts.empty:
            return 0.0
        strikes = pd.concat([calls['strike'], puts['strike']])
        atm = float(strikes.median())
        otm_puts = puts[puts['strike'] < atm]
        otm_calls = calls[calls['strike'] > atm]
        put_iv = float(otm_puts['impliedVolatility'].replace([np.inf, -np.inf], np.nan).dropna().mean()) if 'impliedVolatility' in otm_puts.columns and not otm_puts.empty else 0.0
        call_iv = float(otm_calls['impliedVolatility'].replace([np.inf, -np.inf], np.nan).dropna().mean()) if 'impliedVolatility' in otm_calls.columns and not otm_calls.empty else 0.0
        skew = (put_iv - call_iv) * 100
        return float(skew)
    except Exception as e:
        logger.error(f"calculate_skew error: {e}")
        return 0.0

def calculate_gamma_exposure(options_df: pd.DataFrame, spot_price: float) -> float:
    try:
        if options_df is None or options_df.empty:
            return 0.0
        df = options_df.copy()
        df['distance'] = (df['strike'] - spot_price).abs() + 1.0
        df['gex_proxy'] = df['openInterest'].fillna(0) / df['distance']
        return float(df['gex_proxy'].sum())
    except Exception as e:
        logger.error(f"calculate_gamma_exposure error: {e}")
        return 0.0

# ============================================================
# POLYMARKET: Marginal Polytope + IP oracle + Frank-Wolfe + Bregman Divergence
# ============================================================

# Helper: assemble payoff vectors for a market (binary / categorical)
def build_payoff_matrix(market: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        # market["outcomes"] expected to be list of dicts with 'price' and 'name'
        outcomes = market.get('outcomes') or []
        n = len(outcomes)
        if n == 0:
            return None
        # For categorical markets, payoff vectors are the basis vectors e_i
        # For more complex combinatorial markets, you'd derive payoff vectors Z
        # Here we return identity matrix rows as payoff atoms
        Z = np.eye(n)
        return Z
    except Exception as e:
        logger.error(f"build_payoff_matrix error: {e}")
        return None

# IP oracle: given a linear direction g, find vertex v in conv(Z) that maximizes g dot v.
# We attempt to use pulp for integer programming if constraints exist. For categorical outcomes,
# vertex is simply the basis vector with highest gradient coordinate.
def ip_oracle(g: np.ndarray, Z: np.ndarray, constraints: Optional[Dict]=None) -> np.ndarray:
    try:
        # If constraints are present and pulp available, attempt to solve small IP
        if constraints and PULP_AVAILABLE:
            # build a binary variable for each atom in Z, choose combination
            prob = pulp.LpProblem("IP_Oracle", pulp.LpMaximize)
            n = Z.shape[0]
            x_vars = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]
            # objective maximize g dot (sum x_i * z_i)
            # compute aggregated payoff vector components = sum_i x_i * Z[i,j]
            # since Z is typically identity-like for categorical, this reduces to choose one with highest g_j
            # But we keep general form
            obj_terms = []
            for j in range(Z.shape[1]):
                # aggregated jth payoff: sum_i x_i * Z[i,j]
                obj_terms.append((g[j], pulp.lpSum([x_vars[i] * float(Z[i, j]) for i in range(n)])))
            # pulp does not support vectorized objective easily, flatten:
            # objective = sum_j g_j * sum_i x_i * Z_i_j = sum_i x_i * sum_j g_j * Z_i_j
            coeffs = [float(np.dot(Z[i, :], g)) for i in range(n)]
            prob += pulp.lpSum([coeffs[i] * x_vars[i] for i in range(n)])
            # add constraints from input if any (e.g., cardinality), simple example: at least one
            prob += pulp.lpSum(x_vars) >= 1
            # other possible constraints could be encoded into 'constraints' param, but this is placeholder
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            x_vals = np.array([pulp.value(v) for v in x_vars], dtype=float)
            # build vertex as convex combination normalized
            if x_vals.sum() <= 0:
                # fallback to greedy
                idx = int(np.argmax(np.dot(Z, g)))
                return Z[idx, :]
            weights = x_vals / (x_vals.sum() + 1e-12)
            v = weights @ Z
            return v
        else:
            # simple greedy oracle: choose atom whose dot(g, z) is maximal
            scores = Z.dot(g)
            idx = int(np.argmax(scores))
            return Z[idx, :]
    except Exception as e:
        logger.error(f"ip_oracle error: {e}")
        try:
            scores = Z.dot(g)
            idx = int(np.argmax(scores))
            return Z[idx, :]
        except Exception as ex:
            logger.error(f"ip_oracle fallback error: {ex}")
            return Z[0, :]

def frank_wolfe_projection(theta: np.ndarray, Z: np.ndarray, iterations: int=30, b: float=10.0) -> np.ndarray:
    # Project theta through Bregman divergence onto conv(Z)
    try:
        m = Z.shape[1]
        # initialize mu uniform
        mu = np.ones(m) / m
        for t in range(iterations):
            # gradient of D(mu || theta) wrt mu is grad R(mu) - theta
            # with R(mu) = b * sum mu log mu => grad R = b * (1 + log mu)
            gradR = b * (1.0 + np.log(np.maximum(mu, 1e-12)))
            grad = gradR - theta
            # oracle solves min_{v in conv(Z)} grad dot v  == choose v that minimizes grad dot v
            # equivalently choose v that maximizes (-grad) dot v
            v = ip_oracle(-grad, Z, constraints=None)  # v is in outcome space
            # line search for gamma using simple rule
            gamma = 2.0 / (t + 2.0)
            mu = (1 - gamma) * mu + gamma * v
            # normalize numerically
            mu = np.maximum(mu, 0)
            mu = mu / (mu.sum() + 1e-12)
        return mu
    except Exception as e:
        logger.error(f"frank_wolfe_projection error: {e}")
        # fallback: return softmax(theta)
        exps = np.exp(theta / b)
        return exps / (np.sum(exps) + 1e-12)

def bregman_divergence(mu: np.ndarray, theta: np.ndarray, b: float=10.0) -> float:
    # D(mu || theta) = R(mu) + C(theta) - theta . mu
    # R(mu) = b sum mu log mu
    # C(theta) = b log sum exp(theta / b)
    try:
        R = b * np.sum(mu * np.log(np.maximum(mu, 1e-12)))
        C = b * np.log(np.sum(np.exp(theta / b)))
        return float(R + C - np.dot(theta, mu))
    except Exception as e:
        logger.error(f"bregman_divergence error: {e}")
        return float('inf')

# Polymarket advanced arbitrage attempt: returns candidate trades with estimated profit
def polymarket_arbitrage_engine(market: Dict[str, Any], b: float=10.0, iterations: int=30) -> Dict[str, Any]:
    """
    Input market JSON from Polymarket.
    Steps:
      - Build payoff matrix Z
      - Build theta vector from current prices (log-odds / current market parameterization)
      - Use Frank-Wolfe with IP oracle to find mu*
      - Compute potential profit using VWAP/price approximation and require > 0.05 threshold
    Returns dict with keys: 'arbitrage' (bool), 'profit' (float), 'mu_star' (np.array), 'notes'
    """
    try:
        outcomes = market.get('outcomes') or []
        n = len(outcomes)
        if n == 0:
            return {'arbitrage': False, 'profit': 0.0, 'notes': 'No outcomes'}
        # Build payoff atoms Z: categorical basis
        Z = np.eye(n)
        # Build theta from prices: treat price p as current market implicit probability
        prices = np.array([float(out.get('price', 0.5)) for out in outcomes])
        # Convert prices to 'theta' (log-odds scaled by b)
        # Ensure no zeros
        p_safe = np.clip(prices, 1e-6, 1-1e-6)
        theta = b * np.log(p_safe / (1 - p_safe))
        # Project using Frank-Wolfe with IP oracle onto conv(Z)
        mu_star = frank_wolfe_projection(theta, Z, iterations=iterations, b=b)
        # Estimate VWAP (approx) using current prices, for each outcome use price as VWAP
        vwap = prices
        # Edge per outcome
        edge = mu_star - vwap
        positive_edge = edge[edge > 0]
        profit = float(positive_edge.sum())
        arbitrage = profit > 0.05  # threshold to account for fees and slippage
        notes = f"n={n}, profit={profit:.4f}, arbitrage={arbitrage}"
        return {'arbitrage': arbitrage, 'profit': profit, 'mu_star': mu_star.tolist(), 'notes': notes}
    except Exception as e:
        logger.error(f"polymarket_arbitrage_engine error: {e}")
        return {'arbitrage': False, 'profit': 0.0, 'notes': str(e)}

# ============================================================
# AI BRIEFING: more focused macro content, high-density paragraphs
# ============================================================
def _simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    try:
        risk = "Elevated" if vix_price > 25 or put_call_ratio > 1.2 else "Normal"
        opportunity = "Look for sector momentum around tech if skew compresses" if spx_price > 4000 else "Scan for mean reversion setups near key support"
        algos = "Monitor IV spikes, put-call skew, and directional volume clusters"
        # Build focused paragraphs
        para1 = f"Yield Curve & Fed: With current P/C {put_call_ratio:.2f} and VIX {vix_price:.1f}, yield curve changes suggest liquidity repricing risk. Watch 2s10s for compression, and Fed SOMA adjustments affecting short-end liquidity."
        para2 = f"Cross-asset Correlation: SPX/ES correlation with major crypto pairs can rise rapidly in high vol regimes, amplifying tail risk. Correlation shifts during liquidity events create transient arbitrage windows in prediction markets."
        para3 = f"Prediction Markets & Liquidity: Liquidity cycles drive pricing inefficiencies. Use VWAP-based execution and a minimum profit threshold to avoid gas and slippage erosion."
        overflow = " ".join(news_headlines[3:]) if len(news_headlines) > 3 else ""
        return f"{para1}\n\n{para2}\n\n{para3}\n\nNotes: {overflow}"
    except Exception as e:
        logger.error(f"_simple_briefing error: {e}")
        return "Macro briefing unavailable."

def generate_ai_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    if not gemini_configured:
        return _simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        top_headlines = news_headlines[:6] if news_headlines else []
        prompt = f"""You are an institutional macro strategist. Produce concise, high-density paragraphs (max 250 words) focusing on:
1) Yield curve shifts and Fed balance operations,
2) Empirical correlation coefficients between SPX/ES and major crypto pairs and implication,
3) Liquidity cycle impacts on prediction markets, VWAP-based execution, slippage and gas.
Use bullet-style paragraphs. News: {' | '.join(top_headlines)}"""
        response = model.generate_content(prompt)
        if not response or not getattr(response, "text", None):
            return _simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines)
        return response.text
    except Exception as e:
        logger.error(f"generate_ai_briefing error: {e}")
        return f"⚠️ AI Error, fallback briefing:\n\n{_simple_briefing(spx_price, vix_price, put_call_ratio, news_headlines)}"

# ============================================================
# Service classes
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
    @staticmethod
    def arbitrage_engine(market):
        return polymarket_arbitrage_engine(market)

class FredService:
    @staticmethod
    def liquidity():
        return fetch_fred_liquidity()

# ============================================================
# MAIN APPLICATION UI
# ============================================================
st.title("⚡ ALPHA DECK PRO v4.0 - GOLD MASTER ENHANCED")

# AI briefing section
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
        # Primary viewport: concise; overflow content below as paragraphs
        st.markdown("#### Brief (top-level)")
        lines = briefing.split("\n\n")
        st.write(lines[0] if lines else briefing)
        st.markdown("#### Full briefing")
        st.code(briefing)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MAIN DECK", "📊 LIQUIDITY & INSIDER", "₿ CRYPTO & POLY", "📈 TRADINGVIEW"])

# ---------------------------
# TAB 1: MAIN DECK
# ---------------------------
with tab1:
    st.subheader("📡 MARKET PULSE")
    indices = MarketDataService.indices()
    cols = st.columns(6)
    index_keys = ['SPX', 'NDX', 'VIX', 'HYG', 'US10Y', 'DXY']
    for idx, name in enumerate(index_keys):
        data = indices.get(name, {'success': False})
        with cols[idx]:
            if data.get('success'):
                if name in ['VIX', 'HYG']:
                    abs_str = f"{data['change_abs']:+.2f} pts"
                    value_str = f"{data['price']:.2f}"
                    delta = f"{abs_str} ({data['change_pct']:+.2f}%)"
                else:
                    abs_str = f"${data['change_abs']:+.2f}"
                    value_str = f"${data['price']:.2f}"
                    delta = f"{abs_str} ({data['change_pct']:+.2f}%)"
                st.metric(label=name, value=value_str, delta=delta)
            else:
                st.metric(label=name, value="LOADING", delta="—")

    st.divider()

    # VIX term structure + regime
    st.subheader("📊 VOLATILITY TERM STRUCTURE")
    vix_term = fetch_vix_term_structure()
    col_vix1, col_vix2 = st.columns([3, 7])
    with col_vix1:
        if vix_term.get('backwardation'):
            st.error("⚠️ BACKWARDATION (CRASH SIGNAL)")
        else:
            st.success("✅ CONTANGO (NORMAL)")
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
        fig_vix.update_layout(title="VIX Term Structure", template='plotly_dark', height=200, plot_bgcolor='#000000', paper_bgcolor='#000000',
                              font=dict(color='#FFB000', family='Courier New'), margin=dict(l=0, r=0, t=40, b=0), xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=False, title="Volatility"))
        st.plotly_chart(fig_vix, use_container_width=True)

    st.divider()

    # Quick selectors
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

    # SPX Options intelligence
    st.subheader("🎯 SPX OPTIONS INTELLIGENCE")
    market_is_open = is_market_open()
    ny_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_tz)
    st.caption(f"Current ET Time: {current_time.strftime('%I:%M %p')} | Day: {current_time.strftime('%A')}")
    if not market_is_open:
        st.warning("⏰ Markets Closed, options data available Mon-Fri 9:30 AM - 4:00 PM ET")

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
        # compute skew + gex
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
            fig_volume.update_layout(title="Call vs Put Volume", template='plotly_dark', height=250, plot_bgcolor='#000000', paper_bgcolor='#000000',
                                     font=dict(color='#FFB000', family='Courier New'), margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_volume, use_container_width=True)
        with col_vol2:
            try:
                calls_top = spx_data['calls'].nlargest(10, 'volume') if 'calls' in spx_data and not spx_data['calls'].empty else pd.DataFrame()
                puts_top = spx_data['puts'].nlargest(10, 'volume') if 'puts' in spx_data and not spx_data['puts'].empty else pd.DataFrame()
                fig_iv = go.Figure()
                if not calls_top.empty and 'impliedVolatility' in calls_top.columns:
                    fig_iv.add_trace(go.Scatter(x=calls_top['strike'], y=calls_top['impliedVolatility'] * 100, mode='markers', name='Calls', marker=dict(size=10, color='#00FF00')))
                if not puts_top.empty and 'impliedVolatility' in puts_top.columns:
                    fig_iv.add_trace(go.Scatter(x=puts_top['strike'], y=puts_top['impliedVolatility'] * 100, mode='markers', name='Puts', marker=dict(size=10, color='#FF0000')))
                fig_iv.update_layout(title="IV by Strike", template='plotly_dark', height=250, xaxis_title="Strike", yaxis_title="IV %", plot_bgcolor='#000000', paper_bgcolor='#000000', font=dict(color='#FFB000', family='Courier New'), margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_iv, use_container_width=True)
            except Exception as e:
                logger.error(f"SPX IV chart error: {e}")
    else:
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

    # WATCHLIST: card-based layout rather than Excel table
    st.subheader("⚡ WATCHLIST + TECHNICAL SCANNER")
    watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
    with st.spinner("Loading watchlist..."):
        wl_df = MarketDataService.watchlist(watchlist_tickers)

    if isinstance(wl_df, pd.DataFrame) and not wl_df.empty:
        # show rows of 4 cards per row
        cols_per_row = 4
        rows = int(np.ceil(len(wl_df) / cols_per_row))
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx < len(wl_df):
                    row = wl_df.iloc[idx]
                    ticker = row['Ticker']
                    price = row['Price']
                    prev = row.get('Prev', price - (price * row['Change %']/100 if 'Change %' in row else 0))
                    change_str = format_abs_pct(price, prev)
                    rsi = row['RSI']
                    volume = row['Volume']
                    signals = check_technical_signals({'RSI': rsi, 'Change %': row['Change %'], 'Volume': f"{volume/1e6:.1f}M" if volume else "0M"})
                    card_html = f"""
                        <div style="padding:12px;border-radius:8px;background:#0b0b0b;border:1px solid #222;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div style="font-weight:700;color:#FFB000;">{ticker}</div>
                                <div style="font-weight:700;color:#FFFFFF;">${price:,.2f}</div>
                            </div>
                            <div style="margin-top:6px;color:#FFFFFF;">{change_str}</div>
                            <div style="margin-top:6px;color:#FFB000;font-size:12px;">RSI: {rsi:.1f} | {signals}</div>
                        </div>
                    """
                    cols[c].markdown(card_html, unsafe_allow_html=True)
    else:
        st.error("DATA UNAVAILABLE")

    st.divider()

    # SECTOR HEATMAP with drill-down
    st.subheader("🎨 SECTOR HEAT")
    with st.spinner("Loading sector performance..."):
        sector_df = fetch_sector_performance()
    if isinstance(sector_df, pd.DataFrame) and not sector_df.empty:
        # Build treemap data from sector_df
        fig = px.treemap(sector_df, path=['Sector'], values='Change %', color='Change %', color_continuous_scale='RdYlGn', template='plotly_dark')
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=450)
        st.plotly_chart(fig, use_container_width=True)

        # drill-down selector
        sectors = sector_df['Sector'].tolist()
        selected_sector = st.selectbox("Drill down into sector", options=[''] + sectors)
        if selected_sector:
            # fetch constituents using a small mapping (for performance) or external API if available
            sector_to_tickers = {
                'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC'],
                'Energy': ['XOM', 'CVX', 'SLB'],
                'Financials': ['JPM', 'BAC', 'GS', 'MS']
            }
            assets = sector_to_tickers.get(selected_sector, [])
            if assets:
                cols = st.columns(min(len(assets), 5))
                for i, t in enumerate(assets):
                    p = fetch_ticker_data_reliable(t)
                    if p and p.get('success'):
                        val = p['price']
                        prev = val - (p['change_abs'])
                        cols[i % len(cols)].metric(label=t, value=f"${val:.2f}", delta=f"{p['change_abs']:+.2f} ({p['change_pct']:+.2f}%)")
                    else:
                        cols[i % len(cols)].write(t)
            else:
                st.info("No constituents mapping available for this sector")
    else:
        st.info("Sector performance unavailable")

# ---------------------------
# TAB 2: LIQUIDITY & INSIDER
# ---------------------------
with tab2:
    st.subheader("🏛️ FED LIQUIDITY METRICS")
    liquidity = FredService.liquidity()
    if liquidity and liquidity.get('success'):
        met_cols = st.columns(3)
        met_cols[0].metric("10Y-2Y SPREAD", f"{liquidity['yield_spread']:.2f}%")
        met_cols[1].metric("HY CREDIT SPREAD", f"{liquidity['credit_spread']:.2f}%")
        met_cols[2].metric("FED BALANCE", f"${liquidity['fed_balance']:.2f}T")
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

    st.subheader("🕵️ INSIDER CLUSTER BUYS")
    insider_df = fetch_insider_cluster_buys()
    if isinstance(insider_df, pd.DataFrame) and not insider_df.empty:
        st.dataframe(insider_df, use_container_width=True, height=400, hide_index=True)
    else:
        st.warning("⚠️ Insider Data: Source Blocking or unavailable")

    st.divider()
    st.subheader("📰 NEWS WIRE")
    news = NewsService.headlines()
    if news:
        for article in news:
            st.markdown(f"**[{article['Source']}]** [{article['Title']}]({article['Link']})")
    else:
        st.info("NO NEWS AVAILABLE")

# ---------------------------
# TAB 3: CRYPTO & POLYMARKET
# ---------------------------
with tab3:
    st.subheader("₿ CRYPTO MARKET PULSE")
    crypto_data = fetch_crypto_metrics(['BTC', 'ETH', 'SOL', 'DOGE'])
    row1 = st.columns(2)
    row2 = st.columns(2)
    crypto_order = ['BTC', 'ETH', 'SOL', 'DOGE']
    for idx, crypto in enumerate(crypto_order):
        d = crypto_data.get(crypto)
        if d and d.get('success'):
            if idx < 2:
                row1[idx].metric(label=crypto, value=f"${d['price']:,.2f}", delta=f"{d['change_pct']:+.2f}%")
            else:
                row2[idx-2].metric(label=crypto, value=f"${d['price']:,.2f}", delta=f"{d['change_pct']:+.2f}%")

    st.divider()
    st.subheader("🎲 POLYMARKET ALPHA (Advanced)")
    poly_df = PolymarketService.opportunities()
    if isinstance(poly_df, pd.DataFrame) and not poly_df.empty:
        df = poly_df.copy()
        df['Vol'] = df['Vol'].apply(lambda x: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
        st.dataframe(df[['Event', 'Yes %', 'Vol', 'Score']], use_container_width=True, height=400, hide_index=True)
        st.caption("**Click events to review arbitrage**")
        # check arbitrage on click: user chooses a row
        selected_idx = st.number_input("Select row index to analyze arbitrage", min_value=0, max_value=len(df)-1 if len(df)>0 else 0, value=0)
        if len(df) > 0:
            selected = df.iloc[int(selected_idx)]
            slug = selected.get('slug')
            if slug:
                # fetch market detail if possible via API
                try:
                    url = f"https://gamma-api.polymarket.com/markets/{slug}"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        market_json = resp.json()
                        arb_res = PolymarketService.arbitrage_engine(market_json)
                        st.write("Arbitrage engine result:")
                        st.json(arb_res)
                        if arb_res.get('arbitrage'):
                            st.success(f"Estimated profit: {arb_res.get('profit'):.4f}")
                        else:
                            st.info("No arbitrage above threshold")
                    else:
                        st.error(f"Failed to fetch market {slug}, status {resp.status_code}")
                except Exception as e:
                    logger.error(f"Polymarket fetch error: {e}")
                    st.error("Failed to fetch polymarket market details")
    else:
        st.error("POLYMARKET DATA UNAVAILABLE")

# ---------------------------
# TAB 4: TRADINGVIEW
# ---------------------------
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

# Footer
st.divider()
st.caption("⚡ ALPHA DECK PRO v4.0 - GOLD MASTER ENHANCED")
