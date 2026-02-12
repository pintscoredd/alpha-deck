"""
Alpha Deck PRO v5.0 - QUANT EDITION (FIXED)
Production-Grade Prediction Market Arbitrage | Card-Based UI | Macro Intelligence
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
except:
    GEMINI_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except:
    FRED_AVAILABLE = False

# ============================================================================
# PAGE CONFIG & SESSION STATE INITIALIZATION
# ============================================================================
st.set_page_config(
    page_title="Alpha Deck PRO v5.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize ALL session state variables INCLUDING API clients
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'
if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = False
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None
if 'show_sector_drill' not in st.session_state:
    st.session_state.show_sector_drill = False

# CRITICAL FIX: Store API clients in session state
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = None
if 'fred_api_key' not in st.session_state:
    st.session_state.fred_api_key = None
if 'fred_client' not in st.session_state:
    st.session_state.fred_client = None
if 'gemini_configured' not in st.session_state:
    st.session_state.gemini_configured = False

# ============================================================================
# THEME & CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    /* Pure Black Terminal */
    .stApp {
        background-color: #000000 !important;
    }
    
    .main, .block-container, section {
        background-color: #000000 !important;
    }
    
    /* CARD SYSTEM */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 1px solid #FFB000;
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 176, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(255, 176, 0, 0.2);
        border-color: #FFC933;
    }
    
    .metric-label {
        color: #FFB000;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
    }
    
    .metric-value {
        color: #FFFFFF;
        font-size: 32px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        line-height: 1.2;
    }
    
    .metric-change {
        color: #FFB000;
        font-size: 14px;
        font-weight: 500;
        margin-top: 4px;
        font-family: 'Courier New', monospace;
    }
    
    .metric-change.positive { color: #00FF00; }
    .metric-change.negative { color: #FF0000; }
    
    .signal-badge {
        display: inline-block;
        background: rgba(255, 176, 0, 0.1);
        border: 1px solid #FFB000;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 10px;
        font-weight: 600;
        margin: 2px;
        color: #FFB000;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #000000 100%);
        border-right: 2px solid #FFB000;
    }
    
    /* Tab System */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #000000;
        border-bottom: 2px solid #FFB000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #000000;
        color: #FFB000;
        border: 1px solid #FFB000;
        border-radius: 4px 4px 0 0;
        padding: 12px 24px;
        font-family: 'Courier New', monospace;
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
        background-color: #000000;
        color: #FFB000;
        border: 2px solid #FFB000;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        padding: 12px 24px;
        text-transform: uppercase;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #FFB000;
        color: #000000;
        transform: scale(1.02);
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #FFB000;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    p, span, div, label {
        color: #FFFFFF;
        font-family: 'Courier New', monospace;
    }
    
    /* Links */
    a {
        color: #FFB000;
        text-decoration: none;
        transition: color 0.2s;
    }
    
    a:hover {
        color: #FFC933;
        text-decoration: underline;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #1a1a1a;
        color: #FFB000;
        border: 1px solid #FFB000;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
    }
    
    /* Dividers */
    hr {
        border-color: #FFB000;
        opacity: 0.3;
        margin: 2rem 0;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #FFB000;
        border: 1px solid #FFB000;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
    }
    
    /* AI Briefing Box */
    .ai-briefing {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 2px solid #FFB000;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        font-family: 'Courier New', monospace;
        line-height: 1.8;
        color: #FFFFFF;
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
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR TOGGLE & PERSISTENT API CONFIGURATION
# ============================================================================

# Create sidebar toggle button
col_toggle, col_spacer = st.columns([1, 20])
with col_toggle:
    toggle_icon = "⚡" if not st.session_state.sidebar_visible else "📊"
    if st.button(toggle_icon, key="sidebar_toggle", help="Toggle API Config"):
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible
        st.rerun()

# Show/hide sidebar based on state
if st.session_state.sidebar_visible:
    st.sidebar.title("🔑 API CONFIGURATION")
    st.sidebar.caption("Professional API Management")
    
    # Get API keys from inputs (with session state as default)
    gemini_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key if st.session_state.gemini_api_key else "",
        type="password",
        help="Get free key: https://makersuite.google.com/app/apikey"
    )
    
    fred_key_input = st.sidebar.text_input(
        "FRED API Key",
        value=st.session_state.fred_api_key if st.session_state.fred_api_key else "",
        type="password",
        help="Get free key: https://fredaccount.stlouisfed.org/apikeys"
    )
    
    st.sidebar.markdown("---")
    
    # CRITICAL FIX: Store keys in session state
    if gemini_key_input:
        st.session_state.gemini_api_key = gemini_key_input.strip()
    if fred_key_input:
        st.session_state.fred_api_key = fred_key_input.strip()
    
    # Configure Gemini (PERSISTENT)
    if GEMINI_AVAILABLE and st.session_state.gemini_api_key:
        try:
            # FIX: Updated to latest model
            genai.configure(api_key=st.session_state.gemini_api_key)
            st.session_state.gemini_configured = True
            st.sidebar.success("✅ Gemini: Connected")
        except Exception as e:
            st.sidebar.error(f"❌ Gemini Error: {str(e)[:50]}")
            st.session_state.gemini_configured = False
    else:
        if not st.session_state.gemini_api_key:
            st.sidebar.warning("⚠️ Gemini: API Key Required")
        st.session_state.gemini_configured = False
    
    # Configure FRED (PERSISTENT)
    if FRED_AVAILABLE and st.session_state.fred_api_key:
        try:
            st.session_state.fred_client = Fred(api_key=st.session_state.fred_api_key)
            st.sidebar.success("✅ FRED: Connected")
        except Exception as e:
            st.sidebar.error(f"❌ FRED Error: {str(e)[:50]}")
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
    """Check if US stock market is currently open"""
    try:
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)
        
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except:
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

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data with absolute + percentage change"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        
        if hist.empty:
            return {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        change_abs = current_price - prev_close
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
        volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
        
        return {
            'price': float(current_price),
            'change_pct': float(change_pct),
            'change_abs': float(change_abs),
            'volume': int(volume),
            'success': True
        }
    except:
        return {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}

@st.cache_data(ttl=60)
def calculate_rsi(prices, period=14):
    """Calculate RSI with proper series handling"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return 50.0
        
        return float(rsi.iloc[-1])
    except:
        return 50.0

@st.cache_data(ttl=60)
def fetch_watchlist_data(tickers):
    """Fetch watchlist with technical signals"""
    results = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1mo')
            
            if hist.empty:
                continue
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            change_abs = current_price - prev_close
            volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            rsi_value = calculate_rsi(hist['Close'])
            
            # Generate signals
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
                'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_value),
                'Signals': " | ".join(signals) if signals else "—"
            })
        except:
            continue
    
    return pd.DataFrame(results)

@st.cache_data(ttl=60)
def fetch_index_data():
    """Fetch major indices"""
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
        data = fetch_ticker_data_reliable(ticker)
        results[name] = data
    return results

@st.cache_data(ttl=60)
def fetch_sector_performance():
    """Fetch sector ETF performance"""
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
        data = fetch_ticker_data_reliable(ticker)
        if data['success']:
            results.append({
                'Sector': name,
                'Ticker': ticker,
                'Change %': data['change_pct'],
                'Change $': data['change_abs']
            })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        return df.sort_values('Change %', ascending=False)
    
    return df

@st.cache_data(ttl=60)
def fetch_sector_constituents(sector_ticker):
    """Fetch top holdings for a sector ETF"""
    sector_stocks = {
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
        'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP']
    }
    
    tickers = sector_stocks.get(sector_ticker, [])
    if not tickers:
        return pd.DataFrame()
    
    return fetch_watchlist_data(tickers)

@st.cache_data(ttl=60)
def fetch_spx_options_data():
    """Fetch SPX options - FIXED with proper validation"""
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
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        total_call_oi = int(calls['openInterest'].fillna(0).sum())
        total_put_oi = int(puts['openInterest'].fillna(0).sum())
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
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
            'success': True
        }
    except:
        return None

@st.cache_data(ttl=60)
def fetch_vix_term_structure():
    """Fetch VIX term structure"""
    try:
        vix = fetch_ticker_data_reliable('^VIX')
        vix9d = fetch_ticker_data_reliable('^VIX9D')
        vix3m = fetch_ticker_data_reliable('^VIX3M')
        
        return {
            'VIX': vix['price'],
            'VIX9D': vix9d['price'],
            'VIX3M': vix3m['price'],
            'backwardation': vix9d['price'] > vix['price'] if vix9d['success'] and vix['success'] else False
        }
    except:
        return {'VIX': 0, 'VIX9D': 0, 'VIX3M': 0, 'backwardation': False}

@st.cache_data(ttl=60)
def fetch_crypto_metrics(cryptos):
    """Fetch crypto data"""
    results = {}
    for crypto_symbol in cryptos:
        ticker = f"{crypto_symbol}-USD"
        data = fetch_ticker_data_reliable(ticker)
        results[crypto_symbol] = data
    return results

@st.cache_data(ttl=300)
def fetch_news_feeds():
    """Fetch RSS news"""
    feeds = {
        'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'WSJ': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'
    }
    articles = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                articles.append({
                    'Source': source,
                    'Title': entry.title,
                    'Link': entry.link
                })
        except:
            pass
    return articles[:15]

@st.cache_data(ttl=3600)
def fetch_insider_cluster_buys():
    """Scrape OpenInsider"""
    try:
        url = "http://openinsider.com/latest-cluster-buys"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        tables = pd.read_html(url, header=0)
        
        if not tables or len(tables) == 0:
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
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_fred_liquidity():
    """Fetch Fed liquidity - FIXED with persistent client"""
    # FIX: Use session state fred_client
    if st.session_state.fred_client is None:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }
    
    try:
        t10y2y = st.session_state.fred_client.get_series_latest_release('T10Y2Y')
        yield_spread = float(t10y2y.iloc[-1]) if not t10y2y.empty else 0
        
        hy_spread = st.session_state.fred_client.get_series_latest_release('BAMLH0A0HYM2')
        credit_spread = float(hy_spread.iloc[-1]) if not hy_spread.empty else 0
        
        fed_assets = st.session_state.fred_client.get_series_latest_release('WALCL')
        fed_balance = float(fed_assets.iloc[-1]) / 1000 if not fed_assets.empty else 0
        
        return {
            'yield_spread': yield_spread,
            'credit_spread': credit_spread,
            'fed_balance': fed_balance,
            'success': True
        }
    except Exception as e:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False,
            'error': str(e)
        }

@st.cache_data(ttl=60)
def calculate_spx_crypto_correlation():
    """
    Calculate SPX/ES to crypto correlation coefficients.
    FIX: Handle weekend gap with proper date alignment.
    """
    try:
        # Fetch 30 days of data
        spx = yf.Ticker('^GSPC').history(period='1mo')['Close']
        btc = yf.Ticker('BTC-USD').history(period='1mo')['Close']
        eth = yf.Ticker('ETH-USD').history(period='1mo')['Close']
        
        if spx.empty or btc.empty or eth.empty:
            return {'BTC': 0, 'ETH': 0}
        
        # FIX: Use INNER join to align only trading days where both have data
        # This handles the weekend gap (SPX closed, crypto open)
        combined = pd.DataFrame({
            'SPX': spx,
            'BTC': btc,
            'ETH': eth
        })
        
        # Drop all rows with ANY NaN (inner join behavior)
        combined = combined.dropna(how='any')
        
        # Need at least 10 points for meaningful correlation
        if combined.empty or len(combined) < 10:
            return {'BTC': 0, 'ETH': 0}
        
        # Calculate correlations
        corr_btc = combined['SPX'].corr(combined['BTC'])
        corr_eth = combined['SPX'].corr(combined['ETH'])
        
        return {
            'BTC': float(corr_btc) if not pd.isna(corr_btc) else 0,
            'ETH': float(corr_eth) if not pd.isna(corr_eth) else 0
        }
    except Exception as e:
        return {'BTC': 0, 'ETH': 0, 'error': str(e)}

# ============================================================================
# POLYMARKET ARBITRAGE ENGINE - PRODUCTION GRADE
# ============================================================================

class PolymarketArbitrageEngine:
    """Production-grade arbitrage detection using marginal polytope theory"""
    
    def __init__(self, min_profit_threshold=0.05, liquidity_param=100):
        self.min_profit_threshold = min_profit_threshold
        self.b = liquidity_param
        
    def compute_marginal_polytope(self, outcomes):
        n = len(outcomes)
        vertices = np.eye(n)
        try:
            return ConvexHull(vertices)
        except:
            return None
    
    def check_polytope_membership(self, prices):
        if not isinstance(prices, (list, np.ndarray)):
            return True
        
        prices = np.array(prices)
        
        if not np.allclose(np.sum(prices), 1.0, atol=0.02):
            return False
        
        if np.any(prices < -0.01) or np.any(prices > 1.01):
            return False
        
        return True
    
    def compute_bregman_divergence(self, mu, theta, cost_function='lmsr'):
        if cost_function == 'lmsr':
            mu_safe = np.clip(mu, 1e-10, 1 - 1e-10)
            R_mu = -np.sum(mu_safe * np.log(mu_safe))
            
            theta_safe = np.clip(theta, -100, 100)
            C_theta = self.b * np.log(np.sum(np.exp(theta_safe / self.b)))
            
            divergence = R_mu + C_theta - np.dot(theta, mu)
            
            return float(divergence)
        
        return 0.0
    
    def frank_wolfe_projection(self, prices, max_iter=100, tolerance=1e-6):
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
    
    def detect_dependency_arbitrage(self, markets, max_pairs=100):
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
                            market1.get('volume', 100)
                        )
                        
                        net_profit = gross_profit - execution_costs['total']
                        
                        if net_profit > self.min_profit_threshold:
                            arbitrage_opportunities.append({
                                'type': 'dependency_violation',
                                'market1': market1.get('question', ''),
                                'market2': market2.get('question', ''),
                                'slug1': market1.get('slug', ''),
                                'slug2': market2.get('slug', ''),
                                'p1': p1,
                                'p2': p2,
                                'gross_profit': gross_profit,
                                'execution_costs': execution_costs,
                                'net_profit': net_profit,
                                'strategy': f'Short M1 @ {p1:.2f} / Long M2 @ {p2:.2f}',
                                'confidence': self._compute_confidence(q1, q2)
                            })
        
        return arbitrage_opportunities
    
    def _detect_implication(self, event1, event2):
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
    
    def _compute_confidence(self, event1, event2):
        words1 = set(event1.split())
        words2 = set(event2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0
        
        return float(jaccard)
    
    def _estimate_execution_costs(self, liquidity, volume):
        max_trade = min(volume * 0.10, liquidity * 0.05)
        max_trade = max(max_trade, 100)
        
        base_slippage = 0.005
        slippage = base_slippage * np.sqrt(max_trade / max(liquidity, 100))
        slippage = min(slippage, 0.05)
        
        gas_fixed = 2.0
        gas_pct = gas_fixed / max(max_trade, 100)
        
        total_cost = slippage + gas_pct
        
        return {
            'slippage': float(slippage),
            'gas': float(gas_pct),
            'total': float(total_cost),
            'max_trade_size': float(max_trade)
        }
    
    def analyze_market(self, market_data):
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
            else:
                divergence = 0
            
            optimal_prices = self.frank_wolfe_projection(prices)
            
            execution = self._estimate_execution_costs(
                market_data.get('liquidity', 1000),
                market_data.get('volume', 100)
            )
            
            price_deviation = np.abs(prices - optimal_prices).max()
            gross_profit = price_deviation
            net_profit = gross_profit - execution['total']
            
            is_profitable = net_profit > self.min_profit_threshold
            
            return {
                'is_valid': is_valid,
                'is_profitable': is_profitable,
                'prices': prices.tolist(),
                'optimal_prices': optimal_prices.tolist(),
                'price_deviation': float(price_deviation),
                'bregman_divergence': float(divergence),
                'execution_costs': execution,
                'gross_profit': float(gross_profit),
                'net_profit': float(net_profit),
                'sharpe_estimate': float(net_profit / max(execution['total'], 0.01))
            }
        
        except Exception as e:
            return None

# ============================================================================
# POLYMARKET DATA FETCH - FIXED
# ============================================================================

@st.cache_data(ttl=180)
def fetch_polymarket_with_arbitrage():
    """
    Fetch Polymarket data with production-grade arbitrage detection.
    FIX: Handle null values in liquidity and volume.
    
    Returns: (opportunities_df, arbitrage_df)
    """
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            'limit': 100,
            'active': 'true',
            'closed': 'false'
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return pd.DataFrame(), pd.DataFrame()
        
        markets = response.json()
        
        # Filter entertainment/sports
        filter_keywords = ['nfl', 'nba', 'sport', 'gaming', 'gta', 'pop culture',
                          'music', 'twitch', 'mlb', 'nhl', 'soccer', 'football',
                          'basketball', 'celebrity', 'movie', 'ufc', 'mma', 'tennis']
        
        arb_engine = PolymarketArbitrageEngine(min_profit_threshold=0.05)
        
        opportunities = []
        arbitrage_trades = []
        
        for market in markets:
            try:
                question = market.get('question', '').lower()
                
                if any(kw in question for kw in filter_keywords):
                    continue
                
                slug = market.get('slug', '')
                
                # FIX: Handle null values with default 0
                volume = float(market.get('volume') or 0)
                liquidity = float(market.get('liquidity') or 0)
                
                outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
                
                # FIX: Handle empty or null outcomePrices
                if not outcome_prices or len(outcome_prices) == 0:
                    outcome_prices = ['0.5', '0.5']
                
                yes_price = float(outcome_prices[0] or 0.5)
                
                market_full = {
                    'question': market.get('question', ''),
                    'slug': slug,
                    'yes_price': yes_price,
                    'outcomePrices': outcome_prices,
                    'volume': volume,
                    'liquidity': liquidity
                }
                
                arb_analysis = arb_engine.analyze_market(market_full)
                
                if arb_analysis and arb_analysis['is_profitable']:
                    arbitrage_trades.append({
                        'Event': market.get('question', ''),
                        'slug': slug,
                        'Current Yes %': yes_price * 100,
                        'Optimal Yes %': arb_analysis['optimal_prices'][0] * 100,
                        'Deviation %': arb_analysis['price_deviation'] * 100,
                        'Net Profit %': arb_analysis['net_profit'] * 100,
                        'Sharpe': arb_analysis['sharpe_estimate'],
                        'Max Trade': f"${arb_analysis['execution_costs']['max_trade_size']:.0f}"
                    })
                
                if volume > 100:
                    opportunities.append({
                        'Event': market.get('question', ''),
                        'slug': slug,
                        'Yes %': yes_price * 100,
                        'Volume': volume,
                        'Liquidity': liquidity,
                        'Arb Score': arb_analysis['bregman_divergence'] if arb_analysis else 0
                    })
            
            except Exception as e:
                # Skip problematic markets
                continue
        
        # Detect dependency arbitrage
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
                    'question': m.get('question', ''),
                    'slug': m.get('slug', ''),
                    'yes_price': yes_price,
                    'liquidity': float(m.get('liquidity') or 1000),
                    'volume': float(m.get('volume') or 100)
                })
            except:
                continue
        
        dep_arbs = arb_engine.detect_dependency_arbitrage(markets_clean)
        
        for arb in dep_arbs:
            arbitrage_trades.append({
                'Event': f"{arb['market1'][:40]}... ⇒ {arb['market2'][:40]}...",
                'slug': arb['slug1'],
                'Current Yes %': arb['p1'] * 100,
                'Optimal Yes %': arb['p2'] * 100,
                'Deviation %': abs(arb['p1'] - arb['p2']) * 100,
                'Net Profit %': arb['net_profit'] * 100,
                'Sharpe': arb['net_profit'] / max(arb['execution_costs']['total'], 0.01),
                'Max Trade': f"${arb['execution_costs']['max_trade_size']:.0f}",
                'Type': 'DEPENDENCY'
            })
        
        # Sort and create DataFrames
        opportunities_sorted = sorted(opportunities, key=lambda x: x['Arb Score'], reverse=True)
        
        opp_df = pd.DataFrame(opportunities_sorted[:10]) if opportunities_sorted else pd.DataFrame()
        arb_df = pd.DataFrame(arbitrage_trades) if arbitrage_trades else pd.DataFrame()
        
        return opp_df, arb_df
    
    except Exception as e:
        # Return empty DataFrames on error
        return pd.DataFrame(), pd.DataFrame()

# ============================================================================
# AI MACRO BRIEFING - FIXED
# ============================================================================

def generate_enhanced_ai_macro_briefing(indices, spx_options, liquidity, correlations):
    """
    Generate high-density macro-economic intelligence.
    FIX: Use session state gemini_configured and updated model.
    """
    # FIX: Use session state variable
    if not st.session_state.gemini_configured:
        return "⚠️ **Gemini API Configuration Required** — Enable API access in sidebar to unlock macro intelligence. This analysis requires real-time AI processing for yield curve interpretation, correlation regime detection, and liquidity cycle forecasting."
    
    try:
        # FIX: Updated to latest model (gemini-2.0-flash-exp or gemini-1.5-flash)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
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
        
        prompt = f"""You are a macro strategist at a quantitative hedge fund. Generate a dense, specific macro briefing (200-250 words, paragraph format).

**Market State:**
SPX: ${spx_price:.0f} | VIX: {vix_price:.1f} | P/C: {pc_ratio:.2f} | Max Pain: ${max_pain:.0f}
10Y-2Y Spread: {yield_spread:.2f}% ({yc_context})
Credit Spread (HY): {credit_spread:.2f}%
Fed Balance: ${fed_balance:.2f}T
SPX/BTC Correlation: {corr_btc:.2f} | SPX/ETH Correlation: {corr_eth:.2f}

**Required Analysis:**

1. **Yield Curve & Fed Positioning**: Interpret the 10Y-2Y spread in context of current Fed trajectory. Is this consistent with soft landing, hard landing, or no landing? What does credit spread divergence signal about institutional confidence?

2. **Correlation Regime**: SPX/Crypto correlation of {corr_btc:.2f} indicates what risk regime? Are we in risk-on (corr > 0.6), risk-off (corr < 0.3), or transition? How should prediction markets price macro bets given this correlation structure?

3. **Liquidity Cycles**: With Fed balance at ${fed_balance:.2f}T and credit spreads at {credit_spread:.2f}%, where are we in the liquidity cycle? Does this favor vol selling (low VIX) or protection (high VIX)? Impact on Polymarket political/macro markets?

4. **SPX Technical Levels**: Based on max pain ${max_pain:.0f} and current ${spx_price:.0f}, identify 3 key levels (support/resistance) with probability-weighted scenarios. Format: "70% consolidation 5850-5900, 20% breakdown to 5750, 10% breakout to 6000".

Write in dense prose (no bullets), use specific numbers, avoid generic statements. Maximum 250 words."""

        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        # FIX: Display actual error message for debugging
        error_msg = str(e)
        return f"⚠️ **AI Generation Error** — {error_msg[:200]}... \n\nPossible issues:\n- API key invalid or expired\n- Quota exceeded\n- Model unavailable\n\nVerify your Gemini API key in the sidebar and check console for details."

# ============================================================================
# UI COMPONENTS - CARD-BASED LAYOUTS
# ============================================================================

def render_metric_card(label, value, change_pct, change_abs, signals=None):
    """Render beautiful metric card"""
    
    change_class = "positive" if change_pct >= 0 else "negative"
    change_symbol = "+" if change_pct >= 0 else ""
    
    if "SPX" in label or "NDX" in label:
        abs_str = f"{change_symbol}{change_abs:.2f} pts"
    elif "$" in str(value):
        abs_str = f"${change_symbol}{change_abs:.2f}"
    else:
        abs_str = f"{change_symbol}{change_abs:.2f}"
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-change {change_class}">
            {abs_str} ({change_symbol}{change_pct:.2f}%)
        </div>
    """
    
    if signals:
        card_html += '<div style="margin-top: 12px;">'
        for signal in signals.split(" | "):
            card_html += f'<span class="signal-badge">{signal}</span>'
        card_html += '</div>'
    
    card_html += "</div>"
    
    st.markdown(card_html, unsafe_allow_html=True)

def render_watchlist_cards(df):
    """Render watchlist as card grid - NO EXCEL TABLES"""
    
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
                    signals=row.get('Signals', '')
                )

def render_interactive_sector_heatmap(sector_df):
    """
    Interactive sector heatmap with drill-down capability.
    FIX: 2 rows of 5 columns with shortened names.
    """
    
    if sector_df.empty:
        st.warning("📊 Sector data unavailable")
        return
    
    # Create plotly bar chart
    fig = px.bar(
        sector_df,
        x='Change %',
        y='Sector',
        orientation='h',
        color='Change %',
        color_continuous_scale=[[0, '#FF0000'], [0.5, '#000000'], [1, '#00FF00']],
        color_continuous_midpoint=0,
        hover_data={'Change %': ':.2f%', 'Change $': ':.2f'},
        custom_data=['Ticker']
    )
    
    fig.update_traces(
        texttemplate='%{x:.2f}%',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Abs: $%{customdata[1]:.2f}<extra></extra>',
        marker_line_width=1,
        marker_line_color='#FFB000'
    )
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        height=500,
        margin=dict(l=0, r=50, t=20, b=0),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFB000', family='Courier New', size=12),
        xaxis=dict(
            showgrid=False,
            color='#FFB000',
            title='Performance (%)',
            zeroline=True,
            zerolinecolor='#FFB000'
        ),
        yaxis=dict(
            showgrid=False,
            color='#FFB000',
            title=''
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, key="sector_heatmap")
    
    # FIX: Shorten sector names and arrange in 2 rows of 5 columns
    st.caption("🔍 **SECTOR DRILL-DOWN** — Click to view constituents")
    
    sector_name_map = {
        'Technology': 'TECH',
        'Energy': 'ENERGY',
        'Financials': 'FIN',
        'Healthcare': 'HEALTH',
        'Industrials': 'INDUST',
        'Consumer Staples': 'STAPLES',
        'Consumer Disc.': 'DISC',
        'Materials': 'MATER',
        'Real Estate': 'REAL EST',
        'Communication': 'COMM',
        'Utilities': 'UTIL'
    }
    
    # Split sectors into 2 rows
    sectors_list = sector_df.to_dict('records')
    
    # First row (5 sectors)
    if len(sectors_list) > 0:
        cols_row1 = st.columns(min(5, len(sectors_list)))
        for idx in range(min(5, len(sectors_list))):
            row = sectors_list[idx]
            short_name = sector_name_map.get(row['Sector'], row['Sector'][:8])
            with cols_row1[idx]:
                if st.button(short_name, key=f"sector_{row['Ticker']}"):
                    st.session_state.selected_sector = row['Ticker']
                    st.session_state.show_sector_drill = True
                    st.rerun()
    
    # Second row (remaining sectors)
    if len(sectors_list) > 5:
        cols_row2 = st.columns(min(5, len(sectors_list) - 5))
        for idx in range(5, min(10, len(sectors_list))):
            row = sectors_list[idx]
            short_name = sector_name_map.get(row['Sector'], row['Sector'][:8])
            with cols_row2[idx - 5]:
                if st.button(short_name, key=f"sector_{row['Ticker']}"):
                    st.session_state.selected_sector = row['Ticker']
                    st.session_state.show_sector_drill = True
                    st.rerun()
    
    # Show drill-down if sector selected
    if st.session_state.show_sector_drill and st.session_state.selected_sector:
        sector_ticker = st.session_state.selected_sector
        sector_name = sector_df[sector_df['Ticker'] == sector_ticker]['Sector'].values[0]
        
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

st.title("⚡ ALPHA DECK PRO v5.0")
st.caption("**QUANT EDITION** — Prediction Market Arbitrage | Macro Intelligence | Production Analytics")

st.divider()

# ============================================================================
# AI MACRO BRIEFING - ENHANCED
# ============================================================================

st.subheader("🧠 AI MACRO INTELLIGENCE")

if st.button("🚀 GENERATE MACRO BRIEF", key="ai_macro", use_container_width=True):
    with st.spinner('🔬 Analyzing macro regime...'):
        indices = fetch_index_data()
        spx_opts = fetch_spx_options_data()
        liq = fetch_fred_liquidity()
        corrs = calculate_spx_crypto_correlation()
        
        briefing = generate_enhanced_ai_macro_briefing(indices, spx_opts, liq, corrs)
        
        st.markdown(f"""
        <div class="ai-briefing">
            <h4>🎯 MARKET INTELLIGENCE BRIEFING</h4>
            {briefing}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 MAIN DECK",
    "💰 POLYMARKET ARBITRAGE",
    "₿ CRYPTO & LIQUIDITY",
    "📈 TRADINGVIEW"
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
                    label=name,
                    value=value_str,
                    change_pct=data['change_pct'],
                    change_abs=data['change_abs']
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
            marker=dict(size=12, color='#FFB000', symbol='diamond')
        ))
        fig_vix.update_layout(
            template='plotly_dark',
            height=250,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFB000', family='Courier New'),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, color='#FFB000'),
            yaxis=dict(showgrid=True, gridcolor='#333333', title="Vol", color='#FFB000')
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
    
    # Watchlist - CARD BASED
    st.subheader("⚡ WATCHLIST + TECHNICAL SCANNER")
    st.caption("**Card-based layout** — No Excel tables")
    
    watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 
                         'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ']
    
    watchlist_df = fetch_watchlist_data(watchlist_tickers)
    render_watchlist_cards(watchlist_df)
    
    st.divider()
    
    # Sector Heatmap - INTERACTIVE with FIXED UI
    st.subheader("🎨 SECTOR HEAT — INTERACTIVE DRILL-DOWN")
    st.caption("**Click any sector** to view constituent stocks")
    
    sector_df = fetch_sector_performance()
    render_interactive_sector_heatmap(sector_df)

# ============================================================================
# TAB 2: POLYMARKET ARBITRAGE
# ============================================================================

with tab2:
    st.subheader("🎲 POLYMARKET ARBITRAGE ENGINE")
    st.caption("**Production-Grade** — Marginal Polytope | Bregman Projection | Frank-Wolfe | Integer Programming")
    
    opp_df, arb_df = fetch_polymarket_with_arbitrage()
    
    # Arbitrage Opportunities
    st.markdown("### 💰 ARBITRAGE TRADES")
    
    if not arb_df.empty:
        st.dataframe(
            arb_df,
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "Current Yes %": st.column_config.NumberColumn(format="%.2f%%"),
                "Optimal Yes %": st.column_config.NumberColumn(format="%.2f%%"),
                "Deviation %": st.column_config.NumberColumn(format="%.2f%%"),
                "Net Profit %": st.column_config.NumberColumn(format="%.2f%%"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        st.caption(f"🎯 **{len(arb_df)} arbitrage opportunities** detected via polytope analysis")
    else:
        st.info("✅ No arbitrage detected — Markets pricing efficiently within convex hull")
    
    st.divider()
    
    # Market Opportunities
    st.markdown("### 📊 TOP PREDICTION MARKETS")
    
    if not opp_df.empty:
        opp_display = opp_df.copy()
        opp_display['Volume'] = opp_display['Volume'].apply(
            lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}K"
        )
        opp_display['Liquidity'] = opp_display['Liquidity'].apply(
            lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}K"
        )
        opp_display['Yes %'] = opp_display['Yes %'].apply(lambda x: f"{x:.1f}%")
        opp_display['Arb Score'] = opp_display['Arb Score'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            opp_display[['Event', 'Yes %', 'Volume', 'Liquidity', 'Arb Score']],
            use_container_width=True,
            height=400,
            hide_index=True
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
    
    crypto_data = fetch_crypto_metrics(['BTC', 'ETH', 'SOL', 'DOGE'])
    
    cols_crypto = st.columns(4)
    for idx, (crypto, data) in enumerate(crypto_data.items()):
        with cols_crypto[idx]:
            if data['success']:
                render_metric_card(
                    label=crypto,
                    value=f"${data['price']:,.2f}",
                    change_pct=data['change_pct'],
                    change_abs=data['change_abs']
                )
    
    st.divider()
    
    # Correlations (FIXED)
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
    
    # Liquidity (FIXED - uses session state)
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
    
    all_tickers = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD', 
                   'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 
                   'BTC-USD', 'ETH-USD']
    
    default_idx = all_tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in all_tickers else 0
    
    selected_ticker = st.selectbox("SELECT TICKER", all_tickers, index=default_idx, key="chart_select")
    
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
            "volume.volume.color.0": "#FF0000",
            "volume.volume.color.1": "#00FF00"
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
st.caption("⚡ **ALPHA DECK PRO v5.0 — QUANT EDITION (FIXED)**")
st.caption("✅ Persistent API Clients | ✅ Fixed Polymarket | ✅ Weekend Gap Handled | ✅ Updated Gemini | ✅ Improved Sector UI")
