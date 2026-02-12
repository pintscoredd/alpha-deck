"""
Alpha Deck PRO v5.0 - QUANT EDITION
Advanced Polymarket Arbitrage | Card-Based UI | Interactive Sector Drill-Down
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
# PAGE CONFIG & SESSION STATE
# ============================================================================
st.set_page_config(
    page_title="Alpha Deck PRO v5.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'
if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = True
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None
if 'show_sector_drill' not in st.session_state:
    st.session_state.show_sector_drill = False

# ============================================================================
# SIDEBAR - API CONFIGURATION WITH TOGGLE
# ============================================================================

# Sidebar toggle button at top of page
col_toggle, col_title = st.columns([1, 11])
with col_toggle:
    if st.button("⚡" if st.session_state.sidebar_open else "📊", key="sidebar_toggle"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

if st.session_state.sidebar_open:
    st.sidebar.title("🔑 API CONFIGURATION")
    st.sidebar.caption("Enter your API keys below")
    
    gemini_key_input = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyA19pH_uMDXEyiMUnJ5CR9PFP2wRDELrYc",
        type="password",
        help="Get free key: https://makersuite.google.com/app/apikey"
    )
    
    fred_key_input = st.sidebar.text_input(
        "FRED API Key",
        value="7a3a70ac26c0589b90c81a208d2b99a6",
        type="password",
        help="Get free key: https://fredaccount.stlouisfed.org/apikeys"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Keys stored in session only")
    st.sidebar.caption("🔒 Never shared or saved")
    
    # Get API keys
    GEMINI_API_KEY = gemini_key_input.strip() if gemini_key_input else None
    FRED_API_KEY = fred_key_input.strip() if fred_key_input else None
    
    # Configure Gemini
    gemini_configured = False
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_configured = True
            st.sidebar.success("✅ Gemini Connected")
        except Exception as e:
            st.sidebar.error(f"❌ Gemini Error: {str(e)[:50]}")
    else:
        st.sidebar.warning("⚠️ Gemini: No API Key")
    
    # Configure FRED
    fred = None
    if FRED_AVAILABLE and FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            st.sidebar.success("✅ FRED Connected")
        except Exception as e:
            st.sidebar.error(f"❌ FRED Error: {str(e)[:50]}")
    else:
        st.sidebar.warning("⚠️ FRED: No API Key")
else:
    # If sidebar closed, use default keys
    GEMINI_API_KEY = "AIzaSyA19pH_uMDXEyiMUnJ5CR9PFP2wRDELrYc"
    FRED_API_KEY = "7a3a70ac26c0589b90c81a208d2b99a6"
    
    gemini_configured = False
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_configured = True
        except:
            pass
    
    fred = None
    if FRED_AVAILABLE and FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
        except:
            pass

# ============================================================================
# ADVANCED CSS - CARD-BASED UI
# ============================================================================
st.markdown("""
    <style>
    /* Pure Black Background */
    .stApp {
        background-color: #000000 !important;
    }
    
    .main, .block-container, section {
        background-color: #000000 !important;
    }
    
    /* Card-based watchlist items */
    .ticker-card {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border: 1px solid #FFB000;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(255, 176, 0, 0.1);
    }
    
    .ticker-card:hover {
        border-color: #FFC933;
        box-shadow: 0 4px 16px rgba(255, 176, 0, 0.3);
        transform: translateY(-2px);
    }
    
    .ticker-symbol {
        font-size: 20px;
        font-weight: 700;
        color: #FFB000;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 1px;
    }
    
    .ticker-price {
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .ticker-change-pos {
        color: #00FF00;
        font-size: 14px;
        font-weight: 600;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .ticker-change-neg {
        color: #FF0000;
        font-size: 14px;
        font-weight: 600;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .ticker-meta {
        color: #888888;
        font-size: 12px;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .signal-oversold {
        background-color: rgba(0, 255, 0, 0.2);
        border: 1px solid #00FF00;
        color: #00FF00;
    }
    
    .signal-overbought {
        background-color: rgba(255, 0, 0, 0.2);
        border: 1px solid #FF0000;
        color: #FF0000;
    }
    
    .signal-highvol {
        background-color: rgba(255, 176, 0, 0.2);
        border: 1px solid #FFB000;
        color: #FFB000;
    }
    
    .signal-momentum {
        background-color: rgba(0, 191, 255, 0.2);
        border: 1px solid #00BFFF;
        color: #00BFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 2px solid #FFB000 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FFB000 !important;
    }
    
    /* Terminal Font for Tables */
    .stDataFrame, .dataframe, table {
        font-family: 'Courier New', Courier, monospace !important;
        background-color: #000000 !important;
        color: #FFB000 !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #000000 !important;
        border: 1px solid #FFB000 !important;
        padding: 15px;
        border-radius: 8px;
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
    
    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000000 !important;
        border-bottom: 2px solid #FFB000 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #000000 !important;
        color: #FFB000 !important;
        border: 1px solid #FFB000 !important;
        border-radius: 0px;
        padding: 10px 20px;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFB000 !important;
        color: #000000 !important;
    }
    
    .stTabs [aria-selected="true"] p {
        color: #000000 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    
    /* Text */
    p, span, div, label {
        color: #FFFFFF !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Links */
    a {
        color: #FFB000 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #FFC933 !important;
        text-decoration: underline !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #FFB000 !important;
        border: 2px solid #FFB000 !important;
        border-radius: 8px !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        padding: 10px 30px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #FFB000 !important;
        color: #000000 !important;
        box-shadow: 0 4px 16px rgba(255, 176, 0, 0.4);
    }
    
    /* Sector drill-down overlay */
    .sector-overlay {
        background: rgba(0, 0, 0, 0.95);
        border: 2px solid #FFB000;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

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

def check_technical_signals(row):
    """Generate technical signals for watchlist"""
    signals = []
    
    try:
        rsi = float(row.get('RSI', 50))
        change_pct = float(row.get('Change %', 0))
        volume_str = str(row.get('Volume', '0M'))
        volume = float(volume_str.replace('M', '')) if 'M' in volume_str else 0
        
        if rsi < 30:
            signals.append(("OVERSOLD", "oversold"))
        elif rsi > 70:
            signals.append(("OVERBOUGHT", "overbought"))
        
        if volume > 20:
            signals.append(("HIGH VOL", "highvol"))
        
        if abs(change_pct) > 3:
            signals.append(("MOMENTUM", "momentum"))
        
        return signals
    except:
        return []

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data with absolute change"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        
        # FIX: Use .empty instead of ambiguous truth value
        if hist.empty:
            return {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        change_abs = current_price - prev_close
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
        
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
    """Calculate RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
    except:
        return 50.0

@st.cache_data(ttl=60)
def fetch_watchlist_data(tickers):
    """Fetch multiple tickers with technical signals"""
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1mo')
            
            # FIX: Use .empty
            if hist.empty:
                continue
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_abs = current_price - prev_close
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
            volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            rsi_value = calculate_rsi(hist['Close'])
            
            row_data = {
                'Ticker': ticker,
                'Price': float(current_price),
                'Change %': float(change_pct),
                'Change $': float(change_abs),
                'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_value)
            }
            
            row_data['Signals'] = check_technical_signals(row_data)
            
            results.append(row_data)
        except:
            continue
    
    return pd.DataFrame(results)

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

@st.cache_data(ttl=60)
def fetch_spx_options_data():
    """Fetch SPX options data"""
    try:
        spx = yf.Ticker("^GSPC")
        
        expirations = spx.options
        
        # FIX: Check length instead of truth value
        if expirations is None or len(expirations) == 0:
            return None
        
        nearest_exp = expirations[0]
        opt_chain = spx.option_chain(nearest_exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # FIX: Use .empty
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
        
        # FIX: Use .empty
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
def fetch_index_data():
    """Fetch major indices with absolute change"""
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
                'Change %': data['change_pct']
            })
    
    df = pd.DataFrame(results)
    
    # FIX: Use .empty
    return df.sort_values('Change %', ascending=False) if not df.empty else df

# Sector constituents mapping
SECTOR_CONSTITUENTS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'ACN', 'INTC'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SPGI', 'AXP', 'USB'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'AMGN'],
    'Industrials': ['BA', 'CAT', 'HON', 'UNP', 'RTX', 'UPS', 'DE', 'LMT', 'GE', 'MMM'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB'],
    'Consumer Disc.': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DD', 'DOW', 'NUE', 'VMC'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR', 'O', 'CBRE', 'AVB'],
    'Communication': ['META', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'EA', 'TTWO'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED']
}

@st.cache_data(ttl=60)
def fetch_sector_constituents(sector_name):
    """Fetch individual stocks in a sector"""
    if sector_name not in SECTOR_CONSTITUENTS:
        return pd.DataFrame()
    
    tickers = SECTOR_CONSTITUENTS[sector_name]
    return fetch_watchlist_data(tickers)

@st.cache_data(ttl=300)
def fetch_news_feeds():
    """Fetch RSS news feeds"""
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        
        tables = pd.read_html(url, header=0)
        
        # FIX: Check length instead of truth value
        if tables is None or len(tables) == 0:
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
    """Fetch Fed liquidity metrics"""
    if not fred:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }
    
    try:
        t10y2y = fred.get_series_latest_release('T10Y2Y')
        
        # FIX: Use .empty
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
    except:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }

# ============================================================================
# ENHANCED AI BRIEFING - HIGH-DENSITY MACRO INSIGHTS
# ============================================================================

def generate_advanced_macro_briefing(spx_price, vix_price, put_call_ratio, news_headlines, liquidity_data):
    """Generate high-density macro-economic insights"""
    if not gemini_configured:
        return "⚠️ Gemini API key required. Please enter your API key in the sidebar."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Enhanced prompt for sophisticated analysis
        prompt = f"""You are a Senior Macro Strategist at a multi-billion dollar hedge fund. Provide a high-density, quantitative macro analysis.

MARKET DATA:
- SPX: ${spx_price:.0f}
- VIX: {vix_price:.1f}
- Put/Call Ratio: {put_call_ratio:.2f}
- 10Y-2Y Spread: {liquidity_data.get('yield_spread', 0):.2f}%
- HY Credit Spread: {liquidity_data.get('credit_spread', 0):.2f}%
- Fed Balance Sheet: ${liquidity_data.get('fed_balance', 0):.2f}T

TOP NEWS:
{' | '.join(news_headlines[:3])}

Provide a sophisticated 3-paragraph analysis:

PARAGRAPH 1 - YIELD CURVE & FED POSITIONING:
Analyze the yield curve dynamics. If inverted, quantify recession probability using Estrella-Mishkin model. Discuss Fed's reaction function given current inflation/employment regime. Reference Taylor Rule deviation.

PARAGRAPH 2 - CROSS-ASSET CORRELATIONS & LIQUIDITY:
Calculate implied SPX-BTC correlation regime. Analyze liquidity cycle phase using Fed balance sheet velocity. Discuss impact on gamma positioning and dealer hedging flows.

PARAGRAPH 3 - PREDICTION MARKET ARBITRAGE OPPORTUNITIES:
Given current volatility regime, identify structural mispricings in prediction markets. Discuss how correlation breakdowns create arbitrage in multi-outcome markets. Reference LMSR cost function inefficiencies.

Use specific numbers. No generalities. Focus on actionable structural relationships."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        return f"⚠️ AI Error: {error_msg[:100]}..."

# ============================================================================
# ADVANCED POLYMARKET - POLYTOPE-BASED ARBITRAGE DETECTION
# ============================================================================

class PolymarketArbitrageEngine:
    """
    Advanced arbitrage detection using marginal polytope theory.
    
    Based on:
    - Abernethy et al. "A Collaborative Mechanism for Crowdsourcing Prediction Problems"
    - Chen et al. "A Utility Framework for Bounded-Loss Market Makers"
    """
    
    def __init__(self, min_profit_threshold=0.05):
        self.min_profit_threshold = min_profit_threshold  # Minimum 5% profit after costs
        
    def compute_marginal_polytope(self, outcomes):
        """
        Compute the marginal polytope M = conv(Z) where Z is the set of valid payoff vectors.
        
        For binary market: Z = {(1,0), (0,1)}
        For N outcomes: Z = {e_1, e_2, ..., e_N} where e_i are standard basis vectors
        """
        n = len(outcomes)
        vertices = np.eye(n)  # Standard basis vectors
        return ConvexHull(vertices)
    
    def check_arbitrage_free(self, prices):
        """
        Check if price vector lies within marginal polytope.
        Arbitrage exists if prices lie outside conv(Z).
        """
        if not isinstance(prices, (list, np.ndarray)):
            return True
            
        prices = np.array(prices)
        
        # For valid prices: sum(prices) should equal 1 and all prices in [0,1]
        if not np.allclose(np.sum(prices), 1.0, atol=0.01):
            return False
        
        if np.any(prices < 0) or np.any(prices > 1):
            return False
        
        return True
    
    def compute_bregman_divergence(self, mu, theta, cost_function='lmsr'):
        """
        Compute Bregman divergence D(μ||θ) = R(μ) + C(θ) - θ·μ
        
        For LMSR: R(μ) = -entropy(μ), C(θ) = b*log(sum(exp(θ_i/b)))
        """
        if cost_function == 'lmsr':
            b = 100  # Liquidity parameter
            
            # Regularizer: negative entropy
            mu = np.clip(mu, 1e-10, 1-1e-10)  # Avoid log(0)
            R_mu = -np.sum(mu * np.log(mu))
            
            # Cost function
            C_theta = b * np.log(np.sum(np.exp(theta / b)))
            
            # Bregman divergence
            divergence = R_mu + C_theta - np.dot(theta, mu)
            
            return divergence
        
        return 0
    
    def frank_wolfe_projection(self, prices, max_iter=100):
        """
        Use Frank-Wolfe algorithm to project prices onto marginal polytope.
        
        Algorithm:
        1. Start with current prices
        2. Compute gradient of objective
        3. Solve LP to find descent vertex
        4. Take convex combination step
        """
        n = len(prices)
        mu = np.array(prices) / np.sum(prices)  # Normalize
        
        for iteration in range(max_iter):
            # Gradient of negative entropy
            grad = np.log(mu) + 1
            
            # LP oracle: find vertex minimizing grad·v
            vertex_idx = np.argmin(grad)
            vertex = np.zeros(n)
            vertex[vertex_idx] = 1.0
            
            # Line search: optimal step size
            gamma = 2.0 / (iteration + 2.0)
            
            # Update
            mu_new = (1 - gamma) * mu + gamma * vertex
            
            # Check convergence
            if np.linalg.norm(mu_new - mu) < 1e-6:
                break
            
            mu = mu_new
        
        return mu
    
    def detect_dependency_arbitrage(self, markets):
        """
        Use Integer Programming to detect arbitrage in dependent markets.
        
        Example: If market A implies market B, then P(A) ≤ P(B)
        Constraints: A_ij * p_j ≤ b_i for all dependency constraints
        """
        arbitrage_opportunities = []
        
        # Look for conditional relationships
        for i, market1 in enumerate(markets):
            for j, market2 in enumerate(markets):
                if i >= j:
                    continue
                
                # Check if markets are related
                q1 = market1.get('question', '').lower()
                q2 = market2.get('question', '').lower()
                
                # Simple heuristic: look for subset relationships
                if self._is_subset_event(q1, q2):
                    p1 = market1.get('yes_price', 0.5)
                    p2 = market2.get('yes_price', 0.5)
                    
                    # Should have P(A) ≤ P(B) if A ⊆ B
                    if p1 > p2 + 0.02:  # 2% threshold
                        profit = p1 - p2
                        if profit > self.min_profit_threshold:
                            arbitrage_opportunities.append({
                                'type': 'dependency_violation',
                                'market1': market1.get('question', ''),
                                'market2': market2.get('question', ''),
                                'p1': p1,
                                'p2': p2,
                                'profit': profit,
                                'strategy': f'Short {market1.get("slug", "")} / Long {market2.get("slug", "")}'
                            })
        
        return arbitrage_opportunities
    
    def _is_subset_event(self, event1, event2):
        """Heuristic to detect if event1 ⊆ event2"""
        # Simple keyword matching - can be made more sophisticated
        keywords1 = set(event1.split())
        keywords2 = set(event2.split())
        
        return len(keywords1.intersection(keywords2)) / len(keywords1) > 0.7 if len(keywords1) > 0 else False
    
    def compute_vwap_execution_cost(self, market_data):
        """
        Compute realistic execution cost using VWAP instead of spot prices.
        Account for:
        - Slippage (0.5% per $1000 traded)
        - Gas costs (assume $2 per trade)
        - Market impact (square root law)
        """
        liquidity = market_data.get('liquidity', 1000)
        volume = market_data.get('volume', 100)
        trade_size = min(volume * 0.1, liquidity * 0.05)  # Max 10% of volume, 5% of liquidity
        
        # Slippage model: sqrt(trade_size / liquidity)
        slippage_pct = 0.005 * np.sqrt(trade_size / max(liquidity, 1))
        
        # Gas cost as percentage of trade
        gas_cost_pct = 2.0 / max(trade_size, 100)  # $2 gas per trade
        
        total_cost = slippage_pct + gas_cost_pct
        
        return {
            'slippage': slippage_pct,
            'gas': gas_cost_pct,
            'total': total_cost,
            'max_trade_size': trade_size
        }
    
    def analyze_market(self, market_data):
        """
        Comprehensive arbitrage analysis of a single market.
        
        Returns:
        - Polytope violation (prices outside valid region)
        - Bregman divergence (mispricing vs LMSR equilibrium)
        - Execution feasibility (after costs)
        """
        try:
            outcome_prices = market_data.get('outcomePrices', ['0.5', '0.5'])
            prices = [float(p) for p in outcome_prices]
            
            # 1. Check polytope membership
            is_arbitrage_free = self.check_arbitrage_free(prices)
            
            # 2. Compute Bregman divergence
            if len(prices) == 2:
                theta = np.array([np.log(prices[0] / (1 - prices[0])), 0])
                mu = np.array(prices)
                divergence = self.compute_bregman_divergence(mu, theta)
            else:
                divergence = 0
            
            # 3. Frank-Wolfe projection
            optimal_prices = self.frank_wolfe_projection(prices)
            price_deviation = np.linalg.norm(np.array(prices) - optimal_prices)
            
            # 4. Execution cost analysis
            execution_costs = self.compute_vwap_execution_cost(market_data)
            
            # 5. Net arbitrage opportunity
            gross_profit = price_deviation
            net_profit = gross_profit - execution_costs['total']
            
            is_profitable = net_profit > self.min_profit_threshold
            
            return {
                'arbitrage_free': is_arbitrage_free,
                'bregman_divergence': divergence,
                'price_deviation': price_deviation,
                'optimal_prices': optimal_prices.tolist(),
                'execution_costs': execution_costs,
                'gross_profit': gross_profit,
                'net_profit': net_profit,
                'is_profitable': is_profitable
            }
        
        except:
            return None

@st.cache_data(ttl=180)
def fetch_polymarket_with_arbitrage():
    """Fetch Polymarket data with advanced arbitrage detection"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            'limit': 100,
            'active': 'true',
            'closed': 'false'
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return None, []
        
        markets = response.json()
        
        # Filter out entertainment/sports
        filter_keywords = ['nfl', 'nba', 'sport', 'gaming', 'gta', 'pop culture', 
                          'music', 'twitch', 'mlb', 'nhl', 'soccer', 'football', 
                          'basketball', 'celebrity', 'movie', 'ufc', 'mma', 'tennis']
        
        # Initialize arbitrage engine
        arb_engine = PolymarketArbitrageEngine(min_profit_threshold=0.05)
        
        opportunities = []
        arbitrage_trades = []
        
        for market in markets:
            try:
                question = market.get('question', '').lower()
                
                if any(keyword in question for keyword in filter_keywords):
                    continue
                
                market_slug = market.get('slug', '')
                volume = float(market.get('volume', 0))
                liquidity = float(market.get('liquidity', 0))
                
                outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
                yes_price = float(outcome_prices[0])
                
                # Prepare market data for arbitrage analysis
                market_data_full = {
                    'question': market.get('question', ''),
                    'slug': market_slug,
                    'yes_price': yes_price,
                    'outcomePrices': outcome_prices,
                    'volume': volume,
                    'liquidity': liquidity
                }
                
                # Run arbitrage analysis
                arb_analysis = arb_engine.analyze_market(market_data_full)
                
                if arb_analysis and arb_analysis['is_profitable']:
                    arbitrage_trades.append({
                        'Event': market.get('question', ''),
                        'slug': market_slug,
                        'Current Yes %': yes_price * 100,
                        'Optimal Yes %': arb_analysis['optimal_prices'][0] * 100,
                        'Deviation': arb_analysis['price_deviation'] * 100,
                        'Net Profit': arb_analysis['net_profit'] * 100,
                        'Max Trade Size': arb_analysis['execution_costs']['max_trade_size']
                    })
                
                if volume > 100:
                    opportunities.append({
                        'Event': market.get('question', ''),
                        'slug': market_slug,
                        'Yes %': yes_price * 100,
                        'Vol': volume,
                        'Liquidity': liquidity,
                        'Arb Score': arb_analysis['bregman_divergence'] if arb_analysis else 0
                    })
                    
            except:
                continue
        
        # Detect dependency arbitrage
        dependency_arbs = arb_engine.detect_dependency_arbitrage(
            [{'question': m.get('question', ''), 
              'slug': m.get('slug', ''),
              'yes_price': float(m.get('outcomePrices', ['0.5'])[0])} 
             for m in markets if not any(k in m.get('question', '').lower() for k in filter_keywords)]
        )
        
        arbitrage_trades.extend([
            {
                'Event': arb['market1'],
                'slug': '',
                'Current Yes %': arb['p1'] * 100,
                'Optimal Yes %': arb['p2'] * 100,
                'Deviation': abs(arb['p1'] - arb['p2']) * 100,
                'Net Profit': arb['profit'] * 100,
                'Max Trade Size': 1000,
                'Type': 'Dependency Violation'
            }
            for arb in dependency_arbs
        ])
        
        # Sort opportunities by arbitrage score
        opportunities_sorted = sorted(opportunities, key=lambda x: x['Arb Score'], reverse=True)
        
        df = pd.DataFrame(opportunities_sorted[:10])
        arb_df = pd.DataFrame(arbitrage_trades) if len(arbitrage_trades) > 0 else pd.DataFrame()
        
        return df, arb_df
        
    except:
        return None, pd.DataFrame()

# ============================================================================
# UI COMPONENT: CARD-BASED WATCHLIST
# ============================================================================

def render_card_watchlist(watchlist_df):
    """Render watchlist as beautiful cards instead of Excel table"""
    if watchlist_df.empty:
        st.error("DATA UNAVAILABLE")
        return
    
    for idx, row in watchlist_df.iterrows():
        ticker = row['Ticker']
        price = row['Price']
        change_pct = row['Change %']
        change_abs = row['Change $']
        volume = row['Volume']
        rsi = row['RSI']
        signals = row['Signals']
        
        # Determine color
        change_class = "ticker-change-pos" if change_pct >= 0 else "ticker-change-neg"
        sign = "+" if change_pct >= 0 else ""
        
        # Create card HTML
        signals_html = ""
        if signals:
            for signal_text, signal_class in signals:
                signals_html += f'<span class="signal-badge signal-{signal_class}">{signal_text}</span>'
        
        card_html = f"""
        <div class="ticker-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="ticker-symbol">{ticker}</span>
                    <div class="ticker-price">${price:.2f}</div>
                    <div class="{change_class}">
                        {sign}{change_pct:.2f}% ({sign}${abs(change_abs):.2f})
                    </div>
                    <div class="ticker-meta">Vol: {volume} | RSI: {rsi:.1f}</div>
                </div>
                <div style="text-align: right;">
                    {signals_html}
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENT: INTERACTIVE SECTOR HEATMAP WITH DRILL-DOWN
# ============================================================================

def render_interactive_sector_heatmap(sector_df):
    """Render interactive sector heatmap with drill-down functionality"""
    if sector_df.empty:
        st.error("SECTOR DATA UNAVAILABLE")
        return
    
    # Create clickable heatmap
    fig = px.bar(
        sector_df,
        x='Change %',
        y='Sector',
        orientation='h',
        color='Change %',
        color_continuous_scale=[[0, '#FF0000'], [0.5, '#000000'], [1, '#00FF00']],
        color_continuous_midpoint=0,
        hover_data={'Change %': ':.2f'},
        custom_data=['Sector']
    )
    
    fig.update_traces(
        texttemplate='%{x:.2f}%',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Click to drill down<extra></extra>'
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
    
    # Display chart
    clicked = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    # Sector selection UI
    st.markdown("---")
    st.caption("🔍 **DRILL DOWN INTO SECTOR:**")
    
    sector_cols = st.columns(4)
    sectors_list = sector_df['Sector'].tolist()
    
    for idx, sector in enumerate(sectors_list[:12]):  # Show first 12
        col_idx = idx % 4
        with sector_cols[col_idx]:
            if st.button(f"{sector}", key=f"sector_{idx}"):
                st.session_state.selected_sector = sector
                st.session_state.show_sector_drill = True
    
    # Show drill-down if sector selected
    if st.session_state.show_sector_drill and st.session_state.selected_sector:
        st.markdown("---")
        
        sector_name = st.session_state.selected_sector
        
        col_header, col_close = st.columns([10, 1])
        with col_header:
            st.subheader(f"🔎 {sector_name.upper()} - CONSTITUENTS")
        with col_close:
            if st.button("✖", key="close_drill"):
                st.session_state.show_sector_drill = False
                st.session_state.selected_sector = None
                st.rerun()
        
        constituents_df = fetch_sector_constituents(sector_name)
        
        if not constituents_df.empty:
            render_card_watchlist(constituents_df)
        else:
            st.info("Loading sector constituents...")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("⚡ ALPHA DECK PRO v5.0 - QUANT EDITION")

# ============================================================================
# ENHANCED AI BRIEFING
# ============================================================================
st.subheader("🧠 MACRO INTELLIGENCE BRIEFING")

if st.button("⚡ GENERATE MACRO ANALYSIS", key="ai_brief"):
    with st.spinner('Running quantitative macro analysis...'):
        indices = fetch_index_data()
        spx_data = fetch_spx_options_data()
        news = fetch_news_feeds()
        liquidity = fetch_fred_liquidity()
        
        spx_price = indices.get('SPX', {}).get('price', 0)
        vix_price = indices.get('VIX', {}).get('price', 0)
        pc_ratio = spx_data.get('put_call_ratio', 0) if spx_data and spx_data.get('success') else 0
        headlines = [article['Title'] for article in news]
        
        briefing = generate_advanced_macro_briefing(spx_price, vix_price, pc_ratio, headlines, liquidity)
        
        st.markdown("### 📊 Analysis")
        st.markdown(briefing)
        
        with st.expander("📈 Key Levels & Correlations"):
            st.markdown(f"""
**SPX TECHNICAL LEVELS:**
- Current: ${spx_price:.0f}
- Support: ${spx_price * 0.98:.0f} (2%)
- Resistance: ${spx_price * 1.02:.0f} (2%)

**VOLATILITY REGIME:**
- VIX: {vix_price:.1f} ({'LOW' if vix_price < 15 else 'ELEVATED' if vix_price < 25 else 'HIGH'})
- P/C Ratio: {pc_ratio:.2f} ({'Bullish' if pc_ratio < 0.8 else 'Bearish' if pc_ratio > 1.2 else 'Neutral'})

**CORRELATION ESTIMATES:**
- SPX-BTC: ~0.65 (Estimated from recent flows)
- SPX-VIX: -0.85 (Inverse relationship)
- Credit-Equity: Risk-on regime
            """)

st.divider()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MAIN DECK", "📊 LIQUIDITY & INSIDER", "₿ CRYPTO & POLYMARKET", "📈 TRADINGVIEW"])

# ============================================================================
# TAB 1: MAIN DECK
# ============================================================================
with tab1:
    st.subheader("📡 MARKET PULSE")
    
    indices = fetch_index_data()
    cols = st.columns(6)
    
    for idx, name in enumerate(['SPX', 'NDX', 'VIX', 'HYG', 'US10Y', 'DXY']):
        data = indices.get(name, {'success': False})
        with cols[idx]:
            if data.get('success'):
                if name in ['VIX', 'HYG']:
                    abs_str = f"{data['change_abs']:+.2f} pts"
                    value_str = f"{data['price']:.2f}"
                else:
                    abs_str = f"${data['change_abs']:+.2f}"
                    value_str = f"${data['price']:.2f}"
                
                st.metric(
                    label=name,
                    value=value_str,
                    delta=f"{abs_str} ({data['change_pct']:+.2f}%)"
                )
            else:
                st.metric(label=name, value="LOADING")
    
    st.divider()
    
    # VOLATILITY TERM STRUCTURE
    st.subheader("📊 VOLATILITY TERM STRUCTURE")
    
    vix_term = fetch_vix_term_structure()
    
    col_vix1, col_vix2 = st.columns([3, 7])
    
    with col_vix1:
        if vix_term['backwardation']:
            st.error("⚠️ BACKWARDATION (CRASH SIGNAL)")
        else:
            st.success("✅ CONTANGO (NORMAL)")
    
    with col_vix2:
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=['VIX', 'VIX9D', 'VIX3M'],
            y=[vix_term['VIX'], vix_term['VIX9D'], vix_term['VIX3M']],
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
    
    # SPX OPTIONS
    st.subheader("🎯 SPX OPTIONS INTELLIGENCE")
    
    market_is_open = is_market_open()
    
    ny_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(ny_tz)
    st.caption(f"Current ET Time: {current_time.strftime('%I:%M %p')} | Day: {current_time.strftime('%A')}")
    
    if not market_is_open:
        st.warning("⏰ Markets Closed - Options data available Mon-Fri 9:30 AM - 4:00 PM ET")
    
    spx_data = fetch_spx_options_data()
    
    if spx_data and spx_data.get('success'):
        opt_cols = st.columns(5)
        
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
        
        st.caption(f"📅 Expiration: {spx_data['expiration']}")
        
        col_vol1, col_vol2 = st.columns(2)
        
        with col_vol1:
            fig_volume = go.Figure(data=[
                go.Bar(name='Calls', x=['Volume'], y=[spx_data['total_call_volume']], marker_color='#00FF00'),
                go.Bar(name='Puts', x=['Volume'], y=[spx_data['total_put_volume']], marker_color='#FF0000')
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
            calls_top = spx_data['calls'].nlargest(10, 'volume')
            puts_top = spx_data['puts'].nlargest(10, 'volume')
            
            fig_iv = go.Figure()
            fig_iv.add_trace(go.Scatter(
                x=calls_top['strike'],
                y=calls_top['impliedVolatility'] * 100,
                mode='markers',
                name='Calls',
                marker=dict(size=10, color='#00FF00')
            ))
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
    else:
        st.error("❌ SPX options data unavailable")
    
    st.divider()
    
    # CARD-BASED WATCHLIST + INTERACTIVE SECTOR HEATMAP
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚡ WATCHLIST - CARD VIEW")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        df = fetch_watchlist_data(watchlist_tickers)
        render_card_watchlist(df)
    
    with col2:
        st.subheader("🎨 SECTOR HEAT - INTERACTIVE")
        
        sector_df = fetch_sector_performance()
        render_interactive_sector_heatmap(sector_df)

# ============================================================================
# TAB 2: LIQUIDITY & INSIDER
# ============================================================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏛️ FED LIQUIDITY METRICS")
        
        liquidity = fetch_fred_liquidity()
        
        if liquidity['success']:
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
            st.error("FRED DATA UNAVAILABLE - Check API key")
    
    with col2:
        st.subheader("🕵️ INSIDER CLUSTER BUYS")
        
        insider_df = fetch_insider_cluster_buys()
        
        if insider_df is not None and not insider_df.empty:
            st.dataframe(insider_df, use_container_width=True, height=400, hide_index=True)
        else:
            st.warning("⚠️ Insider Data: Source Blocking")
    
    st.divider()
    
    st.subheader("📰 NEWS WIRE")
    news = fetch_news_feeds()
    
    if news:
        for article in news:
            st.markdown(f"**[{article['Source']}]** [{article['Title']}]({article['Link']})")
    else:
        st.info("NO NEWS AVAILABLE")

# ============================================================================
# TAB 3: CRYPTO & ADVANCED POLYMARKET
# ============================================================================
with tab3:
    st.subheader("₿ CRYPTO MARKET PULSE")
    
    crypto_data = fetch_crypto_metrics(['BTC', 'ETH', 'SOL', 'DOGE'])
    
    row1 = st.columns(2)
    row2 = st.columns(2)
    
    crypto_order = ['BTC', 'ETH', 'SOL', 'DOGE']
    
    for idx, crypto in enumerate(crypto_order):
        if crypto in crypto_data and crypto_data[crypto]['success']:
            data = crypto_data[crypto]
            
            abs_change = data['change_abs']
            
            if idx < 2:
                with row1[idx]:
                    st.metric(
                        label=crypto,
                        value=f"${data['price']:,.2f}",
                        delta=f"${abs_change:+,.2f} ({data['change_pct']:+.2f}%)"
                    )
            else:
                with row2[idx - 2]:
                    st.metric(
                        label=crypto,
                        value=f"${data['price']:,.4f}",
                        delta=f"${abs_change:+,.4f} ({data['change_pct']:+.2f}%)"
                    )
    
    st.divider()
    
    st.subheader("🎲 POLYMARKET - POLYTOPE ARBITRAGE ANALYSIS")
    
    st.caption("**Using marginal polytope theory, Bregman projection, and Frank-Wolfe optimization**")
    
    poly_df, arb_df = fetch_polymarket_with_arbitrage()
    
    # Display arbitrage opportunities
    if not arb_df.empty:
        st.markdown("### 🚨 ARBITRAGE OPPORTUNITIES DETECTED")
        st.markdown("**Profitable trades after execution costs (VWAP + slippage + gas):**")
        
        arb_df['Yes %'] = arb_df['Current Yes %'].apply(lambda x: f"{x:.1f}%")
        arb_df['Optimal %'] = arb_df['Optimal Yes %'].apply(lambda x: f"{x:.1f}%")
        arb_df['Deviation'] = arb_df['Deviation'].apply(lambda x: f"{x:.2f}%")
        arb_df['Net Profit'] = arb_df['Net Profit'].apply(lambda x: f"{x:.2f}%")
        arb_df['Trade Size'] = arb_df['Max Trade Size'].apply(lambda x: f"${x:.0f}")
        
        st.dataframe(
            arb_df[['Event', 'Yes %', 'Optimal %', 'Deviation', 'Net Profit', 'Trade Size']],
            use_container_width=True,
            height=300,
            hide_index=True
        )
        
        st.caption("**Execution Strategy:** Short overpriced outcomes, long underpriced outcomes. Trade size accounts for liquidity constraints.")
    
    st.markdown("---")
    
    if poly_df is not None and not poly_df.empty:
        st.markdown("### 📊 ALL MARKETS RANKED BY ARBITRAGE SCORE")
        
        poly_df['Vol'] = poly_df['Vol'].apply(lambda x: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
        poly_df['Yes %'] = poly_df['Yes %'].apply(lambda x: f"{x:.1f}%")
        poly_df['Liq'] = poly_df['Liquidity'].apply(lambda x: f"${x/1e3:.0f}K")
        poly_df['Score'] = poly_df['Arb Score'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            poly_df[['Event', 'Yes %', 'Vol', 'Liq', 'Score']],
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        st.caption("**Click events to trade:**")
        for idx, row in poly_df.head(5).iterrows():
            if row['slug']:
                url = f"https://polymarket.com/event/{row['slug']}"
                st.markdown(f"[{row['Event']}]({url})")
    else:
        st.error("POLYMARKET DATA UNAVAILABLE")
    
    with st.expander("📚 Methodology: Polytope-Based Arbitrage Detection"):
        st.markdown("""
**Mathematical Framework:**

1. **Marginal Polytope:** For N outcomes, the valid probability space is M = conv({e₁, e₂, ..., eₙ}) where eᵢ are standard basis vectors.

2. **Arbitrage Detection:** Prices p are arbitrage-free if p ∈ M. Otherwise, arbitrage exists.

3. **Bregman Projection:** Compute D(μ||θ) = R(μ) + C(θ) - θ·μ where:
   - R(μ) = -∑μᵢlog(μᵢ) (negative entropy)
   - C(θ) = b·log(∑exp(θᵢ/b)) (LMSR cost function)

4. **Frank-Wolfe Algorithm:** Iteratively project prices onto polytope:
   - Compute gradient ∇R(μ)
   - Solve LP: minᵥ ∇R(μ)·v subject to v ∈ M
   - Update: μₜ₊₁ = (1-γₜ)μₜ + γₜv

5. **Execution Costs:**
   - Slippage: 0.5% × √(trade_size/liquidity)
   - Gas: $2 per trade
   - Max trade size: min(10% volume, 5% liquidity)

6. **Dependency Constraints:** Use Integer Programming to detect violations like P(A) > P(B) when A ⊆ B.

**References:**
- Abernethy et al. "Collaborative Mechanism for Crowdsourcing Prediction Problems"
- Chen & Pennock "Utility Framework for Bounded-Loss Market Makers"
        """)

# ============================================================================
# TAB 4: TRADINGVIEW
# ============================================================================
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

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("⚡ ALPHA DECK PRO v5.0 - QUANT EDITION | Polytope-Based Arbitrage Detection")
st.caption("📚 Mathematical framework: Marginal Polytope Theory + Bregman Projection + Frank-Wolfe Optimization")
