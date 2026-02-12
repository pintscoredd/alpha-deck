"""
Alpha Deck PRO v4.0 - GOLD MASTER (FIXED)
Bloomberg-Style Trading Terminal - All Issues Resolved
"""

import os
import logging
import streamlit as st

# Resilient imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import feedparser
from datetime import datetime
import numpy as np
import pytz

# Optional imports
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AlphaDeckPRO")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Alpha Deck PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'

# ============================================================
# SIDEBAR - API CONFIGURATION
# ============================================================
st.sidebar.title("🔑 API CONFIGURATION")
st.sidebar.caption("Enter your API keys below")

gemini_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value="",
    type="password",
    help="Get free key: https://makersuite.google.com/app/apikey"
)

fred_key_input = st.sidebar.text_input(
    "FRED API Key",
    value="",
    type="password",
    help="Get free key: https://fredaccount.stlouisfed.org/apikeys"
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Keys are stored in session only")
st.sidebar.caption("🔒 Never shared or saved")

# Get API keys with environment variable fallback
GEMINI_API_KEY = gemini_key_input.strip() if gemini_key_input else os.getenv("GEMINI_API_KEY")
FRED_API_KEY = fred_key_input.strip() if fred_key_input else os.getenv("FRED_API_KEY")

# Configure Gemini
gemini_configured = False
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        st.sidebar.success("✅ Gemini Connected")
        logger.info("Gemini configured")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini Error: {str(e)[:50]}")
        logger.error(f"Gemini error: {e}")
else:
    st.sidebar.warning("⚠️ Gemini: No API Key")

# Configure FRED
fred = None
if FRED_AVAILABLE and FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
        st.sidebar.success("✅ FRED Connected")
        logger.info("FRED configured")
    except Exception as e:
        st.sidebar.error(f"❌ FRED Error: {str(e)[:50]}")
        logger.error(f"FRED error: {e}")
else:
    st.sidebar.warning("⚠️ FRED: No API Key")

# ============================================================
# AMBER TERMINAL THEME CSS
# ============================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #000000 !important;
    }
    
    .main, .block-container, section {
        background-color: #000000 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 2px solid #FFB000 !important;
    }
    
    .stDataFrame, .dataframe, table {
        font-family: 'Courier New', Courier, monospace !important;
        background-color: #000000 !important;
        color: #FFB000 !important;
    }
    
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
        text-transform: uppercase;
    }
    
    .stMetric .metric-value {
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
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
    
    h1, h2, h3, h4 {
        color: #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    
    p, span, div, label {
        color: #FFFFFF !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    a {
        color: #FFB000 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #FFC933 !important;
        text-decoration: underline !important;
    }
    
    hr {
        border-color: #FFB000 !important;
        margin: 1rem 0;
    }
    
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
    
    .stCaption {
        color: #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    .stAlert {
        background-color: #1a1a1a !important;
        color: #FFB000 !important;
        border: 1px solid #FFB000 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
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
    """Check if US stock market is currently open"""
    try:
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)
        
        if now.weekday() >= 5:  # Weekend
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
    """Generate technical signals"""
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
    except:
        return "—"

# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data with absolute change"""
    try:
        if not YFINANCE_AVAILABLE:
            return {'price': 0, 'change_pct': 0, 'change_abs': 0, 'volume': 0, 'success': False}
        
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
    """Calculate RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
    except:
        return 50.0

@st.cache_data(ttl=60)
def fetch_watchlist_data(tickers):
    """Fetch watchlist with technical signals"""
    results = []
    
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame(results)
    
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
            logger.warning(f"Watchlist error for {ticker}: {e}")
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
            'backwardation': (vix9d['price'] > vix['price']) if (vix9d.get('success') and vix.get('success')) else False
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
    """Robust SPX/SPY options fetch with fallback"""
    try:
        if not YFINANCE_AVAILABLE:
            return {'success': False, 'error': 'yfinance not installed'}
        
        # Try multiple tickers
        for ticker in ["^GSPC", "SPY"]:
            try:
                opt_ticker = yf.Ticker(ticker)
                expirations = opt_ticker.options
                
                if not expirations:
                    continue
                
                nearest_exp = expirations[0]
                opt_chain = opt_ticker.option_chain(nearest_exp)
                
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                if calls.empty or puts.empty:
                    continue
                
                # Calculate metrics
                total_call_volume = int(calls['volume'].fillna(0).sum())
                total_put_volume = int(puts['volume'].fillna(0).sum())
                put_call_ratio = (total_put_volume / total_call_volume) if total_call_volume > 0 else 0
                
                total_call_oi = int(calls['openInterest'].fillna(0).sum())
                total_put_oi = int(puts['openInterest'].fillna(0).sum())
                put_call_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0
                
                # Max pain
                calls_oi = calls.groupby('strike')['openInterest'].sum()
                puts_oi = puts.groupby('strike')['openInterest'].sum()
                total_oi = calls_oi.add(puts_oi, fill_value=0)
                max_pain = float(total_oi.idxmax()) if not total_oi.empty else 0
                
                # Average IV
                avg_call_iv = float(calls['impliedVolatility'].mean() * 100)
                avg_put_iv = float(puts['impliedVolatility'].mean() * 100)
                
                return {
                    'success': True,
                    'ticker': ticker,
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
                
            except Exception as e:
                logger.warning(f"Options fetch failed for {ticker}: {e}")
                continue
        
        return {'success': False, 'error': 'All tickers failed'}
        
    except Exception as e:
        logger.error(f"SPX options error: {e}")
        return {'success': False, 'error': str(e)}

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
        if data.get('success'):
            results.append({
                'Sector': name,
                'Change %': data['change_pct']
            })
    
    df = pd.DataFrame(results)
    return df.sort_values('Change %', ascending=False) if not df.empty else df

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
                    'Title': entry.get('title', '')[:200],
                    'Link': entry.get('link', '')
                })
        except Exception as e:
            logger.error(f"News fetch error for {source}: {e}")
    
    return articles[:15]

@st.cache_data(ttl=3600)
def fetch_insider_cluster_buys():
    """Scrape OpenInsider"""
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
            
            return df[columns_to_keep].head(10)
        
        return None
        
    except Exception as e:
        logger.error(f"Insider fetch error: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_fred_liquidity():
    """Fetch FRED liquidity metrics"""
    if not fred:
        return {'yield_spread': 0, 'credit_spread': 0, 'fed_balance': 0, 'success': False}
    
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
        logger.error(f"FRED fetch error: {e}")
        return {'yield_spread': 0, 'credit_spread': 0, 'fed_balance': 0, 'success': False}

def generate_ai_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    """Generate AI market briefing"""
    if not gemini_configured:
        return "⚠️ Gemini API key required. Enter in sidebar."
    
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
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"AI briefing error: {e}")
        return f"⚠️ AI Error: {error_msg[:100]}"

@st.cache_data(ttl=180)
def fetch_polymarket_advanced_analytics():
    """Fetch Polymarket opportunities"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {'limit': 100, 'active': 'true', 'closed': 'false'}
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return None
        
        markets = response.json()
        
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
                yes_price = float(outcome_prices[0])
                
                total_prob = yes_price + (1 - yes_price)
                prob_deviation = abs(1.0 - total_prob)
                
                volume_velocity = (volume_24h / volume * 100) if volume > 0 else 0
                liquidity_score = liquidity / 1000
                edge_score = prob_deviation * 100
                activity_score = volume_velocity if volume_24h > 1000 else 0
                
                opportunity_score = (edge_score * 3) + (activity_score * 2) + (liquidity_score * 1)
                
                if volume > 100:
                    opportunities.append({
                        'Event': market.get('question', ''),
                        'slug': market_slug,
                        'Yes %': yes_price * 100,
                        'Vol': volume,
                        'Score': opportunity_score
                    })
                    
            except Exception as e:
                logger.warning(f"Polymarket parsing error: {e}")
                continue
        
        if not opportunities:
            return None
        
        opportunities_sorted = sorted(opportunities, key=lambda x: x['Score'], reverse=True)
        return pd.DataFrame(opportunities_sorted[:10])
        
    except Exception as e:
        logger.error(f"Polymarket fetch error: {e}")
        return None

# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("⚡ ALPHA DECK PRO v4.0")

# AI BRIEFING
st.subheader("🤖 AI MARKET BRIEFING")

if st.button("⚡ GENERATE MORNING BRIEF", key="ai_brief"):
    with st.spinner('Consulting AI...'):
        indices = fetch_index_data()
        spx_data = fetch_spx_options_data()
        news = fetch_news_feeds()
        
        spx_price = indices.get('SPX', {}).get('price', 0)
        vix_price = indices.get('VIX', {}).get('price', 0)
        pc_ratio = spx_data.get('put_call_ratio', 0) if spx_data and spx_data.get('success') else 0
        headlines = [article['Title'] for article in news]
        
        briefing = generate_ai_briefing(spx_price, vix_price, pc_ratio, headlines)
        st.markdown(f"```\n{briefing}\n```")

st.divider()

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MAIN DECK", "📊 LIQUIDITY & INSIDER", "₿ CRYPTO & POLY", "📈 TRADINGVIEW"])

# TAB 1: MAIN DECK
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
                
                st.metric(label=name, value=value_str, delta=f"{abs_str} ({data['change_pct']:+.2f}%)")
            else:
                st.metric(label=name, value="LOADING")
    
    st.divider()
    
    # VIX TERM STRUCTURE
    st.subheader("📊 VOLATILITY TERM STRUCTURE")
    
    vix_term = fetch_vix_term_structure()
    col_vix1, col_vix2 = st.columns([3, 7])
    
    with col_vix1:
        if vix_term.get('backwardation'):
            st.error("⚠️ BACKWARDATION")
        else:
            st.success("✅ CONTANGO")
    
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
    st.caption(f"Current ET Time: {current_time.strftime('%I:%M %p')} | {current_time.strftime('%A')}")
    
    if not market_is_open:
        st.warning("⏰ Markets Closed - Options available Mon-Fri 9:30 AM - 4:00 PM ET")
    
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
        
        st.caption(f"📅 Exp: {spx_data['expiration']} | Source: {spx_data['ticker']}")
        
        # Charts
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
    
    # WATCHLIST & SECTOR
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚡ WATCHLIST + SCANNER")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        df = fetch_watchlist_data(watchlist_tickers)
        
        if not df.empty:
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
        
        sector_df = fetch_sector_performance()
        
        if not sector_df.empty:
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

# TAB 2: LIQUIDITY & INSIDER
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏛️ FED LIQUIDITY")
        
        liquidity = fetch_fred_liquidity()
        
        if liquidity['success']:
            met_cols = st.columns(3)
            
            with met_cols[0]:
                st.metric("10Y-2Y SPREAD", f"{liquidity['yield_spread']:.2f}%")
            
            with met_cols[1]:
                st.metric("HY CREDIT", f"{liquidity['credit_spread']:.2f}%")
            
            with met_cols[2]:
                st.metric("FED BALANCE", f"${liquidity['fed_balance']:.2f}T")
            
            st.markdown("---")
            
            if liquidity['yield_spread'] < 0:
                st.error("⚠️ INVERTED CURVE")
            else:
                st.success("✅ NORMAL CURVE")
        else:
            st.error("FRED DATA UNAVAILABLE")
    
    with col2:
        st.subheader("🕵️ INSIDER BUYS")
        
        insider_df = fetch_insider_cluster_buys()
        
        if insider_df is not None and not insider_df.empty:
            st.dataframe(insider_df, use_container_width=True, height=400, hide_index=True)
        else:
            st.warning("⚠️ Insider Data Blocked")
    
    st.divider()
    
    st.subheader("📰 NEWS WIRE")
    news = fetch_news_feeds()
    
    if news:
        for article in news:
            st.markdown(f"**[{article['Source']}]** [{article['Title']}]({article['Link']})")
    else:
        st.info("NO NEWS")

# TAB 3: CRYPTO & POLYMARKET
with tab3:
    st.subheader("₿ CRYPTO PULSE")
    
    crypto_data = fetch_crypto_metrics(['BTC', 'ETH', 'SOL', 'DOGE'])
    
    row1 = st.columns(2)
    row2 = st.columns(2)
    
    crypto_order = ['BTC', 'ETH', 'SOL', 'DOGE']
    
    for idx, crypto in enumerate(crypto_order):
        if crypto in crypto_data and crypto_data[crypto]['success']:
            data = crypto_data[crypto]
            
            if idx < 2:
                with row1[idx]:
                    st.metric(label=crypto, value=f"${data['price']:,.2f}", delta=f"{data['change_pct']:+.2f}%")
            else:
                with row2[idx - 2]:
                    st.metric(label=crypto, value=f"${data['price']:,.2f}", delta=f"{data['change_pct']:+.2f}%")
    
    st.divider()
    
    st.subheader("🎲 POLYMARKET")
    
    poly_df = fetch_polymarket_advanced_analytics()
    
    if poly_df is not None and not poly_df.empty:
        poly_df['Vol'] = poly_df['Vol'].apply(lambda x: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
        poly_df['Yes %'] = poly_df['Yes %'].apply(lambda x: f"{x:.1f}%")
        poly_df['Score'] = poly_df['Score'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(poly_df[['Event', 'Yes %', 'Vol', 'Score']], use_container_width=True, height=400, hide_index=True)
        
        st.caption("**Click to trade:**")
        for idx, row in poly_df.iterrows():
            if row['slug']:
                url = f"https://polymarket.com/event/{row['slug']}"
                st.markdown(f"[{row['Event']}]({url})")
    else:
        st.error("POLYMARKET UNAVAILABLE")

# TAB 4: TRADINGVIEW
with tab4:
    st.subheader("📈 TRADINGVIEW CHARTS")
    
    all_tickers = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'BTC-USD', 'ETH-USD']
    
    default_idx = all_tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in all_tickers else 0
    selected_ticker = st.selectbox("SELECT TICKER", all_tickers, index=default_idx)
    
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
        st.caption(f"📊 {tv_symbol}")

st.divider()
st.caption("⚡ ALPHA DECK PRO v4.0 - GOLD MASTER")
