"""
Alpha Deck PRO v4.0 - GOLD MASTER
Bloomberg-Style Trading Terminal - Production Ready
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
import google.generativeai as genai
from fredapi import Fred

# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================
st.set_page_config(
    page_title="Alpha Deck PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for cross-tab ticker selection
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'

# ============================================================================
# SIDEBAR - API CONFIGURATION
# ============================================================================
st.sidebar.title("⚙️ API CONFIGURATION")
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

GEMINI_API_KEY = gemini_key_input or st.secrets.get("GEMINI_API_KEY", "AIzaSyA19pH_uMDXEyiMUnJ5CR9PFP2wRDELrYc")
FRED_API_KEY = fred_key_input or st.secrets.get("FRED_API_KEY", "7a3a70ac26c0589b90c81a208d2b99a6")

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        st.sidebar.success("✅ Gemini Connected")
    else:
        st.sidebar.warning("⚠️ Gemini: No API Key")
except:
    st.sidebar.error("❌ Gemini: Invalid Key")

try:
    if FRED_API_KEY:
        fred = Fred(api_key=FRED_API_KEY)
        st.sidebar.success("✅ FRED Connected")
    else:
        st.sidebar.warning("⚠️ FRED: No API Key")
except:
    st.sidebar.error("❌ FRED: Invalid Key")

# ============================================================================
# REFINED AMBER TERMINAL THEME CSS
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

# ============================================================================
# DATA FETCHING FUNCTIONS (UPGRADED WITH ABSOLUTE CHANGE)
# ============================================================================

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
            if hist.empty:
                continue
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
            volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
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
        except:
            continue
    return pd.DataFrame(results)

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
            'backwardation': vix9d['price'] > vix['price'] if vix9d['success'] and vix['success'] else False
        }
    except:
        return {'VIX': 0, 'VIX9D': 0, 'VIX3M': 0, 'backwardation': False}

@st.cache_data(ttl=60)
def fetch_crypto_metrics(cryptos):
    """Fetch crypto data as metrics"""
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
        if not expirations:
            return None
        nearest_exp = expirations[0]
        opt_chain = spx.option_chain(nearest_exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        calls_oi = calls.groupby('strike')['openInterest'].sum()
        puts_oi = puts.groupby('strike')['openInterest'].sum()
        total_oi = calls_oi.add(puts_oi, fill_value=0)
        max_pain = total_oi.idxmax() if not total_oi.empty else 0
        
        avg_call_iv = calls['impliedVolatility'].mean() * 100
        avg_put_iv = puts['impliedVolatility'].mean() * 100
        
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
            'total_put_volume': total_put_volume
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
                    'Title': entry.title,
                    'Link': entry.link
                })
        except:
            pass
    return articles[:15]

@st.cache_data(ttl=3600)  # OPTIMIZED: 1 hour cache
def fetch_insider_cluster_buys():
    """Scrape OpenInsider - optimized caching"""
    try:
        url = "http://openinsider.com/latest-cluster-buys"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
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
        
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_fred_liquidity():
    """Fetch Fed liquidity metrics"""
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
    except:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }

def generate_ai_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    """Generate AI briefing - FIXED with truncation"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # CRITICAL: Truncate news to top 3 only
        top_headlines = news_headlines[:3]
        
        prompt = f"""Act as a cynical hedge fund manager. Analyze this data in 3 bullets max (keep under 150 words total):

Market: SPX ${spx_price:.0f}, VIX {vix_price:.1f}, P/C {put_call_ratio:.2f}
News: {' | '.join(top_headlines)}

Output format:
1. Risk: [one sentence]
2. Opportunity: [one sentence]
3. Algos: [one sentence]"""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"⚠️ AI Briefing Error: {error_msg[:100]}")
        return f"AI unavailable. Error: {error_msg[:50]}..."

@st.cache_data(ttl=180)
def fetch_polymarket_advanced_analytics():
    """Fetch Polymarket with slug for linking"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            'limit': 100,
            'active': 'true',
            'closed': 'false'
        }
        
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
                
                try:
                    yes_price = float(outcome_prices[0])
                except:
                    yes_price = 0.5
                
                total_prob = yes_price + (1 - yes_price)
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
            except:
                continue
        
        if not opportunities:
            return None
        
        opportunities_sorted = sorted(opportunities, key=lambda x: x['Score'], reverse=True)
        
        return pd.DataFrame(opportunities_sorted[:10])
        
    except:
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("⚡ ALPHA DECK PRO v4.0")

# ============================================================================
# AI BRIEFING
# ============================================================================
st.subheader("🤖 AI MARKET BRIEFING")

if st.button("⚡ GENERATE MORNING BRIEF", key="ai_brief"):
    if not GEMINI_API_KEY:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar")
    else:
        with st.spinner('Consulting AI...'):
            indices = fetch_index_data()
            spx_data = fetch_spx_options_data()
            news = fetch_news_feeds()
            
            spx_price = indices.get('SPX', {}).get('price', 0)
            vix_price = indices.get('VIX', {}).get('price', 0)
            pc_ratio = spx_data.get('put_call_ratio', 0) if spx_data else 0
            headlines = [article['Title'] for article in news]
            
            briefing = generate_ai_briefing(spx_price, vix_price, pc_ratio, headlines)
            
            st.markdown(f"```\n{briefing}\n```")

st.divider()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MAIN DECK", "📊 LIQUIDITY & INSIDER", "₿ CRYPTO & POLY", "📈 TRADINGVIEW"])

# ============================================================================
# TAB 1: MAIN DECK
# ============================================================================
with tab1:
    st.subheader("📡 MARKET PULSE")
    
    indices = fetch_index_data()
    
    # Enhanced metrics with absolute change
    cols = st.columns(6)
    for idx, (name, data) in enumerate(indices.items()):
        with cols[idx]:
            if data['success']:
                # Format absolute change
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
            st.caption("VIX9D > VIX = Near-term fear elevated")
        else:
            st.success("✅ CONTANGO (NORMAL)")
            st.caption("VIX curve in normal upward slope")
    
    with col_vix2:
        # VIX term structure chart
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
                st.success(f"✅ {ticker} selected for charting")
    
    st.divider()
    
    # SPX OPTIONS
    st.subheader("🎯 SPX OPTIONS INTELLIGENCE")
    
    market_is_open = is_market_open()
    
    if not market_is_open:
        st.warning("⏰ Markets Closed - Options data available 9:30 AM - 4:00 PM ET")
    
    spx_data = fetch_spx_options_data()
    
    if spx_data:
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
        st.info("💤 Markets closed. Options data available during trading hours.")
    
    st.divider()
    
    # Watchlist and Sector
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚡ WATCHLIST + TECHNICAL SCANNER")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        df = fetch_watchlist_data(watchlist_tickers)
        
        if not df.empty:
            # Format for clean display
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

# ============================================================================
# TAB 2: LIQUIDITY & INSIDER
# ============================================================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏛️ FED LIQUIDITY METRICS")
        
        if not FRED_API_KEY:
            st.warning("⚠️ Please enter your FRED API Key in the sidebar")
        else:
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
                    st.error("⚠️ INVERTED YIELD CURVE - RECESSION RISK")
                else:
                    st.success("✅ NORMAL YIELD CURVE")
                
                if liquidity['credit_spread'] > 5:
                    st.error("⚠️ ELEVATED CREDIT SPREADS")
                else:
                    st.success("✅ CREDIT MARKETS STABLE")
            else:
                st.error("FRED DATA UNAVAILABLE")
    
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
# TAB 3: CRYPTO & POLYMARKET
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
            
            if idx < 2:
                with row1[idx]:
                    st.metric(
                        label=crypto,
                        value=f"${data['price']:,.2f}",
                        delta=f"{data['change_pct']:+.2f}%"
                    )
            else:
                with row2[idx - 2]:
                    st.metric(
                        label=crypto,
                        value=f"${data['price']:,.2f}",
                        delta=f"{data['change_pct']:+.2f}%"
                    )
    
    st.divider()
    
    # POLYMARKET
    st.subheader("🎲 POLYMARKET ALPHA")
    
    poly_df = fetch_polymarket_advanced_analytics()
    
    if poly_df is not None and not poly_df.empty:
        # Format volume
        poly_df['Vol'] = poly_df['Vol'].apply(lambda x: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
        poly_df['Yes %'] = poly_df['Yes %'].apply(lambda x: f"{x:.1f}%")
        poly_df['Score'] = poly_df['Score'].apply(lambda x: f"{x:.1f}")
        
        # Display with LinkColumn
        st.dataframe(
            poly_df,
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "Event": st.column_config.LinkColumn(
                    "Event",
                    display_text="https://polymarket.com/event/(.*)"
                )
            }
        )
        
        # Create clickable links manually
        st.caption("**Click events to trade on Polymarket:**")
        for idx, row in poly_df.iterrows():
            if row['slug']:
                url = f"https://polymarket.com/event/{row['slug']}"
                st.markdown(f"[{row['Event']}]({url})")
    else:
        st.error("POLYMARKET DATA UNAVAILABLE")

# ============================================================================
# TAB 4: TRADINGVIEW
# ============================================================================
with tab4:
    st.subheader("📈 TRADINGVIEW ADVANCED CHARTS")
    
    all_tickers = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'BTC-USD', 'ETH-USD']
    
    # Use session state for default
    default_idx = all_tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in all_tickers else 0
    
    selected_ticker = st.selectbox("SELECT TICKER", all_tickers, index=default_idx, key="chart_select")
    
    # Update session state
    st.session_state.selected_ticker = selected_ticker
    
    if selected_ticker:
        tv_symbol = get_tradingview_symbol(selected_ticker)
        
        # Enhanced TradingView widget with specific studies
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
        
        st.caption(f"📊 Symbol: {tv_symbol} | Volume + SMA 15 + SMA 30")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("⚡ ALPHA DECK PRO v4.0 - GOLD MASTER | OPENINSIDER • FRED • GEMINI • TRADINGVIEW")
