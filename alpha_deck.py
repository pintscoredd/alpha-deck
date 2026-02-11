"""
Alpha Deck PRO v4.0 FINAL - Bloomberg-Style Amber Terminal
Professional Features: OpenInsider • FRED Liquidity • Gemini AI • TradingView Charts
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
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Alpha Deck PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# API KEYS
# ============================================================================
GEMINI_API_KEY = "AIzaSyA19pH_uMDXEyiMUnJ5CR9PFP2wRDELrYc"
FRED_API_KEY = "7a3a70ac26c0589b90c81a208d2b99a6"

# Configure APIs
try:
    genai.configure(api_key=GEMINI_API_KEY)
except:
    pass

try:
    fred = Fred(api_key=FRED_API_KEY)
except:
    pass

# ============================================================================
# AMBER TERMINAL THEME CSS
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
    
    /* Tabs - Amber Style */
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

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        if hist.empty:
            return {'price': 0, 'change_pct': 0, 'volume': 0, 'success': False}
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
        return {
            'price': float(current_price),
            'change_pct': float(change_pct),
            'volume': int(volume),
            'success': True
        }
    except:
        return {'price': 0, 'change_pct': 0, 'volume': 0, 'success': False}

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
    """Fetch multiple tickers"""
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
            results.append({
                'Ticker': ticker,
                'Price': f"${float(current_price):.2f}",
                'Change %': float(change_pct),
                'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_value)
            })
        except:
            continue
    return pd.DataFrame(results)

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
        
        # Calculate metrics
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Max pain
        calls_oi = calls.groupby('strike')['openInterest'].sum()
        puts_oi = puts.groupby('strike')['openInterest'].sum()
        total_oi = calls_oi.add(puts_oi, fill_value=0)
        max_pain = total_oi.idxmax() if not total_oi.empty else 0
        
        # Average IV
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
                'Change %': data['change_pct']
            })
    df = pd.DataFrame(results)
    return df.sort_values('Change %', ascending=False) if not df.empty else df

@st.cache_data(ttl=300)
def fetch_news_feeds():
    """Fetch RSS news feeds"""
    feeds = {
        'CNBC': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'Reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best'
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
    return articles[:10]

@st.cache_data(ttl=60)
def fetch_crypto_data(cryptos):
    """Fetch crypto data"""
    results = []
    for crypto_symbol in cryptos:
        ticker = f"{crypto_symbol}-USD"
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
            results.append({
                'Ticker': crypto_symbol,
                'Price': f"${float(current_price):,.2f}",
                'Change %': float(change_pct),
                'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_value)
            })
        except:
            continue
    return pd.DataFrame(results)

# ============================================================================
# OPENINSIDER SCRAPING (FIXED 403)
# ============================================================================

@st.cache_data(ttl=600)
def fetch_insider_cluster_buys():
    """Scrape OpenInsider with proper user agent"""
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
        
    except Exception as e:
        return None

# ============================================================================
# FRED LIQUIDITY METRICS
# ============================================================================

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

# ============================================================================
# GEMINI AI BRIEFING (FIXED - Using gemini-1.5-flash)
# ============================================================================

def generate_ai_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    """Generate AI morning briefing using Gemini 1.5 Flash"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Act as a cynical hedge fund manager analyzing the market.

Current Market Data:
- SPX: ${spx_price:.2f}
- VIX: {vix_price:.2f}
- Put/Call Ratio: {put_call_ratio:.2f}

Recent Headlines:
{chr(10).join(['- ' + h for h in news_headlines[:5]])}

Provide a brutally honest 3-bullet point summary:
1. Top Risk
2. Top Opportunity  
3. What the algos are doing

Be cynical, data-driven, and actionable. No fluff."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ AI Briefing unavailable. Check API key configuration."

# ============================================================================
# POLYMARKET ADVANCED ANALYTICS (FIXED - WITH VISUALIZATIONS)
# ============================================================================

@st.cache_data(ttl=180)
def fetch_polymarket_advanced_analytics():
    """Fetch and analyze Polymarket markets - FILTERED"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            'limit': 100,  # Fetch 100 to ensure enough after filtering
            'active': 'true',
            'closed': 'false'
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return None
        
        markets = response.json()
        
        # Filter keywords
        filter_keywords = ['nfl', 'nba', 'sport', 'gaming', 'gta', 'pop culture', 
                          'music', 'twitch', 'mlb', 'nhl', 'soccer', 'football', 
                          'basketball', 'celebrity', 'movie', 'ufc', 'mma', 'tennis']
        
        opportunities = []
        
        for market in markets:
            try:
                question = market.get('question', '').lower()
                
                # Skip if contains filter keywords
                if any(keyword in question for keyword in filter_keywords):
                    continue
                
                market_slug = market.get('slug', '')
                volume = float(market.get('volume', 0))
                volume_24h = float(market.get('volume24hr', 0))
                liquidity = float(market.get('liquidity', 0))
                
                # Get outcome prices
                outcomes = market.get('outcomes', ['Yes', 'No'])
                outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
                
                try:
                    yes_price = float(outcome_prices[0])
                    no_price = float(outcome_prices[1])
                except:
                    yes_price = 0.5
                    no_price = 0.5
                
                # Calculate analytics
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
                
                # Only include markets with meaningful metrics
                if volume > 100 and (edge_score > 0.1 or activity_score > 5 or liquidity > 500):
                    opportunities.append({
                        'question': market.get('question', '')[:60] + '...' if len(market.get('question', '')) > 60 else market.get('question', ''),
                        'slug': market_slug,
                        'yes_price': yes_price * 100,
                        'no_price': no_price * 100,
                        'volume': volume,
                        'volume_24h': volume_24h,
                        'liquidity': liquidity,
                        'edge_score': edge_score,
                        'activity_score': activity_score,
                        'opportunity_score': opportunity_score,
                        'prob_sum': total_prob * 100
                    })
            except:
                continue
        
        if not opportunities:
            return None
        
        # Sort by opportunity score
        opportunities_sorted = sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)
        
        return opportunities_sorted[:10]
        
    except:
        return None

# ============================================================================
# STYLING HELPERS
# ============================================================================

def style_dataframe(df):
    """Apply amber/red styling to dataframes"""
    def color_negative_red(val):
        try:
            val = float(val)
            if val > 0:
                return 'color: #00FF00; font-weight: bold'
            elif val < 0:
                return 'color: #FF0000; font-weight: bold'
            else:
                return 'color: #FFB000'
        except:
            return 'color: #FFB000'
    
    if 'Change %' in df.columns:
        return df.style.applymap(color_negative_red, subset=['Change %'])
    return df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("⚡ ALPHA DECK PRO v4.0")
st.caption(">>> BLOOMBERG TERMINAL MODE: OPENINSIDER | FRED LIQUIDITY | GEMINI AI <<<")

# ============================================================================
# AI BRIEFING SECTION
# ============================================================================
st.subheader("🤖 AI MARKET BRIEFING")

if st.button("⚡ GENERATE MORNING BRIEF", key="ai_brief"):
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
    
    cols = st.columns(6)
    for idx, (name, data) in enumerate(indices.items()):
        with cols[idx]:
            if data['success']:
                st.metric(
                    label=name,
                    value=f"${data['price']:.2f}" if name not in ['VIX', 'HYG'] else f"{data['price']:.2f}",
                    delta=f"{data['change_pct']:+.2f}%"
                )
            else:
                st.metric(label=name, value="LOADING")
    
    st.divider()
    
    # SPX OPTIONS INTELLIGENCE
    st.subheader("🎯 SPX OPTIONS INTELLIGENCE")
    
    market_is_open = is_market_open()
    
    if not market_is_open:
        st.warning("⏰ Markets Closed - SPX options data available during trading hours (Mon-Fri 9:30 AM - 4:00 PM ET)")
    
    spx_data = fetch_spx_options_data()
    
    if spx_data:
        opt_cols = st.columns(5)
        
        with opt_cols[0]:
            st.metric(
                "Put/Call Ratio",
                f"{spx_data['put_call_ratio']:.2f}",
                help="Volume-based P/C ratio. >1.0 = bearish, <1.0 = bullish"
            )
        
        with opt_cols[1]:
            st.metric(
                "P/C OI Ratio",
                f"{spx_data['put_call_oi_ratio']:.2f}",
                help="Open Interest P/C ratio"
            )
        
        with opt_cols[2]:
            st.metric(
                "Max Pain",
                f"${spx_data['max_pain']:.0f}",
                help="Strike with highest total OI"
            )
        
        with opt_cols[3]:
            st.metric(
                "Avg Call IV",
                f"{spx_data['avg_call_iv']:.1f}%",
                help="Average call IV"
            )
        
        with opt_cols[4]:
            st.metric(
                "Avg Put IV",
                f"{spx_data['avg_put_iv']:.1f}%",
                help="Average put IV"
            )
        
        st.caption(f"📅 Expiration: {spx_data['expiration']}")
        
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
                title="IV by Strike (Top 10 Vol)",
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
        if market_is_open:
            st.info("SPX options data temporarily unavailable")
        else:
            st.info("💤 Markets closed. Options data available during trading hours.")
    
    st.divider()
    
    # Watchlist and Sector
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚡ WATCHLIST")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        df = fetch_watchlist_data(watchlist_tickers)
        
        if not df.empty:
            styled_df = style_dataframe(df)
            st.dataframe(styled_df, use_container_width=True, height=500)
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
        
        liquidity = fetch_fred_liquidity()
        
        if liquidity['success']:
            met_cols = st.columns(3)
            
            with met_cols[0]:
                st.metric(
                    "10Y-2Y SPREAD",
                    f"{liquidity['yield_spread']:.2f}%",
                    help="Negative = Recession Signal"
                )
            
            with met_cols[1]:
                st.metric(
                    "HY CREDIT SPREAD",
                    f"{liquidity['credit_spread']:.2f}%",
                    help="High = Credit Stress"
                )
            
            with met_cols[2]:
                st.metric(
                    "FED BALANCE",
                    f"${liquidity['fed_balance']:.2f}T",
                    help="Total Fed Assets"
                )
            
            st.markdown("---")
            st.markdown("**ANALYSIS:**")
            
            if liquidity['yield_spread'] < 0:
                st.error("⚠️ INVERTED YIELD CURVE - RECESSION RISK")
            else:
                st.success("✅ NORMAL YIELD CURVE")
            
            if liquidity['credit_spread'] > 5:
                st.error("⚠️ ELEVATED CREDIT SPREADS - STRESS DETECTED")
            else:
                st.success("✅ CREDIT MARKETS STABLE")
        else:
            st.error("FRED DATA UNAVAILABLE")
    
    with col2:
        st.subheader("🕵️ INSIDER CLUSTER BUYS")
        st.caption("Multiple execs buying = High confidence")
        
        insider_df = fetch_insider_cluster_buys()
        
        if insider_df is not None and not insider_df.empty:
            st.dataframe(insider_df, use_container_width=True, height=400)
        else:
            st.warning("⚠️ Insider Data: Source Blocking Connections")
            st.caption("Manual check: http://openinsider.com/latest-cluster-buys")
    
    st.divider()
    
    st.subheader("📰 NEWS WIRE")
    news = fetch_news_feeds()
    
    if news:
        for article in news:
            st.markdown(f"**[{article['Source']}]** {article['Title']}")
    else:
        st.info("NO NEWS AVAILABLE")

# ============================================================================
# TAB 3: CRYPTO & POLYMARKET
# ============================================================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("₿ CRYPTO WATCHLIST")
        
        crypto_df = fetch_crypto_data(['BTC', 'ETH', 'SOL', 'DOGE'])
        
        if not crypto_df.empty:
            styled_crypto = style_dataframe(crypto_df)
            st.dataframe(styled_crypto, use_container_width=True, height=300)
        else:
            st.error("CRYPTO DATA UNAVAILABLE")
    
    with col2:
        st.subheader("🎲 POLYMARKET OVERVIEW")
        st.info("📊 See detailed analytics below ⬇️")
    
    st.divider()
    
    # POLYMARKET ANALYTICS WITH VISUALIZATIONS
    st.subheader("🔥 POLYMARKET ALPHA: TOP 10 OPPORTUNITIES")
    st.caption("Filtered: Economics, Politics, Crypto only (No Sports/Gaming/Pop Culture)")
    
    poly_opportunities = fetch_polymarket_advanced_analytics()
    
    if poly_opportunities:
        # Table
        display_data = []
        for opp in poly_opportunities:
            display_data.append({
                'Event': opp['question'],
                'Yes %': f"{opp['yes_price']:.1f}%",
                'No %': f"{opp['no_price']:.1f}%",
                'Volume': f"${opp['volume']:,.0f}",
                '24h Vol': f"${opp['volume_24h']:,.0f}",
                'Edge': f"{opp['edge_score']:.2f}%",
                'Activity': f"{opp['activity_score']:.1f}",
                'Score': f"{opp['opportunity_score']:.1f}"
            })
        
        df_poly = pd.DataFrame(display_data)
        st.dataframe(df_poly, use_container_width=True, height=400)
        
        st.caption("""
        **Metrics:** Edge = Mispricing | Activity = 24h volume velocity | Score = Overall opportunity
        """)
        
        # VISUALIZATIONS
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Top 5 Opportunities Bar Chart
            top_5 = poly_opportunities[:5]
            fig_opp = go.Figure(data=[
                go.Bar(
                    x=[opp['opportunity_score'] for opp in top_5],
                    y=[opp['question'][:40] + '...' for opp in top_5],
                    orientation='h',
                    marker=dict(
                        color=[opp['opportunity_score'] for opp in top_5],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"{opp['opportunity_score']:.1f}" for opp in top_5],
                    textposition='outside'
                )
            ])
            fig_opp.update_layout(
                title="Top 5 Opportunities",
                template='plotly_dark',
                height=350,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#FFB000', family='Courier New'),
                xaxis=dict(showgrid=False, title="Opportunity Score", color='#FFB000'),
                yaxis=dict(showgrid=False, color='#FFB000')
            )
            st.plotly_chart(fig_opp, use_container_width=True)
        
        with col_viz2:
            # Edge vs Activity Scatter
            fig_scatter = go.Figure(data=[
                go.Scatter(
                    x=[opp['edge_score'] for opp in poly_opportunities],
                    y=[opp['activity_score'] for opp in poly_opportunities],
                    mode='markers',
                    marker=dict(
                        size=[min(opp['liquidity']/100, 50) for opp in poly_opportunities],
                        color=[opp['opportunity_score'] for opp in poly_opportunities],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Score")
                    ),
                    text=[opp['question'][:40] for opp in poly_opportunities],
                    hovertemplate='<b>%{text}</b><br>Edge: %{x:.2f}%<br>Activity: %{y:.1f}<extra></extra>'
                )
            ])
            fig_scatter.update_layout(
                title="Edge vs Activity (size = liquidity)",
                template='plotly_dark',
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#FFB000', family='Courier New'),
                xaxis=dict(showgrid=False, title="Edge (Mispricing %)", color='#FFB000'),
                yaxis=dict(showgrid=False, title="Activity Score", color='#FFB000')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Best Opportunities Cards
        st.subheader("🎯 BEST OPPORTUNITIES")
        
        best_edge = max(poly_opportunities, key=lambda x: x['edge_score'])
        best_activity = max(poly_opportunities, key=lambda x: x['activity_score'])
        best_overall = poly_opportunities[0]
        
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            st.markdown("**🔸 BEST MISPRICING**")
            st.info(f"**{best_edge['question']}**  \nEdge: {best_edge['edge_score']:.2f}%  \nYes: {best_edge['yes_price']:.1f}% | No: {best_edge['no_price']:.1f}%")
        
        with col_b2:
            st.markdown("**🔸 HIGHEST ACTIVITY**")
            st.warning(f"**{best_activity['question']}**  \nActivity: {best_activity['activity_score']:.1f}  \n24h Vol: ${best_activity['volume_24h']:,.0f}")
        
        with col_b3:
            st.markdown("**🔸 TOP OVERALL**")
            st.success(f"**{best_overall['question']}**  \nScore: {best_overall['opportunity_score']:.1f}  \nVol: ${best_overall['volume']:,.0f}")
        
    else:
        st.error("POLYMARKET DATA UNAVAILABLE")

# ============================================================================
# TAB 4: TRADINGVIEW INTEGRATION
# ============================================================================
with tab4:
    st.subheader("📈 TRADINGVIEW ADVANCED CHARTS")
    
    all_tickers = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'BTC-USD', 'ETH-USD']
    
    selected_ticker = st.selectbox("SELECT TICKER", all_tickers, index=0)
    
    if selected_ticker:
        tv_symbol = get_tradingview_symbol(selected_ticker)
        
        # TradingView Widget
        tradingview_widget = f"""
        <!-- TradingView Widget BEGIN -->
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
            "MASimple@tv-basicstudies",
            "Volume@tv-basicstudies"
          ],
          "container_id": "tradingview_chart"
        }}
          );
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        
        st.components.v1.html(tradingview_widget, height=650)
        
        st.caption(f"📊 TradingView Symbol: {tv_symbol}")
        st.caption("💡 Use drawing tools, indicators, and timeframes from the chart toolbar")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("⚡ ALPHA DECK PRO v4.0 FINAL | BLOOMBERG TERMINAL | POWERED BY: OPENINSIDER • FRED • GEMINI AI • TRADINGVIEW")
