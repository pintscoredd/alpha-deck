"""
Alpha Deck PRO v4.0 - Bloomberg-Style Trading Terminal
Professional Features: OpenInsider • FRED Liquidity • Gemini AI • Advanced Charts
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
# API KEYS (Hardcoded as requested)
# ============================================================================
GEMINI_API_KEY = "AIzaSyA19pH_uMDXEyiMUnJ5CR9PFP2wRDELrYc"
FRED_API_KEY = "7a3a70ac26c0589b90c81a208d2b99a6"

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

# ============================================================================
# PURE BLACK TERMINAL THEME CSS
# ============================================================================
st.markdown("""
    <style>
    /* Pure OLED Black Background */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* All sections pure black */
    .main, .block-container, section {
        background-color: #000000 !important;
    }
    
    /* Terminal Font for Tables */
    .stDataFrame, .dataframe, table {
        font-family: 'Courier New', Courier, monospace !important;
        background-color: #000000 !important;
        color: #00FF00 !important;
    }
    
    /* Metrics - Pure Black with Neon */
    .stMetric {
        background-color: #000000 !important;
        border: 1px solid #00FF00 !important;
        padding: 15px;
        border-radius: 0px;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    .stMetric label {
        color: #00FF00 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    .stMetric .metric-value {
        color: #00FF00 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Tabs - Terminal Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000000 !important;
        border-bottom: 2px solid #00FF00 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #000000 !important;
        color: #00FF00 !important;
        border: 1px solid #00FF00 !important;
        border-radius: 0px;
        padding: 10px 20px;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00FF00 !important;
        color: #000000 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #00FF00 !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    
    /* Text */
    p, span, div {
        color: #00FF00 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Dividers */
    hr {
        border-color: #00FF00 !important;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000 !important;
        color: #00FF00 !important;
        border: 2px solid #00FF00 !important;
        border-radius: 0px !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
        padding: 10px 30px;
    }
    
    .stButton > button:hover {
        background-color: #00FF00 !important;
        color: #000000 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00FF00 !important;
    }
    
    /* Remove all gray backgrounds */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
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
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        return {
            'expiration': nearest_exp,
            'put_call_ratio': put_call_ratio,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume
        }
    except:
        return None

@st.cache_data(ttl=60)
def fetch_index_data():
    """Fetch major indices - UPDATED: VVIX replaced with HYG"""
    indices = {
        'SPX': '^GSPC',
        'NDX': '^NDX',
        'VIX': '^VIX',
        'HYG': 'HYG',  # High Yield Bond ETF instead of VVIX
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

@st.cache_data(ttl=60)
def fetch_candlestick_data(ticker, period='3mo'):
    """Fetch candlestick data with SMAs"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        # Calculate SMAs
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        return data
    except:
        return None

# ============================================================================
# NEW: OPENINSIDER SCRAPING
# ============================================================================

@st.cache_data(ttl=600)
def fetch_insider_cluster_buys():
    """Scrape OpenInsider for cluster buys"""
    try:
        url = "http://openinsider.com/latest-cluster-buys"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Read HTML tables
        tables = pd.read_html(url, header=0)
        
        if not tables:
            return None
        
        # Get the main table (usually the first one with data)
        df = tables[0]
        
        # Clean and select relevant columns
        if 'Ticker' in df.columns:
            # Select key columns
            columns_to_keep = []
            for col in ['Ticker', 'Company Name', 'Industry', 'Insider Name', 'Title', 
                       'Trade Type', 'Price', 'Qty', 'Value', 'Trade Date']:
                if col in df.columns:
                    columns_to_keep.append(col)
            
            df = df[columns_to_keep].head(10)
            return df
        
        return None
        
    except Exception as e:
        return None

# ============================================================================
# NEW: FRED LIQUIDITY METRICS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_fred_liquidity():
    """Fetch Fed liquidity metrics from FRED"""
    try:
        # Yield Curve (10Y-2Y Spread)
        t10y2y = fred.get_series_latest_release('T10Y2Y')
        yield_spread = float(t10y2y.iloc[-1]) if not t10y2y.empty else 0
        
        # High Yield Credit Spread
        hy_spread = fred.get_series_latest_release('BAMLH0A0HYM2')
        credit_spread = float(hy_spread.iloc[-1]) if not hy_spread.empty else 0
        
        # Fed Balance Sheet
        fed_assets = fred.get_series_latest_release('WALCL')
        fed_balance = float(fed_assets.iloc[-1]) if not fed_assets.empty else 0
        
        return {
            'yield_spread': yield_spread,
            'credit_spread': credit_spread,
            'fed_balance': fed_balance / 1000,  # Convert to Trillions
            'success': True
        }
    except Exception as e:
        return {
            'yield_spread': 0,
            'credit_spread': 0,
            'fed_balance': 0,
            'success': False
        }

# ============================================================================
# NEW: GEMINI AI BRIEFING
# ============================================================================

def generate_ai_briefing(spx_price, vix_price, put_call_ratio, news_headlines):
    """Generate AI morning briefing using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
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
        return f"AI Briefing unavailable: {str(e)}"

# ============================================================================
# FILTERED POLYMARKET DATA
# ============================================================================

@st.cache_data(ttl=180)
def fetch_polymarket_filtered():
    """Fetch Polymarket - Filter out sports/gaming/pop culture"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {'limit': 50, 'active': 'true', 'closed': 'false'}
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            return None
        
        markets = response.json()
        
        # Filter keywords
        filter_keywords = ['nfl', 'nba', 'sport', 'gaming', 'pop culture', 'mlb', 'nhl', 
                          'soccer', 'football', 'basketball', 'celebrity', 'music', 'movie']
        
        filtered_markets = []
        for market in markets:
            question = market.get('question', '').lower()
            
            # Skip if contains filter keywords
            if any(keyword in question for keyword in filter_keywords):
                continue
            
            volume = float(market.get('volume', 0))
            if volume < 100:  # Skip low volume
                continue
            
            outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
            try:
                yes_price = float(outcome_prices[0])
                no_price = float(outcome_prices[1])
            except:
                yes_price = 0.5
                no_price = 0.5
            
            filtered_markets.append({
                'Event': market.get('question', '')[:60],
                'Yes %': yes_price * 100,
                'No %': no_price * 100,
                'Volume': f"${volume:,.0f}"
            })
        
        if not filtered_markets:
            return None
        
        return pd.DataFrame(filtered_markets[:10])
        
    except:
        return None

# ============================================================================
# STYLING HELPERS
# ============================================================================

def style_dataframe(df):
    """Apply neon green/red styling to dataframes"""
    def color_negative_red(val):
        try:
            val = float(val)
            if val > 0:
                return 'color: #00FF00; font-weight: bold'
            elif val < 0:
                return 'color: #FF0000; font-weight: bold'
            else:
                return 'color: #00FF00'
        except:
            return 'color: #00FF00'
    
    if 'Change %' in df.columns:
        return df.style.applymap(color_negative_red, subset=['Change %'])
    return df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("⚡ ALPHA DECK PRO v4.0")
st.caption(">>> TERMINAL MODE: OPENINSIDER | FRED LIQUIDITY | GEMINI AI <<<")

# ============================================================================
# AI BRIEFING SECTION (TOP)
# ============================================================================
st.subheader("🤖 AI MARKET BRIEFING")

if st.button("⚡ GENERATE MORNING BRIEF", key="ai_brief"):
    with st.spinner('Consulting the AI...'):
        # Fetch current data
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
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MAIN DECK", "📊 LIQUIDITY & INSIDER", "₿ CRYPTO & POLY", "📈 CHARTS"])

# ============================================================================
# TAB 1: MAIN DECK
# ============================================================================
with tab1:
    st.subheader("📡 MARKET PULSE")
    
    indices = fetch_index_data()
    
    # Display metrics
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
    
    # Watchlist
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
            # Updated with hover data
            fig = px.bar(
                sector_df,
                x='Change %',
                y='Sector',
                orientation='h',
                color='Change %',
                color_continuous_scale=[[0, '#FF0000'], [0.5, '#000000'], [1, '#00FF00']],
                color_continuous_midpoint=0,
                hover_data={'Change %': ':.2f'}  # Show exact percentage on hover
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
                font=dict(color='#00FF00', family='Courier New'),
                xaxis=dict(showgrid=False, color='#00FF00'),
                yaxis=dict(showgrid=False, color='#00FF00')
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
            
            # Interpretation
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
            st.warning("OPENINSIDER DATA UNAVAILABLE (403/BLOCK)")
            st.caption("Scraping blocked. Try manual check: http://openinsider.com/latest-cluster-buys")
    
    st.divider()
    
    # News
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
        st.subheader("🎲 POLYMARKET (SERIOUS ONLY)")
        st.caption("Filtered: No Sports/Gaming/Pop Culture")
        
        poly_df = fetch_polymarket_filtered()
        
        if poly_df is not None and not poly_df.empty:
            st.dataframe(poly_df, use_container_width=True, height=300)
        else:
            st.warning("POLYMARKET DATA UNAVAILABLE")

# ============================================================================
# TAB 4: ADVANCED CHARTS (FINVIZ STYLE)
# ============================================================================
with tab4:
    st.subheader("📈 FINVIZ-STYLE CHARTS")
    
    all_tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'BTC-USD', 'ETH-USD']
    
    col1, col2 = st.columns([2, 8])
    
    with col1:
        selected_ticker = st.selectbox("TICKER", all_tickers, index=0)
        period = st.selectbox("PERIOD", ['1mo', '3mo', '6mo', '1y'], index=1)
    
    with col2:
        if selected_ticker:
            chart_data = fetch_candlestick_data(selected_ticker, period=period)
            
            if chart_data is not None and not chart_data.empty:
                # Create subplots: Price + Volume
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        increasing_line_color='#00FF00',
                        decreasing_line_color='#FF0000',
                        name='Price'
                    ),
                    row=1, col=1
                )
                
                # SMA 50 (Blue)
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['SMA_50'],
                        line=dict(color='#0000FF', width=1),
                        name='SMA 50'
                    ),
                    row=1, col=1
                )
                
                # SMA 200 (Yellow)
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['SMA_200'],
                        line=dict(color='#FFFF00', width=1),
                        name='SMA 200'
                    ),
                    row=1, col=1
                )
                
                # Volume
                colors = ['#00FF00' if chart_data['Close'].iloc[i] >= chart_data['Open'].iloc[i] 
                         else '#FF0000' for i in range(len(chart_data))]
                
                fig.add_trace(
                    go.Bar(
                        x=chart_data.index,
                        y=chart_data['Volume'],
                        marker_color=colors,
                        name='Volume',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Layout - Pure Black, No Gridlines
                fig.update_layout(
                    title=f'{selected_ticker} | FINVIZ MODE',
                    template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    height=700,
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='#00FF00', family='Courier New', size=10),
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(
                        orientation='h',
                        yanchor='top',
                        y=0.99,
                        xanchor='left',
                        x=0.01,
                        bgcolor='#000000',
                        font=dict(color='#00FF00')
                    )
                )
                
                # Remove all gridlines
                fig.update_xaxes(showgrid=False, color='#00FF00')
                fig.update_yaxes(showgrid=False, color='#00FF00')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"CHART DATA UNAVAILABLE FOR {selected_ticker}")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("⚡ ALPHA DECK PRO v4.0 | POWERED BY: OPENINSIDER • FRED • GEMINI AI | NOT FINANCIAL ADVICE")
