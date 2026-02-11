"""
Alpha Deck v3.1 - Personal Bloomberg Terminal for Options Traders
Enhanced with Advanced Polymarket Analytics
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

# ============================================================================
# PAGE CONFIG - FORCE DARK MODE
# ============================================================================
st.set_page_config(
    page_title="Alpha Deck",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced dark theme with custom CSS
st.markdown("""
    <style>
    /* Main styling */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metrics styling */
    .stMetric {
        background: linear-gradient(145deg, #1a1d29, #0d0f14);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3139;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #8b92a7 !important;
        letter-spacing: 0.5px;
    }
    
    .stMetric .metric-value {
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d29;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0E1117;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #2d5cff, #1a3dbf);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border-color: #2d3139;
    }
    
    /* Alert boxes */
    .alert-box {
        background: linear-gradient(145deg, #ff4444, #cc0000);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .opportunity-box {
        background: linear-gradient(145deg, #00ff88, #00cc66);
        padding: 15px;
        border-radius: 10px;
        color: black;
        font-weight: 600;
        margin: 10px 0;
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
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within trading hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except:
        return False

# ============================================================================
# CACHED DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data using Ticker.history() method - more reliable"""
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
    except Exception as e:
        return {'price': 0, 'change_pct': 0, 'volume': 0, 'success': False}

@st.cache_data(ttl=60)
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
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
    """Fetch multiple tickers with improved reliability"""
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
            day_low = hist['Low'].iloc[-1] if 'Low' in hist.columns else current_price
            day_high = hist['High'].iloc[-1] if 'High' in hist.columns else current_price
            
            rsi_value = calculate_rsi(hist['Close'])
            
            results.append({
                'Ticker': ticker,
                'Price': f"${float(current_price):.2f}",
                'Change %': float(change_pct),
                'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_value),
                'Day Range': f"${float(day_low):.2f} - ${float(day_high):.2f}"
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

@st.cache_data(ttl=60)
def fetch_spx_options_data():
    """Fetch SPX options data and calculate key metrics"""
    try:
        spx = yf.Ticker("^GSPC")
        
        # Get options expiration dates
        expirations = spx.options
        if not expirations:
            return None
        
        # Get nearest expiration
        nearest_exp = expirations[0]
        opt_chain = spx.option_chain(nearest_exp)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Calculate Put/Call Ratio
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        # Calculate Put/Call Open Interest Ratio
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Get max pain
        calls_oi = calls.groupby('strike')['openInterest'].sum()
        puts_oi = puts.groupby('strike')['openInterest'].sum()
        total_oi = calls_oi.add(puts_oi, fill_value=0)
        max_pain = total_oi.idxmax() if not total_oi.empty else 0
        
        # Calculate average IV
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
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def fetch_index_data():
    """Fetch major indices data"""
    indices = {
        'SPX': '^GSPC',
        'NDX': '^NDX',
        'VIX': '^VIX',
        'VVIX': '^VVIX',
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

@st.cache_data(ttl=300)
def fetch_earnings_calendar(tickers):
    """Check earnings dates for next 7 days"""
    upcoming = []
    today = datetime.now()
    week_ahead = today + timedelta(days=7)
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
                    
                    if pd.notna(earnings_date):
                        if isinstance(earnings_date, str):
                            earnings_date = pd.to_datetime(earnings_date)
                        
                        if today <= earnings_date <= week_ahead:
                            estimate = calendar.loc['Earnings Average'].iloc[0] if 'Earnings Average' in calendar.index else 'N/A'
                            upcoming.append({
                                'Ticker': ticker,
                                'Date': earnings_date.strftime('%Y-%m-%d'),
                                'Estimate': f"${estimate:.2f}" if isinstance(estimate, (int, float)) else 'N/A'
                            })
        except:
            continue
    
    return pd.DataFrame(upcoming) if upcoming else None

@st.cache_data(ttl=60)
def fetch_crypto_data(cryptos):
    """Fetch crypto data in same format as stocks"""
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
            day_low = hist['Low'].iloc[-1] if 'Low' in hist.columns else current_price
            day_high = hist['High'].iloc[-1] if 'High' in hist.columns else current_price
            
            rsi_value = calculate_rsi(hist['Close'])
            
            results.append({
                'Ticker': crypto_symbol,
                'Price': f"${float(current_price):,.2f}",
                'Change %': float(change_pct),
                'Volume': f"{int(volume)/1e6:.1f}M" if volume > 0 else "0M",
                'RSI': float(rsi_value),
                'Day Range': f"${float(day_low):,.2f} - ${float(day_high):,.2f}"
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

@st.cache_data(ttl=180)
def fetch_polymarket_advanced_analytics():
    """
    Fetch Polymarket markets and analyze for:
    - Unusual volume activity
    - Mispriced contracts
    - High liquidity opportunities
    - Insider activity signals
    """
    try:
        # Fetch markets sorted by volume
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
        
        opportunities = []
        
        for market in markets:
            try:
                question = market.get('question', '')
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
                prob_deviation = abs(1.0 - total_prob)  # Should sum to 1.0
                
                # Volume velocity (24h volume as % of total)
                volume_velocity = (volume_24h / volume * 100) if volume > 0 else 0
                
                # Liquidity score
                liquidity_score = liquidity / 1000  # Normalize
                
                # Edge detection (mispricing)
                # If probabilities don't sum to 100%, there's arbitrage
                edge_score = prob_deviation * 100
                
                # Activity score (high recent volume = potential insider info)
                activity_score = volume_velocity if volume_24h > 1000 else 0
                
                # Opportunity score (weighted combination)
                opportunity_score = (
                    (edge_score * 3) +           # Mispricing is most valuable
                    (activity_score * 2) +       # Unusual activity is important
                    (liquidity_score * 1)        # Liquidity enables trading
                )
                
                # Only include markets with meaningful metrics
                if volume > 100 and (edge_score > 0.1 or activity_score > 5 or liquidity > 500):
                    opportunities.append({
                        'question': question[:60] + '...' if len(question) > 60 else question,
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
        
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def fetch_candlestick_data(ticker, period='3mo'):
    """Fetch candlestick data"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        return None

# ============================================================================
# STYLING HELPERS
# ============================================================================

def color_change(val):
    """Color code percentage changes"""
    try:
        val = float(val)
        if val > 0:
            return 'color: #00ff88; font-weight: 600'
        elif val < 0:
            return 'color: #ff4444; font-weight: 600'
        else:
            return 'color: #8b92a7'
    except:
        return 'color: #8b92a7'

def highlight_rsi(row):
    """Highlight RSI values"""
    colors = [''] * len(row)
    if 'RSI' in row.index:
        try:
            idx = row.index.get_loc('RSI')
            rsi_val = float(row['RSI'])
            if rsi_val > 70:
                colors[idx] = 'background-color: #ff4444; font-weight: 700; color: white'
            elif rsi_val < 30:
                colors[idx] = 'background-color: #00ff88; font-weight: 700; color: black'
        except:
            pass
    return colors

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("📊 Alpha Deck - Personal Trading Terminal")
st.caption("🔥 Real-time market intelligence for options traders")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Main Deck", "📰 Macro & Earnings", "₿ Crypto & Poly", "📈 Charts"])

# ============================================================================
# TAB 1: MAIN DECK
# ============================================================================
with tab1:
    st.subheader("📡 Market Pulse")
    
    # Fetch index data
    with st.spinner('Loading market data...'):
        indices = fetch_index_data()
    
    # Display top row metrics
    cols = st.columns(6)
    for idx, (name, data) in enumerate(indices.items()):
        with cols[idx]:
            if data['success']:
                delta_color = "normal" if data['change_pct'] >= 0 else "inverse"
                st.metric(
                    label=name,
                    value=f"${data['price']:.2f}" if name not in ['VIX', 'VVIX'] else f"{data['price']:.2f}",
                    delta=f"{data['change_pct']:.2f}%"
                )
            else:
                st.metric(label=name, value="Loading...")
    
    st.divider()
    
    # SPX OPTIONS ANALYTICS SECTION
    st.subheader("🎯 SPX Options Intelligence")
    
    market_is_open = is_market_open()
    
    if not market_is_open:
        st.warning("⏰ **Markets Closed** - SPX options data only available during trading hours (Mon-Fri 9:30 AM - 4:00 PM ET)")
    
    with st.spinner('Analyzing SPX options flow...'):
        spx_data = fetch_spx_options_data()
    
    if spx_data:
        # Options metrics row
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
                help="Strike with highest total open interest"
            )
        
        with opt_cols[3]:
            st.metric(
                "Avg Call IV",
                f"{spx_data['avg_call_iv']:.1f}%",
                help="Average implied volatility for calls"
            )
        
        with opt_cols[4]:
            st.metric(
                "Avg Put IV",
                f"{spx_data['avg_put_iv']:.1f}%",
                help="Average implied volatility for puts"
            )
        
        st.caption(f"📅 Expiration: {spx_data['expiration']}")
        
        # Volume visualization
        col_vol1, col_vol2 = st.columns(2)
        
        with col_vol1:
            fig_volume = go.Figure(data=[
                go.Bar(name='Calls', x=['Volume'], y=[spx_data['total_call_volume']], marker_color='#00ff88'),
                go.Bar(name='Puts', x=['Volume'], y=[spx_data['total_put_volume']], marker_color='#ff4444')
            ])
            fig_volume.update_layout(
                title="Call vs Put Volume",
                template='plotly_dark',
                height=250,
                showlegend=True,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
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
                marker=dict(size=10, color='#00ff88')
            ))
            fig_iv.add_trace(go.Scatter(
                x=puts_top['strike'],
                y=puts_top['impliedVolatility'] * 100,
                mode='markers',
                name='Puts',
                marker=dict(size=10, color='#ff4444')
            ))
            fig_iv.update_layout(
                title="IV by Strike (Top 10 Volume)",
                template='plotly_dark',
                height=250,
                xaxis_title="Strike",
                yaxis_title="IV %",
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_iv, use_container_width=True)
    else:
        if market_is_open:
            st.info("SPX options data temporarily unavailable - retrying...")
        else:
            st.info("💤 Markets closed. SPX options data will be available during trading hours.")
    
    st.divider()
    
    # Main watchlist and sector heat
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚡ Options Watchlist")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        with st.spinner('Fetching watchlist...'):
            df = fetch_watchlist_data(watchlist_tickers)
        
        if not df.empty:
            styled_df = df.style.applymap(color_change, subset=['Change %']).apply(highlight_rsi, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=500)
        else:
            st.error("Unable to load watchlist data. Please refresh.")
    
    with col2:
        st.subheader("🎨 Sector Heat Map")
        
        with st.spinner('Loading sectors...'):
            sector_df = fetch_sector_performance()
        
        if not sector_df.empty:
            fig = px.bar(
                sector_df,
                x='Change %',
                y='Sector',
                orientation='h',
                color='Change %',
                color_continuous_scale=['#ff4444', '#ffaa00', '#00ff88'],
                color_continuous_midpoint=0,
                text='Change %'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(
                template='plotly_dark',
                showlegend=False,
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#2d3139'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: MACRO & EARNINGS
# ============================================================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Earnings Watch (Next 7 Days)")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR']
        
        with st.spinner('Scanning earnings calendar...'):
            earnings_df = fetch_earnings_calendar(watchlist_tickers)
        
        if earnings_df is not None and not earnings_df.empty:
            st.dataframe(earnings_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No Major Earnings Next 7 Days")
    
    with col2:
        st.subheader("📰 News Wire")
        
        with st.spinner('Fetching latest news...'):
            news = fetch_news_feeds()
        
        if news:
            for article in news:
                st.markdown(f"**{article['Source']}** • [{article['Title']}]({article['Link']})")
        else:
            st.info("No news available")
    
    st.divider()
    st.subheader("🌍 Economic Calendar")
    
    tradingview_widget = """
    <div class="tradingview-widget-container" style="height:100%;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
      {
      "colorTheme": "dark",
      "isTransparent": false,
      "width": "100%",
      "height": "500",
      "locale": "en",
      "importanceFilter": "0,1",
      "countryFilter": "us"
      }
      </script>
    </div>
    """
    st.components.v1.html(tradingview_widget, height=500)

# ============================================================================
# TAB 3: CRYPTO & POLYMARKET
# ============================================================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("₿ Crypto Watchlist")
        
        with st.spinner('Loading crypto data...'):
            crypto_df = fetch_crypto_data(['BTC', 'ETH', 'SOL', 'DOGE'])
        
        if not crypto_df.empty:
            styled_crypto = crypto_df.style.applymap(color_change, subset=['Change %']).apply(highlight_rsi, axis=1)
            st.dataframe(styled_crypto, use_container_width=True, height=300)
        else:
            st.error("Unable to load crypto data")
    
    with col2:
        st.subheader("🎲 Market Sentiment Overview")
        
        # Simple overview chart - this will be expanded below
        st.info("📊 See detailed Polymarket analytics below ⬇️")
    
    st.divider()
    
    # ADVANCED POLYMARKET ANALYTICS SECTION
    st.subheader("🔥 Polymarket Alpha: Top 10 Trading Opportunities")
    st.caption("Analyzing markets for mispriced contracts, unusual activity, and insider signals")
    
    with st.spinner('Analyzing Polymarket markets...'):
        poly_opportunities = fetch_polymarket_advanced_analytics()
    
    if poly_opportunities:
        # Create detailed table
        display_data = []
        for opp in poly_opportunities:
            display_data.append({
                'Event': opp['question'],
                'Yes %': f"{opp['yes_price']:.1f}%",
                'No %': f"{opp['no_price']:.1f}%",
                'Volume': f"${opp['volume']:,.0f}",
                '24h Vol': f"${opp['volume_24h']:,.0f}",
                'Liquidity': f"${opp['liquidity']:,.0f}",
                'Edge': f"{opp['edge_score']:.2f}%",
                'Activity': f"{opp['activity_score']:.1f}",
                'Score': f"{opp['opportunity_score']:.1f}"
            })
        
        df_poly = pd.DataFrame(display_data)
        
        # Display table
        st.dataframe(df_poly, use_container_width=True, height=400)
        
        st.caption("""
        **Metrics Explained:**  
        • **Edge**: Probability mispricing (higher = arbitrage opportunity)  
        • **Activity**: 24h volume velocity (higher = unusual activity/potential insider info)  
        • **Score**: Overall opportunity rating (edge × 3 + activity × 2 + liquidity)
        """)
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Top opportunities by score
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
                title="Top 5 Opportunities (by Score)",
                template='plotly_dark',
                height=350,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#2d3139', title="Opportunity Score"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_opp, use_container_width=True)
        
        with col_viz2:
            # Activity vs Edge scatter
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
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#2d3139', title="Edge (Mispricing %)"),
                yaxis=dict(showgrid=True, gridcolor='#2d3139', title="Activity Score")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Highlight best opportunities
        st.subheader("🎯 Best Opportunities Right Now")
        
        best_edge = max(poly_opportunities, key=lambda x: x['edge_score'])
        best_activity = max(poly_opportunities, key=lambda x: x['activity_score'])
        best_overall = poly_opportunities[0]  # Already sorted by score
        
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            st.markdown("**🔸 Best Mispricing**")
            st.info(f"**{best_edge['question']}**  \nEdge: {best_edge['edge_score']:.2f}%  \nYes: {best_edge['yes_price']:.1f}% | No: {best_edge['no_price']:.1f}%")
        
        with col_b2:
            st.markdown("**🔸 Highest Activity**")
            st.warning(f"**{best_activity['question']}**  \nActivity: {best_activity['activity_score']:.1f}  \n24h Vol: ${best_activity['volume_24h']:,.0f}")
        
        with col_b3:
            st.markdown("**🔸 Top Overall**")
            st.success(f"**{best_overall['question']}**  \nScore: {best_overall['opportunity_score']:.1f}  \nVol: ${best_overall['volume']:,.0f}")
        
    else:
        st.error("Unable to load Polymarket data. API may be temporarily unavailable.")

# ============================================================================
# TAB 4: CHARTS
# ============================================================================
with tab4:
    st.subheader("📈 Advanced Charting")
    
    all_tickers = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'BTC-USD', 'ETH-USD']
    
    col1, col2 = st.columns([3, 7])
    
    with col1:
        selected_ticker = st.selectbox("Select Asset", all_tickers, index=0)
        period = st.selectbox("Time Period", ['1mo', '3mo', '6mo', '1y', 'ytd'], index=1)
    
    with col2:
        if selected_ticker:
            with st.spinner(f'Loading {selected_ticker} chart...'):
                chart_data = fetch_candlestick_data(selected_ticker, period=period)
            
            if chart_data is not None and not chart_data.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                )])
                
                fig.update_layout(
                    title=f'{selected_ticker} Price Action',
                    template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    height=600,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#2d3139'),
                    margin=dict(l=0, r=0, t=40, b=0),
                    font=dict(color='white', size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Unable to load chart data for {selected_ticker}")

# Footer
st.divider()
st.caption("⚡ Alpha Deck v3.1 | SPX Options + Polymarket Analytics | Live Market Data | Not Financial Advice")
