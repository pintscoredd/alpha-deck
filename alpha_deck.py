"""
Alpha Deck v3.0 - Personal Bloomberg Terminal for Options Traders
Enhanced with SPX Options Analytics
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
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(145deg, #1a1d29, #0d0f14);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2d5cff;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHED DATA FUNCTIONS - IMPROVED WITH BETTER ERROR HANDLING
# ============================================================================

@st.cache_data(ttl=60)
def fetch_ticker_data_reliable(ticker):
    """Fetch single ticker data using Ticker.history() method - more reliable"""
    try:
        stock = yf.Ticker(ticker)
        # Use history method instead of download
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
        st.error(f"Error fetching {ticker}: {str(e)}")
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
            
            # Calculate RSI
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
        
        # Get max pain (strike with most open interest)
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
        st.error(f"SPX options data unavailable: {str(e)}")
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
            
            # Calculate RSI
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

@st.cache_data(ttl=300)
def fetch_polymarket_data():
    """Fetch Polymarket trending markets"""
    try:
        url = "https://clob.polymarket.com/markets"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return create_sample_polymarket_data()
        
        data = response.json()
        
        results = []
        for market in data[:8]:
            question = market.get('question', 'N/A')
            outcomes = market.get('outcomes', ['Yes', 'No'])
            outcome_prices = market.get('outcomePrices', ['0.5', '0.5'])
            
            try:
                yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0.5
                no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
            except:
                yes_price = 0.5
                no_price = 0.5
            
            results.append({
                'Event': question[:50] + '...' if len(question) > 50 else question,
                'Yes': yes_price * 100,
                'No': no_price * 100
            })
        
        if not results:
            return create_sample_polymarket_data()
            
        return pd.DataFrame(results)
        
    except Exception as e:
        return create_sample_polymarket_data()

def create_sample_polymarket_data():
    """Create sample data when API is unavailable"""
    return pd.DataFrame({
        'Event': [
            'Presidential Election 2024',
            'Fed Rate Cut March 2025',
            'Bitcoin above $100k EOY',
            'Recession in 2025',
            'AI Breakthrough 2025',
            'Ukraine Conflict End',
            'Market Correction',
            'Tech M&A Deal'
        ],
        'Yes': [52.3, 78.5, 45.2, 34.8, 61.9, 28.3, 42.1, 55.7],
        'No': [47.7, 21.5, 54.8, 65.2, 38.1, 71.7, 57.9, 44.3]
    })

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
            # Options Volume Distribution
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
            # IV Skew
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
        st.info("SPX options data temporarily unavailable")
    
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
        st.subheader("🎲 Polymarket Prediction Markets")
        
        with st.spinner('Loading prediction markets...'):
            poly_df = fetch_polymarket_data()
        
        if not poly_df.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Yes',
                y=poly_df['Event'],
                x=poly_df['Yes'],
                orientation='h',
                marker=dict(color='#00ff88'),
                text=poly_df['Yes'].apply(lambda x: f'{x:.1f}%'),
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Yes: %{x:.1f}%<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='No',
                y=poly_df['Event'],
                x=poly_df['No'],
                orientation='h',
                marker=dict(color='#ff4444'),
                text=poly_df['No'].apply(lambda x: f'{x:.1f}%'),
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>No: %{x:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                barmode='stack',
                template='plotly_dark',
                height=400,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#2d3139', range=[0, 100]),
                yaxis=dict(showgrid=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)

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
st.caption("⚡ Alpha Deck v3.0 | SPX Options Analytics | Live Market Data | Not Financial Advice")
