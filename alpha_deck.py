"""
Alpha Deck - Personal Bloomberg Terminal for Options Traders
Optimized for Streamlit Community Cloud
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Force dark theme with custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #0E1117;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        font-size: 14px !important;
    }
    .stMetric .metric-value {
        font-size: 24px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHED DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_ticker_data(ticker):
    """Fetch single ticker data with 60s cache"""
    try:
        data = yf.download(ticker, period='1d', interval='1d', progress=False)
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        prev_close = info.get('previousClose', current_price)
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        
        return {
            'price': current_price,
            'change_pct': change_pct,
            'volume': info.get('volume', 0)
        }
    except:
        return {'price': 0, 'change_pct': 0, 'volume': 0}

@st.cache_data(ttl=60)
def fetch_multiple_tickers(tickers):
    """Fetch multiple tickers efficiently with RSI calculation"""
    try:
        # Download historical data for RSI calculation
        df = yf.download(tickers, period='1mo', progress=False)
        
        results = []
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
                volume = info.get('volume', 0)
                day_low = info.get('dayLow', 0)
                day_high = info.get('dayHigh', 0)
                
                # Calculate RSI
                try:
                    if len(tickers) > 1:
                        closes = df['Close'][ticker].dropna()
                    else:
                        closes = df['Close'].dropna()
                    
                    delta = closes.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_value = rsi.iloc[-1] if len(rsi) > 0 else 50
                except:
                    rsi_value = 50
                
                results.append({
                    'Ticker': ticker,
                    'Price': f"${current_price:.2f}",
                    'Change %': change_pct,
                    'Volume': f"{volume/1e6:.1f}M" if volume > 0 else "0M",
                    'RSI': rsi_value,
                    'Day Range': f"${day_low:.2f} - ${day_high:.2f}"
                })
            except:
                results.append({
                    'Ticker': ticker,
                    'Price': '$0.00',
                    'Change %': 0,
                    'Volume': '0M',
                    'RSI': 50,
                    'Day Range': '$0.00 - $0.00'
                })
        
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

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
        data = fetch_ticker_data(ticker)
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
        data = fetch_ticker_data(ticker)
        results.append({
            'Sector': name,
            'Change %': data['change_pct']
        })
    
    df = pd.DataFrame(results)
    return df.sort_values('Change %', ascending=False)

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
def fetch_crypto_prices():
    """Fetch crypto prices"""
    cryptos = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    results = []
    
    for crypto in cryptos:
        data = fetch_ticker_data(crypto)
        results.append({
            'Crypto': crypto.replace('-USD', ''),
            'Price': f"${data['price']:,.2f}",
            'Change %': data['change_pct']
        })
    
    return pd.DataFrame(results)

@st.cache_data(ttl=300)
def fetch_polymarket_data():
    """Fetch Polymarket top markets"""
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            'limit': 5,
            'active': 'true',
            'closed': 'false',
            'order': 'volume',
            'ascending': 'false'
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        results = []
        for market in data[:5]:
            question = market.get('question', 'N/A')
            volume = market.get('volume', 0)
            
            # Get outcome probabilities
            outcomes = market.get('outcomes', [])
            outcome_text = ' | '.join([f"{o.get('outcome', 'N/A')}: {float(o.get('price', 0))*100:.1f}%" for o in outcomes[:2]])
            
            results.append({
                'Event': question[:60] + '...' if len(question) > 60 else question,
                'Volume': f"${float(volume):,.0f}",
                'Outcomes': outcome_text
            })
        
        return pd.DataFrame(results)
    except:
        return pd.DataFrame({'Event': ['API Error'], 'Volume': ['N/A'], 'Outcomes': ['N/A']})

@st.cache_data(ttl=60)
def fetch_candlestick_data(ticker):
    """Fetch 3-month candlestick data"""
    try:
        data = yf.download(ticker, period='3mo', interval='1d', progress=False)
        return data
    except:
        return pd.DataFrame()

# ============================================================================
# STYLING HELPERS
# ============================================================================

def color_change(val):
    """Color code percentage changes"""
    if val > 0:
        return 'color: #00ff00'
    elif val < 0:
        return 'color: #ff0000'
    else:
        return 'color: #ffffff'

def highlight_rsi(row):
    """Highlight RSI values"""
    colors = [''] * len(row)
    if 'RSI' in row.index:
        idx = row.index.get_loc('RSI')
        if row['RSI'] > 70:
            colors[idx] = 'background-color: #ff4444'
        elif row['RSI'] < 30:
            colors[idx] = 'background-color: #44ff44'
    return colors

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("📊 Alpha Deck - Personal Trading Terminal")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Main Deck", "📰 Macro & Earnings", "₿ Crypto & Poly", "📈 Charts"])

# ============================================================================
# TAB 1: MAIN DECK
# ============================================================================
with tab1:
    st.subheader("📡 Ticker Tape")
    
    # Fetch index data
    indices = fetch_index_data()
    
    # Display top row metrics
    cols = st.columns(6)
    for idx, (name, data) in enumerate(indices.items()):
        with cols[idx]:
            delta_color = "normal" if data['change_pct'] >= 0 else "inverse"
            st.metric(
                label=name,
                value=f"${data['price']:.2f}" if name != 'VIX' else f"{data['price']:.2f}",
                delta=f"{data['change_pct']:.2f}%"
            )
    
    st.divider()
    
    # Main watchlist and sector heat
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("⚡ Options Watchlist")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        df = fetch_multiple_tickers(watchlist_tickers)
        
        if not df.empty:
            # Style the dataframe
            styled_df = df.style.applymap(color_change, subset=['Change %']).apply(highlight_rsi, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=500)
    
    with col2:
        st.subheader("🎨 Sector Heat")
        sector_df = fetch_sector_performance()
        
        if not sector_df.empty:
            fig = px.bar(
                sector_df,
                x='Change %',
                y='Sector',
                orientation='h',
                color='Change %',
                color_continuous_scale=['red', 'yellow', 'green'],
                color_continuous_midpoint=0
            )
            fig.update_layout(
                template='plotly_dark',
                showlegend=False,
                height=500,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: MACRO & EARNINGS
# ============================================================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Earnings Watch (Next 7 Days)")
        watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
        
        earnings_df = fetch_earnings_calendar(watchlist_tickers)
        
        if earnings_df is not None and not earnings_df.empty:
            st.dataframe(earnings_df, use_container_width=True)
        else:
            st.info("✅ No Major Earnings Next 7 Days")
    
    with col2:
        st.subheader("📰 News Wire")
        news = fetch_news_feeds()
        
        for article in news:
            st.markdown(f"**[{article['Source']}]** [{article['Title']}]({article['Link']})")
    
    st.divider()
    st.subheader("🌍 Economic Calendar")
    
    # TradingView Economic Calendar Widget
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
        st.subheader("₿ Crypto Prices")
        crypto_df = fetch_crypto_prices()
        
        if not crypto_df.empty:
            styled_crypto = crypto_df.style.applymap(color_change, subset=['Change %'])
            st.dataframe(styled_crypto, use_container_width=True)
    
    with col2:
        st.subheader("🎲 Polymarket Alpha")
        poly_df = fetch_polymarket_data()
        
        if not poly_df.empty:
            st.dataframe(poly_df, use_container_width=True, height=400)

# ============================================================================
# TAB 4: CHARTS
# ============================================================================
with tab4:
    st.subheader("📈 Candlestick Charts")
    
    watchlist_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'COIN', 'MSTR', 'SPY', 'QQQ', 'IWM']
    
    selected_ticker = st.selectbox("Select Ticker", watchlist_tickers)
    
    if selected_ticker:
        chart_data = fetch_candlestick_data(selected_ticker)
        
        if not chart_data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close']
            )])
            
            fig.update_layout(
                template='plotly_dark',
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to load chart data")

# Footer
st.divider()
st.caption("⚡ Alpha Deck v1.0 | Data refreshes every 60 seconds | Built with Streamlit")
