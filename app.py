import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="AI 트레이딩 참모 v2.1", page_icon="🤖", layout="wide")

# 1. 지표 계산기 (pandas-ta 없이 직접 계산)
def add_indicators(df):
    # RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD 계산
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD'] - df['Signal']
    
    # ATR 계산
    high_low = df['High'] - df['Low']
    high_cp = abs(df['High'] - df['Close'].shift())
    low_cp = abs(df['Low'] - df['Close'].shift())
    df['ATRr_14'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
    
    # River 엔진 (추세)
    df['River'] = df['MACDh_12_26_9'].ewm(span=5, adjust=False).mean()
    df['River_Slope'] = df['River'].diff()
    df['River_Accel'] = df['River_Slope'].diff()
    return df

@st.cache_data(ttl=60)
def get_analysis(tf):
    exchange = ccxt.kraken()
    raw = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=500)
    df = pd.DataFrame(raw, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    df = add_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    features = ['ATRr_14', 'MACDh_12_26_9', 'RSI_14', 'Volume', 'River', 'River_Slope', 'River_Accel']
    df_clean = df.dropna()
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(df_clean[features], df_clean['Target'])
    
    live_bar = df.iloc[-1]
    prob = model.predict_proba(pd.DataFrame([live_bar[features]]))[0]
    
    return live_bar['Close'], live_bar['Date'], prob[1]*100, prob[0]*100, live_bar

# UI 구성
st.title("🤖 24시간 실시간 AI 참모")
st.write("---")

col1, col2, col3 = st.columns(3)
tf_map = {"1시간": "1h", "4시간": "4h", "1일": "1d"}
selected_tf = None

with col1:
    if st.button("1시간 분석"): selected_tf = "1시간"
with col2:
    if st.button("4시간 분석"): selected_tf = "4시간"
with col3:
    if st.button("1일 분석"): selected_tf = "1일"

if selected_tf:
    with st.spinner('분석 중...'):
        price, time, up, down, bar = get_analysis(tf_map[selected_tf])
        st.success(f"### {selected_tf} 분석 결과 ({time.strftime('%H:%M')})")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"{price:,.1f} USDT")
        m2.metric("상승 확률", f"{up:.1f}%")
        m3.metric("하락 확률", f"{down:.1f}%")
        
        if up >= 60: st.success("🔥 **강력 매수 신호**")
        elif down >= 60: st.error("❄️ **강력 매도 신호**")
        else: st.warning("⚠️ **관망 추천**")
