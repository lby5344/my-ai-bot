import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# [수정] 이지패널 Environment 탭에 넣은 키를 가져옵니다. 
# 만약 설정 안 하셨다면 'st.secrets' 대신 'os.getenv'를 씁니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI 참모 v3.0 (LSTM 딥러닝 탑재)", page_icon="🧠", layout="wide")

# 폰트 및 스타일 설정
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], p, h1, h2, h3, h4, div { font-family: 'Nanum Gothic', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; color: #ff00ff; }
</style>
""", unsafe_allow_html=True)

# 🧠 [신규] 딥러닝 모델 불러오기
@st.cache_resource
def load_ai_brain():
    model_path = 'ai_trader_lstm.h5' # 깃허브에 이 파일이 있어야 합니다!
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"모델 로딩 실패: {e}")
            return None
    return None

def get_safe_ai_briefing(df, up, down):
    if not GEMINI_API_KEY:
        return "🤖 API 키가 설정되지 않았습니다. 이지패널 Environment 설정을 확인하세요."
    
    try:
        latest = df.iloc[-1]
        prompt = f"""
        당신은 암호화폐 전문 분석가 'AI 참모'입니다.
        - 현재 비트코인 가격: ${latest['Close']:,.1f}
        - 딥러닝(LSTM) 예측: 상승 확률 {up:.1f}%, 하락 확률 {down:.1f}%
        - RSI_DK: {latest['RSI_DK']:.1f}
        위 데이터를 바탕으로 현재 시장 상황과 투자 전략을 딱 3줄로 명확하게 브리핑하세요.
        """
        
        # 최신 모델인 gemini-1.5-flash를 기본으로 사용합니다.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        response = requests.post(url, headers=headers, json=data).json()
        
        if 'candidates' in response:
            return response['candidates'][0]['content']['parts'][0]['text']
        else:
            return "🤖 구글 API가 응답하지 않습니다. (키가 만료되었거나 정지되었을 수 있음)"
            
    except Exception as e:
        return f"🤖 브리핑 오류: {str(e)}"

# 보조지표 계산 함수 (마누라님 전용 로직)
def add_indicators_v6(df):
    rsiLen = 14
    df['lo'] = df['Low'].rolling(window=rsiLen).min()
    df['hi'] = df['High'].rolling(window=rsiLen).max()
    denom = df['hi'] - df['lo']
    df['RSI_DK'] = np.where(denom == 0, 50, (df['Close'] - df['lo']) / denom * 100)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['River_Smoothed'] = (ema12 - ema26).ewm(span=5, adjust=False).mean()
    df['River_Slope'] = df['River_Smoothed'].diff()
    df['River_Accel'] = df['River_Slope'].diff()
    
    fortress = np.full(len(df), np.nan)
    curr = np.nan
    for i in range(1, len(df)):
        if df['River_Slope'].iloc[i] < 0 and df['River_Accel'].iloc[i] > (abs(df['River_Slope'].iloc[i]) * 0.1):
            curr = df['Low'].iloc[i]
        elif df['River_Slope'].iloc[i] > 0 and df['River_Accel'].iloc[i] < -(abs(df['River_Slope'].iloc[i]) * 0.1):
            curr = df['High'].iloc[i]
        fortress[i] = curr
    df['Fortress_Price'] = fortress
    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    ex = ccxt.kraken() 
    df = pd.DataFrame(ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    df = add_indicators_v6(df)
    df_c = df.dropna().copy()
    
    lstm_model = load_ai_brain()
    if lstm_model is not None:
        features = ['Close', 'RSI_DK', 'River_Slope', 'River_Accel']
        # 모델 학습시 사용했던 것과 동일한 스케일링 필요
        data = df_c[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        time_steps = 10 
        if len(scaled_data) >= time_steps:
            latest_sequence = scaled_data[-time_steps:]
            latest_sequence = np.expand_dims(latest_sequence, axis=0)
            prob = lstm_model.predict(latest_sequence, verbose=0)[0][0]
            up_prob = prob * 100
            down_prob = (1 - prob) * 100
        else:
            up_prob, down_prob = 50.0, 50.0
    else:
        up_prob, down_prob = 50.0, 50.0
        
    return df, df_c.iloc[-1], up_prob, down_prob

# 메인 UI
st.title("🧠 AI 참모 v3.0 (LSTM 딥러닝 탑재)")
st.write("---")

col1, col2, col3 = st.columns(3)
sel_tf = None
if col1.button("🔄 1시간 분석"): sel_tf = "1h"
if col2.button("🔄 4시간 분석"): sel_tf = "4h"
if col3.button("🔄 1일 분석"): sel_tf = "1d"

if sel_tf:
    with st.spinner('LSTM 딥러닝이 차트의 흐름을 분석 중입니다...'):
        df, latest, up, down = get_analysis_data(sel_tf)
        
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 BTC 가격", f"${latest['Close']:,.1f}")
        m2.metric("상승 확률", f"{up:.1f}%")
        m3.metric("하락 확률", f"{down:.1f}%")
        
        ai_msg = get_safe_ai_briefing(df, up, down)
        st.info(f"🤖 **실시간 브리핑**\n\n{ai_msg}")
        
        # 차트 그리기
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Fortress_Price'], line=dict(color='#ffaa00', width=2), name='성벽'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_DK'], name='RSI_DK'), row=2, col=1)
        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
