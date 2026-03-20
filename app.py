import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

# [중요] 트레이더님이 주신 키를 직접 입력했습니다.
GEMINI_API_KEY = "AIzaSyDxWi6FPNI1UZLpHHFGz9Iquqjcsxbbfps".strip()
genai.configure(api_key=GEMINI_API_KEY)

# 1. 페이지 설정
st.set_page_config(page_title="AI 참모 v2.5 (Clean River v6)", page_icon="🧲", layout="wide")

# 스타일 설정
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], p, h1, h2, h3, h4, div { font-family: 'Nanum Gothic', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; color: #00ffbb; }
</style>
""", unsafe_allow_html=True)

# AI 시황 분석 함수 (오류 방지 강화)
def get_ai_briefing(df, up, down):
    try:
        # 모델 설정 (gemini-1.5-flash가 더 빠르고 안정적일 수 있습니다)
        model = genai.GenerativeModel('gemini-pro')
        latest = df.iloc[-1]
        
        prompt = f"""
        당신은 암호화폐 전문 분석가 'AI 참모'입니다. 아래 데이터를 바탕으로 비트코인 시황을 분석하세요.
        - 현재가: ${latest['Close']:,.1f}
        - AI 예측: 상승 {up:.1f}%, 하락 {down:.1f}%
        - RSI_DK: {latest['RSI_DK']:.1f}
        - 4H 추세: {'BULL' if latest.get('Trend_4H', 0) == 1 else 'BEAR'}
        
        요구사항: 전문가답게 딱 3줄로 핵심 전략(진입/관망/익절)을 제시할 것.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석 중 일시적 오류가 발생했습니다. (사유: {str(e)})"

# 2. 지표 계산기 (V6)
def add_indicators_v6(df, mtf_df=None):
    # RSI_DK 및 기본 지표
    rsiLen = 14
    df['lo'] = df['Low'].rolling(window=rsiLen).min()
    df['hi'] = df['High'].rolling(window=rsiLen).max()
    df['RSI_DK'] = np.where((df['hi'] - df['lo']) == 0, 50, (df['Close'] - df['lo']) / (df['hi'] - df['lo']) * 100)
    
    # River 엔진
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['River_Smoothed'] = (ema12 - ema26).ewm(span=5).mean()
    df['River_Slope'] = df['River_Smoothed'].diff()
    df['River_Accel'] = df['River_Slope'].diff()
    
    # 성벽 라인
    fortress = np.full(len(df), np.nan)
    curr = np.nan
    for i in range(1, len(df)):
        if df['River_Slope'].iloc[i] < 0 and df['River_Accel'].iloc[i] > (abs(df['River_Slope'].iloc[i]) * 0.1):
            curr = df['Low'].iloc[i]
        elif df['River_Slope'].iloc[i] > 0 and df['River_Accel'].iloc[i] < -(abs(df['River_Slope'].iloc[i]) * 0.1):
            curr = df['High'].iloc[i]
        fortress[i] = curr
    df['Fortress_Price'] = fortress
    
    # 4H 추세 병합
    if mtf_df is not None:
        mtf_df['is4hBull'] = (mtf_df['Close'].ewm(span=12).mean() - mtf_df['Close'].ewm(span=26).mean() >= 0).astype(int)
        mtf_merge = mtf_df[['Date', 'is4hBull']].rename(columns={'Date':'Date_4H', 'is4hBull':'Trend_4H'})
        df = pd.merge_asof(df.sort_values('Date'), mtf_merge.sort_values('Date_4H'), left_on='Date', right_on='Date_4H', direction='backward')
    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    # [수정] 바이낸스 대신 바이비트(Bybit)를 사용합니다. 
    # 바이비트는 클라우드 서버 접속에 훨씬 관대합니다.
    ex = ccxt.bybit() 
    
    try:
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200)
    except:
        # 만약 바이비트도 안되면 다시 크라켄으로 우회
        ex = ccxt.kraken()
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200)

    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    # 4시간 데이터도 동일하게 변경
    ohlcv_4h = ex.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100)
    df_4h = pd.DataFrame(ohlcv_4h, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_4h['Date'] = pd.to_datetime(df_4h['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    df = add_indicators_v6(df, df_4h)
    
    # ML 예측 로직 (피처 개수 일치 확인)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI_DK', 'River_Slope', 'River_Accel'] # 피처 단순화
    df_c = df.dropna()
    
    model = RandomForestClassifier(n_estimators=50).fit(df_c[features], df_c['Target'])
    prob = model.predict_proba(pd.DataFrame([df.iloc[-1][features]], columns=features))[0]
    
    return df, df.iloc[-1], prob[1]*100, prob[0]*100
# 3. 메인 UI
st.title("🧲 AI 참모 v2.5 (Clean River v6)")
st.caption("제미나이 1.5 엔진이 실시간으로 시황을 분석합니다.")

col1, col2, col3 = st.columns(3)
tf_map = {"1시간": "1h", "4시간": "4h", "1일": "1d"}
sel_tf = None
if col1.button("🔄 1시간 분석"): sel_tf = "1h"
if col2.button("🔄 4시간 분석"): sel_tf = "4h"
if col3.button("🔄 1일 분석"): sel_tf = "1d"

if sel_tf:
    with st.spinner('차트 분석 및 시황 작성 중...'):
        df, latest, up, down = get_analysis_data(sel_tf)
        ai_msg = get_ai_briefing(df, up, down)
        
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"${latest['Close']:,.1f}")
        m2.metric("상승 확률", f"{up:.1f}%")
        m3.metric("하락 확률", f"{down:.1f}%")
        
        st.info(f"🤖 **AI 참모의 실시간 브리핑**\n\n{ai_msg}")
        
        # 차트 그리기
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Fortress_Price'], line=dict(color='#ffaa00', width=2, shape='hv'), name='성벽'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_DK'], name='RSI', line=dict(color='#00e676')), row=2, col=1)
        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)
