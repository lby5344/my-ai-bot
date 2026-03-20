import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai  # 추가: 제미나이 라이브러리

# 0. 제미나이 AI 설정 (본인의 API 키를 입력하세요)
GEMINI_API_KEY = "AIzaSyDxWi6FPNI1UZLpHHFGz9Iquqjcsxbbfps"
genai.configure(api_key=GEMINI_API_KEY)

# 1. 페이지 설정 및 폰트 스타일
st.set_page_config(page_title="AI 참모 v2.5 (Clean River v6)", page_icon="🧲", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], .st-emotion-cache-zt5idj, p, h1, h2, h3, h4, div {
        font-family: 'Nanum Gothic', sans-serif !important;
    }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 1.1rem; color: #a0a0a0; }
    [data-testid="stPlotlyChart"] { background-color: rgba(0,0,0,0) !important; }
</style>
""", unsafe_allow_html=True)

# --- 신규: AI 시황 분석 브리핑 함수 ---
def get_ai_briefing(df, up, down):
    try:
        model = genai.GenerativeModel('gemini-pro')
        latest = df.iloc[-1]
        
        prompt = f"""
        당신은 암호화폐 전문 분석가 'AI 참모'입니다. 아래 데이터를 바탕으로 현재 비트코인 시황을 전문가처럼 분석해라.
        - 현재가: ${latest['Close']:,.1f}
        - AI 예측: 상승 확률 {up:.1f}%, 하락 확률 {down:.1f}%
        - RSI_DK: {latest['RSI_DK']:.1f}
        - 강물 기울기: {latest['River_Slope']:.4f}
        - 4H 대추세: {'BULL' if latest['Trend_4H'] == 1 else 'BEAR'}
        
        조건: 
        1. 3줄 이내로 핵심만 말할 것.
        2. 말투는 신뢰감 있는 전문가 어조로.
        3. 진입, 관망, 익절 중 하나를 명확히 권고할 것.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ AI 시황 분석을 불러올 수 없습니다. API 키를 확인하세요."

# 2. 지표 계산기 (Clean River v6 로직)
def add_indicators_v6(df, mtf_df=None):
    fastLen = 12
    slowLen = 26
    smoothLen = 5
    rsiLen = 14
    
    # ATR 계산
    high_low = df['High'] - df['Low']
    high_cp = abs(df['High'] - df['Close'].shift())
    low_cp = abs(df['Low'] - df['Close'].shift())
    df['ATRr_14'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(rsiLen).mean()
    
    # River 강물 엔진 (Smoothed MACD)
    ema12 = df['Close'].ewm(span=fastLen, adjust=False).mean()
    ema26 = df['Close'].ewm(span=slowLen, adjust=False).mean()
    macd_raw = ema12 - ema26
    
    df['River_Smoothed'] = macd_raw.ewm(span=smoothLen, adjust=False).mean()
    df['River_Slope'] = df['River_Smoothed'].diff()
    df['River_Accel'] = df['River_Slope'].diff()
    
    # RSI_DK 로직
    df['lo'] = df['Low'].rolling(window=rsiLen).min()
    df['hi'] = df['High'].rolling(window=rsiLen).max()
    denom = df['hi'] - df['lo']
    df['RSI_DK'] = np.where(denom == 0, 50, (df['Close'] - df['lo']) / denom * 100)
    
    # 성벽 라인 (Fortress Price)
    fortress_price = np.full(len(df), np.nan)
    curr_price = np.nan
    
    preAnchor = (df['River_Slope'] < 0) & (df['River_Accel'] > (df['River_Slope'].abs() * 0.1))
    preBarricade = (df['River_Slope'] > 0) & (df['River_Accel'] < -(df['River_Slope'].abs() * 0.1))

    for i in range(rsiLen, len(df)):
        if preAnchor.iloc[i]: curr_price = df['Low'].iloc[i]
        elif preBarricade.iloc[i]: curr_price = df['High'].iloc[i]
        fortress_price[i] = curr_price
        
    df['Fortress_Price'] = fortress_price
    df['Long_Anchor'] = preAnchor.astype(int)
    df['Short_Barricade'] = preBarricade.astype(int)

    if mtf_df is not None:
        mtf_df['River_4H'] = (mtf_df['Close'].ewm(span=12).mean() - mtf_df['Close'].ewm(span=26).mean()).ewm(span=5).mean()
        mtf_df['is4hBull'] = (mtf_df['River_4H'] >= 0).astype(int)
        mtf_merge = mtf_df[['Date', 'is4hBull', 'RSI_DK']].rename(columns={'Date':'Date_4H', 'is4hBull':'Trend_4H', 'RSI_DK':'RSI_4H'})
        df = pd.merge_asof(df.sort_values('Date'), mtf_merge.sort_values('Date_4H'), left_on='Date', right_on='Date_4H', direction='backward')
        df['Trend_4H_Status'] = df['Trend_4H'].map({1: "🟢 BULL", 0: "🔴 BEAR"})
        df['Trend_1H_Status'] = (df['River_Smoothed'] >= 0).map({True: "🟢 BULL", False: "🔴 BEAR"})

    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    exchange = ccxt.kraken()
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=400), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    mtf_df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=150), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    mtf_df['Date'] = pd.to_datetime(mtf_df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    mtf_df = add_indicators_v6(mtf_df) 
    df = add_indicators_v6(df, mtf_df)
    
    features = ['Volume', 'RSI_DK', 'River_Smoothed', 'River_Slope', 'River_Accel', 'Long_Anchor', 'Short_Barricade', 'Trend_4H', 'Fortress_Price']
    df_clean = df.dropna()
    df_clean['Target'] = (df_clean['Close'].shift(-1) > df_clean['Close']).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df_clean[features], df_clean['Target'])
    
    latest_bar = df.iloc[-1]
    prob = model.predict_proba(pd.DataFrame([latest_bar[features]]))[0]
    
    importance_df = pd.DataFrame({'지표 (Feature)': features, '중요도 (Importance)': model.feature_importances_}).sort_values('중요도 (Importance)', ascending=False)
    
    return df, latest_bar, prob[1]*100, prob[0]*100, importance_df

def create_trading_chart(df, tf_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candle', increasing_line_color='#00ffbb', decreasing_line_color='#ff0055'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Fortress_Price'], line=dict(color='#ffaa00', width=2, shape='hv'), name='🧲 성벽'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_DK'], name='RSI_DK', line=dict(color='#00e676')), row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    return fig

# --- 메인 UI ---
st.title("🧲 AI 참모 (Clean River v6 엔진)")
st.write("---")

col1, col2, col3 = st.columns(3)
tf_map = {"🔄 1시간 분석": "1h", "🔄 4시간 분석": "4h", "🔄 1일 분석": "1d"}
selected_tf_label = None

if col1.button("🔄 1시간 분석"): selected_tf_label = "🔄 1시간 분석"
if col2.button("🔄 4시간 분석"): selected_tf_label = "🔄 4시간 분석"
if col3.button("🔄 1일 분석"): selected_tf_label = "🔄 1일 분석"

if selected_tf_label:
    with st.spinner('AI 참모가 데이터를 분석하고 시황을 작성 중입니다...'):
        df, latest_bar, up, down, feature_importance_df = get_analysis_data(tf_map[selected_tf_label])
        
        # 신규: 제미나이 브리핑 가져오기
        ai_msg = get_ai_briefing(df, up, down)
        
        st.write("---")
        st.subheader(f"📊 {selected_tf_label} 분석 보고서")
        
        # 메트릭 표시
        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"${latest_bar['Close']:,.1f}")
        m2.metric("상승 확률", f"{up:.1f}%")
        m3.metric("하락 확률", f"{down:.1f}%")

        # [여기에 AI 브리핑 출력]
        st.info(f"🤖 **AI 참모 브리핑:**\n\n{ai_msg}")

        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.table(pd.DataFrame({"항목": ["4H 대추세", "선행 RSI_DK", "강물 기울기"], 
                                   "값": [latest_bar.get('Trend_4H_Status', 'N/A'), round(latest_bar['RSI_DK'],1), round(latest_bar['River_Slope'],3)]}))
        with c2:
            st.dataframe(feature_importance_df, hide_index=True)

        st.plotly_chart(create_trading_chart(df, selected_tf_label), use_container_width=True)
