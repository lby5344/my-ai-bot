import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 페이지 설정 및 사용자 정의 스타일
st.set_page_config(page_title="AI 트레이딩 참모 v2.4", page_icon="🧲", layout="wide")

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

# 2. 지표 계산기 (+ 🧲 자석 라인 로직 추가)
def add_indicators(df):
    # RSI, MACD, ATR 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD'] - df['Signal']
    
    high_low = df['High'] - df['Low']
    high_cp = abs(df['High'] - df['Close'].shift())
    low_cp = abs(df['Low'] - df['Close'].shift())
    df['ATRr_14'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean().fillna(0)
    
    # River 엔진
    df['norm'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['river_upper'] = df['norm'] + (df['ATRr_14'] * 1.5)
    df['river_lower'] = df['norm'] - (df['ATRr_14'] * 1.5)
    
    river_macd = df['MACDh_12_26_9'].ewm(span=5, adjust=False).mean()
    df['River'] = river_macd
    df['River_Slope'] = river_macd.diff()
    df['River_Accel'] = df['River_Slope'].diff()

    # 🧲 핵심: 자석 라인(Magnet Line - 계단식 추세선) 고속 계산
    hl2 = (df['High'] + df['Low']) / 2
    m_atr = 3.0 * df['ATRr_14'] # 계단 높이 민감도
    ub = hl2 + m_atr
    lb = hl2 - m_atr
    
    close_arr = df['Close'].values
    ub_arr = ub.values
    lb_arr = lb.values
    
    dir_arr = np.ones(len(df))
    magnet_arr = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if close_arr[i] > ub_arr[i-1]:
            dir_arr[i] = 1
        elif close_arr[i] < lb_arr[i-1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i-1]
            
        if dir_arr[i] == 1:
            if lb_arr[i] < lb_arr[i-1]: lb_arr[i] = lb_arr[i-1] # 계단 유지(안 내려감)
            magnet_arr[i] = lb_arr[i]
        else:
            if ub_arr[i] > ub_arr[i-1]: ub_arr[i] = ub_arr[i-1] # 계단 유지(안 올라감)
            magnet_arr[i] = ub_arr[i]
            
    df['Magnet_Trend'] = dir_arr # 상승 1, 하락 -1
    df['Magnet_Line'] = magnet_arr # 자석 라인 가격
    df['Magnet_Dist'] = df['Close'] - df['Magnet_Line'] # 자석과의 거리 (이격도)
    
    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    exchange = ccxt.kraken()
    raw = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=300)
    df = pd.DataFrame(raw, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    df = add_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 🧠 AI 두뇌에 '자석 라인 상태'와 '거리' 추가 학습!
    features = ['ATRr_14', 'MACDh_12_26_9', 'RSI_14', 'Volume', 'River', 'Magnet_Trend', 'Magnet_Dist']
    df_clean = df.dropna()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df_clean[features], df_clean['Target'])
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        '지표 (Feature)': ['ATR (변동성)', 'MACD (모멘텀)', 'RSI (과매수/매도)', '거래량', 'River (강물추세)', '자석방향 (Magnet Trend)', '자석이격도 (Magnet Dist)'],
        '중요도 (Importance)': importances
    }).sort_values(by='중요도 (Importance)', ascending=False)
    
    latest_bar = df.iloc[-1]
    prob = model.predict_proba(pd.DataFrame([latest_bar[features]]))[0]
    
    return df, latest_bar['Close'], latest_bar['Date'], prob[1]*100, prob[0]*100, feature_importance_df, latest_bar['Magnet_Trend']

# 3. 차트 생성 함수 (계단식 자석 라인 추가)
def create_trading_chart(df, tf_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # 캔들스틱
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
    
    # 🧲 계단식 자석 라인 시각화 (shape='hv' 속성으로 완벽한 계단 모양 구현)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Magnet_Line'], 
                             line=dict(color='#ffaa00', width=2, shape='hv'), # 주황색 계단선
                             name='자석 라인 (Magnet)'), row=1, col=1)
    
    # 강물(River) 배경
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_upper'], line=dict(color='rgba(0,180,0,0.0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_lower'], fill='tonexty', fillcolor='rgba(0,180,0,0.1)', line=dict(color='rgba(0,180,0,0.0)'), name='River 강물'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI', line=dict(color='#00e676')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=2, col=1)
    
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=20, r=20, t=50, b=20))
    fig.update_yaxes(side="right")
    return fig

# 중요도 막대그래프 함수
def create_feature_importance_chart(df):
    fig = go.Figure(go.Bar(
        x=df['중요도 (Importance)'], y=df['지표 (Feature)'], orientation='h',
        marker=dict(color=df['중요도 (Importance)'], colorscale='Blues_r')
    ))
    fig.update_layout(template='plotly_dark', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

# UI 구성
st.title("🧲 24시간 AI 참모 (자석 라인 탑재)")
st.write("---")

col1, col2, col3 = st.columns(3)
tf_map = {"1시간": "1h", "4시간": "4h", "1일": "1d"}
selected_tf = None

with col1:
    if st.button("🔄 1시간 분석"): selected_tf = "1시간"
with col2:
    if st.button("🔄 4시간 분석"): selected_tf = "4시간"
with col3:
    if st.button("🔄 1일 분석"): selected_tf = "1일"

if selected_tf:
    with st.spinner(f'AI가 {selected_tf} 차트의 자석 라인을 추적 중...'):
        df, price, time, up, down, feature_importance_df, magnet_dir = get_analysis_data(tf_map[selected_tf])
        
        st.write("---")
        st.subheader(f"📊 {selected_tf} 분석 보고서 ({time.strftime('%Y-%m-%d %H:%M')})")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 BTC 가격", f"${price:,.1f}")
        m2.metric("상승 확률 (📈 LONG)", f"{up:.1f}%")
        m3.metric("하락 확률 (📉 SHORT)", f"{down:.1f}%")
        
        st.write("---")
        # 자석 라인 상태 브리핑 추가
        magnet_status = "🟢 상승 추세 지지 중" if magnet_dir == 1 else "🔴 하락 추세 저항 중"
        st.info(f"🧲 **현재 자석 라인 상태:** {magnet_status} (이 선이 뚫리면 추세가 반전됩니다!)")
        
        if up >= 60: st.success(f"🔥 **최종 결론:** {selected_tf} 차트 확인 후 **LONG 매수** 권장")
        elif down >= 60: st.error(f"❄️ **최종 결론:** {selected_tf} 차트 확인 후 **SHORT 매도** 권장")
        else: st.warning("⚠️ **최종 결론:** 방향성이 불확실하니 **관망** 추천")

        st.write("---")
        st.subheader(f"📊 실시간 {selected_tf} 차트 & 자석 라인")
        fig_chart = create_trading_chart(df, selected_tf)
        st.plotly_chart(fig_chart, use_container_width=True)

        st.write("---")
        st.subheader("📊 지표 중요도 (AI가 무엇을 가장 중요하게 봤을까?)")
        fig_importance = create_feature_importance_chart(feature_importance_df)
        st.plotly_chart(fig_importance, use_container_width=True)
