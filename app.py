import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots # 서브플롯 도구 추가

# 1. 페이지 설정 및 사용자 정의 스타일 (우리가 얘기하는 폰트 적용)
st.set_page_config(page_title="AI 트레이딩 참모 v2.2", page_icon="🤖", layout="wide")

# 대화창 느낌의 깔끔한 나눔고딕 폰트 적용 (Unsafe HTML 활용)
st.markdown("""
<style>
    /* 나눔고딕 폰트 import */
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    
    /* 전체 기본 폰트 적용 */
    html, body, [data-testid="stSidebar"], .st-emotion-cache-zt5idj, p, h1, h2, h3, h4, div {
        font-family: 'Nanum Gothic', sans-serif !important;
    }
    
    /* 메트릭 박스 스타일링 (LLM 챗 느낌) */
    div[data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; color: #f0f0f0; }
    div[data-testid="stMetricLabel"] { font-size: 1.1rem; color: #a0a0a0; }
    
    /* 차트 배경색 투명화 */
    [data-testid="stPlotlyChart"] { background-color: rgba(0,0,0,0) !important; }
</style>
""", unsafe_allow_html=True)

# 2. 지표 계산기 (pandas-ta 없이 직접 계산)
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
    
    # River 엔진 (추세) - 시각화용 가격 기반 밴드로 재정의
    df['norm'] = df['Close'].ewm(span=20, adjust=False).mean() # 중심선
    df['river_upper'] = df['norm'] + (df['ATRr_14'] * 1.5) # 상단 강둑
    df['river_lower'] = df['norm'] - (df['ATRr_14'] * 1.5) # 하단 강둑
    
    # AI 학습용 River 미분 지표 (v2.1 로직 유지)
    river_macd = df['MACDh_12_26_9'].ewm(span=5, adjust=False).mean()
    df['River'] = river_macd
    df['River_Slope'] = river_macd.diff()
    df['River_Accel'] = df['River_Slope'].diff()
    
    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    # 크라켄 거래소 사용 (ccxt)
    exchange = ccxt.kraken()
    raw = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=300)
    df = pd.DataFrame(raw, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    # 지표 추가
    df = add_indicators(df)
    
    # 학습 목표: 다음 캔들이 상승할지(1) 하락할지(0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 학습 피처 (v2.1 동일)
    features = ['ATRr_14', 'MACDh_12_26_9', 'RSI_14', 'Volume', 'River', 'River_Slope', 'River_Accel']
    df_clean = df.dropna()
    
    # AI 모델 학습 (가벼운 랜덤 포레스트)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(df_clean[features], df_clean['Target'])
    
    latest_bar = df.iloc[-1]
    prob = model.predict_proba(pd.DataFrame([latest_bar[features]]))[0]
    
    # 데이터셋 전체를 리턴 (그래프 그리기용)
    return df, latest_bar['Close'], latest_bar['Date'], prob[1]*100, prob[0]*100

# 3. 코랩 스타일 역동적 그래프 생성 함수 (Plotly Subplots 최적화)
def create_trading_chart(df, tf_name):
    # 2층짜리 서브플롯 생성 (메인 차트:RSI = 7:3 비율)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Row 1: 캔들차트 (가격 변화)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='가격'),
                  row=1, col=1)

    # Row 1: River Band (코랩 스타일 강물 배경 시각화)
    # 상단 밴드 Scatter
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_upper'], 
                             line=dict(color='rgba(0,180,0,0.2)'), showlegend=False), row=1, col=1)
    # 하단 밴드 Scatter + fill='tonexty'로 강둑 배경 채우기
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_lower'], 
                             fill='tonexty', fillcolor='rgba(0,180,0,0.1)', 
                             line=dict(color='rgba(0,180,0,0.2)'), name='River 강물'), row=1, col=1)
    # 중심선 (Norm)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['norm'], line=dict(color='orange', width=1), name='중심선'), row=1, col=1)


    # Row 2: RSI 지표 차트
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI', line=dict(color='#00e676')), row=2, col=1)
    # RSI 과매수/과매도 기준선 (70/30)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=2, col=1)

    # 레이아웃 스타일링 (모바일 최적화)
    fig.update_layout(
        template='plotly_dark', # 다크 모드
        xaxis_rangeslider_visible=False, # 하단 레인지 슬라이더 숨기기 (공간 확보)
        height=600, # 그래프 높이
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', # 배경 투명
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=50, b=20) # 여백 최소화
    )
    # 우측 Y축 배치 (모바일에서 가격 확인 용이)
    fig.update_yaxes(side="right")
    return fig

# UI 구성
st.title("🤖 24시간 실시간 AI 참모")
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
    with st.spinner(f'AI가 {selected_tf} 차트 패턴 분석 중...'):
        df, price, time, up, down = get_analysis_data(tf_map[selected_tf])
        
        st.write("---")
        st.subheader(f"📊 {selected_tf} 분석 보고서 ({time.strftime('%Y-%m-%d %H:%M')})")
        
        # 핵심 확률 메트릭 (폰트 적용됨)
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 BTC 가격", f"${price:,.1f}")
        m2.metric("상승 확률 (📈 LONG)", f"{up:.1f}%")
        m3.metric("하락 확률 (📉 SHORT)", f"{down:.1f}%")
        
        # 브리핑 메시지
        st.write("---")
        if up >= 60: st.success(f"🔥 **최종 결론:** {selected_tf} 차트 🟢 확인 후 **LONG 매수** 권장")
        elif down >= 60: st.error(f"❄️ **최종 결론:** {selected_tf} 차트 🔴 확인 후 **SHORT 매도** 권장")
        else: st.warning("⚠️ **최종 결론:** 방향성이 불확실하니 **관망** 추천 (휩소 주의)")

        # 🚀 [핵심 구현] 코랩 스타일 역동적 그래프 표시
        st.write("---")
        st.subheader(f"📊 실시간 {selected_tf} 차트 & River 시각화")
        fig = create_trading_chart(df, selected_tf)
        # use_container_width=True로 폰 화면에 꽉 차게 표시
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) # 차트 툴바 숨기기
