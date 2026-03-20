import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 페이지 설정 및 사용자 정의 스타일 (나눔고딕 폰트)
st.set_page_config(page_title="AI 참모 v2.5 (Clean River v6 탑재)", page_icon="🧲", layout="wide")

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

# 2. 지표 계산기 (🧲 Pine Script v6 'Clean River' 로직 100% 이식)
def add_indicators_v6(df, mtf_df=None):
    # 기본 변수 설정 (Pine Script 설정값 동일)
    fastLen      = 12
    slowLen      = 26
    smoothLen    = 5
    rsiLen       = 14
    
    # ATR 계산 (지지/저항 성벽 구축용)
    high_low = df['High'] - df['Low']
    high_cp = abs(df['High'] - df['Close'].shift())
    low_cp = abs(df['Low'] - df['Close'].shift())
    df['ATRr_14'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(rsiLen).mean()
    
    # 🌊 강물 엔진 (Smoothed MACD)
    ema12 = df['Close'].ewm(span=fastLen, adjust=False).mean()
    ema26 = df['Close'].ewm(span=slowLen, adjust=False).mean()
    macd_raw = ema12 - ema26 # ta.macd rRaw
    
    # 곡선 부드러움 (EMA 5)
    df['River_Smoothed'] = macd_raw.ewm(span=smoothLen, adjust=False).mean() # rSmooth
    
    # 기울기(Slope)와 가속도(Acceleration) 계산 - 후행성 제거 핵심 로직
    df['River_Slope'] = df['River_Smoothed'].diff()
    df['River_Accel'] = df['River_Slope'].diff()
    
    # 📉 RSI_DK (Stochastic RSI 변형) 로직
    df['lo'] = df['Low'].rolling(window=rsiLen).min()
    df['hi'] = df['High'].rolling(window=rsiLen).max()
    denom = df['hi'] - df['lo']
    # 분모가 0일 경우 예외처리
    df['RSI_DK'] = np.where(denom == 0, 50, (df['Close'] - df['lo']) / denom * 100)
    
    # 🧲 v6 '성벽 라인' (Fortress Price - 계단식 황색선) 계산 - 선행성 지지/저항
    # 기울기와 가속도를 분석하여 꺾이는 지점을 선제적으로 포착
    slope_val = df['River_Slope']
    accel_val = df['River_Accel']
    
    # Anchor(Long 신호), Barricade(Short 신호) 조건 계산
    slope_abs = slope_val.abs()
    # 조건 만족 시점에 LOW 또는 HIGH로 지지선 업데이트 (Var Float 구현)
    preAnchor = (slope_val < 0) & (accel_val > (slope_abs * 0.1)) # Long Anchor
    preBarricade = (slope_val > 0) & (accel_val < -(slope_abs * 0.1)) # Short Barricade

    # 계단식 선 구축을 위한 넘파이 고속 루프
    fortress_price = np.full(len(df), np.nan)
    curr_price = np.nan
    
    preAnchor_arr = preAnchor.values
    preBarricade_arr = preBarricade.values
    low_arr = df['Low'].values
    high_arr = df['High'].values
    
    for i in range(rsiLen, len(df)):
        if preAnchor_arr[i]:
            curr_price = low_arr[i]
        elif preBarricade_arr[i]:
            curr_price = high_arr[i]
        
        fortress_price[i] = curr_price
        
    df['Fortress_Price'] = fortress_price # v2.5 최종 계단 지지/저항선
    df['Long_Anchor'] = preAnchor_arr.astype(int) # 학습 피처로 사용
    df['Short_Barricade'] = preBarricade_arr.astype(int) # 학습 피처로 사용

    # MTF (4시간 추세) 동기화 - request.security 구현
    if mtf_df is not None:
        # 큰 강물 흐름 판독 (4H MACD smoothed >= 0 ? 1 : -1)
        ema12_4 = mtf_df['Close'].ewm(span=fastLen, adjust=False).mean()
        ema26_4 = mtf_df['Close'].ewm(span=slowLen, adjust=False).mean()
        macd_raw4 = ema12_4 - ema26_4
        mtf_df['River_4H'] = macd_raw4.ewm(span=smoothLen, adjust=False).mean()
        mtf_df['is4hBull'] = (mtf_df['River_4H'] >= 0).astype(int)
        
        # 4시간 데이터 1시간 차트에 실시간 동기화
        mtf_merge = mtf_df[['Date', 'is4hBull', 'RSI_DK']].rename(columns={'Date':'Date_4H', 'is4hBull':'Trend_4H', 'RSI_DK':'RSI_4H'})
        df = pd.merge_asof(df.sort_values('Date'), mtf_merge.sort_values('Date_4H'), left_on='Date', right_on='Date_4H', direction='backward')
        
        # Dashboard용 가독성 데이터
        df['Trend_4H_Status'] = df['Trend_4H'].map({1: "🟢 BULL", 0: "🔴 BEAR"})
        df['Trend_1H_Status'] = (df['River_Smoothed'] >= 0).map({True: "🟢 BULL", False: "🔴 BEAR"})

    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    # 크라켄 거래소에서 1H(학습용) 및 4H(MTF용) OHLCV 데이터 수집
    exchange = ccxt.kraken()
    limit_counts = 400
    
    # 현재 타임프레임 데이터
    raw_tf = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=limit_counts)
    df = pd.DataFrame(raw_tf, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    # MTF 데이터를 위한 4H 데이터 수집
    raw_4h = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=150)
    mtf_df = pd.DataFrame(raw_4h, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    mtf_df['Date'] = pd.to_datetime(mtf_df['Date'], unit='ms') + pd.Timedelta(hours=9)
    # 4H RSI_DK 등 기본 지표 계산 (병합 전)
    mtf_df = add_indicators_v6(mtf_df, None) 
    
    # v2.5 Clean River 엔진 실행 (병합 및 지표 계산)
    df = add_indicators_v6(df, mtf_df)
    
    # AI 학습 목표: 다음 캔들이 상승할지(1) 하락할지(0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 🧠 v2.5 선행성 지표로 완전히 개조된 AI 두뇌
    features = ['Volume', 'RSI_DK', 'River_Smoothed', 'River_Slope', 'River_Accel', 'Long_Anchor', 'Short_Barricade', 'Trend_4H', 'Fortress_Price']
    df_clean = df.dropna()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df_clean[features], df_clean['Target'])
    
    # 중요도 추출 및 한글화
    importances = model.feature_importances_
    f_map_ko = {'Volume': '거래량 (Volume)', 'RSI_DK': '선행 RSI_DK (Clean)', 'River_Smoothed': '강물 모멘텀 (Smoothed)', 'River_Slope': '강물 기울기 (Slope)', 'River_Accel': '강물 가속도 (Acceleration)', 'Long_Anchor': 'Long 신호(v6)', 'Short_Barricade': 'Short 신호(v6)', 'Trend_4H': '4H 대추세 (MTF)', 'Fortress_Price': '성벽 지지/저항 가격'}
    f_names_ko = [f_map_ko[f] for f in features]
    
    feature_importance_df = pd.DataFrame({
        '지표 (Feature)': f_names_ko,
        '중요도 (Importance)': importances
    }).sort_values(by='중요도 (Importance)', ascending=False)
    
    latest_bar = df.iloc[-1]
    prob = model.predict_proba(pd.DataFrame([latest_bar[features]]))[0]
    
    return df, latest_bar, prob[1]*100, prob[0]*100, feature_importance_df

# 3. 차트 생성 함수 (v6 시각화 이식)
def create_trading_chart(df, tf_name):
    # 가격 차트와 RSI_DK 차트 서브플롯 구성
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # 1시간 Clean 캔들스틱 (청록색/심홍색)
    bColor = np.where(df['Close'] >= df['Open'], '#00ffbb', '#ff0055')
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='1H Clean Candle', marker=dict(color=bColor), line=dict(color=bColor)), row=1, col=1)
    
    # 🧲 v6 성벽 라인 (황색 계단식 계단선) - shape='hv' 중요
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Fortress_Price'], line=dict(color='#ffaa00', width=2, shape='hv'), name='🧲 성벽 지지/저항 (v6)'), row=1, col=1)
    
    # River 강물 배경 시각화
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_upper'], line=dict(color='rgba(0,180,0,0.0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_lower'], fill='tonexty', fillcolor='rgba(0,180,0,0.1)', line=dict(color='rgba(0,180,0,0.0)'), name='River 강물 영역'), row=1, col=1)
    
    # RSI_DK 및 과매수/매도 성벽 배경색
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_DK'], name='선행 RSI_DK', line=dict(color='#00e676', width=1)), row=2, col=1)
    
    # 배경색 과매수/과매도 영역 추가 (v6 코드 bgcolor 구현)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(0,128,0,0.1)", line_width=0, row=2, col=1) # 과매수 (녹색)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(255,0,0,0.1)", line_width=0, row=2, col=1)  # 과매도 (적색)
    
    # 차트 레이아웃 최적화
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

# 4. 대시보드 테이블 함수 (v6 대시보드 구현)
def show_v6_dashboard(bar):
    col1, col2 = st.columns(2)
    with col1:
        st.write("### v6 Clean Dashboard")
        dash_df = pd.DataFrame({
            "항목": ["4H 대추세 (MTF)", "선행 RSI_DK", "강물 기울기 (Slope)", "현재 1H 상태"],
            "값": [bar['Trend_4H_Status'], str(round(bar['RSI_DK'], 1)), str(round(bar['River_Slope'], 3)), bar['Trend_1H_Status']]
        })
        st.table(dash_df)
    with col2:
        st.write("### AI 중요도 분석 결과")
        st.dataframe(feature_importance_df, hide_index=True)

# UI 구성 (전체 최적화)
st.title("🧲 AI 참모 (Clean River v6 엔진 탑재)")
st.write("---")

col1, col2, col3 = st.columns(3)
tf_map = {"🔄 1시간 분석": "1h", "🔄 4시간 분석": "4h", "🔄 1일 분석": "1d"}
selected_tf_label = None

with col1:
    if st.button("🔄 1시간 분석"): selected_tf_label = "🔄 1시간 분석"
with col2:
    if st.button("🔄 4시간 분석"): selected_tf_label = "🔄 4시간 분석"
with col3:
    if st.button("🔄 1일 분석"): selected_tf_label = "🔄 1일 분석"

if selected_tf_label:
    with st.spinner(f'AI가 {selected_tf_label} 차트의 기울기와 가속도를 정밀 추적 중...'):
        # v2.5 데이터 수집 및 AI 분석
        df, latest_bar, up, down, feature_importance_df = get_analysis_data(tf_map[selected_tf_label])
        
        st.write("---")
        st.subheader(f"📊 {selected_tf_label} 분석 보고서 ({latest_bar['Date'].strftime('%Y-%m-%d %H:%M')})")
        
        # 1. 핵심 확률 메트릭
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 BTC 가격", f"${latest_bar['Close']:,.1f}")
        m2.metric("상승 확률 (📈 LONG)", f"{up:.1f}%")
        m3.metric("하락 확률 (📉 SHORT)", f"{down:.1f}%")
        
        # 2. v6 전용 대시보드 표시
        st.write("---")
        show_v6_dashboard(latest_bar)

        st.write("---")
        # 최종 결론 브리핑
        if up >= 60: st.success(f"🔥 **최종 결론:** {selected_tf_label} 차트🟢 확인 후 **LONG 매수** 권장")
        elif down >= 60: st.error(f"❄️ **최종 결론:** {selected_tf_label} 차트 🔴 확인 후 **SHORT 매도** 권장")
        else: st.warning("⚠️ **최종 결론:** 방향성이 불확실하니 **관망** 추천")

        # 3. 실시간 차트 & River v6 시각화
        st.write("---")
        st.subheader(f"📊 실시간 {selected_tf_label} 차트 & v6 성벽 라인")
        fig_chart = create_trading_chart(df, selected_tf_label)
        st.plotly_chart(fig_chart, use_container_width=True)
