import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    slope_val = df['River_Slope']
    accel_val = df['River_Accel']
    slope_abs = slope_val.abs()
    
    preAnchor = (slope_val < 0) & (accel_val > (slope_abs * 0.1))
    preBarricade = (slope_val > 0) & (accel_val < -(slope_abs * 0.1))

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
        
    df['Fortress_Price'] = fortress_price
    df['Long_Anchor'] = preAnchor_arr.astype(int)
    df['Short_Barricade'] = preBarricade_arr.astype(int)

    # MTF (4시간 추세) 동기화
    if mtf_df is not None:
        ema12_4 = mtf_df['Close'].ewm(span=fastLen, adjust=False).mean()
        ema26_4 = mtf_df['Close'].ewm(span=slowLen, adjust=False).mean()
        macd_raw4 = ema12_4 - ema26_4
        mtf_df['River_4H'] = macd_raw4.ewm(span=smoothLen, adjust=False).mean()
        mtf_df['is4hBull'] = (mtf_df['River_4H'] >= 0).astype(int)
        
        mtf_merge = mtf_df[['Date', 'is4hBull', 'RSI_DK']].rename(columns={'Date':'Date_4H', 'is4hBull':'Trend_4H', 'RSI_DK':'RSI_4H'})
        df = pd.merge_asof(df.sort_values('Date'), mtf_merge.sort_values('Date_4H'), left_on='Date', right_on='Date_4H', direction='backward')
        
        df['Trend_4H_Status'] = df['Trend_4H'].map({1: "🟢 BULL", 0: "🔴 BEAR"})
        df['Trend_1H_Status'] = (df['River_Smoothed'] >= 0).map({True: "🟢 BULL", False: "🔴 BEAR"})

    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    exchange = ccxt.kraken()
    # 데이터 수집 (1H, 4H)
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=400), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    mtf_df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=150), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    mtf_df['Date'] = pd.to_datetime(mtf_df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    mtf_df = add_indicators_v6(mtf_df, None) 
    df = add_indicators_v6(df, mtf_df)
    
    # 머신러닝 타겟 및 피처
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['Volume', 'RSI_DK', 'River_Smoothed', 'River_Slope', 'River_Accel', 'Long_Anchor', 'Short_Barricade', 'Trend_4H', 'Fortress_Price']
    df_clean = df.dropna()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df_clean[features], df_clean['Target'])
    
    importances = model.feature_importances_
    f_map_ko = {'Volume': '거래량', 'RSI_DK': '선행 RSI_DK', 'River_Smoothed': '강물 모멘텀', 'River_Slope': '강물 기울기', 'River_Accel': '강물 가속도', 'Long_Anchor': 'Long 닻(v6)', 'Short_Barricade': 'Short 바리케이드(v6)', 'Trend_4H': '4H 대추세', 'Fortress_Price': '성벽 지지/저항'}
    f_names_ko = [f_map_ko[f] for f in features]
    
    feature_importance_df = pd.DataFrame({
        '지표 (Feature)': f_names_ko,
        '중요도 (Importance)': importances
    }).sort_values(by='중요도 (Importance)', ascending=False)
    
    latest_bar = df.iloc[-1]
    prob = model.predict_proba(pd.DataFrame([latest_bar[features]]))[0]
    
    return df, latest_bar, prob[1]*100, prob[0]*100, feature_importance_df

def create_trading_chart(df, tf_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # [수정된 부분] Plotly 전용 양봉(increasing)/음봉(decreasing) 색상 지정 방식 사용
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Clean Candle', 
        increasing_line_color='#00ffbb', decreasing_line_color='#ff0055'
    ), row=1, col=1)
    
    # 🧲 v6 성벽 라인 (계단식)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Fortress_Price'], line=dict(color='#ffaa00', width=2, shape='hv'), name='🧲 성벽 라인'), row=1, col=1)
    
    # 시각화용 River 밴드
    df['norm'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['river_upper'] = df['norm'] + (df['ATRr_14'] * 1.5)
    df['river_lower'] = df['norm'] - (df['ATRr_14'] * 1.5)
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_upper'], line=dict(color='rgba(0,180,0,0.0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['river_lower'], fill='tonexty', fillcolor='rgba(0,180,0,0.1)', line=dict(color='rgba(0,180,0,0.0)'), name='River 구역'), row=1, col=1)
    
    # RSI_DK
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_DK'], name='RSI_DK', line=dict(color='#00e676', width=1)), row=2, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(0,128,0,0.1)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(255,0,0,0.1)", line_width=0, row=2, col=1)
    
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=20, r=20, t=50, b=20))
    fig.update_yaxes(side="right")
    return fig

def create_feature_importance_chart(df):
    fig = go.Figure(go.Bar(
        x=df['중요도 (Importance)'], y=df['지표 (Feature)'], orientation='h',
        marker=dict(color=df['중요도 (Importance)'], colorscale='Blues_r')
    ))
    fig.update_layout(template='plotly_dark', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

st.title("🧲 AI 참모 (Clean River v6 엔진)")
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
    with st.spinner(f'AI가 {selected_tf_label} 차트의 v6 로직을 추적 중...'):
        df, latest_bar, up, down, feature_importance_df = get_analysis_data(tf_map[selected_tf_label])
        
        st.write("---")
        st.subheader(f"📊 {selected_tf_label} 분석 보고서")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 BTC 가격", f"${latest_bar['Close']:,.1f}")
        m2.metric("상승 확률 (📈 LONG)", f"{up:.1f}%")
        m3.metric("하락 확률 (📉 SHORT)", f"{down:.1f}%")
        
        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.write("### 🧲 v6 대시보드")
            # get() 메서드를 사용하여 KeyError 방지
            trend_4h = latest_bar.get('Trend_4H_Status', '계산 중...')
            trend_1h = latest_bar.get('Trend_1H_Status', '계산 중...')
            
            st.table(pd.DataFrame({
                "항목": ["4H 대추세", "선행 RSI_DK", "강물 기울기", "현재 1H 상태"], 
                "값": [trend_4h, str(round(latest_bar['RSI_DK'], 1)), str(round(latest_bar['River_Slope'], 3)), trend_1h]
            }))
        with c2:
            st.write("### 🧠 AI 중요도 분석")
            st.dataframe(feature_importance_df, hide_index=True)

        st.write("---")
        if up >= 60: st.success("🔥 **최종 결론:** 확인 후 **LONG 매수** 권장")
        elif down >= 60: st.error("❄️ **최종 결론:** 확인 후 **SHORT 매도** 권장")
        else: st.warning("⚠️ **최종 결론:** 방향성 불확실, **관망** 추천")

        st.write("---")
        st.plotly_chart(create_trading_chart(df, selected_tf_label), use_container_width=True)
        st.write("---")
        st.plotly_chart(create_feature_importance_chart(feature_importance_df), use_container_width=True)
