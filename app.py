import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.graph_objects as go

# 1. 페이지 설정 및 스타일
st.set_page_config(page_title="다중 타임프레임 AI 참모 v2.0", page_icon="🤖", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; }
div[data-testid="stMetricLabel"] { font-size: 1.1rem; color: #a0a0a0; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 다중 타임프레임 AI 참모 v2.0")
st.markdown("트레이딩뷰 차트 시간에 맞춰 아래 **버튼**을 클릭하세요!")

# 2. 데이터 분석 엔진
@st.cache_data(ttl=60)
def get_data_and_predict(tf):
    # 크라켄 거래소에서 데이터 호출 (ccxt 사용)
    exchange = ccxt.kraken()
    raw_data = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=1000)
    df = pd.DataFrame(raw_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)

    # 지표 계산 (pandas_ta)
    df.ta.atr(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.rsi(length=14, append=True)

    # River 로직 (EMA 기반 추세)
    df['River'] = ta.ema(df['MACDh_12_26_9'], length=5)
    df['River_Slope'] = df['River'] - df['River'].shift(1)
    df['River_Accel'] = df['River_Slope'] - df['River_Slope'].shift(1)

    # 학습 목표: 다음 캔들이 상승할지(1) 하락할지(0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = ['ATRr_14', 'MACDh_12_26_9', 'RSI_14', 'Volume', 'River', 'River_Slope', 'River_Accel']
    
    # 마지막 데이터(현재 진행중인 캔들) 분리
    live_data = df.iloc[-1:]
    df_train = df.dropna()

    X_train = df_train[feature_cols]
    y_train = df_train['Target']

    # AI 모델 학습 (랜덤 포레스트)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 중요도 추출
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        '지표 (Feature)': feature_cols,
        '중요도 (Importance)': importances
    }).sort_values(by='중요도 (Importance)', ascending=False)

    # 실시간 예측
    X_live = live_data[feature_cols]
    live_prob = model.predict_proba(X_live)[0]

    return live_data['Close'].iloc[0], live_data['Date'].iloc[0], live_prob[1]*100, live_prob[0]*100, feature_importance_df, live_data.to_dict('records')[0]

# 3. 사용자 인터페이스 (UI)
st.write("---")
col1, col2, col3 = st.columns(3)

tf_to_run = None
tf_name = ""

with col1:
    if st.button("🔄 1시간 (1h)", use_container_width=True):
        tf_to_run = '1h'; tf_name = '1시간'
with col2:
    if st.button("🔄 4시간 (4h)", use_container_width=True):
        tf_to_run = '4h'; tf_name = '4시간'
with col3:
    if st.button("🔄 1일 (1d)", use_container_width=True):
        tf_to_run = '1d'; tf_name = '1일'

if tf_to_run:
    with st.spinner(f'AI가 {tf_name} 차트 패턴을 실시간 분석 중입니다...'):
        current_price, latest_time, up_prob, down_prob, importance_df, latest_bar = get_data_and_predict(tf_to_run)

        st.write("---")
        st.subheader(f"업데이트: {latest_time.strftime('%Y-%m-%d %H:%M')} (KST)")

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric(label="💰 현재 BTC 가격", value=f"{current_price:,.2f} USDT")
        with summary_cols[1]:
            st.metric(label=f"📈 다음 {tf_name} 상승 확률", value=f"{up_prob:.1f}%")
        with summary_cols[2]:
            st.metric(label=f"📉 다음 {tf_name} 하락 확률", value=f"{down_prob:.1f}%")

        st.write("---")
        briefing_cols = st.columns([1.5, 2.5])

        with briefing_cols[0]:
            st.subheader("❄️ 최종 브리핑")
            if up_prob >= 60:
                msg = (f"🔥 **[강력 매수 권장]**\n\n현재 {tf_name} 상승 확률은 **{up_prob:.1f}%** 입니다.\n\n➔ {tf_name} 차트 🟢 확인 후 진입하세요.")
                st.success(msg)
            elif down_prob >= 60:
                msg = (f"❄️ **[강력 매도 권장]**\n\n현재 {tf_name} 하락 확률은 **{down_prob:.1f}%** 입니다.\n\n➔ {tf_name} 차트 🔴 확인 후 진입하세요.")
                st.error(msg)
            else:
                msg = (f"⚠️ **[관망 추천]**\n\n현재 방향성이 불확실합니다({up_prob:.1f}% vs {down_prob:.1f}%).\n\n휩소에 주의하며 진입을 보류하십시오.")
                st.warning(msg)

        with briefing_cols[1]:
            st.subheader(f"📊 {tf_name} 상세 지표 상태")
            rsi = latest_bar['RSI_14']
            rsi_status = "과매수" if rsi >= 70 else "과매도" if rsi <= 30 else "중립"
            macd = latest_bar['MACDh_12_26_9']
            macd_status = "상승추세" if macd >= 0 else "하락추세"
            
            status_data = pd.DataFrame({
                '지표': ['RSI (14)', 'MACD Hist', 'ATR (14)', 'Volume'],
                '현재 값': [f"{rsi:.2f}", f"{macd:.2f}", f"{latest_bar['ATRr_14']:.2f}", f"{latest_bar['Volume']:,.0f}"],
                '상태': [rsi_status, macd_status, "정상", "분석중"]
            })
            st.table(status_data)

        st.write("---")
        st.subheader("📊 Feature Importance (판단 근거)")
        fig = go.Figure(go.Bar(
            x=importance_df['중요도 (Importance)'],
            y=importance_df['지표 (Feature)'],
            orientation='h',
            marker=dict(color=importance_df['중요도 (Importance)'], colorscale='Blues_r')
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
