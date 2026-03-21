import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests 

# [수정된 부분] 깃허브에 키를 노출하지 않고, 스트림릿 금고에서 가져옵니다!
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 2. 페이지 설정
st.set_page_config(page_title="AI 참모 v2.5 (Clean River v6)", page_icon="🧲", layout="wide")

# ... (아래 코드는 방금 전과 동일하게 그대로 두시면 됩니다!) ...
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], p, h1, h2, h3, h4, div { font-family: 'Nanum Gothic', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; color: #00ffbb; }
</style>
""", unsafe_allow_html=True)

# 3. [핵심] 직통망(REST API) AI 브리핑 함수
# 3. [핵심] 사용 가능한 AI 모델을 자동 검색해서 쏘는 브리핑 함수
def get_safe_ai_briefing(df, up, down):
    try:
        # 1단계: 구글 서버에 "지금 사용 가능한 모델 명단 다 내놔"라고 요청
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        list_response = requests.get(list_url).json()
        
        valid_model = None
        # 명단에서 글쓰기(generateContent)가 가능한 놈을 색출
        if 'models' in list_response:
            for m in list_response['models']:
                if 'supportedGenerationMethods' in m and 'generateContent' in m['supportedGenerationMethods']:
                    # flash나 pro라는 이름이 들어간 놈을 최우선으로 스카웃
                    if 'flash' in m['name'] or 'pro' in m['name']:
                        valid_model = m['name']
                        break
            
            # 만약 flash나 pro가 없으면, 그냥 글 쓸 줄 아는 아무 모델이나 멱살 잡고 끌고 옴
            if not valid_model:
                for m in list_response['models']:
                    if 'supportedGenerationMethods' in m and 'generateContent' in m['supportedGenerationMethods']:
                        valid_model = m['name']
                        break
                        
        if not valid_model:
            return f"🤖 구글 서버에 글쓰기 가능한 AI가 출근하지 않았습니다. (명단: {list_response})"

        # 2단계: 색출해낸 모델(valid_model)에게 바로 차트 분석 지시
        latest = df.iloc[-1]
        prompt = f"""
        당신은 암호화폐 전문 분석가 'AI 참모'입니다.
        - 현재 비트코인 가격: ${latest['Close']:,.1f}
        - AI 예측: 상승 확률 {up:.1f}%, 하락 확률 {down:.1f}%
        - RSI_DK: {latest['RSI_DK']:.1f}
        위 데이터를 바탕으로 현재 시장 상황과 투자 전략(진입/관망/익절)을 딱 3줄로 명확하게 브리핑하세요.
        """
        
        # 찾은 모델 이름으로 주소 자동 세팅
        url = f"https://generativelanguage.googleapis.com/v1beta/{valid_model}:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data).json()
        
        # 3단계: 대답 확인
        if 'candidates' in response and len(response['candidates']) > 0:
            ai_text = response['candidates'][0]['content']['parts'][0]['text']
            return f"(자동 선택된 모델: {valid_model})\n\n{ai_text}"
        elif 'error' in response:
            return f"🤖 서버가 거절했습니다. 사유: {response['error']['message']}"
        else:
            return f"🤖 알 수 없는 응답 구조입니다."
            
    except Exception as e:
        return f"🤖 통신망 오류 발생: {str(e)}"
# 4. 지표 계산기 (Clean River v6)
def add_indicators_v6(df, mtf_df=None):
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
    
    if mtf_df is not None:
        ema12_4 = mtf_df['Close'].ewm(span=12, adjust=False).mean()
        ema26_4 = mtf_df['Close'].ewm(span=26, adjust=False).mean()
        mtf_df['River_4H'] = (ema12_4 - ema26_4).ewm(span=5, adjust=False).mean()
        mtf_df['is4hBull'] = (mtf_df['River_4H'] >= 0).astype(int)
        mtf_merge = mtf_df[['Date', 'is4hBull']].rename(columns={'Date':'Date_4H', 'is4hBull':'Trend_4H'})
        df = pd.merge_asof(df.sort_values('Date'), mtf_merge.sort_values('Date_4H'), left_on='Date', right_on='Date_4H', direction='backward')
    return df

@st.cache_data(ttl=60)
def get_analysis_data(tf):
    ex = ccxt.kraken() 
    df = pd.DataFrame(ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    df_4h = pd.DataFrame(ex.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_4h['Date'] = pd.to_datetime(df_4h['Date'], unit='ms') + pd.Timedelta(hours=9)
    
    df_4h = add_indicators_v6(df_4h)
    df = add_indicators_v6(df, df_4h)
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI_DK', 'River_Slope', 'River_Accel']
    df_c = df.dropna()
    
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(df_c[features], df_c['Target'])
    prob = model.predict_proba(pd.DataFrame([df.iloc[-1][features]], columns=features))[0]
    
    return df, df.iloc[-1], prob[1]*100, prob[0]*100

# 5. 메인 화면 UI
st.title("🧲 AI 참모 v2.5 (Clean River v6)")
st.write("---")

col1, col2, col3 = st.columns(3)
tf_map = {"1시간": "1h", "4시간": "4h", "1일": "1d"}
sel_tf = None
if col1.button("🔄 1시간 분석"): sel_tf = "1h"
if col2.button("🔄 4시간 분석"): sel_tf = "4h"
if col3.button("🔄 1일 분석"): sel_tf = "1d"

if sel_tf:
    with st.spinner('차트를 스캔하고 AI 브리핑을 준비 중입니다...'):
        df, latest, up, down = get_analysis_data(sel_tf)
        
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 BTC 가격", f"${latest['Close']:,.1f}")
        m2.metric("상승 확률 (LONG)", f"{up:.1f}%")
        m3.metric("하락 확률 (SHORT)", f"{down:.1f}%")
        
        # AI 브리핑 출력부
        ai_msg = get_safe_ai_briefing(df, up, down)
        st.info(f"🤖 **제미나이 AI 실시간 브리핑**\n\n{ai_msg}")
        
        # 차트 그리기
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC', increasing_line_color='#00ffbb', decreasing_line_color='#ff0055'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Fortress_Price'], line=dict(color='#ffaa00', width=2, shape='hv'), name='🧲 성벽 라인'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_DK'], name='RSI_DK', line=dict(color='#00e676')), row=2, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(0,128,0,0.1)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(255,0,0,0.1)", line_width=0, row=2, col=1)
        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
