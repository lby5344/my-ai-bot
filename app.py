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
