# 파일명: app.py (또는 main.py 등 회원님 깃허브에 덮어씌울 파이썬 파일)
# 이 코드는 Streamlit이 아닌 [Flask 웹 브레인 서버]입니다!
import os
import ccxt
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import pytz
app = Flask(__name__)
# ==============================================================================
# 🛠️ [1단계: 회원님 전용 매매 설정소 (여기를 입맛대로 고치세요!)]
# ==============================================================================
# 1. 거래소 세팅 (기본: 바이낸스 선물)
# 업비트는 ccxt.upbit() 로 나중에 고치셔도 됩니다.
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY", "여기에_바이낸스_API_키를_넣어주세요"),
    'secret': os.getenv("BINANCE_SECRET_KEY", "여기에_바이낸스_시크릿키를_넣어주세요"),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future' # 'spot'으로 바꾸면 현물, 'future'는 파생(선물)입니다.
    }
})
# 2. 거래 자금 세팅 (1회 타점 진입 시 비율)
TRADE_PERCENT_SIZE = 10  # 달러(USDT) 잔고의 몇 %를 베팅할 것인가? (기본 10%)
# 3. 텔레그램 봇 및 제미나이(Groq) 키 연동
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "123456789:알파벳토큰번호")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "텔레그램채팅방숫자번호")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "제미나이_그록_API키_넣으세요")
# ==============================================================================
def send_telegram(message):
    """텔레그램 스마트폰으로 톡을 쏘는 소음기 함수"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception:
        pass
# ==============================================================================
# 🎯 [2단계: 트레이딩뷰 직통 안테나 수신기 (Webhook 톨게이트)]
# 파인스크립트 머리의 신호(LONG/SHORT)를 받아 CCXT 손발을 움직이는 핵심 심장부!!
# ==============================================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    # 트레이딩뷰에서 넘어오는 데이터(JSON) 받기
    data = request.json
    if not data or "action" not in data:
        return jsonify({"error": "오류: 파인스크립트 신호가 아닙니다."}), 400
    
    action = data["action"].upper() # "LONG" 또는 "SHORT"
    ticker = data.get("ticker", "BTC/USDT") # 기본값 비트코인
    
    # 신호가 포착되면 즉시 텔레파시 발동
    send_telegram(f"🚨 [트레이딩뷰 v7.0 타점 포착!]\n명령: {action}\n종목: {ticker}\n전투 로봇 엔진 결제를 시작합니다.")
    
    try:
        # 1. 지갑 탈탈 털어서 달러(USDT) 잔고 확인
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['info']['availableBalance']) if 'availableBalance' in balance['info'] else float(balance['USDT']['free'])
        
        # 2. 내가 정한 비중(TRADE_PERCENT_SIZE) 만큼 전투 자금 분할 (자금 관리)
        trade_amount_usdt = usdt_balance * (TRADE_PERCENT_SIZE / 100.0)
        
        # 3. 현재 살 수 있는 코인 개수(Size) 정확히 계산
        ticker_info = exchange.fetch_ticker(ticker)
        current_price = ticker_info['last']
        amount_to_buy = trade_amount_usdt / current_price
        
        # 4. 🔥 실전 결제 타격 개시! (시장가 진입) 🔥
        if action == "LONG":
            order = exchange.create_market_buy_order(ticker, amount_to_buy)
        elif action == "SHORT":
            order = exchange.create_market_sell_order(ticker, amount_to_buy)
        else:
            return jsonify({"error": "알 수 없는 명령"}), 400
            
        # 성공하면 텔레그램으로 승전보 알림
        msg = f"✅ [진입 성.공.적.] {action}\n체결가: ${current_price:,.2f}\n투입량: {amount_to_buy:.4f} 코인\n(가자아아!! 🚀)"
        send_telegram(msg)
        return jsonify({"status": "Success", "message": msg}), 200
        
    except Exception as e:
        # 돈이 부족하거나 거래소 점검 등 에러가 나면 텔레그램에 바로 빨간 보고
        error_msg = f"❌ [주문 실패 비상!]\n이유: {str(e)}"
        send_telegram(error_msg)
        return jsonify({"error": str(e)}), 500
# ==============================================================================
# 📊 [3단계: 스마트폰으로 언제든 볼 수 있는 봇 통제실 (Dashboard)]
# 아까 쓰시던 '머신러닝.txt'의 예쁜 관상용 브리핑 화면을 이곳으로 옮겼습니다.
# ==============================================================================
@app.route("/")
def dashboard():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')
    
    current_price = "조회 중..."
    try:
        ticker_info = exchange.fetch_ticker('BTC/USDT')
        current_price = f"${ticker_info['last']:,.2f}"
    except Exception:
        pass
    
    # 제미나이(Groq) 참모의 초스피드 전황 보고
    ai_briefing = "AI 참모가 전쟁터 데이터를 분석 중입니다..."
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            prompt = f"현재 BTC 가격: {current_price}. 너는 최고 베테랑 비트코인 참모다. 전황을 2문장으로 보고해라."
            comp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
            ai_briefing = comp.choices[0].message.content
        except Exception as e:
            ai_briefing = f"참모 통신 두절 (원인: {str(e)})"
    
    # 딥 다크한 프로그래머용 무반사 스텔스 대시보드 화면 생성기
    html = f"""
    <html>
    <head>
        <title>🤖 리버 시스템 자동매매 통제실 (v5.0)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background-color: #0b0c10; color: #66fcf1; font-family: 'Malgun Gothic', Arial, sans-serif; padding: 30px; line-height: 1.6; margin: 0; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .box {{ background-color: #1f2833; padding: 25px; border-radius: 12px; border-left: 6px solid #45a29e; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            h1 {{ color: #ffffff; font-size: 24px; border-bottom: 1px solid #45a29e; padding-bottom: 10px; }}
            .highlight {{ color: #ffffff; font-weight: bold; font-size: 22px; }}
            .webhook-url {{ background-color: #000; padding: 10px; border-radius: 5px; color: #ff007f; font-family: monospace; font-size: 16px; margin-top: 10px; display: inline-block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 River System 100% 자동 결제 봇 (v5.0) 가동 중</h1>
            <div class="box">
                <p>💰 현재 비트코인(BTC) 레이더 가격: <span class="highlight">{current_price}</span></p>
                <p>🕒 서버 기준 시간 (KST): {now}</p>
                <p>🎯 매매 진입 비중 세팅: 지갑의 {TRADE_PERCENT_SIZE}% 자동 타격 설정됨</p>
            </div>
            
            <div class="box">
                <h3 style="color:#ffffff;">💬 6성급 AI 참모 (Llama-3 기반) 브리핑</h3>
                <p>{ai_briefing}</p>
            </div>
            
            <div class="box" style="border-left-color: #ff007f;">
                <h3 style="color:#ffffff;">🚨 트레이딩뷰 텔레파시 수신 안테나 주소 (Webhook)</h3>
                <p>트레이딩뷰 알림의 웹훅 URL 칸에 이 방의 주소를 꽂아주세요.</p>
                <div class="webhook-url">http://회원님-이지패널-도메인.com/webhook</div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)
if __name__ == "__main__":
    # 포트 3000번으로 문을 활짝 엽니다. 이지패널 기본 포트와 완벽하게 호환됩니다.
    app.run(host="0.0.0.0", port=3000)
