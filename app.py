# 파일명: app.py (이 코드를 깃허브에 통째로 복사해서 덮어쓰세요!)
import os
import ccxt
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import pytz

app = Flask(__name__)

# ==============================================================================
# 🛠️ [1단계: 회원님 전용 매매 설정소 (원하시는 대로 수정하세요!)]
# ==============================================================================

exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY", "여기에_바이낸스_API_키를_넣어주세요"),
    'secret': os.getenv("BINANCE_SECRET_KEY", "여기에_바이낸스_시크릿키를_넣어주세요"),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future' # 파생(선물) 거래용
    }
})

TRADE_PERCENT_SIZE = 10  # 달러(USDT) 잔고의 몇 %를 베팅할 것인가? (기본 10%)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "여기에_텔레그램_토큰")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "여기에_채팅방_아이디")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "여기에_제미나이_그록_API_키")

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception:
        pass

# ==============================================================================
# 🎯 [2단계: 트레이딩뷰 직통 안테나 수신기 (Webhook 톨게이트)]
# ==============================================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    if not data or "action" not in data:
        return jsonify({"error": "파인스크립트 신호가 아닙니다."}), 400
    
    action = data["action"].upper()
    ticker = data.get("ticker", "BTC/USDT")
    send_telegram(f"🚨 [트레이딩뷰 타점 포착!]\n명령: {action}\n종목: {ticker}\n전투 로봇 엔진 결제를 시작합니다.")
    
    try:
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['info']['availableBalance']) if 'availableBalance' in balance['info'] else float(balance['USDT']['free'])
        trade_amount_usdt = usdt_balance * (TRADE_PERCENT_SIZE / 100.0)
        
        ticker_info = exchange.fetch_ticker(ticker)
        current_price = ticker_info['last']
        amount_to_buy = trade_amount_usdt / current_price
        
        if action == "LONG":
            order = exchange.create_market_buy_order(ticker, amount_to_buy)
        elif action == "SHORT":
            order = exchange.create_market_sell_order(ticker, amount_to_buy)
        else:
            return jsonify({"error": "알 수 없는 명령"}), 400
            
        msg = f"✅ [진입 성.공.적.] {action}\n체결가: ${current_price:,.2f}\n투입량: {amount_to_buy:.4f} 코인\n(가자아아!! 🚀)"
        send_telegram(msg)
        return jsonify({"status": "Success", "message": msg}), 200
        
    except Exception as e:
        send_telegram(f"❌ [주문 실패 비상!]\n이유: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==============================================================================
# 📊 [3단계: 스마트폰으로 언제든 볼 수 있는 봇 통제실 (Dashboard)]
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
            
    html = f"""
    <html>
    <head>
        <title>🤖 리버 시스템 자동매매 통제실 (v5.0)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background-color: #0b0c10; color: #66fcf1; font-family: 'Malgun Gothic', sans-serif; padding: 30px; margin: 0; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .box {{ background-color: #1f2833; padding: 25px; border-radius: 12px; border-left: 6px solid #45a29e; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            h1 {{ color: #ffffff; border-bottom: 1px solid #45a29e; padding-bottom: 10px; }}
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
                <p>🎯 지갑의 {TRADE_PERCENT_SIZE}% 자동 타격 설정됨</p>
            </div>
            <div class="box">
                <h3 style="color:#ffffff;">💬 6성급 AI 참모 (Llama-3 기반) 브리핑</h3>
                <p>{ai_briefing}</p>
            </div>
            <div class="box" style="border-left-color: #ff007f;">
                <h3 style="color:#ffffff;">🚨 트레이딩뷰 텔레파시 수신소 주소 (Webhook URL)</h3>
                <p>트레이딩뷰 알림 웹훅 칸에 이 주소를 넣으세요:</p>
                <div class="webhook-url">https://여기에_회원님_이지패널_도메인주소를_적으세요/webhook</div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
