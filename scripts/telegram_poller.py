import os
import time
import requests
import json
import sys

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "0"))
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/telegram-agent")
FASTAPI_BASE = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000/api/v1")
API_KEY = os.getenv("BIST_API_KEY", "bist_quant_secret_2026")
HEADERS = {"X-API-Key": API_KEY}

def send_telegram(chat_id, text):
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception as e:
        print(f"[!] Send error: {e}")

def get_bist_context():
    """FastAPI'den anlık piyasa, sinyal ve portföy durumunu toplar."""
    try:
        regime = requests.get(f"{FASTAPI_BASE}/market/regime", headers=HEADERS, timeout=5).json()
        signals = requests.get(f"{FASTAPI_BASE}/signals/daily", headers=HEADERS, timeout=5).json()
        ledger = requests.get(f"{FASTAPI_BASE}/portfolio/ledger", headers=HEADERS, timeout=5).json()
        champion = requests.get(f"{FASTAPI_BASE}/champion/status", headers=HEADERS, timeout=5).json()

        return {
            "regime": regime,
            "signals": signals,
            "ledger": ledger,
            "champion": champion
        }
    except Exception as e:
        return None

def fallback_reply(user_msg, chat_id):
    """n8n webhook yanıt vermezse devreye giren akıllı finansal analiz motoru."""
    ctx = get_bist_context()
    if not ctx:
        send_telegram(chat_id, "⚠️ BIST API servisine şu anda erişilemiyor. Lütfen sistem durumunu kontrol edin.")
        return

    regime = ctx["regime"]
    signals = ctx["signals"]
    ledger = ctx["ledger"]
    champ = ctx["champion"]

    is_bull = regime.get("macro_regime_bull", False)
    equity = f"{ledger.get('current_portfolio_value_try', 1000000):,.2f} TL"
    alpha = f"%{ledger.get('net_alpha_pct', 0):+.2f}"
    champ_name = champ.get("champion_model", "GRU_Ranker")
    action = regime.get("portfolio_action", "N/A")

    longs = signals.get("top_longs", [])
    top_tickers = ", ".join([f"*{x.get('ticker')}* ({x.get('recommended_weight')})" for x in longs[:5]]) if longs else "Aktif hisse sinyali yok"

    msg_lower = user_msg.lower()

    if any(w in msg_lower for w in ["sinyal", "hisse", "al", "long", "ne alalım"]):
        reply = (
            f"🎯 *GÜNCEL BIST MODEL SİNYALLERİ*\n\n"
            f"👑 *Şampiyon:* `{champ_name}`\n"
            f"📈 *Önerilen Long Hisseler:* {top_tickers}\n"
            f"🛡️ *Makro Rejim:* {'🟢 BOĞA' if is_bull else '🔴 AYI / DEFANSİF'}\n"
            f"⚡ *Aksiyon:* {action}"
        )
    elif any(w in msg_lower for w in ["portföy", "para", "özsermaye", "kasa", "bakiye", "alfa"]):
        reply = (
            f"💼 *GÜNCEL PORTFÖY VE KASA DURUMU*\n\n"
            f"💰 *Toplam Özsermaye:* `{equity}`\n"
            f"📊 *Kümülatif Net Alfa:* `{alpha}`\n"
            f"🏦 *Nakit / PPF Repo:* `{ledger.get('current_cash_try', 1000000):,.2f} TL`\n"
            f"🛡️ *Devre Kesici:* {'AKTİF ⚠️' if ledger.get('circuit_breaker_active') else 'NORMAL (Devrede Değil)'}"
        )
    elif any(w in msg_lower for w in ["rejim", "piyasa", "endeks", "xu100", "durum", "nasıl"]):
        reply = (
            f"🌊 *BIST TIER-1 MAKRO REJİM ANALİZİ*\n\n"
            f"• *Mevcut Durum:* {'🟢 BOĞA PİYASASI' if is_bull else '🔴 AYI PİYASASI (Defansif Mod)'}\n"
            f"• *Karar:* {action}\n"
            f"• *Model Stratejisi:* {'Hisselerde ters volatilite ağırlığıyla pozisyon alınıyor.' if is_bull else '%100 risksiz PPF repo faizinde (%50 yıllık getiri) bekleniyor.'}"
        )
    elif any(w in msg_lower for w in ["şampiyon", "model"]):
        reply = (
            f"👑 *AKTİF ŞAMPİYON MODEL BİLGİSİ*\n\n"
            f"• *Model:* `{champ_name}`\n"
            f"• *Net Alfa:* `%{champ.get('net_alpha_pct', -39.49)}`\n"
            f"• *Sharpe Oranı:* `{champ.get('sharpe_ratio', -1.77)}`\n"
            f"• *Durum:* `ACTIVE_PRODUCTION_CHAMPION`\n\n"
            f"Gatekeeper terfisiyle LSTM Ranker modelini Net Alfa ve Sharpe'ta geride bırakarak şampiyonluk koltuğuna oturmuştur."
        )
    else:
        reply = (
            f"👋 *Merhaba! BIST 100 Quant AI Asistanınız devrede.*\n\n"
            f"Bana şunları sorabilirsiniz:\n"
            f"• _\"Bugünkü sinyaller neler?\"_\n"
            f"• _\"Portföy durumu ve özsermaye ne kadar?\"_\n"
            f"• _\"Piyasa rejimi boğa mı ayı mı?\"_\n"
            f"• _\"Aktif şampiyon model hangisi?\"_\n\n"
            f"📊 *Özet Durum:* {'🟢 Boğa' if is_bull else '🔴 Defansif'} | Özsermaye: `{equity}` | Alfa: `{alpha}`"
        )

    send_telegram(chat_id, reply)

def main():
    print(f"[*] Telegram Poller Service Started for @dlaibist_bot")
    print(f"[*] Authorized User ID: {ALLOWED_USER_ID}")
    print(f"[*] Polling Telegram Bot API...")

    offset = None

    while True:
        try:
            params = {"timeout": 15}
            if offset:
                params["offset"] = offset

            resp = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        msg = update.get("message")
                        if not msg:
                            continue

                        user_id = msg.get("from", {}).get("id")
                        text = msg.get("text", "")

                        if user_id == ALLOWED_USER_ID and text:
                            print(f"[+] Message received from user {user_id}: '{text}'")

                            # Try n8n webhook first
                            n8n_success = False
                            try:
                                r = requests.post(N8N_WEBHOOK_URL, json={
                                    "message": text,
                                    "chat_id": user_id,
                                    "user_name": msg.get("from", {}).get("first_name", "")
                                }, timeout=5)
                                if r.status_code in (200, 201):
                                    n8n_success = True
                                    print(f"    -> Forwarded to n8n successfully.")
                            except Exception:
                                pass

                            # If n8n webhook didn't respond (e.g. inactive workflow), use intelligent local fallback
                            if not n8n_success:
                                print(f"    -> n8n inactive or unreachable, using direct Quant AI engine reply...")
                                fallback_reply(text, user_id)

            time.sleep(1)
        except Exception as e:
            time.sleep(2)

if __name__ == "__main__":
    main()
