"""
BIST 100 — 360° Yatırımcı Karar Destek REST API Sunucusu

Bu sunucu, n8n workflow üzerinden Telegram Bot'a bağlanarak
kullanıcıların Telegram'da hisse kodu sorduklarında 360° analiz raporu almasını sağlar.

Çalıştırma:
    pip install flask flask-cors
    python api_server.py

Endpoints:
    GET /api/report?ticker=THYAO    → Tam 360° analiz raporu (JSON)
    GET /api/health?ticker=THYAO    → Mali sağlık analizi
    GET /api/telegram?ticker=THYAO  → Telegram formatında metin rapor
    GET /api/ping                   → Sunucu durumu kontrolü
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask gerekli: pip install flask flask-cors")
    sys.exit(1)

from fundamental_engine import analyze_company_fundamentals
from custody_engine import analyze_custody_and_money_flow
from inference_engine import predict_single_ticker


app = Flask(__name__)
CORS(app)


def format_number(val, suffix="TL"):
    """Sayıları tam rakamlarıyla binlik ayracı ile gösterir."""
    if pd.isna(val) or val is None:
        return f"0 {suffix}"
    
    # Türkçe format: 1.234.567 TL
    formatted = f"{val:,.0f}".replace(",", ".")
    return f"{formatted} {suffix}"


def get_health_data(fund_data):
    """Mali sağlık analizini yapar."""
    import yfinance as yf
    import pandas as pd

    ticker_sym = fund_data.get("ticker", "")
    stock = yf.Ticker(ticker_sym)
    info = getattr(stock, "info", {}) or {}
    balance_sheet = getattr(stock, "quarterly_balance_sheet", pd.DataFrame())
    if balance_sheet is None or balance_sheet.empty:
        balance_sheet = getattr(stock, "balance_sheet", pd.DataFrame())

    cash = fund_data.get("cash_and_equivalents", 0)
    total_debt = fund_data.get("total_debt", 0)
    stockholder_equity = fund_data.get("stockholder_equity", 0)
    share_capital = fund_data.get("share_capital", 0)

    total_assets = 0
    current_assets = 0
    current_liabilities = 0

    if balance_sheet is not None and not balance_sheet.empty:
        latest_col = balance_sheet.columns[0]
        col_data = balance_sheet[latest_col]

        def _get(keys):
            for k in keys:
                if k in col_data.index and pd.notna(col_data[k]):
                    return float(col_data[k])
            return 0.0

        total_assets = _get(["Total Assets"])
        current_assets = _get(["Current Assets", "Total Current Assets"])
        current_liabilities = _get(["Current Liabilities", "Total Current Liabilities"])

    if total_assets == 0:
        total_assets = float(info.get("totalAssets", stockholder_equity * 1.5 if stockholder_equity > 0 else 10e9))
    if current_assets == 0:
        current_assets = cash
    if current_liabilities == 0:
        current_liabilities = total_debt * 0.4

    ebitda = float(info.get("ebitda", 0))
    total_revenue = float(info.get("totalRevenue", 1))
    ebitda_margin = (ebitda / total_revenue * 100) if total_revenue > 0 else 0.0

    net_working_capital = current_assets - current_liabilities
    nwc_ratio = (current_assets / (current_liabilities + 1e-7)) if current_liabilities > 0 else 5.0
    equity_to_assets = (stockholder_equity / (total_assets + 1e-7)) * 100 if total_assets > 0 else 0
    debt_to_assets = (total_debt / (total_assets + 1e-7)) * 100 if total_assets > 0 else 0

    signals = []

    # NWC sinyalleri
    if nwc_ratio < 1.0:
        signals.append("🚨 Dönen varlıklar KVB'yi karşılamıyor — likidite riski!")
    elif nwc_ratio < 1.5:
        signals.append("⚠️ Net işletme sermayesi düşük, nakit akışı izlenmeli")

    # Nakit sinyalleri
    cash_threshold = current_liabilities / 2.0
    if cash < cash_threshold:
        signals.append("🚨 Nakit yetersizliği — temerrüt riski!")

    # Mali yapı sinyalleri
    if equity_to_assets < 40:
        signals.append("🚨 Özkaynaktan ziyade borçla finanse edilmiş — kaldıraç riski")

    # Yapı durumu
    if equity_to_assets >= 60:
        structure = "GÜÇLÜ"
    elif equity_to_assets >= 40:
        structure = "DENGELİ"
    else:
        structure = "GÜÇSÜZ"

    return {
        "net_working_capital": net_working_capital,
        "nwc_ratio": round(nwc_ratio, 2),
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "equity_to_assets_pct": round(equity_to_assets, 1),
        "debt_to_assets_pct": round(debt_to_assets, 1),
        "structure": structure,
        "ebitda": ebitda,
        "ebitda_margin_pct": round(ebitda_margin, 1),
        "signals": signals
    }


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "BIST 360° API"
    })


@app.route("/api/predict", methods=["GET"])
def get_prediction():
    """Hızlı Yapay Zeka Model Tahminleri (GRU Dual + Volatilite)."""
    ticker = request.args.get("ticker", "THYAO").strip().upper()
    try:
        pred_res = predict_single_ticker(ticker)
        return jsonify(pred_res)
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/api/report", methods=["GET"])
def get_report():
    """Tam 360° analiz raporu (Yapay Zeka + Bilanço + Takas)."""
    ticker = request.args.get("ticker", "THYAO").strip().upper()

    try:
        # Merkezi çıkarım motoru üzerinden tam veri çek
        full_pred = predict_single_ticker(ticker)
        fund = full_pred.get("fundamental", {})
        if not fund or "ticker" not in fund:
            fund = analyze_company_fundamentals(ticker)
        
        cust = full_pred.get("custody", {})
        if not cust or "mfi_14" not in cust:
            cust = analyze_custody_and_money_flow(ticker)
            
        health = get_health_data(fund)

        report = {
            "ticker": full_pred.get("ticker", ticker),
            "timestamp": datetime.now().isoformat(),
            "price": full_pred.get("price"),
            "daily_change_pct": full_pred.get("daily_change_pct"),
            "volume": full_pred.get("volume"),
            "rsi_14": full_pred.get("rsi_14"),
            "realized_volatility_20": full_pred.get("realized_volatility_20"),
            "ai_predictions": full_pred.get("ai_predictions", {}),
            "fundamental": {
                "nakit": fund.get("cash_and_equivalents", 0),
                "toplam_borc": fund.get("total_debt", 0),
                "nakit_borc_orani": round(fund.get("cash_to_debt_ratio", 0), 2),
                "net_nakit": fund.get("net_cash_position", 0),
                "ozkaynak_gucu_pct": round(fund.get("equity_ratio_pct", 0), 1),
                "borc_ozkaynak_pct": round(fund.get("debt_to_equity_pct", 0), 1),
                "bedelsiz_potansiyeli_pct": round(fund.get("bonus_issue_potential_pct", 0), 1),
                "faiz_uyum_skoru": round(fund.get("interest_rate_score", 0), 0),
                "durum": fund.get("fundamental_status", "NÖTR")
            },
            "health": health,
            "custody": {
                "mfi_14": round(cust.get("mfi_14", 50.0), 1),
                "obv_trend": cust.get("obv_trend", "NÖTR"),
                "takas_skoru": round(cust.get("custody_proxy_score", 50.0), 0),
                "flow_status": cust.get("flow_status", "NÖTR")
            }
        }
        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/api/health", methods=["GET"])
def get_health():
    """Mali sağlık analizi."""
    ticker = request.args.get("ticker", "THYAO").strip().upper()

    try:
        fund = analyze_company_fundamentals(ticker)
        health = get_health_data(fund)
        return jsonify({"ticker": fund["ticker"], "health": health})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/telegram", methods=["GET"])
def get_telegram_report():
    """Telegram formatında metin rapor (Yapay Zeka + Bilanço + Takas)."""
    ticker = request.args.get("ticker", "THYAO").strip().upper()

    try:
        full_pred = predict_single_ticker(ticker)
        fund = full_pred.get("fundamental", {})
        if not fund or "ticker" not in fund:
            fund = analyze_company_fundamentals(ticker)
            
        cust = full_pred.get("custody", {})
        if not cust or "mfi_14" not in cust:
            cust = analyze_custody_and_money_flow(ticker)
            
        health = get_health_data(fund)
        ai = full_pred.get("ai_predictions", {})

        lines = []
        lines.append(f"📊 *BIST 360° RAPOR — {full_pred.get('ticker', ticker)}*")
        lines.append(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')} | Fiyat: {full_pred.get('price', 0):.2f} TL (%{full_pred.get('daily_change_pct', 0):+.2f})")
        lines.append("")

        if ai:
            lines.append("━━━ 🤖 YAPAY ZEKA TAHMİNLERİ ━━━")
            lines.append(f"• Karar Sinyali : {ai.get('conviction_signal', 'N/A')}")
            lines.append(f"• Yükseliş Olasılığı : %{ai.get('up_probability_pct', 50.0):.1f}")
            lines.append(f"• Beklenen Göreceli Alfa : %{ai.get('expected_excess_return_pct', 0.0):+.2f}")
            lines.append(f"• Tahmin Volatilite (20G): %{ai.get('predicted_volatility_pct', 0.0):.1f} ({ai.get('volatility_regime', 'Normal')})")
            lines.append("")

        lines.append("━━━ 💰 NAKİT & LİKİDİTE ━━━")
        lines.append(f"• Nakit: {format_number(fund.get('cash_and_equivalents', 0))}")
        lines.append(f"• Net İşletme Serm.: {format_number(health['net_working_capital'])}")
        lines.append(f"• Dönen Var./KVB: {health['nwc_ratio']:.2f}x")
        lines.append("")

        lines.append("━━━ 🏛️ MALİ YAPI ━━━")
        lines.append(f"• Özkaynak/Varlık: %{health['equity_to_assets_pct']}")
        lines.append(f"• Borç/Varlık: %{health['debt_to_assets_pct']}")
        lines.append(f"• Yapı: {health['structure']}")
        lines.append("")

        lines.append("━━━ 📈 KARLILIK ━━━")
        lines.append(f"• FAVÖK Marjı: %{health['ebitda_margin_pct']}")
        lines.append(f"• Nakit/Borç: {fund.get('cash_to_debt_ratio', 0):.2f}x")
        lines.append(f"• Bedelsiz Pot.: %{fund.get('bonus_issue_potential_pct', 0):,.1f}")
        lines.append(f"• Faiz Skoru: {fund.get('interest_rate_score', 0):.0f}/100")
        lines.append("")

        lines.append("━━━ 🏦 TAKAS ━━━")
        lines.append(f"• MFI-14: {cust.get('mfi_14', 50.0):.1f}")
        lines.append(f"• OBV: {cust.get('obv_trend', 'NÖTR')}")
        lines.append(f"• Takas Skoru: {cust.get('custody_proxy_score', 50.0):.0f}/100")
        lines.append("")

        if health["signals"]:
            lines.append("━━━ 🚨 UYARILAR ━━━")
            for sig in health["signals"]:
                lines.append(f"• {sig}")
            lines.append("")

        lines.append(f"🎯 {fund.get('fundamental_status', 'NÖTR')}")
        lines.append("_BIST 360° Yatırımcı Platformu_")

        report_text = "\n".join(lines)

        return jsonify({
            "ticker": full_pred.get("ticker", ticker),
            "report": report_text,
            "parse_mode": "Markdown"
        })

    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Tüm portföyü tarayıp sadece riskli olanları (NWC<1 veya Güçsüz Yapı) Telegram mesajı olarak döner."""
    tickers_param = request.args.get("tickers", "THYAO,EREGL")
    tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
    
    alerts = []
    
    for t in tickers:
        try:
            fund = analyze_company_fundamentals(t)
            health = get_health_data(fund)
            
            if health["nwc_ratio"] < 1.0 or health["structure"] == "GÜÇSÜZ":
                alerts.append(f"🚨 *{t}* 🚨\n"
                              f"• Dönen Varlık/KVB: {health['nwc_ratio']:.2f}x\n"
                              f"• Mali Yapı: {health['structure']}\n"
                              f"• Sinyaller: {', '.join(health['signals'])}")
        except:
            continue
            
    if not alerts:
        return jsonify({
            "report": "✅ Portföyünüzdeki şirketlerde şu an için acil bir likidite veya borç riski tespit edilmedi.",
            "parse_mode": "Markdown"
        })
        
    report_text = "⚠️ *PORTFÖY RİSK UYARISI* ⚠️\n\n" + "\n\n".join(alerts)
    
    return jsonify({
        "report": report_text,
        "parse_mode": "Markdown"
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BIST 360° REST API Sunucusu Başlatılıyor...")
    print("=" * 60)
    print("Endpoints:")
    print("  GET http://localhost:5050/api/ping")
    print("  GET http://localhost:5050/api/report?ticker=THYAO")
    print("  GET http://localhost:5050/api/health?ticker=THYAO")
    print("  GET http://localhost:5050/api/telegram?ticker=THYAO")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=True)
