"""
BIST 100 Temel Bilanço Analitiği & Bedelsiz Potansiyeli Motoru

Bu modül:
1. Şirketlerin son çeyrek bilanço verilerini analiz eder.
2. Nakit ve Nakit Benzerleri / Toplam Borç Oranını (Cash-to-Debt Ratio) hesaplar.
3. Özkaynak Gücü ve Borçluluk oranını çıkarır.
4. Bedelsiz Sermaye Artırımı Potansiyelini (%) tespit eder.
5. TCMB Faiz Politikasına Göre "Nakit Pozisyonu Güçlü Şirket" Skoru üretir (0-100 Puan).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf


def analyze_company_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Belirtilen hisse senedinin son çeyrek bilanço ve finansal tablolarını analiz eder.
    """
    ticker_sym = ticker.upper() if ticker.endswith(".IS") else ticker.upper() + ".IS"
    stock = yf.Ticker(ticker_sym)

    info = getattr(stock, "info", {}) or {}
    balance_sheet = getattr(stock, "quarterly_balance_sheet", pd.DataFrame())
    if balance_sheet is None or balance_sheet.empty:
        balance_sheet = getattr(stock, "balance_sheet", pd.DataFrame())

    # Financial Fallbacks / Defaults
    cash_and_equivalents = 0.0
    total_debt = 0.0
    total_assets = 0.0
    stockholder_equity = 0.0
    share_capital = 0.0

    if balance_sheet is not None and not balance_sheet.empty:
        latest_col = balance_sheet.columns[0]
        col_data = balance_sheet[latest_col]

        def _get_val(keys):
            for k in keys:
                if k in col_data.index and pd.notna(col_data[k]):
                    return float(col_data[k])
            return 0.0

        cash_and_equivalents = _get_val(["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"])
        total_debt = _get_val(["Total Debt", "Total Liabilities Net Minority Interest", "Current Debt And Capital Lease Obligation"])
        total_assets = _get_val(["Total Assets"])
        stockholder_equity = _get_val(["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
        share_capital = _get_val(["Share Capital", "Common Stock"])

    # Fallbacks from yfinance info dictionary if balance sheet rows are missing
    if cash_and_equivalents == 0.0:
        cash_and_equivalents = float(info.get("totalCash", 0.0))
    if total_debt == 0.0:
        total_debt = float(info.get("totalDebt", 0.0))
    if total_assets == 0.0:
        total_assets = float(info.get("totalAssets", stockholder_equity * 1.5 if stockholder_equity > 0 else 10e9))
    if stockholder_equity == 0.0:
        stockholder_equity = float(info.get("bookValue", 10.0)) * float(info.get("sharesOutstanding", 1e8))
    if share_capital == 0.0:
        share_capital = float(info.get("sharesOutstanding", 1e8)) * 1.0

    # 1. Nakit ve Nakit Benzerleri / Borç Oranı
    cash_to_debt_ratio = (cash_and_equivalents / (total_debt + 1e-7)) if total_debt > 0 else 5.0
    net_cash_position = cash_and_equivalents - total_debt

    # 2. Özkaynak Gücü (%)
    equity_ratio_pct = (stockholder_equity / (total_assets + 1e-7)) * 100.0 if total_assets > 0 else 50.0
    debt_to_equity_pct = (total_debt / (stockholder_equity + 1e-7)) * 100.0 if stockholder_equity > 0 else 0.0

    # 3. Bedelsiz Sermaye Artırımı Potansiyeli (%)
    # (Özkaynaklar - Ödenmiş Sermaye) / Ödenmiş Sermaye * 100
    free_reserves = max(0.0, stockholder_equity - share_capital)
    bonus_issue_potential_pct = (free_reserves / (share_capital + 1e-7)) * 100.0 if share_capital > 0 else 0.0

    # 4. TCMB Yüksek Faiz Ortamı Uyum Skoru (0 - 100 Puan)
    # Yüksek faizde net nakdi borcundan fazla olan ve özkaynak oranı yüksek şirketler yüksek puan alır
    interest_rate_score = 50.0
    if cash_to_debt_ratio >= 1.5:
        interest_rate_score += 30.0
    elif cash_to_debt_ratio >= 1.0:
        interest_rate_score += 20.0
    elif cash_to_debt_ratio < 0.5:
        interest_rate_score -= 20.0

    if equity_ratio_pct >= 50.0:
        interest_rate_score += 20.0
    elif equity_ratio_pct < 25.0:
        interest_rate_score -= 15.0

    interest_rate_score = float(np.clip(interest_rate_score, 0.0, 100.0))

    # Değerlendirme Özeti
    if interest_rate_score >= 80.0:
        fundamental_status = "🟢 GÜÇLÜ TEMEL / YÜKSEK NAKİT FAYDASI"
    elif interest_rate_score >= 55.0:
        fundamental_status = "🟡 DENGELİ BİLANÇO"
    else:
        fundamental_status = "🔴 YÜKSEK BORÇLULUK RİSKİ"

    return {
        "ticker": ticker_sym,
        "cash_and_equivalents": cash_and_equivalents,
        "total_debt": total_debt,
        "net_cash_position": net_cash_position,
        "cash_to_debt_ratio": cash_to_debt_ratio,
        "equity_ratio_pct": equity_ratio_pct,
        "debt_to_equity_pct": debt_to_equity_pct,
        "stockholder_equity": stockholder_equity,
        "share_capital": share_capital,
        "bonus_issue_potential_pct": bonus_issue_potential_pct,
        "interest_rate_score": interest_rate_score,
        "fundamental_status": fundamental_status
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="THYAO.IS")
    args = parser.parse_args()

    res = analyze_company_fundamentals(args.ticker)
    print("=" * 80)
    print(f"📊 BİLANÇO & TEMEL ANALİZ RAPORU ({res['ticker']})")
    print("=" * 80)
    print(f"  • Nakit ve Nakit Benzerleri  : {res['cash_and_equivalents']:,.0f} TL")
    print(f"  • Toplam Borç              : {res['total_debt']:,.0f} TL")
    print(f"  • Net Nakit Pozisyonu       : {res['net_cash_position']:,.0f} TL")
    print(f"  • Nakit / Borç Oranı       : {res['cash_to_debt_ratio']:.2f}x")
    print(f"  • Özkaynak Güç Oranı       : %{res['equity_ratio_pct']:.1f}")
    print(f"  • Bedelsiz Sermaye Potansiyeli: %{res['bonus_issue_potential_pct']:,.1f}")
    print(f"  • Yüksek Faiz Uyum Skoru    : {res['interest_rate_score']:.0f} / 100")
    print(f"  • Durum                    : {res['fundamental_status']}")
    print("=" * 80)
