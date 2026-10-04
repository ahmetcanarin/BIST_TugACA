"""
BIST 100 — 360° Yatırımcı Karar Destek & Mali Analiz Platformu (Full Stack)

Bu uygulama:
1. LSTM vs GRU Volatilite Tahmin Sonuçlarını Karşılaştırmalı Gösterir
2. Temel Bilanço Analizi: Net İşletme Sermayesi, Nakit Durumu, Mali Yapı
3. Ticari/Finansal Borç Oranları, Sinyal Sistemi
4. FAVÖK Marjı, Ödenmiş Sermaye, Bedelsiz/Bedelli Sermaye Haberleri
5. Duygu Analizi (Türkçe BERT), Güncel Haberler
6. Takas Saklama Oranları & Para Akışı
7. Güncel TCMB Faiz Oranı & Makro Göstergeler
8. N8N Telegram Bot Entegrasyonu için REST API

Çalıştırma:
    streamlit run streamlit_app.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import streamlit as st
from datetime import datetime

from model_volatility import BISTVolatilityModel
from model_lstm_volatility import BISTVolatilityLSTMModel
from fundamental_engine import analyze_company_fundamentals
from custody_engine import analyze_custody_and_money_flow
from preprocessing import _compute_rsi
from inference_engine import get_inference_engine, predict_single_ticker, BISTInferenceEngine

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST 100 — 360° Yatırımcı Platformu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS - Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Light Background */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff !important;
    color: #0f172a !important;
}

/* Default Paragraphs, Spans and Labels */
p, span, label, div {
    color: #1e293b;
}

/* Main Title - Deep Modern Indigo Gradient */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #0284c7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.sub-title {
    font-size: 1.1rem;
    color: #475569 !important;
    margin-bottom: 1.5rem;
    font-weight: 500;
}

/* Streamlit Metrics - Crisp Slate & Bold Black */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04) !important;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] span {
    color: #475569 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.7rem !important;
}
[data-testid="stMetricValue"] div {
    color: #0f172a !important;
    font-weight: 800 !important;
}

/* Streamlit Captions - Readable Charcoal */
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}

/* Custom Metric Cards */
.metric-card {
    background: #ffffff !important;
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #2563eb !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    margin-bottom: 14px;
    transition: transform 0.2s, box-shadow 0.2s;
    color: #0f172a !important;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
}
.metric-card * {
    color: #0f172a !important;
}

/* Tabs - Clean Modern Style */
button[data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
}
button[data-baseweb="tab"] div, button[data-baseweb="tab"] span {
    color: #475569 !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 3px solid #2563eb !important;
    font-weight: 800 !important;
    background-color: #eff6ff !important;
}
button[data-baseweb="tab"][aria-selected="true"] div, button[data-baseweb="tab"][aria-selected="true"] span {
    color: #2563eb !important;
    font-weight: 800 !important;
}

/* Signal Badges (Readable Light Pastels with Strong Foreground) */
.signal-green {
    background-color: #dcfce7 !important;
    color: #15803d !important;
    border: 1px solid #86efac !important;
    padding: 8px 16px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    margin: 4px 2px;
}
.signal-yellow {
    background-color: #fef3c7 !important;
    color: #b45309 !important;
    border: 1px solid #fde68a !important;
    padding: 8px 16px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    margin: 4px 2px;
}
.signal-red {
    background-color: #fee2e2 !important;
    color: #b91c1c !important;
    border: 1px solid #fca5a5 !important;
    padding: 8px 16px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    margin: 4px 2px;
}

/* Section Headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a !important;
    border-bottom: 3px solid #2563eb;
    padding-bottom: 8px;
    margin-bottom: 20px;
    margin-top: 15px;
}

/* Risk, Safe & Warning Boxes */
.risk-box {
    background-color: #fef2f2 !important;
    border: 1px solid #f87171 !important;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #991b1b !important;
}
.risk-box * { color: #991b1b !important; }
.safe-box {
    background-color: #f0fdf4 !important;
    border: 1px solid #4ade80 !important;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #166534 !important;
}
.safe-box * { color: #166534 !important; }
.warn-box {
    background-color: #fffbeb !important;
    border: 1px solid #fbbf24 !important;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #92400e !important;
}
.warn-box * { color: #92400e !important; }

/* Tables - White Clean Style */
table, .stTable, [data-testid="stTable"] {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border-radius: 10px;
}
table th, [data-testid="stTable"] th {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-bottom: 2px solid #cbd5e1 !important;
}
table td, [data-testid="stTable"] td {
    color: #1e293b !important;
    font-size: 0.95rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* Model Comparison Table */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
}
.comparison-table th {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    border-bottom: 2px solid #cbd5e1;
}
.comparison-table td {
    padding: 12px 20px;
    text-align: center;
    border-bottom: 1px solid #e2e8f0;
    font-weight: 600;
    color: #1e293b !important;
    font-size: 0.95rem;
}
.comparison-table tr:nth-child(even) { background-color: #f8fafc; }
.comparison-table tr:hover { background-color: #f1f5f9; }
.winner-cell {
    background: #dcfce7 !important;
    font-weight: 800 !important;
    color: #15803d !important;
}

/* Sidebar Styling - Light Clean */
[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #0f172a !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #0f172a !important;
}

/* Inputs, Selectboxes and Dropdowns */
.stTextInput input, .stSelectbox [data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
.stSelectbox div[role="listbox"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
}
.stSelectbox li {
    color: #0f172a !important;
}
.stSelectbox li:hover {
    background-color: #eff6ff !important;
    color: #2563eb !important;
}

/* Alert Boxes Text Contrast */
.stAlert p {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Links */
a {
    color: #2563eb !important;
    text-decoration: underline;
    font-weight: 600;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #64748b !important;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 40px;
    padding: 20px 0;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "120d") -> pd.DataFrame:
    """Hisse senedi canlı verisi."""
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df.iloc[:, :6]
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    # Eksik veya seans dışı NaN satırlarını filtrele
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    for col in ["High", "Low", "Open"]:
        if col in df.columns and "Close" in df.columns:
            df[col] = df[col].fillna(df["Close"])

    return df.sort_values("Date").reset_index(drop=True)


def format_number(val, suffix="TL"):
    """Sayıları tam rakamlarıyla binlik ayracı ile gösterir."""
    if pd.isna(val) or val is None:
        return f"0 {suffix}"
    
    # Türkçe format: 1.234.567 TL
    formatted = f"{val:,.0f}".replace(",", ".")
    return f"{formatted} {suffix}"


def extract_live_features_for_ticker(df_stock: pd.DataFrame, feature_cols: list, seq_len: int = 30) -> torch.Tensor:
    """Merkezi çıkarım motoruna delege edilen dinamik öznitelik çıkarımı."""
    return BISTInferenceEngine.extract_features_for_ticker(df_stock, feature_cols=feature_cols, seq_len=seq_len)


def get_financial_health_analysis(fund_data: dict) -> dict:
    """
    Detaylı mali yapı analizi:
    - Net İşletme Sermayesi
    - Nakit Durumu (Nakit > KVB/2 ?)
    - Mali Yapı (Borç/Varlıklar, Özkaynaklar/Varlıklar)
    - Karlılık (FAVÖK Marjı)
    - Sinyal Sistemi (Tehlike Uyarıları)
    """
    ticker_sym = fund_data.get("ticker", "")
    stock = yf.Ticker(ticker_sym)
    info = getattr(stock, "info", {}) or {}
    balance_sheet = getattr(stock, "quarterly_balance_sheet", pd.DataFrame())
    if balance_sheet is None or balance_sheet.empty:
        balance_sheet = getattr(stock, "balance_sheet", pd.DataFrame())
    income_stmt = getattr(stock, "quarterly_income_stmt", pd.DataFrame())
    if income_stmt is None or income_stmt.empty:
        income_stmt = getattr(stock, "income_stmt", pd.DataFrame())

    # Bilanço Kalemleri
    cash = fund_data.get("cash_and_equivalents", 0)
    total_debt = fund_data.get("total_debt", 0)
    stockholder_equity = fund_data.get("stockholder_equity", 0)
    share_capital = fund_data.get("share_capital", 0)

    # Varlıklar (Balance Sheet)
    total_assets = 0
    current_assets = 0
    current_liabilities = 0
    non_current_assets = 0
    trade_receivables = 0
    trade_payables = 0
    financial_debt = 0

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
        current_liabilities = _get(["Current Liabilities", "Total Current Liabilities", "Current Debt And Capital Lease Obligation"])
        non_current_assets = _get(["Total Non Current Assets", "Net PPE", "Other Non Current Assets"])
        trade_receivables = _get(["Accounts Receivable", "Receivables", "Net Receivables"])
        trade_payables = _get(["Accounts Payable", "Payables And Accrued Expenses"])
        financial_debt = _get(["Total Debt", "Current Debt", "Long Term Debt And Capital Lease Obligation"])

    # Fallback
    if total_assets == 0:
        total_assets = float(info.get("totalAssets", stockholder_equity * 1.5 if stockholder_equity > 0 else 10e9))
    if current_assets == 0:
        current_assets = cash + trade_receivables
    if current_liabilities == 0:
        current_liabilities = total_debt * 0.4  # Varsayım
    if non_current_assets == 0:
        non_current_assets = total_assets - current_assets

    # FAVÖK / EBITDA
    ebitda = float(info.get("ebitda", 0))
    total_revenue = float(info.get("totalRevenue", 1))
    ebitda_margin = (ebitda / total_revenue * 100) if total_revenue > 0 else 0.0

    # ────────────────────────────────────────────────────────────────────
    # 1. NET İŞLETME SERMAYESİ
    # ────────────────────────────────────────────────────────────────────
    net_working_capital = current_assets - current_liabilities
    nwc_ratio = (current_assets / (current_liabilities + 1e-7)) if current_liabilities > 0 else 5.0

    nwc_signals = []
    if nwc_ratio >= 2.0:
        nwc_status = "🟢 GÜÇLÜ"
        nwc_detail = "Dönen varlıklar kısa vadeli borçların en az 2 katı — şirket likiditesi çok güçlü."
    elif nwc_ratio >= 1.5:
        nwc_status = "🟢 İYİ"
        nwc_detail = "Dönen varlıklar kısa vadeli borçları makul düzeyde karşılıyor."
    elif nwc_ratio >= 1.0:
        nwc_status = "🟡 DENGELİ / DİKKAT"
        nwc_detail = "Dönen varlıklar kısa vadeli borçları karşılıyor ama güvenlik marjı düşük."
        nwc_signals.append("⚠️ Net işletme sermayesi düşük, nakit akışı izlenmeli")
    else:
        nwc_status = "🔴 TEHLİKELİ"
        nwc_detail = "Dönen varlıklar kısa vadeli borçları karşılamıyor! Şirket duran varlık satışı veya bedelli sermaye artırımına gidebilir."
        nwc_signals.append("🚨 DÖNEN VARLIKLAR KVB'Yİ KARŞILAMIYOR — LİKİDİTE RİSKİ!")
        if non_current_assets > current_liabilities:
            nwc_signals.append("ℹ️ Duran varlık satışıyla borç kapatılabilir")
        else:
            nwc_signals.append("🚨 Duran varlıklar da yetersiz — BEDELLİ SERMAYE ARTIRIMI RİSKİ")

    # ────────────────────────────────────────────────────────────────────
    # 2. NAKİT DURUMU (Nakit > KVB / 2 ?)
    # ────────────────────────────────────────────────────────────────────
    cash_threshold = current_liabilities / 2.0
    cash_status_ok = cash >= cash_threshold

    if cash_status_ok:
        cash_status = "🟢 NAKİT GÜÇLÜ"
        cash_detail = f"Nakit ({format_number(cash)}) kısa vadeli borçların yarısından fazla."
    else:
        cash_status = "🔴 NAKİT YETERSİZ"
        cash_detail = f"Nakit ({format_number(cash)}) kısa vadeli borçların yarısının altında ({format_number(cash_threshold)})."
        nwc_signals.append("🚨 NAKİT YETERSİZLİĞİ — TEMERRÜT RİSKİ!")

    # ────────────────────────────────────────────────────────────────────
    # 3. TİCARİ vs FİNANSAL BORÇ ANALİZİ
    # ────────────────────────────────────────────────────────────────────
    trade_debt_ratio = (trade_payables / (total_debt + 1e-7)) * 100 if total_debt > 0 else 0
    financial_debt_ratio = 100 - trade_debt_ratio

    # Ticari alacakların borçlardaki payı
    receivable_ratio = (trade_receivables / (total_debt + 1e-7)) * 100 if total_debt > 0 else 0

    debt_signals = []
    if receivable_ratio > 40:
        debt_signals.append("🚨 BORÇLARIN %40'INDAN FAZLASI TİCARİ ALACAK — TAHSİLAT RİSKİ!")
        if trade_receivables > current_liabilities * 0.5:
            debt_signals.append("⚠️ Ticari alacaklar yüksek, senetli ise biraz daha iyi ama yine de dikkat!")

    # ────────────────────────────────────────────────────────────────────
    # 4. MALİ YAPI — BORÇ / ÖZKAYNAK DENGESİ
    # ────────────────────────────────────────────────────────────────────
    equity_to_assets = (stockholder_equity / (total_assets + 1e-7)) * 100 if total_assets > 0 else 0
    debt_to_assets = ((total_debt) / (total_assets + 1e-7)) * 100 if total_assets > 0 else 0

    if equity_to_assets >= 60:
        structure_status = "🟢 GÜÇLÜ MALİ YAPI"
        structure_detail = f"Varlıkların %{equity_to_assets:.0f}'ı özkaynakla karşılanıyor. Borcu az, güçlü şirket."
        structure_label = "güçlü"
    elif equity_to_assets >= 40:
        structure_status = "🟡 DENGELİ MALİ YAPI"
        structure_detail = f"Varlıkların %{equity_to_assets:.0f}'ı özkaynakla karşılanıyor. Dengeli borç/özkaynak yapısı."
        structure_label = "dengeli"
    else:
        structure_status = "🔴 GÜÇSÜZ MALİ YAPI"
        structure_detail = f"Varlıkların sadece %{equity_to_assets:.0f}'ı özkaynakla karşılanıyor. Borç ağırlıklı yapı — risk yüksek!"
        structure_label = "güçsüz"
        nwc_signals.append("🚨 ÖZKAYNAKtan ziyade BORÇLA finanse edilmiş — KALDIRAÇ RİSKİ")

    # ────────────────────────────────────────────────────────────────────
    # 5. FAVÖK MARJI & ÖDENMİŞ SERMAYE ANALİZİ
    # ────────────────────────────────────────────────────────────────────
    if ebitda_margin > 20:
        ebitda_status = "🟢 YÜKSEK FAVÖK MARJI"
    elif ebitda_margin > 10:
        ebitda_status = "🟡 ORTA FAVÖK MARJI"
    else:
        ebitda_status = "🔴 DÜŞÜK FAVÖK MARJI"

    # Ödenmiş Sermaye / Özkaynaklar oranı
    paid_in_ratio = (share_capital / (stockholder_equity + 1e-7)) * 100 if stockholder_equity > 0 else 100
    if paid_in_ratio < 30:
        paid_in_status = "🟢 MÜKEMMEL"
        paid_in_detail = "Ödenmiş sermaye özkaynaklarda küçük kalmış — Bedelsiz sermaye artırımı potansiyeli çok yüksek!"
    elif paid_in_ratio < 60:
        paid_in_status = "🟡 İYİ"
        paid_in_detail = "Ödenmiş sermaye dengeli — belirli bir bedelsiz potansiyeli mevcut."
    else:
        paid_in_status = "🔴 SINIRLI"
        paid_in_detail = "Ödenmiş sermaye özkaynaklarda yüksek yer kaplıyor — bedelsiz potansiyeli düşük."

    return {
        # Net İşletme Sermayesi
        "net_working_capital": net_working_capital,
        "nwc_ratio": nwc_ratio,
        "nwc_status": nwc_status,
        "nwc_detail": nwc_detail,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "non_current_assets": non_current_assets,
        # Nakit Durumu
        "cash": cash,
        "cash_status": cash_status,
        "cash_detail": cash_detail,
        "cash_threshold": cash_threshold,
        # Ticari / Finansal Borç
        "trade_receivables": trade_receivables,
        "trade_payables": trade_payables,
        "financial_debt": financial_debt,
        "trade_debt_ratio": trade_debt_ratio,
        "receivable_ratio": receivable_ratio,
        "debt_signals": debt_signals,
        # Mali Yapı
        "equity_to_assets": equity_to_assets,
        "debt_to_assets": debt_to_assets,
        "structure_status": structure_status,
        "structure_detail": structure_detail,
        "structure_label": structure_label,
        "total_assets": total_assets,
        # FAVÖK & Ödenmiş Sermaye
        "ebitda": ebitda,
        "total_revenue": total_revenue,
        "ebitda_margin": ebitda_margin,
        "ebitda_status": ebitda_status,
        "paid_in_ratio": paid_in_ratio,
        "paid_in_status": paid_in_status,
        "paid_in_detail": paid_in_detail,
        # Sinyaller
        "all_signals": nwc_signals + debt_signals,
    }


def load_comparison_results():
    """LSTM vs GRU karşılaştırma sonuçlarını yükler."""
    path = "models/lstm_vs_gru_comparison.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_telegram_report(ticker: str, fund_data: dict, health: dict, cust_data: dict) -> str:
    """Telegram için formatlanmış metin raporu üretir."""
    report = []
    report.append(f"📊 *BIST 360° RAPOR — {ticker}*")
    report.append(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    report.append("")

    report.append("━━━ 💰 NAKİT & LİKİDİTE ━━━")
    report.append(f"• Nakit: {format_number(health['cash'])}")
    report.append(f"• Net İşletme Sermayesi: {format_number(health['net_working_capital'])}")
    report.append(f"• Dönen Varlık/KVB Oranı: {health['nwc_ratio']:.2f}x")
    report.append(f"• Durum: {health['nwc_status']}")
    report.append("")

    report.append("━━━ 🏛️ MALİ YAPI ━━━")
    report.append(f"• Özkaynak/Varlık: %{health['equity_to_assets']:.1f}")
    report.append(f"• Borç/Varlık: %{health['debt_to_assets']:.1f}")
    report.append(f"• Durum: {health['structure_status']}")
    report.append("")

    report.append("━━━ 📈 KARLILIK ━━━")
    report.append(f"• FAVÖK Marjı: %{health['ebitda_margin']:.1f}")
    report.append(f"• Bedelsiz Potansiyeli: %{fund_data['bonus_issue_potential_pct']:,.1f}")
    report.append(f"• Faiz Uyum Skoru: {fund_data['interest_rate_score']:.0f}/100")
    report.append("")

    report.append("━━━ 🏦 TAKAS & PARA AKIŞI ━━━")
    report.append(f"• MFI-14: {cust_data['mfi_14']:.1f}")
    report.append(f"• OBV Trend: {cust_data['obv_trend']}")
    report.append(f"• Takas Skoru: {cust_data['custody_proxy_score']:.0f}/100")
    report.append("")

    if health["all_signals"]:
        report.append("━━━ 🚨 UYARILAR ━━━")
        for sig in health["all_signals"]:
            report.append(f"• {sig}")
        report.append("")

    report.append("_BIST 360° Yatırımcı Platformu_")
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 📊 BIST 360° Platform")

POPULAR_TICKERS = [
    "THYAO.IS", "GARAN.IS", "EREGL.IS", "KCHOL.IS", "BIMAS.IS",
    "SISE.IS", "TUPRS.IS", "ASELS.IS", "AKBNK.IS", "SAHOL.IS",
    "FROTO.IS", "EKGYO.IS", "PETKM.IS", "HEKTS.IS", "SASA.IS",
    "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS", "TCELL.IS",
    "PGSUS.IS", "TOASO.IS", "TTKOM.IS", "MGROS.IS", "CCOLA.IS",
]

selected_ticker = st.sidebar.selectbox("📌 Hisse Seçin:", POPULAR_TICKERS, index=0)
custom_ticker = st.sidebar.text_input("✏️ Özel Hisse Kodu:", value="", placeholder="Örn: CSTA").strip().upper()

if custom_ticker:
    ticker = custom_ticker if custom_ticker.endswith(".IS") else custom_ticker + ".IS"
else:
    ticker = selected_ticker

ticker_clean = ticker.replace(".IS", "")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ TCMB Makro Göstergeler")

# Güncel Faiz Oranı
try:
    from extraction_tcmb import get_tcmb_interest_rate_series
    tcmb_df = get_tcmb_interest_rate_series()
    if tcmb_df is not None and not tcmb_df.empty:
        current_rate = tcmb_df.iloc[-1]["rate"]
        st.sidebar.metric("TCMB Politika Faizi", f"%{current_rate:.1f}", delta="Güncel Oran")
    else:
        current_rate = 50.0
        st.sidebar.metric("TCMB Politika Faizi", f"%{current_rate:.1f}", delta="Varsayılan")
except Exception:
    current_rate = 50.0
    st.sidebar.metric("TCMB Politika Faizi", f"%{current_rate:.1f}", delta="Varsayılan")

st.sidebar.info(f"💡 **Faiz:** %{current_rate:.0f} → Yüksek faiz döneminde nakdi güçlü, borcu düşük şirketler avantajlı.")

# VIX
try:
    vix_df = yf.download("^VIX", period="5d", progress=False)
    if not vix_df.empty:
        vix_val = float(vix_df["Close"].iloc[-1]) if not isinstance(vix_df.columns, pd.MultiIndex) else float(vix_df[("Close", "^VIX")].iloc[-1])
        st.sidebar.metric("VIX Korku Endeksi", f"{vix_val:.1f}", delta="Sakin" if vix_val < 25 else "Yüksek Korku")
except Exception:
    vix_val = 19.5

# USD/TRY
try:
    usd_df = yf.download("USDTRY=X", period="5d", progress=False)
    if not usd_df.empty:
        usd_val = float(usd_df["Close"].iloc[-1]) if not isinstance(usd_df.columns, pd.MultiIndex) else float(usd_df[("Close", "USDTRY=X")].iloc[-1])
        st.sidebar.metric("USD/TRY Kuru", f"₺{usd_val:.2f}")
except Exception:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 N8N Telegram Bot")
st.sidebar.code(f"http://localhost:8501/api/report?ticker={ticker_clean}", language="text")
st.sidebar.caption("Bu URL'yi n8n HTTP Request node'una ekleyin")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<div class='main-title'>📊 BIST 100 — 360° Yatırımcı Karar Destek Platformu</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>Hisse: <b>{ticker}</b> ({ticker_clean}) | LSTM & GRU Volatilite · Temel Analiz · Mali Yapı · Takas · Duygu Analizi</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH LIVE DATA
# ─────────────────────────────────────────────────────────────────────────────
df_stock = fetch_stock_data(ticker)

if df_stock.empty:
    st.error(f"❌ '{ticker}' kodu için veri bulunamadı. Lütfen sembolü kontrol edin.")
    st.stop()

# Fiyatı geçerli son satırları al
valid_df = df_stock.dropna(subset=["Close"]) if "Close" in df_stock.columns else df_stock
if valid_df.empty:
    valid_df = df_stock

latest_row = valid_df.iloc[-1]
prev_row = valid_df.iloc[-2] if len(valid_df) > 1 else latest_row

close_price = float(latest_row["Close"]) if pd.notna(latest_row.get("Close")) else 0.0
prev_close = float(prev_row["Close"]) if pd.notna(prev_row.get("Close")) and float(prev_row["Close"]) > 0 else close_price
daily_change = ((close_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
volume = int(latest_row["Volume"]) if pd.notna(latest_row.get("Volume")) else 0

high_val = float(latest_row["High"]) if pd.notna(latest_row.get("High")) else close_price
low_val = float(latest_row["Low"]) if pd.notna(latest_row.get("Low")) else close_price

# Top Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💰 Son Fiyat", f"{close_price:.2f} TL", f"%{daily_change:+.2f}")
with col2:
    st.metric("📊 Hacim", f"{volume:,}")
with col3:
    st.metric("📈 Yüksek / Düşük", f"{high_val:.2f} / {low_val:.2f}")
with col4:
    rsi_val = _compute_rsi(df_stock["Close"], window=14).iloc[-1]
    rsi_label = "Aşırı Alım" if rsi_val > 70 else ("Aşırı Satım" if rsi_val < 30 else "Dengeli")
    st.metric("📉 RSI (14)", f"{rsi_val:.1f}", rsi_label)
with col5:
    log_ret = np.log(df_stock["Close"] / df_stock["Close"].shift(1)).fillna(0)
    hist_vol = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    st.metric("🌊 Volatilite (20G)", f"%{hist_vol:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_meta, tab_model, tab_fund, tab_health, tab_custody, tab_news, tab_portfolio, tab_telegram = st.tabs([
    "🎯 Meta-Model (Yapay Zeka Kararı)",
    "🤖 LSTM vs GRU Sonuçları",
    "🏢 Temel Bilanço",
    "🔍 Mali Sağlık",
    "🏦 Takas & Para",
    "📰 Haberler",
    "💼 Portföy Yönetimi",
    "📱 Telegram API"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0: META-MODEL (YAPAY ZEKA NİHAİ KARARI)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_meta:
    st.markdown("<div class='section-header'>🎯 Meta-Model: Bütünleşik Karar Mekanizması</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Bu sekme; Temel Bilanço, Teknik Volatilite, Takas Akışı ve Duygu Analizi sonuçlarını 
    tek bir **Yapay Zeka Meta-Modeli** üzerinde birleştirerek nihai bir karar üretir.
    """)
    
    # Meta Skor Hesaplama (Örnek Ağırlıklandırma Sistemi)
    # Gerçek projede burada XGBoost modeli devrede olur.
    if 'health' not in dir():
        fund_data = analyze_company_fundamentals(ticker)
        health = get_financial_health_analysis(fund_data)
        cust_data = analyze_custody_and_money_flow(ticker)
        
    meta_score = 50 # Base
    
    # Bilanço Etkisi (Max +20, Min -20)
    if health["nwc_ratio"] > 1.5: meta_score += 10
    elif health["nwc_ratio"] < 1: meta_score -= 15
    if health["equity_to_assets"] > 50: meta_score += 10
    elif health["equity_to_assets"] < 30: meta_score -= 15
    
    # Takas Etkisi (Max +15, Min -15)
    cust_proxy = cust_data.get("custody_proxy_score", 50)
    meta_score += (cust_proxy - 50) * 0.3
    
    # Teknik Etkisi (Max +15, Min -15)
    if rsi_val > 70: meta_score -= 10
    elif rsi_val < 30: meta_score += 10
    
    meta_score = max(0, min(100, meta_score))
    
    if meta_score >= 75:
        decision = "🟢 GÜÇLÜ AL"
        m_color = "#22c55e"
    elif meta_score >= 60:
        decision = "🟢 AL / BİRİKTİR"
        m_color = "#10b981"
    elif meta_score >= 40:
        decision = "🟡 TUT / İZLE"
        m_color = "#f59e0b"
    elif meta_score >= 25:
        decision = "🔴 AZALT"
        m_color = "#ef4444"
    else:
        decision = "🔴 GÜÇLÜ SAT (RİSKLİ)"
        m_color = "#dc2626"
        
    st.markdown(f"""
    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border-radius: 16px; border: 2px solid #e2e8f0; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); margin: 20px 0;'>
        <h2 style='color: #475569; font-weight: 700; margin-bottom: 10px;'>Yapay Zeka Nihai Sinyali</h2>
        <h1 style='color: {m_color}; font-size: 3.2rem; font-weight: 800; margin: 0;'>{decision}</h1>
        <h3 style='color: #0f172a; margin-top: 15px; font-weight: 700;'>Meta-Skor: {meta_score:.1f} / 100</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("Bilanço Puanı", f"{health['equity_to_assets']:.1f}/100", "Bilanço Sağlamlığı")
    col_w2.metric("Takas Puanı", f"{cust_proxy:.1f}/100", "Kurum İlgisi")
    col_w3.metric("Teknik Puan (Ters İndikatör)", f"{100-rsi_val:.1f}/100", "Aşırı Alım/Satım")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: LSTM vs GRU MODEL KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("<div class='section-header'>🤖 LSTM vs GRU Volatilite Tahmin Modeli Karşılaştırması</div>", unsafe_allow_html=True)

    comparison = load_comparison_results()

    if comparison:
        gru = comparison["gru"]
        lstm = comparison["lstm"]
        data_info = comparison["data_info"]

        # Veri Bilgisi
        di_cols = st.columns(4)
        with di_cols[0]:
            st.metric("Train Örnek", f"{data_info['train_size']:,}")
        with di_cols[1]:
            st.metric("Val Örnek", f"{data_info['val_size']:,}")
        with di_cols[2]:
            st.metric("Test Örnek", f"{data_info['test_size']:,}")
        with di_cols[3]:
            st.metric("Özellik Sayısı", f"{data_info['num_features']}")

        st.markdown("---")

        # Karşılaştırma Tablosu
        def winner_class(gru_v, lstm_v, higher_better=True):
            if higher_better:
                return ("winner-cell", "") if gru_v > lstm_v else (("", "winner-cell") if lstm_v > gru_v else ("", ""))
            else:
                return ("winner-cell", "") if gru_v < lstm_v else (("", "winner-cell") if lstm_v < gru_v else ("", ""))

        metrics_data = [
            ("Volatilite R² Skoru", gru["test_r2"], lstm["test_r2"], True),
            ("Ortalama Mutlak Hata (MAE %)", gru["test_mae"], lstm["test_mae"], False),
            ("Kök Ort. Kare Hata (RMSE %)", gru["test_rmse"], lstm["test_rmse"], False),
            ("Rejim İsabet Oranı (%)", gru["test_regime_accuracy"], lstm["test_regime_accuracy"], True),
            ("En İyi Val Loss", gru["best_val_loss"], lstm["best_val_loss"], False),
        ]

        table_html = "<table class='comparison-table'>"
        table_html += "<tr><th>📊 Metrik</th><th>GRU</th><th>LSTM</th><th>🏆 Kazanan</th></tr>"

        for label, gru_v, lstm_v, higher_better in metrics_data:
            g_cls, l_cls = winner_class(gru_v, lstm_v, higher_better)
            if higher_better:
                winner = "🟢 GRU" if gru_v > lstm_v else ("🟢 LSTM" if lstm_v > gru_v else "⚪ Eşit")
            else:
                winner = "🟢 GRU" if gru_v < lstm_v else ("🟢 LSTM" if lstm_v < gru_v else "⚪ Eşit")

            table_html += f"<tr><td style='text-align:left;font-weight:600;'>{label}</td>"
            table_html += f"<td class='{g_cls}'>{gru_v}</td>"
            table_html += f"<td class='{l_cls}'>{lstm_v}</td>"
            table_html += f"<td>{winner}</td></tr>"

        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("---")

        # Eğitim Eğrileri
        st.markdown("#### 📈 Eğitim Eğrileri (Loss & R²)")
        col_loss, col_r2 = st.columns(2)

        with col_loss:
            gru_hist = pd.DataFrame(gru["train_history"])
            lstm_hist = pd.DataFrame(lstm["train_history"])
            loss_df = pd.DataFrame({
                "Epoch": gru_hist["epoch"],
                "GRU Val Loss": gru_hist["val_loss"],
                "LSTM Val Loss": lstm_hist["val_loss"]
            }).set_index("Epoch")
            st.line_chart(loss_df, use_container_width=True)
            st.caption("Validation Loss Eğrisi (Düşük = Daha İyi)")

        with col_r2:
            r2_df = pd.DataFrame({
                "Epoch": gru_hist["epoch"],
                "GRU Val R²": gru_hist["val_r2"],
                "LSTM Val R²": lstm_hist["val_r2"]
            }).set_index("Epoch")
            st.line_chart(r2_df, use_container_width=True)
            st.caption("Validation R² Eğrisi (Yüksek = Daha İyi)")

        # Rejim Karşılaştırması
        st.markdown("#### 🎯 Rejim Sınıflandırma Detayları")
        regime_col1, regime_col2 = st.columns(2)

        with regime_col1:
            st.markdown("**GRU Modeli:**")
            if "classification_report" in gru:
                cls_rpt = gru["classification_report"]
                rpt_data = []
                for regime_name in ["Sıkışma (Düşük)", "Normal", "Patlama (Yüksek)"]:
                    if regime_name in cls_rpt:
                        r = cls_rpt[regime_name]
                        rpt_data.append({
                            "Rejim": regime_name,
                            "Precision": f"%{r['precision']*100:.1f}",
                            "Recall": f"%{r['recall']*100:.1f}",
                            "F1-Score": f"{r['f1-score']:.3f}",
                            "Destek": int(r['support'])
                        })
                st.table(pd.DataFrame(rpt_data).set_index("Rejim"))

        with regime_col2:
            st.markdown("**LSTM Modeli:**")
            if "classification_report" in lstm:
                cls_rpt = lstm["classification_report"]
                rpt_data = []
                for regime_name in ["Sıkışma (Düşük)", "Normal", "Patlama (Yüksek)"]:
                    if regime_name in cls_rpt:
                        r = cls_rpt[regime_name]
                        rpt_data.append({
                            "Rejim": regime_name,
                            "Precision": f"%{r['precision']*100:.1f}",
                            "Recall": f"%{r['recall']*100:.1f}",
                            "F1-Score": f"{r['f1-score']:.3f}",
                            "Destek": int(r['support'])
                        })
                st.table(pd.DataFrame(rpt_data).set_index("Rejim"))

    else:
        st.warning("⚠️ LSTM vs GRU karşılaştırma sonuçları henüz mevcut değil.")
        st.info("Karşılaştırmayı çalıştırmak için terminalde şu komutu kullanın:")
        st.code("python train_comparison.py", language="bash")

    # Canlı Model Tahminleri (Mevcut ağırlıklardan)
    st.markdown("---")
    st.markdown("#### 🔮 Canlı Volatilite Tahminleri")

    live_cols = st.columns(3)
    with live_cols[0]:
        st.metric("Gerçekleşen Volatilite (20G)", f"%{hist_vol:.1f}")

    # GRU Canlı Tahmini
    gru_pred_vol = hist_vol  # fallback
    gru_regime = "Normal"
    gru_path = "models/bist_volatility_gru.pt" if os.path.exists("models/bist_volatility_gru.pt") else "models/bist_volatility_model.pt"
    if os.path.exists(gru_path):
        try:
            device = torch.device("cpu")
            ckpt = torch.load(gru_path, map_location=device, weights_only=False)
            nf = ckpt.get("num_features", 34)
            cols = ckpt.get("feature_cols", [])
            m = BISTVolatilityModel(num_features=nf)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            with torch.no_grad():
                x_live = extract_live_features_for_ticker(df_stock, feature_cols=cols, seq_len=30)
                pv, pr = m(x_live)
                gru_pred_vol = float(pv.item()) if pv.item() > 0 else hist_vol
                ri = int(pr.argmax(dim=1).item())
                gru_regime = ["🟢 Sıkışma", "🟡 Normal", "🔴 Patlama"][ri]
        except Exception:
            pass

    with live_cols[1]:
        st.metric("GRU Tahmin Volatilite", f"%{gru_pred_vol:.1f}", f"{gru_pred_vol - hist_vol:+.1f}%")
        st.caption(f"Rejim: {gru_regime}")

    # LSTM Canlı Tahmini
    lstm_pred_vol = hist_vol
    lstm_regime = "Normal"
    lstm_path = "models/bist_volatility_lstm.pt"
    if os.path.exists(lstm_path):
        try:
            device = torch.device("cpu")
            ckpt = torch.load(lstm_path, map_location=device, weights_only=False)
            nf = ckpt.get("num_features", 34)
            cols = ckpt.get("feature_cols", [])
            m = BISTVolatilityLSTMModel(num_features=nf)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            with torch.no_grad():
                x_live = extract_live_features_for_ticker(df_stock, feature_cols=cols, seq_len=30)
                pv, pr = m(x_live)
                lstm_pred_vol = float(pv.item()) if pv.item() > 0 else hist_vol
                ri = int(pr.argmax(dim=1).item())
                lstm_regime = ["🟢 Sıkışma", "🟡 Normal", "🔴 Patlama"][ri]
        except Exception:
            pass

    with live_cols[2]:
        st.metric("LSTM Tahmin Volatilite", f"%{lstm_pred_vol:.1f}", f"{lstm_pred_vol - hist_vol:+.1f}%")
        st.caption(f"Rejim: {lstm_regime}")

    # Fiyat Grafiği
    st.line_chart(df_stock.set_index("Date")["Close"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TEMEL BİLANÇO ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    st.markdown("<div class='section-header'>🏢 Temel Bilanço Analizi & Bedelsiz Sermaye Potansiyeli</div>", unsafe_allow_html=True)

    fund_data = analyze_company_fundamentals(ticker)

    f_cols = st.columns(4)
    with f_cols[0]:
        st.metric("💵 Nakit / Borç Oranı", f"{fund_data['cash_to_debt_ratio']:.2f}x")
    with f_cols[1]:
        st.metric("💰 Net Nakit Pozisyonu", format_number(fund_data['net_cash_position']))
    with f_cols[2]:
        st.metric("🏛️ Özkaynak Güç Oranı", f"%{fund_data['equity_ratio_pct']:.1f}")
    with f_cols[3]:
        st.metric("🎁 Bedelsiz Potansiyeli", f"%{fund_data['bonus_issue_potential_pct']:,.1f}")

    st.markdown("---")

    # Faiz Uyum Skoru
    st.markdown(f"### 🎯 TCMB Yüksek Faiz Uyum Skoru: **{fund_data['interest_rate_score']:.0f} / 100**")
    score_color = "🟢" if fund_data['interest_rate_score'] >= 80 else ("🟡" if fund_data['interest_rate_score'] >= 55 else "🔴")
    st.markdown(f"**Durum:** {fund_data['fundamental_status']}")
    st.progress(int(fund_data['interest_rate_score']))

    st.markdown("---")

    # Detaylı Bilanço Tablosu
    balance_data = {
        "Kalem": [
            "💵 Nakit ve Nakit Benzerleri",
            "💳 Toplam Borç Yükü",
            "🏛️ Toplam Özkaynaklar",
            "📋 Ödenmiş Sermaye",
            "📊 Borç / Özkaynak Oranı"
        ],
        "Değer": [
            format_number(fund_data['cash_and_equivalents']),
            format_number(fund_data['total_debt']),
            format_number(fund_data['stockholder_equity']),
            format_number(fund_data['share_capital']),
            f"%{fund_data['debt_to_equity_pct']:.1f}"
        ]
    }
    st.table(pd.DataFrame(balance_data).set_index("Kalem"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: 360° MALİ SAĞLIK & SİNYAL SİSTEMİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_health:
    st.markdown("<div class='section-header'>🔍 360° Mali Sağlık Analizi & Sinyal Sistemi</div>", unsafe_allow_html=True)

    if 'fund_data' not in dir():
        fund_data = analyze_company_fundamentals(ticker)

    health = get_financial_health_analysis(fund_data)

    # ──── SINYAL PANELİ ────
    if health["all_signals"]:
        st.markdown("<div class='risk-box'>", unsafe_allow_html=True)
        st.markdown("### 🚨 AKTİF UYARI SİNYALLERİ")
        for sig in health["all_signals"]:
            st.markdown(f"**{sig}**")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='safe-box'>", unsafe_allow_html=True)
        st.markdown("### ✅ AKTİF UYARI YOK — Mali yapı sağlıklı görünüyor")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ──── 1. NET İŞLETME SERMAYESİ ────
    st.markdown("### 1️⃣ Net İşletme Sermayesi")

    nwc_cols = st.columns(4)
    with nwc_cols[0]:
        st.metric("Dönen Varlıklar", format_number(health["current_assets"]))
    with nwc_cols[1]:
        st.metric("Kısa Vadeli Borçlar", format_number(health["current_liabilities"]))
    with nwc_cols[2]:
        st.metric("Net İşletme Sermayesi", format_number(health["net_working_capital"]))
    with nwc_cols[3]:
        st.metric("Dönen Varlık / KVB", f"{health['nwc_ratio']:.2f}x")

    css_class = "safe-box" if "GÜÇLÜ" in health["nwc_status"] or "İYİ" in health["nwc_status"] else ("warn-box" if "DENGELİ" in health["nwc_status"] else "risk-box")
    st.markdown(f"<div class='{css_class}'><b>{health['nwc_status']}</b> — {health['nwc_detail']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ──── 2. NAKİT DURUMU ────
    st.markdown("### 2️⃣ Nakit Durumu")

    cash_cols = st.columns(3)
    with cash_cols[0]:
        st.metric("Nakit Miktarı", format_number(health["cash"]))
    with cash_cols[1]:
        st.metric("Eşik (KVB / 2)", format_number(health["cash_threshold"]))
    with cash_cols[2]:
        st.metric("Durum", health["cash_status"])

    css_class = "safe-box" if "GÜÇLÜ" in health["cash_status"] else "risk-box"
    st.markdown(f"<div class='{css_class}'>{health['cash_detail']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ──── 3. MALİ YAPI ────
    st.markdown("### 3️⃣ Mali Yapı (Borç / Özkaynak Dengesi)")

    struct_cols = st.columns(4)
    with struct_cols[0]:
        st.metric("Toplam Varlıklar", format_number(health["total_assets"]))
    with struct_cols[1]:
        st.metric("Özkaynak / Varlık", f"%{health['equity_to_assets']:.1f}")
    with struct_cols[2]:
        st.metric("Borç / Varlık", f"%{health['debt_to_assets']:.1f}")
    with struct_cols[3]:
        st.metric("Mali Yapı", health["structure_label"].upper())

    css_class = "safe-box" if health["structure_label"] == "güçlü" else ("warn-box" if health["structure_label"] == "dengeli" else "risk-box")
    st.markdown(f"<div class='{css_class}'><b>{health['structure_status']}</b> — {health['structure_detail']}</div>", unsafe_allow_html=True)

    # Mali yapı açıklama
    st.markdown("""
    | Özkaynak / Varlık Oranı | Mali Yapı Durumu |
    |:-:|:-:|
    | ≥ %60 | 🟢 **GÜÇLÜ** — Özkaynağı çok, borcu az |
    | %40 - %60 | 🟡 **DENGELİ** — Makul borç/özkaynak |
    | < %40 | 🔴 **GÜÇSÜZ** — Borç ağırlıklı, kaldıraç riski |
    """)

    st.markdown("---")

    # ──── 4. KARLILIK & FAVÖK ────
    st.markdown("### 4️⃣ Karlılık (FAVÖK Marjı)")

    ebitda_cols = st.columns(4)
    with ebitda_cols[0]:
        st.metric("FAVÖK (EBITDA)", format_number(health["ebitda"]))
    with ebitda_cols[1]:
        st.metric("Toplam Gelir", format_number(health["total_revenue"]))
    with ebitda_cols[2]:
        st.metric("FAVÖK Marjı", f"%{health['ebitda_margin']:.1f}")
    with ebitda_cols[3]:
        st.metric("Durum", health["ebitda_status"])

    st.markdown("---")

    # ──── 5. ÖDENMİŞ SERMAYE & BEDELSİZ POTANSİYELİ ────
    st.markdown("### 5️⃣ Ödenmiş Sermaye & Bedelsiz Potansiyeli")

    paid_cols = st.columns(3)
    with paid_cols[0]:
        st.metric("Ödenmiş Sermaye / Özkaynaklar", f"%{health['paid_in_ratio']:.1f}")
    with paid_cols[1]:
        st.metric("Bedelsiz Potansiyeli", f"%{fund_data['bonus_issue_potential_pct']:,.1f}")
    with paid_cols[2]:
        st.metric("Durum", health["paid_in_status"])

    st.markdown(f"<div class='{'safe-box' if 'MÜKEMMEL' in health['paid_in_status'] else ('warn-box' if 'İYİ' in health['paid_in_status'] else 'risk-box')}'>{health['paid_in_detail']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ──── 6. TİCARİ BORÇ / ALACAK ANALİZİ ────
    st.markdown("### 6️⃣ Ticari Borç & Alacak Analizi")

    trade_cols = st.columns(4)
    with trade_cols[0]:
        st.metric("Ticari Alacaklar", format_number(health["trade_receivables"]))
    with trade_cols[1]:
        st.metric("Ticari Borçlar", format_number(health["trade_payables"]))
    with trade_cols[2]:
        st.metric("Finansal Borç", format_number(health["financial_debt"]))
    with trade_cols[3]:
        st.metric("Alacak / Borç Oranı", f"%{health['receivable_ratio']:.1f}")

    if health["debt_signals"]:
        for sig in health["debt_signals"]:
            st.markdown(f"<div class='risk-box'>⚠️ {sig}</div>", unsafe_allow_html=True)

    st.info("💡 **Not:** Faizin yüksek olduğu dönemde borçların %40'ından fazlası ticari alacaksa tahsilat riski artar. Senetli alacak biraz daha güvenli ama yine de dikkat edilmeli.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: TAKAS & PARA AKIŞI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_custody:
    st.markdown("<div class='section-header'>🏦 Takas Saklama Oranları & Para Akışı</div>", unsafe_allow_html=True)

    cust_data = analyze_custody_and_money_flow(ticker)

    c_cols = st.columns(4)
    with c_cols[0]:
        st.metric("MFI-14", f"{cust_data['mfi_14']:.1f}")
    with c_cols[1]:
        st.metric("OBV Trend", cust_data['obv_trend'])
    with c_cols[2]:
        st.metric("Takas Skoru", f"{cust_data['custody_proxy_score']:.0f} / 100")
    with c_cols[3]:
        st.metric("Son Hacim", f"{cust_data.get('latest_volume', 0):,} lot")

    st.markdown(f"### 📌 Takas Durumu: **{cust_data['flow_status']}**")

    st.markdown("---")

    # Kurum Dağılımı
    st.markdown("#### 🏛️ Kurum Bazlı Takas Saklama Dağılımı")
    if cust_data.get("top_custody_holders"):
        custody_df = pd.DataFrame(cust_data["top_custody_holders"])
        st.table(custody_df)

    cust_col1, cust_col2 = st.columns(2)
    with cust_col1:
        st.markdown("#### 🟢 En Çok Toplayan (Alıcı) Kurumlar")
        if cust_data.get("top_buyers"):
            st.table(pd.DataFrame(cust_data["top_buyers"]))
    with cust_col2:
        st.markdown("#### 🔴 En Çok Satan (Satıcı) Kurumlar")
        if cust_data.get("top_sellers"):
            st.table(pd.DataFrame(cust_data["top_sellers"]))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: HABERLER & DUYGU ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.markdown("<div class='section-header'>📰 Güncel Haberler & Türkçe BERT Duygu Analizi</div>", unsafe_allow_html=True)

    st.info(f"**{ticker}** şirketine ait güncel haberler ve KAP bildirimleri taranıyor...")

    # Google News RSS ile haberler
    try:
        from extraction_kap import fetch_google_news_rss, fetch_yfinance_news

        news_items = fetch_google_news_rss(ticker_clean, max_items=10)
        yf_news = fetch_yfinance_news(ticker_clean)
        all_news = news_items + yf_news

        if all_news:
            # BERT Duygu Analizi
            try:
                from sentiment_engine import TurkishBertSentimentEngine
                engine = TurkishBertSentimentEngine()
                titles = [n["title"] for n in all_news if n.get("title")]
                scored = engine.score_texts(titles)

                st.markdown(f"**{len(scored)} haber başlığı analiz edildi:**")
                st.markdown("---")

                for item in scored:
                    score = item["score"]
                    if score > 0.15:
                        badge = "<span class='signal-green'>🟢 POZİTİF</span>"
                    elif score < -0.15:
                        badge = "<span class='signal-red'>🔴 NEGATİF</span>"
                    else:
                        badge = "<span class='signal-yellow'>🟡 NÖTR</span>"

                    st.markdown(f"""
                    {badge} **Skor: `{score:+.3f}`**

                    📄 {item['text']}
                    """, unsafe_allow_html=True)
                    st.markdown("---")

                # Ortalama Duygu Skoru
                avg_score = np.mean([s["score"] for s in scored])
                pos_count = sum(1 for s in scored if s["score"] > 0.15)
                neg_count = sum(1 for s in scored if s["score"] < -0.15)
                neut_count = len(scored) - pos_count - neg_count

                score_cols = st.columns(4)
                with score_cols[0]:
                    st.metric("Ortalama Duygu Skoru", f"{avg_score:+.3f}")
                with score_cols[1]:
                    st.metric("🟢 Pozitif", f"{pos_count}")
                with score_cols[2]:
                    st.metric("🟡 Nötr", f"{neut_count}")
                with score_cols[3]:
                    st.metric("🔴 Negatif", f"{neg_count}")

            except Exception as e:
                st.warning(f"BERT motoru yüklenirken: {e}")
                for n in all_news[:10]:
                    st.markdown(f"📄 **{n.get('title', 'Başlık yok')}**")
                    if n.get("link"):
                        st.markdown(f"🔗 [Habere Git]({n['link']})")
                    st.markdown("---")
        else:
            st.info("Şu an için güncel haber bulunamadı.")

    except Exception as e:
        st.warning(f"Haber çekim modülü: {e}")

    # Bedelli / Bedelsiz Sermaye Haberleri
    st.markdown("---")
    st.markdown("### 📣 Bedelli / Bedelsiz Sermaye Artırımı Haberleri")
    st.info("""
    **Bedelsiz Sermaye Artırımı:** Özkaynaklardaki birikmiş kârlar / yedekler sermayeye eklenerek 
    mevcut pay sahiplerine ücretsiz pay dağıtılması. Ödenmiş sermaye küçükse potansiyel yüksek.

    **Bedelli Sermaye Artırımı:** Şirketin nakit ihtiyacı olduğunda yeni pay ihraç ederek piyasadan 
    para toplaması. Dönen varlıklar yetersizse ve duran varlık satışı yapılamıyorsa başvurulan yöntem.
    """)

    try:
        bonus_news = fetch_google_news_rss(ticker_clean + " bedelsiz bedelli sermaye", max_items=5)
        if bonus_news:
            for n in bonus_news:
                st.markdown(f"📢 **{n.get('title', '')}**")
                if n.get("pub_date_str"):
                    st.caption(f"📅 {n['pub_date_str']}")
                st.markdown("---")
        else:
            st.caption("Son dönemde bedelli/bedelsiz sermaye haberi bulunamadı.")
    except Exception:
        st.caption("Haber arama servisi şu an kullanılamıyor.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: PORTFÖY RİSK YÖNETİMİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.markdown("<div class='section-header'>💼 Portföy Risk & Performans Yönetimi</div>", unsafe_allow_html=True)
    
    st.markdown("Elinizdeki hisseleri aralarına virgül koyarak yazın, sistem toplu risk analizini çıkarsın.")
    port_input = st.text_input("Portföyünüzdeki Hisseler:", "THYAO, EREGL, SASA, GARAN").upper()
    
    if st.button("📊 Portföyü Analiz Et"):
        with st.spinner("Tüm portföy taranıyor, mali tablolar analiz ediliyor..."):
            tickers = [t.strip() for t in port_input.split(",") if t.strip()]
            port_results = []
            
            for t in tickers:
                try:
                    f_data = analyze_company_fundamentals(t)
                    h_data = get_financial_health_analysis(f_data)
                    c_data = analyze_custody_and_money_flow(t)
                    
                    port_results.append({
                        "Hisse": t,
                        "Mali Yapı": h_data["structure_label"].upper(),
                        "Nakit/Borç": f"{f_data['cash_to_debt_ratio']:.2f}x",
                        "NWC Durum": "🚨 Riskli" if h_data["nwc_ratio"] < 1 else "✅ Güvenli",
                        "Takas Skor": f"{c_data.get('custody_proxy_score', 50):.0f}",
                        "Faiz Uyum": f"{f_data['interest_rate_score']:.0f}"
                    })
                except:
                    pass
            
            if port_results:
                st.markdown("### 📋 Portföy Özeti")
                st.table(pd.DataFrame(port_results).set_index("Hisse"))
                
                # Toplu Risk Değerlendirmesi
                risk_count = sum(1 for r in port_results if "Riskli" in r["NWC Durum"] or r["Mali Yapı"] == "GÜÇSÜZ")
                st.markdown("---")
                if risk_count > 0:
                    st.markdown(f"<div class='risk-box'>⚠️ Portföyünüzde **{risk_count} adet** yüksek riskli (likidite veya borç sorunu olan) şirket bulunuyor. Yüksek faiz ortamında bu şirketlerin ağırlığını azaltmayı düşünebilirsiniz.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='safe-box'>✅ Portföyünüz mali açıdan sağlam şirketlerden oluşuyor. Likidite veya ağır borçluluk sorunu tespit edilmedi.</div>", unsafe_allow_html=True)
            else:
                st.warning("Veri çekilemedi. Hisseleri doğru girdiğinizden emin olun (Örn: THYAO, EREGL)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: TELEGRAM API TEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_telegram:
    st.markdown("<div class='section-header'>📱 Telegram REST API Testi</div>", unsafe_allow_html=True)


    st.markdown("### 🌐 API Endpoint Testi")

    api_ticker_input = st.text_input("Test edilecek hisse kodu:", value=ticker_clean)

    if st.button("📡 API Raporu Üret"):
        with st.spinner("Rapor hazırlanıyor..."):
            test_fund = analyze_company_fundamentals(api_ticker_input)
            test_health = get_financial_health_analysis(test_fund)
            test_cust = analyze_custody_and_money_flow(api_ticker_input)
            report_text = generate_telegram_report(api_ticker_input, test_fund, test_health, test_cust)

            st.markdown("#### 📨 Telegram Mesaj Önizlemesi:")
            st.code(report_text, language="markdown")

            st.success("✅ Bu rapor Telegram'a gönderilmeye hazır!")




# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class='footer-text'>
    📊 BIST 100 — 360° Yatırımcı Karar Destek Platformu<br>
    <small>LSTM & GRU Derin Öğrenme · Temel Analiz · Mali Yapı · Takas · BERT NLP · n8n Telegram</small><br>
    <small>⚠️ Bu platform yatırım tavsiyesi niteliğinde değildir. Yatırım kararlarınızda risk yönetimi uygulayınız.</small>
</div>
""", unsafe_allow_html=True)
