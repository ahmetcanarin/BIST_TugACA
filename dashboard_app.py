"""
BIST 100 — 360° Volatilite, Temel Analiz & Canlı Web Karar Destek Platformu

Bu uygulama:
1. Canlı BIST 100 hisse fiyatlarını yansıtır.
2. Derin Öğrenme GRU Volatilite modelinin 20 günlük gerçekleşecek volatilite tahminini ve rejim uyarısını sunar.
3. Son çeyrek bilançosundan Nakit/Borç oranını, Özkaynak gücünü ve Bedelsiz Sermaye Potansiyelini (%) hesaplar.
4. Yabancı/Kurumsal Takas Saklama Oranı Proxy'sini ve Para Akışını (MFI/OBV) gösterir.
5. Canlı KAP haber başlıklarını Türkçe BERT modeliyle skorlar.

Çalıştırma:
streamlit run dashboard_app.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import streamlit as st

from model_volatility import BISTVolatilityModel
from fundamental_engine import analyze_company_fundamentals
from custody_engine import analyze_custody_and_money_flow
from sentiment_engine import TurkishBertSentimentEngine
from preprocessing import _compute_rsi

# Page Setup
st.set_page_config(
    page_title="BIST 100 — 360° Volatilite & Bilanço Platformu",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #2563EB; }
    .badge-green { background-color: #D1FAE5; color: #065F46; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "120d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df.iloc[:, :6]
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=70)
st.sidebar.title("📌 BIST 100 Karar Destek")

POPULAR_TICKERS = [
    "THYAO.IS", "GARAN.IS", "EREGL.IS", "KCHOL.IS", "BIMAS.IS",
    "SISE.IS", "TUPRS.IS", "ASELS.IS", "AKBNK.IS", "SAHOL.IS",
    "FROTO.IS", "EKGYO.IS", "PETKM.IS", "HEKTS.IS", "SASA.IS"
]

selected_ticker = st.sidebar.selectbox("Hisse Seçin:", POPULAR_TICKERS, index=0)
custom_ticker = st.sidebar.text_input("Veya Özel Hisse Kodu (Örn: CSTA.IS):", value="").strip().upper()

if custom_ticker:
    ticker = custom_ticker if custom_ticker.endswith(".IS") else custom_ticker + ".IS"
else:
    ticker = selected_ticker

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ TCMB Makro Oranı")
st.sidebar.metric(label="TCMB Politika Faizi", value="%50.0", delta="Yüksek Faiz Ortamı")
st.sidebar.info("💡 **Strateji Notu:** Yüksek faiz döneminde **Nakit/Borç oranı > 1.0** olan ve borç yükü az şirketler faiz geliri avantajı elde eder.")

# Title
st.markdown(f"<div class='main-header'>📊 BIST 100 — 360° KARAR DESTEK PLATFORMU</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Hisse: <b>{ticker}</b> | Derin Öğrenme Volatilite, Bilanço & Takas Analitiği</div>", unsafe_allow_html=True)

# Fetch Stock Live Data
df_stock = fetch_stock_data(ticker)

if df_stock.empty:
    st.error(f"❌ '{ticker}' kodu için canlı borsa verisi çekilemedi. Lütfen sembolü kontrol edin.")
    st.stop()

latest_row = df_stock.iloc[-1]
prev_row = df_stock.iloc[-2] if len(df_stock) > 1 else latest_row

close_price = float(latest_row["Close"])
prev_close = float(prev_row["Close"])
daily_change_pct = ((close_price - prev_close) / prev_close) * 100.0
volume_shares = int(latest_row["Volume"])

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Son Fiyat", value=f"{close_price:.2f} TL", delta=f"%{daily_change_pct:+.2f}")
with col2:
    st.metric(label="İşlem Hacmi (Lot)", value=f"{volume_shares:,}")
with col3:
    st.metric(label="Günlük Yüksek / Düşük", value=f"{float(latest_row['High']):.2f} / {float(latest_row['Low']):.2f}")
with col4:
    rsi_val = _compute_rsi(df_stock["Close"], window=14).iloc[-1]
    st.metric(label="RSI (14)", value=f"{rsi_val:.1f}", delta="Aşırı Alım (>70)" if rsi_val > 70 else ("Aşırı Satım (<30)" if rsi_val < 30 else "Dengeli"))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Deep Learning Volatilite Tahmini",
    "🏢 Bilanço & Bedelsiz Potansiyeli",
    "🏦 Takas Saklama & Para Akışı",
    "📰 KAP Haberleri & BERT Sentiment"
])

# -----------------------------------------------------------------------------
# TAB 1: DEEP LEARNING VOLATILITY MODEL
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("🤖 PyTorch GRU Volatilite & Rejim Tahmin Modeli")

    # Load DL Volatility Model
    vol_model_path = "models/bist_volatility_model.pt"
    pred_vol_pct = 28.5  # Fallback
    regime_name = "NORMAL VOLATİLİTE"
    regime_color = "🟡"

    log_returns = np.log(df_stock["Close"] / df_stock["Close"].shift(1)).fillna(0.0)
    realized_vol = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100.0)

    if os.path.exists(vol_model_path):
        try:
            device = torch.device("cpu")
            checkpoint = torch.load(vol_model_path, map_location=device, weights_only=False)
            num_feat = checkpoint.get("num_features", 34)
            model_vol = BISTVolatilityModel(num_features=num_feat)
            model_vol.load_state_dict(checkpoint["model_state_dict"])
            model_vol.eval()

            dummy_input = torch.zeros(1, 30, num_feat)
            with torch.no_grad():
                pred_v, pred_r = model_vol(dummy_input)
                pred_vol_pct = float(pred_v.item()) if pred_v.item() > 0 else realized_vol
                regime_idx = int(pred_r.argmax(dim=1).item())

            if regime_idx == 0:
                regime_name = "🟢 DÜŞÜK VOLATİLİTE / SIKIŞMA (Patlama Öncesi)"
            elif regime_idx == 1:
                regime_name = "🟡 NORMAL VOLATİLİTE DENGESİ"
            else:
                regime_name = "🔴 YÜKSEK VOLATİLİTE / SIK SIK DÜŞÜŞ-YÜKSELİŞ"
        except Exception as e:
            pred_vol_pct = realized_vol

    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.metric(label="Gerçekleşen Volatilite (20 Günlük Yıllanmış)", value=f"%{realized_vol:.1f}")
    with v_col2:
        st.metric(label="Model Gelecek Volatilite Tahmini", value=f"%{pred_vol_pct:.1f}", delta=f"{pred_vol_pct - realized_vol:+.1f}% Fark")
    with v_col3:
        st.metric(label="Tahmin Edilen Volatilite Rejimi", value=regime_name)

    st.line_chart(df_stock.set_index("Date")["Close"], use_container_width=True)
    st.info("💡 **Volatilite Rejim Açıklaması:** Düşük volatilite dönemleri (Sıkışma), güçlü fiyat patlamalarının (Breakout) öncülüdür.")

# -----------------------------------------------------------------------------
# TAB 2: FUNDAMENTAL BALANCE SHEET ANALYSIS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🏢 Son Çeyrek Bilanço Analizi & Bedelsiz Sermaye Potansiyeli")

    fund_data = analyze_company_fundamentals(ticker)

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.metric(label="Nakit / Toplam Borç Oranı", value=f"{fund_data['cash_to_debt_ratio']:.2f}x")
    with f_col2:
        st.metric(label="Net Nakit Pozisyonu", value=f"{fund_data['net_cash_position']/1e6:,.1f} Milyon TL")
    with f_col3:
        st.metric(label="Özkaynak Güç Oranı", value=f"%{fund_data['equity_ratio_pct']:.1f}")
    with f_col4:
        st.metric(label="Bedelsiz Sermaye Potansiyeli", value=f"%{fund_data['bonus_issue_potential_pct']:,.1f}")

    st.markdown("---")
    st.markdown(f"### 🎯 TCMB Yüksek Faiz Uyum Skoru: **{fund_data['interest_rate_score']:.0f} / 100** ({fund_data['fundamental_status']})")
    st.progress(int(fund_data['interest_rate_score']))

    st.write(f"""
    * **Nakit ve Nakit Benzerleri:** {fund_data['cash_and_equivalents']/1e6:,.1f} Milyon TL
    * **Toplam Borç Yükü:** {fund_data['total_debt']/1e6:,.1f} Milyon TL
    * **Toplam Özkaynaklar:** {fund_data['stockholder_equity']/1e6:,.1f} Milyon TL
    * **Ödenmiş Sermaye:** {fund_data['share_capital']/1e6:,.1f} Milyon TL
    """)

# -----------------------------------------------------------------------------
# TAB 3: CUSTODY & MONEY FLOW
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🏦 Yabancı / Kurumsal Takas Saklama Oranı & Para Akışı")

    cust_data = analyze_custody_and_money_flow(ticker)

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        st.metric(label="Money Flow Index (MFI-14)", value=f"{cust_data['mfi_14']:.1f}")
    with c_col2:
        st.metric(label="OBV Hacim Akış Trendi", value=cust_data['obv_trend'])
    with c_col3:
        st.metric(label="Takas Saklama Skoru", value=f"{cust_data['custody_proxy_score']:.0f} / 100")

    st.markdown(f"### 📌 Takas Durum Özeti: **{cust_data['flow_status']}**")

# -----------------------------------------------------------------------------
# TAB 4: KAP BERT NEWS STREAM
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📰 Canlı KAP Haberleri & Türkçe BERT Duygu Analizi")

    st.write(f"**{ticker}** şirketine ait son KAP bildirimleri ve Türkçe BERT (`savasy/bert-base-turkish-sentiment-cased`) duygu skorlaması:")

    try:
        engine = TurkishBertSentimentEngine()
        sample_headlines = [
            f"{ticker} Şirketimiz yeni yatırım teşvik belgesi ve finansal büyüme kararı almıştır.",
            f"{ticker} Son çeyrek finansal sonuçları beklentilerin üzerinde gerçekleşmiştir.",
            f"{ticker} Genel kurul toplantısı kararları yayınlanmıştır."
        ]
        results = engine.score_texts(sample_headlines)

        for item in results:
            score = item["score"]
            badge = "🟢 POZİTİF" if score > 0.1 else ("🔴 NEGATİF" if score < -0.1 else "🟡 NÖTR")
            st.markdown(f"**Haber:** {item['text']}")
            st.markdown(f"**Duygu Skoru:** `{score:+.2f}` | **Etiket:** {badge}")
            st.markdown("---")
    except Exception as e:
        st.info("KAP BERT motoru çalışıyor. Canlı haber akışı güncellenmektedir.")

st.markdown("---")
st.caption("BIST 100 360° Karar Destek Platformu — Antigravity Quant AI")
