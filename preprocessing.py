"""
BIST Hisse Senetleri için Panel Veri ve LSTM/GRU Ön İşleme Hattı (Pipeline)

Bu modül:
1. Yahoo Finance ham verisini uzun/panel (Long / Panel: Date, Ticker) formatına çevirir.
2. Ham fiyatlar yerine durağan (stationary) finansal ve teknik göstergeler türetir.
3. Hisselerin farklı halka arz tarihlerinden kaynaklanan NaN problemlerini yok eder.
4. Zamansal (Chronological / Walk-forward) Train-Val-Test ayrımı yapar (Look-ahead bias yok).
5. RobustScaler ile aykırı değerlere dayanıklı ölçekleme yapar (Sadece Train verisine fit edilir).
6. LSTM ve GRU modellerinin beklediği (Örnek Sayısı, Zaman Adımı, Özellik Sayısı)
   şeklindeki 3 Boyutlu tensörleri (3D Tensors) üretir.
7. KÜRESEL KRONOLOJİK DÜZELTME (Global Chronological Alignment):
   Tarih asla geriye dönmez; tüm hisseler gün gün ileriye akar (Cross-Sectional Time-Series).
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from extraction import (
    tum_hisseleri_cek,
    endeks_verisini_cek,
    makro_verileri_cek,
    sektor_verilerini_cek,
    TICKER_SECTOR_MAP,
    BIST_LIQUID_40
)
from extraction_tcmb import get_tcmb_interest_rate_series, compute_tcmb_features

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# =============================================================================
# 1. HAM VERİYİ PANEL (LONG) FORMATA DÖNÜŞTÜRME
# =============================================================================
def prepare_panel_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo Finance'ten gelen MultiIndex sütunlu geniş tabloyu
    (Date, Ticker) formatında temiz bir panel veriye dönüştürür.
    """
    df = df_raw.copy()

    # MultiIndex sütunları (Price, Ticker) veya (Ticker, Price) panel satırlarına aç (stack)
    if isinstance(df.columns, pd.MultiIndex):
        level_name = "Ticker" if "Ticker" in df.columns.names else 1
        df = df.stack(level=level_name, future_stack=True).reset_index()
        df.columns.name = None

    # Sütun adları standartlaştırma
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df["Date"] = pd.to_datetime(df["Date"])

    # Eksik fiyat satırlarını temizleme (Halka arz öncesi veya işlem görmeyen günler)
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df.dropna(subset=price_cols)

    # İndikatör hesaplaması için hisse içi kronolojik sıralama
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


# =============================================================================
# 2. DURAĞAN VE TEKNİK ÖZNİTELİK MÜHENDİSLİĞİ (STATIONARY FEATURES)
# =============================================================================
def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Klasik RSI (Göreceli Güç Endeksi) hesabı."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=min(5, window)).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=min(5, window)).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def process_index_data(df_index: pd.DataFrame) -> pd.DataFrame:
    """
    XU100.IS verisinden Date, Index_Log_Return, Index_Volatility_20 ve Target_Index_Return üretir.
    Hedef Getiri: Ertesi günkü seans içi getiri (T+1 Açılış -> T+1 Kapanış).
    """
    if df_index is None or df_index.empty:
        return pd.DataFrame(columns=["Date", "Index_Log_Return", "Index_Volatility_20", "Target_Index_Return"])

    idx = df_index.copy()

    def _extract_col(col_name: str) -> pd.Series:
        if isinstance(idx.columns, pd.MultiIndex):
            if col_name in idx.columns.levels[0]:
                sub = idx[col_name]
                return sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            elif col_name in idx.columns.names or any(c == col_name for c in idx.columns.get_level_values(1)):
                sub = idx.xs(col_name, axis=1, level=1)
                return sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            else:
                return idx.iloc[:, 0]
        elif col_name in idx.columns:
            sub = idx[col_name]
            return sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
        return idx.iloc[:, 0]

    close_s = _extract_col("Close")
    open_s = _extract_col("Open")

    date_idx = pd.to_datetime(idx.index)
    if date_idx.tz is not None:
        date_idx = date_idx.tz_localize(None)

    idx_df = pd.DataFrame({
        "Date": date_idx,
        "Index_Close": pd.to_numeric(close_s.values.flatten() if hasattr(close_s, "values") else close_s, errors="coerce"),
        "Index_Open": pd.to_numeric(open_s.values.flatten() if hasattr(open_s, "values") else open_s, errors="coerce")
    }).dropna(subset=["Index_Close", "Index_Open"]).sort_values("Date").reset_index(drop=True)

    idx_df["Index_Log_Return"] = np.log(idx_df["Index_Close"] / idx_df["Index_Close"].shift(1))
    idx_df["Index_Volatility_20"] = idx_df["Index_Log_Return"].rolling(window=20).std()

    # 1. Katman Makro Rejim Göstergeleri: XU100 > SMA50 ve SMA200
    idx_df["Index_SMA50"] = idx_df["Index_Close"].rolling(window=50, min_periods=20).mean()
    idx_df["Index_SMA200"] = idx_df["Index_Close"].rolling(window=200, min_periods=50).mean()
    idx_df["Market_Trend_SMA50"] = (idx_df["Index_Close"] > idx_df["Index_SMA50"]).astype(float)
    idx_df["Market_Trend_SMA200"] = (idx_df["Index_Close"] > idx_df["Index_SMA200"]).astype(float)

    # 5 Günlük ve 1 Günlük Endeks İleri Getirileri
    next_5d_close = idx_df["Index_Close"].shift(-5)
    valid_5d = (next_5d_close > 0) & next_5d_close.notna()
    raw_5d_target = ((next_5d_close - idx_df["Index_Close"]) / idx_df["Index_Close"]) * 100.0
    idx_df["Target_Index_Return_5d"] = raw_5d_target.where(valid_5d, np.nan)

    next_idx_open = idx_df["Index_Open"].shift(-1)
    next_idx_close = idx_df["Index_Close"].shift(-1)
    valid_idx = (next_idx_open > 0) & next_idx_open.notna() & next_idx_close.notna()
    raw_idx_target = ((next_idx_close - next_idx_open) / next_idx_open) * 100.0
    idx_df["Target_Index_Return_1d"] = raw_idx_target.where(valid_idx, np.nan)

    # Standart hedef olarak 5 günlük endeks getirisini kullan
    idx_df["Target_Index_Return"] = idx_df["Target_Index_Return_5d"]
    return idx_df


def process_macro_data(df_macro: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    yfinance'ten gelen Dalga-1 çoklu makro verisini (USDTRY, VIX, BRENT, GOLD)
    temizler, durağan göstergelere dönüştürür ve Date bazlı panele hazırlar.
    """
    cols = [
        "Date", "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return"
    ]
    if df_macro is None or df_macro.empty:
        return pd.DataFrame(columns=cols)

    m = df_macro.copy()
    date_idx = pd.to_datetime(m.index)
    if date_idx.tz is not None:
        date_idx = date_idx.tz_localize(None)

    macro_df = pd.DataFrame({"Date": date_idx})

    def _extract_close(ticker_str: str) -> pd.Series:
        if isinstance(m.columns, pd.MultiIndex):
            if ticker_str in m.columns.levels[0]:
                sub = m[ticker_str]
                s = sub["Close"] if "Close" in sub.columns else sub.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce")
            elif "Close" in m.columns.levels[0] and ticker_str in m["Close"].columns:
                return pd.to_numeric(m["Close"][ticker_str], errors="coerce")
        elif ticker_str in m.columns:
            return pd.to_numeric(m[ticker_str], errors="coerce")
        return pd.Series(np.nan, index=m.index)

    # 1. USD/TRY
    usd_s = _extract_close("USDTRY=X").ffill().fillna(0.0)
    usd_ret = np.log(usd_s / usd_s.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_df["USDTRY_Return"] = usd_ret.values
    macro_df["USDTRY_Vol_20"] = usd_ret.rolling(window=20, min_periods=5).std().fillna(0.0).values

    # 2. VIX (Global Korku & Risk İştahı)
    vix_s = _extract_close("^VIX").ffill().fillna(20.0)
    macro_df["VIX_Level"] = (vix_s / 50.0).fillna(0.4).values
    macro_df["VIX_Change"] = (vix_s.diff() / 10.0).fillna(0.0).values

    # 3. Brent Petrol (Enerji & Maliyet)
    brent_s = _extract_close("BZ=F").ffill().fillna(0.0)
    brent_ret = np.log(brent_s / brent_s.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_df["Brent_Return"] = brent_ret.values

    # 4. Ons Altın (Güvenli Liman & Madencilik)
    gold_s = _extract_close("GC=F").ffill().fillna(0.0)
    gold_ret = np.log(gold_s / gold_s.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_df["Gold_Return"] = gold_ret.values

    # 5. CDS Sovereign Risk Proxy (TUR vs EEM Spread Dinamikleri)
    # TUR: iShares MSCI Turkey ETF (USD)
    # EEM: iShares MSCI Emerging Markets ETF (USD)
    # Spread = -ln(TUR / EEM) = ln(EEM / TUR) -> TR riski arttığında yükselir
    tur_s = _extract_close("TUR").ffill()
    eem_s = _extract_close("EEM").ffill()
    valid_mask = (tur_s > 0) & (eem_s > 0) & tur_s.notna() & eem_s.notna()
    spread = pd.Series(0.0, index=m.index)
    spread[valid_mask] = -np.log(tur_s[valid_mask] / eem_s[valid_mask])
    spread = spread.ffill().fillna(0.0)

    cds_chg5 = spread.diff(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_df["CDS_Proxy_Chg5"] = cds_chg5.values

    roll_mean = spread.rolling(window=60, min_periods=10).mean()
    roll_std = spread.rolling(window=60, min_periods=10).std()
    z_score = ((spread - roll_mean) / (roll_std + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_df["CDS_Proxy_ZScore_60d"] = (z_score.clip(-4.0, 4.0) / 4.0).values

    spread_diff = spread.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_df["CDS_Spread_Diff"] = spread_diff.values

    return macro_df.sort_values("Date").reset_index(drop=True)


def process_sector_data(df_sec: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    yfinance'ten gelen BIST Banka (XBANK.IS) ve BIST Sınai (XUSIN.IS) verilerini
    Date bazlı temiz sütunlara ve relatif getiri/momentum göstergelerine dönüştürür.
    """
    cols = ["Date", "Bank_Log_Return", "Indus_Log_Return", "Bank_Mom10", "Indus_Mom10"]
    if df_sec is None or df_sec.empty:
        return pd.DataFrame(columns=cols)

    s = df_sec.copy()
    date_idx = pd.to_datetime(s.index)
    if date_idx.tz is not None:
        date_idx = date_idx.tz_localize(None)

    sec_df = pd.DataFrame({"Date": date_idx})

    def _extract_close(ticker_str: str) -> pd.Series:
        if isinstance(s.columns, pd.MultiIndex):
            if ticker_str in s.columns.levels[0]:
                sub = s[ticker_str]
                c = sub["Close"] if "Close" in sub.columns else sub.iloc[:, 0]
                return pd.to_numeric(c, errors="coerce")
            elif "Close" in s.columns.levels[0] and ticker_str in s["Close"].columns:
                return pd.to_numeric(s["Close"][ticker_str], errors="coerce")
        elif ticker_str in s.columns:
            return pd.to_numeric(s[ticker_str], errors="coerce")
        return pd.Series(np.nan, index=s.index)

    bank_close = _extract_close("XBANK.IS").ffill().bfill().fillna(1.0)
    indus_close = _extract_close("XUSIN.IS").ffill().bfill().fillna(1.0)

    bank_ret = np.log(bank_close / bank_close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    indus_ret = np.log(indus_close / indus_close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sec_df["Bank_Log_Return"] = bank_ret.values
    sec_df["Indus_Log_Return"] = indus_ret.values

    bank_m10 = (bank_close / bank_close.shift(10) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    indus_m10 = (indus_close / indus_close.shift(10) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sec_df["Bank_Mom10"] = bank_m10.values
    sec_df["Indus_Mom10"] = indus_m10.values

    return sec_df.sort_values("Date").reset_index(drop=True)


def add_technical_features(
    df_panel: pd.DataFrame,
    df_index: Optional[pd.DataFrame] = None,
    df_macro: Optional[pd.DataFrame] = None,
    df_sector: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Ham fiyatlar yerine durağan teknik, makro ve sektörel göstergeleri türetir:
    - Fiyat/Hacim/Momentum (15)
    - Piyasa Bağlamı & Makro Rejim (5)
    - Dalga-1 Makro & Beta (8)
    - Dalga-2 TCMB Politika Faizi (3)
    - Dalga-2 CDS Sovereign Risk Proxy (3)
    - Sektörel & Kurumsal Para Akışı (5): Bank_vs_Market, Indus_vs_Market, Sector_Relative_Strength_10d, MFI_14_Norm, OBV_Trend_10d
    """
    df_p = df_panel.copy()
    df_p["Date"] = pd.to_datetime(df_p["Date"])
    if df_p["Date"].dt.tz is not None:
        df_p["Date"] = df_p["Date"].dt.tz_localize(None)

    # Endeks verisini birleştir
    if df_index is not None and not df_index.empty:
        idx_df = process_index_data(df_index)
        idx_merge_cols = [c for c in ["Date", "Index_Log_Return", "Index_Volatility_20", "Target_Index_Return", "Market_Trend_SMA50", "Market_Trend_SMA200"] if c in idx_df.columns]
        df_p = pd.merge(df_p, idx_df[idx_merge_cols], on="Date", how="left")
    else:
        df_p["Index_Log_Return"] = 0.0
        df_p["Index_Volatility_20"] = 0.0
        df_p["Target_Index_Return"] = 0.0
        df_p["Market_Trend_SMA50"] = 1.0
        df_p["Market_Trend_SMA200"] = 1.0

    df_p["Index_Log_Return"] = df_p["Index_Log_Return"].fillna(0.0)
    df_p["Index_Volatility_20"] = df_p["Index_Volatility_20"].fillna(0.0)
    df_p["Target_Index_Return"] = df_p["Target_Index_Return"].fillna(0.0)
    if "Market_Trend_SMA50" not in df_p.columns:
        df_p["Market_Trend_SMA50"] = 1.0
    else:
        df_p["Market_Trend_SMA50"] = df_p["Market_Trend_SMA50"].fillna(1.0)
    if "Market_Trend_SMA200" not in df_p.columns:
        df_p["Market_Trend_SMA200"] = 1.0
    else:
        df_p["Market_Trend_SMA200"] = df_p["Market_Trend_SMA200"].fillna(1.0)

    # Makro verileri birleştir
    if df_macro is not None and not df_macro.empty:
        macro_df = process_macro_data(df_macro)
        df_p = pd.merge(df_p, macro_df, on="Date", how="left")
    else:
        macro_cols = [
            "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
            "Brent_Return", "Gold_Return", "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "CDS_Spread_Diff"
        ]
        for mc in macro_cols:
            df_p[mc] = 0.0

    # Makro eksikleri forward-fill ve 0 ile doldur
    macro_fill_cols = [
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "CDS_Spread_Diff"
    ]
    df_p[macro_fill_cols] = df_p[macro_fill_cols].ffill().fillna(0.0)

    # TCMB Faiz ve Para Politikası Verilerini Birleştir (Dalga-2)
    start_str = df_p["Date"].min().strftime("%Y-%m-%d")
    tcmb_raw = get_tcmb_interest_rate_series(start_date=start_str)
    tcmb_feat = compute_tcmb_features(tcmb_raw, df_p["Date"])
    df_p = pd.merge(df_p, tcmb_feat, on="Date", how="left")
    tcmb_cols = ["TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision"]
    df_p[tcmb_cols] = df_p[tcmb_cols].ffill().fillna(0.0)

    # Sektör Verilerini Birleştir (XBANK ve XUSIN)
    if df_sector is not None and not df_sector.empty:
        sec_df = process_sector_data(df_sector)
        df_p = pd.merge(df_p, sec_df, on="Date", how="left")
    else:
        df_p["Bank_Log_Return"] = 0.0
        df_p["Indus_Log_Return"] = 0.0
        df_p["Bank_Mom10"] = 0.0
        df_p["Indus_Mom10"] = 0.0

    df_p["Bank_Log_Return"] = df_p["Bank_Log_Return"].ffill().fillna(0.0)
    df_p["Indus_Log_Return"] = df_p["Indus_Log_Return"].ffill().fillna(0.0)
    df_p["Bank_Mom10"] = df_p["Bank_Mom10"].ffill().fillna(0.0)
    df_p["Indus_Mom10"] = df_p["Indus_Mom10"].ffill().fillna(0.0)

    df_p["Bank_vs_Market"] = df_p["Bank_Log_Return"] - df_p["Index_Log_Return"]
    df_p["Indus_vs_Market"] = df_p["Indus_Log_Return"] - df_p["Index_Log_Return"]

    groups = []
    for ticker, group in df_p.groupby("Ticker", group_keys=False):
        g = group.sort_values("Date").copy()

        # 1. Getiri ve Fiyat Oranları (Scale-free)
        g["Log_Return"] = np.log(g["Close"] / g["Close"].shift(1))
        g["HL_Spread"] = (g["High"] - g["Low"]) / g["Close"]
        g["CO_Return"] = (g["Close"] - g["Open"]) / g["Open"]
        g["Excess_Return"] = g["Log_Return"] - g["Index_Log_Return"]

        # 2. Hacim ve Likidite Dinamikleri (Kurumsal Likidite Filtresi için)
        g["Log_Volume"] = np.log1p(g["Volume"])
        g["Volume_Change"] = g["Log_Volume"].diff()
        g["Turnover_TRY"] = g["Volume"] * g["Close"]
        g["ADV20_TRY"] = g["Turnover_TRY"].rolling(window=20, min_periods=3).mean().fillna(0.0)

        # 3. Hareketli Ortalama Oranları
        sma_10 = g["Close"].rolling(window=10, min_periods=2).mean()
        sma_30 = g["Close"].rolling(window=30, min_periods=3).mean()
        g["SMA10_Ratio"] = ((g["Close"] / sma_10) - 1).fillna(0.0)
        g["SMA30_Ratio"] = ((g["Close"] / sma_30) - 1).fillna(0.0)

        # 4. Volatilite (20 günlük yuvarlanan standart sapma)
        g["Volatility_20"] = g["Log_Return"].rolling(window=20, min_periods=3).std().fillna(0.0)

        # 5. RSI (14) - [-1, 1]
        raw_rsi = _compute_rsi(g["Close"], window=14)
        g["RSI_Norm"] = ((raw_rsi - 50.0) / 50.0).fillna(0.0)

        # 6. ATR (14) - Yüzdesel Risk ve Volatilite Ölçütü (Dinamik Stop-Loss için)
        tr1 = g["High"] - g["Low"]
        tr2 = (g["High"] - g["Close"].shift(1)).abs()
        tr3 = (g["Low"] - g["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=3).mean()
        g["ATR_14_Pct"] = ((atr_14 / g["Close"]) * 100.0).fillna(3.0)

        # 7. Dinamik Makro Hassasiyetler (Rolling Betalar)
        cov_fx = g["Log_Return"].rolling(window=60, min_periods=5).cov(g["USDTRY_Return"])
        var_fx = g["USDTRY_Return"].rolling(window=60, min_periods=5).var()
        g["Beta_FX_60d"] = (cov_fx / (var_fx + 1e-7)).fillna(0.0).clip(-5.0, 5.0)

        cov_m = g["Log_Return"].rolling(window=60, min_periods=5).cov(g["Index_Log_Return"])
        var_m = g["Index_Log_Return"].rolling(window=60, min_periods=5).var()
        g["Beta_Market_60d"] = (cov_m / (var_m + 1e-7)).fillna(1.0).clip(-3.0, 5.0)

        cov_cds = g["Log_Return"].rolling(window=60, min_periods=5).cov(g["CDS_Spread_Diff"])
        var_cds = g["CDS_Spread_Diff"].rolling(window=60, min_periods=5).var()
        g["Beta_CDS_60d"] = (cov_cds / (var_cds + 1e-7)).fillna(0.0).clip(-5.0, 5.0)

        # 8. Çok Günlü Trend ve Momentum Öznitelikleri (5-10-20 Günlük Ufuk)
        g["Momentum_5d"] = (g["Close"] / g["Close"].shift(5) - 1.0).fillna(0.0)
        g["Momentum_10d"] = (g["Close"] / g["Close"].shift(10) - 1.0).fillna(0.0)
        g["Momentum_20d"] = (g["Close"] / g["Close"].shift(20) - 1.0).fillna(0.0)

        # Endekse Göre Rölatif Güç (Alpha Momentum)
        if "Index_Close" in g.columns and g["Index_Close"].notna().any():
            idx_mom5 = (g["Index_Close"] / g["Index_Close"].shift(5) - 1.0).fillna(0.0)
            idx_mom10 = (g["Index_Close"] / g["Index_Close"].shift(10) - 1.0).fillna(0.0)
            g["Relative_Strength_5d"] = g["Momentum_5d"] - idx_mom5
            g["Relative_Strength_10d"] = g["Momentum_10d"] - idx_mom10
        else:
            g["Relative_Strength_5d"] = g["Momentum_5d"]
            g["Relative_Strength_10d"] = g["Momentum_10d"]

        # Sektörel Göreceli Güç (Hissenin Kendi Sektörüne Göre 10 Günlük Relatif Momenti)
        t_clean = ticker.replace(".IS", "")
        sec_type = TICKER_SECTOR_MAP.get(t_clean, "OTHER")
        if sec_type == "BANK":
            g["Sector_Relative_Strength_10d"] = g["Momentum_10d"] - g["Bank_Mom10"]
        elif sec_type == "INDUS":
            g["Sector_Relative_Strength_10d"] = g["Momentum_10d"] - g["Indus_Mom10"]
        else:
            g["Sector_Relative_Strength_10d"] = g["Relative_Strength_10d"]

        # Hacim Trendi (Son 5 Gün Hacim / Son 20 Gün Hacim)
        vol_5 = g["Volume"].rolling(window=5, min_periods=2).mean()
        vol_20 = g["Volume"].rolling(window=20, min_periods=5).mean()
        g["Volume_Ratio_5_20"] = ((vol_5 / (vol_20 + 1e-4)) - 1.0).fillna(0.0).clip(-2.0, 5.0)

        # Para Akışı Endeksi (Money Flow Index - MFI 14, [-1, +1] normalize)
        typ_price = (g["High"] + g["Low"] + g["Close"]) / 3.0
        raw_money_flow = typ_price * g["Volume"]
        price_diff = typ_price.diff()
        pos_flow = raw_money_flow.where(price_diff > 0, 0.0)
        neg_flow = raw_money_flow.where(price_diff < 0, 0.0)
        pos_mf = pos_flow.rolling(window=14, min_periods=3).sum()
        neg_mf = neg_flow.rolling(window=14, min_periods=3).sum()
        mfr = pos_mf / (neg_mf + 1e-6)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        g["MFI_14_Norm"] = ((mfi - 50.0) / 50.0).fillna(0.0).clip(-1.0, 1.0)

        # On-Balance Volume (OBV) 10-Günlük Trend Eğimi (Kurumsal Para Girişi Proxy)
        obv_dir = np.sign(g["Close"].diff()).fillna(0.0)
        obv = (obv_dir * g["Volume"]).cumsum()
        obv_mean = obv.rolling(window=10, min_periods=3).mean()
        obv_std = obv.rolling(window=10, min_periods=3).std()
        g["OBV_Trend_10d"] = (((obv - obv_mean) / (obv_std + 1e-4)).fillna(0.0).clip(-3.0, 3.0)) / 3.0

        # ---------------------------------------------------------------------
        # HEDEF DEĞİŞKENLER (TARGETS): 5 Günlük Kapanış-Kapanış (Close_t -> Close_{t+5})
        # Giriş: Gün sonu kapanış müzayedesi (17:50 - 18:00) -> Gecelik Gap primleri portföyde kalır!
        # Çıkış: 5 iş günü sonraki kapanış (Close_{t+5}) -> Haftalık Rebalans
        # Stop: Gün içi whipsaw iptal; gün sonu kapanışlarındaki en düşük getiri izlenir.
        # ---------------------------------------------------------------------
        next_5d_close = g["Close"].shift(-5)
        valid_5d = (g["Close"] > 0) & (next_5d_close > 0) & next_5d_close.notna()
        raw_5d_return = ((next_5d_close - g["Close"]) / g["Close"]) * 100.0
        g["Target_Return_5d"] = raw_5d_return.where(valid_5d, np.nan)

        # 5 Gün boyunca gün sonu kapanışlarında görülen minimum ve maksimum getiri (EOD Stop)
        future_closes = pd.concat([g["Close"].shift(-i) for i in range(1, 6)], axis=1)
        min_future_close = future_closes.min(axis=1)
        max_future_close = future_closes.max(axis=1)
        g["Target_Min_Close_Return_5d"] = (((min_future_close - g["Close"]) / g["Close"]) * 100.0).where(valid_5d, np.nan)
        g["Target_Max_Close_Return_5d"] = (((max_future_close - g["Close"]) / g["Close"]) * 100.0).where(valid_5d, np.nan)

        if "Target_Index_Return_5d" in g.columns:
            g["Target_Excess_Return_5d"] = g["Target_Return_5d"] - g["Target_Index_Return_5d"]
        else:
            g["Target_Excess_Return_5d"] = g["Target_Return_5d"]

        g["Target_Direction_5d"] = (g["Target_Excess_Return_5d"] > 0).astype(float)

        # Standart hedefleri 5 günlük kapanış-kapanış ufkuyla eşle
        g["Target_Return"] = g["Target_Return_5d"]
        g["Target_Excess_Return"] = g["Target_Excess_Return_5d"]
        g["Target_Direction_Alpha"] = g["Target_Direction_5d"]
        g["Target_Low_Return"] = g["Target_Min_Close_Return_5d"]
        g["Target_High_Return"] = g["Target_Max_Close_Return_5d"]

        groups.append(g)

    df_features = pd.concat(groups, ignore_index=True)

    # 1. Katman Makro Rejim Kapısı: XU100 > SMA50 ve Düşük/Normal Kur Oynaklığı
    sma50_ok = (df_features["Market_Trend_SMA50"] == 1.0) if "Market_Trend_SMA50" in df_features.columns else pd.Series(True, index=df_features.index)
    usd_vol_ok = (df_features["USDTRY_Vol_20"] < 0.025) if "USDTRY_Vol_20" in df_features.columns else pd.Series(True, index=df_features.index)
    df_features["Macro_Regime_Bull"] = (sma50_ok & usd_vol_ok).astype(float)

    # 9. Kesitsel Sıralamalar (Cross-Sectional Percentile Rank: [-0.5, +0.5])
    df_features["Cross_Rank_Return"] = df_features.groupby("Date")["Log_Return"].rank(pct=True) - 0.5
    df_features["Target_Rank"] = df_features.groupby("Date")["Target_Return"].rank(pct=True) - 0.5
    df_features["Target_Excess_Rank"] = df_features.groupby("Date")["Target_Excess_Return"].rank(pct=True) - 0.5

    feature_cols = [
        # Hisse Fiyat/Hacim & Momentum (15)
        "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
        "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
        "Momentum_5d", "Momentum_10d", "Momentum_20d", "Relative_Strength_5d", "Relative_Strength_10d", "Volume_Ratio_5_20",
        # Sektörel & Kurumsal Para Akışı (5)
        "Bank_vs_Market", "Indus_vs_Market", "Sector_Relative_Strength_10d", "MFI_14_Norm", "OBV_Trend_10d",
        # Piyasa Bağlamı & Makro Rejim (5)
        "Index_Log_Return", "Excess_Return", "Cross_Rank_Return", "Market_Trend_SMA50", "Macro_Regime_Bull",
        # Dalga-1 Makro & Beta (8)
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "Beta_FX_60d", "Beta_Market_60d",
        # Dalga-2 TCMB Politika Faizi (3)
        "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
        # Dalga-2 CDS Sovereign Risk Proxy (3)
        "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "Beta_CDS_60d"
    ]
    df_features = df_features.dropna(subset=feature_cols + ["Target_Return"]).reset_index(drop=True)

    # Tabloyu küresel zaman akışına göre sıralıyoruz (Date birincil, Ticker ikincil)
    df_features = df_features.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df_features


# =============================================================================
# 3. ZAMANSAL VERİ AYRIMI (TEMPORAL TRAIN-VAL-TEST SPLIT)
# =============================================================================
def split_by_date(
    df: pd.DataFrame,
    train_end: Optional[str] = "2021-12-31",
    val_end: Optional[str] = "2023-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Finansal zaman serilerinde kural: Asla rastgele shuffle yapılmaz!
    Zaman ekseninde geçmiş -> gelecek sıralı bölünür.
    Eğer verinin başlangıç tarihi train_end'den sonraysa otomatik olarak
    tarihlerin %70'i train, %15'i val, %15'i test olarak bölünür.
    """
    min_date = df["Date"].min()
    use_date_strings = (train_end is not None and val_end is not None and min_date <= pd.to_datetime(train_end))

    if use_date_strings:
        train_df = df[df["Date"] <= train_end].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
        val_df = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
        test_df = df[df["Date"] > val_end].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    else:
        # Dinamik yüzdelik ayrım (Tarihe göre kronolojik)
        unique_dates = np.sort(df["Date"].unique())
        n_dates = len(unique_dates)
        if n_dates == 0:
            return df.copy(), df.copy(), df.copy()

        t_idx = min(max(0, int(n_dates * 0.70)), n_dates - 1)
        v_idx = min(max(t_idx, int(n_dates * 0.85)), n_dates - 1)

        t_cutoff = unique_dates[t_idx]
        v_cutoff = unique_dates[v_idx]

        train_df = df[df["Date"] <= t_cutoff].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
        val_df = df[(df["Date"] > t_cutoff) & (df["Date"] <= v_cutoff)].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
        test_df = df[df["Date"] > v_cutoff].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return train_df, val_df, test_df


STOCK_SPECIFIC_COLS = [
    # Hisse Fiyat/Hacim & Momentum (15)
    "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
    "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
    "Momentum_5d", "Momentum_10d", "Momentum_20d", "Relative_Strength_5d", "Relative_Strength_10d", "Volume_Ratio_5_20",
    # Sektörel & Kurumsal Para Akışı (5)
    "Bank_vs_Market", "Indus_vs_Market", "Sector_Relative_Strength_10d", "MFI_14_Norm", "OBV_Trend_10d",
    # Kesitsel Ayrışma & Betalar (5)
    "Excess_Return", "Cross_Rank_Return", "Beta_FX_60d", "Beta_Market_60d", "Beta_CDS_60d"
]

MACRO_MARKET_COLS = [
    # Piyasa Bağlamı & Makro Rejim (3)
    "Index_Log_Return", "Market_Trend_SMA50", "Macro_Regime_Bull",
    # Dalga-1 Makro & Beta (6)
    "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
    "Brent_Return", "Gold_Return",
    # Dalga-2 TCMB Politika Faizi (3)
    "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
    # Dalga-2 CDS Sovereign Risk Proxy (2)
    "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d"
]


# =============================================================================
# 4. ÖLÇEKLEME (GÜNLÜK KESİTSEL Z-SCORE + MAKRO ROBUST SCALER - QLIB STANDARDI)
# =============================================================================
def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    stock_cols: Optional[List[str]] = None,
    macro_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[RobustScaler]]:
    """
    Kantitatif Hibrit Ölçekleme (Qlib / WorldQuant Standardı):
    1. Hisse Bazlı Öznitelikler (stock_cols):
       Her t işlem gününde işlem gören hisseler arasında kesitsel Z-Score (cross-sectional norm)
       uygulanır ve [-3.0, 3.0] aralığına kırpılır.
       Böylece model rejim farklarından arınarak her gün sadece hisselerin birbirine göre üstünlüğünü öğrenir.
    2. Makro & Piyasa Bağlamı Öznitelikleri (macro_cols):
       Zaman serisi boyunca yalnızca Train verisine fit edilen RobustScaler ile ölçeklenir.
       Böylece makro göstergeler (VIX, TCMB faizi, kur oynaklığı) gün içi sıfırlanmaz.
    """
    if stock_cols is None:
        stock_cols = [c for c in STOCK_SPECIFIC_COLS if c in feature_cols]
    if macro_cols is None:
        macro_cols = [c for c in MACRO_MARKET_COLS if c in feature_cols]

    def _cross_sectional_norm(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty or not stock_cols:
            return df_in
        df_out = df_in.copy()
        valid_cols = [c for c in stock_cols if c in df_out.columns]
        if valid_cols and "Date" in df_out.columns:
            grouped = df_out.groupby("Date")[valid_cols]
            mean = grouped.transform("mean")
            std = grouped.transform("std").replace(0, np.nan).fillna(1.0)
            df_out[valid_cols] = ((df_out[valid_cols] - mean) / std).clip(-3.0, 3.0).fillna(0.0)
        return df_out

    # 1. Hisse bazlı göstergelerde her gün kendi içinde normalize edilir (Causal, zero leakage)
    train_scaled = _cross_sectional_norm(train_df)
    val_scaled = _cross_sectional_norm(val_df)
    test_scaled = _cross_sectional_norm(test_df)

    # 2. Makro göstergeler için RobustScaler (sadece train verisine fit)
    scaler = None
    if macro_cols:
        valid_macro = [c for c in macro_cols if c in train_scaled.columns]
        if valid_macro:
            scaler = RobustScaler()
            scaler.fit(train_scaled[valid_macro])
            train_scaled[valid_macro] = scaler.transform(train_scaled[valid_macro])
            val_scaled[valid_macro] = scaler.transform(val_scaled[valid_macro])
            test_scaled[valid_macro] = scaler.transform(test_scaled[valid_macro])

    return train_scaled, val_scaled, test_scaled, scaler


# =============================================================================
# 5. LSTM / GRU İÇİN 3D SIRALI DİZİLER (3D SEQUENCES) ÜRETİMİ
# =============================================================================
def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target_Return",
    seq_len: int = 30,
    sort_chronologically: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Her hisse için bağımsız kayan pencere (sliding window) uygulayarak
    LSTM ve GRU modellerinin girdi formatı olan 3D tensör üretir.

    Parametreler:
        seq_len: Geçmiş kaç günün pencere olarak alınacağı (örn. 30 gün)
        sort_chronologically: True ise üretilen tüm örnekleri küresel tarihe
            (global chronological order) göre sıralar. Böylece tarih ASLA geriye
            dönmez; tüm hisseler gün gün eşzamanlı ileriye akar (Cross-Sectional Stream).

    Çıktı Şekilleri:
        X: (Örnek Sayısı, seq_len, Özellik Sayısı)
        y: (Örnek Sayısı,)
        tickers: Her bir örneğin ait olduğu hisse kodu
        dates: Her bir tahminin yapıldığı günün tarihi (t)
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    tickers_list: List[str] = []
    dates_list: List[pd.Timestamp] = []

    # 1. Adım: Pencereleri her hissenin kendi geçmişi içinde üret (Hisseler birbirine karışmaz)
    for ticker, group in df.groupby("Ticker"):
        g = group.sort_values("Date").reset_index(drop=True)
        n_samples = len(g)

        if n_samples < seq_len:
            # Hisse bu periyotta pencere uzunluğundan daha az işlem gördüyse atlanır
            continue

        feat_values = g[feature_cols].values
        target_values = g[target_col].values
        date_values = g["Date"].values

        # Kayan pencere oluşturma (Sliding Window)
        for i in range(n_samples - seq_len + 1):
            # [t, t+1, ..., t+seq_len-1] günlerinin öznitelikleri (Pencere içi kronolojik)
            window_x = feat_values[i: i + seq_len]
            # t+seq_len-1 gününün hedefi (ertesi günün getirisi)
            window_y = target_values[i + seq_len - 1]
            ref_date = date_values[i + seq_len - 1]

            X_list.append(window_x)
            y_list.append(window_y)
            tickers_list.append(ticker)
            dates_list.append(pd.Timestamp(ref_date))

    if not X_list:
        empty_x = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_x, empty_y, [], []

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # 2. Adım: Küresel Kronolojik Sıralama (Global Chronological Alignment)
    # Farklı hisselerin pencerelerini gün gün yan yana getirir; zaman asla geriye dönmez!
    if sort_chronologically:
        sort_indices = np.argsort(dates_list)
        X = X[sort_indices]
        y = y[sort_indices]
        tickers_list = [tickers_list[idx] for idx in sort_indices]
        dates_list = [dates_list[idx] for idx in sort_indices]

    return X, y, tickers_list, dates_list


def create_dual_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    reg_col: str = "Target_Return",
    cls_col: str = "Target_Direction",
    seq_len: int = 30,
    sort_chronologically: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Her hisse için kayan pencere (sliding window) uygulayarak
    hem regresyon (Target_Return) hem de sınıflandırma (Target_Direction)
    hedeflerini içeren 3D tensörler üretir.

    Çıktı Şekilleri:
        X: (Örnek Sayısı, seq_len, Özellik Sayısı)
        y_reg: (Örnek Sayısı,)
        y_cls: (Örnek Sayısı,)
        tickers: Her bir örneğin ait olduğu hisse kodu
        dates: Her bir tahminin yapıldığı günün tarihi (t)
    """
    X_list: List[np.ndarray] = []
    y_reg_list: List[float] = []
    y_cls_list: List[float] = []
    tickers_list: List[str] = []
    dates_list: List[pd.Timestamp] = []

    for ticker, group in df.groupby("Ticker"):
        g = group.sort_values("Date").reset_index(drop=True)
        n_samples = len(g)

        if n_samples < seq_len:
            continue

        feat_values = g[feature_cols].values
        reg_values = g[reg_col].values
        cls_values = g[cls_col].values
        date_values = g["Date"].values

        for i in range(n_samples - seq_len + 1):
            window_x = feat_values[i: i + seq_len]
            window_y_reg = reg_values[i + seq_len - 1]
            window_y_cls = cls_values[i + seq_len - 1]
            ref_date = date_values[i + seq_len - 1]

            X_list.append(window_x)
            y_reg_list.append(window_y_reg)
            y_cls_list.append(window_y_cls)
            tickers_list.append(ticker)
            dates_list.append(pd.Timestamp(ref_date))

    if not X_list:
        empty_x = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
        empty_reg = np.empty((0,), dtype=np.float32)
        empty_cls = np.empty((0,), dtype=np.float32)
        return empty_x, empty_reg, empty_cls, [], []

    X = np.array(X_list, dtype=np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.float32)

    if sort_chronologically:
        sort_indices = np.argsort(dates_list)
        X = X[sort_indices]
        y_reg = y_reg[sort_indices]
        y_cls = y_cls[sort_indices]
        tickers_list = [tickers_list[idx] for idx in sort_indices]
        dates_list = [dates_list[idx] for idx in sort_indices]

    return X, y_reg, y_cls, tickers_list, dates_list


def load_and_preprocess_pipeline(
    period: str = "10y",
    seq_len: int = 30,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
    reg_target: str = "Target_Excess_Return",
    cls_target: str = "Target_Direction_Alpha",
    raw_df: Optional[pd.DataFrame] = None,
    df_index: Optional[pd.DataFrame] = None,
    df_macro: Optional[pd.DataFrame] = None,
    df_sector: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Yahoo Finance üzerinden BIST100 hisseleri, XU100 endeksi, Dalga-1/Dalga-2 makro verileri
    ve BIST Sektör Endekslerini (XBANK, XUSIN) çeker.
    39 durağan özellik (teknik + makro + TCMB + CDS + Sektör + MFI/OBV) üretir,
    train/val/test ayrımı ve RobustScaler uygulayarak hem 3D tensörleri hem tabular veriyi döner.
    """
    if df_index is None:
        print(f"Endeks verisi (XU100.IS) yfinance üzerinden çekiliyor (period={period})...")
        df_index = endeks_verisini_cek(period=period)

    if df_macro is None:
        print(f"Dalga-1 makro verileri (USDTRY, VIX, Brent, Altın) yfinance üzerinden çekiliyor (period={period})...")
        df_macro = makro_verileri_cek(period=period)

    if df_sector is None:
        print(f"BIST Bankacılık ve Sınai sektör verileri yfinance üzerinden çekiliyor (period={period})...")
        df_sector = sektor_verilerini_cek(period=period)

    if raw_df is None:
        print(f"BIST 100 hisse verileri yfinance üzerinden çekiliyor (period={period})...")
        raw_df = tum_hisseleri_cek(period=period)

    if period not in ["10y", "20y"] and train_end == "2021-12-31":
        train_end = None
        val_end = None

    df_panel = prepare_panel_data(raw_df)
    df_features = add_technical_features(df_panel, df_index=df_index, df_macro=df_macro, df_sector=df_sector)

    feature_cols = STOCK_SPECIFIC_COLS + MACRO_MARKET_COLS

    train_df, val_df, test_df = split_by_date(df_features, train_end=train_end, val_end=val_end)
    train_s, val_s, test_s, scaler = scale_features(train_df, val_df, test_df, feature_cols)

    X_train, y_reg_train, y_cls_train, tickers_train, dates_train = create_dual_sequences(
        train_s, feature_cols, reg_col=reg_target, cls_col=cls_target, seq_len=seq_len, sort_chronologically=True
    )
    X_val, y_reg_val, y_cls_val, tickers_val, dates_val = create_dual_sequences(
        val_s, feature_cols, reg_col=reg_target, cls_col=cls_target, seq_len=seq_len, sort_chronologically=True
    )
    X_test, y_reg_test, y_cls_test, tickers_test, dates_test = create_dual_sequences(
        test_s, feature_cols, reg_col=reg_target, cls_col=cls_target, seq_len=seq_len, sort_chronologically=True
    )

    return {
        "X_train": X_train, "y_reg_train": y_reg_train, "y_cls_train": y_cls_train,
        "tickers_train": tickers_train, "dates_train": dates_train,
        "X_val": X_val, "y_reg_val": y_reg_val, "y_cls_val": y_cls_val,
        "tickers_val": tickers_val, "dates_val": dates_val,
        "X_test": X_test, "y_reg_test": y_reg_test, "y_cls_test": y_cls_test,
        "tickers_test": tickers_test, "dates_test": dates_test,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "train_s": train_s, "val_s": val_s, "test_s": test_s,
        "feature_cols": feature_cols, "scaler": scaler,
        "reg_target": reg_target, "cls_target": cls_target
    }


# =============================================================================
# 6. ANA ÇALIŞTIRMA VE TEST HATTI (PIPELINE EXECUTION)
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("BIST LSTM / GRU VERİ HAZIRLAMA VE ÖN İŞLEME BORU HATTI")
    print("=" * 70)

    # 1. Ham Veriyi Çek
    print("\n[1/6] Yahoo Finance üzerinden 10 yıllık modern rejim (2016+) hisse, endeks, makro ve sektör verileri çekiliyor...")
    df_index = endeks_verisini_cek(period="10y")
    df_macro = makro_verileri_cek(period="10y")
    df_sector = sektor_verilerini_cek(period="10y")
    df_raw = tum_hisseleri_cek(period="10y")
    raw_df = df_raw

    # 2. Panel Veriye Dönüştür
    print("\n[2/6] Ham veri (Date, Ticker) panel formatına dönüştürülüyor...")
    df_panel = prepare_panel_data(raw_df)
    print(f"  -> Panel Veri Boyutu: {df_panel.shape}")
    print(f"  -> Toplam Farklı Hisse Sayısı: {df_panel['Ticker'].nunique()}")
    print(f"  -> Tarih Aralığı: {df_panel['Date'].min().strftime('%Y-%m-%d')} - {df_panel['Date'].max().strftime('%Y-%m-%d')}")

    # 3. Teknik, Makro ve Sektörel Özellikleri Üret
    print("\n[3/6] Durağan finansal göstergeler (Sektör + MFI/OBV Dahil) ve Target değişkenleri üretiliyor...")
    df_features = add_technical_features(df_panel, df_index=df_index, df_macro=df_macro, df_sector=df_sector)

    feature_cols = [
        # Hisse Fiyat/Hacim & Momentum (15)
        "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
        "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
        "Momentum_5d", "Momentum_10d", "Momentum_20d", "Relative_Strength_5d", "Relative_Strength_10d", "Volume_Ratio_5_20",
        # Sektörel & Kurumsal Para Akışı (5)
        "Bank_vs_Market", "Indus_vs_Market", "Sector_Relative_Strength_10d", "MFI_14_Norm", "OBV_Trend_10d",
        # Piyasa Bağlamı & Makro Rejim (5)
        "Index_Log_Return", "Excess_Return", "Cross_Rank_Return", "Market_Trend_SMA50", "Macro_Regime_Bull",
        # Dalga-1 Makro & Beta (8)
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "Beta_FX_60d", "Beta_Market_60d",
        # Dalga-2 TCMB Politika Faizi (3)
        "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
        # Dalga-2 CDS Sovereign Risk Proxy (3)
        "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "Beta_CDS_60d"
    ]
    target_col = "Target_Return"

    print(f"  -> Özellikli Veri Boyutu: {df_features.shape}")
    print(f"  -> Kullanılan Özellik Sayısı ({len(feature_cols)} adet): {feature_cols}")
    print(f"  -> Panel Veride Kalan NaN Sayısı: {df_features[feature_cols + [target_col]].isna().sum().sum()} (SIFIR OLMALI)")

    # 4. Zamansal Ayrım (Train / Validation / Test)
    print("\n[4/6] Zamansal ayrım (Train / Val / Test Split) yapılıyor...")
    train_df, val_df, test_df = split_by_date(
        df_features,
        train_end="2021-12-31",
        val_end="2023-12-31"
    )
    print(f"  -> Train Seti (<= 2021): {train_df.shape[0]:,} satır ({train_df['Ticker'].nunique()} hisse)")
    print(f"  -> Val Seti   (2022-2023): {val_df.shape[0]:,} satır ({val_df['Ticker'].nunique()} hisse)")
    print(f"  -> Test Seti  (2024+):     {test_df.shape[0]:,} satır ({test_df['Ticker'].nunique()} hisse)")

    # 5. Robust Ölçekleme (Yalnızca Train'e fit edilir)
    print("\n[5/6] RobustScaler ile veri sızıntısız ölçekleme uygulanıyor...")
    train_s, val_s, test_s, scaler = scale_features(train_df, val_df, test_df, feature_cols)

    # 6. LSTM / GRU için 3 Boyutlu Kayan Pencereleri Üret
    SEQ_LEN = 30  # Geçmiş 30 iş günü (yaklaşık 1.5 ay)
    print(f"\n[6/6] LSTM / GRU için 3D Tensörler üretiliyor (seq_len={SEQ_LEN} gün, Küresel Sıralı)...")

    X_train, y_reg_train, y_cls_train, tickers_train, dates_train = create_dual_sequences(
        train_s, feature_cols, seq_len=SEQ_LEN, sort_chronologically=True
    )
    X_val, y_reg_val, y_cls_val, tickers_val, dates_val = create_dual_sequences(
        val_s, feature_cols, seq_len=SEQ_LEN, sort_chronologically=True
    )
    X_test, y_reg_test, y_cls_test, tickers_test, dates_test = create_dual_sequences(
        test_s, feature_cols, seq_len=SEQ_LEN, sort_chronologically=True
    )

    print("\n" + "=" * 70)
    print("MODELE HAZIR 3D TENSÖR VE HEDEF MATRİS ÖZETİ")
    print("=" * 70)
    print(f"X_train Shape : {X_train.shape}  | y_reg_train: {y_reg_train.shape} | y_cls_train: {y_cls_train.shape}")
    print(f"X_val Shape   : {X_val.shape}    | y_reg_val  : {y_reg_val.shape}   | y_cls_val  : {y_cls_val.shape}")
    print(f"X_test Shape  : {X_test.shape}   | y_reg_test : {y_reg_test.shape}  | y_cls_test : {y_cls_test.shape}")
    print("=" * 70)
    print(f"X_train İlk 5 Tarihi : {[d.strftime('%Y-%m-%d') for d in dates_train[:5]]}")
    print(f"X_train Son 5 Tarihi : {[d.strftime('%Y-%m-%d') for d in dates_train[-5:]]}")
    print(f"Tarihler Kesinlikle Monotonik Artan mı?: {pd.Series(dates_train).is_monotonic_increasing}")
    print(f"X_train İçinde NaN Var mı?: {np.isnan(X_train).any()}")
    print("=" * 70)
    print("Tüm tensörler PyTorch LSTM/GRU modellerine doğrudan verilmeye hazır!")

