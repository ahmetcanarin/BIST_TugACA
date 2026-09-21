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
from extraction import tum_hisseleri_cek, endeks_verisini_cek, makro_verileri_cek
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
    """
    if df_index is None or df_index.empty:
        return pd.DataFrame(columns=["Date", "Index_Log_Return", "Index_Volatility_20", "Target_Index_Return"])

    idx = df_index.copy()
    if isinstance(idx.columns, pd.MultiIndex):
        if "Close" in idx.columns.levels[0]:
            close_s = idx["Close"].iloc[:, 0] if isinstance(idx["Close"], pd.DataFrame) else idx["Close"]
        elif "Close" in idx.columns.names or any(c == "Close" for c in idx.columns.get_level_values(1)):
            close_s = idx.xs("Close", axis=1, level=1).iloc[:, 0]
        else:
            close_s = idx.iloc[:, 0]
    elif "Close" in idx.columns:
        close_s = idx["Close"]
    else:
        close_s = idx.iloc[:, 0]

    date_idx = pd.to_datetime(idx.index)
    if date_idx.tz is not None:
        date_idx = date_idx.tz_localize(None)

    idx_df = pd.DataFrame({
        "Date": date_idx,
        "Index_Close": pd.to_numeric(close_s.values.flatten() if hasattr(close_s, "values") else close_s, errors="coerce")
    }).dropna(subset=["Index_Close"]).sort_values("Date").reset_index(drop=True)

    idx_df["Index_Log_Return"] = np.log(idx_df["Index_Close"] / idx_df["Index_Close"].shift(1))
    idx_df["Index_Volatility_20"] = idx_df["Index_Log_Return"].rolling(window=20).std()

    # Endeksin bir sonraki günkü yüzde getirisi (Hisse bazında Target_Excess_Return için)
    next_idx_close = idx_df["Index_Close"].shift(-1)
    idx_df["Target_Index_Return"] = ((next_idx_close - idx_df["Index_Close"]) / idx_df["Index_Close"]) * 100.0
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


def add_technical_features(
    df_panel: pd.DataFrame,
    df_index: Optional[pd.DataFrame] = None,
    df_macro: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Ham fiyatlar yerine durağan teknik ve makro göstergeleri türetir:
    
    12 Klasik Özellik:
    - Log_Return, HL_Spread, CO_Return, Log_Volume, Volume_Change,
      SMA10_Ratio, SMA30_Ratio, Volatility_20, RSI_Norm,
      Index_Log_Return, Excess_Return, Cross_Rank_Return
      
    8 Dalga-1 Makro Özellik:
    - USDTRY_Return, USDTRY_Vol_20, VIX_Level, VIX_Change,
      Brent_Return, Gold_Return, Beta_FX_60d, Beta_Market_60d

    Hedef Değişkenler (Target):
    - Target_Return: Bir sonraki günün yüzde getirisi (Ham getiri)
    - Target_Excess_Return: Endeks üstü getiri (Alfa)
    - Target_Direction: Ham yükseliş (1 / 0)
    - Target_Direction_Alpha: Endeksi yenme (1 / 0)
    - Target_Rank: Kesitsel sıralama [-0.5, +0.5]
    """
    df_p = df_panel.copy()
    df_p["Date"] = pd.to_datetime(df_p["Date"])
    if df_p["Date"].dt.tz is not None:
        df_p["Date"] = df_p["Date"].dt.tz_localize(None)

    # Endeks verisini birleştir
    if df_index is not None and not df_index.empty:
        idx_df = process_index_data(df_index)
        df_p = pd.merge(df_p, idx_df[["Date", "Index_Log_Return", "Index_Volatility_20", "Target_Index_Return"]], on="Date", how="left")
    else:
        df_p["Index_Log_Return"] = 0.0
        df_p["Index_Volatility_20"] = 0.0
        df_p["Target_Index_Return"] = 0.0

    df_p["Index_Log_Return"] = df_p["Index_Log_Return"].fillna(0.0)
    df_p["Index_Volatility_20"] = df_p["Index_Volatility_20"].fillna(0.0)
    df_p["Target_Index_Return"] = df_p["Target_Index_Return"].fillna(0.0)

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

    # KAP ve Finansal Haber Türkçe BERT Duygu Verilerini Birleştir (Dalga-2 Adım 3 & V4)
    try:
        from extraction_kap import merge_kap_features_into_panel
        df_p = merge_kap_features_into_panel(df_p)
    except Exception as e:
        print(f"  [!] KAP duygu verisi eklenirken hata: {e}. Nötr sıfırlar atanıyor.")
        df_p["KAP_Sentiment"] = 0.0
        df_p["KAP_News_Count"] = 0.0
        df_p["KAP_Sentiment_Shock_3d"] = 0.0

    groups = []
    for ticker, group in df_p.groupby("Ticker", group_keys=False):
        g = group.sort_values("Date").copy()

        # 1. Getiri ve Fiyat Oranları (Scale-free)
        g["Log_Return"] = np.log(g["Close"] / g["Close"].shift(1))
        g["HL_Spread"] = (g["High"] - g["Low"]) / g["Close"]
        g["CO_Return"] = (g["Close"] - g["Open"]) / g["Open"]
        g["Excess_Return"] = g["Log_Return"] - g["Index_Log_Return"]

        # 2. Hacim Dinamikleri
        g["Log_Volume"] = np.log1p(g["Volume"])
        g["Volume_Change"] = g["Log_Volume"].diff()

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

        # 6. Dinamik Makro Hassasiyetler (Rolling Betalar)
        cov_fx = g["Log_Return"].rolling(window=60, min_periods=5).cov(g["USDTRY_Return"])
        var_fx = g["USDTRY_Return"].rolling(window=60, min_periods=5).var()
        g["Beta_FX_60d"] = (cov_fx / (var_fx + 1e-7)).fillna(0.0).clip(-5.0, 5.0)

        cov_m = g["Log_Return"].rolling(window=60, min_periods=5).cov(g["Index_Log_Return"])
        var_m = g["Index_Log_Return"].rolling(window=60, min_periods=5).var()
        g["Beta_Market_60d"] = (cov_m / (var_m + 1e-7)).fillna(1.0).clip(-3.0, 5.0)

        cov_cds = g["Log_Return"].rolling(window=60, min_periods=5).cov(g["CDS_Spread_Diff"])
        var_cds = g["CDS_Spread_Diff"].rolling(window=60, min_periods=5).var()
        g["Beta_CDS_60d"] = (cov_cds / (var_cds + 1e-7)).fillna(0.0).clip(-5.0, 5.0)

        # 7. Event-Driven Haber Momentum ve Şok Dinamikleri
        s_roll3 = g["KAP_Sentiment"].rolling(window=3, min_periods=1).mean()
        s_roll20 = g["KAP_Sentiment"].rolling(window=20, min_periods=1).mean()
        g["KAP_Sentiment_Shock_3d"] = (s_roll3 - s_roll20).fillna(0.0).clip(-1.0, 1.0)

        # ---------------------------------------------------------------------
        # HEDEF DEĞİŞKENLER (TARGETS): T+1 Getirisi ve Alfa
        # ---------------------------------------------------------------------
        next_close = g["Close"].shift(-1)
        g["Target_Return"] = ((next_close - g["Close"]) / g["Close"]) * 100.0
        g["Target_Excess_Return"] = g["Target_Return"] - g["Target_Index_Return"]
        g["Target_Direction"] = (g["Target_Return"] > 0).astype(float)
        g["Target_Direction_Alpha"] = (g["Target_Excess_Return"] > 0).astype(float)

        groups.append(g)

    df_features = pd.concat(groups, ignore_index=True)

    # 8. Kesitsel Sıralamalar (Cross-Sectional Percentile Rank: [-0.5, +0.5])
    df_features["Cross_Rank_Return"] = df_features.groupby("Date")["Log_Return"].rank(pct=True) - 0.5
    df_features["Target_Rank"] = df_features.groupby("Date")["Target_Return"].rank(pct=True) - 0.5
    df_features["Target_Excess_Rank"] = df_features.groupby("Date")["Target_Excess_Return"].rank(pct=True) - 0.5

    feature_cols = [
        # Hisse Fiyat/Hacim (9)
        "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
        "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
        # Piyasa Bağlamı (3)
        "Index_Log_Return", "Excess_Return", "Cross_Rank_Return",
        # Dalga-1 Makro & Beta (8)
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "Beta_FX_60d", "Beta_Market_60d",
        # Dalga-2 TCMB Politika Faizi (3)
        "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
        # Dalga-2 CDS Sovereign Risk Proxy (3)
        "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "Beta_CDS_60d",
        # Dalga-2 Adım 3: Event-Driven KAP & Haber Türkçe BERT Duygu Öznitelikleri (3)
        "KAP_Sentiment", "KAP_News_Count", "KAP_Sentiment_Shock_3d"
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


# =============================================================================
# 4. ÖLÇEKLEME (ROBUST SCALER - DATA LEAKAGE OLMADAN)
# =============================================================================
def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Öznitelikleri RobustScaler (medyan ve IQR tabanlı) ile ölçekler.
    Kritik İlke: Scaler YALNIZCA Train verisi üzerinde fit edilir.
    Validation ve Test verisi sadece transform edilir (Veri sızıntısını engeller).
    """
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

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
    df_macro: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Yahoo Finance üzerinden BIST100 hisseleri, XU100 endeksi ve Dalga-1/Dalga-2 makro verilerini çeker.
    29 durağan özellik (teknik + makro + TCMB + CDS + KAP Sentiment Shock) üretir, train/val/test ayrımı ve RobustScaler uygulayarak
    hem 3D tensörleri (GRU için) hem de tabular veri çerçevelerini (LightGBM için) döner.
    """
    if df_index is None:
        print(f"Endeks verisi (XU100.IS) yfinance üzerinden çekiliyor (period={period})...")
        df_index = endeks_verisini_cek(period=period)

    if df_macro is None:
        print(f"Dalga-1 makro verileri (USDTRY, VIX, Brent, Altın) yfinance üzerinden çekiliyor (period={period})...")
        df_macro = makro_verileri_cek(period=period)

    if raw_df is None:
        print(f"BIST 100 hisse verileri yfinance üzerinden çekiliyor (period={period})...")
        raw_df = tum_hisseleri_cek(period=period)

    if period not in ["10y", "20y"] and train_end == "2021-12-31":
        train_end = None
        val_end = None

    df_panel = prepare_panel_data(raw_df)
    df_features = add_technical_features(df_panel, df_index=df_index, df_macro=df_macro)

    assert "KAP_Sentiment_Shock_3d" in df_features.columns, "KAP_Sentiment_Shock_3d sütunu df_features içinde bulunamadı!"
    assert df_features["KAP_Sentiment_Shock_3d"].isna().sum() == 0, "KAP_Sentiment_Shock_3d içinde NaN değer bulundu!"

    feature_cols = [
        "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
        "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
        "Index_Log_Return", "Excess_Return", "Cross_Rank_Return",
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "Beta_FX_60d", "Beta_Market_60d",
        "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
        "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "Beta_CDS_60d",
        "KAP_Sentiment", "KAP_News_Count", "KAP_Sentiment_Shock_3d"
    ]

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
    print("\n[1/6] Yahoo Finance üzerinden 10 yıllık modern rejim (2016+) hisse, endeks ve makro veriler çekiliyor...")
    df_index = endeks_verisini_cek(period="10y")
    df_macro = makro_verileri_cek(period="10y")
    df_raw = tum_hisseleri_cek(period="10y")

    # 2. Panel Veriye Dönüştür
    print("\n[2/6] Ham veri (Date, Ticker) panel formatına dönüştürülüyor...")
    df_panel = prepare_panel_data(df_raw)
    print(f"  -> Panel Veri Boyutu: {df_panel.shape}")
    print(f"  -> Toplam Farklı Hisse Sayısı: {df_panel['Ticker'].nunique()}")
    print(f"  -> Tarih Aralığı: {df_panel['Date'].min().strftime('%Y-%m-%d')} - {df_panel['Date'].max().strftime('%Y-%m-%d')}")

    # 3. Teknik ve Durağan Özellikleri Üret
    print("\n[3/6] Durağan finansal göstergeler (29 Özellik) ve Target değişkenleri üretiliyor...")
    df_features = add_technical_features(df_panel, df_index=df_index, df_macro=df_macro)

    feature_cols = [
        "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
        "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
        "Index_Log_Return", "Excess_Return", "Cross_Rank_Return",
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "Beta_FX_60d", "Beta_Market_60d",
        "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
        "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "Beta_CDS_60d",
        "KAP_Sentiment", "KAP_News_Count", "KAP_Sentiment_Shock_3d"
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

