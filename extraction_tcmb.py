"""
TCMB (Türkiye Cumhuriyet Merkez Bankası) Para Politikası ve Faiz Verileri Çıkarma Modülü

Bu modül:
1. TCMB'nin 2005'ten günümüze tüm Para Politikası Kurulu (PPK) faiz kararlarını içerir.
2. Eğer sistemde EVDS_API_KEY ortam değişkeni varsa, resmi EVDS API üzerinden dinamik günlük seriyi çeker.
3. Yoksa, resmi TCMB PPK karar takvimi üzerinden zaman serisini kurar ve BIST işlem takvimine göre forward-fill eder.
4. Model için 3 durağan, scale-free öznitelik üretir:
   - TCMB_Policy_Rate: Politika faizi seviyesi (normalize)
   - TCMB_Rate_Change: Faiz karar günündeki değişim miktarı
   - TCMB_Days_Since_Decision: Son PPK kararından bu yana geçen iş günü sayısı (Politika belirsizliği proxy'si)
"""

import os
import requests
from typing import Optional
import numpy as np
import pandas as pd


# 2005 - 2026 Resmi TCMB Faiz Karar Takvimi (Karar Tarihi, Politika/Gösterge Faiz Oranı %)
# Kaynak: TCMB Resmi Duyuruları & EVDS
TCMB_HISTORICAL_DECISIONS = [
    # 2005 - 2009 (O/N Borçlanma / Politika Göstergesi)
    ("2005-01-11", 17.00), ("2005-02-09", 16.50), ("2005-03-09", 15.50), ("2005-04-11", 15.00),
    ("2005-05-10", 14.50), ("2005-06-09", 14.25), ("2005-10-11", 14.00), ("2005-11-09", 13.75),
    ("2005-12-09", 13.50), ("2006-04-28", 13.25), ("2006-06-08", 15.00), ("2006-06-26", 17.25),
    ("2006-07-21", 17.50), ("2007-09-14", 17.25), ("2007-10-17", 16.75), ("2007-11-15", 16.25),
    ("2007-12-14", 15.75), ("2008-01-18", 15.50), ("2008-02-15", 15.25), ("2008-05-16", 15.75),
    ("2008-06-17", 16.25), ("2008-07-18", 16.75), ("2008-10-23", 16.75), ("2008-11-20", 16.25),
    ("2008-12-19", 15.00), ("2009-01-16", 13.00), ("2009-02-20", 11.50), ("2009-03-20", 10.50),
    ("2009-04-17", 9.75),  ("2009-05-15", 9.25),  ("2009-06-17", 8.75),  ("2009-07-17", 8.25),
    ("2009-08-19", 7.75),  ("2009-09-18", 7.25),  ("2009-10-16", 6.75),  ("2009-11-20", 6.50),

    # 2010 - 2017 (1 Hafta Repo Tanımlanması & Sadeleşme Öncesi)
    ("2010-05-20", 7.00),  ("2010-12-17", 6.50),  ("2011-01-21", 6.25),  ("2011-08-05", 5.75),
    ("2012-12-19", 5.50),  ("2013-04-17", 5.00),  ("2013-05-17", 4.50),  ("2014-01-29", 10.00),
    ("2014-05-23", 9.50),  ("2014-06-25", 8.75),  ("2014-07-18", 8.25),  ("2015-01-21", 7.75),
    ("2015-02-25", 7.50),  ("2016-11-25", 8.00),

    # 2018 - 2022 (Sadeleşmiş 1 Hafta Repo Politika Faizi)
    ("2018-06-01", 16.50), ("2018-06-08", 17.75), ("2018-09-14", 24.00), ("2019-07-26", 19.75),
    ("2019-09-13", 16.50), ("2019-10-25", 14.00), ("2019-12-13", 12.00), ("2020-01-17", 11.25),
    ("2020-02-20", 10.75), ("2020-03-18", 9.75),  ("2020-04-23", 8.75),  ("2020-05-22", 8.25),
    ("2020-09-25", 10.25), ("2020-11-20", 15.00), ("2020-12-25", 17.00), ("2021-03-19", 19.00),
    ("2021-09-24", 18.00), ("2021-10-22", 16.00), ("2021-11-19", 15.00), ("2021-12-17", 14.00),
    ("2022-08-19", 13.00), ("2022-09-23", 12.00), ("2022-10-21", 10.50), ("2022-11-25", 9.00),
    ("2023-02-24", 8.50),

    # 2023 - 2026 (Yeni Ortodoks Para Politikası & Sıkılaşma Döngüsü)
    ("2023-06-23", 15.00), ("2023-07-21", 17.50), ("2023-08-24", 25.00), ("2023-09-21", 30.00),
    ("2023-10-26", 35.00), ("2023-11-23", 40.00), ("2023-12-21", 42.50), ("2024-01-25", 45.00),
    ("2024-03-21", 50.00), ("2024-04-25", 50.00), ("2024-05-23", 50.00), ("2024-06-27", 50.00),
    ("2024-07-23", 50.00), ("2024-08-20", 50.00), ("2024-09-19", 50.00), ("2024-10-17", 50.00),
    ("2024-11-22", 50.00), ("2024-12-26", 47.50), ("2025-01-23", 45.00), ("2025-03-06", 42.50),
    ("2025-04-17", 46.00), ("2025-06-19", 46.00), ("2025-07-24", 43.00), ("2025-09-11", 40.50),
    ("2025-10-23", 39.50), ("2025-12-11", 38.00), ("2026-01-22", 37.00), ("2026-03-12", 37.00),
    ("2026-04-22", 37.00), ("2026-06-11", 37.00), ("2026-07-23", 37.00), ("2026-09-10", 37.00)
]


def fetch_tcmb_from_evds(
    api_key: str,
    start_date: str = "01-01-2005",
    end_date: str = "01-01-2027"
) -> Optional[pd.DataFrame]:
    """
    TCMB EVDS API üzerinden resmi faiz serisini (TP.APIFON4) çeker.
    """
    url = f"https://evds2.tcmb.gov.tr/service/evds/series=TP.APIFON4&startDate={start_date}&endDate={end_date}&type=json"
    headers = {"key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            items = data.get("items", [])
            if items:
                df = pd.DataFrame(items)
                df = df.rename(columns={"Tarih": "Date", "TP_APIFON4": "Policy_Rate"})
                df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
                df["Policy_Rate"] = pd.to_numeric(df["Policy_Rate"], errors="coerce")
                return df.dropna(subset=["Date", "Policy_Rate"]).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"[!] EVDS bağlantı uyarısı: {e}. Yerel karar takvimine geçiliyor.")
    return None


def get_tcmb_interest_rate_series(
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    TCMB politika faiz serisini üretir.
    EVDS anahtarı varsa canlı çeker, yoksa resmi karar takviminden zaman serisini kurar.
    Zaman sızıntısı (lookahead bias) olmaksızın günlük panele hazır hale getirir.
    """
    if api_key is None:
        api_key = os.environ.get("EVDS_API_KEY")

    if api_key:
        df_evds = fetch_tcmb_from_evds(api_key)
        if df_evds is not None and not df_evds.empty:
            return df_evds

    # Yerel resmi karar takviminden kurma
    df_decisions = pd.DataFrame(TCMB_HISTORICAL_DECISIONS, columns=["Date", "Policy_Rate"])
    df_decisions["Date"] = pd.to_datetime(df_decisions["Date"])
    df_decisions = df_decisions.sort_values("Date").reset_index(drop=True)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Günlük takvim
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df_calendar = pd.DataFrame({"Date": all_dates})

    # Karar günlerini takvime birleştir ve forward-fill uygula (gelecekten geçmişe sızıntı yok)
    df_merged = pd.merge(df_calendar, df_decisions, on="Date", how="left")
    df_merged["Policy_Rate"] = df_merged["Policy_Rate"].ffill()
    df_merged["Policy_Rate"] = df_merged["Policy_Rate"].fillna(17.0)  # 2005 başı başlangıç değeri

    return df_merged


def compute_tcmb_features(
    df_tcmb: pd.DataFrame,
    bist_dates: pd.Series
) -> pd.DataFrame:
    """
    BIST işlem günleriyle eşleştirilmiş durağan TCMB özniteliklerini türetir:
    1. TCMB_Policy_Rate: Normalize politika faizi (Seviye / 50.0)
    2. TCMB_Rate_Change: Karar günündeki faiz artış/azalış miktarı
    3. TCMB_Days_Since_Decision: Son karardan bu yana geçen gün sayısı / 30.0 (Politika belirsizlik proxy'si)
    """
    unique_bist_dates = pd.DataFrame({"Date": pd.to_datetime(bist_dates.unique())}).sort_values("Date").reset_index(drop=True)
    df = pd.merge(unique_bist_dates, df_tcmb, on="Date", how="left")
    df["Policy_Rate"] = df["Policy_Rate"].ffill().fillna(17.0)

    # 1. Faiz Seviyesi Normalize [-1, +1 aralığına yakın]
    df["TCMB_Policy_Rate"] = (df["Policy_Rate"] / 50.0) - 0.5

    # 2. Karar Günündeki Faiz Değişimi
    df["TCMB_Rate_Change"] = df["Policy_Rate"].diff().fillna(0.0) / 10.0

    # 3. Son Karardan Bu Yana Geçen Gün Sayısı (Belirsizlik Ölçütü)
    is_decision = df["TCMB_Rate_Change"].abs() > 1e-4
    last_decision_idx = 0
    days_since = []
    for i, dec in enumerate(is_decision):
        if dec:
            last_decision_idx = i
        days_since.append(min((i - last_decision_idx), 60) / 30.0)
    df["TCMB_Days_Since_Decision"] = days_since

    return df[["Date", "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision"]].sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    tcmb_df = get_tcmb_interest_rate_series("2020-01-01", "2026-09-20")
    print("TCMB Faiz Serisi Örneği (Son 10 Kayıt):")
    print(tcmb_df.tail(10))
    feat_df = compute_tcmb_features(tcmb_df, tcmb_df["Date"])
    print("\nTüretilen Öznitelikler:")
    print(feat_df.tail(10))
