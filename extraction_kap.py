"""
KAP ve Finansal Haber Çekim, Önbellekleme ve BERT Duygu Analizi Modülü

BIST 100 hisseleri için:
1. Google News RSS ("{TICKER} KAP borsa") ve yfinance news akışlarını çeker.
2. Yerel Türkçe BERT sentiment motoru ile başlık ve özetleri skorlar.
3. Seans saatleri (18:00 sonrası T+1) kuralıyla sıfır gelecek sızıntılı (lookahead bias free) zaman damgası hizalaması yapar.
4. Panel veriyle birleştirilmeye hazır günlük hisse bazlı duygu öznitelikleri üretir:
   - KAP_Sentiment: Günlük ortalama duygu [-1.0, +1.0] (Haber yoksa 0.0)
   - KAP_News_Count: Günlük log haber adedi ln(1 + N)
   - KAP_Sentiment_Shock_3d: Son 3 günlük duygu momentumu
"""

import os
import sys
import json
import time
from datetime import datetime, time as dtime
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import requests
import yfinance as yf

from sentiment_engine import get_sentiment_engine

CACHE_FILE = "data/kap_sentiment_cache.json"

# Finansal Terim Düzeltmeleri (Loughran-McDonald Türkçe Domain Adaptation)
FIN_POS_TERMS = [
    "kâr", "kar", "ihracat", "büyüme", "sözleşme", "anlaşma", "sipariş", "onay",
    "yatırım", "temettü", "pay geri alım", "artış", "rekor", "kazanç", "ihale", "kapasite artışı"
]
FIN_NEG_TERMS = [
    "zarar", "ceza", "iptal", "dava", "iflas", "haciz", "durdurma", "yangın",
    "soruşturma", "kayıp", "düşüş", "fesih", "devre kesici", "tedbir", "yapılandırma"
]


def adjust_score_with_financial_lexicon(text: str, base_score: float) -> float:
    """Genel Türkçe BERT skorunu finansal terminoloji ile hassaslaştırır."""
    t_lower = text.lower()
    pos_hit = any(term in t_lower for term in FIN_POS_TERMS)
    neg_hit = any(term in t_lower for term in FIN_NEG_TERMS)

    adj = 0.0
    if pos_hit and not neg_hit:
        adj += 0.35
    elif neg_hit and not pos_hit:
        adj -= 0.35

    final_score = np.clip(base_score + adj, -1.0, 1.0)
    return round(float(final_score), 4)


def load_sentiment_cache() -> Dict[str, Any]:
    """Yerel diskteki skorlanmış haber önbelleğini yükler."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sentiment_cache(cache: Dict[str, Any]):
    """Skorlanmış haber önbelleğini diske kaydeder."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_google_news_rss(ticker: str, max_items: int = 15) -> List[Dict[str, Any]]:
    """Google News RSS üzerinden hisseye ait en güncel Türkçe haber ve KAP başlıklarını çeker."""
    query = f"{ticker} KAP borsa"
    url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    items = []
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for el in root.findall("./channel/item")[:max_items]:
                title = el.find("title").text if el.find("title") is not None else ""
                pub_date_str = el.find("pubDate").text if el.find("pubDate") is not None else ""
                link = el.find("link").text if el.find("link") is not None else ""
                if title:
                    items.append({
                        "ticker": ticker,
                        "title": title,
                        "pub_date_str": pub_date_str,
                        "source": "GoogleNews_RSS",
                        "link": link
                    })
    except Exception as e:
        pass
    return items


def fetch_yfinance_news(ticker: str) -> List[Dict[str, Any]]:
    """yfinance API üzerinden hisseye ait haber akışını çeker."""
    yahoo_ticker = f"{ticker}.IS"
    items = []
    try:
        t = yf.Ticker(yahoo_ticker)
        news = t.news
        if news:
            for n in news:
                content = n.get("content", {})
                title = content.get("title", "")
                pub_date_str = content.get("pubDate", "")
                if title:
                    items.append({
                        "ticker": ticker,
                        "title": title,
                        "pub_date_str": pub_date_str,
                        "source": "yfinance_news",
                        "link": content.get("canonicalUrl", {}).get("url", "")
                    })
    except Exception:
        pass
    return items


def parse_and_align_date(pub_date_str: str) -> Optional[pd.Timestamp]:
    """
    Haber zaman damgasını parse eder ve sıfır gelecek sızıntısı kuralıyla seans gününe hizalar:
    - 18:00 sonrası veya hafta sonu gelen haberler T+1 (bir sonraki iş gününe) yazılır.
    """
    if not pub_date_str:
        return None

    dt = None
    try:
        # ISO format: 2026-09-15T17:14:42Z
        if "T" in pub_date_str:
            dt = pd.to_datetime(pub_date_str).tz_localize(None)
        else:
            # RFC 822 format: Wed, 16 Sep 2026 14:31:57 GMT
            dt = pd.to_datetime(pub_date_str).tz_localize(None)
    except Exception:
        return None

    if dt is None:
        return None

    # Seans sonrası kontrolü (Saat 18:00 ve sonrası T+1'e kayar)
    if dt.time() >= dtime(18, 0):
        dt = dt + pd.Timedelta(days=1)

    # Hafta sonu kontrolü (Cumartesi -> Pazartesi, Pazar -> Pazartesi)
    if dt.weekday() == 5:  # Cumartesi
        dt = dt + pd.Timedelta(days=2)
    elif dt.weekday() == 6:  # Pazar
        dt = dt + pd.Timedelta(days=1)

    return dt.floor("D")


def collect_and_score_all_news(
    tickers: List[str],
    max_items_per_ticker: int = 15
) -> pd.DataFrame:
    """
    Tüm BIST 100 hisseleri için haberleri çeker, önbellekle kontrol eder,
    yeni olanları Türkçe BERT ile skorlar ve tarih-hisse panelini döner.
    """
    cache = load_sentiment_cache()
    engine = get_sentiment_engine()

    raw_items = []
    print(f"[KAP & Haber Modülü] {len(tickers)} hisse için haber ve bildirim akışı taranıyor...")

    for i, ticker in enumerate(tickers):
        gn_items = fetch_google_news_rss(ticker, max_items=max_items_per_ticker)
        yf_items = fetch_yfinance_news(ticker)
        all_ticker_items = gn_items + yf_items

        for item in all_ticker_items:
            item_id = f"{ticker}_{item['title'][:40]}"
            if item_id in cache:
                cached = cache[item_id]
                item["score"] = cached["score"]
                item["aligned_date"] = pd.Timestamp(cached["aligned_date"])
            else:
                aligned_dt = parse_and_align_date(item["pub_date_str"])
                if aligned_dt is None:
                    aligned_dt = pd.Timestamp.today().floor("D")

                base_score = engine.score_single(item["title"])
                fin_score = adjust_score_with_financial_lexicon(item["title"], base_score)

                item["score"] = fin_score
                item["aligned_date"] = aligned_dt

                cache[item_id] = {
                    "score": fin_score,
                    "aligned_date": aligned_dt.strftime("%Y-%m-%d"),
                    "title": item["title"]
                }
            raw_items.append(item)

        if (i + 1) % 25 == 0 or (i + 1) == len(tickers):
            print(f"  -> {i + 1}/{len(tickers)} hisse tamamlandı. (Toplam toplanan haber: {len(raw_items)})")

    save_sentiment_cache(cache)

    if not raw_items:
        return pd.DataFrame(columns=["Date", "Ticker", "KAP_Sentiment", "KAP_News_Count"])

    df_news = pd.DataFrame(raw_items)
    df_news["Date"] = pd.to_datetime(df_news["aligned_date"])
    df_news["Ticker"] = df_news["ticker"].str.strip().str.upper()

    # Günlük hisse bazında ortalama sentiment ve haber sayısı
    agg_df = df_news.groupby(["Date", "Ticker"]).agg(
        KAP_Sentiment=("score", "mean"),
        Raw_News_Count=("score", "count")
    ).reset_index()

    agg_df["KAP_News_Count"] = np.log1p(agg_df["Raw_News_Count"])
    agg_df["KAP_Sentiment"] = agg_df["KAP_Sentiment"].clip(-1.0, 1.0)

    return agg_df[["Date", "Ticker", "KAP_Sentiment", "KAP_News_Count"]]


def merge_kap_features_into_panel(df_panel: pd.DataFrame, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    df_panel DataFrame'ine KAP_Sentiment, KAP_News_Count ve KAP_Sentiment_Shock_3d özniteliklerini ekler.
    Haber olmayan günlerde doğal finansal nötrlük (Sentiment=0.0, News_Count=0.0) korunur.
    """
    df = df_panel.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    unique_tickers = tickers if tickers else df["Ticker"].unique().tolist()
    news_df = collect_and_score_all_news(unique_tickers)

    if not news_df.empty:
        news_df["Date"] = pd.to_datetime(news_df["Date"])
        df = pd.merge(df, news_df, on=["Date", "Ticker"], how="left")
    else:
        df["KAP_Sentiment"] = 0.0
        df["KAP_News_Count"] = 0.0

    # Eksik günler = Haber/bildirim olmayan günler -> Doğal nötrlük (0.0)
    df["KAP_Sentiment"] = df["KAP_Sentiment"].fillna(0.0)
    df["KAP_News_Count"] = df["KAP_News_Count"].fillna(0.0)

    # 3 Günlük Üssel Duygu Sönümlemesi (Post-Announcement Drift & Event Decay)
    # Yeni haber yoksa duygu aniden sıfırlanmaz, piyasa sindirme süresini modellemek için
    # her gün %50 oranında sönümlenir (lambda = 0.5)
    groups = []
    for ticker, group in df.groupby("Ticker", group_keys=False):
        g = group.sort_values("Date").copy()
        raw_s = g["KAP_Sentiment"].values
        decayed_s = np.zeros_like(raw_s, dtype=np.float32)
        curr = 0.0
        for idx, val in enumerate(raw_s):
            if abs(val) > 1e-4:
                curr = val
            else:
                curr = curr * 0.5  # 1 günlük yarı ömür sönümlemesi
            decayed_s[idx] = curr

        g["KAP_Sentiment"] = np.clip(decayed_s, -1.0, 1.0)
        s_roll3 = g["KAP_Sentiment"].rolling(window=3, min_periods=1).mean()
        s_roll20 = g["KAP_Sentiment"].rolling(window=20, min_periods=1).mean()
        g["KAP_Sentiment_Shock_3d"] = (s_roll3 - s_roll20).fillna(0.0).clip(-1.0, 1.0)
        groups.append(g)

    df_out = pd.concat(groups, ignore_index=True)
    df_out = df_out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df_out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample_tickers = ["THYAO", "ASELS", "GARAN", "FROTO", "EREGL"]
    print("=" * 80)
    print("KAP & HABER TÜRKÇE BERT ENTEGRASYON VE ÇEKİM TESTİ")
    print("=" * 80)
    test_df = collect_and_score_all_news(sample_tickers, max_items_per_ticker=5)
    print(test_df.tail(15).to_string(index=False))
    print("=" * 80)
