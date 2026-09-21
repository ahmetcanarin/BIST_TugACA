"""
BIST 100 hisse senetleri ve endeks verilerini Yahoo Finance üzerinden çeken modül.

Not: BIST 100 endeksinin içeriği periyodik olarak (yılda birkaç kez) güncellenir.
En güncel listeyi Borsa İstanbul'un resmi sitesinden (borsaistanbul.com/tr/endeksler)
kontrol edip BIST100_TICKERS listesini güncelleyebilirsiniz.
"""

import time
from typing import List, Optional
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# BIST 100 hisse kodları (Yahoo Finance formatı: KOD.IS)
# ---------------------------------------------------------------------------
BIST100_TICKERS: List[str] = [
    "AEFES", "AGHOL", "AKBNK", "AKSA", "AKSEN", "ALARK", "ALFAS", "ANSGR",
    "ARCLK", "ASELS", "ASTOR", "BERA", "BIMAS", "BINHO", "BRSAN", "BRYAT",
    "BTCIM", "BUCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DOAS", "DOHOL",
    "ECILC", "EGEEN", "EKGYO", "ENERY", "ENJSA", "ENKAI", "EREGL", "EUPWR",
    "FROTO", "GARAN", "GESAN", "GUBRF", "GWIND", "HALKB", "HEKTS", "IPEKE",
    "ISCTR", "ISMEN", "IZDMC", "IZENR", "KARSN", "KAYSE", "KCAER", "KCHOL",
    "KLSER", "KONTR", "KONYA", "KORDS", "KOZAA", "KOZAL", "KRDMD", "MAVI",
    "MGROS", "MIATK", "ODAS", "OTKAR", "OYAKC", "PASEU", "PENTA", "PETKM",
    "PGSUS", "QUAGR", "REEDR", "SAHOL", "SASA", "SISE", "SKBNK", "SMRTG",
    "SOKM", "TABGD", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TSKB",
    "TTKOM", "TTRAK", "TUKAS", "TUPRS", "TURSG", "ULKER", "VAKBN", "VESBE",
    "VESTL", "YEOTK", "YKBNK", "YYLGD", "ZOREN",
]

INDEX_TICKER: str = "XU100.IS"  # BIST 100 endeksinin kendisi

# Dalga-1 ve Dalga-2 Makro Varlıklar (yfinance üzerinden sıfır maliyetli ve güvenilir)
MACRO_TICKERS = {
    "USDTRY": "USDTRY=X",  # Dolar/TL Kuru
    "VIX": "^VIX",         # CBOE Volatilite Endeksi (Global Risk İştahı)
    "BRENT": "BZ=F",       # Brent Petrol Vadeli (Enerji & Maliyet)
    "GOLD": "GC=F",        # Ons Altın Vadeli (Emtia & Güvenli Liman)
    "TUR": "TUR",          # iShares MSCI Turkey ETF (USD bazlı Türkiye Risk Primi)
    "EEM": "EEM"           # iShares MSCI Emerging Markets ETF (Gelişmekte Olan Ülkeler Benchmark)
}


def bist_kodu_to_yahoo(kod: str) -> str:
    """BIST kodunu Yahoo Finance formatına çevirir (ör: THYAO -> THYAO.IS)."""
    return f"{kod.strip().upper()}.IS"


def tum_hisseleri_cek(
    tickers: Optional[List[str]] = None,
    period: str = "5d",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    BIST 100'deki hisselerin açılış/kapanış/yüksek/düşük/hacim verilerini tek seferde çeker.

    :param tickers: Çekilecek hisse kodları listesi (Varsayılan: BIST100_TICKERS)
    :param period: "1d", "5d", "1mo", "3mo", "6mo", "1y"... (ne kadar geriye gidileceği)
    :param interval: "1d" (günlük), "1h" (saatlik), "15m" (15 dakikalık) vb.
    :return: İndirilen verileri içeren pandas DataFrame
    """
    if tickers is None:
        tickers = BIST100_TICKERS

    yahoo_kodlari = [bist_kodu_to_yahoo(k) for k in tickers]
    print(f"{len(yahoo_kodlari)} hisse için veri çekiliyor... (biraz sürebilir)")

    # yfinance toplu indirme - tek istekte tüm hisseleri çeker
    veri = yf.download(
        tickers=yahoo_kodlari,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=True,
    )
    return veri


def son_gun_ozet_tablosu(
    veri: pd.DataFrame,
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Çekilen veriden her hisse için SON GÜNÜN açılış / kapanış / yüksek / düşük / hacim
    bilgisini tek satırlık, okunabilir bir tabloya dönüştürür.

    :param veri: yf.download ile indirilmiş MultiIndex veya gruplanmış DataFrame
    :param tickers: İşlenecek hisse kodları listesi
    :return: Özet tablosu DataFrame'i
    """
    if tickers is None:
        tickers = BIST100_TICKERS

    kayitlar = []
    for kod in tickers:
        yahoo_kod = bist_kodu_to_yahoo(kod)
        try:
            if isinstance(veri.columns, pd.MultiIndex):
                if yahoo_kod in veri.columns.levels[0]:
                    hisse_verisi = veri[yahoo_kod].dropna(how="all")
                elif yahoo_kod in veri.columns:
                    hisse_verisi = veri[yahoo_kod].dropna(how="all")
                else:
                    continue
            else:
                if yahoo_kod in veri:
                    hisse_verisi = veri[[yahoo_kod]].dropna(how="all")
                else:
                    continue

            if hisse_verisi.empty:
                continue

            son_satir = hisse_verisi.iloc[-1]
            
            # Sütun isimlerinin güvenli erişimi (büyük/küçük harf duyarlılığı için)
            open_val = son_satir.get("Open")
            close_val = son_satir.get("Close")
            high_val = son_satir.get("High")
            low_val = son_satir.get("Low")
            vol_val = son_satir.get("Volume")

            acilis = float(open_val) if pd.notna(open_val) else None
            kapanis = float(close_val) if pd.notna(close_val) else None
            yuksek = float(high_val) if pd.notna(high_val) else None
            dusuk = float(low_val) if pd.notna(low_val) else None
            hacim = int(vol_val) if pd.notna(vol_val) else None

            degisim_pct = None
            if acilis and kapanis and acilis != 0:
                degisim_pct = round(((kapanis - acilis) / acilis) * 100, 2)

            kayitlar.append({
                "Hisse": kod,
                "Tarih": hisse_verisi.index[-1].strftime("%Y-%m-%d"),
                "Acilis": round(acilis, 2) if acilis is not None else None,
                "Kapanis": round(kapanis, 2) if kapanis is not None else None,
                "Yuksek": round(yuksek, 2) if yuksek is not None else None,
                "Dusuk": round(dusuk, 2) if dusuk is not None else None,
                "Hacim": hacim,
                "Degisim_%": degisim_pct,
            })
        except (KeyError, IndexError, ValueError) as err:
            print(f"  Uyari: {kod} icin veri islenemedi ({err}), atlaniyor.")
            continue

    if not kayitlar:
        return pd.DataFrame(columns=[
            "Hisse", "Tarih", "Acilis", "Kapanis", "Yuksek", "Dusuk", "Hacim", "Degisim_%"
        ])

    return pd.DataFrame(kayitlar).sort_values("Hisse").reset_index(drop=True)


def endeks_verisini_cek(period: str = "5d") -> pd.DataFrame:
    """
    BIST 100 endeksinin (XU100.IS) kendi açılış / kapanış verisini çeker.

    :param period: Ne kadar geriye dönük veri çekileceği
    :return: Endeks fiyat DataFrame'i
    """
    return yf.download(INDEX_TICKER, period=period, interval="1d", progress=False)


def makro_verileri_cek(period: str = "20y") -> pd.DataFrame:
    """
    Dalga-1 makro göstergeleri (USDTRY, VIX, Brent, Altın) yfinance üzerinden çeker.

    :param period: Geriye dönük zaman periyodu (Varsayılan: '20y')
    :return: Tarih bazlı çoklu sütunlu makro fiyat tablosu
    """
    tickers = list(MACRO_TICKERS.values())
    print(f"Makro veriler çekiliyor: {list(MACRO_TICKERS.keys())} (period={period})...")
    df_macro = yf.download(tickers, period=period, interval="1d", group_by="ticker", auto_adjust=False, progress=False)
    return df_macro


if __name__ == "__main__":
    # 1) Tüm BIST100 hisselerinin son 5 günlük verisini çek
    ham_veri = tum_hisseleri_cek(period="5d", interval="1d")

    # 2) Son günün özet tablosunu çıkar
    ozet = son_gun_ozet_tablosu(ham_veri)
    print("\n=== BIST 100 - Son Gün Açılış/Kapanış Özeti ===")
    if not ozet.empty:
        print(ozet.to_string(index=False))
    else:
        print("Özet veri bulunamadı.")

    # 3) CSV olarak kaydet
    cikti_dosyasi = "bist100_son_gun.csv"
    ozet.to_csv(cikti_dosyasi, index=False, encoding="utf-8-sig")
    print(f"\nSonuc '{cikti_dosyasi}' dosyasina kaydedildi.")

    # 4) BIST 100 endeksinin kendisini de çek
    print("\n=== BIST 100 Endeksi (XU100) ===")
    endeks = endeks_verisini_cek(period="5d")
    print(endeks.tail())

