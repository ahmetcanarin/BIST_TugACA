"""
BIST 100 Günlük 09:55 Canlı Sanal İcra ve Paper Trading Koşucusu (live_daily_runner.py)

Bu betik:
1. Her borsa iş günü sabahı saat 09:55'te (açılış öncesi) otomatik veya manuel çalıştırılır.
2. Portföy Risk Devre Kesici Kontrolü:
   - Son 5 günlük zirve özsermaye düşüşünü inceler.
   - Eğer haftalık kayıp >= -%5.0 ise devre kesiciyi tetikler; portföy 3 gün nakitte (PPF / Gecelik Repo) kalır.
3. KAP Gatekeeper ve Şampiyon Model Doğrulaması:
   - 'output/daily_signals.json' dosyasındaki aktif şampiyon model sinyallerini okur.
   - KAP veto listesini kontrol eder; olumsuz haberi olan hisseleri eler.
4. T+1 Açılış Sanal Emir Defteri İcrası:
   - Top 5 Long hisse için Ters Volatilite ağırlıklandırması ve Dinamik ATR stop seviyelerini belirler.
   - Kademeli slippage (BIST30 10 bps, Yan Tahta 25 bps) + 15 bps komisyon uygulayarak sanal emirleri icra eder.
   - Gün içi Trailing Stop (+%2.5 primde kâr kilitleme) ve Dinamik ATR stop kontrollerini yürütür.
5. 'output/paper_trading_ledger.json' defterini günceller ve kurumsal durum raporu yazdırır.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from paper_trading import PaperTradingEngine
from champion_gatekeeper import get_champion_metadata
from extraction import BIST30_TICKERS, BIST_LIQUID_40
from ensemble import check_gatekeeper_veto
from extraction_kap import load_sentiment_cache


def check_ticker_kap_veto(ticker_clean: str) -> Tuple[bool, Optional[str]]:
    """Hisse için önbellekteki KAP açıklamalarını ve duygu skorunu denetler."""
    cache = load_sentiment_cache()
    matches = [v for k, v in cache.items() if k.startswith(f"{ticker_clean}_")]
    if matches:
        last = matches[-1]
        news_info = {
            "headline": last.get("title", ""),
            "news_sentiment": last.get("score", 0.0)
        }
        return check_gatekeeper_veto(news_info)
    return False, None


def load_daily_signals(signals_path: str = "output/daily_signals.json") -> Dict[str, Any]:
    """Aktif şampiyon model tarafından üretilmiş en güncel sinyal paketini yükler."""
    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"Sinyal dosyası bulunamadı: {signals_path}. Lütfen önce walk_forward.py veya ensemble.py çalıştırınız.")
    with open(signals_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_close_auction_execution(
    signals_path: str = "output/daily_signals.json",
    ledger_path: str = "output/paper_trading_ledger.json",
    initial_capital: float = 1_000_000.0,
    force_cash: bool = False,
    rebalance_step: int = 5
) -> Dict[str, Any]:
    """
    17:50 Kapanış Seansı (Close Auction) Kurumsal Sanal İcra Koşucusu.
    
    1. Tier 1 Makro Rejim Kapısı'nı ($XU100 > SMA_{50}$ ve $USDTRY\_Vol_{20} < 0.025$) denetler.
       - Rejim Ayı ise: Portföy 100% PPF Repo Nakte geçer (%50 yıllık repo faizi).
    2. Aday Havuzunu BIST Likit 40 ile sınırlar (sığ yan tahta slippage tuzağını önler).
    3. KAP Gatekeeper ve Şampiyon Ranker doğrulaması yapar.
    4. 17:50 - 18:05 Kapanış Marjında ($Close_t$) emir icrası hazırlar ve defteri günceller.
    """
    print("=" * 95)
    print("    BIST KURUMSAL 17:50 KAPANIŞ SEANSI İCRA KOŞUCUSU (CLOSE AUCTION RUNNER)")
    print(f"    Çalışma Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Kapanış Öncesi Emir Hazırlığı)")
    print("=" * 95)

    engine = PaperTradingEngine(
        initial_capital=initial_capital,
        ledger_path=ledger_path,
        rebalance_step=rebalance_step,
        cash_annual_yield=0.50,
        circuit_breaker_pct=5.0,
        circuit_breaker_window=5,
        circuit_breaker_cooldown=3
    )

    signals_data = load_daily_signals(signals_path)
    champion_name = signals_data.get("champion_model", "Walk-Forward Şampiyon Model")
    signal_date = signals_data.get("signal_date", signals_data.get("report_date", datetime.now().strftime("%Y-%m-%d")))
    execution_date = datetime.now().strftime("%Y-%m-%d")

    raw_candidates = signals_data.get("top_longs", signals_data.get("top_long_candidates", []))
    is_cash_rec = signals_data.get("is_cash_recommended", False)
    macro_bull = signals_data.get("macro_regime_bull", not is_cash_rec)

    is_defensive = force_cash or (not macro_bull) or is_cash_rec

    print(f"[*] Aktif Şampiyon Model       : {champion_name}")
    print(f"[*] Sinyal / Rapor Tarihi      : {signal_date}")
    print(f"[*] Kapanış İcra Tarihi        : {execution_date} (17:50 Kapanış Müzayedesi)")
    print(f"[*] Mevcut Portföy Özsermayesi : {engine.portfolio_value:,.2f} TL")
    print(f"[*] Nakit / PPF Repo Rezervi   : {engine.current_cash:,.2f} TL (Yıllık %50 Repo)")

    # 1. Makro Kapı ve Devre Kesici Kontrolü
    history = engine.ledger.get("daily_history", [])
    recent_equities = [float(h.get("end_equity_try", engine.initial_capital)) for h in history[-5:]] if history else [engine.portfolio_value]
    rolling_peak = max(recent_equities) if recent_equities else engine.portfolio_value
    rolling_dd_pct = ((engine.portfolio_value - rolling_peak) / rolling_peak) * 100.0 if rolling_peak > 0 else 0.0

    print(f"[*] Son Dönem Zirve Değer      : {rolling_peak:,.2f} TL (Dönemsel Düşüş: %{rolling_dd_pct:+.2f})")

    if not macro_bull:
        print("\n[!] TIER 1 MAKRO KAPISI KAPALI: XU100 < SMA50 veya Kur Volatilitesi Yüksek.")
        print("    -> Portföy koruma amaçlı 100% PPF Repo nakitte bekletiliyor.")
        is_defensive = True
    elif engine.cooldown_remaining > 0:
        print(f"\n[!] DEVRE KESİCİ DEVREDE: Portföy 100% nakitte (Kalan: {engine.cooldown_remaining} periyot).")
        is_defensive = True
    elif rolling_dd_pct <= -5.0:
        print(f"\n[!] DİKKAT: -%5 kayıp aşıldı (%{rolling_dd_pct:.2f}). Devre kesici tetiklendi!")
        is_defensive = True

    # 2. Likit 40 Filtresi ve KAP Gatekeeper Veto Denetimi
    qualified_candidates = []
    if not is_defensive:
        print(f"\n[+] BIST Likit 40 & KAP Gatekeeper Filtresi Uygulanıyor...")
        liquid40_set = set(BIST_LIQUID_40)
        for c in raw_candidates:
            ticker = c.get("ticker", "").replace(".IS", "")
            if ticker not in liquid40_set and f"{ticker}.IS" not in liquid40_set:
                print(f"  [X] {ticker} ELENDİ: Likit 40 evreninde değil (Sığ tahta kayma riski).")
                continue

            is_vetoed, reason = check_ticker_kap_veto(ticker)
            if is_vetoed:
                print(f"  [X] {ticker} KAP VETOLANDI: {reason}")
            else:
                qualified_candidates.append(c)
                w_str = c.get('recommended_weight', '%20.0')
                stop_str = c.get('dynamic_stop_loss', '-%5.0')
                tier_str = "BIST 30 (10 bps)" if ticker in BIST30_TICKERS else "Likit 40 (20 bps)"
                print(f"  [OK] {ticker} Onaylandı -> Ağırlık: {w_str} | End-of-Day Stop: {stop_str} | İcra: {tier_str}")

            if len(qualified_candidates) >= 5:
                break

    # 3. Kapanış Seansı İçin Piyasa Defter Kaydı
    market_df_records = []
    for c in qualified_candidates:
        ticker = c.get("ticker", "")
        t_full = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
        market_df_records.append({
            "Ticker": t_full,
            "Target_Return": 0.0,
            "Target_Low_Return": 0.0,
            "Target_High_Return": 0.0,
            "Target_Index_Return": 0.0
        })

    market_day_df = pd.DataFrame(market_df_records) if market_df_records else pd.DataFrame(
        columns=["Ticker", "Target_Return", "Target_Low_Return", "Target_High_Return", "Target_Index_Return"]
    )

    # 4. İcra Defterini Çalıştır
    summary = engine.execute_round(
        signal_date=signal_date,
        execution_date=execution_date,
        top_longs=qualified_candidates if not is_defensive else [],
        market_day_df=market_day_df,
        is_cash_day=is_defensive,
        macro_regime_bull=not is_defensive,
        holding_days=rebalance_step
    )

    print("\n" + "=" * 95)
    print("    17:50 KAPANIŞ SEANSI İCRA VE PORTFÖY ÖZETİ")
    print("=" * 95)
    print(f"İşlem Kararı                  : {summary['action']}")
    print(f"Başlangıç Sermaye             : {summary['start_equity_try']:,.2f} TL")
    print(f"Portföy Değeri                : {summary['end_equity_try']:,.2f} TL")
    print(f"Kümülatif Portföy Getirisi    : %{summary['cumulative_return_pct']:+.2f}")
    print(f"Kümülatif XU100 Benchmark     : %{summary['benchmark_cumulative_pct']:+.2f}")
    cum_alpha = summary['cumulative_return_pct'] - summary['benchmark_cumulative_pct']
    print(f"Toplam Net Alfa               : %{cum_alpha:+.2f}")
    print(f"Devre Kesici Aktif mi?        : {'EVET' if summary['circuit_breaker_active'] else 'HAYIR'}")
    print(f"Defter Dosyası                : '{ledger_path}'")
    print("=" * 95)

    return summary


def run_morning_paper_execution(
    signals_path: str = "output/daily_signals.json",
    ledger_path: str = "output/paper_trading_ledger.json",
    initial_capital: float = 1_000_000.0,
    force_cash: bool = False
) -> Dict[str, Any]:
    """Geriye dönük uyumluluk: 09:55 sabah icra koşucusu."""
    return run_close_auction_execution(
        signals_path=signals_path,
        ledger_path=ledger_path,
        initial_capital=initial_capital,
        force_cash=force_cash,
        rebalance_step=5
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 Kurumsal Sanal İcra ve Paper Trading Koşucusu")
    parser.add_argument("--mode", type=str, choices=["close_auction", "morning"], default="close_auction",
                        help="İcra modu: 'close_auction' (17:50 Kapanış Müzayedesi) veya 'morning' (09:55 Açılış)")
    parser.add_argument("--signals", type=str, default="output/daily_signals.json", help="Sinyal JSON dosya yolu")
    parser.add_argument("--ledger", type=str, default="output/paper_trading_ledger.json", help="Paper trading defter yolu")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Başlangıç sanal sermayesi (TL)")
    parser.add_argument("--cash", action="store_true", help="Zorunlu nakit/repo modu")
    parser.add_argument("--rebalance-step", type=int, default=5, help="Holding / Rebalans adım günü (varsayılan: 5 gün)")
    args = parser.parse_args()

    if args.mode == "close_auction":
        run_close_auction_execution(
            signals_path=args.signals,
            ledger_path=args.ledger,
            initial_capital=args.capital,
            force_cash=args.cash,
            rebalance_step=args.rebalance_step
        )
    else:
        run_morning_paper_execution(
            signals_path=args.signals,
            ledger_path=args.ledger,
            initial_capital=args.capital,
            force_cash=args.cash
        )

