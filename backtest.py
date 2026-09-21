"""
BIST 100 Gerçekçi Portföy Backtest ve Simülasyon Motoru

Bu modül:
1. Model tahminlerini (LightGBM, GRU veya Hibrit Ensemble) kesitsel sıralamaya (Rank) dönüştürür.
2. Her işlem günü için Top-K (örn: En iyi 5 veya 10 hisse) portföyü oluşturur.
3. Gerçekçi piyasa kısıtlarını simüle eder:
   - On binde 15 (%0.15) aracı kurum komisyonu
   - On binde 5 (%0.05) kayma maliyeti (Slippage)
   - BIST taban/tavan (%10) kuralı
4. Kümülatif Getiri, Yıllıklandırılmış Getiri (CAGR), Sharpe Oranı,
   Maksimum Düşüş (Max Drawdown - MDD) ve BIST 100 Karşılaştırmalı Alfa raporlar.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def run_portfolio_backtest(
    df_eval: pd.DataFrame,
    signal_col: str = "pred_reg",
    actual_return_col: str = "Target_Return",
    top_k: int = 10,
    buffer_k: int = 20,            # Hysteresis buffer: İlk 10'a giren hisse, ilk 20'den düşene kadar korunur
    smooth_window: int = 3,        # Sinyal gürültüsünü filtreleyen 3 günlük yumuşatma
    commission_bps: float = 15.0,  # On binde 15
    slippage_bps: float = 5.0,      # On binde 5
    price_limit_pct: float = 10.0,  # BIST %10 marj
    conviction_col: Optional[str] = None,
    min_conviction_threshold: float = 0.0,
    cash_daily_yield: float = 0.0
) -> Dict[str, Any]:
    """
    Günlük eşit ağırlıklı Long-Only Top-K portföy simülasyonu yapar.
    Hysteresis tamponu ve sinyal yumuşatması ile gereksiz komisyon erimesini (turnover drag) engeller.
    Eğer conviction_col verilirse, modelin beklenen alfasının negatif olduğu günlerde nakitte kalır.
    """
    total_cost_per_trade = (commission_bps + slippage_bps) / 10000.0  # Örn: 20 bps = 0.0020

    df = df_eval.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Sinyal Yumuşatma (Gürültü filtreleme)
    if smooth_window > 1:
        df["smooth_signal"] = df.groupby("Ticker")[signal_col].transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
        active_signal_col = "smooth_signal"
    else:
        active_signal_col = signal_col

    dates = np.sort(df["Date"].unique())

    daily_records = []
    current_portfolio = []
    prev_selected_tickers = set()

    cumulative_portfolio = 1.0
    cumulative_benchmark = 1.0

    for dt in dates:
        day_df = df[df["Date"] == dt].copy()
        # Piyasa derinliği yetersiz / yarım günleri atla (en az 20 hisse)
        if len(day_df) < max(top_k * 2, 20):
            continue

        day_sorted = day_df.sort_values(active_signal_col, ascending=False).reset_index(drop=True)

        # İkna / Conviction Filtresi (Sinyal Yoksa Nakit Kal)
        is_cash = False
        if conviction_col is not None and conviction_col in day_sorted.columns:
            avg_conviction = float(day_sorted.head(top_k)[conviction_col].mean())
            if avg_conviction < min_conviction_threshold:
                is_cash = True

        if is_cash:
            # Pozisyonları kapatarak nakite geç veya nakitte bekle
            turnover_fraction = (len(prev_selected_tickers) / float(top_k)) if len(prev_selected_tickers) > 0 else 0.0
            cost_deduction = turnover_fraction * total_cost_per_trade  # Sadece satış komisyonu
            gross_daily_return = cash_daily_yield
            net_daily_return = gross_daily_return - cost_deduction
            selected_tickers = set()
            current_portfolio = []
        else:
            top_candidates = day_sorted.head(top_k)["Ticker"].tolist()
            buffer_candidates = set(day_sorted.head(buffer_k)["Ticker"].tolist())

            # Tampon Mantığı (Turnover Koruması):
            # Mevcut hisseler tampon bölge (buffer_k) içinde kaldığı sürece satılmaz
            retained = [t for t in current_portfolio if t in buffer_candidates]
            needed = top_k - len(retained)
            additions = [t for t in top_candidates if t not in retained][:needed]
            selected_tickers = set(retained + additions)
            current_portfolio = list(selected_tickers)

            # Portföy cirosu (Turnover) hesabı
            if len(prev_selected_tickers) > 0:
                turnover_fraction = len(selected_tickers - prev_selected_tickers) / float(top_k)
            else:
                turnover_fraction = 1.0  # İlk gün kuruluş cirosu

            # Hisselerin gerçekleşen getirileri (%10 tavan/taban kısıtlı)
            selected_df = day_df[day_df["Ticker"].isin(selected_tickers)]
            raw_rets = selected_df[actual_return_col].values
            clipped_rets = np.clip(raw_rets, -price_limit_pct, price_limit_pct)
            gross_daily_return = float(np.mean(clipped_rets) / 100.0)

            # İşlem maliyeti düşümü (Turnover oranına göre alış ve satış komisyonu)
            cost_deduction = turnover_fraction * 2.0 * total_cost_per_trade
            net_daily_return = gross_daily_return - cost_deduction

        # Benchmark: O günkü tüm BIST hisselerinin eşit ağırlıklı ortalaması
        bench_ret = float(np.mean(np.clip(day_df[actual_return_col].values, -price_limit_pct, price_limit_pct)) / 100.0)

        cumulative_portfolio *= (1.0 + net_daily_return)
        cumulative_benchmark *= (1.0 + bench_ret)

        daily_records.append({
            "Date": dt,
            "Gross_Return": gross_daily_return,
            "Net_Return": net_daily_return,
            "Benchmark_Return": bench_ret,
            "Excess_Return": net_daily_return - bench_ret,
            "Cumulative_Portfolio": cumulative_portfolio,
            "Cumulative_Benchmark": cumulative_benchmark,
            "Turnover": turnover_fraction,
            "Is_Cash": is_cash,
            "Selected_Tickers": list(selected_tickers)
        })

        prev_selected_tickers = selected_tickers

    df_results = pd.DataFrame(daily_records)
    if df_results.empty:
        return {"error": "Yeterli veri bulunamadı."}

    # Finansal Performans İstatistikleri (Yıllık 252 iş günü varsayımı)
    n_days = len(df_results)
    years = max(n_days / 252.0, 0.05)

    total_net_return = (cumulative_portfolio - 1.0) * 100.0
    total_bench_return = (cumulative_benchmark - 1.0) * 100.0

    cagr_portfolio = ((cumulative_portfolio ** (1.0 / years)) - 1.0) * 100.0
    cagr_benchmark = ((cumulative_benchmark ** (1.0 / years)) - 1.0) * 100.0

    daily_net_rets = df_results["Net_Return"]
    vol_annual = float(daily_net_rets.std() * np.sqrt(252) * 100.0)

    # Sharpe Oranı (Yıllık %30 risksiz faiz veya baz getiri düşülmüş)
    rf_daily = (1.30 ** (1 / 252.0)) - 1.0
    excess_daily = daily_net_rets - rf_daily
    sharpe = float(np.sqrt(252) * (excess_daily.mean() / (daily_net_rets.std() + 1e-6)))

    # Maksimum Düşüş (Max Drawdown - MDD)
    cum_series = df_results["Cumulative_Portfolio"]
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    max_drawdown = float(drawdown.min() * 100.0)

    # Bilgi Oranı (Information Ratio vs Benchmark)
    tracking_diff = df_results["Excess_Return"]
    info_ratio = float(np.sqrt(252) * (tracking_diff.mean() / (tracking_diff.std() + 1e-6)))

    win_rate = float((df_results["Excess_Return"] > 0).mean() * 100.0)
    cash_days_pct = float(df_results["Is_Cash"].mean() * 100.0) if "Is_Cash" in df_results else 0.0

    return {
        "n_trading_days": n_days,
        "total_net_return": total_net_return,
        "total_bench_return": total_bench_return,
        "cagr_portfolio": cagr_portfolio,
        "cagr_benchmark": cagr_benchmark,
        "volatility_annual": vol_annual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "information_ratio": info_ratio,
        "win_rate_vs_benchmark": win_rate,
        "cash_days_pct": cash_days_pct,
        "daily_history": df_results
    }


def print_backtest_report(backtest_res: Dict[str, Any], title: str = "BIST 100 MODEL PORTFÖYÜ"):
    """Standart kantitatif portföy analiz raporu yazdırır."""
    print("\n" + "=" * 70)
    print(f"KURUMSAL PORTFÖY BACKTEST RAPORU: {title}")
    print("=" * 70)
    print(f"Toplam İşlem Günü (Test)      : {backtest_res['n_trading_days']} gün")
    print(f"Portföy Toplam Net Getiri     : %{backtest_res['total_net_return']:.2f}")
    print(f"BIST 100 Benchmark Getirisi   : %{backtest_res['total_bench_return']:.2f}")
    print(f"Net Alfa (Portföy - BIST100)  : %{backtest_res['total_net_return'] - backtest_res['total_bench_return']:+.2f}")
    if backtest_res.get("cash_days_pct", 0.0) > 0:
        print(f"Nakitte Geçirilen Gün Oranı   : %{backtest_res['cash_days_pct']:.1f}")
    print("-" * 70)
    print(f"Yıllıklandırılmış Getiri (CAGR): %{backtest_res['cagr_portfolio']:.2f} (BIST: %{backtest_res['cagr_benchmark']:.2f})")
    print(f"Yıllıklandırılmış Volatilite  : %{backtest_res['volatility_annual']:.2f}")
    print(f"Sharpe Oranı                  : {backtest_res['sharpe_ratio']:.2f}")
    print(f"Maksimum Düşüş (Max Drawdown) : %{backtest_res['max_drawdown']:.2f}")
    print(f"Bilgi Oranı (Information Ratio): {backtest_res['information_ratio']:.2f}")
    print(f"Benchmark'ı Yenme Oranı       : %{backtest_res['win_rate_vs_benchmark']:.2f}")
    print("=" * 70)


def run_conviction_sensitivity_sweep(
    df: pd.DataFrame,
    signal_col: str,
    conviction_col: str,
    thresholds: List[Optional[float]] = [0.0, -0.10, -0.25, -0.50, -1.0, None],
    cash_daily_yield: float = 0.0,
    top_k: int = 10,
    buffer_k: int = 20
) -> pd.DataFrame:
    """
    Conviction eşiği duyarlılık analizi (Sensitivity Sweep):
    Farklı eşik değerlerinde portföyün nakit oranı, net getiri, alfa, Sharpe ve MDD
    değerlerini hesaplayarak %99 nakit kapanı ile hisse ticareti arasındaki dengeyi gösterir.
    """
    records = []
    for thresh in thresholds:
        c_col = conviction_col if thresh is not None else None
        c_val = thresh if thresh is not None else 0.0
        res = run_portfolio_backtest(
            df,
            signal_col=signal_col,
            top_k=top_k,
            buffer_k=buffer_k,
            conviction_col=c_col,
            min_conviction_threshold=c_val,
            cash_daily_yield=cash_daily_yield
        )
        label = f"%{thresh:+.2f}" if thresh is not None else "Filtresiz (Tam Hisse)"
        records.append({
            "Eşik": label,
            "Net_Getiri": round(res["total_net_return"], 2),
            "Net_Alfa": round(res["total_net_return"] - res["total_bench_return"], 2),
            "Sharpe": round(res["sharpe_ratio"], 2),
            "Max_DD": round(res["max_drawdown"], 2),
            "Win_Rate": round(res["win_rate_vs_benchmark"], 1),
            "Nakit_Pct": round(res["cash_days_pct"], 1)
        })
    sweep_df = pd.DataFrame(records)
    return sweep_df
