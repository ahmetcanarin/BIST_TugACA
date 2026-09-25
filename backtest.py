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
from typing import Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
from extraction import BIST30_TICKERS, BIST_LIQUID_40


def run_portfolio_backtest(
    df_eval: pd.DataFrame,
    signal_col: str = "pred_reg",
    actual_return_col: str = "Target_Return",
    top_k: int = 10,
    buffer_k: int = 20,            # Hysteresis buffer: İlk 10'a giren hisse, ilk 20'den düşene kadar korunur
    smooth_window: int = 3,        # Sinyal gürültüsünü filtreleyen 3 günlük yumuşatma
    commission_bps: float = 15.0,  # On binde 15 aracı kurum komisyonu
    slippage_bps: Optional[float] = None, # Belirtilirse sabit, None ise kademeli (BIST30: 10bps, Yan Tahta: 25bps)
    bist30_slippage_bps: float = 10.0, # BIST 30 yüksek likidite hisseleri için 10 bps kayma
    midcap_slippage_bps: float = 25.0, # BIST 100 yan tahtaları için 25 bps kayma
    bist30_tickers: Optional[Set[str]] = None,
    price_limit_pct: float = 10.0,  # BIST %10 marj
    conviction_col: Optional[str] = None,
    min_conviction_threshold: float = 0.0,
    cash_daily_yield: float = (1.50 ** (1.0 / 252.0)) - 1.0, # %50 risksiz faiz / PPF gecelik getirisi
    benchmark_col: Optional[str] = "Target_Index_Return",
    weighting: str = "inverse_vol", # "inverse_vol" (Ters Volatilite / Risk Parity Lite) veya "equal"
    volatility_col: str = "Volatility_20",
    min_weight: float = 0.04,      # Minimum %4 hisse ağırlığı
    max_weight: float = 0.20,      # Maksimum %20 tek hisse tavanı
    use_stop_loss: bool = True,    # Dinamik ATR Stop-Loss koruması
    stop_loss_mode: str = "dynamic_atr", # "dynamic_atr", "fixed", "none"
    atr_col: str = "ATR_14_Pct",
    atr_multiplier: float = 2.0,   # Stop mesafesi: 2.0 x ATR
    fixed_stop_pct: float = 4.0,   # Sabit stop seçilirse %4
    min_stop_pct: float = 3.0,     # Minimum stop mesafesi %3
    max_stop_pct: float = 7.0,     # Maksimum stop mesafesi %7
    stop_slippage_bps: float = 10.0, # Stop tetiklendiğinde ek kayma maliyeti
    low_return_col: str = "Target_Low_Return",
    high_return_col: str = "Target_High_Return",
    use_trailing_stop: bool = True,
    trailing_activation_pct: float = 2.5, # Hisse en az %2.5 primlenirse trailing stop aktifleşir
    trailing_atr_mult: float = 1.5,      # Zirveden 1.5 x ATR çekilirse kâr realize edilir
    circuit_breaker_pct: Optional[float] = 5.0, # Haftalık %5 kayıpta portföyü nakde geçiren Devre Kesici
    circuit_breaker_window: int = 5,     # 5 işlem günü
    circuit_breaker_cooldown: int = 3,   # 3 işlem günü nakitte bekleme
    min_day_tickers: Optional[int] = None,
    min_adv_try: float = 0.0,       # Minimum günlük ortalama işlem hacmi (TL) filtresi
    adv_col: str = "ADV20_TRY",
    macro_regime_col: Optional[str] = "Macro_Regime_Bull", # 1. Katman Makro Rejim Kapısı (XU100 > SMA50)
    liquid_universe_only: bool = False, # Yalnızca Likit 40 hisseleri ile işlem yap
    rebalance_step: int = 5         # 5 günlük haftalık rebalans periyodu
) -> Dict[str, Any]:
    """
    Kademeli Slippage (BIST 30: 10 bps, Yan Tahta: 25 bps) ve Ters Volatilite (1/sigma)
    özellikli kurumsal Long-Only Top-K portföy simülasyonu yapar.
    Hysteresis tamponu ve sinyal yumuşatması ile gereksiz komisyon erimesini (turnover drag) engeller.
    Eğer conviction_col verilirse, modelin beklenen alfasının negatif olduğu günlerde nakitte kalır.
    İşlemler T+1 açılışında başlar, T+1 kapanışında değerlenir (Open-to-Close gerçekçi icra).
    Dinamik ATR Stop-Loss ile seans içi sert düşüşler erken kesilerek kuyruk riski sınırlandırılır.
    """
    if bist30_tickers is None:
        b30_set = set(BIST30_TICKERS) | {f"{t}.IS" for t in BIST30_TICKERS}
    else:
        b30_set = set(bist30_tickers)

    def _get_ticker_trade_cost(ticker_str: str) -> float:
        if slippage_bps is not None:
            slip = slippage_bps
        else:
            t_clean = ticker_str.replace(".IS", "")
            slip = bist30_slippage_bps if (t_clean in b30_set or ticker_str in b30_set) else midcap_slippage_bps
        return (commission_bps + slip) / 10000.0

    df = df_eval.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Likit 40 Filtresi (BIST 30 + Likit 10 Sanayi/Banka)
    if liquid_universe_only:
        liq_set = set(BIST_LIQUID_40) | {f"{t}.IS" for t in BIST_LIQUID_40}
        df = df[df["Ticker"].isin(liq_set)].copy()

    # Sinyal Yumuşatma (Gürültü filtreleme)
    if smooth_window > 1:
        df["smooth_signal"] = df.groupby("Ticker")[signal_col].transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
        active_signal_col = "smooth_signal"
    else:
        active_signal_col = signal_col

    dates = np.sort(df["Date"].unique())
    # Haftalık / Çok Günlük Rebalans Adımı (5 iş günü)
    if rebalance_step > 1 and len(dates) > rebalance_step:
        eval_dates = dates[::rebalance_step]
    else:
        eval_dates = dates

    daily_records = []
    current_portfolio = []
    prev_selected_tickers = set()

    cumulative_portfolio = 1.0
    cumulative_benchmark = 1.0
    equity_curve = [1.0]

    circuit_cooldown_remaining = 0
    circuit_breaker_tripped_count = 0
    trailing_stopped_count = 0
    step_days = float(rebalance_step) if rebalance_step > 1 else 1.0

    for dt in eval_dates:
        day_df = df[df["Date"] == dt].copy()
        # Piyasa derinliği yetersiz / yarım günleri atla
        min_req = min_day_tickers if min_day_tickers is not None else max(top_k * 2, 10 if liquid_universe_only else 20)
        if len(day_df) < min_req:
            continue

        day_sorted = day_df.sort_values(active_signal_col, ascending=False).reset_index(drop=True)

        # Likidite Filtresi (Sığ / düşük hacimli tahtaları ele)
        if min_adv_try > 0 and adv_col in day_sorted.columns:
            liquid_mask = (day_sorted[adv_col] >= min_adv_try)
            if liquid_mask.sum() >= top_k:
                day_sorted = day_sorted[liquid_mask].reset_index(drop=True)

        # 1. Katman: Makro Rejim Kapısı Kontrolü (XU100 > SMA50 & Sakin Kur Volatilitesi)
        macro_gate_active = False
        if macro_regime_col is not None and macro_regime_col in day_df.columns and day_df[macro_regime_col].notna().any():
            is_bull = float(day_df[macro_regime_col].dropna().iloc[0]) >= 0.5
            if not is_bull:
                macro_gate_active = True

        # Portföy Genelinde Devre Kesici (Haftalık %5 Kayıp Kontrolü)
        circuit_active = False
        if circuit_cooldown_remaining > 0:
            is_cash = True
            circuit_cooldown_remaining -= 1
            circuit_active = True
        elif circuit_breaker_pct is not None and len(equity_curve) >= min(2, circuit_breaker_window):
            rolling_peak = max(equity_curve[-circuit_breaker_window:])
            current_eq = equity_curve[-1]
            rolling_dd_pct = ((current_eq - rolling_peak) / rolling_peak) * 100.0
            if rolling_dd_pct <= -circuit_breaker_pct:
                is_cash = True
                circuit_cooldown_remaining = circuit_breaker_cooldown - 1
                circuit_breaker_tripped_count += 1
                circuit_active = True

        # 2. Katman: İkna / Conviction Filtresi veya Makro Rejim Kapısı
        if not circuit_active:
            if macro_gate_active:
                is_cash = True
                circuit_active = True
            else:
                is_cash = False
                if conviction_col is not None and conviction_col in day_sorted.columns:
                    avg_conviction = float(day_sorted.head(top_k)[conviction_col].mean())
                    if avg_conviction < min_conviction_threshold:
                        is_cash = True

        stopped_out_count = 0

        if is_cash:
            # Pozisyonları kapat veya nakitte bekle (%50 risksiz getiri / PPF)
            if len(prev_selected_tickers) > 0:
                cost_deduction = sum(_get_ticker_trade_cost(t) * (1.0 / float(top_k)) for t in prev_selected_tickers)
                turnover_fraction = len(prev_selected_tickers) / float(top_k)
            else:
                cost_deduction = 0.0
                turnover_fraction = 0.0
            gross_period_return = ((1.0 + cash_daily_yield) ** step_days) - 1.0
            gross_daily_return = gross_period_return
            net_daily_return = gross_period_return - cost_deduction
            selected_tickers = set()
            current_portfolio = []
        else:
            top_candidates = day_sorted.head(top_k)["Ticker"].tolist()
            buffer_candidates = set(day_sorted.head(buffer_k)["Ticker"].tolist())

            # Tampon Mantığı (Turnover Koruması):
            retained = [t for t in current_portfolio if t in buffer_candidates]
            needed = top_k - len(retained)
            additions = [t for t in top_candidates if t not in retained][:needed]
            selected_tickers = set(retained + additions)
            current_portfolio = list(selected_tickers)

            if len(prev_selected_tickers) > 0:
                turnover_fraction = len(selected_tickers - prev_selected_tickers) / float(top_k)
            else:
                turnover_fraction = 1.0  # İlk gün kuruluş cirosu

            selected_df = day_df[day_df["Ticker"].isin(selected_tickers)].copy()

            # Hisselerin gerçekleşen getirileri, Dinamik ATR Stop-Loss ve Trailing Stop Simülasyonu
            realized_returns = []

            for _, srow in selected_df.iterrows():
                close_ret = float(np.clip(srow[actual_return_col], -price_limit_pct, price_limit_pct))

                if use_stop_loss:
                    low_ret = float(np.clip(srow[low_return_col], -price_limit_pct, price_limit_pct)) if (low_return_col in srow and pd.notna(srow[low_return_col])) else close_ret
                    high_ret = float(np.clip(srow[high_return_col], -price_limit_pct, price_limit_pct)) if (high_return_col in srow and pd.notna(srow[high_return_col])) else close_ret

                    if stop_loss_mode == "dynamic_atr" and atr_col in srow and pd.notna(srow[atr_col]):
                        atr_val = float(srow[atr_col])
                        stop_thresh = float(np.clip(atr_multiplier * atr_val, min_stop_pct, max_stop_pct))
                    elif stop_loss_mode == "fixed":
                        stop_thresh = fixed_stop_pct
                        atr_val = fixed_stop_pct / 2.0
                    else:
                        stop_thresh = max_stop_pct
                        atr_val = 2.5

                    # [1. ADIM]: Hisse Başı Dinamik Trailing Stop (İzüren Stop)
                    # Eğer hisse seans içi en az trailing_activation_pct (+%2.5) primlendiyse stop kârlı seviyeye taşınır
                    if use_trailing_stop and high_ret >= trailing_activation_pct:
                        trail_stop_level = max(-stop_thresh, high_ret - (trailing_atr_mult * atr_val))
                        if low_ret <= trail_stop_level:
                            exit_ret = trail_stop_level - (stop_slippage_bps / 100.0)
                            realized_returns.append(exit_ret)
                            trailing_stopped_count += 1
                        else:
                            realized_returns.append(close_ret)
                    # [2. ADIM]: Standart Dinamik ATR Stop-Loss Kontrolü
                    elif low_ret <= -stop_thresh:
                        exit_ret = -stop_thresh - (stop_slippage_bps / 100.0)
                        realized_returns.append(exit_ret)
                        stopped_out_count += 1
                    else:
                        realized_returns.append(close_ret)
                else:
                    realized_returns.append(close_ret)

            realized_returns = np.array(realized_returns)

            # Ağırlıklandırma: Ters Volatilite (1 / sigma) veya Eşit Ağırlık
            if weighting == "inverse_vol" and volatility_col in selected_df.columns and len(realized_returns) > 0:
                vols = selected_df[volatility_col].values.astype(float)
                vols = np.where(np.isnan(vols) | (vols <= 1e-4), 0.02, vols)
                inv_vols = 1.0 / vols
                raw_weights = inv_vols / np.sum(inv_vols)
                # Min / Max tavan kısıtı uygula
                clipped_weights = np.clip(raw_weights, min_weight, max_weight)
                weights = clipped_weights / np.sum(clipped_weights)
            else:
                n_items = max(len(realized_returns), 1)
                weights = np.ones(len(realized_returns)) / float(n_items)

            gross_daily_return = float(np.sum(weights * realized_returns) / 100.0) if len(realized_returns) > 0 else 0.0

            # İşlem maliyeti düşümü (Hisse bazında kademeli komisyon ve slippage)
            if len(prev_selected_tickers) == 0:
                # İlk gün portföy kurulum maliyeti
                cost_deduction = sum(w * _get_ticker_trade_cost(t) for w, t in zip(weights, selected_df["Ticker"]))
            else:
                buys = selected_tickers - prev_selected_tickers
                sells = prev_selected_tickers - selected_tickers
                buy_cost = sum(_get_ticker_trade_cost(t) * (1.0 / float(top_k)) for t in buys)
                sell_cost = sum(_get_ticker_trade_cost(t) * (1.0 / float(top_k)) for t in sells)
                cost_deduction = buy_cost + sell_cost

            net_daily_return = gross_daily_return - cost_deduction

        # Benchmark: Gerçek XU100 endeks seans içi getirisi (Target_Index_Return) varsa onu kullan, yoksa evren ortalamasını al
        if benchmark_col is not None and benchmark_col in day_df.columns and day_df[benchmark_col].notna().any():
            bench_ret = float(day_df[benchmark_col].dropna().iloc[0] / 100.0)
        else:
            bench_ret = float(np.mean(np.clip(day_df[actual_return_col].values, -price_limit_pct, price_limit_pct)) / 100.0)

        cumulative_portfolio *= (1.0 + net_daily_return)
        cumulative_benchmark *= (1.0 + bench_ret)
        equity_curve.append(cumulative_portfolio)

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
            "Circuit_Breaker_Active": circuit_active,
            "Stopped_Out_Count": stopped_out_count,
            "Trailing_Stopped_Count": trailing_stopped_count,
            "Selected_Tickers": list(selected_tickers)
        })

        prev_selected_tickers = selected_tickers

    df_results = pd.DataFrame(daily_records)
    if df_results.empty:
        return {"error": "Yeterli veri bulunamadı."}

    # Finansal Performans İstatistikleri (Yıllık 252 iş günü varsayımı)
    n_steps = len(df_results)
    steps_per_year = 252.0 / step_days
    years = max((n_steps * step_days) / 252.0, 0.05)

    total_net_return = (cumulative_portfolio - 1.0) * 100.0
    total_bench_return = (cumulative_benchmark - 1.0) * 100.0

    cagr_portfolio = ((cumulative_portfolio ** (1.0 / years)) - 1.0) * 100.0
    cagr_benchmark = ((cumulative_benchmark ** (1.0 / years)) - 1.0) * 100.0

    daily_net_rets = df_results["Net_Return"]
    vol_annual = float(daily_net_rets.std() * np.sqrt(steps_per_year) * 100.0)

    # Sharpe Oranı (Yıllık %50 risksiz getiri düşülmüş)
    rf_step = (1.50 ** (step_days / 252.0)) - 1.0
    excess_daily = daily_net_rets - rf_step
    sharpe = float(np.sqrt(steps_per_year) * (excess_daily.mean() / (daily_net_rets.std() + 1e-6)))

    # Maksimum Düşüş (Max Drawdown - MDD)
    cum_series = df_results["Cumulative_Portfolio"]
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    max_drawdown = float(drawdown.min() * 100.0)

    # Bilgi Oranı (Information Ratio vs Benchmark)
    tracking_diff = df_results["Excess_Return"]
    info_ratio = float(np.sqrt(steps_per_year) * (tracking_diff.mean() / (tracking_diff.std() + 1e-6)))

    win_rate = float((df_results["Excess_Return"] > 0).mean() * 100.0)
    cash_days_pct = float(df_results["Is_Cash"].mean() * 100.0) if "Is_Cash" in df_results else 0.0

    n_days = n_steps * step_days
    return {
        "n_trading_days": n_days,
        "n_steps": n_steps,
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
        "circuit_breaker_tripped_count": circuit_breaker_tripped_count,
        "total_stopped_trades": int(df_results["Stopped_Out_Count"].sum()) if "Stopped_Out_Count" in df_results else 0,
        "total_trailing_stopped_trades": int(df_results["Trailing_Stopped_Count"].sum()) if "Trailing_Stopped_Count" in df_results else 0,
        "weighting_mode": weighting,
        "stop_loss_mode": stop_loss_mode if use_stop_loss else "none",
        "bist30_slippage_bps": bist30_slippage_bps if slippage_bps is None else slippage_bps,
        "midcap_slippage_bps": midcap_slippage_bps if slippage_bps is None else slippage_bps,
        "benchmark_name": "BIST 100 Endeksi (XU100.IS Gerçek Seans Getirisi)" if (benchmark_col is not None and benchmark_col in df.columns and df[benchmark_col].notna().any()) else "Eşit Ağırlıklı Hisse Ortalaması",
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
    if backtest_res.get("circuit_breaker_tripped_count", 0) > 0:
        print(f"Devre Kesici Tetiklenme       : {backtest_res['circuit_breaker_tripped_count']} kez (Portföy Haftalık -%5)")
    if backtest_res.get("total_stopped_trades", 0) > 0:
        print(f"Stop-Loss Tetiklenen İşlem    : {backtest_res['total_stopped_trades']} adet ({backtest_res.get('stop_loss_mode', '')})")
    if backtest_res.get("total_trailing_stopped_trades", 0) > 0:
        print(f"Trailing Stop Kâr Koruma     : {backtest_res['total_trailing_stopped_trades']} adet")
    print(f"Portföy Ağırlıklandırması     : {backtest_res.get('weighting_mode', 'equal').upper()}")
    if backtest_res.get("bist30_slippage_bps") is not None:
        print(f"Kademeli Kayma (Slippage)     : BIST30: {backtest_res['bist30_slippage_bps']:.0f} bps | Yan Tahta: {backtest_res['midcap_slippage_bps']:.0f} bps")
    print(f"Kıyaslama Ölçütü (Benchmark)  : {backtest_res.get('benchmark_name', 'XU100.IS')}")
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
    buffer_k: int = 20,
    benchmark_col: Optional[str] = "Target_Index_Return",
    **kwargs
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
            cash_daily_yield=cash_daily_yield,
            benchmark_col=benchmark_col,
            **kwargs
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
