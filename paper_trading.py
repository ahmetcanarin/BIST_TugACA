"""
BIST 100 30 Günlük Sanal Emir Defteri ve Paper Trading İcra Simülatörü

Bu modül:
1. Kurumsal quant modellerinin canlıya geçiş öncesinde gerçek piyasa verileriyle
   birebir eşzamanlı sanal portföy yönetimini (Paper Trading) yürütür.
2. 1.000.000 TL sanal başlangıç sermayesi ile başlar.
3. Her gün üretilen şampiyon model sinyallerini (Top 5 Long):
   - Ters volatilite ağırlıklandırması (recommended_weight)
   - Dinamik ATR Stop-Loss (dynamic_stop_loss)
   - Kademeli slippage (BIST30 10 bps, Yan Tahta 25 bps) + 15 bps komisyon
   ile ertesi günün açılışında ($Open_{t+1}$) icra eder.
4. Seans içi $Low_{t+1}$ seviyesini izler; stop tetiklenirse pozisyonu stop fiyatından keser.
   Stop tetiklenmezse gün sonu $Close_{t+1}$ fiyatından realize/değerler.
5. Gün gün tüm işlem detaylarını, nakit bakiyesini, hisse pozisyonlarını ve XU100
   kıyaslamasını 'output/paper_trading_ledger.json' defterine kaydeder.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from extraction import BIST30_TICKERS, BIST_LIQUID_40


class PaperTradingEngine:
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        ledger_path: str = "output/paper_trading_ledger.json",
        commission_bps: float = 15.0,
        bist30_slippage_bps: float = 10.0,
        midcap_slippage_bps: float = 20.0,
        circuit_breaker_pct: float = 5.0,
        circuit_breaker_window: int = 5,
        circuit_breaker_cooldown: int = 3,
        use_trailing_stop: bool = True,
        trailing_activation_pct: float = 2.5,
        trailing_atr_mult: float = 1.5,
        rebalance_step: int = 5,
        cash_annual_yield: float = 0.50
    ):
        self.initial_capital = initial_capital
        self.ledger_path = ledger_path
        self.commission_bps = commission_bps
        self.bist30_slippage_bps = bist30_slippage_bps
        self.midcap_slippage_bps = midcap_slippage_bps
        self.circuit_breaker_pct = circuit_breaker_pct
        self.circuit_breaker_window = circuit_breaker_window
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_atr_mult = trailing_atr_mult
        self.rebalance_step = rebalance_step
        self.cash_annual_yield = cash_annual_yield
        self.b30_set = set(BIST30_TICKERS) | {f"{t}.IS" for t in BIST30_TICKERS}
        self.liquid40_set = set(BIST_LIQUID_40) | {f"{t}.IS" for t in BIST_LIQUID_40}

        self.current_cash = initial_capital
        self.portfolio_value = initial_capital
        self.ledger = self._load_or_create_ledger()
        self.cooldown_remaining = int(self.ledger.get("cooldown_remaining", 0))

    def _get_trade_cost_rate(self, ticker: str) -> float:
        t_clean = ticker.replace(".IS", "")
        slip = self.bist30_slippage_bps if (t_clean in self.b30_set or ticker in self.b30_set) else self.midcap_slippage_bps
        return (self.commission_bps + slip) / 10000.0

    def _load_or_create_ledger(self) -> Dict[str, Any]:
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_cash = float(data.get("current_cash_try", self.initial_capital))
                    self.portfolio_value = float(data.get("current_portfolio_value_try", self.initial_capital))
                    return data
            except Exception:
                pass

        return {
            "initial_capital_try": self.initial_capital,
            "current_cash_try": self.initial_capital,
            "current_portfolio_value_try": self.initial_capital,
            "total_pnl_try": 0.0,
            "total_return_pct": 0.0,
            "benchmark_cumulative_pct": 0.0,
            "total_trades_count": 0,
            "stopped_out_trades_count": 0,
            "daily_history": [],
            "executed_trades": []
        }

    def save_ledger(self):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(self.ledger, f, ensure_ascii=False, indent=2)

    def execute_round(
        self,
        signal_date: str,
        execution_date: str,
        top_longs: List[Dict[str, Any]],
        market_day_df: pd.DataFrame,
        is_cash_day: bool = False,
        macro_regime_bull: bool = True,
        holding_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Kapanış Seansı (17:50 - 18:05) veya ertesi gün icra edilen portföy turunu simüle eder.
        Multi-day (varsayılan 5 gün / haftalık) holding, End-of-Day stop loss ve
        Tier 1 Makro Rejim Kapısı'nı (%50 yıllık repo faizi) tam entegre eder.
        """
        step_days = holding_days if holding_days is not None else self.rebalance_step
        cash_yield = (1.0 + self.cash_annual_yield) ** (step_days / 252.0) - 1.0

        start_equity = self.portfolio_value
        daily_trades = []
        stopped_count = 0
        trailing_stopped_count = 0

        # Endeks getirisi (Benchmark)
        if "Target_Index_Return" in market_day_df.columns and market_day_df["Target_Index_Return"].notna().any():
            bench_ret_pct = float(market_day_df["Target_Index_Return"].dropna().iloc[0])
        else:
            bench_ret_pct = 0.0

        # Tier 1 Makro Kapısı Kontrolü
        if not macro_regime_bull:
            is_cash_day = True
            action_desc = "MAKRO REJİM KAPISI: AYI PİYASASI (100% Nakit / PPF Repo)"
        else:
            action_desc = ""

        # Portföy Haftalık Devre Kesici Kontrolü (Haftalık -%5 kayıp kontrolü)
        history = self.ledger.get("daily_history", [])
        recent_equities = [float(h.get("end_equity_try", self.initial_capital)) for h in history[-self.circuit_breaker_window:]] if history else [start_equity]
        rolling_peak = max(recent_equities) if recent_equities else start_equity
        rolling_dd_pct = ((start_equity - rolling_peak) / rolling_peak) * 100.0 if rolling_peak > 0 else 0.0

        circuit_active = False
        if self.cooldown_remaining > 0:
            is_cash_day = True
            self.cooldown_remaining -= 1
            circuit_active = True
            action_desc = f"DEVRE KESİCİ DEVREDE (Kalan Soğuma: {self.cooldown_remaining + 1} periyot)"
        elif self.circuit_breaker_pct is not None and len(recent_equities) >= min(2, self.circuit_breaker_window) and rolling_dd_pct <= -self.circuit_breaker_pct:
            is_cash_day = True
            self.cooldown_remaining = self.circuit_breaker_cooldown - 1
            circuit_active = True
            action_desc = f"DEVRE KESİCİ TETİKLENDİ (Düşüş %{abs(rolling_dd_pct):.1f} >= %{self.circuit_breaker_pct:.1f})"

        if is_cash_day or not top_longs:
            # Defansif Periyot: Nakit repo/PPF faizi işletilir (Yıllık %50)
            interest_gain = self.current_cash * cash_yield
            self.current_cash += interest_gain
            self.portfolio_value = self.current_cash
            net_day_pnl = interest_gain
            day_return_pct = (interest_gain / start_equity) * 100.0
            if not action_desc:
                action_desc = "NAKİTTE BEKLENDİ (PPF / Gecelik Repo)"
        else:
            action_desc = f"{len(top_longs)} HİSSE İLE PORTFÖY REBALANSI ({step_days}G Holding)"
            total_allocated = 0.0
            allocated_equities = []

            # 1. Sermaye Dağıtımı (Ters Volatilite Ağırlıkları)
            for s in top_longs:
                w_str = str(s.get("recommended_weight", "20.0")).replace("%", "").strip()
                w_pct = float(w_str) / 100.0 if float(w_str) > 1.0 else float(w_str)
                capital_alloc = start_equity * w_pct
                allocated_equities.append((s, capital_alloc))
                total_allocated += capital_alloc

            # Kalan nakit
            unallocated_cash = max(0.0, start_equity - total_allocated)
            self.current_cash = unallocated_cash * (1.0 + cash_yield)
            period_end_positions_value = 0.0

            # 2. Her hisse için Kapanış İcrası, Stop-Loss ve Getiri Simülasyonu
            for s, capital_alloc in allocated_equities:
                ticker = s["ticker"]
                t_full = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
                match = market_day_df[market_day_df["Ticker"] == t_full]

                cost_rate = self._get_trade_cost_rate(ticker)
                entry_cost_try = capital_alloc * cost_rate

                if not match.empty:
                    m_row = match.iloc[0]
                    close_ret_pct = float(m_row.get("Target_Return", 0.0))
                    low_ret_pct = float(m_row.get("Target_Low_Return", close_ret_pct))
                    high_ret_pct = float(m_row.get("Target_High_Return", close_ret_pct))
                else:
                    close_ret_pct = 0.0
                    low_ret_pct = 0.0
                    high_ret_pct = 0.0

                # Dinamik Stop-Loss Eşiği (End-of-Day Close bazlı)
                stop_str = str(s.get("dynamic_stop_loss", "5.0")).replace("%", "").replace("-", "").strip()
                stop_thresh_pct = float(stop_str)
                atr_pct = float(s.get("atr_pct", stop_thresh_pct / 2.0))

                # [1. ADIM]: Trailing Stop Kontrolü
                is_trailing_stopped = False
                if self.use_trailing_stop and high_ret_pct >= self.trailing_activation_pct:
                    trail_stop_level = max(-stop_thresh_pct, high_ret_pct - (self.trailing_atr_mult * atr_pct))
                    if low_ret_pct <= trail_stop_level:
                        exit_cost_rate = cost_rate
                        realized_ret_pct = trail_stop_level - (exit_cost_rate * 100.0)
                        is_trailing_stopped = True
                        trailing_stopped_count += 1
                        exit_reason = f"Trailing Stop Kâr Koruma Tetiklendi (Seviye: %{trail_stop_level:+.2f})"

                # [2. ADIM]: End-of-Day Stop-Loss Kontrolü
                is_stopped = False
                if not is_trailing_stopped:
                    is_stopped = (low_ret_pct <= -stop_thresh_pct)
                    if is_stopped:
                        exit_cost_rate = cost_rate
                        realized_ret_pct = -stop_thresh_pct - (exit_cost_rate * 100.0)
                        stopped_count += 1
                        exit_reason = f"Dinamik ATR Stop-Loss Tetiklendi (-%{stop_thresh_pct:.1f})"
                    else:
                        exit_cost_rate = cost_rate
                        realized_ret_pct = close_ret_pct - (exit_cost_rate * 100.0)
                        exit_reason = "Periyot Sonu Kapanış Realizasyonu"

                trade_pnl_try = (capital_alloc * (realized_ret_pct / 100.0)) - entry_cost_try
                final_position_val = capital_alloc + trade_pnl_try
                period_end_positions_value += final_position_val

                is_liquid40 = ticker in self.liquid40_set or t_full in self.liquid40_set
                is_b30 = ticker in self.b30_set or t_full in self.b30_set
                tier_label = "BIST 30 (10 bps)" if is_b30 else ("Likit 40 (20 bps)" if is_liquid40 else "Yan Tahta (25 bps)")

                trade_record = {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "ticker": ticker,
                    "allocated_capital_try": round(capital_alloc, 2),
                    "weight_pct": s.get("recommended_weight", "%20.0"),
                    "liquidity_tier": tier_label,
                    "holding_return_high": f"%{high_ret_pct:+.2f}",
                    "holding_return_low": f"%{low_ret_pct:+.2f}",
                    "realized_return_pct": f"%{realized_ret_pct:+.2f}",
                    "pnl_try": round(trade_pnl_try, 2),
                    "is_stopped_out": is_stopped,
                    "is_trailing_stopped": is_trailing_stopped,
                    "exit_reason": exit_reason
                }
                daily_trades.append(trade_record)

            self.portfolio_value = self.current_cash + period_end_positions_value
            self.current_cash = self.portfolio_value
            net_day_pnl = self.portfolio_value - start_equity
            day_return_pct = (net_day_pnl / start_equity) * 100.0

        # Defter Güncellemesi
        tot_return_pct = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100.0
        prev_bench_cum = self.ledger.get("benchmark_cumulative_pct", 0.0)
        new_bench_cum = ((1.0 + (prev_bench_cum / 100.0)) * (1.0 + (bench_ret_pct / 100.0)) - 1.0) * 100.0

        day_summary = {
            "execution_date": execution_date,
            "action": action_desc,
            "start_equity_try": round(start_equity, 2),
            "end_equity_try": round(self.portfolio_value, 2),
            "net_daily_pnl_try": round(net_day_pnl, 2),
            "daily_return_pct": round(day_return_pct, 2),
            "benchmark_return_pct": round(bench_ret_pct, 2),
            "alpha_daily_pct": round(day_return_pct - bench_ret_pct, 2),
            "cumulative_return_pct": round(tot_return_pct, 2),
            "benchmark_cumulative_pct": round(new_bench_cum, 2),
            "circuit_breaker_active": circuit_active,
            "stopped_out_count": stopped_count,
            "trailing_stopped_count": trailing_stopped_count,
            "trades_count": len(daily_trades)
        }

        self.ledger["current_cash_try"] = round(self.current_cash, 2)
        self.ledger["current_portfolio_value_try"] = round(self.portfolio_value, 2)
        self.ledger["total_pnl_try"] = round(self.portfolio_value - self.initial_capital, 2)
        self.ledger["total_return_pct"] = round(tot_return_pct, 2)
        self.ledger["benchmark_cumulative_pct"] = round(new_bench_cum, 2)
        self.ledger["total_trades_count"] += len(daily_trades)
        self.ledger["stopped_out_trades_count"] += stopped_count
        self.ledger["trailing_stopped_trades_count"] = self.ledger.get("trailing_stopped_trades_count", 0) + trailing_stopped_count
        self.ledger["cooldown_remaining"] = self.cooldown_remaining
        self.ledger["daily_history"].append(day_summary)
        self.ledger["executed_trades"].extend(daily_trades)

        self.save_ledger()
        return day_summary

    def execute_daily_round(
        self,
        signal_date: str,
        execution_date: str,
        top_longs: List[Dict[str, Any]],
        market_day_df: pd.DataFrame,
        is_cash_day: bool = False,
        cash_daily_yield: Optional[float] = None
    ) -> Dict[str, Any]:
        """Geriye dönük uyumluluk için execute_round çağrısı."""
        return self.execute_round(
            signal_date=signal_date,
            execution_date=execution_date,
            top_longs=top_longs,
            market_day_df=market_day_df,
            is_cash_day=is_cash_day,
            macro_regime_bull=not is_cash_day,
            holding_days=1
        )


def run_paper_trading_simulation(
    df_features: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    n_days: int = 60,
    rebalance_step: int = 5,
    ledger_path: str = "output/paper_trading_ledger.json"
) -> Dict[str, Any]:
    """
    Son 60 işlem günü üzerinde kurumsal 5 günlük (haftalık) rebalans ve
    Tier 1 Makro Rejim Kapılı Paper Trading simülasyonu yürütür.
    """
    df = df_features.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    unique_dates = np.sort(df["Date"].unique())

    if len(unique_dates) < n_days + rebalance_step:
        n_days = max(10, len(unique_dates) - rebalance_step)

    sim_dates = unique_dates[-n_days:]
    step_indices = list(range(0, len(sim_dates) - 1, rebalance_step))

    engine = PaperTradingEngine(
        initial_capital=initial_capital,
        ledger_path=ledger_path,
        rebalance_step=rebalance_step,
        cash_annual_yield=0.50
    )

    print("\n" + "=" * 95)
    print("BIST KURUMSAL PAPER TRADING SİMÜLATÖRÜ (HAFTALIK MOMENTUM & MAKRO REJİM KAPISI)")
    print(f"Başlangıç Sermayesi: {initial_capital:,.2f} TL | Test Süresi: {n_days} gün ({len(step_indices)} Tur)")
    print(f"Holding Süresi: {rebalance_step} Gün | Likit Evren: BIST Likit 40 | Repo Faizi: %50 Yıllık")
    print("=" * 95)
    print(f"{'İşlem Tarihi':<12} | {'Portföy Değeri (TL)':<20} | {'Dönem PnL':<15} | {'Getiri':<8} | {'XU100':<8} | {'Alfa':<8} | {'Stop':<5}")
    print("-" * 95)

    for idx in step_indices:
        sig_dt = sim_dates[idx]
        exec_idx = min(idx + rebalance_step, len(sim_dates) - 1)
        exec_dt = sim_dates[exec_idx]

        day_sig_df = df[df["Date"] == sig_dt].copy()
        day_exec_df = df[df["Date"] == exec_dt].copy()

        # Makro Rejim Kapısı Kontrolü ($XU100 > SMA_50$)
        is_bull = True
        if "Macro_Regime_Bull" in day_sig_df.columns:
            m_flag = day_sig_df["Macro_Regime_Bull"].dropna()
            if not m_flag.empty:
                is_bull = bool(m_flag.iloc[0] > 0.5)

        # Yalnızca Likit 40 evreninden seç
        liquid_day_df = day_sig_df[day_sig_df["Ticker"].str.replace(".IS", "").isin(BIST_LIQUID_40)]
        if liquid_day_df.empty:
            liquid_day_df = day_sig_df

        score_col = "Cross_Rank_Return" if "Cross_Rank_Return" in liquid_day_df.columns else ("Momentum_5d" if "Momentum_5d" in liquid_day_df.columns else "Log_Return")
        day_sig_sorted = liquid_day_df.sort_values(score_col, ascending=False).reset_index(drop=True)

        top5_longs = []
        for _, row in day_sig_sorted.head(5).iterrows():
            t_clean = row["Ticker"].replace(".IS", "")
            atr_val = float(row.get("ATR_14_Pct", 2.5))
            stop_dist = round(float(np.clip(2.0 * atr_val, 3.0, 7.0)), 1)
            is_b30 = t_clean in BIST30_TICKERS

            top5_longs.append({
                "ticker": t_clean,
                "recommended_weight": "%20.0",
                "dynamic_stop_loss": f"-%{stop_dist}",
                "atr_pct": atr_val,
                "liquidity_tier": "BIST 30 (10 bps)" if is_b30 else "Likit 40 (20 bps)"
            })

        summary = engine.execute_round(
            signal_date=pd.Timestamp(sig_dt).strftime("%Y-%m-%d"),
            execution_date=pd.Timestamp(exec_dt).strftime("%Y-%m-%d"),
            top_longs=top5_longs,
            market_day_df=day_exec_df,
            is_cash_day=not is_bull,
            macro_regime_bull=is_bull,
            holding_days=rebalance_step
        )

        d_str = summary["execution_date"]
        val_str = f"{summary['end_equity_try']:,.2f} TL"
        pnl_str = f"{summary['net_daily_pnl_try']:+,.2f} TL"
        ret_str = f"%{summary['daily_return_pct']:+.2f}"
        bench_str = f"%{summary['benchmark_return_pct']:+.2f}"
        alpha_str = f"%{summary['alpha_daily_pct']:+.2f}"
        stop_str = f"{summary['stopped_out_count']} adet"
        print(f"{d_str:<12} | {val_str:<20} | {pnl_str:<15} | {ret_str:<8} | {bench_str:<8} | {alpha_str:<8} | {stop_str:<5}")

    print("=" * 95)
    print(f"Sanal Portföy Kapanış Bakiyesi : {engine.ledger['current_portfolio_value_try']:,.2f} TL")
    print(f"Toplam Net PnL (Kâr/Zarar)     : {engine.ledger['total_pnl_try']:+,.2f} TL (%{engine.ledger['total_return_pct']:+.2f})")
    print(f"XU100 Benchmark Kümülatif      : %{engine.ledger['benchmark_cumulative_pct']:+.2f}")
    cum_alpha = engine.ledger['total_return_pct'] - engine.ledger['benchmark_cumulative_pct']
    print(f"Gerçekleşen Net Alfa           : %{cum_alpha:+.2f}")
    print(f"Toplam Yapılan İşlem Sayısı    : {engine.ledger['total_trades_count']} adet")
    print(f"Durdurulan (Stop-Loss) İşlem   : {engine.ledger['stopped_out_trades_count']} adet")
    print(f"Detaylı defter '{ledger_path}' dosyasına kaydedildi.")
    print("=" * 95)

    return engine.ledger


def run_30day_paper_trading_simulation(df_features: pd.DataFrame, initial_capital: float = 1_000_000.0) -> Dict[str, Any]:
    """Geriye dönük uyumluluk alias fonksiyonu (varsayılan 60 günlük simülasyon)."""
    return run_paper_trading_simulation(df_features=df_features, initial_capital=initial_capital, n_days=60, rebalance_step=5)


if __name__ == "__main__":
    from preprocessing import load_and_preprocess_pipeline
    print("Canlı veri çekiliyor ve 30 günlük paper trading simülasyonu başlatılıyor...")
    data_dict = load_and_preprocess_pipeline(period="1y", seq_len=10)
    run_30day_paper_trading_simulation(data_dict["test_df"])
