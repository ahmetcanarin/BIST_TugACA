"""
BIST 100 Kayan Pencereli Walk-Forward Validasyon ve Model Eğitim Motoru

Bu modül:
1. Sabit tekil Train/Test ayrımı yerine, finansal rejim değişimlerine (faiz döngüleri,
   enflasyon şokları, kur hareketleri) karşı modeli periyodik kayan/genişleyen pencereler
   (Expanding Walk-Forward) ile eğitir ve out-of-sample performansını test eder.
2. Belirlenen zaman dilimleri (2023, 2024, 2025+) için modeli geçmiş veride eğitir,
   ilgili out-of-sample penceresinde tahmin üretir.
3. Her pencere için Adım 1-4 kurumsal reformlarını (T+1 Open-to-Close icra, Gatekeeper Veto,
   Ters Volatilite ağırlıklandırması, Dinamik ATR Stop-Loss, Kademeli Slippage) çalıştırır.
4. Tüm bağımsız out-of-sample pencerelerini birleştirerek tek bir kesintisiz
   'Walk-Forward Portföy Özsermaye Eğrisi' (Cumulative Equity Curve) üretir ve raporlar.
5. En güncel tüm veri üzerinde canlı üretim modelini (models/lgbm_reg.txt, models/lgbm_cls.txt)
   eğitir ve Champion/Challenger Gatekeeper üzerinden canlıya terfi/onay kararı verir.
6. 'output/daily_signals.json' dosyasını Walk-Forward onaylı şampiyon model sinyalleriyle günceller.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb

from preprocessing import load_and_preprocess_pipeline
from benchmark_tree import generate_tabular_lag_features, compute_relevance_labels
from backtest import run_portfolio_backtest, print_backtest_report
from ensemble import generate_champion_actionable_signals
from champion_gatekeeper import evaluate_and_promote_challenger, initialize_inaugural_champion


def run_walk_forward_validation(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    windows: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    buffer_k: int = 10,
    weighting: str = "inverse_vol",
    use_stop_loss: bool = True,
    bist30_slippage_bps: float = 10.0,
    midcap_slippage_bps: float = 25.0,
    min_day_tickers: Optional[int] = None,
    update_live_model: bool = True,
    rebalance_step: int = 5,
    liquid_universe_only: bool = True,
    macro_regime_col: Optional[str] = "Macro_Regime_Bull"
) -> Dict[str, Any]:
    """
    Kayan pencereli (Expanding Window) Walk-Forward validasyonu ve model eğitimini yürütür.
    """
    if windows is None:
        windows = [
            {
                "name": "Pencere 1 (2023 Rejimi)",
                "train_start": "2016-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-01-01",
                "test_end": "2023-12-31"
            },
            {
                "name": "Pencere 2 (2024 Rejimi)",
                "train_start": "2016-01-01",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31"
            },
            {
                "name": "Pencere 3 (2025+ Modern Rejim)",
                "train_start": "2016-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2026-12-31"
            }
        ]

    df = df_features.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fold_results = []
    oos_predictions = []

    print("=" * 115)
    print("BIST 100 KURUMSAL WALK-FORWARD VALİDASYON VE MODEL EĞİTİM MOTORU (LAMBDARANK)")
    print(f"Pencere Sayısı: {len(windows)} | Ağırlıklandırma: {weighting.upper()} | Dinamik Stop: {use_stop_loss}")
    print(f"Kademeli Kayma: BIST30 {bist30_slippage_bps:.0f} bps / Yan Tahta {midcap_slippage_bps:.0f} bps")
    print("=" * 115)
    print(f"{'Pencere':<30} | {'Eğitim Aralığı':<23} | {'Test Aralığı':<23} | {'Net Getiri':<10} | {'BIST100':<9} | {'Net Alfa':<10} | {'Sharpe':<6}")
    print("-" * 115)

    ranker_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 250,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }

    cls_params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 250,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }

    last_model_reg = None
    last_model_cls = None
    last_full_features = None

    for w in windows:
        w_name = w["name"]
        tr_start = pd.Timestamp(w["train_start"])
        tr_end = pd.Timestamp(w["train_end"])
        te_start = pd.Timestamp(w["test_start"])
        te_end = pd.Timestamp(w["test_end"])

        train_slice = df[(df["Date"] >= tr_start) & (df["Date"] <= tr_end)].copy()
        test_slice = df[(df["Date"] >= te_start) & (df["Date"] <= te_end)].copy()

        if train_slice.empty or test_slice.empty:
            print(f"  [!] {w_name} için yeterli veri yok, atlanıyor.")
            continue

        # Tabular Lag Özellikleri Üret
        train_lag, full_features = generate_tabular_lag_features(train_slice, feature_cols)
        test_lag, _ = generate_tabular_lag_features(test_slice, feature_cols)

        if train_lag.empty or test_lag.empty:
            continue

        train_lag = train_lag.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        test_lag = test_lag.sort_values(["Date", "Ticker"]).reset_index(drop=True)

        train_groups = train_lag.groupby("Date", sort=False).size().to_numpy()
        y_tr_rank = compute_relevance_labels(train_lag, target_col="Target_Excess_Return", num_bins=5)

        X_tr = train_lag[full_features]
        y_tr_cls = train_lag["Target_Direction_Alpha"]
        X_te = test_lag[full_features]

        # Ranker ve Classifier Eğitimi
        model_reg = lgb.LGBMRanker(**ranker_params)
        model_reg.fit(X_tr, y_tr_rank, group=train_groups)

        model_cls = lgb.LGBMClassifier(**cls_params)
        model_cls.fit(X_tr, y_tr_cls)

        last_model_reg = model_reg
        last_model_cls = model_cls
        last_full_features = full_features

        test_lag = test_lag.copy()
        test_lag["pred_reg"] = model_reg.predict(X_te)
        test_lag["pred_cls_prob"] = model_cls.predict_proba(X_te)[:, 1]
        test_lag["rank_pred"] = test_lag.groupby("Date")["pred_reg"].rank(pct=True) - 0.5

        # Portföy Backtesti (5 Günlük Rebalans & Makro Rejim Kapısı Devrede)
        res = run_portfolio_backtest(
            test_lag,
            signal_col="rank_pred",
            actual_return_col="Target_Return",
            top_k=top_k,
            buffer_k=buffer_k,
            benchmark_col="Target_Index_Return",
            weighting=weighting,
            use_stop_loss=use_stop_loss,
            bist30_slippage_bps=bist30_slippage_bps,
            midcap_slippage_bps=midcap_slippage_bps,
            min_day_tickers=min_day_tickers,
            rebalance_step=rebalance_step,
            liquid_universe_only=liquid_universe_only,
            macro_regime_col=macro_regime_col
        )

        if "error" in res:
            print(f"  [!] {w_name} için backtest hatası: {res['error']}, atlanıyor.")
            continue

        fold_res = {
            "window_name": w_name,
            "train_period": f"{w['train_start']} -> {w['train_end']}",
            "test_period": f"{w['test_start']} -> {w['test_end']}",
            "net_return": res["total_net_return"],
            "bench_return": res["total_bench_return"],
            "net_alpha": res["total_net_return"] - res["total_bench_return"],
            "sharpe": res["sharpe_ratio"],
            "max_dd": res["max_drawdown"],
            "stopped_trades": res.get("total_stopped_trades", 0),
            "n_days": res["n_trading_days"]
        }
        fold_results.append(fold_res)
        oos_predictions.append(test_lag)

        tr_str = f"{w['train_start'][:7]}..{w['train_end'][:7]}"
        te_str = f"{w['test_start'][:7]}..{w['test_end'][:7]}"
        net_str = f"%{res['total_net_return']:+.1f}"
        bist_str = f"%{res['total_bench_return']:+.1f}"
        alpha_str = f"%{fold_res['net_alpha']:+.1f}"
        sharpe_str = f"{res['sharpe_ratio']:.2f}"
        print(f"{w_name:<30} | {tr_str:<23} | {te_str:<23} | {net_str:<10} | {bist_str:<9} | {alpha_str:<10} | {sharpe_str:<6}")

    print("-" * 115)

    if not oos_predictions:
        print("[!] Hiçbir Walk-Forward penceresi tamamlanamadı.")
        return {"error": "Walk-Forward tamamlanamadı."}

    # Tüm Out-of-Sample Tahminleri Birleştir (Birleşik Walk-Forward Performansı)
    df_combined = pd.concat(oos_predictions, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    combined_res = run_portfolio_backtest(
        df_combined,
        signal_col="rank_pred",
        actual_return_col="Target_Return",
        top_k=top_k,
        buffer_k=buffer_k,
        benchmark_col="Target_Index_Return",
        weighting=weighting,
        use_stop_loss=use_stop_loss,
        bist30_slippage_bps=bist30_slippage_bps,
        midcap_slippage_bps=midcap_slippage_bps,
        min_day_tickers=min_day_tickers,
        rebalance_step=rebalance_step,
        liquid_universe_only=liquid_universe_only,
        macro_regime_col=macro_regime_col
    )

    wf_net_alpha = combined_res["total_net_return"] - combined_res["total_bench_return"]
    wf_sharpe = combined_res["sharpe_ratio"]

    comb_net = f"%{combined_res['total_net_return']:+.1f}"
    comb_bist = f"%{combined_res['total_bench_return']:+.1f}"
    comb_alpha = f"%{wf_net_alpha:+.1f}"
    comb_sharpe = f"{wf_sharpe:.2f}"
    print(f"{'BİRLEŞİK WALK-FORWARD (OOS)':<30} | {'Tüm Genişleyen':<23} | {'2023 - 2026':<23} | {comb_net:<10} | {comb_bist:<9} | {comb_alpha:<10} | {comb_sharpe:<6}")
    print("=" * 115)

    # Canlı Model Eğitimi & Gatekeeper Entegrasyonu
    if update_live_model:
        print("\n" + "#" * 85)
        print("CANLI ÜRETİM MODELİ: EN GÜNCEL VERİ İLE EĞİTİLİYOR (2016 -> GÜNÜMÜZ)...")
        print("#" * 85)

        full_lag, full_features = generate_tabular_lag_features(df, feature_cols)
        full_lag = full_lag.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        full_groups = full_lag.groupby("Date", sort=False).size().to_numpy()
        y_full_rank = compute_relevance_labels(full_lag, target_col="Target_Excess_Return", num_bins=5)

        prod_reg = lgb.LGBMRanker(**ranker_params)
        prod_reg.fit(full_lag[full_features], y_full_rank, group=full_groups)

        prod_cls = lgb.LGBMClassifier(**cls_params)
        prod_cls.fit(full_lag[full_features], full_lag["Target_Direction_Alpha"])

        os.makedirs("models", exist_ok=True)
        prod_reg_path = "models/lgbm_reg.txt"
        prod_cls_path = "models/lgbm_cls.txt"
        prod_reg.booster_.save_model(prod_reg_path)
        prod_cls.booster_.save_model(prod_cls_path)
        print(f"[+] Canlı üretim modelleri kaydedildi: '{prod_reg_path}' ve '{prod_cls_path}'")

        # Son günün sinyallerini üret
        latest_date = full_lag["Date"].max()
        day_df = full_lag[full_lag["Date"] == latest_date].copy()
        day_df["pred_reg"] = prod_reg.predict(day_df[full_features])
        day_df["pred_prob"] = prod_cls.predict_proba(day_df[full_features])[:, 1]
        day_df["rank_pred"] = day_df["pred_reg"].rank(pct=True) - 0.5

        backtest_summary = {
            "champion_model": "Walk-Forward LightGBM (Genişleyen Model)",
            "trading_days": int(combined_res["n_trading_days"]),
            "portfolio_cagr_pct": round(float(combined_res["cagr_portfolio"]), 2),
            "benchmark_cagr_pct": round(float(combined_res["cagr_benchmark"]), 2),
            "net_alpha_cagr_pct": round(float(combined_res["cagr_portfolio"] - combined_res["cagr_benchmark"]), 2),
            "sharpe_ratio": round(float(combined_res["sharpe_ratio"]), 2),
            "max_drawdown_pct": round(float(combined_res["max_drawdown"]), 2),
            "win_rate_vs_benchmark_pct": round(float(combined_res["win_rate_vs_benchmark"]), 2),
            "cash_days_pct": round(float(combined_res.get("cash_days_pct", 0.0)), 2)
        }

        signal_pack = generate_champion_actionable_signals(
            df_signals=day_df,
            champion_name="Walk-Forward LightGBM",
            score_col="rank_pred",
            reg_col="pred_reg",
            prob_col="pred_prob",
            conviction_threshold=-0.10,
            backtest_metrics=backtest_summary
        )

        # Champion / Challenger Gatekeeper Terfisi
        is_promoted, active_name, final_pack = evaluate_and_promote_challenger(
            challenger_name="Walk-Forward LightGBM (2016-2026)",
            challenger_alpha=round(float(wf_net_alpha), 2),
            challenger_sharpe=round(float(wf_sharpe), 2),
            challenger_weights_path=prod_reg_path,
            challenger_signal_pack=signal_pack
        )

        # Walk-Forward Tablosunu JSON'a ekle
        final_pack["walk_forward_oos_summary"] = {
            "total_net_return_pct": round(float(combined_res["total_net_return"]), 2),
            "benchmark_return_pct": round(float(combined_res["total_bench_return"]), 2),
            "net_alpha_pct": round(float(wf_net_alpha), 2),
            "sharpe_ratio": round(float(wf_sharpe), 2),
            "max_drawdown_pct": round(float(combined_res["max_drawdown"]), 2),
            "fold_results": fold_results
        }

        os.makedirs("output", exist_ok=True)
        with open("output/daily_signals.json", "w", encoding="utf-8") as f:
            json.dump(final_pack, f, ensure_ascii=False, indent=2)
        print("[+] 'output/daily_signals.json' Walk-Forward şampiyon sinyalleriyle güncellendi.")

    return {
        "fold_results": fold_results,
        "combined_performance": combined_res,
        "df_combined_oos": df_combined
    }


def main():
    print("=" * 85)
    print("    BIST 100 10 YILLIK PERİYODİK WALK-FORWARD EĞİTİM VE VALİDASYON HATTI")
    print("=" * 85)
    data_dict = load_and_preprocess_pipeline(period="10y", seq_len=10)

    # 10 yıllık tüm panel verisini birleştir (Train + Val + Test)
    full_df = pd.concat([data_dict["train_df"], data_dict["val_df"], data_dict["test_df"]], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    print(f"[*] Tam 10 Yıllık Panel Verisi: {full_df['Date'].min().strftime('%Y-%m-%d')} -> {full_df['Date'].max().strftime('%Y-%m-%d')} ({len(full_df):,} satır)")

    wf_res = run_walk_forward_validation(
        df_features=full_df,
        feature_cols=data_dict["feature_cols"],
        update_live_model=True
    )


if __name__ == "__main__":
    main()
