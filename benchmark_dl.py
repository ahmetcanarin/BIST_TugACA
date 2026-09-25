"""
BIST 100 Derin Öğrenme Sıralama Modelleri Benchmark Motoru (benchmark_dl.py)
Standart RNN (Vanilla) vs LSTM vs GRU

Bu modül:
1. Ortak Veri ve Tensör Hazırlığı:
   - 10 yıllık panel verisi (2016 - 2026) üzerinden 3D tensörler üretilir (seq_len=15 gün).
   - Hedef: 5 Günlük Endeks Üstü Getiri (Target_Excess_Return) ve Yön (Target_Direction_Alpha).
2. Yarışmacı Modellerin Eğitimi:
   - RNN (Vanilla / Elman): Basit hafıza, düşük parametre (Baseline).
   - LSTM (Long Short-Term Memory): 3 kapılı, hücre durumlu, uzun vadeli trend hafızası.
   - GRU (Gated Recurrent Unit): 2 kapılı, hafif, düşük overfitting riski.
   - Ortak Kayıp: Huber Regresyon + BCE Yön + Diferansiyellenebilir Pearson Korelasyon Kaybı (1 - Corr).
3. Out-of-Sample (2024 - 2026) Baş-Başa Değerlendirme:
   - Spearman Rank IC & Bilgi Oranı (IC IR)
   - 5 Günlük Kapanış-Kapanış Portföy Getirisi (Haftalık Rebalans & Likit 40 Evreni)
   - Tier 1 Makro Rejim Kapısı (%50 Yıllık Repo Getirisi)
   - Net Alfa, Sharpe Oranı, Maksimum Düşüş (MDD) ve Kazanma Oranı
4. Şampiyon Tescili ve Gatekeeper Terfisi:
   - En yüksek Sharpe ve Net Alfa üreten model "Canlı Şampiyon" seçilir.
   - Ağırlıklar 'models/bist_dual_model_best.pt' olarak tescil edilir.
   - 'output/daily_signals.json' güncellenir.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocessing import load_and_preprocess_pipeline
from model import BISTDualTargetModel
from train import BISTDataset, train_pipeline, evaluate
from backtest import run_portfolio_backtest, print_backtest_report
from champion_gatekeeper import evaluate_and_promote_challenger
from ensemble import generate_champion_actionable_signals
from extraction import BIST_LIQUID_40


def run_rnn_lstm_gru_benchmark(
    period: str = "10y",
    seq_len: int = 15,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 5e-4,
    hidden_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.40,
    rebalance_step: int = 5,
    models_to_test: Optional[List[str]] = None
) -> Dict[str, Any]:
    if models_to_test is None:
        models_to_test = ["rnn", "lstm", "gru"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print("    BIST 100 DERİN ÖĞRENME MODEL BENCHMARK'I: RNN vs LSTM vs GRU")
    print(f"    Periyot: {period} | Seq Len: {seq_len} Gün | Rebalans: {rebalance_step} Gün (Haftalık Holding)")
    print(f"    Donanım: {device} | Kayıp: Huber + Date-Wise ListNet/Margin Ranking + BCE")
    print("=" * 105)

    # 1. Veri Hazırlığı (Ortak 10 Yıllık Panel)
    print("\n[*] 10 Yıllık Panel Verisi Çekiliyor ve 3D Tensörler Hazırlanıyor...")
    data_dict = load_and_preprocess_pipeline(
        period=period,
        seq_len=seq_len,
        reg_target="Target_Excess_Return",
        cls_target="Target_Direction_Alpha"
    )

    test_df = data_dict["test_df"].copy()
    test_dates = data_dict.get("dates_test", [])
    test_tickers = data_dict.get("tickers_test", [])
    X_test = data_dict["X_test"]

    os.makedirs("models", exist_ok=True)
    benchmark_results = []
    trained_models = {}

    # 2. Her Model Tipini Eğit ve Test Et
    for m_type in models_to_test:
        m_name = m_type.upper()
        save_path = f"models/{m_type}_dual_model.pt"

        print("\n" + "#" * 85)
        print(f"    MODEL EĞİTİMİ VE TESTİ: {m_name} (Hücre: {m_type})")
        print("#" * 85)

        # Modeli Eğit
        t_start = time.time()
        test_eval_metrics = train_pipeline(
            period=period,
            seq_len=seq_len,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=m_type,
            model_save_path=save_path,
            data_dict=data_dict,
            patience=10
        )
        train_elapsed = time.time() - t_start

        # En iyi ağırlıkları yükle ve Test Kümesi üzerinde tahmin üret
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model = BISTDualTargetModel(
            num_features=X_test.shape[2],
            hidden_dim=checkpoint.get("hidden_dim", hidden_dim),
            num_layers=checkpoint.get("num_layers", num_layers),
            dropout=checkpoint.get("dropout", dropout),
            cell_type=m_type
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Batch inference
        test_dataset = BISTDataset(data_dict["X_test"], data_dict["y_reg_test"], data_dict["y_cls_test"], data_dict.get("dates_test"))
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

        preds_reg = []
        preds_cls_prob = []
        with torch.no_grad():
            for bx, _, _, _ in test_loader:
                bx = bx.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pr, pc = model(bx)
                preds_reg.extend(pr.cpu().numpy().tolist())
                preds_cls_prob.extend(torch.sigmoid(pc).cpu().numpy().tolist())

        # Tahminleri Test DataFrame'i ile eşle
        pred_df = pd.DataFrame({
            "Date": test_dates,
            "Ticker": test_tickers,
            "pred_reg": preds_reg,
            "pred_prob": preds_cls_prob
        })

        # Test veri çerçevesine tahminleri merge et
        merged_test = pd.merge(test_df, pred_df, on=["Date", "Ticker"], how="inner")
        merged_test["rank_pred"] = merged_test.groupby("Date")["pred_reg"].rank(pct=True) - 0.5

        # 3. Portföy Backtesti (5 Günlük Rebalans, Likit 40, Makro Rejim Kapısı)
        bt_res = run_portfolio_backtest(
            df_eval=merged_test,
            signal_col="rank_pred",
            actual_return_col="Target_Return",
            top_k=5,
            buffer_k=10,
            rebalance_step=rebalance_step,
            liquid_universe_only=True,
            macro_regime_col="Macro_Regime_Bull",
            use_stop_loss=True,
            bist30_slippage_bps=10.0,
            midcap_slippage_bps=20.0
        )

        net_ret = bt_res["total_net_return"]
        bench_ret = bt_res["total_bench_return"]
        net_alpha = net_ret - bench_ret
        sharpe = bt_res["sharpe_ratio"]
        mdd = bt_res["max_drawdown"]
        win_rate = bt_res["win_rate_vs_benchmark"]
        ic = test_eval_metrics["daily_ic"]
        ic_ir = test_eval_metrics["daily_ic_ir"]

        res_item = {
            "model_type": m_type,
            "model_name": f"{m_name}_Ranker",
            "weights_path": save_path,
            "test_ic": ic,
            "test_ic_ir": ic_ir,
            "net_return": net_ret,
            "bench_return": bench_ret,
            "net_alpha": net_alpha,
            "sharpe": sharpe,
            "max_dd": mdd,
            "win_rate": win_rate,
            "training_time_sec": round(train_elapsed, 1),
            "backtest_details": bt_res,
            "merged_df": merged_test
        }
        benchmark_results.append(res_item)
        trained_models[m_type] = res_item

    # 4. Karşılaştırmalı Liderlik Tablosu (Leaderboard)
    # Sıralama Kriteri: Sharpe Oranı ve Net Alfa bileşimi
    benchmark_results.sort(key=lambda x: (x["sharpe"], x["net_alpha"]), reverse=True)

    print("\n" + "=" * 105)
    print("    RNN vs LSTM vs GRU RESMİ BENCHMARK SONUÇ TABLOSU (2024-2026 TEST DÖNEMİ)")
    print("=" * 105)
    print(f"{'Sıra':<5} | {'Model':<15} | {'Net Alfa':<10} | {'Net Getiri':<10} | {'BIST 100':<9} | {'Sharpe':<8} | {'Max DD':<8} | {'Daily IC':<9} | {'Süre (s)':<8}")
    print("-" * 105)

    for rank, res in enumerate(benchmark_results, start=1):
        trophy = "🏆 " if rank == 1 else f"#{rank} "
        m_label = trophy + res["model_name"]
        a_str = f"%{res['net_alpha']:+.2f}"
        r_str = f"%{res['net_return']:+.2f}"
        b_str = f"%{res['bench_return']:+.2f}"
        s_str = f"{res['sharpe']:.2f}"
        m_str = f"%{res['max_dd']:.1f}"
        ic_str = f"{res['test_ic']:.3f}"
        t_str = f"{res['training_time_sec']}s"
        print(f"{rank:<5} | {m_label:<15} | {a_str:<10} | {r_str:<10} | {b_str:<9} | {s_str:<8} | {m_str:<8} | {ic_str:<9} | {t_str:<8}")

    print("=" * 105)

    # 5. Şampiyonun Canlıya Terfisi (Gatekeeper)
    champion = benchmark_results[0]
    print(f"\n[+] BENCHMARK KAZANANI: {champion['model_name']} (Sharpe: {champion['sharpe']:.2f}, Net Alfa: %{champion['net_alpha']:+.2f})")

    prod_path = "models/bist_dual_model_best.pt"

    # Son günün sinyallerini üret
    merged_last = champion["merged_df"]
    latest_dt = merged_last["Date"].max()
    day_df = merged_last[merged_last["Date"] == latest_dt].copy()

    backtest_summary = {
        "champion_model": champion["model_name"],
        "trading_days": int(champion["backtest_details"]["n_trading_days"]),
        "portfolio_cagr_pct": round(float(champion["backtest_details"]["cagr_portfolio"]), 2),
        "benchmark_cagr_pct": round(float(champion["backtest_details"]["cagr_benchmark"]), 2),
        "net_alpha_cagr_pct": round(float(champion["backtest_details"]["cagr_portfolio"] - champion["backtest_details"]["cagr_benchmark"]), 2),
        "sharpe_ratio": round(float(champion["sharpe"]), 2),
        "max_drawdown_pct": round(float(champion["max_dd"]), 2),
        "win_rate_vs_benchmark_pct": round(float(champion["win_rate"]), 2),
        "test_spearman_ic": round(float(champion["test_ic"]), 3)
    }

    signal_pack = generate_champion_actionable_signals(
        df_signals=day_df,
        champion_name=champion["model_name"],
        score_col="rank_pred",
        reg_col="pred_reg",
        prob_col="pred_prob",
        conviction_threshold=-0.10,
        backtest_metrics=backtest_summary,
        liquid_universe_only=True
    )

    # Champion Gatekeeper Terfi Denetimi
    is_promoted, active_name, final_pack = evaluate_and_promote_challenger(
        challenger_name=f"{champion['model_name']} (10Y Haftalık Momentum)",
        challenger_alpha=round(float(champion["net_alpha"]), 2),
        challenger_sharpe=round(float(champion["sharpe"]), 2),
        challenger_weights_path=champion["weights_path"],
        challenger_signal_pack=signal_pack
    )

    # Benchmark tablosunu sinyal paketine ekle
    final_pack["rnn_lstm_gru_benchmark_leaderboard"] = [
        {
            "rank": i + 1,
            "model_name": r["model_name"],
            "net_alpha_pct": round(r["net_alpha"], 2),
            "net_return_pct": round(r["net_return"], 2),
            "benchmark_return_pct": round(r["bench_return"], 2),
            "sharpe_ratio": round(r["sharpe"], 2),
            "max_drawdown_pct": round(r["max_dd"], 2),
            "test_ic": round(r["test_ic"], 3),
            "win_rate_pct": round(r["win_rate"], 2)
        }
        for i, r in enumerate(benchmark_results)
    ]

    os.makedirs("output", exist_ok=True)
    with open("output/daily_signals.json", "w", encoding="utf-8") as f:
        json.dump(final_pack, f, ensure_ascii=False, indent=2)
    print("[+] 'output/daily_signals.json' yeni Şampiyon sinyalleriyle güncellendi.")

    return {
        "leaderboard": benchmark_results,
        "champion": champion,
        "final_signal_pack": final_pack
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 RNN vs LSTM vs GRU Benchmark")
    parser.add_argument("--period", type=str, default="10y", help="Veri periyodu (örn: 10y, 5y)")
    parser.add_argument("--seq_len", type=int, default=15, help="Pencere uzunluğu (gün)")
    parser.add_argument("--epochs", type=int, default=35, help="Her model için epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=5e-4, help="Öğrenme oranı")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Gizli katman boyutu")
    parser.add_argument("--num_layers", type=int, default=1, help="Katman sayısı")
    parser.add_argument("--dropout", type=float, default=0.40, help="Dropout oranı")
    parser.add_argument("--rebalance_step", type=int, default=5, help="Haftalık holding süresi (5 gün)")
    args = parser.parse_args()

    run_rnn_lstm_gru_benchmark(
        period=args.period,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rebalance_step=args.rebalance_step
    )
