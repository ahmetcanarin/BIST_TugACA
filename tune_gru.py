"""
BIST 100 Optuna GPU Hiperparametre Optimizasyon Motoru (tune_gru.py)

Bu script:
1. 10 Yıllık modern rejim (2016+) verisini çeker ve Train (<=2021), Val (2022-2023), Test (2024+) olarak ayırır.
2. Optuna ile Bayesian / TPE araması yaparak Nedensel GRU için en yüksek Validation IC ve
   Long/Short yayılımını veren optimal hiperparametreleri (hidden_dim, num_layers, dropout, seq_len, lr, batch_size, vb.) keşfeder.
3. Bulunan en iyi hiperparametreleri 'models/optuna_best_gru_params.json' dosyasına kaydeder.
4. En iyi konfigürasyon ile final modeli eğitip 'models/bist_dual_model_best.pt' olarak saklar.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import time
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

from preprocessing import (
    load_and_preprocess_pipeline,
    split_by_date,
    scale_features,
    create_dual_sequences,
    prepare_panel_data,
    add_technical_features
)
from extraction import tum_hisseleri_cek, endeks_verisini_cek, makro_verileri_cek
from model import BISTDualTargetModel
from train import BISTDataset, evaluate, directional_consistency_loss, pearson_correlation_loss, train_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, base_data: Dict[str, Any], device: torch.device) -> float:
    # 1. Hiperparametre Alanı Tanımlama
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.10, 0.40, step=0.05)
    seq_len = trial.suggest_categorical("seq_len", [10, 15, 20])
    lr = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    alpha_cls = trial.suggest_float("alpha_cls", 0.5, 3.0, step=0.5)
    lambda_cons = trial.suggest_float("lambda_cons", 0.2, 1.0, step=0.2)
    gamma_corr = trial.suggest_float("gamma_corr", 0.2, 1.0, step=0.2)

    train_s = base_data["train_s"]
    val_s = base_data["val_s"]
    feature_cols = base_data["feature_cols"]

    # Kayan pencere tensörlerini seçilen seq_len'e göre üret
    X_train, y_reg_train, y_cls_train, _, dates_train = create_dual_sequences(
        train_s, feature_cols, reg_col="Target_Excess_Return", cls_col="Target_Direction_Alpha",
        seq_len=seq_len, sort_chronologically=True
    )
    X_val, y_reg_val, y_cls_val, _, dates_val = create_dual_sequences(
        val_s, feature_cols, reg_col="Target_Excess_Return", cls_col="Target_Direction_Alpha",
        seq_len=seq_len, sort_chronologically=True
    )

    train_ds = BISTDataset(X_train, y_reg_train, y_cls_train, dates_train)
    val_ds = BISTDataset(X_val, y_reg_val, y_cls_val, dates_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = BISTDualTargetModel(
        num_features=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion_reg = nn.SmoothL1Loss(beta=1.0)
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_score = -float("inf")
    patience = 4
    no_improve_epochs = 0
    max_epochs = 12

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y_reg, batch_y_cls, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_reg = batch_y_reg.to(device, non_blocking=True)
            batch_y_cls = batch_y_cls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_reg, pred_cls = model(batch_x)
                loss_reg = criterion_reg(pred_reg, batch_y_reg)
                loss_cls = criterion_cls(pred_cls, batch_y_cls)
                loss_cons = directional_consistency_loss(pred_reg, pred_cls)
                loss_corr = pearson_correlation_loss(pred_reg, batch_y_reg)
                total_loss = loss_reg + (alpha_cls * loss_cls) + (lambda_cons * loss_cons) + (gamma_corr * loss_corr)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # Val değerlendirmesi
        val_metrics = evaluate(model, val_loader, criterion_reg, criterion_cls, device, alpha_cls, lambda_cons, gamma_corr)
        val_score = val_metrics["daily_ic"] + (val_metrics["long_short_spread"] / 100.0)

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_score > best_val_score:
            best_val_score = val_score
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    return best_val_score


def run_optuna_tuning(n_trials: int = 15, period: str = "10y") -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("BIST 100 NEDENSEL GRU OPTUNA HİPERPARAMETRE OPTİMİZASYON MOTORU")
    print("=" * 80)
    print(f"  -> Cihaz               : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  -> Periyot             : {period} Modern Rejim (2016+)")
    print(f"  -> Deneme Sayısı (Trials): {n_trials}")
    print("=" * 80)

    # 1. Ham veriyi çek ve özellikleri hazırla
    print("\n[1/3] Canlı veri ve 26 özellik hazırlanıyor...")
    df_index = endeks_verisini_cek(period=period)
    df_macro = makro_verileri_cek(period=period)
    df_raw = tum_hisseleri_cek(period=period)

    df_panel = prepare_panel_data(df_raw)
    df_features = add_technical_features(df_panel, df_index=df_index, df_macro=df_macro)

    feature_cols = [
        "Log_Return", "HL_Spread", "CO_Return", "Log_Volume",
        "Volume_Change", "SMA10_Ratio", "SMA30_Ratio", "Volatility_20", "RSI_Norm",
        "Index_Log_Return", "Excess_Return", "Cross_Rank_Return",
        "USDTRY_Return", "USDTRY_Vol_20", "VIX_Level", "VIX_Change",
        "Brent_Return", "Gold_Return", "Beta_FX_60d", "Beta_Market_60d",
        "TCMB_Policy_Rate", "TCMB_Rate_Change", "TCMB_Days_Since_Decision",
        "CDS_Proxy_Chg5", "CDS_Proxy_ZScore_60d", "Beta_CDS_60d"
    ]

    train_df, val_df, test_df = split_by_date(df_features, train_end="2021-12-31", val_end="2023-12-31")
    train_s, val_s, test_s, scaler = scale_features(train_df, val_df, test_df, feature_cols)

    base_data = {
        "train_s": train_s,
        "val_s": val_s,
        "test_s": test_s,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols
    }

    # 2. Optuna Çalışması
    print(f"\n[2/3] Optuna çalışması başlatılıyor ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )

    t0 = time.time()
    def _opt_obj(t):
        return objective(t, base_data, device)

    study.optimize(_opt_obj, n_trials=n_trials, show_progress_bar=True)
    t_elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print(f"OPTUNA ARAMASI TAMAMLANDI! (Toplam Süre: {t_elapsed / 60:.1f} dk)")
    print("=" * 80)
    print(f"En İyi Validation Skoru: {study.best_value:.4f}")
    print("En İyi Parametreler:")
    for k, v in study.best_params.items():
        print(f"  -> {k:<15}: {v}")
    print("=" * 80)

    # Parametreleri kaydet
    os.makedirs("models", exist_ok=True)
    best_params_path = "models/optuna_best_gru_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_value": round(float(study.best_value), 4),
            "best_params": study.best_params,
            "period": period,
            "trials_count": n_trials,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"En iyi hiperparametreler kaydedildi: '{best_params_path}'")

    # 3. En İyi Parametrelerle Tam Eğitimi Çalıştır (100 Epoch, Erken Durdurma)
    print("\n[3/3] En iyi hiperparametrelerle nihai model eğitimi başlatılıyor...")
    bp = study.best_params
    train_pipeline(
        period=period,
        seq_len=bp["seq_len"],
        epochs=80,
        batch_size=bp["batch_size"],
        lr=bp["lr"],
        hidden_dim=bp["hidden_dim"],
        num_layers=bp["num_layers"],
        dropout=bp["dropout"],
        alpha_cls=bp["alpha_cls"],
        lambda_cons=bp["lambda_cons"],
        gamma_corr=bp["gamma_corr"],
        patience=15,
        model_save_path="models/bist_dual_model_best.pt"
    )

    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna GRU Hyperparameter Tuning")
    parser.add_argument("--n_trials", type=int, default=15, help="Optuna deneme sayısı")
    parser.add_argument("--period", type=str, default="10y", help="Veri periyodu")
    args = parser.parse_args()

    run_optuna_tuning(n_trials=args.n_trials, period=args.period)
