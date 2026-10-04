"""
BIST 100 — LSTM vs GRU Volatilite Modelleri Karşılaştırmalı Eğitim ve Başarı Raporu

Bu script:
1. Aynı veriyle hem LSTM hem GRU volatilite modellerini eğitir.
2. Test kümesi üzerinde R², MAE, RMSE ve Rejim Doğruluk skorlarını karşılaştırır.
3. Sonuçları JSON dosyasına kaydeder (Streamlit dashboard'un okuması için).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import json
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_and_preprocess_pipeline
from model_volatility import BISTVolatilityModel
from model_lstm_volatility import BISTVolatilityLSTMModel


class VolatilityDataset(Dataset):
    def __init__(self, X: np.ndarray, y_vol: np.ndarray, y_regime: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_vol = torch.tensor(y_vol, dtype=torch.float32)
        self.y_regime = torch.tensor(y_regime, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_vol[idx], self.y_regime[idx]


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    epochs: int,
    lr: float,
    save_path: str
) -> Dict[str, Any]:
    """Tek bir modeli eğitir ve test sonuçlarını döner."""

    device = torch.device("cpu")
    model = model.to(device)

    criterion_reg = nn.HuberLoss(delta=1.0)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float("inf")

    train_history = []

    print(f"\n{'='*80}")
    print(f"  🚀 {model_name} MODELİ EĞİTİMİ BAŞLIYOR")
    print(f"{'='*80}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch_x, batch_y_vol, batch_y_regime in train_loader:
            optimizer.zero_grad()
            pred_vol, pred_regime = model(batch_x)

            l_reg = criterion_reg(pred_vol, batch_y_vol)
            l_cls = criterion_cls(pred_regime, batch_y_regime)
            loss = l_reg + 2.0 * l_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = train_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_vol = []
        val_targets_vol = []
        val_preds_reg = []
        val_targets_reg = []

        with torch.no_grad():
            for batch_x, batch_y_vol, batch_y_regime in val_loader:
                pred_vol, pred_regime = model(batch_x)
                l_reg = criterion_reg(pred_vol, batch_y_vol)
                l_cls = criterion_cls(pred_regime, batch_y_regime)
                loss = l_reg + 2.0 * l_cls
                val_loss += loss.item() * batch_x.size(0)
                val_preds_vol.extend(pred_vol.numpy())
                val_targets_vol.extend(batch_y_vol.numpy())
                val_preds_reg.extend(pred_regime.argmax(dim=1).numpy())
                val_targets_reg.extend(batch_y_regime.numpy())

        epoch_val_loss = val_loss / len(val_ds)
        val_acc = accuracy_score(val_targets_reg, val_preds_reg) * 100.0
        val_r2 = r2_score(val_targets_vol, val_preds_vol)

        train_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_r2": round(val_r2, 4),
            "val_regime_acc": round(val_acc, 2)
        })

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val R²: {val_r2:.4f} | Rejim İsabeti: %{val_acc:.2f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_features": model.num_features,
                "model_name": model_name
            }, save_path)

    # Test Kümesi Değerlendirmesi
    print(f"\n{'='*80}")
    print(f"  🏆 {model_name} - TEST KÜMESİ (2024+) PERFORMANSI")
    print(f"{'='*80}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds_vol = []
    test_targets_vol = []
    test_preds_reg = []
    test_targets_reg = []

    with torch.no_grad():
        for batch_x, batch_y_vol, batch_y_regime in test_loader:
            pred_vol, pred_regime = model(batch_x)
            test_preds_vol.extend(pred_vol.numpy())
            test_targets_vol.extend(batch_y_vol.numpy())
            test_preds_reg.extend(pred_regime.argmax(dim=1).numpy())
            test_targets_reg.extend(batch_y_regime.numpy())

    r2 = r2_score(test_targets_vol, test_preds_vol)
    mae = mean_absolute_error(test_targets_vol, test_preds_vol)
    rmse = np.sqrt(mean_squared_error(test_targets_vol, test_preds_vol))
    regime_acc = accuracy_score(test_targets_reg, test_preds_reg) * 100.0

    regime_names = ["Sıkışma (Düşük)", "Normal", "Patlama (Yüksek)"]
    cls_report = classification_report(
        test_targets_reg, test_preds_reg,
        target_names=regime_names,
        output_dict=True,
        zero_division=0
    )
    conf_matrix = confusion_matrix(test_targets_reg, test_preds_reg).tolist()

    print(f"  • Volatilite R² Skoru           : {r2:.4f}")
    print(f"  • Ortalama Mutlak Hata (MAE)    : %{mae:.2f}")
    print(f"  • Kök Ortalama Kare Hata (RMSE) : %{rmse:.2f}")
    print(f"  • Rejim Sınıflandırma İsabeti   : %{regime_acc:.2f}")

    results = {
        "model_name": model_name,
        "test_r2": round(r2, 4),
        "test_mae": round(mae, 2),
        "test_rmse": round(rmse, 2),
        "test_regime_accuracy": round(regime_acc, 2),
        "classification_report": cls_report,
        "confusion_matrix": conf_matrix,
        "train_history": train_history,
        "best_val_loss": round(best_val_loss, 4)
    }

    return results


def run_comparison_pipeline(
    period: str = "10y",
    seq_len: int = 30,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    print("=" * 80)
    print("📊 BIST 100 LSTM vs GRU VOLATİLİTE MODELLERİ KARŞILAŞTIRMALI EĞİTİM")
    print("=" * 80)

    # 1. Veri Hazırlığı
    data_dict = load_and_preprocess_pipeline(period=period, seq_len=seq_len)
    X_train, X_val, X_test = data_dict["X_train"], data_dict["X_val"], data_dict["X_test"]
    num_features = X_train.shape[2]

    vol_idx = data_dict["feature_cols"].index("Volatility_20")

    y_vol_train = (X_train[:, -1, vol_idx] * np.sqrt(252) * 100.0).clip(5.0, 150.0)
    y_vol_val = (X_val[:, -1, vol_idx] * np.sqrt(252) * 100.0).clip(5.0, 150.0)
    y_vol_test = (X_test[:, -1, vol_idx] * np.sqrt(252) * 100.0).clip(5.0, 150.0)

    y_reg_train = np.where(y_vol_train < 22.0, 0, np.where(y_vol_train <= 42.0, 1, 2))
    y_reg_val = np.where(y_vol_val < 22.0, 0, np.where(y_vol_val <= 42.0, 1, 2))
    y_reg_test = np.where(y_vol_test < 22.0, 0, np.where(y_vol_test <= 42.0, 1, 2))

    train_ds = VolatilityDataset(X_train, y_vol_train, y_reg_train)
    val_ds = VolatilityDataset(X_val, y_vol_val, y_reg_val)
    test_ds = VolatilityDataset(X_test, y_vol_test, y_reg_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"\nVeri Boyutları:")
    print(f"  -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  -> Özellik Sayısı: {num_features}")

    # 2. GRU Modeli Eğitimi
    gru_model = BISTVolatilityModel(
        num_features=num_features, hidden_dim=64, num_layers=2, dropout=0.20
    )
    gru_results = train_single_model(
        model=gru_model,
        model_name="GRU (Gated Recurrent Unit)",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        epochs=epochs,
        lr=lr,
        save_path="models/bist_volatility_gru.pt"
    )

    # 3. LSTM Modeli Eğitimi
    lstm_model = BISTVolatilityLSTMModel(
        num_features=num_features, hidden_dim=64, num_layers=2, dropout=0.20
    )
    lstm_results = train_single_model(
        model=lstm_model,
        model_name="LSTM (Long Short-Term Memory)",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        epochs=epochs,
        lr=lr,
        save_path="models/bist_volatility_lstm.pt"
    )

    # 4. Karşılaştırma Tablosu
    print("\n" + "=" * 80)
    print("🏆 LSTM vs GRU KARŞILAŞTIRMALI BAŞARI RAPORU")
    print("=" * 80)
    print(f"{'Metrik':<40} {'GRU':>15} {'LSTM':>15} {'Kazanan':>15}")
    print("-" * 85)

    metrics = [
        ("Volatilite R² Skoru", "test_r2", True),
        ("Ortalama Mutlak Hata (MAE %)", "test_mae", False),
        ("Kök Ort. Kare Hata (RMSE %)", "test_rmse", False),
        ("Rejim İsabet Oranı (%)", "test_regime_accuracy", True),
    ]

    for label, key, higher_better in metrics:
        gru_val = gru_results[key]
        lstm_val = lstm_results[key]
        if higher_better:
            winner = "🟢 GRU" if gru_val > lstm_val else ("🟢 LSTM" if lstm_val > gru_val else "⚪ Eşit")
        else:
            winner = "🟢 GRU" if gru_val < lstm_val else ("🟢 LSTM" if lstm_val < gru_val else "⚪ Eşit")
        print(f"  {label:<38} {gru_val:>15} {lstm_val:>15} {winner:>15}")

    print("=" * 80)

    # 5. Sonuçları JSON olarak kaydet (Streamlit için)
    comparison = {
        "gru": gru_results,
        "lstm": lstm_results,
        "data_info": {
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "num_features": num_features,
            "seq_len": seq_len,
            "epochs": epochs,
            "feature_cols": data_dict["feature_cols"]
        }
    }

    results_path = "models/lstm_vs_gru_comparison.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n📁 Karşılaştırma sonuçları kaydedildi: {results_path}")

    return comparison


if __name__ == "__main__":
    run_comparison_pipeline(epochs=15)
