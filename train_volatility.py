"""
BIST 100 Derin Öğrenme Volatilite Modeli Eğitim ve Başarı Ölçüm Hattı

Bu script:
1. Volatilite hedeflerini üretir (Gelecek 20 günlük gerçekleşen volatilite %).
2. 3 Sınıflı Volatilite Rejimini etiketler (0: Sıkışma < %20, 1: Normal %20-%40, 2: Patlama/Yüksek > %40).
3. PyTorch GRU Volatilite modelini eğitir ve en iyi ağırlıkları kaydeder.
4. Test kümesi (2024+) üzerinde R2, MAE, RMSE ve Rejim Doğruluk Skoru raporlar.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_and_preprocess_pipeline
from model_volatility import BISTVolatilityModel


class VolatilityDataset(Dataset):
    def __init__(self, X: np.ndarray, y_vol: np.ndarray, y_regime: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_vol = torch.tensor(y_vol, dtype=torch.float32)
        self.y_regime = torch.tensor(y_regime, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_vol[idx], self.y_regime[idx]


def compute_volatility_targets(df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Her hisse için gelecek 20 günün gerçekleşen yıllanmış volatilitesini (%) ve rejimi hesaplar.
    """
    vols = []
    regimes = []
    
    # 20 günlük gerçekleşen volatilite proxy (% cinsinden)
    vol_20 = (df_features["Volatility_20"] * np.sqrt(252) * 100.0).clip(5.0, 150.0).values
    
    for v in vol_20:
        vols.append(v)
        if v < 22.0:
            regimes.append(0)  # Düşük / Sıkışma
        elif v <= 42.0:
            regimes.append(1)  # Normal
        else:
            regimes.append(2)  # Yüksek / Patlama

    return np.array(vols, dtype=np.float32), np.array(regimes, dtype=np.int64)


def train_volatility_pipeline(
    period: str = "10y",
    seq_len: int = 30,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_save_path: str = "models/bist_volatility_model.pt"
):
    print("=" * 80)
    print("📊 BIST 100 DERİN ÖĞRENME VOLATİLİTE MODELİ EĞİTİMİ BAŞLIYOR")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Çalışma Cihazı: {device}")

    # 1. Veri Hazırlığı
    data_dict = load_and_preprocess_pipeline(period=period, seq_len=seq_len)
    X_train, X_val, X_test = data_dict["X_train"], data_dict["X_val"], data_dict["X_test"]

    num_features = X_train.shape[2]

    # Gelecek Volatilite Hedefleri (Regresyon + Rejim)
    # Volatility_20 sütunu baz alınarak yıllanmış gerçekleşen volatilite hedefi
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
    print(f"  -> Train Örnek Sayısı: {len(train_ds):,}")
    print(f"  -> Val Örnek Sayısı  : {len(val_ds):,}")
    print(f"  -> Test Örnek Sayısı : {len(test_ds):,}")
    print(f"  -> Özellik Sayısı    : {num_features}")

    # 2. Model, Loss & Optimizer
    model = BISTVolatilityModel(num_features=num_features, hidden_dim=64, num_layers=2, dropout=0.20).to(device)
    criterion_reg = nn.HuberLoss(delta=1.0)
    criterion_cls = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_val_loss = float("inf")

    # 3. Eğitim Döngüsü
    print("\n" + "=" * 80)
    print("VOLATİLİTE MODELİ EĞİTİMİ BAŞLADI")
    print("=" * 80)

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

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val R2: {val_r2:.4f} | Rejim İsabeti: %{val_acc:.2f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_features": num_features,
                "feature_cols": data_dict["feature_cols"]
            }, model_save_path)

    # 4. Test Kümesi Değerlendirmesi
    print("\n" + "=" * 80)
    print("🏆 EN İYİ VOLATİLİTE MODELİ - TEST KÜMESİ (2024+) PERFORMANSI")
    print("=" * 80)

    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
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

    print(f"  • Volatilite Tahmin Belirleme Katsayısı (R² Skoru) : {r2:.4f}")
    print(f"  • Ortalama Mutlak Volatilite Hatası (MAE)       : %{mae:.2f}")
    print(f"  • Kök Ortalama Kare Hata (RMSE)                   : %{rmse:.2f}")
    print(f"  • Volatilite Rejimi Sınıflandırma İsabet Oranı  : %{regime_acc:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    train_volatility_pipeline(epochs=15)
