"""
BIST 100 Multi-Task PyTorch GPU Eğitim ve Değerlendirme Hattı

Bu script:
1. Yahoo Finance üzerinden canlı veriyi çeker ve 3D tensörleri üretir.
2. NVIDIA GPU (RTX 3050 Ti) üzerinde AMP (Automatic Mixed Precision - FP16)
   kullanarak bellek tasarruflu ve yüksek hızlı eğitim gerçekleştirir.
3. Regresyon (SmoothL1 / Huber) ve Sınıflandırma (BCEWithLogits) kayıplarını birleştirir.
4. Doğrulama (Validation) kümesinde en iyi skoru veren modeli 'models/bist_dual_model_best.pt'
   olarak kaydeder.
5. Test kümesi (2024+) üzerinde Spearman Rank IC (Information Coefficient),
   Yön İsabet Oranı (Hit Ratio) ve kümülatif getiri simülasyonu raporlar.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from preprocessing import load_and_preprocess_pipeline
from model import BISTDualTargetModel


class BISTDataset(Dataset):
    """3D Zaman Serisi Tensörlerini PyTorch Tensörlerine Sarmalayan Veri Kümesi."""
    def __init__(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        dates: Optional[list] = None
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        if dates is not None and len(dates) == len(X):
            self.date_ts = torch.tensor([int(pd.Timestamp(d).timestamp()) for d in dates], dtype=torch.int64)
        else:
            self.date_ts = torch.zeros(len(X), dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_reg[idx], self.y_cls[idx], self.date_ts[idx]


class DateBatchSampler(Sampler):
    """
    Tarih Bazlı Kesitsel Batch Sampler (Date-Wise Cross-Sectional Sampler).
    Her batch içine aynı güne ait hisseleri toplar.
    Böylece batch içinde kesitsel sıralama (ListNet / Pairwise Ranking)
    kayıpları eksiksiz ve gürültüsüz olarak hesaplanabilir.
    """
    def __init__(self, date_ts_list, dates_per_batch: int = 1, shuffle: bool = True):
        self.dates_per_batch = max(1, dates_per_batch)
        self.shuffle = shuffle

        date_to_indices = {}
        ts_array = date_ts_list.cpu().numpy() if isinstance(date_ts_list, torch.Tensor) else np.array(date_ts_list)
        for idx, ts in enumerate(ts_array):
            if ts not in date_to_indices:
                date_to_indices[ts] = []
            date_to_indices[ts].append(idx)

        self.unique_dates = list(date_to_indices.keys())
        self.date_to_indices = date_to_indices

    def __iter__(self):
        dates = list(self.unique_dates)
        if self.shuffle:
            np.random.shuffle(dates)

        for i in range(0, len(dates), self.dates_per_batch):
            batch_dates = dates[i:i + self.dates_per_batch]
            batch_indices = []
            for d in batch_dates:
                batch_indices.extend(self.date_to_indices[d])
            if batch_indices:
                yield batch_indices

    def __len__(self):
        return (len(self.unique_dates) + self.dates_per_batch - 1) // self.dates_per_batch


def directional_consistency_loss(pred_reg: torch.Tensor, pred_cls: torch.Tensor) -> torch.Tensor:
    """
    Regresyon getiri tahmini ile sınıflandırma logiti arasındaki çelişkiyi cezalandırır.
    Örn: pred_reg < 0 iken pred_cls > 0 ise ceza keser.
    """
    penalty = F.relu(-torch.tanh(pred_reg) * pred_cls)
    return torch.mean(penalty)


def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Diferansiyellenebilir Pearson Korelasyon Kaybı (1 - Corr).
    Modelin ham değer yerine kesitsel ilişkiyi ve hisseleri doğru sıralamayı öğrenmesini sağlar.
    """
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_diff = pred - pred_mean
    target_diff = target - target_mean

    cov = torch.sum(pred_diff * target_diff)
    pred_var = torch.sum(pred_diff ** 2)
    target_var = torch.sum(target_diff ** 2)

    corr = cov / (torch.sqrt(pred_var * target_var) + 1e-7)
    return 1.0 - torch.clamp(corr, -1.0, 1.0)


def date_wise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    date_ts: torch.Tensor,
    temperature: float = 0.5,
    margin: float = 0.02,
    margin_diff: float = 0.005
) -> torch.Tensor:
    """
    Gün Bazlı Kesitsel Sıralama Kaybı (Cross-Sectional ListNet + Pairwise Margin Loss).
    Batch içindeki örnekleri 'date_ts' bazında gruplar ve her gün için:
    1. ListNet Softmax Çapraz Entropi (Top-k hisse getiri dağılım uyumu)
    2. Pairwise Margin Ranking (Gerçekleşen getiri farkı belirgin çiftler arasında ayrım)
    hesaplar.
    """
    unique_dates = torch.unique(date_ts)
    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    valid_count = 0

    for dt in unique_dates:
        mask = (date_ts == dt)
        p_day = pred[mask]
        y_day = target[mask]
        n = p_day.size(0)

        if n < 3:
            continue

        # 1. ListNet Softmax Cross-Entropy
        p_true = F.softmax(y_day / temperature, dim=0)
        p_pred_log = F.log_softmax(p_day / temperature, dim=0)
        listnet = -torch.sum(p_true * p_pred_log)

        # 2. Pairwise Margin Ranking
        diff_y = y_day.unsqueeze(1) - y_day.unsqueeze(0)
        diff_p = p_day.unsqueeze(1) - p_day.unsqueeze(0)
        pair_mask = (diff_y > margin_diff)

        if pair_mask.any():
            pairwise = F.relu(margin - diff_p)[pair_mask].mean()
        else:
            pairwise = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        total_loss = total_loss + listnet + pairwise
        valid_count += 1

    if valid_count > 0:
        return total_loss / valid_count
    else:
        p_true = F.softmax(target / temperature, dim=0)
        p_pred_log = F.log_softmax(pred / temperature, dim=0)
        return -torch.sum(p_true * p_pred_log)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_reg: nn.Module,
    criterion_cls: nn.Module,
    device: torch.device,
    alpha_reg: float = 0.2,
    gamma_rank: float = 1.0,
    alpha_cls: float = 0.3,
    lambda_cons: float = 0.2
) -> Dict[str, float]:
    """Modelin doğrulama veya test kümesi üzerindeki finansal ve kesitsel performansını hesaplar."""
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_rank_loss = 0.0
    total_cons_loss = 0.0

    all_preds_reg = []
    all_targets_reg = []
    all_preds_cls = []
    all_targets_cls = []
    all_date_ts = []
    correct_dirs = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y_reg, batch_y_cls, batch_dates in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_reg = batch_y_reg.to(device, non_blocking=True)
            batch_y_cls = batch_y_cls.to(device, non_blocking=True)
            batch_dates_dev = batch_dates.to(device, non_blocking=True)

            use_amp = (device.type == "cuda")
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_reg, pred_cls = model(batch_x)
                l_reg = criterion_reg(pred_reg, batch_y_reg)
                l_cls = criterion_cls(pred_cls, batch_y_cls)
                l_cons = directional_consistency_loss(pred_reg, pred_cls)
                l_rank = date_wise_ranking_loss(pred_reg, batch_y_reg, batch_dates_dev)
                loss = alpha_reg * l_reg + gamma_rank * l_rank + alpha_cls * l_cls + lambda_cons * l_cons

            b_size = batch_x.size(0)
            total_loss += loss.item() * b_size
            total_reg_loss += l_reg.item() * b_size
            total_cls_loss += l_cls.item() * b_size
            total_rank_loss += l_rank.item() * b_size
            total_cons_loss += l_cons.item() * b_size
            total_samples += b_size

            # Yön İsabeti (Logit > 0 ise Yükseliş: 1)
            pred_dirs = (pred_cls > 0.0).float()
            correct_dirs += (pred_dirs == batch_y_cls).sum().item()

            all_preds_reg.extend(pred_reg.cpu().numpy())
            all_targets_reg.extend(batch_y_reg.cpu().numpy())
            all_preds_cls.extend(pred_cls.cpu().numpy())
            all_targets_cls.extend(batch_y_cls.cpu().numpy())
            all_date_ts.extend(batch_dates.numpy())

    all_preds_reg = np.array(all_preds_reg)
    all_targets_reg = np.array(all_targets_reg)
    all_date_ts = np.array(all_date_ts)

    # Temel Metrikler
    hit_ratio = (correct_dirs / total_samples) * 100.0 if total_samples > 0 else 50.0
    mae = np.mean(np.abs(all_preds_reg - all_targets_reg)) if len(all_preds_reg) > 0 else 0.0
    rmse = np.sqrt(np.mean((all_preds_reg - all_targets_reg) ** 2)) if len(all_preds_reg) > 0 else 0.0

    # Günlük Kesitsel Spearman Rank IC ve Long-Short Spread (Quant Fon Standardı)
    df_eval = pd.DataFrame({
        "ts": all_date_ts,
        "pred_reg": all_preds_reg,
        "target_reg": all_targets_reg
    })

    daily_ics = []
    daily_spreads = []

    for _, group in df_eval.groupby("ts"):
        if len(group) >= 5:
            ic, _ = spearmanr(group["pred_reg"].values, group["target_reg"].values)
            if not np.isnan(ic):
                daily_ics.append(ic)

            # En yüksek %10 beklenen getiri ile en düşük %10 beklenen getiri spreadi
            sorted_group = group.sort_values("pred_reg")
            k = max(1, int(len(sorted_group) * 0.10))
            bottom_ret = sorted_group.iloc[:k]["target_reg"].mean()
            top_ret = sorted_group.iloc[-k:]["target_reg"].mean()
            daily_spreads.append(top_ret - bottom_ret)

    daily_ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
    daily_ic_std = float(np.std(daily_ics)) if daily_ics else 1e-6
    daily_ic_ir = daily_ic_mean / (daily_ic_std + 1e-6)
    long_short_spread = float(np.mean(daily_spreads)) if daily_spreads else 0.0

    return {
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "loss_reg": total_reg_loss / total_samples if total_samples > 0 else 0.0,
        "loss_cls": total_cls_loss / total_samples if total_samples > 0 else 0.0,
        "loss_rank": total_rank_loss / total_samples if total_samples > 0 else 0.0,
        "loss_cons": total_cons_loss / total_samples if total_samples > 0 else 0.0,
        "hit_ratio": hit_ratio,
        "daily_ic": daily_ic_mean,
        "daily_ic_ir": daily_ic_ir,
        "long_short_spread": long_short_spread,
        "mae": mae,
        "rmse": rmse
    }


def train_pipeline(
    period: str = "10y",
    seq_len: int = 15,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 5e-4,
    hidden_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.40,
    alpha_reg: float = 0.2,
    gamma_rank: float = 1.0,
    alpha_cls: float = 0.3,
    lambda_cons: float = 0.2,
    gamma_corr: Optional[float] = None,
    patience: int = 10,
    cell_type: str = "gru",
    model_save_path: str = "models/bist_dual_model_best.pt",
    data_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # Geriye dönük uyumluluk
    if gamma_corr is not None:
        gamma_rank = gamma_corr

    # 1. Cihaz ve Donanım Hazırlığı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("BIST100 MULTI-TASK PYTORCH GPU EĞİTİM BAŞLATILIYOR (V2 - QUANT REFORM)")
    print("=" * 70)
    print(f"Çalışma Cihazı (Device)     : {device}")
    if device.type == "cuda":
        print(f"Ekran Kartı                 : {torch.cuda.get_device_name(0)}")
        print(f"Toplam VRAM                 : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB")
        torch.backends.cudnn.benchmark = True
    print("=" * 70)

    # 2. Canlı Veri Çekimi ve Tensör Üretimi (Hedef: Excess Return & Direction Alpha)
    if data_dict is None:
        data_dict = load_and_preprocess_pipeline(
            period=period,
            seq_len=seq_len,
            reg_target="Target_Excess_Return",
            cls_target="Target_Direction_Alpha"
        )

    train_dataset = BISTDataset(data_dict["X_train"], data_dict["y_reg_train"], data_dict["y_cls_train"], data_dict.get("dates_train"))
    val_dataset = BISTDataset(data_dict["X_val"], data_dict["y_reg_val"], data_dict["y_cls_val"], data_dict.get("dates_val"))
    test_dataset = BISTDataset(data_dict["X_test"], data_dict["y_reg_test"], data_dict["y_cls_test"], data_dict.get("dates_test"))

    # Tarih Bazlı Kesitsel DataLoader
    has_dates = (train_dataset.date_ts > 0).any()
    if has_dates:
        dates_per_batch = max(1, batch_size // 40)
        train_sampler = DateBatchSampler(train_dataset.date_ts, dates_per_batch=dates_per_batch, shuffle=True)
        val_sampler = DateBatchSampler(val_dataset.date_ts, dates_per_batch=dates_per_batch, shuffle=False)
        test_sampler = DateBatchSampler(test_dataset.date_ts, dates_per_batch=dates_per_batch, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, pin_memory=(device.type == "cuda"))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    print(f"\nVeri Boyutları:")
    print(f"  -> Train Örnek Sayısı: {len(train_dataset):,} ({len(train_loader)} batch)")
    print(f"  -> Val Örnek Sayısı  : {len(val_dataset):,} ({len(val_loader)} batch)")
    print(f"  -> Test Örnek Sayısı : {len(test_dataset):,} ({len(test_loader)} batch)")
    print(f"  -> Özellik Sayısı    : {data_dict['X_train'].shape[2]} ({len(data_dict['feature_cols'])} özellik)")

    # 3. Model, Kayıp Fonksiyonu ve Optimizer
    model = BISTDualTargetModel(
        num_features=data_dict["X_train"].shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        cell_type=cell_type
    ).to(device)

    # Huber Loss (SmoothL1) + BCE
    criterion_reg = nn.SmoothL1Loss(beta=1.0)
    criterion_cls = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 4. Eğitim Döngüsü
    print("\n" + "=" * 70)
    print(f"EĞİTİM DÖNGÜSÜ BAŞLIYOR ({cell_type.upper()} + Tarih Bazlı ListNet/Margin Sıralama Kaybı)")
    print("=" * 70)

    best_val_score = -float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_x, batch_y_reg, batch_y_cls, batch_dates in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_reg = batch_y_reg.to(device, non_blocking=True)
            batch_y_cls = batch_y_cls.to(device, non_blocking=True)
            batch_dates_dev = batch_dates.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_reg, pred_cls = model(batch_x)
                loss_reg = criterion_reg(pred_reg, batch_y_reg)
                loss_cls = criterion_cls(pred_cls, batch_y_cls)
                loss_cons = directional_consistency_loss(pred_reg, pred_cls)
                loss_rank = date_wise_ranking_loss(pred_reg, batch_y_reg, batch_dates_dev)
                loss = alpha_reg * loss_reg + gamma_rank * loss_rank + alpha_cls * loss_cls + lambda_cons * loss_cons

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            b_size = batch_x.size(0)
            train_loss += loss.item() * b_size
            train_samples += b_size

        scheduler.step()
        epoch_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
        epoch_time = time.time() - t0

        # Doğrulama (Validation)
        val_metrics = evaluate(model, val_loader, criterion_reg, criterion_cls, device, alpha_reg, gamma_rank, alpha_cls, lambda_cons)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Hit: %{val_metrics['hit_ratio']:.2f} | "
            f"Daily IC: {val_metrics['daily_ic']:.4f} (IR: {val_metrics['daily_ic_ir']:.2f}) | "
            f"L/S Spread: %{val_metrics['long_short_spread']:.2f} | "
            f"MAE: %{val_metrics['mae']:.2f}"
        )

        # En İyi Modeli Kaydetme: Quant Standardı (Val Daily IC + L/S Spread Skoru)
        val_score = val_metrics["daily_ic"] + (val_metrics["long_short_spread"] / 100.0)

        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "val_score": val_score,
                "feature_cols": data_dict["feature_cols"],
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "cell_type": cell_type
            }, model_save_path)
            print(f"  [*] [YENİ EN İYİ MODEL] Epoch {epoch} kaydedildi! (Val IC: {val_metrics['daily_ic']:.4f}, L/S: %{val_metrics['long_short_spread']:.2f}, Skor: {val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[ERKEN DURDURMA] Doğrulama IC skoru {patience} epoch boyunca iyileşmedi.")
                break

    # 5. Test Kümesi Değerlendirmesi (2024+)
    print("\n" + "=" * 70)
    print(f"EN İYİ {cell_type.upper()} MODELİ İLE TEST KÜMESİ (2024+) PERFORMANSI")
    print("=" * 70)
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    eval_model = BISTDualTargetModel(
        num_features=data_dict["X_train"].shape[2],
        hidden_dim=checkpoint.get("hidden_dim", hidden_dim),
        num_layers=checkpoint.get("num_layers", num_layers),
        dropout=checkpoint.get("dropout", dropout),
        cell_type=checkpoint.get("cell_type", cell_type)
    ).to(device)
    eval_model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(eval_model, test_loader, criterion_reg, criterion_cls, device, alpha_reg, gamma_rank, alpha_cls, lambda_cons)
    best_val_m = checkpoint.get("val_metrics", {})
    print(f"Doğrulama (Val) MAE            : %{best_val_m.get('mae', 0.0):.2f}")
    print(f"Doğrulama (Val) RMSE           : %{best_val_m.get('rmse', 0.0):.2f}")
    print(f"Doğrulama (Val) Daily IC       : {best_val_m.get('daily_ic', 0.0):.4f}")
    print(f"Test Toplam Kayıp (Loss)       : {test_metrics['loss']:.4f}")
    print(f"Test Ortalama Mutlak Hata (MAE): %{test_metrics['mae']:.2f}")
    print(f"Test Kök Ort. Kare Hata (RMSE) : %{test_metrics['rmse']:.2f}")
    print(f"Yön İsabet Oranı (Hit Ratio)   : %{test_metrics['hit_ratio']:.2f}")
    print(f"Günlük Kesitsel IC (Daily IC)  : {test_metrics['daily_ic']:.4f}")
    print(f"Bilgi Oranı (Daily IC IR)      : {test_metrics['daily_ic_ir']:.2f}")
    print(f"Long/Short Yayılımı (L/S Spread): %{test_metrics['long_short_spread']:.2f}")
    print("=" * 70)

    # 6. Quant Karar Destek Çıktısı Örneği
    print("\nÖrnek Karar Destek Değerlendirmesi:")
    if test_metrics["hit_ratio"] > 51.5 and test_metrics["daily_ic"] > 0.02:
        print(">> Model piyasa üzerinde pozitif kesitsel alfa sinyali üretmektedir. n8n karar ajanına bağlanabilir.")
    else:
        print(">> Model yön tahmininde baz seviyededir; haber akışı ve makro sentiment filtresiyle desteklenmelidir.")
    print("=" * 70)

    test_metrics["val_mae"] = float(best_val_m.get("mae", 0.0))
    test_metrics["val_rmse"] = float(best_val_m.get("rmse", 0.0))
    test_metrics["val_daily_ic"] = float(best_val_m.get("daily_ic", 0.0))
    test_metrics["val_loss"] = float(checkpoint.get("val_loss", 0.0))

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST Dual-Target GPU Training")
    parser.add_argument("--period", type=str, default="10y", help="Veri periyodu (örn. 10y, 5y, 2y)")
    parser.add_argument("--seq_len", type=int, default=15, help="Geçmiş pencere uzunluğu (gün)")
    parser.add_argument("--cell_type", type=str, choices=["rnn", "lstm", "gru"], default="gru", help="Hücre tipi: rnn, lstm, gru")
    parser.add_argument("--epochs", type=int, default=35, help="Epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=5e-4, help="Öğrenme oranı")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Gizli katman boyutu")
    parser.add_argument("--num_layers", type=int, default=1, help="Katman sayısı")
    parser.add_argument("--dropout", type=float, default=0.40, help="Dropout oranı")
    parser.add_argument("--alpha_reg", type=float, default=0.2, help="Regresyon kaybı ağırlığı")
    parser.add_argument("--gamma_rank", type=float, default=1.0, help="Sıralama (ranking) kaybı ağırlığı")
    parser.add_argument("--alpha_cls", type=float, default=0.3, help="Sınıflandırma kaybı ağırlığı")
    parser.add_argument("--lambda_cons", type=float, default=0.2, help="Tutarlılık kaybı ağırlığı")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping sayısı")
    parser.add_argument("--model_save_path", type=str, default="models/bist_dual_model_best.pt", help="Model kayıt yolu")
    args = parser.parse_args()

    train_pipeline(
        period=args.period,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        alpha_reg=args.alpha_reg,
        gamma_rank=args.gamma_rank,
        alpha_cls=args.alpha_cls,
        lambda_cons=args.lambda_cons,
        patience=args.patience,
        cell_type=args.cell_type,
        model_save_path=args.model_save_path
    )
