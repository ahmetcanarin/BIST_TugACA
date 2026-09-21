"""
BIST 100 LightGBM Tabular Benchmark ve Sinyal Üreticisi

Bu modül:
1. Panel verisinden (Date, Ticker, Features) zamansal lag özellikleri (Lag 1, 2, 3, 5) türetir.
2. LightGBM Regressor (Target_Excess_Return / Target_Rank) ve
   LightGBM Classifier (Target_Direction_Alpha) modellerini eğitir.
3. Test kümesinde kurumsal Quant metriklerini (Spearman Rank IC, IC IR, L/S Spread, Hit Ratio)
   hesaplar ve raporlar.
4. Öznitelik önem düzeylerini (Feature Importance) analiz ederek hangi makro/teknik
   göstergelerin alfa ürettiğini ortaya koyar.
5. Modelleri 'models/lgbm_reg.txt' ve 'models/lgbm_cls.txt' olarak kaydeder.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

from preprocessing import load_and_preprocess_pipeline


def generate_tabular_lag_features(
    df: pd.DataFrame,
    base_features: List[str],
    lags: List[int] = [1, 2, 3, 5]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tabular ağaç modelleri için her hissenin geçmiş adımlarındaki (t-1, t-2, ...)
    öznitelik değerlerini sütun olarak ekler (Lag Features).
    """
    df_out = df.sort_values(["Ticker", "Date"]).copy()
    added_cols = []

    for lag in lags:
        for col in ["Log_Return", "HL_Spread", "Volume_Change", "Excess_Return", "USDTRY_Return"]:
            if col in df_out.columns:
                lag_col = f"{col}_Lag{lag}"
                df_out[lag_col] = df_out.groupby("Ticker")[col].shift(lag)
                added_cols.append(lag_col)

    full_features = [c for c in base_features + added_cols if c in df_out.columns]
    df_out = df_out.dropna(subset=full_features).reset_index(drop=True)
    return df_out, full_features


def evaluate_quant_metrics(
    df_eval: pd.DataFrame,
    pred_col: str = "pred_return",
    target_col: str = "Target_Excess_Return",
    dir_target_col: str = "Target_Direction_Alpha"
) -> Dict[str, float]:
    """
    Günlük bazda kesitsel Spearman Rank IC, IC IR, Long/Short Spread ve Hit Ratio hesaplar.
    """
    daily_ics = []
    daily_spreads = []

    for _, group in df_eval.groupby("Date"):
        if len(group) >= 5:
            ic, _ = spearmanr(group[pred_col].values, group[target_col].values)
            if not np.isnan(ic):
                daily_ics.append(ic)

            sorted_group = group.sort_values(pred_col)
            k = max(1, int(len(sorted_group) * 0.10))
            bottom_ret = sorted_group.iloc[:k][target_col].mean()
            top_ret = sorted_group.iloc[-k:][target_col].mean()
            daily_spreads.append(top_ret - bottom_ret)

    daily_ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
    daily_ic_std = float(np.std(daily_ics)) if daily_ics else 1e-6
    daily_ic_ir = daily_ic_mean / (daily_ic_std + 1e-6)
    long_short_spread = float(np.mean(daily_spreads)) if daily_spreads else 0.0

    hit_ratio = 50.0
    if dir_target_col in df_eval.columns:
        pred_dir = (df_eval[pred_col] > 0).astype(float)
        hit_ratio = float((pred_dir == df_eval[dir_target_col]).mean() * 100.0)

    mae = float(np.mean(np.abs(df_eval[pred_col] - df_eval[target_col])))
    rmse = float(np.sqrt(np.mean((df_eval[pred_col] - df_eval[target_col]) ** 2)))

    return {
        "daily_ic": daily_ic_mean,
        "daily_ic_ir": daily_ic_ir,
        "long_short_spread": long_short_spread,
        "hit_ratio": hit_ratio,
        "mae": mae,
        "rmse": rmse
    }


def train_lightgbm_pipeline(
    data_dict: Dict[str, Any],
    model_dir: str = "models",
    target_reg: str = "Target_Excess_Return",
    target_cls: str = "Target_Direction_Alpha"
) -> Dict[str, Any]:
    """
    LightGBM modellerini eğitir, test performansı raporlar ve model dosyalarını kaydeder.
    """
    os.makedirs(model_dir, exist_ok=True)

    train_df = data_dict["train_df"].copy()
    val_df = data_dict["val_df"].copy()
    test_df = data_dict["test_df"].copy()
    base_features = data_dict["feature_cols"]

    print("=" * 70)
    print("LIGHTGBM TABULAR BENCHMARK EĞİTİMİ BAŞLATILIYOR")
    print("=" * 70)

    # 1. Lag Öznitelikleri Ekleme
    train_lag, full_features = generate_tabular_lag_features(train_df, base_features)
    val_lag, _ = generate_tabular_lag_features(val_df, base_features)
    test_lag, _ = generate_tabular_lag_features(test_df, base_features)

    print(f"Toplam Öznitelik Sayısı: {len(full_features)} (Temel: {len(base_features)} + Lagler: {len(full_features) - len(base_features)})")
    print(f"Train Satır: {len(train_lag):,} | Val Satır: {len(val_lag):,} | Test Satır: {len(test_lag):,}")

    X_tr = train_lag[full_features]
    y_tr_reg = train_lag[target_reg]
    y_tr_cls = train_lag[target_cls]

    X_va = val_lag[full_features]
    y_va_reg = val_lag[target_reg]
    y_va_cls = val_lag[target_cls]

    X_te = test_lag[full_features]
    y_te_reg = test_lag[target_reg]
    y_te_cls = test_lag[target_cls]

    # 2. LightGBM Regressor (Huber Loss ile uç değerlere dayanıklı alfa tahmini)
    print("\n[1/2] LightGBM Regressor (Target: Excess Return) eğitiliyor (Ağaç sayısı artırılıyor)...")
    reg_params = {
        "objective": "huber",
        "huber_alpha": 1.0,
        "metric": "huber",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_estimators": 250,
        "verbose": -1
    }

    reg_model = lgb.LGBMRegressor(**reg_params)
    reg_model.fit(
        X_tr, y_tr_reg,
        eval_set=[(X_va, y_va_reg)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    print(f"  -> Regressor Eğitilen Ağaç Sayısı: {reg_model.booster_.num_trees()}")

    # 3. LightGBM Classifier (Target: Direction Alpha)
    print("\n[2/2] LightGBM Classifier (Target: Direction Alpha) eğitiliyor (Ağaç sayısı artırılıyor)...")
    cls_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_estimators": 250,
        "verbose": -1
    }

    cls_model = lgb.LGBMClassifier(**cls_params)
    cls_model.fit(
        X_tr, y_tr_cls,
        eval_set=[(X_va, y_va_cls)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    print(f"  -> Classifier Eğitilen Ağaç Sayısı: {cls_model.booster_.num_trees()}")

    # 4. Modelleri Kaydetme
    reg_path = os.path.join(model_dir, "lgbm_reg.txt")
    cls_path = os.path.join(model_dir, "lgbm_cls.txt")
    reg_model.booster_.save_model(reg_path)
    cls_model.booster_.save_model(cls_path)
    print(f"\nModeller kaydedildi: '{reg_path}' ({reg_model.booster_.num_trees()} ağaç) ve '{cls_path}' ({cls_model.booster_.num_trees()} ağaç)")

    # 5. Test Kümesi Tahminleri ve Quant Metrikleri
    pred_test_reg = reg_model.predict(X_te)
    pred_test_cls_prob = cls_model.predict_proba(X_te)[:, 1]

    cols_to_keep = ["Date", "Ticker", target_reg, target_cls]
    if "Target_Return" in test_lag.columns and "Target_Return" not in cols_to_keep:
        cols_to_keep.append("Target_Return")

    df_test_eval = test_lag[cols_to_keep].copy()
    df_test_eval["pred_reg"] = pred_test_reg
    df_test_eval["pred_cls_prob"] = pred_test_cls_prob

    metrics = evaluate_quant_metrics(
        df_test_eval,
        pred_col="pred_reg",
        target_col=target_reg,
        dir_target_col=target_cls
    )

    print("\n" + "=" * 70)
    print("LIGHTGBM TEST KÜMESİ (OUT-OF-SAMPLE) QUANT PERFORMANSI")
    print("=" * 70)
    print(f"  -> Daily Spearman IC       : {metrics['daily_ic']:.4f}")
    print(f"  -> IC Information Ratio    : {metrics['daily_ic_ir']:.2f}")
    print(f"  -> Long/Short Spread Günlük: %{metrics['long_short_spread']:.2f}")
    print(f"  -> Yön İsabet Oranı (Hit)  : %{metrics['hit_ratio']:.2f}")
    print(f"  -> MAE                     : %{metrics['mae']:.2f}")
    print(f"  -> RMSE                    : %{metrics['rmse']:.2f}")
    print("=" * 70)

    # 6. En Önemli 10 Özellik (Feature Importance)
    feat_imp = pd.DataFrame({
        "Feature": full_features,
        "Importance": reg_model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nEn Yüksek Alfa Üreten İlk 10 Gösterge (Feature Importance):")
    for i, row in feat_imp.head(10).iterrows():
        print(f"  {i+1:02d}. {row['Feature']:<25}: {row['Importance']}")

    return {
        "reg_model": reg_model,
        "cls_model": cls_model,
        "metrics": metrics,
        "feature_importances": feat_imp,
        "test_eval_df": df_test_eval,
        "full_features": full_features
    }


if __name__ == "__main__":
    print("Canlı veri yfinance üzerinden çekiliyor (10y modern rejim)...")
    data_dict = load_and_preprocess_pipeline(period="10y", seq_len=10)
    results = train_lightgbm_pipeline(data_dict)
