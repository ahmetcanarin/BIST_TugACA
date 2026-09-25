"""
BIST 100 Bağımsız Model Benchmark ve Şampiyon Karar Destek Sistemi (V4)

Bu modül:
1. PyTorch Nedensel GRU ve Tabular LightGBM modellerini birbirinden tamamen bağımsız
   iki yarışmacı olarak konumlandırır (Zararlı ensemble birleştirmesini iptal eder).
2. Out-of-Sample Test Kümesinde (2024+) Head-to-Head Benchmark yarışını yürütür:
   - Model A: Nedensel GRU (Saf Hisse / Filtresiz)
   - Model B: Düzeltilmiş LightGBM (Saf Hisse / Filtresiz - 200+ Ağaç)
   - Nedensel GRU + Kalibre Conviction (Eşik: -%0.25 / -%0.50, %35 PPF)
   - LightGBM + Kalibre Conviction (Eşik: -%0.25 / -%0.50, %35 PPF)
3. Conviction Eşiği Duyarlılık Taraması (Sensitivity Sweep) ile %99 nakit kapanından
   çıkış dengesini modeller.
4. En yüksek net alfa ve Sharpe üreten modeli ŞAMPİYON MODEL ilan eder.
5. Canlı yatırım kararlarını (output/daily_signals.json) YALNIZCA şampiyon modelin
   tahminleri ve KAP haber duygu verileri ile üretir.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

from preprocessing import load_and_preprocess_pipeline
from benchmark_tree import generate_tabular_lag_features, train_lightgbm_pipeline, evaluate_quant_metrics
from model import BISTDualTargetModel
from backtest import run_portfolio_backtest, print_backtest_report, run_conviction_sensitivity_sweep
from extraction_kap import load_sentiment_cache
from extraction import BIST30_TICKERS, BIST_LIQUID_40


def predict_lightgbm(
    df: pd.DataFrame,
    full_features: List[str],
    reg_path: str = "models/lgbm_reg.txt",
    cls_path: str = "models/lgbm_cls.txt"
) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM modelleri üzerinden getiri ve olasılık tahmini yapar."""
    reg_booster = lgb.Booster(model_file=reg_path)
    cls_booster = lgb.Booster(model_file=cls_path)

    X = df[full_features]
    pred_reg = reg_booster.predict(X)
    pred_cls_prob = cls_booster.predict(X)
    return pred_reg, pred_cls_prob


def predict_gru(
    X_seq: np.ndarray,
    feature_cols: List[str],
    model_path: str = "models/bist_dual_model_best.pt",
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """PyTorch Nedensel GRU modeli üzerinden getiri ve yön logiti tahmini yapar."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_features = len(checkpoint.get("feature_cols", feature_cols))
    hidden_dim = checkpoint.get("hidden_dim", 64)
    num_layers = checkpoint.get("num_layers", 2)
    cell_type = checkpoint.get("cell_type", "gru")

    model = BISTDualTargetModel(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        cell_type=cell_type
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_reg_t, pred_cls_t = model(X_tensor)
        pred_reg = pred_reg_t.cpu().numpy()
        pred_cls_prob = torch.sigmoid(pred_cls_t).cpu().numpy()

    return pred_reg, pred_cls_prob


def check_gatekeeper_veto(news_info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    KAP ve Finansal Haber Türkçe BERT Gatekeeper Veto Filtresi.
    Model skoru yüksek olsa bile, şirketin güncel haberlerinde ciddi yasal/finansal risk veya
    aşırı negatif duygu (-0.20 altı) varsa LONG adayını veto eder.
    """
    sentiment = float(news_info.get("news_sentiment", 0.0))
    headline = str(news_info.get("headline", "")).lower()

    # 1. Aşırı Negatif Duygu Eşiği (< -0.20)
    if sentiment < -0.20:
        return True, f"Aşırı Negatif Haber Duygusu (Skor: {sentiment:+.2f} < -0.20)"

    # 2. Kritik Risk Anahtar Kelimeleri (Ceza, Soruşturma, Dava, İflas, Tedbir vb.)
    critical_keywords = [
        "ceza", "dava", "soruşturma", "durdurma", "iflas",
        "haciz", "tedbir", "zarar açıkladı", "konkordato"
    ]
    for kw in critical_keywords:
        if kw in headline:
            return True, f"Kritik Hukuki/Finansal Risk Başlığı ('{kw}')"

    return False, None


def generate_champion_actionable_signals(
    df_signals: pd.DataFrame,
    champion_name: str,
    score_col: str,
    reg_col: str,
    prob_col: str,
    conviction_threshold: float = -0.10,
    latest_date: Optional[str] = None,
    backtest_metrics: Optional[Dict[str, Any]] = None,
    liquid_universe_only: bool = True
) -> Dict[str, Any]:
    """
    YALNIZCA Şampiyon Modelin tahminlerini kullanarak n8n, Telegram ve
    Yatırım Komitesi için Top 5 LONG ve Bottom 5 KAÇIN sinyal paketini üretir.
    BIST Likit 40 evreni, Makro Rejim Kapısı ve KAP Gatekeeper Veto Filtresi tam entegredir.
    """
    df = df_signals.copy()
    if latest_date is None:
        min_tickers = min(50, max(5, int(df["Ticker"].nunique() * 0.7)))
        date_counts = df.groupby("Date").size()
        valid_dates = date_counts[date_counts >= min_tickers].index
        latest_date = valid_dates.max() if len(valid_dates) > 0 else df["Date"].max()

    day_df = df[df["Date"] == latest_date].copy()

    # Tier 1 Makro Rejim Kapısı Denetimi
    macro_bull = True
    if "Macro_Regime_Bull" in day_df.columns:
        m_vals = day_df["Macro_Regime_Bull"].dropna()
        if not m_vals.empty:
            macro_bull = bool(m_vals.iloc[0] > 0.5)

    # Likit 40 Evren Filtresi (Sığ hisse slippage tuzağını ortadan kaldırır)
    if liquid_universe_only:
        liquid_mask = day_df["Ticker"].str.replace(".IS", "").isin(BIST_LIQUID_40)
        if liquid_mask.any():
            day_df = day_df[liquid_mask].copy()

    day_df = day_df.sort_values(score_col, ascending=False)
    avg_expected_excess = float(day_df.head(10)[reg_col].mean()) if not day_df.empty else 0.0

    # KAP Önbelleğinden en son haber başlığını ve skorunu çek
    cache = load_sentiment_cache()
    def _get_news_info(ticker_clean):
        matches = [v for k, v in cache.items() if k.startswith(f"{ticker_clean}_")]
        if matches:
            last = matches[-1]
            return {
                "headline": last.get("title", ""),
                "news_sentiment": last.get("score", 0.0)
            }
        return {"headline": "Son günlerde özel durum açıklaması yok.", "news_sentiment": 0.0}

    gatekeeper_vetoed = []
    top5_long = []

    # Tüm sıralanmış hisseleri tara; Gatekeeper veto edenleri filtrele, temiz olan ilk 5'i al
    for _, row in day_df.iterrows():
        if len(top5_long) >= 5:
            break

        exp_ret = float(row[reg_col])
        conf = float(row[prob_col]) * 100.0
        score = float(row[score_col])
        t_clean = row["Ticker"].replace(".IS", "")
        news_info = _get_news_info(t_clean)

        is_vetoed, veto_reason = check_gatekeeper_veto(news_info)
        if is_vetoed:
            gatekeeper_vetoed.append({
                "ticker": t_clean,
                "model_score": round(score, 3),
                "expected_excess_return": f"%{round(exp_ret, 2)}",
                "veto_reason": veto_reason,
                "headline": news_info["headline"],
                "news_sentiment": news_info["news_sentiment"]
            })
            continue

        if exp_ret > 0 and conf >= 50.0:
            action = "GÜÇLÜ AL (Yüksek Alfa)"
        elif exp_ret > 0 and conf < 50.0:
            action = "KADEMELİ AL / TAKİP ET"
        elif exp_ret <= 0 and score > 0.25:
            action = "DEFANSİF TUT (Endekse Dirençli)"
        else:
            action = "NÖTR / İZLE"

        top5_long.append({
            "ticker": t_clean,
            "model_score": round(score, 3),
            "expected_excess_return": f"%{round(exp_ret, 2)}",
            "confidence": f"%{round(conf, 1)}",
            "action": action,
            "latest_news": news_info["headline"],
            "news_sentiment": news_info["news_sentiment"]
        })

    # Ters Volatilite Ağırlıklandırması ve Dinamik ATR Stop-Loss Parametreleri
    if top5_long:
        vols = []
        for s in top5_long:
            t_orig = s["ticker"] + ".IS"
            match = day_df[day_df["Ticker"] == t_orig]
            vol = float(match["Volatility_20"].iloc[0]) if (not match.empty and "Volatility_20" in match.columns and pd.notna(match["Volatility_20"].iloc[0])) else 0.02
            vols.append(max(vol, 0.005))

        inv_vols = [1.0 / v for v in vols]
        sum_inv = sum(inv_vols)
        norm_weights = [iv / sum_inv for iv in inv_vols]
        # Top 5 hisse için Min %8, Max %30 tekil hisse tavanı
        clipped = [min(0.30, max(0.08, w)) for w in norm_weights]
        sum_clipped = sum(clipped)
        final_weights = [c / sum_clipped for c in clipped]

        for i, s in enumerate(top5_long):
            t_orig = s["ticker"] + ".IS"
            match = day_df[day_df["Ticker"] == t_orig]
            atr_pct = float(match["ATR_14_Pct"].iloc[0]) if (not match.empty and "ATR_14_Pct" in match.columns and pd.notna(match["ATR_14_Pct"].iloc[0])) else 2.5
            stop_dist = float(np.clip(2.0 * atr_pct, 3.0, 7.0))
            is_bist30 = (s["ticker"] in BIST30_TICKERS)
            is_liquid40 = (s["ticker"] in BIST_LIQUID_40)

            s["recommended_weight"] = f"%{round(final_weights[i] * 100.0, 1)}"
            s["dynamic_stop_loss"] = f"-%{round(stop_dist, 1)}"
            s["liquidity_tier"] = "BIST 30 (10 bps)" if is_bist30 else ("Likit 40 (20 bps)" if is_liquid40 else "Yan Tahta (25 bps)")

    bottom5_short = []
    for _, row in day_df.tail(5).iterrows():
        exp_ret = float(row[reg_col])
        conf_avoid = (1.0 - float(row[prob_col])) * 100.0
        score = float(row[score_col])
        t_clean = row["Ticker"].replace(".IS", "")
        news_info = _get_news_info(t_clean)

        if exp_ret < 0 and conf_avoid >= 50.0:
            action = "AŞIRI SATIŞ / RİSKTEN KAÇIN"
        elif exp_ret < 0 and conf_avoid < 50.0:
            action = "AĞIRLIK AZALT (Negatif Alfa)"
        elif score < -0.25:
            action = "YÜKSEK RİSK / ENDEKS ALTI"
        else:
            action = "ZAYIF PERFORMANS"

        bottom5_short.append({
            "ticker": t_clean,
            "model_score": round(score, 3),
            "expected_excess_return": f"%{round(exp_ret, 2)}",
            "confidence": f"%{round(conf_avoid, 1)}",
            "action": action,
            "latest_news": news_info["headline"],
            "news_sentiment": news_info["news_sentiment"]
        })

    is_cash_day = (avg_expected_excess < conviction_threshold) or (not macro_bull)
    if not macro_bull:
        portfolio_action = "DEFANSİF MOD / 100% PPF REPO NAKİT (Tier 1 Makro Kapısı: XU100 < SMA50)"
    elif avg_expected_excess < conviction_threshold:
        portfolio_action = "DEFANSİF MOD / NAKİTTE BEKLE (Piyasa Geneli Negatif Alfa)"
    elif avg_expected_excess < 0:
        portfolio_action = "SEÇİCİ DEFANSİF MOD / KONTROLLÜ HİSSE (Relatif Alfa Arayışı)"
    else:
        portfolio_action = "HİSSE PORTFÖYÜ AÇ (Pozitif Alfa Fırsatı)"

    pack = {
        "report_date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
        "champion_model": champion_name,
        "calibrated_threshold": f"%{conviction_threshold:+.2f}",
        "top10_avg_expected_alpha": f"%{round(avg_expected_excess, 2)}",
        "portfolio_action": portfolio_action,
        "is_cash_recommended": is_cash_day,
        "macro_regime_bull": macro_bull,
        "top_longs": top5_long,
        "bottom_shorts": bottom5_short,
        "gatekeeper_vetoed": gatekeeper_vetoed
    }

    if backtest_metrics:
        pack["champion_backtest_performance"] = backtest_metrics

    return pack


def run_head_to_head_benchmark(
    df_signals: pd.DataFrame,
    top_k: int = 10,
    buffer_k: int = 20,
    smooth_window: int = 3
) -> Dict[str, Any]:
    """
    Nedensel GRU ve LightGBM modellerini bağımsız baş başa yarıştıran Quant karnesi.
    """
    rf_daily = (1.35 ** (1.0 / 252.0)) - 1.0  # Yıllık %35 BIST Gecelik Repo / PPF Getirisi

    configs = [
        ("1. Model A: Nedensel GRU (Saf Hisse / Eşit / Stopsuz)", "rank_gru", None, 0.0, 0.0, "equal", False),
        ("2. Model B: LightGBM (Saf Hisse / Eşit / Stopsuz)", "rank_lgbm", None, 0.0, 0.0, "equal", False),
        ("3. Model A (GRU) + Ters Volatilite + Dinamik ATR Stop", "rank_gru", None, 0.0, 0.0, "inverse_vol", True),
        ("4. Model B (LGBM) + Ters Volatilite + Dinamik ATR Stop", "rank_lgbm", None, 0.0, 0.0, "inverse_vol", True),
        ("5. Model A (GRU) + Kalibre Conv (-%0.10) + Risk Yön.", "rank_gru", "pred_gru_reg", -0.10, rf_daily, "inverse_vol", True),
        ("6. Model B (LGBM) + Kalibre Conv (-%0.10) + Risk Yön.", "rank_lgbm", "pred_lgbm_reg", -0.10, rf_daily, "inverse_vol", True),
        ("7. Model A (GRU) + Aşırı Defansif (Eşik %0.0) [Nakit Kapanı]", "rank_gru", "pred_gru_reg", 0.0, rf_daily, "equal", False)
    ]

    results = {}
    print("\n" + "=" * 115)
    print(f"BIST 100 BAĞIMSIZ MODEL BENCHMARK KARNESİ (Top {top_k}, 2024+ Out-of-Sample)")
    print("=" * 115)
    print(f"{'Strateji Konfigürasyonu':<52} | {'Net Getiri':<11} | {'BIST100':<9} | {'Net Alfa':<10} | {'Sharpe':<7} | {'Max DD':<9} | {'Win%':<6} | {'Nakit%':<6}")
    print("-" * 115)

    for name, sig_col, conv_col, conv_thresh, cash_yield, weighting, use_stop in configs:
        res = run_portfolio_backtest(
            df_signals,
            signal_col=sig_col,
            actual_return_col="Target_Return",
            top_k=top_k,
            buffer_k=buffer_k,
            smooth_window=smooth_window,
            conviction_col=conv_col,
            min_conviction_threshold=conv_thresh,
            cash_daily_yield=cash_yield,
            benchmark_col="Target_Index_Return",
            weighting=weighting,
            use_stop_loss=use_stop
        )
        results[name] = res
        net_ret = f"%{res['total_net_return']:.2f}"
        bist_ret = f"%{res['total_bench_return']:.2f}"
        net_alpha = f"%{res['total_net_return'] - res['total_bench_return']:+.2f}"
        sharpe = f"{res['sharpe_ratio']:.2f}"
        mdd = f"%{res['max_drawdown']:.2f}"
        win_rate = f"%{res['win_rate_vs_benchmark']:.1f}"
        cash_pct = f"%{res.get('cash_days_pct', 0.0):.1f}"

        print(f"{name:<52} | {net_ret:<11} | {bist_ret:<9} | {net_alpha:<10} | {sharpe:<7} | {mdd:<9} | {win_rate:<6} | {cash_pct:<6}")
    print("=" * 115)
    return results


def run_independent_model_benchmark(
    data_dict: Optional[Dict[str, Any]] = None,
    gru_model_path: str = "models/bist_dual_model_best.pt"
) -> Dict[str, Any]:
    print("=" * 75)
    print("BIST 100 BAĞIMSIZ MODEL BENCHMARK VE ŞAMPİYON SEÇİM SİSTEMİ (V4)")
    print("=" * 75)

    # 1. Canlı Veri Çekimi (10 yıllık modern rejim panel verisi 2016+)
    seq_len = 20
    if os.path.exists("models/optuna_best_gru_params.json"):
        try:
            with open("models/optuna_best_gru_params.json", "r", encoding="utf-8") as f:
                opt_info = json.load(f)
                seq_len = opt_info.get("best_params", {}).get("seq_len", 20)
        except Exception:
            pass

    if data_dict is None:
        print(f"Boru hattı yükleniyor (seq_len={seq_len} gün)...")
        data_dict = load_and_preprocess_pipeline(period="10y", seq_len=seq_len)

    # 2. Düzeltilmiş LightGBM Modelleri (En az 150-300 Ağaç Öğrenmeli)
    reg_path = "models/lgbm_reg.txt"
    cls_path = "models/lgbm_cls.txt"
    if not (os.path.exists(reg_path) and os.path.exists(cls_path)):
        print("\n[!] LightGBM modelleri eğitiliyor (Ağaç kapasitesi artırılmış)...")
        train_lightgbm_pipeline(data_dict)
    else:
        print(f"\n[+] Mevcut LightGBM modelleri yüklendi ('{reg_path}', '{cls_path}').")

    # 3. Test Kümesi Tahminleri (2024+ Out-of-Sample)
    test_lag, full_features = generate_tabular_lag_features(data_dict["test_df"], data_dict["feature_cols"])
    pred_lgbm_reg, pred_lgbm_cls_prob = predict_lightgbm(test_lag, full_features, reg_path, cls_path)

    signal_cols = ["Date", "Ticker", "Target_Return", "Target_Excess_Return"]
    for extra_col in ["Target_Index_Return", "Target_Low_Return", "ATR_14_Pct", "Volatility_20", "ADV20_TRY"]:
        if extra_col in test_lag.columns:
            signal_cols.append(extra_col)
    df_signals = test_lag[signal_cols].copy()
    df_signals["pred_lgbm_reg"] = pred_lgbm_reg
    df_signals["pred_lgbm_prob"] = pred_lgbm_cls_prob
    df_signals["rank_lgbm"] = df_signals.groupby("Date")["pred_lgbm_reg"].rank(pct=True) - 0.5

    # GRU Tahminlerini Yükle
    if os.path.exists(gru_model_path):
        p_reg_gru, p_cls_gru = predict_gru(data_dict["X_test"], data_dict["feature_cols"], gru_model_path)
        df_gru = pd.DataFrame({
            "Date": pd.to_datetime(data_dict["dates_test"]),
            "Ticker": data_dict["tickers_test"],
            "pred_gru_reg": p_reg_gru,
            "pred_gru_prob": p_cls_gru
        })
        df_signals = pd.merge(df_signals, df_gru, on=["Date", "Ticker"], how="left")
        df_signals["pred_gru_reg"] = df_signals["pred_gru_reg"].fillna(df_signals["pred_lgbm_reg"])
        df_signals["pred_gru_prob"] = df_signals["pred_gru_prob"].fillna(df_signals["pred_lgbm_prob"])
        df_signals["rank_gru"] = df_signals.groupby("Date")["pred_gru_reg"].rank(pct=True) - 0.5
        print("  [+] PyTorch Nedensel GRU Modeli başarıyla yüklendi.")
    else:
        print("  [!] GRU modeli bulunamadı! Yalnızca LightGBM kullanılacak.")
        df_signals["pred_gru_reg"] = df_signals["pred_lgbm_reg"]
        df_signals["pred_gru_prob"] = df_signals["pred_lgbm_prob"]
        df_signals["rank_gru"] = df_signals["rank_lgbm"]

    # 4. Baş Başa Model Benchmark Karnesi
    benchmark_results = run_head_to_head_benchmark(df_signals, top_k=10, buffer_k=20, smooth_window=3)

    # 5. Conviction Eşiği Duyarlılık Taraması (Sensitivity Sweep)
    print("\n" + "=" * 85)
    print("NEDENSEL GRU: CONVICTION EŞİK DUYARLILIK TARAMASI (%99 Nakit Kapanı Analizi)")
    print("=" * 85)
    rf_daily = (1.35 ** (1.0 / 252.0)) - 1.0
    sweep_df = run_conviction_sensitivity_sweep(
        df_signals,
        signal_col="rank_gru",
        conviction_col="pred_gru_reg",
        thresholds=[0.0, -0.10, -0.25, -0.50, -1.0, None],
        cash_daily_yield=rf_daily,
        top_k=10,
        buffer_k=20,
        weighting="inverse_vol",
        use_stop_loss=True
    )
    print(sweep_df.to_string(index=False))
    print("=" * 85)

    # 6. Şampiyon Model Seçimi (Risk Yönetimli Saf Hisse Alfasını En Yüksek Üreten Model)
    gru_saf = benchmark_results["3. Model A (GRU) + Ters Volatilite + Dinamik ATR Stop"]
    lgbm_saf = benchmark_results["4. Model B (LGBM) + Ters Volatilite + Dinamik ATR Stop"]

    gru_alpha = gru_saf["total_net_return"] - gru_saf["total_bench_return"]
    lgbm_alpha = lgbm_saf["total_net_return"] - lgbm_saf["total_bench_return"]

    if gru_alpha >= lgbm_alpha:
        champion_name = "Nedensel GRU"
        champ_score_col = "rank_gru"
        champ_reg_col = "pred_gru_reg"
        champ_prob_col = "pred_gru_prob"
        champ_bt = benchmark_results["5. Model A (GRU) + Kalibre Conv (-%0.10) + Risk Yön."]
    else:
        champion_name = "LightGBM"
        champ_score_col = "rank_lgbm"
        champ_reg_col = "pred_lgbm_reg"
        champ_prob_col = "pred_lgbm_prob"
        champ_bt = benchmark_results["6. Model B (LGBM) + Kalibre Conv (-%0.10) + Risk Yön."]

    print(f"\n[★] RESMİ ŞAMPİYON MODEL: {champion_name}")
    print(f"     -> Saf Hisse Net Alfas: GRU: %{gru_alpha:+.2f} | LightGBM: %{lgbm_alpha:+.2f}")
    print(f"     -> Kalibre Portföy Getirisi (-%0.10 Eşik, %35 PPF): %{champ_bt['total_net_return']:.2f}")
    print(f"     -> Kalibre Portföy Net Alfas: %{champ_bt['total_net_return'] - champ_bt['total_bench_return']:+.2f}")
    print(f"     -> Kalibre Sharpe: {champ_bt['sharpe_ratio']:.2f} | Max DD: %{champ_bt['max_drawdown']:.2f} | Nakit Gün Oranı: %{champ_bt.get('cash_days_pct', 0.0):.1f}")

    # 7. Canlı Karar Destek Sinyal Paketi (YALNIZCA Şampiyon Model ile Üretilir)
    champ_summary = {
        "champion_model": champion_name,
        "trading_days": int(champ_bt["n_trading_days"]),
        "portfolio_cagr_pct": round(float(champ_bt["cagr_portfolio"]), 2),
        "benchmark_cagr_pct": round(float(champ_bt["cagr_benchmark"]), 2),
        "net_alpha_cagr_pct": round(float(champ_bt["cagr_portfolio"] - champ_bt["cagr_benchmark"]), 2),
        "sharpe_ratio": round(float(champ_bt["sharpe_ratio"]), 2),
        "max_drawdown_pct": round(float(champ_bt["max_drawdown"]), 2),
        "win_rate_vs_benchmark_pct": round(float(champ_bt["win_rate_vs_benchmark"]), 2),
        "cash_days_pct": round(float(champ_bt.get("cash_days_pct", 0.0)), 2)
    }

    signal_pack = generate_champion_actionable_signals(
        df_signals,
        champion_name=champion_name,
        score_col=champ_score_col,
        reg_col=champ_reg_col,
        prob_col=champ_prob_col,
        conviction_threshold=-0.10,
        backtest_metrics=champ_summary
    )

    # Model karşılaştırma tablosunu JSON'a ekle
    comparison_summary = {}
    for name, res in benchmark_results.items():
        comparison_summary[name] = {
            "net_return_pct": round(float(res["total_net_return"]), 2),
            "benchmark_return_pct": round(float(res["total_bench_return"]), 2),
            "net_alpha_pct": round(float(res["total_net_return"] - res["total_bench_return"]), 2),
            "sharpe_ratio": round(float(res["sharpe_ratio"]), 2),
            "max_drawdown_pct": round(float(res["max_drawdown"]), 2),
            "win_rate_vs_benchmark_pct": round(float(res["win_rate_vs_benchmark"]), 2),
            "cash_days_pct": round(float(res.get("cash_days_pct", 0.0)), 2)
        }
    signal_pack["head_to_head_benchmark"] = comparison_summary
    signal_pack["conviction_sensitivity_sweep"] = sweep_df.to_dict(orient="records")

    print("\n" + "=" * 75)
    print(f"CANLI YATIRIMCI KARAR DESTEK RAPORU ({signal_pack['report_date']})")
    print(f"Aktif Model: {signal_pack['champion_model']} (Kalibre Eşik: {signal_pack['calibrated_threshold']})")
    print(f"Genel Portföy Eylemi: {signal_pack['portfolio_action']}")
    print("=" * 75)
    print("EN GÜÇLÜ YÜKSELİŞ BEKLENTİSİ (TOP 5 LONG):")
    for s in signal_pack["top_longs"]:
        w_str = f"Ağırlık: {s.get('recommended_weight', '%20.0')}"
        sl_str = f"Stop: {s.get('dynamic_stop_loss', '-%5.0')}"
        liq_str = f"Likidite: {s.get('liquidity_tier', 'BIST 100')}"
        print(f"  [+] {s['ticker']:<8} | Skor: {s['model_score']:+.3f} | {w_str:<13} | {sl_str:<12} | {liq_str:<26} | Beklenen Alfa: {s['expected_excess_return']:<6} | Güven: {s['confidence']} ({s['action']})")
        if s.get("latest_news") and "özel durum" not in s["latest_news"]:
            print(f"      Haber: \"{s['latest_news'][:65]}...\" (Duygu: {s['news_sentiment']:+.2f})")

    if signal_pack.get("gatekeeper_vetoed"):
        print("\nGATEKEEPER VETO FİLTRESİNE TAKILANLAR (Yüksek Skorlu Ama Riskli):")
        for v in signal_pack["gatekeeper_vetoed"]:
            print(f"  [X] {v['ticker']:<8} | Model Skoru: {v['model_score']:+.3f} | Neden: {v['veto_reason']}")
            print(f"      Haber: \"{v['headline'][:65]}...\" (Duygu: {v['news_sentiment']:+.2f})")

    print("\nEN GÜÇLÜ DÜŞÜŞ / RİSK BEKLENTİSİ (BOTTOM 5 KAÇIN):")
    for s in signal_pack["bottom_shorts"]:
        print(f"  [-] {s['ticker']:<8} | Skor: {s['model_score']:+.3f} | Beklenen Alfa: {s['expected_excess_return']:<6} | Güven: {s['confidence']} ({s['action']})")
        if s.get("latest_news") and "özel durum" not in s["latest_news"]:
            print(f"      Haber: \"{s['latest_news'][:65]}...\" (Duygu: {s['news_sentiment']:+.2f})")
    print("=" * 75)

    os.makedirs("output", exist_ok=True)
    with open("output/daily_signals.json", "w", encoding="utf-8") as f:
        json.dump(signal_pack, f, ensure_ascii=False, indent=2)
    print("JSON sinyal paketi 'output/daily_signals.json' dosyasına kaydedildi (n8n API hazır).")

    return {
        "champion_model": champion_name,
        "champion_alpha": round(float(champ_bt["total_net_return"] - champ_bt["total_bench_return"]), 2),
        "champion_sharpe": round(float(champ_bt["sharpe_ratio"]), 2),
        "signal_pack": signal_pack,
        "benchmark_results": benchmark_results,
        "df_signals": df_signals
    }


if __name__ == "__main__":
    run_independent_model_benchmark()
