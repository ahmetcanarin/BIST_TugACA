"""
BIST 100 Kurumsal Quant Reformu: 10 Yıllık Tam Model Eğitimi ve Champion/Challenger Orkestratörü

Bu script tek bir komutla:
1. 10 yıllık (2016-günümüz) BIST 100 ve XU100 verisini çeker ve 26 saf durağan öznitelik üretir.
2. LightGBM Tabular Regressor ve Classifier modellerini eğitir.
3. NVIDIA GPU (RTX 3050 Ti) üzerinde PyTorch Nedensel GRU modelini 'Meydan Okuyan' (Challenger) olarak eğitir ('models/challenger_gru.pt').
4. Test kümesinde (2024+ Out-of-Sample) yeni modellerin head-to-head yarışını yürütür.
5. [CHAMPION / CHALLENGER GATEKEEPER]:
   - Yeni eğitilen Challenger modelini canlıdaki Mevcut Şampiyon ile karşılaştırır.
   - EĞER Challenger daha yüksek Alfa/Sharpe üretirse: Canlı şampiyon olarak terfi ettirilir ('models/bist_dual_model_best.pt'), eski model arşivlenir.
   - EĞER Challenger eski şampiyonu YENEMEZSE: Yeni model reddedilir; canlı tahminlerde ve portföy üretiminde ESKİ ŞAMPİYON korunur!
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import json
import torch

from preprocessing import load_and_preprocess_pipeline
from benchmark_tree import train_lightgbm_pipeline
from train import train_pipeline
from ensemble import run_independent_model_benchmark
from champion_gatekeeper import (
    initialize_inaugural_champion,
    get_champion_metadata,
    evaluate_and_promote_challenger
)


def main():
    start_time = time.time()
    print("=" * 85)
    print("      BIST 100 KURUMSAL QUANT REFORMU: CHAMPION / CHALLENGER EĞİTİM HATTI")
    print("=" * 85)
    print(f"Başlangıç Zamanı : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch Cihazı   : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"Ekran Kartı      : {torch.cuda.get_device_name(0)}")
    print("=" * 85)

    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # 0. Aşama: Mevcut Şampiyon Kontrolü (Yoksa mevcut canlı modeli tescil et)
    initialize_inaugural_champion(
        champion_name="Nedensel GRU",
        net_alpha=-31.05,
        sharpe=-4.24,
        source_model_path="models/bist_dual_model_best.pt"
    )

    # 1. Aşama: 10 Yıllık Veri Çekimi ve Tensör Üretimi (Tek Seferlik Yükleme)
    print("\n" + "#" * 85)
    print("AŞAMA 1/5: 10 Yıllık BIST100, Endeks ve Makro Veri Boru Hattı Yükleniyor...")
    print("#" * 85)
    seq_len = 20
    if os.path.exists("models/optuna_best_gru_params.json"):
        try:
            with open("models/optuna_best_gru_params.json", "r", encoding="utf-8") as f:
                opt = json.load(f)
                seq_len = opt.get("best_params", {}).get("seq_len", 20)
        except Exception:
            pass

    data_dict = load_and_preprocess_pipeline(
        period="10y",
        seq_len=seq_len,
        reg_target="Target_Excess_Return",
        cls_target="Target_Direction_Alpha"
    )
    print(f"[+] Veri boru hattı hazır: {len(data_dict['feature_cols'])} saf özellik, {len(data_dict['X_train']):,} eğitim dizisi.")

    # 2. Aşama: LightGBM Tabular Modellerinin Eğitimi (200+ Ağaç)
    print("\n" + "#" * 85)
    print("AŞAMA 2/5: LightGBM Tabular Modelleri Eğitiliyor (Huber Loss, 200+ Ağaç)...")
    print("#" * 85)
    lgbm_results = train_lightgbm_pipeline(data_dict)
    print("[+] LightGBM Regressor ve Classifier modelleri eğitildi ve kaydedildi.")

    # 3. Aşama: PyTorch Nedensel GRU Modelinin GPU Üzerinde Eğitimi (Challenger Olarak)
    print("\n" + "#" * 85)
    print("AŞAMA 3/5: PyTorch Nedensel GRU (Challenger) GPU Üzerinde Eğitiliyor...")
    print("#" * 85)
    challenger_gru_path = "models/challenger_gru.pt"
    gru_metrics = train_pipeline(
        period="10y",
        seq_len=seq_len,
        epochs=60,
        patience=15,
        model_save_path=challenger_gru_path,
        data_dict=data_dict
    )
    print(f"[+] Yeni Meydan Okuyan (Challenger) GRU modeli eğitildi: '{challenger_gru_path}'")

    # 4. Aşama: Yeni Modeller Arasında Ön Eleme (Challenger GRU vs LightGBM)
    print("\n" + "#" * 85)
    print("AŞAMA 4/5: Yeni Challenger GRU vs LightGBM Ön Eleme Yarışı...")
    print("#" * 85)
    challenger_benchmark = run_independent_model_benchmark(
        data_dict=data_dict,
        gru_model_path=challenger_gru_path
    )

    challenger_champ_name = challenger_benchmark.get("champion_model", "Nedensel GRU")
    challenger_alpha = challenger_benchmark.get("champion_alpha", 0.0)
    challenger_sharpe = challenger_benchmark.get("champion_sharpe", 0.0)
    challenger_signal_pack = challenger_benchmark.get("signal_pack", {})

    challenger_weights = challenger_gru_path if "GRU" in challenger_champ_name else "models/lgbm_reg.txt"

    # 5. Aşama: [CHAMPION vs CHALLENGER GATEKEEPER] Nihai Terfi Kararı
    print("\n" + "#" * 85)
    print("AŞAMA 5/5: CHAMPION vs CHALLENGER Terfi ve Onay Kararı...")
    print("#" * 85)
    is_promoted, active_model_name, final_signals = evaluate_and_promote_challenger(
        challenger_name=f"Challenger {challenger_champ_name}",
        challenger_alpha=challenger_alpha,
        challenger_sharpe=challenger_sharpe,
        challenger_weights_path=challenger_weights,
        challenger_signal_pack=challenger_signal_pack
    )

    if not is_promoted:
        # Eğer Challenger yenildiyse, günlük canlı sinyaller ESKİ ŞAMPİYON modeliyle yeniden oluşturulur
        print("\n[!] Challenger yenildi. Canlı sinyaller mevcut şampiyon model ile üretiliyor...")
        retained_res = run_independent_model_benchmark(
            data_dict=data_dict,
            gru_model_path="models/bist_dual_model_best.pt"
        )
        final_signals = retained_res.get("signal_pack", {})
        final_signals["promotion_gatekeeper_decision"] = {
            "status": "RETAINED_EXISTING_CHAMPION",
            "active_model": active_model_name,
            "reason": f"Challenger eskisini yenemedi. Eski şampiyon ({active_model_name}) canlıda kaldı."
        }

    # Nihai onaylı sinyalleri kaydet
    with open("output/daily_signals.json", "w", encoding="utf-8") as f:
        json.dump(final_signals, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"CHAMPION / CHALLENGER SÜRECİ TAMAMLANDI! (Toplam Süre: {elapsed/60:.1f} dakika)")
    print("=" * 85)
    print(f"Resmi Aktif Tahminci  : {active_model_name}")
    print(f"Terfi Kararı          : {'★ YENİ MODEL TERFİ ETTİ' if is_promoted else '🛡️ ESKİ ŞAMPİYON KORUNDU'}")
    print(f"Canlı Yatırım Sinyali : output/daily_signals.json")
    print(f"Şampiyon Kayıt Kütüğü : models/champion_metadata.json")
    print("=" * 85)


if __name__ == "__main__":
    main()
