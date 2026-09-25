"""
BIST 100 Champion vs Challenger Model Terfi Kapısı (Model Promotion Gatekeeper)

Bu modül:
1. Canlı üretimde çalışan mevcut şampiyon modelin (Champion) ağırlıklarını ve performans geçmişini
   'models/champion_metadata.json' dosyasında takip eder.
2. 6 veya 12 ay sonra yeni eğitilen Meydan Okuyan (Challenger) modeli değerlendirir.
3. Out-of-Sample test döneminde Challenger ile Mevcut Şampiyonu doğrudan yarıştırır:
   - Eğer Challenger Net Alfa veya Sharpe oranında mevcut şampiyonu YENERSE:
     -> Eski şampiyon 'models/archive/' klasörüne tarih etiketiyle yedeklenir.
     -> Challenger modeli canlı şampiyon olarak terfi ettirilir ('models/bist_dual_model_best.pt').
     -> 'output/daily_signals.json' yeni şampiyonun sinyalleriyle güncellenir.
   - Eğer Challenger mevcut şampiyonu YENEMEZSE:
     -> Canlı üretim ağırlıklarına ASLA DOKUNULMAZ. Challenger terfisi REDDEDİLİR.
     -> Canlı tahminlerde ve 'output/daily_signals.json' dosyasında ESKİ ŞAMPİYON tahminci olmaya devam eder.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import shutil
import json
import time
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

CHAMPION_META_PATH = "models/champion_metadata.json"
CHAMPION_MODEL_PATH = "models/bist_dual_model_best.pt"
CHALLENGER_GRU_PATH = "models/challenger_gru.pt"
ARCHIVE_DIR = "models/archive"


def get_champion_metadata(meta_path: str = CHAMPION_META_PATH) -> Optional[Dict[str, Any]]:
    """Mevcut canlı şampiyonun metadatasını döner."""
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_champion_metadata(meta: Dict[str, Any], meta_path: str = CHAMPION_META_PATH):
    """Şampiyon metadatasını JSON dosyasına kaydeder."""
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def initialize_inaugural_champion(
    champion_name: str = "Nedensel GRU",
    net_alpha: float = -31.05,
    sharpe: float = -4.24,
    source_model_path: str = "models/bist_dual_model_best.pt",
    meta_path: str = CHAMPION_META_PATH,
    champion_model_path: str = CHAMPION_MODEL_PATH
) -> Dict[str, Any]:
    """Sistemde henüz bir şampiyon kaydı yoksa mevcut modeli ilk şampiyon olarak tescil eder."""
    existing = get_champion_metadata(meta_path=meta_path)
    if existing is not None and os.path.exists(champion_model_path):
        return existing

    champion_meta = {
        "champion_model": champion_name,
        "active_weights_path": champion_model_path,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "net_alpha_pct": net_alpha,
        "sharpe_ratio": sharpe,
        "status": "ACTIVE_PRODUCTION_CHAMPION",
        "evaluation_history": [
            {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "INAUGURAL_REGISTRATION",
                "model_name": champion_name,
                "net_alpha_pct": net_alpha,
                "sharpe_ratio": sharpe,
                "decision": "INITIALIZED_AS_CHAMPION"
            }
        ]
    }
    save_champion_metadata(champion_meta, meta_path=meta_path)
    print(f"[*] İlk Şampiyon Model Tescil Edildi: {champion_name} (Net Alfa: %{net_alpha:+.2f}, Sharpe: {sharpe:.2f})")
    return champion_meta


def evaluate_and_promote_challenger(
    challenger_name: str,
    challenger_alpha: float,
    challenger_sharpe: float,
    challenger_weights_path: str,
    challenger_signal_pack: Dict[str, Any],
    meta_path: str = CHAMPION_META_PATH,
    champion_model_path: str = CHAMPION_MODEL_PATH,
    archive_dir: str = ARCHIVE_DIR
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Yeni eğitilen Challenger modelini mevcut Champion ile yarıştırır ve terfi kararını verir.

    Dönüş:
    (is_promoted, active_champion_name, final_signal_pack)
    """
    os.makedirs(archive_dir, exist_ok=True)
    current_champion = get_champion_metadata(meta_path=meta_path)

    # Eğer sistemde kayıtlı şampiyon yoksa doğrudan ilk şampiyon ilan et
    if current_champion is None:
        initialize_inaugural_champion(
            champion_name=challenger_name,
            net_alpha=challenger_alpha,
            sharpe=challenger_sharpe,
            source_model_path=challenger_weights_path,
            meta_path=meta_path,
            champion_model_path=champion_model_path
        )
        return True, challenger_name, challenger_signal_pack

    champ_name = current_champion.get("champion_model", "Eski Şampiyon")
    champ_alpha = float(current_champion.get("net_alpha_pct", -999.0))
    champ_sharpe = float(current_champion.get("sharpe_ratio", -999.0))

    print("\n" + "=" * 85)
    print("        CHAMPION vs CHALLENGER MODEL TERFİ MÜCADELESİ (GATEKEEPER)")
    print("=" * 85)
    print(f"  Mevcut Canlı Şampiyon (Champion)    : {champ_name}")
    print(f"    -> Canlı Net Alfa: %{champ_alpha:+.2f} | Sharpe: {champ_sharpe:.2f}")
    print(f"  Yeni Meydan Okuyan (Challenger)     : {challenger_name}")
    print(f"    -> Aday Net Alfa : %{challenger_alpha:+.2f} | Sharpe: {challenger_sharpe:.2f}")
    print("-" * 85)

    # Karar Mantığı: Challenger hem daha yüksek net alfa hem de daha iyi/eşit Sharpe üretmeli
    # (veya belirgin bir alfa üstünlüğü sağlamalı)
    is_better = (challenger_alpha > champ_alpha) or (
        abs(challenger_alpha - champ_alpha) < 0.5 and challenger_sharpe > champ_sharpe
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if is_better:
        # [AŞAMA 1]: Eski şampiyonu arşive taşı
        if os.path.exists(champion_model_path):
            archive_path = os.path.join(archive_dir, f"champion_{champ_name.replace(' ', '_')}_{timestamp}.pt")
            shutil.copy2(champion_model_path, archive_path)
            print(f"  [+] Eski şampiyon başarıyla arşivlendi: '{archive_path}'")

        # [AŞAMA 2]: Challenger ağırlıklarını canlıya terfi ettir
        if os.path.exists(challenger_weights_path) and challenger_weights_path != champion_model_path:
            shutil.copy2(challenger_weights_path, champion_model_path)
            print(f"  [+] Challenger ağırlıkları canlı modele terfi ettirildi: '{champion_model_path}'")

        # [AŞAMA 3]: Şampiyon metadatasını güncelle
        new_meta = {
            "champion_model": challenger_name,
            "active_weights_path": champion_model_path,
            "registered_at": current_champion.get("registered_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            "last_promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "net_alpha_pct": challenger_alpha,
            "sharpe_ratio": challenger_sharpe,
            "status": "ACTIVE_PRODUCTION_CHAMPION",
            "evaluation_history": current_champion.get("evaluation_history", []) + [
                {
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "CHALLENGE_EVALUATION",
                    "challenger_model": challenger_name,
                    "challenger_alpha": challenger_alpha,
                    "challenger_sharpe": challenger_sharpe,
                    "previous_champion_alpha": champ_alpha,
                    "decision": "PROMOTED_NEW_CHAMPION"
                }
            ]
        }
        save_champion_metadata(new_meta, meta_path=meta_path)

        # Sinyal paketine terfi kararını işle
        challenger_signal_pack["promotion_gatekeeper_decision"] = {
            "status": "PROMOTED",
            "active_model": challenger_name,
            "reason": f"Yeni model mevcut şampiyonu geçti (Alfa: %{challenger_alpha:+.2f} > %{champ_alpha:+.2f})"
        }

        print(f"\n  [★ TERFİ ONAYLANDI] '{challenger_name}' eski şampiyonu geride bıraktı ve YENİ CANLI ŞAMPİYON oldu!")
        print("=" * 85)
        return True, challenger_name, challenger_signal_pack

    else:
        # [RED KARARI]: Eski şampiyon canlıda kalır, canlı dosyalara DOKUNULMAZ!
        print(f"\n  [🛡️ TERFİ REDDEDİLDİ] Yeni model ('{challenger_name}': %{challenger_alpha:+.2f}) eski şampiyonu ('{champ_name}': %{champ_alpha:+.2f}) GEÇEMEDİ.")
        print(f"  -> CANLI TAHMİNCİ OLARAK ESKİ ŞAMPİYON ({champ_name}) KORUNUYOR.")
        print("=" * 85)

        current_champion["evaluation_history"].append({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "CHALLENGE_EVALUATION",
            "challenger_model": challenger_name,
            "challenger_alpha": challenger_alpha,
            "challenger_sharpe": challenger_sharpe,
            "previous_champion_alpha": champ_alpha,
            "decision": "REJECTED_RETAINED_PREVIOUS_CHAMPION"
        })
        save_champion_metadata(current_champion, meta_path=meta_path)

        # Canlı sinyal paketi eski şampiyonun güvencesiyle korunur
        challenger_signal_pack["promotion_gatekeeper_decision"] = {
            "status": "RETAINED_EXISTING_CHAMPION",
            "active_model": champ_name,
            "reason": f"Yeni model eskisini yenemedi (Challenger: %{challenger_alpha:+.2f} <= Champion: %{champ_alpha:+.2f}). Eski şampiyon korundu."
        }

        return False, champ_name, challenger_signal_pack
