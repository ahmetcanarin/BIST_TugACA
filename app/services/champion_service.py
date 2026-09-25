import json
import os
from typing import Dict, Any, Optional
from app.core.config import settings

class ChampionService:
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        path = str(settings.CHAMPION_META_PATH)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                return {
                    "champion_model": "GRU_Ranker (Hata)",
                    "active_weights_path": str(settings.CHAMPION_MODEL_PATH),
                    "registered_at": "N/A",
                    "last_promoted_at": "N/A",
                    "net_alpha_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "status": "ERROR_READING_METADATA",
                    "evaluation_history": []
                }
        return {
            "champion_model": "GRU_Ranker",
            "active_weights_path": str(settings.CHAMPION_MODEL_PATH),
            "registered_at": "N/A",
            "last_promoted_at": "N/A",
            "net_alpha_pct": -39.49,
            "sharpe_ratio": -1.77,
            "status": "DEFAULT_FALLBACK",
            "evaluation_history": []
        }
