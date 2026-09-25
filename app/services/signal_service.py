import json
import os
from datetime import datetime
from typing import Dict, Any
from app.core.config import settings
from ensemble import check_gatekeeper_veto
from extraction_kap import load_sentiment_cache

class SignalService:
    @staticmethod
    def get_daily_signals() -> Dict[str, Any]:
        path = str(settings.DAILY_SIGNALS_PATH)
        if not os.path.exists(path):
            return {
                "report_date": "N/A",
                "champion_model": "GRU_Ranker",
                "portfolio_action": "SİNYAL OLUŞTURULMADI",
                "is_cash_recommended": True,
                "macro_regime_bull": False,
                "top_longs": [],
                "bottom_shorts": []
            }
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return {
            "report_date": data.get("report_date", "N/A"),
            "champion_model": data.get("champion_model", "GRU_Ranker"),
            "portfolio_action": data.get("portfolio_action", "N/A"),
            "is_cash_recommended": data.get("is_cash_recommended", False),
            "macro_regime_bull": data.get("macro_regime_bull", True),
            "calibrated_threshold": data.get("calibrated_threshold"),
            "top10_avg_expected_alpha": data.get("top10_avg_expected_alpha"),
            "top_longs": data.get("top_longs", []),
            "bottom_shorts": data.get("bottom_shorts", [])
        }

    @staticmethod
    def recalculate_or_refresh_signals() -> Dict[str, Any]:
        """Tarihi bugüne günceller, en son KAP haber ve veto kontrollerini tekrar koşturup dosyayı yeniler."""
        path = str(settings.DAILY_SIGNALS_PATH)
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        data["report_date"] = today_str
        
        # Re-check KAP sentiment and veto for candidates
        cache = load_sentiment_cache()
        for c in data.get("top_longs", []):
            ticker = c.get("ticker", "").replace(".IS", "")
            matches = [v for k, v in cache.items() if k.startswith(f"{ticker}_")]
            if matches:
                last = matches[-1]
                title = last.get("title", "")
                score = float(last.get("score", 0.0))
                c["latest_news"] = title[:80]
                c["news_sentiment"] = score
                is_v, reason = check_gatekeeper_veto({"headline": title, "news_sentiment": score})
                if is_v:
                    c["action"] = f"VETO EDİLDİ ({reason})"
        
        # Save back to disk
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return SignalService.get_daily_signals()
