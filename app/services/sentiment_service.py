import json
import os
from typing import Dict, Any, List
from app.core.config import settings
from ensemble import check_gatekeeper_veto

class SentimentService:
    @staticmethod
    def get_kap_sentiment_radar(limit: int = 40) -> Dict[str, Any]:
        path = str(settings.KAP_CACHE_PATH)
        if not os.path.exists(path):
            return {
                "total_cached_news": 0,
                "news_items": [],
                "vetoed_count": 0
            }
            
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
            
        total_count = len(cache)
        news_items = []
        vetoed_count = 0
        
        # Parse items (keys are formatted like TICKER_headline)
        # We sort or take the most recent items
        for key, val in list(cache.items())[-limit:]:
            parts = key.split("_", 1)
            ticker = parts[0] if parts else "BIST"
            title = val.get("title", key)
            score = float(val.get("score", 0.0))
            date_str = val.get("aligned_date", "")
            
            is_vetoed, reason = check_gatekeeper_veto({"headline": title, "news_sentiment": score})
            if is_vetoed:
                vetoed_count += 1
                
            news_items.append({
                "ticker": ticker,
                "title": title,
                "score": score,
                "aligned_date": date_str,
                "is_vetoed": is_vetoed,
                "veto_reason": reason
            })
            
        # Reverse so newest are first
        news_items.reverse()
        
        return {
            "total_cached_news": total_count,
            "news_items": news_items,
            "vetoed_count": vetoed_count
        }
