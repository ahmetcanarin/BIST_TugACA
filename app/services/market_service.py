import json
import os
from typing import Dict, Any
from datetime import datetime
from app.core.config import settings

class MarketService:
    @staticmethod
    def get_macro_regime() -> Dict[str, Any]:
        path = str(settings.DAILY_SIGNALS_PATH)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            macro_bull = data.get("macro_regime_bull", False)
            is_cash = data.get("is_cash_recommended", True)
            action = data.get("portfolio_action", "N/A")
            rep_date = data.get("report_date", datetime.now().strftime("%Y-%m-%d"))
            
            return {
                "macro_regime_bull": macro_bull,
                "portfolio_action": action,
                "is_cash_recommended": is_cash,
                "description": "Boğa Rejimi (XU100 > SMA50 & Düşük Dolar Volatilitesi)" if macro_bull else "Ayı / Defansif Rejim (XU100 < SMA50 veya Yüksek Kur Şoku)",
                "xu100_above_sma50": macro_bull,
                "usdtry_volatility_low": macro_bull,
                "report_date": rep_date
            }
            
        return {
            "macro_regime_bull": False,
            "portfolio_action": "DEFANSİF MOD (Veri Yok)",
            "is_cash_recommended": True,
            "description": "Bilinmeyen Rejim (Varsayılan Koruma: %100 Nakit)",
            "xu100_above_sma50": False,
            "usdtry_volatility_low": False,
            "report_date": datetime.now().strftime("%Y-%m-%d")
        }
