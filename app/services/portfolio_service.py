import json
import os
import shutil
from typing import Dict, Any
from app.core.config import settings

class PortfolioService:
    @staticmethod
    def get_ledger_data() -> Dict[str, Any]:
        path = str(settings.LEDGER_PATH)
        if not os.path.exists(path):
            return {
                "initial_capital_try": 1000000.0,
                "current_cash_try": 1000000.0,
                "current_portfolio_value_try": 1000000.0,
                "total_pnl_try": 0.0,
                "total_return_pct": 0.0,
                "benchmark_cumulative_pct": 0.0,
                "net_alpha_pct": 0.0,
                "total_trades_count": 0,
                "stopped_out_trades_count": 0,
                "rolling_drawdown_pct": 0.0,
                "circuit_breaker_active": False,
                "daily_history": []
            }
        
        with open(path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
            
        history = ledger.get("daily_history", [])
        init_cap = float(ledger.get("initial_capital_try", 1000000.0))
        port_val = float(ledger.get("current_portfolio_value_try", init_cap))
        
        # Calculate rolling drawdown from peak of last 5 days
        recent_equities = [float(h.get("end_equity_try", init_cap)) for h in history[-5:]] if history else [port_val]
        rolling_peak = max(recent_equities) if recent_equities else port_val
        rolling_dd_pct = ((port_val - rolling_peak) / rolling_peak) * 100.0 if rolling_peak > 0 else 0.0
        
        # Circuit breaker trigger if rolling dd <= -5.0%
        circuit_breaker_active = rolling_dd_pct <= -5.0
        
        tot_return = float(ledger.get("total_return_pct", 0.0))
        bench_return = float(ledger.get("benchmark_cumulative_pct", 0.0))
        net_alpha = tot_return - bench_return
        
        return {
            "initial_capital_try": init_cap,
            "current_cash_try": float(ledger.get("current_cash_try", port_val)),
            "current_portfolio_value_try": port_val,
            "total_pnl_try": float(ledger.get("total_pnl_try", 0.0)),
            "total_return_pct": tot_return,
            "benchmark_cumulative_pct": bench_return,
            "net_alpha_pct": round(net_alpha, 2),
            "total_trades_count": int(ledger.get("total_trades_count", len(history))),
            "stopped_out_trades_count": int(ledger.get("stopped_out_trades_count", 0)),
            "rolling_drawdown_pct": round(rolling_dd_pct, 2),
            "circuit_breaker_active": circuit_breaker_active,
            "daily_history": history
        }

    @staticmethod
    def reset_ledger(target_capital: float = 1000000.0) -> Dict[str, Any]:
        """Portföy özsermayesini ve defteri belirtilen tutara (varsayılan: 1.000.000 TL) sıfırlar."""
        path = str(settings.LEDGER_PATH)
        
        # Yedek oluştur
        if os.path.exists(path):
            backup_path = path.replace(".json", "_backup.json")
            try:
                shutil.copyfile(path, backup_path)
            except Exception:
                pass
                
        new_ledger = {
            "initial_capital_try": float(target_capital),
            "current_cash_try": float(target_capital),
            "current_portfolio_value_try": float(target_capital),
            "total_pnl_try": 0.0,
            "total_return_pct": 0.0,
            "benchmark_cumulative_pct": 0.0,
            "total_trades_count": 0,
            "stopped_out_trades_count": 0,
            "cooldown_remaining": 0,
            "daily_history": [],
            "executed_trades": []
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_ledger, f, ensure_ascii=False, indent=2)
            
        return PortfolioService.get_ledger_data()
