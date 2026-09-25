import os
import json
import threading
from datetime import datetime
from typing import Dict, Any
from app.core.config import settings
from live_daily_runner import run_close_auction_execution, run_morning_paper_execution

class DuplicateExecutionError(Exception):
    """Aynı işlem günü için mükerrer icra denendiğinde fırlatılır."""
    pass

class ExecutionService:
    _lock = threading.Lock()

    @staticmethod
    def execute_daily(
        mode: str = "close_auction",
        force_cash: bool = False,
        rebalance_step: int = 5,
        force_execution: bool = False
    ) -> Dict[str, Any]:
        signals_p = str(settings.DAILY_SIGNALS_PATH)
        ledger_p = str(settings.LEDGER_PATH)
        today_str = datetime.now().strftime("%Y-%m-%d")

        with ExecutionService._lock:
            # 1. Idempotency Check (Aynı gün mükerrer icrayı engelleme)
            if not force_execution and os.path.exists(ledger_p):
                try:
                    with open(ledger_p, "r", encoding="utf-8") as f:
                        ledger_data = json.load(f)
                    
                    history = ledger_data.get("daily_history", [])
                    if history:
                        last_entry = history[-1]
                        last_exec_date = last_entry.get("execution_date")
                        
                        if last_exec_date == today_str:
                            raise DuplicateExecutionError(
                                f"Mükerrer İcra Koruması Devrede: {today_str} tarihi için seans icrası zaten yapılmıştır. "
                                f"Tekrar çalıştırmak için 'force_execution: true' parametresini kullanınız."
                            )
                except (json.JSONDecodeError, KeyError) as e:
                    # Defter bozuk veya boşsa logla, devam et
                    pass

            # 2. İcra Koşumu
            if mode == "close_auction":
                res = run_close_auction_execution(
                    signals_path=signals_p,
                    ledger_path=ledger_p,
                    initial_capital=1000000.0,
                    force_cash=force_cash,
                    rebalance_step=rebalance_step
                )
            else:
                res = run_morning_paper_execution(
                    signals_path=signals_p,
                    ledger_path=ledger_p,
                    initial_capital=1000000.0,
                    force_cash=force_cash
                )
            return res
