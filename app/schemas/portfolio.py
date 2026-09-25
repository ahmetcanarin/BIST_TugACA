from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class DailyHistoryEntry(BaseModel):
    execution_date: str
    action: str
    start_equity_try: float
    end_equity_try: float
    net_daily_pnl_try: float
    daily_return_pct: float
    benchmark_return_pct: float
    alpha_daily_pct: float
    cumulative_return_pct: float
    benchmark_cumulative_pct: float
    stopped_out_count: int
    trades_count: int
    trades: Optional[List[Dict[str, Any]]] = None

class PortfolioLedgerResponse(BaseModel):
    initial_capital_try: float
    current_cash_try: float
    current_portfolio_value_try: float
    total_pnl_try: float
    total_return_pct: float
    benchmark_cumulative_pct: float
    net_alpha_pct: float
    total_trades_count: int
    stopped_out_trades_count: int
    rolling_drawdown_pct: float
    circuit_breaker_active: bool
    daily_history: List[DailyHistoryEntry]
