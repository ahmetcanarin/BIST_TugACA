from typing import Optional
from pydantic import BaseModel

class MacroRegimeResponse(BaseModel):
    macro_regime_bull: bool
    portfolio_action: str
    is_cash_recommended: bool
    description: str
    xu100_above_sma50: bool
    usdtry_volatility_low: bool
    report_date: str
