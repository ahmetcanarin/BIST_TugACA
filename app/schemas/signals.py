from typing import List, Optional
from pydantic import BaseModel

class CandidateSignal(BaseModel):
    ticker: str
    model_score: float
    expected_excess_return: str
    confidence: str
    action: str
    latest_news: Optional[str] = None
    news_sentiment: Optional[float] = 0.0
    recommended_weight: Optional[str] = None
    dynamic_stop_loss: Optional[str] = None
    liquidity_tier: Optional[str] = None

class DailySignalsResponse(BaseModel):
    report_date: str
    champion_model: str
    portfolio_action: str
    is_cash_recommended: bool
    macro_regime_bull: bool
    calibrated_threshold: Optional[str] = None
    top10_avg_expected_alpha: Optional[str] = None
    top_longs: List[CandidateSignal]
    bottom_shorts: List[CandidateSignal]
