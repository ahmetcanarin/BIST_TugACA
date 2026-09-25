from typing import List, Optional
from pydantic import BaseModel

class NewsItem(BaseModel):
    ticker: str
    title: str
    score: float
    aligned_date: Optional[str] = None
    is_vetoed: bool
    veto_reason: Optional[str] = None

class SentimentRadarResponse(BaseModel):
    total_cached_news: int
    news_items: List[NewsItem]
    vetoed_count: int
