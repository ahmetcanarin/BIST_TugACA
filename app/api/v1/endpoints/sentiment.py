from fastapi import APIRouter, Query
from app.schemas.sentiment import SentimentRadarResponse
from app.services.sentiment_service import SentimentService

router = APIRouter()

@router.get("/kap", response_model=SentimentRadarResponse, summary="KAP Haber Radarı, Duygu Skorları ve Veto Listesi")
def get_kap_sentiment(limit: int = Query(default=30, ge=5, le=100)):
    return SentimentService.get_kap_sentiment_radar(limit=limit)
