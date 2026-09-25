from fastapi import APIRouter, Depends
from app.schemas.signals import DailySignalsResponse
from app.services.signal_service import SignalService
from app.core.security import verify_api_key

router = APIRouter()

@router.get("/daily", response_model=DailySignalsResponse, summary="Günlük Şampiyon Model Sinyalleri (Top Longs / Shorts)")
def get_daily_signals():
    return SignalService.get_daily_signals()

@router.post("/refresh", response_model=DailySignalsResponse, summary="Sinyalleri Yeniden Hesapla ve KAP Gatekeeper Filtresini Güncelle")
def refresh_daily_signals(api_key: str = Depends(verify_api_key)):
    return SignalService.recalculate_or_refresh_signals()
