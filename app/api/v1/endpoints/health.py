from fastapi import APIRouter
from app.core.config import settings
from app.services.champion_service import ChampionService

router = APIRouter()

@router.get("/health", summary="Sistem Sağlık ve Model Durumu")
def check_health():
    champion = ChampionService.get_metadata()
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "active_champion": champion.get("champion_model", "Unknown"),
        "model_status": champion.get("status", "ACTIVE")
    }
