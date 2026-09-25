from fastapi import APIRouter
from app.schemas.champion import ChampionStatusResponse
from app.services.champion_service import ChampionService

router = APIRouter()

@router.get("/status", response_model=ChampionStatusResponse, summary="Aktif Şampiyon Model ve Terfi Geçmişi")
def get_champion_status():
    return ChampionService.get_metadata()
