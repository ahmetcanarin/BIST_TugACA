from fastapi import APIRouter
from app.schemas.market import MacroRegimeResponse
from app.services.market_service import MarketService

router = APIRouter()

@router.get("/regime", response_model=MacroRegimeResponse, summary="Tier-1 Makro Rejim ve Risk Kapısı")
def get_market_regime():
    return MarketService.get_macro_regime()
