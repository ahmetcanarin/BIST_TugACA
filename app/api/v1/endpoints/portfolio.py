from fastapi import APIRouter, Depends, Query
from app.schemas.portfolio import PortfolioLedgerResponse
from app.services.portfolio_service import PortfolioService
from app.core.security import verify_api_key

router = APIRouter()

@router.get("/ledger", response_model=PortfolioLedgerResponse, summary="1M TL Kurumsal Sanal Portföy Defteri ve PnL")
def get_portfolio_ledger():
    return PortfolioService.get_ledger_data()

@router.post("/reset", response_model=PortfolioLedgerResponse, summary="Portföy Özsermayesini 1 Milyon TL'ye Sıfırla")
def reset_portfolio_ledger(
    capital: float = Query(default=1000000.0, ge=1000.0),
    api_key: str = Depends(verify_api_key)
):
    return PortfolioService.reset_ledger(target_capital=capital)
