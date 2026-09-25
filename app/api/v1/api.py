from fastapi import APIRouter
from app.api.v1.endpoints import (
    health,
    market,
    signals,
    portfolio,
    champion,
    sentiment,
    execution
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(market.router, prefix="/market", tags=["Market & Regime"])
api_router.include_router(signals.router, prefix="/signals", tags=["Signals & Predictions"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio & Ledger"])
api_router.include_router(champion.router, prefix="/champion", tags=["Champion Model"])
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment & KAP"])
api_router.include_router(execution.router, prefix="/execution", tags=["Trading Execution"])
