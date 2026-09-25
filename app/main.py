import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.api.v1.api import api_router
from app.services.champion_service import ChampionService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload and verify champion metadata
    champion = ChampionService.get_metadata()
    print(f"[*] [FASTAPI STARTUP] Aktif Şampiyon Model: {champion.get('champion_model')}")
    print(f"[*] [FASTAPI STARTUP] Durum: {champion.get('status')}")
    yield
    # Shutdown
    print("[*] [FASTAPI SHUTDOWN] Servis kapatılıyor.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="BIST 100 Çok Boyutlu Tahminleme, KAP Sentiment Radarı ve Otonom Portföy Karar Destek Terminali",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & Templates
BASE_PATH = Path(__file__).resolve().parent
static_dir = BASE_PATH / "static"
templates_dir = BASE_PATH / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", response_class=HTMLResponse, summary="Quant Terminal Kokpiti")
async def get_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "project_name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "api_key": settings.API_KEY
        }
    )
