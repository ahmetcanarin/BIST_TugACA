import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "DLAI_BIST Quant Trading Terminal"
    VERSION: str = "2.4.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "bist_quant_secret_2026")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "true").lower() in ("true", "1", "yes")
    
    # Path settings
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    DATA_DIR: Path = BASE_DIR / "data"
    
    CHAMPION_META_PATH: Path = MODELS_DIR / "champion_metadata.json"
    CHAMPION_MODEL_PATH: Path = MODELS_DIR / "bist_dual_model_best.pt"
    DAILY_SIGNALS_PATH: Path = OUTPUT_DIR / "daily_signals.json"
    LEDGER_PATH: Path = OUTPUT_DIR / "paper_trading_ledger.json"
    KAP_CACHE_PATH: Path = DATA_DIR / "kap_sentiment_cache.json"

settings = Settings()
