from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ChampionStatusResponse(BaseModel):
    champion_model: str
    active_weights_path: str
    registered_at: str
    last_promoted_at: str
    net_alpha_pct: float
    sharpe_ratio: float
    status: str
    evaluation_history: List[Dict[str, Any]]
