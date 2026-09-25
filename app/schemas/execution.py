from typing import Optional, Dict, Any
from pydantic import BaseModel

class ExecutionRequest(BaseModel):
    mode: str = "close_auction" # "close_auction" or "morning"
    force_cash: bool = False
    rebalance_step: int = 5
    force_execution: bool = False # Mükerrer işlem kontrolünü bypass etme flag'i

class ExecutionResponse(BaseModel):
    status: str
    message: str
    execution_result: Optional[Dict[str, Any]] = None
