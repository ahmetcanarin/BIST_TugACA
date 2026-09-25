from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool
from app.schemas.execution import ExecutionRequest, ExecutionResponse
from app.services.execution_service import ExecutionService, DuplicateExecutionError
from app.core.security import verify_api_key

router = APIRouter()

@router.post("/run-daily", response_model=ExecutionResponse, summary="17:50 Kapanış / 09:55 Açılış Seansı Sanal İcrasını Tetikle")
async def trigger_daily_execution(
    request: ExecutionRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Run in threadpool so it doesn't block the async event loop
        result = await run_in_threadpool(
            ExecutionService.execute_daily,
            mode=request.mode,
            force_cash=request.force_cash,
            rebalance_step=request.rebalance_step,
            force_execution=request.force_execution
        )
        return ExecutionResponse(
            status="SUCCESS",
            message=f"{request.mode.upper()} icra seansı başarıyla tamamlandı.",
            execution_result=result
        )
    except DuplicateExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"İcra sırasında hata oluştu: {str(e)}"
        )
