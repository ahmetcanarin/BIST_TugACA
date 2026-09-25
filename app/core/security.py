from typing import Optional
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

async def verify_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query)
) -> str:
    """
    X-API-Key başlığını veya ?api_key query parametresini doğrular.
    Güvenli n8n ve harici entegrasyonlar için kullanılır.
    """
    if not settings.REQUIRE_API_KEY:
        return "DEV_MODE_NO_AUTH"

    # Header or Query check
    provided_key = header_key or query_key
    
    if not provided_key or provided_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yetkisiz erişim: Geçersiz veya eksik API Anahtarı (X-API-Key)."
        )
    return provided_key
