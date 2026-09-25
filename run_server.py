import uvicorn
import os
import sys

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print("=" * 80)
    print(f"[*] Starting DLAI_BIST Quant Trading Terminal & Dashboard")
    print(f"[*] Dashboard URL : http://{host}:{port}")
    print(f"[*] API Docs URL  : http://{host}:{port}/docs")
    print(f"[*] ReDoc URL     : http://{host}:{port}/redoc")
    print("=" * 80)
    uvicorn.run("app.main:app", host=host, port=port, reload=False, workers=1)
