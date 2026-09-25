import os
import json
from starlette.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)
VALID_KEY = settings.API_KEY
AUTH_HEADERS = {"X-API-Key": VALID_KEY}
INVALID_HEADERS = {"X-API-Key": "wrong_key_123"}

def test_all():
    print("=" * 70)
    print("      DLAI_BIST API & GÜVENLİK / MUTEX ENTEGRASYON TESTİ")
    print("=" * 70)

    # 1. Public Endpoints
    print("[-] Testing GET /api/v1/health...")
    r = client.get("/api/v1/health")
    assert r.status_code == 200, f"Health failed: {r.status_code}"
    print("    -> Health OK:", r.json())

    print("[-] Testing GET /api/v1/market/regime...")
    r = client.get("/api/v1/market/regime")
    assert r.status_code == 200, f"Regime failed: {r.status_code}"
    print("    -> Market Regime OK:", r.json().get("portfolio_action"))

    print("[-] Testing GET /api/v1/champion/status...")
    r = client.get("/api/v1/champion/status")
    assert r.status_code == 200, f"Champion failed: {r.status_code}"
    print("    -> Champion OK:", r.json().get("champion_model"))

    print("[-] Testing GET /api/v1/signals/daily...")
    r = client.get("/api/v1/signals/daily")
    assert r.status_code == 200, f"Signals failed: {r.status_code}"
    print("    -> Signals OK: Count of Top Longs =", len(r.json().get("top_longs", [])))

    # 2. Security / Authentication Tests on POST /signals/refresh
    print("[-] Testing Security on POST /api/v1/signals/refresh (Unauthorized)...")
    r = client.post("/api/v1/signals/refresh")
    assert r.status_code == 403, f"Expected 403 Forbidden without header, got {r.status_code}"
    print("    -> 403 Forbidden without key verified OK.")

    r = client.post("/api/v1/signals/refresh", headers=INVALID_HEADERS)
    assert r.status_code == 403, f"Expected 403 Forbidden with invalid key, got {r.status_code}"
    print("    -> 403 Forbidden with invalid key verified OK.")

    print("[-] Testing Security on POST /api/v1/signals/refresh (Authorized)...")
    r = client.post("/api/v1/signals/refresh", headers=AUTH_HEADERS)
    assert r.status_code == 200, f"Authorized refresh failed: {r.status_code}"
    print("    -> Authorized signals refresh OK: Report date =", r.json().get("report_date"))

    # 3. Portfolio Endpoints
    print("[-] Testing GET /api/v1/portfolio/ledger...")
    r = client.get("/api/v1/portfolio/ledger")
    assert r.status_code == 200, f"Portfolio failed: {r.status_code}"
    data = r.json()
    print(f"    -> Ledger OK: Equity = {data.get('current_portfolio_value_try')} TL, Alpha = %{data.get('net_alpha_pct')}")

    print("[-] Testing Security on POST /api/v1/portfolio/reset (Unauthorized)...")
    r = client.post("/api/v1/portfolio/reset?capital=1000000")
    assert r.status_code == 403, f"Expected 403 without key, got {r.status_code}"
    print("    -> 403 Forbidden on reset verified OK.")

    print("[-] Testing Security on POST /api/v1/portfolio/reset (Authorized)...")
    r = client.post("/api/v1/portfolio/reset?capital=1000000", headers=AUTH_HEADERS)
    assert r.status_code == 200, f"Reset failed: {r.status_code}"
    data = r.json()
    assert data.get("current_portfolio_value_try") == 1000000.0, "Reset equity mismatch"
    assert data.get("current_cash_try") == 1000000.0, "Reset cash mismatch"
    print(f"    -> Reset OK: Equity reset to {data.get('current_portfolio_value_try')} TL")

    # 4. Sentiment Endpoints
    print("[-] Testing GET /api/v1/sentiment/kap...")
    r = client.get("/api/v1/sentiment/kap?limit=5")
    assert r.status_code == 200, f"Sentiment failed: {r.status_code}"
    print(f"    -> Sentiment OK: {r.json().get('total_cached_news')} total news, {len(r.json().get('news_items', []))} returned")

    # 5. HTML Dashboard & Embedded API Key
    print("[-] Testing GET / (HTML Dashboard)...")
    r = client.get("/")
    assert r.status_code == 200, f"Dashboard failed: {r.status_code}"
    assert "DLAI_BIST" in r.text, "Dashboard HTML title missing"
    assert "window.DLAI_API_KEY" in r.text, "Embedded DLAI_API_KEY missing from HTML"
    assert "resetPortfolioEquity" in r.text, "Reset function missing from HTML"
    print("    -> HTML Dashboard Render & Injected Key OK!")

    # 6. Execution Endpoint Security & Idempotency / Mutex Test
    print("[-] Testing Security on POST /api/v1/execution/run-daily (Unauthorized)...")
    r = client.post("/api/v1/execution/run-daily", json={"mode": "close_auction", "force_cash": False})
    assert r.status_code == 403, f"Expected 403 Forbidden on execution without key, got {r.status_code}"
    print("    -> 403 Forbidden on execution verified OK.")

    print("[-] Testing POST /api/v1/execution/run-daily (Authorized First Run)...")
    r1 = client.post(
        "/api/v1/execution/run-daily",
        headers=AUTH_HEADERS,
        json={"mode": "close_auction", "force_cash": False, "rebalance_step": 5, "force_execution": True}
    )
    assert r1.status_code == 200, f"Execution 1 failed: {r1.status_code} - {r1.text}"
    print("    -> First execution run OK:", r1.json().get("message"))

    print("[-] Testing Idempotency Conflict on POST /api/v1/execution/run-daily (Duplicate Run)...")
    r2 = client.post(
        "/api/v1/execution/run-daily",
        headers=AUTH_HEADERS,
        json={"mode": "close_auction", "force_cash": False, "rebalance_step": 5, "force_execution": False}
    )
    assert r2.status_code == 409, f"Expected 409 Conflict for duplicate run, got {r2.status_code}: {r2.text}"
    print(f"    -> 409 Conflict verified OK: {r2.json().get('detail')}")

    print("[-] Testing Bypass on POST /api/v1/execution/run-daily with force_execution=True...")
    r3 = client.post(
        "/api/v1/execution/run-daily",
        headers=AUTH_HEADERS,
        json={"mode": "close_auction", "force_cash": False, "rebalance_step": 5, "force_execution": True}
    )
    assert r3.status_code == 200, f"Bypass execution failed: {r3.status_code} - {r3.text}"
    print("    -> Force execution bypass verified OK:", r3.json().get("message"))

    print("\n" + "=" * 70)
    print(" [✓] ADIM 1: TÜM GÜVENLİK (API KEY) VE MUTEX / IDEMPOTENCY TESTLERİ BAŞARILI!")
    print("=" * 70)

if __name__ == "__main__":
    test_all()
