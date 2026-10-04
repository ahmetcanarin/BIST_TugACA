"""
BIST 100 — n8n Cloudflare HTTPS Tünel Senkronizasyon Aracı

Bu script, Cloudflare Quick Tunnel'ın ürettiği canlı HTTPS adresini okur,
.env dosyasındaki WEBHOOK_URL ve N8N_WEBHOOK_URL değişkenlerini günceller
ve n8n konteynerinin Telegram ile kesintisiz haberleşmesini sağlar.

Kullanım:
    python sync_tunnel.py
"""

import subprocess
import re
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 60)
    print("🔄 BIST 100 — n8n HTTPS Tünel Kontrolü...")
    print("=" * 60)

    # 1. bist_tunnel loglarından en güncel URL'yi çek
    try:
        res = subprocess.run(
            ["docker", "logs", "bist_tunnel"],
            capture_output=True,
            text=True,
            check=True
        )
        logs = res.stdout + res.stderr
    except Exception as e:
        print(f"❌ Docker logları okunamadı: {e}")
        sys.exit(1)

    urls = re.findall(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', logs)
    if not urls:
        print("⚠️ Aktif trycloudflare.com URL'si bulunamadı. Tünel konteyneri başlatılıyor mu?")
        sys.exit(1)

    active_url = urls[-1].rstrip("/") + "/"
    print(f"🌐 Tespit Edilen Aktif HTTPS Tünel URL: {active_url}")

    # 2. .env dosyasını kontrol et
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    current_webhook = None

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r'WEBHOOK_URL=(https://[^\s]+)', content)
        if m:
            current_webhook = m.group(1).rstrip("/") + "/"

    if current_webhook == active_url:
        print("✅ .env dosyasındaki WEBHOOK_URL zaten güncel. Her şey yolunda!")
        return

    print(f"📝 .env güncelleniyor: {current_webhook} ➔ {active_url}")

    new_env_content = f"""# =============================================================================
# BIST 100 — Ortam Değişkenleri (.env)
# =============================================================================

# Cloudflare HTTPS Tünel Adresi (Telegram Webhook için aktif ve canlı URL)
WEBHOOK_URL={active_url}
N8N_WEBHOOK_URL={active_url}

# Telegram & n8n
N8N_PORT=5678
GENERIC_TIMEZONE=Europe/Istanbul
TZ=Europe/Istanbul
"""
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(new_env_content)

    print("🚀 n8n konteyneri yeni HTTPS URL ile güncelleniyor...")
    subprocess.run(["docker", "compose", "up", "-d", "n8n"], check=True)
    print("=" * 60)
    print("🎉 TAMAMLANDI! n8n artık şu adresten ve Telegram'dan erişilebilir:")
    print(f"👉 {active_url}")
    print("=" * 60)

if __name__ == "__main__":
    main()
