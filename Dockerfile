# =============================================================================
# BIST 100 — 360° Yatırımcı Karar Destek & n8n REST API Dockerfile
# =============================================================================
FROM python:3.11-slim

# Temel ortam ayarları
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# LightGBM, PyTorch ve ağ işlemleri için zorunlu sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılık listesini kopyala
COPY requirements.txt .

# PyTorch CPU versiyonu ile konteyner boyutunu optimize et ve paketleri kur
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını ve önceden eğitilmiş model ağırlıklarını kopyala
COPY . .

# REST API (5050) ve Streamlit Dashboard (8501) portlarını dışa aç
EXPOSE 5050 8501

# Varsayılan olarak API sunucusunu başlat
CMD ["python", "api_server.py"]
