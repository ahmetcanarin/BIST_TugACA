# DLAI_BIST Quant Trading Engine - Production Dockerfile
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları (LightGBM için libgomp1, derlemeler için gcc/g++, healthcheck için curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pip güncellemesi
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CPU versiyonunu optimize kurulması (İmaj boyutunu ve build süresini minimize eder)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Python bağımlılıklarını kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje kaynak kodlarını kopyala
COPY . .

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Port
EXPOSE 8000

# Servis sağlık kontrolü (Healthcheck)
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Web sunucusunu başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
