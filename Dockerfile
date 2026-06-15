# ── Base image ────────────────────────────────────────────────
FROM python:3.12-slim

# ── System deps (LightGBM ต้องการ libgomp) ───────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ก่อน copy code (cache layer) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────
COPY diabetes.csv .
COPY api.py .
COPY app.py .
COPY database.py .

# ── Expose ports ──────────────────────────────────────────────
# 8000 = FastAPI  |  8501 = Streamlit
EXPOSE 8000 8501

# Default: รัน FastAPI (override ได้ใน docker-compose)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
