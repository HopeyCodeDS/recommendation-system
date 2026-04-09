FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy project source (needed for model classes at import time)
COPY src/ ./src/
COPY submit/ ./submit/
COPY scripts/ ./scripts/
COPY api/ ./api/

# Run precompute to train models and generate JSON data
RUN python scripts/precompute.py

EXPOSE 8000

# Railway injects $PORT; fall back to 8000 for other hosts
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
