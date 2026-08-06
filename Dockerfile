FROM python:3.11-slim

WORKDIR /app

# System deps needed to build some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install everything except torch first
RUN grep -v "^torch" requirements.txt > requirements-no-torch.txt \
    && pip install --no-cache-dir -r requirements-no-torch.txt

# Install CPU-only torch — this is a fraction of the size of the default
# CUDA build and is all that's needed to run FinBERT for inference.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY main.py .

ENV PORT=8000
EXPOSE 8000

# Render (and most PaaS providers) inject $PORT at runtime — bind to it.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
