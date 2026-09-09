FROM python:3.11-slim

WORKDIR /app

# System deps needed to build some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Every top-level module main.py imports has to be listed here. A missing
# one is not a soft failure: the import blows up at startup and the
# container exits before serving anything. tools/check_deploy_manifest.py
# checks this file against main.py's imports for exactly that reason.
COPY main.py db.py auth.py .
COPY screener_v2 ./screener_v2
COPY web ./web

ENV PORT=8000
EXPOSE 8000

# Render (and most PaaS providers) inject $PORT at runtime — bind to it.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
