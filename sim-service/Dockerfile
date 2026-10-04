FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl \
    DATA_DIR=/app/data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY docker-entrypoint.sh .

RUN useradd -m sim && mkdir -p /app/data && chown -R sim /app
USER sim

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
ENTRYPOINT ["./docker-entrypoint.sh"]
