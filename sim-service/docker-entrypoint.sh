#!/bin/sh
set -e
# FlyWire data is not baked into the image (CC BY-NC 4.0); fetch & build it on first start.
if [ ! -f "$DATA_DIR/weights.npz" ] || [ ! -f "$DATA_DIR/neurons.parquet" ]; then
  echo "Data not found in $DATA_DIR, downloading and building (one-off, ~2 min)..."
  python scripts/build_data.py
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
