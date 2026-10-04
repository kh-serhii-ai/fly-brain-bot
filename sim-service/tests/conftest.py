import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_READY = (ROOT / "data" / "weights.npz").exists() and (ROOT / "data" / "neurons.parquet").exists()
needs_data = pytest.mark.skipif(not DATA_READY, reason="run scripts/build_data.py first")
