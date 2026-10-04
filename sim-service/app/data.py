"""Loading of the compact artifacts produced by scripts/build_data.py."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))


@lru_cache(maxsize=1)
def load_weights() -> sp.csr_matrix:
    """Signed synapse counts, W[post, pre]."""
    return sp.load_npz(DATA_DIR / "weights.npz").tocsr()


@lru_cache(maxsize=1)
def load_neurons() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "neurons.parquet")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna("").astype(object)   # JSON-safe: no NaN in text fields
    return df
