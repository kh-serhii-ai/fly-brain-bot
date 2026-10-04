"""Download FlyWire v783 data and build compact artifacts for the sim-service.

Sources (all public, no login):
  * Connectivity_783.parquet / Completeness_783.csv — Shiu et al. 2024 model repo
    (github.com/philshiu/Drosophila_brain_model, code MIT; data FlyWire CC BY-NC 4.0)
  * Supplemental_file1_neuron_annotations.tsv — Schlegel et al. 2024 / flyconnectome
    (github.com/flyconnectome/flywire_annotations, CC BY-NC 4.0)

Outputs in data/:
  * weights.npz      — CSR matrix W[post, pre] = signed synapse count (int32)
  * neurons.parquet  — one row per model neuron (index = model index)
  * meta.json        — counts and source info for sanity checks

Usage:
  python scripts/build_data.py            # download (if missing) + build
  python scripts/build_data.py --check    # only print sanity check of built data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW = OUT / "raw"

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
ANN = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files"
FILES = {
    "Completeness_783.csv": f"{SHIU}/Completeness_783.csv",
    "Connectivity_783.parquet": f"{SHIU}/Connectivity_783.parquet",
    "neuron_annotations.tsv": f"{ANN}/Supplemental_file1_neuron_annotations.tsv",
}

# Published v783 numbers (Dorkenwald et al. 2024): ~139k neurons, ~50M synapses
# (Shiu's edge list sums to 54.5M synapses, matching the published total;
# we check ranges to tolerate minor re-releases).
EXPECTED_NEURONS = (138_000, 140_500)


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = RAW / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  ✓ {name} already present ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  ↓ {name} <- {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
        print(f"    {dst.stat().st_size / 1e6:.1f} MB")


def _str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")


def build() -> None:
    comp = pd.read_csv(RAW / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(RAW / "Connectivity_783.parquet")
    ann = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", low_memory=False)

    n = len(comp)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    print(f"neurons in model: {n:,}; edges: {len(con):,}")

    # --- weight matrix: rows = post, cols = pre (so input current = W @ spikes)
    pre = con["Presynaptic_Index"].to_numpy(np.int32)
    post = con["Postsynaptic_Index"].to_numpy(np.int32)
    signed = con["Excitatory x Connectivity"].to_numpy(np.int32)
    W = sp.csr_matrix((signed, (post, pre)), shape=(n, n), dtype=np.int32)
    W.sum_duplicates()
    sp.save_npz(OUT / "weights.npz", W, compressed=True)

    # --- neuron table
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    a = ann.reindex(root_ids)
    missing_ann = int(a["super_class"].isna().sum()) if "super_class" in a else n
    keep = [c for c in [
        "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
        "hemibrain_type", "side", "nerve", "top_nt", "top_nt_conf",
        "known_nt", "synonyms", "status", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z",
    ] if c in a.columns]
    neurons = a[keep].copy()
    for c in neurons.columns:
        if neurons[c].dtype == object:
            neurons[c] = _str(neurons[c])
    neurons.insert(0, "root_id", root_ids)
    neurons.index = pd.RangeIndex(n, name="idx")

    # sign of outgoing synapses as used by the model (from Shiu's 'Excitatory' column)
    exc = con.groupby("Presynaptic_Index")["Excitatory"].first()
    neurons["sign"] = 0
    neurons.loc[exc.index, "sign"] = exc.to_numpy().astype(np.int8)

    in_deg = np.diff(W.indptr)
    out_deg = np.diff(W.tocsc().indptr)
    neurons["n_in"] = in_deg.astype(np.int32)
    neurons["n_out"] = out_deg.astype(np.int32)
    neurons.to_parquet(OUT / "neurons.parquet")

    meta = {
        "dataset": "FlyWire FAFB v783",
        "n_neurons": int(n),
        "n_edges": int(W.nnz),
        "n_synapses": int(np.abs(W.data).sum()),
        "n_excitatory_edges": int((W.data > 0).sum()),
        "n_inhibitory_edges": int((W.data < 0).sum()),
        "neurons_without_annotation": missing_ann,
        "sources": FILES,
        "license": "FlyWire data: CC BY-NC 4.0",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def check() -> bool:
    meta = json.loads((OUT / "meta.json").read_text())
    ok = EXPECTED_NEURONS[0] <= meta["n_neurons"] <= EXPECTED_NEURONS[1]
    print(f"neurons {meta['n_neurons']:,} in {EXPECTED_NEURONS}: {'OK' if ok else 'FAIL'}")
    syn_ok = 40e6 <= meta["n_synapses"] <= 60e6
    print(f"synapses {meta['n_synapses']:,} in [40M, 60M]: {'OK' if syn_ok else 'FAIL'}")
    return ok and syn_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        download()
        build()
    sys.exit(0 if check() else 1)
