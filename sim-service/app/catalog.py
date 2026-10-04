"""Neuron catalog: ready-made groups, name/id resolution and search."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GROUPS_FILE = Path(__file__).with_name("groups.yaml")
TEXT_COLS = ["cell_type", "hemibrain_type", "synonyms", "cell_sub_class", "cell_class", "super_class"]


class UnknownTarget(ValueError):
    def __init__(self, item: str, suggestions: list[str]):
        self.item = item
        self.suggestions = suggestions
        super().__init__(f"unknown neuron/group '{item}'")


@dataclass
class Group:
    key: str
    name: str
    description: str
    role: str
    default_rate_hz: float
    source: str
    idx: np.ndarray

    def info(self, neurons: pd.DataFrame) -> dict:
        sub = neurons.loc[self.idx]
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "n_neurons": int(len(self.idx)),
            "default_rate_hz": self.default_rate_hz,
            "sides": sub["side"].value_counts().to_dict(),
            "source": self.source,
        }


class Catalog:
    def __init__(self, neurons: pd.DataFrame, groups_file: Path = GROUPS_FILE):
        self.neurons = neurons
        self._rid2idx = pd.Series(neurons.index.to_numpy(), index=neurons["root_id"].to_numpy())
        self._type_lower = neurons["cell_type"].fillna("").str.lower()
        self._hb_lower = neurons["hemibrain_type"].fillna("").str.lower()
        self.groups: dict[str, Group] = {}
        for key, g in yaml.safe_load(groups_file.read_text(encoding="utf-8")).items():
            mask = np.ones(len(neurons), bool)
            for col, val in g["select"].items():
                vals = val if isinstance(val, list) else [val]
                mask &= neurons[col].isin(vals).to_numpy()
            idx = neurons.index.to_numpy()[mask]
            if len(idx) == 0:
                raise ValueError(f"group {key} resolved to 0 neurons: {g['select']}")
            self.groups[key] = Group(key, g["name"], g["description"], g["role"],
                                     float(g["default_rate_hz"]), g["source"], idx)
        self.group_of: dict[int, list[str]] = {}
        for key, g in self.groups.items():
            for i in g.idx:
                self.group_of.setdefault(int(i), []).append(key)

    # ------------------------------------------------------------------ resolution
    def resolve(self, items: list[str | int], side: str | None = None) -> tuple[np.ndarray, list[dict]]:
        """Resolve group keys, cell types (case-insensitive) or root ids into model indices.

        An item may carry a side suffix: "lc4:left", "DNp01:right".
        """
        out: list[np.ndarray] = []
        resolved: list[dict] = []
        for raw in items:
            item = str(raw).strip()
            item_side = side
            if ":" in item and item.rsplit(":", 1)[1].lower() in ("left", "right"):
                item, item_side = item.rsplit(":", 1)
                item_side = item_side.lower()
            idx, kind = self._resolve_one(item)
            if item_side:
                idx = idx[self.neurons.loc[idx, "side"].to_numpy() == item_side]
            out.append(idx)
            info = {"item": str(raw), "kind": kind, "n_neurons": int(len(idx))}
            if kind == "group":
                info["name"] = self.groups[item.lower()].name
            if item_side:
                info["side"] = item_side
            resolved.append(info)
        all_idx = np.unique(np.concatenate(out)) if out else np.array([], np.int64)
        return all_idx.astype(np.int64), resolved

    def _resolve_one(self, item: str) -> tuple[np.ndarray, str]:
        key = item.lower()
        if key in self.groups:
            return self.groups[key].idx, "group"
        if item.isdigit():
            rid = int(item)
            if rid in self._rid2idx.index:
                return np.array([self._rid2idx[rid]]), "root_id"
            raise UnknownTarget(item, [])
        mask = (self._type_lower == key) | (self._hb_lower == key)
        if mask.any():
            return self.neurons.index.to_numpy()[mask.to_numpy()], "cell_type"
        raise UnknownTarget(item, [h["name"] for h in self.search(item, limit=5)])

    # ------------------------------------------------------------------ search
    def search(self, q: str, limit: int = 15) -> list[dict]:
        """Search groups and cell types. Returns aggregated hits, not individual neurons."""
        q = q.strip().lower()
        if not q:
            return []
        hits: list[dict] = []
        for g in self.groups.values():
            if q in g.key or q in g.name.lower() or q in g.description.lower():
                score = 4 if q == g.key else 3 if (q in g.key or q in g.name.lower()) else 2
                hits.append({"kind": "group", "name": g.key, "title": g.name,
                             "n_neurons": int(len(g.idx)), "score": score})
        n = self.neurons
        if q.isdigit():
            rid = int(q)
            if rid in self._rid2idx.index:
                row = n.loc[self._rid2idx[rid]]
                hits.append({"kind": "neuron", "name": str(rid), "cell_type": row["cell_type"],
                             "super_class": row["super_class"], "side": row["side"],
                             "top_nt": row["top_nt"], "n_neurons": 1, "score": 3})
            return hits[:limit]
        match = np.zeros(len(n), bool)
        for col in TEXT_COLS:
            match |= n[col].fillna("").str.lower().str.contains(q, regex=False).to_numpy()
        sub = n[match]
        if len(sub):
            agg = (sub.assign(cell_type=sub["cell_type"].replace("", pd.NA).fillna(sub["cell_sub_class"]))
                   .groupby("cell_type", dropna=True)
                   .agg(n_neurons=("root_id", "size"), super_class=("super_class", "first"),
                        cell_class=("cell_class", "first"), top_nt=("top_nt", _mode),
                        hemibrain_type=("hemibrain_type", "first"), synonyms=("synonyms", "first")))
            for ct, row in agg.iterrows():
                exact = q in (str(ct).lower(), str(row["hemibrain_type"]).lower())
                hits.append({"kind": "cell_type", "name": ct, "n_neurons": int(row["n_neurons"]),
                             "super_class": row["super_class"], "cell_class": row["cell_class"],
                             "top_nt": row["top_nt"],
                             "aka": row["hemibrain_type"] if row["hemibrain_type"] not in ("", ct) else None,
                             "score": 3 if exact else 1})
        hits.sort(key=lambda h: (-h["score"], -h["n_neurons"]))
        for h in hits:
            h.pop("score")
        return hits[:limit]

    def describe(self, i: int) -> dict:
        row = self.neurons.loc[i]
        return {
            "root_id": str(row["root_id"]),
            "cell_type": row["cell_type"] or row["cell_sub_class"] or row["cell_class"] or "unannotated",
            "super_class": row["super_class"],
            "cell_class": row["cell_class"],
            "side": row["side"],
            "nt": row["top_nt"],
            "groups": self.group_of.get(int(i), []),
        }


def _mode(s: pd.Series):
    """Most common value, or the full breakdown when neurons of one type disagree."""
    s = s[s != ""]
    if not len(s):
        return None
    counts = s.value_counts()
    return counts.index[0] if len(counts) == 1 else {k: int(v) for k, v in counts.items()}
