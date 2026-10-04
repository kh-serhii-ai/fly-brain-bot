"""Glue between the API and the model: request handling, compact summaries, caching."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from .catalog import Catalog
from .data import load_neurons, load_weights
from .graph import MIN_SYNAPSES, Graph
from .sim import Connectome, SimResult, Simulator

ACTIVE_HZ = 1.0          # a neuron counts as "active" above this mean rate
OUTPUT_GROUPS = ("mn9", "giant_fiber", "descending", "motor")
RUNAWAY_ACTIVE = 5000    # this many active neurons = seizure-like runaway activity
TOP_N = 20
BEHAVIORS = yaml.safe_load((Path(__file__).with_name("behaviors.yaml")).read_text(encoding="utf-8"))
SCENARIOS = yaml.safe_load((Path(__file__).with_name("scenarios.yaml")).read_text(encoding="utf-8"))
QUIZ = yaml.safe_load((Path(__file__).with_name("quiz.yaml")).read_text(encoding="utf-8"))


@dataclass
class SimRecord:
    sim_id: str
    request: dict
    result: SimResult
    stim: np.ndarray
    silenced: np.ndarray
    summary: dict


class LRU(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get_item(self, key):
        with self.lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        return None

    def put(self, key, value):
        with self.lock:
            self[key] = value
            self.move_to_end(key)
            while len(self) > self.maxsize:
                self.popitem(last=False)


class Service:
    def __init__(self):
        t = time.time()
        self.neurons = load_neurons()
        W = load_weights()
        self.catalog = Catalog(self.neurons)
        self.sim = Simulator(Connectome.from_weights(W),
                             n_threads=int(os.environ.get("SIM_THREADS", os.cpu_count() or 2)),
                             max_ops_per_request=float(os.environ.get("SIM_MAX_OPS", 2e8)))
        self._W = W
        self.sim_cache = LRU(int(os.environ.get("SIM_CACHE_SIZE", 64)))
        self.sim_lock = threading.Semaphore(int(os.environ.get("SIM_CONCURRENCY", 1)))
        self.load_seconds = round(time.time() - t, 2)
        # compile the numba kernel now, not on the first user request
        self.sim.run([0], n_trials=1, duration_ms=5)

    @cached_property
    def graph(self) -> Graph:
        return Graph(self._W)

    # ------------------------------------------------------------------ simulate
    def simulate(self, req: dict) -> SimRecord:
        stim, stim_info = self.catalog.resolve(req["stimulate"], req.get("side"))
        silenced, sil_info = self.catalog.resolve(req.get("silence") or [])
        if len(stim) == 0:
            raise ValueError("nothing to stimulate (empty selection)")
        rate = req.get("rate_hz") or self._default_rate(req["stimulate"])
        canon = {
            "stimulate": sorted(map(str, req["stimulate"])), "silence": sorted(map(str, req.get("silence") or [])),
            "side": req.get("side"), "rate_hz": float(rate), "duration_ms": float(req["duration_ms"]),
            "n_trials": int(req["n_trials"]), "seed": int(req["seed"]),
        }
        if req.get("record_spikes"):
            canon["record_spikes"] = True   # part of the key: a cached run without spikes can't be animated
        sim_id = hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]
        if (hit := self.sim_cache.get_item(sim_id)) is not None:
            return hit
        t = time.time()
        with self.sim_lock:
            res = self.sim.run(stim, silence=silenced, rate_hz=rate, duration_ms=canon["duration_ms"],
                               n_trials=canon["n_trials"], seed=canon["seed"],
                               record_spikes=bool(req.get("record_spikes")))
        elapsed = time.time() - t
        summary = self._summarize(sim_id, canon, res, stim, silenced, stim_info, sil_info, elapsed)
        rec = SimRecord(sim_id, canon, res, stim, silenced, summary)
        self.sim_cache.put(sim_id, rec)
        return rec

    def _default_rate(self, items) -> float:
        rates = [self.catalog.groups[str(i).split(":")[0].lower()].default_rate_hz
                 for i in items if str(i).split(":")[0].lower() in self.catalog.groups]
        return min(rates) if rates else 150.0

    def _summarize(self, sim_id, canon, res: SimResult, stim, silenced, stim_info, sil_info, elapsed) -> dict:
        r = res.rates
        stim_set = set(stim.tolist())
        active = np.flatnonzero(r >= ACTIVE_HZ)
        resp = np.array([i for i in active if i not in stim_set], dtype=np.int64)
        resp = resp[np.argsort(-r[resp])] if len(resp) else resp

        def neuron(i):
            d = self.catalog.describe(int(i))
            d["rate_hz"] = round(float(r[i]), 1)
            return d

        sc = self.neurons.loc[resp, "super_class"].replace("", "unannotated").value_counts()
        groups = []
        for key, g in self.catalog.groups.items():
            members = np.setdiff1d(g.idx, stim)
            if len(members) == 0:
                continue
            act = members[r[members] >= ACTIVE_HZ]
            if len(act):
                groups.append({"group": key, "name": g.name, "active": int(len(act)),
                               "of": int(len(members)), "mean_rate_hz": round(float(r[act].mean()), 1)})
        groups.sort(key=lambda x: -x["active"] / x["of"])

        outputs = {}
        for key in OUTPUT_GROUPS:
            g = self.catalog.groups[key]
            members = np.setdiff1d(g.idx, stim)
            act = members[r[members] >= ACTIVE_HZ]
            act = act[np.argsort(-r[act])]
            outputs[key] = {"name": g.name, "active": int(len(act)), "of": int(len(members)),
                            "reached": bool(len(act)), "top": [neuron(i) for i in act[:5]]}

        behavior = self._behavior(r, stim, len(resp), len(resp) > RUNAWAY_ACTIVE)
        stim_rate = float(r[stim].mean()) if len(stim) else 0.0
        return {
            "sim_id": sim_id,
            "params": {**canon, "model": "LIF, Shiu et al. 2024 parameters, FlyWire v783"},
            "stimulated": {"n_neurons": int(len(stim)), "items": stim_info,
                           "mean_rate_hz": round(stim_rate, 1)},
            "silenced": {"n_neurons": int(len(silenced)), "items": sil_info},
            "behavior": behavior,
            "n_active": int(len(resp)),
            # seizure-like activity; a run cut short by the time budget alone is only "truncated"
            "runaway": bool(len(resp) > RUNAWAY_ACTIVE),
            "truncated": res.truncated,
            "simulated_ms": round(res.simulated_ms, 1),
            "outputs": outputs,
            "top_neurons": [neuron(i) for i in resp[:TOP_N]],
            "active_by_super_class": sc.head(10).to_dict(),
            "groups_activated": groups[:12],
            "elapsed_s": round(elapsed, 2),
            "note": "Model prediction (connectome + simplified LIF neurons), not a recording from a real fly.",
        }

    def _behavior(self, r: np.ndarray, stim: np.ndarray, n_responding: int, runaway: bool) -> dict:
        """Behaviour verdicts from validated output neurons and thresholds in behaviors.yaml."""
        out: dict = {}
        for key, b in BEHAVIORS.items():
            idx = self.catalog.groups[b["group"]].idx
            rate = float(r[idx].max())
            out[key] = {
                "active": rate >= b["threshold_hz"],
                "rate_hz": round(rate, 1),
                "neurons": b["neurons"],
                "label": b["label"],
                "icon": b["icon"],
                "threshold_hz": b["threshold_hz"],
                "weak": ACTIVE_HZ <= rate < b["threshold_hz"],
                # the output neurons themselves were stimulated, so the verdict proves nothing
                "stimulated_directly": bool(np.isin(idx, stim).any()),
            }
        out["responding_neurons"] = int(n_responding)
        out["total_neurons"] = int(len(r))
        out["runaway"] = bool(runaway)
        return out

    # ------------------------------------------------------------------ path
    def path(self, req: dict) -> dict:
        src, src_info = self.catalog.resolve([req["from"]], req.get("from_side"))
        dst, dst_info = self.catalog.resolve([req["to"]], req.get("to_side"))
        paths = self.graph.paths(src, dst, max_hops=req["max_hops"], k=req["k"])
        out = []
        for p in paths:
            nodes = [self.catalog.describe(i) for i in p]
            edges = [self.graph.edge(a, b) for a, b in zip(p[:-1], p[1:])]
            strength = float(np.prod([e["input_fraction"] for e in edges]))
            out.append({"hops": len(edges), "nodes": nodes, "edges": edges,
                        "path_strength": strength})
        for info in (src_info[0], dst_info[0]):
            g = self.catalog.groups.get(info["item"].split(":")[0].lower())
            info["name"] = g.name if g else info["item"]
        return {"from": src_info[0], "to": dst_info[0], "max_hops": req["max_hops"],
                "found": bool(out), "paths": out,
                "note": f"Edges with >= {MIN_SYNAPSES} synapses; input_fraction = share of the next neuron's input synapses; path_strength = product."}

    # ------------------------------------------------------------------ compare
    def compare(self, baseline_req: dict, variant_req: dict, scenario: dict | None = None) -> dict:
        """Same stimulus protocol for a normal and a changed brain; seed/trials/duration are shared."""
        shared = {k: baseline_req[k] for k in ("duration_ms", "n_trials", "seed")}
        base = self.simulate(baseline_req)
        var = self.simulate({**variant_req, **shared})
        rb, rv = base.result.rates, var.result.rates

        behavior = {}
        for key in BEHAVIORS:
            b, v = base.summary["behavior"][key], var.summary["behavior"][key]
            behavior[key] = {"baseline": b["active"], "variant": v["active"],
                             "rate_baseline": b["rate_hz"], "rate_variant": v["rate_hz"],
                             "changed": b["active"] != v["active"]}

        skip = set(base.stim.tolist()) | set(var.stim.tolist())
        act_b, act_v = rb >= ACTIVE_HZ, rv >= ACTIVE_HZ

        def neurons(mask, order_rates):
            idx = np.array([i for i in np.flatnonzero(mask) if i not in skip], dtype=np.int64)
            idx = idx[np.argsort(-order_rates[idx])][:10] if len(idx) else idx
            return [{**self.catalog.describe(int(i)), "rate_baseline": round(float(rb[i]), 1),
                     "rate_variant": round(float(rv[i]), 1)} for i in idx]

        groups = []
        for key, g in self.catalog.groups.items():
            members = np.setdiff1d(g.idx, np.fromiter(skip, np.int64, len(skip)))
            nb, nv = int(act_b[members].sum()), int(act_v[members].sum())
            if nb != nv:
                groups.append({"group": key, "name": g.name, "baseline_active": nb, "variant_active": nv,
                               "of": int(len(members))})
        groups.sort(key=lambda x: -abs(x["baseline_active"] - x["variant_active"]))

        labels = {"baseline": "Звичайна муха", "variant": self._variant_label(base.summary, var.summary)}
        if scenario:
            labels = {"baseline": scenario["baseline_label"], "variant": scenario["variant_label"]}
        return {
            "compare_id": f"{base.sim_id}-{var.sim_id}",
            "scenario": scenario,
            "labels": labels,
            "baseline": self._compact(base.summary),
            "variant": self._compact(var.summary),
            "diff": {
                "behavior": behavior,
                "responding": {"baseline": base.summary["n_active"], "variant": var.summary["n_active"]},
                "lost": neurons(act_b & ~act_v, rb),
                "gained": neurons(act_v & ~act_b, rv),
                "groups": groups[:8],
            },
            "note": "Same stimulus protocol and random seed for both brains; model prediction, not a recording.",
        }

    @staticmethod
    def _compact(s: dict) -> dict:
        keep = ("sim_id", "params", "stimulated", "silenced", "behavior", "n_active", "runaway", "truncated",
                "simulated_ms")
        return {**{k: s[k] for k in keep}, "top_neurons": s["top_neurons"][:8]}

    @staticmethod
    def _variant_label(base: dict, var: dict) -> str:
        names = lambda items: ", ".join(i.get("name", i["item"]) for i in items)
        parts = []
        if var["silenced"]["items"]:
            parts.append(f"вимкнено: {names(var['silenced']['items'])}")
        extra = [i for i in var["stimulated"]["items"] if i["item"] not in {b["item"] for b in base["stimulated"]["items"]}]
        if extra:
            parts.append(f"додано: {names(extra)}")
        return "Змінена муха" + (f" ({'; '.join(parts)})" if parts else "")
