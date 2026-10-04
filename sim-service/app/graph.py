"""Strongest-path search in the connectome graph.

Edge pre -> post is kept if it has >= MIN_SYNAPSES synapses (FlyWire's usual threshold).
Its strength is the fraction of post's input synapses that come from pre, and its cost is
-log(fraction), so the cheapest path is the chain of the strongest relative connections.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

MIN_SYNAPSES = 5


class Graph:
    def __init__(self, W_post_pre: sp.csr_matrix, min_synapses: int = MIN_SYNAPSES):
        absW = abs(W_post_pre).astype(np.float64).tocsr()
        in_total = np.asarray(absW.sum(axis=1)).ravel()          # input synapses per post
        A = W_post_pre.T.tocsr()                                   # [pre, post], signed
        A.data = A.data.astype(np.float64)
        keep = np.abs(A.data) >= min_synapses
        A = sp.csr_matrix((A.data[keep], A.indices[keep], _filtered_indptr(A.indptr, keep)),
                          shape=A.shape)
        self.signed = A
        frac = A.copy()
        post = frac.indices
        frac.data = np.abs(frac.data) / in_total[post]
        self.frac = frac
        cost = frac.copy()
        cost.data = -np.log(cost.data) + 1e-6                      # strictly positive
        self.cost = cost

    def paths(self, sources: np.ndarray, targets: np.ndarray, max_hops: int = 6,
              k: int = 3) -> list[list[int]]:
        """Up to k strongest paths from any source to distinct targets, at most max_hops edges."""
        sources = np.asarray(sources)
        targets = np.setdiff1d(np.asarray(targets), sources)
        if len(sources) == 0 or len(targets) == 0:
            return []
        dist, pred, src = dijkstra(self.cost, directed=True, indices=sources,
                                   min_only=True, return_predecessors=True)
        reachable = targets[np.isfinite(dist[targets])]
        order = reachable[np.argsort(dist[reachable])]
        out: list[list[int]] = []
        for t in order:
            path = [int(t)]
            while pred[path[-1]] >= 0:
                path.append(int(pred[path[-1]]))
            path.reverse()
            if len(path) - 1 <= max_hops:
                out.append(path)
            if len(out) >= k:
                break
        if not out and len(order):
            # strongest paths are too long: fall back to the fewest-hop path
            hop = self.cost.copy()
            hop.data[:] = 1.0
            d2, p2, _ = dijkstra(hop, directed=True, indices=sources, min_only=True,
                                 return_predecessors=True)
            for t in targets[np.argsort(d2[targets])][:k]:
                if not np.isfinite(d2[t]) or d2[t] > max_hops:
                    break
                path = [int(t)]
                while p2[path[-1]] >= 0:
                    path.append(int(p2[path[-1]]))
                out.append(path[::-1])
        return out

    def edge(self, pre: int, post: int) -> dict:
        syn = self.signed[pre, post]
        return {"synapses": int(abs(syn)), "sign": "excitatory" if syn > 0 else "inhibitory",
                "input_fraction": round(float(self.frac[pre, post]), 4)}


def _filtered_indptr(indptr: np.ndarray, keep: np.ndarray) -> np.ndarray:
    csum = np.concatenate([[0], np.cumsum(keep)])
    return csum[indptr]
