"""Leaky integrate-and-fire simulation of the whole FlyWire brain.

Re-implementation of the Shiu et al. 2024 Brian2 model (github.com/philshiu/Drosophila_brain_model)
in numba, so a full-brain run takes about a second instead of minutes:

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike: v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    synapse: g_post += w_syn * n_synapses * sign(pre),  delay t_dly
    stimulus: Poisson input to v with weight w_syn * f_poi (enough to cause a spike)

The ODE is linear, so it is integrated exactly (same as Brian2 method='linear'), with the
Brian2 per-step order: state update -> threshold -> synaptic/Poisson input -> reset.

Speed trick (exact, no approximation): with u = v - v_0, the free trajectory is
    u(s) = A e^{-s/tau} + (u_0 - A) e^{-s/t_mbr},   A = g_0 tau / (tau - t_mbr)
whose maximum has a closed form. A neuron whose maximum stays below threshold cannot spike
until it gets new input, so it is left "passive" and its state is advanced analytically only
when input arrives. Only neurons that may cross threshold are stepped every dt.
Spike delivery then dominates the cost, so the per-neuron state is packed into one row and
decay factors come from a lookup table (see _run_trial).
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Params:
    # Values from Shiu et al. 2024 (model.py default_params)
    v_0: float = -52.0      # mV, resting potential
    v_rst: float = -52.0    # mV, reset potential
    v_th: float = -45.0     # mV, spike threshold
    t_mbr: float = 20.0     # ms, membrane time constant
    tau: float = 5.0        # ms, synaptic time constant
    t_rfc: float = 2.2      # ms, refractory period
    t_dly: float = 1.8      # ms, synaptic delay
    w_syn: float = 0.275    # mV per synapse
    f_poi: float = 250.0    # Poisson input weight multiplier
    dt: float = 0.1         # ms, Brian2 default


@dataclass
class Connectome:
    """Outgoing connections in CSR form: for pre neuron i, posts are idx[ptr[i]:ptr[i+1]]."""
    ptr: np.ndarray       # int64 [n+1]
    idx: np.ndarray       # int32 [nnz]
    syn: np.ndarray       # float64 [nnz], signed synapse counts
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.ptr) - 1

    @classmethod
    def from_weights(cls, W_post_pre: sp.csr_matrix) -> "Connectome":
        # W is [post, pre]; outgoing lists need [pre, post] in CSR == W in CSC
        out = W_post_pre.T.tocsr()
        out.sort_indices()
        return cls(out.indptr.astype(np.int64), out.indices.astype(np.int32),
                   out.data.astype(np.float64))


@nb.njit(cache=True, inline="always")
def _may_spike(u0, g0, u_th, tau, tm):
    """True if the free trajectory starting at (u0, g0) can reach u_th."""
    if u0 >= u_th:
        return True
    if g0 <= 0.0:
        return False                      # u decays from u0 (< u_th) or is pushed down
    # cheap bound: u(s) <= max(u0, 0) + integral of g/t_mbr = max(u0, 0) + g0 * tau / t_mbr
    if max(u0, 0.0) + g0 * tau / tm < u_th:
        return False
    A = g0 * tau / (tau - tm)
    B = u0 - A
    ratio = -B * tau / (A * tm)
    if ratio <= 0.0:
        return False
    s = math.log(ratio) / (1.0 / tm - 1.0 / tau)
    if s <= 0.0:
        return False
    peak = A * math.exp(-s / tau) + B * math.exp(-s / tm)
    return peak >= u_th - 1e-9


@nb.njit(cache=True, nogil=True)
def _run_trial(ptr, idx, w, n, n_steps, stim, stim_events, silenced,
               dt, tau, tm, v_0, v_rst, v_th, rfc_steps, dly_steps, w_poi, max_ops, rec_cap):
    """One trial. Returns (spike counts per neuron int32[n], number of simulated steps,
    recorded spike neurons, recorded spike steps).

    The first rec_cap spikes are recorded (neuron, step) for animation; rec_cap = 0 disables it.
    Recording only copies data out and never changes the dynamics.

    The trial stops early once more than max_ops synaptic events have been delivered
    (runaway, seizure-like activity), so a request has a bounded cost.

    stim_events: uint8[n_steps, len(stim)] — Poisson events per step and stimulated neuron.
    w: synaptic weights in mV (already multiplied by w_syn).
    """
    a = math.exp(-dt / tm)
    c = math.exp(-dt / tau)
    b = tau / (tau - tm) * (c - a)
    kA = tau / (tau - tm)
    u_th = v_th - v_0
    # decay factors for advancing a passive neuron by k steps (table instead of exp per synapse)
    ks = np.arange(n_steps + 1) * dt
    pow_a = np.exp(-ks / tm)
    pow_c = np.exp(-ks / tau)

    # Per-neuron state packed in one row so a synapse touches one cache line:
    # S[i, 0] = u = v - v_0,  S[i, 1] = g,  S[i, 2] = step at which (u, g) of a passive
    # neuron is valid,  S[i, 3] = 1.0 if the neuron is in the active (stepped) set.
    S = np.zeros((n, 4))
    counts = np.zeros(n, np.int32)
    refr_until = np.full(n, -1, np.int64)    # last step of refractoriness
    rfc = np.full(n, rfc_steps, np.int64)
    for k in range(len(stim)):
        rfc[stim[k]] = 0                     # Poisson targets have no refractory period
    is_silenced = np.zeros(n, np.bool_)
    for k in range(len(silenced)):
        is_silenced[silenced[k]] = True

    active = np.empty(n, np.int32)
    n_active = 0

    ring = dly_steps + 1                     # spikes awaiting delivery after the delay
    q_len = np.zeros(ring, np.int64)
    q_buf = np.empty((ring, n), np.int32)
    spiked = np.empty(n, np.int32)
    ops = 0
    rec_i = np.empty(rec_cap, np.int32)
    rec_t = np.empty(rec_cap, np.int32)
    n_rec = 0

    for t in range(n_steps):
        if ops > max_ops:
            return counts, t, rec_i[:n_rec], rec_t[:n_rec]
        # 1) state update + threshold for active neurons
        n_spk = 0
        new_n = 0
        for j in range(n_active):
            i = active[j]
            if t > refr_until[i]:
                ui = S[i, 0]
                gi = S[i, 1]
                S[i, 0] = a * ui + b * gi
                S[i, 1] = c * gi
                S[i, 2] = t
                if S[i, 0] > u_th:
                    spiked[n_spk] = i
                    n_spk += 1
                elif not _may_spike(S[i, 0], S[i, 1], u_th, tau, tm):
                    S[i, 3] = 0.0     # becomes passive, state valid at step t
                    continue
            else:
                S[i, 2] = t                  # frozen while refractory
            active[new_n] = i
            new_n += 1
        n_active = new_n

        # 2) enqueue this step's spikes
        slot = t % ring
        q_len[slot] = 0
        for s in range(n_spk):
            i = spiked[s]
            counts[i] += 1
            if n_rec < rec_cap:
                rec_i[n_rec] = i
                rec_t[n_rec] = t
                n_rec += 1
            if not is_silenced[i]:
                q_buf[slot, q_len[slot]] = i
                q_len[slot] += 1

        # 3) deliver spikes emitted dly_steps ago, then Poisson input
        if t >= dly_steps:
            src = (t - dly_steps) % ring
            for s in range(q_len[src]):
                pre = q_buf[src, s]
                ops += ptr[pre + 1] - ptr[pre]
                for e in range(ptr[pre], ptr[pre + 1]):
                    post = idx[e]
                    if S[post, 3] == 0.0:
                        k = t - int(S[post, 2])   # advance passive neuron analytically to step t
                        if k > 0:
                            g0 = S[post, 1]
                            A = kA * g0
                            S[post, 0] = A * pow_c[k] + (S[post, 0] - A) * pow_a[k]
                            S[post, 1] = g0 * pow_c[k]
                            S[post, 2] = t
                    S[post, 1] += w[e]
                    # inhibition cannot make a passive neuron able to spike
                    if w[e] > 0.0 and S[post, 3] == 0.0 and _may_spike(S[post, 0], S[post, 1], u_th, tau, tm):
                        S[post, 3] = 1.0
                        active[n_active] = post
                        n_active += 1

        for k_s in range(len(stim)):
            ev = stim_events[t, k_s]
            if ev:
                i = stim[k_s]
                if S[i, 3] == 0.0:
                    k = t - int(S[i, 2])
                    if k > 0:
                        g0 = S[i, 1]
                        A = kA * g0
                        S[i, 0] = A * pow_c[k] + (S[i, 0] - A) * pow_a[k]
                        S[i, 1] = g0 * pow_c[k]
                        S[i, 2] = t
                    S[i, 3] = 1.0
                    active[n_active] = i
                    n_active += 1
                S[i, 0] += ev * w_poi

        # 4) reset
        for s in range(n_spk):
            i = spiked[s]
            S[i, 0] = v_rst - v_0
            S[i, 1] = 0.0
            refr_until[i] = t + rfc[i]
    return counts, n_steps, rec_i[:n_rec], rec_t[:n_rec]


@dataclass
class SpikeRecord:
    """Spikes of one trial: neuron index and time in ms, in time order."""
    neuron: np.ndarray       # int32
    t_ms: np.ndarray         # float32
    complete: bool           # False if the recording buffer filled up


@dataclass
class SimResult:
    rates: np.ndarray        # Hz, mean over trials, float32[n]
    rates_std: np.ndarray    # Hz, std over trials
    n_trials: int
    duration_ms: float       # requested duration
    simulated_ms: float      # mean actually simulated duration (< duration_ms if truncated)
    spikes: SpikeRecord | None = None   # first trial, only when recording was requested

    @property
    def truncated(self) -> bool:
        return self.simulated_ms < self.duration_ms - 1e-6


class Simulator:
    def __init__(self, conn: Connectome, params: Params | None = None, n_threads: int | None = None,
                 max_ops_per_request: float = 4e8):
        self.conn = conn
        self.p = params = params or Params()
        self.w = conn.syn * params.w_syn
        self.rfc_steps = int(round(params.t_rfc / params.dt))
        self.dly_steps = int(round(params.t_dly / params.dt))
        self.n_threads = n_threads
        self.max_ops_per_request = max_ops_per_request

    def _trial(self, stim: np.ndarray, silenced: np.ndarray, rate_hz: float,
               n_steps: int, seed: np.random.SeedSequence, max_ops: int, rec_cap: int = 0):
        rng = np.random.default_rng(seed)
        prob = rate_hz * self.p.dt * 1e-3
        events = (rng.random((n_steps, len(stim))) < prob).astype(np.uint8)
        p = self.p
        return _run_trial(self.conn.ptr, self.conn.idx, self.w, self.conn.n, n_steps,
                          stim, events, silenced, p.dt, p.tau, p.t_mbr,
                          p.v_0, p.v_rst, p.v_th, self.rfc_steps, self.dly_steps,
                          p.w_syn * p.f_poi, max_ops, rec_cap)

    def run(self, stimulate, silence=(), rate_hz: float = 150.0, duration_ms: float = 1000.0,
            n_trials: int = 10, seed: int = 0, record_spikes: bool = False,
            record_cap: int = 300_000) -> SimResult:
        stim = np.unique(np.asarray(stimulate, np.int32))
        slnc = np.unique(np.asarray(silence, np.int32))
        n_steps = int(round(duration_ms / self.p.dt))
        seeds = np.random.SeedSequence(seed).spawn(n_trials)
        max_ops = int(self.max_ops_per_request / n_trials)
        with ThreadPoolExecutor(self.n_threads) as ex:
            caps = [record_cap if (record_spikes and k == 0) else 0 for k in range(n_trials)]
            out = list(ex.map(lambda sc: self._trial(stim, slnc, rate_hz, n_steps, sc[0], max_ops, sc[1]),
                              zip(seeds, caps)))
        steps = np.array([o[1] for o in out], np.float32)
        rates = np.stack([o[0] for o in out]).astype(np.float32) / (steps[:, None] * self.p.dt * 1e-3)
        spikes = None
        if record_spikes:
            rec_i, rec_t = out[0][2], out[0][3]
            spikes = SpikeRecord(rec_i.copy(), (rec_t * self.p.dt).astype(np.float32), len(rec_i) < record_cap)
        return SimResult(rates.mean(0), rates.std(0), n_trials, duration_ms,
                         float(steps.mean() * self.p.dt), spikes)
