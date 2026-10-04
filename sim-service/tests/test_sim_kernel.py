"""The lazy (event-driven) kernel must give exactly the same spikes as naive stepping."""
import math

import numpy as np
import scipy.sparse as sp

from app.sim import Connectome, Params, Simulator


def naive_counts(W, stim, events, silenced, p: Params, n_steps):
    """Straightforward Brian2-order simulation, every neuron updated every step."""
    n = W.shape[0]
    a, c = math.exp(-p.dt / p.t_mbr), math.exp(-p.dt / p.tau)
    b = p.tau / (p.tau - p.t_mbr) * (c - a)
    dly, rfc_steps = round(p.t_dly / p.dt), round(p.t_rfc / p.dt)
    Wm = W.toarray().astype(float) * p.w_syn
    Wm[:, silenced] = 0
    v, g = np.full(n, p.v_0), np.zeros(n)
    rfc = np.full(n, rfc_steps); rfc[stim] = 0
    refr_until = np.full(n, -1)
    counts = np.zeros(n, int)
    history = []
    for t in range(n_steps):
        free = t > refr_until
        u = v - p.v_0
        v = np.where(free, p.v_0 + a * u + b * g, v)
        g = np.where(free, c * g, g)
        spk = free & (v > p.v_th)
        counts += spk
        history.append(spk.copy())
        if t >= dly:
            g = g + Wm @ history[t - dly]
        v[stim] += events[t] * p.w_syn * p.f_poi
        v[spk] = p.v_rst
        g[spk] = 0
        refr_until[spk] = t + rfc[spk]
    return counts


def test_lazy_kernel_matches_naive():
    rng = np.random.default_rng(1)
    n = 300
    dense = (rng.random((n, n)) < 0.05) * rng.integers(1, 40, (n, n))
    sign = np.where(rng.random(n) < 0.75, 1, -1)          # per presynaptic neuron
    W = sp.csr_matrix(dense * sign[None, :])
    p = Params()
    sim = Simulator(Connectome.from_weights(W), p)
    stim = np.arange(10, dtype=np.int32)
    silenced = np.array([50, 51, 52], np.int32)
    n_steps = 3000
    seed = np.random.SeedSequence(7)
    ev_rng = np.random.default_rng(seed)
    events = (ev_rng.random((n_steps, len(stim))) < 150 * p.dt * 1e-3).astype(np.uint8)

    fast, steps, rec_i, rec_t = sim._trial(stim, silenced, 150.0, n_steps, seed, max_ops=10**12, rec_cap=10**6)
    slow = naive_counts(W, stim, events, silenced, p, n_steps)
    assert steps == n_steps
    assert slow.sum() > 500, "network should be active for a meaningful comparison"
    np.testing.assert_array_equal(fast, slow)
    # the recording holds exactly the counted spikes, in time order
    np.testing.assert_array_equal(np.bincount(rec_i, minlength=n), fast)
    assert np.all(np.diff(rec_t) >= 0)


def test_deterministic_seed():
    rng = np.random.default_rng(0)
    W = sp.random(200, 200, density=0.05, random_state=1, data_rvs=lambda k: rng.integers(1, 30, k)).tocsr()
    sim = Simulator(Connectome.from_weights(W.astype(np.int32)))
    r1 = sim.run(range(5), n_trials=4, duration_ms=200, seed=3)
    r2 = sim.run(range(5), n_trials=4, duration_ms=200, seed=3)
    r3 = sim.run(range(5), n_trials=4, duration_ms=200, seed=4)
    np.testing.assert_array_equal(r1.rates, r2.rates)
    assert not np.array_equal(r1.rates, r3.rates)


def test_recording_does_not_change_results():
    rng = np.random.default_rng(0)
    W = sp.random(300, 300, density=0.05, random_state=2, data_rvs=lambda k: rng.integers(1, 30, k)).tocsr()
    sim = Simulator(Connectome.from_weights(W.astype(np.int32)))
    plain = sim.run(range(8), n_trials=3, duration_ms=300, seed=5)
    rec = sim.run(range(8), n_trials=3, duration_ms=300, seed=5, record_spikes=True)
    np.testing.assert_array_equal(plain.rates, rec.rates)
    assert rec.spikes is not None and rec.spikes.complete and len(rec.spikes.neuron) > 0
    capped = sim.run(range(8), n_trials=1, duration_ms=300, seed=5, record_spikes=True, record_cap=10)
    assert len(capped.spikes.neuron) == 10 and not capped.spikes.complete
