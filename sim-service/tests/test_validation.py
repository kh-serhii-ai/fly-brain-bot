"""Reproduce Shiu et al. 2024 (Nature 634:210), Fig. 1 & 3, on FlyWire v783.

* Activating labellar sugar GRNs drives MN9 (proboscis extension motor neuron, type CB0701).
* Bitter GRNs alone do not drive MN9, and adding bitter input suppresses the sugar response.

Sugar GRN ids are the paper's list (figures.ipynb, v630); 20 of 21 still exist in v783.
"""
import pytest

from conftest import needs_data

SUGAR_SHIU = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
BITTER_SHIU = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
MN9_SHIU = 720575940660219265


@pytest.fixture(scope="module")
def model():
    from app.data import load_neurons, load_weights
    from app.sim import Connectome, Simulator
    neurons = load_neurons()
    return neurons, Simulator(Connectome.from_weights(load_weights()))


def _idx(neurons, root_ids):
    pos = dict(zip(neurons.root_id, neurons.index))
    return [pos[r] for r in root_ids if r in pos]


@needs_data
def test_sugar_grn_activates_mn9(model):
    neurons, sim = model
    sugar = _idx(neurons, SUGAR_SHIU)
    assert len(sugar) >= 20
    mn9 = _idx(neurons, [MN9_SHIU])[0]
    assert neurons.loc[mn9, "cell_type"] == "CB0701"  # MN9 per FlyWire annotations

    r = sim.run(sugar, rate_hz=200, n_trials=30, seed=0)
    assert not r.truncated
    assert r.rates[mn9] > 40, f"MN9 rate {r.rates[mn9]:.1f} Hz"
    # sparse response: a few hundred neurons, as in the paper ("about 400")
    n_active = int((r.rates > 0).sum())
    assert 200 < n_active < 1000, n_active

    # dose dependence: weaker stimulation -> weaker MN9 response (Fig. 1D)
    r_low = sim.run(sugar, rate_hz=50, n_trials=30, seed=0)
    assert r_low.rates[mn9] < r.rates[mn9]


@needs_data
def test_bitter_does_not_activate_mn9_and_suppresses_sugar(model):
    neurons, sim = model
    sugar, bitter = _idx(neurons, SUGAR_SHIU), _idx(neurons, BITTER_SHIU)
    mn9 = _idx(neurons, [MN9_SHIU])[0]

    r_bitter = sim.run(bitter, rate_hz=200, n_trials=30, seed=0)
    assert r_bitter.rates[mn9] == 0

    r_sugar = sim.run(sugar, rate_hz=100, n_trials=30, seed=0)
    r_both = sim.run(sugar + bitter, rate_hz=100, n_trials=30, seed=0)
    assert r_both.rates[mn9] < 0.5 * r_sugar.rates[mn9], (r_both.rates[mn9], r_sugar.rates[mn9])
