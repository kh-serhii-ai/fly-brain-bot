"""Behaviour verdicts ("what did the fly do") must match the validated cases."""
import pytest

from conftest import needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def svc():
    from app.service import Service
    return Service()


def run(svc, stimulate, silence=(), rate_hz=None):
    req = {"stimulate": list(stimulate), "silence": list(silence), "side": None, "rate_hz": rate_hz,
           "duration_ms": 1000, "n_trials": 10, "seed": 0}
    return svc.simulate(req).summary["behavior"]


def test_sugar_100hz_feeds_and_does_not_escape(svc):
    b = run(svc, ["sugar_grn:left"], rate_hz=100)
    assert b["feeding"]["active"] and not b["escape"]["active"]
    assert b["feeding"]["rate_hz"] >= 49


def test_sugar_25hz_does_not_feed(svc):
    b = run(svc, ["sugar_grn:left"], rate_hz=25)
    assert not b["feeding"]["active"] and not b["feeding"]["weak"]


def test_bitter_does_not_feed(svc):
    b = run(svc, ["bitter_grn"])
    assert not b["feeding"]["active"] and not b["escape"]["active"]


def test_bitter_cancels_sugar(svc):
    assert run(svc, ["sugar_grn:left"], rate_hz=150)["feeding"]["active"]
    assert not run(svc, ["sugar_grn:left", "bitter_grn:left"], rate_hz=150)["feeding"]["active"]


@pytest.mark.parametrize("group", ["lplc2", "lc4"])
def test_looming_escapes_and_does_not_feed(svc, group):
    b = run(svc, [group])
    assert b["escape"]["active"] and not b["feeding"]["active"]
    assert not b["escape"]["stimulated_directly"]


def test_direct_stimulation_is_flagged(svc):
    b = run(svc, ["giant_fiber"])
    assert b["escape"]["stimulated_directly"]


def test_counts(svc):
    b = run(svc, ["lplc2"])
    assert b["total_neurons"] == 138_639
    assert 300 < b["responding_neurons"] < 1000
    assert b["runaway"] is False


def test_strong_but_normal_response_is_not_runaway(svc):
    # LPLC2+LC4 together can hit the time budget (truncated) without being seizure-like
    b = run(svc, ["lplc2", "lc4"])
    assert b["escape"]["active"] and b["runaway"] is False


def test_gaba_silencing_is_runaway(svc):
    b = run(svc, ["sugar_grn:left"], silence=["gaba"])
    assert b["runaway"] is True


def test_jo_ce_triggers_grooming_jo_f_does_not(svc):
    b = run(svc, ["jo_ce"], rate_hz=220)
    assert b["grooming"]["active"] and not b["escape"]["active"] and not b["feeding"]["active"]
    assert b["grooming"]["rate_hz"] >= 45
    assert run(svc, ["jo_f"], rate_hz=220)["grooming"]["rate_hz"] < 1


@pytest.mark.parametrize("stim", [["sugar_grn"], ["lplc2"], ["lc4"], ["bitter_grn"], ["water_grn"]])
def test_no_grooming_from_other_inputs(svc, stim):
    assert run(svc, stim)["grooming"]["rate_hz"] < 1
