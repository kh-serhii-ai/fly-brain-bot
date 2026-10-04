"""The schematic fly must do exactly what the simulation decided, at the moment it decided."""
import pytest

from conftest import needs_data

from app import fly


def test_scene_props():
    assert fly.scene_for(["lplc2"]).prop == "newspaper"
    assert fly.scene_for(["sugar_grn", "bitter_grn"]).prop == "drop_mixed"
    assert fly.scene_for(["jo_ce"]).prop == "dust"
    assert fly.scene_for(["mn9"]).prop is None


def test_frames_render_and_escape_removes_the_fly():
    frames = fly.render_frames(fly.scene_for(["lplc2"]), {"escape": 0.5}, 30, 15, t_hit=1.5)
    assert len(frames) == 30 and frames[0].shape == (fly.PH, fly.PW, 3)
    # before the action the fly is on the table; long after take-off the table area is empty
    body = (slice(250, 360), slice(120, 300))
    dark = lambda fr: int((fr[body].mean(axis=2) < 110).sum())
    assert dark(frames[0]) > 500
    assert dark(frames[-1]) < dark(frames[0]) / 5


@needs_data
@pytest.mark.parametrize("stim,rate,expected", [
    (["lplc2"], 100, {"escape"}), (["sugar_grn"], 150, {"feeding"}), (["jo_ce"], 220, {"grooming"}),
    (["bitter_grn"], 150, set()),
])
def test_actions_follow_the_verdict(stim, rate, expected):
    from app.animation import fly_actions
    svc = _svc()
    rec = svc.simulate({"stimulate": stim, "silence": [], "side": None, "rate_hz": rate, "duration_ms": 1000,
                        "n_trials": 10, "seed": 0, "record_spikes": True})
    acts = fly_actions(svc, rec, window_ms=100, seconds=5)
    assert set(acts) == expected
    if "escape" in acts:   # Giant Fiber fires within ~10 ms of the looming input
        assert acts["escape"] < 0.6


_SVC = None


def _svc():
    global _SVC
    if _SVC is None:
        from app.service import Service
        _SVC = Service()
    return _SVC


def test_no_escape_means_the_newspaper_lands_on_the_fly():
    frames = fly.render_frames(fly.scene_for(["lc4"]), {}, 30, 15, t_hit=1.0, header="Муха без PVLP122b")
    before, after = frames[0], frames[-1]
    # the standing fly is gone; a flattened body is left at table level
    standing = (slice(250, 340), slice(120, 300))
    assert (after[standing].mean(axis=2) < 110).sum() < (before[standing].mean(axis=2) < 110).sum() / 5
