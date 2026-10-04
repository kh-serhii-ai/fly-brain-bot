"""Every "break the fly" scenario must produce the behaviour change it promises."""
import pytest
import yaml

from conftest import ROOT, needs_data

pytestmark = needs_data
SCENARIOS = yaml.safe_load((ROOT / "app" / "scenarios.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("key", list(SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scenario_flips_behaviour(client, key, seed):
    exp = SCENARIOS[key]["expect"]
    d = client.post("/compare", json={"scenario": key, "seed": seed}).json()
    b = d["diff"]["behavior"][exp["behavior"]]
    assert b["baseline"] is exp["baseline"] and b["variant"] is exp["variant"], b
    assert b["changed"]
    # a clear, not borderline, difference
    assert b["rate_baseline"] >= 45 and b["rate_variant"] <= 20, b
    # the other behaviour does not appear out of nowhere and nothing runs away
    for other, ob in d["diff"]["behavior"].items():
        if other != exp["behavior"]:
            assert not ob["variant"]
    assert not d["baseline"]["runaway"] and not d["variant"]["runaway"]


def test_scenarios_listing(client):
    keys = [s["key"] for s in client.get("/scenarios").json()]
    assert keys == list(SCENARIOS)


def test_compare_explicit_requests_share_the_seed(client):
    body = {"baseline": {"stimulate": ["sugar_grn"], "seed": 3},
            "variant": {"stimulate": ["sugar_grn"], "silence": ["CB0553"], "seed": 99}}
    d = client.post("/compare", json=body).json()
    assert d["variant"]["params"]["seed"] == 3
    assert d["labels"]["variant"].startswith("Змінена муха (вимкнено:")
    assert any(n["cell_type"] == "CB0701" for n in d["diff"]["lost"])  # MN9 among neurons that went quiet


def test_compare_errors_and_render(client):
    assert client.post("/compare", json={"scenario": "nope"}).status_code == 404
    assert client.post("/compare", json={}).status_code == 400
    png = client.post("/render/compare", json={"scenario": "cut_escape"})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_compare_animation(client):
    r = client.post("/render/compare_animation", json={"scenario": "cut_escape"})
    assert r.status_code == 200 and r.content[4:8] == b"ftyp" and len(r.content) <= 2_000_000
