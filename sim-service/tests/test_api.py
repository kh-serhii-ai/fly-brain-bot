import pytest

from conftest import needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health_and_groups(client):
    assert client.get("/health").json()["neurons"] > 138_000
    groups = {g["key"]: g for g in client.get("/groups").json()}
    for key in ["sugar_grn", "bitter_grn", "water_grn", "orn_dm1", "lc4", "lplc2",
                "giant_fiber", "mn9", "descending", "gaba"]:
        assert groups[key]["n_neurons"] > 0, key
    assert groups["giant_fiber"]["n_neurons"] == 2
    assert groups["mn9"]["n_neurons"] == 2


def test_search(client):
    res = client.get("/neurons/search", params={"q": "giant fiber"}).json()["results"]
    assert res[0]["name"] == "giant_fiber"
    res = client.get("/neurons/search", params={"q": "DNp01"}).json()["results"]
    assert any(r["name"] == "DNp01" and r["n_neurons"] == 2 for r in res)


def test_simulate_sugar_reaches_mn9(client):
    r = client.post("/simulate", json={"stimulate": ["sugar_grn:left"], "n_trials": 5})
    assert r.status_code == 200
    d = r.json()
    assert d["outputs"]["mn9"]["reached"]
    assert not d["outputs"]["giant_fiber"]["reached"]
    assert 0 < len(d["top_neurons"]) <= 20
    assert all(isinstance(n["root_id"], str) for n in d["top_neurons"])  # 64-bit ids as strings
    # cached second call returns the same id
    assert client.post("/simulate", json={"stimulate": ["sugar_grn:left"], "n_trials": 5}).json()["sim_id"] == d["sim_id"]


def test_simulate_looming_reaches_giant_fiber(client):
    d = client.post("/simulate", json={"stimulate": ["lplc2"], "n_trials": 5}).json()
    assert d["outputs"]["giant_fiber"]["reached"]


def test_unknown_target_is_404_with_suggestions(client):
    r = client.post("/simulate", json={"stimulate": ["LC"]})
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_target"


def test_path(client):
    d = client.post("/path", json={"from": "lplc2", "to": "giant_fiber", "max_hops": 3}).json()
    assert d["found"] and d["paths"][0]["hops"] == 1
    assert d["paths"][0]["nodes"][-1]["cell_type"] == "DNp01"


def test_render(client):
    sim_id = client.post("/simulate", json={"stimulate": ["bitter_grn"], "n_trials": 3}).json()["sim_id"]
    png = client.post("/render/activity", json={"sim_id": sim_id})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    png = client.post("/render/path", json={"from": "sugar_grn", "to": "mn9"})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_animation(client):
    import time
    sim_id = client.post("/simulate", json={"stimulate": ["lplc2"], "record_spikes": True}).json()["sim_id"]
    t = time.time()
    r = client.post("/render/animation", json={"sim_id": sim_id})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.content[4:8] == b"ftyp"                       # an MP4 container
    assert len(r.content) <= 2_000_000
    assert time.time() - t < 10
    assert 40 <= int(r.headers["x-window-ms"]) <= 150
    # a simulation without recorded spikes cannot be animated
    plain = client.post("/simulate", json={"stimulate": ["lplc2"]}).json()["sim_id"]
    assert client.post("/render/animation", json={"sim_id": plain}).status_code == 409
