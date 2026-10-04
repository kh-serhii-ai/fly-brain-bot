"""Every "guess what the fly does" scenario must give the outcome its explanation describes."""
import pytest
import yaml

from conftest import ROOT, needs_data

pytestmark = needs_data
QUIZ = yaml.safe_load((ROOT / "app" / "quiz.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("key", list(QUIZ))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quiz_outcome(client, key, seed):
    q = QUIZ[key]
    b = client.post("/simulate", json={**q["simulate"], "seed": seed}).json()["behavior"]
    for behaviour, expected in q["expect"].items():
        assert b[behaviour]["active"] is expected, (key, behaviour, b[behaviour])
        # no borderline answers in a game: a "no" must not be a weak near-threshold signal
        if not expected:
            assert b[behaviour]["rate_hz"] < 15, (key, behaviour, b[behaviour]["rate_hz"])
    assert not b["runaway"]


def test_quiz_listing(client):
    items = client.get("/quiz").json()
    assert [i["key"] for i in items] == list(QUIZ)
    assert all(i["prompt"] and i["explain"] and i["button"] for i in items)
