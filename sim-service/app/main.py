"""sim-service HTTP API."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import animation, render
from .catalog import UnknownTarget
from .service import QUIZ, SCENARIOS, Service

log = logging.getLogger("sim-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Side = Literal["left", "right"]


class SimulateRequest(BaseModel):
    stimulate: list[str] = Field(..., min_length=1, max_length=20,
                                 description="Group keys, cell types or FlyWire root ids. Suffix ':left'/':right' to pick a side.")
    silence: list[str] = Field(default_factory=list, max_length=20)
    side: Side | None = Field(None, description="Restrict stimulated neurons to one side")
    rate_hz: float | None = Field(None, ge=1, le=300, description="Poisson rate; default from the group")
    duration_ms: float = Field(1000, ge=50, le=2000)
    n_trials: int = Field(10, ge=1, le=30)
    seed: int = Field(0, ge=0)
    record_spikes: bool = Field(False, description="Record spike times of the first trial (for /render/animation)")


class PathRequest(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    from_side: Side | None = None
    to_side: Side | None = None
    max_hops: int = Field(6, ge=1, le=10)
    k: int = Field(3, ge=1, le=5)

    def as_dict(self) -> dict:
        return {"from": self.from_, "to": self.to, "from_side": self.from_side,
                "to_side": self.to_side, "max_hops": self.max_hops, "k": self.k}


class RenderActivityRequest(SimulateRequest):
    stimulate: list[str] = Field(default_factory=list, max_length=20)
    sim_id: str | None = Field(None, description="Id from /simulate; otherwise the simulation is (re)run")
    top_n: int = Field(15, ge=5, le=30)


class CompareRequest(BaseModel):
    scenario: str | None = Field(None, description="Key from GET /scenarios; otherwise pass baseline and variant")
    baseline: SimulateRequest | None = None
    variant: SimulateRequest | None = None
    n_trials: int = Field(10, ge=1, le=30)
    seed: int = Field(0, ge=0)

    def resolve(self) -> tuple[dict, dict, dict | None]:
        if self.scenario:
            sc = SCENARIOS.get(self.scenario)
            if sc is None:
                raise HTTPException(404, f"unknown scenario '{self.scenario}', see GET /scenarios")
            # spikes are recorded so /render/compare_animation reuses the cached runs
            common = {"side": None, "duration_ms": 1000, "n_trials": self.n_trials, "seed": self.seed,
                      "record_spikes": True}
            return {**common, **sc["baseline"]}, {**common, **sc["variant"]}, {"key": self.scenario, **sc}
        if not (self.baseline and self.variant):
            raise HTTPException(400, "pass scenario, or both baseline and variant")
        return ({**self.baseline.model_dump(), "record_spikes": True},
                {**self.variant.model_dump(), "record_spikes": True}, None)


svc: Service | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc
    svc = Service()
    log.info("loaded %d neurons in %.1fs", len(svc.neurons), svc.load_seconds)
    yield


app = FastAPI(title="Fly brain sim-service", version="0.1.0", lifespan=lifespan)


@app.exception_handler(UnknownTarget)
async def unknown_target(_: Request, exc: UnknownTarget):
    return JSONResponse(status_code=404, content={
        "error": "unknown_target", "item": exc.item,
        "message": f"Не знайдено нейрон або групу '{exc.item}'.",
        "suggestions": exc.suggestions})


@app.exception_handler(ValueError)
async def bad_value(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "bad_request", "message": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "neurons": len(svc.neurons), "cached_simulations": len(svc.sim_cache)}


@app.get("/groups")
def groups():
    return [g.info(svc.neurons) for g in svc.catalog.groups.values()]


@app.get("/neurons/search")
def search(q: str = Query(..., min_length=1, max_length=64), limit: int = Query(15, ge=1, le=50)):
    return {"query": q, "results": svc.catalog.search(q, limit)}


@app.post("/simulate")
def simulate(req: SimulateRequest):
    return svc.simulate(req.model_dump()).summary


@app.get("/scenarios")
def scenarios():
    return [{"key": k, **{f: v[f] for f in ("title", "button", "baseline_label", "variant_label", "story", "expect")}}
            for k, v in SCENARIOS.items()]


@app.get("/quiz")
def quiz():
    """"Guess what the fly does" scenarios: run `simulate` and compare the guess with `behavior`."""
    return [{"key": k, **v} for k, v in QUIZ.items()]


@app.post("/compare")
def compare(req: CompareRequest):
    return svc.compare(*req.resolve())


@app.post("/render/compare", response_class=Response)
def render_compare(req: CompareRequest):
    res = svc.compare(*req.resolve())
    return Response(render.compare_png(svc, res), media_type="image/png")


@app.post("/render/compare_animation", response_class=Response)
def render_compare_animation(req: CompareRequest):
    res = svc.compare(*req.resolve())
    recs = [svc.sim_cache.get_item(res[k]["sim_id"]) for k in ("baseline", "variant")]
    return Response(animation.compare_mp4(svc, recs[0], recs[1], res["labels"]), media_type="video/mp4")


@app.post("/path")
def path(req: PathRequest):
    return svc.path(req.as_dict())


@app.post("/render/activity", response_class=Response)
def render_activity(req: RenderActivityRequest):
    if req.sim_id:
        rec = svc.sim_cache.get_item(req.sim_id)
        if rec is None:
            raise HTTPException(404, "sim_id not in cache; pass the simulation parameters instead")
    elif req.stimulate:
        rec = svc.simulate(req.model_dump(exclude={"sim_id", "top_n"}))
    else:
        raise HTTPException(400, "pass sim_id or stimulate")
    png = render.activity_png(svc, rec, top_n=req.top_n)
    return Response(png, media_type="image/png")


class RenderAnimationRequest(SimulateRequest):
    stimulate: list[str] = Field(default_factory=list, max_length=20)
    sim_id: str | None = Field(None, description="Id from /simulate run with record_spikes=true")
    window_ms: float | None = Field(None, ge=30, le=500,
                                    description="Biological time shown; default = recruitment phase (40-150 ms)")
    seconds: float = Field(5, ge=2, le=8, description="Length of the brain part of the clip")
    with_fly: bool = Field(True, description="Add the schematic fly driven by the verdict on the left")


@app.post("/render/animation", response_class=Response)
def render_animation(req: RenderAnimationRequest):
    if req.sim_id:
        rec = svc.sim_cache.get_item(req.sim_id)
        if rec is None:
            raise HTTPException(404, "sim_id not in cache; pass the simulation parameters instead")
    elif req.stimulate:
        rec = svc.simulate({**req.model_dump(exclude={"sim_id", "window_ms", "seconds"}), "record_spikes": True})
    else:
        raise HTTPException(400, "pass sim_id or stimulate")
    if rec.result.spikes is None:
        raise HTTPException(409, "this simulation was run without record_spikes=true")
    window = req.window_ms or animation.auto_window(rec, [svc.catalog.groups[k].idx for k in animation.OUTPUT_KEYS])
    mp4 = animation.animation_mp4(svc, rec, window_ms=window, seconds=req.seconds, with_fly=req.with_fly)
    return Response(mp4, media_type="video/mp4",
                    headers={"X-Window-Ms": str(int(window)), "X-Slowdown": str(round(req.seconds * 1000 / window))})


@app.post("/render/path", response_class=Response)
def render_path(req: PathRequest):
    res = svc.path(req.as_dict())
    if not res["found"]:
        raise HTTPException(404, "no path found within max_hops")
    return Response(render.path_png(res), media_type="image/png")
