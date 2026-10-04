"""Signal animation: neurons flash on a frontal brain projection as they spike (MP4 for Telegram).

The static parts (backdrop, titles, labels) are drawn once with matplotlib; each frame only
blends glowing dots into that image with numpy, so a 5 s clip renders in about a second.
Positions are FlyWire annotation coordinates (pos_x / pos_y); nothing is invented.
"""
from __future__ import annotations

import os
import tempfile

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np

from . import fly
from .render import BLUE, CLOUD, GOOD, INK, INK_2, MUTED, ORANGE, _stim_title

W, H = 720, 480
FPS = 15
MAX_BYTES = 2_000_000
OUTPUT_KEYS = ("giant_fiber", "mn9", "grooming_dn")   # behaviour outputs shown green
TAU_GLOW_MS = 4.0          # how long a flash fades, in biological time
TRACE_ALPHA = 0.28         # neurons that already fired stay faintly visible: the spreading wave


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r + r


def auto_window(rec, outputs: list[np.ndarray]) -> float:
    """Show the recruitment phase: until 75 % of responding neurons first fired (or an output fired)."""
    sp = rec.result.spikes
    resp = ~np.isin(sp.neuron, rec.stim)
    if not resp.any():
        return 60.0
    _, first = np.unique(sp.neuron[resp], return_index=True)
    w = float(np.percentile(sp.t_ms[resp][first], 75))
    for idx in outputs:
        hit = np.isin(sp.neuron, idx)
        if hit.any():
            w = max(w, float(sp.t_ms[hit].min()) + 15)
    return float(np.clip(np.ceil(w / 5) * 5, 40, 150))


def _verdict_text(rec) -> str:
    b = rec.summary["behavior"]
    yes = lambda k: "ТАК" if b[k]["active"] else "НІ"
    return (f"Що зробила муха:  втеча — {yes('escape')},  хоботок — {yes('feeding')},  "
            f"чистить вусики — {yes('grooming')}")


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def _times_word(n: int) -> str:
    if n % 10 in (2, 3, 4) and not 12 <= n % 100 <= 14:
        return "рази"
    return "раз" if n % 10 == 1 and n % 100 != 11 else "разів"


def _base(svc, rec, window_ms: float, slowdown: int):
    """Static frame and the pixel position of every neuron."""
    n = svc.neurons
    x, y = n["pos_x"].to_numpy(), -n["pos_y"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    stim = ", ".join(_stim_title(svc, it["item"]) for it in rec.summary["stimulated"]["items"])
    fig.text(0.03, 0.935, "Як сигнал біжить мозком мухи", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.03, 0.885, f"Подразник: {stim}", fontsize=9.5, color=INK_2)
    fig.text(0.03, 0.845, f"Перші {window_ms:.0f} мс, сповільнено в {slowdown} {_times_word(slowdown)}",
             fontsize=9.5, color=INK_2)
    ax = fig.add_axes([0.02, 0.13, 0.96, 0.69])
    bg = np.random.default_rng(0).choice(np.flatnonzero(ok), size=min(int(ok.sum()), 45000), replace=False)
    ax.scatter(x[bg], y[bg], s=0.25, c=CLOUD, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    outputs = {}
    for key, name in (("giant_fiber", "Giant Fiber"), ("mn9", "MN9"), ("grooming_dn", "aDN1")):
        idx = svc.catalog.groups[key].idx
        idx = idx[ok[idx]]
        ax.scatter(x[idx], y[idx], s=70, facecolors="none", edgecolors=MUTED, linewidths=1.0)
        i = idx[np.argmin(x[idx])]
        ax.annotate(name, (x[i], y[i]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK)
        outputs[key] = idx
    # legend and time axis labels
    for k, (col, label) in enumerate([(ORANGE, "подразник"), (BLUE, "нейрон спалахнув"), (GOOD, "вихід поведінки")]):
        fig.text(0.05 + k * 0.22, 0.085, "●", color=col, fontsize=11, va="center")
        fig.text(0.075 + k * 0.22, 0.085, label, color=INK_2, fontsize=8.5, va="center")
    fig.text(0.03, 0.022, "0 мс", fontsize=8, color=MUTED)
    fig.text(0.97, 0.022, f"{window_ms:.0f} мс", fontsize=8, color=MUTED, ha="right")
    fig.text(0.5, 0.022, "Модель FlyWire + LIF, не запис живої мухи", fontsize=8, color=MUTED, ha="center")
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255
    # the final frame shows the verdict instead of the legend
    fig.patches.append(plt.Rectangle((0, 0.06), 1, 0.05, transform=fig.transFigure, color="#fcfcfb", zorder=10))
    fig.text(0.03, 0.085, _verdict_text(rec), fontsize=10.5, fontweight="bold", color=INK, va="center", zorder=11)
    fig.canvas.draw()
    final = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255
    px = np.full((len(x), 2), -1, np.int64)
    pts = ax.transData.transform(np.c_[x[ok], y[ok]])
    px[ok, 0] = np.clip(np.round(pts[:, 0]), 0, W - 1)
    px[ok, 1] = np.clip(np.round(H - pts[:, 1]), 0, H - 1)
    plt.close(fig)
    return base, final, px, outputs


BEHAVIOR_OUTPUT = {"escape": "giant_fiber", "feeding": "mn9", "grooming": "grooming_dn"}
AFTER_S_FLY = 2.0           # extra seconds so a take-off / grooming can finish after the brain part


def fly_actions(svc, rec, window_ms: float, seconds: float) -> dict[str, float]:
    """Clip time (s) at which each behaviour with a 'yes' verdict starts: its output's first spike."""
    sp, out = rec.result.spikes, {}
    for key, group in BEHAVIOR_OUTPUT.items():
        if not rec.summary["behavior"][key]["active"]:
            continue
        hit = np.isin(sp.neuron, svc.catalog.groups[group].idx)
        t_ms = float(sp.t_ms[hit].min()) if hit.any() else window_ms
        out[key] = min(t_ms, window_ms) / window_ms * seconds
    return out


def animation_mp4(svc, rec, window_ms: float | None = None, seconds: float = 5.0,
                  with_fly: bool = True) -> bytes:
    spikes = rec.result.spikes
    if spikes is None:
        raise ValueError("this simulation has no recorded spikes; run it with record_spikes=true")
    if window_ms is None:
        window_ms = auto_window(rec, [svc.catalog.groups[k].idx for k in OUTPUT_KEYS])
    n_frames = int(round(FPS * seconds))
    slowdown = int(round(seconds * 1000 / window_ms))
    base, final, px, outputs = _base(svc, rec, window_ms, slowdown)

    sel = spikes.t_ms < window_ms
    neu, t = spikes.neuron[sel], spikes.t_ms[sel]
    keep = px[neu, 0] >= 0
    neu, t = neu[keep], t[keep]
    ids, local = np.unique(neu, return_inverse=True)
    category = np.ones(len(ids), np.int8)                       # 1 = responding neuron
    category[np.isin(ids, rec.stim)] = 0                        # 0 = stimulated
    out_idx = np.concatenate(list(outputs.values()))
    category[np.isin(ids, out_idx)] = 2                         # 2 = behaviour output
    colors = [_hex(ORANGE), _hex(BLUE), _hex(GOOD)]
    cols, rows = px[ids, 0], px[ids, 1]

    frame_dt = window_ms / n_frames
    frame_of = np.minimum((t / frame_dt).astype(np.int64), n_frames - 1)
    decay = float(np.exp(-frame_dt / TAU_GLOW_MS))
    glow = np.zeros(len(ids), np.float32)
    seen = np.zeros(len(ids), bool)
    bar_row = slice(H - 32, H - 27)
    track = _hex("#e1e0d9")
    ink = _hex(INK)

    # Sprites instead of full-frame filters: a solid core plus a gaussian halo, stamped only where
    # neurons glow. The faint trace of neurons that already fired is stamped once into a layer.
    R = 9
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    halo = np.exp(-(xx ** 2 + yy ** 2) / (2 * 3.0 ** 2)).astype(np.float32) * 9 * (1 / (2 * np.pi * 9))
    cores = {c: np.pad(_disk(5 if c == 2 else 2).astype(np.float32), R - (5 if c == 2 else 2)) for c in (0, 1, 2)}
    trace = np.zeros((3, H + 2 * R, W + 2 * R), np.float32)    # padded so stamps never clip

    def stamp(layer, c, r, k, sprite):
        sl = layer[r:r + 2 * R + 1, c:c + 2 * R + 1]
        np.maximum(sl, sprite, out=sl)

    frames = []
    for f in range(n_frames):
        glow *= decay
        hit = local[frame_of == f]
        if len(hit):
            u = np.unique(hit)
            glow[u] = 1.0
            for i in u[~seen[u]]:
                stamp(trace[category[i]], cols[i], rows[i], category[i], cores[category[i]] * TRACE_ALPHA)
            seen[u] = True
        img = (final if f == n_frames - 1 else base).copy()
        lit = np.flatnonzero(glow > TRACE_ALPHA)
        for cat in (1, 0, 2):                                   # outputs drawn last, on top
            layer = trace[cat].copy()
            for i in lit[category[lit] == cat]:
                g = float(glow[i])
                stamp(layer, cols[i], rows[i], cat, np.maximum(cores[cat] * g, np.minimum(halo * g, 1.0)))
            alpha = layer[R:R + H, R:R + W]
            if not alpha.any():
                continue
            img = img * (1 - alpha[..., None]) + colors[cat] * alpha[..., None]
        # progress bar of biological time
        img[bar_row, 22:W - 22] = track
        img[bar_row, 22:22 + int((W - 44) * (f + 1) / n_frames)] = ink
        frames.append((img * 255).astype(np.uint8))
    if not with_fly:
        frames += [frames[-1]] * FPS                           # hold the last frame for a second
        width = W
    else:
        frames += [frames[-1]] * int(AFTER_S_FLY * FPS)
        stim_keys = [it["item"].split(":")[0].lower() for it in rec.summary["stimulated"]["items"]]
        fly_frames = fly.render_frames(fly.scene_for(stim_keys), fly_actions(svc, rec, window_ms, seconds),
                                       len(frames), FPS, t_hit=seconds + 0.8)
        frames = [np.hstack([a, b]) for a, b in zip(fly_frames, frames)]
        width = W + fly.PW

    for crf in (26, 32, 38):
        data = _encode(frames, crf, width)
        if len(data) <= MAX_BYTES:
            return data
    return data


def compare_mp4(svc, base_rec, var_rec, labels: dict, seconds: float = 5.0) -> bytes:
    """Two schematic flies side by side: the normal brain and the changed one, same stimulus and seed.
    Each fly is driven by its own simulation (verdict + first output spike)."""
    if base_rec.result.spikes is None or var_rec.result.spikes is None:
        raise ValueError("both simulations need record_spikes=true")
    outputs = [svc.catalog.groups[k].idx for k in OUTPUT_KEYS]
    window_ms = max(auto_window(base_rec, outputs), auto_window(var_rec, outputs))
    n_frames = int(round(FPS * (seconds + AFTER_S_FLY)))
    panels = []
    for rec, label in ((base_rec, labels["baseline"]), (var_rec, labels["variant"])):
        stim_keys = [it["item"].split(":")[0].lower() for it in rec.summary["stimulated"]["items"]]
        panels.append(fly.render_frames(fly.scene_for(stim_keys), fly_actions(svc, rec, window_ms, seconds),
                                        n_frames, FPS, t_hit=seconds + 0.8, header=label))
    gap = np.full((H, 8, 3), 225, np.uint8)
    frames = [np.hstack([a, gap, b]) for a, b in zip(*panels)]
    width = 2 * fly.PW + 8
    for crf in (26, 32, 38):
        data = _encode(frames, crf, width)
        if len(data) <= MAX_BYTES:
            return data
    return data


def _encode(frames: list[np.ndarray], crf: int, width: int = W) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = imageio_ffmpeg.write_frames(
            path, (width, H), fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=8, ffmpeg_log_level="error",
            output_params=["-crf", str(crf), "-preset", "veryfast", "-movflags", "+faststart", "-an"],
        )
        writer.send(None)
        for fr in frames:
            writer.send(np.ascontiguousarray(fr))
        writer.close()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.remove(path)
