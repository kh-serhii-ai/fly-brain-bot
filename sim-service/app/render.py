"""PNG rendering for Telegram: activity card and path diagram (labels in Ukrainian)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

# palette (dataviz reference instance, light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"       # slot 1: responding neurons / excitatory
ORANGE = "#eb6834"     # slot 2: stimulated / outputs
RED = "#e34948"        # inhibitory (diverging pole)
CLOUD = "#d6d5ce"      # whole-brain backdrop

FOOTER = "Модель (коннектом + спрощені LIF-нейрони), а не запис живої мухи · Дані: FlyWire v783, CC BY-NC 4.0"
SUPER_UK = {
    "optic": "зорова частка", "central": "центральний мозок", "sensory": "сенсорний",
    "visual_projection": "зоровий проєкційний", "ascending": "висхідний", "descending": "низхідний",
    "sensory_ascending": "сенсорний висхідний", "visual_centrifugal": "зоровий відцентровий",
    "motor": "мотонейрон", "endocrine": "ендокринний", "unannotated": "без анотації",
}
NT_UK = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "dopamine": "DA",
         "serotonin": "5-HT", "octopamine": "OA"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


ALIASES = {"mn9": "MN9", "giant_fiber": "Giant Fiber"}


def _label(n: dict) -> str:
    side = {"left": "L", "right": "R"}.get(n.get("side") or "", "")
    name = next((ALIASES[g] for g in n.get("groups", []) if g in ALIASES), n["cell_type"])
    return name + (f" ({side})" if side else "")


def _short(name: str) -> str:
    return name.split(" — ")[0]


def activity_png(svc, rec, top_n: int = 15) -> bytes:
    s = rec.summary
    neurons = svc.neurons
    r = rec.result.rates
    top = s["top_neurons"][:top_n]
    output_ids = {n["root_id"] for k in ("descending", "motor", "mn9", "giant_fiber")
                  for n in s["outputs"][k]["top"]}
    out_keys = {"descending", "motor", "mn9", "giant_fiber"}

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.17, right=0.98, top=0.77, bottom=0.1, wspace=0.12)

    # --- title
    stim_names = ", ".join(_stim_title(svc, it["item"]) for it in s["stimulated"]["items"])
    fig.text(0.02, 0.95, f"Стимуляція: {stim_names}", fontsize=15, fontweight="bold", color=INK)
    sub = (f"{s['params']['rate_hz']:.0f} Гц · {s['simulated_ms']:.0f} мс · {s['params']['n_trials']} прогонів · "
           f"відгукнулось нейронів: {s['n_active']:,}".replace(",", " "))
    if s["silenced"]["n_neurons"]:
        sub += f" · вимкнено: {s['silenced']['n_neurons']:,}".replace(",", " ")
    reached = [svc.catalog.groups[k].name.split(" — ")[0].split(" (")[0]
               for k in ("mn9", "giant_fiber") if s["outputs"][k]["reached"]]
    if reached:
        sub += " · активовано: " + ", ".join(reached)
    fig.text(0.02, 0.905, sub, fontsize=10.5, color=INK_2)
    if s["runaway"]:
        fig.text(0.02, 0.87, "⚠ Лавиноподібна активність: тисячі нейронів палять на максимумі"
                 + (" · прогін обрізано за лімітом часу" if s["truncated"] else ""), fontsize=9.5, color=INK_2)

    # --- left: top responding neurons
    ax = fig.add_subplot(gs[0])
    if top:
        labels = [_label(n) for n in top][::-1]
        vals = [n["rate_hz"] for n in top][::-1]
        is_out = [(n["root_id"] in output_ids) or bool(set(n["groups"]) & out_keys) for n in top][::-1]
        y = np.arange(len(top))
        ax.barh(y, vals, height=0.62, color=[ORANGE if o else BLUE for o in is_out], zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, f"{v:.0f}", va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.15)
        if any(is_out):
            ax.legend(handles=[Patch(color=BLUE, label="інші нейрони"),
                               Patch(color=ORANGE, label="вихід мозку (низхідні / мотонейрони)")],
                      loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
                      labelcolor=INK_2, handlelength=1, borderaxespad=0.2)
    else:
        ax.text(0.5, 0.5, "Жоден нейрон, крім стимульованих,\nне відгукнувся", ha="center",
                va="center", color=INK_2, transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_title(f"Найактивніші нейрони (топ-{len(top)})", loc="left", fontsize=11, color=INK,
                 pad=24 if top else 8)
    ax.set_xlabel("частота спайків, Гц")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # --- right: frontal projection of the brain
    ax2 = fig.add_subplot(gs[1])
    x, yy = neurons["pos_x"].to_numpy(), neurons["pos_y"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(yy)
    rng = np.random.default_rng(0)
    bg = np.flatnonzero(ok)
    bg = rng.choice(bg, size=min(len(bg), 40000), replace=False)
    ax2.scatter(x[bg], -yy[bg], s=0.3, c=CLOUD, linewidths=0, rasterized=True)
    act = np.flatnonzero(ok & (r >= 1.0))
    act = np.setdiff1d(act, rec.stim)
    if len(act):
        act = act[np.argsort(r[act])]
        sizes = 4 + 26 * np.sqrt(r[act] / r[act].max())
        ax2.scatter(x[act], -yy[act], s=sizes, c=BLUE, alpha=0.75, linewidths=0.4,
                    edgecolors=SURFACE, label="відгукнулись (розмір = частота)")
    st = rec.stim[ok[rec.stim]]
    ax2.scatter(x[st], -yy[st], s=22, c=ORANGE, linewidths=0.6, edgecolors=SURFACE, marker="o",
                label="стимульовані")
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Де вони в мозку (фронтальна проєкція)", loc="left", fontsize=11, color=INK, pad=8)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK_2, markerscale=1.2)

    fig.text(0.02, 0.02, FOOTER, fontsize=8, color=MUTED)
    return _png(fig)


def _stim_title(svc, item: str) -> str:
    key = item.split(":")[0].lower()
    side = {"left": " · ліва сторона", "right": " · права сторона"}.get(item.split(":")[-1].lower(), "") if ":" in item else ""
    g = svc.catalog.groups.get(key)
    return (g.name if g else item.split(":")[0]) + side


def path_png(res: dict) -> bytes:
    paths = res["paths"]
    max_len = max(len(p["nodes"]) for p in paths)
    w = max(8.0, 2.3 * max_len + 0.6)
    h = 1.45 + 1.35 * len(paths)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    fig.text(0.3 / w, 1 - 0.38 / h, f"Найсильніші шляхи: {_short(res['from'].get('name', res['from']['item']))} → "
             f"{_short(res['to'].get('name', res['to']['item']))}",
             fontsize=14 if w >= 10 else 12.5, fontweight="bold", color=INK)
    fig.text(0.3 / w, 1 - 0.68 / h,
             "Число на стрілці — кількість синапсів; % — частка всіх входів наступного нейрона",
             fontsize=9.5, color=INK_2)

    bw, bh, gap = 1.75, 0.78, 2.3
    for row, p in enumerate(paths):
        yc = h - 1.55 - row * 1.35
        x0 = 0.3
        for j, n in enumerate(p["nodes"]):
            xc = x0 + j * gap
            ax.add_patch(FancyBboxPatch((xc, yc - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor="#f0efec", edgecolor=AXIS, linewidth=0.8))
            ax.text(xc + bw / 2, yc + 0.13, _label(n), ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=INK)
            meta = SUPER_UK.get(n["super_class"] or "unannotated", n["super_class"] or "")
            nt = NT_UK.get(n["nt"] or "", "")
            ax.text(xc + bw / 2, yc - 0.18, f"{meta}" + (f" · {nt}" if nt else ""), ha="center",
                    va="center", fontsize=7.5, color=INK_2)
            if j < len(p["edges"]):
                e = p["edges"][j]
                col = BLUE if e["sign"] == "excitatory" else RED
                ax.add_patch(FancyArrowPatch((xc + bw + 0.04, yc), (xc + gap - 0.04, yc),
                                             arrowstyle="-|>", mutation_scale=12, color=col, linewidth=2))
                ax.text(xc + bw + (gap - bw) / 2, yc + 0.17, f"{e['synapses']}", ha="center",
                        fontsize=8.5, color=INK)
                ax.text(xc + bw + (gap - bw) / 2, yc - 0.27, _pct(e["input_fraction"]),
                        ha="center", fontsize=7.5, color=INK_2)
    # legend
    ly = 0.42
    for i, (col, txt) in enumerate([(BLUE, "збуджувальний синапс"), (RED, "гальмівний синапс")]):
        lx = 0.3 + i * 2.6
        ax.add_patch(FancyArrowPatch((lx, ly), (lx + 0.45, ly), arrowstyle="-|>", mutation_scale=10,
                                     color=col, linewidth=2))
        ax.text(lx + 0.55, ly, txt, va="center", fontsize=8.5, color=INK_2)
    ax.text(0.3, 0.13, FOOTER, fontsize=7, color=MUTED)
    return _png(fig)


def _pct(x: float) -> str:
    x *= 100
    return f"{x:.0f}%" if x >= 10 else f"{x:.1f}%" if x >= 0.1 else "<0.1%"


GOOD = "#0ca30c"       # status "good": behaviour happened
PILL_OFF = "#e1e0d9"
SILENCED = "#52514e"
BEHAVIOR_ORDER = (("escape", "giant_fiber", "Giant Fiber"), ("feeding", "mn9", "MN9"),
                  ("grooming", "grooming_dn", "aDN1"))


def _brain_map(ax, svc, rates, stim, silenced, outputs):
    """Frontal projection: backdrop, responding neurons, stimulated, silenced, labelled outputs."""
    n = svc.neurons
    x, y = n["pos_x"].to_numpy(), -n["pos_y"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    bg = np.random.default_rng(0).choice(np.flatnonzero(ok), size=min(int(ok.sum()), 40000), replace=False)
    ax.scatter(x[bg], y[bg], s=0.3, c=CLOUD, linewidths=0, rasterized=True)
    act = np.setdiff1d(np.flatnonzero(ok & (rates >= 1.0)), np.concatenate([stim, silenced]))
    if len(act):
        act = act[np.argsort(rates[act])]
        ax.scatter(x[act], y[act], s=4 + 26 * np.sqrt(rates[act] / max(rates[act].max(), 1)), c=BLUE,
                   alpha=0.75, linewidths=0.4, edgecolors=SURFACE)
    sl = silenced[ok[silenced]]
    if len(sl):
        ax.scatter(x[sl], y[sl], s=60 if len(sl) < 50 else 4, c=SILENCED, marker="x", linewidths=1.6)
    st = stim[ok[stim]]
    ax.scatter(x[st], y[st], s=22, c=ORANGE, linewidths=0.6, edgecolors=SURFACE)
    for idx, name, active in outputs:
        idx = idx[ok[idx]]
        ax.scatter(x[idx], y[idx], s=110, facecolors="none", edgecolors=GOOD if active else MUTED,
                   linewidths=1.8 if active else 1.2, zorder=5)
        # label once, beside the left-most neuron of the pair
        i = idx[np.argmin(x[idx])]
        ax.annotate(name, (x[i], y[i]), xytext=(-9, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    ax.set_aspect("equal")
    ax.axis("off")


def _verdict_rows(fig, left, top, width, behavior):
    for k, (key, _, _) in enumerate(BEHAVIOR_ORDER):
        b = behavior[key]
        yy = top - k * 0.045
        fig.text(left, yy, b["label"], fontsize=11, color=INK, va="center")
        pill = "ТАК" if b["active"] else "НІ"
        fig.text(left + width * 0.62, yy, f" {pill} ", fontsize=10.5, fontweight="bold", va="center",
                 color="white" if b["active"] else INK,
                 bbox=dict(boxstyle="round,pad=0.35", fc=GOOD if b["active"] else PILL_OFF, ec="none"))
        fig.text(left + width * 0.76, yy, f"{b['rate_hz']:.0f} Гц", fontsize=9.5, color=INK_2, va="center")


def compare_png(svc, res: dict) -> bytes:
    recs = [svc.sim_cache.get_item(res[k]["sim_id"]) for k in ("baseline", "variant")]
    sc = res.get("scenario") or {}
    base = res["baseline"]
    fig = plt.figure(figsize=(12, 6.2))
    fig.text(0.02, 0.94, sc.get("title", "Порівняння двох мозків"), fontsize=16, fontweight="bold", color=INK)
    stim_names = ", ".join(_stim_title(svc, it["item"]) for it in base["stimulated"]["items"])
    same = base["params"]["stimulate"] == res["variant"]["params"]["stimulate"]
    lead = "Однаковий подразник для обох" if same else "Подразник звичайної мухи"
    fig.text(0.02, 0.895, f"{lead}: {stim_names} · {base['params']['rate_hz']:.0f} Гц · "
                          f"{base['params']['n_trials']} прогонів · однаковий випадковий шум", fontsize=10.5, color=INK_2)

    for col, key in enumerate(("baseline", "variant")):
        left = 0.03 + col * 0.5
        s, rec = res[key], recs[col]
        fig.text(left, 0.815, res["labels"][key], fontsize=13, fontweight="bold", color=INK)
        _verdict_rows(fig, left, 0.75, 0.44, s["behavior"])
        fig.text(left, 0.615, f"Відгукнулось нейронів: {s['n_active']:,}".replace(",", " "), fontsize=10,
                 color=INK_2)
        outputs = [(svc.catalog.groups[g].idx, name, s["behavior"][b]["active"]) for b, g, name in BEHAVIOR_ORDER]
        ax = fig.add_axes([left - 0.01, 0.15, 0.46, 0.46])
        _brain_map(ax, svc, rec.result.rates, rec.stim, rec.silenced, outputs)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.15, 0.84], color=GRID, linewidth=1))

    changed = [b for b in res["diff"]["behavior"].values() if b["changed"]]
    if changed:
        names = {k: BEHAVIORS_UK[k] for k in res["diff"]["behavior"]}
        parts = [f"{names[k]}: {'так → ні' if b['baseline'] else 'ні → так'} "
                 f"({b['rate_baseline']:.0f} → {b['rate_variant']:.0f} Гц)"
                 for k, b in res["diff"]["behavior"].items() if b["changed"]]
        fig.text(0.02, 0.09, "Змінилось — " + "; ".join(parts), fontsize=11, fontweight="bold", color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="стимульовані"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, label="відгукнулись"),
               plt.Line2D([], [], marker="x", ls="", color=SILENCED, markeredgewidth=1.6, label="вимкнені"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=GOOD,
                          markeredgewidth=1.8, label="вихід поведінки (зелений = спрацював)")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.07), ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK_2)
    fig.text(0.02, 0.025, FOOTER, fontsize=8, color=MUTED)
    return _png(fig)


BEHAVIORS_UK = {"escape": "стрибок-втеча", "feeding": "витягує хоботок", "grooming": "чистить вусики"}
