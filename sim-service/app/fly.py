"""Schematic side-view fly, driven by the simulation.

The model has no body, muscles or physics, so the fly is a drawing. What it does is decided by the
simulation: an action plays only if the code verdict for it is "yes" (behaviors.yaml), and it
starts at the first spike of that behaviour's output neuron in the animated trial:
  escape   -> Giant Fiber (DNp01): take-off jump, then flight away
  feeding  -> MN9: proboscis extends to the drop
  grooming -> aDN1 (DNg62): front legs clean the antenna, then rub each other
The props (newspaper, drop, dust) only illustrate which sensory neurons were stimulated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 2                       # supersampling factor for smooth edges
PW, PH = 480, 480           # panel size
TABLE_Y = 372
ANCHOR = (205.0, 300.0)     # thorax centre of the standing fly
F = 1.4                     # drawing scale of the fly

FONT_PATH = f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans.ttf"
FONT_BOLD_PATH = f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf"

BG = (252, 252, 251)
INK = (11, 11, 11)
INK_2 = (82, 81, 78)
MUTED = (137, 135, 129)
GOOD = (12, 163, 12)
BODY = (92, 74, 58)
BODY_DARK = (60, 48, 38)
LEG = (52, 42, 34)
LEG_FAR = (150, 138, 126)
EYE = (178, 44, 40)
WING = (196, 222, 240, 120)
WING_EDGE = (120, 150, 175, 200)

ACTION_TEXT = {
    "escape": "Giant Fiber спрацював → злітає!",
    "feeding": "MN9 спрацював → п'є краплю",
    "grooming": "aDN1 спрацював → чистить вусик",
}
SWATTED_TEXT = "Giant Fiber не спрацював → не встигла злетіти"
RED = (208, 59, 59)


@dataclass
class Scene:
    prop: str | None          # newspaper | drop_sugar | drop_bitter | drop_mixed | drop_water | dust | None
    title: str


def scene_for(stim_keys: list[str]) -> Scene:
    k = set(stim_keys)
    if k & {"lplc2", "lc4", "giant_fiber"}:
        return Scene("newspaper", "На муху летить згорнута газета")
    if "sugar_grn" in k and "bitter_grn" in k:
        return Scene("drop_mixed", "Крапля: солодке з гірким")
    if "sugar_grn" in k:
        return Scene("drop_sugar", "Крапля солодкого біля хоботка")
    if "bitter_grn" in k:
        return Scene("drop_bitter", "Гірка крапля біля хоботка")
    if "water_grn" in k:
        return Scene("drop_water", "Крапля води біля хоботка")
    if k & {"jo_ce", "jo_f"}:
        return Scene("dust", "На вусик сіла порошинка")
    return Scene(None, "Муха на столі")


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _rot(points, center, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in points]


def _ellipse(cx, cy, rx, ry, angle=0.0, hinge=None, n=40):
    pts = [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return _rot(pts, hinge or (cx, cy), angle) if angle else pts


class FlyAnimator:
    def __init__(self, scene: Scene, actions: dict[str, float], header: str | None = None):
        """actions: behaviour -> clip time (s) when it starts (only behaviours with a 'yes' verdict).
        header: panel title (e.g. "Звичайна муха"); the scene title then becomes the subtitle."""
        self.scene, self.header = scene, header
        # one visible action: escape wins (it removes the fly), then grooming, then feeding
        self.action, self.t_action = None, None
        for key in ("escape", "grooming", "feeding"):
            if key in actions:
                self.action, self.t_action = key, actions[key]
                break
        self.font_title = ImageFont.truetype(FONT_BOLD_PATH, 15 * S)
        self.font = ImageFont.truetype(FONT_PATH, 11 * S)
        self.font_small = ImageFont.truetype(FONT_PATH, 9 * S)
        self.font_paper = ImageFont.truetype(FONT_BOLD_PATH, 9 * S)

    # ------------------------------------------------------------------ geometry helpers
    def _p(self, x, y, ox=0.0, oy=0.0):
        """Fly coordinates (unscaled drawing units) + screen offset -> supersampled pixels."""
        return ((ANCHOR[0] + F * x + ox) * S, (ANCHOR[1] + F * y + oy) * S)

    def frame(self, t: float, t_hit: float) -> np.ndarray:
        img = Image.new("RGB", (PW * S, PH * S), BG)
        d = self._draw = ImageDraw.Draw(img, "RGBA")
        # table
        d.rectangle([0, TABLE_Y * S, PW * S, PH * S], fill=(238, 233, 224))
        d.line([0, TABLE_Y * S, PW * S, TABLE_Y * S], fill=(196, 188, 176), width=2 * S)

        a = (t - self.t_action) if self.action and t >= self.t_action else None
        fly_hidden = False
        ox = oy = 0.0
        wings_open = 0.0
        flap = 0.0
        legs = "stand"
        proboscis = 0.0
        groom = None
        if self.action == "escape" and a is not None:
            rise = _ease(a / 0.25) * 22                     # legs push the body up
            fly_t = max(0.0, a - 0.2)
            ox, oy = -fly_t * 190 - (fly_t ** 2) * 60, -rise - fly_t * 150
            wings_open = _ease(a / 0.15)
            flap = math.sin(a * 2 * math.pi * 7)          # visible beat (real flies: ~200 Hz)
            legs = "push" if a < 0.25 else "dangle"
            fly_hidden = ANCHOR[1] + oy < -80
        elif self.action == "feeding" and a is not None:
            proboscis = _ease(a / 0.5) * (1 + 0.06 * math.sin(a * 2 * math.pi * 2.5))
        elif self.action == "grooming" and a is not None:
            groom = a

        self._draw_prop_back(d, t, t_hit, groom)
        if self._squashed(t, t_hit):
            self._draw_squashed(d)
        elif not fly_hidden:
            self._draw_fly(d, ox, oy, wings_open, flap, legs, proboscis, groom)
        self._draw_prop_front(d, t, t_hit)
        self._draw_text(d, t, a, t_hit)
        return np.asarray(img.resize((PW, PH), Image.LANCZOS))

    def _squashed(self, t, t_hit):
        return self.scene.prop == "newspaper" and self.action != "escape" and t >= t_hit

    def _draw_squashed(self, d):
        """The fly flattened on the table under the newspaper (no escape command came in time)."""
        P = lambda x, y: self._p(x, y)
        flat = (TABLE_Y - ANCHOR[1]) / F - 6
        for x0, x1 in ((-10, -46), (6, 22), (24, 58), (-30, -70)):        # splayed legs
            d.line([P(x0, flat), P(x1, flat + 4)], fill=LEG, width=int(2.5 * S * F))
        d.polygon([P(x, y) for x, y in _ellipse(-40, flat - 4, 44, 5)], fill=WING[:3] + (150,), outline=WING_EDGE)
        d.ellipse([*P(-90, flat - 7), *P(46, flat + 3)], fill=BODY_DARK)
        d.ellipse([*P(30, flat - 9), *P(56, flat + 3)], fill=EYE)

    # ------------------------------------------------------------------ the fly
    def _draw_fly(self, d, ox, oy, wings_open, flap, legs, proboscis, groom):
        P = lambda x, y: self._p(x, y, ox, oy)
        foot_y = (TABLE_Y - ANCHOR[1] - oy) / F         # table level in fly coordinates

        def leg(hip, knee, foot, color, w):
            d.line([P(*hip), P(*knee), P(*foot)], fill=color, width=int(w * S * F), joint="curve")

        def leg_set(color, w, shift):
            hips = [(20, 16), (2, 22), (-16, 18)]
            if legs == "stand":
                knees, feet = [(42, 34), (6, 42), (-36, 36)], [(54, foot_y), (2, foot_y), (-46, foot_y)]
            elif legs == "push":
                knees, feet = [(44, 38), (8, 48), (-34, 44)], [(56, foot_y), (4, foot_y), (-44, foot_y)]
            else:                                        # dangling in flight
                knees, feet = [(30, 36), (-2, 40), (-30, 34)], [(40, 56), (-12, 60), (-48, 52)]
            for k in range(3):
                if groom is not None and k == 0:
                    continue                             # front legs are drawn by the grooming pose
                h, kn, f = hips[k], knees[k], feet[k]
                leg((h[0] + shift, h[1]), (kn[0] + shift, kn[1]), (f[0] + shift, f[1]), color, w)

        leg_set(LEG_FAR, 2.2, -6)                        # far-side legs
        # wings behind the body when folded
        hinge = (-2, -22)
        for k, alpha in ((0, 1.0),) if wings_open == 0 else ((-1, 0.35), (0, 1.0), (1, 0.35)):
            ang = 175 - wings_open * (95 + 35 * (flap + 0.4 * k))
            pts = _ellipse(-54, -22, 54, 13, ang - 180, hinge=hinge)
            col = WING[:3] + (int(WING[3] * alpha),)
            d.polygon([P(x, y) for x, y in pts], fill=col, outline=WING_EDGE)
        # abdomen with stripes
        d.ellipse([*P(-104, -20), *P(-12, 30)], fill=BODY)
        for sx in (-90, -74, -58, -42):
            d.line([P(sx, -16), P(sx + 4, 26)], fill=BODY_DARK, width=int(3 * S * F))
        # thorax and head
        d.ellipse([*P(-34, -30), *P(32, 26)], fill=BODY)
        d.ellipse([*P(-30, -30), *P(26, -6)], fill=(120, 98, 78))
        d.ellipse([*P(22, -28), *P(62, 12)], fill=BODY_DARK)
        d.ellipse([*P(30, -26), *P(60, 4)], fill=EYE)
        d.ellipse([*P(38, -20), *P(46, -12)], fill=(236, 140, 130))
        # antenna with arista
        d.line([P(54, -24), P(62, -36)], fill=LEG, width=int(3 * S * F))
        d.line([P(62, -36), P(76, -42)], fill=LEG, width=int(1 * S * F))
        # proboscis: retracted stub -> extended to the drop
        drop_y = (TABLE_Y - 12 - ANCHOR[1] - oy) / F
        tip = (50 + 36 * proboscis, 14 + (drop_y - 14) * proboscis)
        d.line([P(48, 10), P(*tip)], fill=BODY_DARK, width=int(5 * S * F))
        if proboscis > 0.9:
            d.ellipse([*P(tip[0] - 5, tip[1] - 3), *P(tip[0] + 5, tip[1] + 4)], fill=BODY_DARK)
        leg_set(LEG, 3.2, 0)                             # near-side legs
        if groom is not None:
            self._draw_grooming_legs(P, groom, foot_y)

    def _draw_grooming_legs(self, P, a, foot_y):
        rise = _ease(a / 0.35)
        # front legs move from the table to the antenna and sweep along it
        for shift, color, w, phase in ((-6, LEG_FAR, 2.2, math.pi), (0, LEG, 3.2, 0.0)):
            r = math.sin(a * 2 * math.pi * 3.0 + phase)
            foot = (54 + (4 + 6 * r) * rise, foot_y + (-34 - foot_y - 6 * r) * rise)
            knee = (42 + 22 * rise, 34 - 36 * rise)
            d_hip = (20 + shift, 16)
            pts = [d_hip, (knee[0] + shift, knee[1]), (foot[0] + shift, foot[1])]
            self._draw.line([P(*q) for q in pts], fill=color, width=int(w * S * F), joint="curve")

    # ------------------------------------------------------------------ props
    def _drop_color(self):
        return {"drop_sugar": (255, 236, 244), "drop_bitter": (196, 228, 170), "drop_mixed": (232, 236, 200),
                "drop_water": (190, 220, 245)}.get(self.scene.prop)

    def _draw_prop_back(self, d, t, t_hit, groom):
        prop = self.scene.prop
        if prop and prop.startswith("drop"):
            cx, cy = ANCHOR[0] + 92 * F, TABLE_Y - 9
            shrink = 1.0
            if self.action == "feeding" and self.t_action is not None and t > self.t_action + 0.5:
                shrink = max(0.55, 1 - (t - self.t_action - 0.5) * 0.15)
            rx, ry = 16 * shrink * F, 10 * shrink * F
            d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=self._drop_color(),
                      outline=(170, 170, 170))
            d.ellipse([(cx - rx * 0.4) * S, (cy - ry * 0.6) * S, (cx - rx * 0.1) * S, (cy - ry * 0.2) * S],
                      fill=(255, 255, 255))
        if prop == "dust":
            x, y = ANCHOR[0] + 70 * F, ANCHOR[1] - 40 * F
            if groom is not None and groom > 0.9:            # brushed off: falls to the table
                fall = min(1.0, (groom - 0.9) / 0.6)
                x, y = x + 12 * fall, y + (TABLE_Y - 4 - y) * _ease(fall)
            d.ellipse([(x - 4) * S, (y - 4) * S, (x + 4) * S, (y + 4) * S], fill=(150, 145, 135))
            d.ellipse([(x - 2) * S, (y + 3) * S, (x + 1) * S, (y + 6) * S], fill=(170, 165, 155))

    def _draw_prop_front(self, d, t, t_hit):
        if self.scene.prop != "newspaper":
            return
        # a rolled newspaper swinging down onto the spot where the fly sits
        k = _ease(min(t / t_hit, 1.0)) ** 1.6
        cx = 430 + (ANCHOR[0] + 5 - 430) * k
        cy = -40 + (TABLE_Y - 14 - (-40)) * k
        ang = -58 + 50 * k
        L, R = 150, 15
        body = [(cx - L, cy - R), (cx + L, cy - R), (cx + L, cy + R), (cx - L, cy + R)]
        d.polygon([(x * S, y * S) for x, y in _rot(body, (cx, cy), ang)], fill=(236, 232, 220), outline=(120, 116, 106))
        for k2 in range(-3, 4):
            if k2 == 0:
                continue
            seg = [(cx - L + 20, cy + k2 * 4), (cx + L - 20, cy + k2 * 4)]
            d.line([(x * S, y * S) for x, y in _rot(seg, (cx, cy), ang)], fill=(180, 176, 166), width=S)
        end = _rot([(cx + L, cy)], (cx, cy), ang)[0]
        d.ellipse([(end[0] - 9) * S, (end[1] - R) * S, (end[0] + 9) * S, (end[1] + R) * S], fill=(214, 210, 198))
        lab = _rot([(cx - 30, cy - 6)], (cx, cy), ang)[0]
        d.text((lab[0] * S, lab[1] * S), "НОВИНИ", font=self.font_paper, fill=(90, 86, 78))
        if t >= t_hit and t < t_hit + 0.5:
            d.text(((ANCHOR[0] + 60) * S, (TABLE_Y - 70) * S), "ЛЯП!", font=self.font_title, fill=INK)

    # ------------------------------------------------------------------ text
    def _draw_text(self, d, t, a, t_hit):
        y = 12
        if self.header:
            d.text((14 * S, y * S), self.header, font=self.font_title, fill=INK)
            d.text((14 * S, (y + 22) * S), self.scene.title, font=self.font, fill=INK_2)
            y += 42
        else:
            d.text((14 * S, y * S), self.scene.title, font=self.font_title, fill=INK)
            y += 24
        if a is not None:
            d.text((14 * S, y * S), ACTION_TEXT[self.action], font=self.font, fill=GOOD)
        elif self._squashed(t, t_hit):
            d.text((14 * S, y * S), SWATTED_TEXT, font=self.font, fill=RED)
        elif self.action is None and t > 1.0:
            d.text((14 * S, y * S), "Жодна з поведінок не спрацювала", font=self.font, fill=INK_2)
        d.text((14 * S, (PH - 40) * S), "Схематична анімація: тіла в моделі немає.", font=self.font_small, fill=MUTED)
        d.text((14 * S, (PH - 26) * S), "Що робити, вирішила симуляція мозку →", font=self.font_small, fill=MUTED)


def render_frames(scene: Scene, actions: dict[str, float], n_frames: int, fps: int, t_hit: float,
                  header: str | None = None) -> list[np.ndarray]:
    fa = FlyAnimator(scene, actions, header)
    return [fa.frame(f / fps, t_hit) for f in range(n_frames)]
