"""
The figures of the blog article, as SVG files whose colours are CSS variables of the website.

The SVGs are meant to be inlined in the page: text and axes use --color-fg, --color-muted, --color-rule,
and the series use --fig-1 (blue), --fig-2 (orange) and --fig-3 (aqua), which the website defines for the light
and the dark theme. Every mark carries a <title>, so hovering it shows its exact value.

    uv run python experiments/article/figures.py [--out experiments/article/figures]

Figures:
- warm_start.svg  PPO from scratch vs PPO started from the hand-written networks (checkpoint evaluations).
- scoreboard.svg  all approaches, one panel per game.
- weights.svg     share of non-zero weights and of integer weights, hand-written vs PPO networks.
"""

import argparse
import json
from html import escape
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.weights import load_weights

ROOT = Path(__file__).resolve().parents[2]
WARM = ROOT / "experiments" / "ppo_warm_start" / "results" / "summary.json"
FONT = "font-family: var(--font-sans, system-ui, sans-serif);"
INK, MUTED, RULE = "var(--color-fg)", "var(--color-muted)", "var(--color-rule)"
BLUE, ORANGE, AQUA = "var(--fig-1)", "var(--fig-2)", "var(--fig-3)"


class Svg:
    def __init__(self, width, height):
        self.w, self.h, self.items = width, height, []

    def text(self, x, y, s, size=11, color=MUTED, anchor="start", weight=400, baseline="auto"):
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="{baseline}" '
            f'style="{FONT} font-size: {size}px; font-weight: {weight}; fill: {color};">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color=RULE, width=1, dash=None):
        d = f" stroke-dasharray: {dash};" if dash else ""
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'style="stroke: {color}; stroke-width: {width};{d}" />')

    def path(self, pts, color, width=2, title=None):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        t = f"<title>{escape(title)}</title>" if title else ""
        self.items.append(f'<path d="{d}" style="fill: none; stroke: {color}; stroke-width: {width}; '
                          f'stroke-linejoin: round; stroke-linecap: round;">{t}</path>')

    def circle(self, x, y, r, color, title=None, ring=True):
        t = f"<title>{escape(title)}</title>" if title else ""
        stroke = " stroke: var(--color-bg); stroke-width: 2;" if ring else ""
        self.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill: {color};{stroke}">{t}</circle>')

    def rect(self, x, y, w, h, color, title=None, radius=2):
        t = f"<title>{escape(title)}</title>" if title else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{radius}" '
                          f'style="fill: {color};">{t}</rect>')

    def save(self, path):
        body = "\n".join(self.items)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}">\n{body}\n</svg>\n')


def nice_ticks(vmax, n=4):
    raw = vmax / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    return [i * step for i in range(int(np.floor(vmax / step)) + 1)]


def fmt(v):
    return f"{v:g}" if v < 1000 else f"{v / 1000:g}k"


# ---------------------------------------------------------------- figure 1: warm start curves

def curve(checkpoints):
    pts = sorted((int(k), v["mean"]) for k, v in checkpoints.items())
    return [p for p in pts]


def warm_start(out):
    data = json.load(open(WARM))
    by = {(c["game"], c["hand"]): c for c in data["comparisons"]}
    panels = [
        ("Breakout", by[("breakout", "paper_astra")], [("ppo_from_astra", BLUE)]),
        ("Seaquest", by[("seaquest", "paper_astra")], [("ppo_from_astra", BLUE)]),
        ("Asterix", by[("asterix", "paper_astra")], [("ppo_from_astra", BLUE)]),
        ("Low Tide, Astra's net", by[("low_tide", "gpt")], [("ppo_from_gpt", BLUE), ("ppo_from_gpt_cw", AQUA)]),
        ("Low Tide, Claude's net", by[("low_tide", "claude")],
         [("ppo_from_claude", BLUE), ("ppo_from_claude_cw", AQUA)]),
    ]
    W, cols, pw, ph = 672, 3, 186, 132
    left, top, gapx, gapy = 34, 40, 32, 58
    H = top + 2 * ph + gapy + 22
    s = Svg(W, H)
    s.text(0, 12, "Mean score per episode (200 episodes per checkpoint) against training steps", 11, MUTED)
    # legend in the empty sixth cell
    lx, ly = left + 2 * (pw + gapx), top + ph + gapy + 4
    for k, (label, color, dash) in enumerate([("PPO from scratch, same shape", ORANGE, None),
                                              ("PPO from the LLM-written net", BLUE, None),
                                              ("same, after 2M steps", AQUA, None),
                                              ("of critic-only warm-up", None, None),
                                              ("LLM-written net", MUTED, "4 3")]):
        y = ly + (0, 20, 40, 54, 78)[k]
        if color:
            s.line(lx, y, lx + 18, y, color, 2, dash)
        s.text(lx + 24, y, label, 11, INK, baseline="middle")
    for i, (title, comp, warms) in enumerate(panels):
        r, c = divmod(i, cols)
        x0, y0 = left + c * (pw + gapx), top + r * (ph + gapy)
        series = [(curve(comp["scratch_checkpoints"]), ORANGE, f"PPO from scratch ({comp['scratch']})")]
        for name, color in warms:
            series.append((curve(comp["warm"][name]["checkpoints"]), color, name))
        hand = comp["hand_score"]["mean"]
        xmax = max(p[0] for ser, _, _ in series for p in ser)
        ymax = max(max(p[1] for ser, _, _ in series for p in ser), hand) * 1.08
        X = lambda v: x0 + v / xmax * pw
        Y = lambda v: y0 + ph - v / ymax * ph
        s.text(x0, y0 - 12, title, 12, INK, weight=600)
        for t in nice_ticks(ymax / 1.08):
            s.line(x0, Y(t), x0 + pw, Y(t), RULE, 1)
            s.text(x0 - 5, Y(t), fmt(t), 10, MUTED, "end", baseline="middle")
        for t in nice_ticks(xmax / 1e6, 4):
            s.text(X(t * 1e6), y0 + ph + 14, f"{t:g}M", 10, MUTED, "middle")
        s.line(x0, y0 + ph, x0 + pw, y0 + ph, MUTED, 1)
        s.line(x0, Y(hand), x0 + pw, Y(hand), MUTED, 1.5, "4 3")
        for pts, color, name in series:
            s.path([(X(a), Y(b)) for a, b in pts], color, 2, f"{name}: final {pts[-1][1]:.1f}")
            for a, b in pts:
                s.circle(X(a), Y(b), 2.2, color, f"{name}, {a / 1e6:g}M steps: {b:.1f}", ring=False)
    s.save(out / "warm_start.svg")


# ---------------------------------------------------------------- figure 2: scoreboard

SCOREBOARD = {
    # game: [(label, score, stderr, colour)], from the READMEs of the experiments
    "Low Tide": [
        ("Code written by Astra", 188.9, 0.7, BLUE),
        ("PPO from scratch, best (100M)", 166.9, 2.7, ORANGE),
        ("BFS heuristic as weights (ours)", 166.7, 2.0, AQUA),
        ("BFS heuristic (ours)", 164.0, 1.5, AQUA),
        ("PPO from Astra's net (100M)", 147.3, 2.9, ORANGE),
        ("Weights written by Claude", 132.3, 0.9, BLUE),
        ("Weights written by Astra", 79.9, 1.0, BLUE),
    ],
    "Breakout": [
        ("Code written by Astra", 786.0, 2.6, BLUE),
        ("Weights written by Astra", 512.5, 7.6, BLUE),
        ("PPO from Astra's net (200M)", 512.5, 7.6, ORANGE),
        ("PPO from scratch, best (200M)", 63.2, 1.0, ORANGE),
        ("PPO, article (500M)", 25.3, None, ORANGE),
    ],
    "Seaquest": [
        ("Code written by Astra", 187.3, 2.8, BLUE),
        ("PPO, article (500M)", 88.9, None, ORANGE),
        ("PPO from Astra's net (200M)", 82.5, 2.1, ORANGE),
        ("PPO from scratch, best (200M)", 55.9, 1.3, ORANGE),
        ("Weights written by Astra", 51.4, 1.4, BLUE),
    ],
    "Asterix": [
        ("Code written by Astra", 83.2, 1.8, BLUE),
        ("PPO, article (500M)", 55.2, None, ORANGE),
        ("Weights written by Astra", 54.8, 1.1, BLUE),
        ("PPO from Astra's net (200M)", 50.8, 1.0, ORANGE),
        ("PPO from scratch, best (200M)", 24.6, 0.5, ORANGE),
    ],
}


def scoreboard(out):
    W, labelw, barw, row, gap = 672, 210, 400, 17, 34
    H = 40 + sum(len(v) * row + gap for v in SCOREBOARD.values())
    s = Svg(W, H)
    lx = 0
    for label, color in [("written by an LLM", BLUE), ("trained with PPO", ORANGE), ("written by hand (ours)", AQUA)]:
        s.rect(lx, 6, 11, 11, color)
        s.text(lx + 16, 12, label, 11, INK, baseline="middle")
        lx += 16 + 6.2 * len(label) + 22
    y = 44
    for game, rows in SCOREBOARD.items():
        vmax = max(v for _, v, _, _ in rows)
        s.text(0, y, game, 12, INK, weight=600)
        y += 10
        for label, v, se, color in rows:
            s.text(labelw - 8, y + row / 2, label, 11, INK, "end", baseline="middle")
            w = v / vmax * barw
            se_txt = f" ± {se}" if se is not None else " (reported)"
            s.rect(labelw, y + 2, w, row - 4, color, f"{game}, {label}: {v}{se_txt}")
            s.text(labelw + w + 5, y + row / 2, f"{v:g}", 10.5, MUTED, baseline="middle")
            y += row
        y += gap - 10
    s.save(out / "scoreboard.svg")


# ---------------------------------------------------------------- figure 3: weights

NETS = [
    ("Low Tide, Claude", "low_tide", "claude", BLUE),
    ("Low Tide, Astra", "low_tide", "gpt", BLUE),
    ("Breakout, Astra", "breakout", "paper_astra", BLUE),
    ("Seaquest, Astra", "seaquest", "paper_astra", BLUE),
    ("Asterix, Astra", "asterix", "paper_astra", BLUE),
    ("Low Tide, PPO (Astra's shape)", "low_tide", "ppo_arch_gpt", ORANGE),
    ("Low Tide, PPO (Claude's shape)", "low_tide", "ppo_arch_claude", ORANGE),
    ("Breakout, PPO", "breakout", "ppo_ent05", ORANGE),
    ("Seaquest, PPO", "seaquest", "ppo_ent05", ORANGE),
    ("Asterix, PPO", "asterix", "ppo_ent05", ORANGE),
]


def weight_stats():
    rows = []
    for label, game, name, color in NETS:
        layers = load_weights(ROOT / "weights" / game / f"{name}.json", get_game(game))
        v = np.concatenate([np.concatenate([W.ravel(), b]) for W, b, _ in layers])
        nz = v[v != 0]
        rows.append((label, color, 100 * nz.size / v.size, 100 * np.mean(nz == np.round(nz)), len(np.unique(nz)),
                     v.size))
    return rows


def weights(out):
    rows = weight_stats()
    W, labelw, pw, gapx, row = 672, 190, 190, 52, 19
    H = 46 + len(rows) * row + 26
    s = Svg(W, H)
    panels = [("Non-zero weights (% of all parameters)", 2, 100), ("Integer weights (% of the non-zero ones)", 3, 100)]
    for p, (title, idx, vmax) in enumerate(panels):
        x0 = labelw + p * (pw + gapx)
        s.text(x0, 12, title, 11.5, INK, weight=600)
        y0 = 30
        for t in (0, 25, 50, 75, 100):
            x = x0 + t / vmax * pw
            s.line(x, y0, x, y0 + len(rows) * row, RULE, 1)
            s.text(x, y0 + len(rows) * row + 14, f"{t}", 10, MUTED, "middle")
        for i, r in enumerate(rows):
            y = y0 + i * row + row / 2 + (8 if i >= 5 else 0)
            if p == 0:
                s.text(labelw - 10, y, r[0], 11, INK, "end", baseline="middle")
            v = r[idx]
            s.circle(x0 + v / vmax * pw, y, 4.5, r[1],
                     f"{r[0]}: {v:.1f}% ({r[4]} distinct non-zero values out of {r[5]} parameters)")
    s.save(out / "weights.svg")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "article" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    warm_start(args.out)
    scoreboard(args.out)
    for r in weights(args.out):
        print(f"{r[0]:32s} non-zero {r[2]:5.1f}%  integers {r[3]:5.1f}%  distinct {r[4]:6d}  params {r[5]}")
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
