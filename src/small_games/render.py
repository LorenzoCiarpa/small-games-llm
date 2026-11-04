"""
Pygame drawing shared by every game.

`Renderer` draws the grid area of one game and can add a block of its own to
the side panel. A game gets the generic `GridRenderer` (one shape per
observation channel) unless it names a custom renderer in `Game.renderer`.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

CELL = 60
PANEL_W = 330
MIN_H = 600

PANEL_BG = (16, 44, 58)
PANEL_LINE = (34, 74, 90)
TEXT = (232, 240, 236)
MUTED = (138, 168, 176)
DANGER = (214, 74, 58)
GOOD = (242, 200, 90)
STICKY = (240, 150, 40)
BAR_TRACK = (28, 62, 78)
BAR_FILL = (116, 196, 196)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def darker(c, k=0.9):
    return tuple(int(v * k) for v in c)


class Fonts:
    def __init__(self):
        self.title = pygame.font.Font(None, 48)
        self.score = pygame.font.Font(None, 72)
        self.text = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 21)
        self.tiny = pygame.font.Font(None, 16)


class Panel:
    """Writes into the side panel; `x` is the left edge of the content, `w` its width."""

    def __init__(self, surface, fonts, x, w):
        self.s, self.fonts, self.x, self.w = surface, fonts, x, w

    def text(self, font, txt, color, dx, y):
        img = font.render(txt, True, color)
        self.s.blit(img, (self.x + dx, y))
        return img.get_height()

    def line(self, y):
        pygame.draw.line(self.s, PANEL_LINE, (self.x, y), (self.x + self.w, y), 1)

    def fit(self, font, small_font, txt, width):
        """`font`, or `small_font` if the text does not fit in `width` pixels."""
        return font if font.size(txt)[0] <= width else small_font

    def wrap(self, font, items, sep=", "):
        """Greedily join `items` with `sep` into lines that fit the panel width."""
        lines, cur = [], ""
        for item in items:
            cand = f"{cur}{sep}{item}" if cur else item
            if cur and font.size(cand)[0] > self.w:
                lines.append(cur)
                cur = item
            else:
                cur = cand
        return lines + ([cur] if cur else [])

    def pips(self, dx, y, n_on, n_total, color_on, color_off=PANEL_LINE, size=9, gap=6):
        for i in range(n_total):
            pygame.draw.circle(self.s, color_on if i < n_on else color_off,
                               (self.x + dx + size + i * (2 * size + gap), y + size), size)


class Renderer:
    """Base class: reward floaters, coordinates and the default panel texts."""

    def __init__(self, game, env, fonts):
        self.game, self.env, self.fonts = game, env, fonts
        self.obs = None
        self.floaters = []
        self.show_coords = False

    # ------------------------------------------------------------ to override
    @property
    def size(self):
        """(width, height) in pixels of the grid area."""
        raise NotImplementedError

    def draw(self, surf, t, last, done):
        """Draw the grid area on `surf` (its own origin is the top-left corner)."""
        raise NotImplementedError

    def anchor_cell(self):
        """(row, col) where reward floaters appear, or None."""
        return None

    def before_step(self):
        pass

    def panel_section(self, panel, y):
        """Game-specific block of the side panel; returns the next free y."""
        return y

    def score_details(self):
        return []

    def step_message(self, last):
        """(text, color) describing the last step, or None for the default message."""
        return None

    def end_title(self, terminated):
        return f"Partita finita al passo {self.env.t}" if terminated else f"Fine dei {self.env.t} passi"

    # ------------------------------------------------------------ shared
    def reset(self, obs):
        self.obs = obs
        self.floaters.clear()

    def after_step(self, obs, last):
        self.obs = obs
        r = last["reward"]
        cell = self.anchor_cell()
        if r != 0 and cell is not None:
            text = f"+{r:g}" if r > 0 else f"{r:g}"
            self.floaters.append((text, GOOD if r > 0 else DANGER, cell[0], cell[1],
                                  pygame.time.get_ticks() / 1000))

    def draw_floaters(self, surf, cell=CELL):
        now = pygame.time.get_ticks() / 1000
        keep = []
        for text, color, r, c, t0 in self.floaters:
            age = now - t0
            if age < 0.8:
                img = self.fonts.text.render(text, True, darker(color, 0.75))
                img.set_alpha(int(255 * (1 - age / 0.8)))
                surf.blit(img, (c * cell + cell // 2 - img.get_width() // 2, r * cell - 4 - 26 * age))
                keep.append((text, color, r, c, t0))
        self.floaters = keep

    def draw_coords(self, surf, rows, cols, color, cell=CELL):
        if self.show_coords:
            for r in range(rows):
                for c in range(cols):
                    surf.blit(self.fonts.tiny.render(f"{r},{c}", True, color), (c * cell + 3, r * cell + 2))


# ---------------------------------------------------------------- generic grid
GRID_BG = (18, 24, 34)
GRID_LINE = (30, 39, 52)
# role -> (color, shape); channels with other names get a color from EXTRA
ROLE_STYLE = {
    "gauge": ((72, 168, 108), "bar"),
    "reward": ((238, 190, 70), "square"),
    "danger": ((226, 70, 58), "circle"),
    "projectile": ((245, 245, 245), "small"),
    "player": ((84, 204, 232), "player"),
    "trail": ((120, 136, 160), "dot"),
}
DRAW_ORDER = ("gauge", "reward", "danger", "projectile", "player", "trail")
EXTRA = [(196, 120, 230), (120, 220, 140), (240, 140, 180), (150, 150, 250)]


class GridRenderer(Renderer):
    """Any game whose observation is a grid of binary channels (Game.grid)."""

    def __init__(self, game, env, fonts):
        super().__init__(game, env, fonts)
        self.spec = game.grid
        if self.spec is None:
            raise ValueError(f"{game.name}: no Game.grid and no custom renderer")
        self.cell = max(24, min(80, 660 // max(self.spec.height, self.spec.width)))
        names = list(self.spec.channels)
        order = [n for n in DRAW_ORDER if n in names] + [n for n in names if n not in DRAW_ORDER]
        # trail is drawn last, as a dot on top: it shows which way things move
        self.order = [(names.index(n), *self.style(n)) for n in order]

    def style(self, name):
        if name in ROLE_STYLE:
            return ROLE_STYLE[name]
        return EXTRA[self.spec.channels.index(name) % len(EXTRA)], "square"

    @property
    def size(self):
        return self.spec.width * self.cell, self.spec.height * self.cell

    def planes(self):
        return self.spec.planes(self.obs)

    def anchor_cell(self):
        if "player" not in self.spec.channels:
            return None
        cells = np.argwhere(self.planes()[self.spec.channels.index("player")])
        return tuple(int(v) for v in cells[0]) if len(cells) else None

    def shape(self, surf, kind, color, rect):
        """Draw one channel's mark in the cell `rect` (sizes scale with the cell)."""
        cx, cy = rect.center
        u = rect.w / 60
        if kind == "bar":
            pygame.draw.rect(surf, color, (rect.x + 2, rect.bottom - round(22 * u), rect.w - 4, round(18 * u)),
                             border_radius=4)
        elif kind == "square":
            pygame.draw.rect(surf, color, rect.inflate(-round(14 * u), -round(14 * u)), border_radius=round(7 * u))
        elif kind == "circle":
            pygame.draw.circle(surf, color, (cx, cy), round(13 * u))
            pygame.draw.circle(surf, darker(color, 0.6), (cx, cy), round(13 * u), 2)
        elif kind == "small":
            pygame.draw.circle(surf, color, (cx, cy), round(8 * u))
        elif kind == "player":
            box = rect.inflate(-round(8 * u), -round(8 * u))
            pygame.draw.rect(surf, color, box, border_radius=round(10 * u))
            pygame.draw.rect(surf, darker(color, 0.6), box, max(2, round(3 * u)), border_radius=round(10 * u))
        elif kind == "dot":
            pygame.draw.circle(surf, color, (cx, cy), round(6 * u))
            pygame.draw.circle(surf, darker(color, 0.5), (cx, cy), round(6 * u), 2)

    def draw(self, surf, t, last, done):
        surf.fill(GRID_BG)
        H, W, cell = self.spec.height, self.spec.width, self.cell
        for r in range(H):
            for c in range(W):
                pygame.draw.rect(surf, GRID_LINE, (c * cell, r * cell, cell, cell), 1)
        planes = self.planes()
        for k, color, kind in self.order:
            for r, c in np.argwhere(planes[k]):
                self.shape(surf, kind, color, pygame.Rect(int(c) * cell, int(r) * cell, cell, cell))
        self.draw_coords(surf, H, W, MUTED, cell)
        self.draw_floaters(surf, cell)

    def panel_section(self, panel, y):
        for name, label in zip(self.spec.channels, self.spec.labels):
            if label is None:
                continue
            color, kind = self.style(name)
            box = pygame.Rect(panel.x - 6, y - 7, 34, 34)
            self.shape(panel.s, kind, color, box)
            font = panel.fit(self.fonts.small, self.fonts.tiny, label, panel.w - 34)
            panel.text(font, label, TEXT, 34, y + 3 + (2 if font is self.fonts.tiny else 0))
            y += 26
        return y
