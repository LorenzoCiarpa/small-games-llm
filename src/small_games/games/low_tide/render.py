"""Low Tide's own look for `play`: beach, tide, shells, jellyfish and the bucket."""

import math

import numpy as np
import pygame

from ...render import CELL, DANGER, MUTED, TEXT, BAR_FILL, Renderer, darker, lerp
from .env import H, STUN_STEPS, W

GRID_PX = CELL * W

SAND_DRY = (226, 207, 160)
SAND_WET = (196, 172, 124)
WATER_SHALLOW = (54, 140, 150)
WATER_DEEP = (22, 78, 98)
FOAM = (226, 242, 238)
SHELL = (242, 162, 140)
SHELL_RIDGE = (172, 96, 78)
JELLY = (162, 130, 216)
JELLY_RIM = (104, 80, 168)
BUCKET = (214, 64, 54)
BUCKET_RIM = (150, 40, 34)
BUCKET_STUN = (166, 132, 126)
STARS = (246, 202, 72)
STICKY = (240, 150, 40)


class LowTideRenderer(Renderer):
    was_stunned = False

    @property
    def size(self):
        return GRID_PX, CELL * H

    def anchor_cell(self):
        return self.env.agent_r, self.env.agent_c

    def before_step(self):
        self.was_stunned = self.env.stun > 0

    def after_step(self, obs, last):
        last["stunned"] = self.was_stunned
        r = last["reward"]
        if r:
            color = SHELL if r > 0 else JELLY
            self.floaters.append((f"{r:+g}", color, self.env.agent_r, self.env.agent_c,
                                  pygame.time.get_ticks() / 1000))
        self.obs = obs

    # -------------------------------------------------------------- grid
    def draw(self, s, t, last, done):
        env = self.env
        waterline = H - env.level            # first water row
        for r in range(H):
            for c in range(W):
                rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
                if r >= waterline:
                    depth = (r - waterline) / max(1, H - 1)
                    pygame.draw.rect(s, lerp(WATER_SHALLOW, WATER_DEEP, min(1.0, depth * 1.6)), rect)
                else:
                    base = SAND_DRY if r < 3 else SAND_WET
                    pygame.draw.rect(s, base, rect)
                    pygame.draw.rect(s, darker(base, 0.93), rect, 1)

        # row about to be flooded
        if env.direction > 0 and waterline - 1 >= 0 and env.countdown <= 2:
            alpha = 120 if env.countdown == 1 else 55
            tint = pygame.Surface((GRID_PX, CELL), pygame.SRCALPHA)
            tint.fill((*DANGER, alpha))
            s.blit(tint, (0, (waterline - 1) * CELL))
            if env.countdown == 1:
                pygame.draw.rect(s, DANGER, (0, (waterline - 1) * CELL, GRID_PX, CELL), 2)

        # waterline with foam
        if env.level > 0:
            y0 = waterline * CELL
            pts = [(x, y0 + 3 * math.sin(x * 0.07 + t * 2.0)) for x in range(0, GRID_PX + 7, 6)]
            pygame.draw.polygon(s, WATER_SHALLOW, pts + [(GRID_PX, y0 + 8), (0, y0 + 8)])
            pygame.draw.lines(s, FOAM, False, pts, 3)

        for r in range(H):
            for c in range(W):
                cx, cy = c * CELL + CELL // 2, r * CELL + CELL // 2
                if env.shell[r, c]:
                    self.draw_shell(s, cx, cy)
                elif env.jelly[r, c]:
                    self.draw_jelly(s, cx, cy, t + 0.37 * (r * W + c))

        self.draw_agent(s, t, last, done)

        if self.show_coords:
            for r in range(H):
                for c in range(W):
                    col = FOAM if r >= waterline else darker(SAND_WET, 0.6)
                    s.blit(self.fonts.tiny.render(f"{r},{c}", True, col), (c * CELL + 3, r * CELL + 2))
        self.draw_floaters(s)

    def draw_shell(self, s, cx, cy):
        base = (cx, cy + 12)
        arc = [(cx + 17 * math.cos(a), cy + 6 - 17 * math.sin(a))
               for a in np.linspace(math.radians(15), math.radians(165), 12)]
        pygame.draw.polygon(s, SHELL, [base] + arc)
        for a in np.linspace(math.radians(35), math.radians(145), 5):
            pygame.draw.line(s, SHELL_RIDGE, base, (cx + 16 * math.cos(a), cy + 6 - 16 * math.sin(a)), 2)
        pygame.draw.polygon(s, SHELL_RIDGE, [base] + arc, 2)
        pygame.draw.rect(s, SHELL_RIDGE, (cx - 5, cy + 10, 10, 5), border_radius=2)

    def draw_jelly(self, s, cx, cy, phase):
        for k in range(4):
            x = cx - 9 + 6 * k
            pts = [(x + 2.5 * math.sin(phase * 3 + j * 0.9 + k), cy + 2 + j * 4) for j in range(5)]
            pygame.draw.lines(s, JELLY_RIM, False, pts, 2)
        dome = pygame.Surface((36, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(dome, (*JELLY, 225), (0, 0, 36, 40))
        s.blit(dome, (cx - 18, cy - 16))
        pygame.draw.line(s, JELLY_RIM, (cx - 17, cy + 5), (cx + 17, cy + 5), 2)
        pygame.draw.circle(s, (232, 220, 250), (cx - 6, cy - 8), 3)

    def draw_agent(self, s, t, last, done):
        env = self.env
        cx, cy = env.agent_c * CELL + CELL // 2, env.agent_r * CELL + CELL // 2 + 4
        stunned = env.stun > 0
        body = BUCKET_STUN if stunned else BUCKET
        pygame.draw.arc(s, darker(body, 0.6), (cx - 14, cy - 24, 28, 26), 0, math.pi, 3)
        pygame.draw.polygon(s, body, [(cx - 15, cy - 10), (cx + 15, cy - 10), (cx + 11, cy + 14), (cx - 11, cy + 14)])
        pygame.draw.rect(s, BUCKET_RIM if not stunned else darker(body, 0.7), (cx - 16, cy - 13, 32, 6), border_radius=2)
        if stunned:
            for k in range(3):
                a = t * 4 + k * 2 * math.pi / 3
                pygame.draw.circle(s, STARS, (cx + 14 * math.cos(a), cy - 28 + 4 * math.sin(a)), 3)

        if last and last["executed"] != 0 and not last.get("stunned") and not done:
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = dirs[last["executed"]]
            ax, ay = cx + dx * 25, cy - 4 + dy * 25
            px, py = -dy, dx
            color = STICKY if last["sticky"] else darker(SAND_WET, 0.5)
            pygame.draw.polygon(s, color, [(ax + dx * 6, ay + dy * 6),
                                           (ax + px * 6, ay + py * 6), (ax - px * 6, ay - py * 6)])

    # -------------------------------------------------------------- panel
    def score_details(self):
        n_sh, n_st = self.env.shells_collected, self.env.stings
        return [f"{n_sh} shell" + ("" if n_sh == 1 else "s") + ", "
                + f"{n_st} sting" + ("" if n_st == 1 else "s")]

    def panel_section(self, panel, y):
        env, s = self.env, panel.s
        rising = env.direction > 0
        ax, ay = panel.x + 8, y + 11
        if rising:
            pygame.draw.polygon(s, DANGER, [(ax, ay - 9), (ax - 8, ay + 6), (ax + 8, ay + 6)])
        else:
            pygame.draw.polygon(s, BAR_FILL, [(ax, ay + 9), (ax - 8, ay - 6), (ax + 8, ay - 6)])
        panel.text(self.fonts.text, "the tide is rising" if rising else "the tide is receding", TEXT, 26, y + 2)
        y += 30
        panel.text(self.fonts.small, "next change in", MUTED, 0, y + 3)
        panel.pips(150, y, env.countdown, 5, DANGER if rising and env.countdown == 1 else BAR_FILL,
                   size=7, gap=5)
        y += 24
        if rising and env.countdown == 1:
            panel.text(self.fonts.small, "the red row is about to flood", DANGER, 0, y)
        elif env.stun > 0:
            panel.text(self.fonts.small, f"stunned: {env.stun} more step" + ("" if env.stun == 1 else "s"), STARS, 0, y)
        return y + 22

    def step_message(self, last):
        if last.get("stunned"):
            return "action ignored: stunned", STARS
        if last["sticky"]:
            return None
        if last["reward"] > 0:
            return "shell collected", SHELL
        if last["reward"] < 0:
            return f"jellyfish: stunned for {STUN_STEPS} steps", JELLY
        return None

    def end_title(self, terminated):
        if terminated and self.env.drowned:
            return f"Drowned at step {self.env.t}"
        return super().end_title(terminated)
