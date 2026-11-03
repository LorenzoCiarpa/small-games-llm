"""
Low Tide - graphical interface (pygame).

    python play.py                              # play yourself
    python play.py --weights weights.json       # watch the network play
    python play.py --baseline heuristic         # watch the heuristic (or: random)

Options: --seed N, --sticky 0.1, --speed 6 (steps per second), --greedy

Keys (also shown in the window):
    arrows           move            space   stay still
    1 2 3 4          you / network / heuristic / random  (switch at any moment,
                     the episode continues from the current state)
    P  pause         N  one step (when paused)       + -  speed
    T  turn-based / real-time (when you play)
    R  restart same seed     E  next seed
    S  sticky actions on/off     G  greedy/sampled (network)
    C  cell coordinates      Esc  quit
"""

import argparse
import math
import sys

import numpy as np
import pygame

from low_tide_env import LowTideEnv, H, W, N_ACTIONS, STUN_STEPS, STICKY_PROB
from evaluate import load_weights, ReferenceMLP, heuristic_policy, random_policy

# ------------------------------------------------------------------ layout
CELL = 60
GRID_PX = CELL * W
PANEL_W = 330
WIN_W, WIN_H = GRID_PX + PANEL_W, CELL * H
PX = GRID_PX + 22                  # left edge of panel content
PANEL_CONTENT_W = PANEL_W - 44

# ------------------------------------------------------------------ palette
SAND_DRY = (226, 207, 160)
SAND_WET = (196, 172, 124)
WATER_SHALLOW = (54, 140, 150)
WATER_DEEP = (22, 78, 98)
FOAM = (226, 242, 238)
PANEL_BG = (16, 44, 58)
PANEL_LINE = (34, 74, 90)
TEXT = (232, 240, 236)
MUTED = (138, 168, 176)
DANGER = (214, 74, 58)
SHELL = (242, 162, 140)
SHELL_RIDGE = (172, 96, 78)
JELLY = (162, 130, 216)
JELLY_RIM = (104, 80, 168)
BUCKET = (214, 64, 54)
BUCKET_RIM = (150, 40, 34)
BUCKET_STUN = (166, 132, 126)
STARS = (246, 202, 72)
STICKY = (240, 150, 40)
BAR_TRACK = (28, 62, 78)
BAR_FILL = (116, 196, 196)

ACTION_LABELS = ("fermo", "su", "giù", "sinistra", "destra")
MODE_LABELS = {"human": "gioca tu", "net": "rete", "heuristic": "euristica", "random": "casuale"}
KEY_ACTIONS = {pygame.K_SPACE: 0, pygame.K_UP: 1, pygame.K_DOWN: 2,
               pygame.K_LEFT: 3, pygame.K_RIGHT: 4}


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def darker(c, k=0.9):
    return tuple(int(v * k) for v in c)


class App:
    def __init__(self, args):
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Low Tide")
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.Font(None, 48)
        self.f_score = pygame.font.Font(None, 72)
        self.f_text = pygame.font.Font(None, 26)
        self.f_small = pygame.font.Font(None, 21)
        self.f_tiny = pygame.font.Font(None, 16)

        self.mlp = None
        self.weights_name = None
        if args.weights:
            self.mlp = ReferenceMLP(load_weights(args.weights))
            self.weights_name = args.weights.replace("\\", "/").split("/")[-1]
        if args.baseline:
            self.mode = args.baseline
        elif self.mlp is not None:
            self.mode = "net"
        else:
            self.mode = "human"

        self.base_sticky = args.sticky if args.sticky > 0 else STICKY_PROB
        self.env = LowTideEnv(sticky_prob=args.sticky)
        self.rng = np.random.default_rng(args.seed + 999)
        self.greedy = args.greedy
        self.speed = args.speed
        self.realtime = False
        self.paused = False
        self.show_coords = False
        self.pending = 0
        self.timer = 0.0
        self.floaters = []
        self.reset(args.seed)

    # -------------------------------------------------------------- game flow
    def reset(self, seed):
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.total = 0.0
        self.done = False
        self.end_reason = None
        self.last = None
        self.floaters.clear()
        self.update_net_view()

    def update_net_view(self):
        self.probs = self.logits = None
        if self.mlp is not None:
            z = self.mlp.logits(self.obs)
            p = np.exp(z - z.max())
            self.logits, self.probs = z, p / p.sum()

    def choose_action(self):
        if self.mode == "net":
            if self.greedy:
                return int(np.argmax(self.probs))
            return int(self.rng.choice(N_ACTIONS, p=self.probs))
        if self.mode == "heuristic":
            return heuristic_policy(self.obs, self.rng)
        if self.mode == "random":
            return random_policy(self.obs, self.rng)
        return self.pending

    def do_step(self, action):
        if self.done:
            return
        was_stunned = self.env.stun > 0
        self.obs, r, term, trunc, info = self.env.step(action)
        executed = self.env.last_action
        self.last = {"chosen": action, "executed": executed,
                     "sticky": executed != action, "stunned": was_stunned, "reward": r}
        self.total += r
        now = pygame.time.get_ticks() / 1000
        if r > 0:
            self.floaters.append(("+1", SHELL, self.env.agent_r, self.env.agent_c, now))
        elif r < 0:
            self.floaters.append(("-1", JELLY, self.env.agent_r, self.env.agent_c, now))
        if term or trunc:
            self.done = True
            self.end_reason = "drowned" if info["drowned"] else "time"
        self.update_net_view()

    def auto_running(self):
        if self.done or self.paused:
            return False
        return self.mode != "human" or self.realtime

    # -------------------------------------------------------------- input
    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_1:
            self.mode = "human"
        elif key == pygame.K_2 and self.mlp is not None:
            self.mode = "net"
        elif key == pygame.K_3:
            self.mode = "heuristic"
        elif key == pygame.K_4:
            self.mode = "random"
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_n and self.paused and self.mode != "human":
            self.do_step(self.choose_action())
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(60.0, self.speed * 1.5)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(0.5, self.speed / 1.5)
        elif key == pygame.K_t:
            self.realtime = not self.realtime
            self.timer = 0.0
        elif key == pygame.K_r:
            self.reset(self.seed)
        elif key == pygame.K_e:
            self.reset(self.seed + 1)
        elif key == pygame.K_s:
            self.env.sticky_prob = 0.0 if self.env.sticky_prob > 0 else self.base_sticky
        elif key == pygame.K_g:
            self.greedy = not self.greedy
        elif key == pygame.K_c:
            self.show_coords = not self.show_coords
        elif self.mode == "human" and key in KEY_ACTIONS:
            a = KEY_ACTIONS[key]
            if self.realtime:
                self.pending = a
            else:
                self.do_step(a)
        return True

    # -------------------------------------------------------------- drawing: grid
    def draw_grid(self, t):
        s, env = self.screen, self.env
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
                    self.draw_shell(cx, cy)
                elif env.jelly[r, c]:
                    self.draw_jelly(cx, cy, t + 0.37 * (r * W + c))

        self.draw_agent(t)

        if self.show_coords:
            for r in range(H):
                for c in range(W):
                    col = FOAM if r >= waterline else darker(SAND_WET, 0.6)
                    s.blit(self.f_tiny.render(f"{r},{c}", True, col), (c * CELL + 3, r * CELL + 2))

        now = pygame.time.get_ticks() / 1000
        keep = []
        for text, color, r, c, t0 in self.floaters:
            age = now - t0
            if age < 0.8:
                img = self.f_text.render(text, True, darker(color, 0.75))
                img.set_alpha(int(255 * (1 - age / 0.8)))
                s.blit(img, (c * CELL + CELL // 2 - img.get_width() // 2, r * CELL - 4 - 26 * age))
                keep.append((text, color, r, c, t0))
        self.floaters = keep

        if self.done:
            self.draw_end_overlay()

    def draw_shell(self, cx, cy):
        s = self.screen
        base = (cx, cy + 12)
        arc = [(cx + 17 * math.cos(a), cy + 6 - 17 * math.sin(a))
               for a in np.linspace(math.radians(15), math.radians(165), 12)]
        pygame.draw.polygon(s, SHELL, [base] + arc)
        for a in np.linspace(math.radians(35), math.radians(145), 5):
            pygame.draw.line(s, SHELL_RIDGE, base, (cx + 16 * math.cos(a), cy + 6 - 16 * math.sin(a)), 2)
        pygame.draw.polygon(s, SHELL_RIDGE, [base] + arc, 2)
        pygame.draw.rect(s, SHELL_RIDGE, (cx - 5, cy + 10, 10, 5), border_radius=2)

    def draw_jelly(self, cx, cy, phase):
        s = self.screen
        for k in range(4):
            x = cx - 9 + 6 * k
            pts = [(x + 2.5 * math.sin(phase * 3 + j * 0.9 + k), cy + 2 + j * 4) for j in range(5)]
            pygame.draw.lines(s, JELLY_RIM, False, pts, 2)
        dome = pygame.Surface((36, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(dome, (*JELLY, 225), (0, 0, 36, 40))
        s.blit(dome, (cx - 18, cy - 16))
        pygame.draw.line(s, JELLY_RIM, (cx - 17, cy + 5), (cx + 17, cy + 5), 2)
        pygame.draw.circle(s, (232, 220, 250), (cx - 6, cy - 8), 3)

    def draw_agent(self, t):
        s, env = self.screen, self.env
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

        if self.last and self.last["executed"] != 0 and not self.last["stunned"] and not self.done:
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = dirs[self.last["executed"]]
            ax, ay = cx + dx * 25, cy - 4 + dy * 25
            px, py = -dy, dx
            color = STICKY if self.last["sticky"] else darker(SAND_WET, 0.5)
            pygame.draw.polygon(s, color, [(ax + dx * 6, ay + dy * 6),
                                           (ax + px * 6, ay + py * 6), (ax - px * 6, ay - py * 6)])

    def draw_end_overlay(self):
        s = self.screen
        veil = pygame.Surface((GRID_PX, WIN_H), pygame.SRCALPHA)
        veil.fill((10, 30, 40, 170))
        s.blit(veil, (0, 0))
        if self.end_reason == "drowned":
            head = f"Annegato al passo {self.env.t}"
        else:
            head = f"Fine dei {self.env.t} passi"
        lines = [(self.f_title, head, TEXT),
                 (self.f_text, f"punteggio {self.total:g}", FOAM),
                 (self.f_small, "R per ricominciare, E per un nuovo seed", MUTED)]
        y = WIN_H // 2 - 50
        for font, text, color in lines:
            img = font.render(text, True, color)
            s.blit(img, (GRID_PX // 2 - img.get_width() // 2, y))
            y += img.get_height() + 14

    # -------------------------------------------------------------- drawing: panel
    def text(self, font, txt, color, x, y):
        img = font.render(txt, True, color)
        self.screen.blit(img, (x, y))
        return img.get_height()

    def pips(self, x, y, n_on, n_total, color_on, color_off=PANEL_LINE, size=9, gap=6):
        for i in range(n_total):
            pygame.draw.circle(self.screen, color_on if i < n_on else color_off,
                               (x + size + i * (2 * size + gap), y + size), size)

    def draw_panel(self):
        s, env = self.screen, self.env
        pygame.draw.rect(s, PANEL_BG, (GRID_PX, 0, PANEL_W, WIN_H))
        y = 18
        self.text(self.f_title, "Low Tide", TEXT, PX, y)
        y += 42
        mode = MODE_LABELS[self.mode]
        if self.mode == "human":
            mode += ", in tempo reale" if self.realtime else ", a turni"
        elif self.mode == "net":
            mode += f" {self.weights_name}, " + ("greedy" if self.greedy else "campionata")
        if self.paused and self.mode != "human":
            mode += ", in pausa"
        self.text(self.f_small, mode, MUTED, PX, y)
        y += 20
        sticky = f"sticky {env.sticky_prob:g}" if env.sticky_prob > 0 else "sticky spente"
        self.text(self.f_small, f"{sticky}, seed {self.seed}, {self.speed:.1f} passi/s", MUTED, PX, y)

        y += 32
        self.text(self.f_score, f"{self.total:g}", TEXT, PX, y)
        n_sh, n_st = env.shells_collected, env.stings
        shells = f"{n_sh} conchigli" + ("a" if n_sh == 1 else "e")
        stings = f"{n_st} puntur" + ("a" if n_st == 1 else "e")
        self.text(self.f_small, f"{shells}, {stings}",
                  MUTED, PX + 110, y + 14)
        self.text(self.f_small, f"passo {env.t} di {env.max_steps}", MUTED, PX + 110, y + 34)

        y += 72
        pygame.draw.line(s, PANEL_LINE, (PX, y), (PX + PANEL_CONTENT_W, y), 1)
        y += 12
        rising = env.direction > 0
        ax, ay = PX + 8, y + 11
        if rising:
            pygame.draw.polygon(s, DANGER, [(ax, ay - 9), (ax - 8, ay + 6), (ax + 8, ay + 6)])
        else:
            pygame.draw.polygon(s, BAR_FILL, [(ax, ay + 9), (ax - 8, ay - 6), (ax + 8, ay - 6)])
        self.text(self.f_text, "la marea sale" if rising else "la marea scende", TEXT, PX + 26, y + 2)
        y += 30
        self.text(self.f_small, "prossimo cambio tra", MUTED, PX, y + 3)
        self.pips(PX + 150, y, env.countdown, 5, DANGER if rising and env.countdown == 1 else BAR_FILL, size=7, gap=5)
        y += 24
        if rising and env.countdown == 1 and not self.done:
            self.text(self.f_small, "la riga rossa sta per allagarsi", DANGER, PX, y)
        elif env.stun > 0 and not self.done:
            self.text(self.f_small, f"paralizzato: ancora {env.stun} passi", STARS, PX, y)
        y += 22
        msg, col = self.last_message()
        if msg:
            self.text(self.f_small, msg, col, PX, y)

        y += 30
        pygame.draw.line(s, PANEL_LINE, (PX, y), (PX + PANEL_CONTENT_W, y), 1)
        y += 12
        if self.probs is not None:
            title = "la rete adesso sceglierebbe" if self.mode == "net" else "la rete qui sceglierebbe"
            self.text(self.f_small, title, MUTED, PX, y)
            y += 22
            best = int(np.argmax(self.probs))
            for a in range(N_ACTIONS):
                color = TEXT if a == best else MUTED
                self.text(self.f_small, ACTION_LABELS[a], color, PX, y + 2)
                bx, bw = PX + 72, 140
                pygame.draw.rect(s, BAR_TRACK, (bx, y + 3, bw, 13), border_radius=3)
                pygame.draw.rect(s, BAR_FILL if a == best else darker(BAR_FILL, 0.7),
                                 (bx, y + 3, max(2, int(bw * self.probs[a])), 13), border_radius=3)
                self.text(self.f_tiny, f"{100 * self.probs[a]:.0f}%  z {self.logits[a]:+.1f}",
                          color, bx + bw + 8, y + 4)
                y += 20
        else:
            self.text(self.f_small, "nessuna rete caricata (--weights)", MUTED, PX, y)
            y += 22

        help_lines = []
        if self.mode == "human":
            help_lines += ["frecce per muoverti, spazio per restare fermo",
                           "T a turni o in tempo reale"]
        else:
            help_lines += ["P pausa, N un passo, + e - velocità"]
        help_lines += ["1 tu, 2 rete, 3 euristica, 4 casuale",
                       "R ricomincia, E nuovo seed",
                       "S sticky, G greedy, C coordinate, Esc esci"]
        y = WIN_H - 14 - 19 * len(help_lines)
        for line in help_lines:
            self.text(self.f_small, line, MUTED, PX, y)
            y += 19

    def last_message(self):
        L = self.last
        if not L:
            return None, MUTED
        if L["stunned"]:
            return "azione ignorata: paralizzato", STARS
        if L["sticky"]:
            return (f"sticky: {ACTION_LABELS[L['executed']]} invece di {ACTION_LABELS[L['chosen']]}",
                    STICKY)
        if L["reward"] > 0:
            return "conchiglia raccolta", SHELL
        if L["reward"] < 0:
            return "medusa: paralizzato per 3 passi", JELLY
        return f"azione: {ACTION_LABELS[L['executed']]}", MUTED

    # -------------------------------------------------------------- main loop
    def draw(self, t):
        self.draw_grid(t)
        self.draw_panel()

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    running = self.handle_key(ev.key)
            if self.auto_running():
                self.timer += dt
                while self.timer >= 1.0 / self.speed and not self.done:
                    self.timer -= 1.0 / self.speed
                    self.do_step(self.choose_action())
                    self.pending = 0
            else:
                self.timer = 0.0
            self.draw(pygame.time.get_ticks() / 1000)
            pygame.display.flip()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Low Tide: gioca o guarda una policy")
    ap.add_argument("--weights", help="weights.json della rete da guardare")
    ap.add_argument("--baseline", choices=("heuristic", "random"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sticky", type=float, default=STICKY_PROB)
    ap.add_argument("--speed", type=float, default=6.0, help="passi al secondo in riproduzione")
    ap.add_argument("--greedy", action="store_true", help="argmax invece di campionare")
    App(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
