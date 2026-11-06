"""
Play any registered game, or watch a network or a baseline play it (pygame).

    uv run play low_tide                          # play yourself
    uv run play low_tide --weights claude         # watch a network (file or name in weights/<game>/)
    uv run play low_tide --baseline heuristic     # watch a baseline
    uv run play breakout --weights paper_astra

Options: --seed N, --sticky 0.1, --speed 6 (steps per second), --greedy, --list

Reproducibility: the episode with seed s depends only on s (the game, the
sticky actions and the policy's sampling all restart from it), so R replays
it exactly and `play <game> --seed s` shows episode s - seed0 of `evaluate`.
The default seed is the game's seed0, so `play` opens on episode 0 of
`evaluate` and E steps through episodes 1, 2, ...
This holds while the same policy plays the whole episode with the same
settings: switching who plays, G or S in the middle change what follows.

Keys (also shown in the window):
    the game's keys  (arrows, space, ...) when you play
    1 2 3 ...        you / network / baselines (switch at any moment,
                     the episode continues from the current state)
    P  pause         N  one step (when paused)       + -  speed
    T  turn-based / real-time (when you play)
    R  restart same seed     E  next seed (seed + 1)
    S  sticky actions on/off     G  greedy/sampled (network)
    C  cell coordinates      Esc  quit
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .core import get_game, list_games
from .policy import baselines, games_overview, load_mlp, policy_rng, sample_action, softmax
from .render import (BAR_FILL, BAR_TRACK, DANGER, GOOD, MIN_H, MUTED, PANEL_BG, PANEL_W, STICKY,
                     TEXT, Fonts, GridRenderer, Panel, darker)

MODE_LABELS = {"human": "gioca tu", "net": "rete", "random": "casuale",
               "heuristic": "euristica", "astra_code": "codice di Astra"}
VEIL = (10, 30, 40, 170)


def load_renderer(game):
    if game.renderer is None:
        return GridRenderer
    module, _, cls = game.renderer.partition(":")
    return getattr(importlib.import_module(module), cls)


class App:
    def __init__(self, game, args):
        self.game = game
        pygame.display.init()
        pygame.font.init()
        self.fonts = Fonts()

        self.mlp = None
        self.weights_name = None
        if args.weights:
            path, self.mlp = load_mlp(game, args.weights)
            self.weights_name = path.name
        self.policies = baselines(game)
        self.modes = ["human"] + (["net"] if self.mlp else []) + list(self.policies)
        if args.baseline:
            self.mode = args.baseline
        else:
            self.mode = "net" if self.mlp else "human"
        self.keys = {pygame.key.key_code(name): a for name, a in game.human_keys.items()}

        self.base_sticky = args.sticky if args.sticky > 0 else game.protocol.sticky_prob
        self.env = game.make_env(sticky_prob=args.sticky, max_steps=game.protocol.max_steps)
        self.renderer = load_renderer(game)(game, self.env, self.fonts)

        grid_w, grid_h = self.renderer.size
        self.win_h = max(grid_h, MIN_H)
        self.grid_w = grid_w
        self.screen = pygame.display.set_mode((grid_w + PANEL_W, self.win_h))
        self.grid_surf = self.screen.subsurface(((0, (self.win_h - grid_h) // 2), (grid_w, grid_h)))
        self.panel = Panel(self.screen, self.fonts, grid_w + 22, PANEL_W - 44)
        pygame.display.set_caption(game.title)
        self.clock = pygame.time.Clock()

        self.greedy = args.greedy
        self.speed = args.speed
        self.realtime = False
        self.paused = False
        self.pending = 0
        self.timer = 0.0
        self.reset(args.seed)

    # -------------------------------------------------------------- game flow
    def reset(self, seed):
        self.seed = seed
        self.rng = policy_rng(seed)        # the same generator `evaluate` uses for this seed
        self.obs, _ = self.env.reset(seed=seed)
        self.total = 0.0
        self.done = False
        self.terminated = False
        self.last = None
        self.renderer.reset(self.obs)
        self.update_net_view()

    def update_net_view(self):
        self.probs = self.logits = None
        if self.mlp is not None:
            self.logits = self.mlp.logits(self.obs)
            self.probs = softmax(self.logits)

    def choose_action(self):
        if self.mode == "net":
            return sample_action(self.logits, self.rng, self.greedy)
        if self.mode in self.policies:
            return self.policies[self.mode](self.obs, self.rng)
        return self.pending

    def do_step(self, action):
        if self.done:
            return
        self.renderer.before_step()
        self.obs, r, term, trunc, info = self.env.step(action)
        executed = self.env.last_action
        self.last = {"chosen": action, "executed": executed, "sticky": executed != action,
                     "reward": r, "info": info}
        self.total += r
        self.renderer.after_step(self.obs, self.last)
        if term or trunc:
            self.done = True
            self.terminated = term
        self.update_net_view()

    def auto_running(self):
        if self.done or self.paused:
            return False
        return self.mode != "human" or self.realtime

    # -------------------------------------------------------------- input
    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if pygame.K_1 <= key <= pygame.K_9 and key - pygame.K_1 < len(self.modes):
            self.mode = self.modes[key - pygame.K_1]
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_n and self.paused and self.mode != "human":
            self.do_step(self.choose_action())
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(600.0, self.speed * 1.5)
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
            self.renderer.show_coords = not self.renderer.show_coords
        elif self.mode == "human" and key in self.keys:
            a = self.keys[key]
            if self.realtime:
                self.pending = a
            else:
                self.do_step(a)
        return True

    # -------------------------------------------------------------- drawing
    def draw_end_overlay(self):
        s = self.grid_surf
        w, h = s.get_size()
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill(VEIL)
        s.blit(veil, (0, 0))
        lines = [(self.fonts.title, self.renderer.end_title(self.terminated), TEXT),
                 (self.fonts.text, f"punteggio {self.total:g}", TEXT),
                 (self.fonts.small, "R per ricominciare, E per un nuovo seed", MUTED)]
        y = h // 2 - 50
        for font, text, color in lines:
            img = font.render(text, True, color)
            s.blit(img, (w // 2 - img.get_width() // 2, y))
            y += img.get_height() + 14

    def mode_label(self, mode):
        return MODE_LABELS.get(mode, mode)

    def last_message(self):
        L = self.last
        if not L:
            return None, MUTED
        custom = self.renderer.step_message(L)
        if custom:
            return custom
        names = self.game.action_names
        if L["sticky"]:
            return f"sticky: {names[L['executed']]} invece di {names[L['chosen']]}", STICKY
        if L["reward"] > 0:
            return f"ricompensa +{L['reward']:g}", GOOD
        if L["reward"] < 0:
            return f"penalità {L['reward']:g}", DANGER
        return f"azione: {names[L['executed']]}", MUTED

    def draw_panel(self):
        p, f, env = self.panel, self.fonts, self.env
        pygame.draw.rect(self.screen, PANEL_BG, (self.grid_w, 0, PANEL_W, self.win_h))
        y = 18
        p.text(f.title, self.game.title, TEXT, 0, y)
        y += 42
        mode = self.mode_label(self.mode)
        if self.mode == "human":
            mode += ", in tempo reale" if self.realtime else ", a turni"
        elif self.mode == "net":
            mode += f" {self.weights_name}, " + ("greedy" if self.greedy else "campionata")
        if self.paused and self.mode != "human":
            mode += ", in pausa"
        p.text(f.small, mode, MUTED, 0, y)
        y += 20
        sticky = f"sticky {env.sticky_prob:g}" if env.sticky_prob > 0 else "sticky spente"
        p.text(f.small, f"{sticky}, seed {self.seed}, {self.speed:.1f} passi/s", MUTED, 0, y)

        y += 32
        p.text(f.score, f"{self.total:g}", TEXT, 0, y)
        details = self.renderer.score_details() + [f"passo {env.t} di {env.max_steps}"]
        for k, line in enumerate(details[-2:]):
            p.text(f.small, line, MUTED, 110, y + 14 + 20 * k + (10 if len(details) == 1 else 0))

        y += 72
        p.line(y)
        y += 12
        y = self.renderer.panel_section(p, y)
        msg, col = self.last_message()
        if msg:
            p.text(f.small, msg, col, 0, y + 4)

        y += 34
        p.line(y)
        y += 12
        if self.probs is not None:
            title = "la rete adesso sceglierebbe" if self.mode == "net" else "la rete qui sceglierebbe"
            p.text(f.small, title, MUTED, 0, y)
            y += 22
            best = int(np.argmax(self.probs))
            for a, name in enumerate(self.game.action_names):
                color = TEXT if a == best else MUTED
                p.text(f.small, name, color, 0, y + 2)
                bx, bw = p.x + 72, 140
                pygame.draw.rect(self.screen, BAR_TRACK, (bx, y + 3, bw, 13), border_radius=3)
                pygame.draw.rect(self.screen, BAR_FILL if a == best else darker(BAR_FILL, 0.7),
                                 (bx, y + 3, max(2, int(bw * self.probs[a])), 13), border_radius=3)
                p.text(f.tiny, f"{100 * self.probs[a]:.0f}%  z {self.logits[a]:+.1f}", color, 72 + bw + 8, y + 4)
                y += 20
        else:
            p.text(f.small, "nessuna rete caricata (--weights)", MUTED, 0, y)

        if self.mode == "human":
            groups = [self.game.human_help.split(", "), ["T a turni o in tempo reale"]]
        else:
            groups = [["P pausa", "N un passo", "+ e - velocità"]]
        groups += [[f"{k + 1} " + ("tu" if m == "human" else self.mode_label(m)) for k, m in enumerate(self.modes)],
                   ["R ricomincia", "E nuovo seed"],
                   ["S sticky", "G greedy", "C coordinate", "Esc esci"]]
        help_lines = [line for g in groups for line in p.wrap(f.small, g)]
        y = self.win_h - 14 - 19 * len(help_lines)
        for line in help_lines:
            p.text(f.small, line, MUTED, 0, y)
            y += 19

    def draw(self, t):
        self.screen.fill(PANEL_BG)
        self.renderer.draw(self.grid_surf, t, self.last, self.done)
        if self.done:
            self.draw_end_overlay()
        self.draw_panel()

    # -------------------------------------------------------------- main loop
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gioca a un gioco o guarda una policy",
        epilog="giochi:\n" + games_overview() + "\n\n--list descrive ogni gioco e ogni baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", choices=list_games(), help="il gioco (vedi sotto)")
    ap.add_argument("--list", action="store_true", help="descrive i giochi e le loro baseline ed esce")
    ap.add_argument("--weights", help="la rete da guardare: un file, o un nome in weights/<game>/")
    ap.add_argument("--baseline", help="una baseline da guardare (vedi sotto)")
    ap.add_argument("--seed", type=int,
                    help="seed del primo episodio (default: il seed0 del gioco, cioè l'episodio 0 di "
                         "evaluate); con lo stesso seed l'episodio si ripete identico, ed è l'episodio "
                         "seed - seed0 di evaluate")
    ap.add_argument("--sticky", type=float, help="probabilità di sticky action (default: quella del gioco)")
    ap.add_argument("--speed", type=float, default=6.0, help="passi al secondo in riproduzione")
    ap.add_argument("--greedy", action="store_true", help="argmax invece di campionare")
    args = ap.parse_args(argv)

    if args.list:
        print(games_overview(verbose=True))
        return 0
    if not args.game:
        ap.error(f"scegli un gioco: {', '.join(list_games())}")
    game = get_game(args.game)
    if args.sticky is None:
        args.sticky = game.protocol.sticky_prob
    if args.seed is None:
        args.seed = game.protocol.seed0
    known = baselines(game)
    if args.baseline and args.baseline not in known:
        ap.error(f"{game.name}: baseline sconosciuta {args.baseline!r}; disponibili: {', '.join(known)}")
    try:
        App(game, args).run()
    except (FileNotFoundError, ValueError) as e:
        ap.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
