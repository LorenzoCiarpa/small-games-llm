"""
Short gameplay videos for the blog article: one MP4 per game, played by a network written by an LLM.

The frames are drawn by the same renderers as `uv run play` (pygame, without a window): Low Tide with its own
renderer, the MinAtar games with the grid renderer, which shows the 6-channel observation the network receives.
The episodes are those of the evaluation protocol (seed 10000, 10001, ...), with sticky actions and sampled actions.

    uv run python experiments/article/videos.py            # -> experiments/article/videos/<game>.mp4

Needs ffmpeg on the PATH.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import importlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pygame

from small_games.core import get_game
from small_games.policy import load_mlp, policy_rng, sample_action
from small_games.render import MUTED, PANEL_BG, TEXT, Fonts, GridRenderer

HERE = Path(__file__).resolve().parent
OUT = HERE / "videos"
STRIP = 48          # height of the status strip under the grid, in pixels

# game -> (weight file, label of the player, frames, frames per second)
CLIPS = {
    "low_tide": ("claude", "network written by Claude Opus 5.5", 240, 8),
    "breakout": ("paper_astra", "network written by GPT-6 Astra", 300, 12),
    "seaquest": ("paper_astra", "network written by GPT-6 Astra", 300, 10),
    "asterix": ("paper_astra", "network written by GPT-6 Astra", 300, 10),
}

_clock = [0.0]      # simulated time in seconds: the reward floaters fade with it, not with the wall clock
pygame.time.get_ticks = lambda: int(_clock[0] * 1000)


def renderer_class(game):
    if game.renderer is None:
        return GridRenderer
    module, _, cls = game.renderer.partition(":")
    return getattr(importlib.import_module(module), cls)


def record(name, frames_dir):
    weights, _, n_frames, fps = CLIPS[name]
    game = get_game(name)
    _, mlp = load_mlp(game, weights)
    env = game.make_env(sticky_prob=game.protocol.sticky_prob, max_steps=game.protocol.max_steps)
    fonts = Fonts()
    rend = renderer_class(game)(game, env, fonts)
    w, h = rend.size
    grid = pygame.Surface((w, h))
    frame = pygame.Surface((w, h + STRIP))

    seed = game.protocol.seed0
    obs, _ = env.reset(seed=seed)
    rng, total, last, done, hold = policy_rng(seed), 0.0, None, False, 0
    rend.reset(obs)
    for k in range(n_frames):
        _clock[0] = k / fps
        rend.draw(grid, _clock[0], last, done)
        frame.fill(PANEL_BG)
        frame.blit(grid, (0, 0))
        left = fonts.text.render(game.title.split(" (")[0], True, MUTED)
        right = fonts.text.render(f"score {total:g}   step {env.t}", True, TEXT)
        y = h + (STRIP - left.get_height()) // 2
        frame.blit(left, (14, y))
        frame.blit(right, (w - right.get_width() - 14, y))
        pygame.image.save(frame, str(frames_dir / f"{k:05d}.png"))

        if done:                                   # hold the last frame for one second, then the next episode
            hold += 1
            if hold >= fps:
                seed += 1
                obs, _ = env.reset(seed=seed)
                rng, total, last, done, hold = policy_rng(seed), 0.0, None, False, 0
                rend.reset(obs)
            continue
        action = sample_action(mlp.logits(obs), rng)
        rend.before_step()
        obs, r, term, trunc, info = env.step(action)
        executed = env.last_action
        last = {"chosen": action, "executed": executed, "sticky": executed != action, "reward": r, "info": info}
        total += r
        rend.after_step(obs, last)
        done = term or trunc
    return fps


def encode(frames_dir, fps, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=list(CLIPS), choices=list(CLIPS))
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")
    pygame.init()
    OUT.mkdir(exist_ok=True)
    for name in args.games:
        with tempfile.TemporaryDirectory() as tmp:
            fps = record(name, Path(tmp))
            out = OUT / f"{name}.mp4"
            encode(Path(tmp), fps, out)
        print(f"{out.relative_to(HERE.parents[1])}: {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
