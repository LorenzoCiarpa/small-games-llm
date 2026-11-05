"""
PPO (Proximal Policy Optimization) for any registered game, with a policy network that a weight file can describe.

The actor is exactly the MLP that `weights.py` reads: obs -> hidden layers (one activation) -> linear logits, within
the game's NetSpec. It is exported as a dense weight file, so `evaluate` and `play` score it with the reference
forward pass like any hand-written network. The critic is a separate network and is not exported.

Training uses the game's own environment and protocol settings (sticky actions, step cap), but never the evaluation
seeds: the training environments are seeded from `train_seed0` (default 1_000_000), far from the protocol's
seed0 + 0..episodes.

The algorithm is the standard clipped PPO with GAE (as in CleanRL's ppo.py): `n_envs` environments stepped in
lockstep, rollouts of `n_steps` steps, `epochs` passes of minibatch SGD with Adam, linearly annealed learning
rate, advantage normalisation, gradient clipping. A truncated episode (step cap) is bootstrapped with the value
of its last observation instead of being treated as a terminal state.

`actions` restricts the policy to a subset of the actions (the game's minimal action set, as usual for Atari and
MinAtar): in Breakout four of the six actions do nothing, and with all six a random policy moves only a third of
the time; PPO then settled on "almost never move" (see experiments/breakout/ppo/README.md).

Interrupting and resuming: every `save_every` steps the full state (actor, critic, optimizer, the games themselves,
random generators) goes to results/<name>/state.pt, and the actor to checkpoint_<N>M.json. SIGTERM or Ctrl-C stop
at the end of the current update after saving; `--resume` continues from state.pt exactly where it stopped (same
result as an uninterrupted run).

Needs the `ppo` dependency group:  uv run --group ppo python experiments/<game>/ppo/train.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .core import get_game


@dataclass(frozen=True)
class PPOConfig:
    game: str
    hidden: tuple[int, ...] = (32, 32)       # actor hidden layers (must fit the game's NetSpec)
    actions: tuple[int, ...] | None = None   # actions the policy may use (None: all); the others get logit -100
    activation: str = "tanh"                 # actor hidden activation
    critic_hidden: tuple[int, ...] = (64, 64)
    total_steps: int = 10_000_000
    n_envs: int = 64
    n_steps: int = 128
    epochs: int = 4
    minibatches: int = 8
    lr: float = 5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    sticky: float | None = None              # sticky actions during training (None: the protocol's)
    seed: int = 0
    train_seed0: int = 1_000_000             # env seeds for training (the protocol uses 10_000..)
    threads: int = 1
    log_every: int = 10                      # updates between log lines
    save_every: int = 5_000_000              # steps between saves of the full state and of the weights


def _check_spec(cfg: PPOConfig, game):
    spec = game.net
    if not 1 <= len(cfg.hidden) <= spec.max_hidden_layers:
        raise ValueError(f"{game.name}: {len(cfg.hidden)} hidden layers, allowed 1..{spec.max_hidden_layers}")
    if any(not 1 <= h <= spec.max_hidden_units for h in cfg.hidden):
        raise ValueError(f"{game.name}: hidden sizes {cfg.hidden}, max {spec.max_hidden_units}")
    if cfg.activation not in spec.hidden_activations:
        raise ValueError(f"{game.name}: activation {cfg.activation!r} not in {spec.hidden_activations}")


def _mlp(sizes, activation, out_gain):
    import torch.nn as nn

    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, out_gain if last else math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(act())
    return nn.Sequential(*layers)


UNUSED_LOGIT = -100.0


def export_actor(actor, cfg: PPOConfig, game, path: Path, source: str):
    """Write the actor as a dense weight file in the format of weights.py.

    Actions outside `cfg.actions` get zero weights and bias UNUSED_LOGIT in the output layer, so the exported
    network (still obs -> ... -> n_actions) never picks them, exactly as during training.
    """
    lins = [m for m in actor if hasattr(m, "weight")]
    layers = []
    for i, lin in enumerate(lins):
        W = lin.weight.detach().double().numpy().copy()
        b = lin.bias.detach().double().numpy().copy()
        last = i == len(lins) - 1
        if last and cfg.actions is not None:
            unused = [a for a in range(game.n_actions) if a not in cfg.actions]
            W[unused] = 0.0
            b[unused] = UNUSED_LOGIT
        layers.append({
            "in": int(W.shape[1]), "out": int(W.shape[0]),
            "activation": "linear" if last else cfg.activation,
            "W": {"dense": [[float(f"{x:.6g}") for x in row] for row in W]},
            "b": {"dense": [float(f"{x:.6g}") for x in b]},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"source": source, "layers": layers}, f, separators=(",", ":"))
    # the file must load and validate like any other weight file
    from .weights import load_weights
    load_weights(path, game)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


LOG_HEADER = ["update", "steps", "seconds", "episodes", "mean_return", "mean_length",
              "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "lr"]


class _StopFlag:
    """SIGTERM / SIGINT set a flag; the training loop saves and exits at the end of the current update."""

    def __init__(self):
        self.stop = False
        self.old = {}

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.old[sig] = signal.signal(sig, self._handler)
        return self

    def _handler(self, signum, frame):
        if self.stop:                        # second signal: give up immediately
            raise KeyboardInterrupt
        self.stop = True
        print(f"segnale {signum}: salvo lo stato alla fine dell'aggiornamento in corso ed esco", flush=True)

    def __exit__(self, *exc):
        for sig, h in self.old.items():
            signal.signal(sig, h)


def train(cfg: PPOConfig, weights_out: Path, results_dir: Path, resume=False):
    """Train, export the actor to `weights_out`. Returns "done", or "stopped" if interrupted (state saved)."""
    import torch
    import torch.nn as nn

    torch.set_num_threads(cfg.threads)
    game = get_game(cfg.game)
    _check_spec(cfg, game)
    proto = game.protocol
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = results_dir / "state.pt"
    log_path = results_dir / "train_log.csv"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    actor = _mlp((game.obs_dim, *cfg.hidden, game.n_actions), cfg.activation, 0.01)
    mask = torch.zeros(game.n_actions)        # added to the logits: -1e9 on the actions the policy may not use
    if cfg.actions is not None:
        mask[[a for a in range(game.n_actions) if a not in cfg.actions]] = -1e9
    critic = _mlp((game.obs_dim, *cfg.critic_hidden, 1), "tanh", 1.0)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)

    T, N = cfg.n_steps, cfg.n_envs
    batch = T * N
    mb = batch // cfg.minibatches
    n_updates = cfg.total_steps // batch
    source = (f"PPO trained by small_games.ppo on {cfg.game} (git {_git_commit()}), "
              f"config {json.dumps(asdict(cfg))}")

    if resume and state_path.is_file():
        st = torch.load(state_path, weights_only=False)
        if st["config"] != asdict(cfg):
            diff = {k: (st["config"].get(k), v) for k, v in asdict(cfg).items() if st["config"].get(k) != v}
            raise SystemExit(f"--resume: la configurazione è diversa da quella salvata {diff}; usa un altro --name")
        actor.load_state_dict(st["actor"])
        critic.load_state_dict(st["critic"])
        opt.load_state_dict(st["opt"])
        envs = pickle.loads(st["envs"])
        obs, ep_ret, ep_len = st["obs"], st["ep_ret"], st["ep_len"]
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        start, t_prev, source = st["update"] + 1, st["seconds"], st["source"]
        rows = list(csv.reader(open(log_path))) if log_path.is_file() else [LOG_HEADER]
        rows = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= st["update"]]
        with open(log_path, "w", newline="") as f:          # drop log lines written after the saved state
            csv.writer(f).writerows(rows)
        log_f = open(log_path, "a", newline="")
        print(f"[{cfg.game}] riprendo da {st['update'] * batch / 1e6:.2f}M passi", flush=True)
    else:
        sticky = proto.sticky_prob if cfg.sticky is None else cfg.sticky
        envs = [game.make_env(sticky_prob=sticky, max_steps=proto.max_steps) for _ in range(N)]
        obs = np.stack([e.reset(seed=cfg.train_seed0 + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
        ep_ret, ep_len = np.zeros(N), np.zeros(N, dtype=int)
        start, t_prev = 1, 0.0
        log_f = open(log_path, "w", newline="")
        csv.writer(log_f).writerow(LOG_HEADER)
    log = csv.writer(log_f)

    def save_state(update, seconds):
        tmp = state_path.with_suffix(".tmp")
        torch.save({"config": asdict(cfg), "update": update, "seconds": seconds, "source": source,
                    "actor": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                    "envs": pickle.dumps(envs), "obs": obs.copy(), "ep_ret": ep_ret.copy(), "ep_len": ep_len.copy(),
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()}, tmp)
        os.replace(tmp, state_path)          # atomic: a crash while saving keeps the previous state

    b_obs = torch.zeros((T, N, game.obs_dim))
    b_act = torch.zeros((T, N), dtype=torch.long)
    b_logp = torch.zeros((T, N))
    b_rew = torch.zeros((T, N))
    b_done = torch.zeros((T, N))
    b_val = torch.zeros((T, N))
    finished = []                     # (return, length) of episodes completed since the last log line
    t0 = time.time()

    with _StopFlag() as flag:
        for update in range(start, n_updates + 1):
            if cfg.anneal_lr:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)

            # ---- rollout
            with torch.no_grad():
                for t in range(T):
                    o = torch.from_numpy(obs)
                    logits = actor(o) + mask
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    b_obs[t] = o
                    b_act[t] = a
                    b_logp[t] = dist.log_prob(a)
                    b_val[t] = critic(o).squeeze(-1)
                    acts = a.numpy()
                    rew = np.zeros(N, dtype=np.float32)
                    done = np.zeros(N, dtype=np.float32)
                    for i, env in enumerate(envs):
                        o2, r, term, trunc, _ = env.step(int(acts[i]))
                        ep_ret[i] += r
                        ep_len[i] += 1
                        if trunc and not term:   # step cap: bootstrap with the value of the last observation
                            r += cfg.gamma * float(critic(torch.from_numpy(o2.astype(np.float32))).item())
                        rew[i] = r
                        if term or trunc:
                            finished.append((ep_ret[i], ep_len[i]))
                            ep_ret[i], ep_len[i] = 0.0, 0
                            o2, _ = env.reset()
                            done[i] = 1.0
                        obs[i] = o2
                    b_rew[t] = torch.from_numpy(rew)
                    b_done[t] = torch.from_numpy(done)

                # ---- GAE
                next_val = critic(torch.from_numpy(obs)).squeeze(-1)
                adv = torch.zeros((T, N))
                last = torch.zeros(N)
                for t in reversed(range(T)):
                    nv = next_val if t == T - 1 else b_val[t + 1]
                    nonterm = 1.0 - b_done[t]
                    delta = b_rew[t] + cfg.gamma * nv * nonterm - b_val[t]
                    last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                    adv[t] = last
                ret = adv + b_val

            # ---- update
            f_obs, f_act, f_logp = b_obs.reshape(batch, -1), b_act.reshape(batch), b_logp.reshape(batch)
            f_adv, f_ret = adv.reshape(batch), ret.reshape(batch)
            stats = []
            for _ in range(cfg.epochs):
                perm = torch.randperm(batch)
                for s in range(0, batch, mb):
                    idx = perm[s:s + mb]
                    logits = actor(f_obs[idx]) + mask
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(f_act[idx])
                    ratio = (logp - f_logp[idx]).exp()
                    a_ = f_adv[idx]
                    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                    pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    v = critic(f_obs[idx]).squeeze(-1)
                    vl = 0.5 * ((v - f_ret[idx]) ** 2).mean()
                    ent = dist.entropy().mean()
                    loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        lr_ = logp - f_logp[idx]
                        kl = ((lr_.exp() - 1) - lr_).mean().item()
                        cf = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    stats.append((pg.item(), vl.item(), ent.item(), kl, cf))

            steps = update * batch
            seconds = t_prev + time.time() - t0
            if update % cfg.log_every == 0 or update == n_updates:
                s = np.mean(stats, axis=0)
                mr = float(np.mean([f[0] for f in finished])) if finished else float("nan")
                ml = float(np.mean([f[1] for f in finished])) if finished else float("nan")
                log.writerow([update, steps, round(seconds, 1), len(finished), round(mr, 3), round(ml, 1),
                              *[round(float(x), 5) for x in s], opt.param_groups[0]["lr"]])
                log_f.flush()
                print(f"[{cfg.game}] {steps / 1e6:7.2f}M passi  {seconds:6.0f}s  "
                      f"ritorno {mr:8.2f} ({len(finished)} ep)  entropia {s[2]:.3f}  kl {s[3]:.4f}", flush=True)
                finished = []
            if steps // cfg.save_every > (steps - batch) // cfg.save_every and update < n_updates:
                export_actor(actor, cfg, game, results_dir / f"checkpoint_{steps // 1_000_000:04d}M.json",
                             source + f", checkpoint at {steps} steps")
                save_state(update, seconds)
            if flag.stop and update < n_updates:
                save_state(update, seconds)
                log_f.close()
                print(f"[{cfg.game}] fermato a {steps / 1e6:.2f}M passi, stato in {state_path}: "
                      f"riprendi con --resume", flush=True)
                return "stopped"

    log_f.close()
    export_actor(actor, cfg, game, weights_out, source)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "config": asdict(cfg)},
               results_dir / "final.pt")
    if state_path.is_file():
        state_path.unlink()                  # finished: final.pt and the weight file are what remains
    return "done"


def _parse_sets(cfg: PPOConfig, items):
    """--set field=value pairs, each value cast like the field's current value (tuples: comma-separated)."""
    out = {}
    for item in items:
        key, _, val = item.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"--set: PPOConfig has no field {key!r}")
        cur = getattr(cfg, key)
        if val.lower() == "none":
            out[key] = None
        elif isinstance(cur, bool):
            out[key] = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, tuple) or key == "actions":
            out[key] = tuple(int(x) if x.lstrip("-").isdigit() else float(x) for x in val.split(","))
        elif isinstance(cur, int):
            out[key] = int(val)
        elif isinstance(cur, float) or cur is None:
            out[key] = float(val)
        else:
            out[key] = val
    return out


def main(cfg: PPOConfig, here: Path):
    """The CLI of experiments/<game>/ppo/train.py: train, export, score with the game's protocol."""
    ap = argparse.ArgumentParser(description=f"PPO on {cfg.game}: trains the policy, writes "
                                             f"weights/{cfg.game}/ppo.json and scores it with `evaluate`.")
    ap.add_argument("--total-steps", type=int, default=cfg.total_steps)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--name", default="ppo", help="weight file name in weights/<game>/ (default: ppo)")
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-checkpoints", action="store_true",
                    help="also score every checkpoint_*.json (200 episodes each)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from results/<name>/state.pt if it exists (else start); if training already "
                         "finished, only run the missing evaluation")
    ap.add_argument("--set", action="append", default=[], metavar="CAMPO=VALORE",
                    help="change a PPOConfig field, e.g. --set ent_coef=0.05 --set sticky=0.0 (repeatable)")
    args = ap.parse_args()
    cfg = replace(cfg, total_steps=args.total_steps, seed=args.seed, **_parse_sets(cfg, args.set))
    print(f"[{cfg.game}] config: {json.dumps(asdict(cfg))}", flush=True)

    root = here.resolve().parents[2]
    weights_out = root / "weights" / cfg.game / f"{args.name}.json"
    results_dir = here / "results" / args.name
    finished = (results_dir / "final.pt").is_file() and weights_out.is_file() \
        and not (results_dir / "state.pt").is_file()
    if args.resume and finished:
        print(f"[{cfg.game}] addestramento già finito: {weights_out.relative_to(root)}", flush=True)
    else:
        t0 = time.time()
        if train(cfg, weights_out, results_dir, resume=args.resume) == "stopped":
            return
        print(f"[{cfg.game}] training: {(time.time() - t0) / 60:.1f} min, pesi in {weights_out.relative_to(root)}",
              flush=True)
    if args.no_eval or (args.resume and (results_dir / "eval.json").is_file()):
        return
    from .evaluate import evaluate
    from .policy import MLPPolicy
    from .weights import ReferenceMLP, load_weights

    game = get_game(cfg.game)
    summary = {"final": evaluate(game, MLPPolicy(ReferenceMLP(load_weights(weights_out, game))),
                                 workers=args.eval_workers)}
    fin = summary["final"]
    print(f"[{cfg.game}] valutazione ({fin['protocol']['episodes']} episodi): "
          f"{fin['mean_return']:.2f} ± {fin['stderr']:.2f}", flush=True)
    if args.eval_checkpoints:
        for ck in sorted(results_dir.glob("checkpoint_*.json")):
            summary[ck.stem] = evaluate(game, MLPPolicy(ReferenceMLP(load_weights(ck, game))),
                                        episodes=200, workers=args.eval_workers)
            print(f"[{cfg.game}] {ck.stem}: {summary[ck.stem]['mean_return']:.2f}", flush=True)
    with open(results_dir / "eval.json", "w") as f:
        json.dump(summary, f, indent=2)
