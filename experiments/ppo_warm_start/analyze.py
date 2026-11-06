"""
PPO che parte dai pesi scritti a mano: migliora? quanto si muovono i pesi? quanto tempo si risparmia?

Confronta ogni corsa che parte da una rete scritta a mano (experiments/ppo_runs.py: <gioco>/ppo_from_<rete>,
e su Low Tide anche _cw, con 2M passi di solo critico all'inizio) con la corsa da zero di riferimento: per i giochi
MinAtar la migliore delle tre (ppo_ent05, stessa configurazione e stessa architettura), per Low Tide quella con la
stessa architettura della rete a mano (ppo_arch_<rete>).

    uv run --group ppo --group analysis python experiments/ppo_warm_start/analyze.py
    uv run --group ppo --group analysis python experiments/ppo_warm_start/analyze.py --workers 8

Scrive results/: summary.json (tutti i numeri), tables.md (le tabelle del README), curves.png, scores.json (cache
delle valutazioni: si rifanno solo quelle che mancano).
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from small_games.analysis import collect
from small_games.core import get_game
from small_games.evaluate import evaluate
from small_games.policy import MLPPolicy, baselines
from small_games.weights import ReferenceMLP, load_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
GAMES = ("low_tide", "breakout", "seaquest", "asterix")
HAND = {"low_tide": ("claude", "gpt"), "breakout": ("paper_astra",), "seaquest": ("paper_astra",),
        "asterix": ("paper_astra",)}
COMPARISONS = [
    dict(game="breakout", hand="paper_astra", warm=("ppo_from_astra",), scratch="ppo_ent05"),
    dict(game="seaquest", hand="paper_astra", warm=("ppo_from_astra",), scratch="ppo_ent05"),
    dict(game="asterix", hand="paper_astra", warm=("ppo_from_astra",), scratch="ppo_ent05"),
    dict(game="low_tide", hand="gpt", warm=("ppo_from_gpt", "ppo_from_gpt_cw"), scratch="ppo_arch_gpt",
         extra="ppo_100m"),
    dict(game="low_tide", hand="claude", warm=("ppo_from_claude", "ppo_from_claude_cw"), scratch="ppo_arch_claude",
         extra="ppo_100m"),
]
CKPT_EPISODES = 200
SMOOTH_STEPS = 1_000_000          # window of the training-return curve
LABEL = {"paper_astra": "Astra", "gpt": "GPT", "claude": "Claude"}


# ------------------------------------------------------------------------------------------------ data access
def run_dir(game, run):
    return ROOT / "experiments" / game / "ppo" / "results" / run


def run_config(game, run):
    src = json.load(open(ROOT / "weights" / game / f"{run}.json"))["source"]
    return json.loads(re.search(r"config (\{.*\})", src).group(1))


def read_log(game, run):
    rows = list(csv.DictReader(open(run_dir(game, run) / "train_log.csv")))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(math.nan)
        out[k] = np.array(vals)
    return out


def smoothed_return(log):
    """Episode-weighted mean training return over a window of SMOOTH_STEPS (centred on nothing: trailing)."""
    steps, ret, n = log["steps"], log["mean_return"], log["episodes"]
    per_line = steps[1] - steps[0] if len(steps) > 1 else steps[0]
    w = max(1, int(round(SMOOTH_STEPS / per_line)))
    ok = np.isfinite(ret) & (n > 0)
    num = np.where(ok, ret * n, 0.0)
    den = np.where(ok, n, 0.0)
    cn, cd = np.cumsum(num), np.cumsum(den)
    sn = cn - np.concatenate([np.zeros(w), cn[:-w]])
    sd = cd - np.concatenate([np.zeros(w), cd[:-w]])
    with np.errstate(invalid="ignore"):
        return np.where(sd > 0, sn / np.maximum(sd, 1e-12), np.nan)


def first_step(steps, curve, level, after=0):
    idx = np.flatnonzero((curve >= level) & (steps >= after))
    return int(steps[idx[0]]) if len(idx) else None


def checkpoints(game, run):
    """{steps: path} of the saved actors (checkpoint_<N>M.json, plus the initial one when it was saved)."""
    out = {}
    for p in run_dir(game, run).glob("checkpoint_*M.json"):
        out[int(p.stem.split("_")[1][:-1]) * 1_000_000] = p
    return dict(sorted(out.items()))


def initial_actor(game, run):
    """The run's starting actor: checkpoint_0000M.json, or (older runs) rebuilt from the config and the seed."""
    p = run_dir(game, run) / "checkpoint_0000M.json"
    if p.is_file():
        return p
    import torch

    from small_games.ppo import PPOConfig, _mlp, export_actor, hidden_activations
    cfg = run_config(game, run)
    cfg = PPOConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()})
    g = get_game(game)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    actor = _mlp((g.obs_dim, *cfg.hidden, g.n_actions), hidden_activations(cfg), 0.01)
    out = OUT / "init" / f"{game}_{run}.json"
    if not out.is_file():
        export_actor(actor, cfg, g, out, f"initial actor of {game}/{run}, rebuilt from its config and seed")
    return out


def last_actor(game, run):
    """The final weight file of a finished run, else its latest checkpoint."""
    p = ROOT / "weights" / game / f"{run}.json"
    return p if final_eval(game, run) and p.is_file() else list(checkpoints(game, run).values())[-1]


def flat(layers):
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b, _ in layers])


# ------------------------------------------------------------------------------------------------ scores (cached)
class Scores:
    def __init__(self, workers):
        self.path = OUT / "scores.json"
        self.cache = json.load(open(self.path)) if self.path.is_file() else {}
        self.workers = workers

    def get(self, game, key, policy_fn, episodes=None):
        k = f"{game}|{key}|{episodes or 'protocol'}"
        if k not in self.cache:
            s = evaluate(get_game(game), policy_fn(), episodes=episodes, workers=self.workers)
            self.cache[k] = {"mean": s["mean_return"], "stderr": s["stderr"], "episodes": s["protocol"]["episodes"]}
            print(f"  {k}: {s['mean_return']:.2f} ± {s['stderr']:.2f}", flush=True)
            json.dump(self.cache, open(self.path, "w"), indent=1)
        return self.cache[k]

    def net(self, game, path, episodes=None):
        path = Path(path).resolve()
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        return self.get(game, rel, lambda: MLPPolicy(ReferenceMLP(load_weights(path, get_game(game)))), episodes)

    def baseline(self, game, name):
        return self.get(game, f"baseline:{name}", lambda: baselines(get_game(game))[name])


def final_eval(game, run):
    p = run_dir(game, run) / "eval.json"
    if not p.is_file():
        return None
    f = json.load(open(p))["final"]
    return {"mean": f["mean_return"], "stderr": f["stderr"], "episodes": f["protocol"]["episodes"]}


# ------------------------------------------------------------------------------------------------ probes
def probe_stats(game, label):
    d = OUT / "probe" / game / label
    if not (d / "done").is_file():
        return None
    log = {k: v for k, v in read_log_path(d / "train_log.csv").items()}
    n = len(log["steps"])
    tail = slice(n - max(1, n // 4), n)
    out = {k: float(np.nanmean(log[k][tail])) for k in
           ("actor_grad", "actor_grad_full", "pg_grad_full", "ent_grad_full", "critic_grad", "value_loss", "entropy",
            "mean_return")}
    out["actor_grad_full_first"] = float(log["actor_grad_full"][0])
    out["pg_grad_full_first"] = float(log["pg_grad_full"][0])
    init = (d / "done").read_text().strip()
    g = get_game(game)
    if init != "None":
        out["theta_norm"] = float(np.linalg.norm(flat(load_weights(ROOT / init, g))))
    else:                                   # PPO's random initial actor, rebuilt from the probe's config and seed
        import torch
        from dataclasses import replace

        from probe import base_config, probes
        from small_games.ppo import _mlp, hidden_activations
        cfg = replace(base_config(game), **probes()[game, label][0])
        torch.manual_seed(cfg.seed)
        actor = _mlp((g.obs_dim, *cfg.hidden, g.n_actions), hidden_activations(cfg), 0.01)
        out["theta_norm"] = float(torch.cat([p.detach().reshape(-1) for p in actor.parameters()]).norm())
    return out


def read_log_path(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else math.nan for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------------------------------------ one comparison
def layer_rel(a, b):
    return [float(np.linalg.norm(flat([lb]) - flat([la])) / max(np.linalg.norm(flat([la])), 1e-12))
            for la, lb in zip(a, b)]


def agreement(game, a_layers, b_layers, on_layers, episodes=20):
    X, _ = collect(game, on_layers, episodes, game.protocol.seed0 + 7000, max_states=40_000)
    A = np.argmax(np.stack([ReferenceMLP(a_layers).logits(x) for x in X]), axis=1)
    B = np.argmax(np.stack([ReferenceMLP(b_layers).logits(x) for x in X]), axis=1)
    return float(np.mean(A == B))


SAME_WEIGHTS = 1e-4                # a checkpoint this close to the start (max |Δw|) is the starting policy


def ckpt_curve(game, run, scores):
    """{steps: score} of every saved actor (every 5M steps, CKPT_EPISODES episodes), from the start to the final
    weights. A checkpoint whose weights are the starting ones (max |Δw| < SAME_WEIGHTS) gets the starting score
    without being played again (Breakout from Astra: the weights move by 1e-5 in 200M steps)."""
    g = get_game(game)
    pts = dict(checkpoints(game, run))
    pts.setdefault(0, initial_actor(game, run))
    fin = ROOT / "weights" / game / f"{run}.json"
    if final_eval(game, run) and fin.is_file():
        pts[int(run_config(game, run)["total_steps"])] = fin
    start = flat(load_weights(pts[0], g))
    out = {}
    for s, p in sorted(pts.items()):
        same = s > 0 and np.max(np.abs(flat(load_weights(p, g)) - start)) < SAME_WEIGHTS
        out[s] = out[0] if same else scores.net(game, p, CKPT_EPISODES)
    return out


def curve_arrays(ck):
    s = np.array(sorted(ck), dtype=float)
    return s, np.array([ck[int(k)]["mean"] for k in s])


def grad_stats(log):
    """Full-batch actor-gradient norm during a run: first update (random critic), median of the first 1M steps and
    of the last 10M, and the fraction of minibatches cut by the gradient clipping in the same windows."""
    steps = log["steps"]
    if "actor_grad_full" not in log or not np.isfinite(log["actor_grad_full"]).any():
        return None
    first, last = steps <= 1_000_000, steps >= steps[-1] - 10_000_000
    return {"first": float(log["actor_grad_full"][0]),
            "first1M": float(np.nanmedian(log["actor_grad_full"][first])),
            "last10M": float(np.nanmedian(log["actor_grad_full"][last])),
            "pg_first1M": float(np.nanmedian(log["pg_grad_full"][first])),
            "ent_first1M": float(np.nanmedian(log["ent_grad_full"][first])),
            "clipped_first1M": float(np.nanmean(log["clipped"][first])),
            "clipped_last10M": float(np.nanmean(log["clipped"][last]))}


def compare(c, scores):
    game, g = c["game"], get_game(c["game"])
    hand_path = ROOT / "weights" / game / f"{c['hand']}.json"
    hand_layers = load_weights(hand_path, g)
    hand_score = scores.net(game, hand_path)
    hand = hand_score["mean"]
    sc = c["scratch"]
    sc_log = read_log(game, sc)
    sc_final = final_eval(game, sc)
    sc_sec_per_step = sc_log["seconds"][-1] / sc_log["steps"][-1]
    sc_init = load_weights(initial_actor(game, sc), g)
    sc_last = load_weights(last_actor(game, sc), g)
    sc_ck = ckpt_curve(game, sc, scores)
    sc_s, sc_v = curve_arrays(sc_ck)
    level = 0.95 * sc_final["mean"]
    res = {"game": game, "hand": c["hand"], "hand_score": hand_score, "scratch": sc, "scratch_final": sc_final,
           "scratch_steps": int(sc_log["steps"][-1]), "scratch_hours": float(sc_log["seconds"][-1] / 3600),
           "scratch_sec_per_step": float(sc_sec_per_step),
           "scratch_reach_hand": first_step(sc_s, sc_v, hand), "level95": level,
           "scratch_reach_level": first_step(sc_s, sc_v, level),
           "scratch_move": float(np.linalg.norm(flat(sc_last) - flat(sc_init))),
           "scratch_move_rel": float(np.linalg.norm(flat(sc_last) - flat(sc_init)) / np.linalg.norm(flat(sc_init))),
           "scratch_checkpoints": sc_ck, "scratch_grad": grad_stats(sc_log), "warm": {}}
    if c.get("extra") and final_eval(game, c["extra"]):
        res["extra"] = {"run": c["extra"], "final": final_eval(game, c["extra"]),
                        "checkpoints": ckpt_curve(game, c["extra"], scores)}
    for run in c["warm"]:
        log = read_log(game, run)
        steps = log["steps"]
        curve = smoothed_return(log)
        init = load_weights(run_dir(game, run) / "checkpoint_0000M.json", g)
        fin = final_eval(game, run)
        last = load_weights(last_actor(game, run), g)
        ck = ckpt_curve(game, run, scores)
        s, v = curve_arrays(ck)
        low = int(np.argmin(v))
        # the training curve sees what the 5M-spaced checkpoints miss (a collapse in the first few million steps);
        # its first 1M steps are left out: the trailing window is not full yet and holds only the episodes that
        # ended early, the short ones
        early = (steps >= SMOOTH_STEPS) & (steps <= 0.1 * steps[-1])
        dip_i = int(np.nanargmin(np.where(early, curve, np.inf)))
        best = int(np.argmax(v))
        w = {"start_ckpt": ck[0], "final": fin, "steps": int(steps[-1]), "hours": float(log["seconds"][-1] / 3600),
             "checkpoints": ck,
             "ck_min": float(v[low]), "ck_min_step": int(s[low]),
             "ck_best": float(v[best]), "ck_best_step": int(s[best]),
             "back_to_hand_step": (first_step(s, v, hand, after=s[low]) if v[low] < hand else 0),
             "improve10_step": first_step(s, v, 1.1 * hand),
             "train_dip": float(curve[dip_i]), "train_dip_step": int(steps[dip_i]),
             "gain": fin["mean"] - hand, "gain_rel": (fin["mean"] - hand) / hand,
             "warm_reach_level": first_step(s, v, level),
             "scratch_reach_final": first_step(sc_s, sc_v, fin["mean"]),
             "theta0_norm": float(np.linalg.norm(flat(init))),
             "move": float(np.linalg.norm(flat(last) - flat(init))),
             "move_rel": float(np.linalg.norm(flat(last) - flat(init)) / np.linalg.norm(flat(init))),
             "move_rel_layers": layer_rel(init, last),
             "grad": grad_stats(log),
             "entropy_first": float(log["entropy"][0]), "entropy_last": float(np.nanmean(log["entropy"][-50:])),
             }
        zeros = flat(hand_layers) == 0
        lw = flat(last)
        w["hand_zeros_now_nonzero"] = float(np.mean(np.abs(lw[zeros]) > 1e-3)) if zeros.any() else 0.0
        w["agree_on_hand_states"] = agreement(g, last, hand_layers, hand_layers)
        w["agree_on_own_states"] = agreement(g, last, hand_layers, last)
        if res["scratch_reach_level"] is not None and w["warm_reach_level"] is not None:
            w["saved_steps_level"] = res["scratch_reach_level"] - w["warm_reach_level"]
            w["saved_hours_level"] = w["saved_steps_level"] * sc_sec_per_step / 3600
        res["warm"][run] = w
    return res


# ------------------------------------------------------------------------------------------------ summary table
def all_scores(scores):
    rows = []
    for game in GAMES:
        g = get_game(game)
        for b in baselines(g):
            rows.append((game, b, "baseline", "", scores.baseline(game, b)))
        for h in HAND[game]:
            rows.append((game, h, "scritta a mano", "", scores.net(game, ROOT / "weights" / game / f"{h}.json")))
        for d in sorted((ROOT / "experiments" / game / "ppo" / "results").iterdir()):
            fin = final_eval(game, d.name)
            if fin is None or not (ROOT / "weights" / game / f"{d.name}.json").is_file():
                continue
            cfg = run_config(game, d.name)
            kind = "PPO dai pesi a mano" if cfg.get("init") else "PPO da zero"
            arch = "-".join(map(str, cfg["hidden"]))
            acts = cfg.get("activations") or [cfg["activation"]]
            desc = (f"{arch} {'/'.join(acts)}, ent {cfg['ent_coef']}, lr {cfg['lr']}, "
                    f"{cfg['total_steps'] / 1e6:.0f}M passi" + (", 2M passi di solo critico"
                                                               if cfg.get("critic_warmup") else ""))
            rows.append((game, d.name, kind, desc, fin))
    return rows


def fmt_steps(s):
    if s is None:
        return "mai"
    if s == 0:
        return "0"
    return f"{s / 1e6:.1f}M" if s < 10_000_000 else f"{s / 1e6:.0f}M"


def tables(rows, comps, probes):
    out = ["## Punteggi di tutti i modelli\n",
           "Protocollo completo di ogni gioco (1000 episodi MinAtar, 500 Low Tide), media ± errore standard.\n"]
    for game in GAMES:
        out += [f"\n### {game}\n", "| modello | tipo | configurazione | punteggio |", "|---|---|---|---|"]
        for gm, name, kind, desc, s in sorted((r for r in rows if r[0] == game), key=lambda r: -r[4]["mean"]):
            out.append(f"| `{name}` | {kind} | {desc} | {s['mean']:.1f} ± {s['stderr']:.1f} |")
    out += ["\n## Partire dai pesi scritti a mano: quanto migliora\n",
            f"Checkpoint ogni 5M passi valutati su {CKPT_EPISODES} episodi; inizio e fine con il protocollo completo. "
            "La curva di addestramento (media sugli ultimi 1M passi) vede anche quello che succede fra due checkpoint, "
            "ma è più rumorosa: su Breakout la policy non cambia e oscilla comunque di ±15%.\n",
            "| gioco | corsa | rete a mano | minimo della curva di addestramento nel primo 10% (passo) | "
            "checkpoint peggiore (passo) | "
            "torna al livello della rete a mano a | +10% a | miglior checkpoint (passo) | fine | fine − rete a mano | "
            "da zero, stessa arch. |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        for run, w in c["warm"].items():
            hand = c["hand_score"]["mean"]
            out.append(f"| {c['game']} | `{run}` | {LABEL[c['hand']]} {hand:.1f} | "
                       f"{w['train_dip']:.1f} ({fmt_steps(w['train_dip_step'])}) | "
                       f"{w['ck_min']:.1f} ({fmt_steps(w['ck_min_step'])}) | {fmt_steps(w['back_to_hand_step']) if w['ck_min'] < hand else 'non scende'} | "
                       f"{fmt_steps(w['improve10_step'])} | {w['ck_best']:.1f} ({fmt_steps(w['ck_best_step'])}) | "
                       f"{w['final']['mean']:.1f} ± {w['final']['stderr']:.1f} | {w['gain']:+.1f} ({w['gain_rel']:+.0%}) | "
                       f"{c['scratch_final']['mean']:.1f} ± {c['scratch_final']['stderr']:.1f} |")
    out += ["\n## Partire dai pesi scritti a mano: quanto tempo si risparmia\n",
            "Soglia: 95% del punteggio finale della corsa da zero; passi al primo checkpoint che la supera. Le ore "
            "sono quelle della corsa da zero su questa macchina (1 core per corsa).\n",
            "| gioco | corsa | soglia | da mano la supera a | da zero la supera a | risparmio | "
            "da zero raggiunge la rete a mano a | da zero raggiunge il finale di questa corsa a |",
            "|---|---|---|---|---|---|---|---|"]
    for c in comps:
        for run, w in c["warm"].items():
            if w.get("saved_steps_level") is not None:
                saved = (f"{fmt_steps(w['saved_steps_level'])} passi ≈ {w['saved_hours_level']:.1f} h"
                         if w["saved_steps_level"] > 0 else "nessuno")
            else:
                saved = "nessuno: non ci arriva"
            out.append(f"| {c['game']} | `{run}` | {c['level95']:.1f} | {fmt_steps(w['warm_reach_level'])} | "
                       f"{fmt_steps(c['scratch_reach_level'])} | {saved} | {fmt_steps(c['scratch_reach_hand'])} | "
                       f"{fmt_steps(w['scratch_reach_final'])} |")
    out += ["\n## Il gradiente durante l'addestramento\n",
            "Norma del gradiente dell'attore sul batch intero (8192 passi), prima del clipping; mediane. Il clipping "
            "taglia la norma congiunta di attore e critico a 0.5.\n",
            "| gioco | corsa | primo aggiornamento (critico casuale) | primi 1M passi | di cui policy gradient / "
            "entropia | ultimi 10M passi | minibatch tagliati: primi 1M / ultimi 10M |",
            "|---|---|---|---|---|---|---|"]
    seen = set()
    for c in comps:
        for run, gs in [*((r, w["grad"]) for r, w in c["warm"].items()), (c["scratch"], c["scratch_grad"])]:
            if (c["game"], run) in seen:
                continue
            seen.add((c["game"], run))
            if gs is None:
                out.append(f"| {c['game']} | `{run}` | non registrato (corsa precedente: vedi la tabella a pesi fissi) "
                           "| | | | |")
                continue
            out.append(f"| {c['game']} | `{run}` | {gs['first']:.3g} | {gs['first1M']:.3g} | "
                       f"{gs['pg_first1M']:.3g} / {gs['ent_first1M']:.3g} | {gs['last10M']:.3g} | "
                       f"{gs['clipped_first1M']:.0%} / {gs['clipped_last10M']:.0%} |")
    out += ["\n## Quanto si muovono i pesi\n",
            "| gioco | corsa | ‖θ₀‖ | ‖θ_fine − θ₀‖ | relativo | per strato | da zero: ‖θ_fine − θ₀‖ (relativo) | "
            "zeri della rete a mano ora ≠ 0 | stessa mossa della rete a mano (sui suoi stati / sui propri) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        for run, w in c["warm"].items():
            ag = (f"{w['agree_on_hand_states']:.0%} / {w['agree_on_own_states']:.0%}"
                  if "agree_on_hand_states" in w else "-")
            out.append(f"| {c['game']} | `{run}` | {w['theta0_norm']:.1f} | {w['move']:.3g} | {w['move_rel']:.2%} | "
                       f"{' · '.join(f'{x:.1%}' for x in w['move_rel_layers'])} | {c['scratch_move']:.3g} ({c['scratch_move_rel']:.0%}) | "
                       f"{w['hand_zeros_now_nonzero']:.0%} | {ag} |")
    out += ["\n## Il gradiente a pesi fissi\n",
            "Attore fermo, critico adattato per 2M passi; media dell'ultimo quarto. Norme prima del clipping "
            "(max_grad_norm 0.5 su attore + critico).\n",
            "| gioco | rete | entropia | ‖θ‖ | ‖g‖ minibatch | ‖g‖ batch intero | di cui policy gradient | "
            "di cui entropia | ‖g‖/‖θ‖ | batch intero col critico casuale |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (game, label), p in probes.items():
        th = p.get("theta_norm")
        out.append(f"| {game} | `{label}` | {p['entropy']:.3f} | {th:.1f} | {p['actor_grad']:.3g} | "
                   f"{p['actor_grad_full']:.3g} | {p['pg_grad_full']:.3g} | {p['ent_grad_full']:.3g} | "
                   f"{p['actor_grad_full'] / th:.2g} | {p['actor_grad_full_first']:.3g} |" if th else
                   f"| {game} | `{label}` | {p['entropy']:.3f} | - | {p['actor_grad']:.3g} | "
                   f"{p['actor_grad_full']:.3g} | {p['pg_grad_full']:.3g} | {p['ent_grad_full']:.3g} | - | "
                   f"{p['actor_grad_full_first']:.3g} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ plot
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def binned_median(steps, values, width=SMOOTH_STEPS):
    """Median of `values` in consecutive windows of `width` steps (the per-update gradient is very noisy)."""
    ok = np.isfinite(values) & (values > 0)
    b = (steps // width).astype(int)
    keys = np.unique(b[ok])
    return (np.array([steps[ok & (b == k)].mean() for k in keys]),
            np.array([np.median(values[ok & (b == k)]) for k in keys]))


def plot(comps, probes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(comps), 3, figsize=(16, 3.6 * len(comps)), squeeze=False)
    for row, c in zip(axes, comps):
        game = c["game"]
        runs = [(r, f"da {LABEL[c['hand']]}" + (" + 2M solo critico" if r.endswith("_cw") else ""))
                for r in c["warm"]] + [(c["scratch"], "da zero, stessa arch.")]
        if c.get("extra"):
            runs.append((c["extra"]["run"], "da zero, 64-64"))
        g = get_game(game)
        for i, (run, lab) in enumerate(runs):
            log = read_log(game, run)
            s = log["steps"] / 1e6
            row[0].plot(s, smoothed_return(log), color=COLORS[i], lw=0.8, alpha=0.35)
            ck = (c["warm"][run]["checkpoints"] if run in c["warm"] else
                  c["scratch_checkpoints"] if run == c["scratch"] else c["extra"]["checkpoints"])
            ks, kv = curve_arrays(ck)
            row[0].plot(ks / 1e6, kv, "o-", color=COLORS[i], lw=1.8, ms=3, label=lab)
            if "dist_init" in log and np.isfinite(log["dist_init"]).any():
                d = np.where(log["dist_init"] > 0, log["dist_init"], np.nan)
                row[1].plot(s, d, color=COLORS[i], lw=1.8, label=lab)
                bs, bv = binned_median(log["steps"], log["actor_grad_full"])
                row[2].plot(bs / 1e6, bv, color=COLORS[i], lw=1.6, label=lab)
            elif run == c["scratch"]:      # older run: distance from its checkpoints, gradient from the fixed-weight probes
                th0 = flat(load_weights(initial_actor(game, run), g))
                cks = checkpoints(game, run)
                row[1].plot([k / 1e6 for k in cks], [np.linalg.norm(flat(load_weights(p, g)) - th0) for p in cks.values()],
                            "o", color=COLORS[i], ms=4, label=lab + " (checkpoint)")
                pts = [(0, "random"), (5, "ppo_5M"), (50, "ppo_50M"), (200, "ppo_final")]
                pts = [(x, probes[game, k]["actor_grad_full"]) for x, k in pts if (game, k) in probes]
                row[2].plot(*zip(*pts), "o", color=COLORS[i], ms=6, label=lab + " (a pesi fissi)")
        row[0].axhline(c["hand_score"]["mean"], color="#52514e", ls="--", lw=1.2,
                       label=f"rete a mano ({LABEL[c['hand']]}) {c['hand_score']['mean']:.0f}")
        row[0].set_title(f"{game}: punteggio (linea: checkpoint; sottile: addestramento)", fontsize=10, loc="left")
        row[1].set_title("distanza dei pesi dalla partenza ‖θ − θ₀‖", fontsize=10, loc="left")
        row[2].set_title("‖gradiente dell'attore‖, batch intero (mediana su 1M passi)", fontsize=10, loc="left")
        row[1].set_yscale("log")
        row[2].set_yscale("log")
        for ax in row:
            ax.set_xlabel("milioni di passi", fontsize=9)
            ax.grid(alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
        legend = dict(fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none", loc="best")
        row[1].legend(*row[0].get_legend_handles_labels(), **legend)    # same colours across the row
        if c["scratch_grad"] is None:
            row[2].legend(*[[x for x, lab in zip(*row[2].get_legend_handles_labels()) if "pesi fissi" in lab],
                            [lab for lab in row[2].get_legend_handles_labels()[1] if "pesi fissi" in lab]], **legend)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scores = Scores(args.workers)
    rows = all_scores(scores)
    comps = [compare(c, scores) for c in COMPARISONS]
    from probe import probes as probe_list
    probes = {k: p for k in probe_list() if (p := probe_stats(*k)) is not None}
    summary = {"scores": [dict(game=r[0], model=r[1], kind=r[2], config=r[3], **r[4]) for r in rows],
               "comparisons": comps, "probes": {f"{k[0]}/{k[1]}": v for k, v in probes.items()}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
    (OUT / "tables.md").write_text(tables(rows, comps, probes))
    plot(comps, probes, OUT / "curves.png")
    print((OUT / "tables.md").read_text())
