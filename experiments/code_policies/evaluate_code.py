"""
Valuta il codice scritto dagli LLM (answers/<modello>/<gioco>/policy.py) con il protocollo di `evaluate`.

Per ogni policy.py:
1. **import:** quali moduli importa (il prompt ammette solo standard library e numpy);
2. **controlli su 2000 stati** raccolti giocando:
   - errori ed azioni non valide (nel punteggio valgono come azione 0, come dice il prompt);
   - tempo medio per chiamata;
   - **memoria tra chiamate:** gli stessi stati, in ordine inverso e con il modulo ricaricato, devono dare le stesse
     azioni (con un generatore nuovo e uguale per ogni stato, così il caso non conta);
3. **punteggio:** `small_games.evaluate.evaluate`, stessi seed, sticky actions e limite di passi delle reti.
   L'azione è quella restituita dalla funzione.

Il codice viene dagli LLM: va letto prima di eseguirlo.

    uv run python experiments/code_policies/evaluate_code.py [--model gpt-6-astra] [--games low_tide breakout]
    uv run python experiments/code_policies/evaluate_code.py --games breakout --file percorso/policy.py
"""

import argparse
import ast
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.evaluate import evaluate
from small_games.policy import policy_rng

HERE = Path(__file__).resolve().parent
GAMES = ("low_tide", "breakout", "seaquest", "asterix")


def load_module(path, name="llm_policy"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CodePolicy:
    """La funzione policy(obs, rng) di un file; errori e azioni non valide diventano 0. Si può passare ai processi."""

    def __init__(self, path, n_actions):
        self.path, self.n_actions, self._fn = str(path), n_actions, None

    def __getstate__(self):
        return {"path": self.path, "n_actions": self.n_actions, "_fn": None}

    def __call__(self, obs, rng):
        if self._fn is None:
            self._fn = load_module(self.path).policy
        try:
            a = self._fn(obs.copy(), rng)
            a = int(a)
            return a if 0 <= a < self.n_actions and a == a else 0
        except Exception:
            return 0


def imports(path):
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return sorted(found)


def collect_states(game, path, n=2000):
    """Stati raccolti giocando con la policy (episodi del protocollo, dal primo)."""
    pol = CodePolicy(path, game.n_actions)
    env = game.make_env(sticky_prob=game.protocol.sticky_prob, max_steps=game.protocol.max_steps)
    states, seed = [], game.protocol.seed0
    while len(states) < n:
        obs, _ = env.reset(seed=seed)
        rng, done = policy_rng(seed), False
        while not done and len(states) < n:
            states.append(obs.copy())
            obs, _, term, trunc, _ = env.step(pol(obs, rng))
            done = term or trunc
        seed += 1
    return states


def checks(game, path):
    states = collect_states(game, path)
    fn = load_module(path, "llm_policy_a").policy
    errors, invalid, actions, t = 0, 0, [], 0.0
    for k, s in enumerate(states):
        t0 = time.perf_counter()
        try:
            a = fn(s.copy(), np.random.default_rng(k))
            ok = isinstance(a, (int, np.integer)) and 0 <= int(a) < game.n_actions
            invalid += not ok
            actions.append(int(a) if ok else 0)
        except Exception:
            errors += 1
            actions.append(0)
        t += time.perf_counter() - t0
    fn_b = load_module(path, "llm_policy_b").policy
    memory = 0
    for k in reversed(range(len(states))):
        try:
            a = fn_b(states[k].copy(), np.random.default_rng(k))
            a = int(a) if isinstance(a, (int, np.integer)) and 0 <= int(a) < game.n_actions else 0
        except Exception:
            a = 0
        memory += a != actions[k]
    return {"states": len(states), "errors": errors, "invalid": invalid, "us_per_call": round(t / len(states) * 1e6, 1),
            "order_dependent": memory, "action_counts": np.bincount(actions, minlength=game.n_actions).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--games", nargs="+", default=list(GAMES), choices=GAMES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--file", type=Path, help="un file qualsiasi invece di answers/: serve un solo gioco in --games")
    args = ap.parse_args()
    if args.file and len(args.games) != 1:
        ap.error("con --file serve un solo gioco in --games")
    rows = []
    for name in args.games:
        path = args.file.resolve() if args.file else HERE / "answers" / args.model / name / "policy.py"
        if not path.exists():
            print(f"{name}: manca {path}")
            continue
        game = get_game(name)
        print(f"{name}: import {imports(path)}", flush=True)
        c = checks(game, path)
        print(f"  controlli su {c['states']} stati: errori {c['errors']}, azioni non valide {c['invalid']}, "
              f"{c['us_per_call']} µs per chiamata, azioni diverse con l'ordine inverso {c['order_dependent']}, "
              f"azioni {c['action_counts']}", flush=True)
        s = evaluate(game, CodePolicy(path, game.n_actions), workers=args.workers)
        print(f"  punteggio {s['mean_return']:.2f} ± {s['stderr']:.2f} su {s['protocol']['episodes']} episodi "
              f"(lunghezza media {s['mean_length']:.0f}, {s['seconds']:.0f} s)", flush=True)
        result = {"game": name, "model": args.model, "imports": imports(path), "checks": c, "score": s}
        json.dump(result, open(path.with_name(path.stem + "_score.json" if args.file else "score.json"), "w"), indent=1)
        rows.append(result)
    if rows:
        print("\n| gioco | punteggio | µs/chiamata | errori | dipende dall'ordine |\n|---|---|---|---|---|")
        for r in rows:
            s, c = r["score"], r["checks"]
            print(f"| {r['game']} | {s['mean_return']:.1f} ± {s['stderr']:.1f} | {c['us_per_call']} | "
                  f"{c['errors'] + c['invalid']} | {c['order_dependent']} |")


if __name__ == "__main__":
    sys.exit(main())
