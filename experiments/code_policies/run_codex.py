"""
Manda un prompt di prompts/ a un modello con `codex exec`, senza tool, e salva la risposta.

Le condizioni sono quelle dell'articolo: una conversazione nuova, una sola risposta, nessun tool.

**Come si tolgono i tool (Codex 0.164).**
1. **Configurazione.** config.toml = codex_no_tools.toml, in una CODEX_HOME temporanea. Contiene solo quello, le
   istruzioni neutre dell'articolo ("Answer the user's request.") e un link alle credenziali di ~/.codex. Niente
   server MCP, skill, plugin o AGENTS.md personali. Il modello lavora in una cartella temporanea vuota.
2. **Catalogo dei modelli.** Code mode, agenti, apply_patch, shell, ricerca e orologio arrivano dal catalogo
   (`codex debug models`). Se ne scrive una copia in cui, per il modello scelto:
   - tool_mode = "direct", shell_type = "disabled";
   - tolti multi_agent_version, apply_patch_tool_type, web_search_tool_type, experimental_supported_tools e il
     messaggio sulle "collaboration modes".
3. **Controllo prima di mandare.** `codex debug prompt-input` mostra cosa vedrebbe il modello senza chiamarlo. Se
   c'è anche un solo tool, o un messaggio diverso dalle istruzioni neutre e dal prompt, lo script si ferma.
4. **Controllo dopo.** Si contano gli eventi della risposta: `tool_free` è true solo se non c'è nessuna chiamata
   a un tool.

Salva in answers/<modello>/<gioco>/:
- answer.md, l'ultimo messaggio del modello;
- policy.py, l'ultimo blocco ```python della risposta;
- events.jsonl, gli eventi grezzi di Codex;
- run.json, il comando, il tempo, i token e i due controlli.

    uv run python experiments/code_policies/run_codex.py low_tide --dry-run   # solo il controllo, nessuna chiamata
    uv run python experiments/code_policies/run_codex.py low_tide             # gpt-6-astra, xhigh
    uv run python experiments/code_policies/run_codex.py --check              # chiamata breve: quali tool vede?
"""

import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES = ("low_tide", "breakout", "seaquest", "asterix")
INSTRUCTIONS = "Answer the user's request."
MESSAGE_ITEMS = {"reasoning", "agent_message", "error"}  # ogni altro tipo di evento è un tool
CHECK_PROMPT = ("List the names of every tool or function you can call in this conversation, one per line. "
                "If you cannot call any, answer exactly NONE. Do not call any tool.")


def find_codex():
    found = shutil.which("codex")
    if found:
        return found
    # l'estensione di VS Code porta con sé il binario: la versione più recente
    found = sorted(glob.glob(os.path.expanduser("~/.vscode/extensions/openai.chatgpt-*/bin/*/codex")))
    if not found:
        raise SystemExit("codex non trovato: installalo o passa --codex /percorso/codex")
    return found[-1]


def no_tools_catalog(codex, model, path):
    """Il catalogo dei modelli di Codex, con i tool del modello scelto tolti."""
    raw = subprocess.run([codex, "debug", "models"], capture_output=True, text=True, check=True).stdout
    catalog = json.loads(raw)
    entry = next((m for m in catalog["models"] if m.get("slug") == model), None)
    if entry is None:
        raise SystemExit(f"modello {model!r} non nel catalogo di Codex")
    entry["tool_mode"] = "direct"
    entry["shell_type"] = "disabled"
    entry["experimental_supported_tools"] = []
    entry["supports_search_tool"] = False
    for key in ("multi_agent_version", "apply_patch_tool_type", "web_search_tool_type"):
        entry.pop(key, None)
    entry.get("model_messages", {}).pop("collaboration_modes", None)
    path.write_text(json.dumps(catalog))


def visible_input(codex, env, work, config_args, prompt):
    """Cosa vede il modello (codex debug prompt-input): nomi dei tool e messaggi."""
    p = subprocess.run([codex, "debug", "prompt-input", *config_args, prompt], capture_output=True, text=True,
                       env=env, cwd=work)
    if p.returncode != 0:
        raise SystemExit(f"codex debug prompt-input non riesce:\n{p.stderr[-2000:]}")
    tools, messages = [], []

    def walk(ts, prefix=""):
        for t in ts:
            name = prefix + t.get("name", t.get("type", "?"))
            walk(t["tools"], name + ".") if "tools" in t else tools.append(name)

    for item in json.loads(p.stdout):
        if "tools" in item:
            walk(item["tools"])
        else:
            text = "".join(c.get("text", "") for c in item.get("content") or [])
            messages.append({"role": item.get("role"), "text": text})
    return tools, messages


def audit(events_path):
    item_types, tools, usage, errors = collections.Counter(), collections.Counter(), {}, []
    for line in open(events_path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed":
            kind = ev.get("item", {}).get("type", "?")
            item_types[kind] += 1
            if kind not in MESSAGE_ITEMS:
                tools[kind] += 1
            if kind == "error":
                errors.append(ev["item"].get("message", ""))
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage", {})
    return {"item_types": dict(item_types), "tool_items": dict(tools), "tool_free": not tools,
            "usage": usage, "errors": errors}


def last_python_block(text):
    blocks = re.findall(r"```python[^\n]*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", choices=GAMES)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="xhigh", help="model_reasoning_effort (l'articolo: xhigh)")
    ap.add_argument("--codex", default=None, help="percorso del binario codex")
    ap.add_argument("--dry-run", action="store_true", help="solo il controllo di cosa vede il modello")
    ap.add_argument("--check", action="store_true", help="chiamata breve: chiede al modello quali tool vede")
    args = ap.parse_args()
    if not args.check and not args.game:
        ap.error("serve il gioco (o --check)")

    codex = args.codex or find_codex()
    if args.check:
        prompt, out, effort = CHECK_PROMPT, HERE / "answers" / "_check" / args.model, "low"
    else:
        prompt = (HERE / "prompts" / f"{args.game}.txt").read_text()
        out, effort = HERE / "answers" / args.model / args.game, args.effort

    with tempfile.TemporaryDirectory(dir=Path.home() / ".cache") as home, tempfile.TemporaryDirectory() as work:
        home = Path(home)
        shutil.copy(HERE / "codex_no_tools.toml", home / "config.toml")
        (home / "instructions.md").write_text(INSTRUCTIONS + "\n")
        os.symlink(os.path.expanduser("~/.codex/auth.json"), home / "auth.json")
        no_tools_catalog(codex, args.model, home / "catalog.json")
        env = {**os.environ, "CODEX_HOME": str(home)}
        config_args = ["-c", f'model="{args.model}"',
                       "-c", f'model_reasoning_effort="{effort}"',
                       "-c", f'model_instructions_file="{home / "instructions.md"}"',
                       "-c", f'model_catalog_json="{home / "catalog.json"}"',
                       "-c", 'model_reasoning_summary="detailed"']

        # 1. prima di mandare: il modello non deve vedere nessun tool, e solo le istruzioni neutre e il prompt
        tools, messages = visible_input(codex, env, work, config_args, prompt)
        expected = [{"role": "developer", "text": INSTRUCTIONS}, {"role": "user", "text": prompt}]
        clean = not tools and messages == expected
        print(f"cosa vede il modello: {len(tools)} tool {tools if tools else ''}; "
              f"messaggi {[m['role'] for m in messages]}; "
              f"{'solo istruzioni neutre e prompt' if messages == expected else 'MESSAGGI IN PIÙ'}")
        if not clean:
            for m in messages:
                print(f"--- {m['role']}: {m['text'][:500]}")
            raise SystemExit("fermato: il modello vedrebbe tool o messaggi in più")
        if args.dry_run:
            return

        # 2. la risposta
        out.mkdir(parents=True, exist_ok=True)
        cmd = [codex, "exec", "--strict-config", "--skip-git-repo-check", "--ephemeral", "--json", "-C", work,
               *config_args, "-o", str(out / "answer.md"), "-"]
        t0 = time.time()
        with open(out / "events.jsonl", "w") as ev, open(out / "stderr.log", "w") as err:
            rc = subprocess.run(cmd, input=prompt, text=True, stdout=ev, stderr=err, env=env).returncode
        wall = time.time() - t0

    record = {"rc": rc, "wall_s": round(wall, 1), "model": args.model, "effort": effort,
              "codex": subprocess.run([codex, "--version"], capture_output=True, text=True).stdout.strip(),
              "cmd": ["codex"] + [c.replace(str(home), "<codex_home>").replace(work, "<work>") for c in cmd[1:]],
              "visible_tools": tools, "visible_messages": [m["role"] for m in messages],
              "audit": audit(out / "events.jsonl")}
    answer = (out / "answer.md").read_text() if (out / "answer.md").exists() else ""
    code = last_python_block(answer)
    if code is not None and not args.check:
        (out / "policy.py").write_text(code)
    record["policy_py"] = code is not None
    json.dump(record, open(out / "run.json", "w"), indent=1)

    a = record["audit"]
    print(f"{out.relative_to(HERE)}: rc {rc}, {wall:.0f} s, token {a['usage']}, tool_free {a['tool_free']}"
          + (f", tool usati {a['tool_items']}" if a["tool_items"] else "")
          + ("" if args.check else f", policy.py {'scritto' if code is not None else 'MANCANTE'}"))
    if rc != 0:
        print((out / "stderr.log").read_text()[-2000:])
    if args.check:
        print("risposta:", answer.strip())


if __name__ == "__main__":
    main()
