uv run --with numpy python evaluate.py weights.json --policy policy.py
uv run --with numpy python evaluate.py --baseline heuristic

uv run --with numpy --with pygame python play.py                          # giochi tu
uv run --with numpy --with pygame python play.py --weights weights.json   # guardi la rete di Astra
uv run --with numpy --with pygame python play.py --baseline heuristic     # guardi l'euristica
