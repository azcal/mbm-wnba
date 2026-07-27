"""
Telegram push gate, shared across MBM boards.

First run of a given day sends the whole board and caches it. Every later run that day
compares against the cache and sends only what moved, so an afternoon refresh is silent
unless something actually repriced.

The models take no odds input, so a book's line moving is invisible to them. What is
visible is the model's own number changing: a scratched starter, a confirmed goalie, a
posted lineup, an injury-report update. Those move the minimum odds worth demanding, and
that is what this sends.

Drop this file next to the runner in each repo. No dependencies beyond the standard library.
"""
import json, os

CACHE_NAME = "push_cache.json"
DEFAULT_TOL = 0.02          # required-odds move that counts as news


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def gate(logs_dir, datestr, board, tol=DEFAULT_TOL, cache_name=CACHE_NAME):
    """Decide what to push.

    board: {key: {"line": <str to send>, "min_odds": <float>}}
           key must be stable within a day, so use the game id, not a matchup string.

    Returns (mode, lines) where mode is "full" or "delta" and lines is what to send.
    An empty list means send nothing at all.

    The cache is written on every call, so a run that sends nothing still updates the
    baseline and a later run compares against the most recent state rather than the morning.
    """
    path = os.path.join(logs_dir, cache_name)
    prev = _load(path)
    first = prev.get("date") != datestr
    old = {} if first else prev.get("board", {})

    lines = []
    if first:
        lines = [v["line"] for v in board.values()]
    else:
        for k, v in board.items():
            o = old.get(str(k))
            if o is None:
                lines.append(f"{v['line']}   [new]")
                continue
            try:
                moved = abs(float(v["min_odds"]) - float(o.get("min_odds")))
            except (TypeError, ValueError):
                continue
            if moved >= tol:
                lines.append(f"{v['line']}   [was {float(o['min_odds']):.2f}]")

    os.makedirs(logs_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"date": datestr, "board": {str(k): v for k, v in board.items()}}, fh)
    return ("full" if first else "delta"), lines
