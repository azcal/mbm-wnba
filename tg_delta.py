"""
Push gate, shared across MBM boards.

First run of a given day sends the whole board and caches it. Every later run that day
compares against the cache and sends only the entries that moved, each carrying a marker
saying what moved, so an afternoon refresh is silent unless something actually repriced.

The models take no odds input, so a book's line moving is invisible to them. What is
visible is the model's own number changing: a scratched starter, a confirmed goalie, a
posted lineup, an injury-report update, a replaced opponent. Those move the minimum odds
worth demanding, and that is what this sends.

Two things count as a change:

  price    the required minimum odds moved by at least `tol`
  match    the matchup itself is different under the same id, which is how a late
           opponent replacement shows up. Pass an optional "match" field to enable it.
           A board that omits the field keeps the old price-only behaviour.

Markers go on their own indented line rather than on the end of a board row. Appending
one to a row puts a second bracket after "or better", which the notify regexes reject,
and the whole row then renders as unformatted text.

Drop this file next to the runner in each repo. No dependencies beyond the standard library.
"""
import json, os

VERSION = "2026-08-09"

CACHE_NAME = "push_cache.json"
DEFAULT_TOL = 0.02          # required-odds move that counts as news
MARK_INDENT = "      "      # matches the live-note indent, so notify reads it as subtext


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def gate(logs_dir, datestr, board, tol=DEFAULT_TOL, cache_name=CACHE_NAME,
         indent=MARK_INDENT):
    """Decide what to push.

    board: {key: {"line": <str to send>, "min_odds": <float>, "match": <optional str>}}
           key must be stable within a day, so use the game or fight id, not a matchup
           string. "match" is the matchup in reader-facing form; supply it and a swapped
           opponent under the same id is reported even when the price barely moves.

    Returns (mode, lines) where mode is "full" or "delta" and lines is what to send.
    An empty list means send nothing at all.

    The cache is written on every call, so a run that sends nothing still updates the
    baseline and a later run compares against the most recent state rather than the morning.
    """
    path = os.path.join(logs_dir, cache_name)
    prev = _load(path)
    # FORCE_PUSH=1 makes a run behave as the first of the day and send the whole board.
    # Without it a refresh that finds nothing changed correctly sends nothing, which is
    # indistinguishable from a broken notifier when you are testing.
    forced = os.environ.get("FORCE_PUSH", "").strip() in ("1", "true", "yes")
    first = forced or prev.get("date") != datestr
    if forced:
        print("FORCE_PUSH set, sending the full board")
    old = {} if first else prev.get("board", {})

    lines = []
    if first:
        lines = [v["line"] for v in board.values()]
    else:
        for k, v in board.items():
            o = old.get(str(k))
            if o is None:
                lines.append(f"{v['line']}\n{indent}new: not on the last board")
                continue
            why = []
            was_match = o.get("match")
            if v.get("match") and was_match and was_match != v["match"]:
                # Report only the part that moved. The match field often carries the pick
                # AND the matchup so both a side flip and a replaced opponent are caught,
                # and echoing the unchanged half twice makes the note twice as long and
                # half as clear.
                new_s, old_s = str(v["match"]), str(was_match)
                cut = 0
                while (cut < min(len(new_s), len(old_s))
                       and new_s[len(new_s) - 1 - cut] == old_s[len(old_s) - 1 - cut]):
                    cut += 1
                if cut and cut < min(len(new_s), len(old_s)):
                    tail = new_s[len(new_s) - cut:].lstrip()
                    new_s, old_s = new_s[:len(new_s) - cut].rstrip(), \
                        old_s[:len(old_s) - cut].rstrip()
                    why.append(f"now {new_s}, was {old_s}" + (f" ({tail})" if tail else ""))
                else:
                    why.append(f"now {new_s}, was {old_s}")
            try:
                moved = abs(float(v["min_odds"]) - float(o.get("min_odds")))
            except (TypeError, ValueError):
                moved = 0.0
            if moved >= tol:
                why.append(f"price was {float(o['min_odds']):.2f}")
            if why:
                lines.append(f"{v['line']}\n{indent}changed: " + ", ".join(why))

    os.makedirs(logs_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"date": datestr, "board": {str(k): v for k, v in board.items()}}, fh)
    return ("full" if first else "delta"), lines
