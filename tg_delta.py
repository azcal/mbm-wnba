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
import datetime as dt
import json, os

import pandas as pd

VERSION = "2026-08-18"

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
         indent=MARK_INDENT, with_keys=False, report_gone=False):
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
    # A full board only when there is nothing to compare against, or when it is asked for.
    # NOT on a date rollover, which is what this used to do: `prev.get("date") != datestr`
    # meant every midnight reposted the whole board. CS2 carries a two-day horizon and
    # soccer a seven-day one, so most fixtures survive the rollover unchanged and were being
    # republished in full every morning at prices nobody had moved. A fixture should post
    # once, then only when its price or its side actually changes.
    #
    # The cache still records the date, for reading the file by hand. It just no longer
    # decides anything. Set FORCE_PUSH=1 to get a full board deliberately.
    first = forced or not prev.get("board")
    if first:
        # Name the reason, because "the whole board reposted" has more than one cause and
        # they need different fixes. A missing cache means push_cache.json is not being
        # committed back from logs/, so every run looks like the first.
        print("full board: " + ("FORCE_PUSH set" if forced
                                else "no cache to compare against"))
    old = {} if first else prev.get("board", {})

    lines, keyed = [], []
    if first:
        lines = [v["line"] for v in board.values()]
        keyed = [(k, v["line"]) for k, v in board.items()]
    else:
        for k, v in board.items():
            o = old.get(str(k))
            if o is None:
                lines.append(f"{v['line']}\n{indent}new: not on the last board")
                keyed.append((k, lines[-1]))
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
                keyed.append((k, lines[-1]))

    os.makedirs(logs_dir, exist_ok=True)
    # A fixture that was on the last board and is not on this one. Happens when an
    # opponent is replaced and the source issues a new id, so the old bout simply vanishes
    # rather than changing. Without this the channel keeps a published pick for a fight
    # that will never happen, and nothing ever says otherwise. Measured 2026-08-08: two
    # UFC bouts were boarded, both opponents were swapped, and neither was withdrawn.
    # report_gone is OFF by default and only MMA should turn it on. A fighter withdrawing
    # is real news: the bout is off, the pick is void, and nothing else says so. A team
    # sport fixture leaving the board is never that -- the game started, and the pregame
    # filter drops it by design. Reporting that fires on every game every day, trains the
    # reader to skip the notice, buries the MMA case it exists for, and makes the board
    # post something on every run when nothing has actually changed.
    if not first and report_gone:
        # A fixture on the last board and not on this one has left for one of two reasons,
        # and they are not the same news.
        #
        # It STARTED. Every runner now filters the slate to pre-game only, so a game that
        # tips between two runs drops off by design. Reporting that as "withdrawn" fires on
        # every board every day and trains the reader to ignore the notice, which defeats
        # the point of having it.
        #
        # It was WITHDRAWN. The fixture is gone before its scheduled start: an opponent
        # replaced, a card reshuffled, a bracket slot resolving to different teams. That is
        # real news and it is why this block exists -- two UFC bouts were boarded on
        # 2026-08-08 whose opponents were both swapped, and nothing said so.
        #
        # The two are separable by the scheduled start time, which the runner puts in the
        # push entry as "starts". Without it there is no way to tell, so an entry carrying
        # no start time is reported with hedged wording rather than a false claim.
        now = dt.datetime.now(dt.timezone.utc)
        gone = [(k, v) for k, v in old.items() if k not in board]
        drop = []
        for k, v in gone:
            starts = v.get("starts")
            if starts:
                try:
                    ts = pd.Timestamp(starts)
                    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                    if ts.to_pydatetime() <= now:
                        continue          # under way or done: expected, not news
                except Exception:
                    pass
            drop.append((k, v, bool(starts)))
        if drop:
            lines.append("")              # its own block, not a note on the row above
            keyed.append(("", ""))
            for k, v, known in drop:
                what = v.get("match") or v.get("line", "").strip()
                # Two labels, because they are two different pieces of news and the first
                # word is what the reader acts on. A pre-match market closing at kickoff is
                # routine bookkeeping; a fixture pulled before it was due to start is the
                # thing worth knowing. Calling both "withdrawn" made the routine case shout.
                if known:
                    lines.append(f"{indent}withdrawn: {what} was pulled before its "
                                 f"start time")
                else:
                    lines.append(f"{indent}closed: {what}, pre-match market no longer "
                                 f"open (started, or the fixture changed)")
                keyed.append((k, lines[-1]))
                # Blank line between consecutive notices: Discord collapses adjacent
                # subtext lines into one run-on block.
                if (k, v, known) != drop[-1]:
                    lines.append("")
                    keyed.append(("", ""))

    with open(path, "w") as fh:
        json.dump({"date": datestr, "board": {str(k): v for k, v in board.items()}}, fh)
    # with_keys hands back (key, line) pairs so a caller can regroup what survived the
    # gate, which is how the soccer board puts league headings on a delta push where only
    # some fixtures are sent. Off by default: the other eight runners unpack two values.
    return ("full" if first else "delta"), (keyed if with_keys else lines)
