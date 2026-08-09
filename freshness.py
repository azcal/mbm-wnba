"""
Is this board about to price tonight on data that is already out of date.

Every board builds its features from a stored history, and several of those histories come
from sources that run behind. Measured 2026-08-08: the wehoop mirror's latest WNBA game was
2026-08-01, so eighteen completed games were invisible to the Elo, the form decays and the
availability features. The board published anyway and nothing said a word.

**"Stale" is not "older than today."** A league that did not play yesterday has old data and
is perfectly current. The test that means something is whether the live source knows about
a completed game the local history does not. That is what `check` asks.

Usage in a runner, after the history is loaded and before anything is boarded:

    ok, msg = FR.check("WNBA", newest_local_date, "basketball/wnba")
    print(msg)
    if not ok and not FR.override():
        raise SystemExit("refusing to board on stale data")

Set MBM_ALLOW_STALE=1 to board anyway, which is there for the day a source changes shape
and the check is wrong rather than the data.
"""
import datetime as dt
import os

import requests

VERSION = "2026-08-08"

# ESPN 403s any custom User-Agent. See the note in espn_roster.py.
H = {}
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# How far back to look for the most recent completed event. Long enough to survive an
# All-Star break or an international window without giving up and reporting "unknown".
LOOKBACK_DAYS = 21


def override():
    return os.environ.get("MBM_ALLOW_STALE", "").strip() in ("1", "true", "yes")


# Season types the models actually train on. ESPN marks preseason as 1, regular 2, post 3.
# Without this the NFL board reads as stale all summer because a Hall of Fame preseason
# game completed in August, which no model has ever used.
# Pass season_types=None to accept everything, which is what soccer needs: ESPN puts a
# league id there (14277 for Liga MX) rather than the 1/2/3 US convention, so filtering on
# (2, 3) would reject every match ever played.
LIVE_TYPES = (2, 3)

# Never walk more than this many days looking for the gap. Without a cap, a board whose
# season ended in February probes every day since, which is 179 requests to learn nothing.
MAX_WALK = 40


def _completed_on(slug, day, season_types, timeout=25):
    try:
        r = requests.get(f"{ESPN}/{slug}/scoreboard", headers=H,
                         params={"dates": day.strftime("%Y%m%d")}, timeout=timeout).json()
    except Exception:
        return 0
    n = 0
    for e in r.get("events", []):
        if not e.get("status", {}).get("type", {}).get("completed"):
            continue
        if season_types is not None and \
                (e.get("season", {}) or {}).get("type") not in season_types:
            continue
        n += 1
    return n


def newest_completed(slug, lookback=LOOKBACK_DAYS, season_types=LIVE_TYPES, timeout=25):
    """Date of the most recent completed event, walking back from today.

    Backwards rather than forwards, so an off-season board stops after `lookback` calls
    instead of probing every day since its last game.
    """
    today = dt.date.today()
    for back in range(lookback + 1):
        day = today - dt.timedelta(days=back)
        n = _completed_on(slug, day, season_types, timeout)
        if n:
            return day, n
    return None, 0


def missed_games(slug, since, until=None, season_types=LIVE_TYPES, timeout=25):
    """Completed events after `since`, and the date of the last one.

    Finds the newest completed day first and returns early if it is not after `since`,
    which is the common case and costs a handful of calls rather than hundreds.
    """
    newest, _ = newest_completed(slug, season_types=season_types, timeout=timeout)
    if newest is None or newest <= since:
        return 0, None
    until = min(until or dt.date.today(), newest)
    start = max(since + dt.timedelta(days=1), until - dt.timedelta(days=MAX_WALK))
    total, last, capped = 0, None, start > since + dt.timedelta(days=1)
    day = start
    while day <= until:
        n = _completed_on(slug, day, season_types, timeout)
        if n:
            total += n
            last = day
        day += dt.timedelta(days=1)
    if capped and total:
        total = -total          # negative flags "at least this many", see check()
    return total, last


def check(label, local_newest, slug, max_missed=0, season_types=LIVE_TYPES):
    """(ok, message). Not ok when the source has completed games the history lacks.

    `max_missed` is a tolerance for boards whose source legitimately trails by a game or
    two. Zero means any missed game is a failure, which is the right default.
    """
    if local_newest is None:
        return True, f"{label}: no local history to check"
    if isinstance(local_newest, dt.datetime):
        local_newest = local_newest.date()
    try:
        n, last = missed_games(slug, local_newest, season_types=season_types)
    except Exception as ex:
        # A probe failure is not evidence of staleness, so it must not block a board.
        return True, (f"{label}: freshness probe failed ({type(ex).__name__}), "
                      f"continuing on local data to {local_newest}")
    at_least = n < 0
    n = abs(n)
    if n <= max_missed:
        age = (dt.date.today() - local_newest).days
        return True, (f"{label}: data current to {local_newest} ({age}d), "
                      f"no completed games missing")
    return False, (f"{label}: STALE. History stops {local_newest} and the source has "
                   f"{'at least ' if at_least else ''}{n} completed game(s) since, "
                   f"through {last}. Any pick made now is "
                   f"priced on a league table that is {(last - local_newest).days} day(s) "
                   f"out of date.")


def check_multi(entries, season_types=LIVE_TYPES):
    """Freshness across several leagues on one board.

    Returns (ok, lines). Not ok if ANY league is behind, because a board that publishes
    Brasileirao and Liga MX together is only as trustworthy as its worst input.
    """
    ok_all, lines = True, []
    for label, newest, slug in entries:
        ok, msg = check(label, newest, slug, season_types=season_types)
        ok_all &= ok
        lines.append(msg)
    return ok_all, lines


def board_line(local_newest, slug=None, season_types=LIVE_TYPES, checked=None):
    """One short line for the Discord footer, so the board carries its own provenance.

    The standard is end of yesterday: today's card should be priced on a history that
    includes every game finished last night. This says whether it is.
    """
    if checked is not None:
        newest, missing = checked
    else:
        missing, _ = missed_games(slug, local_newest, season_types=season_types)
        newest = local_newest
    if isinstance(newest, dt.datetime):
        newest = newest.date()
    yday = dt.date.today() - dt.timedelta(days=1)
    if missing:
        return (f"data through {newest} · {abs(missing)} completed game(s) MISSING · "
                f"treat these prices with caution")
    through = "end of yesterday" if newest >= yday else f"{newest}"
    return f"data through {through} · complete"


def season_note(min_gp, full_at, label="both sides"):
    """Flag a board still inside its early-season shrink, or None once it is out.

    Every model except CS2 and MMA leans on a season's accumulated form, and early on
    there is not enough of it, so forecasts are pulled toward the base rate. That is
    working as intended and it is worth saying out loud, because a 58% early in a season
    is a different statement from a 58% in March.
    """
    if min_gp is None or full_at is None or min_gp >= full_at:
        return None
    return (f"early season: {label} {int(min_gp)} game(s) in, forecasts shrunk toward the "
            f"base rate until {int(full_at)}")


def guard(label, local_newest, slug, max_missed=0, season_types=LIVE_TYPES):
    """Print the verdict and stop the run if the data is stale.

    Called after the history is loaded and before anything reaches a board, so a stale run
    produces no forecast at all rather than a forecast nobody should act on.
    """
    ok, msg = check(label, local_newest, slug, max_missed, season_types)
    print(msg)
    if ok:
        return True
    if override():
        print(f"{label}: MBM_ALLOW_STALE set, boarding anyway")
        return True
    raise SystemExit(f"{label}: refusing to board on stale data")
