"""
Is this board about to price tonight on data that is already out of date.

Every board builds its features from a stored history, and several of those histories come
from sources that run behind. Measured 2026-08-08: the wehoop mirror's latest WNBA game was
2026-08-01, so eighteen completed games were invisible to the Elo, the form decays and the
availability features. The board published anyway and nothing said a word.

**"Stale" is not "older than today."** A league that did not play yesterday has old data and
is perfectly current. The test that means something is whether the live source knows about
a completed game the local history does not. That is what `check` asks.

**A probe that fails is not a probe that found nothing.** (2026-08-09) The old `_completed_on`
returned 0 both when a day had no games AND when the ESPN call errored or was rate-limited.
So a throttled run reported every board FRESH and stamped "complete" on data that was nine
matches stale -- the guard asserting a provenance claim it never verified. Now a failed
probe returns a sentinel (None), is retried with backoff, and if it still cannot be
completed the verdict is UNVERIFIED, never "complete". Whether an unverified board publishes
depends on how old the local history already is (see `_verdict`).

Usage in a runner, after the history is loaded and before anything is boarded:

ok, msg, status = FR.check("WNBA", newest_local_date, "basketball/wnba")
print(msg)
if not ok and not FR.override():
raise SystemExit("refusing to board on stale data")

Set MBM_ALLOW_STALE=1 to board anyway, which is there for the day a source changes shape
and the check is wrong rather than the data.
"""
import datetime as dt
import os
import time

import requests

VERSION = "2026-08-09"

# ESPN 403s any custom User-Agent. See the note in espn_roster.py.
H = {}
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# How far back to look for the most recent completed event. Long enough to survive an
# All-Star break or an international window without giving up and reporting "unknown".
LOOKBACK_DAYS = 21

# Season types the models actually train on. ESPN marks preseason as 1, regular 2, post 3.
# Pass season_types=None to accept everything, which is what soccer needs: ESPN puts a
# league id there rather than the 1/2/3 US convention, so filtering on (2, 3) rejects
# every match.
LIVE_TYPES = (2, 3)

# Never walk more than this many days looking for the gap.
MAX_WALK = 40

# Transient-error handling for a single day's probe.
PROBE_RETRIES = 2          # 3 attempts total
PROBE_BACKOFF = 0.5        # seconds, doubled each retry

# How recent the local history must be to publish when the probe could NOT be completed.
# Within this many days of today, an unverified board publishes with a flag and a WARN,
# because taking a board down on every ESPN hiccup gets it trusted less than one that says
# what it does not know. Older than this, an unverified probe blocks: we already have
# independent reason to suspect staleness and no way to rule it out, which is exactly when
# publishing is indefensible.
#
# This MUST be set per board, and the value is the league's LONGEST NORMAL GAP between
# fixtures, not its typical one. The block rule ("probe failed AND history older than grace")
# assumes old history plus no verification is suspicious -- but that premise breaks during a
# legitimate break in play: an international window, an All-Star break, a gap between CS2
# tournaments. During those, old history is entirely normal, and a strict threshold would
# block an unverified probe (ESPN throttled during the break) when nothing is wrong. A
# soccer board that silently goes dark every March is a real cost.
#
#   Big 5 soccer (in season): ~14   # covers an international window
#   MMA, CS2:                  ~7    # weekly cards / gaps between tournaments
#   WNBA, NBA, NHL (daily):    1-2
#
# Blocking is the safe direction, so erring tight is fine. The DEFAULT is strict (1 day) on
# purpose: a repo that forgets to set it fails safe by blocking rather than publishing an
# unverified old board.
STALE_GRACE_DEFAULT = 1


def override():
    return os.environ.get("MBM_ALLOW_STALE", "").strip() in ("1", "true", "yes")


def _completed_on(slug, day, season_types, timeout=25, retries=PROBE_RETRIES):
    """Number of completed events on `day`, or None if the probe could not be completed.

    None is the whole point of the 2026-08-09 fix: it is distinct from 0. A non-200
    response or an exception is retried with backoff (most ESPN throttling is transient),
    and only if every attempt fails does it return None so the caller can say UNVERIFIED
    rather than silently reporting the day empty.
    """
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{ESPN}/{slug}/scoreboard", headers=H,
                             params={"dates": day.strftime("%Y%m%d")}, timeout=timeout)
            if r.status_code != 200:
                raise IOError(f"HTTP {r.status_code}")
            data = r.json()
        except Exception:
            if attempt < retries:
                time.sleep(PROBE_BACKOFF * (2 ** attempt))
                continue
            return None
        n = 0
        for e in data.get("events", []):
            if not e.get("status", {}).get("type", {}).get("completed"):
                continue
            if season_types is not None and \
               (e.get("season", {}) or {}).get("type") not in season_types:
                continue
            n += 1
        return n
    return None


def newest_completed(slug, lookback=LOOKBACK_DAYS, season_types=LIVE_TYPES, timeout=25):
    """(date, count, probe_failed) for the most recent completed event, walking back.

    If a completed day is found, the answer is reliable and probe_failed is False. If no
    completed day is found, probe_failed says whether that is because the days were
    genuinely empty (False) or because the probes could not be completed (True) -- the
    caller must not treat the second as "no games".
    """
    today = dt.date.today()
    errored = False
    for back in range(lookback + 1):
        day = today - dt.timedelta(days=back)
        n = _completed_on(slug, day, season_types, timeout)
        if n is None:
            errored = True
            continue
        if n:
            return day, n, False
    return None, 0, errored


def missed_games(slug, since, until=None, season_types=LIVE_TYPES, timeout=25):
    """(total, last, probe_failed): completed events after `since`, and the last date.

    probe_failed is True only when freshness could not be determined at all (the newest
    probe could not be completed). Once we reliably know the source is ahead of `since`,
    staleness is certain regardless of a hiccup later in the window, so probe_failed is
    False and only the count may undercount.
    """
    newest, _, failed = newest_completed(slug, season_types=season_types, timeout=timeout)
    if failed:
        return 0, None, True
    if newest is None or newest <= since:
        return 0, None, False
    until = min(until or dt.date.today(), newest)
    start = max(since + dt.timedelta(days=1), until - dt.timedelta(days=MAX_WALK))
    total, last, capped = 0, None, start > since + dt.timedelta(days=1)
    day = start
    while day <= until:
        n = _completed_on(slug, day, season_types, timeout)
        if n:                       # None (probe error) counts as 0 here; staleness is
            total += n              # already established, this only affects the count
            last = day
        day += dt.timedelta(days=1)
    if capped and total:
        total = -total              # negative flags "at least this many"
    return total, last, False


def _verdict(label, local_newest, missed, last, status, max_missed=0,
             grace_days=STALE_GRACE_DEFAULT):
    """Turn a probe result into (ok, message, verdict) with the two-way unverified rule.

    status: "measured" (probe completed) or "unverified" (probe could not complete).
    grace_days: how old the local history may be and still publish on an unverified probe.
    verdict: "fresh" | "stale" | "unverified", for the footer and the MBM-CHECK line.
    """
    if local_newest is None:
        return True, f"{label}: no local history to check", "fresh"
    if isinstance(local_newest, dt.datetime):
        local_newest = local_newest.date()
    age = (dt.date.today() - local_newest).days

    if status == "unverified":
        if age <= grace_days:
            return True, (f"{label}: freshness UNVERIFIED (probe failed after retries); "
                          f"local history is recent ({age}d <= {grace_days}d grace), "
                          f"publishing with flag"), "unverified"
        return False, (f"{label}: freshness UNVERIFIED (probe failed after retries) and "
                       f"local history is {age}d old (> {grace_days}d grace); refusing to "
                       f"board without a check"), "unverified"

    at_least = missed < 0
    missed = abs(missed)
    if missed <= max_missed:
        return True, (f"{label}: data current to {local_newest} ({age}d), "
                      f"no completed games missing"), "fresh"
    return False, (f"{label}: STALE. History stops {local_newest} and the source has "
                   f"{'at least ' if at_least else ''}{missed} completed game(s) since, "
                   f"through {last}. Any pick made now is priced on a league table that is "
                   f"{(last - local_newest).days} day(s) out of date."), "stale"


def check(label, local_newest, slug, max_missed=0, season_types=LIVE_TYPES,
          grace_days=STALE_GRACE_DEFAULT):
    """(ok, message, verdict). Not ok when the source has games the history lacks, or when
    the probe could not be completed and the local history is already older than the grace.

    grace_days: set per board (~1 daily, ~7 for MMA/CS2). Defaults strict.
    """
    if local_newest is None:
        return True, f"{label}: no local history to check", "fresh"
    if isinstance(local_newest, dt.datetime):
        local_newest = local_newest.date()
    try:
        missed, last, failed = missed_games(slug, local_newest, season_types=season_types)
    except Exception as ex:
        # An unexpected error is treated as an unverified probe, not as staleness.
        return _verdict(label, local_newest, 0, None, "unverified", max_missed, grace_days)
    status = "unverified" if failed else "measured"
    return _verdict(label, local_newest, missed, last, status, max_missed, grace_days)


def check_multi(entries, season_types=LIVE_TYPES, precomputed=None,
                grace_days=STALE_GRACE_DEFAULT, reporter=None):
    """Freshness across several leagues on one board.

    Returns (ok, lines). Not ok if ANY league is behind (or blocked-unverified), because a
    board that publishes Brasileirao and Liga MX together is only as trustworthy as its
    worst input.

    precomputed: optional {label: (missed, last, probe_failed)} so a runner that has
    already probed each league (to build its footer) does not probe again. This halves the
    per-league probe count, which directly shrinks the window in which a probe can be
    throttled -- the reason a seven-league board was the most exposed to the fail-open.
    grace_days: applies to every league on the board (they share the board's cadence).
    reporter: pass the mbm_checks Reporter and this registers the freshness guard as having
    executed, which is the only thing that satisfies require("freshness").
    """
    ok_all, lines, worst = True, [], "fresh"
    order = {"fresh": 0, "stale": 1, "unverified": 2}
    for label, newest, slug in entries:
        if precomputed and label in precomputed:
            missed, last, failed = precomputed[label]
            ok, msg, v = _verdict(label, newest, missed, last,
                                  "unverified" if failed else "measured",
                                  grace_days=grace_days)
        else:
            ok, msg, v = check(label, newest, slug, season_types=season_types,
                               grace_days=grace_days)
        ok_all &= ok
        lines.append(msg)
        if order.get(v, 0) > order.get(worst, 0):
            worst = v
    if reporter is not None:
        reporter.register_guard("freshness", ok_all, " | ".join(lines), worst)
    return ok_all, lines


def board_line(local_newest, slug=None, season_types=LIVE_TYPES, checked=None):
    """One short line for the Discord footer, so the board carries its own provenance.

    `checked` may be (newest, missing) or (newest, missing, status) where status is
    "unverified" for a probe that could not complete. An unverified board NEVER prints
    "complete": the footer is a provenance claim and must not assert what was not checked.
    """
    status = "measured"
    if checked is not None:
        if len(checked) == 3:
            newest, missing, status = checked
        else:
            newest, missing = checked
    else:
        missing, _, failed = missed_games(slug, local_newest, season_types=season_types)
        newest = local_newest
        status = "unverified" if failed else "measured"
    if isinstance(newest, dt.datetime):
        newest = newest.date()

    if status == "unverified":
        return f"data through {newest} · freshness unverified, probe failed"
    yday = dt.date.today() - dt.timedelta(days=1)
    if missing:
        return (f"data through {newest} · {abs(missing)} completed game(s) MISSING · "
                f"treat these prices with caution")
    through = "end of yesterday" if newest >= yday else f"{newest}"
    return f"data through {through} · complete"


def season_note(min_gp, full_at, label="both sides"):
    """Flag a board still inside its early-season shrink, or None once it is out."""
    if min_gp is None or full_at is None or min_gp >= full_at:
        return None
    return (f"early season: {label} {int(min_gp)} game(s) in, forecasts shrunk toward the "
            f"base rate until {int(full_at)}")


def guard(label, local_newest, slug, max_missed=0, season_types=LIVE_TYPES,
          grace_days=STALE_GRACE_DEFAULT, reporter=None):
    """Print the verdict and stop the run if the data is stale or unverifiably old.

    Called after the history is loaded and before anything reaches a board, so a bad run
    produces no forecast at all rather than a forecast nobody should act on.

    grace_days: longest normal gap for this board (~14 Big 5, ~7 MMA/CS2, 1-2 daily).
    reporter: pass the mbm_checks Reporter and this registers the freshness guard as having
    executed -- the only thing that satisfies require("freshness"), so wiring the guard and
    satisfying the requirement are the same act and cannot come apart.
    """
    ok, msg, verdict = check(label, local_newest, slug, max_missed, season_types,
                             grace_days)
    print(msg)
    if reporter is not None:
        reporter.register_guard("freshness", ok, msg, verdict)
    if ok:
        return True
    if override():
        print(f"{label}: MBM_ALLOW_STALE set, boarding anyway")
        return True
    raise SystemExit(f"{label}: refusing to board on stale data")
