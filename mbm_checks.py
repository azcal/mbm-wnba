"""
MBM board invariant checks -- shared across every runner.

The defect class this exists to catch: a board that is WRONG looks exactly like a board
that is RIGHT. None of the nine incidents crashed; each produced plausible output. So a
static read cannot find them and neither can exception handling. These checks turn the
invariants into assertions that run inside the board, against the real feed, and FAIL
LOUD in the Actions log.

Output is a fixed, greppable line protocol so a pasted Actions log is enough to work from:

    MBM-CHECK BEGIN <label> <datestr>
    MBM-CHECK <PASS|FAIL|WARN|SKIP> <check[:detailkey]> | <human detail>
    MBM-CHECK END <label> pass=<n> fail=<n> warn=<n> skip=<n>

Grep the log with:  grep '^MBM-CHECK'
A FAIL anywhere makes rep.finish(strict=True) raise SystemExit, so the workflow goes red.

Wiring: build one Reporter per run, call the relevant checks at the point where the data
they guard actually exists (freshness after history load, pregame_only before boarding,
log_board_agreement after the push gate), then rep.finish() at the end.

No dependencies beyond pandas, which every repo already imports.
"""
import os
import re
import datetime as dt

import pandas as pd

VERSION = "2026-08-09"

# ESPN status carries a type.state that is one of exactly three values: "pre", "in",
# "post". Only "pre" is boardable. We key on STATE, never on the status NAME: there are
# many names (scheduled, postponed, delayed, forfeit, TBD, ...) and a name whitelist fails
# OPEN on any name nobody anticipated, which is the exact defect class this audit exists to
# stop. Three states, one of them safe.
PREGAME_STATE = "pre"

_LEVELS = ("PASS", "WARN", "SKIP", "FAIL")


class Reporter:
    """Accumulates check results and prints the fixed protocol.

    strict finish raises when any check FAILed, which is what turns a silent wrong board
    into a red workflow. A WARN never fails the run; it is for things that are inert today
    but are the fingerprint of a real incident (unsorted set iteration, an empty result
    that could not be distinguished from an error).
    """

    def __init__(self, label, datestr=None):
        self.label = label
        self.datestr = datestr or dt.date.today().strftime("%Y%m%d")
        self.counts = {k: 0 for k in _LEVELS}
        self.seen = set()          # base check names that ran (any check method)
        self._guards = set()        # guards that ACTUALLY EXECUTED (registered by the guard
                                    # itself, e.g. FR.guard) -- what require() trusts
        print(f"MBM-CHECK BEGIN {self.label} {self.datestr}")

    def _emit(self, level, check, detail):
        self.counts[level] += 1
        self.seen.add(str(check).split(":")[0])
        print(f"MBM-CHECK {level} {check} | {detail}")
        return level != "FAIL"

    # -- individual invariants -------------------------------------------------

    def register_guard(self, name, ok, msg, verdict=None):
        """Record that a shared GUARD actually executed. This is the only thing require()
        trusts, and it is called by the guard itself (FR.guard / FR.check_multi with a
        reporter=), never by the runner. That closes the loophole where a runner satisfies
        require("freshness") by calling a check method by hand while never probing anything
        -- the same "present in form, absent in effect" shape as the MMA freshness bug.

        verdict "unverified" + ok True (published within grace) is a WARN; blocked or stale
        is a FAIL.
        """
        self._guards.add(name)
        if verdict == "unverified":
            return self._emit("WARN" if ok else "FAIL", name, msg)
        return self._emit("PASS" if ok else "FAIL", name, msg)

    def freshness(self, ok, msg, verdict=None, check="freshness"):
        """Manual freshness reporting for a board that probes inline. NOTE: this does NOT
        satisfy require("freshness") -- only an executed FR.guard (via register_guard) does,
        so a hand-written freshness line cannot fake the guard.
        """
        if verdict == "unverified":
            return self._emit("WARN" if ok else "FAIL", check, msg)
        return self._emit("PASS" if ok else "FAIL", check, msg)

    def cross_source_match(self, name, unmapped, check="cross_source_match"):
        """Names that failed to join across two sources must be zero.

        Incident 9: a fighter or club named differently in the fixtures source than the
        results source is silently left ungraded forever. The soccer Americas path already
        skips-and-prints unmatched names; this turns that log line into a WARN so a feed
        rename surfaces in the report instead of scrolling past in the Actions output.
        """
        items = list(unmapped) if hasattr(unmapped, "__iter__") and not isinstance(unmapped, str) \
            else ([unmapped] if unmapped else [])
        n = len(items) if not (len(items) == 1 and isinstance(items[0], int)) else items[0]
        if n:
            sample = ", ".join(str(x) for x in items[:4]) if items and not isinstance(items[0], int) else ""
            return self._emit("WARN", f"{check}:{name}",
                              f"{n} name(s) unmatched across sources"
                              f"{' -- e.g. ' + sample if sample else ''}; add to the alias "
                              f"file or these rows never join")
        return self._emit("PASS", f"{check}:{name}", "all names matched across sources")

    def pregame_only(self, slate, history_ids=None, id_key="game_id",
                     state_key="state", check="pregame_only"):
        """Nothing started, finished, or already in history reaches the board.

        slate: iterable of dict-like rows for the games about to be boarded. Each row must
               carry a `state` of "pre"/"in"/"post" (ESPN status.type.state). The runner's
               fetch_slate should surface it; boarding on the state is the fix, this is the
               guard that the fix held.
        history_ids: ids the results history already holds as completed. A slate id that is
               also here is a game the feed still calls "pre" while your history has it
               finished -- belt-and-braces the state filter alone does not catch.

        Catches incident 2 (finished games boarded; the feed's mid-game state is "in", not
        "pre") and the duplicate-row double count that follows when a re-boarded completed
        game is also in history.
        """
        history_ids = {str(x) for x in (history_ids or set())}
        slate = list(slate)
        started, in_hist, unknown = [], [], []
        for row in slate:
            gid = str(row.get(id_key, ""))
            state = str(row.get(state_key, "")).lower()
            if not state:
                unknown.append(gid)          # cannot judge -> WARN, never fail open to a pass
            elif state != PREGAME_STATE:
                started.append(f"{gid}:{state}")
            if gid in history_ids:
                in_hist.append(gid)
        if started or in_hist:
            detail = []
            if started:
                detail.append(f"{len(started)} not pregame [{', '.join(started[:6])}]")
            if in_hist:
                detail.append(f"{len(in_hist)} already in history [{', '.join(in_hist[:6])}]")
            return self._emit("FAIL", check, "; ".join(detail))
        if unknown:
            return self._emit("WARN", check,
                              f"{len(unknown)} slate row(s) carry no state field -- cannot "
                              f"confirm pregame [{', '.join(unknown[:6])}]")
        return self._emit("PASS", check, f"{len(slate)} slate game(s) all state=pre, none in history")

    def key_join(self, name, left, right, check="key_join"):
        """The two sides of a join agree AFTER normalisation, measured on overlap.

        NOT a dtype test. Under pandas 3 a string column is StringDtype and an int key is
        int64, and both join perfectly, so asserting object-dtype would FAIL a healthy
        board -- and a check that fires on healthy systems trains you to ignore it. The
        real invariant is that normalising the keys does not recover matches the raw types
        are silently dropping.

        Incident 4: a stored key '2998938.0' (float that went through a blank cell) never
        matches a fresh '2998938'. Raw string reprs differ, so raw overlap misses them;
        stripping the trailing '.0' recovers the match. If normalisation recovers matches,
        the live join is losing rows -- FAIL. If it recovers none, the keys agree -- PASS.
        """
        l_raw = {str(v) for v in pd.Series(left).dropna()}
        r_raw = {str(v) for v in pd.Series(right).dropna()}
        if not l_raw or not r_raw:
            return self._emit("WARN", f"{check}:{name}", "one side empty, nothing to join yet")
        norm = lambda s: {re.sub(r"\.0$", "", x) for x in s}
        raw_overlap = len(l_raw & r_raw)
        norm_overlap = len(norm(l_raw) & norm(r_raw))
        recovered = norm_overlap - raw_overlap
        if recovered > 0:
            example = next((x for x in (l_raw | r_raw) if x.endswith(".0")), "e.g. '2998938.0'")
            return self._emit("FAIL", f"{check}:{name}",
                              f"normalising recovers {recovered} match(es) the raw keys drop "
                              f"(raw overlap {raw_overlap}, normalised {norm_overlap}); a key "
                              f"like {example!r} is not matching its plain form")
        return self._emit("PASS", f"{check}:{name}",
                          f"raw overlap {raw_overlap} == normalised, keys agree")

    def column_contract(self, name, series, kind, check="column_contract"):
        """A consumer asserts a column's MEANING instead of assuming it.

        kind: 'binary' (outcome 0/1), 'prob' (0..1), 'margin' (signed, may exceed 1).
        Incident 5: a scoring margin stored where 0/1 was assumed produced plausible
        numbers. This makes the assumption explicit and loud.
        """
        s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
        if not len(s):
            return self._emit("WARN", f"{check}:{name}", "no non-null values to check")
        lo, hi = float(s.min()), float(s.max())
        if kind == "binary":
            bad = s[~s.isin([0, 1])]
            if len(bad):
                return self._emit("FAIL", f"{check}:{name}",
                                  f"expected 0/1, found {len(bad)} other value(s) "
                                  f"range [{lo:.3g},{hi:.3g}] e.g. {bad.iloc[0]:.3g}")
        elif kind == "prob":
            bad = s[(s < 0) | (s > 1)]
            if len(bad):
                return self._emit("FAIL", f"{check}:{name}",
                                  f"expected 0..1, found {len(bad)} out of range "
                                  f"[{lo:.3g},{hi:.3g}]")
        elif kind == "margin":
            if lo >= 0 and hi <= 1:
                return self._emit("WARN", f"{check}:{name}",
                                  f"declared margin but all values in [0,1] "
                                  f"-- is this actually a 0/1 outcome column?")
        return self._emit("PASS", f"{check}:{name}", f"{kind} ok, range [{lo:.3g},{hi:.3g}]")

    def no_silent_empty(self, name, is_empty, call_failed, check="no_silent_empty"):
        """An empty result must distinguish 'nothing to do' from 'the call failed'.

        Incident 3: slate() swallowed an ESPN 403 and returned {}, so 'no games today'
        printed on a four-game day. The caller must pass an explicit failure flag it
        captured at the call site (e.g. an exception was caught, or status != 200).
        """
        if is_empty and call_failed:
            return self._emit("FAIL", f"{check}:{name}",
                              "empty result AND the upstream call failed -- this is an "
                              "error masquerading as 'nothing to do'")
        if is_empty:
            return self._emit("WARN", f"{check}:{name}", "empty, upstream reported ok (nothing to do)")
        return self._emit("PASS", f"{check}:{name}", "non-empty")

    def assert_sorted(self, name, seq, key=None, check="determinism"):
        """A sequence a downstream step depends on must have a stated order.

        Incident 7: order relied on but never stated (set iteration, list(some_set)[:1]).
        Inert wherever the consumer only sums, but the fingerprint of a real inversion
        elsewhere, so it is a WARN not a FAIL: sort the source and it goes quiet.
        """
        items = list(seq)
        keyed = [key(x) for x in items] if key else items
        if keyed != sorted(keyed):
            return self._emit("WARN", f"{check}:{name}",
                              f"{len(items)} item(s) not in sorted order -- sort at the "
                              f"source for run-to-run reproducibility")
        return self._emit("PASS", f"{check}:{name}", f"{len(items)} item(s) sorted")

    def model_identity(self, logged_tag, expected_tag, check="model_identity"):
        """The model tag on a logged row matches the head that produced it.

        Incident 6: a backfill held v1 while v2 shipped, so the dashboard reported a
        retired model's calibration for weeks. expected_tag should come from the frozen
        spec, not a second hardcoded literal.
        """
        if str(logged_tag) != str(expected_tag):
            return self._emit("FAIL", check,
                              f"board tags rows {logged_tag!r} but the shipping head is "
                              f"{expected_tag!r}")
        return self._emit("PASS", check, f"tag {logged_tag!r} matches shipping head")

    def log_board_agreement(self, pushed_ids, logged_ids_today, check="log_board_agreement"):
        """What the log says was picked matches what was last published.

        Incident 8: the delta gate suppresses unchanged rows, and a game that dropped off
        the card or flipped side left Discord and the log disagreeing. Every id that
        reached the board must have a log row for today; a floor-cleared log row that was
        never pushed is a divergence.
        """
        pushed = {str(x) for x in pushed_ids}
        logged = {str(x) for x in logged_ids_today}
        missing_log = pushed - logged
        if missing_log:
            return self._emit("FAIL", check,
                              f"{len(missing_log)} boarded game(s) have no log row today "
                              f"[{', '.join(sorted(missing_log)[:6])}]")
        return self._emit("PASS", check, f"{len(pushed)} boarded game(s) all present in today's log")

    def skip(self, check, reason):
        """Record an invariant that could not be tested at all (e.g. out of season)."""
        return self._emit("SKIP", check, reason)

    # -- close out -------------------------------------------------------------

    def require(self, *names):
        """Assert that named GUARDS actually executed this session (via register_guard).

        This is what turns "a protection was written but never exercised" into a failing
        check instead of a discovery. MMA shipped `freshness.py` in the repo but never
        imported it, so the guard never ran and nobody knew. It checks self._guards, which
        only an executed guard populates -- a runner cannot satisfy it by calling a check
        method by hand, because that is the exact bug (form without effect) we are catching.
        """
        for name in names:
            if name not in self._guards:
                self._emit("FAIL", f"required:{name}",
                           f"the '{name}' guard did not execute this session -- it is not "
                           f"wired into this runner (a hand-called check does not count)")
        return self

    def finish(self, strict=True, require=None):
        if require:
            self.require(*(require if isinstance(require, (list, tuple, set)) else [require]))
        c = self.counts
        print(f"MBM-CHECK END {self.label} pass={c['PASS']} fail={c['FAIL']} "
              f"warn={c['WARN']} skip={c['SKIP']}")
        if strict and c["FAIL"]:
            raise SystemExit(f"{self.label}: {c['FAIL']} invariant check(s) FAILED, "
                             f"refusing to trust this board")
        return c["FAIL"] == 0
