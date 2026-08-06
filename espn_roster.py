"""
Roster membership and injury status from ESPN, for the basketball boards.

Two problems this exists to fix.

**Team membership was inferred from box scores.** Both game boards built a roster as
"whoever appeared in this team's last three games". That is wrong for exactly as long as
it takes a traded player to show up in a box score, and wrong in both directions at once:
she is missing from her new team while still being counted for her old one. The props
board had the same lag through `latest.team_id`, which reads off a player's most recent
appearance. The roster endpoint answers the question directly.

**The injury feed is current-state only.** It says who is Day-To-Day right now. It never
says who was Day-To-Day last Tuesday, or whether they played. Without that history there
is no way to weight a Day-To-Day player by anything except assertion, so `log_statuses`
starts accumulating one. It changes no forecast today.

Each athlete on the roster response carries their own `injuries` array, so one call per
team replaces both the `/teams` sweep and the separate `/injuries` call the runners used
to make.

Not one of the universal shared files. It lives only in the three basketball repos, and
those runners include it in their own version check.
"""
import os

import pandas as pd
import requests

VERSION = "2026-08-03"

# ESPN 403s ANY custom User-Agent and only serves the requests default. Measured
# 2026-08-05 on the WNBA scoreboard: no header 200, "Mozilla/5.0" 403, an honest
# identifier with contact details 403, a full browser header set 403. mbm-wnba has
# always worked because it never set one. Do not "fix" this by adding a UA.
ESPN_H = {}
H = ESPN_H

# Statuses that remove a player from tonight's actives.
#
# Measured against the live WNBA feed on 2026-08-03, the only strings in use were
# "Out" and "Day-To-Day". The others are here because the NBA feed carries them and
# costs nothing to match. Day-To-Day is deliberately absent: it is genuinely ambiguous,
# and forcing it to out is as unmeasured as forcing it to in. It stays counted as
# available until the log below can say how often it becomes played.
OUT_STATUSES = {"out", "suspension", "doubtful"}


def _flat(athletes):
    """ESPN returns a flat list for both basketball leagues, and a position-nested one
    for some others. Handle both rather than depending on which."""
    if athletes and isinstance(athletes[0], dict) and "items" in athletes[0]:
        return [x for grp in athletes for x in grp.get("items", [])]
    return athletes


def fetch(base, timeout=30, only=None):
    """Pull every team's roster.

    Returns a dict with:
        by_ab   {team abbreviation: set of athlete id}
        by_id   {espn team id: set of athlete id}
        ab      {espn team id: team abbreviation}
        status  {athlete id: raw status string}
        name    {athlete id: display name}
        team    {athlete id: team abbreviation}

    Empty dicts on any failure, so a caller can fall back to its old path rather than
    posting an empty board.
    """
    out = {"by_ab": {}, "by_id": {}, "ab": {}, "status": {}, "name": {}, "team": {}}
    try:
        t = requests.get(f"{base}/teams", headers=H, timeout=timeout).json()
        teams = t["sports"][0]["leagues"][0]["teams"]
    except Exception as ex:
        print(f"roster: team list unavailable ({type(ex).__name__})")
        return out
    for entry in teams:
        tm = entry.get("team", {})
        tid, ab = str(tm.get("id", "")), tm.get("abbreviation", "")
        if not tid or not ab:
            continue
        # `only` limits the sweep to the teams a caller actually needs, which on a
        # four-game night is eight calls instead of thirty.
        if only is not None and tid not in {str(x) for x in only}:
            continue
        try:
            r = requests.get(f"{base}/teams/{tid}/roster", headers=H, timeout=timeout).json()
        except Exception:
            continue
        ids = set()
        for a in _flat(r.get("athletes", [])):
            aid = str(a.get("id", ""))
            if not aid:
                continue
            ids.add(aid)
            out["name"][aid] = a.get("displayName", "")
            out["team"][aid] = ab
            inj = a.get("injuries") or []
            if inj:
                out["status"][aid] = str(inj[0].get("status", ""))
        if ids:
            out["by_ab"][ab] = ids
            out["by_id"][tid] = ids
            out["ab"][tid] = ab
    print(f"roster: {len(out['by_ab'])} teams, {len(out['name'])} players, "
          f"{len(out['status'])} carrying an injury status")
    return out


def out_ids(rosters, overrides_path=None):
    """Athlete ids to remove from tonight's actives, feed plus manual overrides.

    overrides.json keeps its existing shape, {"out": {"PHX": ["id", ...]}}, and unions
    with the feed rather than replacing it.
    """
    bad = {aid for aid, st in rosters.get("status", {}).items()
           if str(st).lower() in OUT_STATUSES}
    if overrides_path and os.path.exists(overrides_path):
        try:
            import json
            for _ab, ids in json.load(open(overrides_path)).get("out", {}).items():
                bad.update(str(x) for x in ids)
        except Exception as ex:
            print(f"roster: overrides.json unreadable ({type(ex).__name__}), ignoring")
    return bad


def log_statuses(rosters, logs_dir, day, league):
    """Append today's statuses to logs/injuries.csv, one row per listed player.

    Deduped on date and athlete, last write winning, so the afternoon run updates the
    morning's row rather than doubling it. Nothing reads this yet. In a couple of months
    it is the sample that lets a Day-To-Day weight be measured instead of asserted.
    """
    rows = [{"date": str(day), "league": league, "athlete_id": aid,
             "player": rosters["name"].get(aid, ""), "team": rosters["team"].get(aid, ""),
             "status": st}
            for aid, st in rosters.get("status", {}).items()]
    if not rows:
        return
    path = os.path.join(logs_dir, "injuries.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, dtype={"athlete_id": str})
            new = pd.concat([old, new], ignore_index=True)
        except Exception as ex:
            print(f"roster: injuries.csv unreadable ({type(ex).__name__}), starting fresh")
    os.makedirs(logs_dir, exist_ok=True)
    new = new.drop_duplicates(["date", "athlete_id"], keep="last")
    new.to_csv(path, index=False)
    print(f"injury log: {len(rows)} listed today, {len(new)} rows total")


def recent_minutes(pbox_hist, g_hist, day, days=21):
    """athlete id -> minutes in their most recent appearance, from any team's box score.

    Keyed by athlete rather than by team, which is the other half of the trade fix: a
    player traded in this week has no minutes with her new team, and looking them up by
    team would score her at zero. Looking them up by athlete carries the minutes she has
    actually been playing, for whoever she played them.

    The window matches the props board's recency rule, so a player who has not logged
    minutes in three weeks drops out of both the same way.
    """
    cut = pd.Timestamp(day) - pd.Timedelta(days=days)
    recent = g_hist[pd.to_datetime(g_hist.game_date) >= cut]
    mins = {}
    for gid in recent.sort_values("game_date").game_id:
        for lst in pbox_hist.get(gid, {}).values():
            for aid, m, _gs, _pm in lst:
                if m and m > 0:
                    mins[str(aid)] = float(m)      # later games overwrite earlier ones
    return mins
