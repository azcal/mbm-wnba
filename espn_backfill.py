"""
Fill the gap between the sportsdataverse mirror and today, from ESPN.

The WNBA and NBA boards build their features from wehoop and hoopR parquet mirrors. Those
are community rebuilds and they run behind. Measured 2026-08-08: wehoop's latest WNBA team
box was 2026-08-01, so **eighteen completed games** were invisible to the Elo, the form
decays and the availability features. The board was pricing tonight on a league table a
week out of date.

That is a separate problem from grading. `grade_from_espn` in the runner fills the `result`
column of the forecast log, which fixes the record. This fixes the *inputs*: it returns
games, team box and player box rows in the mirror's own shape, for the days the mirror has
not reached, so they can be concatenated straight onto the loaded frames.

One summary call per missing game, so a week's gap is a couple of dozen requests. On a
normal day the mirror is current, nothing is missing, and this makes a single scoreboard
call and returns empty.

Fails safe: any error returns empty frames and the board runs on the mirror alone, exactly
as it did before.
"""
import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

VERSION = "2026-08-08"

# ESPN 403s any custom User-Agent, so send none. See the note in espn_roster.py.
H = {}


def _pair(v):
    """'26-64' -> (26.0, 64.0). ESPN packs made and attempted into one string."""
    try:
        a, b = str(v).split("-")
        return float(a), float(b)
    except Exception:
        return np.nan, np.nan


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _minutes(v):
    """Player minutes. ESPN gives whole minutes here, but tolerate 'MM:SS' too."""
    s = str(v).strip()
    if ":" in s:
        try:
            m, sec = s.split(":")
            return float(m) + float(sec) / 60.0
        except Exception:
            return np.nan
    return _num(s)


def completed_since(base, after, until=None, timeout=30):
    """Completed game ids and their metadata for every day after `after`.

    `after` is the mirror's latest date, so the first day fetched is the one following it.
    """
    until = until or dt.date.today()
    out = []
    day = after + dt.timedelta(days=1)
    while day <= until:
        try:
            r = requests.get(f"{base}/scoreboard", headers=H,
                             params={"dates": day.strftime("%Y%m%d")},
                             timeout=timeout).json()
        except Exception:
            day += dt.timedelta(days=1)
            continue
        for e in r.get("events", []):
            if not e.get("status", {}).get("type", {}).get("completed"):
                continue
            c = e["competitions"][0]
            sides = {x["homeAway"]: x for x in c.get("competitors", [])}
            if "home" not in sides or "away" not in sides:
                continue
            try:
                hs, as_ = float(sides["home"]["score"]), float(sides["away"]["score"])
            except (TypeError, ValueError):
                continue
            out.append({
                "game_id": str(e["id"]),
                "game_date": pd.Timestamp(day),
                "game_date_time": pd.Timestamp(day),
                "season": day.year,
                "season_type": 2,
                "home_abbreviation": sides["home"]["team"].get("abbreviation"),
                "away_abbreviation": sides["away"]["team"].get("abbreviation"),
                "home_score": hs, "away_score": as_,
                "home_win": 1.0 if hs > as_ else 0.0,
                "neutral_site": bool(c.get("neutralSite", False)),
                "neutral": bool(c.get("neutralSite", False)),
                "notes_headline": "", "notes": ""})
        day += dt.timedelta(days=1)
    cols = ["game_id", "game_date", "game_date_time", "season", "season_type",
            "home_abbreviation", "away_abbreviation", "home_score", "away_score",
            "home_win", "neutral_site", "neutral", "notes_headline", "notes"]
    # Always the same columns, so an empty result is still a usable frame.
    return pd.DataFrame(out, columns=cols)


def boxes(base, games, timeout=30, pause=0.3):
    """Team and player box rows for the given games, in the mirror's column shape."""
    tb, pb = [], []
    for g in games.itertuples():
        try:
            j = requests.get(f"{base}/summary", headers=H,
                             params={"event": g.game_id}, timeout=timeout).json()
        except Exception:
            continue
        bs = j.get("boxscore", {})

        for t in bs.get("teams", []):
            ab = t.get("team", {}).get("abbreviation")
            if not ab:
                continue
            st = {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])}
            fgm, fga = _pair(st.get("fieldGoalsMade-fieldGoalsAttempted"))
            _, _ = _pair(st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            tpm, _tpa = _pair(
                st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            ftm, fta = _pair(st.get("freeThrowsMade-freeThrowsAttempted"))
            opp = g.away_abbreviation if ab == g.home_abbreviation else g.home_abbreviation
            own = g.home_score if ab == g.home_abbreviation else g.away_score
            oth = g.away_score if ab == g.home_abbreviation else g.home_score
            tb.append({
                # game_date has to be carried onto the box rows: the mirror stores it
                # there, and without it the appended rows read as null and every
                # freshness check still thinks the file stops where it did.
                "game_id": g.game_id, "game_date": g.game_date.date(),
                "season": g.season, "season_type": g.season_type,
                "team_abbreviation": ab,
                "opponent_team_abbreviation": opp,
                "team_score": own, "opponent_team_score": oth,
                "field_goals_made": fgm, "field_goals_attempted": fga,
                "three_point_field_goals_made": tpm,
                "free_throws_made": ftm, "free_throws_attempted": fta,
                "offensive_rebounds": _num(st.get("offensiveRebounds")),
                "defensive_rebounds": _num(st.get("defensiveRebounds")),
                "total_turnovers": _num(st.get("totalTurnovers"))})

        for blk in bs.get("players", []):
            ab = blk.get("team", {}).get("abbreviation")
            for s in blk.get("statistics", []):
                lab = [str(x).upper() for x in (s.get("labels") or [])]
                idx = {k: lab.index(k) for k in
                       ("MIN", "PTS", "AST", "REB", "OREB", "DREB", "TO", "STL", "BLK",
                        "PF", "FG", "FT", "+/-") if k in lab}
                if "MIN" not in idx or "PTS" not in idx:
                    continue
                for a in s.get("athletes", []):
                    if a.get("didNotPlay"):
                        continue
                    v = a.get("stats") or []
                    if len(v) < len(lab):
                        continue
                    gv = lambda k: v[idx[k]] if k in idx else None      # noqa: E731
                    fgm, fga = _pair(gv("FG"))
                    ftm, fta = _pair(gv("FT"))
                    orb, drb = _num(gv("OREB")), _num(gv("DREB"))
                    if np.isnan(orb) and "REB" in idx:
                        orb, drb = 0.0, _num(gv("REB"))
                    pb.append({
                        "game_id": g.game_id, "game_date": g.game_date.date(),
                        "season": g.season, "season_type": g.season_type,
                        "team_abbreviation": ab,
                        "athlete_id": str((a.get("athlete") or {}).get("id", "")),
                        "min": _minutes(gv("MIN")), "points": _num(gv("PTS")),
                        "assists": _num(gv("AST")),
                        "offensive_rebounds": orb, "defensive_rebounds": drb,
                        "field_goals_made": fgm, "field_goals_attempted": fga,
                        "free_throws_made": ftm, "free_throws_attempted": fta,
                        "turnovers": _num(gv("TO")), "steals": _num(gv("STL")),
                        "blocks": _num(gv("BLK")), "fouls": _num(gv("PF")),
                        "plus_minus": _num(str(gv("+/-")).replace("+", ""))})
        time.sleep(pause)
    return pd.DataFrame(tb), pd.DataFrame(pb)


def patch_parquets(base, data_dir, year, label=""):
    """Top up the downloaded parquets with whatever the mirror has not reached.

    Appending to the files on disk rather than to the loaded frames means features.py
    needs no change: it derives game score, possessions and the rest from these columns
    exactly as it does for mirror rows.

    Three things this has to get right, each of which broke a first attempt:

    The team box is the anchor, not the schedule. wehoop's schedule carries every game of
    the season up front, so its latest date is the end of September and nothing ever looks
    missing.

    Unplayed schedule rows hold 0-0 rather than null, and features.py drops those on the
    `score > 0` test. So a game that has now been played needs its schedule row UPDATED,
    not appended, or the row stays 0-0 and the game stays invisible.

    Box rows are genuinely absent and are appended.

    Returns games added. Zero on any failure, and the board runs on the mirror alone.
    """
    import os
    sched = os.path.join(data_dir, f"wnba_schedule_{year}.parquet")
    tbp = os.path.join(data_dir, f"team_box_{year}.parquet")
    pbp = os.path.join(data_dir, f"player_box_{year}.parquet")
    if not all(os.path.exists(p) for p in (sched, tbp, pbp)):
        return 0
    try:
        tbox = pd.read_parquet(tbp)
        last = pd.to_datetime(tbox.game_date, errors="coerce").max().date()
        new = completed_since(base, last)
        have_box = set(tbox.game_id.astype(str))
        new = new[~new.game_id.astype(str).isin(have_box)]
        if not len(new):
            print(f"{label}feature data current to {last}")
            return 0
        tb, pb = boxes(base, new)
        if not len(tb) or not len(pb):
            print(f"{label}found {len(new)} game(s) but no box data, leaving files alone")
            return 0

        sch = pd.read_parquet(sched)
        sch["game_id"] = sch.game_id.astype(str)
        idx = sch.set_index("game_id")
        upd = 0
        for r in new.itertuples():
            if r.game_id in idx.index:
                for col, val in (("home_score", r.home_score), ("away_score", r.away_score)):
                    if col in sch.columns:
                        sch.loc[sch.game_id == r.game_id, col] = val
                upd += 1
        missing = new[~new.game_id.astype(str).isin(set(sch.game_id))]
        if len(missing):
            sch = pd.concat([sch, _align(missing, sch)], ignore_index=True)
        sch.to_parquet(sched, index=False)

        pd.concat([tbox, _align(tb, tbox)], ignore_index=True).to_parquet(tbp, index=False)
        pbox = pd.read_parquet(pbp)
        pd.concat([pbox, _align(pb, pbox)], ignore_index=True).to_parquet(pbp, index=False)

        print(f"{label}feature data stopped at {last}; added {len(new)} game(s) from ESPN "
              f"through {new.game_date.max().date()} "
              f"({upd} schedule row(s) updated, {len(tb)} team, {len(pb)} player)")
        return len(new)
    except Exception as ex:
        print(f"{label}could not top up feature data ({type(ex).__name__}: {ex}), "
              f"running on the mirror alone")
        return 0


def _align(add, old):
    """Give `add` exactly `old`'s columns and dtypes.

    Column names alone are not enough: parquet stores game_id as int64 while these frames
    build it as text, and pyarrow refuses to concatenate the two. Anything that will not
    cast is left as-is rather than dropped, so a surprise never silently loses a column.
    """
    out = add.copy()
    for c in old.columns:
        if c not in out.columns:
            out[c] = np.nan
            continue
        want = old[c].dtype
        if out[c].dtype == want:
            continue
        try:
            if pd.api.types.is_integer_dtype(want):
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(want)
            elif pd.api.types.is_float_dtype(want):
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(want)
            elif pd.api.types.is_bool_dtype(want):
                out[c] = out[c].fillna(False).astype(bool)
            else:
                out[c] = out[c].astype(want)
        except Exception:
            pass
    return out[old.columns.tolist()]


def fill(base, g, tbox_rows, pbox_rows, label=""):
    """Everything the mirror is missing, as (games, team_box, player_box).

    `g` is the loaded games frame; its latest date decides where to start. Empty frames on
    any failure, so the caller can concatenate unconditionally.
    """
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    try:
        if g is None or not len(g):
            return empty
        last = pd.to_datetime(g.game_date).max().date()
        if last >= dt.date.today():
            return empty
        new = completed_since(base, last)
        if not len(new):
            print(f"{label}mirror current to {last}, nothing to fill")
            return empty
        new = new[~new.game_id.astype(str).isin(set(g.game_id.astype(str)))]
        if not len(new):
            return empty
        tb, pb = boxes(base, new)
        print(f"{label}mirror stops at {last}; filled {len(new)} game(s) from ESPN "
              f"through {new.game_date.max().date()} "
              f"({len(tb)} team rows, {len(pb)} player rows)")
        return new, tb, pb
    except Exception as ex:
        print(f"{label}ESPN fill failed ({type(ex).__name__}), running on the mirror alone")
        return empty
