"""
MBM WNBA daily runner (rebuild v1.1, frozen spec).
- Downloads wehoop parquet history, fetches today's slate + ESPN injury report
- Builds features as of now, fits frozen-window model (train 2019-2025 only), predicts slate
- Applies early-season shrink (a0 from model_spec.json, fitted pre-2026)
- Grades all past logged predictions vs finals, reports rolling metrics vs Elo/base-rate floors
Usage: python daily_wnba.py [--date YYYYMMDD]
"""
import argparse, json, os, sys, datetime as dt
import numpy as np, pandas as pd, requests
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import board_format as BF
import espn_roster as R
import espn_backfill as EB
import freshness as FR
import notify

# A stale copy of a shared file shows up as a wrong-looking board or an AttributeError
# halfway through a run. This turns it into a named failure in the workflow's import
# guard, which takes seconds. Sits above the features import so it fires before anything
# heavy loads. A minimum rather than an exact match, so a later improvement to a
# shared file does not force an edit here; raise it only for a change this runner needs.
SHARED_MIN = "2026-08-03"   # board_format.line() field separator, which this board uses
for _mod in (BF, notify, tg_delta, R, EB, FR):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

os.environ.setdefault("MBM_DATA", os.path.join(HERE, "data"))
import features as F
F.DATA = os.environ["MBM_DATA"]
F.YEARS = list(range(2018, dt.date.today().year + 1))

RAW = "https://raw.githubusercontent.com/sportsdataverse/wehoop-wnba-data/main/wnba"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
LEAGUE = "WNBA"          # label on the injury log rows
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "model_spec.json")))
CFG, C, A0 = SPEC["cfg"], SPEC["C"], SPEC["a0"]
TRAIN_LO, TRAIN_HI = 2019, 2025  # frozen through end of 2025; rest of 2026 = forward test

# Tracks board_format rather than restating it. This is the console-side copy, and a
# local 0.05 here while board_format ran at 0.00 meant the Actions log and Discord
# quoted different prices for the same game.
MIN_EDGE = BF.MIN_EDGE

def min_odds(p):
    import math
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0


def download_data():
    os.makedirs(F.DATA, exist_ok=True)
    cur = dt.date.today().year
    for y in F.YEARS:
        for kind, path in (("team_box", f"team_box/parquet/team_box_{y}.parquet"),
                           ("sched", f"schedules/parquet/wnba_schedule_{y}.parquet"),
                           ("player", f"player_box/parquet/player_box_{y}.parquet")):
            fn = os.path.join(F.DATA, os.path.basename(path))
            if os.path.exists(fn) and y < cur:
                continue
            r = requests.get(f"{RAW}/{path}", timeout=60)
            if r.status_code == 200:
                open(fn, "wb").write(r.content)

def _avail_years():
    """Keep only seasons with a COMPLETE parquet set, not just a schedule.

    The mirror publishes a season's schedule as soon as the fixtures are announced and the
    box scores only once games are played, so between those two moments a season has a
    schedule and nothing else. Keying on the schedule alone puts it in YEARS and
    load_team_box then opens a file that was never downloaded.

    Not hypothetical: the NBA board died exactly this way on 2026-08-19, when hoopR served
    nba_schedule_2027 (HTTP 200) alongside a 404 for team_box_2027. WNBA is mid-season so
    all three files exist today, and it would have hit the same wall next May.
    """
    keep = []
    for y in F.YEARS:
        names = (f"wnba_schedule_{y}.parquet", f"team_box_{y}.parquet",
                 f"player_box_{y}.parquet")
        if all(os.path.exists(os.path.join(F.DATA, n)) for n in names):
            keep.append(y)
        elif os.path.exists(os.path.join(F.DATA, names[0])):
            missing = [n for n in names[1:]
                       if not os.path.exists(os.path.join(F.DATA, n))]
            print(f"season {y}: schedule present but {', '.join(missing)} not published "
                  f"yet, skipping until it is")
    F.YEARS = keep

def fetch_slate(datestr):
    r = requests.get(f"{ESPN}/scoreboard", params={"dates": datestr}, timeout=30).json()
    out = []
    for e in r.get("events", []):
        # Only games that have not started. `completed` alone is not enough: it is False
        # for a game currently in progress, which would put a pre-game price on a board
        # for something already under way. state is "pre", "in" or "post".
        if e.get("status", {}).get("type", {}).get("state") != "pre":
            continue
        c = e["competitions"][0]
        t = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
        h, a = F.norm_abbr(t["home"]), F.norm_abbr(t["away"])
        if h in F.CITY and a in F.CITY:
            out.append({"game_id": str(e["id"]), "home": h, "away": a,
                        "neutral": bool(c.get("neutralSite", False)),
                        "tip": e.get("date", ""), "status": e["status"]["type"]["name"]})
    return out

def rosters_and_outs(day):
    """Tonight's rosters and who is unavailable, in one sweep of the roster endpoint.

    Membership used to be inferred from who appeared in the team's last three box scores.
    That is wrong for as long as it takes a traded player to appear in one, and wrong in
    both directions at once: missing from her new team, still counted for her old one.
    Each athlete on the roster response carries their own injuries array, so this also
    replaces the separate /injuries call.
    """
    ros = R.fetch(ESPN)
    outs = R.out_ids(ros, os.path.join(HERE, "overrides.json"))
    if ros["by_ab"]:
        R.log_statuses(ros, LOGS, day, LEAGUE)
    return ros, outs


def expected_actives(minutes, team, ros, outs):
    """Players on tonight's roster who have logged minutes recently and are not out.

    `minutes` is keyed by athlete across the whole league, so a player traded in this
    week is scored on the minutes she has actually been playing rather than on the zero
    she has with her new team.
    """
    ids = ros["by_ab"].get(team, set())
    return [(aid, minutes[aid], 0.0, 0.0) for aid in ids
            if aid in minutes and aid not in outs]


def expected_actives_from_box(pbox_hist, g_hist, team, out_ids):
    """Fallback when the roster feed is unreachable: the old last-3-games rotation.

    Kept so a feed outage degrades to the previous behaviour instead of posting an
    empty board.
    """
    games = g_hist[(g_hist.home_abbreviation == team) | (g_hist.away_abbreviation == team)]
    recent = games.sort_values("game_date").game_id.tail(3).tolist()
    seen = {}
    for gid in recent:
        for aid, m, gs, pm in pbox_hist.get(gid, {}).get(team, []):
            if m and m > 0:
                seen[str(aid)] = max(seen.get(str(aid), 0.0), float(m))
    return [(aid, m, 0.0, 0.0) for aid, m in seen.items() if aid not in out_ids]


def grade_from_espn(log):
    """Fill in results the wehoop mirror has not published yet.

    Grading rides on wehoop's parquet mirrors, which lag. Measured 2026-08-06: its latest
    completed WNBA game was 2026-08-01, while ESPN had every game through 2026-08-05 with
    final scores. Four days of forecasts sat ungraded, which stalls the rolling metrics,
    the tripwire and anything downstream that reads this log.

    ESPN is already the fixture source for this board, so this adds no new dependency. It
    only touches rows the mirror has left blank, so a wehoop-graded row is never
    overwritten, and it walks back at most two weeks.
    """
    if "result" not in log.columns or log.empty:
        return log
    missing = log[log.result.isna()]
    if missing.empty:
        return log
    days = sorted({str(d)[:8].replace("-", "")
                   for d in missing.date.astype(str)}, reverse=True)[:14]
    found = {}
    for day in days:
        try:
            r = requests.get(f"{ESPN}/scoreboard", params={"dates": day}, timeout=30).json()
        except Exception:
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
            if hs == as_:
                continue
            found[str(e["id"])] = 1.0 if hs > as_ else 0.0
    if not found:
        return log
    fill = log.game_id.astype(str).map(found)
    n = int((log.result.isna() & fill.notna()).sum())
    if n:
        log["result"] = log.result.fillna(fill)
        print(f"graded {n} row(s) from ESPN that the wehoop mirror has not published yet")
    return log


def run(datestr):
    download_data()
    # The wehoop mirror runs behind, so top the downloaded parquets up from ESPN before
    # anything reads them. Measured 2026-08-08: the mirror stopped at 2026-08-01 and
    # eighteen completed games were invisible to the Elo, the form decays and the
    # availability features. This is separate from grading: that fixes the record, this
    # fixes the inputs.
    EB.patch_parquets(ESPN, F.DATA, dt.datetime.strptime(datestr, "%Y%m%d").year,
                      label="WNBA: ")
    _avail_years()
    g = F.load_games()
    tbox = F.load_team_box()
    pbox = F.load_player_box()
    pbox = {gid: {tm: [(str(a), m, gs, pm) for a, m, gs, pm in v] for tm, v in d.items()}
            for gid, d in pbox.items()}

    # Refuse to price tonight on a history the source knows is incomplete. This is not an
    # "is it today's data" test: a league that did not play yesterday is current with old
    # data. It asks whether any completed game exists that this history has not seen.
    _newest = pd.to_datetime(g[g.home_win.notna()].game_date).max().date()
    _miss, _ = FR.missed_games("basketball/wnba", _newest)
    FR.guard("WNBA", _newest, "basketball/wnba")
    FRESH_LINE = FR.board_line(_newest, checked=(_newest, _miss))
    SEASON_LINE = None

    slate = fetch_slate(datestr)
    ros, outs = rosters_and_outs(datestr)
    gdate = pd.Timestamp(dt.datetime.strptime(datestr, "%Y%m%d").date())
    # Minutes are looked up by athlete across the league, not within a team, so a player
    # traded in this week is scored on what she has actually been playing.
    mins = R.recent_minutes(pbox, g, gdate)

    pseudo = []
    for s in slate:
        pbox[s["game_id"]] = {
            tm: (expected_actives(mins, tm, ros, outs) if ros["by_ab"].get(tm)
                 else expected_actives_from_box(pbox, g, tm, outs))
            for tm in (s["home"], s["away"])}
        pseudo.append({"game_id": s["game_id"], "game_date": gdate, "game_date_time": gdate,
                       "season": gdate.year, "season_type": 2, "home_abbreviation": s["home"],
                       "away_abbreviation": s["away"], "home_score": np.nan, "away_score": np.nan,
                       "neutral_site": s["neutral"], "home_win": np.nan, "neutral": s["neutral"]})
    g_all = pd.concat([g, pd.DataFrame(pseudo)], ignore_index=True) if pseudo else g
    g_all = g_all.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    df = F.add_diffs(F.build_features(g_all, tbox, pbox, CFG))
    tr = (df.season >= TRAIN_LO) & (df.season <= TRAIN_HI) & df.home_win.notna()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(df.loc[tr, F.NEW_FEATS].values)
    m = LogisticRegression(C=C, max_iter=2000).fit(
        sc.transform(df.loc[tr, F.NEW_FEATS].values), df.loc[tr, "home_win"].astype(int).values)

    def predict(rows):
        p = m.predict_proba(sc.transform(rows[F.NEW_FEATS].values))[:, 1]
        mg = rows[["h_gp", "a_gp"]].min(axis=1).values
        alpha = A0 + (1 - A0) * np.minimum(mg, 10) / 10.0
        lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-alpha * lo))

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date", "game_id", "home", "away", "p_home", "p_home_elo", "model"])

    # Defined before the branch. On a day with no games the delivery block below still
    # runs, and an undefined `push` would raise NameError after the model has already
    # graded everything. This was hidden while delivery was gated behind the TG secrets.
    push, skipped = {}, 0
    if slate:
        rows = df[df.game_id.isin([s["game_id"] for s in slate])].copy()
        rows["p_home"] = predict(rows)
        eh = 1 / (1 + 10 ** (-((rows.h_elo + np.where(rows.neutral == 1, 0, CFG["elo_ha"])) - rows.a_elo) / 400))
        rows["p_home_elo"] = eh
        board = rows[["game_id", "home", "away", "p_home", "p_home_elo"]].copy()
        board.insert(0, "date", datestr)
        board["model"] = "v1.1"
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board], ignore_index=True)
        rows = [(datestr[:4] + "-" + datestr[4:6] + "-" + datestr[6:],
                 f"{r.away} @ {r.home}", r.home if r.p_home >= .5 else r.away,
                 max(r.p_home, 1 - r.p_home), r.game_id) for _, r in board.iterrows()]
        wg = max([len(x[1]) for x in rows] + [10]); wp = max([len(x[2]) for x in rows] + [8])
        for d_, gm, pk, q, gid in rows:
            if BF.min_odds(q) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            ln = BF.line(d_, gm, pk, q, w_game=wg, w_pick=wp)
            note = BF.live_note("WNBA", q)
            if note:
                ln = "\n".join([ln] + note)
            push[gid] = {"line": ln, "min_odds": BF.min_odds(q)}
        print(f"\n=== MBM WNBA board {datestr} ===")
        for v in push.values():
            print(v["line"])
        if skipped:
            print(f"({skipped} game{'s' if skipped != 1 else ''} skipped, "
                  f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)})")
    else:
        print(f"no games on {datestr}")

    # grade everything gradeable
    fin = g.set_index("game_id")["home_win"]
    log["result"] = log.game_id.map(fin)
    log = grade_from_espn(log)
    graded = log[log.result.notna()].copy()
    log.to_csv(pred_path, index=False)
    if len(graded) >= 5:
        y = graded.result.astype(int).values
        for name, p in (("model v1.1", graded.p_home.values), ("Elo floor", graded.p_home_elo.values)):
            p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
            print(f"{name:>12}: n={len(y)}  LL={log_loss(y, p):.4f}  Brier={brier_score_loss(y, p):.4f}  "
                  f"hit={np.mean((p > .5) == (y == 1)):.3f}")
        base = df.loc[tr, "home_win"].mean()
        print(f"{'base rate':>12}: LL={log_loss(y, np.full_like(y, base, dtype=float)):.4f}")
        w = graded.tail(200)
        if len(w) >= 200:
            yw = w.result.astype(int).values
            edge = log_loss(yw, np.full(len(yw), float(base))) - \
                log_loss(yw, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-200 edge vs base rate: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("WNBA winner")
        graded.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    # Delivery runs unconditionally. notify.send() decides for itself whether a webhook is
    # configured and prints when one is not, which is the check that used to sit here as a
    # Telegram secrets test. Gating on TG_TOKEN meant a Discord-only repo posted nothing
    # while printing a message that read like the delta gate had done its job.
    mode, lines = tg_delta.gate(LOGS, datestr, push)
    if lines or alerts:
        body = list(lines) if lines else ["no price changes since the last push"]
        foot = (f"{skipped} game{'s' if skipped != 1 else ''} skipped, "
                f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)}")\
               if skipped else None
        title = "MBM WNBA" + (" · changes since the last push"
                                 if mode == "delta" and lines else "")
        colour = notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " below base rate, stop betting this board"]
            colour = notify.RED
        # The board carries its own provenance. A reader should not have to trust that
        # the data was current; the row says so.
        foot = " · ".join([x for x in (FRESH_LINE, SEASON_LINE, foot) if x])
        notify.send(title, body, foot, colour)
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.datetime.now().strftime("%Y%m%d"))
    run(ap.parse_args().date)
