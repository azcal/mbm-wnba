"""
MBM Big-5 soccer DNB daily runner (rebuild v1, frozen spec).
- Refreshes current-season results from football-data.co.uk (tolerates preseason 404s)
- Ingests upcoming Big-5 fixtures (4-day horizon), predicts P(home | no draw)
- Grades logged predictions (draws = push), rolling metrics + last-300 tripwire
Train window frozen through 2025-26; 2026-27 is the forward test.
Usage: python daily_soccer.py   (env TEST_FIXTURES=path.csv to inject a test slate)
"""
import io, json, os, sys, datetime as dt
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import engine as E

H = {"User-Agent": "Mozilla/5.0"}
DIVS = {"E0": "EPL", "SP1": "LaLiga", "I1": "SerieA", "D1": "Bundesliga", "F1": "Ligue1"}
MATCHES = os.path.join(HERE, "data", "matches.csv")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C, A0 = SPEC["cfg"], SPEC["C"], SPEC["a0"]
TRAIN_THROUGH = "2526"

MIN_EDGE = 0.05  # required margin: threshold odds pay 5% above model breakeven

def min_odds(p):
    import math
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0


def season_code(d):
    y = d.year % 100
    return f"{y:02d}{y+1:02d}" if d.month >= 7 else f"{y-1:02d}{y:02d}"

def extend_seasons(code):
    while code not in E.SEASONS:
        last = E.SEASONS[-1]
        a = int(last[:2]) + 1
        E.SEASONS.append(f"{a:02d}{a+1:02d}")

def refresh_results():
    m = pd.read_csv(MATCHES, parse_dates=["date"])
    m["season"] = m.season.astype(str).str.zfill(4)
    cur = season_code(dt.date.today())
    extend_seasons(cur)
    new = []
    for div, lg in DIVS.items():
        r = requests.get(f"https://www.football-data.co.uk/mmz4281/{cur}/{div}.csv",
                         headers=H, timeout=60)
        if r.status_code != 200 or len(r.text) < 200:
            continue
        d = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip", encoding_errors="ignore")
        keep = [c for c in ("Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                            "HS", "AS", "HST", "AST") if c in d.columns]
        d = d[keep].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        d["league"] = lg
        d["season"] = cur
        new.append(d)
    if new:
        n = pd.concat(new, ignore_index=True)
        n["date"] = pd.to_datetime(n.Date, format="mixed", dayfirst=True)
        key = lambda x: x.league + "|" + x.date.dt.strftime("%Y-%m-%d") + "|" + x.HomeTeam + "|" + x.AwayTeam
        m = m[~key(m).isin(set(key(n)))]
        m = pd.concat([m, n], ignore_index=True).sort_values("date").reset_index(drop=True)
        m.to_csv(MATCHES, index=False)
        print(f"results refreshed: +{len(n)} rows this season")
    else:
        print("no current-season results yet (preseason)")
    return m

def fetch_fixtures(horizon_days=4):
    tf = os.environ.get("TEST_FIXTURES")
    if tf and os.path.exists(tf):
        d = pd.read_csv(tf)
    else:
        r = requests.get("https://www.football-data.co.uk/fixtures.csv", headers=H, timeout=60)
        d = pd.read_csv(io.StringIO(r.text.lstrip("\ufeff")), on_bad_lines="skip",
                        encoding_errors="ignore") if r.status_code == 200 else pd.DataFrame()
    if d.empty or "Div" not in d.columns:
        return pd.DataFrame()
    d = d[d.Div.isin(DIVS)].copy()
    if d.empty:
        return d
    d["league"] = d.Div.map(DIVS)
    d["date"] = pd.to_datetime(d.Date, format="mixed", dayfirst=True)
    today = pd.Timestamp(dt.date.today())
    d = d[(d.date >= today) & (d.date <= today + pd.Timedelta(days=horizon_days))]
    return d[["league", "date", "HomeTeam", "AwayTeam"]]

def run():
    m = refresh_results()
    fx = fetch_fixtures()
    pseudo = []
    for r in fx.itertuples():
        pseudo.append(dict(Date=r.date.strftime("%d/%m/%Y"), HomeTeam=r.HomeTeam,
                           AwayTeam=r.AwayTeam, FTHG=np.nan, FTAG=np.nan, HS=np.nan,
                           AS=np.nan, HST=np.nan, AST=np.nan, league=r.league,
                           season=season_code(r.date.date()), date=r.date))
    g_all = pd.concat([m, pd.DataFrame(pseudo)], ignore_index=True) if pseudo else m
    for c in ("HS", "AS", "HST", "AST"):
        g_all[c] = pd.to_numeric(g_all.get(c), errors="coerce")
    g_all = g_all.sort_values("date").reset_index(drop=True)
    for s in g_all.season.unique():
        extend_seasons(str(s).zfill(4))
    f = E.add_diffs(E.build(g_all, CFG))
    tr = f.season.apply(lambda s: E.SEASONS.index(str(s).zfill(4)) <= E.SEASONS.index(TRAIN_THROUGH)) \
        & f.hg.notna() & (~f.draw)
    sc = StandardScaler().fit(f.loc[tr, E.FEATS].values)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(
        sc.transform(f.loc[tr, E.FEATS].values), f.loc[tr, "y"].values)
    base = float(f.loc[tr, "y"].mean())

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["id", "date", "league", "home", "away", "p_home", "p_elo", "model"])

    if pseudo:
        rows = f[f.hg.isna()].copy()
        p = mdl.predict_proba(sc.transform(rows[E.FEATS].values))[:, 1]
        p = E.shrink(p, rows.min_gp.values, A0)
        pe = 1/(1+10**(-((rows.h_elo + CFG["HA"]) - rows.a_elo)/400))
        b = pd.DataFrame({"id": rows.league + "|" + pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d")
                                + "|" + rows.home + "|" + rows.away,
                          "date": pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d"),
                          "league": rows.league, "home": rows.home, "away": rows.away,
                          "p_home": p, "p_elo": pe.values, "model": "v1"})
        log = pd.concat([log[~log.id.isin(b.id)], b], ignore_index=True)
        print(f"\n=== MBM soccer DNB board ({len(b)} fixtures) ===")
        for _, r in b.sort_values(["date", "league"]).iterrows():
            print(f"{r.date} {r.league:<10} {r.home} vs {r.away}  p(home|no draw)={r.p_home:.3f}  (elo {r.p_elo:.3f})  "
                  f"DNB value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}")
    else:
        print("no upcoming Big-5 fixtures in horizon")

    res = m.assign(id=m.league + "|" + m.date.dt.strftime("%Y-%m-%d") + "|" + m.HomeTeam + "|" + m.AwayTeam)
    exact = res.set_index("id")[["FTHG", "FTAG"]]
    def grade(row):
        if row.id in exact.index:
            hg, ag = exact.loc[row.id]
        else:
            c = res[(res.league == row.league) & (res.HomeTeam == row.home) & (res.AwayTeam == row.away)]
            c = c[(c.date - pd.Timestamp(row.date)).abs() <= pd.Timedelta(days=14)]
            if c.empty:
                return np.nan
            hg, ag = c.iloc[0][["FTHG", "FTAG"]]
        return -1.0 if hg == ag else float(hg > ag)
    log["result"] = log.apply(grade, axis=1)
    log.to_csv(pred_path, index=False)
    dec = log[log.result.isin([0.0, 1.0])]
    if len(dec) >= 30:
        y = dec.result.astype(int).values
        print(f"\nrolling grade, decided only (pushes: {(log.result == -1).sum():.0f}):")
        for lbl, p in (("model v1 ", dec.p_home), ("elo floor", dec.p_elo),
                       ("base rate", pd.Series(np.full(len(y), base)))):
            pc = np.clip(p.values.astype(float), 1e-6, 1-1e-6)
            print(f"{lbl}: n={len(y)}  LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        w = dec.tail(300)
        if len(w) >= 300:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-300 edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("Soccer DNB")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch:
        push = {}
        if pseudo:
            for _, r in b.iterrows():
                push[r.id] = {"line": (f"{r.league} {r.home} vs {r.away}: {r.p_home:.3f} | "
                                       f"DNB {r.home} >= {min_odds(r.p_home):.2f} / "
                                       f"{r.away} >= {min_odds(1-r.p_home):.2f}"),
                              "min_odds": min_odds(r.p_home)}
        mode, lines = tg_delta.gate(LOGS, dt.date.today().strftime("%Y%m%d"), push)
        if lines or alerts:
            txt = "\n".join(lines) if lines else "no price changes since the last push"
            if mode == "delta" and lines:
                txt = "changes since the last push:\n" + txt
            if alerts:
                txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
            requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                          data={"chat_id": ch, "text": f"MBM soccer DNB\n{txt}"}, timeout=20)

if __name__ == "__main__":
    run()
