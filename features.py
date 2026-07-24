"""
MustBeMoose WNBA rebuild v1 -- standalone predictive engine, no market inputs.
Data: sportsdataverse/wehoop ESPN mirrors (schedules, team box, player box), 2018-2026.
Protocol: tune on 2024 only. 2025 and 2026 YTD are frozen holdouts, touched once.
Outputs: comparison table (rebuild vs old-trait reimpl vs baselines), per-game 2026 log.
"""
import numpy as np, pandas as pd, math, json
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

DATA = "/home/claude/data"
YEARS = list(range(2018, 2027))
HHL = dict()  # tuned hyperparams land here

CITY = {  # team -> (lat, lon), current-era home cities
 "ATL": (33.75, -84.39), "CHI": (41.88, -87.63), "CON": (41.49, -72.09),
 "DAL": (32.73, -97.11), "IND": (39.77, -86.16), "LA": (34.04, -118.27),
 "LV": (36.10, -115.18), "MIN": (44.98, -93.27), "NY": (40.68, -73.97),
 "PHX": (33.45, -112.07), "SEA": (47.62, -122.35), "WSH": (38.90, -77.02),
 "GS": (37.77, -122.39), "TOR": (43.64, -79.38), "POR": (45.52, -122.67),
}
def hav(a, b):
    if a is None or b is None: return 0.0
    la1, lo1 = a; la2, lo2 = b
    p = math.pi / 180
    x = 0.5 - math.cos((la2-la1)*p)/2 + math.cos(la1*p)*math.cos(la2*p)*(1-math.cos((lo2-lo1)*p))/2
    return 12742 * math.asin(math.sqrt(x))

def norm_abbr(x):
    return {"CONN": "CON"}.get(x, x)

def load_games():
    rows = []
    for y in YEARS:
        s = pd.read_parquet(f"{DATA}/wnba_schedule_{y}.parquet")
        s = s[s.season_type.isin([2, 3])].copy()
        s["notes"] = s.get("notes_headline", pd.Series("", index=s.index)).fillna("")
        s = s[~s.notes.str.contains("All-Star", case=False)]
        s = s[~s.notes.str.contains("Commissioner's Cup Championship", case=False)]
        s = s[s.home_score.notna() & s.away_score.notna()]
        s = s[(s.home_score.astype(float) > 0) | (s.away_score.astype(float) > 0)]
        for c in ["home_abbreviation", "away_abbreviation"]:
            s[c] = s[c].map(norm_abbr)
        s = s[s.home_abbreviation.isin(CITY) & s.away_abbreviation.isin(CITY)]
        s["season"] = y
        rows.append(s[["game_id", "game_date", "game_date_time", "season", "season_type",
                       "home_abbreviation", "away_abbreviation", "home_score", "away_score",
                       "neutral_site"]])
    g = pd.concat(rows, ignore_index=True)
    g["game_id"] = g.game_id.astype(str)
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    g["home_win"] = (g.home_score.astype(float) > g.away_score.astype(float)).astype(int)
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    g.loc[g.season == 2020, "neutral"] = True  # bubble
    return g

def load_team_box():
    t = pd.concat([pd.read_parquet(f"{DATA}/team_box_{y}.parquet") for y in YEARS], ignore_index=True)
    t["game_id"] = t.game_id.astype(str)
    t["team_abbreviation"] = t.team_abbreviation.map(norm_abbr)
    t["opponent_team_abbreviation"] = t.opponent_team_abbreviation.map(norm_abbr)
    for c in ["field_goals_made","field_goals_attempted","three_point_field_goals_made",
              "free_throws_attempted","offensive_rebounds","defensive_rebounds",
              "total_turnovers","team_score","opponent_team_score"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["poss"] = t.field_goals_attempted - t.offensive_rebounds + t.total_turnovers + 0.44*t.free_throws_attempted
    t["efg"] = (t.field_goals_made + 0.5*t.three_point_field_goals_made) / t.field_goals_attempted
    t["tovr"] = t.total_turnovers / t.poss
    d = {}
    for _, r in t.iterrows():
        d[(r.game_id, r.team_abbreviation)] = r
    return d

def parse_min(m):
    try:
        if m is None or (isinstance(m, float) and np.isnan(m)): return 0.0
        s = str(m)
        if ":" in s:
            a, b = s.split(":"); return float(a) + float(b)/60
        return float(s)
    except Exception:
        return 0.0

def load_player_box():
    p = pd.concat([pd.read_parquet(f"{DATA}/player_box_{y}.parquet") for y in YEARS], ignore_index=True)
    p["game_id"] = p.game_id.astype(str)
    p["team_abbreviation"] = p.team_abbreviation.map(norm_abbr)
    p["min"] = p.minutes.map(parse_min)
    for c in ["points","field_goals_made","field_goals_attempted","free_throws_attempted",
              "free_throws_made","offensive_rebounds","defensive_rebounds","steals","assists",
              "blocks","fouls","turnovers","plus_minus"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    p["gmsc"] = (p.points + 0.4*p.field_goals_made - 0.7*p.field_goals_attempted
                 - 0.4*(p.free_throws_attempted - p.free_throws_made) + 0.7*p.offensive_rebounds
                 + 0.3*p.defensive_rebounds + p.steals + 0.7*p.assists + 0.7*p.blocks
                 - 0.4*p.fouls - p.turnovers)
    by_game = defaultdict(dict)
    for (gid, tm), grp in p.groupby(["game_id", "team_abbreviation"]):
        by_game[gid][tm] = grp[["athlete_id", "min", "gmsc", "plus_minus"]].values
    return by_game

class Decay:
    """exp-decayed weighted mean with season-boundary discount"""
    def __init__(self, hl): self.d = 0.5 ** (1.0/hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x, w=1.0): self.sw = self.sw*self.d + w; self.sx = self.sx*self.d + w*x
    def season_break(self, keep): self.sw *= keep; self.sx *= keep
    def val(self, default): return self.sx/self.sw if self.sw > 1e-9 else default

def build_features(g, tbox, pbox, cfg):
    HL_F, HL_S, CARRY, ELO_K, ELO_HA, TAU = (cfg["hl_fast"], cfg["hl_slow"], cfg["carry"],
                                             cfg["elo_k"], cfg["elo_ha"], cfg["tau"])
    LG_GMSC_PM = 0.36  # league mean game score per minute, computed from 2018 (pre-train)
    elo = defaultdict(lambda: 1500.0)
    fast = defaultdict(lambda: {k: Decay(HL_F) for k in ["net"]})
    slow = defaultdict(lambda: {k: Decay(HL_S) for k in ["net","ortg","drtg","efg","efga","tovr","orb"]})
    season_seen = {}
    std = defaultdict(lambda: defaultdict(list))     # season-to-date lists per team
    last10 = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))
    prev = {}                                        # team -> dict(date, city, dates deque)
    prior_season_net = defaultdict(lambda: 0.0)
    pstate = {}                                      # athlete -> dict of decayed sums
    last_lineup = {}                                 # team -> list of (athlete, minshare, rating)
    team_l10 = {}                                    # team -> deque of {athlete: minutes} per game
    lg_or = Decay(200)

    def p_rating(aid):
        st = pstate.get(aid)
        if st is None: return 0.0, 0.0
        pm = st["gmsc"]/(st["min"] + TAU) if st["min"] > 0 else 0.0
        return (pm - LG_GMSC_PM), st["mins_dec"].val(12.0)

    feats = []
    for _, r in g.iterrows():
        y, h, a, gid, date = r.season, r.home_abbreviation, r.away_abbreviation, r.game_id, r.game_date
        for tm in (h, a):
            if season_seen.get(tm) != y:
                if tm in season_seen:
                    for D in list(fast[tm].values()) + list(slow[tm].values()): D.season_break(CARRY)
                    elo[tm] = 1500 + CARRY*(elo[tm]-1500)
                    prior_season_net[tm] = np.mean(std[tm]["net"]) if std[tm]["net"] else 0.0
                else:
                    elo[tm] = 1500 - 100  # expansion / first appearance
                std[tm] = defaultdict(list); last10[tm] = defaultdict(lambda: deque(maxlen=10))
                prev.pop(tm, None)
                season_seen[tm] = y

        row = {"game_id": gid, "season": y, "date": date, "home": h, "away": a,
               "home_win": r.home_win, "neutral": int(r.neutral)}
        city = CITY[h]
        for side, tm in (("h", h), ("a", a)):
            pv = prev.get(tm)
            rest = min((date - pv["date"]).days, 10) if pv else 7
            b2b = 1 if (pv and (date - pv["date"]).days == 1) else 0
            g4 = sum(1 for d0 in (pv["dates"] if pv else []) if (date - d0).days <= 3)
            trav = 0.0 if r.season == 2020 else hav(pv["city"], city) if pv else 0.0
            gp = len(std[tm]["net"])
            row.update({f"{side}_rest": rest, f"{side}_b2b": b2b, f"{side}_3in4": int(g4 >= 2),
                        f"{side}_travel": trav/1000.0, f"{side}_gp": min(gp, 15),
                        f"{side}_elo": elo[tm],
                        f"{side}_net_f": fast[tm]["net"].val(0.0), f"{side}_net_s": slow[tm]["net"].val(0.0),
                        f"{side}_efg_s": slow[tm]["efg"].val(0.44), f"{side}_efga_s": slow[tm]["efga"].val(0.44),
                        f"{side}_tovr_s": slow[tm]["tovr"].val(0.17), f"{side}_orb_s": slow[tm]["orb"].val(0.28),
                        f"{side}_prior_net": prior_season_net[tm],
                        f"{side}_std_ortg": np.mean(std[tm]["ortg"]) if gp else 100.0,
                        f"{side}_std_drtg": np.mean(std[tm]["drtg"]) if gp else 100.0,
                        f"{side}_l10_ortg": np.mean(last10[tm]["ortg"]) if last10[tm]["ortg"] else 100.0,
                        f"{side}_l10_drtg": np.mean(last10[tm]["drtg"]) if last10[tm]["drtg"] else 100.0,
                        f"{side}_std_efg": np.mean(std[tm]["efg"]) if gp else 0.44,
                        f"{side}_std_efga": np.mean(std[tm]["efga"]) if gp else 0.44,
                        f"{side}_std_tovr": np.mean(std[tm]["tovr"]) if gp else 0.17,
                        f"{side}_std_orb": np.mean(std[tm]["orb"]) if gp else 0.28})
            # availability from today's actives (deployment: injury report + confirmed rotation)
            plist = pbox.get(gid, {}).get(tm, [])
            act = [(aid, m) for aid, m, gs, pm in plist if m and m > 0]
            window = team_l10.get(tm, deque())
            l10min = defaultdict(float)
            for gm in window:
                for aid, m in gm.items(): l10min[aid] += m
            team_l10_total = sum(l10min.values())
            act_minshare = (sum(l10min.get(aid, 0.0) for aid, m in act)/team_l10_total) if team_l10_total > 0 else 1.0
            wsum = 0.0; wr = 0.0
            for aid, m in act:
                rt, pmin = p_rating(aid)
                wsum += pmin; wr += pmin*rt
            avail_strength = (wr/wsum) if wsum > 0 else 0.0
            miss = 0.0
            for aid, share, rt in last_lineup.get(tm, []):
                if aid not in {x[0] for x in act} and rt > 0:
                    miss += share*rt
            row.update({f"{side}_availshare": act_minshare, f"{side}_availstr": avail_strength,
                        f"{side}_missing": miss})
        feats.append(row)

        # ---- post-game state updates ----
        hb, ab = tbox.get((gid, h)), tbox.get((gid, a))
        if hb is None or ab is None: continue
        lgm = lg_or.val(100.0)
        for tm, mine, opp in ((h, hb, ab), (a, ab, hb)):
            poss = (mine.poss + opp.poss)/2
            if not poss or poss <= 0: continue
            ortg = 100*mine.team_score/poss; drtg = 100*opp.team_score/poss
            opp_slow_d = slow[opp.team_abbreviation]["drtg"].val(lgm)
            opp_slow_o = slow[opp.team_abbreviation]["ortg"].val(lgm)
            adj_o = ortg - (opp_slow_d - lgm); adj_d = drtg - (opp_slow_o - lgm)
            fast[tm]["net"].add(adj_o - adj_d); slow[tm]["net"].add(adj_o - adj_d)
            slow[tm]["ortg"].add(adj_o); slow[tm]["drtg"].add(adj_d)
            slow[tm]["efg"].add(mine.efg); slow[tm]["efga"].add(opp.efg)
            slow[tm]["tovr"].add(mine.tovr)
            orb = mine.offensive_rebounds/(mine.offensive_rebounds + opp.defensive_rebounds)
            slow[tm]["orb"].add(orb)
            for k, v in (("ortg", ortg), ("drtg", drtg), ("net", ortg-drtg), ("efg", mine.efg),
                         ("efga", opp.efg), ("tovr", mine.tovr), ("orb", orb)):
                std[tm][k].append(v); last10[tm][k].append(v)
            lg_or.add(ortg)
            pv = prev.get(tm, {"dates": deque(maxlen=6)})
            pv["dates"] = pv.get("dates", deque(maxlen=6)); pv["dates"].append(date)
            pv["date"] = date; pv["city"] = city
            prev[tm] = pv
        eh = 1.0/(1+10**(-((elo[h]+(0 if r.neutral else ELO_HA))-elo[a])/400))
        elo[h] += ELO_K*(r.home_win - eh); elo[a] -= ELO_K*(r.home_win - eh)
        for tm in (h, a):
            plist = pbox.get(gid, {}).get(tm, [])
            tmmin = sum(m for _, m, _, _ in plist) or 1.0
            lineup = []
            for aid, m, gs, pm in plist:
                st = pstate.setdefault(aid, {"min": 0.0, "gmsc": 0.0, "mins_dec": Decay(HL_F)})
                if m and m > 0:
                    rt, _ = p_rating(aid)
                    lineup.append((aid, m/tmmin, max(rt, 0.0)))
                    st["min"] = st["min"]*0.97 + m; st["gmsc"] = st["gmsc"]*0.97 + gs
                    st["mins_dec"].add(m)

            lineup.sort(key=lambda x: -x[1])
            last_lineup[tm] = lineup[:8]
            team_l10.setdefault(tm, deque(maxlen=10)).append({aid: (m or 0.0) for aid, m, _, _ in plist})
    return pd.DataFrame(feats)

def metrics(y, p, label, n=None):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, int)
    hit = float(np.mean((p > 0.5) == (y == 1)))
    return {"model": label, "n": len(y) if n is None else n,
            "log_loss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
            "hit_rate": hit}

OLD_FEATS_BASE = ["std_ortg_d","std_drtg_d","l10_ortg_d","l10_drtg_d","std_efg_d","std_efga_d",
                  "std_tovr_d","std_orb_d","rest_d","h_b2b","a_b2b","h_travel","a_travel",
                  "h_availshare","a_availshare"]
NEW_FEATS = ["elo_d","net_f_d","net_s_d","efg_d","efga_d","tovr_d","orb_d","prior_net_d",
             "availstr_d","h_missing","a_missing","availshare_d","rest_d","h_b2b","a_b2b",
             "h_3in4","a_3in4","h_travel","a_travel","neutral","h_gp","a_gp"]

def add_diffs(df):
    df = df.copy()
    for base, out in [("elo","elo_d"),("net_f","net_f_d"),("net_s","net_s_d"),("efg_s","efg_d"),
                      ("efga_s","efga_d"),("tovr_s","tovr_d"),("orb_s","orb_d"),
                      ("prior_net","prior_net_d"),("availstr","availstr_d"),
                      ("availshare","availshare_d"),("rest","rest_d"),
                      ("std_ortg","std_ortg_d"),("std_drtg","std_drtg_d"),
                      ("l10_ortg","l10_ortg_d"),("l10_drtg","l10_drtg_d"),
                      ("std_efg","std_efg_d"),("std_efga","std_efga_d"),
                      ("std_tovr","std_tovr_d"),("std_orb","std_orb_d")]:
        df[out] = df[f"h_{base}"] - df[f"a_{base}"]
    return df

def fit_eval(df, feats, train_mask, test_mask, C=0.2):
    Xtr, ytr = df.loc[train_mask, feats].values, df.loc[train_mask, "home_win"].values
    Xte, yte = df.loc[test_mask, feats].values, df.loc[test_mask, "home_win"].values
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1], yte, (m, sc)

def elo_prob(df, mask, ha):
    d = df.loc[mask]
    e = 1.0/(1+10**(-((d.h_elo + np.where(d.neutral == 1, 0, ha)) - d.a_elo)/400))
    return e.values, d.home_win.values

def run():
    g = load_games()
    print("games loaded:", len(g), "| by season:", g.groupby("season").size().to_dict())
    tbox = load_team_box(); pbox = load_player_box()
    cfg = {"hl_fast": 5, "hl_slow": 15, "carry": 0.6, "elo_k": 20, "elo_ha": 60, "tau": 150}
    # light tuning on 2024 only
    best = None
    for hl_s in (10, 15):
        for carry in (0.5, 0.7):
            for k in (16, 24):
                c = dict(cfg, hl_slow=hl_s, carry=carry, elo_k=k)
                df = add_diffs(build_features(g, tbox, pbox, c))
                tr = (df.season >= 2019) & (df.season <= 2023)
                va = df.season == 2024
                for C in (0.05, 0.2, 1.0):
                    p, y, _ = fit_eval(df, NEW_FEATS, tr, va, C)
                    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                    if best is None or ll < best[0]:
                        best = (ll, dict(c), C, df)
    ll24, cfg, C, df = best
    print(f"tuned on 2024: log_loss={ll24:.4f} cfg={{hl_slow:{cfg['hl_slow']}, carry:{cfg['carry']}, elo_k:{cfg['elo_k']}}} C={C}")

    results = []
    for holdname, test_season, train_hi in (("2025 holdout", 2025, 2024), ("2026 YTD holdout", 2026, 2025)):
        tr = (df.season >= 2019) & (df.season <= train_hi)
        te = df.season == test_season
        y = df.loc[te, "home_win"].values
        base = df.loc[tr, "home_win"].mean()
        res = [metrics(y, np.full(y.shape, 0.5), "coin flip 0.5"),
               metrics(y, np.full(y.shape, base), f"base rate ({base:.3f} home)")]
        pe, ye = elo_prob(df, te, cfg["elo_ha"])
        res.append(metrics(ye, pe, "Elo (floor)"))
        fav = (pe > 0.5).astype(int)
        res.append({"model": "always Elo favorite", "n": len(ye), "log_loss": None, "brier": None,
                    "hit_rate": float(np.mean(fav == ye))})
        res.append({"model": "always Elo dog", "n": len(ye), "log_loss": None, "brier": None,
                    "hit_rate": float(np.mean((1-fav) == ye))})
        res.append({"model": "always home team", "n": len(ye), "log_loss": None, "brier": None,
                    "hit_rate": float(np.mean(ye == 1))})
        po, yo, _ = fit_eval(df, OLD_FEATS_BASE, tr, te, C=1.0)
        res.append(metrics(yo, po, "old v5.0 trait model (reimpl)"))
        pn, yn, fitted = fit_eval(df, NEW_FEATS, tr, te, C)
        res.append(metrics(yn, pn, "REBUILD v1"))
        R = pd.DataFrame(res); R.insert(0, "holdout", holdname)
        results.append(R)
        if test_season == 2026:
            out = df.loc[te, ["date", "home", "away", "home_win"]].copy()
            out["p_home_rebuild"] = pn
            out.to_csv("/home/claude/wnba_2026_ytd_predictions.csv", index=False)
            m, sc = fitted
            coefs = pd.Series(m.coef_[0], index=NEW_FEATS).sort_values(key=abs, ascending=False)
            print("\nrebuild coefficients (standardized):"); print(coefs.round(3).to_string())
    allr = pd.concat(results, ignore_index=True)
    allr.to_csv("/home/claude/wnba_rebuild_comparison.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + allr.round(4).to_string(index=False))

if __name__ == "__main__":
    run()
