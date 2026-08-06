#!/usr/bin/env python3
"""
Why the model made each WNBA pick, in plain English. One-off, prints and exits.

This is not a second model and it does not guess. The board is a logistic regression on
22 standardised features, so each feature's contribution to the pick is exactly
`coefficient x standardised value`, and those contributions sum to the log-odds the model
published. What you read below is arithmetic on the model that ran, not a story told
afterwards to fit the answer.

It reuses daily_wnba end to end: same data, same spec, same fit, same shrink. It does not
write to logs/, does not touch push_cache.json, and never calls Discord. Run it as often as
you like.

    python explain_wnba.py                    today
    python explain_wnba.py --date 20260805    a specific day
    python explain_wnba.py --words 60         shorter blurbs
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import daily_wnba as D                                   # noqa: E402
import features as F                                     # noqa: E402
import board_format as BF                                # noqa: E402

# Plain-language name for each feature, and which side a positive value favours. The board
# is always oriented home-minus-away, so a positive contribution pushes toward the home
# team and a negative one toward the away team.
LABEL = {
    "elo_d":       "the power rating",
    "net_f_d":     "recent scoring margin",
    "net_s_d":     "season-long scoring margin",
    "efg_d":       "shooting efficiency",
    "efga_d":      "defensive shooting allowed",
    "tovr_d":      "turnover rate",
    "orb_d":       "offensive rebounding",
    "prior_net_d": "last season's margin",
    "availstr_d":  "the quality of who is available",
    "h_missing":   "who the home side is missing",
    "a_missing":   "who the away side is missing",
    "availshare_d": "the share of usual minutes available",
    "rest_d":      "rest",
    "h_b2b":       "the home side on a back-to-back",
    "a_b2b":       "the away side on a back-to-back",
    "h_3in4":      "the home side playing three in four nights",
    "a_3in4":      "the away side playing three in four nights",
    "h_travel":    "home travel",
    "a_travel":    "away travel",
    "neutral":     "the neutral floor",
    "h_gp":        "how far into the season the home side is",
    "a_gp":        "how far into the season the away side is",
}


def blurb(home, away, p, contrib, words):
    """One paragraph naming the biggest drivers, in the order the model weighted them."""
    pick, opp = (home, away) if p >= .5 else (away, home)
    conf = max(p, 1 - p)
    toward = 1.0 if p >= .5 else -1.0          # sign that helps the pick

    ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    helps = [(k, v) for k, v in ranked if v * toward > 0][:3]
    hurts = [(k, v) for k, v in ranked if v * toward < 0][:1]
    total = sum(abs(v) for v in contrib.values()) or 1.0

    # "of the case" once, not once per driver.
    bits = [f"{LABEL.get(k, k)} at {abs(v) / total:.0%}" for k, v in helps]
    spread = max(abs(v) for v in contrib.values())
    if not bits or spread < 0.05:
        return (f"{pick} over {opp} at {conf:.0%}, fair price {BF.price(conf)}. Nothing "
                f"separates these teams. Every input is close to neutral, so this is a "
                f"coin flip the model is barely leaning on, and the price says so.")

    lead = (f"{pick} over {opp} at {conf:.0%}, fair price {BF.price(conf)}. It leans "
            f"mostly on {bits[0]} of the weight behind the pick")
    if len(bits) > 1:
        lead += ", then " + " and ".join(bits[1:])
    lead += "."
    if hurts:
        k, v = hurts[0]
        lead += (f" Pushing back, {LABEL.get(k, k)} favours {opp} at "
                 f"{abs(v) / total:.0%}, which is why it is not stronger.")
    if conf < 0.58:
        lead += " Thin enough that a small price move makes it a pass."
    return " ".join(lead.split()[:words])


def main(datestr, words):
    slate = D.fetch_slate(datestr)
    if not slate:
        print(f"no WNBA games on {datestr}")
        return 0

    # Rebuild exactly what the board rebuilds, then reach into the fitted model rather
    # than re-deriving anything.
    D.download_data()
    D._avail_years()
    g = F.load_games()
    tbox = F.load_team_box()
    pbox = F.load_player_box()
    pbox = {gid: {tm: [(str(a), m, gs, pm) for a, m, gs, pm in v] for tm, v in d.items()}
            for gid, d in pbox.items()}
    ros, outs = D.rosters_and_outs(datestr)
    gdate = pd.Timestamp(pd.to_datetime(datestr, format="%Y%m%d").date())
    mins = D.R.recent_minutes(pbox, g, gdate)
    pseudo = []
    for s in slate:
        pbox[s["game_id"]] = {
            tm: (D.expected_actives(mins, tm, ros, outs) if ros["by_ab"].get(tm)
                 else D.expected_actives_from_box(pbox, g, tm, outs))
            for tm in (s["home"], s["away"])}
        pseudo.append({"game_id": s["game_id"], "game_date": gdate, "game_date_time": gdate,
                       "season": gdate.year, "season_type": 2,
                       "home_abbreviation": s["home"], "away_abbreviation": s["away"],
                       "home_score": np.nan, "away_score": np.nan,
                       "neutral_site": s["neutral"], "home_win": np.nan,
                       "neutral": s["neutral"]})
    g_all = pd.concat([g, pd.DataFrame(pseudo)], ignore_index=True)
    g_all = g_all.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    df = F.add_diffs(F.build_features(g_all, tbox, pbox, D.CFG))
    tr = (df.season >= D.TRAIN_LO) & (df.season <= D.TRAIN_HI) & df.home_win.notna()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(df.loc[tr, F.NEW_FEATS].values)
    m = LogisticRegression(C=D.C, max_iter=2000).fit(
        sc.transform(df.loc[tr, F.NEW_FEATS].values),
        df.loc[tr, "home_win"].astype(int).values)

    rows = df[df.game_id.isin([s["game_id"] for s in slate])].copy()
    if rows.empty:
        print("games found on the slate but no features built for them")
        return 0
    X = sc.transform(rows[F.NEW_FEATS].values)
    coef = m.coef_[0]
    raw = m.predict_proba(X)[:, 1]
    mg = rows[["h_gp", "a_gp"]].min(axis=1).values
    alpha = D.A0 + (1 - D.A0) * np.minimum(mg, 10) / 10.0
    lo = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    p = 1 / (1 + np.exp(-alpha * lo))

    print(f"\nWNBA, {datestr}. {len(rows)} game(s). "
          f"Contributions are coefficient x standardised value and sum to the pick.\n")
    for i, (_, r) in enumerate(rows.iterrows()):
        contrib = {f: float(coef[j] * X[i, j]) for j, f in enumerate(F.NEW_FEATS)}
        print(f"{'=' * 74}")
        print(f"{r.away_abbreviation} at {r.home_abbreviation}")
        print(blurb(r.home_abbreviation, r.away_abbreviation, float(p[i]), contrib, words))
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
        print("\n   biggest inputs, signed toward the home side:")
        for k, v in top:
            print(f"     {LABEL.get(k, k):<44}{v:+.3f}")
        print(f"   {'model p(home)':<46}{p[i]:.3f}"
              + ("   [below the price floor, would not board]"
                 if BF.min_odds(max(p[i], 1 - p[i])) < BF.PRICE_FLOOR else ""))
    print(f"{'=' * 74}")
    return 0


if __name__ == "__main__":
    import datetime as dt
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--words", type=int, default=100)
    sys.exit(main(ap.parse_args().date, ap.parse_args().words))
