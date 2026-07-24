# MBM WNBA daily engine (rebuild v1.1)

Standalone predictive model. No market inputs. Frozen spec: trained on 2019-2025 only;
the rest of 2026 is the forward test. Early-season shrink (a0=0.4) fitted on pre-2026
out-of-sample games. Grades the full slate daily on log loss / Brier vs Elo and base-rate floors.

## Setup (one time, ~5 min)
1. Create a private GitHub repo, upload this folder's contents (keep the .github folder).
2. Actions tab: enable workflows. Done. It runs twice daily (morning board + pre-tip
   refresh with the latest ESPN injury report) and commits results to `logs/`.
3. Optional Telegram ping: repo Settings > Secrets > add `TG_TOKEN` (BotFather token)
   and `TG_CHAT` (your chat id).

## Manual run
    pip install -r requirements.txt
    python daily_wnba.py                # today
    python daily_wnba.py --date 20260724

## Availability logic
Expected actives = players who appeared in the team's last 3 games, minus anyone the ESPN
injury feed lists as Out, minus anyone in `overrides.json` (`{"out": {"LV": ["athlete_id"]}}`).
Day-to-day / questionable players count as in; use overrides for known scratches.

## Files
- `daily_wnba.py` runner; `features.py` feature/model pipeline; `model_spec.json` frozen params
- `logs/predictions.csv` every prediction (2026 YTD backfilled); `logs/graded.csv` scored games

## Season rollover
New season: nothing to do, years auto-extend. Revisit train window + shrink after 2026 ends.
