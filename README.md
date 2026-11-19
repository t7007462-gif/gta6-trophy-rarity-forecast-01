# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/t7007462-gif/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/t7007462-gif/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, how many players earn it in week one — five quantiles of each trophy's earn rate on Exophase at the case's target, from the history of reads of the game's trophy page.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only, no network) reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## How it works

A trophy's earn rate is a ratio, `r = E / N`, of two growing quantities: the earned count `E` (players progress and earn trophies) and the tracked population `N` (new players buy the game and get tracked, mostly before they have earned anything, which dilutes every rate). Extrapolating the rate alone confuses the two motions, so the program models them separately and recombines them.

It is an ensemble of four forecasting families, weighted per trophy by an out-of-sample backtest on the case's own history (fit on the earlier reads, scored on the last ~40% of the span):

- **persistence** — the rate stays at its last read.
- **logit-linear** — `logit(r)` linear in `ln(1 + hours since release)`, fitted on the last 24 hours of reads only (the house curve baseline's window; fitting the whole history anchors the slope on the flat launch-night segment and over-extrapolates).
- **decomp-weibull** — `N(t)` and `E(t)` each a saturating Weibull CDF, fitted by grid search, recombined as a rate.
- **cohort** — `N(t)` a Weibull CDF gives the join rate; each trophy's earners follow a time-to-earn curve over a player's age (with a delay before the trophy is earnable at all), so `E(t)` is the convolution of joins with earning. This is the only family that knows new joiners start with no trophies, so it dilutes correctly when `N` jumps, and the only one whose rates saturate instead of diverging. It carries a modest prior boost in the blend.

Power-law and quadratic extrapolations are deliberately not included: with no saturation they diverge to ~100% over a 4.5-day horizon whenever earned counts outgrow the tracked population in-sample.

The spread is honest: a random-walk term growing with the horizon, the backtest error scaled to the forecast horizon, and the disagreement between families. A page that has never refreshed in the history is treated as stale, not stable — the first refresh re-rates everything, so the band widens and the median shifts down (a refresh dilutes rates). A trophy every tracked player has earned (100% in the last read) keeps a median near 100% with the band reaching down for fresh joiners.

Validated on the kit's practice cases and on synthetic final-like cases (two different truth-generating shapes): the blend beats the house baseline and the house curve baseline on every variant.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7pcj4z9y&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
