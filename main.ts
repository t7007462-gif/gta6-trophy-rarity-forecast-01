// gta6-trophy-rarity-forecast-01 — week-one GTA VI trophy-rarity forecast.
//
// For every trophy of Grand Theft Auto VI on PlayStation 5, forecast five
// quantiles of its earn rate on Exophase at the case's target, from the
// case's history of reads of the game's trophy page.
//
// What the data is: each read gives the players Exophase tracks (N) and every
// trophy's earn rate r_i (percent of tracked players who earned it). The rate
// is a ratio, r_i = E_i / N, of two growing quantities: the earned count E_i
// (players progress through the game and earn trophies) and the tracked
// population N (new players buy the game and get tracked, mostly before they
// have earned anything, which dilutes every rate). A forecast that only
// extrapolates the rate itself confuses the two motions; this program models
// them separately and recombines them.
//
// Model: an ensemble of four forecasting families, fitted per case and
// weighted per trophy by an out-of-sample backtest on the case's own history
// (fit on the earlier reads, scored on the last ~40% of the span), so the
// weighting adapts to whatever the history actually does:
//
//   0 persistence      r stays at its last read.
//   1 logit-linear     logit(r_i) linear in ln(1 + hours since release),
//                      fitted on the last 24 hours of reads only, unweighted.
//                      (Fitting the whole history anchors the slope on the
//                      flat launch-night segment and over-extrapolates; the
//                      last 24h is the house curve baseline's window.)
//   2 decomp-weibull   N(t) and E_i(t) each a saturating Weibull CDF
//                      A * (1 - exp(-(t / tau)^alpha)), fitted by grid search.
//   3 cohort           N(t) a Weibull CDF gives the join rate n(t); each
//                      trophy's earners follow a time-to-earn curve
//                      F_i(a) = c_i * (1 - exp(-max(0, a - d_i) / tau_i))
//                      over a player's age a (d_i: the delay before the
//                      trophy is earnable at all), so
//                      E_i(t) = int_0^t n(s) F_i(t - s) ds. This is the only
//                      family that knows new joiners start with no trophies,
//                      so it dilutes correctly when N jumps, and it is the
//                      only one whose rates saturate instead of diverging.
//
// Deliberately NOT included: a power law for E_i, or a quadratic in log time.
// Either has no saturation and diverges to ~100% over a 4.5-day horizon
// whenever earned counts outgrow the tracked population in-sample, which is
// the common case for a rising trophy. On a synthetic final-like case those
// two families scored far worse than persistence alone.
//
// The median is the backtest-weighted blend of the families' logit forecasts.
// The spread is honest: a random-walk term that grows with the horizon, the
// backtest error scaled up to the forecast horizon, the disagreement between
// families, and an extra term when the page has never refreshed in the
// history (a stale page: the first refresh re-rates everything, as the
// launch-night refresh did, so a flat history is not evidence of a flat
// future). In that stale-page case the median is also shifted down, because a
// refresh dilutes rates: new tracked players have earned almost nothing yet.
//
// Everything is deterministic: no clock, no randomness, fixed iteration
// order. Standard library only; runs offline in the grader's sandbox.
//
// Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Spread of a rate's move, in logit units per square-root day (house baseline's scale). */
const SIGMA_RW = 0.5;
/** Release day begins 2026-11-19T00:00:00Z; time is measured in hours from here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Array<{
    slot: string;
    at: string;
    players_tracked: number;
    trophies_earned: number;
    rates: Array<number | null>;
  }>;
}

/** One read: t in hours since release, N players tracked, rates per trophy. */
interface Read {
  t: number;
  n: number;
  rates: Array<number | null>;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampRate = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const logistic = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -12, 12)));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** u = ln(1 + t): the time scale the trends are fitted on. */
const uOf = (t: number): number => Math.log(1 + Math.max(t, -0.9));

/** Solve a small dense linear system by Gaussian elimination with partial pivoting. */
function solveLinear(a: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    if (piv !== col) {
      const tmp = m[piv];
      m[piv] = m[col];
      m[col] = tmp;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let c = col; c <= n; c++) m[r][c] -= f * m[col][c];
    }
  }
  return m.map((row, i) => row[n] / m[i][i]);
}

/** Weighted least squares for y ~ X beta. Returns beta or null if degenerate. */
function wls(xs: number[][], ys: number[], ws: number[]): number[] | null {
  const k = xs[0].length;
  const ata: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const atb: number[] = new Array<number>(k).fill(0);
  for (let j = 0; j < xs.length; j++) {
    const w = ws[j];
    for (let a = 0; a < k; a++) {
      atb[a] += w * xs[j][a] * ys[j];
      for (let b = 0; b < k; b++) ata[a][b] += w * xs[j][a] * xs[j][b];
    }
  }
  return solveLinear(ata, atb);
}

/** A fitted forecasting family: predicts logit(rate) of trophy i at time t (hours since release). */
interface Family {
  name: string;
  predict: (i: number, t: number) => number;
}

type Fitter = (reads: Read[], trophies: Trophy[], tStar: number) => Family;

/** Last read's rates (persistence). */
const fitPersistence: Fitter = (reads) => {
  const last = reads[reads.length - 1];
  return {
    name: 'persistence',
    predict: (i) => logit(last.rates[i] ?? 1),
  };
};

/**
 * Linear fit of logit(rate) on u = ln(1+t), per trophy, on the last 24 hours
 * of the reads only, unweighted. The whole history must not be used: the
 * flat launch-night segment anchors the slope and the fit over-extrapolates
 * the steep middle of the S-curve. This is the house curve baseline's window.
 */
function fitLogitLinear(reads: Read[]): Family {
  const tLast = reads[reads.length - 1].t;
  const window = reads.filter((r) => r.t >= tLast - 24);
  const use = window.length >= 3 ? window : reads;
  const perTrophy: Array<{ beta: number[] | null; last: number }> = [];
  const nT = reads[0].rates.length;
  for (let i = 0; i < nT; i++) {
    const xs: number[][] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    for (const r of use) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      xs.push([1, uOf(r.t)]);
      ys.push(logit(rate));
      ws.push(1);
    }
    const beta = xs.length >= 3 ? wls(xs, ys, ws) : null;
    const lastRate = [...reads].reverse().find((r) => r.rates[i] !== null && r.rates[i] !== undefined)?.rates[i] ?? 1;
    perTrophy.push({ beta, last: logit(lastRate) });
  }
  return {
    name: 'logit-linear',
    predict: (i, t) => {
      const f = perTrophy[i];
      if (!f.beta) return f.last;
      return clamp(f.beta[0] + f.beta[1] * uOf(t), -9.21, 9.21);
    },
  };
}

/** Weibull CDF saturating fit for a positive series y(t): y = A (1 - exp(-(t/tau)^alpha)). */
function fitWeibullSeries(ts: number[], ys: number[]): { a: number; tau: number; alpha: number } | null {
  const pts = ts.map((t, j) => ({ t: Math.max(t, 0.02), y: ys[j] })).filter((p) => p.y > 0);
  if (pts.length < 4) return null;
  const alphas = [0.4, 0.55, 0.7, 0.85, 1.0, 1.25, 1.5];
  const taus: number[] = [];
  for (let k = 0; k < 25; k++) taus.push(Math.exp(Math.log(6) + (k / 24) * (Math.log(3000) - Math.log(6))));
  let best: { a: number; tau: number; alpha: number; sse: number } | null = null;
  for (const alpha of alphas) {
    for (const tau of taus) {
      let sumLogA = 0;
      let cnt = 0;
      const shape = pts.map((p) => 1 - Math.exp(-Math.pow(p.t / tau, alpha)));
      for (let j = 0; j < pts.length; j++) {
        if (shape[j] < 1e-6) continue;
        sumLogA += Math.log(pts[j].y) - Math.log(shape[j]);
        cnt++;
      }
      if (cnt < 4) continue;
      const logA = sumLogA / cnt;
      const a = Math.exp(logA);
      let sse = 0;
      for (let j = 0; j < pts.length; j++) {
        const pred = a * shape[j];
        const d = Math.log(Math.max(pts[j].y, 1e-6)) - Math.log(Math.max(pred, 1e-6));
        sse += d * d;
      }
      if (!best || sse < best.sse) best = { a, tau, alpha, sse };
    }
  }
  return best ? { a: best.a, tau: best.tau, alpha: best.alpha } : null;
}

const weibullAt = (f: { a: number; tau: number; alpha: number }, t: number): number =>
  f.a * (1 - Math.exp(-Math.pow(Math.max(t, 0.02) / f.tau, f.alpha)));

/** Decomposition family: forecast N and E_i each as a saturating Weibull CDF, recombine as a rate. */
function makeDecompFamily(reads: Read[]): Family {
  const nT = reads[0].rates.length;
  const ts = reads.map((r) => r.t);
  const ns = reads.map((r) => r.n);
  const nWeib = fitWeibullSeries(ts, ns);
  const nLast = ns[ns.length - 1];
  const perTrophy: Array<{ eWeib: { a: number; tau: number; alpha: number } | null; last: number }> = [];
  for (let i = 0; i < nT; i++) {
    const es: number[] = [];
    const ets: number[] = [];
    for (const r of reads) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      ets.push(r.t);
      es.push(Math.max((rate / 100) * r.n, 0.05));
    }
    const lastRate = [...reads].reverse().find((r) => r.rates[i] !== null && r.rates[i] !== undefined)?.rates[i] ?? 1;
    perTrophy.push({
      eWeib: fitWeibullSeries(ets, es),
      last: logit(lastRate),
    });
  }
  const predictN = (t: number): number => (nWeib ? Math.max(weibullAt(nWeib, t), 1) : nLast);
  return {
    name: 'decomp-weibull',
    predict: (i, t) => {
      const f = perTrophy[i];
      if (!f.eWeib) return f.last;
      const e = Math.max(weibullAt(f.eWeib, t), 0.05);
      const n = predictN(t);
      return logit(clampRate((100 * e) / Math.max(n, 1)));
    },
  };
}

/**
 * Cohort family: N(t) is a Weibull CDF, so the join rate n(t) is known. Each
 * trophy's earners follow F_i(a) = c_i (1 - exp(-max(0, a - d_i) / tau_i))
 * over a player's age a (d_i: the delay before the trophy is earnable at all,
 * e.g. a late story chapter or a grind that starts after some progression),
 * so E_i(t) = int_0^t n(s) F_i(t - s) ds. For a grid of (tau, d) pairs the
 * kernel H(t) = int_0^t n(s) (1 - exp(-max(0, t - s - d) / tau)) ds is
 * computed once on a time grid; per trophy c_i is fitted by least squares and
 * the best (tau, d) chosen by SSE. E_i(t*) = c_i H(t*), r = 100 E / N.
 */
function fitCohort(reads: Read[], tStar: number): Family {
  const nT = reads[0].rates.length;
  const ts = reads.map((r) => r.t);
  const ns = reads.map((r) => r.n);
  const nWeib = fitWeibullSeries(ts, ns);
  const nLast = ns[ns.length - 1];
  const predictN = (t: number): number => (nWeib ? Math.max(weibullAt(nWeib, t), 1) : nLast);

  // Join rate n(t) = dN/dt for the fitted Weibull (0 if no fit: fall back to flat).
  const joinRate = (t: number): number => {
    if (!nWeib) return 0;
    const tt = Math.max(t, 0.02);
    const { a, tau, alpha } = nWeib;
    return (a * alpha) / tau * Math.pow(tt / tau, alpha - 1) * Math.exp(-Math.pow(tt / tau, alpha));
  };

  // Time grid 0..tStar for the kernels; H(t) by trapezoid convolution.
  const G = 241;
  const grid: number[] = [];
  for (let k = 0; k < G; k++) grid.push((k / (G - 1)) * Math.max(tStar, 1));
  const taus = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
  const delays = [0, 6, 12, 24, 48];
  // One kernel per (tau, delay) pair: kernels[ki][gridIdx].
  const kernels: number[][] = [];
  for (const tau of taus) {
    for (const d of delays) {
      const h = new Array<number>(G).fill(0);
      for (let k = 1; k < G; k++) {
        const t = grid[k];
        const tEff = t - d;
        if (tEff <= 0) continue;
        // Trapezoid integral over s in [0, t-d] of n(s) (1 - exp(-(t - d - s)/tau)).
        let sum = 0;
        const m = Math.min(k, 200);
        for (let j = 0; j <= m; j++) {
          const s = (j / m) * tEff;
          const wgt = j === 0 || j === m ? 0.5 : 1;
          sum += wgt * joinRate(s) * (1 - Math.exp(-(tEff - s) / tau));
        }
        h[k] = sum * (tEff / m);
      }
      kernels.push(h);
    }
  }
  const hAt = (ki: number, t: number): number => {
    if (t <= 0) return 0;
    const h = kernels[ki];
    if (t >= grid[G - 1]) return h[G - 1];
    const pos = (t / grid[G - 1]) * (G - 1);
    const k = Math.floor(pos);
    const frac = pos - k;
    return h[k] * (1 - frac) + h[Math.min(k + 1, G - 1)] * frac;
  };

  const perTrophy: Array<{ c: number; kernelIdx: number; last: number }> = [];
  for (let i = 0; i < nT; i++) {
    const lastRate = [...reads].reverse().find((r) => r.rates[i] !== null && r.rates[i] !== undefined)?.rates[i] ?? 1;
    if (!nWeib) {
      perTrophy.push({ c: 0, kernelIdx: 0, last: logit(lastRate) });
      continue;
    }
    let best: { c: number; kernelIdx: number; sse: number } | null = null;
    for (let ki = 0; ki < kernels.length; ki++) {
      let num = 0;
      let den = 0;
      const obs: Array<{ e: number; h: number }> = [];
      for (const r of reads) {
        const rate = r.rates[i];
        if (rate === null || rate === undefined) continue;
        const e = (rate / 100) * r.n;
        const h = hAt(ki, r.t);
        obs.push({ e, h });
        num += e * h;
        den += h * h;
      }
      if (den <= 0 || obs.length < 2) continue;
      const c = clamp(num / den, 0, 1);
      let sse = 0;
      for (const o of obs) {
        const d = o.e - c * o.h;
        sse += d * d;
      }
      if (!best || sse < best.sse) best = { c, kernelIdx: ki, sse };
    }
    perTrophy.push(best ? { c: best.c, kernelIdx: best.kernelIdx, last: logit(lastRate) } : { c: 0, kernelIdx: 0, last: logit(lastRate) });
  }
  return {
    name: 'cohort',
    predict: (i, t) => {
      const f = perTrophy[i];
      if (!nWeib || f.c <= 0) return f.last;
      const e = f.c * hAt(f.kernelIdx, t);
      const n = predictN(t);
      return logit(clampRate((100 * e) / Math.max(n, 1)));
    },
  };
}

const FITTERS: Fitter[] = [
  fitPersistence,
  fitLogitLinear,
  makeDecompFamily,
  (reads, _t, tStar) => fitCohort(reads, tStar),
];

/** Forecast one case. Returns rows {id, q:[q05,q25,q50,q75,q95]}. */
function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const tStar = (Date.parse(c.target) - RELEASE_MS) / 3600000;
  const horizonDays = Math.max(c.horizon_minutes, 1) / 1440;

  // Dedupe consecutive identical reads: a stale page repeats itself, and
  // counting each repeat would overweight the flat stretch in every fit.
  const reads: Read[] = [];
  for (const h of c.history) {
    if (!h || typeof h.players_tracked !== 'number' || h.players_tracked <= 0 || !Array.isArray(h.rates)) continue;
    const t = (Date.parse(h.at) - RELEASE_MS) / 3600000;
    const prev = reads[reads.length - 1];
    if (prev && prev.n === h.players_tracked && prev.rates.length === h.rates.length && prev.rates.every((v, j) => v === h.rates[j])) {
      continue;
    }
    reads.push({ t, n: h.players_tracked, rates: h.rates });
  }
  const nT = c.trophies.length;
  const lastRates: Array<number | null> = reads.length
    ? reads[reads.length - 1].rates
    : c.trophies.map(() => null);

  // No usable history at all: flat 1% median, very wide spread.
  if (reads.length === 0) {
    return c.trophies.map((t) => ({ id: t.id, q: [0.01, 0.1, 1, 10, 50] }));
  }

  const nChanged = new Set(reads.map((r) => r.n)).size > 1;
  const stalePage = !nChanged;

  // ---- Backtest: fit on reads up to tCut, score on the reads after it. ----
  const tFirst = reads[0].t;
  const tLast = reads[reads.length - 1].t;
  const span = tLast - tFirst;
  // Holdout is the last ~40% of the span: a short holdout cannot tell a
  // family that extrapolates well from one that explodes a day out.
  const btHorizonDays = Math.max(0.25, 0.4 * span) / 24;
  const tCut = tLast - Math.max(6, 0.4 * span);
  const early = reads.filter((r) => r.t <= tCut);
  const holdout = reads.filter((r) => r.t > tCut);
  // weights[f][i], and per-trophy backtest RMSE, from the holdout.
  let weights: number[][] | null = null;
  let btRmse: number[] | null = null;
  if (early.length >= 4 && holdout.length >= 2 && span >= 4) {
    const fams = FITTERS.map((f) => f(early, c.trophies, tStar));
    const F = fams.length;
    weights = [];
    btRmse = [];
    for (let i = 0; i < nT; i++) {
      const mses: number[] = [];
      for (let f = 0; f < F; f++) {
        let sse = 0;
        let cnt = 0;
        for (const r of holdout) {
          const rate = r.rates[i];
          if (rate === null || rate === undefined) continue;
          const d = fams[f].predict(i, r.t) - logit(rate);
          sse += d * d;
          cnt++;
        }
        mses.push(cnt > 0 ? sse / cnt : Number.POSITIVE_INFINITY);
      }
      const w = mses.map((m) => (Number.isFinite(m) ? 1 / (m + 0.04) : 0));
      const wMax = Math.max(...w);
      // Cap the weight ratio so no single family dominates on a short holdout.
      const wCapped = w.map((x) => (wMax > 0 ? Math.max(x, wMax / 10) : 0));
      const wSum = wCapped.reduce((a, b) => a + b, 0);
      weights.push(wSum > 0 ? wCapped.map((x) => x / wSum) : wCapped.map(() => 1 / F));
      let rmse = 0;
      for (let f = 0; f < F; f++) rmse += weights[i][f] * (Number.isFinite(mses[f]) ? mses[f] : 4);
      btRmse.push(Math.sqrt(Math.max(rmse, 0)));
    }
  }
  if (!weights) {
    // No backtest possible (too little history, or a perfectly flat page):
    // roughly equal weights; the spread terms below carry the uncertainty.
    weights = c.trophies.map(() => [0.15, 0.25, 0.15, 0.45]);
    btRmse = c.trophies.map(() => 0);
  }
  // Prior boost for the cohort family (index 3): it is the only family with
  // the dilution mechanism and saturating rates. A modest 1.5x, not a veto.
  const COHORT_BOOST = 2.5;
  weights = weights.map((w) => {
    const boosted = w.map((x, j) => (j === 3 ? x * COHORT_BOOST : x));
    const ssum = boosted.reduce((a, b) => a + b, 0);
    return boosted.map((x) => x / ssum);
  });
  // Debug/validation hook only: MAKEFUN_ONLY_FAMILY=n forces the blend to
  // use family n alone. Never set in the grader.
  const onlyFamily = process.env.MAKEFUN_ONLY_FAMILY;
  if (onlyFamily !== undefined && onlyFamily !== '') {
    const f = Number(onlyFamily);
    weights = weights.map((w) => w.map((_, j) => (j === f ? 1 : 0)));
  }

  // ---- Full-history fits, blended per trophy. ----
  const fams = FITTERS.map((f) => f(reads, c.trophies, tStar));
  const DEBUG = process.env.MAKEFUN_DEBUG === '1';
  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nT; i++) {
    const preds = fams.map((fam) => fam.predict(i, tStar));
    if (DEBUG) {
      console.error(
        `trophy ${c.trophies[i].id} last=${lastRates[i]} preds(logit)=${preds.map((p) => p.toFixed(2)).join(',')} ` +
          `w=${weights[i].map((w) => w.toFixed(2)).join(',')} btRmse=${btRmse ? btRmse[i].toFixed(2) : '-'}`,
      );
    }
    let med = 0;
    for (let f = 0; f < fams.length; f++) med += weights[i][f] * preds[f];
    // Disagreement between families, in logit units.
    let dis = 0;
    for (let f = 0; f < fams.length; f++) dis += weights[i][f] * (preds[f] - med) * (preds[f] - med);
    dis = Math.sqrt(dis);

    // A page that never refreshed is stale, not stable: the first refresh
    // re-rates everything, so widen the band and shift the median down
    // (a refresh dilutes rates: new tracked players have earned almost
    // nothing yet). The launch-night refresh cut every rate but the 100% one.
    if (stalePage) {
      const last = lastRates[i];
      if (last !== null && last < 99.99) med -= 0.9;
    }

    const sRw = SIGMA_RW * Math.sqrt(horizonDays);
    const sBt = btRmse ? Math.min(0.5, btRmse[i] * Math.sqrt(Math.max(1, horizonDays / btHorizonDays))) : 0;
    const sStale = stalePage ? 1.2 : 0;
    const s = clamp(Math.sqrt(sRw * sRw + sBt * sBt + Math.min(dis, 0.35) ** 2 + sStale * sStale), 0.35, 1.3);

    let q = Z.map((z) => round4(logistic(med + z * s)));
    // Enforce non-decreasing after rounding.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    q = q.map((v) => clamp(v, 0.01, 99.99));

    // A trophy every tracked player has earned (100% in the last read) is
    // almost certainly still earned by nearly everyone at the target: the
    // tracked set is players who have played the opening. Keep the median
    // very high, with the band reaching down to allow for fresh joiners.
    const last = lastRates[i];
    if (last !== null && last >= 99.99) {
      q = [98.0, 99.0, 99.9, 99.97, 99.99];
    }
    rows.push({ id: c.trophies[i].id, q });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
