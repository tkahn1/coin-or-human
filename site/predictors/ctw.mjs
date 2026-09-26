// ctw: Context Tree Weighting experts on stay/switch + displayed-hit contexts, raw keys, and a
// "response" expert that models how the player reacts to the machine's displayed lean (g).
// Experts are combined by Fixed-Share on log-loss. g is chosen to maximise expected accuracy
// under the model *including* the player's modelled response to g itself.

const LOG_HALF = Math.log(0.5);

// ---------- CTW with KT estimators (beta-ratio form) ----------
class CTW {
  constructor(depth, gamma = 1, alpha = 0.5) {
    this.D = depth;
    this.gamma = gamma; // count discount (1 = stationary KT)
    this.alpha = alpha;
    this.nodes = new Map(); // key -> [a0, a1, logBeta]
    this.path = new Array(depth + 1);
    this.pe = new Float64Array(depth + 1);
    this.pw = new Float64Array(depth + 1);
  }
  node(key) {
    let n = this.nodes.get(key);
    if (!n) { n = [0, 0, 0]; this.nodes.set(key, n); }
    return n;
  }
  // ctx: function(i) -> i-th context bit (i=0 most significant / closest). Returns P(sym=1).
  predict(ctx) {
    const D = this.D, al = this.alpha;
    let key = 1;
    this.path[0] = this.node(1);
    for (let d = 1; d <= D; d++) {
      key = (key << 1) | ctx(d - 1);
      this.path[d] = this.node(key);
    }
    // leaf
    let n = this.path[D];
    let p = (n[1] + al) / (n[0] + n[1] + 2 * al);
    this.pe[D] = p; this.pw[D] = p;
    for (let d = D - 1; d >= 0; d--) {
      n = this.path[d];
      const pe = (n[1] + al) / (n[0] + n[1] + 2 * al);
      const lb = n[2];
      // pw = (beta*pe + pchild)/(beta+1)
      let pw;
      if (lb > 30) pw = pe;
      else if (lb < -30) pw = p;
      else { const b = Math.exp(lb); pw = (b * pe + p) / (b + 1); }
      this.pe[d] = pe; this.pw[d] = pw;
      p = pw;
    }
    return p;
  }
  // must be called right after predict() with the same context
  update(sym) {
    const D = this.D, g = this.gamma;
    for (let d = 0; d <= D; d++) {
      const n = this.path[d];
      if (d < D) {
        const pe = sym ? this.pe[d] : 1 - this.pe[d];
        const pc = sym ? this.pw[d + 1] : 1 - this.pw[d + 1];
        let lb = n[2] + Math.log(pe) - Math.log(pc);
        if (lb > 40) lb = 40; else if (lb < -40) lb = -40;
        n[2] = lb;
      }
      if (g < 1) { n[0] *= g; n[1] *= g; }
      n[sym]++;
    }
  }
}

// ---------- predictor ----------
function erf(x) { // Abramowitz-Stegun 7.1.26
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const sig = (z) => 1 / (1 + Math.exp(-z));
const logit = (p) => { p = Math.min(1 - 1e-9, Math.max(1e-9, p)); return Math.log(p / (1 - p)); };
const normLog = (a) => {
  let m = -Infinity; for (const v of a) m = Math.max(m, v);
  let s = 0; for (const v of a) s += Math.exp(v - m);
  const l = m + Math.log(s); for (let i = 0; i < a.length; i++) a[i] -= l;
};
const shareLog = (a, rate, prior) => { if (rate > 0) for (let i = 0; i < a.length; i++) a[i] = Math.log((1 - rate) * Math.exp(a[i]) + rate * prior[i]); };

// A combination pipeline: experts -> fixed-share mixture -> temperature calibration -> response model.
// Two copies run side by side with different forgetting rates: one for the guess decision (tracks
// fast), one for the betting probability p (forgets slowly, so it pays little on stationary players).
class Pipeline {
  constructor(N, O, R) {
    this.O = O; this.R = R; this.N = N;
    this.prior = O.prior ? O.prior.map((v) => v / O.prior.reduce((a, b) => a + b, 0)) : new Array(N).fill(1 / N);
    this.logw = Float64Array.from(this.prior, Math.log);
    this.SC = O.scales;
    this.calPrior = new Array(this.SC.length).fill(1 / this.SC.length);
    this.calw = Float64Array.from(this.calPrior, Math.log);
    // response hypotheses h = (theta, tau): the player's log-odds of pressing the displayed lean is
    // shifted by theta whenever the displayed confidence level >= tau. theta = 0: "ignores g".
    const L = O.levels.length, TH = [0], TAU = [0], lp0 = [O.theta0];
    const nz = O.thetas.filter((t) => t !== 0);
    for (const t of nz) for (let tau = 0; tau < L; tau++) {
      TH.push(t); TAU.push(tau);
      lp0.push((1 - O.theta0) / nz.length * (tau === 0 ? O.tau0 : (1 - O.tau0) / (L - 1)));
    }
    this.TH = TH; this.TAU = TAU; this.thPrior = lp0; this.K = TH.length;
    this.grid = Float64Array.from(lp0, Math.log);
    this.gw = new Float64Array(this.K);
  }
  // base probability from expert predictions ps; returns calibrated logit z
  prepare(ps) {
    const { N, logw, SC, calw, O } = this;
    let m = -Infinity; for (let i = 0; i < N; i++) m = Math.max(m, logw[i]);
    let s = 0, p = 0; for (let i = 0; i < N; i++) { const w = Math.exp(logw[i] - m); s += w; p += w * ps[i]; }
    const z0 = Math.max(-O.zClip, Math.min(O.zClip, logit(p / s)));
    let pc = 0; for (let j = 0; j < SC.length; j++) pc += Math.exp(calw[j]) * sig(SC[j] * z0);
    this.z0 = z0; this.z = Math.max(-O.zClip, Math.min(O.zClip, logit(pc)));
    for (let k = 0; k < this.K; k++) this.gw[k] = Math.exp(this.grid[k]);
    return this.z;
  }
  // P(key = 1) if we display lean sgn (+1 = lean to 1) at level lvl; lam < 1 tempers non-null hypotheses
  pAt(sgn, lvl, lam = 1) {
    const { z, TH, TAU, gw, K } = this;
    let p = 0, s = 0;
    for (let k = 0; k < K; k++) { const w = gw[k] * (k ? lam : 1); s += w; p += w * sig(z + (lvl >= TAU[k] ? TH[k] * sgn : 0)); }
    return p / s;
  }
  // posterior mass on "player counters our lean at level lvl"
  counterMass(lvl) { let m = 0; for (let k = 1; k < this.K; k++) if (this.TH[k] < 0 && lvl >= this.TAU[k]) m += this.gw[k]; return m; }
  update(ps, x, sgn, lvl, t) {
    const { N, logw, SC, calw, R, z, z0, TH, TAU, grid, K } = this;
    for (let i = 0; i < N; i++) { const q = Math.min(1 - 1e-6, Math.max(1e-6, ps[i])); logw[i] += (R.eta ?? 1) * Math.log(x ? q : 1 - q); }
    normLog(logw);
    shareLog(logw, R.alphaDecay ? Math.min(0.5, R.alphaDecay / (t + 2)) : R.alpha, this.prior);
    for (let j = 0; j < SC.length; j++) { const q = sig(SC[j] * z0); calw[j] += Math.log(x ? q : 1 - q); }
    normLog(calw); shareLog(calw, R.cal, this.calPrior);
    for (let k = 0; k < K; k++) { const q = sig(z + (lvl >= TAU[k] ? TH[k] * sgn : 0)); grid[k] += Math.log(x ? q : 1 - q); }
    normLog(grid); shareLog(grid, R.resp, this.thPrior);
  }
}

export default function createPredictor({ rng }, opts = {}) {
  const O = {
    levels: [0.005, 0.1, 0.25, 0.5], // candidate |g - 1/2| offsets of the displayed guess
    experts: ['sw:12:1', 'sw:6:0.95', 'raw:12:1', 'raw:4:0.95', 'cp:0:0.01', 'cp:2:0.01', 'cp:0:0.005:0.05', 'lagmix:2:10:6',
      'majmix:1:6', 'olr:0.3:0.0005', 'runmix:8', 'tally', 'cpraw:0.01:0.03', 'phasemix:2:16', 'winmix:4:8',
      'cp:0:0.005:0.01:0.1:0.2', 'cp:2:0.005:0.01:0.1:0.2'],
    dec: { alpha: 0.005, cal: 0.01, resp: 0.01 },      // forgetting rates, decision pipeline
    bet: { alphaDecay: 0.5, cal: 0.003, resp: 0.003, eta: 0.5 }, // forgetting rates, betting pipeline
    thetas: [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4], // response shifts in logit space
    theta0: 0.7,         // prior mass on theta = 0 (player ignores g)
    tau0: 0.5,           // prior mass (within theta != 0) on responding at every level
    scales: [0.4, 0.6, 0.8, 1, 1.2, 1.5], // calibration temperatures on the base logit
    kt: 0.5,             // KT pseudo-count
    pmix: 0, pmixShare: 0,
    probe: 1, probeStart: 100, probeBudget: 40, probeThr: 0.2,
    kappa: 0,            // risk premium per unit of displayed confidence
    lam: 0.3,            // decision tempering: demand more evidence before avoiding a level
    zClip: 5,
    ...opts,
  };
  const keys = [], sw = [], hits = [];
  let guessesLast = -1;
  let tallyK = 0, tallyS = 0; // running key (1 minus 0) and switch-minus-stay counts
  const bit = (arr, i) => { const j = arr.length - 1 - i; return j >= 0 ? arr[j] : 0; };

  // E1: CTW on switch bits, context interleaved switch/hit.
  const mkSwitch = (depth, gamma) => {
    const t = new CTW(depth, gamma, O.kt);
    const ctx = (i) => ((i & 1) ? bit(hits, i >> 1) : bit(sw, i >> 1));
    return {
      name: `sw${depth}g${gamma}`,
      p() { if (!keys.length) return 0.5; const s = t.predict(ctx); return keys[keys.length - 1] ? 1 - s : s; },
      upd(x) { if (!keys.length) return; t.predict(ctx); t.update(x ^ keys[keys.length - 1]); },
    };
  };
  // E4: change-tracking estimator on switch bits: per-context posterior over a grid of switch
  // probabilities with fixed-share forgetting (Bayesian tracking of a drifting/jumping bias).
  const CPG = [0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97];
  const recentHit = (w) => { const n = hits.length; if (n < w) return 0; let h = 0; for (let i = n - w; i < n; i++) h += hits[i]; return h / w; };
  const mkCP = (depth, rho, mu = 0, hotRho = 0, hotMu = 0, hotW = 12, hotT = 0.65) => {
    const tabs = new Map();
    const ctxKey = () => { let k = 1; for (let i = 0; i < depth; i++) k = (k << 1) | ((i & 1) ? bit(hits, i >> 1) : bit(sw, i >> 1)); return k; };
    const get = () => { const k = ctxKey(); let t = tabs.get(k); if (!t) { t = new Float64Array(CPG.length).fill(1 / CPG.length); tabs.set(k, t); } return t; };
    return {
      p() { if (!keys.length) return 0.5; const t = get(); let s = 0; for (let j = 0; j < CPG.length; j++) s += t[j] * CPG[j]; return keys[keys.length - 1] ? 1 - s : s; },
      upd(x) {
        if (!keys.length) return;
        const t = get(), y = x ^ keys[keys.length - 1];
        let z = 0; for (let j = 0; j < CPG.length; j++) { t[j] *= y ? CPG[j] : 1 - CPG[j]; z += t[j]; }
        const n = CPG.length, u = t.map((v) => v / z);
        const hot = hotRho + hotMu > 0 && recentHit(hotW) >= hotT;
        const r = hot ? hotRho : rho, m = hot ? hotMu : mu;
        for (let j = 0; j < n; j++) t[j] = (1 - r - m) * u[j] + m * u[n - 1 - j] + r / n;
      },
    };
  };
  // E5: lag-L copy model: CTW on d_t = x_t XOR x_{t-L}, context = previous d's (periodic habits, chunks)
  const mkLag = (L, depth) => {
    const t = new CTW(depth, 1, O.kt);
    const d = [];
    const ctx = (i) => bit(d, i);
    return {
      p() { if (keys.length < L) return 0.5; const s = t.predict(ctx); return keys[keys.length - L] ? 1 - s : s; },
      upd(x) { if (keys.length < L) return; t.predict(ctx); const y = x ^ keys[keys.length - L]; t.update(y); d.push(y); },
    };
  };
  // Group: a Bayesian (fixed-share) sub-mixture, so a family of experts costs one slot of prior.
  const mkGroup = (subs, share = 0.005) => {
    const n = subs.length, lw = new Float64Array(n).fill(-Math.log(n));
    let ps = new Float64Array(n);
    return {
      p() { let m = -Infinity; for (const v of lw) m = Math.max(m, v); let s = 0, p = 0; for (let i = 0; i < n; i++) { ps[i] = subs[i].p(); const w = Math.exp(lw[i] - m); s += w; p += w * ps[i]; } return p / s; },
      upd(x) {
        this.p();
        for (let i = 0; i < n; i++) { const q = Math.min(1 - 1e-6, Math.max(1e-6, ps[i])); lw[i] += Math.log(x ? q : 1 - q); }
        let m = -Infinity; for (const v of lw) m = Math.max(m, v); let s = 0; for (const v of lw) s += Math.exp(v - m);
        const l = m + Math.log(s); for (let i = 0; i < n; i++) lw[i] = Math.log((1 - share) * Math.exp(lw[i] - l) + share / n);
        for (const e of subs) e.upd(x);
      },
    };
  };
  const mkLagMix = (lo, hi, depth) => { const subs = []; for (let L = lo; L <= hi; L++) subs.push(mkLag(L, depth)); return mkGroup(subs); };
  // E6: frequency-majority agreement: y = [x == majority key seen after the current order-k context].
  // Captures players who (knowingly or not) play against / along an n-gram frequency predictor,
  // e.g. informed players beating the classic Aaronson oracle. KT per (tie?, margin>1?) state.
  const mkMaj = (k) => {
    const counts = new Map(), kt = [[0, 0], [0, 0], [0, 0]];
    let st = -1, m = 0;
    const ctxKey = () => { let c = 1; for (let i = 0; i < k; i++) c = (c << 1) | bit(keys, i); return c; };
    const prep = () => {
      st = -1; if (keys.length < k) return;
      const v = counts.get(ctxKey()); if (!v || v[0] === v[1]) return;
      m = v[1] > v[0] ? 1 : 0; st = Math.abs(v[1] - v[0]) > 1 ? 1 : 0;
    };
    return {
      p() { prep(); if (st < 0) return 0.5; const c = kt[st], a = (c[1] + 0.5) / (c[0] + c[1] + 1); return m ? a : 1 - a; },
      upd(x) {
        prep(); if (st >= 0) kt[st][x === m ? 1 : 0]++;
        if (keys.length >= k) { const key = ctxKey(); const v = counts.get(key) || [0, 0]; v[x]++; counts.set(key, v); }
      },
    };
  };
  const mkMajMix = (lo, hi) => { const subs = []; for (let k = lo; k <= hi; k++) subs.push(mkMaj(k)); return mkGroup(subs); };
  // E7: Bayesian online probit regression (AdPredictor-style ADF) predicting the switch bit from a
  // handful of human-bias features; shares strength across contexts, so it learns fast.
  const mkOLR = (prior2 = 0.3, drift = 0.0005, ext = 0) => {
    const F = ext ? 36 : 16, mu = new Float64Array(F), v2 = new Float64Array(F).fill(prior2), f = new Float64Array(F);
    const feats = () => {
      const n = keys.length, lk = keys[n - 1];
      f[0] = 1;
      f[1] = n >= 2 ? (sw[n - 1] ? 1 : -1) : 0;          // s-1 (sw[n-1] is switch into key n-1)
      f[2] = n >= 3 ? (sw[n - 2] ? 1 : -1) : 0;
      f[3] = n >= 4 ? (sw[n - 3] ? 1 : -1) : 0;
      f[4] = n >= 1 ? (hits[n - 1] ? 1 : -1) : 0;         // h-1
      f[5] = f[4] * f[1];
      let r = 0; for (let i = n - 1; i >= 0 && keys[i] === lk; i--) r++;
      f[6] = Math.min(r, 5) / 2.5 - 1;                    // run length
      let c = 0, w = Math.min(6, n); for (let i = n - w; i < n; i++) c += keys[i] === lk ? 1 : -1;
      f[7] = w ? c / w : 0;                               // local balance vs last key
      f[8] = lk ? 1 : -1;                                 // key-dependent switching (key bias)
      let c2 = 0, w2 = Math.min(20, n); for (let i = n - w2; i < n; i++) c2 += keys[i] === lk ? 1 : -1;
      f[9] = w2 ? c2 / w2 : 0;                            // longer-window balance
      // session tallies: key imbalance (relative to last key) and switch-vs-stay imbalance
      f[10] = Math.tanh(tallyK / 4) * (lk ? 1 : -1);
      f[11] = Math.tanh(tallyS / 4);
      const st = (v, k) => (v >= k ? 1 : v <= -k ? -1 : 0);
      f[12] = st(tallyK, 2) * (lk ? 1 : -1); f[13] = st(tallyS, 2);
      f[14] = st(tallyK, 4) * (lk ? 1 : -1); f[15] = st(tallyS, 4);
      if (!ext) return;
      const S = (j) => (n - j >= 1 ? (sw[n - j] ? 1 : -1) : 0), H = (j) => (n - j >= 0 ? (hits[n - j] ? 1 : -1) : 0);
      f[16] = S(4); f[17] = f[1] * f[2]; f[18] = f[1] * f[3]; f[19] = f[2] * f[3]; f[20] = f[1] * f[2] * f[3];
      f[21] = H(2); f[22] = H(2) * f[2]; f[23] = f[4] * H(2); f[24] = f[4] * f[8]; // hit x key
      for (let k = 1; k <= 5; k++) f[24 + k] = (r === k || (k === 5 && r >= 5)) ? 1 : 0; // run-length one-hot
      const bal = (w) => { let c = 0; const m = Math.min(w, n); for (let i = n - m; i < n; i++) c += keys[i] === lk ? 1 : -1; return m ? c / m : 0; };
      f[30] = bal(3); f[31] = bal(10); f[32] = bal(40);
      const swr = (w) => { let c = 0; const m = Math.min(w, n - 1); for (let i = n - m; i < n; i++) c += sw[i] ? 1 : -1; return m > 0 ? c / m : 0; };
      f[33] = swr(8); f[34] = swr(32);
      f[35] = n >= 1 && guessesLast >= 0 ? (guessesLast === lk ? 1 : -1) : 0; // last displayed guess vs last key
    };
    const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
    let sM = 0, sV = 1;
    const pre = () => { feats(); sM = 0; sV = 1; for (let i = 0; i < F; i++) { sM += mu[i] * f[i]; sV += v2[i] * f[i] * f[i]; } };
    return {
      p() { if (!keys.length) return 0.5; pre(); const ps = Phi(sM / Math.sqrt(sV)); return keys[keys.length - 1] ? 1 - ps : ps; },
      upd(x) {
        if (!keys.length) return;
        pre();
        const y = (x ^ keys[keys.length - 1]) ? 1 : -1, sd = Math.sqrt(sV), t = y * sM / sd;
        const pdf = Math.exp(-t * t / 2) / Math.sqrt(2 * Math.PI), cdf = Math.max(1e-12, Phi(t));
        const vv = pdf / cdf, ww = vv * (vv + t);
        for (let i = 0; i < F; i++) {
          if (!f[i]) continue;
          mu[i] += y * v2[i] * f[i] / sd * vv;
          v2[i] *= Math.max(1e-6, 1 - v2[i] * f[i] * f[i] / sV * ww);
          v2[i] += drift;
        }
      },
    };
  };
  // Small KT table expert: sym() -> 0/1 symbol being predicted relative to a reference (ref() -> key it maps
  // to when sym = 0 means "same as ref"), ctx() -> integer context. Returns P(key = 1).
  const mkTable = (ctx, ref, alpha = 0.5, decay = 1) => {
    const T = new Map();
    const get = () => { const c = ctx(); let v = T.get(c); if (!v) { v = [0, 0]; T.set(c, v); } return v; };
    return {
      p() { if (!keys.length) return 0.5; const v = get(), a = (v[1] + alpha) / (v[0] + v[1] + 2 * alpha); return ref() ? 1 - a : a; },
      upd(x) { if (!keys.length) return; const v = get(); if (decay < 1) { v[0] *= decay; v[1] *= decay; } v[x ^ ref()]++; },
    };
  };
  const runLen = (off = 0) => { // length of the run ending off keys ago
    let n = keys.length - off; if (n <= 0) return 0; const k = keys[n - 1]; let r = 0;
    while (n > 0 && keys[n - 1] === k) { r++; n--; } return r;
  };
  const lastKey = () => keys[keys.length - 1];
  // E8: run-length hazard experts: P(switch | current run length [, previous run length, key])
  const mkRunMix = (R = 8) => {
    const cur = () => Math.min(runLen(), R);
    const prev = () => Math.min(runLen(runLen()), 4);
    return mkGroup([
      mkTable(() => cur(), lastKey),
      mkTable(() => cur() * 8 + prev(), lastKey),
      mkTable(() => cur() * 2 + lastKey(), lastKey),
      mkTable(() => cur(), lastKey, 0.5, 0.95),
    ]);
  };
  // E9: tally keepers: P(key | bucketed key imbalance, bucketed switch-vs-stay imbalance [, last key]),
  // over the whole session and over a 30-key window.
  const bucket = (v, a) => (v <= -2 * a ? 0 : v < 0 ? 1 : v === 0 ? 2 : v < 2 * a ? 3 : 4);
  const tallies = (w) => {
    const n = keys.length, lo = Math.max(0, n - w); let kb = 0, sb = 0;
    for (let i = lo; i < n; i++) { kb += keys[i] ? 1 : -1; if (i > lo) sb += keys[i] !== keys[i - 1] ? 1 : -1; }
    return [kb, sb];
  };
  const mkTallyMix = () => {
    const cl = (v) => Math.max(-3, Math.min(3, v)) + 3;
    const c1 = (w) => () => { const [kb, sb] = tallies(w); return cl(kb) * 7 + cl(sb); };
    const kOnly = () => cl(tallyK), sOnly = () => cl(tallyS) * 2 + lastKey();
    return mkGroup([
      mkTable(kOnly, () => 0),                       // key tally -> key
      mkTable(() => cl(tallyS), lastKey),            // switch tally -> switch
      mkTable(sOnly, lastKey),
      mkTable(() => kOnly() * 7 + cl(tallyS), () => 0),
      mkTable(() => (kOnly() * 7 + cl(tallyS)) * 2 + lastKey(), () => 0),
      mkTable(c1(30), () => 0),
    ]);
  };
  // E10: score-reactive experts: a low-order switch model whose counts are reset or inverted when the
  // machine's visible hit rate over the last w keys exceeds thr (players who bail out when read).
  const mkHot = (w, thr, invert) => {
    let v = [[0, 0], [0, 0]], cool = 0;
    const hot = () => { const n = hits.length; if (n < w) return false; let h = 0; for (let i = n - w; i < n; i++) h += hits[i]; return h >= thr * w; };
    const c = () => bit(hits, 0);
    return {
      p() { if (!keys.length) return 0.5; const t = v[c()], a = (t[1] + 0.5) / (t[0] + t[1] + 1); return lastKey() ? 1 - a : a; },
      upd(x) {
        if (!keys.length) return;
        v[c()][x ^ lastKey()]++;
        if (cool > 0) cool--;
        else if (hot()) { v = invert ? v.map(([a, b]) => [b * 0.7, a * 0.7]) : [[0, 0], [0, 0]]; cool = w >> 1; }
      },
    };
  };
  const mkHotMix = () => mkGroup([[8, 0.75], [12, 0.67], [16, 0.62], [24, 0.6]].flatMap(([w, t]) => [mkHot(w, t, 0), mkHot(w, t, 1)]));
  // E11: change-tracking key-bias estimator (with mirror jumps: preference noticed and overcorrected)
  const mkCPRaw = (rho, mu) => {
    const t = new Float64Array(CPG.length).fill(1 / CPG.length), n = CPG.length;
    return {
      p() { let s = 0; for (let j = 0; j < n; j++) s += t[j] * CPG[j]; return s; },
      upd(x) {
        let z = 0; for (let j = 0; j < n; j++) { t[j] *= x ? CPG[j] : 1 - CPG[j]; z += t[j]; }
        const u = t.map((v) => v / z);
        for (let j = 0; j < n; j++) t[j] = (1 - rho - mu) * u[j] + mu * u[n - 1 - j] + rho / n;
      },
    };
  };
  // E12: phase tables: for period P, KT on key given (t mod P, x_{t-P}, x_{t-2P}) -- fixed-width mental
  // algorithms (counting, chunk recitation) where each position in the block has its own rule.
  const mkPhaseMix = (lo, hi) => {
    const subs = [];
    for (let P = lo; P <= hi; P++) subs.push(mkTable(() => ((keys.length % P) * 2 + bit(keys, P - 1)) * 2 + bit(keys, 2 * P - 1), () => 0));
    return mkGroup(subs, 0.01);
  };
  // E13: window-count experts: P(key | #ones in the last w keys [, last key]) -- local representativeness.
  const mkWinMix = (lo, hi) => {
    const subs = [];
    const cnt = (w) => { let c = 0; const n = keys.length; for (let i = Math.max(0, n - w); i < n; i++) c += keys[i]; return c; };
    for (let w = lo; w <= hi; w++) { subs.push(mkTable(() => cnt(w), () => 0)); subs.push(mkTable(() => cnt(w) * 2 + lastKey(), () => 0)); }
    return mkGroup(subs);
  };
  // E2: CTW on raw keys
  const mkRaw = (depth, gamma) => {
    const t = new CTW(depth, gamma, O.kt);
    const ctx = (i) => bit(keys, i);
    return {
      name: `raw${depth}g${gamma}`,
      p() { return t.predict(ctx); },
      upd(x) { t.predict(ctx); t.update(x); },
    };
  };
  const MK = { phasemix: mkPhaseMix, winmix: mkWinMix, runmix: mkRunMix, tally: mkTallyMix, hot: mkHotMix, cpraw: mkCPRaw, olr: mkOLR, majmix: mkMajMix, maj: mkMaj, lagmix: mkLagMix, lag: mkLag, cp: mkCP, sw: mkSwitch, raw: mkRaw };
  const experts = [];
  for (const spec of O.experts) { const [k, ...a] = spec.split(':'); experts.push(MK[k](...a.map(Number))); }
  const N = experts.length;
  const dec = new Pipeline(N, O, O.dec), bet = new Pipeline(N, O, O.bet);
  const L = O.levels.length;
  let cache = null;
  let lwM = 0, lwF = 0, lwPeak = 0, probeUsed = 0, llB = Math.log(0.5), llD = Math.log(0.5);

  return {
    predict() {
      const ps = new Float64Array(N);
      for (let i = 0; i < N; i++) ps[i] = experts[i].p();
      dec.prepare(ps); bet.prepare(ps);
      // choose the displayed guess probability g maximising expected accuracy under the model of how
      // the player responds to g: acc = 1/2 + off * (2 P(key = lean) - 1)
      let best = null;
      for (let lean = 0; lean < 2; lean++) for (let st = 0; st < L; st++) {
        const sgn = lean ? 1 : -1, off = O.levels[st];
        const pd = dec.pAt(sgn, st, O.lam);
        const acc = 0.5 + off * (2 * (lean ? pd : 1 - pd) - 1 - O.kappa);
        if (!best || acc > best.acc + 1e-12) best = { acc, sgn, st, g: lean ? 0.5 + off : 0.5 - off };
      }
      // Detection probe: if we suspect the player counters confident guesses but have not yet proven it
      // (own copy of the referee's wealth W < 20), expose a confident guess on a limited budget of keys.
      // The counter-play then shows up in p and the wealth grows; costs accuracy only on suspected counterers.
      const top = L - 1;
      if (O.probe && best.st < top && keys.length >= O.probeStart && probeUsed < O.probeBudget && lwPeak < Math.log(20)
          && bet.counterMass(top) > O.probeThr) {
        const sgn = dec.z >= 0 ? 1 : -1;
        best = { acc: 0, sgn, st: top, g: sgn > 0 ? 0.5 + O.levels[top] : 0.5 - O.levels[top], probe: 1 };
      }
      // p: Bayesian mixture of the slow (bet) and fast (dec) pipelines' predictions at the chosen g
      const pB = bet.pAt(best.sgn, best.st), pD = dec.pAt(best.sgn, best.st);
      const wB = O.pmix ? 1 / (1 + Math.exp(llD - llB)) : 1;
      const p = wB * pB + (1 - wB) * pD;
      cache = { ps, best, p, pB, pD };
      return { p, g: best.g };
    },
    update(x, guess) {
      if (!cache) this.predict();
      const { ps, best } = cache;
      if (best.probe) probeUsed++;
      llB += Math.log(x ? cache.pB : 1 - cache.pB); llD += Math.log(x ? cache.pD : 1 - cache.pD);
      if (O.pmixShare) { const d = llD - llB, c = Math.log(O.pmixShare); if (d > -c) llB = llD + c; else if (d < c) llD = llB + c; }
      const q = Math.min(0.99, Math.max(0.01, cache.p));
      lwM += Math.log(x ? 2 * q : 2 * (1 - q)); lwF += Math.log(x ? 2 * (1 - q) : 2 * q);
      const mx = Math.max(lwM, lwF);
      lwPeak = Math.max(lwPeak, mx + Math.log(Math.exp(lwM - mx) + Math.exp(lwF - mx)) - Math.log(2));
      dec.update(ps, x, best.sgn, best.st, keys.length);
      bet.update(ps, x, best.sgn, best.st, keys.length);
      for (const e of experts) e.upd(x);
      if (keys.length) { sw.push(x ^ keys[keys.length - 1]); tallyS += x !== keys[keys.length - 1] ? 1 : -1; } else sw.push(0);
      tallyK += x ? 1 : -1; guessesLast = guess;
      keys.push(x); hits.push(guess === x ? 1 : 0);
      cache = null;
    },
  };
}
