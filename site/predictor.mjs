// ensemble: pool of simple experts (context counts, lag/match, n-gram-majority agreement, CTW incl. phase-CTW,
// score-driven style-flip HMMs) -> tempered Bayes mixture + level bandit/probe for g; Bayes mixture for p. See ALGORITHM.md.
const LOG = Math.log, EXP = Math.exp;
const stretch = (p) => LOG(p / (1 - p));
const squash = (x) => (x > 30 ? 1 : x < -30 ? 0 : 1 / (1 + EXP(-x)));
const clampP = (p, e = 1e-3) => (p < e ? e : p > 1 - e ? 1 - e : p);
function logAddExp(a, b) { return a > b ? a + LOG(1 + EXP(b - a)) : b + LOG(1 + EXP(a - b)); }
const LHALF = Math.log(0.5);

// Binary CTW with KT estimators. ctx: array of D bits, most recent first.
function makeCTW(D) {
  const nodes = new Map();
  const path = new Array(D + 1);
  function keyOf(ctx, d) { let k = 1; for (let i = 0; i < d; i++) k = (k << 1) | ctx[i]; return k; }
  function prob(ctx) {
    for (let d = 0; d <= D; d++) path[d] = keyOf(ctx, d);
    let lpw1 = 0, lpwOld = 0;
    for (let d = D; d >= 0; d--) {
      const n = nodes.get(path[d]);
      const a = n ? n[0] : 0, b = n ? n[1] : 0, lpe = n ? n[2] : 0, lpw = n ? n[3] : 0;
      const lpe1 = lpe + LOG((b + 0.5) / (a + b + 1));
      if (d === D) lpw1 = lpe1;
      else {
        const sib = nodes.get((path[d] << 1) | (1 - ctx[d]));
        lpw1 = LHALF + logAddExp(lpe1, lpw1 + (sib ? sib[3] : 0));
      }
      lpwOld = lpw;
    }
    return EXP(lpw1 - lpwOld);
  }
  function update(ctx, x) {
    let childW = 0;
    for (let d = D; d >= 0; d--) {
      let n = nodes.get(path[d]);
      if (!n) { n = [0, 0, 0, 0]; nodes.set(path[d], n); }
      n[2] += LOG(((x ? n[1] : n[0]) + 0.5) / (n[0] + n[1] + 1));
      n[x]++;
      if (d === D) n[3] = n[2];
      else {
        const sib = nodes.get((path[d] << 1) | (1 - ctx[d]));
        n[3] = LHALF + logAddExp(n[2], childW + (sib ? sib[3] : 0));
      }
      childW = n[3];
    }
  }
  return { prob, update }; // update() must follow prob() with the same ctx (reuses path)
}

// Fixed-share Bayesian mixture with prior weights.
function makeMix(prior, alpha, eta = 1, alphaT = 0) {
  const n = prior.length;
  const lp = prior.map((w) => LOG(w));
  const lw = Float64Array.from(lp);
  const pr = new Float64Array(n);
  let out = 0.5, steps = 0;
  return {
    pr,
    mix() {
      let mx = -Infinity; for (let i = 0; i < n; i++) if (lw[i] > mx) mx = lw[i];
      let s = 0, sp = 0;
      for (let i = 0; i < n; i++) { const w = EXP(lw[i] - mx); s += w; sp += w * pr[i]; }
      out = sp / s; return out;
    },
    update(bit) {
      steps++;
      if (alphaT) alpha = alphaT / (steps + 2);
      let mx = -Infinity;
      for (let i = 0; i < n; i++) { lw[i] += eta * LOG(bit ? pr[i] : 1 - pr[i]); if (lw[i] > mx) mx = lw[i]; }
      let s = 0; for (let i = 0; i < n; i++) s += EXP(lw[i] - mx);
      const lz = mx + LOG(s);
      for (let i = 0; i < n; i++) lw[i] = LOG((1 - alpha) * EXP(lw[i] - lz) + alpha * EXP(lp[i]));
    },
  };
}

export default function createPredictor({ rng, opts = {} }) {
  const O = (k, d) => (k in opts ? opts[k] : d);
  const keys = [], hits = [], sws = []; // sws[t] = keys[t] != keys[t-1] (sws[0] = 0)
  const T = () => keys.length;

  // ---------- context functions (evaluated at predict time) ----------
  const rawK = (k) => () => {
    const t = T(); if (t < k) return -1;
    let c = 0; for (let i = t - k; i < t; i++) c = (c << 1) | keys[i]; return c;
  };
  const swK = (k) => () => {
    const t = T(); if (t < k + 1) return -1;
    let c = 0; for (let i = t - k; i < t; i++) c = (c << 1) | sws[i]; return c;
  };
  const swHitK = (k) => () => {
    const t = T(); if (t < k + 1) return -1;
    let c = 0; for (let i = t - k; i < t; i++) c = (c << 2) | (sws[i] << 1) | hits[i]; return c;
  };
  const keyHitK = (k) => () => {
    const t = T(); if (t < k) return -1;
    let c = 0; for (let i = t - k; i < t; i++) c = (c << 2) | (keys[i] << 1) | hits[i]; return c;
  };
  const imbal = (w) => () => {
    const t = T(); if (t < w) return -1;
    let c = 0; for (let i = t - w; i < t; i++) c += keys[i]; return c;
  };
  const imbalB = (w) => () => { // bucketed imbalance over a long window
    const t = T(); if (t < w) return -1;
    let c = 0; for (let i = t - w; i < t; i++) c += keys[i];
    const d = (2 * c - w) / Math.sqrt(w);
    return d < -1.5 ? 0 : d < -0.75 ? 1 : d < -0.25 ? 2 : d <= 0.25 ? 3 : d <= 0.75 ? 4 : d <= 1.5 ? 5 : 6;
  };
  const runCtx = (withKey) => () => {
    const t = T(); if (t < 1) return -1;
    let r = 0; const l = keys[t - 1];
    for (let i = t - 1; i >= 0 && keys[i] === l && r < 8; i--) r++;
    return withKey ? r * 2 + l : r;
  };
  const scoreCtx = (w) => () => { // recent visible hit-rate bucket x last switch
    const t = T(); if (t < w + 1) return -1;
    let h = 0; for (let i = t - w; i < t; i++) h += hits[i];
    const b = h * 10 < w * 4 ? 0 : h * 10 < w * 6 ? 1 : h * 10 < w * 7 ? 2 : 3;
    return b * 2 + sws[t - 1];
  };

  const lagCtx = (L, k) => () => { // last k "differs from lag L" bits
    const t = T(); if (t < L + k) return -1;
    let c = 0; for (let i = t - k; i < t; i++) c = (c << 1) | (keys[i] !== keys[i - L] ? 1 : 0); return c;
  };
  const specs = [];
  const DECAYS = O('decays', [1, 0.9]);
  const add = (fn, size, target, ds = DECAYS) => { for (const d of ds) specs.push({ fn, size, target, d }); };
  for (let k = 0; k <= 6; k++) add(rawK(k), 1 << k, 0);
  for (let k = 1; k <= 6; k++) add(swK(k), 1 << k, 1);
  for (let k = 1; k <= 4; k++) add(swHitK(k), 1 << (2 * k), 1);
  for (let k = 1; k <= 3; k++) add(keyHitK(k), 1 << (2 * k), 0);
  for (let w = 4; w <= 8; w++) add(imbal(w), w + 1, 0);
  for (const w of O('longW', [12, 16, 24])) add(imbalB(w), 7, 0);
  for (const L of O('lags', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])) add(lagCtx(L, O('lagK', 1)), 1 << O('lagK', 1), L, O('lagD', [0.95]));
  add(runCtx(false), 9, 1);
  add(runCtx(true), 18, 1);
  add(scoreCtx(10), 8, 1);
  add(scoreCtx(20), 8, 1);
  const models = specs.map((s) => ({ ...s, tab: new Float64Array(2 * s.size), cur: -1 }));
  const KT = O('kt', 0.4);

  // ---------- match models ----------
  const MATCH_L = [3, 5, 8, 12, 16, 24];
  const match = MATCH_L.map((L) => ({ L, map: new Map(), pred: -1, n: 0, c: 0 }));
  const ctxKey = (L) => { const t = T(); if (t < L) return null; return keys.slice(t - L).join(''); };

  // ---------- n-gram-majority agreement experts ----------
  // ref = the key a simple order-k frequency oracle would predict; learn P(key == ref | k, strength).
  // Catches players who play along with / against what a naive model (or their own model) predicts.
  const MAJ = O('maj', [1, 2, 3, 4, 5, 6]).flatMap((k) => O('majD', [1, 0.9]).map((d) => ({
    k, d, n0: new Float64Array(1 << k), n1: new Float64Array(1 << k), ag: new Float64Array(6), ctx: -1, ref: -1, b: 0,
  })));
  function majPredict(m) {
    const t = T(); m.ctx = -1; m.ref = -1;
    if (t < m.k) return 0.5;
    let c = 0; for (let i = t - m.k; i < t; i++) c = (c << 1) | keys[i];
    m.ctx = c;
    const a = m.n0[c], b = m.n1[c];
    if (Math.abs(a - b) < 0.5) return 0.5;
    m.ref = b > a ? 1 : 0;
    const dif = Math.abs(a - b);
    m.b = dif < 1.5 ? 0 : dif < 3.5 ? 1 : 2;
    const q = (m.ag[2 * m.b + 1] + 0.5) / (m.ag[2 * m.b] + m.ag[2 * m.b + 1] + 1);
    return m.ref ? q : 1 - q;
  }
  function majUpdate(m, bit) {
    if (m.ref >= 0) { const i = 2 * m.b + (bit === m.ref ? 1 : 0); m.ag[i] += 1; }
    if (m.ctx >= 0) {
      if (m.d < 1) { m.n0[m.ctx] *= m.d; m.n1[m.ctx] *= m.d; }
      if (bit) m.n1[m.ctx]++; else m.n0[m.ctx]++;
    }
  }

  // ---------- style-flip HMMs ----------
  // Hidden switch rate theta on a grid; each step the style may invert (theta -> 1 - theta) or reset (uniform)
  // with a hazard that depends smoothly on the machine's visible recent hit rate r:
  // h = sigmoid(a + b (r - 0.5)). Models players who change style when they see they are being predicted.
  const TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], NTH = TH.length;
  const HMM = [];
  for (const w of O('hmmW', [10, 25])) for (const [a, b] of O('hmmAB', [[-4.5, 0], [-4.5, 15], [-3.5, 25]])) for (const mode of [0, 1]) {
    if (b === 0 && w !== 10) continue;
    HMM.push({ w, a, b, mode, key: 0, pi: new Float64Array(NTH).fill(1 / NTH), ps: 0.5 });
  }
  // key-bias variants: hidden P(key=1) that inverts with a constant hazard, or with a hazard driven by the
  // player's own recent key imbalance (noticing a preference and overcorrecting)
  for (const [a, b] of O('khmm', [])) for (const mode of [0, 1])
    HMM.push({ w: 16, a, b, mode, key: 1, pi: new Float64Array(NTH).fill(1 / NTH), ps: 0.5 });
  function hmmPredict(h) {
    const t = T(); if (t < 1 && !h.key) return 0.5;
    let s = 0; for (let i = 0; i < NTH; i++) s += h.pi[i] * TH[i];
    h.ps = s; return h.key ? s : keys[t - 1] ? 1 - s : s;
  }
  function hmmUpdate(h, sw) {
    const pi = h.pi; let z = 0;
    for (let i = 0; i < NTH; i++) { pi[i] *= sw ? TH[i] : 1 - TH[i]; z += pi[i]; }
    for (let i = 0; i < NTH; i++) pi[i] /= z;
  }
  function hmmTransit(h) { // after hits has the newest entry
    const n = hits.length, w = Math.min(h.w, n), src = h.key ? keys : hits;
    let r = 0.5; if (w >= 5) { let c = 0; for (let i = n - w; i < n; i++) c += src[i]; r = c / w; }
    if (h.key) r = 0.5 + Math.abs(r - 0.5);
    const hz = squash(h.a + h.b * (r - 0.5)), pi = h.pi;
    const old = Float64Array.from(pi);
    for (let i = 0; i < NTH; i++) pi[i] = (1 - hz) * old[i] + hz * (h.mode ? 1 / NTH : old[NTH - 1 - i]);
    for (let i = 0; i < NTH; i++) pi[i] = 0.995 * pi[i] + 0.005 / NTH;
  }

  // ---------- CTW experts ----------
  const ctwDefs = [
    { D: 10, target: 1, ctx: (c) => { const t = T(); for (let i = 0; i < 5; i++) { const j = t - 1 - i; c[2 * i] = j >= 1 ? sws[j] : 0; c[2 * i + 1] = j >= 0 ? hits[j] : 0; } } },
    { D: 12, target: 0, ctx: (c) => { const t = T(); for (let i = 0; i < 12; i++) { const j = t - 1 - i; c[i] = j >= 0 ? keys[j] : 0; } } },
    { D: 8, target: 1, ctx: (c) => { const t = T(); for (let i = 0; i < 8; i++) { const j = t - 1 - i; c[i] = j >= 1 ? sws[j] : 0; } } },
  ];
  const ctws = ctwDefs.map((d) => ({ ...d, m: makeCTW(d.D), buf: new Array(d.D).fill(0), p: 0.5 }));
  // phase-conditioned CTWs: context = (t mod L) then the previous L keys oldest-first (fixed-width words)
  const pctws = O('phaseL', [4, 5, 6, 7, 8]).map((L) => ({
    D: 3 + L, target: 0, p: 0.5, m: makeCTW(3 + L), buf: new Array(3 + L).fill(0),
    ctx: (c) => {
      const t = T(), ph = t % L;
      c[0] = (ph >> 2) & 1; c[1] = (ph >> 1) & 1; c[2] = ph & 1;
      for (let i = 0; i < L; i++) { const j = t - L + i; c[3 + i] = j >= 0 ? keys[j] : 0; }
    },
  }));

  const NE = models.length + match.length + MAJ.length + HMM.length + pctws.length + ctws.length; // pool size
  const P = new Float64Array(NE);   // pool probabilities P(key=1)
  const S = new Float64Array(NE + 1); // stretched inputs (+ bias) for the logistic mixer

  // ---------- PAQ-style logistic mixer (one pool member) ----------
  const NSETS = 4;
  const W = [0, 1].map(() => new Float64Array((NE + 1) * NSETS));
  const LR = O('LR', [0.02, 0.006]);
  let wsel = 0, pA = 0.5;
  const pm = [0.5, 0.5], fw = new Float64Array([0.5, 0.5]), x2 = [0, 0];

  function computePool() {
    const t = T();
    let j = 0;
    const last = t ? keys[t - 1] : 0;
    for (const m of models) {
      const c = m.fn(); m.cur = c;
      let p = 0.5;
      if (c >= 0) {
        const n0 = m.tab[2 * c], n1 = m.tab[2 * c + 1];
        p = (n1 + KT) / (n0 + n1 + 2 * KT);
        if (m.target >= 1 && keys[t - m.target]) p = 1 - p; // p(differs from lag) -> p(key=1)
      }
      P[j++] = clampP(p);
    }
    for (const mm of match) {
      const k = ctxKey(mm.L);
      mm.pred = k !== null && mm.map.has(k) ? mm.map.get(k) : -1;
      let p = 0.5;
      if (mm.pred >= 0) { const r = (mm.c + 0.5) / (mm.n + 1); p = mm.pred ? r : 1 - r; }
      P[j++] = clampP(p);
    }
    for (const m of MAJ) P[j++] = clampP(majPredict(m));
    for (const h of HMM) P[j++] = clampP(hmmPredict(h));
    for (const c of pctws) { c.ctx(c.buf); P[j++] = c.p = clampP(c.m.prob(c.buf)); }
    for (const c of ctws) {
      c.ctx(c.buf);
      let p = c.m.prob(c.buf);
      if (c.target === 1 && last) p = 1 - p;
      P[j++] = c.p = clampP(p);
    }
    for (let i = 0; i < NE; i++) S[i] = stretch(P[i]);
    S[NE] = 0.3;
    wsel = t >= 1 ? sws[t - 1] * 2 + hits[t - 1] : 0;
    // logistic mixer
    const NI = NE + 1;
    for (let q = 0; q < 2; q++) {
      const w = W[q], o = wsel * NI;
      let dot = 0;
      for (let i = 0; i < NI; i++) dot += w[o + i] * S[i];
      x2[q] = dot; pm[q] = squash(dot);
    }
    pA = clampP(squash(fw[0] * x2[0] + fw[1] * x2[1]));
  }
  function updateMixerA(bit) {
    const NI = NE + 1;
    for (let q = 0; q < 2; q++) {
      const err = bit - pm[q], w = W[q], o = wsel * NI, lr = LR[q];
      for (let i = 0; i < NI; i++) w[o + i] += lr * err * S[i];
    }
    const errF = bit - pA;
    for (let q = 0; q < 2; q++) fw[q] += 0.002 * errF * x2[q];
  }

  // ---------- guess mixture: Bayes over pool + A ----------
  const gPrior = new Array(NE + 1).fill(0);
  {
    const wc = O('gCore', 0.3), wa = O('gA', 1 / (NE + 1)); // main CTWs get wc, logistic mixer wa
    for (let i = 0; i < NE - 3; i++) gPrior[i] = (1 - wc - wa) / (NE - 3);
    for (let i = NE - 3; i < NE; i++) gPrior[i] = wc / 3;
    gPrior[NE] = wa;
  }
  const mixG = makeMix(gPrior, O('gAlpha', 0.01), O('gEta', 0.5));
  // ---------- probability mixture (for p): Bayes over pool + A + counter-expert + uniform + guess mixtures ----------
  const mixG1 = makeMix(new Array(NE + 1).fill(1 / (NE + 1)), O('gAlpha', 0.01), 1);
  const IDX_A = NE, IDX_C = NE + 1, IDX_U = NE + 2, IDX_G = NE + 3, IDX_GC = NE + 4, IDX_G1 = NE + 5;
  const pPrior = new Array(NE + 6).fill(0);
  {
    const wCore = O('wCore', 0.2), wU = O('wU', 0.05), wC = O('wC', 0.1), wA = O('wA', 0.05);
    const wG = O('wG', 0.05), wGC = O('wGC', 0.3), wG1 = O('wG1', 0.05);
    const core = [NE - 3, NE - 2, NE - 1]; // CTWs (last pool members)
    const rest = 1 - wCore - wU - wC - wA - wG - wGC - wG1;
    for (let i = 0; i < NE; i++) pPrior[i] = rest / (NE - core.length);
    for (const i of core) pPrior[i] = wCore / core.length;
    pPrior[IDX_A] = wA; pPrior[IDX_C] = wC; pPrior[IDX_U] = wU;
    pPrior[IDX_G] = wG; pPrior[IDX_GC] = wGC; pPrior[IDX_G1] = wG1;
  }
  // online logistic calibration of the guess mixture: p = sigma(ca * stretch(pG) + cb)
  let ca = 1, cb = 0, sG = 0, pGC = 0.5;
  const CLR = O('cLR', 0.02);
  const mixP = makeMix(pPrior, O('pAlpha', 0.005), 1, O('pAlphaT', 0.5));

  // ---------- guess-level bandit (robust to players who see g) ----------
  const LEVELS = O('levels', [0.005, 0.15, 0.3, 0.5]);
  const NL = LEVELS.length;
  const lvN = new Float64Array(NL), lvC = new Float64Array(NL);
  const LVD = O('lvDecay', 0.995), LVZ = O('lvZ', 2), LVZR = O('lvZR', 1);
  const LVCLIMB = O('climb', 1), LVCN = O('climbN', 8), LVCZ = O('climbZ', 0.5);
  const lvBan = new Uint8Array(NL), lvL = new Float64Array(NL);
  const LVB = O('lvB', 2.5), LA = LOG(O('aC', 0.2) / 0.55), LD = LOG((1 - O('aC', 0.2)) / 0.45);
  let side = -1, lvl = NL - 1, pOut = 0.5;
  const PRB = O('probe', 1), PRB_START = O('probeStart', 60), PRB_MAX = O('probeMax', 40), PRB_MARGIN = O('probeMargin', 0.7);
  const LN2 = Math.log(2), LN20 = Math.log(20);
  let probing = false, probes = 0, lwM = 0, lwF = 0;

  function finalize() {
    for (let i = 0; i < NE; i++) mixG.pr[i] = P[i];
    mixG.pr[NE] = pA;
    const pG = mixG.mix();
    for (let i = 0; i <= NE; i++) mixG1.pr[i] = mixG.pr[i];
    const pG1 = mixG1.mix();
    sG = stretch(clampP(pG)); pGC = clampP(squash(ca * sG + cb));
    side = pG > 0.5 ? 1 : pG < 0.5 ? 0 : -1;
    // choose level
    let best = 0;
    const zs = [];
    for (let l = 0; l < NL; l++) {
      const z = (lvC[l] - 0.5 * lvN[l]) / Math.sqrt(0.25 * lvN[l] + 1);
      zs.push(z);
      if (z < -LVZ || lvL[l] > LVB) lvBan[l] = 1; else if (z > -LVZR && lvL[l] < 0) lvBan[l] = 0;
    }
    if (LVCLIMB) {
      best = lvl;
      if (lvBan[best]) best = 0;
      else if (best < NL - 1 && !lvBan[best + 1] && lvN[best] >= LVCN && zs[best] > LVCZ) best++;
      while (best > 0 && lvBan[best]) best--;
    } else {
      for (let l = 0; l < NL; l++) { if (lvBan[l] && l > 0) break; best = l; }
    }
    lvl = best;
    // detection probe: if a higher level is banned (player seems to counter confident guesses) and our own
    // copy of the referee's betting score has not reached 20 yet, expose that level for a few keys so the
    // counter-expert in p can collect evidence. For a falsely banned normal player this is just argmax.
    probing = false;
    if (PRB && T() >= PRB_START && probes < PRB_MAX && Math.max(lwM, lwF) - LN2 < LN20 + PRB_MARGIN) {
      for (let l = NL - 1; l > lvl; l--) if (lvBan[l]) { lvl = l; probing = true; break; }
    }
    const aL = (lvC[lvl] + 1) / (lvN[lvl] + 2);
    for (let i = 0; i < NE; i++) mixP.pr[i] = P[i];
    mixP.pr[IDX_A] = pA;
    mixP.pr[IDX_C] = side < 0 ? 0.5 : clampP(side ? aL : 1 - aL, 0.02);
    mixP.pr[IDX_U] = 0.5;
    mixP.pr[IDX_G] = clampP(pG); mixP.pr[IDX_GC] = pGC; mixP.pr[IDX_G1] = clampP(pG1);
    pOut = mixP.mix();
    if (side < 0) return 0.5;
    const m = LEVELS[lvl];
    return side ? 0.5 + m : 0.5 - m;
  }

  let computed = false;
  return {
    predict() {
      computePool(); const g = finalize(); computed = true;
      return { p: clampP(pOut, 0.01), g };
    },
    update(bit, guess) {
      if (!computed) { computePool(); finalize(); }
      computed = false;
      mixG.update(bit); mixP.update(bit); mixG1.update(bit);
      { const q = Math.min(0.99, Math.max(0.01, pOut)); lwM += LOG(bit ? 2 * q : 2 * (1 - q)); lwF += LOG(bit ? 2 * (1 - q) : 2 * q); }
      if (probing) probes++;
      { const e = bit - pGC; ca += CLR * e * sG; cb += CLR * e; }
      for (let l = 0; l < NL; l++) { lvN[l] *= LVD; lvC[l] *= LVD; lvL[l] *= LVD; }
      if (side >= 0) { const ag = bit === side ? 1 : 0; lvN[lvl]++; lvC[lvl] += ag; lvL[lvl] += ag ? LA : LD; }
      updateMixerA(bit);
      const t = T();
      const last = t ? keys[t - 1] : 0;
      const sw = t ? (bit !== last ? 1 : 0) : 0;
      for (const m of models) {
        if (m.cur < 0) continue;
        const y = m.target >= 1 ? (bit !== keys[t - m.target] ? 1 : 0) : bit;
        const tab = m.tab, c = m.cur;
        if (m.d < 1) { tab[2 * c] *= m.d; tab[2 * c + 1] *= m.d; }
        tab[2 * c + y] += 1;
      }
      for (const mm of match) {
        if (mm.pred >= 0) { mm.n++; mm.c += mm.pred === bit ? 1 : 0; }
        const k = ctxKey(mm.L);
        if (k !== null) mm.map.set(k, bit);
      }
      for (const c of ctws) c.m.update(c.buf, c.target === 1 ? sw : bit);
      for (const c of pctws) c.m.update(c.buf, bit);
      for (const m of MAJ) majUpdate(m, bit);
      for (const h of HMM) if (h.key) hmmUpdate(h, bit); else if (t >= 1) hmmUpdate(h, sw);
      keys.push(bit); hits.push(guess === bit ? 1 : 0); sws.push(sw);
      for (const h of HMM) hmmTransit(h);
    },
  };
}
