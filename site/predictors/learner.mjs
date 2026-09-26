// "learner": feature-based online learning + context-tree experts, logistic mixing,
// and a guess layer that learns how the player reacts to the displayed guess probability g.
// Dependency-free ES module; uses only the supplied rng (currently not needed at all).

const LN2 = Math.log(2);
const squash = (x) => 1 / (1 + Math.exp(-x));
const stretch = (p) => Math.log(p / (1 - p));
const clip = (x, a, b) => (x < a ? a : x > b ? b : x);
const logAddExp = (a, b) => (a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b)));

// ---------------------------------------------------------------------------------------------
// Context Tree Weighting with (optionally discounted) KT estimators. Binary symbols, binary contexts.
class CTW {
  constructor(depth, gamma = 1, alpha = 0.5) {
    this.D = depth; this.gamma = gamma; this.alpha = alpha;
    const cap = 1 << 14;
    this.c0 = new Float64Array(cap); this.c1 = new Float64Array(cap);
    this.lpe = new Float64Array(cap); this.lpw = new Float64Array(cap);
    this.ch = new Int32Array(cap * 2).fill(-1);
    this.n = 1; // node 0 = root
    this.path = new Int32Array(depth + 1);
    this.pe1 = new Float64Array(depth + 1); this.pe0 = new Float64Array(depth + 1);
    this.pw1 = new Float64Array(depth + 1); this.pw0 = new Float64Array(depth + 1);
    this.bits = new Int8Array(depth);
    this.p = 0.5;
  }
  grow() {
    const cap = this.c0.length * 2;
    const g = (a, T) => { const b = new T(cap); b.set(a); return b; };
    this.c0 = g(this.c0, Float64Array); this.c1 = g(this.c1, Float64Array);
    this.lpe = g(this.lpe, Float64Array); this.lpw = g(this.lpw, Float64Array);
    const ch = new Int32Array(cap * 2).fill(-1); ch.set(this.ch); this.ch = ch;
  }
  // ctx(i) returns the i-th context bit (i = 0 most recent); returns P(next = 1).
  predict(ctx) {
    const D = this.D, a = this.alpha;
    let node = 0;
    this.path[0] = 0;
    for (let d = 0; d < D; d++) {
      const b = ctx(d);
      let c = this.ch[node * 2 + b];
      if (c < 0) {
        if (this.n >= this.c0.length) this.grow();
        c = this.n++;
        this.ch[node * 2 + b] = c;
      }
      node = c; this.path[d + 1] = node;
      this.bits[d] = b;
    }
    for (let d = D; d >= 0; d--) {
      const v = this.path[d];
      const n0 = this.c0[v], n1 = this.c1[v], tot = n0 + n1 + 2 * a;
      const e1 = this.lpe[v] + Math.log((n1 + a) / tot), e0 = this.lpe[v] + Math.log((n0 + a) / tot);
      this.pe1[d] = e1; this.pe0[d] = e0;
      if (d === D) { this.pw1[d] = e1; this.pw0[d] = e0; }
      else {
        const b = this.bits[d];
        const other = this.ch[v * 2 + (1 - b)];
        const lo = other >= 0 ? this.lpw[other] : 0;
        this.pw1[d] = logAddExp(e1, this.pw1[d + 1] + lo) - LN2;
        this.pw0[d] = logAddExp(e0, this.pw0[d + 1] + lo) - LN2;
      }
    }
    const l1 = this.pw1[0], l0 = this.pw0[0];
    this.p = 1 / (1 + Math.exp(l0 - l1));
    return this.p;
  }
  update(x) {
    const g = this.gamma;
    for (let d = 0; d <= this.D; d++) {
      const v = this.path[d];
      this.lpe[v] = x ? this.pe1[d] : this.pe0[d];
      this.lpw[v] = x ? this.pw1[d] : this.pw0[d];
      if (g < 1) { this.c0[v] *= g; this.c1[v] *= g; }
      if (x) this.c1[v]++; else this.c0[v]++;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Online logistic regression with per-feature AdaGrad-ish step sizes and L2 toward 0.
class LogReg {
  constructor(nf, lr, l2, decay = 1) {
    this.decay = decay;
    this.w = new Float64Array(nf); this.g2 = new Float64Array(nf).fill(1);
    this.f = new Float64Array(nf); this.nf = nf; this.lr = lr; this.l2 = l2; this.p = 0.5;
  }
  predict() {
    let z = 0;
    for (let i = 0; i < this.nf; i++) z += this.w[i] * this.f[i];
    this.z = clip(z, -8, 8);
    this.p = squash(this.z);
    return this.p;
  }
  update(y) {
    const e = y - this.p;
    for (let i = 0; i < this.nf; i++) {
      const fi = this.f[i];
      if (fi === 0) { this.w[i] *= 1 - this.l2; continue; }
      const gr = e * fi;
      this.g2[i] = this.decay * this.g2[i] + gr * gr;
      this.w[i] += (this.lr * gr) / Math.sqrt(this.g2[i]) - this.l2 * this.w[i];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Adaptive probability table: context -> probability, learned by count-based rates.
// Change-point tracker for a Bernoulli rate: grid posterior with fixed-share restarts and a
// "mirror" transition theta -> 1 - theta (a player who notices a bias and overcorrects).
class CP {
  constructor(G, share, mirror) {
    this.G = G; this.a = share; this.b = mirror;
    this.th = new Float64Array(G); this.w = new Float64Array(G).fill(1 / G); this.t = new Float64Array(G);
    for (let i = 0; i < G; i++) this.th[i] = (i + 0.5) / G;
    this.p = 0.5;
  }
  predict() { let s = 0; for (let i = 0; i < this.G; i++) s += this.w[i] * this.th[i]; return (this.p = s); }
  update(y) {
    const G = this.G, w = this.w, t = this.t;
    let z = 0;
    for (let i = 0; i < G; i++) { w[i] *= y ? this.th[i] : 1 - this.th[i]; z += w[i]; }
    for (let i = 0; i < G; i++) t[i] = w[i] / z;
    for (let i = 0; i < G; i++) w[i] = (1 - this.a - this.b) * t[i] + this.a / G + this.b * t[G - 1 - i];
  }
}

class APT {
  constructor(n, rateMin = 0.02) { this.p = new Float64Array(n).fill(0.5); this.n = new Float64Array(n); this.rmin = rateMin; }
  get(i) { return this.p[i]; }
  up(i, y) { this.n[i]++; const r = Math.max(1 / (this.n[i] + 1.5), this.rmin); this.p[i] += (y - this.p[i]) * r; }
}

export const DEFAULTS = {
  // CTW experts: [type, depth, KT discount]
  ctws: [['sh', 12, 1], ['raw', 12, 1], ['sw', 10, 1], ['sh', 6, 0.9]],
  cps: [], cpInMix: true,
  betPipe: true, hotReset: false, hotBet: true, slowPool: false, hotMirror: 0.3, spAgree: false, guessFromBet: true, hotShare: 0, slowEta: 1, slowShare: 0.001, hotW: 12, hotThr: 0.67, hotShrink: 0.2, tallyFeat: false, spCP: true, spHaz: true, spTally: true, cpA: 0.01, cpB: 0.01, mainPrior: 0.7, betShare: 0.002,
  useLR: true, lrLr: 0.15, lrL2: 0.002, lrDecay: 1, ngramFeat: true, useMatch: true, usePeriod: true,
  mixLr: 0.02, mixInit: 0.15,                 // logistic mixer (weight set chosen by match length)
  bayes: true, bEta: 0.4, bShare: 0.01, mixPrior: 1, // tempered Bayes / fixed-share pool over experts + mixer
  thetas: [0.7, 0.85, 0.95], antiPrior: 0.02, share: 0.01, halfAnti: 0.5, // g-counter hypotheses
  gCands: [0.5, 0, 1],
  probe: true, probeMax: 40, probeMargin: 0.7, probeThresh: 0.2, probeStart: 100,
};

export function make(cfg = {}) {
  const C = { ...DEFAULTS, ...cfg };
  return function createPredictor({ rng }) {
    const keys = [], hits = [], sw = [], guessesArr = [];
    const t = () => keys.length;
    const X = (i) => { const n = keys.length; return n > i ? 2 * keys[n - 1 - i] - 1 : 0; }; // ±1, 0 if missing
    const K = (i) => { const n = keys.length; return n > i ? keys[n - 1 - i] : 0; };
    const S = (i) => { const n = sw.length; return n > i ? sw[n - 1 - i] : 0; };
    const Hh = (i) => { const n = hits.length; return n > i ? hits[n - 1 - i] : 0; };

    // CTW experts: [type, depth, gamma]; types: sh = switch target, ctx [s1,h1,s2,h2..];
    // sw = switch target, ctx switches; raw = key target, ctx keys; rh = key target, ctx [x1,h1,x2,h2..]
    const ctws = C.ctws.map(([ty, d, gm, al]) => ({ ty, m: new CTW(d, gm, al ?? 0.5) }));
    const NF = 56;
    const lr = new LogReg(NF, C.lrLr, C.lrL2, C.lrDecay);

    // match model: longest match of recent suffix, predicted bit, confidence by length bucket
    const matchT = new APT(16);
    let matchPred = -1, matchLenB = 0;

    // periodic expert: for periods P = 2..16, table ctx (x_{t-P}, x_{t-2P}) per period;
    // track discounted log-loss per period, output best-period prediction.
    const PMIN = 2, PMAX = 16;
    const perT = []; const perLoss = new Float64Array(PMAX + 1); const perP = new Float64Array(PMAX + 1);
    for (let P = 0; P <= PMAX; P++) perT.push(new APT(4 * 32, 0.05));
    const perCtx = new Int32Array(PMAX + 1);

    // change-point trackers: [target 'key'|'sw', grid, share, mirror]
    const cps = C.cps.map(([ty, G, a, b]) => ({ ty, m: new CP(G, a, b) }));
    // hot-reset experts: low-order switch tables whose counts shrink when the machine gets "hot"
    // (the player is likely to notice and bail out of the current scheme)
    const HR = C.hotReset || C.hotBet ? 3 : 0, HRmix = C.hotReset ? 3 : 0;
    const hr0 = new Float64Array(3 * 16), hr1 = new Float64Array(3 * 16); const hrCtx = new Int32Array(3);
    let hotCool = 0;
    const nIn = ctws.length + 4 + cps.length + HRmix;
    const inp = new Float64Array(nIn + 1);
    const W = [];
    const NSETS = 4;
    for (let s = 0; s < NSETS; s++) W.push(new Float64Array(nIn + 1).fill(C.mixInit));
    let wSet = 0, pMix = 0.5, zMix = 0;

    // response hypotheses: H0 = base model; H1(theta) = player plays against the side of g.
    const THETAS = C.thetas;
    const nA = 1 + THETAS.length, nH = nA;
    const lw = new Float64Array(nH); // log posterior weights
    const hPrior = new Float64Array(nH);
    hPrior[0] = 1 - C.antiPrior;
    for (let i = 1; i < nA; i++) hPrior[i] = C.antiPrior / THETAS.length;
    for (let i = 0; i < nH; i++) lw[i] = Math.log(hPrior[i]);
    const hp = new Float64Array(nH);
    let pFinal = 0.5, gOut = 0.5, probes = 0, lwM = 0, lwF = 0;
    // ---- betting pipeline: Bayes mixture of the guess pipeline's p and detection specialists
    const spKey = new CP(20, C.cpA, C.cpB), spSw = new CP(20, C.cpA, C.cpB), spAg = new CP(20, C.cpA, C.cpB);
    let agSide = -1;
    const hz0 = new Float64Array(64), hz1 = new Float64Array(64); let hzCtx = 0;
    const tallyLR = new LogReg(8, 0.1, 0.001);
    let lwS = null;
    let kb = 0, sbal = 0, kbD = 0, sbD = 0, prevRun = 0;
    const spP = new Float64Array(16); let nSp = 0, lwB = null, pBet = 0.5;
    function hotPreds(out, k) {
      hrCtx[0] = 0; hrCtx[1] = 16 + S(0) * 2 + Hh(0); hrCtx[2] = 32 + S(0) * 2 + S(1);
      for (let j = 0; j < 3; j++) { const a = hr0[hrCtx[j]], b = hr1[hrCtx[j]]; out[k + j] = j === 0 ? (b + 0.5) / (a + b + 1) : toX((b + 0.5) / (a + b + 1)); }
      return k + 3;
    }
    function specialists() {
      const n = keys.length;
      let k = 0;
      if (C.spCP) { spP[k++] = spKey.predict(); spP[k++] = toX(spSw.predict()); }
      if (C.spAgree) {
        // agreement of the key with the base model's side, tracked with (hot-driven) mirror flips
        agSide = pMix > 0.5 ? 1 : pMix < 0.5 ? 0 : -1;
        const th = spAg.predict();
        spP[k++] = agSide < 0 ? 0.5 : agSide ? th : 1 - th;
      }
      if (C.spHaz) {
        let rl = 0; for (let i = n - 1; i >= 0 && keys[i] === keys[n - 1]; i--) rl++;
        hzCtx = Math.min(rl, 8) * 6 + Math.min(prevRun, 5);
        const a = hz0[hzCtx], b = hz1[hzCtx]; // b = switch counts
        spP[k++] = n ? toX((b + 0.5) / (a + b + 1)) : 0.5;
      }
      if (C.spTally) {
        const f = tallyLR.f, x1 = X(0);
        f[0] = 1; f[1] = clip(kb, -6, 6) / 3; f[2] = (clip(sbal, -6, 6) / 3) * -x1;
        f[3] = clip(kbD, -6, 6) / 3; f[4] = (clip(sbD, -6, 6) / 3) * -x1;
        f[5] = Math.abs(kb) >= 2 ? -Math.sign(kb) : 0; f[6] = Math.abs(sbal) >= 2 ? Math.sign(sbal) * x1 : 0; f[7] = x1;
        spP[k++] = tallyLR.predict();
      }
      if (C.hotBet) k = hotPreds(spP, k);
      if (C.slowPool) {
        const ps = lastPs, L = ps.length;
        if (!lwS) { lwS = new Float64Array(L).fill(-Math.log(L)); lwS[L - 1] += C.mixPrior; }
        let m = -Infinity; for (let i = 0; i < L; i++) m = Math.max(m, lwS[i]);
        let z = 0, sp = 0; for (let i = 0; i < L; i++) { const w = Math.exp(lwS[i] - m); z += w; sp += w * ps[i]; }
        spP[k++] = sp / z;
      }
      nSp = k;
    }
    function betMix(pMain) {
      const L = 1 + nSp;
      if (!lwB) { lwB = new Float64Array(L); lwB[0] = Math.log(C.mainPrior); for (let i = 1; i < L; i++) lwB[i] = Math.log((1 - C.mainPrior) / nSp); }
      let m = -Infinity; for (let i = 0; i < L; i++) m = Math.max(m, lwB[i]);
      let z = 0, sp = 0;
      for (let i = 0; i < L; i++) { const w = Math.exp(lwB[i] - m); z += w; sp += w * (i ? spP[i - 1] : pMain); }
      return sp / z;
    }
    function hypPreds(g, out) {
      out[0] = pMix;
      for (let i = 1; i < nA; i++) {
        const th = THETAS[i - 1];
        out[i] = g > 0.5 ? 1 - th : g < 0.5 ? th : C.halfAnti;
      }
    }
    function postMix(g) {
      hypPreds(g, hp);
      let m = -Infinity; for (let i = 0; i < nH; i++) m = Math.max(m, lw[i]);
      let z = 0, s = 0;
      for (let i = 0; i < nH; i++) { const w = Math.exp(lw[i] - m); z += w; s += w * hp[i]; }
      return s / z;
    }
    function features(f) {
      let k = 0;
      f[k++] = 1;
      const x1 = X(0), x2 = X(1), x3 = X(2), x4 = X(3);
      f[k++] = x1; f[k++] = x2; f[k++] = x3; f[k++] = x4;
      f[k++] = x1 * x2; f[k++] = x1 * x3; f[k++] = x2 * x3; f[k++] = x1 * x2 * x3;
      const h1 = t() > 0 ? 2 * Hh(0) - 1 : 0, h2 = t() > 1 ? 2 * Hh(1) - 1 : 0;
      f[k++] = h1 * x1; f[k++] = h1 * x1 * x2; f[k++] = h2 * x1; f[k++] = h1 * h2 * x1; f[k++] = h1;
      // run length one-hot times last key
      let rl = 0;
      for (let i = keys.length - 1; i >= 0 && keys[i] === keys[keys.length - 1]; i--) rl++;
      for (let r = 1; r <= 6; r++) f[k++] = (r < 6 ? rl === r : rl >= 6) ? x1 : 0;
      // window balances
      for (const w of [3, 4, 5, 6, 8, 12, 20, 40]) {
        const m = Math.min(w, keys.length);
        let s = 0;
        for (let i = 0; i < m; i++) s += X(i);
        f[k++] = m ? s / Math.sqrt(w) : 0;
      }
      // switch-rate windows times last key (alternation tendency at scales)
      for (const w of [8, 32]) {
        const m = Math.min(w, sw.length);
        let s = 0;
        for (let i = 0; i < m; i++) s += 2 * S(i) - 1;
        f[k++] = m ? (s / Math.sqrt(w)) * -x1 : 0;
      }
      // last guess sign and last guess*last key
      const gl = guessesArr.length ? 2 * guessesArr[guessesArr.length - 1] - 1 : 0;
      f[k++] = gl; f[k++] = gl * x1;
      // match prediction as a feature
      f[k++] = matchPred < 0 ? 0 : (2 * matchPred - 1) * Math.min(matchLenB, 8) / 8;
      if (C.ngramFeat) for (let o = 1; o <= 6; o++) {
        const n = keys.length;
        if (n < o) { f[k++] = 0; continue; }
        let c = 1; for (let i = n - o; i < n; i++) c = (c << 1) | keys[i];
        const a = ng0[c], b = ng1[c];
        f[k++] = a === b ? 0 : b > a ? 1 : -1;
      }
      if (C.tallyFeat) {
        f[k++] = clip(kb, -6, 6) / 3; f[k++] = (clip(sbal, -6, 6) / 3) * -x1;
        f[k++] = Math.abs(kb) >= 2 ? -Math.sign(kb) : 0; f[k++] = Math.abs(sbal) >= 2 ? Math.sign(sbal) * x1 : 0;
      }
      while (k < NF) f[k++] = 0;
    }
    const ng0 = new Float64Array(256), ng1 = new Float64Array(256);
    function ngUpdate(bit) {
      const n = keys.length;
      for (let o = 1; o <= 6; o++) {
        if (n < o) break;
        let c = 1; for (let i = n - o; i < n; i++) c = (c << 1) | keys[i];
        if (bit) ng1[c]++; else ng0[c]++;
      }
    }

    function matchFind() {
      // longest suffix match (up to 24) ending before the end, most recent occurrence
      const n = keys.length;
      matchPred = -1; matchLenB = 0;
      if (n < 3) return;
      let bestLen = 0, bestPos = -1;
      for (let j = n - 2; j >= 0 && n - j < 300; j--) {
        let L = 0;
        while (L < 24 && j - L >= 0 && keys[j - L] === keys[n - 1 - L]) L++;
        if (L > bestLen) { bestLen = L; bestPos = j + 1; if (L >= 24) break; }
      }
      if (bestLen >= 2) { matchPred = keys[bestPos]; matchLenB = Math.min(15, bestLen); }
    }

    function periodPredict() {
      const n = keys.length;
      let best = -1, bestL = Infinity;
      for (let P = PMIN; P <= PMAX; P++) {
        if (n < P) { perCtx[P] = -1; continue; }
        const a = keys[n - P], b = n >= 2 * P ? keys[n - 2 * P] : 0;
        const c = (n % P) * 4 + a * 2 + b;
        perCtx[P] = c;
        perP[P] = perT[P].get(c);
        if (perLoss[P] < bestL) { bestL = perLoss[P]; best = P; }
      }
      return best > 0 && perCtx[best] >= 0 ? perP[best] : 0.5;
    }

    const toX = (pSwitch) => (keys.length ? (K(0) ? 1 - pSwitch : pSwitch) : 0.5);

    function predict() {
      const shCtx = (i) => (i & 1 ? Hh(i >> 1) : S(i >> 1));
      const rhCtx = (i) => (i & 1 ? Hh(i >> 1) : K(i >> 1));
      const ps = [];
      for (const e of ctws) {
        if (e.ty === 'sh') ps.push(toX(e.m.predict(shCtx)));
        else if (e.ty === 'sw') ps.push(toX(e.m.predict(S)));
        else if (e.ty === 'raw') ps.push(e.m.predict(K));
        else if (e.ty === 'rh') ps.push(e.m.predict(rhCtx));
      }
      matchFind();
      features(lr.f);
      ps.push(C.useLR ? lr.predict() : 0.5);
      ps.push(C.useMatch && matchPred >= 0 ? (matchPred ? matchT.get(matchLenB) : 1 - matchT.get(matchLenB)) : 0.5);
      ps.push(C.usePeriod ? periodPredict() : 0.5);
      for (const e of cps) ps.push(e.ty === 'key' ? e.m.predict() : toX(e.m.predict()));
      if (HRmix) hotPreds(ps, ps.length);
      for (let k = 0; k < ps.length; k++) inp[k] = clip(stretch(clip(ps[k], 1e-4, 1 - 1e-4)), -8, 8);
      inp[ps.length] = 0; inp[nIn] = 0.3; // bias
      if (!C.cpInMix) for (let i = ps.length - cps.length; i < ps.length; i++) inp[i] = 0;
      wSet = Math.min(3, matchLenB >> 2);
      const w = W[wSet];
      let z = 0;
      for (let i = 0; i <= nIn; i++) z += w[i] * inp[i];
      zMix = clip(z, -8, 8);
      pMix = squash(zMix);
      pLog = pMix;
      if (C.bayes) {
        ps.push(pLog);
        if (!lwE) { lwE = new Float64Array(ps.length); lwE.fill(-Math.log(ps.length)); lwE[ps.length - 1] += C.mixPrior; }
        let m = -Infinity; for (let i = 0; i < ps.length; i++) m = Math.max(m, lwE[i]);
        let zz = 0, sp = 0;
        for (let i = 0; i < ps.length; i++) { const w = Math.exp(lwE[i] - m); zz += w; sp += w * ps[i]; }
        pMix = clip(sp / zz, 1e-4, 1 - 1e-4); zMix = stretch(pMix);
      }
      lastPs = ps;

      // choose g to maximize expected hit under the posterior over response hypotheses
      let g = 0.5, best = -1;
      const useBet = C.betPipe && C.guessFromBet;
      if (useBet) specialists();
      for (const c of C.gCands) {
        const p1 = useBet && nSp ? betMix(postMix(c)) : postMix(c);
        const h = c * p1 + (1 - c) * (1 - p1);
        if (h > best + 1e-9) { best = h; g = c; }
      }
      // detection probe: if the player looks like a g-counter and the verdict is not yet HUMAN,
      // spend a few keys exposing a side so the betting score can collect evidence.
      if (C.probe && g === 0.5 && keys.length >= C.probeStart && probes < C.probeMax && Math.max(lwM, lwF) - LN2 < Math.log(20) + C.probeMargin) {
        let pa = 0, m = -Infinity; for (let i = 0; i < nH; i++) m = Math.max(m, lw[i]);
        let z = 0; for (let i = 0; i < nH; i++) { const w = Math.exp(lw[i] - m); z += w; if (i && i < nA) pa += w; }
        if (pa / z > C.probeThresh) { g = zMix >= 0 ? 1 : 0; probes++; }
      }
      gOut = g;
      pFinal = postMix(g);
      pBet = pFinal;
      if (C.betPipe) { if (!useBet) specialists(); if (nSp) pBet = betMix(pFinal); }
      return { p: pBet, g };
    }
    let lastPs = null, lwE = null, pLog = 0.5;

    function update(bit, guess) {
      const n = keys.length;
      const s = n ? (bit ^ keys[n - 1]) : 0;
      for (const e of ctws) e.m.update(e.ty === 'raw' || e.ty === 'rh' ? bit : s);
      for (const e of cps) if (e.ty === 'key' || n) e.m.update(e.ty === 'key' ? bit : s);
      if (HR) {
        if (bit) hr1[hrCtx[0]]++; else hr0[hrCtx[0]]++;
        if (n) for (let j = 1; j < 3; j++) { if (s) hr1[hrCtx[j]]++; else hr0[hrCtx[j]]++; }
        const w = C.hotW, m = hits.length + 1;
        if (--hotCool <= 0 && m >= w) {
          let h = guess === bit ? 1 : 0; for (let i = m - w; i < m - 1; i++) h += hits[i];
          if (h / w >= C.hotThr) { for (let i = 0; i < hr0.length; i++) { hr0[i] *= C.hotShrink; hr1[i] *= C.hotShrink; } hotCool = w; }
        }
      }
      if (C.useLR) lr.update(bit);
      if (matchPred >= 0) matchT.up(matchLenB, matchPred === bit ? 1 : 0);
      if (C.usePeriod) for (let P = PMIN; P <= PMAX; P++) {
        const c = perCtx[P];
        if (c < 0) continue;
        const q = clip(perP[P], 0.02, 0.98);
        perLoss[P] = 0.95 * perLoss[P] - Math.log(bit ? q : 1 - q);
        perT[P].up(c, bit);
      }
      // mixer
      const err = bit - pLog;
      if (C.bayes) {
        const ps = lastPs, L = ps.length;
        let m = -Infinity;
        for (let i = 0; i < L; i++) { lwE[i] += C.bEta * Math.log(clip(bit ? ps[i] : 1 - ps[i], 1e-4, 1)); m = Math.max(m, lwE[i]); }
        let zz = 0; for (let i = 0; i < L; i++) zz += Math.exp(lwE[i] - m);
        const lz = m + Math.log(zz);
        for (let i = 0; i < L; i++) lwE[i] = Math.log((1 - C.bShare) * Math.exp(lwE[i] - lz) + C.bShare / L);
      }
      const w = W[wSet];
      for (let i = 0; i <= nIn; i++) w[i] += C.mixLr * err * inp[i];
      if (C.betPipe && nSp) {
        const L = 1 + nSp; let m = -Infinity;
        for (let i = 0; i < L; i++) { const q = i ? spP[i - 1] : pFinal; lwB[i] += Math.log(clip(bit ? q : 1 - q, 1e-6, 1)); m = Math.max(m, lwB[i]); }
        let z = 0; for (let i = 0; i < L; i++) z += Math.exp(lwB[i] - m);
        const lz = m + Math.log(z);
        for (let i = 0; i < L; i++) { const pr = i ? (1 - C.mainPrior) / nSp : C.mainPrior; lwB[i] = Math.log((1 - C.betShare) * Math.exp(lwB[i] - lz) + C.betShare * pr); }
      }
      if (C.betPipe) {
        if (C.spCP) {
          if (C.hotMirror) {
            const w = Math.min(16, hits.length + 1); let h = guess === bit ? 1 : 0;
            for (let i = hits.length - w + 1; i < hits.length; i++) h += hits[i];
            const hot = w >= 8 ? clip((h / w - 0.55) / 0.25, 0, 1) : 0;
            spKey.b = spSw.b = spAg.b = C.cpB + C.hotMirror * hot;
            spKey.a = spSw.a = C.cpA + C.hotShare * hot;
          }
          spKey.update(bit); if (n) spSw.update(s);
          if (C.spAgree && agSide >= 0) spAg.update(bit === agSide ? 1 : 0);
        }
        if (C.spHaz && n) { if (s) hz1[hzCtx]++; else hz0[hzCtx]++; }
        if (C.spTally) tallyLR.update(bit);
        if (C.slowPool) {
          const ps = lastPs, L = ps.length; let m = -Infinity;
          for (let i = 0; i < L; i++) { lwS[i] += C.slowEta * Math.log(clip(bit ? ps[i] : 1 - ps[i], 1e-4, 1)); m = Math.max(m, lwS[i]); }
          let z = 0; for (let i = 0; i < L; i++) z += Math.exp(lwS[i] - m);
          const lz = m + Math.log(z);
          for (let i = 0; i < L; i++) lwS[i] = Math.log((1 - C.slowShare) * Math.exp(lwS[i] - lz) + C.slowShare / L);
        }
        if (n && s) { let rl = 0; for (let i = n - 1; i >= 0 && keys[i] === keys[n - 1]; i--) rl++; prevRun = rl; }
        kb += bit ? 1 : -1; kbD = 0.95 * kbD + (bit ? 1 : -1);
        if (n) { sbal += s ? 1 : -1; sbD = 0.95 * sbD + (s ? 1 : -1); }
      }
      { const q = clip(pBet, 0.01, 0.99); lwM += Math.log(bit ? 2 * q : 2 * (1 - q)); lwF += Math.log(bit ? 2 * (1 - q) : 2 * q); }
      // posterior update over response hypotheses (+ fixed share)
      hypPreds(gOut, hp);
      let m = -Infinity;
      for (let i = 0; i < nH; i++) { lw[i] += Math.log(clip(bit ? hp[i] : 1 - hp[i], 1e-6, 1)); m = Math.max(m, lw[i]); }
      let z = 0; for (let i = 0; i < nH; i++) z += Math.exp(lw[i] - m);
      const lz = m + Math.log(z), a = C.share;
      for (let i = 0; i < nH; i++) {
        const post = Math.exp(lw[i] - lz);
        lw[i] = Math.log((1 - a) * post + a * hPrior[i]);
      }
      ngUpdate(bit);
      if (n) sw.push(s);
      keys.push(bit); guessesArr.push(guess); hits.push(guess === bit ? 1 : 0);
    }

    return { predict, update };
  };
}

export default make();
