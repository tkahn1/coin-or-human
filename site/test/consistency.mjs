// Headless checks: node site/test/consistency.mjs
// 1. score.mjs reproduces the reference betting-score formulas bit-for-bit on random sequences.
// 2. game.mjs reproduces the referee's loop (predict -> draw guess -> key -> hits/score -> update).
// 3. Null check: 2000 fair-coin sessions x 300 keys through the site's own game loop and predictor
//    -> false HUMAN rate <= ~5%, accuracy ~ 50%.
import assert from 'node:assert/strict';
import { createScore, LOG_THRESH } from '../score.mjs';
import { createGame } from '../game.mjs';
import { createCryptoRng } from '../rng.mjs';
import createPredictor from '../predictor.mjs';

// Seeded PRNG for reproducible comparisons (mulberry32).
function seeded(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Reference formulas (independent transcription) ---
const LN2 = Math.log(2), THRESH = Math.log(20);
function logAdd(a, b) { const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); }
function refereeScore(ps, xs) {
  let lwM = 0, lwF = 0, peak = 0;
  for (let t = 0; t < xs.length; t++) {
    const p = ps[t], x = xs[t];
    const q = Math.min(0.99, Math.max(0.01, p));
    lwM += Math.log(x ? 2 * q : 2 * (1 - q));
    lwF += Math.log(x ? 2 * (1 - q) : 2 * q);
    peak = Math.max(peak, logAdd(lwM, lwF) - LN2);
  }
  return { lwM, lwF, peak, det: peak >= THRESH };
}
function refereeSession(pred, drawRng, keyFn, n) {
  const keys = [], guesses = [];
  let hits = 0, lwM = 0, lwF = 0, peak = 0;
  for (let t = 0; t < n; t++) {
    const out = pred.predict();
    const p = Number(out.p), g = Number(out.g);
    const guess = drawRng() < g ? 1 : 0;
    const q = Math.min(0.99, Math.max(0.01, p));
    const x = keyFn(t);
    hits += guess === x;
    lwM += Math.log(x ? 2 * q : 2 * (1 - q));
    lwF += Math.log(x ? 2 * (1 - q) : 2 * q);
    peak = Math.max(peak, logAdd(lwM, lwF) - LN2);
    keys.push(x); guesses.push(guess);
    pred.update(x, guess);
  }
  return { hits, peak, guesses };
}

// 1. Score formulas on random sequences, including extreme and out-of-clip p values.
{
  assert.equal(LOG_THRESH, THRESH);
  const r = seeded(12345);
  let humans = 0;
  for (let trial = 0; trial < 500; trial++) {
    const n = 1 + Math.floor(r() * 400);
    const bias = r();
    const ps = [], xs = [];
    for (let t = 0; t < n; t++) {
      const u = r();
      ps.push(u < 0.05 ? 0 : u < 0.1 ? 1 : u < 0.15 ? 0.5 : r());
      xs.push(r() < bias ? 1 : 0);
    }
    const ref = refereeScore(ps, xs);
    const s = createScore();
    let st;
    for (let t = 0; t < n; t++) st = s.update(ps[t], xs[t]);
    assert.equal(st.lwM, ref.lwM); assert.equal(st.lwF, ref.lwF);
    assert.equal(st.peak, ref.peak); assert.equal(st.human, ref.det);
    humans += ref.det;
  }
  console.log(`score.mjs matches run.mjs formulas exactly on 500 random sequences (${humans} reached W>=20)`);
}

// 2. Game loop vs referee loop with identical seeded streams.
{
  for (let s = 0; s < 200; s++) {
    const keyRng = seeded(s * 7 + 1), keys = Array.from({ length: 300 }, () => (keyRng() < 0.6 ? 1 : 0));
    const ref = refereeSession(createPredictor({ rng: seeded(s * 13 + 5) }), seeded(s ^ 0x2545f491), (t) => keys[t], 300);
    const game = createGame({ createPredictor, predRng: seeded(s * 13 + 5), drawRng: seeded(s ^ 0x2545f491) });
    let st;
    for (const k of keys) st = game.press(k);
    assert.equal(st.hits, ref.hits); assert.equal(st.peak, ref.peak);
    assert.deepEqual(st.history.map((h) => h.guess), ref.guesses);
  }
  console.log('game.mjs loop matches referee loop (hits, guesses, peak) on 200 seeded sessions');
}

// 3. Null check with the site's crypto RNGs and game loop.
{
  const SESSIONS = 2000, N = 300;
  const coin = createCryptoRng();
  let fp = 0, hits = 0;
  for (let s = 0; s < SESSIONS; s++) {
    const game = createGame({ createPredictor, predRng: createCryptoRng(), drawRng: createCryptoRng() });
    let st;
    for (let t = 0; t < N; t++) st = game.press(coin() < 0.5 ? 1 : 0);
    fp += st.verdict === 'HUMAN';
    hits += st.hits;
  }
  const fpr = fp / SESSIONS, acc = hits / (SESSIONS * N);
  const z = (hits - (SESSIONS * N) / 2) / Math.sqrt((SESSIONS * N) / 4);
  console.log(`null: ${SESSIONS} fair-coin sessions x ${N} keys: false HUMAN rate = ${(100 * fpr).toFixed(2)}%, accuracy = ${(100 * acc).toFixed(2)}% (z=${z.toFixed(2)})`);
  // Ville bound is 5%; allow ~3 sigma of binomial noise at n=2000 (sd ~0.49%).
  assert.ok(fpr <= 0.05 + 3 * Math.sqrt(0.05 * 0.95 / SESSIONS), `false HUMAN rate too high: ${fpr}`);
  assert.ok(Math.abs(z) < 4, `null accuracy off 50%: ${acc}`);
}
console.log('ALL OK');
