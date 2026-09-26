// Game loop. Order of operations:
//   predict() -> draw displayed guess from g (secret stream) -> [key arrives] -> hits, score
//   -> update(bit, guess) -> predict() for the next key.
// Key mapping: "f" = 1, "d" = 0.
import { createScore, verdict } from './score.mjs';

export const KEY_TO_BIT = { f: 1, d: 0 };
export const BIT_TO_KEY = ['d', 'f'];

export function createGame({ createPredictor, predRng, drawRng }) {
  const pred = createPredictor({ rng: predRng });
  const score = createScore();
  const history = []; // { bit, guess, hit }
  let hits = 0;
  let pending = commit();

  // Commit p and the displayed guess before the key is known. The UI never reads this.
  function commit() {
    const out = pred.predict();
    const p = Number(out.p), g = Number(out.g);
    if (!(p >= 0 && p <= 1)) throw new Error(`predictor returned bad p=${out.p}`);
    if (!(g >= 0 && g <= 1)) throw new Error(`predictor returned bad g=${out.g}`);
    const guess = drawRng() < g ? 1 : 0;
    return { p, g, guess };
  }

  function stats() {
    const s = score.state();
    return {
      n: history.length, hits, accuracy: history.length ? hits / history.length : null,
      logW: s.logW, peak: s.peak, verdict: verdict(s.peak), history,
    };
  }

  return {
    press(bit) {
      if (bit !== 0 && bit !== 1) throw new Error('bit must be 0 or 1');
      const { p, guess } = pending;
      const hit = guess === bit;
      hits += hit;
      score.update(p, bit);
      history.push({ bit, guess, hit });
      pred.update(bit, guess);
      pending = commit();
      return { bit, guess, hit, ...stats() };
    },
    stats,
  };
}
