// Betting score (e-process) and verdict logic. Pure. See ALGORITHM.md.
//
// W = (W_q + W_{1-q}) / 2, q = clip(p, 0.01, 0.99), where W_q bets that the key follows p and
// W_{1-q} bets the opposite. Under a fair coin E[W] stays 1, so by Ville's inequality
// P(sup_t W_t >= 20) <= 1/20 = 5%. Verdict: HUMAN iff the peak ever reaches 20.

export const LN2 = Math.log(2);
export const THRESHOLD_W = 20;
export const LOG_THRESH = Math.log(THRESHOLD_W);

export const clipP = (p) => Math.min(0.99, Math.max(0.01, p));

export function logAdd(a, b) {
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

export function createScore() {
  let lwM = 0, lwF = 0, logW = 0, peak = 0;
  const state = () => ({ lwM, lwF, logW, peak, human: peak >= LOG_THRESH });
  return {
    // p = predictor's committed P(bit = 1); bit = actual key (1 = f, 0 = d).
    update(p, bit) {
      const q = clipP(p);
      lwM += Math.log(bit ? 2 * q : 2 * (1 - q));
      lwF += Math.log(bit ? 2 * (1 - q) : 2 * q);
      logW = logAdd(lwM, lwF) - LN2;
      peak = Math.max(peak, logW);
      return state();
    },
    state,
  };
}

// Once HUMAN, always HUMAN (peak is monotone).
export const verdict = (peakLogW) => (peakLogW >= LOG_THRESH ? 'HUMAN' : 'RANDOM');
