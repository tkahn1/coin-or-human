# Algorithm

Keys are bits: f = 1, d = 0. There are three adversaries to choose from. They share the game loop and the verdict, and differ in how they predict.

Code: `site/predictors/` (the three adversaries), `site/game.mjs` (loop), `site/score.mjs` (verdict).

## Game loop

For each key:

1. The adversary outputs two numbers from the past keys only:
   - `p`: probability the next key is f, used for the verdict.
   - `g`: probability of guessing f. The displayed guess is drawn from `g` with `crypto.getRandomValues`, before the key is pressed.
2. The key arrives. The guess is scored and `p` is used to update the verdict.
3. The adversary is updated with the key and the guess that was shown.

Because the guess is fixed before the key, a true coin is guessed right exactly 50% of the time on average.

All three adversaries also defend against players who try to press the opposite of the guess. Each can lower its confidence (`g` near 0.5) against such a player, holding them to about 50%. Each also runs a probe: after a point, if countering is suspected and the verdict is not yet human, it shows confident guesses for up to 40 keys so `p` can collect evidence of the countering.

## Verdict

With q = clip(p, 0.01, 0.99), after each key x:

```
W₊ ← W₊ · 2q      if x = f,  W₊ · 2(1 − q)  if x = d
W₋ ← W₋ · 2(1 − q) if x = f,  W₋ · 2q        if x = d
W  = (W₊ + W₋) / 2          (W₊ = W₋ = 1 at the start)
```

Human if W ever reaches 20; otherwise random so far. Once human, it stays human.

If the keys are fair coin flips, each factor has expected value 1 whatever p is. W is then a nonnegative martingale starting at 1, and by Ville's inequality P(W ever ≥ 20) ≤ 1/20. So true randomness is called human at most 5% of the time, however long the game runs, for any adversary. W₋ bets against the adversary, so players who are predictably wrong are also caught.

## Adversary 1 (ensemble)

**Predictors.** 111 small predictors, each giving P(next key = f):

- **Count tables (75).** Counts per context, estimate (n₁ + 0.4) / (n + 0.8). Contexts:
  - last k keys, k = 0–6
  - last k switch/stay bits, k = 1–6
  - last k (switch, guess was right) pairs, k = 1–4
  - last k (key, guess was right) pairs, k = 1–3
  - number of f in the last w keys, w = 4–8
  - bucketed f/d balance over the last 12, 16, 24 keys
  - whether the last key equaled the key L back, L = 2–12
  - current run length, with and without the key
  - guess hit rate over the last 10 and 20 keys, with the last switch

  Each table exists twice: plain counts, and counts decayed by 0.9 per use (0.95 for the L-back tables).
- **Match (6).** For L = 3, 5, 8, 12, 16, 24: what followed the last occurrence of the most recent L keys, weighted by how often that has been right.
- **Frequency-oracle agreement (12).** For k = 1–6: the key a simple order-k majority-count predictor would guess, and how often the player matches it. Plain and decayed (0.9).
- **Style-flip models (10).** A hidden switch rate on a grid 0.1–0.9 that may flip (θ → 1 − θ) or reset after each key, more likely when the machine's recent hit rate (window 10 or 25) is high.
- **Context tree weighting (3).** Depth 10 on interleaved (switch, hit) bits; depth 12 on keys; depth 8 on switch bits. KT estimators.
- **Phase CTW (5).** For L = 4–8: context is the key position mod L plus the previous L keys.

A logistic mixer combines all 111. It has two learning rates, 0.02 and 0.006, and picks its weight set by (last switch, last hit).

**Guess.** A tempered (η = 0.5), fixed-share (α = 0.01) Bayesian mixture over the predictors and the mixer picks the side. Confidence is `g = 0.5 ± m`, m ∈ {0.005, 0.15, 0.3, 0.5}:

- For each level, it tracks how often the key matched the side shown (decay 0.995).
- A level is dropped when the player matches it significantly less than half the time.
- It climbs back one level at a time.
- The probe starts at key 60.

**Probability.** A fixed-share Bayesian mixture (η = 1, α = 0.5 / (t + 2)) over the predictors, the mixer, a constant 0.5, the guess mixture (raw, calibrated and untempered), and a counter model: how often the key matches the shown side at the current level.

## Adversary 2 (learner)

**Predictors:**

- **Context tree weighting (4).** Depth 12 on interleaved (switch, hit) bits; depth 12 on keys; depth 10 on switch bits; depth 6 on (switch, hit) with counts decayed by 0.9.
- **Online logistic regression (1).** AdaGrad-style steps, L2 0.002. Features:
  - last 4 keys and their products
  - guess-was-right × key interactions
  - run length × last key
  - f/d balance over the last 3–40 keys
  - switch rate over the last 8 and 32 keys
  - last guess
  - the match prediction
  - order 1–6 majority votes
- **Match (1).** The longest match (up to 24 keys) of the recent keys in the last 300. Its confidence is learned per match length.
- **Period (1).** For P = 2–16, a table on (t mod P, key P back, key 2P back). Uses the period with the lowest recent log-loss.

A logistic mixer (4 weight sets, by match length) combines these. Then a tempered (η = 0.4), fixed-share (0.01) Bayesian mixture combines the predictors and the mixer.

**Guess.** Two kinds of hypothesis are compared:

- the model above;
- "the player plays against the shown side" with probability 0.7, 0.85 or 0.95, with a prior of 2% in total.

`g` is chosen from {0, 0.5, 1} to maximize expected hits under this posterior. The probe starts at key 100.

**Probability.** A Bayesian mixture (fixed share 0.002) of the guess model's prediction (prior 0.7) and detection specialists:

- change-point trackers on the key rate and on the switch rate; they can jump to the mirror rate (θ → 1 − θ), more often when the machine is on a hit streak
- a run-length table: P(switch | current run length, previous run length)
- a tally regression on the session's f/d balance and switch/stay balance
- low-order switch tables whose counts shrink ×0.2 when 9 or more of the last 12 guesses hit

## Adversary 3 (context tree)

**Predictors (17).** Families of similar predictors are grouped into fixed-share sub-mixtures:

- **Context tree weighting on switch bits.** Context is interleaved (switch, hit) bits: depth 12, plus depth 6 with count decay 0.95.
- **Context tree weighting on keys.** Depth 12, plus depth 4 with decay 0.95.
- **Switch-rate trackers (5).** Per context of the last 0 or 2 (switch, hit) bits, a posterior over switch rates 0.03–0.97 with restarts. Some can also jump to the mirror rate. Two restart and mirror much faster when 65% or more of the last 12 guesses hit.
- **Lag models.** For L = 2–10: depth-6 CTW on whether each key equals the key L back.
- **Frequency-oracle agreement.** k = 1–6, as in Adversary 1.
- **Bayesian probit regression.** Features:
  - last 3 switches
  - last hit, and hit × switch
  - run length
  - f/d balance over 6 and 20 keys
  - last key
  - session f/d and switch/stay tallies, with threshold indicators
- **Run-length tables.** Current run length, alone and with the previous run length or the key.
- **Tally tables.** Session and 30-key f/d and switch/stay balances.
- **Key-bias tracker.** A posterior over P(f) with restarts and mirror jumps.
- **Phase tables.** For P = 2–16: (t mod P, key P back, key 2P back).
- **Window counts.** Number of f in the last 4–8 keys, alone and with the last key.

**Combination.** Two copies run side by side, with the same structure but different forgetting:

1. A fixed-share mixture over the 17.
2. Temperature calibration, a mixture over scales 0.4–1.5.
3. A response model. Under hypothesis (θ, τ), the player's log-odds of pressing the shown side shift by θ ∈ {±0.5, …, ±4} whenever the shown confidence level is at least τ. The prior on θ = 0 is 0.7.

The decision copy forgets fast (α = 0.005). The betting copy forgets slowly (α = 0.5 / (t + 2), η = 0.5).

**Guess.** Over both sides and levels m ∈ {0.005, 0.1, 0.25, 0.5}, it picks the one with the highest expected accuracy, 0.5 + m (2 P(key = side) − 1). Countering hypotheses are down-weighted by 0.3 in this choice, so it needs more evidence before giving up confident guesses. The probe starts at key 100.

**Probability.** The betting copy's prediction at the chosen guess.

## Measured

Simulated players, 150 sessions each. Figures are accuracy / % called human by 400 keys.

| | Adversary 1 | Adversary 2 | Adversary 3 | 5-gram baseline (aaronson-oracle) |
|---|---|---|---|---|
| Fair coin | 50% / 2% | 50% / 1.5% | 50% / 1.5% | 50% / 4% |
| 11 simulated human types | 57% / 81% | 57% / 80% | 58% / 79% | 52% / 49% |
| 12 held-out informed strategies | 63% / 90% | 63% / 86% | 62% / 81% | 57% / 59% |

The three are about equal at guessing; Adversary 1 is somewhat quicker to call humans. All test players are simulated. Players reciting memorized digits (e.g. of π) are not detectable by any of them.
