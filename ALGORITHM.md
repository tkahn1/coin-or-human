# Algorithm

Keys are bits: f = 1, d = 0. Code: `site/predictor.mjs` (predictor), `site/game.mjs` (loop), `site/score.mjs` (verdict).

## Game loop

For each key:

1. The predictor outputs two numbers from the past keys only:
   - `p`: probability the next key is f, used for the verdict.
   - `g`: probability of guessing f. The displayed guess is drawn from `g` with `crypto.getRandomValues`, before the key is pressed.
2. The key arrives. The guess is scored and `p` is used to update the verdict.
3. The predictor is updated with the key and the guess that was shown.

Because the guess is fixed before the key, a true coin is guessed right exactly 50% of the time on average.

## Predictors

111 small predictors, each giving P(next key = f):

- **Count tables (75).** A table of counts per context, estimate (n₁ + 0.4) / (n + 0.8). Contexts:
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
- **Match (6).** For L = 3, 5, 8, 12, 16, 24: find the last time the most recent L keys occurred and predict what followed, weighted by how often that has been right.
- **Frequency-oracle agreement (12).** For k = 1–6: the key a simple order-k majority-count predictor would guess, and how often the player has matched it. Catches players who play along with or against a simple predictor. Plain and decayed (0.9) versions.
- **Style-flip models (10).** A hidden switch rate on a grid 0.1–0.9. After each key it may flip (θ → 1 − θ) or reset, with a probability that rises with the machine's recent hit rate (window 10 or 25). Catches players who change style when they notice they're being predicted.
- **Context tree weighting (3).** Depth 10 on interleaved (switch, hit) bits; depth 12 on keys; depth 8 on switch bits. KT estimators.
- **Phase CTW (5).** For L = 4–8: context is the key position mod L plus the previous L keys. Catches fixed-length patterns such as counting in binary.

A logistic mixer combines all 111. Two learning rates, 0.02 and 0.006. The weight set is chosen by (last switch, last hit).

## Guess (`g`)

1. Mixture over the 111 predictors plus the logistic mixer. Bayesian weights on log-loss, tempered (η = 0.5), fixed-share α = 0.01. Its output picks the side: f if above 0.5, d if below.
2. Confidence level: `g = 0.5 ± m`, m ∈ {0.005, 0.15, 0.3, 0.5}. For each level it tracks, with decay 0.995, how often the key matched the side shown at that level.
   - A level is dropped if the player matches it significantly less than half the time, meaning they are countering the guess.
   - The level steps back up once matching looks normal.
   - Against ordinary players this ends at m = 0.5, a plain best guess.
   - Against players who counter the guess it ends near 0.5, which holds them to about 50%.
3. Probe: from key 60, if a level has been dropped and the verdict is not yet human, that level is shown for up to 40 keys. This lets `p` detect the countering.

## Probability (`p`)

Fixed-share Bayesian mixture on log-loss (η = 1, α = 0.5 / (t + 2)) over:

- the 111 predictors and the logistic mixer
- a constant 0.5
- the guess mixture from step 1 above, raw and after online logistic calibration
- an untempered version of that mixture
- a counter model: how often the key has matched the shown side at the current level

Output is clipped to [0.01, 0.99].

## Verdict

With q = clip(p, 0.01, 0.99), after each key x:

```
W₊ ← W₊ · 2q      if x = f,  W₊ · 2(1 − q)  if x = d
W₋ ← W₋ · 2(1 − q) if x = f,  W₋ · 2q        if x = d
W  = (W₊ + W₋) / 2          (W₊ = W₋ = 1 at the start)
```

Human if W ever reaches 20; otherwise random so far. Once human, it stays human.

If the keys are fair coin flips, each factor has expected value 1 whatever p is. W is then a nonnegative martingale starting at 1, and by Ville's inequality P(W ever ≥ 20) ≤ 1/20. So true randomness is called human at most 5% of the time, however long the game runs. W₋ bets against the predictor, so players who are predictably wrong are also caught.

## Measured

Simulated players, 150 sessions each; figures are accuracy / % called human by 400 keys.

| | This algorithm | 5-gram baseline (aaronson-oracle) |
|---|---|---|
| Fair coin (false human rate) | 50.0% / 2% | 50.0% / 4% |
| 11 simulated human types | 57% / 81% | 52% / 49% |
| 12 held-out informed strategies | 63% / 90% | 57% / 59% |

All test players are simulated. Players reciting memorized digits (e.g. of π) are not detectable by any method.
