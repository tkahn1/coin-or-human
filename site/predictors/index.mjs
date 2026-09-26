// The three adversaries. Each module: export default function createPredictor({ rng }) -> { predict, update }.
import ensemble from './ensemble.mjs';
import learner from './learner.mjs';
import ctw from './ctw.mjs';

export const OPPONENTS = [
  {
    id: 'ensemble', name: 'Adversary 1', create: ensemble,
    desc: 'About 110 simple models, each predicting your next key from one feature: your last few keys, your '
      + 'switch/stay history, run length, F/D balance over the last few keys, or whether its last guess was right. '
      + 'A few look for repeated chunks or fixed-length patterns like counting in binary, and some assume you change '
      + 'style when it starts winning. Their predictions are combined with weights that track each model\'s recent '
      + 'accuracy, so it shifts quickly when you change strategy.',
  },
  {
    id: 'learner', name: 'Adversary 2', create: learner,
    desc: 'Online logistic regression on about 50 features of your recent keys: the last few presses, run length, '
      + 'F/D balance over several window sizes, and whether its recent guesses were right. This is mixed with '
      + 'context-tree models of your key and switch sequences, a longest-match lookup and a period detector. Separate '
      + 'change-point trackers pick up shifts in how often you press F or switch keys, including overcorrecting after '
      + 'favoring one key.',
  },
  {
    id: 'ctw', name: 'Adversary 3', create: ctw,
    desc: 'Context tree weighting: predicts from your last 1 to 12 keys and switches, averaging over all history '
      + 'lengths, so short patterns count early and longer ones take over later. Extra models cover run lengths, '
      + 'running F/D and switch/stay tallies, repeating periods, and keys that copy the one a few steps back. It also '
      + 'models how you react to the guess it shows, and stops showing confident guesses if you seem to press the '
      + 'opposite.',
  },
];
