import { OPPONENTS } from './predictors/index.mjs';
import { createGame, KEY_TO_BIT, BIT_TO_KEY } from './game.mjs';
import { createCryptoRng } from './rng.mjs';

const $ = (id) => document.getElementById(id);

const GOAL = 400;

let opponent = OPPONENTS[0];
try {
  const saved = localStorage.getItem('opponent');
  opponent = OPPONENTS.find((o) => o.id === saved) ?? opponent;
} catch {}

let game;
let caughtAt = null; // key number at which the verdict became HUMAN

function newGame() {
  $('error').hidden = true;
  try {
    // Separate crypto streams: one for the predictor, one (secret) for the displayed-guess draw.
    game = createGame({ createPredictor: opponent.create, predRng: createCryptoRng(), drawRng: createCryptoRng() });
  } catch (e) {
    showError(e);
    game = null;
  }
  caughtAt = null;
  $('history').replaceChildren();
  render(game ? game.stats() : null);
}

function showError(e) {
  console.error(e);
  $('error').textContent = `Predictor error: ${e.message}. Press Reset to try again.`;
  $('error').hidden = false;
}

function render(s) {
  if (!s) return;
  const { n, accuracy, verdict } = s;
  const human = verdict === 'HUMAN';
  const v = $('verdict');
  v.dataset.state = human ? 'human' : 'random';
  v.innerHTML = human ? 'human' : 'random<span class="soft"> so far</span>';
  if (human && caughtAt == null) caughtAt = n;
  $('n').innerHTML = `${n}<span class="of"> / ${GOAL}</span>`;
  $('result').textContent = human ? `Detected at key ${caughtAt}.`
    : n >= GOAL ? `${GOAL} keys undetected. You pass as a coin.` : '';
  $('result').dataset.state = human ? 'human' : n >= GOAL ? 'pass' : '';
  $('acc').textContent = n ? `${Math.round(100 * accuracy)}%` : '–';
}

function press(key) {
  if (!game) return;
  const bit = KEY_TO_BIT[key];
  let r;
  try { r = game.press(bit); } catch (e) { showError(e); game = null; return; }

  const cell = document.createElement('b');
  if (r.hit) cell.className = 'hit';
  cell.title = `#${r.n}: you ${key.toUpperCase()}, machine guessed ${BIT_TO_KEY[r.guess].toUpperCase()}`;
  const hist = $('history');
  const atBottom = hist.scrollTop + hist.clientHeight >= hist.scrollHeight - 4;
  hist.append(cell);
  if (atBottom) hist.scrollTop = hist.scrollHeight;
  render(r);

  const btn = $(`btn-${key}`);
  btn.classList.add('pressed');
  setTimeout(() => btn.classList.remove('pressed'), 90);
}

document.addEventListener('keydown', (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
  const k = e.key.toLowerCase();
  if (k !== 'f' && k !== 'd') return;
  e.preventDefault();
  press(k);
});
$('btn-f').addEventListener('click', () => press('f'));
$('btn-d').addEventListener('click', () => press('d'));

// Auto-play with true randomness, to show that coin flips are (almost always) called random.
const coinRng = createCryptoRng();
let autoTimer = null;
function setAuto(on) {
  clearInterval(autoTimer);
  autoTimer = on ? setInterval(() => press(coinRng() < 0.5 ? 'f' : 'd'), 50) : null;
  const b = $('auto');
  b.textContent = on ? 'Stop flipping' : 'Flip coins';
  b.setAttribute('aria-pressed', String(on));
}
$('auto').addEventListener('click', (e) => { setAuto(!autoTimer); e.currentTarget.blur(); });

function renderOpponents() {
  for (const b of $('opponents').querySelectorAll('button')) {
    b.setAttribute('aria-checked', String(b.dataset.id === opponent.id));
  }
}
for (const o of OPPONENTS) {
  const b = Object.assign(document.createElement('button'), { type: 'button', className: 'pill', textContent: o.name });
  b.dataset.id = o.id;
  b.setAttribute('role', 'radio');
  b.addEventListener('click', (e) => {
    e.currentTarget.blur();
    if (o.id === opponent.id) return;
    opponent = o;
    try { localStorage.setItem('opponent', o.id); } catch {}
    renderOpponents();
    setAuto(false);
    newGame();
  });
  $('opponents').append(b);
}
renderOpponents();

for (const o of OPPONENTS) {
  $('adversaries').append(
    Object.assign(document.createElement('dt'), { textContent: o.name }),
    Object.assign(document.createElement('dd'), { textContent: o.desc }),
  );
}

$('reset').addEventListener('click', (e) => { setAuto(false); newGame(); e.currentTarget.blur(); });

$('how-open').addEventListener('click', (e) => {
  const open = $('how').hidden;
  $('how').hidden = !open;
  e.currentTarget.setAttribute('aria-expanded', String(open));
  e.currentTarget.blur();
  if (open) $('how').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

newGame();
