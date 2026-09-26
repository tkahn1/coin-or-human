# Coin or Human?

Press F or D as randomly as you can. The machine guesses each key before you press it. Can you reach 400 keys without being called human?

Inspired by [aaronson-oracle](https://github.com/elsehow/aaronson-oracle). How it works: [ALGORITHM.md](ALGORITHM.md).

## Run

```sh
python3 -m http.server --directory site
```

## Test

```sh
node site/test/consistency.mjs
```

## Deploy

Settings → Pages → Source: GitHub Actions, then push to `main`.
