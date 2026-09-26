// crypto.getRandomValues-backed uniforms in [0, 1). Each call returns an independent stream.
export function createCryptoRng(bufSize = 256) {
  const buf = new Uint32Array(bufSize);
  let i = buf.length;
  return function rng() {
    if (i === buf.length) { globalThis.crypto.getRandomValues(buf); i = 0; }
    return buf[i++] / 4294967296;
  };
}
