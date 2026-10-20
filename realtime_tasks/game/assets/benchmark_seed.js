// Seeded randomness for every browser-game document.
//
// The game runtime registers this file with Page.addScriptToEvaluateOnNewDocument, replacing the
// placeholder below with the task seed, so it runs in each document and same-origin frame (and
// again after every reload) before any game script. Games therefore never ship their own PRNG:
// Math.random, crypto.getRandomValues/randomUUID and argument-less seedrandom() all draw from
// mulberry32 streams keyed by the seed, restarting from it in each new document.
//
// Published API (the gameAPI observation shims call the __*DeterministicSeed/__resetRandom names):
//   window.__benchmarkSeed               seed driving this document; the runtime verifies it
//   window.__setDeterministicSeed(seed)  restart all streams from seed and publish it
//   window.__resetRandom(seed?)          same; defaults to the current seed
//   window.__getDeterministicSeed()      current seed
//   window.__restartRandomStream(state)  restart Math.random from a derived state, keeping the
//                                        published seed (e.g. one stream per level)
(function (initialSeed) {
  "use strict";
  const CRYPTO_STREAM = 0x9e3779b9;

  function readSeed(value) {
    const number = Number(value);
    if (!Number.isInteger(number) || number < 0 || number > 0xffffffff) {
      throw new Error("benchmark seed must be an unsigned 32-bit integer");
    }
    return number;
  }

  function mulberry32(state) {
    state >>>= 0;
    return function () {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  let seed = readSeed(initialSeed);
  let random = mulberry32(seed);
  let cryptoRandom = mulberry32(seed ^ CRYPTO_STREAM);

  function applySeed(value) {
    seed = readSeed(value);
    random = mulberry32(seed);
    cryptoRandom = mulberry32(seed ^ CRYPTO_STREAM);
    window.__benchmarkSeed = seed;
    return seed;
  }

  function fillRandomBytes(array) {
    if (!array || typeof array.length !== "number") throw new TypeError("Expected typed array");
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (cryptoRandom() * 256) | 0;
    return array;
  }

  function randomUUID() {
    const bytes = fillRandomBytes(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  Math.random = function () { return random(); };
  if (window.crypto) {
    try { window.crypto.getRandomValues = fillRandomBytes; } catch (_) {}
    try { window.crypto.randomUUID = randomUUID; } catch (_) {}
  }

  // seedrandom() called without a seed would otherwise mix in time and native entropy.
  function wrapSeedrandom(fn) {
    if (typeof fn !== "function") return fn;
    const wrapped = function (value, options) {
      return fn.call(this, value === undefined || value === null || value === "" ? String(seed) : value, options);
    };
    Object.assign(wrapped, fn);
    return wrapped;
  }
  for (const target of [Math, window]) {
    let current = wrapSeedrandom(target.seedrandom);
    try {
      Object.defineProperty(target, "seedrandom", {
        configurable: true,
        get() { return current; },
        set(fn) { current = wrapSeedrandom(fn); }
      });
    } catch (_) {}
  }

  window.__benchmarkSeed = seed;
  window.__setDeterministicSeed = applySeed;
  window.__resetRandom = (value) => applySeed(value === undefined ? seed : value);
  window.__getDeterministicSeed = () => seed;
  window.__restartRandomStream = (state) => { random = mulberry32(readSeed(state)); };
}(__BENCHMARK_SEED__));
