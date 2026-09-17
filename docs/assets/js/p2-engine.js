/* =============================================================================================
   p2-engine.js — GPT forward pass with a KV cache, in JavaScript.

   WHY A HAND-WRITTEN ENGINE AGAIN
   ------------------------------
   Same reason as Project 1 (decision D-003): the demo has to expose internals — per-step
   distributions, what top-k actually removed, attention per head — and a black-box runtime gives
   outputs without them. It is checked against PyTorch on load; see `verifyParity`.

   WHY THE KV CACHE IS NOT OPTIONAL HERE
   ------------------------------------
   Project 1 generated 12 tokens over a 30-character context and a naive re-encode was fine. This
   model has 6 layers, d_model 192, and a 192-token context, and a demo generates hundreds of tokens.

   Per new token, re-encoding the whole context costs roughly
       6 layers x (12 x T x d^2  +  T^2 x d)  ≈  550 M multiply-accumulates at T = 192.
   With a cache, the keys and values of positions already generated are reused — they depend only on
   their own input, which has not changed — leaving
       6 layers x (12 x d^2  +  2 x T x d)    ≈  3 M multiply-accumulates.

   That is a ~180x reduction, and it is the difference between roughly 2.7 seconds per token and
   roughly 15 milliseconds. Without it there is no usable demo, so the cache is part of the project
   rather than an optimisation bolted on afterwards.

   Cache layout: one flat Float32Array per layer per tensor, indexed
       [head][position][dim]  ->  h * (blockSize * headDim) + pos * headDim + d
   Preallocated at full context length so no array grows during generation — reallocating mid-
   generation would cause a visible stutter in the interface.
   ============================================================================================= */

'use strict';

// ---------------------------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------------------------

function mat(rows, cols) {
  return { data: new Float32Array(rows * cols), rows, cols };
}

/** y = x @ W^T + b, with x [n, in], W [out, in] (PyTorch layout), b [out] or null. */
function linear(x, W, b) {
  const outF = W.shape[0];
  const inF = W.shape[1];
  if (x.cols !== inF) throw new Error(`linear: x.cols=${x.cols} != in_features=${inF}`);
  const y = mat(x.rows, outF);
  for (let r = 0; r < x.rows; r++) {
    const xo = r * inF;
    const yo = r * outF;
    for (let o = 0; o < outF; o++) {
      let acc = b ? b.data[o] : 0;
      const wo = o * inF;
      for (let i = 0; i < inF; i++) acc += x.data[xo + i] * W.data[wo + i];
      y.data[yo + o] = acc;
    }
  }
  return y;
}

/** LayerNorm over the last axis. Biased variance, eps inside the sqrt — must match torch. */
function layerNorm(x, gamma, beta, eps) {
  const y = mat(x.rows, x.cols);
  for (let r = 0; r < x.rows; r++) {
    const o = r * x.cols;
    let mean = 0;
    for (let c = 0; c < x.cols; c++) mean += x.data[o + c];
    mean /= x.cols;
    let v = 0;
    for (let c = 0; c < x.cols; c++) { const d = x.data[o + c] - mean; v += d * d; }
    v /= x.cols;
    const inv = 1 / Math.sqrt(v + eps);
    for (let c = 0; c < x.cols; c++) {
      y.data[o + c] = (x.data[o + c] - mean) * inv * gamma.data[c] + beta.data[c];
    }
  }
  return y;
}

/**
 * Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
 *
 * PyTorch's default `gelu` is the exact erf form, not the tanh approximation, and the two differ by
 * up to ~1e-3 — which is far above our parity tolerance. Using tanh here would fail the parity check
 * for a reason that looks like a weight-loading bug, so the erf version is implemented properly.
 * Abramowitz & Stegun 7.1.26 gives erf to ~1.5e-7, comfortably inside tolerance.
 */
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
    + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

function geluInPlace(x) {
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    x.data[i] = 0.5 * v * (1 + erf(v / Math.SQRT2));
  }
  return x;
}

function addInPlace(a, b) {
  for (let i = 0; i < a.data.length; i++) a.data[i] += b.data[i];
  return a;
}

/** Deterministic PRNG so the demo's seed control actually reproduces a generation. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------------------------
// byte-level BPE, mirroring labs/p2_gpt/tokenizer.py
// ---------------------------------------------------------------------------------------------

const PRETOKEN_RE = /'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/gu;

class BPE {
  constructor(spec) {
    this.specials = spec.specials || [];
    this.specialIds = spec.special_ids || {};
    this.vocabSize = spec.vocab_size;
    this.rank = new Map();
    this.pairToId = new Map();
    spec.merges.forEach(([a, b], i) => {
      this.rank.set(`${a},${b}`, i);
      this.pairToId.set(`${a},${b}`, 256 + i);
    });
    // Bytes arrive latin-1 encoded, which round-trips 0-255 to code points 0-255 exactly.
    this.tokenBytes = spec.token_bytes_latin1.map((s) => {
      const out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
      return out;
    });
    this.encoder = new TextEncoder();
    // fatal:false so a sequence ending mid-UTF-8-character renders a replacement char instead of
    // throwing. That happens constantly during streaming generation and must never break the UI.
    this.decoder = new TextDecoder('utf-8', { fatal: false });
  }

  encodeChunk(bytes) {
    const ids = Array.from(bytes);
    while (ids.length >= 2) {
      let bestRank = Infinity;
      let bestI = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.rank.get(`${ids[i]},${ids[i + 1]}`);
        if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
      }
      if (bestI < 0) break;
      ids.splice(bestI, 2, this.pairToId.get(`${ids[bestI]},${ids[bestI + 1]}`));
    }
    return ids;
  }

  encode(text) {
    const out = [];
    // Split out special-token literals so they map to their reserved id.
    const parts = this.specials.length
      ? text.split(new RegExp(`(${this.specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`))
      : [text];
    for (const part of parts) {
      if (!part) continue;
      if (this.specialIds[part] !== undefined) { out.push(this.specialIds[part]); continue; }
      for (const m of part.matchAll(PRETOKEN_RE)) {
        out.push(...this.encodeChunk(this.encoder.encode(m[0])));
      }
    }
    return out;
  }

  decode(ids) {
    let total = 0;
    for (const id of ids) total += (this.tokenBytes[id] || []).length;
    const buf = new Uint8Array(total);
    let o = 0;
    for (const id of ids) {
      const b = this.tokenBytes[id];
      if (!b) continue;
      buf.set(b, o);
      o += b.length;
    }
    return this.decoder.decode(buf);
  }

  /** Human-readable form of one token, with whitespace made visible for the token inspector. */
  display(id) {
    const raw = this.decode([id]);
    return raw.replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/ /g, '·');
  }
}

// ---------------------------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------------------------

class Weights {
  constructor(manifest, buffer) {
    this.tensors = manifest.weights.tensors;
    this.floats = new Float32Array(buffer);
    this.cache = new Map();
  }

  get(name) {
    if (this.cache.has(name)) return this.cache.get(name);
    const e = this.tensors[name];
    if (!e) throw new Error(`missing tensor: ${name}`);
    const count = e.shape.reduce((a, b) => a * b, 1);
    const t = { data: this.floats.subarray(e.offset, e.offset + count), shape: e.shape };
    this.cache.set(name, t);
    return t;
  }

  has(name) { return Boolean(this.tensors[name]); }
  maybe(name) { return this.has(name) ? this.get(name) : null; }
}

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

class P2GPT {
  constructor(manifest, buffer, tokenizerSpec, parity) {
    this.cfg = manifest.config;
    this.manifest = manifest;
    this.W = new Weights(manifest, buffer);
    this.bpe = new BPE(tokenizerSpec);
    this.parity = parity;
    this.eps = this.cfg.layer_norm_eps;
    this.headDim = this.cfg.d_model / this.cfg.n_head;
    this.resetCache();
  }

  /** Preallocate the cache at full context so nothing reallocates during generation. */
  resetCache() {
    const { n_layer, n_head, block_size } = this.cfg;
    const per = n_head * block_size * this.headDim;
    this.cacheK = Array.from({ length: n_layer }, () => new Float32Array(per));
    this.cacheV = Array.from({ length: n_layer }, () => new Float32Array(per));
    this.cacheLen = 0;
    this.lastAttention = null;
  }

  attnW(layer) {
    const p = `blocks.${layer}.attn`;
    return {
      wq: this.W.get(`${p}.w_q.weight`), wqb: this.W.maybe(`${p}.w_q.bias`),
      wk: this.W.get(`${p}.w_k.weight`), wkb: this.W.maybe(`${p}.w_k.bias`),
      wv: this.W.get(`${p}.w_v.weight`), wvb: this.W.maybe(`${p}.w_v.bias`),
      wo: this.W.get(`${p}.w_o.weight`), wob: this.W.maybe(`${p}.w_o.bias`),
    };
  }

  /**
   * Run `ids` through the model, appending to the cache.
   * @param captureAttention when true, store per-layer attention rows for the viewer.
   * @returns logits for the LAST position only, as a Float32Array of length vocab_size.
   */
  forward(ids, { captureAttention = false } = {}) {
    const { d_model: d, n_head: h, n_layer: L, block_size: B } = this.cfg;
    const dh = this.headDim;
    const nNew = ids.length;
    const nPast = this.cacheLen;
    if (nPast + nNew > B) throw new Error(`context overflow: ${nPast} + ${nNew} > ${B}`);

    // token + learned position embeddings. Positions are nPast..nPast+nNew-1 -- taken from the
    // cache length, NOT from zero. Using zero would give every generated token position 0, which
    // produces output that is fluent for a few tokens then degenerates.
    const wte = this.W.get('wte.weight');
    const wpe = this.W.get('wpe.weight');
    let x = mat(nNew, d);
    for (let i = 0; i < nNew; i++) {
      const t = ids[i] * d;
      const p = (nPast + i) * d;
      for (let c = 0; c < d; c++) x.data[i * d + c] = wte.data[t + c] + wpe.data[p + c];
    }

    const attnRows = captureAttention ? [] : null;
    const nTotal = nPast + nNew;

    for (let l = 0; l < L; l++) {
      const g1 = this.W.get(`blocks.${l}.ln_1.gamma`);
      const b1 = this.W.get(`blocks.${l}.ln_1.beta`);
      const normed = layerNorm(x, g1, b1, this.eps);
      const A = this.attnW(l);

      const q = linear(normed, A.wq, A.wqb);
      const kNew = linear(normed, A.wk, A.wkb);
      const vNew = linear(normed, A.wv, A.wvb);

      // append new k/v into the preallocated cache
      const K = this.cacheK[l];
      const V = this.cacheV[l];
      for (let i = 0; i < nNew; i++) {
        const pos = nPast + i;
        for (let hh = 0; hh < h; hh++) {
          const src = i * d + hh * dh;
          const dst = hh * (B * dh) + pos * dh;
          for (let c = 0; c < dh; c++) { K[dst + c] = kNew.data[src + c]; V[dst + c] = vNew.data[src + c]; }
        }
      }

      const context = mat(nNew, d);
      const scale = 1 / Math.sqrt(dh);
      const scores = new Float32Array(nTotal);
      for (let i = 0; i < nNew; i++) {
        const qpos = nPast + i;
        // Causal: query at absolute position qpos may attend to keys 0..qpos inclusive.
        const limit = qpos + 1;
        for (let hh = 0; hh < h; hh++) {
          const qo = i * d + hh * dh;
          const base = hh * (B * dh);
          let max = -Infinity;
          for (let j = 0; j < limit; j++) {
            let dot = 0;
            const ko = base + j * dh;
            for (let c = 0; c < dh; c++) dot += q.data[qo + c] * K[ko + c];
            const s = dot * scale;
            scores[j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j < limit; j++) { const e = Math.exp(scores[j] - max); scores[j] = e; sum += e; }
          for (let j = 0; j < limit; j++) scores[j] /= sum;

          const co = i * d + hh * dh;
          for (let j = 0; j < limit; j++) {
            const w = scores[j];
            if (w === 0) continue;
            const vo = base + j * dh;
            for (let c = 0; c < dh; c++) context.data[co + c] += w * V[vo + c];
          }
          if (attnRows && i === nNew - 1) {
            attnRows.push({ layer: l, head: hh, row: Float32Array.from(scores.subarray(0, limit)) });
          }
        }
      }

      x = addInPlace(linear(context, A.wo, A.wob), x);

      const g2 = this.W.get(`blocks.${l}.ln_2.gamma`);
      const b2 = this.W.get(`blocks.${l}.ln_2.beta`);
      const n2 = layerNorm(x, g2, b2, this.eps);
      const hidden = geluInPlace(linear(n2, this.W.get(`blocks.${l}.mlp.fc.weight`),
                                       this.W.maybe(`blocks.${l}.mlp.fc.bias`)));
      x = addInPlace(linear(hidden, this.W.get(`blocks.${l}.mlp.proj.weight`),
                            this.W.maybe(`blocks.${l}.mlp.proj.bias`)), x);
    }

    this.cacheLen = nTotal;
    if (attnRows) this.lastAttention = attnRows;

    // Only the final position's logits matter for next-token prediction.
    const last = mat(1, d);
    last.data.set(x.data.subarray((nNew - 1) * d, nNew * d));
    const normed = layerNorm(last, this.W.get('ln_f.gamma'), this.W.get('ln_f.beta'), this.eps);
    return linear(normed, this.W.get('lm_head.weight'), null).data;
  }

  /**
   * Apply the sampling controls. Order is temperature -> top-k -> top-p, matching
   * labs/p2_gpt/sample.py. Reversing it would let temperature reintroduce mass that truncation had
   * already removed, which is not what either control is supposed to do.
   */
  filter(logits, { temperature = 1.0, topK = 0, topP = 0 } = {}) {
    const n = logits.length;
    const probs = new Float32Array(n);

    if (temperature === 0) {
      let best = 0;
      for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
      probs[best] = 1;
      return probs;
    }

    const scaled = new Float32Array(n);
    for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

    let allowed = null;   // null = everything
    if (topK > 0 && topK < n) {
      // Select by index, not by comparing against the k-th value. Ties at the cut would otherwise
      // all survive and top-k would silently keep more than k -- the same bug the Python side had.
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
      allowed = new Uint8Array(n);
      for (let i = 0; i < topK; i++) allowed[order[i]] = 1;
    }

    let max = -Infinity;
    for (let i = 0; i < n; i++) if ((!allowed || allowed[i]) && scaled[i] > max) max = scaled[i];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      if (allowed && !allowed[i]) { probs[i] = 0; continue; }
      const e = Math.exp(scaled[i] - max);
      probs[i] = e;
      sum += e;
    }
    for (let i = 0; i < n; i++) probs[i] /= sum;

    if (topP > 0 && topP < 1) {
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
      let cum = 0;
      let cut = order.length;
      for (let r = 0; r < order.length; r++) {
        cum += probs[order[r]];
        if (cum > topP) { cut = r + 1; break; }   // keep the token that crosses p
      }
      let renorm = 0;
      const keep = new Uint8Array(n);
      for (let r = 0; r < cut; r++) { keep[order[r]] = 1; renorm += probs[order[r]]; }
      for (let i = 0; i < n; i++) probs[i] = keep[i] ? probs[i] / renorm : 0;
    }

    return probs;
  }

  sampleFrom(probs, rand) {
    const r = rand();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r < acc) return i;
    }
    // Floating-point shortfall: fall back to the last nonzero entry rather than returning -1.
    for (let i = probs.length - 1; i >= 0; i--) if (probs[i] > 0) return i;
    return 0;
  }

  /**
   * Generate tokens, yielding after each one so the caller can stream to the DOM.
   * @param onToken called with ({ id, text, step, entropy, nCandidates, rawTop, filteredTop }).
   */
  async generate(prompt, {
    maxNewTokens = 200, temperature = 0.8, topK = 40, topP = 0, seed = 1234,
    onToken = null, captureAttention = false, shouldStop = null,
  } = {}) {
    this.resetCache();
    const rand = mulberry32(seed);
    const t0 = performance.now();

    let ids = this.bpe.encode(prompt);
    if (ids.length === 0) ids = this.bpe.encode('\n');
    if (ids.length > this.cfg.block_size - 1) ids = ids.slice(-(this.cfg.block_size - 1));

    const promptLength = ids.length;
    const generated = [];
    let feed = ids.slice();
    let nTokens = 0;

    for (let step = 0; step < maxNewTokens; step++) {
      if (shouldStop && shouldStop()) break;

      const logits = this.forward(feed, { captureAttention });
      const raw = this.filter(logits, { temperature: 1.0 });
      const probs = this.filter(logits, { temperature, topK, topP });
      const id = this.sampleFrom(probs, rand);

      let entropy = 0;
      for (let i = 0; i < raw.length; i++) if (raw[i] > 0) entropy -= raw[i] * Math.log(raw[i]);
      let nCandidates = 0;
      for (let i = 0; i < probs.length; i++) if (probs[i] > 0) nCandidates++;

      generated.push(id);
      nTokens++;

      if (onToken) {
        onToken({
          id, step, text: this.bpe.decode([id]), entropy, nCandidates,
          rawTop: topIndices(raw, 8).map((i) => ({ id: i, p: raw[i], token: this.bpe.display(i) })),
          filteredTop: topIndices(probs, 8).filter((i) => probs[i] > 0)
            .map((i) => ({ id: i, p: probs[i], token: this.bpe.display(i) })),
          attention: captureAttention ? this.lastAttention : null,
          cacheLen: this.cacheLen,
        });
        // Yield to the event loop so the page paints and stays responsive. Without this the whole
        // generation blocks the main thread and the UI freezes -- unacceptable for something being
        // demonstrated live.
        if (step % 4 === 3) await new Promise((r) => setTimeout(r, 0));
      }

      if (this.cacheLen >= this.cfg.block_size) {
        // Context full: rebuild from the most recent window. The model has no position embedding
        // beyond block_size, so older context is genuinely unrepresentable.
        const recent = ids.concat(generated).slice(-(this.cfg.block_size - 1));
        this.resetCache();
        feed = recent;
      } else {
        feed = [id];
      }
    }

    return {
      promptIds: ids,
      generatedIds: generated,
      promptLength,
      text: this.bpe.decode(ids.concat(generated)),
      completion: this.bpe.decode(generated),
      elapsedMs: performance.now() - t0,
      tokensPerSecond: nTokens / ((performance.now() - t0) / 1000),
    };
  }

  /**
   * Check against PyTorch reference outputs.
   *
   * Note what is and is not comparable. Sampling uses a JavaScript PRNG that cannot match
   * `torch.multinomial`, so **sampled text is not compared**. What is compared is everything
   * deterministic: the logits for fixed prompts, and greedy (temperature 0) continuations. That is
   * the honest boundary, and the page states it.
   */
  verifyParity() {
    const tol = this.parity.tolerance;
    const results = [];
    let worstLogit = 0;
    let pass = true;

    for (const c of this.parity.cases) {
      this.resetCache();
      const logits = this.forward(c.prompt_ids);
      let dev = 0;
      for (let i = 0; i < c.logits_head.length; i++) {
        dev = Math.max(dev, Math.abs(logits[i] - c.logits_head[i]));
      }
      let argmax = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;

      // greedy continuation, deterministic in both implementations
      this.resetCache();
      let feed = c.prompt_ids.slice();
      const greedy = [];
      for (let s = 0; s < c.greedy_ids.length; s++) {
        const lg = this.forward(feed);
        let best = 0;
        for (let i = 1; i < lg.length; i++) if (lg[i] > lg[best]) best = i;
        greedy.push(best);
        feed = [best];
      }
      const greedyMatch = greedy.length === c.greedy_ids.length
        && greedy.every((v, i) => v === c.greedy_ids[i]);

      const ok = dev <= tol.logits_abs && argmax === c.argmax && greedyMatch;
      pass = pass && ok;
      worstLogit = Math.max(worstLogit, dev);
      results.push({
        prompt: c.prompt, logitsMaxAbsDev: dev, argmaxMatches: argmax === c.argmax,
        greedyMatch, pytorchGreedy: this.bpe.decode(c.greedy_ids),
        jsGreedy: this.bpe.decode(greedy), pass: ok,
      });
    }

    this.resetCache();
    return { pass, worstLogitDev: worstLogit, tolerance: tol, results };
  }
}

function topIndices(arr, k) {
  return Array.from({ length: arr.length }, (_, i) => i)
    .sort((a, b) => arr[b] - arr[a]).slice(0, k);
}

async function loadP2Model(base = '../assets/models/p2') {
  const j = (n) => fetch(`${base}/${n}`).then((r) => {
    if (!r.ok) throw new Error(`${n}: HTTP ${r.status}`);
    return r.json();
  });
  const [manifest, tokenizer, parity] = await Promise.all([
    j('manifest.json'), j('tokenizer.json'), j('parity.json'),
  ]);
  const buffer = await fetch(`${base}/weights.bin`).then((r) => {
    if (!r.ok) throw new Error(`weights.bin: HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  const expected = manifest.weights.total_floats * 4;
  if (buffer.byteLength !== expected) {
    throw new Error(`weights.bin is ${buffer.byteLength} bytes, manifest expects ${expected}`);
  }
  return new P2GPT(manifest, buffer, tokenizer, parity);
}

export { P2GPT, loadP2Model, BPE, mulberry32, layerNorm, linear, geluInPlace, erf };
