/* =============================================================================================
   p1-engine.js — the Transformer forward pass, re-implemented in JavaScript.

   WHY THIS EXISTS (decision D-003)
   --------------------------------
   The demo has to show genuine model output *and* expose internals: attention weights per head per
   layer, tensor shapes at every boundary, and the probability distribution at each decoding step.
   A black-box runtime gives outputs but not internals, so the forward pass is written out here.

   THE RISK, AND HOW IT IS CONTROLLED
   ----------------------------------
   A re-implementation that quietly disagrees with the trained PyTorch model would turn "genuine
   model output" into a fabrication. So `verifyParity()` checks this engine against reference
   outputs exported from PyTorch (`parity.json`) at load time, and the page displays the measured
   deviation. If parity fails, the demo says so instead of showing plausible numbers.

   SCOPE SIMPLIFICATION, STATED RATHER THAN HIDDEN
   -----------------------------------------------
   The demo runs exactly one unpadded sequence at a time. Padding masks are therefore all-true
   (identity) and are omitted — not because padding is unimportant, but because there is none here.
   The causal mask IS implemented, because it is never trivial. The Python side is what handles
   batches with padding, and `tests/test_p1_attention.py` proves that path.

   LAYOUT CONVENTIONS
   ------------------
   Activations are row-major Float32Array with explicit dims: {data, rows, cols}.
   PyTorch `nn.Linear` stores weight as [out_features, in_features], and computes x @ W^T + b.
   We keep that layout so no transposition happens at export time and the manifest shapes match
   the checkpoint exactly.
   ============================================================================================= */

'use strict';

// ---------------------------------------------------------------------------------------------
// small matrix helpers
// ---------------------------------------------------------------------------------------------

function mat(rows, cols) {
  return { data: new Float32Array(rows * cols), rows, cols };
}

/** y = x @ W^T + b   with x [n, in], W [out, in], b [out] (or null). Returns [n, out]. */
function linear(x, W, b) {
  const outFeatures = W.shape[0];
  const inFeatures = W.shape[1];
  if (x.cols !== inFeatures) {
    throw new Error(`linear: x.cols=${x.cols} does not match W in_features=${inFeatures}`);
  }
  const y = mat(x.rows, outFeatures);
  for (let r = 0; r < x.rows; r++) {
    const xOff = r * inFeatures;
    const yOff = r * outFeatures;
    for (let o = 0; o < outFeatures; o++) {
      let acc = b ? b.data[o] : 0;
      const wOff = o * inFeatures;
      for (let i = 0; i < inFeatures; i++) acc += x.data[xOff + i] * W.data[wOff + i];
      y.data[yOff + o] = acc;
    }
  }
  return y;
}

/**
 * Layer normalization over the last dimension, matching labs/p1_transformer/layers.py.
 * Two details must match or parity breaks: variance is BIASED (divide by n, not n-1), and eps is
 * added INSIDE the square root.
 */
function layerNorm(x, gamma, beta, eps = 1e-5) {
  const y = mat(x.rows, x.cols);
  for (let r = 0; r < x.rows; r++) {
    const off = r * x.cols;
    let mean = 0;
    for (let c = 0; c < x.cols; c++) mean += x.data[off + c];
    mean /= x.cols;
    let variance = 0;
    for (let c = 0; c < x.cols; c++) {
      const d = x.data[off + c] - mean;
      variance += d * d;
    }
    variance /= x.cols;                       // biased, as in torch
    const inv = 1 / Math.sqrt(variance + eps); // eps inside the sqrt
    for (let c = 0; c < x.cols; c++) {
      y.data[off + c] = (x.data[off + c] - mean) * inv * gamma.data[c] + beta.data[c];
    }
  }
  return y;
}

function reluInPlace(x) {
  for (let i = 0; i < x.data.length; i++) if (x.data[i] < 0) x.data[i] = 0;
  return x;
}

function addInPlace(a, b) {
  for (let i = 0; i < a.data.length; i++) a.data[i] += b.data[i];
  return a;
}

/** Numerically stable softmax over a slice, subtracting the max first. */
function softmaxRow(arr, start, length) {
  let max = -Infinity;
  for (let i = 0; i < length; i++) if (arr[start + i] > max) max = arr[start + i];
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const e = Math.exp(arr[start + i] - max);
    arr[start + i] = e;
    sum += e;
  }
  if (sum === 0) { for (let i = 0; i < length; i++) arr[start + i] = 0; return; }
  for (let i = 0; i < length; i++) arr[start + i] /= sum;
}

/** Sinusoidal positional encoding, section 3.5. Same exp/log form as the Python implementation. */
function sinusoidalPE(maxLen, dModel) {
  const pe = mat(maxLen, dModel);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dModel; i += 2) {
      const invFreq = Math.exp(-(i / dModel) * Math.log(10000));
      const angle = pos * invFreq;
      pe.data[pos * dModel + i] = Math.sin(angle);
      if (i + 1 < dModel) pe.data[pos * dModel + i + 1] = Math.cos(angle);
    }
  }
  return pe;
}

// ---------------------------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------------------------

/**
 * Multi-head attention, Equation 1 plus section 3.2.2.
 *
 * @param causal when true, query i may only attend to keys j <= i.
 * @returns {{out: object, weights: Float32Array, heads: number, qLen: number, kLen: number}}
 *          `weights` is [heads, qLen, kLen] flattened — what the head viewer renders.
 */
function multiHeadAttention(qIn, kIn, vIn, W, numHeads, causal) {
  const q = linear(qIn, W.wq, W.wqb);
  const k = linear(kIn, W.wk, W.wkb);
  const v = linear(vIn, W.wv, W.wvb);

  const dModel = q.cols;
  const headDim = dModel / numHeads;
  const scale = 1 / Math.sqrt(headDim);
  const qLen = q.rows;
  const kLen = k.rows;

  const context = mat(qLen, dModel);
  const weights = new Float32Array(numHeads * qLen * kLen);

  for (let h = 0; h < numHeads; h++) {
    const base = h * headDim;
    for (let i = 0; i < qLen; i++) {
      const wOff = h * qLen * kLen + i * kLen;

      // scores = q_i . k_j / sqrt(d_k), with forbidden positions set to -Infinity so that
      // exp() gives exactly zero and no probability mass leaks to them.
      for (let j = 0; j < kLen; j++) {
        if (causal && j > i) { weights[wOff + j] = -Infinity; continue; }
        let dot = 0;
        for (let c = 0; c < headDim; c++) {
          dot += q.data[i * dModel + base + c] * k.data[j * dModel + base + c];
        }
        weights[wOff + j] = dot * scale;
      }
      softmaxRow(weights, wOff, kLen);

      // context_i = sum_j w_ij * v_j, within this head's subspace only.
      for (let c = 0; c < headDim; c++) {
        let acc = 0;
        for (let j = 0; j < kLen; j++) acc += weights[wOff + j] * v.data[j * dModel + base + c];
        context.data[i * dModel + base + c] = acc;
      }
    }
  }

  // Concat(head_1..head_h) is already implicit: the heads wrote into disjoint feature ranges of
  // `context`. W^O then mixes across them, which is the only place heads interact.
  return { out: linear(context, W.wo, W.wob), weights, heads: numHeads, qLen, kLen };
}

// ---------------------------------------------------------------------------------------------
// weight access
// ---------------------------------------------------------------------------------------------

class Weights {
  constructor(manifest, buffer) {
    this.manifest = manifest;
    this.floats = new Float32Array(buffer);
    this.cache = new Map();
  }

  get(name) {
    if (this.cache.has(name)) return this.cache.get(name);
    const entry = this.manifest.weights.tensors[name];
    if (!entry) throw new Error(`missing tensor in manifest: ${name}`);
    const count = entry.shape.reduce((a, b) => a * b, 1);
    const view = this.floats.subarray(entry.offset, entry.offset + count);
    const t = { data: view, shape: entry.shape };
    this.cache.set(name, t);
    return t;
  }

  has(name) {
    return Boolean(this.manifest.weights.tensors[name]);
  }

  /** Collect the four projections of one attention module, tolerating absent biases. */
  attn(prefix) {
    return {
      wq: this.get(`${prefix}.w_q.weight`), wqb: this.has(`${prefix}.w_q.bias`) ? this.get(`${prefix}.w_q.bias`) : null,
      wk: this.get(`${prefix}.w_k.weight`), wkb: this.has(`${prefix}.w_k.bias`) ? this.get(`${prefix}.w_k.bias`) : null,
      wv: this.get(`${prefix}.w_v.weight`), wvb: this.has(`${prefix}.w_v.bias`) ? this.get(`${prefix}.w_v.bias`) : null,
      wo: this.get(`${prefix}.w_o.weight`), wob: this.has(`${prefix}.w_o.bias`) ? this.get(`${prefix}.w_o.bias`) : null,
    };
  }
}

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

class P1Transformer {
  constructor(manifest, weightsBuffer, tokenizer, parity) {
    this.cfg = manifest.config;
    this.W = new Weights(manifest, weightsBuffer);
    this.tok = tokenizer;
    this.parity = parity;
    this.manifest = manifest;
    this.pe = sinusoidalPE(this.cfg.max_len, this.cfg.d_model);
    if (this.cfg.norm_style !== 'post') {
      throw new Error(`this engine implements post-norm only; checkpoint says ${this.cfg.norm_style}`);
    }
    this.trace = { shapes: [], encoderSelf: [], decoderSelf: [], cross: [] };
  }

  encode(text) {
    const map = new Map(this.tok.itos.map((t, i) => [t, i]));
    return Array.from(text).map((ch) => (map.has(ch) ? map.get(ch) : this.tok.unk_id));
  }

  decodeIds(ids) {
    const out = [];
    for (const id of ids) {
      const tk = this.tok.itos[id];
      if (tk === '<eos>') break;
      if (tk === '<pad>' || tk === '<bos>' || tk === '<unk>') continue;
      out.push(tk);
    }
    return out.join('');
  }

  /** Embedding lookup, scaled by sqrt(d_model) (section 3.4), then + positional encoding. */
  embed(ids) {
    const d = this.cfg.d_model;
    const emb = this.W.get('embedding.weight');
    const x = mat(ids.length, d);
    const scale = this.cfg.scale_embeddings ? Math.sqrt(d) : 1;
    for (let i = 0; i < ids.length; i++) {
      const src = ids[i] * d;
      for (let c = 0; c < d; c++) {
        x.data[i * d + c] = emb.data[src + c] * scale + this.pe.data[i * d + c];
      }
    }
    return x;
  }

  /** Post-norm sub-layer: y = LayerNorm(x + sublayer(x)). Dropout is inference-off. */
  subLayer(x, inner, normPrefix) {
    const gamma = this.W.get(`${normPrefix}.norm.gamma`);
    const beta = this.W.get(`${normPrefix}.norm.beta`);
    const sub = inner(x);
    return layerNorm(addInPlace(sub, x), gamma, beta, this.cfg.layer_norm_eps);
  }

  ffn(x, prefix) {
    const h = linear(x, this.W.get(`${prefix}.w_1.weight`), this.W.get(`${prefix}.w_1.bias`));
    reluInPlace(h);
    return linear(h, this.W.get(`${prefix}.w_2.weight`), this.W.get(`${prefix}.w_2.bias`));
  }

  /** Run the encoder. Returns the memory that every decoder layer will attend to. */
  runEncoder(srcIds) {
    this.trace.encoderSelf = [];
    this.trace.shapes = [
      { label: 'source token ids', shape: [1, srcIds.length] },
      { label: 'after embedding + positional encoding', shape: [1, srcIds.length, this.cfg.d_model] },
    ];
    let x = this.embed(srcIds);

    for (let l = 0; l < this.cfg.num_encoder_layers; l++) {
      const attnW = this.W.attn(`encoder.layers.${l}.self_attn`);
      let captured = null;
      x = this.subLayer(x, (t) => {
        const r = multiHeadAttention(t, t, t, attnW, this.cfg.num_heads, false);
        captured = r;
        return r.out;
      }, `encoder.layers.${l}.sub_attn`);
      this.trace.encoderSelf.push(captured);
      x = this.subLayer(x, (t) => this.ffn(t, `encoder.layers.${l}.ffn`), `encoder.layers.${l}.sub_ffn`);
      this.trace.shapes.push({ label: `after encoder layer ${l}`, shape: [1, srcIds.length, this.cfg.d_model] });
    }
    return x;
  }

  /**
   * Run the decoder over the whole prefix and return logits for the LAST position only.
   * That is the distribution over the next token, which is all greedy decoding needs.
   */
  runDecoderStep(tgtIds, memory) {
    let x = this.embed(tgtIds);
    const selfMaps = [];
    const crossMaps = [];

    for (let l = 0; l < this.cfg.num_decoder_layers; l++) {
      const selfW = this.W.attn(`decoder.layers.${l}.self_attn`);
      let cap = null;
      x = this.subLayer(x, (t) => {
        const r = multiHeadAttention(t, t, t, selfW, this.cfg.num_heads, true); // causal
        cap = r;
        return r.out;
      }, `decoder.layers.${l}.sub_self`);
      selfMaps.push(cap);

      const crossW = this.W.attn(`decoder.layers.${l}.cross_attn`);
      let capX = null;
      x = this.subLayer(x, (t) => {
        const r = multiHeadAttention(t, memory, memory, crossW, this.cfg.num_heads, false);
        capX = r;
        return r.out;
      }, `decoder.layers.${l}.sub_cross`);
      crossMaps.push(capX);

      x = this.subLayer(x, (t) => this.ffn(t, `decoder.layers.${l}.ffn`), `decoder.layers.${l}.sub_ffn`);
    }

    this.trace.decoderSelf = selfMaps;
    this.trace.cross = crossMaps;

    // Only the final row matters for the next-token distribution.
    const last = mat(1, this.cfg.d_model);
    last.data.set(x.data.subarray((tgtIds.length - 1) * this.cfg.d_model, tgtIds.length * this.cfg.d_model));
    return linear(last, this.W.get('generator.weight'), null);
  }

  /**
   * Greedy decode, recording the top-k distribution and entropy at every step.
   * Mirrors labs/p1_transformer/generate.py, including recomputing the full prefix each step
   * (no KV cache — see that file for why that is a deliberate omission, not an oversight).
   */
  generate(text, { maxNewTokens = 12, topK = 5 } = {}) {
    const srcIds = this.encode(text);
    if (srcIds.length === 0) return { output: '', steps: [], srcIds, memory: null };
    if (srcIds.length > this.cfg.max_len) {
      throw new Error(`input of ${srcIds.length} characters exceeds max_len ${this.cfg.max_len}`);
    }

    const t0 = performance.now();
    const memory = this.runEncoder(srcIds);
    const ys = [this.tok.bos_id];
    const steps = [];

    for (let s = 0; s < maxNewTokens; s++) {
      const logits = this.runDecoderStep(ys, memory);
      const v = logits.cols;

      // softmax over the vocabulary, stable
      let max = -Infinity;
      for (let i = 0; i < v; i++) if (logits.data[i] > max) max = logits.data[i];
      let sum = 0;
      const probs = new Float32Array(v);
      for (let i = 0; i < v; i++) { probs[i] = Math.exp(logits.data[i] - max); sum += probs[i]; }
      for (let i = 0; i < v; i++) probs[i] /= sum;

      let chosen = 0;
      for (let i = 1; i < v; i++) if (logits.data[i] > logits.data[chosen]) chosen = i;

      const order = Array.from({ length: v }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
      let entropy = 0;
      for (let i = 0; i < v; i++) if (probs[i] > 0) entropy -= probs[i] * Math.log(probs[i]);

      steps.push({
        step: s,
        chosen,
        chosenToken: this.tok.itos[chosen],
        entropy,
        topK: order.slice(0, topK).map((i) => ({ id: i, token: this.tok.itos[i], p: probs[i] })),
        selfAttention: this.trace.decoderSelf.map((m) => ({ weights: m.weights, heads: m.heads, qLen: m.qLen, kLen: m.kLen })),
        crossAttention: this.trace.cross.map((m) => ({ weights: m.weights, heads: m.heads, qLen: m.qLen, kLen: m.kLen })),
      });

      ys.push(chosen);
      if (chosen === this.tok.eos_id) break;
    }

    return {
      output: this.decodeIds(ys),
      steps,
      srcIds,
      generatedIds: ys,
      encoderSelfAttention: this.trace.encoderSelf,
      shapes: this.trace.shapes,
      elapsedMs: performance.now() - t0,
    };
  }

  /**
   * Check this engine against PyTorch reference outputs.
   * Returns per-case deviations and an overall pass/fail. The page shows these numbers rather
   * than asserting correctness, so a visitor can see the evidence instead of a claim.
   */
  verifyParity() {
    const tol = this.parity.tolerance;
    const results = [];
    let worstEnc = 0;
    let worstLogit = 0;
    let allPass = true;

    for (const c of this.parity.cases) {
      const memory = this.runEncoder(c.src_ids);
      let encDev = 0;
      for (let i = 0; i < c.encoder_output_head.length; i++) {
        encDev = Math.max(encDev, Math.abs(memory.data[i] - c.encoder_output_head[i]));
      }

      const logits = this.runDecoderStep([this.tok.bos_id], memory);
      let logitDev = 0;
      for (let i = 0; i < c.first_step_logits_head.length; i++) {
        logitDev = Math.max(logitDev, Math.abs(logits.data[i] - c.first_step_logits_head[i]));
      }

      let argmax = 0;
      for (let i = 1; i < logits.cols; i++) if (logits.data[i] > logits.data[argmax]) argmax = i;

      const gen = this.generate(c.source, { maxNewTokens: 12 });
      const stringMatch = gen.output === c.greedy_output;
      const pass = encDev <= tol.encoder_output_abs
        && logitDev <= tol.logits_abs
        && argmax === c.first_step_argmax
        && stringMatch;

      allPass = allPass && pass;
      worstEnc = Math.max(worstEnc, encDev);
      worstLogit = Math.max(worstLogit, logitDev);

      results.push({
        source: c.source,
        encoderMaxAbsDev: encDev,
        logitsMaxAbsDev: logitDev,
        argmaxMatches: argmax === c.first_step_argmax,
        pytorchOutput: c.greedy_output,
        jsOutput: gen.output,
        stringMatch,
        pass,
      });
    }

    return { pass: allPass, worstEncoderDev: worstEnc, worstLogitDev: worstLogit, tolerance: tol, results };
  }
}

// ---------------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------------

async function loadP1Model(base = 'assets/models/p1') {
  const [manifest, tokenizer, parity] = await Promise.all([
    fetch(`${base}/manifest.json`).then((r) => { if (!r.ok) throw new Error(`manifest ${r.status}`); return r.json(); }),
    fetch(`${base}/tokenizer.json`).then((r) => { if (!r.ok) throw new Error(`tokenizer ${r.status}`); return r.json(); }),
    fetch(`${base}/parity.json`).then((r) => { if (!r.ok) throw new Error(`parity ${r.status}`); return r.json(); }),
  ]);
  const buffer = await fetch(`${base}/weights.bin`).then((r) => {
    if (!r.ok) throw new Error(`weights ${r.status}`);
    return r.arrayBuffer();
  });
  const expected = manifest.weights.total_floats * 4;
  if (buffer.byteLength !== expected) {
    throw new Error(`weights.bin is ${buffer.byteLength} bytes, manifest expects ${expected}`);
  }
  return new P1Transformer(manifest, buffer, tokenizer, parity);
}

export { P1Transformer, loadP1Model, sinusoidalPE, layerNorm, linear, multiHeadAttention, mat };
