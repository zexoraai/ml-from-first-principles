/* =============================================================================================
   p1-demo.js — interactive demo for Project 1.

   Everything on screen comes from a real forward pass of the trained weights, executed in this
   browser by `p1-engine.js`. Nothing is precomputed, animated, or illustrative unless it is
   explicitly labelled so (only the mask panel is, and it says as much).

   The parity badge is the honesty mechanism: on load, the engine is checked against reference
   outputs exported from PyTorch, and the measured deviation is shown. If it fails, the demo
   refuses to present its output as the model's.
   ============================================================================================= */

'use strict';

import { loadP1Model } from './p1-engine.js';

const $ = (id) => document.getElementById(id);

const PRESETS = [
  'March 3, 2019',
  '3 Mar 1987',
  'Sunday, September 22, 2001',
  '22nd December 1999',
  '2035 July 4',
  '01-May-1968',
];

let model = null;
let lastResult = null;

// ---------------------------------------------------------------------------------------------
// heatmap rendering
// ---------------------------------------------------------------------------------------------

/** Blue-scale colour for a probability in [0, 1]. Lightness carries the value. */
function heatColour(v) {
  const t = Math.max(0, Math.min(1, v));
  // interpolate from the page background to a bright accent
  const r = Math.round(13 + t * (110 - 13));
  const g = Math.round(17 + t * (168 - 17));
  const b = Math.round(23 + t * (254 - 23));
  return `rgb(${r},${g},${b})`;
}

/**
 * Render an attention matrix as an accessible table.
 * @param weights Float32Array laid out [heads, qLen, kLen]
 * @param head    which head to draw
 * @param rowLabels labels for query positions, colLabels for key positions
 */
function renderHeatmap(container, weights, head, qLen, kLen, rowLabels, colLabels, caption) {
  const table = document.createElement('table');
  table.className = 'heatmap';
  const cap = document.createElement('caption');
  cap.textContent = caption;
  table.appendChild(cap);

  const thead = document.createElement('thead');
  const hrow = document.createElement('tr');
  hrow.appendChild(document.createElement('th'));
  for (let j = 0; j < kLen; j++) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = colLabels[j] === ' ' ? '␣' : colLabels[j];
    hrow.appendChild(th);
  }
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  const base = head * qLen * kLen;
  for (let i = 0; i < qLen; i++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.scope = 'row';
    th.textContent = rowLabels[i] === ' ' ? '␣' : rowLabels[i];
    tr.appendChild(th);
    for (let j = 0; j < kLen; j++) {
      const v = weights[base + i * kLen + j];
      const td = document.createElement('td');
      td.style.background = heatColour(v);
      const pct = (v * 100).toFixed(1);
      td.title = `${rowLabels[i]} → ${colLabels[j]}: ${pct}%`;
      td.setAttribute('aria-label', `row ${rowLabels[i]}, column ${colLabels[j]}, ${pct} percent`);
      if (v > 0.35) td.classList.add('strong');
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  container.replaceChildren(table);
}

// ---------------------------------------------------------------------------------------------
// panels
// ---------------------------------------------------------------------------------------------

function renderOutput(result, sourceText) {
  $('demo-output').textContent = result.output || '(nothing generated)';
  $('demo-latency').textContent = `${result.elapsedMs.toFixed(0)} ms in your browser`;

  const looksIso = /^\d{4}-\d{2}-\d{2}$/.test(result.output);
  const badge = $('demo-shape-badge');
  badge.textContent = looksIso ? 'well-formed ISO-8601' : 'malformed — see limitations';
  badge.className = `pill ${looksIso ? 'pill--done' : 'pill--blocked'}`;
}

function renderTokens(result, sourceText) {
  const chars = Array.from(sourceText);
  const rows = chars.map((c, i) =>
    `<tr><td class="num">${i}</td><td class="mono">${c === ' ' ? '␣' : escapeHtml(c)}</td><td class="num">${result.srcIds[i]}</td></tr>`
  ).join('');
  $('demo-tokens').innerHTML = `
    <table><caption>Character-level tokenisation of the input (${chars.length} tokens)</caption>
      <thead><tr><th scope="col">pos</th><th scope="col">char</th><th scope="col">id</th></tr></thead>
      <tbody>${rows}</tbody></table>`;

  const d = model.cfg.d_model;
  const h = model.cfg.num_heads;
  const S = result.srcIds.length;
  const T = result.generatedIds.length;
  const shapeRows = [
    ['source token ids', `(1, ${S})`],
    ['after embedding × √d_model', `(1, ${S}, ${d})`],
    ['after + positional encoding', `(1, ${S}, ${d})`],
    ['q / k / v after split_heads', `(1, ${h}, ${S}, ${d / h})`],
    ['encoder self-attention scores', `(1, ${h}, ${S}, ${S})`],
    ['encoder memory', `(1, ${S}, ${d})`],
    ['decoder self-attention scores', `(1, ${h}, ${T}, ${T})`],
    ['cross-attention scores', `(1, ${h}, ${T}, ${S})`],
    ['FFN hidden', `(1, ${S}, ${model.cfg.d_ff})`],
    ['logits per step', `(1, ${model.cfg.vocab_size})`],
  ].map(([k, v]) => `<tr><td>${k}</td><td class="mono num">${v}</td></tr>`).join('');

  $('demo-shapes').innerHTML = `
    <table><caption>Tensor shapes for this input, computed live</caption>
      <thead><tr><th scope="col">boundary</th><th scope="col">shape</th></tr></thead>
      <tbody>${shapeRows}</tbody></table>`;
}

function renderAttention() {
  if (!lastResult) return;
  const kind = $('attn-kind').value;
  const layer = parseInt($('attn-layer').value, 10);
  const head = parseInt($('attn-head').value, 10);
  const container = $('attn-map');

  const srcChars = Array.from($('demo-input').value);
  const outChars = lastResult.generatedIds.map((id) => model.tok.itos[id])
    .map((t) => (t === '<bos>' ? '⟨s⟩' : t === '<eos>' ? '⟨/s⟩' : t));

  if (kind === 'cross') {
    const step = lastResult.steps[lastResult.steps.length - 1];
    const m = step.crossAttention[layer];
    renderHeatmap(container, m.weights, head, m.qLen, m.kLen,
      outChars.slice(0, m.qLen), srcChars,
      `Cross-attention, decoder layer ${layer}, head ${head}. Rows are output positions, columns are input characters.`);
    $('attn-note').textContent =
      'Each row shows where that output character drew its information from in the input. This is the map that matters for this task: to emit the year, the model has to find the year.';
  } else if (kind === 'decoder-self') {
    const step = lastResult.steps[lastResult.steps.length - 1];
    const m = step.selfAttention[layer];
    renderHeatmap(container, m.weights, head, m.qLen, m.kLen,
      outChars.slice(0, m.qLen), outChars.slice(0, m.kLen),
      `Decoder self-attention, layer ${layer}, head ${head}. Strictly lower-triangular by construction.`);
    $('attn-note').textContent =
      'The upper triangle is exactly zero — not small, zero. Forbidden scores are set to −∞ before the softmax, so exp() returns exactly 0 and no probability mass leaks to the future.';
  } else {
    const m = lastResult.encoderSelfAttention[layer];
    renderHeatmap(container, m.weights, head, m.qLen, m.kLen, srcChars, srcChars,
      `Encoder self-attention, layer ${layer}, head ${head}. Rows and columns are both input characters.`);
    $('attn-note').textContent =
      'Encoder self-attention has no causal restriction: every input character may look at every other, in both directions.';
  }
}

function renderSteps() {
  if (!lastResult) return;
  const rows = lastResult.steps.map((s) => {
    const top = s.topK.map((t, idx) => {
      const label = t.token === '<eos>' ? '⟨/s⟩' : t.token === ' ' ? '␣' : escapeHtml(t.token);
      const cls = idx === 0 ? ' class="t-good"' : '';
      return `<span${cls}>${label} ${(t.p * 100).toFixed(1)}%</span>`;
    }).join(' · ');
    const chosen = s.chosenToken === '<eos>' ? '⟨/s⟩' : escapeHtml(s.chosenToken);
    return `<tr><td class="num">${s.step}</td><td class="mono">${chosen}</td>
            <td class="small">${top}</td><td class="num">${s.entropy.toFixed(3)}</td></tr>`;
  }).join('');

  $('demo-steps').innerHTML = `
    <table><caption>Per-step distribution over the vocabulary. Entropy is in nats: low means confident, high means undecided.</caption>
      <thead><tr><th scope="col">step</th><th scope="col">chosen</th>
        <th scope="col">top 5</th><th scope="col">entropy</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
}

function renderMaskPanel() {
  const n = 7;
  const kind = $('mask-kind').value;
  const labels = Array.from({ length: n }, (_, i) => String(i));
  const weights = new Float32Array(n * n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let keep;
      if (kind === 'causal') keep = j <= i;
      else if (kind === 'padding') keep = j < 4;                 // positions 4..6 are padding
      else keep = j <= i && j < 4;                                // combined
      weights[i * n + j] = keep ? 1 : 0;
    }
  }
  renderHeatmap($('mask-map'), weights, 0, n, n, labels, labels,
    kind === 'causal' ? 'Causal keep mask: True where j ≤ i'
      : kind === 'padding' ? 'Padding keep mask: positions 4–6 are padding, so those columns are forbidden'
        : 'Combined: a pair is kept only if BOTH masks keep it — a logical AND');

  $('mask-note').textContent = kind === 'combined'
    ? 'The decoder needs both at once. Applying only one is the most consequential single bug available here: drop causality and the model trains by copying the answer out of its own input, reaching near-zero loss while being useless at generation time.'
    : kind === 'causal'
      ? 'Note the diagonal is permitted. Position i may attend to itself, because decoder inputs are the targets shifted right — "itself" is the previous target token, not the answer.'
      : 'Masks constrain which positions may be read FROM, never which may ask. Mask queries as well and a padded row would have nothing to attend to, making softmax undefined.';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------------------------

function run() {
  const text = $('demo-input').value.trim();
  const status = $('demo-status');
  if (!text) { status.textContent = 'Enter a date first.'; return; }
  if (text.length > model.cfg.max_len) {
    status.textContent = `Input capped at ${model.cfg.max_len} characters (this model's positional table).`;
    return;
  }
  try {
    status.textContent = 'running…';
    lastResult = model.generate(text, { maxNewTokens: 12, topK: 5 });
    renderOutput(lastResult, text);
    renderTokens(lastResult, text);

    // populate layer/head selectors once we know the shapes
    const layers = $('attn-kind').value === 'encoder-self'
      ? model.cfg.num_encoder_layers : model.cfg.num_decoder_layers;
    fillSelect($('attn-layer'), layers, 'layer');
    fillSelect($('attn-head'), model.cfg.num_heads, 'head');

    renderAttention();
    renderSteps();
    status.textContent = '';
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

function fillSelect(sel, n, label) {
  const prev = sel.value;
  if (sel.options.length === n) return;
  sel.replaceChildren(...Array.from({ length: n }, (_, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `${label} ${i}`;
    return o;
  }));
  if (prev && Number(prev) < n) sel.value = prev;
}

// ---------------------------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------------------------

async function boot() {
  const status = $('demo-status');
  const parityEl = $('parity-report');
  status.textContent = 'loading weights…';

  try {
    model = await loadP1Model('../assets/models/p1');
  } catch (err) {
    status.textContent = '';
    $('demo-unavailable').hidden = false;
    $('demo-live').hidden = true;
    $('demo-unavailable-reason').textContent = err.message;
    return;
  }

  $('model-params').textContent = model.manifest.n_parameters.toLocaleString();
  $('model-size').textContent = `${(model.manifest.weights.bytes / 1e6).toFixed(2)} MB`;
  $('model-config').textContent =
    `${model.cfg.num_encoder_layers}+${model.cfg.num_decoder_layers} layers · d_model ${model.cfg.d_model} · ${model.cfg.num_heads} heads · d_ff ${model.cfg.d_ff} · ${model.cfg.norm_style}-norm`;
  $('model-run').textContent = model.manifest.source_run;

  // Parity check before anything is shown as "the model's output".
  status.textContent = 'verifying against PyTorch…';
  const parity = model.verifyParity();
  const rows = parity.results.map((r) => `
    <tr>
      <td class="mono">${escapeHtml(r.source)}</td>
      <td class="mono">${escapeHtml(r.pytorchOutput)}</td>
      <td class="mono">${escapeHtml(r.jsOutput)}</td>
      <td class="num">${r.encoderMaxAbsDev.toExponential(2)}</td>
      <td class="num">${r.logitsMaxAbsDev.toExponential(2)}</td>
      <td>${r.pass ? '<span class="pill pill--done">match</span>' : '<span class="pill pill--blocked">differs</span>'}</td>
    </tr>`).join('');

  parityEl.innerHTML = `
    <table><caption>
      This page's JavaScript forward pass versus the trained PyTorch model, on fixed inputs.
      Tolerances: encoder ${parity.tolerance.encoder_output_abs}, logits ${parity.tolerance.logits_abs} —
      loose enough for float32 accumulation-order differences, far tighter than any logic error.
    </caption>
    <thead><tr><th scope="col">input</th><th scope="col">PyTorch</th><th scope="col">this browser</th>
      <th scope="col">enc max |Δ|</th><th scope="col">logit max |Δ|</th><th scope="col">verdict</th></tr></thead>
    <tbody>${rows}</tbody></table>`;

  const badge = $('parity-badge');
  if (parity.pass) {
    badge.textContent = `parity verified · worst deviation ${Math.max(parity.worstEncoderDev, parity.worstLogitDev).toExponential(1)}`;
    badge.className = 'pill pill--done';
  } else {
    badge.textContent = 'PARITY FAILED — output below is not trustworthy';
    badge.className = 'pill pill--blocked';
    $('parity-failed-warning').hidden = false;
  }

  // presets
  $('demo-presets').replaceChildren(...PRESETS.map((p) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'preset';
    b.textContent = p;
    b.addEventListener('click', () => { $('demo-input').value = p; run(); });
    return b;
  }));

  $('demo-run').addEventListener('click', run);
  $('demo-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  $('attn-kind').addEventListener('change', () => {
    const layers = $('attn-kind').value === 'encoder-self'
      ? model.cfg.num_encoder_layers : model.cfg.num_decoder_layers;
    fillSelect($('attn-layer'), layers, 'layer');
    renderAttention();
  });
  $('attn-layer').addEventListener('change', renderAttention);
  $('attn-head').addEventListener('change', renderAttention);
  $('mask-kind').addEventListener('change', renderMaskPanel);

  renderMaskPanel();
  $('demo-input').value = PRESETS[0];
  run();
}

// Results/metrics are read from the evidence file rather than typed into the page, so a number on
// screen cannot drift from the run that produced it.
async function loadResults() {
  try {
    const r = await fetch('../assets/models/p1/results.json');
    if (!r.ok) throw new Error(`results.json ${r.status}`);
    const data = await r.json();

    const m = data.final_metrics;
    $('results-table').innerHTML = `
      <table><caption>Held-out metrics from run <code>${escapeHtml(data.run_id)}</code>,
        commit <code>${escapeHtml(String(data.source_commit).slice(0, 10))}</code>. Single run — no variance estimate.</caption>
        <thead><tr><th scope="col">metric</th><th scope="col">validation</th><th scope="col">test</th></tr></thead>
        <tbody>
          <tr><td>cross-entropy (nats/token, unsmoothed)</td>
              <td class="num">${m.val.val_loss_nats_per_token.toFixed(4)}</td>
              <td class="num">${m.test.val_loss_nats_per_token.toFixed(4)}</td></tr>
          <tr><td>token accuracy (teacher forced)</td>
              <td class="num">${(m.val.token_accuracy_teacher_forced * 100).toFixed(2)}%</td>
              <td class="num">${(m.test.token_accuracy_teacher_forced * 100).toFixed(2)}%</td></tr>
          <tr><td><strong>exact match (free running)</strong></td>
              <td class="num"><strong>${(m.val.exact_match_free_running * 100).toFixed(2)}%</strong></td>
              <td class="num"><strong>${(m.test.exact_match_free_running * 100).toFixed(2)}%</strong></td></tr>
          <tr><td>examples evaluated</td>
              <td class="num">${m.val.n_examples.toLocaleString()}</td>
              <td class="num">${m.test.n_examples.toLocaleString()}</td></tr>
        </tbody></table>`;

    $('run-facts').innerHTML = [
      ['parameters', data.n_parameters.toLocaleString()],
      ['training steps', data.steps_completed.toLocaleString()],
      ['target tokens seen', data.tokens_seen.toLocaleString()],
      ['wall clock', `${(data.duration_s / 60).toFixed(1)} min`],
      ['throughput', `${(data.tokens_seen / data.duration_s).toFixed(0)} tok/s`],
      ['seeds', `python/numpy/torch = ${data.seeds.torch}`],
      ['train / val / test', `${data.split_sizes.train.toLocaleString()} / ${data.split_sizes.val.toLocaleString()} / ${data.split_sizes.test.toLocaleString()}`],
      ['unique dates per split', `${data.unique_dates.train.toLocaleString()} / ${data.unique_dates.val.toLocaleString()} / ${data.unique_dates.test.toLocaleString()}`],
    ].map(([k, v]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd></div>`).join('');

    // Real failures, found by scanning the whole test split. At ~99.9% accuracy a small sample
    // contains none, and "here are 24 correct outputs" would not be a failure analysis.
    const scan = data.failure_scan;
    if (scan && scan.n_failures > 0) {
      $('failure-table').innerHTML = `
        <table><caption>
          Every one of these is a real held-out test example, found by scanning the full split.
          <strong>${scan.n_failures} of ${scan.n_test.toLocaleString()}</strong> test examples were
          wrong (${((1 - scan.exact_match) * 100).toFixed(2)}% error rate).
          Showing ${Math.min(scan.failures.length, 12)}.
        </caption>
        <thead><tr><th scope="col">input</th><th scope="col">expected</th>
          <th scope="col">model output</th><th scope="col">wrong chars</th></tr></thead>
        <tbody>${scan.failures.slice(0, 12).map((p) => `<tr>
          <td class="mono">${escapeHtml(p.source)}</td>
          <td class="mono">${escapeHtml(p.target)}</td>
          <td class="mono t-bad">${escapeHtml(p.prediction)}</td>
          <td class="num">${p.n_wrong_chars} (pos ${p.wrong_positions.join(', ')})</td>
        </tr>`).join('')}</tbody></table>`;
    } else if (scan) {
      $('failure-table').innerHTML = `
        <div class="note good"><span class="note-label">No failures found</span>
        <p>A full scan of all ${scan.n_test.toLocaleString()} held-out test examples produced
        <strong>zero</strong> exact-match failures. That is a statement about how easy this
        synthetic task is, not about the model's general ability — see the limitations above.</p></div>`;
    } else {
      const failures = (data.sample_predictions || []).filter((p) => !p.correct);
      const shown = (data.sample_predictions || []).slice(0, 10);
      $('failure-table').innerHTML = `
        <table><caption>Sampled test predictions (${failures.length} of
          ${data.sample_predictions.length} wrong). No full-split failure scan in this run.</caption>
        <thead><tr><th scope="col">input</th><th scope="col">expected</th>
          <th scope="col">model output</th><th scope="col"></th></tr></thead>
        <tbody>${shown.map((p) => `<tr>
          <td class="mono">${escapeHtml(p.source)}</td>
          <td class="mono">${escapeHtml(p.target)}</td>
          <td class="mono ${p.correct ? '' : 't-bad'}">${escapeHtml(p.prediction)}</td>
          <td>${p.correct ? '<span class="pill pill--done">correct</span>' : '<span class="pill pill--blocked">wrong</span>'}</td>
        </tr>`).join('')}</tbody></table>`;
    }

    renderCurve(data.history);
  } catch (err) {
    $('results-unavailable').hidden = false;
    $('results-unavailable-reason').textContent = err.message;
  }
}

/** Training curve as an inline SVG. Plots the smoothed training loss and unsmoothed val loss. */
function renderCurve(history) {
  const train = history.filter((h) => h.train_loss_smoothed !== undefined);
  const val = history.filter((h) => h.val_loss_nats_per_token !== undefined);
  if (!train.length) return;

  const W = 720, H = 260, pad = 44;
  const maxStep = Math.max(...train.map((h) => h.step));
  const maxLoss = Math.max(...train.map((h) => h.train_loss_smoothed));
  const x = (s) => pad + (s / maxStep) * (W - pad - 12);
  const y = (l) => H - pad - (l / maxLoss) * (H - pad - 16);

  const path = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ');
  const trainPts = train.map((h) => [h.step, h.train_loss_smoothed]);
  const valPts = val.map((h) => [h.step, h.val_loss_nats_per_token]);

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const l = f * maxLoss;
    return `<line x1="${pad}" y1="${y(l)}" x2="${W - 12}" y2="${y(l)}" stroke="#2a323d" stroke-width="1"/>
            <text x="${pad - 8}" y="${y(l) + 4}" fill="#9aa4b2" font-size="11" text-anchor="end">${l.toFixed(1)}</text>`;
  }).join('');

  $('curve').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Training and validation loss against step count. Both fall steeply then flatten.">
      ${ticks}
      <path d="${path(trainPts)}" fill="none" stroke="#6ea8fe" stroke-width="1.6"/>
      <path d="${path(valPts)}" fill="none" stroke="#5dd39e" stroke-width="1.8" stroke-dasharray="4 3"/>
      <text x="${pad}" y="${H - 12}" fill="#9aa4b2" font-size="11">step 0</text>
      <text x="${W - 12}" y="${H - 12}" fill="#9aa4b2" font-size="11" text-anchor="end">step ${maxStep.toLocaleString()}</text>
      <text x="${W - 12}" y="20" fill="#6ea8fe" font-size="11" text-anchor="end">— training (label-smoothed)</text>
      <text x="${W - 12}" y="36" fill="#5dd39e" font-size="11" text-anchor="end">-- validation (unsmoothed)</text>
    </svg>
    <p class="small muted">The two curves are not directly comparable and the axis labels say so:
    training uses label smoothing (ε=0.1), which the paper notes deliberately hurts perplexity,
    while validation is plain cross-entropy. The shapes are informative; the gap between them is
    mostly the smoothing, not overfitting.</p>`;
}

document.addEventListener('DOMContentLoaded', () => { boot(); loadResults(); });
