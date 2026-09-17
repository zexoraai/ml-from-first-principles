/* =============================================================================================
   p2-demo.js — interactive interface for Project 2.

   Built to be operated live in front of an audience, which sets the requirements:
     * generation streams token by token and the page never freezes (the engine yields to the
       event loop every few tokens),
     * a Stop button that works mid-generation,
     * every control has a visible, immediate effect and its own explanation,
     * every failure state is a labelled panel, never a blank area or a silent no-op.
   ============================================================================================= */

'use strict';

import { loadP2Model } from './p2-engine.js';

const $ = (id) => document.getElementById(id);

const PRESETS = [
  { label: 'Empty (unconditioned)', text: '\n' },
  { label: 'ROMEO:', text: 'ROMEO:\n' },
  { label: 'First Citizen:', text: 'First Citizen:\nBefore we proceed any further, ' },
  { label: 'A stage direction', text: '\nEnter KING RICHARD, attended.\n' },
  { label: 'Out-of-domain prose', text: 'The quarterly earnings report indicated that ' },
];

let model = null;
let running = false;
let stopRequested = false;
let lastRun = null;

const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ---------------------------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------------------------

function readControls() {
  return {
    temperature: parseFloat($('ctl-temp').value),
    topK: parseInt($('ctl-topk').value, 10),
    topP: parseFloat($('ctl-topp').value),
    maxNewTokens: parseInt($('ctl-tokens').value, 10),
    seed: parseInt($('ctl-seed').value, 10) || 0,
  };
}

function syncLabels() {
  const c = readControls();
  $('lbl-temp').textContent = c.temperature.toFixed(2);
  $('lbl-topk').textContent = c.topK === 0 ? 'off' : String(c.topK);
  $('lbl-topp').textContent = c.topP === 0 ? 'off' : c.topP.toFixed(2);
  $('lbl-tokens').textContent = String(c.maxNewTokens);

  // Explain the *interaction*, which is where people go wrong: raising temperature does much less
  // when a narrow top-k has already deleted the tail it would have amplified.
  let note;
  if (c.temperature === 0) {
    note = 'Temperature 0 is greedy: always the single most likely token. Top-k and top-p have no effect, and the seed is irrelevant — output is fully deterministic.';
  } else if (c.topK > 0 && c.topK <= 5 && c.temperature > 1.1) {
    note = `High temperature with a narrow top-${c.topK}: the tail that temperature would amplify has already been removed, so this is far less random than the temperature alone suggests.`;
  } else if (c.topK === 0 && c.topP === 0 && c.temperature > 1.1) {
    note = 'No truncation and temperature above 1: the full tail is in play, including tokens the model considers very unlikely. Expect occasional nonsense characters.';
  } else if (c.topP > 0 && c.topK > 0) {
    note = 'Both active: top-k caps the candidate count, then top-p trims further where the model is confident. Top-p is the binding constraint whenever the model is sure.';
  } else if (c.topP > 0) {
    note = 'Nucleus sampling adapts: few candidates where the model is confident, many where it is not. Watch the candidate count change as it generates.';
  } else {
    note = 'Temperature rescales the model\u2019s own ranking without reordering it; top-k truncates the tail.';
  }
  $('ctl-note').textContent = note;
}

// ---------------------------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------------------------

function renderTokenisation(prompt) {
  const ids = model.bpe.encode(prompt);
  const chips = ids.map((id) =>
    `<span class="tok" title="id ${id}">${escapeHtml(model.bpe.display(id))}</span>`).join('');
  const chars = Array.from(prompt).length;
  $('tokenisation').innerHTML = `
    <p class="small muted">${chars} characters &rarr; <strong>${ids.length} tokens</strong>
      (${(chars / Math.max(ids.length, 1)).toFixed(2)} chars/token).
      Each chip is one BPE token; <code>·</code> marks a leading space, <code>\\n</code> a newline.</p>
    <div class="tokstrip">${chips}</div>`;
  return ids;
}

function renderStepTable(steps) {
  const rows = steps.slice(-40).reverse().map((s) => {
    const raw = s.rawTop.slice(0, 5).map((t, i) =>
      `<span class="${i === 0 ? 't-good' : ''}">${escapeHtml(t.token)} ${(t.p * 100).toFixed(1)}%</span>`
    ).join(' · ');
    const chosen = escapeHtml(model.bpe.display(s.id));
    return `<tr>
      <td class="num">${s.step}</td>
      <td class="mono">${chosen}</td>
      <td class="num">${s.nCandidates}</td>
      <td class="num">${s.entropy.toFixed(2)}</td>
      <td class="small">${raw}</td></tr>`;
  }).join('');
  $('steps').innerHTML = `
    <table><caption>Most recent 40 steps, newest first. “Candidates” is how many tokens survived
      your top-k / top-p settings; “entropy” is of the model’s <em>raw</em> distribution in nats, so
      it reflects the model’s own uncertainty rather than your sampling choices.</caption>
      <thead><tr><th scope="col">step</th><th scope="col">chosen</th><th scope="col">cand.</th>
        <th scope="col">entropy</th><th scope="col">model’s top 5 (untruncated)</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
}

function heatColour(v) {
  const t = Math.max(0, Math.min(1, Math.sqrt(v)));   // sqrt so small weights stay visible
  return `rgb(${Math.round(13 + t * 97)},${Math.round(17 + t * 151)},${Math.round(23 + t * 231)})`;
}

function renderAttention(attention, contextIds) {
  if (!attention || !attention.length) { $('attn').innerHTML = ''; return; }
  const layer = parseInt($('attn-layer').value, 10);
  const head = parseInt($('attn-head').value, 10);
  const entry = attention.find((a) => a.layer === layer && a.head === head);
  if (!entry) { $('attn').innerHTML = ''; return; }

  const row = entry.row;
  const start = Math.max(0, row.length - 48);
  const cells = [];
  for (let j = start; j < row.length; j++) {
    const tokenText = model.bpe.display(contextIds[j] ?? 0);
    cells.push(`<span class="attncell" style="background:${heatColour(row[j])}"
      title="position ${j} (${escapeHtml(tokenText)}): ${(row[j] * 100).toFixed(2)}%">${escapeHtml(tokenText)}</span>`);
  }
  $('attn').innerHTML = `
    <p class="small muted">Layer ${layer}, head ${head}: where the <strong>most recent</strong>
      token looked, across the last ${row.length - start} positions of context. Brighter is higher
      weight.</p>
    <div class="attnstrip">${cells.join('')}</div>`;
}

// ---------------------------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------------------------

async function run() {
  if (running) return;
  const prompt = $('prompt').value.length ? $('prompt').value : '\n';
  const controls = readControls();

  running = true;
  stopRequested = false;
  $('go').disabled = true;
  $('stop').disabled = false;
  $('status').textContent = 'generating…';
  $('output').textContent = prompt;
  $('output').dataset.prompt = prompt;

  const promptIds = renderTokenisation(prompt);
  const steps = [];
  let contextIds = promptIds.slice();
  let lastAttention = null;

  try {
    const result = await model.generate(prompt, {
      ...controls,
      captureAttention: true,
      shouldStop: () => stopRequested,
      onToken: (info) => {
        steps.push(info);
        contextIds.push(info.id);
        lastAttention = info.attention;
        $('output').textContent += info.text;
        $('live-count').textContent = `${steps.length} tokens`;
        if (steps.length % 8 === 0) {
          renderStepTable(steps);
          renderAttention(lastAttention, contextIds);
        }
      },
    });

    renderStepTable(steps);
    renderAttention(lastAttention, contextIds);
    lastRun = { result, steps, contextIds };

    $('status').textContent = '';
    $('throughput').textContent =
      `${result.generatedIds.length} tokens in ${(result.elapsedMs / 1000).toFixed(2)} s `
      + `= ${result.tokensPerSecond.toFixed(1)} tokens/s in your browser`
      + (stopRequested ? ' (stopped early)' : '');
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    running = false;
    $('go').disabled = false;
    $('stop').disabled = true;
  }
}

// ---------------------------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------------------------

async function boot() {
  $('status').textContent = 'downloading weights…';
  try {
    model = await loadP2Model('../assets/models/p2');
  } catch (err) {
    $('demo-live').hidden = true;
    $('demo-unavailable').hidden = false;
    $('demo-unavailable-reason').textContent = err.message;
    $('status').textContent = '';
    return;
  }

  const m = model.manifest;
  $('m-params').textContent = m.n_parameters.toLocaleString();
  $('m-size').textContent = `${(m.weights.bytes / 1e6).toFixed(2)} MB`;
  $('m-arch').textContent =
    `${m.config.n_layer}L · d${m.config.d_model} · ${m.config.n_head}h · ctx ${m.config.block_size}`;
  $('m-vocab').textContent = `${m.config.vocab_size} BPE`;
  $('m-run').textContent = m.source_run;

  $('status').textContent = 'verifying against PyTorch…';
  const parity = model.verifyParity();
  $('parity').innerHTML = `
    <table><caption>
      This page’s JavaScript engine versus the trained PyTorch model. Tolerance on logits:
      ${parity.tolerance.logits_abs}. <strong>Sampled text is not compared</strong> —
      ${escapeHtml(parity.tolerance.not_compared)}
    </caption>
    <thead><tr><th scope="col">prompt</th><th scope="col">max |Δ| logits</th>
      <th scope="col">argmax</th><th scope="col">greedy continuation</th>
      <th scope="col">verdict</th></tr></thead>
    <tbody>${parity.results.map((r) => `<tr>
      <td class="mono">${escapeHtml(r.prompt.replace(/\n/g, '\\n'))}</td>
      <td class="num">${r.logitsMaxAbsDev.toExponential(2)}</td>
      <td>${r.argmaxMatches ? 'match' : '<span class="t-bad">differs</span>'}</td>
      <td>${r.greedyMatch ? 'identical' : '<span class="t-bad">differs</span>'}</td>
      <td>${r.pass ? '<span class="pill pill--done">ok</span>'
                   : '<span class="pill pill--blocked">FAIL</span>'}</td>
    </tr>`).join('')}</tbody></table>`;

  const badge = $('parity-badge');
  if (parity.pass) {
    badge.textContent = `parity verified · worst logit Δ ${parity.worstLogitDev.toExponential(1)}`;
    badge.className = 'pill pill--done';
  } else {
    badge.textContent = 'PARITY FAILED — output is not trustworthy';
    badge.className = 'pill pill--blocked';
    $('parity-warning').hidden = false;
  }

  // layer/head selectors
  const fill = (sel, n, label) => {
    sel.replaceChildren(...Array.from({ length: n }, (_, i) => {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = `${label} ${i}`;
      return o;
    }));
  };
  fill($('attn-layer'), m.config.n_layer, 'layer');
  fill($('attn-head'), m.config.n_head, 'head');
  $('attn-layer').value = String(m.config.n_layer - 1);   // last layer is usually most legible

  // presets
  $('presets').replaceChildren(...PRESETS.map((p) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'preset';
    b.textContent = p.label;
    b.addEventListener('click', () => { $('prompt').value = p.text; renderTokenisation(p.text); });
    return b;
  }));

  ['ctl-temp', 'ctl-topk', 'ctl-topp', 'ctl-tokens'].forEach((id) =>
    $(id).addEventListener('input', syncLabels));
  $('go').addEventListener('click', run);
  $('stop').addEventListener('click', () => { stopRequested = true; $('status').textContent = 'stopping…'; });
  $('attn-layer').addEventListener('change', () => lastRun && renderAttention(lastRun.steps.at(-1).attention, lastRun.contextIds));
  $('attn-head').addEventListener('change', () => lastRun && renderAttention(lastRun.steps.at(-1).attention, lastRun.contextIds));
  $('prompt').addEventListener('input', () => { if (!running) renderTokenisation($('prompt').value || '\n'); });
  $('reseed').addEventListener('click', () => {
    $('ctl-seed').value = String(Math.floor(Math.random() * 100000));
  });

  syncLabels();
  $('prompt').value = PRESETS[2].text;
  renderTokenisation($('prompt').value);
  $('status').textContent = '';
  $('go').disabled = false;
}

// ---------------------------------------------------------------------------------------------
// results, read from the evidence file rather than typed into the page
// ---------------------------------------------------------------------------------------------

async function loadResults() {
  try {
    const r = await fetch('../assets/models/p2/results.json');
    if (!r.ok) throw new Error(`results.json HTTP ${r.status}`);
    const d = await r.json();
    const f = d.final_metrics;

    $('results').innerHTML = `
      <table><caption>Run <code>${escapeHtml(d.run_id)}</code>, commit
        <code>${escapeHtml(String(d.source_commit).slice(0, 10))}</code>.
        Single run — no variance estimate.</caption>
      <thead><tr><th scope="col">metric</th><th scope="col">value</th></tr></thead>
      <tbody>
        <tr><td>validation cross-entropy (nats/token)</td><td class="num">${f.val_loss.toFixed(4)}</td></tr>
        <tr><td>validation perplexity (per BPE token)</td><td class="num">${f.val_perplexity.toFixed(2)}</td></tr>
        <tr><td>training cross-entropy</td><td class="num">${f.train_loss.toFixed(4)}</td></tr>
        <tr><td>train − val gap</td><td class="num">${(f.val_loss - f.train_loss).toFixed(4)}</td></tr>
        <tr><td>evaluation windows averaged</td><td class="num">${f.eval_iters}</td></tr>
      </tbody></table>`;

    $('runfacts').innerHTML = [
      ['parameters', d.n_parameters.toLocaleString()],
      ['non-embedding', d.n_parameters_non_embedding.toLocaleString()],
      ['steps', d.steps_completed.toLocaleString()],
      ['tokens seen', d.tokens_seen.toLocaleString()],
      ['epochs over the corpus', (d.tokens_seen / d.data.train_tokens).toFixed(1)],
      ['wall clock', `${(d.duration_s / 60).toFixed(1)} min`],
      ['throughput', `${(d.tokens_seen / d.duration_s).toFixed(0)} tok/s`],
      ['context', `${d.model_config.block_size} tokens`],
      ['compression', `${d.data.compression_chars_per_token.toFixed(2)} chars/token`],
      ['attention share of FLOPs', `${(d.flop_estimate_per_token.attention_share * 100).toFixed(1)}%`],
    ].map(([k, v]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd></div>`).join('');

    if (d.samples) {
      $('samples').innerHTML = d.samples.map((s) => `
        <div class="sample">
          <h4>temperature ${s.temperature}${s.top_k ? `, top-k ${s.top_k}` : ', no truncation'}</h4>
          <pre>${escapeHtml(s.text.slice(0, 700))}</pre>
        </div>`).join('');
    }

    renderCurve(d.history);
  } catch (err) {
    $('results-unavailable').hidden = false;
    $('results-unavailable-reason').textContent = err.message;
  }
}

function renderCurve(history) {
  const evals = history.filter((h) => h.eval_val_loss !== undefined);
  const train = history.filter((h) => h.train_loss !== undefined);
  if (!evals.length) return;

  const W = 760, H = 280, pad = 46;
  const maxStep = Math.max(...history.map((h) => h.step));
  const all = [...train.map((h) => h.train_loss), ...evals.map((h) => h.eval_val_loss)];
  const maxL = Math.max(...all);
  const minL = Math.min(...all);
  const x = (s) => pad + (s / maxStep) * (W - pad - 14);
  const y = (l) => H - pad - ((l - minL) / Math.max(maxL - minL, 1e-6)) * (H - pad - 20);
  const path = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ');

  const best = evals.reduce((a, b) => (b.eval_val_loss < a.eval_val_loss ? b : a));
  const grid = [0, 0.25, 0.5, 0.75, 1].map((fr) => {
    const l = minL + fr * (maxL - minL);
    return `<line x1="${pad}" y1="${y(l)}" x2="${W - 14}" y2="${y(l)}" stroke="#2a323d"/>
            <text x="${pad - 8}" y="${y(l) + 4}" fill="#9aa4b2" font-size="11" text-anchor="end">${l.toFixed(2)}</text>`;
  }).join('');

  $('curve').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training and validation cross-entropy against step. Both fall; validation flattens and then rises slightly as the model begins to overfit the small corpus.">
      ${grid}
      <path d="${path(train.map((h) => [h.step, h.train_loss]))}" fill="none" stroke="#6ea8fe" stroke-width="1.4" opacity="0.85"/>
      <path d="${path(evals.map((h) => [h.step, h.eval_train_loss]))}" fill="none" stroke="#9aa4b2" stroke-width="1.6" stroke-dasharray="2 2"/>
      <path d="${path(evals.map((h) => [h.step, h.eval_val_loss]))}" fill="none" stroke="#5dd39e" stroke-width="2"/>
      <circle cx="${x(best.step)}" cy="${y(best.eval_val_loss)}" r="4" fill="#f0b849"/>
      <text x="${x(best.step)}" y="${y(best.eval_val_loss) - 10}" fill="#f0b849" font-size="11" text-anchor="middle">best ${best.eval_val_loss.toFixed(3)} @ ${best.step}</text>
      <text x="${W - 14}" y="18" fill="#6ea8fe" font-size="11" text-anchor="end">— per-step training loss</text>
      <text x="${W - 14}" y="34" fill="#9aa4b2" font-size="11" text-anchor="end">-- train loss (averaged)</text>
      <text x="${W - 14}" y="50" fill="#5dd39e" font-size="11" text-anchor="end">— validation loss</text>
      <text x="${pad}" y="${H - 12}" fill="#9aa4b2" font-size="11">step 0</text>
      <text x="${W - 14}" y="${H - 12}" fill="#9aa4b2" font-size="11" text-anchor="end">step ${maxStep.toLocaleString()}</text>
    </svg>`;
}

document.addEventListener('DOMContentLoaded', () => { boot(); loadResults(); });
