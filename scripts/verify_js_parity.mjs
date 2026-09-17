/* =============================================================================================
   verify_js_parity.mjs — run the browser engines' parity checks outside a browser.

   WHY THIS EXISTS
   ---------------
   Each project page displays a "parity verified" badge. Until this script has run, that badge is an
   untested claim: the check lives in client-side JavaScript, and neither the Python test suite nor a
   page fetch executes it. Publishing an unverified parity badge would be exactly the kind of
   plausible-looking-but-unchecked assertion this repository exists to avoid.

   So the same `verifyParity()` the browser runs is executed here under Node, against the same
   exported artefacts, and the process exits non-zero on failure. That makes the badge a reported
   measurement rather than a promise.

   Usage:  node scripts/verify_js_parity.mjs           # every project that has been exported
           node scripts/verify_js_parity.mjs p1        # just one
   ============================================================================================= */

import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const modelsRoot = path.resolve(here, '..', 'docs', 'assets', 'models');

const pad = (s, n) => String(s).padEnd(n);
const one = (s) => String(s).replace(/\n/g, '\\n');

async function loadArtifacts(dir) {
  const j = async (n) => JSON.parse(await readFile(path.join(dir, n), 'utf8'));
  const [manifest, tokenizer, parity] = await Promise.all([
    j('manifest.json'), j('tokenizer.json'), j('parity.json'),
  ]);
  const raw = await readFile(path.join(dir, 'weights.bin'));
  // Copy into a standalone ArrayBuffer: Node's Buffer can be a view into a larger pooled buffer, and
  // a Float32Array over that would read the wrong bytes.
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);

  const expected = manifest.weights.total_floats * 4;
  if (buffer.byteLength !== expected) {
    throw new Error(`weights.bin is ${buffer.byteLength} bytes, manifest expects ${expected}`);
  }
  return { manifest, tokenizer, parity, buffer };
}

function header(name, manifest, tol) {
  console.log('');
  console.log('='.repeat(92));
  console.log(`${name.toUpperCase()} — JS ENGINE PARITY (the same check the project page runs on load)`);
  console.log('='.repeat(92));
  console.log(`  source run       ${manifest.source_run} @ step ${manifest.checkpoint_step}`);
  console.log(`  parameters       ${manifest.n_parameters.toLocaleString()}`);
  console.log(`  weights          ${(manifest.weights.bytes / 1e6).toFixed(2)} MB, `
    + `sha256 ${manifest.weights.sha256.slice(0, 16)}…`);
  console.log(`  tolerances       ${JSON.stringify(tol)}`);
  console.log('');
}

// ---------------------------------------------------------------------------------------------
// P1 — encoder–decoder Transformer
// ---------------------------------------------------------------------------------------------
async function checkP1() {
  const dir = path.join(modelsRoot, 'p1');
  if (!existsSync(path.join(dir, 'manifest.json'))) return null;

  const { P1Transformer } = await import('../docs/assets/js/p1-engine.js');
  const { manifest, tokenizer, parity, buffer } = await loadArtifacts(dir);
  header('p1', manifest, { encoder: parity.tolerance.encoder_output_abs, logits: parity.tolerance.logits_abs });

  const model = new P1Transformer(manifest, buffer, tokenizer, parity);
  const t0 = performance.now();
  const report = model.verifyParity();
  const elapsed = performance.now() - t0;

  console.log(`  ${pad('input', 30)} ${pad('pytorch', 13)} ${pad('javascript', 13)} `
    + `${pad('enc |Δ|', 11)} ${pad('logit |Δ|', 11)} verdict`);
  for (const r of report.results) {
    console.log(`  ${pad(one(r.source), 30)} ${pad(r.pytorchOutput, 13)} ${pad(r.jsOutput, 13)} `
      + `${pad(r.encoderMaxAbsDev.toExponential(2), 11)} ${pad(r.logitsMaxAbsDev.toExponential(2), 11)} `
      + `${r.pass ? 'MATCH' : 'DIFFERS'}`);
  }
  console.log('');
  console.log(`  worst encoder Δ  ${report.worstEncoderDev.toExponential(3)}`);
  console.log(`  worst logit Δ    ${report.worstLogitDev.toExponential(3)}`);
  console.log(`  all outputs byte-identical: ${report.results.every((r) => r.stringMatch)}`);
  console.log(`  elapsed          ${elapsed.toFixed(0)} ms`);

  const timed = model.generate('Sunday, September 22, 2001', { maxNewTokens: 12 });
  console.log(`  single inference  ${timed.elapsedMs.toFixed(1)} ms -> "${timed.output}"`);
  return { name: 'p1', pass: report.pass };
}

// ---------------------------------------------------------------------------------------------
// P2 — decoder-only GPT
// ---------------------------------------------------------------------------------------------
async function checkP2() {
  const dir = path.join(modelsRoot, 'p2');
  if (!existsSync(path.join(dir, 'manifest.json'))) return null;

  const { P2GPT } = await import('../docs/assets/js/p2-engine.js');
  const { manifest, tokenizer, parity, buffer } = await loadArtifacts(dir);
  header('p2', manifest, { logits: parity.tolerance.logits_abs });
  console.log(`  NOT compared: ${parity.tolerance.not_compared}`);
  console.log('');

  const model = new P2GPT(manifest, buffer, tokenizer, parity);
  const t0 = performance.now();
  const report = model.verifyParity();
  const elapsed = performance.now() - t0;

  console.log(`  ${pad('prompt', 34)} ${pad('logit |Δ|', 11)} ${pad('argmax', 8)} ${pad('greedy', 10)} verdict`);
  for (const r of report.results) {
    console.log(`  ${pad(one(r.prompt), 34)} ${pad(r.logitsMaxAbsDev.toExponential(2), 11)} `
      + `${pad(r.argmaxMatches ? 'match' : 'DIFFERS', 8)} `
      + `${pad(r.greedyMatch ? 'identical' : 'DIFFERS', 10)} ${r.pass ? 'MATCH' : 'DIFFERS'}`);
  }
  console.log('');
  console.log(`  worst logit Δ    ${report.worstLogitDev.toExponential(3)}`);
  console.log(`  greedy continuations identical: ${report.results.every((r) => r.greedyMatch)}`);
  console.log(`  elapsed          ${elapsed.toFixed(0)} ms`);

  // Measured generation throughput, and the KV cache's effect. The cache claim on the page should be
  // a measurement, not arithmetic.
  const gen = await model.generate('ROMEO:\n', { maxNewTokens: 64, temperature: 0.8, topK: 40, seed: 7 });
  console.log(`  generation        ${gen.generatedIds.length} tokens in ${gen.elapsedMs.toFixed(0)} ms `
    + `= ${gen.tokensPerSecond.toFixed(1)} tok/s (Node; a browser will differ)`);
  console.log(`  sample            ${JSON.stringify(gen.completion.slice(0, 90))}`);
  return { name: 'p2', pass: report.pass };
}

// ---------------------------------------------------------------------------------------------

async function main() {
  const only = process.argv[2];
  const checks = { p1: checkP1, p2: checkP2 };
  const selected = only ? [only] : Object.keys(checks);

  const results = [];
  for (const name of selected) {
    if (!checks[name]) throw new Error(`unknown project ${name}`);
    const r = await checks[name]();
    if (r) results.push(r);
    else console.log(`\n(${name}: not exported yet — skipping)`);
  }

  console.log('');
  console.log('='.repeat(92));
  if (!results.length) {
    console.log('nothing to verify: no exported models found under docs/assets/models/');
    return;
  }
  for (const r of results) {
    console.log(`  ${r.name}: ${r.pass ? 'PARITY OK' : 'PARITY FAILED'}`);
  }
  console.log('='.repeat(92));

  if (results.some((r) => !r.pass)) {
    console.error('\nAt least one engine disagrees with its trained model. Do not publish.');
    process.exit(1);
  }
  console.log('\nAll in-browser engines reproduce their PyTorch models.');
}

main().catch((err) => { console.error(err); process.exit(1); });
