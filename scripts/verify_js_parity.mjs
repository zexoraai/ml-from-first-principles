/* =============================================================================================
   verify_js_parity.mjs — run the browser engine's parity check outside a browser.

   WHY THIS EXISTS
   ---------------
   The project page displays a "parity verified" badge. Until this script has run, that badge is an
   untested claim: the check lives in client-side JavaScript, and neither the Python test suite nor
   a page fetch executes it. Publishing an unverified parity badge would be exactly the kind of
   plausible-looking-but-unchecked assertion this repository is supposed to avoid.

   So the same `verifyParity()` the browser runs is executed here under Node, against the same
   exported artefacts, and the process exits non-zero on failure. That makes the badge a reported
   measurement rather than a promise.

   Usage:  node scripts/verify_js_parity.mjs
   ============================================================================================= */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { P1Transformer } from '../docs/assets/js/p1-engine.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const base = path.resolve(here, '..', 'docs', 'assets', 'models', 'p1');

const json = async (name) => JSON.parse(await readFile(path.join(base, name), 'utf8'));

async function main() {
  const [manifest, tokenizer, parity] = await Promise.all([
    json('manifest.json'), json('tokenizer.json'), json('parity.json'),
  ]);
  const raw = await readFile(path.join(base, 'weights.bin'));
  // Copy into a standalone ArrayBuffer: Node's Buffer may be a view into a larger pooled buffer,
  // and Float32Array over that would read the wrong bytes.
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);

  const expectedBytes = manifest.weights.total_floats * 4;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`weights.bin is ${buffer.byteLength} bytes, manifest expects ${expectedBytes}`);
  }

  console.log('='.repeat(84));
  console.log('JS ENGINE PARITY CHECK  (the same check the project page runs on load)');
  console.log('='.repeat(84));
  console.log(`  source run       ${manifest.source_run} @ step ${manifest.checkpoint_step}`);
  console.log(`  parameters       ${manifest.n_parameters.toLocaleString()}`);
  console.log(`  weights          ${(manifest.weights.bytes / 1e6).toFixed(2)} MB, sha256 ${manifest.weights.sha256.slice(0, 16)}…`);
  console.log(`  architecture     ${manifest.config.num_encoder_layers}+${manifest.config.num_decoder_layers} layers, `
    + `d_model ${manifest.config.d_model}, ${manifest.config.num_heads} heads, ${manifest.config.norm_style}-norm`);
  console.log(`  tolerances       encoder ${parity.tolerance.encoder_output_abs}, logits ${parity.tolerance.logits_abs}`);
  console.log('');

  const model = new P1Transformer(manifest, buffer, tokenizer, parity);
  const t0 = performance.now();
  const report = model.verifyParity();
  const elapsed = performance.now() - t0;

  const pad = (s, n) => String(s).padEnd(n);
  console.log(`  ${pad('input', 28)} ${pad('pytorch', 12)} ${pad('javascript', 12)} `
    + `${pad('enc |Δ|', 10)} ${pad('logit |Δ|', 10)} verdict`);
  for (const r of report.results) {
    console.log(`  ${pad(r.source, 28)} ${pad(r.pytorchOutput, 12)} ${pad(r.jsOutput, 12)} `
      + `${pad(r.encoderMaxAbsDev.toExponential(2), 10)} ${pad(r.logitsMaxAbsDev.toExponential(2), 10)} `
      + `${r.pass ? 'MATCH' : 'DIFFERS'}`);
  }

  console.log('');
  console.log(`  worst encoder deviation  ${report.worstEncoderDev.toExponential(3)}`);
  console.log(`  worst logit deviation    ${report.worstLogitDev.toExponential(3)}`);
  console.log(`  all ${report.results.length} cases string-identical: `
    + `${report.results.every((r) => r.stringMatch)}`);
  console.log(`  elapsed                  ${elapsed.toFixed(0)} ms`);
  console.log('');

  // A latency figure for the page's honesty about in-browser cost, measured not guessed.
  const timed = model.generate('Sunday, September 22, 2001', { maxNewTokens: 12 });
  console.log(`  single inference         ${timed.elapsedMs.toFixed(1)} ms for "${timed.output}" `
    + `(Node on this CPU; a browser will differ)`);

  if (!report.pass) {
    console.error('\nPARITY FAILED — the browser engine disagrees with the trained model.');
    process.exit(1);
  }
  console.log('\nPARITY OK — the in-browser forward pass reproduces the PyTorch model.');
}

main().catch((err) => { console.error(err); process.exit(1); });
