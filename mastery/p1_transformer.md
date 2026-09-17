# Mastery pack — Project 1, the encoder–decoder Transformer

Answers live in `mastery/p1_transformer_answers.md`. **Read the questions first and write your own
answer before opening it.** Recognising a correct answer is not the same skill as producing one, and
only the second one survives an interview.

How this is scored: a good answer names the mechanism, states *why it exists*, and predicts what
would break without it. An answer that recites the right words without the causal story counts as a
gap and gets retaught.

---

## A. Ten oral defence questions

**A1.** Why is there a `1/√d_k` in Equation 1? Give the variance argument, then say what you would
observe during training if it were removed at `d_k = 256`, and what you would observe if it were
applied twice.

**A2.** Attention has three inputs called query, key and value. For each of the three places
attention appears in this architecture, say where each of Q, K and V comes from — and explain why
cross-attention takes queries from the decoder rather than the encoder.

**A3.** Self-attention with no positional information is permutation-equivariant. State precisely
what that means, prove it in one sentence, and explain why sinusoids in particular were chosen over
learned embeddings when the paper reports the two perform about the same.

**A4.** The paper says layer normalization goes *after* the residual add. The authors' own released
code puts it *before*. Describe both, explain which one needs learning-rate warmup and why, and say
which one this project implements and on what authority.

**A5.** Walk through the shapes of a single decoder forward pass with `batch=2`, `src_len=10`,
`tgt_len=5`, `d_model=128`, `h=4`. Name every tensor whose shape is *not* square and say why.

**A6.** Why does the padding mask constrain keys rather than queries? What specifically goes wrong if
you mask query positions too, and why is the failure worse than producing garbage?

**A7.** Label smoothing improves BLEU but the paper says it *hurts* perplexity. Explain the
mechanism behind both halves of that sentence, and state the reporting error it creates if you are
careless.

**A8.** Someone shows you a model with training loss 0.003 and free-running exact-match accuracy of
4%. Give your three most likely diagnoses in order, and the single cheapest experiment that
distinguishes them.

**A9.** Tied embeddings: which three matrices are tied, what does that make the logit for token *t*
mean geometrically, and why is the parameter saving *not* the main reason to do it here?

**A10.** You have attention maps showing head 2 of layer 1 attending strongly from output position 0
to the four year digits of the input. State exactly what that does and does not license you to
claim, and describe an intervention that would actually establish the causal claim.

---

## B. Five tensor-shape exercises

Configuration for all five: `d_model = 128`, `h = 4`, `d_ff = 512`, `V = 71`, 2 encoder layers,
2 decoder layers. Write the shape at every arrow.

**B1.** `src` ids `(3, 12)` → embedding → ×√d_model → +PE → `w_q` → `split_heads` → scores →
weights → `@v` → `merge_heads` → `w_o`.

**B2.** Cross-attention with `src_len = 12`, `tgt_len = 6`. Give the shapes of Q, K, V *after* head
splitting, the score tensor, and the context. Which axis is the softmax over?

**B3.** Give the shape of each of these masks and say which axes broadcast:
source padding; causal for `tgt_len = 6`; the combined decoder mask for a batch of 3.

**B4.** The FFN for a batch of 3 and `src_len = 12`. Shapes at the input, the hidden activation, and
the output. How many multiply-accumulates does one FFN sub-layer perform for this input?

**B5.** During greedy decoding at step 4 (so 5 tokens exist including BOS), with `src_len = 12`:
give the shapes of the decoder self-attention scores, the cross-attention scores, the hidden state
you slice out, and the final logits.

---

## C. Three numerical exercises

**C1.** By hand, `d_k = 4`.
```
q = [1, 1, 0, 0]
k_0 = [1, 1, 0, 0]     v_0 = [2, 0]
k_1 = [0, 0, 1, 1]     v_1 = [0, 2]
k_2 = [1, 0, 1, 0]     v_2 = [1, 1]
```
Compute the scaled scores, the softmax weights to 4 decimal places, and the output. Then state which
value vector dominates and why.

**C2.** Positional encoding with `d_model = 4`. Compute `PE(0)`, `PE(1)`, `PE(2)` exactly. Then take
the dimension-0/1 pair at `pos = 1` and, using only the rotation identity, predict the pair at
`pos = 3`. Verify against a direct computation.

**C3.** Label smoothing with `V = 6`, `pad_id = 0`, `ε = 0.1`, correct token `3`. Write out the full
target distribution `q`. Then compute the loss for logits `[0,0,0,10,0,0]` and for
`[0,0,0,2,0,0]`, and explain which is lower and why that is the opposite of the unsmoothed case.

---

## D. Three deliberate bugs to diagnose

For each: **predict the symptom before running anything**, then apply the patch, run the suite, and
compare what actually broke against your prediction. The prediction is the exercise; the test result
is just the marking scheme.

**D1 — the axis.** In `attention.py`, change `torch.softmax(scores, dim=-1)` to `dim=-2`.
Predict: which tests fail, and would a training run still show a falling loss?

**D2 — the mask.** In `model.py::target_keep_mask`, drop the causal term:
```python
return padding_key_mask(tgt, self.cfg.pad_id)
```
Predict: what happens to training loss, what happens to greedy decoding, and which single test
catches it first.

**D3 — the shift.** In `data.py::DateDataset.__getitem__`, change `labels` to
`torch.tensor([self.tok.bos_id] + tgt, dtype=torch.long)` — the same tensor as `tgt_in`.
Predict: the loss value it converges to, and whether the failure is visible from the loss curve
alone.

Restore with `git checkout -- <file>` after each.

---

## E. Two components to implement without looking

Delete the body, keep the signature, make the tests pass. No peeking at the original, no reading the
tests for hints beyond their names.

**E1.** `layers.LayerNorm.forward`. Must pass `test_p1_layers.py` including the parity test against
`F.layer_norm` and the two trap tests about biased variance and eps placement.

**E2.** `attention.split_heads` and `attention.merge_heads`. Must pass the round-trip test, the
contiguous-slicing test, and the `nn.MultiheadAttention` parity test — which is the one that will
actually catch you if the head layout is wrong.

---

## F. One experiment whose outcome you must predict first

**Write your prediction down before running.** Predictions you can revise after seeing the result
teach nothing.

Train two models identical except for normalisation placement:

```
.\run.cmd python scripts/train_p1.py --run-name pred-post --norm-style post --steps 1500 --warmup 400
.\run.cmd python scripts/train_p1.py --run-name pred-pre  --norm-style pre  --steps 1500 --warmup 400
```

Predict, with a number and a reason for each:

1. Which reaches lower validation loss at step 1500?
2. Which has the noisier loss curve in the first 200 steps, and why?
3. Now predict what changes if both are rerun with `--warmup 1`. Which one degrades more, and by
   roughly how much?
4. The pre-norm model has two extra `LayerNorm` modules (the final norms). Predict the exact
   parameter-count difference before checking.

Then run all four and compare. Where you were wrong, write down *which belief* was wrong — not just
the corrected number.

---

## G. Spaced review schedule

Recall beats rereading. At each checkpoint, answer from memory first, then check.

| When | What | Pass condition |
|---|---|---|
| **Day 0** | A1, A3, A6; B1; C1 | Explain the `√d_k` argument and the key-vs-query masking rule with no notes |
| **Day 1** | A2, A5; B2, B3; D1 (predict + run) | Draw all three attention uses with correct Q/K/V sources from memory |
| **Day 3** | A4, A7; C2; E1 (blind LayerNorm) | State the pre/post-norm + warmup link unprompted; LayerNorm passes first try |
| **Day 7** | A8, A9; B4, B5; D2 (predict + run) | Diagnose low-loss/bad-generation without hints |
| **Day 14** | A10; C3; E2 (blind head split/merge) | Give the correct epistemic limits of attention maps, and name a real intervention |
| **Day 30** | Experiment F, all four runs; then all of A cold | Every A answer includes mechanism + purpose + failure mode |
| **Day 60** | Teach section 7 of `docs/teaching/p1-composition.md` aloud to someone else | They can ask one "why" follow-up per step and you answer without notes |

---

## H. Self-assessment rubric

Score each A-question 0–3. Anything scoring 0–1 goes back into the next review slot.

| Score | Meaning |
|---|---|
| **0** | Cannot answer, or the answer is confidently wrong |
| **1** | Names the mechanism, no causal story ("it scales the scores") |
| **2** | Mechanism + why it exists, but cannot predict the failure mode |
| **3** | Mechanism + purpose + predicted failure mode + how you would test it |

The bar for "I can defend this project" is **3 on A1, A4, A6, A8, A10** and **≥2 everywhere else**.
Those five are the ones that separate having built it from understanding it: the scaling argument,
the paper-versus-code discrepancy, the masking asymmetry, the diagnostic reasoning, and the
epistemic limits of interpretability pictures.

Mastery is tracked separately from implementation in `records/PROGRESS.md`. A finished project is
not an understood one, and that table only moves after a back-explanation has been evaluated.
