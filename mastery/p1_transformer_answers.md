# Answer notes — Project 1

**Write your own answer first.** These are marking notes, not a script to memorise. An answer that
differs in wording but contains the mechanism, the purpose and the failure mode is a full-credit
answer.

---

## A1 · Why `1/√d_k`?

**The argument.** Assume the components of `q` and `k` are independent, mean 0, variance 1. Then
`q·k = Σᵢ qᵢkᵢ` over `d_k` terms has mean 0 and variance `d_k`, so its typical magnitude grows like
`√d_k`. Softmax is invariant to an additive shift but *not* to a multiplicative one: feed it scores
of typical size `√d_k` and as `d_k` grows the distribution approaches one-hot. Dividing by `√d_k`
restores unit variance.

**Why that matters mechanically.** The Jacobian of softmax is `diag(p) − ppᵀ`. As `p → one-hot`,
every entry → 0. So the gradient with respect to the scores vanishes precisely in the regime where
the scores are wrong and need to change. The model cannot learn its way out, because the saturation
that kills the gradient is itself the thing that must change.

**If removed at `d_k = 256`:** attention collapses to near-selection immediately. Measured in
`test_large_dk_without_scaling_saturates_softmax`: mean maximum row probability > 0.9 unscaled
versus < 0.5 scaled. Training would show a loss that falls briefly then flattens well above the
achievable floor, and attention maps that look like hard pointers from step one.

**If applied twice** (divide by `d_k` instead of `√d_k`): the opposite failure. Scores shrink toward
zero, softmax → uniform, and every position becomes an equally weighted average of all values.
Attention stops distinguishing anything; the model degenerates towards a bag-of-positions and the
FFN has to do all the work. Loss falls slowly to a poor plateau.

**Full credit** requires naming both failure directions. Only naming the saturation case is a 2.

---

## A2 · Where Q, K and V come from

| Use | Q | K | V | Mask |
|---|---|---|---|---|
| Encoder self-attention | encoder states | encoder states | encoder states | source padding |
| Decoder self-attention | decoder states | decoder states | decoder states | target padding **AND** causal |
| Cross-attention | **decoder** states | encoder memory | encoder memory | source padding |

**Why queries come from the decoder in cross-attention.** The query is the *question*; keys and
values are the *material being searched*. The decoder is the party with a question — "I am about to
emit output position 4, which part of the input do I need?" The encoder holds the material. Swap
them and you would be asking the source what it wants from the partially generated output, which
inverts the information flow and, worse, would make the encoder's representation depend on the
decoding step — destroying the property that the memory is computed once and reused.

Note also that all three are the *same module* with different arguments. That is the paper's actual
structural claim and the reason the architecture is as small as it is.

---

## A3 · Permutation equivariance

**Statement.** Let `P` be a permutation matrix. Self-attention `A` without positional information
satisfies `A(Px) = P·A(x)`. Permuting the inputs permutes the outputs identically, with the same
values.

**One-sentence proof.** Every operation in `softmax(QKᵀ/√d_k)V` is either applied per position
(the projections) or is a sum over positions weighted by content-derived scores — no step reads a
position index — so relabelling positions relabels rows of the output and nothing else.

**Consequence.** `"the cat sat"` and `"sat cat the"` are indistinguishable. Recurrence got order for
free by consuming tokens sequentially; attention discarded recurrence and must buy order back.
`test_attention_is_permutation_equivariant_without_positional_encoding` demonstrates the weakness,
and the next test shows PE repairing it.

**Why sinusoids over learned embeddings** when Table 3 row (E) reports near-identical performance:
the paper's stated reason is extrapolation — sinusoids are defined at positions never seen in
training, whereas a learned embedding table simply has no row for position 5001. Note honestly that
*whether* extrapolation actually works is an empirical question the paper does not settle.

The deeper reason to like them: for fixed offset `k`, each (sin, cos) pair at frequency `w` rotates
by the fixed angle `wk`:

```
[ sin(w(p+k)) ]   [ cos wk   sin wk ] [ sin wp ]
[ cos(w(p+k)) ] = [ −sin wk  cos wk ] [ cos wp ]
```

The matrix depends on `k` but not on `p`. So a single learned linear map — which is exactly what
`W^Q` and `W^K` are — can implement "attend 3 positions back" uniformly across the sequence. That is
the bridge from absolute encodings to relative addressing, and
`test_relative_offset_is_a_fixed_linear_map` checks the identity numerically.

---

## A4 · Post-norm versus pre-norm

**Post-norm** (paper §3.1): `y = LayerNorm(x + Sublayer(x))`.
**Pre-norm** (`tensor2tensor`, and everything since GPT-2): `y = x + Sublayer(LayerNorm(x))`.

**Which needs warmup: post-norm.** In pre-norm there is an unbroken identity path from the input of
the stack to its output — the residual branch is added without passing through any normalisation. So
`∂L/∂(early layer)` contains a term unscaled by any LayerNorm Jacobian, and deep stacks train from
initialisation. In post-norm every residual sum is immediately normalised, so the identity path is
rescaled at each of the `N` layers. Early in training the normalisation statistics and the parameters
are mismatched, gradients are badly scaled, and a large initial learning rate diverges. **Warmup is
not a decorative addition to the original recipe — it is what makes post-norm trainable.**

`test_pre_norm_leaves_an_unnormalised_identity_path` shows the structural difference concretely: with
a sub-layer that outputs exactly zero, pre-norm returns `x` untouched while post-norm returns
`LayerNorm(x)`.

**Which this project implements and on what authority:** post-norm, because the brief says "original"
and §3.1 of arXiv:1706.03762v7 states the sub-layer output is `LayerNorm(x + Sublayer(x))`. Pre-norm
is available behind `norm_style="pre"` so the discrepancy is a runnable experiment. Recorded as
decision D-006.

**The point worth making unprompted:** papers and their reference code disagree, and most people
follow the code while citing the paper. Knowing which you implemented, and saying so, is the actual
skill being tested.

---

## A5 · Shapes for `batch=2, src_len=10, tgt_len=5, d_model=128, h=4`

```
src ids                        (2, 10)
tgt_in ids                     (2, 5)
embed(src) × √128              (2, 10, 128)
+ PE                           (2, 10, 128)
w_q / w_k / w_v                (2, 10, 128)
split_heads                    (2, 4, 10, 32)
encoder scores                 (2, 4, 10, 10)
encoder weights                (2, 4, 10, 10)
weights @ v                    (2, 4, 10, 32)
merge_heads                    (2, 10, 128)
FFN hidden                     (2, 10, 512)
encoder memory                 (2, 10, 128)

decoder self scores            (2, 4, 5, 5)
cross Q after split            (2, 4, 5, 32)
cross K/V after split          (2, 4, 10, 32)
cross scores                   (2, 4, 5, 10)   ← not square
decoder output                 (2, 5, 128)
logits                         (2, 5, V)
```

**Non-square tensors and why.**
- `cross scores (2, 4, 5, 10)` — one row per *output* position, one column per *input* position.
  Source and target lengths are independent, so this is rectangular by nature. It is the tensor the
  demo's cross-attention view draws.
- `FFN hidden (2, 10, 512)` — non-square in the feature axis because `d_ff = 4·d_model`.
- `logits (2, 5, V)` — the feature axis becomes the vocabulary.

Masks: source padding `(2, 1, 1, 10)`, causal `(1, 1, 5, 5)`, combined decoder `(2, 1, 5, 5)`.

---

## A6 · Why padding masks constrain keys, not queries

**The rule.** Mask which positions may be *read from*, never which may *ask*.

**What goes wrong if you mask queries too.** A padded query row would have every key forbidden.
Softmax is then being asked to normalise over the empty set, which is undefined. In floating point
you get `NaN` across the row.

**Why that failure is worse than garbage.** Garbage at a padded output position is harmless — the
loss ignores those positions, so nothing reads it. `NaN` is not local: it propagates through the
weighted sum, into `W^O`, into every downstream layer, and in the backward pass into the gradient of
every *shared* parameter. One undefined row destroys the entire model in a single step, and the
symptom (all-`NaN` loss) points nowhere near the cause.

Because masks constrain keys only, every query row retains at least one permitted key as long as the
sequence has at least one real token, so the condition never arises in correct usage.
`fully_masked_rows` exists to detect it anyway — if the counter is ever nonzero, that is a bug
signal, not a routine event — and the implementation writes exact zeros rather than `NaN` so the
failure stays findable.

---

## A7 · Label smoothing hurts perplexity and helps BLEU

**Why it helps accuracy.** With a one-hot target, the loss is only minimised as the correct logit
runs to `+∞`. The model is pushed toward unattainable, badly calibrated confidence, and toward
over-fitting the exact token identity of the training data. Capping the target at `1 − ε` gives the
loss a finite minimum at a finite logit gap, which regularises and improves the ranking decisions
that BLEU and accuracy actually measure.

**Why it hurts perplexity.** Perplexity *is* a measure of the model's certainty on held-out data.
Smoothing deliberately trains the model to be less certain — to reserve `ε` of its mass for tokens it
believes are wrong. So it must score worse on a metric that rewards certainty. The paper states this
outright in §5.4; it is a trade, not a side effect.

**The reporting error it creates.** A label-smoothed training loss and an unsmoothed validation loss
are different objectives and must not be compared, plotted on the same axis without labels, or used
to diagnose overfitting. The apparent gap is mostly the smoothing. This project therefore defines
`val_loss_nats_per_token` as *unsmoothed* cross-entropy, computed separately from the training
objective, and the project page's curve says so next to the plot.

**Bonus, worth knowing:** the smoothed loss has a nonzero floor. For `ε = 0.1`, `V = 70`, a model
that predicts `q` exactly still scores about `0.747` nats. So a smoothed training loss plateauing
near `0.75–0.80` is *converged*, not stuck — and mistaking that plateau for a bug is a common waste
of an afternoon.

---

## A8 · Loss 0.003, exact match 4%

**Diagnoses, in order of likelihood.**

1. **Causal mask missing, transposed, or applied to the wrong axis.** The model is reading the answer
   out of its own decoder input. Teacher-forced loss goes to ~0 because the task becomes copying;
   free-running generation collapses because at inference the future is not there to copy. This is by
   far the most common cause of exactly this symptom pair.
2. **Off-by-one in the shift.** If `labels == tgt_in` rather than shifted, the model is trained to
   emit its own input. Same signature.
3. **Genuine error compounding.** If both of the above are clean, the model may legitimately be good
   at next-token prediction and bad at free running — one early mistake derails the rest. This is
   real, but it does not usually produce loss as low as 0.003.

**The cheapest distinguishing experiment.** Take one batch. Run a forward pass, record the logits.
Perturb only the *later* target tokens and run again. If logits at earlier positions change, the
future is leaking → diagnosis 1 or 2. That is a two-line intervention, needs no training, and
separates the leak hypotheses from the compounding hypothesis immediately. To then separate 1 from 2,
assert `labels[i] == tgt_in[i+1]` directly on a single example.

`test_decoder_output_is_invariant_to_future_target_tokens` is exactly this check, run in CI.

---

## A9 · Tied embeddings

**The three matrices** (§3.4): the encoder input embedding, the decoder input embedding, and the
pre-softmax output projection. All one matrix. Embeddings are additionally multiplied by `√d_model`.

**Geometric meaning of the logit.** With tying, `logit_t = h · e_t` where `h` is the final hidden
state and `e_t` is token `t`'s own embedding. So the model scores "how much does my current state
look like token `t`" — the output space and the input space are the same space, and generation
becomes nearest-neighbour search in embedding space under a dot product.

**Why the parameter saving is not the main reason here.** At `V = 70` and `d_model = 128`, tying saves
`2 × 70 × 128 = 17,920` parameters out of ~932k — under 2%. Irrelevant. The reason is the inductive
bias: one representation is forced to serve both "which token is this" and "how much do I want to
emit this token", which couples the two and regularises both. In the paper's setting (`V ≈ 37,000`,
`d_model = 512`) the saving is ~38M parameters and does matter — so the honest answer distinguishes
*our* motivation from *theirs*.

**Implementation detail worth mentioning:** it must be the same `Parameter` object, not a copy.
A copy would receive two separate gradients and drift apart. `test_tying_shares_one_parameter_object_not_a_copy`
checks identity with `is`, and a companion test checks the tie survives a `state_dict` round trip.

---

## A10 · What an attention map licenses you to claim

**What it shows.** In one head, at one layer, for one input, the weights of a convex combination over
value vectors. That is all. It is a faithful record of a computation that happened.

**What it does not show.**
- **That the information was used.** The head's output passes through `W^O`, which can project it
  into a subspace the rest of the network ignores. High attention weight with zero causal
  contribution is entirely possible.
- **That the head is doing the job the picture suggests.** The residual stream carries information
  *around* the head; a later layer can overwrite it; the same output could arise for other reasons.
- **That it generalises.** One input is one input. Patterns often vary across examples in ways a
  single heatmap hides.
- **Anything about the other heads.** Attention is a sum over heads; reading one in isolation ignores
  the rest of the layer.

**Interventions that would actually establish the claim.**
- **Ablation.** Zero this head's output (or replace it with its mean over the dataset) and measure the
  change in the target logit. A causally load-bearing head produces a measurable drop.
- **Activation patching / causal tracing.** Run the model on input A, cache activations, then run on
  input B while substituting this head's output from A. If the prediction moves toward A's answer,
  the head carries that information causally.
- **Attention knockout.** Force specific attention weights to zero and see whether the output changes.

**Full credit** requires naming at least one intervention, not just listing caveats. The distinction
being tested is between *inspection* and *intervention* — correlational evidence about a computation
versus causal evidence about its role.

---

# Section B — shape exercises

**B1.** `(3,12)` → `(3,12,128)` → `(3,12,128)` → `(3,12,128)` → `(3,12,128)` → `(3,4,12,32)` →
scores `(3,4,12,12)` → weights `(3,4,12,12)` → `(3,4,12,32)` → `(3,12,128)` → `(3,12,128)`.

**B2.** Q `(B,4,6,32)`, K `(B,4,12,32)`, V `(B,4,12,32)`, scores `(B,4,6,12)`, context `(B,4,6,32)`.
Softmax over the **last** axis (keys, length 12).

**B3.** source padding `(B,1,1,12)` — broadcasts over heads and queries. Causal `(1,1,6,6)` —
broadcasts over batch and heads. Combined decoder `(3,1,6,6)` — broadcasts over heads only.

**B4.** in `(3,12,128)`, hidden `(3,12,512)`, out `(3,12,128)`.
MACs = `3·12·(128·512 + 512·128) = 3·12·131,072 = 4,718,592`.

**B5.** decoder self scores `(1,4,5,5)`; cross scores `(1,4,5,12)`; hidden slice `(1,1,128)` (last
position only); logits `(1,70)`.

---

# Section C — numerical exercises

**C1.** `√d_k = 2`. Raw dots: `q·k₀ = 2`, `q·k₁ = 0`, `q·k₂ = 1`. Scaled: `1.0, 0.0, 0.5`.
`exp: 2.71828, 1.0, 1.64872`, sum `5.36700`.
Weights: `0.50648, 0.18goal63, 0.30724` → precisely `[0.5065, 0.1864, 0.3072]`.
Output `= 0.5065·[2,0] + 0.1864·[0,2] + 0.3072·[1,1] = [1.3202, 0.6800]`.
`v₀` dominates because `k₀` is the only key exactly aligned with `q`; `k₂` overlaps in one of the two
active dimensions so it gets intermediate weight; `k₁` is orthogonal to `q` and gets the residual.

**C2.** `d_model = 4`, so `i ∈ {0,1}` and `inv_freq = [1, 0.01]`.
`PE(0) = [0, 1, 0, 1]`.
`PE(1) = [sin 1, cos 1, sin 0.01, cos 0.01] = [0.84147, 0.54030, 0.01000, 0.99995]`.
`PE(2) = [sin 2, cos 2, sin 0.02, cos 0.02] = [0.90930, −0.41615, 0.02000, 0.99980]`.
Rotation from `pos=1` to `pos=3` with `k=2`, `w=1`: `cos 2 = −0.41615`, `sin 2 = 0.90930`.
`sin(3) = cos2·sin1 + sin2·cos1 = (−0.41615)(0.84147) + (0.90930)(0.54030) = 0.14112` ✓
`cos(3) = cos2·cos1 − sin2·sin1 = (−0.41615)(0.54030) − (0.90930)(0.84147) = −0.98999` ✓

**C3.** `V = 6`, `ε = 0.1`, correct `= 3`, pad `= 0`. Smoothing recipients: `V − 2 = 4`, each
`0.025`. So `q = [0, 0.025, 0.025, 0.9, 0.025, 0.025]`, summing to 1.

Logits `[0,0,0,10,0,0]`: `log-softmax` ≈ `[−10.0067, −10.0067, −10.0067, −0.0067, −10.0067, −10.0067]`.
Loss `= −(0.9·(−0.0067) + 4·0.025·(−10.0067)) = 0.0060 + 1.0007 = 1.0067`.

Logits `[0,0,0,2,0,0]`: `log-softmax` ≈ `[−2.3308, −2.3308, −2.3308, −0.3308, −2.3308, −2.3308]`.
Loss `= −(0.9·(−0.3308) + 4·0.025·(−2.3308)) = 0.2977 + 0.2331 = 0.5308`.

The **moderate** logit has lower loss. Unsmoothed, the extreme logit would win (loss `0.0067` versus
`0.3308`). That inversion is the whole mechanism: smoothing makes over-confidence *cost* something,
so the loss has a finite minimum.

---

# Section D — expected outcomes for the planted bugs

**D1 · `dim=-2`.** Fails immediately: `test_weights_form_a_probability_distribution_over_keys`,
`test_softmax_is_over_the_key_axis_not_the_query_axis`, both SDPA parity tests, the naive-transcription
test, and the `nn.MultiheadAttention` parity test. Note `test_softmax_is_over_the_key_axis` uses
`q_len ≠ k_len` specifically so this cannot hide behind a square matrix. **Would a training run still
show a falling loss? Yes** — that is the point. Columns summing to 1 still produces a differentiable
mixing operation, so the model learns *something* and the curve looks plausible. This bug is
detectable by test and nearly invisible from a loss curve.

**D2 · dropped causal mask.** Training loss falls faster and further than the correct model, plausibly
below `0.1`, because the task degenerates to copying visible target tokens. Greedy decoding collapses
to garbage. First test to fail:
`test_decoder_output_is_invariant_to_future_target_tokens`, then `test_overfits_a_tiny_batch` fails on
its *second* assertion (exact greedy match) while passing the loss assertion — which is precisely why
that test checks both.

**D3 · `labels == tgt_in`.** The model is trained to reproduce its own input, shifted zero. Loss
converges to roughly the label-smoothing floor (~`0.75` at `ε=0.1, V=70`) because copying is trivially
learnable. **Is it visible from the loss curve alone? No** — the curve looks like a healthy
convergence to a floor. It is visible from exact-match (0%) and from
`test_decoder_input_and_labels_are_offset_by_exactly_one`, which fails instantly. This is the strongest
argument in the pack for reporting free-running exact match alongside loss.

---

# Section F — notes on the prediction experiment

No answer key: the point is that **you** commit first. But two things to hold yourself to.

**On question 4, the parameter count is derivable, not guessable.** Pre-norm adds one `LayerNorm` to
the encoder and one to the decoder, each `2·d_model` parameters. At `d_model = 128` that is exactly
`4 × 128 = 512` extra parameters. If your prediction was vague, that is a gap — the count is
arithmetic you already have from B4-style reasoning.

**On question 3, be specific about the mechanism.** "Post-norm degrades more without warmup" is the
expected direction, and the reason is A4: post-norm rescales the identity path at every layer, so it
depends on warmup to survive the early mismatch between normalisation statistics and parameters. If
your prediction named the direction but not the mechanism, reread A4 before running.

If the measured outcome contradicts your prediction, write down **which belief** was wrong, not just
the corrected number. At two layers deep, the pre/post-norm difference may well be too small to
resolve — and "my prediction assumed a depth effect that does not appear at N=2" is a more valuable
note than a corrected decimal.
