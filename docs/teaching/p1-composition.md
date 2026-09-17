# Project 1 as a composition — the Transformer in fourteen steps

Every claim below points at a function you can open. Where a number appears, it is either
arithmetic you can redo by hand or it is labelled with the run that produced it.

Fidelity: **tier E** for the mechanisms (implemented from the equations, verified against
independent oracles). The trained model is **tier R** at most. No comparison to the paper's WMT14
results is made anywhere.

---

## 1. The problem in plain language

You have a sequence and you want a different sequence. Not the same length, not the same alphabet,
and the answer's pieces are scattered through the input in an order that changes from example to
example.

Concretely, our task:

```
"March 3, 2019"              ->  "2019-03-03"
"Sunday, 3 Mar 2019"         ->  "2019-03-03"
"22nd December 1999"         ->  "1999-12-22"
```

To emit the very first output character, the model must find the year — which may be at the end, or
at the start. To emit character 5 it must find the month, which may be a word of three to nine
letters, or already a number. Nothing about "which input position matters now" is fixed.

Before 2017 the standard answer was a recurrent network: read the input one token at a time,
maintain a hidden state, then emit. That works, but every token must wait for the previous one, so
training cannot be parallelised along the sequence. The paper's claim is that you can throw
recurrence away entirely and get better results faster, if you let every position look directly at
every other position. The mechanism for "look directly at" is attention.

---

## 2. A small numerical example

Attention is a weighted average where the model chooses the weights. Take three positions with
two-dimensional vectors, and let one query ask about them.

```
q  = [1, 0]

k_0 = [1, 0]      v_0 = [10, 0]
k_1 = [0, 1]      v_1 = [ 0, 10]
k_2 = [1, 0]      v_2 = [ 5, 5]
```

Dot products, then divide by `sqrt(d_k) = sqrt(2) ≈ 1.4142`:

```
q·k_0 = 1   ->  0.7071
q·k_1 = 0   ->  0.0
q·k_2 = 1   ->  0.7071
```

Softmax over those three scores:

```
exp(0.7071) = 2.0281      exp(0) = 1.0      exp(0.7071) = 2.0281
sum = 5.0562
weights = [0.4011, 0.1978, 0.4011]
```

Output is the weighted sum of the values:

```
out = 0.4011*[10,0] + 0.1978*[0,10] + 0.4011*[5,5]
    = [4.011, 0] + [0, 1.978] + [2.006, 2.006]
    = [6.017, 3.984]
```

Three things are already visible. The query pulled hardest on positions 0 and 2 because their keys
pointed the same way as the query. Position 1 was not ignored, only down-weighted — softmax never
outputs exactly zero for a finite score. And the output is a blend, not a selection: attention
cannot pick one position, only concentrate on it.

Now the same query with `d_k = 128` instead of 2. If the components are unit-variance and
independent, the dot products have variance 128, so scores land around ±11 instead of ±1. Softmax of
`[11, 0, 11]` is roughly `[0.5, 0.00002, 0.5]` — nearly one-hot, and its gradient is nearly zero.
That is why the `sqrt(d_k)` divisor is in Equation 1 and not an afterthought.
`tests/test_p1_attention.py::test_scaling_keeps_score_variance_near_one` measures both halves of
this.

---

## 3. The necessary mathematical objects

Five, and no more.

**Equation 1 — scaled dot-product attention** (§3.2.1) → `attention.scaled_dot_product_attention`

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

**Multi-head attention** (§3.2.2) → `attention.MultiHeadAttention`

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Position-wise feed-forward** (§3.3) → `layers.PositionwiseFeedForward`

$$\mathrm{FFN}(x) = \max(0,\, xW_1 + b_1)W_2 + b_2$$

**Layer normalization with residual** (§3.1, §5.4) → `layers.SublayerConnection`

$$y = \mathrm{LayerNorm}\big(x + \mathrm{Dropout}(\mathrm{Sublayer}(x))\big),\qquad
\mathrm{LayerNorm}(z) = \frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}\odot\gamma + \beta$$

with $\mu,\sigma^2$ taken over the feature axis of each token, and $\sigma^2$ **biased**.

**Sinusoidal positional encoding** (§3.5) → `positional.sinusoidal_positional_encoding`

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Two more appear only in training:

**Noam schedule** (§5.3) → `optim.NoamSchedule`

$$lrate = d_{\text{model}}^{-0.5}\cdot\min\!\left(step^{-0.5},\; step\cdot warmup^{-1.5}\right)$$

**Label smoothing** (§5.4) → `optim.LabelSmoothingLoss`, with $q(\text{correct}) = 1-\epsilon_{ls}$.

---

## 4. Tensor shapes at every major boundary

Our trained configuration: `d_model = 128`, `h = 4`, `d_k = d_v = 32`, `d_ff = 512`,
2 encoder + 2 decoder layers, `V = 71`.

| Boundary | Shape |
|---|---|
| source ids | `(B, S)` |
| target-in ids (right-shifted) | `(B, T)` |
| after embedding | `(B, S, 128)` |
| after ×`sqrt(d_model)` and + PE | `(B, S, 128)` |
| `w_q(x)` | `(B, S, 128)` |
| after `split_heads` | `(B, 4, S, 32)` |
| scores `q @ k^T / sqrt(d_k)` | `(B, 4, S, S)` |
| attention weights (softmax over last axis) | `(B, 4, S, S)` |
| `weights @ v` | `(B, 4, S, 32)` |
| after `merge_heads` | `(B, S, 128)` |
| after `w_o` | `(B, S, 128)` |
| FFN hidden | `(B, S, 512)` |
| encoder memory | `(B, S, 128)` |
| cross-attention scores | `(B, 4, T, S)` ← the only non-square one |
| decoder output | `(B, T, 128)` |
| logits | `(B, T, 71)` |

The cross-attention row is the one worth memorising. `(T, S)` says: one row per output position,
one column per input position. Read row 4 and you are reading "where did output character 4 look in
the input". That tensor is what the demo's cross-attention view draws.

Masks broadcast against the score tensors:

| Mask | Shape | Broadcasts over |
|---|---|---|
| source padding | `(B, 1, 1, S)` | heads, queries |
| causal | `(1, 1, T, T)` | batch, heads |
| target combined | `(B, 1, T, T)` | heads |

---

## 5. One component implemented in isolation

`scaled_dot_product_attention`, in full:

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

if keep_mask is not None:
    empty_rows = fully_masked_rows(keep_mask)
    scores = scores.masked_fill(~keep_mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)

if empty_rows is not None and bool(empty_rows.any()):
    weights = torch.where(empty_rows.unsqueeze(-1).expand_as(weights),
                          torch.zeros_like(weights), weights)

output = torch.matmul(weights, value)
```

Four decisions in nine lines.

`transpose(-2, -1)` contracts over the feature axis, not the sequence axis, so the result is one
score per (query, key) pair.

`dim=-1` normalises over **keys**. Over queries instead and columns would sum to 1 rather than
rows — a bug that runs happily and trains a nonsense model.
`test_softmax_is_over_the_key_axis_not_the_query_axis` uses `q_len ≠ k_len` so the mistake cannot
hide behind a square matrix.

`-inf` rather than `-1e9`, because `exp(-inf)` is exactly `0`: forbidden positions receive exactly
no probability mass, and no renormalisation step is needed. `-1e9` leaks a little mass and is not
representable in fp16.

The fully-masked-row branch handles softmax over the empty set, which is undefined and shows up as
NaN. NaN in attention weights propagates into the gradient of every shared parameter and destroys
the model, so we write exact zeros and keep the failure local and findable.

---

## 6. How components connect

```
                    ┌──────────── encoder ────────────┐
source ids ─► embed ─► +PE ─► [self-attn ─► FFN] × 2 ─► memory
                                                          │
                                                          │ (keys and values)
                                                          ▼
target-in ─► embed ─► +PE ─► [masked self-attn ─► cross-attn ─► FFN] × 2 ─► logits
```

Two structural facts do most of the explanatory work.

**Attention is the only thing that moves information between positions.** The FFN, layer
normalization and the residual add all operate on one position at a time.
`test_ffn_does_not_mix_across_positions` proves this by perturbing position 3 and requiring every
other position's output to be bit-identical. So if you want to know how information travelled, look
at attention; there is nowhere else it could have gone.

**Every decoder layer attends to the same memory** — the output of the *last* encoder layer, not
its own depth-matched layer. This is why the encoder is computed once and reused for all decoding
steps, and `test_encoder_output_is_independent_of_the_target` pins it.

---

## 7. One complete forward pass

Input `"3 Mar 2019"`, `S = 10` characters, and we are about to produce output position 0.

1. **Tokenise.** Character level. 10 ids, no padding for a single sequence.
2. **Embed.** `(1, 10)` → `(1, 10, 128)`.
3. **Scale.** Multiply by `sqrt(128) ≈ 11.31`. Without this the embeddings, initialised at std
   `d_model^-0.5 ≈ 0.088`, are dwarfed by positional encodings bounded in `[-1, 1]`, and token
   identity is drowned by position.
4. **Add PE.** Same shape. Order information now present.
5. **Encoder layer 0, self-attention.** Project to q, k, v; split into 4 heads of 32; scores
   `(1, 4, 10, 10)`; softmax over keys; weighted sum of values; merge; `w_o`. Add the residual,
   then LayerNorm — post-norm, per §3.1.
6. **Encoder layer 0, FFN.** `128 → 512 → ReLU → 128`, per position. Residual, LayerNorm.
7. **Encoder layer 1.** The same again. Output is the memory, `(1, 10, 128)`.
8. **Decoder input.** Just `[BOS]`, so `T = 1`.
9. **Decoder self-attention.** `(1, 4, 1, 1)` scores. With one position the causal mask is
   vacuous — but it is still applied, and at step 7 it will not be vacuous.
10. **Cross-attention.** Queries from the decoder `(1, 1, 128)`, keys and values from the memory
    `(1, 10, 128)`. Scores `(1, 4, 1, 10)`: one row, ten columns. This row is where the model
    decides which input characters matter for output position 0. For a well-trained model it should
    concentrate on the year digits.
11. **Decoder FFN**, residual, LayerNorm.
12. **Project to logits.** `(1, 1, 128) @ W^T` with `W` the shared embedding matrix, giving
    `(1, 1, 71)`. Because weights are tied (§3.4), the logit for token *t* is the dot product of the
    hidden state with *t*'s own embedding — literally "how much does my current state look like
    token *t*".
13. **Argmax** → the first output character. Append and repeat from step 8 with `T = 2`.

---

## 8. Loss calculation

The shift is where the whole training scheme lives:

```
target      y      = "2019-03-03"
tgt_in             = [BOS, '2','0','1','9','-','0','3','-','0','3']
labels             = ['2','0','1','9','-','0','3','-','0','3', EOS]
```

`labels[i] == tgt_in[i+1]`, asserted by
`test_p1_data.py::test_decoder_input_and_labels_are_offset_by_exactly_one`. Combined with the causal
mask, position `i` predicts `labels[i]` from `tgt_in[0..i]` only — so one forward pass over an
11-token target trains eleven next-token problems at once. That is the parallelism recurrence could
not offer.

The loss is label-smoothed cross-entropy over non-padding positions, averaged by token count rather
than by tensor size — otherwise the reported loss depends on how sequences happened to be batched.

Two traps, both worth stating out loud:

*Getting the shift backwards* feeds the model the token it is meant to predict. Training loss
collapses towards zero and generation is garbage, because at inference the answer is not there to
copy.

*Comparing a smoothed loss with an unsmoothed one.* §5.4 says plainly that smoothing hurts
perplexity — the model is deliberately made less certain. So evaluation here defines its own
unsmoothed cross-entropy, computed separately from the training objective.

---

## 9. Gradient flow and parameter updates

Backward, in reverse:

logits → generator (which is the embedding matrix, so it receives gradient from *both* the output
projection and the token lookups) → decoder layers → **the memory**, where cross-attention sends
gradient back into the encoder → encoder layers → embeddings.

The residual connections matter here more than anywhere. Because $y = x + f(x)$, the derivative is
$I + f'(x)$: the identity term guarantees a gradient path around every sub-layer even when $f'$ is
badly conditioned. `test_residual_connection_gives_the_input_a_direct_gradient_path` uses a
sub-layer with exactly zero gradient and requires the input to still receive one.

Post-norm complicates this. Each residual sum is immediately normalised, so the identity path is
rescaled at every layer — and that is precisely why the Noam warmup exists. Early in training the
normalisation statistics and parameters are mismatched and a large learning rate diverges.
Pre-norm, $y = x + f(\mathrm{LN}(x))$, leaves an unrescaled identity path to the top;
`test_pre_norm_leaves_an_unnormalised_identity_path` demonstrates the difference with a zero
sub-layer.

Updates use Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$ (§5.3) — all three
differ from PyTorch's defaults. Gradients are clipped to norm 1.0; that is ours, not the paper's,
and it is labelled as such.

---

## 10. Training and checkpoint recovery

A checkpoint saves weights, optimizer moments, scheduler position, and the RNG states of
python/numpy/torch. The RNG states are not paranoia: dropout masks and shuffling come from them, so
a resume without them silently continues a *different* experiment, and the loss curve gets a hidden
discontinuity at the resume point.

`test_p1_training.py::test_resume_reproduces_the_next_step_loss_exactly` saves mid-run, reloads into
fresh objects, and requires the next five losses to match to `1e-12`. Its companion,
`test_omitting_rng_state_causes_divergence`, shows the first test is not vacuous.

---

## 11. Evaluation

Three metrics, defined before any value is quoted:

**`val_loss_nats_per_token`** — mean unsmoothed cross-entropy per non-padding target token.

**`token_accuracy_teacher_forced`** — fraction of target positions whose argmax is correct *given
the correct prefix*. The optimistic metric.

**`exact_match_free_running`** — fraction of examples where greedy decoding from BOS alone, with no
access to the target, reproduces the entire ISO string. The honest metric, and the headline.

Both accuracy figures are reported because the **gap between them** is the informative quantity: a
wide gap means the model predicts well one step at a time but compounds its own errors once it has
to consume its own output.

The split is drawn over **calendar dates, not rendered strings**, so every rendering of a date sits
in one split. The naive alternative leaks: the model would meet `"March 3, 2019"` in training and be
scored on `"3 Mar 2019"` having already memorised that date's answer, and validation accuracy would
measure memorisation while looking excellent.
`test_no_iso_date_appears_in_more_than_one_split` enforces it.

---

## 12. Inference and deployment

Greedy decoding, not beam search. The paper used beam 4 with length penalty 0.6 (§6.1); we do not,
and do not imply we do. Greedy suits a task with one correct output and makes the per-step
probability display honest, since the shown distribution is exactly the one the token came from.

Cost is `O(n²)` decoder work for `n` tokens, because the whole prefix is recomputed each step. A KV
cache would make it linear. It is not implemented, and no efficient-generation claim is made — at
`n ≤ 12` the quadratic term is irrelevant, and a second code path would be one more thing for the
browser demo to disagree with.

Deployment runs the forward pass in JavaScript in the visitor's browser, which bounds cost
structurally: there is no shared resource to exhaust. The risk is that the re-implementation drifts
from the trained model, so the export ships a parity fixture, the page checks itself against it on
load, and the measured deviation is displayed.

---

## 13. Failure diagnosis

A symptom-first table, from mistakes actually made or actively guarded against here:

| Symptom | Likely cause | How to confirm |
|---|---|---|
| Train loss → 0, generation is garbage | causal mask missing or transposed | perturb a future target token; earlier logits must not move |
| Loss stuck near `log V` | shift wrong; model asked to predict unseeable tokens | check `labels[i] == tgt_in[i+1]` |
| NaN loss after a few steps | fully-masked row, or `-inf` reaching softmax on every key | count `fully_masked_rows`; check masks constrain keys only |
| Val accuracy suspiciously high | split leakage | check the same date across splits |
| Loss diverges in the first hundred steps | post-norm with no warmup, or `lr` base ≠ 1.0 with `LambdaLR` | print the actual `lr` each step |
| Loss jumps at a resume point | RNG state not checkpointed | resume-determinism test |
| Attention looks uniform everywhere | `sqrt(d_k)` scaling applied twice, or logits collapsed | measure score variance |
| Token accuracy high, exact match low | error compounding under free running | compare the two metrics directly |

---

## 14. Performance trade-offs

**Attention is quadratic in sequence length.** Scores are `(B, h, S, S)`: at `S = 32` that is
trivial, at `S = 4096` it dominates everything. Project 5 is about making exactly this term cheaper
without changing its result.

**Most parameters are in the FFN.** Per layer, attention holds `4·d²` and the FFN `2·d·d_ff`. With
`d_ff = 4d` that is a 2:1 split in the FFN's favour. So "make the model smaller" usually means
"touch `d_ff`", while "make long sequences faster" means "touch attention".

**Heads are free, width is not.** `h` heads of `d_model/h` each cost the same total as one head of
full width, because the per-head dimension shrinks as `h` grows. Heads buy multiple attention
patterns at no FLOP cost — but each head gets a narrower subspace, so past some point they become
individually too weak.

**Measured on this machine** (run `env-bench-02`, contended, so a lower bound): the 0.53 M-parameter
configuration trains at 3,672–10,194 tokens/second using **2 threads**. Six threads is three to six
times *slower* at this size — the parallel-region overhead exceeds the work available. Large matmuls
do scale to 6 threads. Thread count is therefore a per-workload setting, not a global one.

---

## The three explanations

### 30 seconds

A Transformer turns one sequence into another by letting every output position look directly at
every input position and take a weighted average of what it finds — with the model choosing the
weights. Doing that in several "heads" at once lets it track several kinds of relationship
simultaneously. Because nothing has to wait for anything else, the whole thing trains in parallel,
which is why it replaced recurrent networks.

### Five minutes, at a whiteboard

Start with the problem: map `"March 3, 2019"` to `"2019-03-03"`. The pieces of the answer are
scattered through the input in positions that move between examples.

Draw three rows — queries, keys, values. Every position emits a key ("here is what I am") and a
value ("here is what I would contribute"). Every position also emits a query ("here is what I need").
Dot a query against all keys, softmax the results, and you have weights that sum to one. Take the
weighted average of the values. That is attention, and it is one matrix multiply, one softmax, one
more matrix multiply.

Two annotations. Divide the scores by `sqrt(d_k)`, because in high dimensions dot products grow like
`sqrt(d_k)` and a saturated softmax has no usable gradient. And do it `h` times in parallel on
narrower slices — one head can track "which characters are digits", another "where does the month
word end".

Now the problem: shuffle the inputs and nothing above changes, because no step depends on position.
So add a fixed sinusoidal pattern to each embedding. Sinusoids because shifting position by `k`
rotates each sine/cosine pair by a fixed angle, so "look three back" is a single linear map the
model can learn once and apply everywhere.

Stack it: encoder reads the source and produces a memory. Decoder attends to what it has already
written — masked, so it cannot see the future — then attends to the encoder's memory, then applies a
small per-position MLP. Wrap every sub-layer in `x + f(x)` followed by normalization: the skip keeps
gradients alive, the normalization keeps scales sane.

Finally, the training trick: feed the decoder the target shifted right by one and mask the future.
Then a single pass over an eleven-token target trains eleven next-token predictions at once.

### Technical walkthrough

Sections 1–14 above. Read §3 for the objects, §4 for the shapes, §5 for the one function that
matters most, §8 for why the shift and the mask are the same idea seen twice, and §13 when
something breaks.
