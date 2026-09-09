# qa-mask-bench — results

Model: `Qwen/Qwen2.5-72B-Instruct-AWQ`. Dataset: 5 rounds × 6 synthetic
self-contained facts; 5 random 2-question asks per round = **25 asks / 50
questions** per case. KV page size 16. Latency averaged per ask.

## Headline table

| Case | EM | F1 | Reg ms | Prefill ms (TTFT) | Decode ms | µs/token | Attend tok | Resident tok |
|---|---|---|---|---|---|---|---|---|
| 1 all-cached (cascade) | **1.000** | **1.000** | 660 | 60 | 493 | 31633 | 648 | 648 |
| 2 mask+drop (cascade)  | 0.660 | 0.660 | 867 | 59 | 493 | 31473 | 317 | 340 |
| 3 mask+drop (exclusive)| 0.660 | 0.660 | 868 | 59 | 494 | 31543 | 317 | 340 |
| 4 inline               | **1.000** | **1.000** | 92 | **194** | 492 | 31549 | 317 | 317 |
| 5 pure-mask (cascade)  | **1.000** | **1.000** | 657 | 60 | 493 | 31641 | 317 | 648 |
| 6 pure-mask (exclusive)| 0.700 | 0.716 | 653 | 59 | 497 | 31621 | 317 | 648 |

Cases 1–4 are the originally requested strategies; 5–6 were added to isolate the
cause of the case 2/3 accuracy loss.

## Verdict on the original hypotheses

1. **"Case 4 has the biggest prefill"** — ✅ confirmed. Inline prefill of the two
   used infos every ask costs ~194 ms TTFT vs ~60 ms for the cached cases (which
   prefill only the two questions).

2. **"Cases 2/3 give a slight decode improvement"** — ❌ not at this scale.
   Decode is ~493 ms for *every* case even though case 1 carries 648 resident KV
   tokens vs ~320–340 for the masked cases. A 72B decode step is **weight-bound**:
   attention over a few hundred KV tokens is negligible next to the FFN. The
   per-token cost (~31.5 ms) is identical across cases. The masking/drop decode
   win only appears when the cached context is large (thousands of tokens, i.e.
   long documents), not ~80-token facts.

3. **Cascade vs exclusive prefill** — exclusive prefill (isolating each info to
   attend only to the system prompt) **hurts** accuracy: pure-mask cascade = 1.00
   vs pure-mask exclusive = 0.70. Cascade contamination from neighbouring facts
   did not hurt here, while isolating an info from the positions it physically
   occupies did.

## The key finding: mask+drop corrupts multi-token answers

Cases 2/3 dropped to EM 0.66. The failure mode is specific — **multi-token name
answers get their word boundaries mangled; single-token numbers are unharmed**:

```
gold:    'Doran Drummel'   'Karsil Hallow'
case 1:  'Doran Drummel'   'Karsil Hallow'   ✓ (contiguous)
case 2:  'Dorandrummel'    'Kars Silhallow'  ✗ (mask+drop)
case 3:  'Dorandrummel'    'Kars Silhallow'  ✗ (mask+drop)
case 4:  'Doran Drummel'   'Karsil Hallow'   ✓ (inline, contiguous)
case 5:  'Doran Drummel'   'Karsil Hallow'   ✓ (pure-mask, contiguous)
```

Mechanism: RoPE is applied to K **before** it is written to the paged KV cache
(`qwen2.py`: `apply_rope_pos_ids_inplace` → `append_paged_kv_cache`), so each
key is stored *pre-rotated* at the position it was prefilled at and **cannot be
renumbered afterwards**. To make `drop_masked_kv_pages` work, each info is padded
to a page boundary; that padding inserts masked-but-position-consuming tokens
**between** infos. Even an ~11-token gap perturbs the model enough to drop spaces
inside multi-token names. Tightening slot padding (uniform `max+96` → per-info
page boundary) only moved EM 0.50 → 0.66 — the gap is intrinsic to slot+drop.

**Pure masking avoids it entirely (case 5 = 1.00):** pack the infos contiguously
like case 1, then mask the unused ones at query time *without* dropping. The used
infos and the query keep the exact positions they have in case 1, so there is no
position perturbation. Because decode is weight-bound here, *not* dropping the
masked pages costs no decode time — so pure masking is strictly better than
mask+drop at this scale: same latency, full accuracy.

## Recommendation

- For short cached contexts (this regime): **cache everything (case 1) or
  pure-mask the distractors (case 5)** — both 1.00 accuracy, ~60 ms prefill,
  ~493 ms decode. Avoid mask+drop (accuracy cost, no decode benefit) and avoid
  inline (3× prefill).
- To actually exercise the decode/drop benefit, rerun with **long infos**
  (≥1k tokens each) so the KV cache is large enough that attention — not the
  FFN — drives decode. Then mask+drop's resident-KV reduction should translate
  into real decode savings, and the position-gap accuracy cost can be weighed
  against it.
- Prefer **cascade** over **exclusive** prefill for cached infos.
