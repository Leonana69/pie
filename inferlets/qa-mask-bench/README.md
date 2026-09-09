# qa-mask-bench

Prefix-cache **masking** experiment: how four KV-cache strategies trade off
accuracy vs prefill/decode latency when many documents are cached but each
request needs only a few of them — accessed at random positions.

## Setup

A **round** is 6 self-contained `(info, question, answer)` items. The infos are
fictional facts about uniquely named institutions (so the model can only answer
by attending to the *right* document — it can't guess). All 6 infos share one
cached base context (system prompt + the 6 infos).

Within a round we issue **asks**, each posing **two** of the round's questions at
once. The two infos those questions need sit at two random positions among the
cached 6 — that is the "random KV-cache access" we study. There are
`C(6,2)=15` possible pairs per round; we sample 5 of them. With 5 rounds that is
`5×5 = 25` asks = **50 questions** total.

## The four cases

All cases build the round's base once, then reuse it across the round's 5 asks.

| Case | Base cache | Per ask | Expected |
|---|---|---|---|
| **1** all-cached | system + all 6 infos (cascade prefill) | send only the 2 questions; attend to **all 6** infos | cheapest prefill, heaviest decode |
| **2** mask+drop (cascade) | system + 6 infos in page-aligned slots, cascade prefill | mask the 4 unused slots, `drop_masked_kv_pages`, attend to system + 2 used infos | cheap prefill, lighter decode |
| **3** mask+drop (exclusive) | same, but each info prefilled attending **only to system** | identical to case 2 at query time | like 2, but cleaner K/V → potentially better accuracy |
| **4** inline | system only | prefill the 2 used infos **inline** with the questions every ask | heaviest prefill, light decode |

Decode-latency note: PIE's attention kernel iterates over every KV page handed to
the forward pass and applies the mask on top — so masking *alone* does not reduce
decode cost. Cases 2/3 therefore mask **and** `drop_masked_kv_pages`, which
physically removes the unused pages so decode genuinely shrinks.

Cases 2 and 3 differ only in how the cached infos' K/V was computed (cascade vs
exclusive), which tests whether cross-info contamination during prefill hurts
quality when you later attend to a random subset.

## Architecture

```
inferlets/qa-mask-bench/
├── src/lib.rs     # universal inferlet (built once); input {case, round, max_tokens?}
├── Pie.toml       # inferlet manifest (required by the build + runtime)
├── prepare.py     # synthetic dataset generator → rounds.json (no network)
├── rounds.json    # generated dataset (one entry per round)
├── run.py         # driver: builds the wasm once, one `pie run` per (round, case)
├── eval.py        # EM/F1 + prefill/decode latency table
├── Makefile
└── results.jsonl  # appended output (one JSON row per ask)
```

`src/lib.rs` never changes (only the `{case, round}` input does), so `run.py`
builds it once to a `wasm32-wasip2` component and reuses the artifact for every
invocation.

## Run

The model is whatever `~/.pie/config.toml` targets (this host: `Qwen/Qwen3-0.6B`
on `cuda_native`); `pie run` selects it from the config, not the CLI.

```bash
cd inferlets/qa-mask-bench

# 1. Generate the dataset (deterministic, no download).
make prepare N_ROUNDS=5 N_PARAGRAPHS=6 ASKS_PER_ROUND=5

# 2. Run all rounds × 6 cases. First invocation builds the inferlet (~10s).
make run                       # add CASES=1,2,5 to subset; MAX_TOKENS=64 default

# 3. Score: accuracy + prefill/decode latency per case.
make eval
```

Smoke test on a single round (all 6 cases):

```bash
make smoke
```

`run.py` builds the wasm with `cargo build --release --target wasm32-wasip2`
(what `pie build` does for Rust) and invokes
`pie run --path <wasm> --manifest Pie.toml --input '{"case":…,"round":…}' --stdout`.
Pass `--wasm <path>` to reuse a prebuilt component, or `--pie <path>` to point at
a specific `pie` binary (default: the repo's `target/release/pie`).

## SDK-port notes

This inferlet was ported from a two-generations-old SDK. The current API has no
persistent context mask and no `drop_masked_kv_pages`; the experiment is
re-expressed as:

- **KV reuse** — the base context is `save`d once per round and `open`ed (forked)
  per ask (replaces the old `export_kv_pages` / `import_kv_pages`).
- **Masking** — a per-forward-pass `attention_mask` (one BRLE per query position)
  supplied on every prefill/decode pass (replaces `mask_token_range`).
- **Page drop** — masking an *entire page* of KV triggers the runtime's
  **page-trim** optimization, which physically excludes that page from the
  kernel. The slotted, page-aligned layout (cases 2/3) is built so masking an
  unused info masks whole pages → page-trim shrinks resident KV (replaces
  `drop_masked_kv_pages`). Cases 5/6 pack infos contiguously, so masked infos sit
  mid-page and are *not* trimmed — decode KV stays full, by design.

The query carries a trailing `/no_think` so Qwen3 emits the short `A1:/A2:`
answer directly instead of a reasoning block (the original targeted a
non-thinking model); models that don't recognize the marker ignore it.

## Output

`eval.py` prints a per-case table:

```
Case               Nq      EM      F1   Reg ms  Prefill ms  Decode ms   Tok µs  OutTok  Attend  Resident
1 all-cached       50   0.xxx   0.xxx     ....        ....       ....     ....    ....    ....      ....
2 mask+drop casc   50   ...
3 mask+drop excl   50   ...
4 inline           50   ...
```

`Attend` = tokens actually attended at decode; `Resident` = tokens physically in
the KV cache during decode (the figure that drives decode latency).
