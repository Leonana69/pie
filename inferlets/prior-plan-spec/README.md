# Prior-plan speculative decoding

This inferlet tests whether a plan returned for a previous, similar request can
reduce decode latency for a new request. The previous plan is not included in
the model prompt. A custom Pie speculator uses it only to propose tokens, and
the target model verifies every proposal before it becomes output.

## Build and configure from a Pie checkout

From the repository root, build the Metal/CPU-capable server and the inferlet:

```bash
cargo build -p pie-server --release
cargo build --manifest-path inferlets/prior-plan-spec/Cargo.toml \
  --target wasm32-wasip2 --release
```

Create a disposable config and point it at a locally cached 7–8B model:

```bash
target/release/pie config init --path /tmp/pie-prior-plan.toml
target/release/pie config set --path /tmp/pie-prior-plan.toml \
  model.0.hf_repo Qwen/Qwen3-8B

# Only needed when the model is not already in the Hugging Face cache.
target/release/pie model download Qwen/Qwen3-8B
```

`Qwen/Qwen2.5-7B-Instruct` is another supported 7B option. On Apple Silicon,
the default portable driver selects Metal when the server was built on macOS.

## Run the A/B comparison

```bash
target/release/pie run \
  --config /tmp/pie-prior-plan.toml \
  --path inferlets/prior-plan-spec/target/wasm32-wasip2/release/prior_plan_spec.wasm \
  --manifest inferlets/prior-plan-spec/Pie.toml \
  -- --mode both
```

Run it with your own adjacent requests:

```bash
target/release/pie run \
  --config /tmp/pie-prior-plan.toml \
  --path inferlets/prior-plan-spec/target/wasm32-wasip2/release/prior_plan_spec.wasm \
  --manifest inferlets/prior-plan-spec/Pie.toml \
  -- \
  --mode both \
  --previous-plan $'1. Inspect the existing export flow.\n2. Add CSV serialization.\n3. Add metrics and tests.' \
  --current-request 'Add JSON export to the same reporting service.' \
  --draft-len 2 \
  --match-tokens 2
```

The returned object contains baseline and speculative prefill/decode latency,
verifier-step counts, draft acceptance, and a comparison. Check these fields:

- `comparison.outputs_match` must be `true` before treating a speedup as
  lossless.
- `speculated.draft_acceptance_rate` shows how reusable the old plan was.
- `speculated.average_tokens_per_step` should be greater than the baseline's
  value of roughly 1.0 when drafts are useful.
- `comparison.decode_speedup` is the primary latency result. Values above 1.0
  are faster; repeat runs after a warm-up and compare medians.

For less order-sensitive timings, collect `--mode baseline` and
`--mode speculated` in separate invocations. Compare their returned `text`
fields as well as their decode medians: some backends can choose a different
greedy token for larger multi-token verification shapes because of numerical
kernel differences. If that happens, shorten `--draft-len` and rerun the
correctness check. A larger `--match-tokens` is safer for plans with repeated
phrases; a smaller value realigns sooner after changed words.
