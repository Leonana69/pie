# Prior-plan speculative decoding

This inferlet tests whether a plan returned for a previous, similar request can
reduce decode latency for a new request. The previous plan is not included in
the model prompt. A custom Pie speculator uses it only to propose tokens, and
the target model verifies every proposal before it becomes output.

Run the built-in A/B example from the repository root:

```bash
./pie-serve run inferlets/prior-plan-spec/src/lib.rs \
  --model Qwen/Qwen3-8B -- --mode both
```

Run it with your own adjacent requests:

```bash
./pie-serve run inferlets/prior-plan-spec/src/lib.rs \
  --model Qwen/Qwen3-8B -- \
  --mode both \
  --previous-plan $'1. Inspect the existing export flow.\n2. Add CSV serialization.\n3. Add metrics and tests.' \
  --current-request 'Add JSON export to the same reporting service.' \
  --draft-len 8 \
  --match-tokens 2
```

The returned object contains baseline and speculative prefill/decode latency,
verifier-step counts, draft acceptance, and a comparison. Check these fields:

- `comparison.outputs_match` should be `true`. Greedy speculative decoding must
  preserve the verifier model's result.
- `speculated.draft_acceptance_rate` shows how reusable the old plan was.
- `speculated.average_tokens_per_step` should be greater than the baseline's
  value of roughly 1.0 when drafts are useful.
- `comparison.decode_speedup` is the primary latency result. Values above 1.0
  are faster; repeat runs after a warm-up and compare medians.

Use `--mode baseline` and `--mode speculated` in separate invocations when you
want independent timing samples. A larger `--match-tokens` is safer for plans
with repeated phrases; a smaller value realigns sooner after changed words.
