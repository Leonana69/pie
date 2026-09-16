# OpenAI HTTP service for prior plans and example masking

This ports `pie-runpod-setup/inferlets/s1-prior-plan-mask` into PIE's existing WASI HTTP service pattern. It exposes `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`, plus prompt-cache creation and deletion. The service uses the same suffix matcher, greedy draft verification, rejected-KV rollback, and masks on every dynamic-prefill and verification row as the combined benchmark.

## Build this inferlet

```bash
cd /workspace/pie
cargo +1.90.0 test --locked --manifest-path inferlets/openai-plan-mask-service/Cargo.toml
cargo +1.90.0 build --locked --manifest-path inferlets/openai-plan-mask-service/Cargo.toml --target wasm32-wasip2 --release
```

The following deployment and integration-test commands live in the companion `/workspace/pie-runpod-setup` repository, which also contains the original benchmark and raw S1 results. Its launcher builds its own copy of this inferlet. To install the artifact built from this checkout on that running engine:

```bash
cd /workspace/pie-runpod-setup
source env.sh
python3 scripts/launch_spec_daemon.py --ws-url ws://127.0.0.1:18088 --http-port 8002 \
  --inferlet openai-plan-mask-service@0.1.0 \
  --wasm /workspace/pie/inferlets/openai-plan-mask-service/target/wasm32-wasip2/release/openai_plan_mask_service.wasm \
  --manifest /workspace/pie/inferlets/openai-plan-mask-service/Pie.toml
```

Use an unused HTTP port when starting a second daemon. The example's base URL is `http://127.0.0.1:8002/v1`.

## Start

From `/workspace/pie-runpod-setup`:

```bash
python3 scripts/serve_plan_mask_service.py
```

The launcher builds the inferlet and starts the installed native CUDA engine with the local Qwen3-Coder-30B-A3B-Instruct BF16 checkpoint. Defaults: HTTP `http://127.0.0.1:8000/v1`, engine control port `18088`. Override with `--http-port` and `--engine-port`. Stop the foreground process with Ctrl-C; it stops its engine too. It refuses occupied ports.

Prerequisites are the workspace's PIE source/SDK, Rust 1.90.0 with `wasm32-wasip2`, CUDA libraries, and the saved native binary. The binary's filename ends in `qwen38-fp8` because that was the build task; this launcher loads **A3B BF16**, not Qwen3.8. `--binary` and `--model-path` override the paths. The launcher checks for the tested Qwen3 MoE model family and refuses the Qwen3.8 hybrid path, whose masking and prior-plan failures remain unresolved.

The dependency lock pins `wasip2=1.0.1` (WASI HTTP 0.2.4), matching this host. Rust 1.90.0 also keeps the standard-library HTTP bindings at 0.2.4; Rust 1.97.1 introduces an additional HTTP 0.2.9 import even with the crate pin. Install the compatible toolchain with `rustup toolchain install 1.90.0 --profile minimal --target wasm32-wasip2`. Newer 1.0.x releases import HTTP 0.2.12; the pins keep this build on the host's declared incoming-handler interface. Initial missing-binding errors also exposed the need to enable the HTTP linker through the runtime setting above.

The native host must include daemon-scoped scratch storage and serialized HTTP request handling, already present in local PIE source `1dd90602`. The launcher enables `runtime.allow_fs` and assigns a service scratch directory. It also sets `allow_network=true` with an empty outbound host allowlist: this host omits all HTTP bindings when `allow_network=false`, including the types needed to serve incoming requests. The inferlet does not make outbound requests. Each HTTP request uses a fresh WASM instance; cache metadata survives in daemon scratch and KV snapshots live in the model runtime.

## Cache the static prompt and examples once

```bash
curl http://127.0.0.1:8000/v1/prompt_caches \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "system": "Generate a JSON action plan for the robot.",
    "header": "Examples:\n",
    "examples": [
      "Task: turn left. Plan: {\"queue\":[{\"act\":\"turn_left(90)\"}]}",
      "Task: bark. Plan: {\"queue\":[{\"act\":\"bark()\"}]}"
    ]
  }'
```

The response contains `id` (`pm-…`), `example_count`, `prompt_tokens`, `cache_hit`, and `build_ms`. Repeating identical model/system/header/examples reuses the snapshot. Changing content or example order creates a different ID. IDs are immutable; dynamic information, selected indexes, and previous plans do not change the cache.

## Generate with either optimization enabled or disabled

Use the returned ID:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "cache_id": "pm-REPLACE_WITH_RETURNED_ID",
    "messages": [{"role":"user","content":"Current task: turn left 90 degrees. Position: hallway."}],
    "selected_examples": [0],
    "masking": true,
    "prior_plan": true,
    "previous_plan": "{\"queue\":[{\"act\":\"turn_left(90)\"}]}",
    "temperature": 0,
    "max_tokens": 256
  }'
```

With the OpenAI Python SDK, extensions go in `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Current task: turn left 90 degrees."}],
    temperature=0,
    max_tokens=256,
    extra_body={
        "cache_id": cache_id,  # From POST /v1/prompt_caches
        "selected_examples": [0],
        "masking": True,
        "prior_plan": True,
        "previous_plan": previous_plan,
    },
)
print(response.choices[0].message.content)
```

| Request field | Behavior |
|---|---|
| `masking` | Default `false`. When true, dynamic information and generated tokens can directly attend only to selected catalog examples. `enable_masking` is an alias. |
| `prior_plan` | Default `false`. When true, draft from `previous_plan` and verify with the target model. `enable_prior_plan` is an alias. |
| `previous_plan` | Required nonempty text when prior-plan is enabled. It is a draft source, not an addition to the prompt. The service does not silently reuse another request's plan. |
| `selected_examples` | Zero-based catalog indexes. Omitted/null selects all; `[]` selects none. Duplicate indexes are deduplicated; invalid indexes return HTTP 400. With masking off, all examples remain visible, regardless of this selection. |
| `cache_id` | Cached system/header/catalog. With this field, send exactly one user message containing dynamic information. Without it, ordinary text chat with system/developer/user/assistant messages works, but example masking is unavailable. |
| `draft_len`, `match_tokens` | Defaults 8 and 2; each in 1–32. Same matcher as the benchmark. |
| `temperature`, `top_p` | Default greedy temperature 0. Prior-plan requires greedy decoding; when disabled, Top-P sampling is available. |
| `max_tokens`, `max_completion_tokens` | Default 256, range 1–8192. `max_completion_tokens` takes precedence. |
| `stop` | String or list of strings. Stops during generation, including within an accepted draft batch. |
| `stream` | Default false. True emits OpenAI SSE framing, buffered until generation finishes, as in the existing HTTP inferlet. |

`stream_options={"include_usage":true}` adds the standard final SSE usage chunk with empty choices.

Only text chat and `n=1` are supported. Unsupported fields such as tool calling, image content, and logprobs are rejected rather than silently ignored. Unknown model names return HTTP 404. Responses include normal `choices` and `usage`, plus `pie` metadata for flags, selected indexes, cache reuse, prefill/decode timing, and accepted/rejected drafts. `usage.prompt_tokens_details.cached_tokens` reports reused prefix tokens. Completion usage counts generated non-EOS tokens, including generated stop-string text that is omitted from returned content.

## Prior-plan output agreement

Every draft is checked against the target model's batched greedy picks, but **exact output preservation relative to single-token baseline decoding is not established on this runtime**. The original A3B BF16 benchmark matched 15/30 unmasked pairs and 24/30 masked pairs. The HTTP port retains this behavior; its waypoint-sequence baseline and prior-plan outputs matched the original benchmark exactly, including their difference from each other. Treat agreement as a diagnostic, not an accuracy score. The precise runtime cause has not been isolated. Both optimizations default to off.

## Cache lifetime and masking semantics

A cache contains the system prompt, header, and the full catalog prefetched in order, matching the ablation's contiguous cascade layout. Requests open an independent copy of the saved context; dynamic tokens and generated output never overwrite the base snapshot. Switching either flag or the selection reuses that same base.

Masking applies to direct attention from dynamic and generated tokens to unselected example spans. **It does not erase indirect influence already encoded in later examples during the shared catalog prefill.** This preserves the existing benchmark semantics; it is not equivalent to reconstructing the prompt with only selected examples or a mechanism for isolating confidential examples. The system/header always remain visible.

The daemon keeps at most 16 snapshots and 65,536 total cached tokens, evicting least-recently-used entries before building another cache. Recreate an evicted or expired ID by posting its original cache payload. A missing registry entry returns 404; an expired runtime snapshot returns 409. Engine/daemon restarts require cache recreation. Cache contents are shared by clients of this daemon, as in the existing service; cache IDs are references, not credentials.

```bash
curl -X DELETE http://127.0.0.1:8000/v1/prompt_caches/pm-REPLACE_WITH_RETURNED_ID
```

## Verification

```bash
source env.sh
cargo +1.90.0 test --manifest-path inferlets/openai-plan-mask-service/Cargo.toml
python3 scripts/test_plan_mask_service.py \
  --base-url http://127.0.0.1:8000 \
  --output /tmp/plan-mask-http-test
```

The integration suite uses the same 10-case S1 catalog and prior-plan inputs as the saved A3B ablation. It checks all four flag combinations, records exact output agreement and agreement with the original benchmark, and checks KV cache reuse, all/none/subset selection, draft rejection, stop strings, token limits, SSE, errors, concurrent selection isolation, and cache deletion. It does not grade plan accuracy. Per-request inputs, outputs, timings and the final summary are saved in the chosen output folder.

For an additional SDK compatibility check, install the OpenAI Python SDK in your test environment and run `python3 scripts/test_plan_mask_openai_sdk.py --output /tmp/plan-mask-sdk.json`. The service itself does not need the OpenAI SDK or an external API key.
