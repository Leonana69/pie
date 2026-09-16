# HTTP service validation — A3B BF16

2026-09-14. Service: `openai-plan-mask-service`; model: Qwen3-Coder-30B-A3B-Instruct BF16 on one H100 80 GB, native CUDA PIE source `1dd90602`.

## Outcome

The combined prior-plan/example-masking inferlet is available at `http://127.0.0.1:8000/v1`. It runs in tmux session `pie-plan-mask`; the engine control port is 18088. Startup command: `python3 scripts/serve_plan_mask_service.py` from the setup repository. [API and usage](README.md).

- All **40 S1 generation requests** (10 cases × four flag combinations) completed with HTTP 200 and normal stop, using a shared system/header/27-example KV snapshot.
- **40/40 outputs exactly matched the original combined benchmark** for the corresponding case and mode. This validates the port against the recorded inferlet behavior.
- Exact prior-plan on/off agreement: **5/10 without masking**, **8/10 with masking**, **13/20 total**. This reproduces the earlier benchmark's incomplete agreement; it does not establish exact output preservation or measure plan accuracy.
- **78 recorded HTTP requests** cover generation and API checks, including intentionally invalid requests with expected 400/404 responses. Nine Rust unit tests passed.
- **OpenAI Python SDK 3.13.0 passed**: model listing, `extra_body` flags, chat responses, cached-token usage, buffered SSE including its usage chunk, and structured error parsing.

## Verified behavior

Cache creation prefills the static prefix once; subsequent requests reuse 11,403 cached tokens. Repeating the same catalog returns the same ID and a cache hit. Changing system content creates a different ID. Tests cover selecting all examples, no examples, and a subset, then switching selection on the same cache. Concurrent requests retained their own selections.

The prior-plan and masking flags are independent. Drafts are target-verified in batches and rejected KV is truncated. A deliberately altered prior exercised both accepted and rejected drafts. Its output differed from its baseline, consistent with the unresolved output-preservation limitation; a following baseline request reproduced its original output, confirming that the shared base cache was unchanged in this check. Stop strings, a one-token cap, normal chat without a cache, cache deletion and missing-cache errors also passed.

The catalog retains the benchmark's cascade prefill: masking blocks direct attention to unselected spans during dynamic prefill and generation. It does not remove indirect influence already encoded in later cached examples. Streaming uses OpenAI SSE framing but buffers generation before sending chunks.

## Startup fixes

The initial HTTP health request closed before reaching inference because the launcher set `allow_network=false`. In this host, that setting omits the WASI HTTP linker bindings needed for incoming requests too. The working configuration sets `allow_network=true` and `network_allowed_hosts=[]`, providing the bindings while denying outbound destinations.

The inferlet also pins `wasip2=1.0.1` and uses Rust 1.90.0 so its HTTP imports and incoming-handler export are 0.2.4. Newer dependency/toolchain builds contained HTTP 0.2.12/0.2.9 imports. Those were observed while the HTTP linker was disabled; this test does not independently establish that each newer version would fail with the linker enabled. Pins make the tested build reproducible without changing the host.

The first test harness incorrectly required complete prior-plan/baseline output identity. After checking the differing outputs against the saved benchmark, the harness was corrected to record this known limitation as a diagnostic. The completed 40-request matrix was reused without changing the inferlet; edge checks were rerun. `matrix_reused` in the summary records this reuse. A fresh default test run executes the entire matrix and edge checks.

## Evidence and reproduction

Raw requests/responses, aggregate diagnostics, SDK output, build logs, and runtime configuration are in `/workspace/pie-runpod-setup/results/plan-mask-http-a3b-bf16-2026-09-14/`. These are functional checks with one matrix pass and additional edge probes; their timing metadata is not a controlled throughput comparison. The original S1 dataset and prior-plan proxy inputs were reused unchanged.

```bash
source env.sh
cargo +1.90.0 test --locked --manifest-path inferlets/openai-plan-mask-service/Cargo.toml
python3 scripts/test_plan_mask_service.py --output /tmp/plan-mask-http-retest
python3 scripts/test_plan_mask_openai_sdk.py --output /tmp/plan-mask-sdk-retest.json
```

The last command needs the OpenAI SDK in the test environment. It contacts only the local service. Engine and daemon restarts require cache recreation. The launcher remains restricted to the tested Qwen3 MoE family; it does not enable the unresolved Qwen3.8 hybrid masking/speculation paths.
