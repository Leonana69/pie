# openai-spec-service

A PIE 0.4.0 WASI HTTP inferlet exposing OpenAI-compatible chat completions
with a persistent dictionary of model-approved responses.

## Endpoints

- POST /v1/chat/completions
- GET /v1/models
- POST /v1/speculation/clear
- POST /v1/prefix_cache/clear
- GET /health

Setting stream to true uses OpenAI SSE framing, but the current implementation
buffers generation before sending the chunks.

With the OpenAI Python SDK, pass PIE extensions through extra_body:

    from openai import OpenAI

    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="unused",
    )
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "..."}],
        temperature=0,
        extra_body={"speculation": True, "spec_clear": False},
    )

## Request extensions

- speculation (boolean, default true): enable target-verified dictionary
  drafts. speculative_decoding is accepted as an alias.
- spec_clear (boolean, default false): clear the selected dictionary before
  this request.
- spec_key (string, default default): independent dictionary namespace.
- spec_store (boolean, default true): retain this response for later calls.
- spec_draft_len (1 to 32, default 8): maximum tokens in one draft.
- spec_match_tokens (1 to 32, default 2): generated suffix length used for
  dictionary lookup.
- spec_max_entries (default 32) and spec_max_dictionary_tokens (default
  32768): state bounds.
- prefix_cache (boolean, default true): reuse the longest saved KV snapshot
  that is an exact token prefix of the new prompt.
- prefix_cache_clear (boolean, default false): clear the selected prompt-cache
  namespace before this request.
- prefix_cache_key (string, default default): independent prompt-cache
  namespace.
- prefix_cache_store (boolean, default true): save the selected prompt prefix
  as a KV snapshot for later requests.
- prefix_cache_prefix_tokens (optional): cache exactly this many tokens from
  the start of the rendered chat prompt. Use this when the prompt has a stable
  prefix followed by a dynamic suffix. When omitted, the entire rendered
  prompt is cached for backward compatibility. The value must be between 1
  and the response's usage.prompt_tokens value.
- prefix_cache_max_entries (default 32) and prefix_cache_max_tokens (default
  32768): LRU bounds for saved prompt snapshots.

Dictionary speculation currently requires greedy decoding (temperature 0).
When speculation is false, ordinary Top-P sampling remains available.

The response is a normal OpenAI chat.completion object plus pie_speculation and
pie_prefix_cache. The latter reports the selected prefix_tokens boundary,
whether a KV snapshot hit, how many prompt tokens were reused, how many were
actually prefilled, and the bounded registry size. A client may clear either
cache without generating:

    POST /v1/speculation/clear
    {"spec_key": "my-test"}

    POST /v1/prefix_cache/clear
    {"prefix_cache_key": "my-test"}

## Persistence model

PIE 0.4 creates a fresh WASM instance per HTTP request. The accompanying host
patch mounts one daemon-scoped /scratch directory into every request instance
and serializes requests for that daemon. This keeps clear/read/generate/write
atomic and allows later requests to reuse earlier approved tokens. One-shot PIE
processes keep their original isolated, auto-cleaned scratch directories.

Prompt KV snapshots live in the model runtime and are keyed by a SHA-256 digest
of the cache schema, model, namespace, and exact prompt token sequence. The
scratch registry survives HTTP request instances; stale registry entries after
an engine restart are discarded automatically on the next request.

Drafts are never emitted without target-model verification. The implementation
is lossless relative to PIE greedy decoding; it is not a second draft model.
