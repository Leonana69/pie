# mask-serving

An OpenAI-compatible PIE HTTP inferlet that hides selected chat messages with
custom attention masks. When a hidden span covers complete KV pages, PIE's
page-trim optimization leaves those pages out of the decode kernel. Logical
token positions stay unchanged, so trimming is exactly equivalent to executing
the same attention mask without the optimization.

## Endpoint

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

The standard chat-completion fields supported here are `model`, `messages`,
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stream`,
`stop`, and `n=1`.

## Mask extensions

- `masking` (boolean, default `true`): enables or disables the supplied mask.
- `mask_message_indices` (array of integers, default `[]`): zero-based indices
  into `messages` that later visible prompt tokens and generated tokens must not
  attend to. `mask_messages` is accepted as an alias.

System/developer messages cannot be masked. Some model templates render the
system message and first conversational turn as one atomic token span; when
that happens, the service also rejects masking that bootstrap turn. Put a small,
always-visible example first and mask later demonstration pairs.

The response is a normal OpenAI `chat.completion` object with an additional
`pie_mask` object. It reports the resolved token ranges, masked-token count,
number of complete pages eligible for trimming, and prefill/decode timings.

## OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "Answer using the relevant examples."},
        {"role": "user", "content": "Always-visible bootstrap example."},
        {"role": "assistant", "content": "Bootstrap answer."},
        {"role": "user", "content": "Long irrelevant example input ..."},
        {"role": "assistant", "content": "Long irrelevant example output ..."},
        {"role": "user", "content": "The current request."},
    ],
    max_tokens=128,
    extra_body={"mask_message_indices": [3, 4]},
)

print(response.choices[0].message.content)
print(response.model_extra["pie_mask"])
```

Mask both sides of a demonstration turn. Long page-aligned or naturally
long messages provide the largest decode benefit. The current native CUDA pure-decode path trims complete hidden pages
but does not yet consume custom masks for partial boundary pages. Use
page-aligned hidden spans or a driver with custom-mask decode support when
strict boundary-token isolation is required.

## Correctness model

The mask is applied during the prompt prefill and every generation step. This
is important: masking only during decode would allow a later visible prompt
token's KV to encode information from a supposedly hidden earlier message.

Rows belonging to a hidden message may use ordinary causal attention while
their KV is built, because no later visible row is allowed to read those KV
positions. Visible rows always exclude every hidden range. The runtime may then
drop a page only when the whole page is hidden for every row in that forward
pass; it preserves position IDs and never drops the current write page.

With a driver that honors custom masks during decode, page trimming is
lossless relative to the declared masked-attention program. It does **not**
claim that hiding an example produces the same answer
as showing it—the prompt has intentionally changed.

Custom attention masks require a mask-aware driver. Bridges that advertise
`supports_user_attention_mask = false` are not suitable for this inferlet. The
native CUDA boundary-page limitation above also applies even though complete
hidden pages are trimmed correctly.

## Build

```bash
cargo build --manifest-path inferlets/mask-serving/Cargo.toml \
  --target wasm32-wasip2 --release
```

Serve the resulting component with the same HTTP setup used by
`openai-spec-service`.
