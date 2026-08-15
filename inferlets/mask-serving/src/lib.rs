//! OpenAI-compatible chat completions with message-level KV masking.
//!
//! `mask_message_indices` identifies messages that should be invisible to all
//! later, unmasked prompt tokens and to every generated token. The inferlet
//! supplies the mask during prefill as well as decode, which prevents hidden
//! messages from leaking through the KV of later prompt tokens. Pie's runtime
//! can then omit fully-masked KV pages from decode kernels (page-trim) without
//! changing the logical masked-attention result.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use inferlet::{Context, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};
use serde_json::json;
use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};
use wstd::io::AsyncRead;

const DEFAULT_MAX_TOKENS: usize = 256;
/// Keep sampled prefill rows within the CUDA driver logit-row capacity.
/// Intermediate chunks do not sample, so their KV can be committed before
/// the final chunk requests the first completion token.
const PREFILL_CHUNK_TOKENS: usize = 512;

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    n: Option<usize>,

    /// Pie extension: apply `mask_message_indices` when true.
    #[serde(default = "default_true")]
    masking: bool,
    /// Pie extension: zero-based indices into `messages` to hide.
    #[serde(default, alias = "mask_messages")]
    mask_message_indices: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type", default)]
    _kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn values(&self) -> Vec<&str> {
        match self {
            Self::One(value) => vec![value.as_str()],
            Self::Many(values) => values.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct TokenRange {
    start: u32,
    end: u32,
}

impl TokenRange {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start: start as u32,
            end: end as u32,
        }
    }

    fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    fn contains(self, position: u32) -> bool {
        self.start <= position && position < self.end
    }
}

struct RenderedPrompt {
    token_ids: Vec<u32>,
    message_ranges: Vec<Option<TokenRange>>,
    /// A system message can be folded into the first conversation turn by the
    /// model's template. That turn is atomic and cannot be masked safely.
    atomic_with_system: Vec<bool>,
}

#[derive(Debug, Serialize)]
struct MaskMetadata {
    enabled: bool,
    requested_message_indices: Vec<usize>,
    masked_message_indices: Vec<usize>,
    masked_token_ranges: Vec<TokenRange>,
    masked_tokens: usize,
    page_size: u32,
    fully_masked_pages: usize,
    trimmed_tokens_per_decode: usize,
    prompt_tokens: usize,
    prefill_ms: u128,
    decode_ms: u128,
    total_ms: u128,
    decode_tokens_per_second: f64,
}

struct GenerationResult {
    text: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    hit_max_tokens: bool,
    page_size: u32,
    masked_ranges: Vec<TokenRange>,
    prefill_elapsed: Duration,
    decode_elapsed: Duration,
}

#[wstd::http_server]
async fn main(mut req: Request<IncomingBody>, responder: Responder) -> Finished {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    match (method, path.as_str()) {
        (Method::POST, "/v1/chat/completions") | (Method::POST, "/chat/completions") => {
            let body = match read_body(req.body_mut()).await {
                Ok(body) => body,
                Err(message) => return error_response(responder, 400, &message, None).await,
            };
            handle_chat_completions(body, responder).await
        }
        (Method::GET, "/v1/models") => models_response(responder).await,
        (Method::GET, "/health") => json_response(responder, 200, json!({"status": "ok"})).await,
        (Method::GET, "/") => {
            json_response(
                responder,
                200,
                json!({
                    "name": "Pie mask-serving service",
                    "version": "0.1.0",
                    "endpoints": [
                        "POST /v1/chat/completions",
                        "GET /v1/models",
                        "GET /health"
                    ],
                    "extensions": ["masking", "mask_message_indices"]
                }),
            )
            .await
        }
        (Method::OPTIONS, _) => cors_response(responder).await,
        _ => error_response(responder, 404, "Endpoint not found", None).await,
    }
}

async fn handle_chat_completions(body: Vec<u8>, responder: Responder) -> Finished {
    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error_response(
                responder,
                400,
                &format!("Invalid JSON request: {error}"),
                None,
            )
            .await;
        }
    };

    if let Err((message, param)) = validate_request(&request) {
        return error_response(responder, 400, &message, Some(param)).await;
    }

    let models = runtime::models();
    let Some(model_name) = models.first().cloned() else {
        return error_response(responder, 503, "No models are configured", None).await;
    };
    let model = match Model::load(&model_name) {
        Ok(model) => model,
        Err(error) => {
            return error_response(
                responder,
                500,
                &format!("Failed to load model: {error}"),
                None,
            )
            .await;
        }
    };

    let rendered = match render_prompt(&model, &request.messages) {
        Ok(rendered) => rendered,
        Err(error) => {
            return error_response(
                responder,
                400,
                &format!("Invalid prompt: {error}"),
                Some("messages"),
            )
            .await;
        }
    };

    let (masked_message_indices, requested_ranges) = if request.masking {
        match resolve_masked_ranges(&request, &rendered) {
            Ok(resolved) => resolved,
            Err((message, param)) => {
                return error_response(responder, 400, &message, Some(param)).await;
            }
        }
    } else {
        (Vec::new(), Vec::new())
    };

    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = request.temperature.unwrap_or(0.0);
    let top_p = request.top_p.unwrap_or(1.0);

    let mut generation = match generate(
        &model,
        &rendered.token_ids,
        &requested_ranges,
        max_tokens,
        temperature,
        top_p,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return error_response(responder, 500, &format!("Generation failed: {error}"), None)
                .await;
        }
    };

    if let Some(stop) = &request.stop
        && let Some(index) = earliest_stop(&generation.text, stop)
    {
        generation.text.truncate(index);
        generation.token_ids = model.tokenizer().encode(&generation.text);
        generation.hit_max_tokens = false;
    }

    let completion_tokens = generation.token_ids.len();
    let total_elapsed = generation.prefill_elapsed + generation.decode_elapsed;
    let fully_masked_pages = count_fully_masked_pages(
        &generation.masked_ranges,
        generation.prompt_tokens as u32,
        generation.page_size,
    );
    let metadata = MaskMetadata {
        enabled: !generation.masked_ranges.is_empty(),
        requested_message_indices: request.mask_message_indices.clone(),
        masked_message_indices,
        masked_token_ranges: generation.masked_ranges.clone(),
        masked_tokens: generation
            .masked_ranges
            .iter()
            .map(|range| range.len() as usize)
            .sum(),
        page_size: generation.page_size,
        fully_masked_pages,
        trimmed_tokens_per_decode: fully_masked_pages * generation.page_size as usize,
        prompt_tokens: generation.prompt_tokens,
        prefill_ms: generation.prefill_elapsed.as_millis(),
        decode_ms: generation.decode_elapsed.as_millis(),
        total_ms: total_elapsed.as_millis(),
        decode_tokens_per_second: completion_tokens.saturating_sub(1) as f64
            / generation.decode_elapsed.as_secs_f64().max(1e-9),
    };

    let id = completion_id(&body);
    let created = now_unix_secs();
    let finish_reason = if generation.hit_max_tokens {
        "length"
    } else {
        "stop"
    };
    let response_model = request.model.as_deref().unwrap_or(&model_name);

    if request.stream {
        streaming_response(
            responder,
            &id,
            created,
            response_model,
            &generation.text,
            finish_reason,
            &metadata,
        )
        .await
    } else {
        let payload = json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generation.text
                },
                "logprobs": null,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": generation.prompt_tokens + completion_tokens
            },
            "pie_mask": metadata
        });
        json_response_with_mask_headers(responder, 200, payload, &metadata).await
    }
}

async fn generate(
    model: &Model,
    prompt_token_ids: &[u32],
    masked_ranges: &[TokenRange],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> inferlet::Result<GenerationResult> {
    if prompt_token_ids.is_empty() {
        return Err("chat template produced an empty prompt".into());
    }

    let mut ctx = Context::new(model)?;
    let page_size = ctx.page_size();
    let masked_ranges = merge_ranges(masked_ranges, prompt_token_ids.len() as u32);

    // Prefill in bounded chunks and sample the first completion token from the
    // final chunk. Every unmasked row gets a causal mask with the requested
    // message ranges removed. Rows inside a hidden range keep ordinary causal
    // attention: their KV is never visible to a later unmasked row, and keeping
    // their local attention avoids an all-false row at the beginning of a
    // prompt.
    let prefill_start = Instant::now();
    let mut first_token = None;
    for (chunk_index, chunk) in prompt_token_ids.chunks(PREFILL_CHUNK_TOKENS).enumerate() {
        let chunk_start = chunk_index * PREFILL_CHUNK_TOKENS;
        let is_final_chunk = chunk_start + chunk.len() == prompt_token_ids.len();
        let mut pass = ctx.forward();
        pass.input(chunk);
        if !masked_ranges.is_empty() {
            let masks = prompt_masks_range(chunk_start, chunk.len(), &masked_ranges);
            pass.attention_mask(&masks);
        }
        let handle = is_final_chunk
            .then(|| pass.sample(&[(chunk.len() - 1) as u32], sampler(temperature, top_p)));
        let output = pass.execute().await?;
        if let Some(handle) = handle {
            first_token = output.token(handle);
        }
    }
    let first_token = first_token.ok_or("prefill produced no completion token")?;
    let prefill_elapsed = prefill_start.elapsed();

    let stop_tokens = chat::stop_tokens(model);
    let first_is_stop = stop_tokens.contains(&first_token);
    let mut token_ids = if first_is_stop {
        Vec::new()
    } else {
        vec![first_token]
    };
    let mut pending = first_token;

    let decode_start = Instant::now();
    while !first_is_stop && token_ids.len() < max_tokens {
        let next_token = {
            let mut pass = ctx.forward();
            let total_after = pass.start_position() + 1;
            pass.input(&[pending]);
            if !masked_ranges.is_empty() {
                let mask = build_visibility_mask(total_after, &masked_ranges);
                pass.attention_mask(&[mask]);
            }
            let handle = pass.sample(&[0], sampler(temperature, top_p));
            let output = pass.execute().await?;
            output.token(handle).ok_or("decode produced no token")?
        };

        if stop_tokens.contains(&next_token) {
            break;
        }
        token_ids.push(next_token);
        pending = next_token;
    }
    let decode_elapsed = decode_start.elapsed();
    let text = model.tokenizer().decode(&token_ids)?;

    Ok(GenerationResult {
        text,
        hit_max_tokens: token_ids.len() >= max_tokens,
        token_ids,
        prompt_tokens: prompt_token_ids.len(),
        page_size,
        masked_ranges,
        prefill_elapsed,
        decode_elapsed,
    })
}

fn render_prompt(model: &Model, messages: &[ChatMessage]) -> inferlet::Result<RenderedPrompt> {
    let system_indices = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(message.role.as_str(), "system" | "developer"))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let system = system_indices
        .iter()
        .filter_map(|&index| messages[index].content.as_ref())
        .map(MessageContent::text)
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut ctx = Context::new(model)?;
    if !system.is_empty() {
        ctx.system(&system);
    }

    let mut message_ranges = vec![None; messages.len()];
    let mut atomic_with_system = vec![false; messages.len()];
    let mut system_pending = !system.is_empty();

    for (index, message) in messages.iter().enumerate() {
        let text = message
            .content
            .as_ref()
            .map(MessageContent::text)
            .unwrap_or_default();
        if matches!(message.role.as_str(), "system" | "developer") {
            continue;
        }

        let start = ctx.buffer().len();
        match message.role.as_str() {
            "user" => {
                ctx.user(&text);
            }
            "assistant" => {
                ctx.assistant(&text);
            }
            "tool" => {
                ctx.user(&format!("Tool result:\n{text}"));
            }
            role => return Err(format!("Unsupported message role: {role}")),
        }
        let end = ctx.buffer().len();
        message_ranges[index] = Some(TokenRange::new(start, end));
        if system_pending {
            // Context may render system + first turn through a combined model
            // template. There is no universally valid token boundary between
            // those two logical messages, so keep this bootstrap block atomic.
            atomic_with_system[index] = true;
            system_pending = false;
        }
    }
    ctx.cue();

    Ok(RenderedPrompt {
        token_ids: ctx.buffer().to_vec(),
        message_ranges,
        atomic_with_system,
    })
}

fn resolve_masked_ranges(
    request: &ChatCompletionRequest,
    rendered: &RenderedPrompt,
) -> Result<(Vec<usize>, Vec<TokenRange>), (String, &'static str)> {
    let mut indices = request.mask_message_indices.clone();
    indices.sort_unstable();
    indices.dedup();

    let mut ranges = Vec::with_capacity(indices.len());
    for &index in &indices {
        let Some(message) = request.messages.get(index) else {
            return Err((
                format!(
                    "mask_message_indices contains {index}, but messages has length {}",
                    request.messages.len()
                ),
                "mask_message_indices",
            ));
        };
        if matches!(message.role.as_str(), "system" | "developer") {
            return Err((
                format!(
                    "message {index} has role '{}' and cannot be masked",
                    message.role
                ),
                "mask_message_indices",
            ));
        }
        if rendered.atomic_with_system[index] {
            return Err((
                format!(
                    "message {index} shares an atomic chat-template span with the system prompt and cannot be masked; keep the bootstrap turn visible"
                ),
                "mask_message_indices",
            ));
        }
        let Some(range) = rendered.message_ranges[index] else {
            return Err((
                format!("message {index} has no maskable token span"),
                "mask_message_indices",
            ));
        };
        ranges.push(range);
    }

    let prompt_len = rendered.token_ids.len() as u32;
    Ok((indices, merge_ranges(&ranges, prompt_len)))
}

#[cfg(test)]
fn prompt_masks(prompt_tokens: usize, hidden: &[TokenRange]) -> Vec<Vec<u32>> {
    prompt_masks_range(0, prompt_tokens, hidden)
}

fn prompt_masks_range(
    start_position: usize,
    query_tokens: usize,
    hidden: &[TokenRange],
) -> Vec<Vec<u32>> {
    (start_position..start_position + query_tokens)
        .map(|position| {
            let position = position as u32;
            if hidden.iter().any(|range| range.contains(position)) {
                // The row is itself hidden from later visible rows, so its
                // internal value cannot leak into the generated continuation.
                vec![0, position + 1]
            } else {
                build_visibility_mask(position + 1, hidden)
            }
        })
        .collect()
}

/// Build BRLE over `[0, total)`, where false is hidden and true is visible.
/// BRLE begins with a false run, so `[0, total]` is the all-visible mask.
fn build_visibility_mask(total: u32, hidden: &[TokenRange]) -> Vec<u32> {
    let hidden = merge_ranges(hidden, total);
    if hidden.is_empty() {
        return vec![0, total];
    }

    let mut visible = Vec::new();
    let mut cursor = 0u32;
    for range in hidden {
        if cursor < range.start {
            visible.push(TokenRange {
                start: cursor,
                end: range.start,
            });
        }
        cursor = cursor.max(range.end);
    }
    if cursor < total {
        visible.push(TokenRange {
            start: cursor,
            end: total,
        });
    }

    if visible.is_empty() {
        return vec![total];
    }

    let mut brle = Vec::with_capacity(visible.len() * 2 + 1);
    let mut encoded_to = 0u32;
    for range in visible {
        brle.push(range.start - encoded_to);
        brle.push(range.end - range.start);
        encoded_to = range.end;
    }
    if encoded_to < total {
        brle.push(total - encoded_to);
    }
    brle
}

fn merge_ranges(ranges: &[TokenRange], total: u32) -> Vec<TokenRange> {
    let mut clipped = ranges
        .iter()
        .map(|range| TokenRange {
            start: range.start.min(total),
            end: range.end.min(total),
        })
        .filter(|range| range.start < range.end)
        .collect::<Vec<_>>();
    clipped.sort_by_key(|range| range.start);

    let mut merged: Vec<TokenRange> = Vec::new();
    for range in clipped {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

fn count_fully_masked_pages(ranges: &[TokenRange], prompt_tokens: u32, page_size: u32) -> usize {
    if page_size == 0 {
        return 0;
    }
    let ranges = merge_ranges(ranges, prompt_tokens);
    let committed_prompt_pages = prompt_tokens / page_size;
    (0..committed_prompt_pages)
        .filter(|&page| {
            let start = page * page_size;
            let end = start + page_size;
            ranges
                .iter()
                .any(|range| range.start <= start && end <= range.end)
        })
        .count()
}

fn validate_request(request: &ChatCompletionRequest) -> Result<(), (String, &'static str)> {
    if request.messages.is_empty() {
        return Err(("messages must not be empty".into(), "messages"));
    }
    if !request
        .messages
        .iter()
        .any(|message| message.role == "user")
    {
        return Err(("at least one user message is required".into(), "messages"));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(("only n=1 is supported".into(), "n"));
    }
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 || max_tokens > 8192 {
        return Err(("max tokens must be in 1..=8192".into(), "max_tokens"));
    }
    let temperature = request.temperature.unwrap_or(0.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(("temperature must be in 0..=2".into(), "temperature"));
    }
    let top_p = request.top_p.unwrap_or(1.0);
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(("top_p must be in (0, 1]".into(), "top_p"));
    }
    if request.mask_message_indices.len() > request.messages.len() {
        return Err((
            "mask_message_indices cannot contain more entries than messages".into(),
            "mask_message_indices",
        ));
    }
    Ok(())
}

fn sampler(temperature: f32, top_p: f32) -> Sampler {
    if temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature,
            p: top_p,
        }
    }
}

fn earliest_stop(text: &str, stop: &StopSequences) -> Option<usize> {
    stop.values()
        .into_iter()
        .filter(|value| !value.is_empty())
        .filter_map(|value| text.find(value))
        .min()
}

fn default_true() -> bool {
    true
}

fn completion_id(body: &[u8]) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-pie-mask-{nanos:x}-{:016x}", fnv1a(body))
}

fn fnv1a(value: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

async fn models_response(responder: Responder) -> Finished {
    let created = now_unix_secs();
    let data = runtime::models()
        .into_iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "created": created,
                "owned_by": "pie"
            })
        })
        .collect::<Vec<_>>();
    json_response(responder, 200, json!({"object": "list", "data": data})).await
}

async fn streaming_response(
    responder: Responder,
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish_reason: &str,
    metadata: &MaskMetadata,
) -> Finished {
    let first = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "logprobs": null,
            "finish_reason": null
        }],
        "pie_mask": metadata
    });
    let content = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "logprobs": null,
            "finish_reason": null
        }]
    });
    let done = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "logprobs": null,
            "finish_reason": finish_reason
        }]
    });
    let body = format!("data: {first}\n\ndata: {content}\n\ndata: {done}\n\ndata: [DONE]\n\n");
    let response = Response::builder()
        .status(200)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Access-Control-Allow-Origin", "*")
        .header("X-Pie-Mask-Enabled", metadata.enabled.to_string())
        .header(
            "X-Pie-Mask-Fully-Masked-Pages",
            metadata.fully_masked_pages.to_string(),
        )
        .body(body.into_body())
        .unwrap();
    responder.respond(response).await
}

async fn read_body(body: &mut IncomingBody) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match body.read(&mut chunk).await {
            Ok(0) => return Ok(output),
            Ok(count) => output.extend_from_slice(&chunk[..count]),
            Err(error) => return Err(format!("Failed to read request body: {error}")),
        }
    }
}

async fn json_response(responder: Responder, status: u16, payload: serde_json::Value) -> Finished {
    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .body(payload.to_string().into_body())
        .unwrap();
    responder.respond(response).await
}

async fn json_response_with_mask_headers(
    responder: Responder,
    status: u16,
    payload: serde_json::Value,
    metadata: &MaskMetadata,
) -> Finished {
    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .header("X-Pie-Mask-Enabled", metadata.enabled.to_string())
        .header("X-Pie-Masked-Tokens", metadata.masked_tokens.to_string())
        .header(
            "X-Pie-Mask-Fully-Masked-Pages",
            metadata.fully_masked_pages.to_string(),
        )
        .header("X-Pie-Prefill-Ms", metadata.prefill_ms.to_string())
        .header("X-Pie-Decode-Ms", metadata.decode_ms.to_string())
        .body(payload.to_string().into_body())
        .unwrap();
    responder.respond(response).await
}

async fn error_response(
    responder: Responder,
    status: u16,
    message: &str,
    param: Option<&str>,
) -> Finished {
    json_response(
        responder,
        status,
        json!({
            "error": {
                "message": message,
                "type": if status >= 500 { "server_error" } else { "invalid_request_error" },
                "param": param,
                "code": null
            }
        }),
    )
    .await
}

async fn cors_response(responder: Responder) -> Finished {
    let response = Response::builder()
        .status(StatusCode::NO_CONTENT)
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        .header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )
        .body("".into_body())
        .unwrap();
    responder.respond(response).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_brle(brle: &[u32]) -> Vec<bool> {
        let mut value = false;
        let mut output = Vec::new();
        for &run in brle {
            output.extend(std::iter::repeat_n(value, run as usize));
            value = !value;
        }
        output
    }

    #[test]
    fn visibility_mask_removes_multiple_holes() {
        let mask = build_visibility_mask(
            12,
            &[
                TokenRange { start: 2, end: 5 },
                TokenRange { start: 8, end: 10 },
            ],
        );
        assert_eq!(
            decode_brle(&mask),
            vec![
                true, true, false, false, false, true, true, true, false, false, true, true,
            ]
        );
    }

    #[test]
    fn ranges_are_clipped_sorted_and_merged() {
        assert_eq!(
            merge_ranges(
                &[
                    TokenRange { start: 8, end: 14 },
                    TokenRange { start: 2, end: 5 },
                    TokenRange { start: 4, end: 9 },
                ],
                10,
            ),
            vec![TokenRange { start: 2, end: 10 }]
        );
    }

    #[test]
    fn hidden_prompt_rows_keep_local_causal_attention() {
        let masks = prompt_masks(6, &[TokenRange { start: 1, end: 4 }]);
        assert_eq!(decode_brle(&masks[0]), vec![true]);
        assert_eq!(decode_brle(&masks[2]), vec![true, true, true]);
        assert_eq!(
            decode_brle(&masks[5]),
            vec![true, false, false, false, true, true]
        );
    }

    #[test]
    fn counts_only_pages_fully_inside_hidden_ranges() {
        let count = count_fully_masked_pages(&[TokenRange { start: 3, end: 21 }], 30, 4);
        assert_eq!(count, 4); // [4,8), [8,12), [12,16), [16,20)
    }

    #[test]
    fn stop_sequence_uses_the_earliest_match() {
        let stop = StopSequences::Many(vec!["later".into(), "stop".into()]);
        assert_eq!(earliest_stop("one stop then later", &stop), Some(4));
    }
}
