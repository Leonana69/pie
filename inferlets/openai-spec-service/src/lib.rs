//! OpenAI-compatible chat-completions service with stateful prior-response
//! speculative decoding.
//!
//! Every successful response can be retained as model-approved token IDs in a
//! daemon-scoped dictionary. A later greedy request may use those token IDs as
//! custom drafts; the target model still verifies every draft before it is
//! emitted. `speculation=false` bypasses drafting but still learns the response,
//! which makes it useful as the baseline/seed request in performance tests.
//! `spec_clear=true` clears the selected dictionary before the request runs.

use std::fs;
use std::io::ErrorKind;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use inferlet::{Context, Speculator, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};
use wstd::io::AsyncRead;

const STATE_VERSION: u32 = 1;
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_DRAFT_LEN: usize = 8;
const DEFAULT_MATCH_TOKENS: usize = 2;
const DEFAULT_MAX_ENTRIES: usize = 32;
const DEFAULT_MAX_DICTIONARY_TOKENS: usize = 32_768;
const PREFIX_CACHE_STATE_VERSION: u32 = 1;
const PREFIX_CACHE_SCHEMA: &str = "openai-spec-prefix-v1";
const DEFAULT_PREFIX_CACHE_MAX_ENTRIES: usize = 32;
const DEFAULT_PREFIX_CACHE_MAX_TOKENS: usize = 32_768;

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

    /// Pie extension: use the saved prior-response dictionary for drafting.
    #[serde(default = "default_true", alias = "speculative_decoding")]
    speculation: bool,
    /// Pie extension: clear this key's dictionary before generation.
    #[serde(default)]
    spec_clear: bool,
    /// Pie extension: independent dictionary namespace.
    #[serde(default = "default_spec_key")]
    spec_key: String,
    /// Pie extension: retain this response for later requests.
    #[serde(default = "default_true")]
    spec_store: bool,
    #[serde(default = "default_draft_len")]
    spec_draft_len: usize,
    #[serde(default = "default_match_tokens")]
    spec_match_tokens: usize,
    #[serde(default = "default_max_entries")]
    spec_max_entries: usize,
    #[serde(default = "default_max_dictionary_tokens")]
    spec_max_dictionary_tokens: usize,

    /// Pie extension: reuse a saved prompt KV snapshot when it is a token prefix.
    #[serde(default = "default_true")]
    prefix_cache: bool,
    /// Pie extension: clear this prompt-cache namespace before generation.
    #[serde(default)]
    prefix_cache_clear: bool,
    /// Pie extension: independent prompt-cache namespace.
    #[serde(default = "default_prefix_cache_key")]
    prefix_cache_key: String,
    /// Pie extension: retain the selected prompt prefix as a KV snapshot.
    #[serde(default = "default_true")]
    prefix_cache_store: bool,
    /// Pie extension: cache only this many rendered prompt tokens. When
    /// omitted, the complete rendered prompt is the cacheable prefix.
    #[serde(default)]
    prefix_cache_prefix_tokens: Option<usize>,
    #[serde(default = "default_prefix_cache_max_entries")]
    prefix_cache_max_entries: usize,
    #[serde(default = "default_prefix_cache_max_tokens")]
    prefix_cache_max_tokens: usize,
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

#[derive(Debug, Default, Deserialize, Serialize)]
struct DictionaryState {
    version: u32,
    entries: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PrefixCacheEntry {
    name: String,
    tokens: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct PrefixCacheState {
    version: u32,
    entries: Vec<PrefixCacheEntry>,
}

impl Default for PrefixCacheState {
    fn default() -> Self {
        Self {
            version: PREFIX_CACHE_STATE_VERSION,
            entries: Vec::new(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
struct SpecStats {
    draft_rounds: usize,
    drafts_proposed: usize,
    drafts_accepted: usize,
    longest_accepted_draft: usize,
}

#[derive(Debug, Serialize)]
struct SpeculationMetadata {
    enabled: bool,
    clear_requested: bool,
    key: String,
    dictionary_entries_before: usize,
    dictionary_entries_after: usize,
    dictionary_tokens_before: usize,
    dictionary_tokens_after: usize,
    draft_len: usize,
    match_tokens: usize,
    verifier_steps: usize,
    draft_rounds: usize,
    drafts_proposed: usize,
    drafts_accepted: usize,
    draft_acceptance_rate: f64,
    longest_accepted_draft: usize,
    prefill_ms: u128,
    decode_ms: u128,
    total_ms: u128,
    decode_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct PrefixCacheMetadata {
    enabled: bool,
    clear_requested: bool,
    key: String,
    prefix_tokens: usize,
    hit: bool,
    cached_tokens: usize,
    prefilled_tokens: usize,
    stored: bool,
    entries_before: usize,
    entries_after: usize,
    prefill_tokens_per_second: f64,
}

struct GenerationResult {
    text: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    hit_max_tokens: bool,
    verifier_steps: usize,
    stats: SpecStats,
    prefill_elapsed: Duration,
    decode_elapsed: Duration,
    prefix_cache_hit: bool,
    prefix_cache_cached_tokens: usize,
    prefix_cache_stored: bool,
    prefix_cache_prefix_tokens: usize,
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
        (Method::POST, "/v1/speculation/clear") => {
            let body = match read_body(req.body_mut()).await {
                Ok(body) => body,
                Err(message) => return error_response(responder, 400, &message, None).await,
            };
            handle_clear(body, responder).await
        }
        (Method::POST, "/v1/prefix_cache/clear") => {
            let body = match read_body(req.body_mut()).await {
                Ok(body) => body,
                Err(message) => return error_response(responder, 400, &message, None).await,
            };
            handle_prefix_cache_clear(body, responder).await
        }
        (Method::GET, "/v1/models") => models_response(responder).await,
        (Method::GET, "/health") => json_response(responder, 200, json!({"status": "ok"})).await,
        (Method::GET, "/") => {
            json_response(
                responder,
                200,
                json!({
                    "name": "Pie OpenAI speculative service",
                    "version": "0.1.0",
                    "endpoints": [
                        "POST /v1/chat/completions",
                        "GET /v1/models",
                        "POST /v1/speculation/clear",
                        "POST /v1/prefix_cache/clear",
                        "GET /health"
                    ]
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

    let prompt_token_ids = match render_prompt_token_ids(&model, &request.messages) {
        Ok(tokens) => tokens,
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
    let prefix_cache_prefix_tokens = match resolve_prefix_cache_tokens(
        request.prefix_cache_prefix_tokens,
        prompt_token_ids.len(),
    ) {
        Ok(tokens) => tokens,
        Err(error) => {
            return error_response(responder, 400, &error, Some("prefix_cache_prefix_tokens"))
                .await;
        }
    };

    let prefix_state_path = prefix_cache_state_path(&model_name, &request.prefix_cache_key);
    if request.prefix_cache_clear {
        if let Err(error) = clear_prefix_cache(&model, &prefix_state_path) {
            return error_response(
                responder,
                500,
                &format!("Failed to clear prompt prefix cache: {error}"),
                Some("prefix_cache_clear"),
            )
            .await;
        }
    }
    let mut prefix_cache_state = match load_prefix_cache_state(&prefix_state_path) {
        Ok(state) => state,
        Err(error) => {
            return error_response(
                responder,
                500,
                &format!("Failed to load prompt prefix cache: {error}"),
                Some("prefix_cache_key"),
            )
            .await;
        }
    };
    let prefix_entries_before = prefix_cache_state.entries.len();

    let state_path = dictionary_path(&model_name, &request.spec_key);
    if request.spec_clear {
        if let Err(error) = remove_state(&state_path) {
            return error_response(
                responder,
                500,
                &format!("Failed to clear speculation dictionary: {error}"),
                Some("spec_clear"),
            )
            .await;
        }
    }

    let mut dictionary = match load_state(&state_path) {
        Ok(state) => state,
        Err(error) => {
            return error_response(
                responder,
                500,
                &format!("Failed to load speculation dictionary: {error}"),
                Some("spec_key"),
            )
            .await;
        }
    };
    let entries_before = dictionary.entries.len();
    let tokens_before = dictionary_token_count(&dictionary);

    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = request.temperature.unwrap_or(0.0);
    let top_p = request.top_p.unwrap_or(1.0);
    let references = if request.speculation {
        dictionary.entries.clone()
    } else {
        Vec::new()
    };

    let mut generation = match generate(
        &model,
        &model_name,
        &prompt_token_ids,
        max_tokens,
        temperature,
        top_p,
        request.speculation,
        references,
        request.spec_match_tokens,
        request.spec_draft_len,
        request.prefix_cache,
        request.prefix_cache_store,
        &request.prefix_cache_key,
        prefix_cache_prefix_tokens,
        request.prefix_cache_max_entries,
        request.prefix_cache_max_tokens,
        &mut prefix_cache_state,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return error_response(responder, 500, &format!("Generation failed: {error}"), None)
                .await;
        }
    };

    if request.prefix_cache {
        if let Err(error) = save_prefix_cache_state(&prefix_state_path, &prefix_cache_state) {
            return error_response(
                responder,
                500,
                &format!("Failed to save prompt prefix cache: {error}"),
                Some("prefix_cache_store"),
            )
            .await;
        }
    }

    if let Some(stop) = &request.stop {
        if let Some(index) = earliest_stop(&generation.text, stop) {
            generation.text.truncate(index);
            generation.token_ids = model.tokenizer().encode(&generation.text);
            generation.hit_max_tokens = false;
        }
    }

    if request.spec_store && !generation.token_ids.is_empty() {
        dictionary.entries.push(generation.token_ids.clone());
        trim_dictionary(
            &mut dictionary,
            request.spec_max_entries,
            request.spec_max_dictionary_tokens,
        );
        if let Err(error) = save_state(&state_path, &dictionary) {
            return error_response(
                responder,
                500,
                &format!("Failed to save speculation dictionary: {error}"),
                Some("spec_store"),
            )
            .await;
        }
    }

    let entries_after = dictionary.entries.len();
    let tokens_after = dictionary_token_count(&dictionary);
    let stats = generation.stats;
    let completion_tokens = generation.token_ids.len();
    let total_elapsed = generation.prefill_elapsed + generation.decode_elapsed;
    let metadata = SpeculationMetadata {
        enabled: request.speculation,
        clear_requested: request.spec_clear,
        key: request.spec_key.clone(),
        dictionary_entries_before: entries_before,
        dictionary_entries_after: entries_after,
        dictionary_tokens_before: tokens_before,
        dictionary_tokens_after: tokens_after,
        draft_len: request.spec_draft_len,
        match_tokens: request.spec_match_tokens,
        verifier_steps: generation.verifier_steps,
        draft_rounds: stats.draft_rounds,
        drafts_proposed: stats.drafts_proposed,
        drafts_accepted: stats.drafts_accepted,
        draft_acceptance_rate: if stats.drafts_proposed == 0 {
            0.0
        } else {
            stats.drafts_accepted as f64 / stats.drafts_proposed as f64
        },
        longest_accepted_draft: stats.longest_accepted_draft,
        prefill_ms: generation.prefill_elapsed.as_millis(),
        decode_ms: generation.decode_elapsed.as_millis(),
        total_ms: total_elapsed.as_millis(),
        decode_tokens_per_second: completion_tokens as f64
            / generation.decode_elapsed.as_secs_f64().max(1e-9),
    };
    let prefilled_tokens = generation
        .prompt_tokens
        .saturating_sub(generation.prefix_cache_cached_tokens);
    let prefix_cache_metadata = PrefixCacheMetadata {
        enabled: request.prefix_cache,
        clear_requested: request.prefix_cache_clear,
        key: request.prefix_cache_key.clone(),
        prefix_tokens: generation.prefix_cache_prefix_tokens,
        hit: generation.prefix_cache_hit,
        cached_tokens: generation.prefix_cache_cached_tokens,
        prefilled_tokens,
        stored: generation.prefix_cache_stored,
        entries_before: prefix_entries_before,
        entries_after: prefix_cache_state.entries.len(),
        prefill_tokens_per_second: if prefilled_tokens == 0 {
            0.0
        } else {
            prefilled_tokens as f64 / generation.prefill_elapsed.as_secs_f64().max(1e-9)
        },
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
            &prefix_cache_metadata,
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
            "pie_speculation": metadata,
            "pie_prefix_cache": prefix_cache_metadata
        });
        json_response_with_spec_headers(responder, 200, payload, &metadata, &prefix_cache_metadata)
            .await
    }
}

async fn generate(
    model: &Model,
    model_name: &str,
    prompt_token_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    speculation: bool,
    references: Vec<Vec<u32>>,
    match_tokens: usize,
    draft_len: usize,
    prefix_cache: bool,
    prefix_cache_store: bool,
    prefix_cache_key: &str,
    prefix_cache_prefix_tokens: usize,
    prefix_cache_max_entries: usize,
    prefix_cache_max_tokens: usize,
    prefix_cache_state: &mut PrefixCacheState,
) -> inferlet::Result<GenerationResult> {
    let prompt_tokens = prompt_token_ids.len();
    let prefill_start = Instant::now();
    let cached = if prefix_cache {
        open_longest_cached_prefix(
            model,
            model_name,
            prefix_cache_key,
            prompt_token_ids,
            prefix_cache_prefix_tokens,
            prefix_cache_state,
        )
    } else {
        None
    };
    let (mut ctx, prefix_cache_hit, prefix_cache_cached_tokens, opened_name) =
        if let Some((cached_ctx, cached_tokens, name)) = cached {
            (cached_ctx, true, cached_tokens, Some(name))
        } else {
            (Context::new(model)?, false, 0, None)
        };

    let cache_boundary = if prefix_cache {
        prefix_cache_prefix_tokens
    } else {
        prompt_tokens
    };
    if prefix_cache_cached_tokens < cache_boundary {
        ctx.append(&prompt_token_ids[prefix_cache_cached_tokens..cache_boundary]);
        ctx.flush().await?;
    }

    if ctx.seq_len() as usize != cache_boundary {
        return Err(format!(
            "prefix context length mismatch: got {}, expected {cache_boundary}",
            ctx.seq_len()
        ));
    }

    if let Some(name) = opened_name.as_deref() {
        touch_prefix_cache_entry(prefix_cache_state, name, prefix_cache_cached_tokens);
    }
    let prefix_cache_stored = if prefix_cache
        && prefix_cache_store
        && prefix_cache_prefix_tokens <= prefix_cache_max_tokens
    {
        store_prefix_snapshot(
            model,
            &ctx,
            model_name,
            prefix_cache_key,
            &prompt_token_ids[..prefix_cache_prefix_tokens],
            prefix_cache_state,
        )
    } else {
        false
    };
    trim_prefix_cache(
        model,
        prefix_cache_state,
        prefix_cache_max_entries,
        prefix_cache_max_tokens,
    );

    if cache_boundary < prompt_tokens {
        ctx.append(&prompt_token_ids[cache_boundary..]);
        ctx.flush().await?;
    }
    if ctx.seq_len() as usize != prompt_tokens {
        return Err(format!(
            "prompt context length mismatch: got {}, expected {prompt_tokens}",
            ctx.seq_len()
        ));
    }
    let prefill_elapsed = prefill_start.elapsed();

    let decode_start = Instant::now();
    let cue = chat::cue(model);
    let trigger = *cue.last().ok_or("chat template produced an empty cue")?;
    let first_token = {
        let mut pass = ctx.forward();
        pass.input(&[trigger]);
        let handle = pass.sample(&[0], sampler(temperature, top_p));
        let output = pass.execute().await?;
        output
            .token(handle)
            .ok_or("bootstrap decode produced no token")?
    };

    let stop_tokens = chat::stop_tokens(model);
    let first_is_stop = stop_tokens.contains(&first_token);
    let mut token_ids = if first_is_stop {
        Vec::new()
    } else {
        vec![first_token]
    };
    let mut verifier_steps = 1usize;

    ctx.append(&[first_token]);
    let cursor = ctx.seq_len() + ctx.buffer().len() as u32;
    let stats = Arc::new(Mutex::new(SpecStats::default()));
    let drafter = DictionaryDrafter::new(
        references,
        vec![first_token],
        match_tokens,
        draft_len,
        cursor,
        Arc::clone(&stats),
    );

    if !first_is_stop && token_ids.len() < max_tokens {
        let mut generator = ctx
            .generate(sampler(temperature, top_p))
            .max_tokens(max_tokens - token_ids.len())
            .stop(&stop_tokens);
        generator = if speculation {
            generator.speculator(drafter)
        } else {
            generator.disable_system_speculation()
        };

        while let Some(step) = generator.next()? {
            let output = step.execute().await?;
            verifier_steps += 1;
            token_ids.extend_from_slice(&output.tokens);
        }
    }

    let decode_elapsed = decode_start.elapsed();
    let final_stats = *stats.lock().unwrap();
    let text = model.tokenizer().decode(&token_ids)?;
    Ok(GenerationResult {
        text,
        hit_max_tokens: token_ids.len() >= max_tokens,
        token_ids,
        prompt_tokens,
        verifier_steps,
        stats: final_stats,
        prefill_elapsed,
        decode_elapsed,
        prefix_cache_hit,
        prefix_cache_cached_tokens,
        prefix_cache_stored,
        prefix_cache_prefix_tokens,
    })
}

fn render_prompt_token_ids(model: &Model, messages: &[ChatMessage]) -> inferlet::Result<Vec<u32>> {
    let mut ctx = Context::new(model)?;
    fill_messages(&mut ctx, messages)?;
    ctx.cue();
    Ok(ctx.buffer().to_vec())
}

fn fill_messages(ctx: &mut Context, messages: &[ChatMessage]) -> inferlet::Result<()> {
    let system = messages
        .iter()
        .filter(|message| matches!(message.role.as_str(), "system" | "developer"))
        .filter_map(|message| message.content.as_ref())
        .map(MessageContent::text)
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system.is_empty() {
        ctx.system(&system);
    }

    for message in messages {
        let text = message
            .content
            .as_ref()
            .map(MessageContent::text)
            .unwrap_or_default();
        match message.role.as_str() {
            "system" | "developer" => {}
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

struct DictionaryDrafter {
    references: Vec<Vec<u32>>,
    history: Vec<u32>,
    match_tokens: usize,
    draft_len: usize,
    cursor: u32,
    last_proposed: usize,
    stats: Arc<Mutex<SpecStats>>,
}

impl DictionaryDrafter {
    fn new(
        references: Vec<Vec<u32>>,
        history: Vec<u32>,
        match_tokens: usize,
        draft_len: usize,
        cursor: u32,
        stats: Arc<Mutex<SpecStats>>,
    ) -> Self {
        Self {
            references,
            history,
            match_tokens,
            draft_len,
            cursor,
            last_proposed: 0,
            stats,
        }
    }

    fn continuation(&self) -> Vec<u32> {
        let n = self.match_tokens;
        if self.history.len() < n {
            return Vec::new();
        }
        let suffix = &self.history[self.history.len() - n..];
        let mut best: Option<(usize, usize, usize)> = None;

        for (reference_index, reference) in self.references.iter().enumerate() {
            if reference.len() <= n {
                continue;
            }
            for start in 0..=reference.len() - n {
                let continuation_start = start + n;
                if continuation_start >= reference.len()
                    || &reference[start..continuation_start] != suffix
                {
                    continue;
                }

                let mut backward_match = n;
                while backward_match < self.history.len()
                    && backward_match < continuation_start
                    && self.history[self.history.len() - backward_match - 1]
                        == reference[continuation_start - backward_match - 1]
                {
                    backward_match += 1;
                }

                let candidate = (backward_match, reference_index, continuation_start);
                if best.is_none_or(|current| candidate > current) {
                    best = Some(candidate);
                }
            }
        }

        let Some((_, reference_index, start)) = best else {
            return Vec::new();
        };
        let reference = &self.references[reference_index];
        let end = (start + self.draft_len).min(reference.len());
        reference[start..end].to_vec()
    }
}

impl Speculator for DictionaryDrafter {
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        let drafts = self.continuation();
        self.last_proposed = drafts.len();
        if !drafts.is_empty() {
            let mut stats = self.stats.lock().unwrap();
            stats.draft_rounds += 1;
            stats.drafts_proposed += drafts.len();
        }
        let positions = (self.cursor..self.cursor + drafts.len() as u32).collect();
        (drafts, positions)
    }

    fn accept(&mut self, accepted: &[u32]) {
        let accepted_drafts = accepted.len().saturating_sub(1).min(self.last_proposed);
        {
            let mut stats = self.stats.lock().unwrap();
            stats.drafts_accepted += accepted_drafts;
            stats.longest_accepted_draft = stats.longest_accepted_draft.max(accepted_drafts);
        }
        self.last_proposed = 0;
        self.history.extend_from_slice(accepted);
        self.cursor += accepted.len() as u32;
    }
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
    if !temperature.is_finite() || temperature < 0.0 || temperature > 2.0 {
        return Err(("temperature must be in 0..=2".into(), "temperature"));
    }
    if request.speculation && temperature > 0.0 {
        return Err((
            "speculation=true currently requires temperature=0 for lossless greedy verification"
                .into(),
            "temperature",
        ));
    }
    let top_p = request.top_p.unwrap_or(1.0);
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(("top_p must be in (0, 1]".into(), "top_p"));
    }
    if request.spec_key.trim().is_empty() || request.spec_key.len() > 256 {
        return Err(("spec_key must contain 1..=256 bytes".into(), "spec_key"));
    }
    if request.spec_draft_len == 0 || request.spec_draft_len > 32 {
        return Err(("spec_draft_len must be in 1..=32".into(), "spec_draft_len"));
    }
    if request.spec_match_tokens == 0 || request.spec_match_tokens > 32 {
        return Err((
            "spec_match_tokens must be in 1..=32".into(),
            "spec_match_tokens",
        ));
    }
    if request.spec_max_entries == 0 || request.spec_max_entries > 256 {
        return Err((
            "spec_max_entries must be in 1..=256".into(),
            "spec_max_entries",
        ));
    }
    if request.spec_max_dictionary_tokens == 0 || request.spec_max_dictionary_tokens > 1_000_000 {
        return Err((
            "spec_max_dictionary_tokens must be in 1..=1000000".into(),
            "spec_max_dictionary_tokens",
        ));
    }
    if request.prefix_cache_key.trim().is_empty() || request.prefix_cache_key.len() > 256 {
        return Err((
            "prefix_cache_key must contain 1..=256 bytes".into(),
            "prefix_cache_key",
        ));
    }
    if request.prefix_cache_prefix_tokens == Some(0) {
        return Err((
            "prefix_cache_prefix_tokens must be at least 1".into(),
            "prefix_cache_prefix_tokens",
        ));
    }
    if request.prefix_cache_max_entries == 0 || request.prefix_cache_max_entries > 256 {
        return Err((
            "prefix_cache_max_entries must be in 1..=256".into(),
            "prefix_cache_max_entries",
        ));
    }
    if request.prefix_cache_max_tokens == 0 || request.prefix_cache_max_tokens > 1_000_000 {
        return Err((
            "prefix_cache_max_tokens must be in 1..=1000000".into(),
            "prefix_cache_max_tokens",
        ));
    }
    Ok(())
}

fn default_true() -> bool {
    true
}
fn default_spec_key() -> String {
    "default".into()
}
fn default_draft_len() -> usize {
    DEFAULT_DRAFT_LEN
}
fn default_match_tokens() -> usize {
    DEFAULT_MATCH_TOKENS
}
fn default_max_entries() -> usize {
    DEFAULT_MAX_ENTRIES
}
fn default_max_dictionary_tokens() -> usize {
    DEFAULT_MAX_DICTIONARY_TOKENS
}
fn default_prefix_cache_key() -> String {
    "default".into()
}
fn default_prefix_cache_max_entries() -> usize {
    DEFAULT_PREFIX_CACHE_MAX_ENTRIES
}
fn default_prefix_cache_max_tokens() -> usize {
    DEFAULT_PREFIX_CACHE_MAX_TOKENS
}

fn dictionary_path(model: &str, key: &str) -> String {
    let hash = fnv1a(&format!("{model}\u{1f}{key}"));
    format!("/scratch/pie-spec-{hash:016x}.json")
}

fn prefix_cache_state_path(model: &str, key: &str) -> String {
    let digest = sha256_components(&[
        PREFIX_CACHE_SCHEMA.as_bytes(),
        model.as_bytes(),
        key.as_bytes(),
    ]);
    format!("/scratch/pie-prefix-cache-{}.json", &digest[..32])
}

fn prefix_snapshot_name(model: &str, key: &str, tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    update_sha_field(&mut hasher, PREFIX_CACHE_SCHEMA.as_bytes());
    update_sha_field(&mut hasher, model.as_bytes());
    update_sha_field(&mut hasher, key.as_bytes());
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("pie-prefix-{}", digest_hex(&hasher.finalize()))
}

fn resolve_prefix_cache_tokens(
    requested: Option<usize>,
    prompt_tokens: usize,
) -> Result<usize, String> {
    let prefix_tokens = requested.unwrap_or(prompt_tokens);
    if prefix_tokens == 0 {
        return Err("prefix_cache_prefix_tokens must be at least 1".into());
    }
    if prefix_tokens > prompt_tokens {
        return Err(format!(
            "prefix_cache_prefix_tokens ({prefix_tokens}) exceeds the rendered prompt length ({prompt_tokens})"
        ));
    }
    Ok(prefix_tokens)
}

fn sha256_components(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        update_sha_field(&mut hasher, part);
    }
    digest_hex(&hasher.finalize())
}

fn update_sha_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn digest_hex(value: &[u8]) -> String {
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn fnv1a(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn load_state(path: &str) -> Result<DictionaryState, String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Ok(DictionaryState {
                version: STATE_VERSION,
                entries: Vec::new(),
            });
        }
        Err(error) => return Err(error.to_string()),
    };
    let state: DictionaryState = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if state.version != STATE_VERSION {
        return Err(format!(
            "unsupported state version {} (expected {})",
            state.version, STATE_VERSION
        ));
    }
    Ok(state)
}

fn save_state(path: &str, state: &DictionaryState) -> Result<(), String> {
    let bytes = serde_json::to_vec(state).map_err(|e| e.to_string())?;
    let temporary = format!("{path}.tmp");
    fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    fs::rename(&temporary, path).map_err(|e| e.to_string())
}

fn remove_state(path: &str) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn load_prefix_cache_state(path: &str) -> Result<PrefixCacheState, String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Ok(PrefixCacheState::default());
        }
        Err(error) => return Err(error.to_string()),
    };
    let state: PrefixCacheState = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if state.version != PREFIX_CACHE_STATE_VERSION {
        return Err(format!(
            "unsupported prefix cache state version {} (expected {})",
            state.version, PREFIX_CACHE_STATE_VERSION
        ));
    }
    Ok(state)
}

fn save_prefix_cache_state(path: &str, state: &PrefixCacheState) -> Result<(), String> {
    let bytes = serde_json::to_vec(state).map_err(|e| e.to_string())?;
    let temporary = format!("{path}.tmp");
    fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    fs::rename(&temporary, path).map_err(|e| e.to_string())
}

fn clear_prefix_cache(model: &Model, path: &str) -> Result<usize, String> {
    let state = load_prefix_cache_state(path)?;
    let count = state.entries.len();
    for entry in state.entries {
        let _ = Context::delete(model, &entry.name);
    }
    remove_state(path)?;
    Ok(count)
}

fn open_longest_cached_prefix(
    model: &Model,
    model_name: &str,
    key: &str,
    prompt_tokens: &[u32],
    max_prefix_tokens: usize,
    state: &mut PrefixCacheState,
) -> Option<(Context, usize, String)> {
    let mut candidates = state.entries.clone();
    candidates.sort_by(|left, right| right.tokens.cmp(&left.tokens));
    let mut stale = Vec::new();

    for entry in candidates {
        if entry.tokens == 0
            || entry.tokens > max_prefix_tokens
            || entry.tokens > prompt_tokens.len()
        {
            continue;
        }
        let expected = prefix_snapshot_name(model_name, key, &prompt_tokens[..entry.tokens]);
        if expected != entry.name {
            continue;
        }
        match Context::open(model, &entry.name) {
            Ok(ctx) if ctx.seq_len() as usize == entry.tokens => {
                if !stale.is_empty() {
                    state
                        .entries
                        .retain(|candidate| !stale.contains(&candidate.name));
                }
                return Some((ctx, entry.tokens, entry.name));
            }
            Ok(ctx) => {
                drop(ctx);
                let _ = Context::delete(model, &entry.name);
                stale.push(entry.name);
            }
            Err(_) => stale.push(entry.name),
        }
    }

    if !stale.is_empty() {
        state
            .entries
            .retain(|candidate| !stale.contains(&candidate.name));
    }
    None
}

fn touch_prefix_cache_entry(state: &mut PrefixCacheState, name: &str, tokens: usize) {
    state.entries.retain(|entry| entry.name != name);
    state.entries.push(PrefixCacheEntry {
        name: name.to_string(),
        tokens,
    });
    state.version = PREFIX_CACHE_STATE_VERSION;
}

fn store_prefix_snapshot(
    model: &Model,
    ctx: &Context,
    model_name: &str,
    key: &str,
    prompt_tokens: &[u32],
    state: &mut PrefixCacheState,
) -> bool {
    let name = prefix_snapshot_name(model_name, key, prompt_tokens);
    let available = if state.entries.iter().any(|entry| entry.name == name) {
        true
    } else if ctx.save(&name).is_ok() {
        true
    } else {
        Context::open(model, &name).is_ok()
    };
    if available {
        touch_prefix_cache_entry(state, &name, prompt_tokens.len());
    }
    available
}

fn trim_prefix_cache(
    model: &Model,
    state: &mut PrefixCacheState,
    max_entries: usize,
    max_tokens: usize,
) {
    while state.entries.len() > max_entries
        || state
            .entries
            .iter()
            .map(|entry| entry.tokens)
            .sum::<usize>()
            > max_tokens
    {
        let removed = state.entries.remove(0);
        let _ = Context::delete(model, &removed.name);
    }
    state.version = PREFIX_CACHE_STATE_VERSION;
}

fn dictionary_token_count(state: &DictionaryState) -> usize {
    state.entries.iter().map(Vec::len).sum()
}

fn trim_dictionary(state: &mut DictionaryState, max_entries: usize, max_tokens: usize) {
    while state.entries.len() > max_entries {
        state.entries.remove(0);
    }
    while state.entries.len() > 1 && dictionary_token_count(state) > max_tokens {
        state.entries.remove(0);
    }
    if let Some(entry) = state.entries.last_mut() {
        entry.truncate(max_tokens);
    }
    state.version = STATE_VERSION;
}

fn earliest_stop(text: &str, stop: &StopSequences) -> Option<usize> {
    stop.values()
        .into_iter()
        .filter(|value| !value.is_empty())
        .filter_map(|value| text.find(value))
        .min()
}

fn completion_id(body: &[u8]) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-pie-{nanos:x}-{:016x}", fnv1a_bytes(body))
}

fn fnv1a_bytes(value: &[u8]) -> u64 {
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

#[derive(Debug, Deserialize)]
struct ClearRequest {
    #[serde(default = "default_spec_key")]
    spec_key: String,
}

async fn handle_clear(body: Vec<u8>, responder: Responder) -> Finished {
    let request = if body.is_empty() {
        ClearRequest {
            spec_key: default_spec_key(),
        }
    } else {
        match serde_json::from_slice::<ClearRequest>(&body) {
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
        }
    };
    if request.spec_key.trim().is_empty() || request.spec_key.len() > 256 {
        return error_response(
            responder,
            400,
            "spec_key must contain 1..=256 bytes",
            Some("spec_key"),
        )
        .await;
    }
    let models = runtime::models();
    let Some(model_name) = models.first() else {
        return error_response(responder, 503, "No models are configured", None).await;
    };
    let path = dictionary_path(model_name, &request.spec_key);
    match remove_state(&path) {
        Ok(()) => {
            json_response(
                responder,
                200,
                json!({"cleared": true, "spec_key": request.spec_key}),
            )
            .await
        }
        Err(error) => {
            error_response(
                responder,
                500,
                &format!("Failed to clear speculation dictionary: {error}"),
                Some("spec_key"),
            )
            .await
        }
    }
}

#[derive(Debug, Deserialize)]
struct PrefixCacheClearRequest {
    #[serde(default = "default_prefix_cache_key")]
    prefix_cache_key: String,
}

async fn handle_prefix_cache_clear(body: Vec<u8>, responder: Responder) -> Finished {
    let request = if body.is_empty() {
        PrefixCacheClearRequest {
            prefix_cache_key: default_prefix_cache_key(),
        }
    } else {
        match serde_json::from_slice::<PrefixCacheClearRequest>(&body) {
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
        }
    };
    if request.prefix_cache_key.trim().is_empty() || request.prefix_cache_key.len() > 256 {
        return error_response(
            responder,
            400,
            "prefix_cache_key must contain 1..=256 bytes",
            Some("prefix_cache_key"),
        )
        .await;
    }
    let models = runtime::models();
    let Some(model_name) = models.first() else {
        return error_response(responder, 503, "No models are configured", None).await;
    };
    let model = match Model::load(model_name) {
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
    let path = prefix_cache_state_path(model_name, &request.prefix_cache_key);
    match clear_prefix_cache(&model, &path) {
        Ok(entries_cleared) => {
            json_response(
                responder,
                200,
                json!({
                    "cleared": true,
                    "prefix_cache_key": request.prefix_cache_key,
                    "entries_cleared": entries_cleared
                }),
            )
            .await
        }
        Err(error) => {
            error_response(
                responder,
                500,
                &format!("Failed to clear prompt prefix cache: {error}"),
                Some("prefix_cache_key"),
            )
            .await
        }
    }
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
    metadata: &SpeculationMetadata,
    prefix_cache_metadata: &PrefixCacheMetadata,
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
        "pie_speculation": metadata,
        "pie_prefix_cache": prefix_cache_metadata
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
        .header("X-Pie-Speculation-Enabled", metadata.enabled.to_string())
        .header(
            "X-Pie-Spec-Drafts-Accepted",
            metadata.drafts_accepted.to_string(),
        )
        .header(
            "X-Pie-Prefix-Cache-Hit",
            prefix_cache_metadata.hit.to_string(),
        )
        .header(
            "X-Pie-Prefix-Cache-Cached-Tokens",
            prefix_cache_metadata.cached_tokens.to_string(),
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

async fn json_response_with_spec_headers(
    responder: Responder,
    status: u16,
    payload: serde_json::Value,
    metadata: &SpeculationMetadata,
    prefix_cache_metadata: &PrefixCacheMetadata,
) -> Finished {
    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .header("X-Pie-Speculation-Enabled", metadata.enabled.to_string())
        .header(
            "X-Pie-Spec-Drafts-Proposed",
            metadata.drafts_proposed.to_string(),
        )
        .header(
            "X-Pie-Spec-Drafts-Accepted",
            metadata.drafts_accepted.to_string(),
        )
        .header("X-Pie-Prefill-Ms", metadata.prefill_ms.to_string())
        .header("X-Pie-Decode-Ms", metadata.decode_ms.to_string())
        .header(
            "X-Pie-Prefix-Cache-Hit",
            prefix_cache_metadata.hit.to_string(),
        )
        .header(
            "X-Pie-Prefix-Cache-Cached-Tokens",
            prefix_cache_metadata.cached_tokens.to_string(),
        )
        .header(
            "X-Pie-Prefill-Tokens",
            prefix_cache_metadata.prefilled_tokens.to_string(),
        )
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

    fn drafter(references: Vec<Vec<u32>>, history: &[u32]) -> DictionaryDrafter {
        DictionaryDrafter::new(
            references,
            history.to_vec(),
            2,
            3,
            100,
            Arc::new(Mutex::new(SpecStats::default())),
        )
    }

    #[test]
    fn drafts_from_a_saved_response() {
        let drafter = drafter(vec![vec![10, 11, 12, 13, 14]], &[10, 11]);
        assert_eq!(drafter.continuation(), vec![12, 13, 14]);
    }

    #[test]
    fn prefers_newer_response_when_alignment_is_equal() {
        let drafter = drafter(vec![vec![1, 2, 7, 8], vec![1, 2, 9, 10]], &[1, 2]);
        assert_eq!(drafter.continuation(), vec![9, 10]);
    }

    #[test]
    fn clear_then_seed_limits_state() {
        let mut state = DictionaryState {
            version: STATE_VERSION,
            entries: vec![vec![1, 2], vec![3, 4], vec![5, 6, 7]],
        };
        trim_dictionary(&mut state, 2, 4);
        assert_eq!(state.entries, vec![vec![5, 6, 7]]);
    }

    #[test]
    fn prefix_snapshot_names_are_content_addressed_and_namespaced() {
        let base = prefix_snapshot_name("model-a", "tenant-a", &[1, 2, 3]);
        assert_eq!(
            base,
            prefix_snapshot_name("model-a", "tenant-a", &[1, 2, 3])
        );
        assert_ne!(
            base,
            prefix_snapshot_name("model-a", "tenant-a", &[1, 2, 4])
        );
        assert_ne!(
            base,
            prefix_snapshot_name("model-a", "tenant-b", &[1, 2, 3])
        );
        assert_ne!(
            base,
            prefix_snapshot_name("model-b", "tenant-a", &[1, 2, 3])
        );
    }

    #[test]
    fn touching_prefix_entries_updates_lru_order_without_duplicates() {
        let mut state = PrefixCacheState::default();
        touch_prefix_cache_entry(&mut state, "first", 10);
        touch_prefix_cache_entry(&mut state, "second", 20);
        touch_prefix_cache_entry(&mut state, "first", 10);
        assert_eq!(state.entries.len(), 2);
        assert_eq!(state.entries[0].name, "second");
        assert_eq!(state.entries[1].name, "first");
    }

    #[test]
    fn explicit_prefix_boundary_is_bounded_by_rendered_prompt() {
        assert_eq!(resolve_prefix_cache_tokens(None, 25).unwrap(), 25);
        assert_eq!(resolve_prefix_cache_tokens(Some(10), 25).unwrap(), 10);
        assert!(resolve_prefix_cache_tokens(Some(0), 25).is_err());
        assert!(resolve_prefix_cache_tokens(Some(26), 25).is_err());
    }
}
