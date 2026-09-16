//! HTTP port of the S1 combined prior-plan/masking inferlet.
mod api;
mod draft;
use api::*;
use inferlet::{Context, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::ErrorKind,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response};
use wstd::io::AsyncRead;

const REGISTRY: &str = "/scratch/plan-mask-caches-v1.json";
const CHUNK: usize = 512;
const MAX_CACHE_TOKENS: usize = 65536;
const MAX_CACHES: usize = 16;
const MAX_BODY: usize = 2 * 1024 * 1024;
#[derive(Clone, Deserialize, Serialize)]
struct Cache {
    id: String,
    model: String,
    input: CacheRequest,
    common: u32,
    slots: Vec<u32>,
    suffix: Vec<u32>,
    tokens: usize,
}
#[derive(Default, Deserialize, Serialize)]
struct Registry {
    entries: Vec<Cache>,
}
fn load_registry() -> Result<Registry> {
    match fs::read(REGISTRY) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| internal(e.to_string())),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(Registry::default()),
        Err(e) => Err(internal(format!(
            "cache registry: {e}; enable runtime.allow_fs"
        ))),
    }
}
fn save_registry(registry: &Registry) -> Result<()> {
    let bytes = serde_json::to_vec(registry).map_err(|e| internal(e.to_string()))?;
    fs::write(format!("{REGISTRY}.tmp"), bytes).map_err(|e| internal(e.to_string()))?;
    fs::rename(format!("{REGISTRY}.tmp"), REGISTRY).map_err(|e| internal(e.to_string()))
}
fn load_model(requested: Option<&str>) -> Result<(String, Model)> {
    let models = runtime::models();
    let name = match requested {
        Some(name) if models.iter().any(|m| m == name) => name.to_string(),
        None => models
            .first()
            .cloned()
            .ok_or_else(|| Error(503, "No models configured".into(), None))?,
        Some(_) => {
            return Err(Error(
                404,
                "Unknown model; see /v1/models".into(),
                Some("model"),
            ));
        }
    };
    let model = Model::load(&name).map_err(internal)?;
    Ok((name, model))
}
fn sampler(req: &ChatRequest) -> Sampler {
    if req.temperature == 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature: req.temperature,
            p: req.top_p.unwrap_or(1.0),
        }
    }
}
async fn feed(
    ctx: &mut Context,
    tokens: &[u32],
    region: Option<&[(bool, u32)]>,
    sampling: Option<Sampler>,
) -> inferlet::Result<Option<u32>> {
    let mut first = None;
    let count = tokens.len().div_ceil(CHUNK);
    for (ci, chunk) in tokens.chunks(CHUNK).enumerate() {
        let mut pass = ctx.forward();
        pass.input(chunk);
        if let Some(region) = region {
            let masks: Vec<_> = (0..chunk.len())
                .map(|j| {
                    let mut runs = region.to_vec();
                    runs.push((true, (ci * CHUNK + j + 1) as u32));
                    brle(&runs)
                })
                .collect();
            pass.attention_mask(&masks);
        }
        let h = if ci + 1 == count {
            sampling
                .clone()
                .map(|s| pass.sample(&[chunk.len() as u32 - 1], s))
        } else {
            None
        };
        let out = pass.execute().await?;
        if let Some(h) = h {
            first = out.token(h);
        }
    }
    Ok(first)
}
fn render_cache(model: &Model, input: &CacheRequest) -> Result<(Vec<Vec<u32>>, Vec<u32>)> {
    if input.examples.len() > 256 || input.examples.iter().any(|e| e.trim().is_empty()) {
        return Err(bad("provide at most 256 nonempty examples", "examples"));
    }
    let tok = model.tokenizer();
    let marker = tok.encode("S1_RAG_MESSAGE_BOUNDARY_73f5");
    let wrapped = chat::user(model, "S1_RAG_MESSAGE_BOUNDARY_73f5");
    let hits: Vec<_> = wrapped
        .windows(marker.len())
        .enumerate()
        .filter(|(_, w)| *w == marker.as_slice())
        .map(|(i, _)| i)
        .collect();
    if hits.len() != 1 {
        return Err(internal("cannot split model user template"));
    }
    let split = hits[0];
    let mut prefix = chat::system(model, &input.system);
    prefix.extend_from_slice(&wrapped[..split]);
    prefix.extend(tok.encode(&input.header));
    let mut suffix = wrapped[split + marker.len()..].to_vec();
    suffix.extend(chat::cue(model));
    let mut blocks = vec![prefix];
    for (i, example) in input.examples.iter().enumerate() {
        let text = if i + 1 < input.examples.len() {
            format!("{example}\n\n------------\n")
        } else {
            example.clone()
        };
        blocks.push(tok.encode(&text));
    }
    Ok((blocks, suffix))
}
async fn create_cache(input: CacheRequest) -> Result<Value> {
    let (name, model) = load_model(input.model.as_deref())?;
    let id = cache_id(&name, &input);
    let (blocks, suffix) = render_cache(&model, &input)?;
    let tokens: usize = blocks.iter().map(Vec::len).sum();
    if tokens == 0 || tokens > MAX_CACHE_TOKENS {
        return Err(bad("cache must contain 1..=65536 tokens", "examples"));
    }
    let mut registry = load_registry()?;
    let started = Instant::now();
    let hit = registry.entries.iter().any(|c| c.id == id)
        && Context::open(&model, &id).is_ok_and(|c| c.seq_len() as usize == tokens);
    if !hit {
        // Bound live snapshot memory before allocating the new prefix.
        while registry.entries.len() >= MAX_CACHES
            || registry.entries.iter().map(|c| c.tokens).sum::<usize>() + tokens > MAX_CACHE_TOKENS
        {
            let old = registry.entries.remove(0);
            if let Ok(m) = Model::load(&old.model) {
                let _ = Context::delete(&m, &old.id);
            }
        }
        // Persist eviction immediately, including if prefill subsequently fails.
        save_registry(&registry)?;
        let _ = Context::delete(&model, &id);
        let mut base = Context::new(&model).map_err(internal)?;
        for block in &blocks {
            feed(&mut base, block, None, None).await.map_err(internal)?;
        }
        base.save(&id).map_err(internal)?;
    }
    let common = blocks[0].len() as u32;
    let slots = blocks[1..].iter().map(|b| b.len() as u32).collect();
    let cache = Cache {
        id: id.clone(),
        model: name,
        input,
        common,
        slots,
        suffix,
        tokens,
    };
    registry.entries.retain(|c| c.id != id);
    registry.entries.push(cache.clone());
    save_registry(&registry)?;
    Ok(
        json!({"id":id,"object":"pie.prompt_cache","model":cache.model,"example_count":cache.slots.len(),"prompt_tokens":tokens,"cache_hit":hit,"build_ms":started.elapsed().as_secs_f64()*1000.0}),
    )
}
async fn completion(req: &ChatRequest) -> Result<Value> {
    req.validate()?;
    let (model_name, model) = load_model(req.model.as_deref())?;
    let tok = model.tokenizer();
    let started = Instant::now();
    let (mut ctx, tail, r, base_len, selected, cache_hit) = if let Some(id) = &req.cache_id {
        let mut registry = load_registry()?;
        let cache = registry
            .entries
            .iter()
            .find(|c| &c.id == id && c.model == model_name)
            .cloned()
            .ok_or_else(|| {
                Error(
                    404,
                    "Cache not found; create it with POST /v1/prompt_caches".into(),
                    Some("cache_id"),
                )
            })?;
        let selected = selection(req.selected_examples.as_deref(), cache.slots.len())?;
        let ctx = Context::open(&model, id).map_err(|_| {
            Error(
                409,
                "KV snapshot expired; recreate the prompt cache".into(),
                Some("cache_id"),
            )
        })?;
        if ctx.seq_len() as usize != cache.tokens {
            return Err(internal("cached prefix length mismatch"));
        }
        let r = region(cache.common, &cache.slots, &selected);
        let mut tail = tok.encode(&format!("\n\n{}", req.messages[0].content));
        tail.extend_from_slice(&cache.suffix);
        registry.entries.retain(|c| &c.id != id);
        registry.entries.push(cache.clone());
        save_registry(&registry)?;
        (ctx, tail, r, cache.tokens as u32, selected, true)
    } else {
        let mut render = Context::new(&model).map_err(internal)?;
        let system = req
            .messages
            .iter()
            .filter(|m| matches!(m.role.as_str(), "system" | "developer"))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        if !system.is_empty() {
            render.system(&system);
        }
        for m in &req.messages {
            match m.role.as_str() {
                "user" => {
                    render.user(&m.content);
                }
                "assistant" => {
                    render.assistant(&m.content);
                }
                _ => {}
            }
        }
        render.cue();
        let tail = render.buffer().to_vec();
        drop(render);
        (
            Context::new(&model).map_err(internal)?,
            tail,
            vec![],
            0,
            vec![],
            false,
        )
    };
    let prompt_tokens = base_len as usize + tail.len();
    let mut cur = feed(
        &mut ctx,
        &tail,
        req.masking.then_some(r.as_slice()),
        Some(sampler(req)),
    )
    .await
    .map_err(internal)?
    .ok_or_else(|| internal("no first token"))?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    let reference = if req.prior_plan {
        tok.encode(req.previous_plan.as_deref().unwrap())
    } else {
        vec![]
    };
    let stop_tokens = chat::stop_tokens(&model);
    let mut generated = vec![];
    let mut sampled = 1usize;
    let mut proposed = 0;
    let mut accepted_drafts = 0;
    let mut steps = 0;
    let mut finished = stop_tokens.contains(&cur);
    if !finished {
        generated.push(cur);
    }
    let mut text = tok.decode(&generated).map_err(internal)?;
    if let Some(pos) = req.stop.as_ref().and_then(|s| s.earliest(&text)) {
        text.truncate(pos);
        finished = true;
    }
    let decode_start = Instant::now();
    while !finished && sampled < req.max() {
        let limit = req.draft_len.min(req.max().saturating_sub(sampled + 1));
        let drafts = if req.prior_plan {
            draft::continuation(&reference, &generated, req.match_tokens, limit)
        } else {
            vec![]
        };
        proposed += drafts.len();
        let mut tokens = vec![cur];
        tokens.extend_from_slice(&drafts);
        let start = ctx.seq_len();
        let mut pass = ctx.forward();
        pass.defer_commit();
        pass.input(&tokens);
        if req.masking {
            let masks: Vec<_> = (0..tokens.len())
                .map(|j| {
                    let mut runs = r.clone();
                    runs.push((true, start - base_len + j as u32 + 1));
                    brle(&runs)
                })
                .collect();
            pass.attention_mask(&masks);
        }
        let indices: Vec<_> = (0..tokens.len() as u32).collect();
        let h = pass.sample(&indices, sampler(req));
        let out = pass.execute().await.map_err(internal)?;
        let picks = out.tokens_at(h);
        if picks.len() != tokens.len() {
            return Err(internal("incomplete verification samples"));
        }
        let accepted = draft::accepted_prefix(&drafts, &picks);
        ctx.truncate((drafts.len() - accepted) as u32);
        if ctx.seq_len() != start + 1 + accepted as u32 {
            return Err(internal("rollback length mismatch"));
        }
        accepted_drafts += accepted;
        steps += 1;
        for &next in &picks[..accepted + 1] {
            cur = next;
            sampled += 1;
            if stop_tokens.contains(&cur) {
                finished = true;
                break;
            }
            generated.push(cur);
            text = tok.decode(&generated).map_err(internal)?;
            if let Some(pos) = req.stop.as_ref().and_then(|s| s.earliest(&text)) {
                text.truncate(pos);
                finished = true;
                break;
            }
            if sampled >= req.max() {
                break;
            }
        }
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(
        json!({"id":format!("chatcmpl-plan-mask-{:x}",created.as_nanos()),"object":"chat.completion","created":created.as_secs(),"model":model_name,
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"logprobs":null,"finish_reason":if finished{"stop"}else{"length"}}],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":generated.len(),"total_tokens":prompt_tokens+generated.len(),"prompt_tokens_details":{"cached_tokens":base_len}},
        "pie":{"prior_plan":req.prior_plan,"masking":req.masking,"selected_examples":selected,"cache_id":req.cache_id,"cache_hit":cache_hit,"cached_tokens":base_len,"prefilled_tokens":tail.len(),
        "attended_prompt_tokens":if req.masking {r.iter().filter(|(v,_)|*v).map(|(_,n)|*n as usize).sum::<usize>()+tail.len()}else{prompt_tokens},
        "drafts_proposed":proposed,"drafts_accepted":accepted_drafts,"drafts_rejected":proposed-accepted_drafts,"decode_steps":steps,"decode_accounted_tokens":sampled-1,
        "prefill_ms":prefill_ms,"decode_ms":decode_ms,"decode_tokens_per_second":(sampled-1) as f64*1000.0/decode_ms.max(1e-9)}}),
    )
}

#[wstd::http_server]
async fn main(mut req: Request<IncomingBody>, responder: Responder) -> Finished {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let result:Result<(Value,bool,bool)>=async {
        if method==Method::OPTIONS {return Ok((json!({}),false,false));}
        if method==Method::GET && path=="/health" {return Ok((json!({"status":"ok","service":"openai-plan-mask-service"}),false,false));}
        if method==Method::GET && path=="/v1/models" {return Ok((json!({"object":"list","data":runtime::models().iter().map(|id|json!({"id":id,"object":"model","created":0,"owned_by":"pie"})).collect::<Vec<_>>()}),false,false));}
        if method==Method::DELETE && path.starts_with("/v1/prompt_caches/") {
            let id=path.trim_start_matches("/v1/prompt_caches/");let mut registry=load_registry()?;
            let cache=registry.entries.iter().find(|c|c.id==id).cloned().ok_or_else(||Error(404,"Cache not found".into(),Some("cache_id")))?;
            if let Ok(model)=Model::load(&cache.model) {let _=Context::delete(&model,id);}
            registry.entries.retain(|c|c.id!=id);save_registry(&registry)?;
            return Ok((json!({"id":id,"object":"pie.prompt_cache","deleted":true}),false,false));
        }
        if method!=Method::POST || !matches!(path.as_str(),"/v1/chat/completions"|"/chat/completions"|"/v1/prompt_caches") {return Err(Error(404,"Endpoint not found".into(),None));}
        let mut bytes=vec![];let mut chunk=[0;8192];
        loop {let n=req.body_mut().read(&mut chunk).await.map_err(|e|bad(e.to_string(),"body"))?;if n==0 {break} if bytes.len()+n>MAX_BODY {return Err(Error(413,"Request body exceeds 2 MiB".into(),None));} bytes.extend_from_slice(&chunk[..n]);}
        if path=="/v1/prompt_caches" {
            let input:CacheRequest=serde_json::from_slice(&bytes).map_err(|e|bad(e.to_string(),"body"))?;
            Ok((create_cache(input).await?,false,false))
        } else {
            let input:ChatRequest=serde_json::from_slice(&bytes).map_err(|e|bad(e.to_string(),"body"))?;
            Ok((completion(&input).await?,input.stream,input.stream_options.as_ref().is_some_and(|o|o.include_usage)))
        }
    }.await;
    let (status,body,content_type)=match result {
        Ok((payload,true,usage))=>(200,sse(&payload,usage),"text/event-stream"),
        Ok((payload,false,_))=>(200,payload.to_string(),"application/json"),
        Err(Error(status,message,param))=>(status,json!({"error":{"message":message,"type":if status>=500{"server_error"}else{"invalid_request_error"},"param":param,"code":null}}).to_string(),"application/json"),
    };
    responder
        .respond(
            Response::builder()
                .status(status)
                .header("Content-Type", content_type)
                .header("Cache-Control", "no-cache")
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                .header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization",
                )
                .body(body.into_body())
                .unwrap(),
        )
        .await
}
/// Same buffered SSE convention as the existing openai-spec-service.
fn sse(response: &Value, include_usage: bool) -> String {
    let chunk = |delta: Value, finish: Value| json!({"id":response["id"],"object":"chat.completion.chunk","created":response["created"],"model":response["model"],"choices":[{"index":0,"delta":delta,"finish_reason":finish,"logprobs":null}]});
    let first = chunk(json!({"role":"assistant","content":""}), Value::Null);
    let content = chunk(
        json!({"content":response["choices"][0]["message"]["content"]}),
        Value::Null,
    );
    let mut last = chunk(json!({}), response["choices"][0]["finish_reason"].clone());
    last["pie"] = response["pie"].clone();
    let usage = if include_usage {
        let mut chunk = chunk(json!({}), Value::Null);
        chunk["choices"] = json!([]);
        chunk["usage"] = response["usage"].clone();
        format!("data: {chunk}\n\n")
    } else {
        String::new()
    };
    format!("data: {first}\n\ndata: {content}\n\ndata: {last}\n\n{usage}data: [DONE]\n\n")
}
