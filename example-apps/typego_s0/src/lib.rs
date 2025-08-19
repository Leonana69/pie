use inferlet::{self, context::Context, traits::allocate::Allocate, wstd};
use pico_args::Arguments;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::time::Instant;
use blake3; // for hashing unique prefix keys

const DEFAULT_PREFIX: &str = r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant."#;

#[derive(Serialize, Deserialize)]
struct CachedPrefixState {
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // --- Args ---
    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "What is the capital of Washington State?".to_string());

    let max_num_outputs: usize = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(128);

    let invalidate_cache: bool = args.contains(["-i", "--invalidate-cache"]);

    let prefix_arg = args
        .opt_value_from_str(["-x", "--prefix"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| DEFAULT_PREFIX.to_string());

    // --- Cache keys depend on prefix hash ---
    let prefix_hash = blake3::hash(prefix_arg.as_bytes()).to_hex().to_string();
    let cache_flag_key = format!("prefix_loaded_{}", prefix_hash);
    let cache_export_name = format!("prefix_export_{}", prefix_hash);
    let cache_state_key = format!("prefix_state_{}", prefix_hash);

    // --- Setup ---
    let start = Instant::now();
    let model = inferlet::get_auto_model();
    let queue = model.create_queue();
    let tokenizer = model.get_tokenizer();
    let mut ctx: Context;

    // --- Cache invalidation ---
    if invalidate_cache && inferlet::store_get(&cache_flag_key) == Some("true".to_string()) {
        queue.unexport_kv_pages(cache_export_name.clone());
        inferlet::store_set(&cache_flag_key, "false");
    }

    // --- Cache hit ---
    if inferlet::store_get(&cache_flag_key) == Some("true".to_string()) {
        println!("✅ Cache HIT. Loading prefix from KV store.");

        let imported_page_ids = queue.import_kv_pages(cache_export_name.clone());
        let state_json =
            inferlet::store_get(&cache_state_key).ok_or("Cache Inconsistency: State missing")?;
        let state: CachedPrefixState = serde_json::from_str(&state_json).unwrap();

        ctx = Context::from_imported_state(
            &model,
            imported_page_ids,
            state.token_ids,
            state.kv_page_last_len,
        );
    } else {
        // --- Cache miss ---
        println!("🐌 Cache MISS. Computing and caching prefix.");

        let mut prefill_ctx = model.create_context();
        prefill_ctx.fill(&prefix_arg);
        prefill_ctx.flush();

        let page_ids = prefill_ctx.get_kv_page_ids().to_vec();
        let state_to_cache = CachedPrefixState {
            token_ids: prefill_ctx.get_token_ids().to_vec(),
            kv_page_last_len: prefill_ctx.get_kv_page_last_len(),
        };

        prefill_ctx
            .queue()
            .export_kv_pages(&page_ids, cache_export_name.clone(), true);

        let state_json = serde_json::to_string(&state_to_cache).unwrap();
        inferlet::store_set(&cache_state_key, &state_json);
        inferlet::store_set(&cache_flag_key, "true");

        ctx = prefill_ctx;
    }

    // --- Generation & Output ---
    let final_prompt = format!(
        "{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt
    );
    ctx.fill(&final_prompt);

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;
    let token_ids = tokenizer.tokenize(&text);
    println!("Output: {:?} (total elapsed: {:?})", text, start.elapsed());

    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    inferlet::send(&text);
    Ok(())
}
