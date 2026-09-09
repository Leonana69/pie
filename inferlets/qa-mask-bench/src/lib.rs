//! Universal "prefix-cache masking" experiment inferlet (current-SDK port).
//!
//! One invocation handles a single (round, case). A *round* is 6 self-contained
//! `(info, question, answer)` items plus a list of *asks*; each ask poses two of
//! the round's questions at once. The two infos a given ask needs sit at two
//! random positions among the 6 cached infos — that is the "random KV-cache
//! access" pattern we study.
//!
//! ## Porting note (old SDK → current SDK)
//!
//! The original inferlet targeted a two-generations-old SDK whose primitives
//! (`Queue::export_kv_pages` / `import_kv_pages`, `Context::from_imported_state`,
//! the persistent `Context::mask_token_range`, `drop_masked_kv_pages`, the manual
//! `decode_step` loop) no longer exist. This port re-expresses the same
//! experiment on the current API:
//!
//! * **KV reuse** — the round's base context is built once, `save`d under a name,
//!   and re-`open`ed (implicit fork) per ask. Replaces export/import.
//! * **Masking** — there is no persistent context mask anymore. Masking is now a
//!   *per-forward-pass* attention mask: one BRLE (binary run-length encoding,
//!   `[n_false, n_true, n_false, …]`) per query position, supplied on every
//!   prefill/decode pass via `Forward::attention_mask`.
//! * **Page drop** — there is no explicit `drop_masked_kv_pages`. Instead, when an
//!   attention mask marks an *entire page* of KV as unattended, the runtime's
//!   **page-trim optimization** physically excludes that page from the kernel.
//!   The slotted, page-aligned layout (cases 2 & 3) is built precisely so that
//!   masking an unused info masks whole pages → page-trim shrinks resident KV.
//!   Cases 5 & 6 pack infos contiguously, so a masked info sits mid-page and is
//!   *not* trimmed (decode KV stays full) — matching their original semantics.
//!
//! Four+two caching strategies (all share one base context built once per round,
//! then reused across that round's asks):
//!
//! * Case 1 — system + all 6 infos cached (cascade prefill); every ask attends to
//!   all 6 infos (plain causal, no mask). Cheapest prefill, heaviest decode.
//! * Case 2 — system + 6 infos cascade-prefilled into page-aligned slots. Per ask
//!   the 4 unused slots are masked (whole pages → page-trim) so they leave the
//!   forward pass; attend system + 2 used infos.
//! * Case 3 — like Case 2 but infos are prefilled *exclusively* (each info attends
//!   only to the system prompt), so its K/V carries no cross-info contamination.
//! * Case 4 — only the system prompt is cached; per ask the 2 used infos are
//!   prefilled inline alongside the questions. Heaviest prefill, no cached infos.
//! * Case 5 — infos packed CONTIGUOUSLY (like case 1); per ask the unused infos are
//!   masked but NOT page-aligned, so nothing is trimmed (decode KV stays full).
//!   Cascade base.
//! * Case 6 — like Case 5 but exclusive base (each info prefilled attending only to
//!   the system prompt).

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use std::collections::HashSet;
use std::time::{Duration, Instant};

const SYSTEM_PROMPT: &str = "You are a precise question-answering assistant. You are given several numbered documents and then asked two questions. Use ONLY the documents to answer. Each answer must be the shortest exact span the documents support (a name, a year, a number, or a city). Reply with exactly two lines and nothing else, in this format:\nA1: <answer to question 1>\nA2: <answer to question 2>";

const DEFAULT_MAX_OUTPUT_TOKENS: usize = 128;

#[derive(Deserialize)]
struct Item {
    info: String,
    question: String,
    answer: String,
}

#[derive(Deserialize)]
struct Round {
    round_id: u64,
    items: Vec<Item>,
    /// Each ask is a pair of indices into `items` (the two questions asked together).
    asks: Vec<Vec<usize>>,
    /// Optional neutral filler appended to the system prompt (token-scale sweep).
    #[serde(default)]
    sys_extra: Option<String>,
}

#[derive(Deserialize)]
struct Input {
    case: usize,
    round: Round,
    /// Cap on generated tokens per answer (default 128).
    #[serde(default)]
    max_tokens: Option<usize>,
}

/// The system prompt for this round, optionally inflated with neutral filler.
fn effective_system(round: &Round) -> String {
    match &round.sys_extra {
        Some(s) if !s.is_empty() => format!("{SYSTEM_PROMPT}\n\n{s}"),
        _ => SYSTEM_PROMPT.to_string(),
    }
}

fn make_info_text(idx: usize, content: &str) -> String {
    format!("Document {}:\n{}", idx + 1, content)
}

fn make_query_text(q1: &str, q2: &str) -> String {
    // Trailing `/no_think` disables Qwen3's reasoning block so the model emits
    // the short `A1:/A2:` answer directly (the original targeted a non-thinking
    // model). Models that don't recognize the marker treat it as inert text.
    format!("Q1: {q1}\nQ2: {q2}\n/no_think")
}

// ============================================================
// BRLE attention-mask construction
// ============================================================
//
// An attention mask is one BRLE per query position. A BRLE is alternating run
// lengths over the KV positions the query may attend, **starting with a False
// (masked) run**: `[n_false, n_true, n_false, n_true, …]`. We build masks from a
// list of `(attended, len)` runs and let this encoder normalize them.

/// Encode `(attended, len)` runs into a BRLE (`[n_false, n_true, …]`). Adjacent
/// same-value runs are merged and zero-length runs dropped; a leading `0` is
/// emitted when the first run is attended (True), to keep the false-first
/// alternation the host expects.
fn brle_from_runs(runs: &[(bool, u32)]) -> Vec<u32> {
    let mut merged: Vec<(bool, u32)> = Vec::new();
    for &(v, l) in runs {
        if l == 0 {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            if last.0 == v {
                last.1 += l;
                continue;
            }
        }
        merged.push((v, l));
    }

    let mut out = Vec::new();
    let mut expect = false; // the next run we emit represents this attended-value
    let mut idx = 0;
    while idx < merged.len() {
        let (v, l) = merged[idx];
        if v == expect {
            out.push(l);
            idx += 1;
        } else {
            // No run of `expect` here — emit an empty one to preserve alternation.
            out.push(0);
        }
        expect = !expect;
    }
    out
}

/// One info's footprint inside the base context. `span == real_len` for the
/// contiguous layout (cases 1/5/6); for the slotted layout (cases 2/3) `span` is
/// padded up to the next page boundary, so a masked slot covers whole pages.
#[derive(Clone, Copy)]
struct Slot {
    real_len: u32,
    span: u32,
}

/// The round's reusable base context, saved under `name`.
struct Base {
    name: String,
    base_len: u32,
    sys_real_len: u32,
    /// System footprint including page padding (`== sys_real_len` when not slotted).
    sys_span: u32,
    slots: Vec<Slot>,
    register_ms: u128,
}

/// Runs covering `[0 .. base_len)` for a query that attends `used` infos.
/// Attended: system real tokens + each used info's real tokens. Masked: system
/// padding, used infos' padding, and every unused info's whole slot.
fn region_runs(base: &Base, used: &HashSet<usize>) -> Vec<(bool, u32)> {
    let mut runs: Vec<(bool, u32)> = vec![(true, base.sys_real_len)];
    if base.sys_span > base.sys_real_len {
        runs.push((false, base.sys_span - base.sys_real_len));
    }
    for (i, s) in base.slots.iter().enumerate() {
        if used.contains(&i) {
            runs.push((true, s.real_len));
            if s.span > s.real_len {
                runs.push((false, s.span - s.real_len));
            }
        } else {
            runs.push((false, s.span));
        }
    }
    runs
}

/// Prefix runs over `[0 .. next_slot_start)` for prefilling info `i`. Cascade
/// (`exclusive == false`): attend system + every earlier info's real tokens, mask
/// all padding. Exclusive: attend only the system's real tokens, mask everything
/// after (earlier infos and all padding).
fn prefill_prefix_runs(
    sys_real_len: u32,
    sys_span: u32,
    slots: &[Slot],
    exclusive: bool,
) -> Vec<(bool, u32)> {
    let mut runs: Vec<(bool, u32)> = vec![(true, sys_real_len)];
    if sys_span > sys_real_len {
        runs.push((false, sys_span - sys_real_len));
    }
    for s in slots {
        if exclusive {
            runs.push((false, s.span));
        } else {
            runs.push((true, s.real_len));
            if s.span > s.real_len {
                runs.push((false, s.span - s.real_len));
            }
        }
    }
    runs
}

// ============================================================
// Low-level forward helpers
// ============================================================

/// Prefill `tokens` with the default causal mask (no custom attention).
async fn feed_plain(ctx: &mut Context, tokens: &[u32]) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let mut pass = ctx.forward();
    pass.input(tokens);
    pass.execute().await?;
    Ok(())
}

/// Prefill `tokens` with a custom attention mask: token `j` (at absolute
/// position `chunk_start + j`) attends `prefix_runs` (covering `[0..chunk_start)`)
/// plus a causal True run over `[chunk_start ..= chunk_start + j]`.
async fn feed_masked(ctx: &mut Context, tokens: &[u32], prefix_runs: &[(bool, u32)]) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let mut pass = ctx.forward();
    pass.input(tokens);
    let masks: Vec<Vec<u32>> = (0..tokens.len() as u32)
        .map(|j| {
            let mut r = prefix_runs.to_vec();
            r.push((true, j + 1));
            brle_from_runs(&r)
        })
        .collect();
    pass.attention_mask(&masks);
    pass.execute().await?;
    Ok(())
}

/// Advance the context to the next page boundary with masked-everywhere filler.
async fn pad_to_page_boundary(ctx: &mut Context) -> Result<()> {
    let page = ctx.page_size();
    let rem = ctx.seq_len() % page;
    if rem == 0 {
        return Ok(());
    }
    let pad = (page - rem) as usize;
    let zeros = vec![0u32; pad];
    feed_plain(ctx, &zeros).await
}

struct Decoded {
    text: String,
    prefill: Duration,
    decode: Duration,
    output_tokens: usize,
}

/// Prefill `prefill_tokens` (sampling the first answer token off the last one),
/// then greedily decode until a stop token or `max_out`. When `region` is `Some`,
/// every pass carries an attention mask = `region` (covering `[0..base_len)`)
/// plus a causal True tail over the query/generated tokens; when `None`, the
/// runtime's causal mask is used (attend everything).
async fn generate_answer(
    ctx: &mut Context,
    model: &Model,
    prefill_tokens: &[u32],
    region: Option<&[(bool, u32)]>,
    base_len: u32,
    stop: &[u32],
    max_out: usize,
) -> Result<Decoded> {
    // ── Prefill (TTFT) ──
    let prefill_start = Instant::now();
    let first = {
        let mut pass = ctx.forward();
        pass.input(prefill_tokens);
        if let Some(rr) = region {
            let masks: Vec<Vec<u32>> = (0..prefill_tokens.len() as u32)
                .map(|j| {
                    let mut r = rr.to_vec();
                    r.push((true, j + 1));
                    brle_from_runs(&r)
                })
                .collect();
            pass.attention_mask(&masks);
        }
        let h = pass.sample(&[prefill_tokens.len() as u32 - 1], Sampler::Argmax);
        let out = pass.execute().await?;
        out.token(h).ok_or("empty prefill output")?
    };
    let prefill = prefill_start.elapsed();

    // ── Decode loop ──
    let mut decoder = chat::Decoder::new(model);
    let mut text = String::new();
    let mut output_tokens = 0usize;
    let mut cur = first;

    let decode_start = Instant::now();
    loop {
        output_tokens += 1;
        match decoder.feed(&[cur])? {
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            chat::Event::Delta(s) => text.push_str(&s),
            _ => {}
        }
        if stop.contains(&cur) || output_tokens >= max_out {
            break;
        }

        let p = ctx.seq_len();
        let mut pass = ctx.forward();
        pass.input(&[cur]);
        if let Some(rr) = region {
            let mut r = rr.to_vec();
            r.push((true, p + 1 - base_len));
            pass.attention_mask(&[brle_from_runs(&r)]);
        }
        let h = pass.sample(&[0], Sampler::Argmax);
        let out = pass.execute().await?;
        cur = out.token(h).ok_or("empty decode output")?;
    }
    let decode = decode_start.elapsed();

    Ok(Decoded {
        text,
        prefill,
        decode,
        output_tokens,
    })
}

// ============================================================
// Answer parsing + reporting
// ============================================================

/// Drop `<think>…</think>` reasoning blocks (Qwen3 etc. emit them by default).
fn strip_think(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    loop {
        match rest.find("<think>") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(i) => {
                out.push_str(&rest[..i]);
                let after = &rest[i + "<think>".len()..];
                match after.find("</think>") {
                    Some(e) => rest = &after[e + "</think>".len()..],
                    None => break,
                }
            }
        }
    }
    out
}

/// Extract the value on the same line after `tag` (e.g. "A1:").
fn extract_after(text: &str, tag: &str) -> String {
    match text.find(tag) {
        Some(pos) => text[pos + tag.len()..]
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string(),
        None => String::new(),
    }
}

fn parse_two_answers(raw: &str) -> (String, String) {
    let clean = strip_think(raw)
        .replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .replace("</s>", "");
    let a1 = extract_after(&clean, "A1:");
    let a2 = extract_after(&clean, "A2:");
    (a1, a2)
}

struct AskReport<'a> {
    round_id: u64,
    case: usize,
    ask_idx: usize,
    used: &'a [usize],
    items: &'a [Item],
    register_ms: u128,
    sys_tokens: u32,
    info_tokens: u32,
    attended_tokens: u32,
    kv_resident_tokens: u32,
    prompt_tokens: u32,
    decoded: Decoded,
}

fn print_ask(r: &AskReport) {
    let (a1, a2) = parse_two_answers(&r.decoded.text);
    let i1 = r.used[0];
    let i2 = r.used[1];
    let per_token_us = if r.decoded.output_tokens > 1 {
        (r.decoded.decode.as_micros() as u64) / (r.decoded.output_tokens as u64 - 1)
    } else {
        0
    };
    println!("[QA_RESULT]");
    println!("round_id={}", r.round_id);
    println!("case={}", r.case);
    println!("ask_idx={}", r.ask_idx);
    println!("used={:?}", r.used);
    println!("q1_idx={i1}");
    println!("q2_idx={i2}");
    println!("gold1={}", r.items[i1].answer.replace('\n', " "));
    println!("gold2={}", r.items[i2].answer.replace('\n', " "));
    println!("ans1={}", a1.replace('\n', " "));
    println!("ans2={}", a2.replace('\n', " "));
    println!("sys_tokens={}", r.sys_tokens);
    println!("info_tokens={}", r.info_tokens);
    println!("attended_tokens={}", r.attended_tokens);
    println!("kv_resident_tokens={}", r.kv_resident_tokens);
    println!("prompt_tokens={}", r.prompt_tokens);
    println!("register_ms={}", r.register_ms);
    println!("prefill_ms={}", r.decoded.prefill.as_millis());
    println!("decode_ms={}", r.decoded.decode.as_millis());
    println!("output_tokens={}", r.decoded.output_tokens);
    println!("per_token_us={per_token_us}");
    println!("raw={}", r.decoded.text.replace('\n', " | "));
    println!("[/QA_RESULT]");
}

// ============================================================
// Base construction (cases 1/2/3/5/6 — system + 6 infos)
// ============================================================

/// Build the round's base context (system + 6 infos), save it under a name, and
/// return its layout. `slotted` page-aligns each info; `exclusive` prefills each
/// info attending only to the system prompt.
async fn build_base(
    model: &Model,
    round: &Round,
    case: usize,
    slotted: bool,
    exclusive: bool,
) -> Result<Base> {
    let name = format!("qamask_{}_c{}", round.round_id, case);
    let mut ctx = Context::new(model)?;

    let reg_start = Instant::now();

    // System prompt.
    let sys_tokens = chat::system(model, &effective_system(round));
    feed_plain(&mut ctx, &sys_tokens).await?;
    let sys_real_len = ctx.seq_len();
    let mut sys_span = sys_real_len;
    if slotted {
        pad_to_page_boundary(&mut ctx).await?;
        sys_span = ctx.seq_len();
    }

    // Infos.
    let mut slots: Vec<Slot> = Vec::with_capacity(round.items.len());
    for (i, it) in round.items.iter().enumerate() {
        let slot_start = ctx.seq_len();
        let toks = chat::user(model, &make_info_text(i, &it.info));

        if slotted || exclusive {
            let prefix = prefill_prefix_runs(sys_real_len, sys_span, &slots, exclusive);
            feed_masked(&mut ctx, &toks, &prefix).await?;
        } else {
            // Cascade + contiguous: plain causal prefill attends system + all
            // earlier infos automatically.
            feed_plain(&mut ctx, &toks).await?;
        }
        let real_len = ctx.seq_len() - slot_start;

        let mut span = real_len;
        if slotted {
            pad_to_page_boundary(&mut ctx).await?;
            span = ctx.seq_len() - slot_start;
        }
        slots.push(Slot { real_len, span });
    }
    let register_ms = reg_start.elapsed().as_millis();
    let base_len = ctx.seq_len();

    let _ = Context::delete(model, &name);
    ctx.save(&name)?;
    drop(ctx);

    Ok(Base {
        name,
        base_len,
        sys_real_len,
        sys_span,
        slots,
        register_ms,
    })
}

/// Cases 1/2/3/5/6: build the base once, then per ask reopen it, attend the two
/// used infos (masking the rest when `mask_unused`), and answer.
async fn run_cached_case(
    model: &Model,
    round: &Round,
    case: usize,
    slotted: bool,
    exclusive: bool,
    mask_unused: bool,
    max_out: usize,
) -> Result<()> {
    let base = build_base(model, round, case, slotted, exclusive).await?;
    let stop = chat::stop_tokens(model);

    for (ask_idx, used) in round.asks.iter().enumerate() {
        let used_set: HashSet<usize> = used.iter().copied().collect();
        let region = if mask_unused {
            Some(region_runs(&base, &used_set))
        } else {
            None
        };

        let mut ctx = Context::open(model, &base.name)?;

        let mut prefill = chat::user(
            model,
            &make_query_text(&round.items[used[0]].question, &round.items[used[1]].question),
        );
        prefill.extend(chat::cue(model));
        let prompt_tokens = prefill.len() as u32;

        let decoded = generate_answer(
            &mut ctx,
            model,
            &prefill,
            region.as_deref(),
            base.base_len,
            &stop,
            max_out,
        )
        .await?;

        let info_tokens: u32 = used.iter().map(|&i| base.slots[i].real_len).sum();
        let attended_tokens = if case == 1 {
            base.base_len + prompt_tokens
        } else {
            base.sys_real_len + info_tokens + prompt_tokens
        };
        // Resident KV during decode: cases 2/3 page-trim the unused slots away;
        // cases 1/5/6 keep the full base resident.
        let kv_resident_tokens = if slotted {
            base.sys_span + used.iter().map(|&i| base.slots[i].span).sum::<u32>() + prompt_tokens
        } else {
            base.base_len + prompt_tokens
        };

        print_ask(&AskReport {
            round_id: round.round_id,
            case,
            ask_idx,
            used,
            items: &round.items,
            register_ms: base.register_ms,
            sys_tokens: base.sys_real_len,
            info_tokens,
            attended_tokens,
            kv_resident_tokens,
            prompt_tokens,
            decoded,
        });
    }

    let _ = Context::delete(model, &base.name);
    Ok(())
}

// ============================================================
// Case 4: only the system prompt cached; the 2 used infos are prefilled inline
// with the questions on every ask.
// ============================================================
async fn run_case4(model: &Model, round: &Round, max_out: usize) -> Result<()> {
    let name = format!("qamask_{}_c4", round.round_id);
    let mut ctx = Context::new(model)?;

    let reg_start = Instant::now();
    let sys_tokens = chat::system(model, &effective_system(round));
    feed_plain(&mut ctx, &sys_tokens).await?;
    let register_ms = reg_start.elapsed().as_millis();
    let sys_len = ctx.seq_len();

    let _ = Context::delete(model, &name);
    ctx.save(&name)?;
    drop(ctx);

    let stop = chat::stop_tokens(model);

    for (ask_idx, used) in round.asks.iter().enumerate() {
        let mut ctx = Context::open(model, &name)?;

        let mut prefill: Vec<u32> = Vec::new();
        let mut info_tokens = 0u32;
        for &i in used {
            let t = chat::user(model, &make_info_text(i, &round.items[i].info));
            info_tokens += t.len() as u32;
            prefill.extend(t);
        }
        let before_query = prefill.len() as u32;
        prefill.extend(chat::user(
            model,
            &make_query_text(&round.items[used[0]].question, &round.items[used[1]].question),
        ));
        prefill.extend(chat::cue(model));
        let prompt_tokens = prefill.len() as u32 - before_query;

        let decoded =
            generate_answer(&mut ctx, model, &prefill, None, sys_len, &stop, max_out).await?;

        let attended_tokens = sys_len + info_tokens + prompt_tokens;
        print_ask(&AskReport {
            round_id: round.round_id,
            case: 4,
            ask_idx,
            used,
            items: &round.items,
            register_ms,
            sys_tokens: sys_len,
            info_tokens,
            attended_tokens,
            kv_resident_tokens: attended_tokens,
            prompt_tokens,
            decoded,
        });
    }

    let _ = Context::delete(model, &name);
    Ok(())
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let round = &input.round;
    if round.items.is_empty() || round.asks.is_empty() {
        return Err("round has no items or asks".to_string());
    }
    let max_out = input.max_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS).max(1);

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    match input.case {
        1 => run_cached_case(&model, round, 1, false, false, false, max_out).await?,
        2 => run_cached_case(&model, round, 2, true, false, true, max_out).await?,
        3 => run_cached_case(&model, round, 3, true, true, true, max_out).await?,
        4 => run_case4(&model, round, max_out).await?,
        5 => run_cached_case(&model, round, 5, false, false, true, max_out).await?,
        6 => run_cached_case(&model, round, 6, false, true, true, max_out).await?,
        c => return Err(format!("invalid case {c} (expected 1..6)")),
    }

    Ok(String::new())
}
