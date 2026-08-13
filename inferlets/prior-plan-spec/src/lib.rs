//! A/B test for reusing a previous response as a speculative draft.
//!
//! The verifier sees only `current_request`. `previous_plan` is tokenized
//! separately and is available only to the custom [`Speculator`], so the
//! experiment does not accidentally improve generation by adding the old
//! answer to the prompt.
//!
//! The drafter aligns the suffix of the new output with any matching span in
//! the previous plan and proposes the next `draft_len` tokens. A mismatch costs
//! one verification round, after which suffix matching can align again later in
//! the plan. Greedy sampling preserves the target model's output exactly.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use inferlet::{Context, Result, Speculator, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_current_request")]
    current_request: String,
    #[serde(default = "default_previous_plan")]
    previous_plan: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_draft_len")]
    draft_len: usize,
    #[serde(default = "default_match_tokens")]
    match_tokens: usize,
}

fn default_current_request() -> String {
    "Add email verification to the account signup flow. Include the data-model change, API behavior, resend handling, observability, rollout, and tests."
        .into()
}

fn default_previous_plan() -> String {
    "1. Inspect the existing password-reset flow, data model, API handlers, and test coverage.\n\
     2. Define the password-reset token lifecycle, validation rules, expiration behavior, and abuse limits.\n\
     3. Add the data-model changes and migrations needed to store password-reset state safely.\n\
     4. Implement the API handlers, delivery path, retry behavior, and user-facing error responses.\n\
     5. Add metrics and structured logs for requests, deliveries, completions, failures, and abuse controls.\n\
     6. Roll out behind a feature flag, monitor the new metrics, and document the rollback procedure.\n\
     7. Add unit, integration, and end-to-end tests for success, expiry, retries, invalid tokens, and rate limits."
        .into()
}

fn default_mode() -> String {
    "both".into()
}

fn default_system() -> String {
    "You are a software-planning assistant. Return only a concise numbered implementation plan. Use one complete sentence per step and cover architecture, implementation, observability, rollout, and tests. /no_think"
        .into()
}

fn default_max_tokens() -> usize {
    256
}

fn default_draft_len() -> usize {
    8
}

fn default_match_tokens() -> usize {
    2
}

#[derive(Serialize)]
struct Output {
    model: String,
    mode: String,
    previous_plan_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline: Option<RunMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speculated: Option<RunMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<Comparison>,
}

#[derive(Serialize)]
struct RunMetrics {
    text: String,
    generated_tokens: usize,
    verifier_steps: usize,
    prefill_ms: u128,
    decode_ms: u128,
    total_ms: u128,
    decode_tokens_per_second: f64,
    average_tokens_per_step: f64,
    draft_rounds: usize,
    drafts_proposed: usize,
    drafts_accepted: usize,
    draft_acceptance_rate: f64,
    longest_accepted_draft: usize,
    #[serde(skip)]
    token_ids: Vec<u32>,
    #[serde(skip)]
    decode_nanos: u128,
    #[serde(skip)]
    total_nanos: u128,
}

#[derive(Serialize)]
struct Comparison {
    outputs_match: bool,
    decode_speedup: f64,
    end_to_end_speedup: f64,
    decode_ms_saved: i128,
    total_ms_saved: i128,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let mode = input.mode.trim().to_ascii_lowercase();
    if !matches!(
        mode.as_str(),
        "baseline" | "plain" | "speculated" | "spec" | "both"
    ) {
        return Err(format!(
            "unknown mode '{}': expected 'baseline', 'speculated', or 'both'",
            input.mode
        ));
    }
    if input.max_tokens == 0 {
        return Err("max_tokens must be greater than zero".into());
    }
    if input.draft_len == 0 {
        return Err("draft_len must be greater than zero".into());
    }
    if input.match_tokens == 0 {
        return Err("match_tokens must be greater than zero".into());
    }

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;
    let previous_tokens = model.tokenizer().encode(&input.previous_plan);
    if previous_tokens.is_empty() && matches!(mode.as_str(), "speculated" | "spec" | "both") {
        return Err("previous_plan must contain at least one token in speculative mode".into());
    }

    let run_baseline = matches!(mode.as_str(), "baseline" | "plain" | "both");
    let run_speculated = matches!(mode.as_str(), "speculated" | "spec" | "both");

    let baseline = if run_baseline {
        Some(run_once(&model, &input, &previous_tokens, false).await?)
    } else {
        None
    };
    let speculated = if run_speculated {
        Some(run_once(&model, &input, &previous_tokens, true).await?)
    } else {
        None
    };

    let comparison = baseline
        .as_ref()
        .zip(speculated.as_ref())
        .map(|(plain, spec)| Comparison {
            outputs_match: plain.token_ids == spec.token_ids,
            decode_speedup: ratio(plain.decode_nanos, spec.decode_nanos),
            end_to_end_speedup: ratio(plain.total_nanos, spec.total_nanos),
            decode_ms_saved: plain.decode_ms as i128 - spec.decode_ms as i128,
            total_ms_saved: plain.total_ms as i128 - spec.total_ms as i128,
        });

    Ok(Output {
        model: model_name,
        mode,
        previous_plan_tokens: previous_tokens.len(),
        baseline,
        speculated,
        comparison,
    })
}

async fn run_once(
    model: &Model,
    input: &Input,
    previous_tokens: &[u32],
    use_previous_plan: bool,
) -> Result<RunMetrics> {
    let mut ctx = Context::new(model)?;
    ctx.system(&input.system);
    ctx.user(&format!(
        "Create an implementation plan for this request:\n\n{}",
        input.current_request
    ));
    ctx.cue();

    // Keep prefill separate from decode. Both A/B paths use the same split so
    // custom verification kernel shapes are the only intentional difference.
    let prefill_start = Instant::now();
    ctx.flush().await?;
    let prefill_elapsed = prefill_start.elapsed();

    let decode_start = Instant::now();
    let cue = chat::cue(model);
    let trigger = *cue.last().ok_or("chat template produced an empty cue")?;
    let first_token = {
        let mut pass = ctx.forward();
        pass.input(&[trigger]);
        let handle = pass.sample(&[0], Sampler::Argmax);
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

    // The sampled token becomes the pending verifier anchor for the first
    // Generator iteration. Drafts begin at the next absolute position.
    ctx.append(&[first_token]);
    let cursor = ctx.seq_len() + ctx.buffer().len() as u32;
    let spec_stats = Arc::new(Mutex::new(SpecStats::default()));
    let drafter = PreviousPlanDrafter::new(
        previous_tokens.to_vec(),
        vec![first_token],
        input.match_tokens,
        input.draft_len,
        cursor,
        Arc::clone(&spec_stats),
    );

    if !first_is_stop && token_ids.len() < input.max_tokens {
        let mut generator = ctx
            .generate(Sampler::Argmax)
            .max_tokens(input.max_tokens - token_ids.len())
            .stop(&stop_tokens);
        generator = if use_previous_plan {
            generator.speculator(drafter)
        } else {
            // A model can advertise a host-side default drafter. Disable it so
            // the baseline is truly one-token-per-verifier-step.
            generator.disable_system_speculation()
        };

        while let Some(step) = generator.next()? {
            let output = step.execute().await?;
            verifier_steps += 1;
            token_ids.extend_from_slice(&output.tokens);
        }
    }

    let decode_elapsed = decode_start.elapsed();
    let total_elapsed = prefill_elapsed + decode_elapsed;
    let stats = spec_stats.lock().unwrap();
    let generated_tokens = token_ids.len();
    let draft_acceptance_rate = if stats.proposed == 0 {
        0.0
    } else {
        stats.accepted as f64 / stats.proposed as f64
    };

    Ok(RunMetrics {
        text: model.tokenizer().decode(&token_ids)?,
        generated_tokens,
        verifier_steps,
        prefill_ms: prefill_elapsed.as_millis(),
        decode_ms: decode_elapsed.as_millis(),
        total_ms: total_elapsed.as_millis(),
        decode_tokens_per_second: throughput(generated_tokens, decode_elapsed),
        average_tokens_per_step: generated_tokens as f64 / verifier_steps.max(1) as f64,
        draft_rounds: stats.draft_rounds,
        drafts_proposed: stats.proposed,
        drafts_accepted: stats.accepted,
        draft_acceptance_rate,
        longest_accepted_draft: stats.longest_accepted_draft,
        token_ids,
        decode_nanos: decode_elapsed.as_nanos(),
        total_nanos: total_elapsed.as_nanos(),
    })
}

fn throughput(tokens: usize, elapsed: Duration) -> f64 {
    tokens as f64 / elapsed.as_secs_f64().max(1e-9)
}

fn ratio(baseline_ms: u128, candidate_ms: u128) -> f64 {
    baseline_ms as f64 / (candidate_ms as f64).max(1.0)
}

#[derive(Default)]
struct SpecStats {
    draft_rounds: usize,
    proposed: usize,
    accepted: usize,
    longest_accepted_draft: usize,
}

/// Drafts continuations from a fixed previous response. `history` contains
/// only target-model-approved tokens from the current response.
struct PreviousPlanDrafter {
    reference: Vec<u32>,
    history: Vec<u32>,
    match_tokens: usize,
    draft_len: usize,
    cursor: u32,
    last_proposed: usize,
    stats: Arc<Mutex<SpecStats>>,
}

impl PreviousPlanDrafter {
    fn new(
        reference: Vec<u32>,
        history: Vec<u32>,
        match_tokens: usize,
        draft_len: usize,
        cursor: u32,
        stats: Arc<Mutex<SpecStats>>,
    ) -> Self {
        Self {
            reference,
            history,
            match_tokens,
            draft_len,
            cursor,
            last_proposed: 0,
            stats,
        }
    }

    /// Find a reference occurrence of the current suffix. When a suffix occurs
    /// more than once, prefer the candidate with the longest matching history
    /// before it; this avoids jumping to the wrong repeated list marker.
    fn continuation(&self) -> Vec<u32> {
        let n = self.match_tokens;
        if self.history.len() < n || self.reference.len() <= n {
            return Vec::new();
        }

        let suffix = &self.history[self.history.len() - n..];
        let mut best: Option<(usize, usize)> = None; // (backward match, continuation start)

        for start in 0..=self.reference.len() - n {
            let continuation_start = start + n;
            if continuation_start >= self.reference.len()
                || &self.reference[start..continuation_start] != suffix
            {
                continue;
            }

            let mut backward_match = n;
            while backward_match < self.history.len()
                && backward_match < continuation_start
                && self.history[self.history.len() - backward_match - 1]
                    == self.reference[continuation_start - backward_match - 1]
            {
                backward_match += 1;
            }

            if best.is_none_or(|(best_match, best_start)| {
                backward_match > best_match
                    || (backward_match == best_match && continuation_start > best_start)
            }) {
                best = Some((backward_match, continuation_start));
            }
        }

        let Some((_, start)) = best else {
            return Vec::new();
        };
        let end = (start + self.draft_len).min(self.reference.len());
        self.reference[start..end].to_vec()
    }
}

impl Speculator for PreviousPlanDrafter {
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        let drafts = self.continuation();
        self.last_proposed = drafts.len();
        if !drafts.is_empty() {
            self.stats.lock().unwrap().draft_rounds += 1;
        }
        let positions = (self.cursor..self.cursor + drafts.len() as u32).collect();
        (drafts, positions)
    }

    fn accept(&mut self, accepted: &[u32]) {
        // accepted[0] is the target model's free pick. Any remaining tokens,
        // up to `last_proposed`, correspond to accepted previous-plan drafts.
        let accepted_drafts = accepted.len().saturating_sub(1).min(self.last_proposed);
        {
            let mut stats = self.stats.lock().unwrap();
            stats.proposed += self.last_proposed;
            stats.accepted += accepted_drafts;
            stats.longest_accepted_draft = stats.longest_accepted_draft.max(accepted_drafts);
        }
        self.last_proposed = 0;
        self.history.extend_from_slice(accepted);
        self.cursor += accepted.len() as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter(reference: &[u32], history: &[u32], match_tokens: usize) -> PreviousPlanDrafter {
        PreviousPlanDrafter::new(
            reference.to_vec(),
            history.to_vec(),
            match_tokens,
            3,
            100,
            Arc::new(Mutex::new(SpecStats::default())),
        )
    }

    #[test]
    fn drafts_the_previous_plan_continuation() {
        let d = drafter(&[10, 11, 12, 13, 14], &[10, 11], 2);
        assert_eq!(d.continuation(), vec![12, 13, 14]);
    }

    #[test]
    fn resumes_after_request_specific_tokens_diverge() {
        let d = drafter(&[1, 2, 90, 4, 5, 6, 7], &[1, 2, 99, 4, 5], 2);
        assert_eq!(d.continuation(), vec![6, 7]);
    }

    #[test]
    fn repeated_suffix_uses_the_longest_backward_alignment() {
        let d = drafter(&[7, 1, 2, 9, 4, 1, 2, 8], &[7, 1, 2], 2);
        assert_eq!(d.continuation(), vec![9, 4, 1]);
    }

    #[test]
    fn tracks_only_verified_draft_tokens_as_accepted() {
        let stats = Arc::new(Mutex::new(SpecStats::default()));
        let mut d =
            PreviousPlanDrafter::new(vec![1, 2, 3, 4], vec![1], 1, 3, 100, Arc::clone(&stats));
        let (drafts, positions) = d.draft();
        assert_eq!(drafts, vec![2, 3, 4]);
        assert_eq!(positions, vec![100, 101, 102]);

        d.accept(&[2, 3]);
        let stats = stats.lock().unwrap();
        assert_eq!(stats.proposed, 3);
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.longest_accepted_draft, 1);
    }
}
