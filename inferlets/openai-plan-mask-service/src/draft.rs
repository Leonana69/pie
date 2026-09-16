//! Suffix matcher ported from pie/inferlets/prior-plan-spec (same tie-breaking).
pub fn continuation(
    reference: &[u32],
    history: &[u32],
    match_tokens: usize,
    draft_len: usize,
) -> Vec<u32> {
    let n = match_tokens;
    if history.len() < n || reference.len() <= n {
        return Vec::new();
    }

    let suffix = &history[history.len() - n..];
    let mut best: Option<(usize, usize)> = None; // (backward match, continuation start)

    for start in 0..=reference.len() - n {
        let continuation_start = start + n;
        if continuation_start >= reference.len() || &reference[start..continuation_start] != suffix
        {
            continue;
        }

        let mut backward_match = n;
        while backward_match < history.len()
            && backward_match < continuation_start
            && history[history.len() - backward_match - 1]
                == reference[continuation_start - backward_match - 1]
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
    let end = (start + draft_len).min(reference.len());
    reference[start..end].to_vec()
}
/// A draft is accepted only if the target's pick at its preceding position matches.
pub fn accepted_prefix(drafts: &[u32], picks: &[u32]) -> usize {
    drafts
        .iter()
        .zip(picks)
        .take_while(|(draft, pick)| draft == pick)
        .count()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn suffix_realigns_after_divergence() {
        assert_eq!(
            continuation(&[1, 2, 90, 4, 5, 6, 7], &[1, 2, 99, 4, 5], 2, 3),
            vec![6, 7]
        );
    }
    #[test]
    fn longest_backward_match_wins() {
        assert_eq!(
            continuation(&[7, 1, 2, 9, 4, 1, 2, 8], &[7, 1, 2], 2, 3),
            vec![9, 4, 1]
        );
    }
    #[test]
    fn rejects_at_first_mismatch() {
        assert_eq!(accepted_prefix(&[10, 11, 12], &[10, 99, 12, 13]), 1);
        assert_eq!(accepted_prefix(&[10, 11], &[10, 11, 12]), 2);
        assert_eq!(accepted_prefix(&[10, 11], &[9, 11, 12]), 0);
    }
    #[test]
    fn empty_or_short_history_has_no_draft() {
        assert!(continuation(&[1, 2, 3], &[1], 2, 8).is_empty());
        assert!(continuation(&[1, 2, 3], &[1, 2], 2, 0).is_empty());
    }
}
