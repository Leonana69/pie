use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub struct Error(pub u16, pub String, pub Option<&'static str>);
pub type Result<T> = std::result::Result<T, Error>;
pub fn bad(message: impl Into<String>, param: &'static str) -> Error {
    Error(400, message.into(), Some(param))
}
pub fn internal(message: impl Into<String>) -> Error {
    Error(500, message.into(), None)
}
pub fn default_max() -> usize {
    256
}
pub fn default_draft() -> usize {
    8
}
pub fn default_match() -> usize {
    2
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub system: String,
    #[serde(default)]
    pub header: String,
    pub examples: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}
#[derive(Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}
impl Stop {
    pub fn earliest(&self, text: &str) -> Option<usize> {
        let values = match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v.iter().collect(),
        };
        values
            .into_iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| text.find(s.as_str()))
            .min()
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub cache_id: Option<String>,
    #[serde(default)]
    pub selected_examples: Option<Vec<usize>>,
    #[serde(default, alias = "enable_masking")]
    pub masking: bool,
    #[serde(default, alias = "enable_prior_plan")]
    pub prior_plan: bool,
    #[serde(default)]
    pub previous_plan: Option<String>,
    #[serde(default = "default_draft")]
    pub draft_len: usize,
    #[serde(default = "default_match")]
    pub match_tokens: usize,
    #[serde(default = "default_max")]
    pub max_tokens: usize,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub n: Option<usize>,
}
impl ChatRequest {
    pub fn max(&self) -> usize {
        self.max_completion_tokens.unwrap_or(self.max_tokens)
    }
    pub fn validate(&self) -> Result<()> {
        if self.messages.is_empty() || !self.messages.iter().any(|m| m.role == "user") {
            return Err(bad("at least one user message is required", "messages"));
        }
        if self.messages.iter().any(|m| {
            !matches!(
                m.role.as_str(),
                "system" | "developer" | "user" | "assistant"
            )
        }) {
            return Err(bad(
                "only text system/developer/user/assistant messages are supported",
                "messages",
            ));
        }
        if self.cache_id.is_some() && (self.messages.len() != 1 || self.messages[0].role != "user")
        {
            return Err(bad(
                "with cache_id, send exactly one user message containing dynamic information",
                "messages",
            ));
        }
        if self.cache_id.is_none() && (self.masking || self.selected_examples.is_some()) {
            return Err(bad(
                "example selection and masking require cache_id",
                "cache_id",
            ));
        }
        if self.n.unwrap_or(1) != 1 {
            return Err(bad("only n=1 is supported", "n"));
        }
        if !(1..=8192).contains(&self.max()) {
            return Err(bad("token limit must be 1..=8192", "max_tokens"));
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(bad("temperature must be 0..=2", "temperature"));
        }
        if self
            .top_p
            .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(bad("top_p must be in (0,1]", "top_p"));
        }
        if self.prior_plan && self.temperature != 0.0 {
            return Err(bad(
                "prior_plan requires greedy temperature=0",
                "temperature",
            ));
        }
        if self.prior_plan
            && self
                .previous_plan
                .as_ref()
                .is_none_or(|s| s.trim().is_empty())
        {
            return Err(bad(
                "prior_plan=true requires a nonempty previous_plan",
                "previous_plan",
            ));
        }
        if !(1..=32).contains(&self.draft_len) || !(1..=32).contains(&self.match_tokens) {
            return Err(bad(
                "draft_len and match_tokens must be 1..=32",
                "draft_len",
            ));
        }
        Ok(())
    }
}
pub fn selection(requested: Option<&[usize]>, count: usize) -> Result<Vec<usize>> {
    let mut selected = requested.map_or_else(|| (0..count).collect(), |v| v.to_vec());
    if selected.iter().any(|&i| i >= count) {
        return Err(bad(
            "selected example index is outside the catalog",
            "selected_examples",
        ));
    }
    selected.sort_unstable();
    selected.dedup();
    Ok(selected)
}
/// Length-delimited serialized content includes model, schema, system, header and catalog order.
pub fn cache_id(model: &str, req: &CacheRequest) -> String {
    let content = serde_json::to_vec(&(
        "plan-mask-cascade-v1",
        model,
        &req.system,
        &req.header,
        &req.examples,
    ))
    .unwrap();
    format!("pm-{:x}", Sha256::digest(content))
}
pub fn brle(runs: &[(bool, u32)]) -> Vec<u32> {
    let mut out = vec![0];
    let mut value = false;
    for &(v, n) in runs {
        if n == 0 {
            continue;
        }
        if v == value {
            *out.last_mut().unwrap() += n;
        } else {
            out.push(n);
            value = v;
        }
    }
    out
}
pub fn region(common: u32, slots: &[u32], selected: &[usize]) -> Vec<(bool, u32)> {
    let mut r = vec![(true, common)];
    r.extend(
        slots
            .iter()
            .enumerate()
            .map(|(i, &n)| (selected.contains(&i), n)),
    );
    r
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_distinguishes_none_and_empty() {
        assert_eq!(selection(None, 3).unwrap(), vec![0, 1, 2]);
        assert!(selection(Some(&[]), 3).unwrap().is_empty());
        assert_eq!(selection(Some(&[2, 0, 2]), 3).unwrap(), vec![0, 2]);
        assert!(selection(Some(&[3]), 3).is_err());
    }
    #[test]
    fn visibility_preserves_system_selected_slots_and_causal_tail() {
        let mut r = region(2, &[3, 2, 4], &[1]);
        r.push((true, 3));
        let mut value = false;
        let mut visibility = vec![];
        for n in brle(&r) {
            visibility.extend(std::iter::repeat_n(value, n as usize));
            value = !value;
        }
        assert_eq!(
            visibility,
            vec![
                true, true, false, false, false, true, true, false, false, false, false, true,
                true, true
            ]
        );
    }
    #[test]
    fn cache_identity_includes_content_order_and_model() {
        let mut req = CacheRequest {
            model: None,
            system: "s".into(),
            header: "h".into(),
            examples: vec!["a".into(), "b".into()],
        };
        let id = cache_id("model-a", &req);
        assert_ne!(id, cache_id("model-b", &req));
        req.examples.reverse();
        assert_ne!(id, cache_id("model-a", &req));
        req.examples.reverse();
        req.system.push('x');
        assert_ne!(id, cache_id("model-a", &req));
    }
    #[test]
    fn rejects_invalid_drafting_and_cached_message_shapes() {
        for extra in [
            r#", "prior_plan":true"#,
            r#", "cache_id":"x", "messages":[{"role":"system","content":"s"},{"role":"user","content":"q"}]"#,
        ] {
            let raw = format!(r#"{{"messages":[{{"role":"user","content":"q"}}]{extra}}}"#);
            let req = serde_json::from_str::<ChatRequest>(&raw);
            assert!(req.is_err() || req.unwrap().validate().is_err());
        }
        let req: ChatRequest=serde_json::from_str(r#"{"messages":[{"role":"user","content":"q"}],"prior_plan":true,"previous_plan":"p","temperature":0.5}"#).unwrap();
        assert!(req.validate().is_err());
    }
    #[test]
    fn stop_uses_earliest_text_boundary() {
        let stop = Stop::Many(vec!["end".into(), "停止".into()]);
        assert_eq!(stop.earliest("a停止 then end"), Some(1));
    }
}
