#include "gguf_hparams.hpp"
#include "gguf_archive.hpp"

#include <cmath>
#include <stdexcept>

namespace pie_portable_driver {

namespace {

// Map llama.cpp's `general.architecture` strings onto our PieArch enum.
// llama.cpp uses an arch-internal name (e.g. "qwen3", "llama"). For the
// hybrid Qwen 3.5 / 3.6 family the name is "qwen35" (per llama.cpp PR
// #19468) — we accept both that and a few seen-in-the-wild variants.
PieArch arch_from_gguf(const std::string& s) {
    if (s == "qwen3")              return PieArch::Qwen3;
    if (s == "qwen2")              return PieArch::Qwen2;
    if (s == "qwen3moe")           return PieArch::Qwen3Moe;
    if (s == "qwen35"
     || s == "qwen3_5"
     || s == "qwen3_5_text"
     || s == "qwen35moe"
     || s == "qwen3_5_moe"
     || s == "qwen3_5_moe_text")   return PieArch::Qwen3_5;
    if (s == "llama")              return PieArch::Llama3;
    if (s == "mistral")            return PieArch::Mistral3;
    if (s == "mixtral")            return PieArch::Mixtral;
    if (s == "phi3")               return PieArch::Phi3;
    if (s == "gemma2")             return PieArch::Gemma2;
    if (s == "gemma3")             return PieArch::Gemma3;
    if (s == "gemma4")             return PieArch::Gemma4;
    if (s == "olmo3" || s == "olmo2") return PieArch::Olmo3;
    if (s == "gpt-oss" || s == "gptoss") return PieArch::GptOss;
    if (s == "glm_moe_dsa")        return PieArch::GlmMoeDsa;
    throw std::runtime_error(
        "gguf: unsupported general.architecture '" + s + "'");
}

// Try `<arch>.<suffix>` first, then `<suffix>` (some converters strip
// the prefix on a few keys). Returns 0 / nullptr when missing.
const GgufMeta::KV* find_kv(const GgufMeta& m,
                             const std::string& arch,
                             const char* suffix) {
    const std::string a = arch + "." + suffix;
    if (auto it = m.kv.find(a); it != m.kv.end()) return &it->second;
    if (auto it = m.kv.find(suffix); it != m.kv.end()) return &it->second;
    return nullptr;
}

double get_num(const GgufMeta& m, const std::string& arch, const char* key,
               double dflt) {
    const auto* kv = find_kv(m, arch, key);
    return kv ? kv->num_value : dflt;
}

bool get_bool(const GgufMeta& m, const std::string& arch, const char* key,
              bool dflt) {
    const auto* kv = find_kv(m, arch, key);
    return kv ? kv->bool_value : dflt;
}

const std::vector<std::int32_t>* get_i32_array(
        const GgufMeta& m, const std::string& arch, const char* key) {
    const auto* kv = find_kv(m, arch, key);
    if (kv == nullptr || kv->i32_array_value.empty()) return nullptr;
    return &kv->i32_array_value;
}

}  // namespace

Hparams parse_gguf_hparams(const GgufMeta& meta) {
    Hparams h{};

    // ---- Identity / arch dispatch --------------------------------------
    h.hf_model_type = meta.general_architecture;
    h.arch          = arch_from_gguf(meta.general_architecture);
    const std::string a = meta.general_architecture;

    // ---- Common transformer hparams ------------------------------------
    h.num_hidden_layers   = static_cast<std::int32_t>(get_num(meta, a, "block_count",        0));
    h.num_attention_heads = static_cast<std::int32_t>(get_num(meta, a, "attention.head_count", 0));
    h.num_key_value_heads = static_cast<std::int32_t>(get_num(meta, a, "attention.head_count_kv",
                                                              h.num_attention_heads));
    h.hidden_size         = static_cast<std::int32_t>(get_num(meta, a, "embedding_length",   0));
    h.intermediate_size   = static_cast<std::int32_t>(get_num(meta, a, "feed_forward_length", 0));
    h.max_position_embeddings = static_cast<std::int32_t>(
        get_num(meta, a, "context_length", 4096));

    // GGUF carries head_dim split as key/value lengths; HF combines into
    // a single head_dim. Default to hidden / num_heads when missing.
    const int kl = static_cast<int>(get_num(meta, a, "attention.key_length", 0));
    if (kl > 0) {
        h.head_dim = kl;
    } else if (h.num_attention_heads > 0) {
        h.head_dim = h.hidden_size / h.num_attention_heads;
    }

    h.rms_norm_eps = static_cast<float>(
        get_num(meta, a, "attention.layer_norm_rms_epsilon", 1e-6));
    h.rope_theta = static_cast<float>(get_num(meta, a, "rope.freq_base", 1e6f));

    // tie_word_embeddings: GGUF doesn't store this flag explicitly
    // (llama.cpp probes for the `output.weight` tensor). We use the
    // archive's pre-computed `has_output_weight`.
    h.tie_word_embeddings = !meta.has_output_weight;

    // Vocab size: GGUF doesn't always store it explicitly either. The
    // tokenizer.ggml.tokens array length is the canonical source, and
    // most converters DO write `<arch>.vocab_size`. Take the explicit
    // arch key when present, otherwise fall back to the tokens-array
    // length captured at archive construction (Qwen3-MoE GGUFs in the
    // wild omit the arch key).
    h.vocab_size = static_cast<std::int32_t>(get_num(meta, a, "vocab_size", 0));
    if (h.vocab_size == 0) {
        h.vocab_size = static_cast<std::int32_t>(meta.tokens_count);
    }

    // ---- MoE -----------------------------------------------------------
    h.num_experts         = static_cast<std::int32_t>(get_num(meta, a, "expert_count",        0));
    h.num_experts_per_tok = static_cast<std::int32_t>(get_num(meta, a, "expert_used_count",   0));
    h.moe_intermediate_size = static_cast<std::int32_t>(
        get_num(meta, a, "expert_feed_forward_length", h.intermediate_size));
    // norm_topk_prob is on by default for Qwen-MoE / Mixtral.
    h.norm_topk_prob = get_bool(meta, a, "expert_norm_topk_prob", true);

    // ---- Sliding window (gemma2 / gemma3 / older mistrals) -------------
    if (auto v = static_cast<int>(get_num(meta, a, "attention.sliding_window", 0)); v > 0) {
        h.sliding_window = v;
    }

    // ---- Qwen 3.5 / 3.6 hybrid (gated-delta-rule linear attn + GQA) -----
    // The safetensors path fills these from config.json (hf_config.cpp);
    // for a GGUF the same facts live under the `qwen35.*` kv namespace.
    // Without them `model.cpp::build_qwen3_5_` throws "layer_types missing".
    if (h.arch == PieArch::Qwen3_5) {
        // Recent Qwen3.5-family GGUFs can append one or more NextN/MTP
        // predictor blocks and include them in `<arch>.block_count`. They are
        // auxiliary draft-model layers rather than layers in the primary
        // transformer. Pie does not execute those predictors through the
        // portable path, so exclude them from the main model depth; otherwise
        // the regular layer loop tries to load a linear-attention block from
        // an MTP block (for example Qwen3.8-27B's blk.64).
        const std::int32_t nextn_predict_layers = static_cast<std::int32_t>(
            get_num(meta, a, "nextn_predict_layers", 0));
        if (nextn_predict_layers < 0 ||
            nextn_predict_layers >= h.num_hidden_layers) {
            throw std::runtime_error(
                "gguf: invalid nextn_predict_layers " +
                std::to_string(nextn_predict_layers) + " for block_count " +
                std::to_string(h.num_hidden_layers));
        }
        h.num_hidden_layers -= nextn_predict_layers;

        // Linear-attention dims. The GGUF `ssm.*` keys carry the same
        // quantities the HF config exposes as `linear_*`:
        //   group_count    -> num K heads        (ssm_n_group)
        //   time_step_rank -> num V heads         (ssm_dt_rank)
        //   state_size     -> K and V head dim    (ssm_d_state)
        //   conv_kernel    -> depthwise conv width
        // Verified against real blobs by the identity
        //   ssm.inner_size == num_v_heads * v_head_dim
        // (0.8B: 16*128=2048; 27B: 48*128=6144).
        h.qwen35_linear_num_k_heads =
            static_cast<std::int32_t>(get_num(meta, a, "ssm.group_count", 0));
        h.qwen35_linear_num_v_heads =
            static_cast<std::int32_t>(get_num(meta, a, "ssm.time_step_rank", 0));
        h.qwen35_linear_k_head_dim =
            static_cast<std::int32_t>(get_num(meta, a, "ssm.state_size", 0));
        h.qwen35_linear_v_head_dim = h.qwen35_linear_k_head_dim;
        h.qwen35_linear_conv_kernel =
            static_cast<std::int32_t>(get_num(meta, a, "ssm.conv_kernel", 4));
        // A GGUF always carries the converter's tiled V-head order; the
        // graph must expand Q/K with a tiled broadcast to match.
        h.qwen35_linear_v_tiled = true;

        // Qwen3.5-family GGUF uses interleaved multimodal RoPE. The three
        // section lengths are stored as an INT32 array and dimension_count
        // records the actually rotated prefix of each attention head.
        const auto* mrope_sections =
            get_i32_array(meta, a, "rope.dimension_sections");
        if (mrope_sections == nullptr || mrope_sections->size() < 3) {
            throw std::runtime_error(
                "gguf: qwen3_5 missing rope.dimension_sections");
        }
        for (std::size_t i = 0; i < 3; ++i) {
            h.qwen35_mrope_section[i] = (*mrope_sections)[i];
        }
        h.qwen35_mrope_interleaved = true;
        const auto rope_dimension_count = static_cast<std::int32_t>(
            get_num(meta, a, "rope.dimension_count", h.head_dim));
        if (rope_dimension_count <= 0 || rope_dimension_count > h.head_dim) {
            throw std::runtime_error(
                "gguf: invalid qwen3_5 rope.dimension_count " +
                std::to_string(rope_dimension_count) + " for head_dim " +
                std::to_string(h.head_dim));
        }
        h.qwen35_partial_rotary_factor =
            static_cast<float>(rope_dimension_count) /
            static_cast<float>(h.head_dim);

        // qwen35moe shared expert (always-active dense path alongside the
        // routed experts). The generic MoE block above already read
        // expert_count / expert_used_count / expert_feed_forward_length.
        h.shared_expert_intermediate_size = static_cast<std::int32_t>(
            get_num(meta, a, "expert_shared_feed_forward_length", 0));

        // Per-layer attention type. The GGUF stores only the stride
        // `full_attention_interval`; llama.cpp derives the pattern as
        // "layer i is full attention iff (i+1) % interval == 0, else
        // linear" (every interval-th layer, last of each group). pie
        // encodes full='g', linear='l' in `layer_types`.
        const std::int32_t interval = static_cast<std::int32_t>(
            get_num(meta, a, "full_attention_interval", 4));
        h.qwen35_full_attn_interval = interval;
        if (interval > 0 && h.num_hidden_layers > 0) {
            h.layer_types.assign(static_cast<std::size_t>(h.num_hidden_layers), 'l');
            for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
                if (((i + 1) % interval) == 0) {
                    h.layer_types[static_cast<std::size_t>(i)] = 'g';
                }
            }
        }

        // Qwen 3.5 / 3.6 full-attention layers always carry an output gate
        // FUSED into a double-wide `q_proj` (out = 2 * n_heads * head_dim:
        // half query, half gate), so there is no separate gate tensor in
        // the GGUF. `graph_qwen3_5.cpp` splits it when this flag is set;
        // leaving it false reshapes the double-wide Q to the single-width
        // head layout and aborts (ggml_nelements mismatch).
        h.qwen35_attn_output_gate = true;
    }

    return h;
}

}  // namespace pie_portable_driver
