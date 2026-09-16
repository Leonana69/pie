#include "gguf_hparams.hpp"
#include "gguf_archive.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

using pie_portable_driver::GgufMeta;
using pie_portable_driver::parse_gguf_hparams;

namespace {

void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

void set_num(GgufMeta& meta, const std::string& key, double value) {
    GgufMeta::KV kv;
    kv.key = key;
    kv.num_value = value;
    meta.kv.emplace(key, std::move(kv));
}

void set_i32_array(GgufMeta& meta, const std::string& key,
                   std::vector<std::int32_t> value) {
    GgufMeta::KV kv;
    kv.key = key;
    kv.i32_array_value = std::move(value);
    meta.kv.emplace(key, std::move(kv));
}

GgufMeta qwen35_meta(std::int32_t block_count,
                     std::int32_t nextn_predict_layers) {
    GgufMeta meta;
    meta.general_architecture = "qwen35";
    set_num(meta, "qwen35.block_count", block_count);
    set_num(meta, "qwen35.nextn_predict_layers", nextn_predict_layers);
    set_num(meta, "qwen35.full_attention_interval", 4);
    set_num(meta, "qwen35.ssm.group_count", 16);
    set_num(meta, "qwen35.ssm.time_step_rank", 48);
    set_num(meta, "qwen35.ssm.state_size", 128);
    set_num(meta, "qwen35.ssm.conv_kernel", 4);
    set_num(meta, "qwen35.attention.head_count", 24);
    set_num(meta, "qwen35.embedding_length", 5120);
    set_num(meta, "qwen35.attention.key_length", 256);
    set_num(meta, "qwen35.rope.dimension_count", 64);
    set_i32_array(meta, "qwen35.rope.dimension_sections", {11, 11, 10, 0});
    return meta;
}

void test_nextn_layers_are_not_primary_transformer_layers() {
    const auto h = parse_gguf_hparams(qwen35_meta(65, 1));
    require(h.num_hidden_layers == 64,
            "Qwen3.8 main transformer should exclude its MTP layer");
    require(h.layer_types.size() == 64,
            "layer_types should describe only main transformer layers");
    require(h.layer_types[62] == 'l', "layer 62 should be linear attention");
    require(h.layer_types[63] == 'g', "layer 63 should be full attention");
    require(h.qwen35_mrope_interleaved,
            "Qwen3.5-family GGUF should use interleaved mRoPE");
    require(h.qwen35_mrope_section[0] == 11 &&
            h.qwen35_mrope_section[1] == 11 &&
            h.qwen35_mrope_section[2] == 10,
            "mRoPE sections should come from GGUF metadata");
    require(h.qwen35_partial_rotary_factor == 0.25f,
            "rope.dimension_count should set the partial rotary factor");
}

void test_zero_nextn_preserves_block_count() {
    const auto h = parse_gguf_hparams(qwen35_meta(64, 0));
    require(h.num_hidden_layers == 64,
            "ordinary Qwen3.5 block count should remain unchanged");
}

void test_invalid_nextn_is_rejected() {
    bool threw = false;
    try {
        (void)parse_gguf_hparams(qwen35_meta(1, 1));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "nextn_predict_layers must be smaller than block_count");
}

}  // namespace

int main() {
    try {
        test_nextn_layers_are_not_primary_transformer_layers();
        test_zero_nextn_preserves_block_count();
        test_invalid_nextn_is_rejected();
        std::puts("portable gguf_hparams ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "portable gguf_hparams failed: %s\n", e.what());
        return 1;
    }
}
