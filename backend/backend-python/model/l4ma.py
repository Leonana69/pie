"""Llama-Like Large Language Model Architecture (L4MA).

This module now focuses on the architecture itself and relies on an injected
runtime backend for kernel-specific behaviour (e.g. FlashInfer or Metal).
"""

from __future__ import annotations
from typing import Sequence, Optional

import torch
from torch import nn

# Safe import of adapter functionality
from adapter_utils import AdapterSubpass
from config.l4ma import L4maArch
from model.l4ma_runtime import L4maBackend, L4maForwardContext, RuntimeInputs
from profiler import start_profile

VERSION = "0.1.0"


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights.

    Returns:
        A dictionary mapping {fused_tensor_name: {"sources": [source_names], "dim": cat_dim}}.
    """

    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for L4maAttention QKV Fusion ---
        if isinstance(module, L4maAttention):
            # Handle weights
            target_w = f"{name}.qkv_proj.weight"
            sources_w = [
                f"{name}.q_proj.weight",
                f"{name}.k_proj.weight",
                f"{name}.v_proj.weight",
            ]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

            # Handle biases if they exist
            if module.qkv_proj.bias is not None:
                target_b = f"{name}.qkv_proj.bias"
                sources_b = [
                    f"{name}.q_proj.bias",
                    f"{name}.k_proj.bias",
                    f"{name}.v_proj.bias",
                ]
                fusion_map[target_b] = {"sources": sources_b, "dim": 0, "op": "fusion"}

        # --- Rule for L4maMlp Gate/Up Fusion ---
        elif isinstance(module, L4maMlp):
            target_w = f"{name}.gate_up_proj.weight"
            sources_w = [f"{name}.gate_proj.weight", f"{name}.up_proj.weight"]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0, "op": "fusion"}

    return fusion_map


class L4maMlp(nn.Module):
    """Feed-forward network block used in each decoder layer."""

    def __init__(self, config: L4maArch):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,  # Double the output dimension
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """Forward pass through the MLP layer."""
        gate_up_proj_out = self._gate_up_projection(x)
        gate_proj, up_proj = gate_up_proj_out.chunk(2, dim=-1)
        interim = self._silu_activation(gate_proj, up_proj)
        down_proj = self._down_projection(interim)
        return down_proj

    def _gate_up_projection(self, x):
        """Gate/Up projection for Metal GEMM kernel comparison."""
        return self.gate_up_proj(x)

    def _silu_activation(self, gate_proj, up_proj):
        """SiLU activation for Metal activation kernel comparison."""
        return self.act_fn(gate_proj) * up_proj

    def _down_projection(self, interim):
        """Down projection for Metal GEMM kernel comparison."""
        return self.down_proj(interim)


class L4maAttention(nn.Module):
    """Multi-head attention block for the decoder."""

    def __init__(self, config: L4maArch, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + self.k_size + self.v_size,
            bias=config.use_qkv_bias,
            device=config.device,
            dtype=config.dtype,
        )

        self.o_proj = nn.Linear(
            config.num_query_heads * config.head_size,
            config.hidden_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        runtime: L4maForwardContext,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: Sequence[torch.Tensor],
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Attention forward pass that delegates runtime specifics."""

        n, _ = hidden_states.size()
        qkv_states = self.qkv_proj(hidden_states)

        query_states, key_states, value_states = torch.split(
            qkv_states, [self.q_size, self.k_size, self.v_size], dim=-1
        )

        # apply adapters if provided
        if adapter_subpass is not None:
            adapter_subpass.execute(
                self.layer_idx,
                hidden_states,
                q_state=query_states,
                k_state=key_states,
                v_state=value_states,
            )

        # Reshape for multi-head attention
        query_states = query_states.view(
            n, self.config.num_query_heads, self.config.head_size
        )
        key_states = key_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )
        value_states = value_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )

        runtime.apply_rope(query_states, key_states, position_ids)

        if query_states.dtype != self.config.dtype:
            query_states = query_states.to(self.config.dtype)

        runtime.append_kv_cache(
            layer_idx=self.layer_idx,
            key_states=key_states,
            value_states=value_states,
            kv_cache_layer=kv_cache_at_layer[self.layer_idx],
        )

        attn_output = runtime.run_attention(
            layer_idx=self.layer_idx,
            query_states=query_states,
            kv_cache_layer=kv_cache_at_layer[self.layer_idx],
        )

        attn_output = self.o_proj(attn_output)

        return attn_output


class L4maDecoderLayer(nn.Module):
    """Single decoder layer consisting of attention + MLP."""

    def __init__(self, config: L4maArch, layer_idx: int):
        super().__init__()

        self.self_attn = L4maAttention(config, layer_idx)

        self.mlp = L4maMlp(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        runtime: L4maForwardContext,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: Sequence[torch.Tensor],
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Run the decoder layer using the provided runtime context."""

        with start_profile("input_norm"):
            residual = hidden_states
            hidden_states = self._input_normalization(hidden_states)

        with start_profile("attention"):
            hidden_states = self.self_attn(
                runtime=runtime,
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                adapter_subpass=adapter_subpass,
            )

        with start_profile("attention_residual"):
            hidden_states = residual + hidden_states

        with start_profile("post_attn_norm"):
            residual = hidden_states
            hidden_states = self._post_attention_normalization(hidden_states)

        with start_profile("mlp"):
            hidden_states = self.mlp(hidden_states)

        with start_profile("mlp_residual"):
            hidden_states = residual + hidden_states

        return hidden_states

    def _input_normalization(self, hidden_states):
        """Input RMSNorm for Metal normalization kernel comparison."""
        return self.input_layernorm(hidden_states)

    def _post_attention_normalization(self, hidden_states):
        """Post-attention RMSNorm for Metal normalization kernel comparison."""
        return self.post_attention_layernorm(hidden_states)


class L4maModel(nn.Module):
    """Backbone model for the L4MA architecture."""

    def __init__(self, config: L4maArch, backend: L4maBackend):
        super().__init__()
        self.config = config
        self.backend = backend
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
            device=config.device,
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            [
                L4maDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        # input
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: Sequence[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Forward pass through all decoder layers using the injected backend."""

        with start_profile("model_setup"):
            hidden_states = input_embeds
            n, _ = hidden_states.size()

            runtime_inputs = RuntimeInputs(
                num_tokens=n,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                qo_indptr=qo_indptr,
                custom_mask=custom_mask,
                single_token_inference_mode=single_token_inference_mode,
            )

            runtime = self.backend.create_forward_context(
                config=self.config,
                inputs=runtime_inputs,
            )

        with start_profile("decoder_layers"):
            for layer_idx, decoder_layer in enumerate(self.layers):
                with start_profile(f"layer_{layer_idx}"):
                    hidden_states = decoder_layer(
                        runtime=runtime,
                        hidden_states=hidden_states,
                        position_ids=position_ids,
                        kv_cache_at_layer=kv_cache_at_layer,
                        adapter_subpass=adapter_subpass,
                    )

        with start_profile("final_norm"):
            hidden_states = self.norm(hidden_states)

        return hidden_states


class L4maForCausalLM(nn.Module):
    """Top-level causal language model wrapper for L4MA architecture."""

    def __init__(self, config: L4maArch, backend: L4maBackend):
        super().__init__()
        self.config = config
        self.model = L4maModel(config, backend)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self):  # pragma: no cover - interface parity placeholder
        """The handler uses dedicated methods rather than Module.forward."""
        raise NotImplementedError("Should not be called")


__all__ = [
    "L4maForCausalLM",
    "L4maModel",
    "L4maDecoderLayer",
    "L4maAttention",
    "L4maMlp",
    "create_fusion_map",
    "VERSION",
]
