"""
Real PyTorch MPS integration using torch.mps.compile_shader.

This module compiles and executes our existing Metal attention kernels
directly through PyTorch's MPS backend, enabling true zero-copy operations.
"""

import torch
from typing import Optional
from .mps_attention import AttentionCompiler
from .mps_rope import RoPECompiler
from .mps_append_kv_cache import AppendKVCacheCompiler


class MPSShaderCompiler:
    """Unified facade for all MPS Metal kernel operations."""

    def __init__(self):
        print("Initializing MPS Shader Compiler...")
        self.attention_compiler = AttentionCompiler()
        self.rope_compiler = RoPECompiler()
        self.append_kv_cache_compiler = AppendKVCacheCompiler()
        print("✅ Metal shaders compiled successfully")

    @property
    def compiled_libraries(self):
        """Unified view of all compiled libraries from all compilers."""
        libraries = {}
        libraries.update(self.attention_compiler.compiled_libraries)
        libraries.update(self.rope_compiler.compiled_libraries)
        libraries.update(self.append_kv_cache_compiler.compiled_libraries)
        return libraries

    def can_use_mps_kernels(self) -> bool:
        """Check if we can use compiled MPS kernels."""
        return (self.attention_compiler.can_use_mps_kernels() or
                self.rope_compiler.can_use_mps_kernels() or
                self.append_kv_cache_compiler.can_use_mps_kernels())

    def run_attention_mps(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        custom_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run attention using compiled MPS kernels."""
        return self.attention_compiler.run_attention_mps(
            query, kv_cache, kv_page_indices, kv_page_indptr,
            kv_last_page_lens, qo_indptr, custom_mask
        )

    def run_rope_mps(
        self,
        input_qk: torch.Tensor,
        position_ids: torch.Tensor,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        interleaved: bool = False
    ) -> None:
        """Run RoPE using compiled MPS kernels."""
        self.rope_compiler.run_rope_mps(
            input_qk, position_ids, rope_theta, rope_factor, interleaved
        )

    def run_append_paged_kv_cache_mps(
        self,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_batch_indices: torch.Tensor,
        kv_positions: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        page_size: int
    ) -> None:
        """Run append_paged_kv_cache using compiled MPS kernels with unified buffer."""
        self.append_kv_cache_compiler.run_append_paged_kv_cache_mps(
            k_input, v_input, paged_kv_cache,
            kv_batch_indices, kv_positions, kv_page_indices,
            kv_page_indptr, kv_last_page_lens,
            num_kv_heads, head_size, page_size
        )


# Global compiler instance
_shader_compiler: Optional[MPSShaderCompiler] = None


def get_mps_compiler() -> MPSShaderCompiler:
    """Get or create the global MPS shader compiler."""
    global _shader_compiler
    if _shader_compiler is None:
        _shader_compiler = MPSShaderCompiler()
    return _shader_compiler


def run_mps_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    qo_indptr: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Run attention using PyTorch MPS compiled shaders."""

    compiler = get_mps_compiler()

    if compiler.can_use_mps_kernels():
        return compiler.run_attention_mps(
            query, kv_cache, kv_page_indices, kv_page_indptr,
            kv_last_page_lens, qo_indptr, custom_mask
        )
    else:
        raise RuntimeError(
            "MPS shader compilation not available. "
            "Please ensure you have PyTorch 2.0+ with MPS support."
        )