"""Shared interfaces for L4MA runtime backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from config.l4ma import L4maArch


@dataclass(frozen=True)
class RuntimeInputs:
    """Inputs required by a backend to execute a single forward pass."""

    num_tokens: int
    kv_cache_at_layer: Sequence[torch.Tensor]
    kv_page_indices: torch.Tensor
    kv_page_indptr: torch.Tensor
    kv_last_page_lens: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: Optional[torch.Tensor]
    single_token_inference_mode: bool


class L4maForwardContext(ABC):
    """Backend-specific context used for each decoder layer iteration."""

    @property
    @abstractmethod
    def batch_indices(self) -> torch.Tensor:
        """Indices describing which request each token belongs to."""

    @property
    @abstractmethod
    def batch_positions(self) -> torch.Tensor:
        """Positions of tokens within their corresponding sequences."""

    @abstractmethod
    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Apply rotary position embeddings in place."""

    @abstractmethod
    def append_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> None:
        """Append key/value tensors into the backend's paged KV cache."""

    @abstractmethod
    def run_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the attention kernel for the given layer."""


class L4maBackend(ABC):
    """Factory interface that produces backend-specific forward contexts."""

    @abstractmethod
    def create_forward_context(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
    ) -> L4maForwardContext:
        """Create a context object to drive a single forward pass."""


__all__ = ["L4maBackend", "L4maForwardContext", "RuntimeInputs"]
