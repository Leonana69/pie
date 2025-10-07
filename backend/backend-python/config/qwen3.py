"""
Qwen3 model configuration module.
"""

from dataclasses import dataclass

from .common import (
    CommonArch,
    ModelConfig,
)


@dataclass
class Qwen3Arch(CommonArch):
    """Qwen3 specific architecture configuration."""

    rope_theta: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "Qwen3Arch":
        """Parse Qwen3-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get RoPE configuration (Qwen3 uses simpler RoPE with only theta)
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_theta = cfg.get_required_key(rope_dict, "theta")

        return Qwen3Arch(
            # Common fields
            **common_arch_dict,
            # Qwen3-specific fields
            rope_theta=rope_theta,
        )
