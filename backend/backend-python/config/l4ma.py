"""
L4MA/Llama model configuration module.
"""

from dataclasses import dataclass

from .common import (
    CommonArch,
    ModelConfig,
)


@dataclass
class L4maArch(CommonArch):
    """L4MA/Llama specific architecture configuration."""

    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_theta: float

    @staticmethod
    def from_config(cfg: ModelConfig) -> "L4maArch":
        """Parse L4MA/Llama-specific architecture configuration."""
        # Get common architecture fields
        common_arch_dict = cfg.get_common_arch_dict()

        # Get all the fields for the architecture section to grab other
        # architecture-specific fields
        arch_dict = cfg.get_required_key(cfg.root, "architecture")

        # Get RoPE configuration (Llama-style with factor-based scaling)
        rope_dict = cfg.get_required_key(arch_dict, "rope")
        rope_factor = cfg.get_required_key(rope_dict, "factor")
        rope_high_frequency_factor = cfg.get_required_key(
            rope_dict, "high_frequency_factor"
        )
        rope_low_frequency_factor = cfg.get_required_key(
            rope_dict, "low_frequency_factor"
        )
        rope_theta = cfg.get_required_key(rope_dict, "theta")

        return L4maArch(
            # Common fields
            **common_arch_dict,
            # L4MA-specific fields
            rope_factor=rope_factor,
            rope_high_frequency_factor=rope_high_frequency_factor,
            rope_low_frequency_factor=rope_low_frequency_factor,
            rope_theta=rope_theta,
        )
