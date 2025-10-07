#!/usr/bin/env python3
"""
Setup script for pie-metal package.

This script handles:
1. PyTorch MPS shader compilation at runtime (no build-time compilation needed)
2. Package installation with Metal kernel sources included
3. Graceful handling of non-macOS platforms
"""

import os
import sys
from pathlib import Path
from setuptools import setup


def check_torch_mps_availability():
    """Check if PyTorch MPS functionality is available."""
    if sys.platform != "darwin":
        print("ℹ️  pie-metal requires macOS with Apple Silicon for Metal acceleration")
        return False

    try:
        import torch
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print("✅ PyTorch MPS backend available")
                return True
            else:
                print("⚠️  PyTorch MPS backend not available on this system")
                return False
        else:
            print("⚠️  PyTorch MPS backend not found - please update PyTorch to 2.0+")
            return False
    except ImportError:
        print("⚠️  PyTorch not found - will be required for Metal acceleration")
        return False


def verify_metal_kernels():
    """Verify that Metal kernel source files are present."""
    kernel_dir = Path(__file__).parent / "src" / "pie_metal" / "_internal" / "metal" / "kernels"

    if not kernel_dir.exists():
        print("⚠️  Metal kernels directory not found")
        return False

    metal_files = list(kernel_dir.glob("*.metal"))
    if not metal_files:
        print("⚠️  No Metal kernel source files found")
        return False

    print(f"✅ Found {len(metal_files)} Metal kernel source files")
    print("   Kernels will be compiled at runtime using torch.mps.compile_shader")
    return True


if __name__ == "__main__":
    print("🔧 Setting up pie-metal package...")

    # Check system compatibility
    check_torch_mps_availability()
    verify_metal_kernels()

    print("ℹ️  pie-metal uses runtime Metal shader compilation via PyTorch MPS")
    print("   No build-time compilation required!")

    setup(
        zip_safe=False,  # Required for proper Metal kernel source file access
    )