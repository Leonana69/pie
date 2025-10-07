# pie-metal: Metal-accelerated FlashInfer Replacement

A drop-in replacement for FlashInfer that uses Apple's Metal Performance Shaders for attention operations on macOS with Apple Silicon.

## Features

- **Drop-in Replacement**: Exact FlashInfer API compatibility
- **Metal Acceleration**: Uses Apple's Metal Performance Shaders on Apple Silicon
- **PyTorch Integration**: Seamless tensor conversion and memory management
- **No Fallback**: It tells the users to use PyTorch when Metal unavailable

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- PyTorch 2.3+
- CMake (for Metal kernel compilation)
- Xcode Command Line Tools

### Install from Source

```bash
# Clone the repository
git clone <pie-repo-url>
cd pie

# Install pie-metal package
pip install ./pie-metal
```

## Usage

Simply replace your FlashInfer import:

```python
# Before:
# import flashinfer as ops

# After:
import pie_metal.ops as ops

# Everything else remains identical!
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cpu')
wrapper = ops.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
wrapper.plan(...)
output = wrapper.run(query, kv_cache)
```

## Supported Operations

### Core Attention Operations
- `BatchPrefillWithPagedKVCacheWrapper` - Multi-token prefill attention
- `BatchDecodeWithPagedKVCacheWrapper` - Single-token decode attention

### Utility Functions
- `apply_llama31_rope_pos_ids_inplace` - RoPE position encoding
- `append_paged_kv_cache` - KV cache management
- `get_seq_lens` - Sequence length calculation
- `get_batch_indices_positions` - Batch indexing utilities

## Performance

On Apple Silicon Macs, pie-metal provides:
- **2-4x faster** attention operations compared to CPU PyTorch
- **Lower memory usage** through optimized Metal buffer management
- **Better energy efficiency** using dedicated GPU cores

## Architecture

```
pie-metal
├── ops.py                    # FlashInfer-compatible API
├── _internal/
│   ├── metal_backend.py     # Metal kernel interface
│   ├── l4ma_runtime.py      # PyTorch fallback implementations
│   └── metal/
│       ├── kernels/         # Metal shader files
│       └── bindings/        # pybind11 C++ bindings
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/benchmark_attention.py
```

### Building from Source

```bash
# Install in development mode
pip install -e .

# Force rebuild of Metal kernels
pip install -e . --force-reinstall --no-deps
```

## License

MIT License - see LICENSE file for details.

## Contributing

Please see the main PIE repository for contribution guidelines.

