"""Shared model loading utilities for PIE backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Tuple

import torch
import ztensor
from tqdm import tqdm

from config.common import ModelInfo


CreateModelFn = Callable[[ModelInfo], Tuple[torch.nn.Module, dict]]


def load_model(
    config: dict, create_model_fn: CreateModelFn
) -> tuple[torch.nn.Module, ModelInfo]:
    """Load a model using the provided factory function and fusion metadata."""

    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config["device"]
    model_dtype = getattr(torch, config["dtype"])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    try:
        model, fusion_map = create_model_fn(model_info)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to instantiate model for architecture {model_info.architecture.type}: {exc}"
        ) from exc

    source_to_fusion_target = {
        source: target
        for target, details in fusion_map.items()
        for source in details["sources"]
    }

    pending_fusion_tensors = {}
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    try:
        for param_file in model_info.parameters:
            weights_path = model_path / model_name / param_file
            with ztensor.Reader(str(weights_path)) as reader:
                tensor_names = reader.get_tensor_names()
                pbar_desc = (
                    f"Loading {param_file[:30]}..."
                    if len(param_file) > 30
                    else f"Loading {param_file}"
                )
                for name in tqdm(tensor_names, desc=pbar_desc, unit="tensors"):
                    if name in source_to_fusion_target:
                        pending_fusion_tensors[name] = reader.read_tensor(
                            name, to="torch"
                        )
                        continue

                    if name in model_state_keys and name not in loaded_keys:
                        tensor_data = reader.read_tensor(name, to="torch")
                        param = model.state_dict()[name]

                        if tensor_data.shape != param.shape:
                            print(
                                f"    Warning: Shape mismatch for tensor '{name}'. Skipping."
                            )
                            continue
                        if tensor_data.dtype != param.dtype:
                            tensor_data = tensor_data.to(dtype=param.dtype)
                        if tensor_data.device != param.device:
                            tensor_data = tensor_data.to(device=param.device)

                        if name == "model.layers.0.input_layernorm.weight":
                            print("[ModelLoaderDebug] copying", name)
                            print(
                                "[ModelLoaderDebug] source",
                                name,
                                "dtype=",
                                tensor_data.dtype,
                                "device=",
                                tensor_data.device,
                                "min=",
                                float(tensor_data.min()),
                                "max=",
                                float(tensor_data.max()),
                            )
                            print(
                                "[ModelLoaderDebug] target before copy",
                                name,
                                "dtype=",
                                param.dtype,
                                "device=",
                                param.device,
                            )
                        with torch.no_grad():
                            param.copy_(tensor_data, non_blocking=True)
                        if name == "model.layers.0.input_layernorm.weight":
                            print(
                                "[ModelLoaderDebug] target after copy",
                                name,
                                "min=",
                                float(param.min()),
                                "max=",
                                float(param.max()),
                            )
                        loaded_keys.add(name)

        for target_name, details in fusion_map.items():
            source_names = details["sources"]
            if all(s in pending_fusion_tensors for s in source_names):
                tensors_to_fuse = [pending_fusion_tensors[s] for s in source_names]
                fused_tensor = torch.cat(tensors_to_fuse, dim=details["dim"])
                param = model.state_dict()[target_name]
                if fused_tensor.shape != param.shape:
                    print(
                        f"    Warning: Shape mismatch for fused tensor '{target_name}'. Skipping."
                    )
                    continue
                with torch.no_grad():
                    param.copy_(fused_tensor, non_blocking=True)
                loaded_keys.add(target_name)

        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            model.state_dict()["lm_head.weight"].copy_(
                model.model.embed_tokens.weight, non_blocking=True
            )
            loaded_keys.add("lm_head.weight")

        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        model.eval()
        return model, model_info

    except ztensor.ZTensorError as exc:
        print(
            f"Fatal Error: Failed to read a ztensor file. Error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(
            f"An unexpected fatal error occurred during model loading: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


__all__ = ["load_model"]
