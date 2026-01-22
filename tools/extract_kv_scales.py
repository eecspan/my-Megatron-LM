#!/usr/bin/env python3

"""
python tools/extract_kv_scales.py \
  --model-dir /path/to/hf-model \
  --output /path/to/kv_scales.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch

try:
    from safetensors.torch import safe_open
except ImportError:
    safe_open = None


LAYER_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.self_attn\\.(?:attn\\.)?(k_scale|v_scale|kv_scale)$")


def _get_scalar(tensor: torch.Tensor) -> float:
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar tensor, got shape {tuple(tensor.shape)}")
    return float(tensor.reshape(-1)[0].item())


def _assign_scale(
    k_scales: List[float | None],
    v_scales: List[float | None],
    layer_idx: int,
    kind: str,
    value: float,
) -> None:
    needed = layer_idx + 1
    if len(k_scales) < needed:
        k_scales.extend([None] * (needed - len(k_scales)))
    if len(v_scales) < needed:
        v_scales.extend([None] * (needed - len(v_scales)))
    if kind == "kv_scale":
        k_scales[layer_idx] = value
        v_scales[layer_idx] = value
    elif kind == "k_scale":
        k_scales[layer_idx] = value
    elif kind == "v_scale":
        v_scales[layer_idx] = value


def _extract_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[List[float | None], List[float | None]]:
    k_scales: List[float | None] = []
    v_scales: List[float | None] = []
    for key, tensor in state_dict.items():
        match = LAYER_RE.search(key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        kind = match.group(2)
        value = _get_scalar(tensor)
        _assign_scale(k_scales, v_scales, layer_idx, kind, value)
    return k_scales, v_scales


def _normalize_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise ValueError("Unsupported checkpoint format; expected a dict-like state dict.")


def _load_index_weight_map(index_path: str) -> Dict[str, str]:
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weight_map", {})


def _extract_from_torch_bin(path: str) -> Tuple[List[float | None], List[float | None]]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    state_dict = _normalize_state_dict(state)
    return _extract_from_state_dict(state_dict)


def _extract_from_safetensors(path: str) -> Tuple[List[float | None], List[float | None]]:
    if safe_open is None:
        raise RuntimeError("safetensors is not installed; cannot read .safetensors files.")
    k_scales: List[float | None] = []
    v_scales: List[float | None] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            match = LAYER_RE.search(key)
            if not match:
                continue
            layer_idx = int(match.group(1))
            kind = match.group(2)
            value = _get_scalar(f.get_tensor(key))
            _assign_scale(k_scales, v_scales, layer_idx, kind, value)
    return k_scales, v_scales


def _merge_scales(
    target: Tuple[List[float | None], List[float | None]],
    incoming: Tuple[List[float | None], List[float | None]],
) -> None:
    k_scales, v_scales = target
    in_k, in_v = incoming
    max_len = max(len(k_scales), len(in_k), len(v_scales), len(in_v))
    if len(k_scales) < max_len:
        k_scales.extend([None] * (max_len - len(k_scales)))
    if len(v_scales) < max_len:
        v_scales.extend([None] * (max_len - len(v_scales)))
    for i in range(len(in_k)):
        if in_k[i] is not None:
            k_scales[i] = in_k[i]
    for i in range(len(in_v)):
        if in_v[i] is not None:
            v_scales[i] = in_v[i]


def _finalize_scales(
    k_scales: List[float | None], v_scales: List[float | None], strict: bool
) -> Tuple[List[float], List[float]]:
    max_len = max(len(k_scales), len(v_scales))
    if len(k_scales) < max_len:
        k_scales.extend([None] * (max_len - len(k_scales)))
    if len(v_scales) < max_len:
        v_scales.extend([None] * (max_len - len(v_scales)))

    missing = []
    for i in range(max_len):
        if k_scales[i] is None or v_scales[i] is None:
            missing.append(i)
            if not strict:
                k_scales[i] = k_scales[i] if k_scales[i] is not None else 1.0
                v_scales[i] = v_scales[i] if v_scales[i] is not None else 1.0

    if missing and strict:
        raise ValueError(f"Missing k/v scales for layers: {missing}")

    return [float(x) for x in k_scales], [float(x) for x in v_scales]


def _find_safetensors_files(model_dir: str) -> List[str]:
    files = []
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            files.append(os.path.join(model_dir, name))
    return sorted(files)


def _find_torch_bin_files(model_dir: str) -> List[str]:
    files = []
    for name in os.listdir(model_dir):
        if name.endswith(".bin"):
            files.append(os.path.join(model_dir, name))
    return sorted(files)


def _extract_scales(model_dir: str, strict: bool) -> Tuple[List[float], List[float]]:
    merged = ([], [])
    index_safetensors = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_safetensors):
        weight_map = _load_index_weight_map(index_safetensors)
        file_to_keys = defaultdict(list)
        for key, filename in weight_map.items():
            if LAYER_RE.search(key):
                file_to_keys[filename].append(key)
        for filename in sorted(file_to_keys.keys()):
            path = os.path.join(model_dir, filename)
            _merge_scales(merged, _extract_from_safetensors(path))
        if merged[0] or merged[1]:
            return _finalize_scales(*merged, strict=strict)

    safetensors_files = _find_safetensors_files(model_dir)
    if safetensors_files:
        for path in safetensors_files:
            _merge_scales(merged, _extract_from_safetensors(path))
        if merged[0] or merged[1]:
            return _finalize_scales(*merged, strict=strict)

    index_bin = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if os.path.exists(index_bin):
        weight_map = _load_index_weight_map(index_bin)
        file_to_keys = defaultdict(list)
        for key, filename in weight_map.items():
            if LAYER_RE.search(key):
                file_to_keys[filename].append(key)
        for filename in sorted(file_to_keys.keys()):
            path = os.path.join(model_dir, filename)
            _merge_scales(merged, _extract_from_torch_bin(path))
        if merged[0] or merged[1]:
            return _finalize_scales(*merged, strict=strict)

    bin_files = _find_torch_bin_files(model_dir)
    if bin_files:
        for path in bin_files:
            _merge_scales(merged, _extract_from_torch_bin(path))
        if merged[0] or merged[1]:
            return _finalize_scales(*merged, strict=strict)

    raise FileNotFoundError("No checkpoint files found or no k/v scales present.")


def _write_output(
    output_path: str, k_scales: List[float], v_scales: List[float], fmt: str
) -> None:
    if fmt == "list":
        payload = {"k_scale": k_scales, "v_scale": v_scales}
    else:
        layers = {}
        for i, (k, v) in enumerate(zip(k_scales, v_scales, strict=True)):
            layers[str(i)] = {"k_scale": k, "v_scale": v}
        payload = {"layers": layers}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-layer KV cache scales from a HF checkpoint."
    )
    parser.add_argument("--model-dir", required=True, help="Path to HF model directory.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument(
        "--format",
        choices=["list", "layers"],
        default="list",
        help="Output format: 'list' or 'layers'.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any layer is missing k/v scales (default fills missing with 1.0).",
    )
    args = parser.parse_args()

    k_scales, v_scales = _extract_scales(args.model_dir, strict=args.strict)

    _write_output(args.output, k_scales, v_scales, args.format)
    print(f"Wrote {len(k_scales)} layers to {args.output}")


if __name__ == "__main__":
    main()
