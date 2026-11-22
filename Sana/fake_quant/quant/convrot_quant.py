"""ConvRot and QuaRot Hadamard-rotation quantization for Sana models."""

from __future__ import annotations

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import (
    SANA_BLOCK_LINEAR_PATTERNS,
    _iter_sana_block_quant_modules,
    quantize_sana_act,
    quantize_sana_weight,
    quantize_tensor,
)

CONVROT_TARGET_PATTERNS = tuple(
    pattern for pattern in SANA_BLOCK_LINEAR_PATTERNS if pattern != ".cross_attn.q_linear"
)


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "blocks"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model
    return model


def get_regular_hadamard_matrix_kronecker(dim: int, device: torch.device) -> torch.Tensor:
    """Generate a normalized regular Hadamard matrix using Kronecker products."""
    if dim <= 0:
        raise ValueError(f"Dimension {dim} must be positive.")

    temp_dim = dim
    while temp_dim > 1:
        if temp_dim % 4 != 0:
            raise ValueError(f"Dimension {dim} must be a power of 4 for Regular Hadamard Kronecker construction.")
        temp_dim //= 4

    h4 = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    if dim == 4:
        return h4 / 2.0

    h = h4
    curr_dim = 4
    while curr_dim < dim:
        h = torch.kron(h, h4)
        curr_dim *= 4

    return h / math.sqrt(dim)


def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform over the last dimension."""
    dim = x.shape[-1]
    if dim <= 0:
        raise ValueError("Input last dimension must be positive.")
    if (dim & (dim - 1)) != 0:
        raise ValueError(f"Dimension {dim} must be a power of 2 for Walsh-Hadamard transform.")

    orig_shape = x.shape
    x = x.reshape(-1, dim).contiguous()

    block_size = 1
    while block_size < dim:
        x = x.reshape(-1, dim // (2 * block_size), 2, block_size)
        left = x[:, :, 0, :].clone()
        right = x[:, :, 1, :].clone()
        x[:, :, 0, :] = left + right
        x[:, :, 1, :] = left - right
        x = x.reshape(-1, dim)
        block_size *= 2

    x = x / math.sqrt(dim)
    return x.reshape(orig_shape)


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    if (n & (n - 1)) == 0:
        return n
    return 1 << (n - 1).bit_length()


def _clear_rotation_hooks(model: nn.Module) -> None:
    handles = getattr(model, "_sana_convrot_handles", None)
    if not handles:
        return
    for handle in handles:
        handle.remove()
    model._sana_convrot_handles = []


def _apply_group_rotation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    in_features = x.shape[-1]
    group_size = hadamard.shape[0]
    num_groups = in_features // group_size
    x_r = x.reshape(*x.shape[:-1], num_groups, group_size)
    x_r = torch.matmul(x_r, hadamard)
    return x_r.reshape(*x.shape[:-1], in_features)


def _quantize_with_config(
    x: torch.Tensor,
    *,
    bits: int,
    group_size: int | None,
    quant_type: str,
    sym: bool,
    clip: bool,
    clip_percentile: float,
) -> torch.Tensor:
    return quantize_tensor(
        x,
        n_bits=bits,
        group_size=group_size,
        quant_type=quant_type,
        sym=sym,
        clip_percentile=clip_percentile if clip else None,
    )


def _error_stats(reference: torch.Tensor, approx: torch.Tensor) -> tuple[float, float, float]:
    diff = (approx.float() - reference.float()).reshape(-1)
    rmse = torch.sqrt(torch.mean(diff.pow(2))).item()
    mae = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    return rmse, mae, max_abs


def _module_weight_to_2d(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.data
    if isinstance(module, nn.Conv2d):
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        return module.weight.data.squeeze(-1).squeeze(-1)
    raise TypeError(f"Unsupported module type: {type(module).__name__}")


def _module_weight_from_2d(module: nn.Module, weight_2d: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return weight_2d
    if isinstance(module, nn.Conv2d):
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        return weight_2d.unsqueeze(-1).unsqueeze(-1)
    raise TypeError(f"Unsupported module type: {type(module).__name__}")


def _input_to_channel_last(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return x
    if isinstance(module, nn.Conv2d):
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        return x.permute(0, 2, 3, 1).contiguous()
    raise TypeError(f"Unsupported module type: {type(module).__name__}")


def _input_from_channel_last(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return x
    if isinstance(module, nn.Conv2d):
        return x.permute(0, 3, 1, 2).contiguous()
    raise TypeError(f"Unsupported module type: {type(module).__name__}")


@torch.no_grad()
def _debug_weight_quant_error(
    name: str,
    module: nn.Module,
    hadamard: torch.Tensor,
    quant_config: dict[str, tp.Any],
) -> None:
    if not quant_config.get("weight_quant", True):
        return

    bits = quant_config.get("weight_bits", 4)
    quant_type = quant_config.get("weight_quant_type", "per_group")
    group_size = quant_config.get("weight_group_size", 128)
    sym = quant_config.get("weight_sym", True)
    clip = quant_config.get("weight_clip", False)
    clip_percentile = quant_config.get("weight_clip_percentile", 99.9)

    weight = _module_weight_to_2d(module).detach()
    direct_quant = _quantize_with_config(
        weight,
        bits=bits,
        group_size=group_size,
        quant_type=quant_type,
        sym=sym,
        clip=clip,
        clip_percentile=clip_percentile,
    )
    weight_rot = _apply_group_rotation(weight, hadamard)
    rotated_quant = _quantize_with_config(
        weight_rot,
        bits=bits,
        group_size=group_size,
        quant_type=quant_type,
        sym=sym,
        clip=clip,
        clip_percentile=clip_percentile,
    )
    recovered = _apply_group_rotation(rotated_quant, hadamard.transpose(0, 1))

    direct_rmse, direct_mae, direct_max = _error_stats(weight, direct_quant)
    rot_rmse, rot_mae, rot_max = _error_stats(weight, recovered)
    print(
        f"[ConvRot Debug][Weight] {name}: "
        f"direct rmse={direct_rmse:.6e}, mae={direct_mae:.6e}, max={direct_max:.6e}; "
        f"rotated rmse={rot_rmse:.6e}, mae={rot_mae:.6e}, max={rot_max:.6e}"
    )


@torch.no_grad()
def _register_group_rotate_hook(
    module: nn.Module,
    hadamard: torch.Tensor,
    *,
    name: str,
    quant_config: dict[str, tp.Any] | None = None,
) -> torch.utils.hooks.RemovableHandle:
    quant_config = quant_config or {}
    weight_dtype = module.weight.dtype
    hadamard = hadamard.to(module.weight.device, weight_dtype)
    in_features = module.weight.shape[1]

    if quant_config.get("convrot_debug", False) and quant_config.get("convrot_debug_weight", True):
        _debug_weight_quant_error(name, module, hadamard, quant_config)

    weight_2d = _module_weight_to_2d(module)
    weight_rot = _apply_group_rotation(weight_2d, hadamard)
    module.weight.data = _module_weight_from_2d(module, weight_rot).to(module.weight.device, weight_dtype)

    debug_act_enabled = (
        quant_config.get("convrot_debug", False)
        and quant_config.get("convrot_debug_act", True)
        and quant_config.get("act_quant", False)
    )
    debug_sample_rows = quant_config.get("convrot_debug_act_sample_rows", 256)
    debug_call_limit = quant_config.get("convrot_debug_act_call_limit", 1)
    debug_call_count = 0

    def pre_hook_fn(_module: nn.Module, args: tuple[tp.Any, ...]):
        nonlocal debug_call_count
        if not args:
            return args
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return args

        x_c = _input_to_channel_last(x, _module)
        if x_c.shape[-1] != in_features:
            return args

        hadamard_local = hadamard
        if hadamard_local.device != x_c.device:
            hadamard_local = hadamard_local.to(x_c.device)
        if hadamard_local.dtype != x_c.dtype:
            hadamard_local = hadamard_local.to(x_c.dtype)
        x_r = _apply_group_rotation(x_c, hadamard_local)

        if debug_act_enabled and debug_call_count < debug_call_limit:
            bits = quant_config.get("act_bits", 8)
            quant_type = quant_config.get("act_quant_type", "per_token")
            group_size = quant_config.get("act_group_size", None)
            sym = quant_config.get("act_sym", True)
            clip = quant_config.get("act_clip", False)
            clip_percentile = quant_config.get("act_clip_percentile", 99.9)

            x_sample = x_c.reshape(-1, in_features)
            x_r_sample = x_r.reshape(-1, in_features)
            if debug_sample_rows is not None and x_sample.shape[0] > debug_sample_rows:
                x_sample = x_sample[:debug_sample_rows]
                x_r_sample = x_r_sample[:debug_sample_rows]

            direct_quant = _quantize_with_config(
                x_sample,
                bits=bits,
                group_size=group_size,
                quant_type=quant_type,
                sym=sym,
                clip=clip,
                clip_percentile=clip_percentile,
            )
            rotated_quant = _quantize_with_config(
                x_r_sample,
                bits=bits,
                group_size=group_size,
                quant_type=quant_type,
                sym=sym,
                clip=clip,
                clip_percentile=clip_percentile,
            )
            recovered = _apply_group_rotation(rotated_quant, hadamard_local.transpose(0, 1))

            direct_rmse, direct_mae, direct_max = _error_stats(x_sample, direct_quant)
            rot_rmse, rot_mae, rot_max = _error_stats(x_sample, recovered)
            print(
                f"[ConvRot Debug][Act] {name}: "
                f"direct rmse={direct_rmse:.6e}, mae={direct_mae:.6e}, max={direct_max:.6e}; "
                f"rotated rmse={rot_rmse:.6e}, mae={rot_mae:.6e}, max={rot_max:.6e}; "
                f"sample_rows={x_sample.shape[0]}"
            )
            debug_call_count += 1

        x_r = _input_from_channel_last(x_r, _module)
        return (x_r, *args[1:])

    return module.register_forward_pre_hook(pre_hook_fn)


@torch.no_grad()
def _register_quarot_hook(
    module: nn.Module,
    original_in_features: int,
    padded_dim: int,
) -> torch.utils.hooks.RemovableHandle:
    weight_2d = _module_weight_to_2d(module)
    if original_in_features < padded_dim:
        weight_2d = F.pad(weight_2d, (0, padded_dim - original_in_features))
    weight_2d = fast_walsh_hadamard_transform(weight_2d)
    module.weight.data = _module_weight_from_2d(module, weight_2d).to(module.weight.device, module.weight.dtype)

    if isinstance(module, nn.Linear):
        module.in_features = padded_dim
    elif isinstance(module, nn.Conv2d):
        module.in_channels = padded_dim

    def pre_hook_fn(_module: nn.Module, args: tuple[tp.Any, ...]):
        if not args:
            return args
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return args

        x_c = _input_to_channel_last(x, _module)
        if x_c.shape[-1] != original_in_features:
            return args

        if original_in_features < padded_dim:
            x_c = F.pad(x_c, (0, padded_dim - original_in_features))

        x_c = fast_walsh_hadamard_transform(x_c)
        x_c = _input_from_channel_last(x_c, _module)
        return (x_c, *args[1:])

    return module.register_forward_pre_hook(pre_hook_fn)


@torch.no_grad()
def rotate_sana(
    model: nn.Module,
    rotate_group_size: int = 128,
    rotation_type: str = "convrot",
    target_patterns: tp.Sequence[str] | None = None,
    quant_config: dict[str, tp.Any] | None = None,
) -> nn.Module:
    """Apply Hadamard rotation to Sana transformer block quant modules."""
    model = _unwrap_model(model)
    _clear_rotation_hooks(model)
    quant_config = quant_config or {}
    if target_patterns is None:
        target_patterns = (
            CONVROT_TARGET_PATTERNS if rotation_type == "convrot" else SANA_BLOCK_LINEAR_PATTERNS
        )

    print(f"Applying {rotation_type.upper()} Hadamard rotation to Sana...")
    count = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []
    debug_max_layers = quant_config.get("convrot_debug_max_layers", None)

    hadamard_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    for name, module in _iter_sana_block_quant_modules(model, target_patterns=target_patterns):
        in_features = module.weight.shape[1]

        if rotation_type == "convrot":
            if in_features % rotate_group_size != 0:
                print(
                    f"Skip {name}: in_features={in_features} is not divisible by "
                    f"rotate_group_size={rotate_group_size}."
                )
                continue

            cache_key = (rotate_group_size, module.weight.device, module.weight.dtype)
            if cache_key not in hadamard_cache:
                hadamard_cache[cache_key] = get_regular_hadamard_matrix_kronecker(
                    rotate_group_size,
                    module.weight.device,
                ).to(module.weight.dtype)
            layer_quant_config = quant_config
            if debug_max_layers is not None and count >= debug_max_layers:
                layer_quant_config = {**quant_config, "convrot_debug": False}
            handle = _register_group_rotate_hook(
                module,
                hadamard_cache[cache_key],
                name=name,
                quant_config=layer_quant_config,
            )
        elif rotation_type == "quarot":
            padded_dim = next_power_of_2(in_features)
            handle = _register_quarot_hook(module, in_features, padded_dim)
        else:
            raise ValueError(f"Unknown rotation_type: {rotation_type}. Use 'convrot' or 'quarot'.")

        handles.append(handle)
        count += 1

    model._sana_convrot_handles = handles
    model._sana_convrot_applied = True
    print(f"Applied rotation to {count} Sana block quantized modules.")
    return model


def quant_sana_convrot(
    model: nn.Module,
    quant_config: dict[str, tp.Any] | None = None,
) -> nn.Module:
    """Apply ConvRot or QuaRot, then weight and activation quantization to Sana."""
    if quant_config is None:
        quant_config = {}

    model = _unwrap_model(model)
    rotate_group_size = quant_config.get("rotate_group_size", 64)
    rotation_type = quant_config.get("rotation_type", "convrot")

    print(f"\n[Step 1] Applying {rotation_type.upper()} Hadamard Rotation...")
    rotate_sana(
        model,
        rotate_group_size=rotate_group_size,
        rotation_type=rotation_type,
        quant_config=quant_config,
    )

    print("\n[Step 2] Quantizing weights...")
    quantize_sana_weight(model, quant_config)

    print("\n[Step 3] Quantizing activations...")
    quantize_sana_act(model, quant_config)

    return model
