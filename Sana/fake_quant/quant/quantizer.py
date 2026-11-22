from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


SANA_BLOCK_LINEAR_PATTERNS = (
    ".attn.qkv",
    ".attn.proj",
    ".cross_attn.q_linear",
    ".cross_attn.kv_linear",
    ".cross_attn.image_kv_linear",
    ".cross_attn.proj",
    ".mlp.inverted_conv.linear",
    ".mlp.point_conv.linear",
    ".mlp.inverted_conv.conv",
    ".mlp.point_conv.conv",
    ".mlp.fc1",
    ".mlp.fc2",
)

# Backward-compatible alias for the original PixArt name.
PIXART_BLOCK_LINEAR_PATTERNS = SANA_BLOCK_LINEAR_PATTERNS

NVFP4_BLOCK_SIZE = 16
NVFP4_E2M1_MAX = 6.0
NVFP4_E4M3_MAX = 448.0
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def cast_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to finite FP4 E2M1 with round-to-nearest, ties-to-even."""
    dtype = x.dtype
    values = torch.tensor(_E2M1_MAGNITUDES, device=x.device, dtype=torch.float32)
    midpoints = (values[:-1] + values[1:]) * 0.5
    magnitude = x.float().abs().clamp(max=NVFP4_E2M1_MAX)
    indices = torch.bucketize(magnitude, midpoints, right=False)

    # bucketize chooses the lower code at a midpoint. E2M1 ties-to-even chooses
    # the code whose least-significant mantissa bit is zero.
    for lower_code, midpoint in enumerate(midpoints):
        if lower_code % 2 == 1:
            indices = torch.where(magnitude == midpoint, lower_code + 1, indices)
    rounded = values[indices] * x.float().sign()
    return rounded.to(dtype)


def cast_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round positive block scales to finite FP8 E4M3."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not provide torch.float8_e4m3fn.")
    bounded = x.float().clamp(min=0.0, max=NVFP4_E4M3_MAX)
    return bounded.to(torch.float8_e4m3fn).float()


def quantize_tensor_nvfp4(
    x: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Numerically emulate NVFP4 E2M1 + E4M3 block scaling.

    One E4M3 scale is shared by 16 consecutive values along the last
    dimension, and one FP32 global scale is shared by the full tensor.
    The returned tensor is dequantized to the input dtype for fake inference.
    """
    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 requires block_size={NVFP4_BLOCK_SIZE}, got {block_size}.")
    dtype = x.dtype
    work = x.float()
    original_shape = work.shape
    last_dim = original_shape[-1]
    pad_len = (-last_dim) % block_size
    if pad_len:
        work = F.pad(work, (0, pad_len))

    global_amax = work.abs().amax()
    if global_amax == 0:
        return torch.zeros(original_shape, device=x.device, dtype=dtype)

    global_scale = global_amax / (NVFP4_E4M3_MAX * NVFP4_E2M1_MAX)
    blocked = work.reshape(*work.shape[:-1], work.shape[-1] // block_size, block_size)
    block_amax = blocked.abs().amax(dim=-1, keepdim=True)
    block_scale_unquantized = (block_amax / NVFP4_E2M1_MAX) / global_scale
    block_scale_e4m3 = cast_to_e4m3(block_scale_unquantized)
    effective_scale = global_scale * block_scale_e4m3

    safe_scale = torch.where(effective_scale > 0, effective_scale, torch.ones_like(effective_scale))
    e2m1 = cast_to_e2m1(blocked / safe_scale)
    dequantized = e2m1.float() * effective_scale
    dequantized = dequantized.reshape(work.shape)
    if pad_len:
        dequantized = dequantized[..., :last_dim]
    return dequantized.reshape(original_shape).to(dtype)


def _infer_reduce_dim(x: torch.Tensor, quant_type: str, dim: int | None) -> int | None:
    if quant_type == "per_tensor":
        return None
    if dim is not None:
        return dim
    if quant_type == "per_group":
        return -1
    if quant_type == "per_channel":
        if x.dim() == 2:
            return 1
        return -1
    if quant_type == "per_token":
        return -1
    raise ValueError(f"Unknown quant_type: {quant_type}")


def _apply_dynamic_clipping(
    x: torch.Tensor,
    reduce_dim: int | None,
    clip_percentile: float | None,
) -> torch.Tensor:
    if clip_percentile is None:
        return x
    if not (50 < clip_percentile <= 100):
        raise ValueError("clip_percentile should be in (50, 100].")

    lower_q = (100 - clip_percentile) / 100
    upper_q = clip_percentile / 100

    if reduce_dim is None:
        clip_min = torch.quantile(x.reshape(-1), lower_q)
        clip_max = torch.quantile(x.reshape(-1), upper_q)
    else:
        clip_min = torch.quantile(x, lower_q, dim=reduce_dim, keepdim=True)
        clip_max = torch.quantile(x, upper_q, dim=reduce_dim, keepdim=True)
    return torch.clamp(x, clip_min, clip_max)


def quantize_tensor(
    x: torch.Tensor,
    n_bits: int = 4,
    group_size: int | None = None,
    quant_type: str = "per_group",
    sym: bool = True,
    dim: int | None = None,
    clip_percentile: float | None = None,
    data_format: str = "int",
) -> torch.Tensor:
    """Core fake-quant function supporting per-tensor, group, channel, and token modes."""
    if data_format == "nvfp4":
        if clip_percentile is not None:
            raise ValueError("Dynamic percentile clipping is not part of the NVFP4 recipe.")
        return quantize_tensor_nvfp4(x, block_size=group_size or NVFP4_BLOCK_SIZE)
    if data_format != "int":
        raise ValueError(f"Unknown quantization data format: {data_format}")
    dtype = x.dtype
    x = x.float()

    original_shape = x.shape
    pad_len = 0

    if quant_type == "per_group":
        if group_size is None:
            raise ValueError("group_size must be specified for per_group quantization")
        if group_size <= 0:
            raise ValueError("group_size must be positive for per_group quantization")

        last_dim = original_shape[-1]
        if last_dim % group_size != 0:
            pad_len = group_size - (last_dim % group_size)
            x = F.pad(x, (0, pad_len))
        current_shape = list(x.shape)
        new_shape = current_shape[:-1] + [current_shape[-1] // group_size, group_size]
        x_reshaped = x.view(new_shape)
    else:
        x_reshaped = x

    reduce_dim = _infer_reduce_dim(x_reshaped, quant_type, dim)
    x_reshaped = _apply_dynamic_clipping(x_reshaped, reduce_dim, clip_percentile)

    if sym:
        if reduce_dim is None:
            xmax = x_reshaped.abs().max()
        else:
            xmax = x_reshaped.abs().amax(dim=reduce_dim, keepdim=True)
        xmax = xmax.clamp(min=1e-5)
        qmax = 2 ** (n_bits - 1) - 1
        scale = (xmax / qmax).clamp(min=1e-5)
        x_q = torch.clamp(torch.round(x_reshaped / scale), -qmax, qmax)
        x_dequant = x_q * scale
    else:
        qmax = 2 ** n_bits - 1
        if reduce_dim is None:
            xmax = x_reshaped.max()
            xmin = x_reshaped.min()
        else:
            xmax = x_reshaped.amax(dim=reduce_dim, keepdim=True)
            xmin = x_reshaped.amin(dim=reduce_dim, keepdim=True)
        scale = ((xmax - xmin) / qmax).clamp(min=1e-5)
        zero = torch.round(-xmin / scale)
        x_q = torch.clamp(torch.round(x_reshaped / scale) + zero, 0, qmax)
        x_dequant = (x_q - zero) * scale

    if quant_type == "per_group":
        x_dequant = x_dequant.view(x.shape)
        if pad_len > 0:
            x_dequant = x_dequant[..., : original_shape[-1]]

    return x_dequant.to(dtype)


def _quantize_module_input(
    x: torch.Tensor,
    *,
    module: nn.Module,
    n_bits: int,
    group_size: int | None,
    quant_type: str,
    sym: bool,
    clip_percentile: float | None,
    data_format: str,
) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return quantize_tensor(
            x,
            n_bits=n_bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=clip_percentile,
            data_format=data_format,
        )

    if isinstance(module, nn.Conv2d):
        # Treat 1x1 convolutions like per-token linear layers by quantizing
        # over the channel dimension for each spatial location.
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        x_q = x.permute(0, 2, 3, 1)
        x_q = quantize_tensor(
            x_q,
            n_bits=n_bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=clip_percentile,
            data_format=data_format,
        )
        return x_q.permute(0, 3, 1, 2).contiguous()

    raise TypeError(f"Unsupported module type for activation quantization: {type(module).__name__}")


def _quantize_module_weight(
    module: nn.Module,
    *,
    n_bits: int,
    group_size: int | None,
    quant_type: str,
    sym: bool,
    clip_percentile: float | None,
    data_format: str,
) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data = quantize_tensor(
            module.weight.data,
            n_bits=n_bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=clip_percentile,
            data_format=data_format,
        )
        return

    if isinstance(module, nn.Conv2d):
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        weight_2d = module.weight.data.squeeze(-1).squeeze(-1)
        weight_2d = quantize_tensor(
            weight_2d,
            n_bits=n_bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=clip_percentile,
            data_format=data_format,
        )
        module.weight.data = weight_2d.unsqueeze(-1).unsqueeze(-1)
        return

    raise TypeError(f"Unsupported module type for weight quantization: {type(module).__name__}")


def _iter_sana_block_quant_modules(
    model: nn.Module,
    target_patterns: tp.Sequence[str] = SANA_BLOCK_LINEAR_PATTERNS,
):
    for name, module in model.named_modules():
        if not name.startswith("blocks."):
            continue
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            continue
        if any(name.endswith(pattern) for pattern in target_patterns):
            yield name, module


def _clear_act_quant_hooks(model: nn.Module) -> None:
    handles = getattr(model, "_sana_act_quant_handles", None)
    if not handles:
        return
    for handle in handles:
        handle.remove()
    model._sana_act_quant_handles = []


def quantize_sana_act(
    model: nn.Module,
    quant_config: dict[str, tp.Any] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register activation fake-quant pre-hooks on supported Sana block modules."""
    if quant_config is None:
        quant_config = {}
    _clear_act_quant_hooks(model)
    if not quant_config.get("act_quant", False):
        print("Skip activation quantization...")
        return []

    bits = quant_config.get("act_bits", 8)
    quant_type = quant_config.get("act_quant_type", "per_token")
    group_size = quant_config.get("act_group_size", None)
    sym = quant_config.get("act_sym", True)
    clip = quant_config.get("act_clip", False)
    clip_percentile = quant_config.get("act_clip_percentile", 99.9)
    data_format = quant_config.get("quant_format", "int")

    if clip:
        print(
            f"Quantizing Sana activations with dynamic clipping (percentile={clip_percentile}): "
            f"bits={bits}, type={quant_type}, group={group_size}, sym={sym}"
        )
    else:
        print(f"Quantizing Sana activations: bits={bits}, type={quant_type}, group={group_size}, sym={sym}")

    handles = []

    def pre_hook_fn(module: nn.Module, args: tuple[torch.Tensor, ...]):
        if len(args) == 0:
            return args
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return args
        x_q = _quantize_module_input(
            x,
            module=module,
            n_bits=bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=clip_percentile if clip else None,
            data_format=data_format,
        )
        return (x_q, *args[1:])

    count = 0
    for _, module in _iter_sana_block_quant_modules(model):
        if quant_config.get("linear_only", False) and not isinstance(module, nn.Linear):
            continue
        handle = module.register_forward_pre_hook(pre_hook_fn)
        handles.append(handle)
        count += 1
    model._sana_act_quant_handles = handles
    print(f"Registered activation quantization pre-hooks for {count} Sana block quantized modules.")
    return handles


def quantize_sana_weight(model: nn.Module, quant_config: dict[str, tp.Any] | None = None) -> int:
    """Quantize supported Sana transformer block weights in place."""
    if quant_config is None:
        quant_config = {}
    if not quant_config.get("weight_quant", True):
        print("Skip weight quantization...")
        return 0

    bits = quant_config.get("weight_bits", 4)
    quant_type = quant_config.get("weight_quant_type", "per_group")
    group_size = quant_config.get("weight_group_size", 128)
    sym = quant_config.get("weight_sym", True)
    clip = quant_config.get("weight_clip", False)
    clip_percentile = quant_config.get("weight_clip_percentile", 99.9)
    data_format = quant_config.get("quant_format", "int")

    if clip:
        print(
            f"Quantizing Sana weights with clipping (percentile={clip_percentile}): "
            f"bits={bits}, type={quant_type}, group={group_size}, sym={sym}"
        )
    else:
        print(f"Quantizing Sana weights: bits={bits}, type={quant_type}, group={group_size}, sym={sym}")

    count = 0
    for _, module in _iter_sana_block_quant_modules(model):
        if quant_config.get("linear_only", False) and not isinstance(module, nn.Linear):
            continue
        _quantize_module_weight(
            module,
            n_bits=bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=clip_percentile if clip else None,
            data_format=data_format,
        )
        count += 1
    print(f"Quantized weights for {count} Sana block quantized modules.")
    return count


def quantize_pixart_act(
    model: nn.Module,
    quant_config: dict[str, tp.Any] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    return quantize_sana_act(model, quant_config)


def quantize_pixart_weight(model: nn.Module, quant_config: dict[str, tp.Any] | None = None) -> int:
    return quantize_sana_weight(model, quant_config)
