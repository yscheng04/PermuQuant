import torch
import torch.nn as nn


def _quantize_e4m3_scale(scale):
    """Quantize positive block scales to E4M3 when PyTorch exposes float8 support."""
    scale = scale.clamp(min=1e-12)
    if hasattr(torch, "float8_e4m3fn"):
        return scale.to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-12)

    # Fallback finite E4M3 approximation for environments without float8 dtype support.
    exponent_bias = 7
    max_normal_exp = 7
    min_normal_exp = -6
    exponent = torch.floor(torch.log2(scale)).clamp(min_normal_exp, max_normal_exp)
    mantissa = torch.round((scale / torch.pow(2.0, exponent) - 1.0) * 8.0).clamp(0, 7)
    quantized = torch.pow(2.0, exponent) * (1.0 + mantissa / 8.0)
    min_subnormal = 2.0 ** (1 - exponent_bias - 3)
    return quantized.clamp(min=min_subnormal)


def _quantize_nvfp4(x, group_size):
    """Fake NVFP4 quantization with E2M1 values and E4M3 block scaling."""
    if group_size is None:
        raise ValueError("group_size must be specified for nvfp4 quantization")

    dtype = x.dtype
    x = x.float()
    original_shape = x.shape
    pad_len = 0
    if original_shape[-1] % group_size != 0:
        pad_len = group_size - (original_shape[-1] % group_size)
        x = torch.nn.functional.pad(x, (0, pad_len))

    grouped_shape = list(x.shape[:-1]) + [x.shape[-1] // group_size, group_size]
    x_grouped = x.view(grouped_shape)
    block_amax = x_grouped.abs().amax(dim=-1, keepdim=True)
    scale = _quantize_e4m3_scale(block_amax / 6.0)

    codebook = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    normalized_abs = (x_grouped.abs() / scale).clamp(max=6.0)
    nearest = torch.argmin((normalized_abs.unsqueeze(-1) - codebook).abs(), dim=-1)
    quantized_abs = codebook[nearest]
    x_dequant = torch.sign(x_grouped) * quantized_abs * scale
    x_dequant = x_dequant.view(x.shape)
    if pad_len > 0:
        x_dequant = x_dequant[..., : original_shape[-1]]
    return x_dequant.to(dtype)


def quantize_tensor(
    x,
    n_bits=4,
    group_size=None,
    quant_type="per_group",
    sym=True,
    dim=None,
    clip_percentile=None,
):
    """
    Core quantization function supporting various modes.

    Args:
        x: Input tensor
        n_bits: Number of bits
        group_size: Group size for per-group quantization
        quant_type: "per_tensor", "per_channel", "per_token", "per_group"
        sym: Symmetric quantization or not
        dim: Dimension to quantize along (for per-channel/per-token).
             If None, inferred from quant_type.
        clip_percentile: If not None, clip values to [percentile, 100-percentile] range
                         before quantization. Clipping is applied at the same granularity
                         as quant_type (per-tensor/per-channel/per-token/per-group).
    """
    dtype = x.dtype
    if quant_type == "nvfp4":
        return _quantize_nvfp4(x, group_size)

    x = x.float()
    # 1. Shape Handling for Group Quantization
    original_shape = x.shape
    pad_len = 0
    if quant_type == "per_group":
        if group_size is None:
            raise ValueError("group_size must be specified for per_group quantization")

        last_dim = original_shape[-1]

        if last_dim % group_size != 0:
            pad_len = group_size - (last_dim % group_size)
            x = torch.nn.functional.pad(x, (0, pad_len))

        # Reshape to (..., num_groups, group_size)
        current_shape = list(x.shape)
        new_shape = current_shape[:-1] + [current_shape[-1] // group_size, group_size]
        x_reshaped = x.view(new_shape)

        # Calculate stats over the last dimension (the group)
        reduce_dim = -1

    elif quant_type == "per_tensor":
        x_reshaped = x
        reduce_dim = None  # Global min/max

    elif quant_type in ["per_channel", "per_token"]:
        x_reshaped = x

        if dim is None:
            if x.dim() == 2:  # Weight (Out, In)
                reduce_dim = 1  # Reduce along In -> stats per Out channel
            elif x.dim() == 3:  # Activation (B, S, H)
                reduce_dim = 2  # Reduce along H -> stats per Token
            else:
                reduce_dim = -1
        else:
            reduce_dim = dim
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    # 2. Dynamic Clipping (optional)
    if clip_percentile is not None:
        assert 50 < clip_percentile <= 100, "clip_percentile should be in (50, 100)"
        lower_p = (100 - clip_percentile) / 100
        upper_p = clip_percentile / 100

        x_flat = x_reshaped.flatten()
        n = x_flat.numel()
        kth_low = int(lower_p * (n - 1))
        kth_high = int(upper_p * (n - 1))
        x_sorted, _ = torch.sort(x_flat)
        clip_min = x_sorted[kth_low]
        clip_max = x_sorted[kth_high]

        x_reshaped = torch.clamp(x_reshaped, clip_min, clip_max)

    # 3. Calculate Scale and Zero
    if sym:
        if reduce_dim is None:
            xmax = x_reshaped.abs().max()
        else:
            xmax = x_reshaped.abs().max(dim=reduce_dim, keepdim=True)[0]

        xmax = xmax.clamp(min=1e-5)
        qmax = 2 ** (n_bits - 1) - 1
        scale = xmax / qmax
        scale = scale.clamp(min=1e-5)
        x_q = torch.clamp(torch.round(x_reshaped / scale), -qmax, qmax)
        x_dequant = x_q * scale

    else:
        # Asymmetric: range is [min, max]
        qmax = 2**n_bits - 1
        if reduce_dim is None:
            xmax = x_reshaped.max()
            xmin = x_reshaped.min()
        else:
            xmax = x_reshaped.max(dim=reduce_dim, keepdim=True)[0]
            xmin = x_reshaped.min(dim=reduce_dim, keepdim=True)[0]

        scale = (xmax - xmin) / qmax
        scale = scale.clamp(min=1e-5)
        zero = torch.round(-xmin / scale)

        x_q = torch.clamp(torch.round(x_reshaped / scale) + zero, 0, qmax)
        x_dequant = (x_q - zero) * scale

    # 4. Restore Shape
    if quant_type == "per_group":
        x_dequant = x_dequant.view(x.shape)
        if pad_len > 0:
            x_dequant = x_dequant[..., : original_shape[-1]]

    return x_dequant.to(dtype)


def quantize_flux_act(model, quant_config=None):
    """
    Apply activation quantization to FLUX model.

    Args:
        model: FLUX transformer model (Flux or nn.Module)
        quant_config: Quantization configuration dict
    """
    if quant_config is None:
        quant_config = {}
    act_quant = quant_config.get("act_quant", False)
    if not act_quant:
        print("Skip activation quantization...")
        return

    # Default Config
    bits = quant_config.get("act_bits", 8)
    quant_type = quant_config.get("act_quant_type", "per_token")
    group_size = quant_config.get("act_group_size", None)
    sym = quant_config.get("act_sym", True)

    # Clipping config (with defaults)
    act_clip = quant_config.get("act_clip", False)
    act_clip_percentile = quant_config.get("act_clip_percentile", 99.9)

    if act_clip:
        print(
            f"Quantizing Activations with dynamic clipping (percentile={act_clip_percentile}): "
            f"bits={bits}, type={quant_type}, group={group_size}, sym={sym}"
        )
    else:
        print(f"Quantizing Activations: bits={bits}, type={quant_type}, group={group_size}, sym={sym}")

    # FLUX target modules for activation quantization
    # Target Linear layers that receive activations
    target_modules = [
        "img_attn.qkv",
        "img_attn.proj",
        "txt_attn.qkv",
        "txt_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mlp.0",
        "txt_mlp.2",
        "linear1",
        "linear2",
    ]

    count = 0

    def pre_hook_fn(module, args):
        x = args[0]
        x_q = quantize_tensor(
            x,
            n_bits=bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=act_clip_percentile if act_clip else None,
        )
        return (x_q, *args[1:])

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_target = any(target in name for target in target_modules)
            if is_target:
                module.register_forward_pre_hook(pre_hook_fn)
                count += 1

    print(f"Registered activation quantization pre-hooks for {count} layers.")


def quantize_flux_weight(model, quant_config=None):
    """
    Apply weight quantization to FLUX model.

    Args:
        model: FLUX transformer model (Flux or nn.Module)
        quant_config: Quantization configuration dict
    """
    if quant_config is None:
        quant_config = {}
    weight_quant = quant_config.get("weight_quant", True)
    if not weight_quant:
        print("Skip weight quantization...")
        return

    # Default Config
    bits = quant_config.get("weight_bits", 4)
    quant_type = quant_config.get("weight_quant_type", "per_group")
    group_size = quant_config.get("weight_group_size", 128)
    sym = quant_config.get("weight_sym", True)

    # Clipping config (with defaults)
    weight_clip = quant_config.get("weight_clip", False)
    weight_clip_percentile = quant_config.get("weight_clip_percentile", 99.9)

    if weight_clip:
        print(
            f"Quantizing Weights with clipping (percentile={weight_clip_percentile}): "
            f"bits={bits}, type={quant_type}, group={group_size}, sym={sym}"
        )
    else:
        print(f"Quantizing Weights: bits={bits}, type={quant_type}, group={group_size}, sym={sym}")

    # FLUX target modules for weight quantization
    target_modules = [
        "img_attn.qkv",
        "img_attn.proj",
        "txt_attn.qkv",
        "txt_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mlp.0",
        "txt_mlp.2",
        "linear1",
        "linear2",  # LastLayer has norm_final and linear
    ]

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_target = any(target in name for target in target_modules)
            if is_target:
                # Quantize weight in-place with dynamic clipping if enabled
                w = module.weight.data
                w_q = quantize_tensor(
                    w,
                    n_bits=bits,
                    group_size=group_size,
                    quant_type=quant_type,
                    sym=sym,
                    clip_percentile=weight_clip_percentile if weight_clip else None,
                )
                module.weight.data = w_q
                count += 1

    print(f"Quantized weights for {count} layers.")
