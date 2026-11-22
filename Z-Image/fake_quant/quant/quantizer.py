import torch
import torch.nn as nn


def _e2m1_codebook(device):
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=torch.float32)


def _e4m3_positive_codebook(device):
    """Positive finite E4M3 values with NVIDIA-style max 448."""
    values = [0.0]
    # Subnormals for exponent -6 with 3 mantissa bits.
    values.extend([(mant / 8.0) * (2.0 ** -6) for mant in range(1, 8)])
    for exp in range(-6, 9):
        for mant in range(8):
            values.append((1.0 + mant / 8.0) * (2.0 ** exp))
    values.append(448.0)
    return torch.tensor(sorted(set(values)), device=device, dtype=torch.float32)


def _quantize_to_codebook_abs(x_abs, codebook):
    flat = x_abs.reshape(-1)
    # Codebooks are tiny; chunk to avoid large [N, K] temporaries for image-token activations.
    out = torch.empty_like(flat)
    chunk_size = 1_000_000
    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        chunk = flat[start:end]
        idx = (chunk[:, None] - codebook[None, :]).abs().argmin(dim=1)
        out[start:end] = codebook[idx]
    return out.view_as(x_abs)


def _quantize_e4m3_positive(x):
    codebook = _e4m3_positive_codebook(x.device)
    x = x.float().clamp(min=0.0, max=448.0)
    return _quantize_to_codebook_abs(x, codebook)


def _quantize_e2m1_signed(x):
    codebook = _e2m1_codebook(x.device)
    sign = torch.sign(x)
    return sign * _quantize_to_codebook_abs(x.abs().clamp(max=6.0), codebook)


def _nvfp4_quantize_1d_blocks(x, block_size=16):
    """Fake-quantize with NVFP4 1D 16-element blocks along the last dimension."""
    original_shape = x.shape
    last_dim = original_shape[-1]
    pad_len = (block_size - last_dim % block_size) % block_size
    if pad_len:
        x = torch.nn.functional.pad(x, (0, pad_len))

    x_blocks = x.view(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    fp4_max = 6.0
    fp8_max = 448.0
    global_amax = x_blocks.abs().amax()
    if global_amax <= 0:
        x_dequant = torch.zeros_like(x_blocks).view(x.shape)
    else:
        s_global = (global_amax / (fp8_max * fp4_max)).clamp(min=1e-12)
        block_amax = x_blocks.abs().amax(dim=-1, keepdim=True)
        s_block = (block_amax / fp4_max) / s_global
        s_block_q = _quantize_e4m3_positive(s_block).clamp(min=1e-12)
        x_scaled = x_blocks / (s_block_q * s_global)
        x_e2m1 = _quantize_e2m1_signed(x_scaled)
        x_dequant = (x_e2m1 * s_block_q * s_global).view(x.shape)

    if pad_len:
        x_dequant = x_dequant[..., :last_dim]
    return x_dequant.view(original_shape)


def _nvfp4_quantize_weight_2d_blocks(w, block_size=16):
    """Fake-quantize weights with NVFP4 16x16 two-dimensional block scaling."""
    if w.dim() != 2:
        return _nvfp4_quantize_1d_blocks(w, block_size=block_size)

    out_features, in_features = w.shape
    pad_out = (block_size - out_features % block_size) % block_size
    pad_in = (block_size - in_features % block_size) % block_size
    if pad_out or pad_in:
        w = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))

    out_blocks = w.shape[0] // block_size
    in_blocks = w.shape[1] // block_size
    w_blocks = w.view(out_blocks, block_size, in_blocks, block_size).permute(0, 2, 1, 3)
    fp4_max = 6.0
    fp8_max = 448.0
    global_amax = w_blocks.abs().amax()
    if global_amax <= 0:
        w_dequant = torch.zeros_like(w_blocks)
    else:
        s_global = (global_amax / (fp8_max * fp4_max)).clamp(min=1e-12)
        block_amax = w_blocks.abs().amax(dim=(-1, -2), keepdim=True)
        s_block = (block_amax / fp4_max) / s_global
        s_block_q = _quantize_e4m3_positive(s_block).clamp(min=1e-12)
        w_scaled = w_blocks / (s_block_q * s_global)
        w_e2m1 = _quantize_e2m1_signed(w_scaled)
        w_dequant = w_e2m1 * s_block_q * s_global

    w_dequant = w_dequant.permute(0, 2, 1, 3).contiguous().view(w.shape)
    if pad_out:
        w_dequant = w_dequant[:-pad_out, :]
    if pad_in:
        w_dequant = w_dequant[:, :-pad_in]
    return w_dequant


def quantize_nvfp4_tensor(x, group_size=16, weight_2d=None):
    """Fake NVFP4 E2M1 + E4M3 block scaling.

    Activations use 1x16 blocks along the last dimension. 2D weights use 16x16
    block scaling by default, matching the NVFP4 inference recipe.
    """
    dtype = x.dtype
    x_float = x.float()
    if weight_2d is None:
        weight_2d = x_float.dim() == 2
    if group_size is None:
        group_size = 16
    if weight_2d:
        x_q = _nvfp4_quantize_weight_2d_blocks(x_float, block_size=group_size)
    else:
        x_q = _nvfp4_quantize_1d_blocks(x_float, block_size=group_size)
    return x_q.to(dtype)

def quantize_tensor(x, n_bits=4, group_size=None, quant_type="per_group", sym=True, dim=None,
                    clip_percentile=None):
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
    if quant_type in {"nvfp4", "nvfp4_1d", "nvfp4_2d"}:
        weight_2d = None
        if quant_type == "nvfp4_1d":
            weight_2d = False
        elif quant_type == "nvfp4_2d":
            weight_2d = True
        return quantize_nvfp4_tensor(x, group_size=group_size or 16, weight_2d=weight_2d)

    dev = x.device
    dtype = x.dtype
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
        reduce_dim = None # Global min/max
        
    elif quant_type in ["per_channel", "per_token"]:
        x_reshaped = x
        
        if dim is None:
            if x.dim() == 2: # Weight (Out, In)
                reduce_dim = 1 # Reduce along In -> stats per Out channel
            elif x.dim() == 3: # Activation (B, S, H)
                reduce_dim = 2 # Reduce along H -> stats per Token
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
        # if quant_type == "per_group":
        #     # x_reshaped: (..., num_groups, group_size)
        #     # Compute percentile along last dim (group_size)
        #     x_flat = x_reshaped.reshape(-1, group_size)
        #     kth_low = int(lower_p * (x_flat.size(1) - 1))
        #     kth_high = int(upper_p * (x_flat.size(1) - 1))
        #     x_sorted, _ = torch.sort(x_flat, dim=1)
        #     clip_min = x_sorted[:, kth_low].view(x_reshaped.shape[:-1] + (1,))
        #     clip_max = x_sorted[:, kth_high].view(x_reshaped.shape[:-1] + (1,))
        # elif quant_type == "per_tensor":
        #     x_flat = x_reshaped.flatten()
        #     n = x_flat.numel()
        #     kth_low = int(lower_p * (n - 1))
        #     kth_high = int(upper_p * (n - 1))
        #     x_sorted, _ = torch.sort(x_flat)
        #     clip_min = x_sorted[kth_low]
        #     clip_max = x_sorted[kth_high]
        # elif quant_type in ["per_channel", "per_token"]:
        #     # x_reshaped: original shape, reduce along reduce_dim
        #     x_transposed = x_reshaped.transpose(reduce_dim, -1).contiguous()
        #     x_flat = x_transposed.reshape(-1, x_transposed.size(-1))
        #     kth_low = int(lower_p * (x_flat.size(1) - 1))
        #     kth_high = int(upper_p * (x_flat.size(1) - 1))
        #     x_sorted, _ = torch.sort(x_flat, dim=1)
        #     clip_min = x_sorted[:, kth_low].view(x_transposed.shape[:-1]).transpose(reduce_dim, -1)
        #     clip_max = x_sorted[:, kth_high].view(x_transposed.shape[:-1]).transpose(reduce_dim, -1)

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
        qmax = 2 ** n_bits - 1
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
            x_dequant = x_dequant[..., :original_shape[-1]]
            
    return x_dequant.to(dtype)

def quantize_zimage_act(model, quant_config=None):
    if quant_config is None:
        quant_config = {}
    act_quant = quant_config['act_quant']
    if not act_quant:
        print("Skip activation quantization...")
        return

    # Default Config
    bits = quant_config['act_bits']
    quant_type = quant_config['act_quant_type']
    group_size = quant_config['act_group_size']
    sym = quant_config['act_sym']

    # Clipping config (with defaults)
    act_clip = quant_config.get('act_clip', False)
    act_clip_percentile = quant_config.get('act_clip_percentile', 99.9)

    if act_clip:
        print(f"Quantizing Activations with dynamic clipping (percentile={act_clip_percentile}): bits={bits}, type={quant_type}, group={group_size}, sym={sym}")
    else:
        print(f"Quantizing Activations: bits={bits}, type={quant_type}, group={group_size}, sym={sym}")

    # Handle wrapped model
    target_model = model
    if hasattr(target_model, "language_model"):
        target_model = target_model.language_model
    if hasattr(target_model, "model"):
        target_model = target_model.model

    target_modules = [
        "w1", "w2", "w3",
        "to_q", "to_k", "to_v",
        "to_out.0",
    ]

    # Skip refiner layers
    skip_prefixes = ["noise_refiner", "context_refiner"]

    count = 0

    def pre_hook_fn(module, args):
        x = args[0]
        x_q = quantize_tensor(
            x,
            n_bits=bits,
            group_size=group_size,
            quant_type=quant_type,
            sym=sym,
            clip_percentile=act_clip_percentile if act_clip else None
        )
        return (x_q, *args[1:])

    for name, module in target_model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            is_target = any(name.endswith(t) for t in target_modules)
            if is_target:
                module.register_forward_pre_hook(pre_hook_fn)
                count += 1

    print(f"Registered activation quantization pre-hooks for {count} layers.")


def quantize_zimage_weight(model, quant_config=None):
    if quant_config is None:
        quant_config = {}
    weight_quant = quant_config.get('weight_quant', True)
    if not weight_quant:
        print("Skip weight quantization...")
        return

    # Default Config
    bits = quant_config.get('weight_bits', 4)
    quant_type = quant_config.get('weight_quant_type', 'per_group')
    group_size = quant_config.get('weight_group_size', 128)
    sym = quant_config.get('weight_sym', True)

    # Clipping config (with defaults)
    weight_clip = quant_config.get('weight_clip', False)
    weight_clip_percentile = quant_config.get('weight_clip_percentile', 99.9)

    if weight_clip:
        print(f"Quantizing Weights with clipping (percentile={weight_clip_percentile}): bits={bits}, type={quant_type}, group={group_size}, sym={sym}")
    else:
        print(f"Quantizing Weights: bits={bits}, type={quant_type}, group={group_size}, sym={sym}")

    if hasattr(model, "language_model"):
        model = model.language_model
    if hasattr(model, "model"):
        model = model.model

    target_modules = [
        "w1", "w2", "w3",
        "to_q", "to_k", "to_v",
        "to_out.0",
    ]

    # Skip refiner layers
    skip_prefixes = ["noise_refiner", "context_refiner"]

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip refiner layers
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            is_target = any(name.endswith(t) for t in target_modules)
            if is_target:
                # Quantize weight in-place with dynamic clipping if enabled
                w = module.weight.data
                w_q = quantize_tensor(
                    w,
                    n_bits=bits,
                    group_size=group_size,
                    quant_type=quant_type,
                    sym=sym,
                    clip_percentile=weight_clip_percentile if weight_clip else None
                )
                module.weight.data = w_q
                count += 1

    print(f"Quantized weights for {count} layers.")
