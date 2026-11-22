"""ConvRot and QuaRot Hadamard-rotation quantization for FLUX model."""

import math

import torch
import torch.nn as nn
from .quantizer import quantize_flux_weight, quantize_flux_act


def get_regular_hadamard_matrix_kronecker(dim: int, device: torch.device) -> torch.Tensor:
    """Generate Regular Hadamard matrix using Kronecker product.

    Args:
        dim: Dimension, must be a power of 4
        device: Target device for the tensor

    Returns:
        Hadamard matrix of shape (dim, dim)
    """
    if dim <= 0:
        raise ValueError(f"Dimension {dim} must be positive.")

    temp_dim = dim
    while temp_dim > 1:
        if temp_dim % 4 != 0:
            raise ValueError(f"Dimension {dim} must be a power of 4 for Regular Hadamard Kronecker construction.")
        temp_dim //= 4

    H4 = torch.tensor(
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
        H = H4 / 2.0
        return H

    H = H4
    curr_dim = 4
    while curr_dim < dim:
        H = torch.kron(H, H4)
        curr_dim *= 4

    H = H / (dim ** 0.5)
    return H


def get_walsh_hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Generate a normalized Walsh-Hadamard matrix using Sylvester construction."""
    if dim <= 0:
        raise ValueError(f"Dimension {dim} must be positive.")
    if (dim & (dim - 1)) != 0:
        raise ValueError(f"Dimension {dim} must be a power of 2 for Walsh-Hadamard matrix.")

    if dim == 1:
        return torch.ones((1, 1), dtype=torch.float32, device=device)

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32, device=device)
    curr_dim = 2
    while curr_dim < dim:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
        curr_dim *= 2

    return H / (dim ** 0.5)


def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform over the last dimension.

    This avoids materializing the full Hadamard matrix, which is prohibitively
    expensive for QuaRot on FLUX MLP layers.
    """
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


@torch.no_grad()
def register_rotate_hook(module, H):
    """Register rotation hook for a Linear module.

    Args:
        module: nn.Linear module
        H: Hadamard rotation matrix
    """
    H = H.to(module.weight.device).to(module.weight.dtype)
    in_features = module.weight.shape[1]
    group_size = H.shape[0]
    num_groups = in_features // group_size

    # Reshape weight and apply rotation
    weight = module.weight.data.reshape(module.weight.shape[0], num_groups, group_size)
    weight = torch.matmul(weight, H)
    module.weight.data = weight.reshape(module.weight.shape[0], in_features)

    def pre_hook_fn(m, args):
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.shape[-1] != in_features:
            return args
        H_local = H
        if H_local.device != x.device:
            H_local = H_local.to(x.device)
        if H_local.dtype != x.dtype:
            H_local = H_local.to(x.dtype)
        x_r = x.reshape(*x.shape[:-1], num_groups, group_size)
        x_r = torch.matmul(x_r, H_local)
        x_r = x_r.reshape(*x.shape[:-1], in_features)
        return (x_r, *args[1:])

    module.register_forward_pre_hook(pre_hook_fn)


@torch.no_grad()
def register_quarot_hook(module, original_in_features: int, padded_dim: int):
    """Register QuaRot hook for a Linear module.

    QuaRot applies a full-dimension Walsh-Hadamard rotation. If the input
    dimension is not a power of 2, both weight and activation are padded to the
    next power of 2 before rotation.
    """
    if original_in_features < padded_dim:
        pad_size = padded_dim - original_in_features
        weight = torch.nn.functional.pad(module.weight.data, (0, pad_size))
    else:
        weight = module.weight.data

    module.weight.data = fast_walsh_hadamard_transform(weight)
    module.in_features = padded_dim

    def pre_hook_fn(m, args):
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.shape[-1] != original_in_features:
            return args

        if original_in_features < padded_dim:
            x = torch.nn.functional.pad(x, (0, padded_dim - original_in_features))

        x = fast_walsh_hadamard_transform(x)
        return (x, *args[1:])

    module.register_forward_pre_hook(pre_hook_fn)


@torch.no_grad()
def rotate_flux(model, rotate_group_size=128, rotation_type="convrot"):
    """Apply Hadamard rotation to FLUX transformer model.

    This function applies Hadamard rotation to target Linear layers. For
    ``convrot`` it uses grouped regular Hadamard matrices. For ``quarot`` it
    uses full-dimension Walsh-Hadamard rotation with padding to the next power
    of 2 when needed.

    Args:
        model: FLUX transformer model (Flux or the transformer component)
        rotate_group_size: Group size for grouped rotation (convrot only)
        rotation_type: "convrot" or "quarot"

    Returns:
        model: The model with rotation applied
    """
    # Handle wrapped model
    if hasattr(model, "transformer"):
        model = model.transformer

    # FLUX target modules for rotation
    target_modules = [
        "img_attn.qkv", "img_attn.proj",
        "txt_attn.qkv", "txt_attn.proj",
        "img_mlp.0", "img_mlp.2",
        "txt_mlp.0", "txt_mlp.2",
        "linear1", "linear2",
    ]

    print(f"Applying {rotation_type.upper()} Hadamard rotation...")
    count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue

        in_features = module.weight.shape[1]
        if rotation_type == "convrot":
            if in_features % rotate_group_size != 0:
                continue
            H = get_regular_hadamard_matrix_kronecker(rotate_group_size, module.weight.device)
            register_rotate_hook(module, H)
        elif rotation_type == "quarot":
            padded_dim = next_power_of_2(in_features)
            register_quarot_hook(module, in_features, padded_dim)
        else:
            raise ValueError(f"Unknown rotation_type: {rotation_type}. Use 'convrot' or 'quarot'.")
        count += 1

    print(f"Applied rotation to {count} modules.")
    return model


def quant_flux_convrot(model, quant_config=None):
    """Hadamard-rotation quantization function for FLUX model.

    Applies either ConvRot or QuaRot followed by quantization.

    Args:
        model: FLUX transformer model
        quant_config: Quantization configuration dict, containing:
            - weight_quant: Whether to quantize weights (default: True)
            - weight_bits: Bits for weight quantization (default: 4)
            - weight_quant_type: 'per_tensor', 'per_channel', 'per_group' (default: 'per_group')
            - weight_group_size: Group size for per-group quantization (default: 128)
            - weight_sym: Symmetric quantization for weights (default: True)
            - act_quant: Whether to quantize activations (default: False)
            - act_bits: Bits for activation quantization (default: 8)
            - act_quant_type: Quantization type for activations (default: 'per_token')
            - act_group_size: Group size for activation per-group quantization (default: None)
            - act_sym: Symmetric quantization for activations (default: True)
            - rotate_group_size: Group size for Hadamard rotation (default: 128, convrot only)
            - rotation_type: 'convrot' or 'quarot' (default: 'convrot')

    Returns:
        model: The model with rotation applied and quantized weights/activations
    """
    if quant_config is None:
        quant_config = {}

    rotate_group_size = quant_config.get("rotate_group_size", 64)
    rotation_type = quant_config.get("rotation_type", "convrot")

    # Apply rotation
    print(f"\n[Step 1] Applying {rotation_type.upper()} Hadamard Rotation...")
    rotate_flux(model, rotate_group_size=rotate_group_size, rotation_type=rotation_type)

    # Apply weight quantization (in-place)
    print("\n[Step 2] Quantizing weights...")
    quantize_flux_weight(model, quant_config)

    # Apply activation quantization (register hooks)
    print("\n[Step 3] Quantizing activations...")
    quantize_flux_act(model, quant_config)

    return model
