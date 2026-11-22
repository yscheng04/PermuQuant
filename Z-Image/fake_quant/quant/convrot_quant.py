"""ConvRot/QuaRot Hadamard-rotation quantization for Z-Image model."""

import math
import os
import torch
import torch.nn as nn
from .quantizer import quantize_zimage_weight, quantize_zimage_act


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
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    if dim == 4:
        H = H4 / 2.0
        signs = (torch.randint(0, 2, (dim,), device=device, dtype=H.dtype) * 2 - 1)
        H = H * signs.unsqueeze(0)
        return H

    H = H4
    curr_dim = 4
    while curr_dim < dim:
        H = torch.kron(H, H4)
        curr_dim *= 4

    H = H / (dim ** 0.5)
    signs = (torch.randint(0, 2, (dim,), device=device, dtype=H.dtype) * 2 - 1)
    H = H * signs.unsqueeze(0)
    return H


def get_walsh_hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Generate Walsh-Hadamard matrix using Sylvester construction.

    Args:
        dim: Dimension, must be a power of 2
        device: Target device for the tensor

    Returns:
        Hadamard matrix of shape (dim, dim), normalized by 1/sqrt(dim)
    """
    if dim <= 0:
        raise ValueError(f"Dimension {dim} must be positive.")

    # Check if dim is a power of 2
    if (dim & (dim - 1)) != 0:
        raise ValueError(f"Dimension {dim} must be a power of 2 for Walsh-Hadamard matrix.")

    # Base case: H_2 = [[1, 1], [1, -1]]
    if dim == 1:
        return torch.ones((1, 1), dtype=torch.float32, device=device)

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32, device=device)

    if dim == 2:
        H = H / (2 ** 0.5)
        return H

    # Build H_n recursively using Sylvester construction
    curr_dim = 2
    while curr_dim < dim:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
        curr_dim *= 2

    H = H / (dim ** 0.5)
    return H


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    if (n & (n - 1)) == 0:
        return n
    return 1 << (n - 1).bit_length()


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
    weight = module.weight.data.view(module.weight.shape[0], num_groups, group_size)
    weight = torch.matmul(weight, H)
    module.weight.data = weight.view(module.weight.shape[0], in_features)

    def pre_hook_fn(m, args):
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.shape[-1] != in_features:
            return args
        H_local = H
        if H_local.device != x.device:
            H_local = H_local.to(x.device)
        if H_local.dtype != x.dtype:
            H_local = H_local.to(x.dtype)
        x_r = x.view(*x.shape[:-1], num_groups, group_size)
        x_r = torch.matmul(x_r, H_local)
        x_r = x_r.view(*x.shape[:-1], in_features)
        return (x_r, *args[1:])

    handle = module.register_forward_pre_hook(pre_hook_fn)
    module._rotate_hook_id = handle.id


@torch.no_grad()
def register_quarot_hook(module, original_in_features, padded_dim):
    """Register QuaRot rotation hook for a Linear module.

    QuaRot uses full-dimension Walsh-Hadamard rotation. If the input dimension
    is not a power of 2, it pads the input to the next power of 2. The transform
    is computed with FWHT instead of materializing a dense H matrix.

    Args:
        module: nn.Linear module
        original_in_features: Original input dimension (may not be power of 2)
        padded_dim: Padded dimension (power of 2)
    """
    # Pad weight matrix if necessary
    if original_in_features < padded_dim:
        # Weight shape: (out_features, in_features)
        # Pad to (out_features, padded_dim)
        pad_size = padded_dim - original_in_features
        weight_padded = torch.nn.functional.pad(module.weight.data, (0, pad_size))
    else:
        weight_padded = module.weight.data

    # Apply rotation without storing H: W_rot = W @ H, and Walsh-Hadamard is symmetric.
    weight_rot = fast_walsh_hadamard_transform(weight_padded)

    # Note: We keep the padded dimension to maintain equivariance
    # Both weight and activation remain at padded_dim
    module.weight.data = weight_rot
    module.in_features = padded_dim

    # Store original and padded dimensions for hook
    module._quarot_original_in_features = original_in_features
    module._quarot_padded_dim = padded_dim

    def pre_hook_fn(m, args):
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.shape[-1] != original_in_features:
            return args

        # Pad activation if necessary
        if original_in_features < padded_dim:
            pad_size = padded_dim - original_in_features
            x_padded = torch.nn.functional.pad(x, (0, pad_size))
        else:
            x_padded = x

        # Note: Keep padded dimension to match rotated weight shape
        x_rot = fast_walsh_hadamard_transform(x_padded)

        return (x_rot, *args[1:])

    handle = module.register_forward_pre_hook(pre_hook_fn)
    module._quarot_hook_id = handle.id


@torch.no_grad()
def rotate_zimage(model, rotate_cache=None, rotate_group_size=128, rotation_type="convrot"):
    """Apply Hadamard rotation to Z-Image transformer model.

    This function applies Hadamard rotation to target Linear layers.
    The rotation is fused into weights directly, and forward hooks are
    registered to rotate activations.

    Args:
        model: ZImageTransformer2DModel or the transformer component
        rotate_cache: Cache of rotation matrices (if None, will generate)
        rotate_group_size: Group size for rotation (must be power of 4, used for convrot only)
        rotation_type: "convrot" for grouped Regular Hadamard, "quarot" for full-dimension Walsh-Hadamard

    Returns:
        rotate_cache: Dictionary of rotation matrices and metadata
    """
    # Handle wrapped model
    if hasattr(model, "transformer"):
        model = model.transformer

    target_modules = [
        "w1", "w2", "w3",                   # Feed-forward gate and up proj
        "to_q", "to_k", "to_v",       # Attention QKV projections
        "to_out.0",                   # Attention output projection
    ]

    # Skip refiner layers - only apply rotation to main transformer layers
    skip_prefixes = ["noise_refiner", "context_refiner"]

    if rotation_type == "quarot" and rotate_cache is not None:
        print("Ignoring provided QuaRot cache and regenerating deterministic rotation metadata.")
        rotate_cache = None

    if rotate_cache is None:
        rotate_cache = {}
        rotate_cache['_metadata'] = {'rotation_type': rotation_type}

        if rotation_type == "convrot":
            print(f"Generating ConvRot Hadamard rotation matrices (group_size={rotate_group_size})...")

            for name, module in model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                # Skip refiner layers
                if any(name.startswith(p) for p in skip_prefixes):
                    continue
                if not any(name.endswith(t) for t in target_modules):
                    continue

                in_features = module.weight.shape[1]
                if in_features % rotate_group_size != 0:
                    continue

                # Generate Hadamard rotation matrix
                H = get_regular_hadamard_matrix_kronecker(rotate_group_size, module.weight.device)
                rotate_cache[name] = H.cpu()

                # Apply rotation and register hook
                register_rotate_hook(module, H)

        elif rotation_type == "quarot":
            print("Applying QuaRot Walsh-Hadamard rotation with FWHT...")

            for name, module in model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                # Skip refiner layers
                if any(name.startswith(p) for p in skip_prefixes):
                    continue
                if not any(name.endswith(t) for t in target_modules):
                    continue

                in_features = module.weight.shape[1]
                original_in_features = in_features

                # Compute padded dimension (next power of 2)
                padded_dim = next_power_of_2(in_features)

                rotate_cache[name] = {
                    'original_in_features': original_in_features,
                    'padded_dim': padded_dim
                }

                # Apply rotation and register hook with padding support.
                register_quarot_hook(module, original_in_features, padded_dim)

        else:
            raise ValueError(f"Unknown rotation_type: {rotation_type}. Use 'convrot' or 'quarot'.")

        print(f"Generated rotation matrices for {len(rotate_cache) - 1} modules.")  # -1 for _metadata
        return rotate_cache

    else:
        # Loading from cache
        cached_rotation_type = rotate_cache.get('_metadata', {}).get('rotation_type', 'convrot')
        print(f"Applying rotation matrices from cache ({len(rotate_cache) - 1} entries, type={cached_rotation_type})...")

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Skip refiner layers
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            if not any(name.endswith(t) for t in target_modules):
                continue

            if name in rotate_cache and name != '_metadata':
                cache_entry = rotate_cache[name]

                if cached_rotation_type == "convrot":
                    # ConvRot cache stores H directly
                    H = cache_entry
                    register_rotate_hook(module, H)
                elif cached_rotation_type == "quarot":
                    # QuaRot cache may be an old dict with H, but FWHT only needs metadata.
                    original_in_features = cache_entry['original_in_features']
                    padded_dim = cache_entry['padded_dim']
                    register_quarot_hook(module, original_in_features, padded_dim)

        return rotate_cache


def quant_zimage_convrot(model, quant_config=None, rotate_cache_path=None):
    """ConvRot/QuaRot quantization function for Z-Image model.

    Applies Hadamard rotation followed by quantization.

    Args:
        model: Z-Image transformer model
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
            - rotate_group_size: Group size for Hadamard rotation (default: 128, for convrot only)
            - rotation_type: 'convrot' or 'quarot' (default: 'convrot')
            - save_rotate_cache: Whether to load/save generated rotation cache (default: False).
              QuaRot ignores cache loading because its transform is deterministic.
        rotate_cache_path: Path to save/load rotation cache

    Returns:
        model: The model with rotation applied and quantized weights/activations
    """
    if quant_config is None:
        quant_config = {}

    rotate_group_size = quant_config.get('rotate_group_size', 128)
    rotation_type = quant_config.get('rotation_type', 'convrot')
    save_rotate_cache = quant_config.get('save_rotate_cache', False)

    # Apply rotation
    print(f"\n[Step 1] Applying {rotation_type.upper()} Hadamard Rotation...")
    rotate_cache = None
    should_load_rotate_cache = rotation_type == "convrot"
    if should_load_rotate_cache and save_rotate_cache and rotate_cache_path and os.path.exists(rotate_cache_path):
        print(f"Loading rotation cache from {rotate_cache_path}...")
        rotate_cache = torch.load(rotate_cache_path)
    else:
        print("Generating rotation without loading a saved cache...")

    rotate_cache = rotate_zimage(model, rotate_cache, rotate_group_size=rotate_group_size, rotation_type=rotation_type)

    if save_rotate_cache and rotate_cache_path and not os.path.exists(rotate_cache_path):
        os.makedirs(os.path.dirname(rotate_cache_path), exist_ok=True)
        torch.save(rotate_cache, rotate_cache_path)
        print(f"Saved rotation cache to {rotate_cache_path}")

    # Apply weight quantization (in-place)
    print("\n[Step 2] Quantizing weights...")
    quantize_zimage_weight(model, quant_config)

    # Apply activation quantization (register hooks)
    print("\n[Step 3] Quantizing activations...")
    quantize_zimage_act(model, quant_config)

    return model
