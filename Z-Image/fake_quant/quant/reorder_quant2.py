"""Reorder Quantization v2 for Z-Image model.

This quantization method reorders activation and weight channels based on
combined channel importance from both activation and weight tensors.
The importance is computed as: act_importance^alpha * weight_importance^(1-alpha)
Alpha values are enumerated to find the best reordering that minimizes quantization error.
The "rr" metric uses absmax importance with round-robin placement across quant groups,
and "random" uses a random channel permutation as a baseline.
"""

import os
import torch
import torch.nn as nn
import gc

from .convrot_quant import rotate_zimage
from .quantizer import quantize_tensor, quantize_zimage_act, quantize_zimage_weight
from fake_quant.calib.calib_cache_loader import CalibCacheLoader


IMPORTANCE_METRICS = ("absmax", "variance", "second_moment")
REORDER_METRICS = IMPORTANCE_METRICS + ("rr", "random")


def _format_metric_options():
    return "', '".join(REORDER_METRICS)


@torch.no_grad()
def compute_channel_importance(X_list, metric="absmax"):
    """Compute channel importance from activation tensors.

    Args:
        X_list: List of activation tensors, each of shape [*, in_features]
        metric: "absmax", "variance", or "second_moment"

    Returns:
        importance: Tensor of shape [in_features] with importance per channel
    """
    if metric == "absmax":
        # Compute max abs per channel incrementally
        max_val = None
        for X in X_list:
            X_flat = X.flatten(0, -2)
            local_max = X_flat.abs().max(dim=0).values
            if max_val is None:
                max_val = local_max
            else:
                max_val = torch.maximum(max_val, local_max)
        return max_val
    elif metric == "variance":
        # Compute variance using Welford's online algorithm
        mean = None
        M2 = None
        count = 0

        for X in X_list:
            X_flat = X.flatten(0, -2)
            n = X_flat.shape[0]

            if mean is None:
                mean = X_flat.mean(dim=0)
                M2 = ((X_flat - mean.unsqueeze(0)) ** 2).sum(dim=0)
                count = n
            else:
                # Welford's algorithm for incremental variance
                delta = X_flat - mean.unsqueeze(0)
                count_new = count + n
                mean = mean + delta.mean(dim=0) * n / count_new
                delta2 = X_flat - mean.unsqueeze(0)
                M2 = M2 + (delta * delta2).sum(dim=0)
                count = count_new

        variance = M2 / count if count > 0 else M2
        return variance
    elif metric == "second_moment":
        # Compute E[x^2] incrementally
        sum_sq = None
        count = 0

        for X in X_list:
            X_flat = X.flatten(0, -2)
            local_sum_sq = (X_flat ** 2).sum(dim=0)
            if sum_sq is None:
                sum_sq = local_sum_sq
            else:
                sum_sq = sum_sq + local_sum_sq
            count += X_flat.shape[0]

        return sum_sq / count if count > 0 else sum_sq
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Use 'absmax', 'variance', or 'second_moment'."
        )


@torch.no_grad()
def compute_weight_channel_importance(W, metric="absmax"):
    """Compute channel importance from weight tensor.

    Args:
        W: Weight tensor of shape [out_features, in_features]
        metric: "absmax", "variance", or "second_moment"

    Returns:
        importance: Tensor of shape [in_features] with importance per channel
    """
    if metric == "absmax":
        return W.abs().max(dim=0).values
    elif metric == "variance":
        return W.var(dim=0)
    elif metric == "second_moment":
        return (W ** 2).mean(dim=0)
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Use 'absmax', 'variance', or 'second_moment'."
        )


def build_round_robin_indices(sorted_indices, group_size):
    """Spread sorted channels across quantization groups in round-robin order."""
    num_channels = sorted_indices.numel()
    if group_size is None or group_size <= 0 or group_size >= num_channels:
        return sorted_indices

    num_groups = (num_channels + group_size - 1) // group_size
    groups = [[] for _ in range(num_groups)]
    for rank, idx in enumerate(sorted_indices.tolist()):
        groups[rank % num_groups].append(idx)

    return torch.tensor(
        [idx for group in groups for idx in group],
        dtype=sorted_indices.dtype,
        device=sorted_indices.device,
    )


def get_reorder_group_size(quant_config):
    """Pick the group size that defines contiguous channels after reorder."""
    if quant_config.get("weight_quant_type", "per_group") in {"per_group", "nvfp4", "nvfp4_1d", "nvfp4_2d"}:
        return quant_config.get("weight_group_size", 128)
    if quant_config.get("act_quant_type", "per_token") in {"per_group", "nvfp4", "nvfp4_1d", "nvfp4_2d"}:
        return quant_config.get("act_group_size", None)
    return None


def enumerate_alphas(alpha_start=0.0, alpha_end=1.0, alpha_step=0.2):
    """Enumerate alpha values from start to end with inclusive stepped endpoint."""
    if alpha_step <= 0:
        raise ValueError(f"alpha_step must be positive, got {alpha_step}.")
    if alpha_start > alpha_end:
        raise ValueError(
            f"alpha_start must be <= alpha_end, got alpha_start={alpha_start}, alpha_end={alpha_end}."
        )

    eps = 1e-9
    alphas = []
    alpha = float(alpha_start)
    alpha_end = float(alpha_end)
    alpha_step = float(alpha_step)

    while alpha <= alpha_end + eps:
        alphas.append(alpha_end if abs(alpha - alpha_end) <= eps else alpha)
        alpha += alpha_step

    if not alphas:
        raise ValueError(
            f"No alpha values enumerated for alpha_start={alpha_start}, alpha_end={alpha_end}, alpha_step={alpha_step}."
        )

    return alphas


def iter_reorder_calib_layers(model, loader, needs_inputs_fn, layerwise=True):
    """Yield calibration caches with bounded CPU memory.

    CalibCacheLoader caches layer inputs before collecting per-module activations.
    Passing the whole ModuleList makes it keep every transformer block input at
    once, which is very expensive for image-token calibration. Layerwise mode
    replays calibration per block and keeps only one block's tensors live.
    """
    layers = getattr(model, "layers", None)
    if not layerwise or layers is None:
        yield from loader.iter_layer_activations(model, needs_inputs_fn=needs_inputs_fn)
        return

    for layer in layers:
        yield from loader.iter_layer_activations(
            model,
            needs_inputs_fn=needs_inputs_fn,
            layers=[layer],
        )


@torch.no_grad()
def compute_reorder_quant_error(X_list, W, reorder_indices, quant_config, batch_size=4096):
    """Compute quantization error after reordering.

    Args:
        X_list: List of input activation tensors
        W: Weight tensor (out_features, in_features)
        reorder_indices: Reordering indices for input channels
        quant_config: Quantization configuration
        batch_size: Batch size for GPU computation

    Returns:
        mean_error: Mean L1 error per output element
    """
    act_bits = quant_config.get("act_bits", 8)
    act_quant_type = quant_config.get("act_quant_type", "per_token")
    act_group_size = quant_config.get("act_group_size", None)
    act_sym = quant_config.get("act_sym", True)
    weight_bits = quant_config.get("weight_bits", 4)
    weight_quant_type = quant_config.get("weight_quant_type", "per_group")
    weight_group_size = quant_config.get("weight_group_size", 128)
    weight_sym = quant_config.get("weight_sym", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reorder weight
    W_reordered = W[:, reorder_indices]

    # Move weight to GPU and quantize
    W_reordered = W_reordered.to(device)
    W_q = quantize_tensor(W_reordered, n_bits=weight_bits, group_size=weight_group_size,
                          quant_type=weight_quant_type, sym=weight_sym)
    W_q_t = W_q.T
    W_t = W_reordered.T

    # Flatten all X tensors on CPU
    X_flat_list = [x.flatten(0, -2) for x in X_list]
    total_samples = sum(x.shape[0] for x in X_flat_list)
    if total_samples == 0:
        return 0.0

    # Process in batches
    l1_error_sum = 0.0
    sample_count = 0
    current_tensor_idx = 0
    current_offset = 0

    for batch_start in range(0, total_samples, batch_size):
        batch_samples = []
        batch_remaining = batch_size

        while batch_remaining > 0 and current_tensor_idx < len(X_flat_list):
            X_tensor = X_flat_list[current_tensor_idx]
            remaining_in_tensor = X_tensor.shape[0] - current_offset
            take = min(batch_remaining, remaining_in_tensor)
            batch_samples.append(X_tensor[current_offset:current_offset + take])

            current_offset += take
            batch_remaining -= take

            if current_offset >= X_tensor.shape[0]:
                current_tensor_idx += 1
                current_offset = 0

        if not batch_samples:
            break

        # Move batch to GPU and reorder
        X_batch = torch.cat(batch_samples, dim=0).to(device)
        X_batch = X_batch[:, reorder_indices]  # Reorder input channels
        batch_actual_size = X_batch.shape[0]

        # Compute original output
        Y_orig = X_batch @ W_t

        # Compute quantized output
        X_q = quantize_tensor(X_batch, n_bits=act_bits, group_size=act_group_size,
                              quant_type=act_quant_type, sym=act_sym)
        Y_q = X_q @ W_q_t

        # Accumulate L1 error
        l1_error_sum += (Y_orig - Y_q).abs().sum().item()
        sample_count += batch_actual_size

        # Clean up
        del X_batch, Y_orig, Y_q, X_q
        if batch_start % (batch_size * 4) == 0:
            torch.cuda.empty_cache()

    mean_l1_error = l1_error_sum / (sample_count * W.shape[0]) if sample_count > 0 else 0.0

    # Cleanup
    del W_reordered, W_q, W_q_t, W_t
    torch.cuda.empty_cache()
    gc.collect()

    return mean_l1_error


@torch.no_grad()
def compute_quant_error(X_list, W, quant_config, batch_size=4096):
    """Compute quantization error without reordering.

    Args:
        X_list: List of input activation tensors
        W: Weight tensor (out_features, in_features)
        quant_config: Quantization configuration
        batch_size: Batch size for GPU computation

    Returns:
        mean_error: Mean L1 error per output element
    """
    return compute_reorder_quant_error(X_list, W, torch.arange(W.shape[1]), quant_config, batch_size)


@torch.no_grad()
def find_and_validate_reorder(X_list, W, quant_config, metric="absmax",
                              alpha_step=0.2, alpha_start=0.0, alpha_end=1.0, batch_size=4096,
                              threshold=0.0, log_print=print):
    """Find best reorder indices by enumerating alpha values.

    For importance metrics, each alpha in [alpha_start, alpha_start + step, ...]:
    1. Compute combined importance = act^alpha * weight^(1-alpha)
    2. Sort channels by importance, or round-robin the absmax order for metric="rr"
    3. Compute quantization error
    4. Select alpha with minimum error

    Args:
        X_list: List of input activation tensors
        W: Weight tensor (out_features, in_features)
        quant_config: Quantization configuration
        metric: "absmax", "variance", "second_moment", "rr", or "random"
        alpha_step: Step size for alpha enumeration (default: 0.2)
        alpha_start: Start of alpha enumeration range (default: 0.0)
        alpha_end: End of alpha enumeration range (default: 1.0)
        batch_size: Batch size for GPU computation
        threshold: Minimum relative error reduction to accept reorder
        log_print: Logging function

    Returns:
        reorder_indices: Tensor of reorder indices if improves error, else None
        info: Dict with error_before, error_after, improvement_ratio, relative_reduction, best_alpha
    """
    if metric not in REORDER_METRICS:
        raise ValueError(f"Unknown metric: {metric}. Use '{_format_metric_options()}'.")

    # Compute base error (without reordering)
    error_before = compute_quant_error(X_list, W, quant_config, batch_size=batch_size)

    if metric == "random":
        generator = None
        seed = quant_config.get("reorder_random_seed", None)
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        reorder_indices = torch.randperm(W.shape[1], generator=generator)
        original_indices = torch.arange(W.shape[1])
        if torch.equal(reorder_indices, original_indices):
            return None, {
                "error_before": error_before,
                "error_after": error_before,
                "improvement_ratio": 1.0,
                "relative_reduction": 0.0,
                "best_alpha": 0.0,
            }

        error_after = compute_reorder_quant_error(
            X_list, W, reorder_indices, quant_config, batch_size=batch_size
        )
        log_print(f"    random: error={error_after:.6f}")

        improvement_ratio = error_before / (error_after + 1e-10)
        relative_reduction = (error_before - error_after) / (error_before + 1e-10)
        info = {
            "error_before": error_before,
            "error_after": error_after,
            "improvement_ratio": improvement_ratio,
            "relative_reduction": relative_reduction,
            "best_alpha": 0.0,
        }
        if relative_reduction >= threshold:
            return reorder_indices, info
        return None, info

    # Pre-compute activation and weight importance (cached for all alpha values)
    importance_metric = "absmax" if metric == "rr" else metric
    act_importance = compute_channel_importance(X_list, metric=importance_metric)
    weight_importance = compute_weight_channel_importance(W, metric=importance_metric)

    # Ensure both are on CPU to avoid device mismatch
    act_importance = act_importance.cpu()
    weight_importance = weight_importance.cpu()

    # Normalize to same scale
    act_importance = act_importance / (act_importance.max() + 1e-10)
    weight_importance = weight_importance / (weight_importance.max() + 1e-10)

    # Enumerate alpha values
    alphas = enumerate_alphas(alpha_start=alpha_start, alpha_end=alpha_end, alpha_step=alpha_step)

    best_indices = None
    best_error = error_before
    best_alpha = alphas[0]
    group_size = get_reorder_group_size(quant_config)

    for alpha in alphas:
        # Compute combined importance using cached values
        combined = (act_importance ** alpha) * (weight_importance ** (1 - alpha))
        sorted_indices = torch.argsort(combined, descending=True)
        if metric == "rr":
            reorder_indices = build_round_robin_indices(sorted_indices, group_size)
        else:
            reorder_indices = sorted_indices

        # Skip if reorder is identity
        original_indices = torch.arange(W.shape[1])
        if torch.equal(reorder_indices, original_indices):
            continue

        # Compute error
        error_after = compute_reorder_quant_error(
            X_list, W, reorder_indices, quant_config, batch_size=batch_size
        )

        if metric == "rr":
            log_print(
                f"    alpha={alpha:.2f}, rr_group={group_size}: "
                f"error={error_after:.6f}"
            )
        else:
            log_print(f"    alpha={alpha:.2f}: error={error_after:.6f}")

        # Track best
        if error_after < best_error:
            best_error = error_after
            best_indices = reorder_indices
            best_alpha = alpha

    # Check if best improvement meets threshold
    relative_reduction = (error_before - best_error) / (error_before + 1e-10)
    should_apply = relative_reduction >= threshold

    info = {
        "error_before": error_before,
        "error_after": best_error,
        "improvement_ratio": error_before / (best_error + 1e-10),
        "relative_reduction": relative_reduction,
        "best_alpha": best_alpha
    }

    if should_apply and best_indices is not None:
        return best_indices, info
    else:
        return None, info


class ReorderHook:
    """Forward pre-hook to reorder input activations."""

    def __init__(self, reorder_indices):
        """Initialize with reorder indices.

        Args:
            reorder_indices: Tensor of indices to reorder input channels
        """
        self.reorder_indices = reorder_indices

    def __call__(self, module, args):
        """Reorder input activations.

        Args:
            module: The nn.Linear module
            args: Tuple of input tensors

        Returns:
            Tuple with reordered input tensor
        """
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return args

        # Move indices to same device as input
        indices = self.reorder_indices
        if indices.device != x.device:
            indices = indices.to(x.device)

        # Reorder input channels (last dimension)
        x_reordered = x[..., indices]
        return (x_reordered,) + args[1:]


@torch.no_grad()
def register_reorder_hook(module, reorder_indices):
    """Register reorder hook for a Linear module.

    Args:
        module: nn.Linear module
        reorder_indices: Reordering indices for input channels
    """
    hook = ReorderHook(reorder_indices)
    handle = module.register_forward_pre_hook(hook)
    module._reorder_hook_id = handle.id


@torch.no_grad()
def reorder_zimage(model, quant_config=None, reorder_cache=None, reorder_cache_path=None):
    """Apply reorder quantization to Z-Image transformer model.

    Args:
        model: ZImageTransformer2DModel or the transformer component
        quant_config: Quantization configuration dict
        reorder_cache: Cache of reorder indices (if None, will compute)
        reorder_cache_path: Path to save reorder cache

    Returns:
        reorder_cache: Dictionary of reorder info for each layer
    """
    if quant_config is None:
        quant_config = {}

    # Handle wrapped model
    if hasattr(model, "transformer"):
        model = model.transformer

    # Configuration
    metric = quant_config.get("reorder_metric", "absmax")
    alpha_step = quant_config.get("alpha_step", 0.2)
    alpha_start = quant_config.get("alpha_start", 0.0)
    alpha_end = quant_config.get("alpha_end", 1.0)
    threshold = quant_config.get("reorder_threshold", 0.0)
    batch_size = quant_config.get("reorder_batch_size", 4096)
    calib_ratio = quant_config.get("calib_ratio", 1.0)
    layerwise_calib = quant_config.get("reorder_layerwise_calib", True)

    target_modules = [
        "w1", "w2", "w3",
        "to_q", "to_k", "to_v",
        "to_out.0",
    ]

    skip_prefixes = ["noise_refiner", "context_refiner"]

    if reorder_cache is None:
        # Calibration mode: compute reorder indices
        reorder_cache = {}
        loader = CalibCacheLoader(ratio=calib_ratio)

        def needs_inputs_fn(name, module):
            """Identify modules that need input activation caching."""
            return any(name.endswith(k) for k in target_modules)

        # Setup logger
        log_file = None
        if reorder_cache_path:
            log_path = reorder_cache_path.replace(".pt", ".log").replace(".pth", ".log")
            log_file = open(log_path, "w")

        def log_print(msg):
            """Print to console and write to log file."""
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()

        log_print(
            f"Calibrating reorder indices (metric={metric}, alpha_start={alpha_start}, "
            f"alpha_end={alpha_end}, alpha_step={alpha_step}, "
            f"calib_ratio={calib_ratio}, layerwise_calib={layerwise_calib})..."
        )

        # Iterate through layers to collect activations
        for layer_name, (layer, cache, _) in iter_reorder_calib_layers(
            model,
            loader,
            needs_inputs_fn=needs_inputs_fn,
            layerwise=layerwise_calib,
        ):
            # Process each Linear module in this layer
            for module_name, module_cache in cache.items():
                # Extract activation data from cache
                X_list = None

                if module_cache.inputs is None:
                    continue

                # Handle different cache structures
                if hasattr(module_cache.inputs, 'data') and isinstance(module_cache.inputs.data, list):
                    X_list = module_cache.inputs.data
                elif hasattr(module_cache.inputs, 'tensors') and isinstance(module_cache.inputs.tensors, dict):
                    all_data = []
                    for tc in module_cache.inputs.tensors.values():
                        if hasattr(tc, 'data') and isinstance(tc.data, list):
                            all_data.extend(tc.data)
                    X_list = all_data if all_data else None
                elif hasattr(module_cache.inputs, 'values'):
                    X_list = []
                    for tc in module_cache.inputs.values():
                        if hasattr(tc, 'data') and isinstance(tc.data, list):
                            X_list.extend(tc.data)
                    if len(X_list) == 0:
                        X_list = None

                if X_list is None or len(X_list) == 0:
                    continue

                # Get the actual module
                named_modules = dict(layer.named_modules())
                if module_name.startswith(layer_name + "."):
                    relative_name = module_name[len(layer_name) + 1:]
                else:
                    relative_name = module_name
                module = named_modules.get(relative_name, None)
                if module is None or not isinstance(module, nn.Linear):
                    continue

                # Skip refiner layers
                # Avoid duplicate layer names: module_name may already include layer_name as prefix
                if module_name.startswith(layer_name + "."):
                    full_name = module_name
                else:
                    full_name = f"{layer_name}.{module_name}"
                if any(p in full_name for p in skip_prefixes):
                    continue

                log_print(f"Processing {full_name}...")

                # Get weight
                W = module.weight.data

                # Find best reorder with alpha enumeration, except random which has one candidate.
                reorder_indices, info = find_and_validate_reorder(
                    X_list, W, quant_config, metric=metric, alpha_step=alpha_step,
                    alpha_start=alpha_start, alpha_end=alpha_end,
                    batch_size=batch_size, threshold=threshold, log_print=log_print
                )

                log_print(f"  Error before: {info['error_before']:.6f}, after: {info['error_after']:.6f}, "
                          f"reduction: {info['relative_reduction']:.2%}, best_alpha: {info['best_alpha']:.2f}")

                # Store in cache
                reorder_cache[full_name] = reorder_indices.cpu() if reorder_indices is not None else None

                # Apply reorder if it reduces error
                if reorder_indices is not None:
                    module.weight.data = W[:, reorder_indices]
                    register_reorder_hook(module, reorder_indices)

                del X_list, W, reorder_indices, info
                gc.collect()

            # Clear layer cache to free memory
            for module_cache in cache.values():
                module_cache.clear()
            del cache
            gc.collect()
            torch.cuda.empty_cache()

        log_print(f"Calibrated reorder indices for {len(reorder_cache)} modules.")

        # Close log file
        if log_file:
            log_file.close()

        return reorder_cache

    else:
        # Apply mode: use cached reorder indices
        print(f"Applying reorder from cache ({len(reorder_cache)} entries)...")

        # Normalize cache keys to handle duplicates (e.g., "layers.0.layers.0.xxx" -> "layers.0.xxx")
        normalized_cache = {}
        for key, value in reorder_cache.items():
            normalized_key = key
            # Fix duplicate layer names like "layers.0.layers.0.xxx" -> "layers.0.xxx"
            parts = key.split(".")
            if len(parts) >= 4:
                # Check for pattern like "layers.N.layers.N"
                for i in range(len(parts) - 2):
                    if parts[i] == "layers" and parts[i+2] == "layers" and parts[i+1] == parts[i+3]:
                        # Found duplicate: remove the second "layers.N"
                        normalized_key = ".".join(parts[:i+2] + parts[i+4:])
                        break
            normalized_cache[normalized_key] = value

        if len(normalized_cache) != len(reorder_cache):
            print(f"Normalized cache keys: {len(reorder_cache)} -> {len(normalized_cache)} entries")

        applied_count = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Skip refiner layers
            if any(p in name for p in skip_prefixes):
                continue

            if name in normalized_cache:
                reorder_indices = normalized_cache[name]

                if reorder_indices is not None:
                    W = module.weight.data
                    module.weight.data = W[:, reorder_indices.to(W.device)]
                    register_reorder_hook(module, reorder_indices.to(module.weight.device))
                    applied_count += 1

        print(f"Applied reorder to {applied_count} modules")
        return normalized_cache


def quant_zimage_reorder(
    model,
    quant_config=None,
    smooth_cache_path=None,
    reorder_cache_path=None,
    svd_cache_path=None,
    rotate_cache_path=None,
):
    """Reorder quantization function for Z-Image model.

    This function applies channel reordering based on combined activation and weight
    importance to reduce quantization error. Alpha values are enumerated to find
    the best reordering for each layer. When enabled, ConvRot-style grouped
    Hadamard rotation is applied before reorder calibration/application.

    Args:
        model: Z-Image transformer model
        quant_config: Quantization configuration dict, containing:
            - reorder_metric: 'absmax', 'variance', 'second_moment', 'rr', or 'random' (default: 'absmax')
              'rr' uses absmax importance with round-robin placement across quant groups.
              'random' uses a random channel permutation as a baseline.
            - alpha_step: Alpha step size for enumeration (default: 0.2)
            - alpha_start: Start of alpha enumeration range (default: 0.0)
            - alpha_end: End of alpha enumeration range (default: 1.0)
            - reorder_threshold: Minimum error reduction threshold (default: 0.0)
            - reorder_batch_size: Batch size for error computation (default: 4096)
            - calib_ratio: Fraction of calibration cache samples to use (default: 1.0)
            - reorder_layerwise_calib: Replay calibration one block at a time to reduce CPU memory (default: True)
            - smooth: Whether to apply SmoothQuant first (default: False)
            - hadamard: Whether to apply Hadamard rotation first (default: False)
            - smooth_alpha: SmoothQuant alpha value (default: 0.5)
            - rotate_group_size: Group size for grouped Hadamard rotation (default: 128, convrot only)
            - rotation_type: 'convrot' or 'quarot' (default: 'convrot')
            - save_rotate_cache: Whether to save generated rotation metadata/cache (default: False).
              QuaRot ignores cache loading because its transform is deterministic.
            - svd: Whether to use SVD-based weight quantization (default: False)
            - weight_rank: SVD low-rank dimension (default: 32)
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
        smooth_cache_path: Path to save/load SmoothQuant cache
        reorder_cache_path: Path to save/load reorder cache
        svd_cache_path: Path to save/load SVD quantization cache
        rotate_cache_path: Path to save/load Hadamard rotation cache

    Returns:
        model: The model with reorder quantization applied
    """
    if quant_config is None:
        quant_config = {}

    if quant_config.get("smooth", False) or quant_config.get("svd", False):
        raise ValueError("This release supports PermuQuant and PermuQuant-H only")
    smooth = quant_config.get("smooth", False)
    hadamard = quant_config.get("hadamard", False)
    rotate_group_size = quant_config.get("rotate_group_size", 128)
    rotation_type = quant_config.get("rotation_type", "convrot")
    save_rotate_cache = quant_config.get("save_rotate_cache", False)

    if smooth and hadamard:
        raise ValueError("SmoothQuant and Hadamard rotation are mutually exclusive for reorder2.")

    # Step 1: Apply SmoothQuant or Hadamard rotation if enabled
    if hadamard:
        print(f"\n[Step 1] Applying {rotation_type.upper()} Hadamard rotation before reorder...")

        rotate_cache = None
        should_load_rotate_cache = rotation_type == "convrot"
        if should_load_rotate_cache and save_rotate_cache and rotate_cache_path and os.path.exists(rotate_cache_path):
            print(f"Loading rotation cache from {rotate_cache_path}...")
            rotate_cache = torch.load(rotate_cache_path)
        else:
            print("Generating rotation without loading a saved cache...")

        rotate_cache = rotate_zimage(
            model,
            rotate_cache=rotate_cache,
            rotate_group_size=rotate_group_size,
            rotation_type=rotation_type,
        )

        if save_rotate_cache and rotate_cache_path and not os.path.exists(rotate_cache_path):
            os.makedirs(os.path.dirname(rotate_cache_path), exist_ok=True)
            torch.save(rotate_cache, rotate_cache_path)
            print(f"Saved rotation cache to {rotate_cache_path}")
    else:
        print("\n[Step 1] Skipping SmoothQuant/Hadamard (smooth=False, hadamard=False)")

    # Step 2: Apply reorder quantization
    print("\n[Step 2] Running Reorder Quantization (with alpha enumeration)...")

    reorder_cache = None
    if reorder_cache_path and os.path.exists(reorder_cache_path):
        print(f"Loading reorder cache from {reorder_cache_path}...")
        reorder_cache = torch.load(reorder_cache_path)
        reorder_cache = reorder_zimage(model, quant_config, reorder_cache=reorder_cache)
    else:
        print("Reorder cache not found. Computing reorder indices...")
        reorder_cache = reorder_zimage(model, quant_config, reorder_cache=None, reorder_cache_path=reorder_cache_path)

    if reorder_cache_path and not os.path.exists(reorder_cache_path):
        os.makedirs(os.path.dirname(reorder_cache_path), exist_ok=True)
        torch.save(reorder_cache, reorder_cache_path)
        print(f"Saved reorder cache to {reorder_cache_path}")

    # Step 3: Apply weight quantization
    use_svd = quant_config.get("svd", False)
    print("\n[Step 3] Applying weight quantization...")
    quantize_zimage_weight(model, quant_config)

    # Step 4: Apply activation quantization
    print("\n[Step 4] Applying activation quantization (if enabled)...")
    quantize_zimage_act(model, quant_config)

    return model
