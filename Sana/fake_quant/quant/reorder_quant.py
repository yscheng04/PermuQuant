"""Reorder Quantization for Sana models."""

from __future__ import annotations

import gc
import os
import typing as tp

import torch
import torch.nn as nn

from fake_quant.calib.calib_cache_loader import CalibCacheLoader

from .convrot_quant import rotate_sana
from .quantizer import (
    SANA_BLOCK_LINEAR_PATTERNS,
    _iter_sana_block_quant_modules,
    quantize_sana_act,
    quantize_sana_weight,
    quantize_tensor,
)

IMPORTANCE_METRICS = ("absmax", "variance", "second_moment")
REORDER_METRICS = IMPORTANCE_METRICS + ("rr", "random")


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "blocks"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model
    return model


def _format_metric_options():
    return "', '".join(REORDER_METRICS)


def _activation_to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return x.flatten(0, -2)


def _module_weight_to_2d(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.data
    if isinstance(module, nn.Conv2d):
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        return module.weight.data.squeeze(-1).squeeze(-1)
    raise TypeError(f"Unsupported module type: {type(module).__name__}")


def _set_module_weight_from_2d(module: nn.Module, weight_2d: torch.Tensor) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data = weight_2d.to(module.weight.device, module.weight.dtype)
        return
    if isinstance(module, nn.Conv2d):
        if module.kernel_size != (1, 1):
            raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
        module.weight.data = weight_2d.unsqueeze(-1).unsqueeze(-1).to(module.weight.device, module.weight.dtype)
        return
    raise TypeError(f"Unsupported module type: {type(module).__name__}")


def _reordered_input_for_error(x: torch.Tensor, reorder_indices: torch.Tensor) -> torch.Tensor:
    x_2d = _activation_to_2d(x)
    return x_2d[:, reorder_indices]


@torch.no_grad()
def compute_channel_importance(X_list, metric="absmax"):
    """Compute channel importance from activation tensors."""
    if metric == "absmax":
        max_val = None
        for X in X_list:
            X_flat = _activation_to_2d(X)
            local_max = X_flat.abs().max(dim=0).values
            max_val = local_max if max_val is None else torch.maximum(max_val, local_max)
        return max_val
    if metric == "variance":
        mean = None
        M2 = None
        count = 0

        for X in X_list:
            X_flat = _activation_to_2d(X)
            n = X_flat.shape[0]
            if mean is None:
                mean = X_flat.mean(dim=0)
                M2 = ((X_flat - mean.unsqueeze(0)) ** 2).sum(dim=0)
                count = n
            else:
                delta = X_flat - mean.unsqueeze(0)
                count_new = count + n
                mean = mean + delta.mean(dim=0) * n / count_new
                delta2 = X_flat - mean.unsqueeze(0)
                M2 = M2 + (delta * delta2).sum(dim=0)
                count = count_new

        return M2 / count if count > 0 else M2
    if metric == "second_moment":
        sum_sq = None
        count = 0
        for X in X_list:
            X_flat = _activation_to_2d(X)
            local_sum_sq = (X_flat**2).sum(dim=0)
            sum_sq = local_sum_sq if sum_sq is None else sum_sq + local_sum_sq
            count += X_flat.shape[0]
        return sum_sq / count if count > 0 else sum_sq

    raise ValueError(f"Unknown metric: {metric}. Use 'absmax', 'variance', or 'second_moment'.")


@torch.no_grad()
def compute_weight_channel_importance(W, metric="absmax"):
    """Compute channel importance from weight tensor."""
    if metric == "absmax":
        return W.abs().max(dim=0).values
    if metric == "variance":
        return W.var(dim=0)
    if metric == "second_moment":
        return (W**2).mean(dim=0)
    raise ValueError(f"Unknown metric: {metric}. Use 'absmax', 'variance', or 'second_moment'.")


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
    if quant_config.get("weight_quant_type", "per_group") == "per_group":
        return quant_config.get("weight_group_size", 128)
    if quant_config.get("act_quant_type", "per_token") == "per_group":
        return quant_config.get("act_group_size", None)
    return None


def _get_default_reorder_layers(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "blocks"):
        return list(model.blocks)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


def iter_reorder_calib_layers(model, loader, needs_inputs_fn, layerwise=True):
    """Yield calibration caches with bounded CPU memory."""
    layers = _get_default_reorder_layers(model)
    if not layerwise or not layers:
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
    """Compute quantization error after reordering."""
    act_bits = quant_config.get("act_bits", 8)
    act_quant_type = quant_config.get("act_quant_type", "per_token")
    act_group_size = quant_config.get("act_group_size", None)
    act_sym = quant_config.get("act_sym", True)
    weight_bits = quant_config.get("weight_bits", 4)
    weight_quant_type = quant_config.get("weight_quant_type", "per_group")
    weight_group_size = quant_config.get("weight_group_size", 128)
    weight_sym = quant_config.get("weight_sym", True)
    data_format = quant_config.get("quant_format", "int")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W_reordered = W[:, reorder_indices].to(device)
    W_q = quantize_tensor(
        W_reordered,
        n_bits=weight_bits,
        group_size=weight_group_size,
        quant_type=weight_quant_type,
        sym=weight_sym,
        data_format=data_format,
    )
    W_q_t = W_q.T
    W_t = W_reordered.T

    X_flat_list = [_reordered_input_for_error(x, reorder_indices) for x in X_list]
    total_samples = sum(x.shape[0] for x in X_flat_list)
    if total_samples == 0:
        return 0.0

    l1_error_sum = 0.0
    sample_count = 0

    if data_format == "nvfp4":
        # NVFP4 has one FP32 global scale per runtime tensor. Preserve each
        # cached module-call boundary instead of concatenating calibration
        # tensors, which would change that global scale and bias acceptance.
        for X_tensor in X_flat_list:
            X_full = X_tensor.to(device)
            X_q = quantize_tensor(
                X_full,
                n_bits=act_bits,
                group_size=act_group_size,
                quant_type=act_quant_type,
                sym=act_sym,
                data_format=data_format,
            )
            for start in range(0, X_full.shape[0], batch_size):
                X_batch = X_full[start : start + batch_size]
                X_q_batch = X_q[start : start + batch_size]
                Y_orig = X_batch @ W_t
                Y_q = X_q_batch @ W_q_t
                l1_error_sum += (Y_orig - Y_q).abs().sum().item()
                sample_count += X_batch.shape[0]
                del X_batch, X_q_batch, Y_orig, Y_q
            del X_full, X_q
        mean_l1_error = l1_error_sum / (sample_count * W.shape[0]) if sample_count > 0 else 0.0
        del W_reordered, W_q, W_q_t, W_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return mean_l1_error

    current_tensor_idx = 0
    current_offset = 0

    for batch_start in range(0, total_samples, batch_size):
        batch_samples = []
        batch_remaining = batch_size

        while batch_remaining > 0 and current_tensor_idx < len(X_flat_list):
            X_tensor = X_flat_list[current_tensor_idx]
            remaining_in_tensor = X_tensor.shape[0] - current_offset
            take = min(batch_remaining, remaining_in_tensor)
            batch_samples.append(X_tensor[current_offset : current_offset + take])

            current_offset += take
            batch_remaining -= take

            if current_offset >= X_tensor.shape[0]:
                current_tensor_idx += 1
                current_offset = 0

        if not batch_samples:
            break

        X_batch = torch.cat(batch_samples, dim=0).to(device)
        batch_actual_size = X_batch.shape[0]

        Y_orig = X_batch @ W_t
        X_q = quantize_tensor(
            X_batch,
            n_bits=act_bits,
            group_size=act_group_size,
            quant_type=act_quant_type,
            sym=act_sym,
            data_format=data_format,
        )
        Y_q = X_q @ W_q_t

        l1_error_sum += (Y_orig - Y_q).abs().sum().item()
        sample_count += batch_actual_size

        del X_batch, Y_orig, Y_q, X_q
        if torch.cuda.is_available() and batch_start % (batch_size * 4) == 0:
            torch.cuda.empty_cache()

    mean_l1_error = l1_error_sum / (sample_count * W.shape[0]) if sample_count > 0 else 0.0

    del W_reordered, W_q, W_q_t, W_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return mean_l1_error


@torch.no_grad()
def compute_quant_error(X_list, W, quant_config, batch_size=4096):
    """Compute quantization error without reordering."""
    return compute_reorder_quant_error(X_list, W, torch.arange(W.shape[1]), quant_config, batch_size=batch_size)


@torch.no_grad()
def find_and_validate_reorder(
    X_list,
    W,
    quant_config,
    metric="absmax",
    alpha_step=0.2,
    batch_size=4096,
    threshold=0.0,
    log_print=print,
):
    """Find best reorder indices by enumerating alpha values."""
    if metric not in REORDER_METRICS:
        raise ValueError(f"Unknown metric: {metric}. Use '{_format_metric_options()}'.")

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

        error_after = compute_reorder_quant_error(X_list, W, reorder_indices, quant_config, batch_size=batch_size)
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

    importance_metric = "absmax" if metric == "rr" else metric
    act_importance = compute_channel_importance(X_list, metric=importance_metric).cpu()
    weight_importance = compute_weight_channel_importance(W, metric=importance_metric).cpu()

    act_importance = act_importance / (act_importance.max() + 1e-10)
    weight_importance = weight_importance / (weight_importance.max() + 1e-10)

    num_steps = int(1.0 / alpha_step) + 1
    alphas = [i * alpha_step for i in range(num_steps)]

    best_indices = None
    best_error = error_before
    best_alpha = 0.0
    group_size = get_reorder_group_size(quant_config)

    for alpha in alphas:
        combined = (act_importance**alpha) * (weight_importance ** (1 - alpha))
        sorted_indices = torch.argsort(combined, descending=True)
        reorder_indices = build_round_robin_indices(sorted_indices, group_size) if metric == "rr" else sorted_indices

        original_indices = torch.arange(W.shape[1])
        if torch.equal(reorder_indices, original_indices):
            continue

        error_after = compute_reorder_quant_error(X_list, W, reorder_indices, quant_config, batch_size=batch_size)
        if metric == "rr":
            log_print(f"    alpha={alpha:.2f}, rr_group={group_size}: error={error_after:.6f}")
        else:
            log_print(f"    alpha={alpha:.2f}: error={error_after:.6f}")

        if error_after < best_error:
            best_error = error_after
            best_indices = reorder_indices
            best_alpha = alpha

    relative_reduction = (error_before - best_error) / (error_before + 1e-10)
    info = {
        "error_before": error_before,
        "error_after": best_error,
        "improvement_ratio": error_before / (best_error + 1e-10),
        "relative_reduction": relative_reduction,
        "best_alpha": best_alpha,
    }

    if relative_reduction >= threshold and best_indices is not None:
        return best_indices, info
    return None, info


class ReorderHook:
    """Forward pre-hook to reorder input activations."""

    def __init__(self, reorder_indices):
        self.reorder_indices = reorder_indices

    def __call__(self, module, args):
        if not args:
            return args
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return args

        indices = self.reorder_indices.to(x.device) if self.reorder_indices.device != x.device else self.reorder_indices

        if isinstance(module, nn.Linear):
            x_reordered = x[..., indices]
        elif isinstance(module, nn.Conv2d):
            if module.kernel_size != (1, 1):
                raise ValueError(f"Only 1x1 Conv2d is supported, got kernel_size={module.kernel_size}")
            x_reordered = x[:, indices, :, :]
        else:
            return args

        return (x_reordered,) + args[1:]


@torch.no_grad()
def register_reorder_hook(module, reorder_indices):
    """Register reorder hook for a supported module."""
    hook = ReorderHook(reorder_indices)
    handle = module.register_forward_pre_hook(hook)
    module._reorder_hook_id = handle.id
    return handle


def _clear_reorder_hooks(model: nn.Module) -> None:
    handles = getattr(model, "_sana_reorder_handles", None)
    if not handles:
        return
    for handle in handles:
        handle.remove()
    model._sana_reorder_handles = []


def _cache_to_input_list(inputs) -> list[torch.Tensor]:
    if inputs is None:
        return []
    if hasattr(inputs, "data") and isinstance(inputs.data, list):
        return list(inputs.data)
    if hasattr(inputs, "tensors") and isinstance(inputs.tensors, dict):
        all_data = []
        for tc in inputs.tensors.values():
            if hasattr(tc, "data") and isinstance(tc.data, list):
                all_data.extend(tc.data)
        return all_data
    if hasattr(inputs, "values"):
        all_data = []
        for tc in inputs.values():
            if hasattr(tc, "data") and isinstance(tc.data, list):
                all_data.extend(tc.data)
        return all_data
    return []


def reorder_sana(model, quant_config=None, reorder_cache=None, reorder_cache_path=None):
    """Apply reorder quantization to Sana transformer model."""
    if quant_config is None:
        quant_config = {}

    model = _unwrap_model(model)
    _clear_reorder_hooks(model)

    metric = quant_config.get("reorder_metric", "absmax")
    alpha_step = quant_config.get("alpha_step", 0.2)
    threshold = quant_config.get("reorder_threshold", 0.0)
    batch_size = quant_config.get("reorder_batch_size", 4096)
    calib_ratio = quant_config.get("calib_ratio", 1.0)
    layerwise_calib = quant_config.get("reorder_layerwise_calib", True)

    if reorder_cache is None:
        reorder_cache = {}
        sample_path = quant_config.get("calib_sample_path", "fake_quant/collect/output/cache")
        loader = CalibCacheLoader(sample_path=sample_path, ratio=calib_ratio)

        target_patterns = tuple(quant_config.get("reorder_target_patterns", SANA_BLOCK_LINEAR_PATTERNS))

        def needs_inputs_fn(name, module):
            return (
                name.startswith("blocks.")
                and isinstance(module, (nn.Linear, nn.Conv2d))
                and not (isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1))
                and any(name.endswith(pattern) for pattern in target_patterns)
            )

        log_file = None
        if reorder_cache_path:
            log_path = reorder_cache_path.replace(".pt", ".log").replace(".pth", ".log")
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_file = open(log_path, "w")

        def log_print(msg):
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()

        log_print(
            f"Calibrating reorder indices (metric={metric}, alpha_step={alpha_step}, "
            f"calib_ratio={calib_ratio}, layerwise={layerwise_calib})..."
        )

        named_modules = dict(model.named_modules())
        hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        for _layer_name, (_layer, cache, _) in iter_reorder_calib_layers(
            model,
            loader,
            needs_inputs_fn=needs_inputs_fn,
            layerwise=layerwise_calib,
        ):
            for module_name, module_cache in cache.items():
                X_list = _cache_to_input_list(module_cache.inputs)
                if not X_list:
                    continue

                module = named_modules.get(module_name, None)
                if module is None or not isinstance(module, (nn.Linear, nn.Conv2d)):
                    continue
                if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                    continue

                log_print(f"Processing {module_name}...")
                W = _module_weight_to_2d(module).detach()

                reorder_indices, info = find_and_validate_reorder(
                    X_list,
                    W,
                    quant_config,
                    metric=metric,
                    alpha_step=alpha_step,
                    batch_size=batch_size,
                    threshold=threshold,
                    log_print=log_print,
                )

                log_print(
                    f"  Error before: {info['error_before']:.6f}, after: {info['error_after']:.6f}, "
                    f"reduction: {info['relative_reduction']:.2%}, best_alpha: {info['best_alpha']:.2f}"
                )

                reorder_cache[module_name] = reorder_indices.cpu() if reorder_indices is not None else None

                if reorder_indices is not None:
                    reorder_indices_dev = reorder_indices.to(W.device)
                    reordered_weight = _module_weight_to_2d(module)[:, reorder_indices_dev]
                    _set_module_weight_from_2d(module, reordered_weight)
                    hook_handles.append(register_reorder_hook(module, reorder_indices.to(module.weight.device)))

                del X_list, W, reorder_indices, info
                gc.collect()

            for layer_cache in cache.values():
                layer_cache.clear()
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model._sana_reorder_handles = hook_handles
        log_print(f"Calibrated reorder indices for {len(reorder_cache)} modules.")
        if log_file:
            log_file.close()
        return reorder_cache

    print(f"Applying reorder from cache ({len(reorder_cache)} entries)...")
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    applied_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            continue

        reorder_indices = reorder_cache.get(name)
        if reorder_indices is None:
            continue

        weight_2d = _module_weight_to_2d(module)
        reorder_indices_dev = reorder_indices.to(weight_2d.device)
        reordered_weight = weight_2d[:, reorder_indices_dev]
        if isinstance(module, nn.Linear):
            module.weight.data = reordered_weight.to(module.weight.device, module.weight.dtype)
        else:
            module.weight.data = reordered_weight.unsqueeze(-1).unsqueeze(-1).to(module.weight.device, module.weight.dtype)
        hook_handles.append(register_reorder_hook(module, reorder_indices.to(module.weight.device)))
        applied_count += 1

    model._sana_reorder_handles = hook_handles
    print(f"Applied reorder to {applied_count} modules")
    return reorder_cache
def quant_sana_reorder(
    model,
    quant_config=None,
    smooth_cache_path=None,
    reorder_cache_path=None,
):
    """Reorder quantization function for Sana."""
    if quant_config is None:
        quant_config = {}

    model = _unwrap_model(model)

    if quant_config.get("smooth", False) or quant_config.get("svd", False):
        raise ValueError("This release supports PermuQuant and PermuQuant-H only")
    smooth = quant_config.get("smooth", False)
    hadamard = quant_config.get("hadamard", False)
    rotate_group_size = quant_config.get("rotate_group_size", 128)
    rotation_type = quant_config.get("rotation_type", "quarot")
    use_svd = quant_config.get("svd", False)

    if smooth and hadamard:
        raise ValueError("SmoothQuant and Hadamard rotation are mutually exclusive for reorder.")

    if hadamard:
        print(f"\n[Step 1] Applying {rotation_type.upper()} Hadamard rotation before reorder...")
        rotate_sana(
            model,
            rotate_group_size=rotate_group_size,
            rotation_type=rotation_type,
            quant_config=quant_config,
        )
    else:
        print("\n[Step 1] Skipping SmoothQuant/Hadamard (smooth=False, hadamard=False)")

    print("\n[Step 2] Running Reorder Quantization...")
    reorder_cache = None
    if reorder_cache_path and os.path.exists(reorder_cache_path):
        print(f"Loading reorder cache from {reorder_cache_path}...")
        reorder_cache = torch.load(reorder_cache_path, map_location="cpu")
        reorder_cache = reorder_sana(model, quant_config, reorder_cache=reorder_cache)
    else:
        print("Reorder cache not found. Computing reorder indices...")
        reorder_cache = reorder_sana(model, quant_config, reorder_cache=None, reorder_cache_path=reorder_cache_path)

    if reorder_cache_path and not os.path.exists(reorder_cache_path):
        cache_dir = os.path.dirname(reorder_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(reorder_cache, reorder_cache_path)
        print(f"Saved reorder cache to {reorder_cache_path}")

    print("\n[Step 3] Applying weight quantization...")
    quantize_sana_weight(model, quant_config)

    print("\n[Step 4] Applying activation quantization (if enabled)...")
    quantize_sana_act(model, quant_config)
    return model
