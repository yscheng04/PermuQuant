"""Model-local helpers retained from the validated generation path."""


def default_quant_config():
    return {
        # Weight quantization
        "weight_quant": True,
        "weight_bits": 3,
        "weight_quant_type": "per_group",
        "weight_group_size": 32,
        "weight_sym": True,
        # Weight clipping
        "weight_clip": False,
        "weight_clip_percentile": 99.9995,
        # SVD specific
        "weight_rank": 32,
        # Activation quantization
        "act_quant": True,
        "act_bits": 3,
        "act_quant_type": "per_group",
        "act_group_size": 32,
        "act_sym": True,
        # Activation clipping
        "act_clip": False,
        "act_clip_percentile": 99.9999,
        # SmoothQuant
        "smooth_alpha": 0.4,
        # Hadamard rotation / ConvRot
        "rotate_group_size": 64,
        "rotation_type": "quarot",
        "hadamard": False,
        "save_rotate_cache": False,
        # OmniQuant specific
        "lwc": True,
        "let": True,
        "block_recon_epochs": 20,
        "block_recon_lr": 1e-3,
        "block_recon_batch_size": 1,
        "num_calib_samples": 32,
        "let_alpha": 0.5,
        "calib_ratio": 1.0,
        # Reorder specific
        "reorder_metric": "second_moment",
        "mixed_alpha_step": 0.2,
        "alpha_step": 0.2,
        "alpha_start": 0.0,
        "alpha_end": 1.0,
        "reorder_random_seed": None,
        "reorder_batch_size": 4096,
        "reorder_layerwise_calib": True,
        "reorder_threshold": 0,
        "smooth": True,
        "svd": False,
        # Debug
        "debug": False,
    }
