

import typing as tp
import torch.nn as nn

def get_needs_inputs_fn(
    model: nn.Module = None, config: tp.Any = None
) -> tp.Callable[[str, nn.Module], bool]:
    """Get function that checks whether the module needs to cache inputs.

    Args:
        model (`nn.Module`):
            The model.
        config (`Any`):
            The quantization configuration.

    Returns:
        `Callable[[str, nn.Module], bool]`:
            The function that checks whether the module needs to cache inputs.
    """

    def needs_inputs(name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # Check if the module is one of the linear layers in the DecoderLayer
        # Common names in Qwen2/Llama architecture
        target_names = [
            "self_attn",
            "gate_proj", "up_proj", "down_proj"
        ]
        return any(key in name for key in target_names)

    return needs_inputs