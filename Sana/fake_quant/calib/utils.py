import typing as tp

import torch.nn as nn


def get_needs_inputs_fn(
    model: nn.Module = None,
    config: tp.Any = None,
) -> tp.Callable[[str, nn.Module], bool]:
    del model, config

    def needs_inputs(name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        target_names = [
            "qkv",
            "q_linear",
            "kv_linear",
            "proj",
            "fc1",
            "fc2",
            "linear",
        ]
        return any(key in name for key in target_names)

    return needs_inputs
