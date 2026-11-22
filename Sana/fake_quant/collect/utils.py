# -*- coding: utf-8 -*-
"""Common utilities for collecting data."""

import typing as tp

import numpy as np
import torch
import torch.nn as nn

__all__ = ["CollectHook", "tree_map", "process"]


def tree_map(func: tp.Callable[[tp.Any], tp.Any], tree: tp.Any) -> tp.Any:
    """Apply a function to tree-structured data."""
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, v) for v in tree)
    elif isinstance(tree, (torch.Tensor, np.ndarray)):
        return func(tree)
    else:
        return tree


class CollectHook:
    def __init__(self, caches: list[dict[str, tp.Any]] = None, zero_redundancy: bool = False) -> None:
        self.caches = [] if caches is None else caches
        self.zero_redundancy = zero_redundancy

    def __call__(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        del module
        cache = tree_map(
            lambda x: x.cpu(),
            {"input_args": input_args, "input_kwargs": input_kwargs, "outputs": output},
        )
        self.caches.append(cache)


def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)
