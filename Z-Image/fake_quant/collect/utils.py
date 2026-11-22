# -*- coding: utf-8 -*-
"""Common utilities for collecting data."""

import inspect
import typing as tp

import torch
import torch.nn as nn

from fake_quant.utils import tree_map

__all__ = ["CollectHook"]

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
        # 将输入与输出移动到cpu并存储
        cache = tree_map(lambda x: x.cpu(), {"input_args": input_args, "input_kwargs": input_kwargs, "outputs": output})
        # 跳过理解模态下 prediction-by-token 的情况
        # if cache['input_kwargs']['packed_query_sequence'].shape[0] > 1:
        #     self.caches.append(cache)
        self.caches.append(cache)