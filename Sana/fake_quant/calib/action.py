# -*- coding: utf-8 -*-
"""Actions for caching inputs and outputs."""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .cache import IOTensorsCache, TensorsCache
from ..hooks import BaseInputPackager, BaseOutputPackager, Hook, IOHook, KeyedInputPackager, KeyedOutputPackager


class CacheHook(IOHook):
    def __init__(
        self,
        name: str,
        module: nn.Module,
        action: "CacheAction",
        cache: TensorsCache,
        info_mode: bool,
        is_output: bool,
    ):
        super().__init__(
            pre=not is_output,
            post=is_output,
            input_packager=None if is_output else action.get_input_packager(name, module, cache),
            output_packager=action.get_output_packager(name, module, cache) if is_output else None,
        )
        self.name = name
        self.action = action
        self.cache = cache
        self.info_mode = info_mode

    def pre_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> None:
        tensors = self.input_packager.unpack(module, input_args, input_kwargs)
        if self.info_mode:
            self.action.info(self.name, module, tensors, self.cache)
        assert len(tensors) == self.cache.num_tensors, f"Expected {self.cache.num_tensors} args, but got {len(tensors)}"
        if not self.info_mode:
            self.action.apply(self.name, module, tensors, self.cache)

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        tensors = self.output_packager.unpack(module, input_args, input_kwargs, output)
        if self.info_mode:
            self.action.info(self.name, module, tensors, self.cache)
        assert len(tensors) == self.cache.num_tensors, f"Expected {self.cache.num_tensors} args, but got {len(tensors)}"
        if not self.info_mode:
            self.action.apply(self.name, module, tensors, self.cache)


class CacheAction(ABC):
    device: torch.device | None = None

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device

    @abstractmethod
    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        ...

    @abstractmethod
    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        ...

    def get_input_packager(self, name: str, module: nn.Module, cache: TensorsCache) -> BaseInputPackager:
        return KeyedInputPackager(module=module, index_or_keys=list(cache.keys()))

    def get_output_packager(self, name: str, module: nn.Module, cache: TensorsCache) -> BaseOutputPackager:
        return KeyedOutputPackager(index_or_keys=list(cache.keys()))

    def register(
        self,
        name: str,
        module: nn.Module,
        cache: IOTensorsCache,
        info_mode: bool,
        needs_inputs: bool,
        needs_outputs: bool,
    ) -> list[Hook]:
        hooks = []
        if needs_inputs:
            assert cache.inputs is not None
            hooks.append(CacheHook(name, module, self, cache.inputs, info_mode, is_output=False).register(module))
        if needs_outputs:
            assert cache.outputs is not None
            hooks.append(CacheHook(name, module, self, cache.outputs, info_mode, is_output=True).register(module))
        return hooks


class ConcatCacheAction(CacheAction):
    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        del name, module
        for k, c in cache.tensors.items():
            x = tensors[k]
            device = self.device or x.device
            c.data.append(x.detach().to(device=device).clone())
            c.num_cached += x.shape[0]

    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        for k, c in cache.tensors.items():
            x = tensors[k]
            c.num_total += x.shape[0]
            c.orig_device = x.device
