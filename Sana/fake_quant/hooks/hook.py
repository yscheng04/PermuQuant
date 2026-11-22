# -*- coding: utf-8 -*-
"""nn.Module Hook."""

from __future__ import annotations

import typing as tp
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.hooks

from .packager import BaseInputPackager, BaseOutputPackager, SimpleInputPackager, SimpleOutputPackager


class Hook:
    handles: dict[nn.Module, list[torch.utils.hooks.RemovableHandle]]

    def __init__(self, *, pre: bool, post: bool) -> None:
        self.handles = defaultdict(list)
        self.pre = pre
        self.post = post
        self.activated = True
        assert self.pre or self.post, "At least one of pre and post must be True."

    def activate(self) -> "Hook":
        self.activated = True
        return self

    def deactivate(self) -> "Hook":
        self.activated = False
        return self

    def pre_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tp.Any:
        return None

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        return None

    def __call__(self, *args, **kwargs) -> tp.Any:
        if not self.activated:
            return None
        n = len(args) + len(kwargs)
        if n == 3:
            return self.pre_forward(*args, **kwargs)
        if n == 4:
            return self.post_forward(*args, **kwargs)
        raise ValueError(f"Invalid number of arguments: {n}")

    def register(
        self,
        module: nn.Module | tp.Iterable[nn.Module],
        prepend: bool | tuple[bool, bool] = False,
        always_call: bool = False,
    ) -> "Hook":
        if isinstance(module, nn.Module):
            module = [module]
        prepends = (prepend, prepend) if isinstance(prepend, bool) else prepend
        if self.pre:
            for mod in module:
                self.handles[mod].append(mod.register_forward_pre_hook(self, prepend=prepends[0], with_kwargs=True))
        if self.post:
            for mod in module:
                self.handles[mod].append(
                    mod.register_forward_hook(self, prepend=prepends[1], with_kwargs=True, always_call=always_call)
                )
        return self

    def remove(self, module: nn.Module | tp.Iterable[nn.Module] | None = None) -> "Hook":
        if module is None:
            for handles in self.handles.values():
                for handle in handles:
                    handle.remove()
                handles.clear()
            self.handles.clear()
            return self
        if isinstance(module, nn.Module):
            module = [module]
        for mod in module:
            handles = self.handles.pop(mod, [])
            for handle in handles:
                handle.remove()
            handles.clear()
        return self


class EarlyStopException(Exception):
    pass


class EarlyStopHook(Hook):
    def __init__(self):
        super().__init__(pre=False, post=True)

    def post_forward(self, *args, **kwargs) -> None:
        raise EarlyStopException()


class IOHook(Hook):
    def __init__(
        self,
        *,
        pre: bool,
        post: bool,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ):
        super().__init__(pre=pre, post=post)
        self.input_packager = input_packager or SimpleInputPackager() if pre else None
        self.output_packager = output_packager or SimpleOutputPackager() if post else None
