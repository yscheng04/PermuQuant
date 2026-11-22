from __future__ import annotations

import math
import typing as tp
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


class ReshapeFn:
    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        return x


class LinearReshapeFn(ReshapeFn):
    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1]).permute(int(not ic_last), int(ic_last))


@dataclass
class ModuleForwardInput:
    args: list[tp.Any] = field(default_factory=list)
    kwargs: dict[str, tp.Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ModuleForwardInput":
        def move(x):
            if isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(move(v) for v in x)
            if isinstance(x, torch.Tensor):
                return x.to(device=device)
            return x

        return ModuleForwardInput(args=move(self.args), kwargs=move(self.kwargs))

    def update(self, x: dict[str | int, tp.Any] | None = None) -> "ModuleForwardInput":
        args, kwargs = list(self.args), dict(self.kwargs)
        if x is not None:
            for k, v in x.items():
                if isinstance(k, int):
                    args[k] = v
                else:
                    kwargs[k] = v
        return ModuleForwardInput(args=args, kwargs=kwargs)


@dataclass
class TensorCache:
    data: list[torch.Tensor] = field(default_factory=list)
    channels_dim: int = 1
    reshape: ReshapeFn = ReshapeFn()
    num_cached: int = 0
    num_total: int = 0
    num_samples: int = 0
    orig_device: torch.device | str = torch.device("cpu")

    @property
    def num_cached_samples(self) -> int:
        return sum(x.shape[0] for x in self.data)

    def clear(self):
        self.data.clear()
        self.num_cached = 0

    def get_standardized_data(self, reshape: bool = False) -> list[torch.Tensor]:
        if reshape:
            return [self.reshape(x.reshape(-1, *x.shape[self.channels_dim :])) for x in self.data]
        return [x.reshape(-1, *x.shape[self.channels_dim :]) for x in self.data]

    def repartition(self, max_batch_size: int, max_size: int, standardize: bool, reshape: bool) -> "TensorCache":
        assert len(self.data) > 0, "No data to relocate."
        assert max_batch_size != 0, "max_batch_size must be non-zero."
        assert max_size != 0, "max_size must be non-zero."
        assert all(x.ndim == self.data[0].ndim for x in self.data), "All tensors must have the same #dims."
        assert all(x.shape == self.data[0].shape for x in self.data), "All tensors must have the same shape."
        data, dim, fn = self.data, self.channels_dim, self.reshape
        if standardize:
            data = [x.reshape(-1, *x.shape[dim:]) for x in data]
            dim = 1
            if reshape:
                data = [fn(x) for x in data]
                dim = -1
                fn = ReshapeFn()
        dim = dim % data[0].ndim
        orig_total = data[0].shape[0] * len(data)
        if max_batch_size > 0:
            batch_size = data[0].shape[0]
            if batch_size > max_batch_size:
                data = [
                    x[i * max_batch_size : (i + 1) * max_batch_size]
                    for x in data
                    for i in range(int(batch_size // max_batch_size))
                ]
            batch_size = data[0].shape[0]
            if max_size > 0 and batch_size * len(data) > max_size:
                assert max_size >= batch_size, "max_size must be greater than or equal to batch_size."
                data = data[:: int(len(data) // (max_size // batch_size))]
        else:
            assert max_size < 0, "max_size must be negative if max_batch_size is negative."
        used_total = data[0].shape[0] * len(data)
        ratio = used_total / orig_total
        return TensorCache(
            data,
            channels_dim=dim,
            reshape=fn,
            orig_device=self.orig_device,
            num_cached=int(math.ceil(ratio * self.num_cached)),
            num_total=int(math.ceil(ratio * self.num_total)),
            num_samples=int(math.ceil(ratio * self.num_samples)),
        )


class TensorsCache:
    def __init__(self, tensors: OrderedDict[str | int, TensorCache] | TensorCache) -> None:
        self.tensors = OrderedDict({0: tensors}) if isinstance(tensors, TensorCache) else tensors

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    def front(self) -> TensorCache:
        return next(iter(self.tensors.values()))

    def items(self) -> tp.ItemsView[str | int, TensorCache]:
        return self.tensors.items()

    def keys(self) -> tp.KeysView[str | int]:
        return self.tensors.keys()

    def values(self) -> tp.ValuesView[str | int, TensorCache]:
        return self.tensors.values()

    def __getitem__(self, key: str | int) -> TensorCache:
        return self.tensors[key]

    def __iter__(self) -> tp.Iterator[TensorCache]:
        return iter(self.tensors.values())

    def __len__(self) -> int:
        return len(self.tensors)

    def clear(self):
        for tensor in self.tensors.values():
            tensor.clear()

    def set_num_samples(self, num_samples: int):
        for tensor in self.tensors.values():
            tensor.num_samples = num_samples

    @property
    def num_cached_samples(self) -> int:
        if self.num_tensors == 0:
            return 0
        return self.front().num_cached_samples

    def extract(self, index: int, kwargs: dict[str, tp.Any]) -> ModuleForwardInput:
        args, out_kwargs = [], {}
        out_kwargs.update(kwargs)
        for key, tensor in self.tensors.items():
            if isinstance(key, int):
                assert len(args) == key, f"Expected {key} args, but got {len(args)}"
                args.append(tensor.data[index].to(tensor.orig_device, non_blocking=True))
            else:
                out_kwargs[key] = tensor.data[index].to(tensor.orig_device, non_blocking=True)
        return ModuleForwardInput(args=args, kwargs=out_kwargs)


class IOTensorsCache:
    def __init__(
        self,
        inputs: TensorCache | TensorsCache | None = None,
        outputs: TensorCache | TensorsCache | None = None,
    ):
        self.inputs = TensorsCache(inputs) if isinstance(inputs, TensorCache) else inputs
        self.outputs = TensorsCache(outputs) if isinstance(outputs, TensorCache) else outputs

    def clear(self):
        if self.inputs is not None:
            self.inputs.clear()
        if self.outputs is not None:
            self.outputs.clear()

    def set_num_samples(self, num_samples: int):
        if self.inputs is not None:
            self.inputs.set_num_samples(num_samples)
        if self.outputs is not None:
            self.outputs.set_num_samples(num_samples)

    @property
    def num_cached_samples(self) -> int:
        if self.inputs is not None:
            return self.inputs.num_cached_samples
        if self.outputs is not None:
            return self.outputs.num_cached_samples
        return 0
