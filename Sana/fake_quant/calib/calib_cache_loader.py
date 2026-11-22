from __future__ import annotations

import contextlib
import functools
import gc
import typing as tp
from collections import OrderedDict

import psutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.hooks
from tqdm import tqdm

from ..dataset.calib_sample_dataset import CalibSampleDataset
from ..hooks import EarlyStopException, EarlyStopHook, Hook
from ..utils import tree_copy_with_ref, tree_map
from .action import CacheAction, ConcatCacheAction
from .cache import IOTensorsCache, LinearReshapeFn, ModuleForwardInput, TensorCache


class CalibCacheLoader:
    def __init__(self, sample_path="fake_quant/collect/output/cache", layers=None, ratio: float = 1.0):
        del layers
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be in (0, 1].")
        self.dataset = CalibSampleDataset(sample_path)
        total_samples = len(self.dataset)
        self._num_samples = max(1, int(total_samples * ratio)) if total_samples > 0 else 0

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def iter_samples(self, *args, **kwargs):
        del args, kwargs
        dataloader = self.dataset.build_loader()
        for idx, data in enumerate(dataloader):
            if idx >= self._num_samples:
                break
            yield ModuleForwardInput(args=data["input_args"], kwargs=data["input_kwargs"])

    def _get_default_layers(self, model: nn.Module) -> list[nn.Module]:
        if hasattr(model, "blocks"):
            return list(model.blocks)
        if hasattr(model, "layers"):
            return list(model.layers)
        raise AttributeError("Cannot infer default layers from model. Expected `.blocks` or `.layers`.")

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        del name
        if isinstance(module, nn.Linear):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        if isinstance(module, nn.Conv2d):
            if module.kernel_size != (1, 1):
                raise NotImplementedError(
                    f"Only 1x1 Conv2d is supported for calibration cache, got kernel_size={module.kernel_size}."
                )
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=1),
                outputs=TensorCache(channels_dim=1),
            )
        raise NotImplementedError(f"Cache for module type {type(module)} is not implemented.")

    def _convert_layer_inputs(
        self,
        m: nn.Module,
        args: tuple[tp.Any, ...],
        kwargs: dict[str, tp.Any],
        save_all: bool = False,
    ) -> ModuleForwardInput:
        del m, save_all
        return ModuleForwardInput(
            args=tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, args),
            kwargs=tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, kwargs),
        )

    def _convert_layer_outputs(self, m: nn.Module, outputs: tp.Any) -> dict[str | int, tp.Any]:
        del m
        if isinstance(outputs, torch.Tensor):
            return {0: outputs.detach().cpu()}
        if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
            return {0: outputs[0].detach().cpu()}
        raise NotImplementedError(f"Cache for layer output type {type(outputs)} is not implemented.")

    def _layer_forward_pre_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        cache: list[ModuleForwardInput],
        save_all: bool = False,
    ) -> None:
        inputs = self._convert_layer_inputs(m, args, kwargs, save_all=save_all)
        if len(cache) > 0:
            inputs.args = tree_copy_with_ref(inputs.args, cache[0].args)
            inputs.kwargs = tree_copy_with_ref(inputs.kwargs, cache[0].kwargs)
        else:
            inputs.args = tree_map(lambda x: x, inputs.args)
            inputs.kwargs = tree_map(lambda x: x, inputs.kwargs)
        cache.append(ModuleForwardInput(args=inputs.args, kwargs=inputs.kwargs))

    @torch.inference_mode()
    def _iter_layer_activations(
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        layers: tp.Sequence[nn.Module] | None = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        recomputes: list[bool] | None = None,
        use_prev_layer_outputs: list[bool] | None = None,
        early_stop_module: nn.Module | None = None,
        clear_after_yield: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
        None,
        None,
    ]:
        del args, kwargs
        if needs_inputs_fn is None:
            raise ValueError("`needs_inputs_fn` must be provided for CalibCacheLoader.")
        if needs_outputs_fn is None:
            needs_outputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_outputs_fn, bool):
            if needs_outputs_fn:
                needs_outputs_fn = lambda name, module: isinstance(module, nn.Linear)  # noqa: E731
            else:
                needs_outputs_fn = lambda name, module: False  # noqa: E731
        if isinstance(needs_inputs_fn, bool):
            if needs_inputs_fn:
                needs_inputs_fn = lambda name, module: isinstance(module, nn.Linear)  # noqa: E731
            else:
                needs_inputs_fn = lambda name, module: False  # noqa: E731

        if layers is None:
            layers = self._get_default_layers(model)
        assert isinstance(layers, (nn.Sequential, nn.ModuleList, list, tuple))
        if recomputes is None:
            recomputes = [False] * len(layers)
        elif isinstance(recomputes, bool):
            recomputes = [recomputes] * len(layers)
        if use_prev_layer_outputs is None:
            use_prev_layer_outputs = [True] * len(layers)
        elif isinstance(use_prev_layer_outputs, bool):
            use_prev_layer_outputs = [use_prev_layer_outputs] * len(layers)
        use_prev_layer_outputs[0] = False
        assert len(recomputes) == len(use_prev_layer_outputs) == len(layers)

        def _move_sample_to_device(sample: ModuleForwardInput, device):
            return ModuleForwardInput(
                args=tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, sample.args),
                kwargs=tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, sample.kwargs),
            )

        cache: dict[str, dict[str, IOTensorsCache]] = {}
        module_names: dict[str, list[str]] = {"": []}
        named_layers: OrderedDict[str, nn.Module] = {"": model}
        forward_cache: dict[str, list[ModuleForwardInput]] = {}
        info_hooks: list[Hook] = []
        forward_hooks: list[torch.utils.hooks.RemovableHandle] = []
        hook_args: dict[str, list[tuple[str, nn.Module, bool, bool]]] = {}
        layer_name = ""

        for module_name, module in model.named_modules():
            if module_name and module in layers:
                layer_name = module_name
                assert layer_name not in module_names
                named_layers[layer_name] = module
                module_names[layer_name] = []
                forward_cache[layer_name] = []
            if layer_name and module_name.startswith(layer_name):
                needs_inputs = needs_inputs_fn(module_name, module)
                needs_outputs = needs_outputs_fn(module_name, module)
                if needs_inputs or needs_outputs:
                    module_names[layer_name].append(module_name)
                    cache.setdefault(layer_name, {})[module_name] = self._init_cache(module_name, module)
                    hook_args.setdefault(layer_name, []).append((module_name, module, needs_inputs, needs_outputs))
                    info_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=True,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
        if len(cache) == 0:
            return

        module_names.pop("")
        named_layers.pop("")
        assert layer_name, "No layer in the given layers is found in the model"
        assert "" not in cache, "The model should not have empty layer name"

        ordered_named_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for layer in layers:
            for name, module in named_layers.items():
                if module is layer:
                    ordered_named_layers[name] = module
                    break
        assert len(ordered_named_layers) == len(named_layers) == len(layers)
        named_layers = ordered_named_layers

        for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
            forward_hooks.append(
                layer.register_forward_pre_hook(
                    functools.partial(
                        self._layer_forward_pre_hook,
                        cache=forward_cache[layer_name],
                        save_all=not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx],
                    ),
                    with_kwargs=True,
                )
            )

        with contextlib.nullcontext():
            if early_stop_module is not None:
                forward_hooks.append(early_stop_module.register_forward_hook(EarlyStopHook()))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tbar = tqdm(
                desc="collecting acts info",
                leave=False,
                total=self.num_samples,
                unit="samples",
                dynamic_ncols=True,
            )
            num_samples = 0
            for sample in self.iter_samples():
                num_samples += 1
                sample = _move_sample_to_device(sample, device)
                try:
                    model(*sample.args, **sample.kwargs)
                except EarlyStopException:
                    pass
                tbar.update(1)
                tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                if psutil.virtual_memory().percent > 90:
                    raise RuntimeError("memory usage > 90%, aborting")

            for layer_cache in cache.values():
                for module_cache in layer_cache.values():
                    module_cache.set_num_samples(num_samples)
            for hook in forward_hooks:
                hook.remove()
            for hook in info_hooks:
                hook.remove()
            del info_hooks, forward_hooks

            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                layer_hooks: list[Hook] = []
                for module_name, module, needs_inputs, needs_outputs in hook_args[layer_name]:
                    layer_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=False,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
                hook_args.pop(layer_name)

                if recomputes[layer_idx]:
                    layer_hooks.append(EarlyStopHook().register(layer if early_stop_module is None else early_stop_module))
                    tbar = tqdm(
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        total=self.num_samples,
                        unit="samples",
                        dynamic_ncols=True,
                    )
                    for sample in self.iter_samples():
                        sample = _move_sample_to_device(sample, device)
                        try:
                            model(*sample.args, **sample.kwargs)
                        except EarlyStopException:
                            pass
                        tbar.update(1)
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%, aborting")
                        gc.collect()
                else:
                    param_device = next(layer.parameters()).device
                    layer_device = torch.device("cuda") if param_device.type == "meta" else param_device
                    layer_outputs: list[tp.Any] = []
                    tbar = tqdm(
                        forward_cache[layer_name],
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        unit="batches",
                        dynamic_ncols=True,
                    )
                    if not use_prev_layer_outputs[layer_idx]:
                        prev_layer_outputs: list[dict[str | int, tp.Any]] = [None] * len(tbar)
                    for i, inputs in enumerate(tbar):
                        inputs = inputs.update(prev_layer_outputs[i]).to(device=layer_device)
                        outputs = layer(*inputs.args, **inputs.kwargs)
                        layer_outputs.append(self._convert_layer_outputs(layer, outputs))
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%, aborting")
                    prev_layer_outputs = layer_outputs
                    del inputs, outputs, layer_outputs
                    if (layer_idx == len(named_layers) - 1) or not use_prev_layer_outputs[layer_idx + 1]:
                        del prev_layer_outputs

                for hook in layer_hooks:
                    hook.remove()
                del layer_hooks
                layer_inputs = forward_cache.pop(layer_name, [])
                if not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx]:
                    layer_inputs = [self._convert_layer_inputs(layer, inputs.args, inputs.kwargs) for inputs in layer_inputs]
                gc.collect()
                torch.cuda.empty_cache()
                yield layer_name, (layer, cache[layer_name], layer_inputs)
                if clear_after_yield:
                    for module_cache in cache[layer_name].values():
                        module_cache.clear()
                cache.pop(layer_name)
                del layer_inputs
                gc.collect()
                torch.cuda.empty_cache()

    def iter_layer_activations(
        self,
        model: nn.Module,
        *args,
        action: CacheAction = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        **kwargs,
    ) -> tp.Generator[
        tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
        None,
        None,
    ]:
        del args
        if needs_inputs_fn is None:
            raise ValueError("`needs_inputs_fn` must be provided for CalibCacheLoader.")
        action = ConcatCacheAction("cpu") if action is None else action
        layers = kwargs.pop("layers", None)
        if layers is None:
            layers = self._get_default_layers(model)
        recomputes = [False] * len(layers)
        use_prev_layer_outputs = [True] * len(layers)
        use_prev_layer_outputs[0] = False

        for layer_name, (layer, layer_cache, layer_inputs) in self._iter_layer_activations(
            model,
            action=action,
            layers=layers,
            needs_inputs_fn=needs_inputs_fn,
            needs_outputs_fn=needs_outputs_fn,
            recomputes=recomputes,
            use_prev_layer_outputs=use_prev_layer_outputs,
            **kwargs,
        ):
            yield layer_name, (layer, layer_cache, layer_inputs)
