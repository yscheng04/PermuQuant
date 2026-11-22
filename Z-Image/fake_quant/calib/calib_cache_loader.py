import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.hooks
import psutil
import os
import functools
import gc
import contextlib
from tqdm import tqdm
from collections import OrderedDict
from ..dataset.calib_sample_dataset import CalibSampleDataset
from .cache import IOTensorsCache, TensorCache, LinearReshapeFn, ModuleForwardInput
import typing as tp
from ..utils import tree_map, tree_copy_with_ref
from ..hooks import EarlyStopException, EarlyStopHook, Hook
from .action import CacheAction, ConcatCacheAction
from zimage.transformer import ZImageTransformerBlock

class CalibCacheLoader():
    def __init__(self, sample_path="fake_quant/collect/output/cache", layers=None, ratio: float = 1.0):
        self.dataset = CalibSampleDataset(sample_path)
        self.ratio = ratio

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset (after applying ratio)."""
        return int(len(self.dataset) * self.ratio)

    def iter_samples(self, *args, **kwargs):
        dataloader = self.dataset.build_loader()
        max_samples = int(len(self.dataset) * self.ratio)
        for idx, data in enumerate(dataloader):
            if idx >= max_samples:
                break
            yield ModuleForwardInput(args=data["input_args"], kwargs=data["input_kwargs"])

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        if isinstance(module, (nn.Linear,)):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        else:
            raise NotImplementedError(f"Cache for module type {type(module)} is not implemented.")

    def _convert_layer_inputs(
        self, m: nn.Module, args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any], save_all: bool = False
    ) -> ModuleForwardInput:
        """Convert inputs for Z-Image transformer blocks.

        For ZImageTransformerBlock, inputs are x, attn_mask, freqs_cis, and optional adaln_input.
        When `save_all=True`, save CPU snapshots of tensors.
        When `save_all=False`, do not include heavy fields in the returned input.
        """
        kw = {k: v for k, v in kwargs.items()}

        if isinstance(m, ZImageTransformerBlock):
            # Z-Image inputs: x, attn_mask, freqs_cis, adaln_input (optional)
            # x should be in args[0], attn_mask and freqs_cis in kwargs
            # Save tensors to CPU if needed
            if save_all:
                for k in list(kw.keys()):
                    if isinstance(kw[k], torch.Tensor):
                        kw[k] = kw[k].detach().cpu()
                new_args = tuple(
                    arg.detach().cpu() if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                )
            else:
                new_args = args

            return ModuleForwardInput(args=new_args, kwargs=kw)

        return ModuleForwardInput(args=(), kwargs=kw)

    def _convert_layer_outputs(self, m: nn.Module, outputs: tp.Any) -> dict[str | int, tp.Any]:
        """Convert layer outputs to dictionary for updating the next layer inputs.

        Args:
            m (`nn.Module`):
                Layer.
            outputs (`Any`):
                Layer outputs.

        Returns:
            `dict[str | int, Any]`:
                Dictionary for updating the next layer inputs.
        """
        if isinstance(m, ZImageTransformerBlock):
            # ZImageTransformerBlock returns x (tensor)
            # Use int key 0 so it goes to args[0] in update(), not kwargs
            assert isinstance(outputs, torch.Tensor)
            return {0: outputs.detach().cpu()}
        else:
            raise NotImplementedError(f"Cache for module type {type(m)} is not implemented.")
        
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

        inputs = ModuleForwardInput(args=inputs.args, kwargs=inputs.kwargs)
        # FIX: Always move cached inputs to CPU to prevent GPU OOM
        # This is needed because save_all=False for some layers doesn't move tensors to CPU
        inputs.args = tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, inputs.args)
        inputs.kwargs = tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, inputs.kwargs)
        cache.append(inputs)

    @torch.inference_mode()
    def _iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        layers: tp.Sequence[nn.Module] | None = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        recomputes: list[bool] | None = None,
        use_prev_layer_outputs: list[bool] | None = None,
        early_stop_module: nn.Module | None = None,
        clear_after_yield: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            layers (`Sequence[nn.Module]` or `None`, *optional*, defaults to `None`):
                Layers to cache activations. If `None`, cache all layers.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            recomputes (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to recompute the activations for each layer.
            use_prev_layer_outputs (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to use the previous layer outputs as inputs for the current layer.
            early_stop_module (`nn.Module` or `None`, *optional*, defaults to `None`):
                Module for early stopping.
            clear_after_yield (`bool`, *optional*, defaults to `True`):
                Whether to clear the cache after yielding the activations.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - layer input arguments
        """
        if needs_outputs_fn is None:
            needs_outputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_outputs_fn, bool):
            if needs_outputs_fn:
                needs_outputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_outputs_fn = lambda name, module: False  # noqa: E731
        if needs_inputs_fn is None:
            needs_inputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_inputs_fn, bool):
            if needs_inputs_fn:
                needs_inputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_inputs_fn = lambda name, module: False  # noqa: E731
        if layers is None:
            recomputes = [True]
            use_prev_layer_outputs = [False]
        else:
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

        # Pre-calculate layer devices for NaiveCache distribution
        layer_devices = {}
        # Try to find layers to determine device map
        layers_container = getattr(model, "layers", None)
        if layers_container is None and hasattr(model, "model"):
             layers_container = getattr(model.model, "layers", None)
        
        if layers_container is not None:
            for idx, layer in enumerate(layers_container):
                try:
                    # Get device of the layer
                    if hasattr(layer, "parameters"):
                        p = next(layer.parameters(), None)
                        if p is not None:
                            layer_devices[idx] = p.device
                except Exception:
                    pass

        def _move_sample_to_device(sample, device):
            new_args = tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, sample.args)
            new_kwargs = {}

            packed_indexes_empty = False
            if "packed_key_value_indexes" in sample.kwargs and "past_key_values" in sample.kwargs:
                idx = sample.kwargs["packed_key_value_indexes"]
                pkv = sample.kwargs["past_key_values"]
                
                # Universal Fix: Restore past_key_values to pre-forward state
                # The saved past_key_values is post-forward (contains new keys), causing shape mismatch.
                # We slice it using packed_key_value_indexes to retrieve exactly the "old" keys.
                if isinstance(idx, torch.Tensor) and hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                    for i in range(len(pkv.key_cache)):
                        if pkv.key_cache[i] is not None:
                            # Ensure index device matches cache device for slicing
                            idx_dev = idx.to(pkv.key_cache[i].device)
                            pkv.key_cache[i] = pkv.key_cache[i][idx_dev]
                        
                        if pkv.value_cache[i] is not None:
                            idx_dev = idx.to(pkv.value_cache[i].device)
                            pkv.value_cache[i] = pkv.value_cache[i][idx_dev]

            for k, v in sample.kwargs.items():
                if "indexes" in k:
                    # Ensure indexes are on CPU
                    new_kwargs[k] = tree_map(lambda x: x.to("cpu") if isinstance(x, torch.Tensor) else x, v)
                elif "past_key_values" in k and hasattr(v, "key_cache") and hasattr(v, "value_cache"):
                    # Distribute past_key_values to correct devices per layer
                    # Note: We modify in-place or assume v is a transient object from dataset
                    # Ideally we should clone v, but NaiveCache might not support clone easily.
                    # Since samples are usually transient, this is likely fine.
                    for layer_idx, tensor in v.key_cache.items():
                        if isinstance(tensor, torch.Tensor):
                            target_device = layer_devices.get(layer_idx, device)
                            v.key_cache[layer_idx] = tensor.to(target_device)
                    
                    for layer_idx, tensor in v.value_cache.items():
                         if isinstance(tensor, torch.Tensor):
                            target_device = layer_devices.get(layer_idx, device)
                            v.value_cache[layer_idx] = tensor.to(target_device)
                    
                    new_kwargs[k] = v
                else:
                    new_kwargs[k] = tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, v)
            return ModuleForwardInput(new_args, new_kwargs)

        cache: dict[str, dict[str, IOTensorsCache]] = {}
        module_names: dict[str, list[str]] = {"": []}
        named_layers: OrderedDict[str, nn.Module] = {"": model}
        # region we first collect infomations for yield modules
        forward_cache: dict[str, list[ModuleForwardInput]] = {}
        info_hooks: list[Hook] = []
        forward_hooks: list[torch.utils.hooks.RemovableHandle] = []
        hook_args: dict[str, list[tuple[str, nn.Module, bool, bool]]] = {}
        layer_name = ""
        for module_name, module in model.named_modules():
            if layers is not None and module_name and module in layers:
                layer_name = module_name
                assert layer_name not in module_names
                named_layers[layer_name] = module
                module_names[layer_name] = []
                forward_cache[layer_name] = []
            if layers is None or (layer_name and module_name.startswith(layer_name)):
                # we only cache modules in the layer
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
        if layers is not None:
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
            assert len(ordered_named_layers) == len(named_layers)
            assert len(ordered_named_layers) == len(layers)
            named_layers = ordered_named_layers
            del ordered_named_layers
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
        else:
            assert len(named_layers) == 1 and "" in named_layers
            assert len(module_names) == 1 and "" in module_names
            assert len(cache) == 1 and "" in cache
        # endregion
        with contextlib.nullcontext():
            if early_stop_module is not None:
                forward_hooks.append(early_stop_module.register_forward_hook(EarlyStopHook()))
            with torch.inference_mode():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tbar = tqdm(
                    desc="collecting acts info",
                    leave=False,
                    total=self.num_samples,
                    unit="samples",
                    dynamic_ncols=True,
                )
                num_samples = 0
                for sample in self.iter_samples(*args, **kwargs):
                    num_samples += 1
                    sample = _move_sample_to_device(sample, device)
                    try:
                        model(*sample.args, **sample.kwargs)
                    except EarlyStopException:
                        pass
                    # Free GPU memory after each sample to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tbar.update(1)
                    tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                    # if psutil.virtual_memory().percent > 90:
                    #     raise RuntimeError("memory usage > 90%%, aborting")
            for layer_cache in cache.values():
                for module_cache in layer_cache.values():
                    module_cache.set_num_samples(num_samples)
            for hook in forward_hooks:
                hook.remove()
            for hook in info_hooks:
                hook.remove()
            del info_hooks, forward_hooks
            # endregion
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                # region we first register hooks for caching activations
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
                # endregion
                if recomputes[layer_idx]:
                    if layers is None:
                        if early_stop_module is not None:
                            layer_hooks.append(EarlyStopHook().register(early_stop_module))
                    else:
                        layer_hooks.append(EarlyStopHook().register(layer))
                    tbar = tqdm(
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        total=self.num_samples,
                        unit="samples",
                        dynamic_ncols=True,
                    )
                    for sample in self.iter_samples(*args, **kwargs):
                        sample = _move_sample_to_device(sample, device)
                        try:
                            model(*sample.args, **sample.kwargs)
                        except EarlyStopException:
                            pass
                        tbar.update(1)
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        # if psutil.virtual_memory().percent > 90:
                        #     raise RuntimeError("memory usage > 90%%, aborting")
                        gc.collect()
                else:
                    # region we then forward the layer to collect activations
                    # Check if layer is on meta device (offloaded via accelerate)
                    # If so, we must place inputs on CUDA to trigger the hook that loads weights.
                    param_device = next(layer.parameters()).device
                    if param_device.type == "meta":
                        device = torch.device("cuda")
                    else:
                        device = param_device

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
                        inputs = inputs.update(prev_layer_outputs[i]).to(device=device)

                        outputs = layer(*inputs.args, **inputs.kwargs)
                        layer_outputs.append(self._convert_layer_outputs(layer, outputs))
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        # if psutil.virtual_memory().percent > 90:
                        #     raise RuntimeError("memory usage > 90%%, aborting")
                    prev_layer_outputs = layer_outputs
                    del inputs, outputs, layer_outputs
                    if (layer_idx == len(named_layers) - 1) or not use_prev_layer_outputs[layer_idx + 1]:
                        del prev_layer_outputs
                    # endregion
                for hook in layer_hooks:
                    hook.remove()
                del layer_hooks
                layer_inputs = forward_cache.pop(layer_name, [])
                if not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx]:
                    layer_inputs = [
                        self._convert_layer_inputs(layer, inputs.args, inputs.kwargs) for inputs in layer_inputs
                    ]
                gc.collect()
                torch.cuda.empty_cache()
                yield layer_name, (layer, cache[layer_name], layer_inputs)
                # region clear layer cache
                if clear_after_yield:
                    for module_cache in cache[layer_name].values():
                        module_cache.clear()
                cache.pop(layer_name)
                del layer_inputs
                gc.collect()
                torch.cuda.empty_cache()
                # endregion

    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - list of layer input arguments for each sample
        """
        action = ConcatCacheAction("cpu") if action is None else action
        if kwargs.get("layers", None) is None:
            layers = model.layers
        else:
            layers = kwargs["layers"]
            kwargs.pop("layers")
        recomputes = [False] * len(layers)
        use_prev_layer_outputs = [True] * len(layers)
        use_prev_layer_outputs[0] = False
        
        for layer_idx, (layer_name, (layer, layer_cache, layer_inputs)) in enumerate(
            self._iter_layer_activations(
                model,
                *args,
                action=action,
                layers=layers,
                needs_inputs_fn=needs_inputs_fn,
                needs_outputs_fn=needs_outputs_fn,
                recomputes=recomputes,
                use_prev_layer_outputs=use_prev_layer_outputs,
                **kwargs,
            )
        ):
            yield layer_name, (layer, layer_cache, layer_inputs)


if __name__ == "__main__":
    from fake_quant.collect.collect import get_model

    # Load model
    print("Loading model...")
    model, _, _, _, _, _ = get_model(None)

    # Initialize loader with default path
    loader = CalibCacheLoader()
    print(f"Loader initialized with {loader.num_samples} samples.")

    # Test iteration
    print("Starting iteration over layer activations...")

    # Determine the correct module to pass (one that has .layers)
    if hasattr(model, "transformer"):
        # Z-Image model
        target_model = model.transformer
    elif hasattr(model, "language_model"):
        # Bagel model
        target_model = model.language_model
        if not hasattr(target_model, "layers") and hasattr(target_model, "model"):
            target_model = target_model.model
    else:
        target_model = model

    # Define filter function
    from fake_quant.calib.utils import get_needs_inputs_fn
    needs_inputs_fn = get_needs_inputs_fn()

    try:
        for name, (layer, cache, inputs) in loader.iter_layer_activations(
            target_model,
            needs_inputs_fn=needs_inputs_fn
        ):
            print(f"Layer: {name}")
            print(f"  Layer type: {type(layer)}")
            print(f"  Cache keys: {list(cache.keys())}")
            for k,v in cache.items():
                if isinstance(v, IOTensorsCache):
                    print(f"  {k} cached: {v.num_cached_samples} samples")
            if "linear" in cache:
                print(f"  Linear inputs cached: {cache['linear'].inputs.num_cached} samples")

        print("Test finished successfully.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
