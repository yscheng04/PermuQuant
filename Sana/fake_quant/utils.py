from __future__ import annotations

import typing as tp

import numpy as np
import torch


def tree_map(func: tp.Callable[[tp.Any], tp.Any], tree: tp.Any) -> tp.Any:
    """Apply a function to tree-structured data."""
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, v) for v in tree)
    if isinstance(tree, (torch.Tensor, np.ndarray)):
        return func(tree)
    return tree


def tree_copy_with_ref(
    tree: tp.Any,
    /,
    ref: tp.Any,
    copy_func: tp.Callable[[tp.Any, tp.Any], tp.Any] | None = None,
) -> tp.Any:
    """Reuse reference tensors when values are unchanged."""
    if isinstance(tree, dict):
        return {k: tree_copy_with_ref(v, ref[k]) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_copy_with_ref(v, ref[i]) for i, v in enumerate(tree))
    if isinstance(tree, torch.Tensor):
        if not isinstance(ref, torch.Tensor):
            return tree
        if tree.shape != ref.shape:
            return tree
        if tree.data_ptr() == ref.data_ptr() or tree.allclose(ref):
            return ref
        return tree
    if copy_func is not None:
        return copy_func(tree, ref)
    return tree
