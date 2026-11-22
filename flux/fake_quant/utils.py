import typing as tp

import numpy as np
import torch
import enum

class TensorType(enum.Enum):
    """The tensor type."""

    Weights = enum.auto()
    Inputs = enum.auto()
    Outputs = enum.auto()

def infer_view_shape(
    tensor_shape: torch.Size,
    /,
    group_shape: tp.Sequence[int],
    skip_first_dim: bool = False,
) -> torch.Size:
    """Infer the view shape from the tensor shape and the group shape.

    Args:
        tensor_shape (`torch.Size`):
            The tensor shape.
        group_shape (`Sequence[int]`):
            The group shape.
        skip_first_dim (`bool`, *optional*, defaults to `False`):
            Whether to skip the first dimension.

    Returns:
        `torch.Size`:
            The view shape of (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    """
    m, view_shape = len(group_shape) - 1, []
    for i, ts in enumerate(tensor_shape):
        gs = group_shape[min(i, m)]
        gs = ts if gs <= 0 else gs
        view_shape.append(ts // gs)
        view_shape.append(gs)
    if skip_first_dim:
        view_shape[0], view_shape[1] = 1, tensor_shape[0]
    return torch.Size(view_shape)

def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int

def tree_map(func: tp.Callable[[tp.Any], tp.Any], tree: tp.Any) -> tp.Any:
    """Apply a function to tree-structured data."""
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, v) for v in tree)
    elif isinstance(tree, (torch.Tensor, np.ndarray)):
        return func(tree)
    # 处理 NaiveCache
    # elif isinstance(tree, NaiveCache):
    #     return {
    #         "key_cache": tree_map(func, tree.key_cache),
    #         "value_cache": tree_map(func, tree.value_cache),
    #     }
    else:
        return tree

def tree_copy_with_ref(
    tree: tp.Any, /, ref: tp.Any, copy_func: tp.Callable[[tp.Any, tp.Any], tp.Any] | None = None
) -> tp.Any:
    """Copy tree-structured data with reference."""
    if isinstance(tree, dict):
        return {k: tree_copy_with_ref(v, ref[k]) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_copy_with_ref(v, ref[i]) for i, v in enumerate(tree))
    elif isinstance(tree, torch.Tensor):
        if not isinstance(ref, torch.Tensor):
            return tree
        if tree.shape != ref.shape:
            return tree
        # FIX: Ensure ref is on the same device as tree before comparing
        ref_on_tree_device = ref.to(tree.device)
        if tree.data_ptr() == ref.data_ptr() or tree.allclose(ref_on_tree_device):
            return ref
        else:
            return tree
    elif copy_func is not None:
        return copy_func(tree, ref)
    else:
        return tree
    
def tree_split(tree: tp.Any) -> list[tp.Any]:
    """Split tree-structured data into a list of data samples."""

    def get_batch_size(tree: tp.Any) -> int | None:
        if isinstance(tree, dict):
            for v in tree.values():
                b = get_batch_size(v)
                if b is not None:
                    return b
        elif isinstance(tree, (list, tuple)):
            for samples in tree:
                b = get_batch_size(samples)
                if b is not None:
                    return b
        elif isinstance(tree, torch.Tensor) and tree.ndim > 0:
            return tree.shape[0]
        return None

    def get_batch(tree: tp.Any, batch_id: int, batch_size: int) -> tp.Any:
        if isinstance(tree, dict):
            return {k: get_batch(v, batch_id, batch_size=batch_size) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            return [get_batch(samples, batch_id, batch_size=batch_size) for samples in tree]
        elif isinstance(tree, torch.Tensor) and tree.ndim > 0:
            if tree.shape[0] == batch_size:
                return tree[batch_id : batch_id + 1]
            else:
                return tree
        else:
            return tree

    ret = []
    batch_size = get_batch_size(tree)
    assert batch_size is not None, "Cannot determine batch size"
    for i in range(batch_size):
        ret.append(get_batch(tree, i, batch_size=batch_size))
    return ret
