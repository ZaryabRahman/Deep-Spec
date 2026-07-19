import numpy as np
from typing import Any
from .exceptions import ShapeMismatchError

def aggregate_layers(attention_tensor: Any, method: str = "mean") -> np.ndarray:
    """
    Reduces a multi-dimensional attention tensor to a 2D text-to-image affinity matrix.
    
    Args:
        attention_tensor (Any): Attention tensor of shape (L, H, T, I), (H, T, I), or (T, I).
                                Accepts NumPy arrays or framework tensors (PyTorch/JAX).
        method (str): Reduction method. Defaults to 'mean' (Semantic Gestalt).

    Returns:
        np.ndarray: A 2D NumPy array of shape (T, I).
    """
    if hasattr(attention_tensor, "detach"):
        attention_tensor = attention_tensor.detach().cpu().numpy()
    elif hasattr(attention_tensor, "numpy"):
        attention_tensor = np.array(attention_tensor)
    
    if not isinstance(attention_tensor, np.ndarray):
        attention_tensor = np.array(attention_tensor)

    ndim = attention_tensor.ndim
    if ndim not in [2, 3, 4]:
        raise ShapeMismatchError(f"Expected attention tensor of 2, 3, or 4 dims. Got {ndim} dims.")

    if method != "mean":
        raise ValueError(f"Aggregation method '{method}' is not supported.")

    if ndim == 4:
        return attention_tensor.mean(axis=(0, 1))
    elif ndim == 3:
        return attention_tensor.mean(axis=0)
    
    return attention_tensor