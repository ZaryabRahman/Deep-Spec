import numpy as np
from scipy.linalg import eigh
from typing import List, Tuple, Optional
from .exceptions import ShapeMismatchError, EigenDecompositionError

class DeepSpecExplainer:
    """
    Core mathematical engine for Deep-Spec: Spectral Bipartite Graph Partitioning.
    """

    @staticmethod
    def compute_heatmap(
        attention_matrix: np.ndarray,
        grid_size: Tuple[int, int],
        target_indices: List[int],
        k: int = 5,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Projects cross-attention into a low-rank Fiedler subspace to extract spatial semantics.

        Args:
            attention_matrix (np.ndarray): 2D array of shape (T_tokens, I_patches).
            grid_size (Tuple[int, int]): Height and width of the visual patch grid.
            target_indices (List[int]): Row indices corresponding to the target linguistic concept.
            k (int): Number of principal components for Low-Rank Spectral Truncation.
            epsilon (float): Minimal connectivity regularizer.

        Returns:
            np.ndarray: A 2D normalized spatial heatmap of shape `grid_size`.
        """
        if attention_matrix.ndim != 2:
            raise ShapeMismatchError(f"Attention matrix must be 2D. Got shape {attention_matrix.shape}")

        t_tokens, i_patches = attention_matrix.shape
        grid_h, grid_w = grid_size

        if grid_h * grid_w != i_patches:
            raise ShapeMismatchError(
                f"Grid size {grid_size} ({grid_h * grid_w} patches) does not match "
                f"attention matrix patch dimension ({i_patches})."
            )

        if max(target_indices) >= t_tokens:
            raise ShapeMismatchError("Target index exceeds textual token dimension.")

        w_c = attention_matrix / (attention_matrix.max() + 1e-10)

        n_nodes = t_tokens + i_patches
        w = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        w[:t_tokens, t_tokens:] = w_c
        w[t_tokens:, :t_tokens] = w_c.T
        w += epsilon

        d = np.sum(w, axis=1)
        d = np.maximum(d, 1e-10)
        d_inv_sqrt = 1.0 / np.sqrt(d)

        # Vectorized Laplacian construction: L_sym = I - D^{-1/2} W D^{-1/2}
        l_sym = np.eye(n_nodes, dtype=np.float32) - (w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :])

        try:
            # eigh is highly optimized for symmetric matrices
            _, eigvecs = eigh(l_sym)
        except Exception as e:
            raise EigenDecompositionError(f"Laplacian eigendecomposition failed: {str(e)}")

        k_eff = min(k, t_tokens - 1) if t_tokens > k else 1
        
        if k_eff <= 0:
            return np.zeros(grid_size, dtype=np.float32)

        v_text = eigvecs[:t_tokens, 1:k_eff + 1]
        v_img = eigvecs[t_tokens:, 1:k_eff + 1]

        denoised_affinity = v_text @ v_img.T
        target_affinity = np.abs(denoised_affinity[target_indices, :])

        heatmap = target_affinity.max(axis=0).reshape(grid_size)

        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max - hm_min == 0:
            return heatmap
            
        heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-10)
        
        return heatmap