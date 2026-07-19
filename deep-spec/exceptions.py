class DeepSpecError(Exception):
    """Base exception for all Deep-Spec errors."""
    pass

class ShapeMismatchError(DeepSpecError):
    """Raised when attention matrix dimensions do not align with grid sizes or target indices."""
    pass

class EigenDecompositionError(DeepSpecError):
    """Raised when the Laplacian eigendecomposition fails to converge."""
    pass