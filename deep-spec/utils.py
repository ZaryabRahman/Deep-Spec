import numpy as np
from PIL import Image
import matplotlib.cm as cm

def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, colormap: str = "jet") -> Image.Image:
    """
    Overlays a 2D heatmap onto a PIL Image.

    Args:
        image (PIL.Image.Image): The base image.
        heatmap (np.ndarray): 2D heatmap array (values in [0, 1]).
        alpha (float): Transparency of the heatmap overlay.
        colormap (str): Matplotlib colormap string.

    Returns:
        PIL.Image.Image: The composite image.
    """
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    heatmap_img = Image.fromarray(heatmap_colored).resize(image.size, resample=Image.BICUBIC)
    
    composite = Image.blend(image.convert("RGB"), heatmap_img, alpha=alpha)
    return composite