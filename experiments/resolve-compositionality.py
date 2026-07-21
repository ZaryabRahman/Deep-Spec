import os
import torch
import requests
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image

warnings.filterwarnings("ignore")

def resolve_compositionality_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)
    model.eval()

    image_path = "/content/dog-cat.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=1000"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    text = "a tabby cat and a white dog"
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"][0].tolist()
    special_ids = processor.tokenizer.all_special_ids
    valid_indices = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
    valid_tokens = [processor.tokenizer.convert_ids_to_tokens(tid) for tid in input_ids if tid not in special_ids]

    cat_idx = valid_tokens.index("cat")
    dog_idx = valid_tokens.index("dog")
    grid_size = 24

    attn_maps = []
    hooks = []
    
    def hook_fn(module, inp, out):
        attn_maps.append(inp[0].detach().cpu())
        
    for name, module in model.named_modules():
        if "crossattention.self.dropout" in name:
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(**inputs)

    for hook in hooks:
        hook.remove()

    all_layers_attn = torch.stack(attn_maps)
    A_global = all_layers_attn.mean(dim=(0, 1, 2))

    A_valid = A_global[valid_indices, 1:]
    A_valid = A_valid / (A_valid.max() + 1e-10)

    base_cat = A_valid[cat_idx, :].reshape(grid_size, grid_size).numpy()
    base_cat = (base_cat - base_cat.min()) / (base_cat.max() - base_cat.min() + 1e-10)

    base_dog = A_valid[dog_idx, :].reshape(grid_size, grid_size).numpy()
    base_dog = (base_dog - base_dog.min()) / (base_dog.max() - base_dog.min() + 1e-10)

    W_c = A_valid.numpy()
    T, I = W_c.shape
    N = T + I

    W = np.zeros((N, N))
    W[:T, T:] = W_c
    W[T:, :T] = W_c.T
    W += 1e-5

    D = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    _, eigvecs = eigh(L_sym)

    k = 5
    V_text = eigvecs[:T, 1:k+1]
    V_img = eigvecs[T:, 1:k+1]
    denoised_affinity = V_text @ V_img.T

    ours_cat = np.abs(denoised_affinity[cat_idx, :]).reshape(grid_size, grid_size)
    ours_cat = (ours_cat - ours_cat.min()) / (ours_cat.max() - ours_cat.min() + 1e-10)

    ours_dog = np.abs(denoised_affinity[dog_idx, :]).reshape(grid_size, grid_size)
    ours_dog = (ours_dog - ours_dog.min()) / (ours_dog.max() - ours_dog.min() + 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    img_resized = image.resize((384, 384))

    def plot_heatmap(ax, heatmap, title, subtitle):
        ax.imshow(img_resized, alpha=0.6)
        heatmap_img = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((384, 384), resample=Image.BICUBIC))
        ax.imshow(heatmap_img, cmap='jet', alpha=0.5)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.text(0.5, -0.1, subtitle, size=14, ha="center", transform=ax.transAxes, color='maroon' if 'Bleeding' in subtitle else 'green')
        ax.axis('off')

    plot_heatmap(axes[0, 0], base_cat, "Baseline: 'cat'", "[Concept Bleeding]")
    plot_heatmap(axes[0, 1], base_dog, "Baseline: 'dog'", "[Concept Bleeding]")

    plot_heatmap(axes[1, 0], ours_cat, "Deep-Spec (Ours): 'cat'", "[Strict Binding]")
    plot_heatmap(axes[1, 1], ours_dog, "Deep-Spec (Ours): 'dog'", "[Strict Binding]")

    plt.suptitle("Resolving Multi-Object Compositionality via Spectral Graph Partitioning", fontsize=20, fontweight='bold', y=0.95)
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("compositional_binding.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    resolve_compositionality_comparison()
