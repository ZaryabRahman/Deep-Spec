import os
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image

def resolve_compositionality():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)
    
    url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=1000"
    image_path = "/content/dog-cat.jpg"
    
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        
    text = "a tabby cat and a white dog"
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    attn_maps = []
    
    def hook_fn(module, inp, out):
        attn_maps.append(inp[0].detach().cpu())
        
    hooks = []
    for name, module in model.named_modules():
        if "crossattention.self.dropout" in name:
            hooks.append(module.register_forward_hook(hook_fn))
            
    with torch.no_grad():
        model(**inputs)
        
    for hook in hooks:
        hook.remove()
        
    all_layers_attn = torch.stack(attn_maps)
    A_global = all_layers_attn.mean(dim=(0, 1, 2))
    
    input_ids = inputs["input_ids"][0].tolist()
    special_ids = processor.tokenizer.all_special_ids
    
    valid_indices = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
    valid_tokens = [processor.tokenizer.convert_ids_to_tokens(tid) for tid in input_ids if tid not in special_ids]
    
    cat_idx = valid_tokens.index("cat")
    dog_idx = valid_tokens.index("dog")
    
    W_c = A_global[valid_indices, 1:]
    W_c = W_c / (W_c.max() + 1e-10)
    W_c = W_c.numpy()
    
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
    
    grid_size = int(np.sqrt(I))
    
    cat_heatmap = np.abs(denoised_affinity[cat_idx, :]).reshape(grid_size, grid_size)
    cat_heatmap = (cat_heatmap - cat_heatmap.min()) / (cat_heatmap.max() - cat_heatmap.min() + 1e-10)
    
    dog_heatmap = np.abs(denoised_affinity[dog_idx, :]).reshape(grid_size, grid_size)
    dog_heatmap = (dog_heatmap - dog_heatmap.min()) / (dog_heatmap.max() - dog_heatmap.min() + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    def plot_heatmap(ax, heatmap, title):
        ax.imshow(image.resize((384, 384)), alpha=0.6)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((384, 384), resample=Image.BICUBIC)
        ax.imshow(np.array(heatmap_img), cmap='jet', alpha=0.5, vmin=0, vmax=255)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
    plot_heatmap(axes[0], cat_heatmap, "Spectral Binding: 'cat'")
    plot_heatmap(axes[1], dog_heatmap, "Spectral Binding: 'dog'")
    
    plt.suptitle("Resolving Multi-Object Compositionality via Spectral Graph Partitioning", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("multi_object_compositionality.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    resolve_compositionality()
