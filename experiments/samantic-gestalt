import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image

def generate_semantic_gestalt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)
    
    url = "https://images.unsplash.com/photo-1537151608828-ea2b11777ee8?q=80&w=1000"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    text = "a cute dog sitting on an orange background"
    
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
    
    input_ids = inputs["input_ids"][0].tolist()
    special_ids = processor.tokenizer.all_special_ids
    
    valid_indices = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
    valid_tokens = [processor.tokenizer.convert_ids_to_tokens(tid) for tid in input_ids if tid not in special_ids]
    target_idx = valid_tokens.index("dog")
    
    layers_to_visualize = [0, 4, 8, 11]
    fig, axes = plt.subplots(1, len(layers_to_visualize), figsize=(20, 5))
    
    for i, layer_idx in enumerate(layers_to_visualize):
        layer_attn = all_layers_attn[layer_idx].mean(dim=(0, 1))
        
        W_c = layer_attn[valid_indices, 1:]
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
        heatmap = np.abs(denoised_affinity[target_idx, :]).reshape(grid_size, grid_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        ax = axes[i]
        ax.imshow(image.resize((384, 384)), alpha=0.6)
        
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((384, 384), resample=Image.BICUBIC)
        
        ax.imshow(np.array(heatmap_img), cmap='jet', alpha=0.5, vmin=0, vmax=255)
        ax.set_title(f"Layer {layer_idx + 1}\nConcept: '{valid_tokens[target_idx]}'", fontsize=14, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("Semantic Gestalt: Dynamic Evolution of Concepts across Transformer Depth", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("semantic_gestalt.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_semantic_gestalt()
