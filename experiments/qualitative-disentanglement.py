import torch
import requests
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

warnings.filterwarnings("ignore")

def generate_qualitative_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)
    model.eval()

    url = "https://images.unsplash.com/photo-1537151608828-ea2b11777ee8?q=80&w=1000"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    text = "a cute dog sitting on an orange background"

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0].tolist()
    special_ids = processor.tokenizer.all_special_ids
    valid_indices = [i for i, idx in enumerate(input_ids) if idx not in special_ids]

    grid_size = 24

    def get_raw_attention():
        attn_maps = []
        hooks = []

        def hook_fn(module, inp, out):
            attn_maps.append(inp[0].detach().cpu())

        for name, module in model.named_modules():
            if "crossattention.self.dropout" in name:
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        all_attn = torch.stack(attn_maps)
        avg_attn = all_attn.mean(dim=(0, 1, 2))
        avg_attn = avg_attn[valid_indices, 1:]

        global_hm = avg_attn.mean(dim=0).reshape(grid_size, grid_size)
        global_hm = global_hm / (global_hm.max() + 1e-10)
        return global_hm.numpy()

    def get_rollout_attention():
        attn_maps = []
        hooks = []

        def hook_fn(module, inp, out):
            attn_maps.append(inp[0].detach().cpu())

        for name, module in model.named_modules():
            if "crossattention.self.dropout" in name:
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        all_attn = torch.stack(attn_maps)
        T, I = all_attn.shape[-2], all_attn.shape[-1]
        rollout = torch.eye(T + I)

        for layer_attn in all_attn:
            avg_heads = layer_attn.mean(dim=(0, 1))
            full = torch.zeros((T + I, T + I))
            full[:T, T:] = avg_heads
            full[T:, :T] = avg_heads.T
            full = 0.5 * full + 0.5 * torch.eye(T + I)
            rollout = rollout @ full

        text_to_img_rollout = rollout[:T, T:]
        filtered_rollout = text_to_img_rollout[valid_indices, 1:]
        heatmap = filtered_rollout.mean(dim=0).reshape(grid_size, grid_size)
        heatmap = heatmap / (heatmap.max() + 1e-10)

        return heatmap.numpy()

    def get_gradcam_heatmap():
        activations = None
        gradients = None

        def forward_hook(module, inp, out):
            nonlocal activations
            activations = out[0].detach()

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        target_layer = model.vision_model.encoder.layers[-1]
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        model.zero_grad()
        outputs = model(**inputs)
        itm_score = outputs.itm_score[0, 1]
        itm_score.backward()

        fwd_handle.remove()
        bwd_handle.remove()

        weights = gradients.mean(dim=(0, 1))
        cam = (weights * activations).sum(dim=-1).squeeze(0)
        cam = torch.relu(cam)[1:]
        cam = cam.reshape(grid_size, grid_size)
        cam = cam / (cam.max() + 1e-10)
        
        return cam.detach().cpu().numpy()

    def get_chefer_heatmap():
        attn_maps = []
        grad_maps = []

        def forward_hook(module, inp, out):
            attn = inp[0]
            attn_maps.append(attn)
            attn.retain_grad()
            attn.register_hook(lambda g: grad_maps.append(g))

        hooks = []
        for name, module in model.named_modules():
            if "crossattention.self.dropout" in name:
                hooks.append(module.register_forward_hook(forward_hook))

        model.zero_grad()
        outputs = model(**inputs)
        itm_score = outputs.itm_score[0, 1]
        itm_score.backward()

        for h in hooks:
            h.remove()

        heatmap = torch.zeros_like(attn_maps[0].mean(dim=(0, 1)))
        for attn, grad in zip(attn_maps, grad_maps):
            heatmap += (attn * grad).mean(dim=(0, 1))

        heatmap = heatmap / len(attn_maps)
        heatmap = heatmap[valid_indices, 1:]
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap.mean(dim=0).reshape(grid_size, grid_size)
        heatmap = heatmap / (heatmap.max() + 1e-10)
        
        return heatmap.detach().cpu().numpy()

    def get_deepspec_heatmaps():
        attn_maps = []
        hooks = []
        
        def hook_fn(module, inp, out):
            attn_maps.append(inp[0].detach().cpu())

        for name, module in model.named_modules():
            if "crossattention.self.dropout" in name:
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        all_layers_attn = torch.stack(attn_maps)
        A_global = all_layers_attn.mean(dim=(0, 1, 2))
        A_global = A_global[valid_indices, 1:]
        A_global = A_global / (A_global.max() + 1e-10)
        W_c = A_global.numpy()

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

        global_hm = np.abs(denoised_affinity).mean(axis=0).reshape(grid_size, grid_size)
        global_hm = (global_hm - global_hm.min()) / (global_hm.max() - global_hm.min() + 1e-10)

        return global_hm

    raw_hm = get_raw_attention()
    rollout_hm = get_rollout_attention()
    gradcam_hm = get_gradcam_heatmap()
    chefer_hm = get_chefer_heatmap()
    deepspec_hm = get_deepspec_heatmaps()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    img_resized = image.resize((grid_size * 16, grid_size * 16))

    def overlay(ax, heatmap, title, cmap='jet'):
        ax.imshow(img_resized, alpha=0.6)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_resized.size, resample=Image.BICUBIC)
        ax.imshow(heatmap_img, cmap=cmap, alpha=0.5, vmin=0, vmax=255)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

    overlay(axes[0], raw_hm, "Raw Attention")
    overlay(axes[1], rollout_hm, "Attention Rollout")
    overlay(axes[2], gradcam_hm, "ViT-Grad-CAM")
    overlay(axes[3], chefer_hm, "Attn x Grad (Chefer)")
    overlay(axes[4], deepspec_hm, "Deep-Spec (Ours)")

    plt.tight_layout()
    plt.savefig("qualitative_disentanglement.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generate_qualitative_comparison()
