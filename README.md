# Deep-Spec: Unentangling Vision-Language Models via Spectral Bipartite Graph Partitioning

[![PyPI version](https://badge.fury.io/py/deep-spec.svg)](https://badge.fury.io/py/deep-spec)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deep-Spec** is a zero-shot, post-hoc interpretability framework for Vision-Language Models (VLMs). This repository contains the official Python package and experiment codebase for our manuscript, which is currently under review for the International Journal of Computer Vision (IJCV) - Springer.

## Abstract

While Vision-Language Models (VLMs) demonstrate strong zero-shot capabilities, understanding their internal cross-modal reasoning remains a significant challenge. Current Explainable AI (XAI) methods typically rely on linear aggregations of attention weights to generate 2D spatial heatmaps. However, these spatial heuristics are susceptible to concept entanglement in multi-object scenes and Vision Transformer (ViT) topological artifacts, such as “attention sinks.” 

In this paper, we propose a paradigm shift from spatial aggregation to topological graph partitioning. We introduce Deep-Spec, a zero-shot, post-hoc interpretability framework that reformulates VLM cross-attention as a deep global bipartite graph. By computing the normalized Graph Laplacian, we project multimodal interactions into a continuous Fiedler subspace. Deep-Spec utilizes strict Low-Rank Spectral Truncation (bounding the manifold to the top K principal components) to mathematically act as a spatial low-pass filter—isolating complex semantic structures while explicitly discarding high-frequency ViT topological noise and generative chat-template artifacts. 

Extensive evaluations demonstrate that our framework resolves multimodal attribute binding, captures abstract syntactic compositionality (e.g., action/verb grounding), and visualizes the layer-by-layer mechanistic “Semantic Gestalt.” Furthermore, large-scale quantitative benchmarking (N = 1,000) across datasets (MS-COCO, Flickr30k Entities) and architectures—ranging from fusion encoders (BLIP, BridgeTower, ViLT) to deep causal decoders (LLaVA-1.5 7B, Qwen2.5-VL)—indicates that Deep-Spec provides consistent spatial localization. Operating entirely on the forward pass, the method reduces peak VRAM consumption, executes faster than gradient alternatives, and remains fully compatible with 4-bit quantized hardware where backward-pass methods are structurally blocked. Finally, double-blind human evaluations (N = 1,250) confirm a 79.8% preference for the structural coherence of the generated explanations.


## Installation

Install the package directly from PyPI for regular use, or clone the repository for development and benchmarking:

```bash
pip install deep-spec
```




## How to Use

The package is framework-agnostic and works with raw attention matrices from PyTorch, JAX, or NumPy – no backward pass required.

```python
import torch
from deep_spec import aggregate_layers, DeepSpecExplainer, overlay_heatmap
from PIL import Image

# 1. extract cross-attention from your VLM
# shape: (layers, heads, text_tokens, image_patches)
# example: 12 layers, 8 heads, 15 tokens, 576 patches (24x24 grid)
raw_attention = torch.rand((12, 8, 15, 576))

# 2. aggregate across layers (Semantic Gestalt)
attention_matrix = aggregate_layers(raw_attention, method="mean")

# 3. compute the spectral heatmap
heatmap = DeepSpecExplainer.compute_heatmap(
    attention_matrix=attention_matrix,
    grid_size=(24, 24),
    target_indices=[5],      # token index to ground
    k=5,                     # low-rank spectral truncation
    epsilon=1e-3             # connectivity regularizer
)

# 4. overlay and visualize
image = Image.open("sample_image.jpg")
composite = overlay_heatmap(image, heatmap, alpha=0.5, colormap="jet")
composite.show()

```



## Citation

If you use this code in your research, please cite our manuscript. Citation details will be updated upon publication.
