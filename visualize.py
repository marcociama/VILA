"""
VILA — Visualizations

plot_attention_rollout — overlay attention map on image.
Highlights which scene elements drove the latent vector.
This is VILA's explainability layer: "I recognized Bruno, the ring, the window."
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import VilaEncoder

OUTPUT_DIR = Path("outputs")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])


def _denorm(tensor_chw: np.ndarray) -> np.ndarray:
    """Denormalise [C,H,W] → [H,W,C] clipped to [0,1]."""
    img = tensor_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_attention_rollout(
    model: VilaEncoder,
    image_path: str | Path,
    device: torch.device,
    title: str = "",
    save_as: str = "attention_rollout.png",
) -> np.ndarray:
    """
    Overlay attention rollout heatmap on a single image.
    Returns the attention map as numpy array [H, W].
    Saves the figure to outputs/{save_as}.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_pil   = Image.open(image_path).convert("RGB")
    img_t     = _transform(img_pil).unsqueeze(0).to(device)
    img_np    = _transform(img_pil).numpy()   # [3, 224, 224] normalised

    model.eval()
    model.register_attention_hooks()
    model.clear_attention_maps()

    with torch.no_grad():
        model(img_t)

    rollout   = model.get_attention_rollout()   # [1, num_patches]
    model.remove_attention_hooks()

    n_patches = rollout.shape[1]
    g         = int(n_patches ** 0.5)           # 28 for ViT-S/8 @ 224

    attn_map  = rollout[0].numpy().reshape(g, g)
    attn_map  = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    attn_up   = F.interpolate(
        torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    img_show  = _denorm(img_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_show)
    axes[0].axis("off")
    axes[0].set_title("Input scene", fontsize=11)

    axes[1].imshow(img_show)
    axes[1].imshow(attn_up, cmap="inferno", alpha=0.55)
    axes[1].axis("off")
    axes[1].set_title("What VILA sees", fontsize=11)

    suptitle = title if title else "VILA — Attention Rollout"
    fig.suptitle(
        f"{suptitle}\n\"I recognized these regions to generate your vault key\"",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / save_as
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    return attn_up


def plot_authentication_result(
    model: VilaEncoder,
    image_path: str | Path,
    device: torch.device,
    similarity: float,
    granted: bool,
    threshold: float = 0.82,
    save_as: str = "auth_result.png",
):
    """
    Full authentication visual: scene + attention rollout + result banner.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(image_path).convert("RGB")
    img_t   = _transform(img_pil).unsqueeze(0).to(device)
    img_np  = _transform(img_pil).numpy()

    model.eval()
    model.register_attention_hooks()
    model.clear_attention_maps()

    with torch.no_grad():
        model(img_t)

    rollout = model.get_attention_rollout()
    model.remove_attention_hooks()

    n_patches = rollout.shape[1]
    g         = int(n_patches ** 0.5)
    attn_map  = rollout[0].numpy().reshape(g, g)
    attn_map  = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_up   = F.interpolate(
        torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    img_show = _denorm(img_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_show)
    axes[0].axis("off")
    axes[0].set_title("Input scene", fontsize=11)

    axes[1].imshow(img_show)
    axes[1].imshow(attn_up, cmap="inferno", alpha=0.55)
    axes[1].axis("off")
    axes[1].set_title("Regions driving the key", fontsize=11)

    color   = "#2ecc71" if granted else "#e74c3c"
    verdict = "✓  WALLET RESTORED" if granted else "✗  SCENE NOT RECOGNIZED"
    fig.suptitle(
        f"{verdict}\nSimilarity: {similarity:.3f}  (threshold: {threshold})",
        fontsize=13, fontweight="bold", color=color,
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / save_as
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
