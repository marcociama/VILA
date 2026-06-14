"""
VILA — Evaluation & Authentication

Core operations:
  encode_image   — image path/tensor → 384-dim latent vector
  similarity     — cosine similarity between two vectors
  authenticate   — compare new scene against saved vault vector
  scene_test     — measure intra-scene (same scene) vs inter-scene (different) distances
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import VilaEncoder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_image(path: str | Path) -> torch.Tensor:
    """Load image from disk → [1, 3, 224, 224] normalised tensor."""
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0)


def encode_image(
    model: VilaEncoder,
    image: str | Path | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a single image into its 384-dim latent vector.
    Accepts: file path (str/Path) or pre-loaded tensor [1, 3, 224, 224].
    Returns: [384] float tensor on CPU.
    """
    if not isinstance(image, torch.Tensor):
        image = load_image(image)
    model.eval()
    with torch.no_grad():
        vec = model.encode(image.to(device))
    return vec.squeeze(0).cpu()   # [384]


def similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine similarity between two 384-dim vectors. Range: [-1, 1]."""
    return float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())


def encode_full(
    model: VilaEncoder,
    image: str | Path | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (cls [384], patches [N,384]) — both L2-normalised.
    CLS used for fast global comparison; patches stored in vault for cross-attention.
    """
    if not isinstance(image, torch.Tensor):
        image = load_image(image)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        cls, patches = model.encode_with_patches(image)   # [1,384], [1,N,384]
    cls     = F.normalize(cls[0],     dim=-1).cpu()       # [384]
    patches = F.normalize(patches[0], dim=-1).cpu()       # [N,384]
    return cls, patches


def cross_attention_sim(
    patches_reg: torch.Tensor,   # [N,384]  — registration patches (SAM-cleaned)
    patches_auth: torch.Tensor,  # [M,384]  — authentication patches (raw)
    top_k_frac: float = 0.35,
) -> float:
    """
    For each registration patch, find the most similar auth patch.
    Returns the mean best-match similarity across the top-K reg patches
    (ranked by attention score, i.e. the most-attended registration regions).
    Stays entirely in R^384 — no dictionary, not brute-forceable.
    """
    # similarity matrix [N, M]
    sim_matrix = patches_reg @ patches_auth.T          # [N, M]
    best_match = sim_matrix.max(dim=1).values          # [N]  best auth patch per reg patch

    # Focus on the top-K most discriminative reg patches
    K = max(1, int(len(patches_reg) * top_k_frac))
    top_vals, _ = best_match.topk(K)
    return float(top_vals.mean())


def authenticate(
    model: VilaEncoder,
    image: str | Path | torch.Tensor,
    device: torch.device,
    top_k_frac: float = 0.35,
) -> torch.Tensor:
    """
    Foreground-aware encoding: attention-weighted average of top-K patch tokens.
    Background-invariant — focuses on the most attended (salient) regions.
    top_k_frac: fraction of patches to keep (default 35% of 256 = ~90 patches).
    """
    if not isinstance(image, torch.Tensor):
        image = load_image(image)
    image = image.to(device)

    model.eval()
    model.register_attention_hooks()
    model.clear_attention_maps()

    with torch.no_grad():
        _, patches = model.encode_with_patches(image)   # [B, N, 384]

    rollout = model.get_attention_rollout()              # [B, N]
    model.remove_attention_hooks()

    N = patches.shape[1]
    K = max(1, int(N * top_k_frac))
    topk_idx = rollout[0].topk(K).indices               # [K] highest-attention patches

    fg_feat = patches[0][topk_idx].mean(dim=0)          # [384]
    return F.normalize(fg_feat, dim=-1).cpu()


def authenticate(
    vector_new: torch.Tensor,
    vector_saved: torch.Tensor,
    threshold: float = 0.82,
) -> tuple[bool, float]:
    """
    Compare a new scene vector against the saved vault vector.
    Returns (access_granted, similarity_score).
    Default threshold 0.82 — tunable via --threshold flag.
    """
    sim = similarity(vector_new, vector_saved)
    return sim >= threshold, sim


# ── Scene test ─────────────────────────────────────────────────────────────────

def scene_test(
    model: VilaEncoder,
    same_scene_paths: list[str],
    different_scene_paths: list[str],
    device: torch.device,
) -> dict:
    """
    Measure intra-scene similarity (same scene, different photos)
    vs inter-scene similarity (different scenes).

    Good system: intra >> inter.
    """
    vecs_same = [encode_image(model, p, device) for p in same_scene_paths]
    vecs_diff = [encode_image(model, p, device) for p in different_scene_paths]

    intra_sims = []
    for i in range(len(vecs_same)):
        for j in range(i + 1, len(vecs_same)):
            intra_sims.append(similarity(vecs_same[i], vecs_same[j]))

    inter_sims = []
    for va in vecs_same:
        for vb in vecs_diff:
            inter_sims.append(similarity(va, vb))

    result = {
        "intra_mean": float(np.mean(intra_sims)) if intra_sims else None,
        "intra_min":  float(np.min(intra_sims))  if intra_sims else None,
        "inter_mean": float(np.mean(inter_sims)) if inter_sims else None,
        "inter_max":  float(np.max(inter_sims))  if inter_sims else None,
    }

    print("\n=== VILA Scene Similarity Test ===")
    if intra_sims:
        print(f"  Intra-scene (same scene, diff photos): mean={result['intra_mean']:.3f}  min={result['intra_min']:.3f}")
    if inter_sims:
        print(f"  Inter-scene (different scenes):         mean={result['inter_mean']:.3f}  max={result['inter_max']:.3f}")
    print("==================================\n")

    return result
