"""
VILA — Threshold Calibration

Measures three similarity distributions:
  1. Intra-scene:  same image, augmented (simulates "same objects, new photo")
  2. Intra-class:  different images of the same STL-10 category
  3. Inter-class:  images from completely different categories

The ideal threshold sits in the gap between intra-scene (should be HIGH)
and intra-class / inter-class (should be LOW).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

from model import VilaEncoder
from main import select_device

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_PAIRS   = 300   # pairs per distribution
N_IMAGES  = 150   # raw images to load

# ── augmentation that simulates "same scene, new photo" ───────────────────────
_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def load_stl10(bin_path: str, n: int) -> list[np.ndarray]:
    data = np.frombuffer(open(bin_path, "rb").read(), dtype=np.uint8)
    imgs = data.reshape(-1, 3, 96, 96)[:n]
    return [imgs[i].transpose(1,2,0) for i in range(n)]  # list of [H,W,3]


def encode_pil(model, pil_img, transform, device):
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.normalize(model.encode(t), dim=1).squeeze(0).cpu()


def main():
    device = select_device()
    model  = VilaEncoder(pretrained=True).to(device).eval()

    print(f"Loading {N_IMAGES} STL-10 images…")
    raw   = load_stl10("data/stl10_binary/train_X.bin", N_IMAGES)
    labels = np.frombuffer(open("data/stl10_binary/train_y.bin","rb").read(), dtype=np.uint8)[:N_IMAGES]

    pil_imgs = [Image.fromarray(arr) for arr in raw]

    print("Encoding base features…")
    base_feats = torch.stack([encode_pil(model, img, _base, device) for img in pil_imgs])

    # ── 1. Intra-scene: same image, augmented twice ────────────────────────────
    print("Computing intra-scene similarities (augmented views)…")
    intra_scene = []
    for _ in range(N_PAIRS):
        idx = random.randrange(N_IMAGES)
        pil = pil_imgs[idx]
        v1 = encode_pil(model, pil, _aug, device)
        v2 = encode_pil(model, pil, _aug, device)
        intra_scene.append(float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))))

    # ── 2. Intra-class: different images, same label ───────────────────────────
    print("Computing intra-class similarities…")
    intra_class = []
    by_class = {c: np.where(labels == c)[0].tolist() for c in np.unique(labels)}
    for _ in range(N_PAIRS):
        cls   = random.choice(list(by_class.keys()))
        idxs  = random.sample(by_class[cls], min(2, len(by_class[cls])))
        if len(idxs) < 2:
            continue
        v1, v2 = base_feats[idxs[0]], base_feats[idxs[1]]
        intra_class.append(float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))))

    # ── 3. Inter-class: different labels ──────────────────────────────────────
    print("Computing inter-class similarities…")
    inter_class = []
    all_classes = list(by_class.keys())
    for _ in range(N_PAIRS):
        c1, c2 = random.sample(all_classes, 2)
        i1 = random.choice(by_class[c1])
        i2 = random.choice(by_class[c2])
        v1, v2 = base_feats[i1], base_feats[i2]
        inter_class.append(float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))))

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(name, vals):
        a = np.array(vals)
        print(f"\n{name}")
        print(f"  mean={a.mean():.3f}  std={a.std():.3f}  min={a.min():.3f}  max={a.max():.3f}  p5={np.percentile(a,5):.3f}  p95={np.percentile(a,95):.3f}")
        return a

    a_is = stats("Intra-scene  (same image, aug)", intra_scene)
    a_ic = stats("Intra-class  (diff image, same label)", intra_class)
    a_xc = stats("Inter-class  (diff label)", inter_class)

    # Suggested threshold: midpoint between intra-scene p5 and intra-class p95
    gap_lo = np.percentile(a_ic, 95)
    gap_hi = np.percentile(a_is, 5)
    suggested = round((gap_lo + gap_hi) / 2, 3) if gap_hi > gap_lo else 0.75
    print(f"\n→ Intra-class p95:  {gap_lo:.3f}")
    print(f"→ Intra-scene p5:   {gap_hi:.3f}")
    print(f"→ Suggested threshold: {suggested}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(-0.2, 1.05, 60)
    ax.hist(inter_class, bins=bins, alpha=0.7, color="#DC2626", label="Inter-class (different scenes)")
    ax.hist(intra_class, bins=bins, alpha=0.7, color="#F59E0B", label="Intra-class (same category, diff image)")
    ax.hist(intra_scene, bins=bins, alpha=0.8, color="#16A34A", label="Intra-scene (same image, augmented)")

    ax.axvline(suggested, color="#2563EB", linewidth=2.5, linestyle="--", label=f"Suggested threshold: {suggested}")
    ax.set_xlabel("Cosine similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("VILA — Similarity Distribution & Threshold Calibration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.15, 1.05)

    plt.tight_layout()
    out = Path("outputs/threshold_calibration.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")
    print(f"\nRecommended threshold for server.py: THRESHOLD = {suggested}")


if __name__ == "__main__":
    main()
