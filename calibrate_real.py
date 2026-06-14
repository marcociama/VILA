"""
VILA — Calibrazione soglia su foto reali.

Scenario reale:
  - Registrazione: foto su sfondo bianco + parole → SAM → vettore
  - Autenticazione: foto raw (qualsiasi sfondo) → vettore → compare

Uso:
  1. Crea le cartelle:
       photos/reg/       → 1 foto su sfondo bianco per scena (come faresti alla registrazione)
       photos/auth_same/ → 3-5 foto degli stessi oggetti, sfondo qualsiasi
       photos/auth_diff/ → 3-5 foto di scene completamente diverse

  2. Lancia:
       conda run -n vila python calibrate_real.py --words "libro penna evidenziatore"

  3. Leggi la soglia consigliata nell'output.
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from model import VilaEncoder
from eval import encode_image
from main import select_device
from ui.backend.segment import segment_scene

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def load_paths(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in EXTS]) if d.exists() else []


def encode_with_sam(model, img_path, words, device):
    img = Image.open(img_path).convert("RGB")
    cleaned = segment_scene(img, words, device)
    tmp = Path("/tmp/vila_calib_tmp.jpg")
    cleaned.save(tmp)
    return F.normalize(encode_image(model, tmp, device), dim=0)


def encode_raw(model, img_path, device):
    return F.normalize(encode_image(model, img_path, device), dim=0)


def sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", default="", help="Parole oggetti usate alla registrazione")
    args = parser.parse_args()

    words = [w.strip() for w in args.words.replace(",", " ").split() if w.strip()]

    device = select_device()
    model  = VilaEncoder(pretrained=True, backbone="dinov2").to(device).eval()

    reg_paths       = load_paths(Path("photos/reg"))
    auth_same_paths = load_paths(Path("photos/auth_same"))
    auth_diff_paths = load_paths(Path("photos/auth_diff"))

    if not reg_paths:
        print("\nCrea le cartelle con le foto:")
        print("  photos/reg/        — 1+ foto su sfondo bianco (come registrazione)")
        print("  photos/auth_same/  — 3+ foto stessi oggetti, sfondo qualsiasi")
        print("  photos/auth_diff/  — 3+ foto scene diverse")
        print("\nEsempio:")
        print("  python calibrate_real.py --words 'libro penna evidenziatore'")
        return

    # Encode registrations (with SAM if words given)
    print(f"\nRegistrazione con SAM (parole: {words or 'nessuna'}):")
    reg_vecs = []
    for p in reg_paths:
        v = encode_with_sam(model, p, words, device) if words else encode_raw(model, p, device)
        reg_vecs.append(v)
        print(f"  ✓ {p.name}")

    # Encode auth_same (raw — no SAM)
    print(f"\nAutenticazione stessa scena (raw, {len(auth_same_paths)} foto):")
    same_vecs = [encode_raw(model, p, device) for p in auth_same_paths]
    for p in auth_same_paths:
        print(f"  ✓ {p.name}")

    # Encode auth_diff (raw)
    print(f"\nScene diverse (raw, {len(auth_diff_paths)} foto):")
    diff_vecs = [encode_raw(model, p, device) for p in auth_diff_paths]
    for p in auth_diff_paths:
        print(f"  ✓ {p.name}")

    # Compute similarities
    intra, inter = [], []
    for rv in reg_vecs:
        for sv in same_vecs:
            intra.append(sim(rv, sv))
        for dv in diff_vecs:
            inter.append(sim(rv, dv))

    print(f"\n{'='*50}")
    print("RISULTATI:")
    print(f"  Stessa scena  (reg SAM vs auth raw):")
    print(f"    min={min(intra):.3f}  mean={np.mean(intra):.3f}  max={max(intra):.3f}")
    if inter:
        print(f"  Scene diverse (reg SAM vs auth raw):")
        print(f"    min={min(inter):.3f}  mean={np.mean(inter):.3f}  max={max(inter):.3f}")

    # Threshold: min intra - safety margin
    safety   = 0.05
    threshold = round(min(intra) - safety, 2)
    gap       = min(intra) - (max(inter) if inter else 0)
    print(f"\n  Gap min_intra - max_inter: {gap:.3f}")
    print(f"  → Soglia consigliata: {threshold}  (min_intra={min(intra):.3f} - {safety} margine)")
    print(f"{'='*50}\n")

    # Plot
    if inter:
        Path("outputs").mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        bins = np.linspace(0, 1.05, 40)
        ax.hist(inter, bins=bins, alpha=0.75, color="#DC2626", label="Scene diverse")
        ax.hist(intra, bins=bins, alpha=0.85, color="#16A34A", label="Stessa scena")
        ax.axvline(threshold, color="#2563EB", lw=2.5, linestyle="--",
                   label=f"Soglia: {threshold}")
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Count")
        ax.set_title("VILA — Calibrazione reale (SAM reg / raw auth)")
        ax.legend(); plt.tight_layout()
        out = "outputs/calibration_real.png"
        plt.savefig(out, dpi=150); plt.close()
        print(f"Grafico salvato: {out}")
        print(f"\nAggiorna server.py → THRESHOLD = {threshold}")
