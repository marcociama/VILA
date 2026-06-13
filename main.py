"""
VILA — Visual Interactive Latent Alignment
Entry point for wallet scene registration and authentication.

Usage:
  python main.py register   --image my_scene.jpg --vault wallet.vila
  python main.py auth       --image new_photo.jpg --vault wallet.vila
  python main.py visualize  --image my_scene.jpg
  python main.py demo       --scene1 photo1.jpg --scene2 photo2.jpg

The .vila vault file stores the 384-dim latent vector of your registered scene.
It is useless without the matching image — the vector alone cannot derive a key.
"""

import argparse
from pathlib import Path

import torch

from model import VilaEncoder
from eval import encode_image, authenticate, load_image, scene_test
from visualize import plot_attention_rollout, plot_authentication_result


# ── Device ────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Vault I/O ─────────────────────────────────────────────────────────────────

def save_vault(vector: torch.Tensor, path: Path):
    """Save latent vector to disk as a .vila vault file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vector, "vila_version": "1.0"}, path)
    print(f"Vault saved: {path}  ({vector.shape[0]}-dim latent vector)")


def load_vault(path: Path) -> torch.Tensor:
    """Load latent vector from a .vila vault file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vault not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["vector"]


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_register(args, model, device):
    """Register a scene: image → latent vector → save to vault."""
    print(f"\nRegistering scene from: {args.image}")
    vector = encode_image(model, args.image, device)
    save_vault(vector, args.vault)
    print("\nScene registered. Keep your vault file safe.")
    print("Your recovery method: photograph this same scene again.")

    if not args.no_viz:
        plot_attention_rollout(model, args.image, device,
                               title="Registration scene",
                               save_as="register_rollout.png")


def cmd_auth(args, model, device):
    """Authenticate: compare new photo against saved vault."""
    print(f"\nAuthenticating scene from: {args.image}")
    vector_new   = encode_image(model, args.image, device)
    vector_saved = load_vault(args.vault)

    granted, sim = authenticate(vector_new, vector_saved, threshold=args.threshold)

    print(f"\nSimilarity score : {sim:.4f}")
    print(f"Threshold        : {args.threshold}")
    print(f"Result           : {'✓ WALLET RESTORED' if granted else '✗ SCENE NOT RECOGNIZED'}\n")

    if not args.no_viz:
        plot_authentication_result(model, args.image, device,
                                   similarity=sim, granted=granted,
                                   threshold=args.threshold,
                                   save_as="auth_result.png")


def cmd_visualize(args, model, device):
    """Visualize attention rollout for a single image."""
    print(f"\nVisualizing attention for: {args.image}")
    plot_attention_rollout(model, args.image, device,
                           title=args.title or "",
                           save_as="attention_rollout.png")


def cmd_demo(args, model, device):
    """
    Demo: register scene1, then authenticate scene2.
    Shows whether scene2 is recognized as the same scene as scene1.
    """
    print("\n── VILA Demo ──")
    print(f"  Scene 1 (registration): {args.scene1}")
    print(f"  Scene 2 (authentication): {args.scene2}")

    v1 = encode_image(model, args.scene1, device)
    v2 = encode_image(model, args.scene2, device)

    granted, sim = authenticate(v2, v1, threshold=args.threshold)

    print(f"\n  Similarity: {sim:.4f}  (threshold: {args.threshold})")
    print(f"  Result: {'✓  SAME SCENE — access granted' if granted else '✗  DIFFERENT SCENE — access denied'}\n")

    plot_attention_rollout(model, args.scene1, device,
                           title="Scene 1 — Registration",
                           save_as="demo_scene1.png")
    plot_attention_rollout(model, args.scene2, device,
                           title="Scene 2 — Authentication attempt",
                           save_as="demo_scene2.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VILA — Visual Interactive Latent Alignment")
    p.add_argument("--no-pretrained", action="store_true")
    sub = p.add_subparsers(dest="command")

    # register
    r = sub.add_parser("register", help="Register a scene to a vault file")
    r.add_argument("--image",  required=True, help="Path to scene photo")
    r.add_argument("--vault",  default="wallet.vila", help="Output vault file (.vila)")
    r.add_argument("--no-viz", action="store_true")

    # auth
    a = sub.add_parser("auth", help="Authenticate against a saved vault")
    a.add_argument("--image",     required=True, help="Path to new scene photo")
    a.add_argument("--vault",     default="wallet.vila")
    a.add_argument("--threshold", type=float, default=0.82)
    a.add_argument("--no-viz",    action="store_true")

    # visualize
    v = sub.add_parser("visualize", help="Show attention rollout for an image")
    v.add_argument("--image", required=True)
    v.add_argument("--title", default="")

    # demo
    d = sub.add_parser("demo", help="Register scene1, authenticate scene2")
    d.add_argument("--scene1",    required=True)
    d.add_argument("--scene2",    required=True)
    d.add_argument("--threshold", type=float, default=0.82)

    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device()

    print(f"VILA  |  device={device}")
    model = VilaEncoder(pretrained=not args.no_pretrained).to(device)
    model.eval()

    if args.command == "register":
        cmd_register(args, model, device)
    elif args.command == "auth":
        cmd_auth(args, model, device)
    elif args.command == "visualize":
        cmd_visualize(args, model, device)
    elif args.command == "demo":
        cmd_demo(args, model, device)
    else:
        print("No command specified. Use: register | auth | visualize | demo")
        print("Example: python main.py register --image my_scene.jpg")


if __name__ == "__main__":
    main()
