import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import base64
import io
import uuid
import tempfile

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from eval import encode_image, encode_full, cross_attention_sim, authenticate
from model import VilaEncoder
from main import select_device
from visualize import plot_attention_rollout
from ui.backend.segment import segment_scene, segment_auto

THRESHOLD = 0.50  # empirical — DINOv2 CLS encodes full scene (rotation, light, angle)
FRONTEND  = Path(__file__).parent.parent / "frontend"
OUTPUTS   = Path(__file__).parent.parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI()

device = select_device()
model  = VilaEncoder(pretrained=True).to(device)
model.eval()


def _parse_words(raw: str) -> list[str]:
    return [w.strip() for w in raw.replace(",", " ").split() if w.strip()]


def _apply_sam(img_pil: Image.Image, words: list[str], tmp_path: Path) -> Path:
    """Segment with SAM, save result, return path. Falls back to original on failure."""
    if not words:
        return tmp_path
    cleaned = segment_scene(img_pil, words, device)
    clean_path = tmp_path.with_suffix(".sam.jpg")
    cleaned.save(clean_path)
    return clean_path


@app.get("/", response_class=HTMLResponse)
async def root():
    return (FRONTEND / "index.html").read_text()


@app.post("/api/register")
async def register(
    image:   UploadFile = File(...),
    objects: str        = Form(default=""),
):
    suffix = Path(image.filename or "scene.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)

    clean_path = None
    try:
        words      = _parse_words(objects)
        print(f"[register] words={words}", flush=True)
        img_pil    = Image.open(tmp_path).convert("RGB")
        encode_path = _apply_sam(img_pil, words, tmp_path)
        if encode_path != tmp_path:
            clean_path = encode_path

        cls, patches = encode_full(model, encode_path, device)

        save_as = f"reg_{uuid.uuid4().hex[:8]}.png"
        plot_attention_rollout(model, encode_path, device, title="Registration scene", save_as=save_as)
        rollout_bytes = (OUTPUTS / save_as).read_bytes()
        (OUTPUTS / save_as).unlink(missing_ok=True)

        buf = io.BytesIO()
        # Vault: CLS for fast compare + patches for cross-attention. No words — security.
        torch.save({"vector": cls, "patches": patches, "vila_version": "1.0"}, buf)

        return {
            "rollout_b64": base64.b64encode(rollout_bytes).decode(),
            "vault_b64":   base64.b64encode(buf.getvalue()).decode(),
        }
    finally:
        tmp_path.unlink(missing_ok=True)
        if clean_path:
            clean_path.unlink(missing_ok=True)


@app.post("/api/auth")
async def auth(image: UploadFile = File(...), vault_b64: str = Form(...)):
    suffix = Path(image.filename or "scene.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)

    try:
        vault = torch.load(io.BytesIO(base64.b64decode(vault_b64)), weights_only=False)

        cls_new, patches_new = encode_full(model, tmp_path, device)

        # CLS similarity (global scene)
        sim_cls = float(F.cosine_similarity(cls_new.unsqueeze(0), vault["vector"].unsqueeze(0)))

        # Cross-attention: reg patches (SAM-clean) vs auth patches (raw) — stays in R^384
        sim_xattn = cross_attention_sim(vault["patches"], patches_new) if "patches" in vault else sim_cls

        # Combined score: weighted average — cross-attention compensates background drift
        sim = round(0.5 * sim_cls + 0.5 * sim_xattn, 4)
        granted = sim >= THRESHOLD

        print(f"[auth] cls={sim_cls:.3f}  xattn={sim_xattn:.3f}  combined={sim:.3f}  {'GRANTED' if granted else 'DENIED'}", flush=True)

        save_as = f"auth_{uuid.uuid4().hex[:8]}.png"
        plot_attention_rollout(model, tmp_path, device, title="Authentication scene", save_as=save_as)
        rollout_bytes = (OUTPUTS / save_as).read_bytes()
        (OUTPUTS / save_as).unlink(missing_ok=True)

        return {
            "granted":     granted,
            "similarity":  round(sim, 4),
            "threshold":   THRESHOLD,
            "rollout_b64": base64.b64encode(rollout_bytes).decode(),
        }
    finally:
        tmp_path.unlink(missing_ok=True)
        if 'clean_path' in locals():
            clean_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
