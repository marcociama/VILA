"""
VILA — SAM-guided foreground segmentation for registration.

Pipeline:
  1. GroundingDINO (text → bounding boxes for the declared objects)
  2. SAM (boxes → precise pixel masks)
  3. Union of masks → replace background with neutral gray
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from PIL import Image

# Lazy globals — loaded once on first call
_gdino_proc  = None
_gdino_model = None
_sam_model   = None
_sam_proc    = None

BG_COLOR = (255, 255, 255)   # white — best for DINOv2 stability
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID   = "facebook/sam-vit-base"


def _load_models(device):
    global _gdino_proc, _gdino_model, _sam_model, _sam_proc
    if _gdino_model is None:
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            SamModel,
            SamProcessor,
        )
        print("Loading GroundingDINO…")
        _gdino_proc  = AutoProcessor.from_pretrained(GDINO_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device)
        print("Loading SAM…")
        _sam_proc  = SamProcessor.from_pretrained(SAM_ID)
        _sam_model = SamModel.from_pretrained(SAM_ID)  # CPU — avoids float64 MPS issue
        print("Segmentation models ready.")


def _boxes_from_text(image_pil: Image.Image, words: list[str], device, threshold: float = 0.2):
    """Run GroundingDINO and return bounding boxes [[x1,y1,x2,y2], ...] in pixel coords."""
    W, H = image_pil.size
    img_area = W * H

    text_prompt = " . ".join(w.strip().lower() for w in words) + " ."
    inputs = _gdino_proc(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _gdino_model(**inputs)
    results = _gdino_proc.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    raw_boxes  = results["boxes"].cpu().tolist()
    raw_scores = results["scores"].cpu().tolist()
    labels     = results.get("labels", ["?"] * len(raw_boxes))

    # Filter: discard boxes covering >65% of image (spurious full-image detections)
    boxes = []
    for b, s, l in sorted(zip(raw_boxes, raw_scores, labels), key=lambda x: -x[1]):
        x1, y1, x2, y2 = b
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / img_area > 0.65:
            print(f"  [skip] {l} score={s:.3f} — covers {box_area/img_area*100:.0f}% of image")
            continue
        boxes.append(b)
        print(f"  [keep] {l} score={s:.3f}  box={[round(v) for v in b]}")
        if len(boxes) == 5:   # max 5 boxes
            break

    print(f"[GDINO] '{text_prompt}' → {len(boxes)} valid boxes (from {len(raw_boxes)} raw)")
    return boxes


def _masks_from_boxes(image_pil: Image.Image, boxes: list, device):
    """Run SAM with bounding box prompts and return list of binary numpy masks."""
    if not boxes:
        return []
    input_boxes = [boxes]   # SAM expects list-of-list
    inputs = _sam_proc(image_pil, input_boxes=input_boxes, return_tensors="pt")  # CPU
    with torch.no_grad():
        outputs = _sam_model(**inputs)
    # post_process_masks → list[Tensor[num_boxes, 3, H, W]]
    masks = _sam_proc.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # [num_boxes, 3, H, W]
    # iou_scores: [1, num_boxes, 3] → [num_boxes, 3]
    iou = outputs.iou_scores[0].cpu()          # [num_boxes, 3]
    best = iou.argmax(dim=-1)                  # [num_boxes]
    result = []
    for i in range(len(boxes)):
        result.append(masks[i, best[i].item()].numpy().astype(bool))
    return result


def segment_auto(
    image_pil: Image.Image,
    device: torch.device,
) -> Image.Image:
    """
    Segment salient objects automatically — no text prompt needed.
    Used at authentication: user uploads photo, SAM finds objects silently.
    """
    _load_models(device)
    image_pil = image_pil.convert("RGB")

    # Generic prompt — finds the most prominent objects in any scene
    boxes = _boxes_from_text(image_pil, ["object"], device, threshold=0.08)

    if not boxes:
        # Fallback: try even broader prompt
        boxes = _boxes_from_text(image_pil, ["item", "thing"], device, threshold=0.07)

    if not boxes:
        print("[segment_auto] No objects found. Returning original.")
        return image_pil

    masks = _masks_from_boxes(image_pil, boxes, device)
    if not masks:
        return image_pil

    np_img = np.array(image_pil).copy()
    H, W   = np_img.shape[:2]
    fg     = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m.shape == (H, W):
            fg |= m

    if not fg.any():
        return image_pil

    np_img[~fg] = BG_COLOR

    # Crop + pad to square
    rows, cols = np.where(fg)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    pad_y = int((y2 - y1) * 0.10);  pad_x = int((x2 - x1) * 0.10)
    y1 = max(0, y1 - pad_y);  y2 = min(H, y2 + pad_y)
    x1 = max(0, x1 - pad_x);  x2 = min(W, x2 + pad_x)
    crop   = np_img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    side   = max(ch, cw)
    square = np.full((side, side, 3), BG_COLOR[0], dtype=np.uint8)
    oy = (side - ch) // 2;  ox = (side - cw) // 2
    square[oy:oy+ch, ox:ox+cw] = crop
    return Image.fromarray(square)


def segment_scene(
    image_pil: Image.Image,
    words: list[str],
    device: torch.device,
    crop_to_object: bool = True,
    pad_frac: float = 0.10,
) -> Image.Image:
    """
    Segment declared objects, replace background with white, crop to object bbox.
    Cropping normalises object size across photos taken at different distances.
    Falls back to original image if segmentation fails.
    """
    _load_models(device)

    image_pil = image_pil.convert("RGB")
    words = [w for w in words if w.strip()]
    if not words:
        return image_pil

    boxes = _boxes_from_text(image_pil, words, device)
    if not boxes:
        print(f"[segment] No boxes found for: {words}. Returning original.")
        return image_pil

    masks = _masks_from_boxes(image_pil, boxes, device)
    if not masks:
        return image_pil

    np_img = np.array(image_pil).copy()
    H, W   = np_img.shape[:2]
    fg     = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m.shape == (H, W):
            fg |= m

    if not fg.any():
        return image_pil

    # Replace background with white
    np_img[~fg] = BG_COLOR

    if not crop_to_object:
        return Image.fromarray(np_img)

    # Crop to bounding box of foreground + padding
    rows, cols = np.where(fg)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    pad_y = int((y2 - y1) * pad_frac)
    pad_x = int((x2 - x1) * pad_frac)
    y1 = max(0, y1 - pad_y);  y2 = min(H, y2 + pad_y)
    x1 = max(0, x1 - pad_x);  x2 = min(W, x2 + pad_x)

    # Pad to square so resize doesn't distort aspect ratio
    crop = np_img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    side   = max(ch, cw)
    square = np.full((side, side, 3), BG_COLOR[0], dtype=np.uint8)
    oy     = (side - ch) // 2
    ox     = (side - cw) // 2
    square[oy:oy+ch, ox:ox+cw] = crop

    return Image.fromarray(square)
