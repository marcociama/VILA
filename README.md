# VILA
### Visual Interactive Latent Alignment

> *Your personal object is your password.*

---

## The Problem

An estimated **20% of all Bitcoin in existence is permanently inaccessible** — not stolen, not hacked. Just lost.

The culprit is always the same: a misplaced seed phrase.

BIP-39 gives users 12–24 random words in a precise order. When the phone dies, the user must find a piece of paper written years ago, with the exact words, in the exact sequence, with no typos. One forgotten word. One wrong order. Everything gone. Forever.

This is not a niche failure mode. It is the **#1 cause of permanent crypto loss**.

---

## The Insight

Human memory is not good at ordered sequences of abstract words.  
Human memory is excellent at **personal objects**.

You will remember your childhood plushie, your custom photo frame, your unique travel souvenir — in twenty years.  
You will not remember *"horse battery staple correct"* in twenty days.

**VILA replaces the seed phrase with a personal object.**  
Point the camera at something only you own. You're in.  
No paper. No sequence. No typos possible.

---

## How It Works

```
DAILY USE
  Encrypted vault (stored locally or on USB)
  PIN / biometric → decrypts vault → derives private key → wallet accessible

RECOVERY (lost or broken device)
  Photograph your personal object
  VILA encodes it into the same latent vector
  Import vault on new device → wallet restored

BACKUP
  Export encrypted vault to USB
  Store anywhere — useless without your object
```

The secret is **never a string, never a file, never a number to memorize**.  
It is computed on-device at the moment of use and immediately discarded.

---

## Architecture

```
your object photo
      │
      ▼ (registration only)
 GroundingDINO  ←── object names ("plushie frame coin")
      │
      ▼
    SAM  →  white background + bbox crop  →  normalized object image
      │
      ▼
 DINOv2 ViT-S/14  (frozen, 22M params, pretrained)
      │
      ├── CLS token  [384-dim]   ← global scene fingerprint
      └── Patch tokens [256×384] ← local feature map
      │
      ▼
  vault  =  { vector: [384], patches: [256×384] }
  (object names discarded — security)
```

**At authentication (no text input, any background):**
```
new photo → DINOv2 → CLS_new + patches_new
                          │
    ┌─────────────────────┴──────────────────────┐
    │ sim_cls   = cosine(CLS_new, CLS_vault)      │  global match
    │ sim_xattn = cross_attention(patches_vault,  │  object-local match
    │                             patches_new)    │
    └─────────────────────┬──────────────────────┘
                          │
              combined = 0.5 × sim_cls + 0.5 × sim_xattn
                          │
              threshold 0.50  →  GRANTED / DENIED
```

**Cross-attention**: for each vault patch (clean object), find the most similar patch in the auth photo. Compensates for background drift in the global CLS — the object is found even in a noisy scene.

No training. No fine-tuning. No custom loss function.

---

## Explainability: Attention Rollout

When VILA authenticates you, it highlights **which regions of your object drove the latent vector** — last-layer DINOv2 attention, sharp and object-focused.

You don't just get access. You see *why* you got access:

> worn corner of the plushie ✓ &nbsp; specific color gradient ✓ &nbsp; unique texture ✓

No authentication system in the world offers this.  
SHA-256 cannot tell you what it looked at. VILA can.

---

## Security Model

**Two compounding layers:**

| | BIP-39 | VILA |
|---|---|---|
| Secret type | Abstract word sequence | Physical personal object |
| Attack space | Dictionary (~2048 words) | R^384 — continuous, infinite |
| Brute-forceable? | Yes — C(2048,12) combinations | No — inverting a 22M-param network |
| Enumerable from text? | Yes | No — verbal description ≠ visual features |
| Shoulder-surfable? | Yes | No — an object cannot be transcribed |
| Forgettable? | Very | No — episodic memory for personal objects |

**Threat model (honest):**  
VILA uses the *"something you have"* security model — identical to a YubiKey or house key.  
If an attacker physically steals your object AND knows it is your VILA key → risk.  
Mitigation: the object is needed only for recovery, not for daily transactions (PIN handles those).

**Kerckhoffs compliant:** algorithm fully public, security entirely in the object.

---

## Why Not Just a Passphrase?

Even if you use a memorable phrase ("my red plushie"):

- A phrase lives in **text space** — finite, enumerable, shoulder-surfable
- VILA lives in **visual latent space** — the exact hue, wear pattern, texture, and silhouette of YOUR specific object, encoded in 384 continuous dimensions
- A generative model cannot reproduce your object from its description
- Even a photo of a similar object produces a completely different vector

---

## Usage

```bash
pip install torch torchvision timm fastapi uvicorn python-multipart \
            transformers segment-anything pillow matplotlib

# Run the web app (landing page + live demo)
python ui/backend/server.py
# → http://localhost:8000

# CLI (optional)
python main.py register --image my_object.jpg --vault wallet.vila
python main.py auth     --image my_object_new.jpg --vault wallet.vila
python main.py visualize --image my_object.jpg
```

---

## File Structure

```
VILA/
├── model.py         VilaEncoder — DINOv2 backbone + attention rollout
├── eval.py          encode_image, encode_full, cross_attention_sim, authenticate
├── visualize.py     plot_attention_rollout (last-layer, object-focused)
├── main.py          CLI: register | auth | visualize | demo
├── calibrate_threshold.py  Threshold calibration on STL-10 (synthetic)
├── calibrate_real.py       Threshold calibration on real photos
└── ui/
    ├── backend/
    │   ├── server.py    FastAPI — /api/register, /api/auth
    │   └── segment.py   GroundingDINO + SAM segmentation pipeline
    └── frontend/
        └── index.html   Landing page + live demo (vanilla HTML/CSS/JS)
```

---

## Dependencies

```
torch ≥ 2.0  ·  torchvision  ·  timm ≥ 0.9  ·  transformers ≥ 5.0
segment-anything  ·  fastapi  ·  uvicorn  ·  pillow  ·  matplotlib
```

---

## Track

Built for **ctrl/shift Hackathon 2026** — Main Track: *New primitives for identity, ownership, and digital trust.*

> **Your personal object is your password.**  
> The key lives in your home and in your memory — not on a piece of paper.
