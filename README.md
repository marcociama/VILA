# VILA
### Visual Interactive Latent Alignment

> *Your memory is your backup.*

---

## The Problem

An estimated **20% of all Bitcoin in existence is permanently inaccessible** — not stolen, not hacked. Just lost.

The culprit is always the same: a misplaced seed phrase.

BIP-39 gives users 12–24 random words in a precise order. When the phone dies, the user must find a piece of paper written years ago, with the exact words, in the exact sequence, with no typos. One forgotten word. One wrong order. Everything gone. Forever.

This is not a niche failure mode. It is the **#1 cause of permanent crypto loss**.

---

## The Insight

Human memory is not good at ordered sequences of abstract words.  
Human memory is excellent at **visual scenes and spatial arrangements**.

You will remember *"my marker, a tin can, and a football — arranged on my desk"* in twenty years.  
You will not remember *"horse battery staple correct"* in twenty days.

**VILA replaces the seed phrase with a scene of ordinary, unrelated objects.**  
No paper. No sequence. No typos possible.  
Your recovery key lives in the spatial arrangement of objects — not in a string, not in a file.

---

## How It Works

```
DAILY USE
  Encrypted vault file (stored locally or on USB)
  PIN / biometric → decrypts vault → derives private key → wallet accessible

RECOVERY (lost or broken device)
  Photograph your personal scene
  VILA encodes it into the same latent vector
  Import vault file on new device → wallet restored

BACKUP
  Export encrypted vault to USB drive
  Store it anywhere — useless without your scene
```

The secret is **never a string, never a file, never a number you need to memorize**.  
It is computed on-device at the moment of use and immediately discarded.

---

## Architecture

VILA uses a frozen **ViT-S/8** (Vision Transformer, patch size 8, pretrained on ImageNet via timm) to encode any image into a **384-dimensional latent vector**.

```
your scene photo
      │
      ▼
 ViT-S/8 (frozen, 22M params, pretrained)
      │
      ▼
  [384-dim float vector]  ←  this IS the vault key
      │
      ▼
  cosine similarity with saved vector
      │
      ▼
  similarity ≥ threshold  →  access granted
```

No training. No fine-tuning. No custom loss function.  
The pretrained ViT encodes the full visual composition: shapes, textures, colors, and spatial relationships between objects.  
Two photos of the same arrangement (different angle, different lighting) → similar vectors → same wallet.  
Two photos of different arrangements → distant vectors → access denied.

**Threshold flexibility is by design.** You don't need to reproduce the exact photo — you just need your objects in frame. Small variations in angle, lighting, and distance stay within the cosine similarity threshold. The ViT is tolerant of the natural variation in how you photograph a scene, but intolerant of a different scene entirely.

**We build only the encoder and the vault mechanism.**  
Key derivation, signing, and blockchain interaction use standard BIP-32 infrastructure unchanged.

---

## Explainability: Attention Rollout

When VILA authenticates you, it shows **which regions of your scene drove the latent vector** — the attention rollout from all 12 transformer layers.

You don't just get access. You see *why* you got access:

> "Marker ✓ &nbsp; Can ✓ &nbsp; Football ✓"

No authentication system in the world offers this.  
SHA-256 cannot tell you what it looked at. VILA can.

---

## Why Not Just a Passphrase?

Some wallets support a custom passphrase instead of seed words.

| | Passphrase | VILA |
|---|---|---|
| Exact order required | Yes | No |
| Tolerates variation | No | Yes — any angle, any lighting |
| Lives in dictionary space | Yes — enumerable | No — lives in R^384 |
| Can be shoulder-surfed | Yes | No — a scene cannot be transcribed |
| Memory type required | Verbal sequential | Visual episodic (strongest) |

An attacker who knows your passphrase concept can enumerate combinations in seconds.  
An attacker who knows your scene concept must generate images that land in the correct region of a continuous 384-dimensional space — inverting a 22-million parameter neural network with no known efficient algorithm.

---

## Why Not Just Save the Vector?

The vault file stores the encrypted latent vector.  
Without the matching scene, the encryption cannot be unlocked.  
The vector file alone is cryptographically useless — exactly like a hardware wallet without its PIN.

**Stolen phone / USB?** → Attacker has an encrypted blob. Cannot proceed without your scene.  
**Forgotten scene?** → Same consequence as a forgotten seed phrase. User's responsibility.  
**New phone?** → Import vault from USB, photograph your scene, restored in seconds.

---

## Security Properties

- **No dictionary attack**: object arrangements are not in any vocabulary — knowing the objects (marker, can, football) does not help enumerate the visual configuration
- **No enumeration**: visual latent space is continuous and infinite (R^384); the attack surface is all possible spatial compositions, not a finite word list
- **No digital artifact to steal**: the secret is a physical arrangement in the real world
- **Objects can be completely ordinary**: security does not require rare or personal objects — it comes from the specific, unexpected combination and arrangement
- **Kerckhoffs compliant**: algorithm fully public, security entirely in the spatial configuration
- **Plausible deniability**: nothing proves which scene is your key

---

## vs BIP-39 &nbsp;|&nbsp; vs unforgettable.app

| | BIP-39 | unforgettable.app | VILA |
|---|---|---|---|
| Recovery method | 12 words, exact order | Memorable phrase | Personal visual scene |
| Forgettable? | Very | Somewhat | No — episodic memory |
| Order matters | Yes | Yes | No |
| Enumerable by attacker | Yes | Yes | No |
| Explainability | None | None | Attention rollout |
| Paper required | Yes | Yes | No |

---

## Usage

```bash
pip install torch torchvision timm matplotlib pillow

# Register your scene
python main.py register --image my_scene.jpg --vault wallet.vila

# Recover on a new device
python main.py auth --image my_scene_new_photo.jpg --vault wallet.vila

# Visualize what VILA sees
python main.py visualize --image my_scene.jpg

# Demo: compare two scenes
python main.py demo --scene1 photo1.jpg --scene2 photo2.jpg
```

Vault files (`.vila`) store the encrypted latent vector locally.  
Back up to USB. Import on any device. Recover with your scene.

---

## File Structure

```
VILA/
├── model.py       VilaEncoder — ViT-S/8 backbone + attention rollout
├── eval.py        encode_image, similarity, authenticate, scene_test
├── visualize.py   plot_attention_rollout, plot_authentication_result
├── main.py        CLI: register | auth | visualize | demo
└── motivation.txt Full pitch and security analysis
```

---

## Dependencies

```
torch >= 2.0  ·  torchvision >= 0.15  ·  timm >= 0.9  ·  matplotlib  ·  pillow
```

---

## Track

Built for **ctrl/shift Hackathon 2026** — Main Track: *New primitives for identity, ownership, and digital trust.*

> **Build what comes after the interface.**  
> The interface is a camera. The key is a memory.

---

*VILA — because your home deserves a key you cannot lose.*
