"""
VILA — VilaEncoder

Frozen vision backbone (pretrained via timm).
Returns a 384-dim float vector — the latent identity of the scene.
No training. No head. No binarization. The vector IS the key.

Attention rollout (Abnar & Zuidema 2020) exposes which regions
of the scene drove the latent vector — the explainability layer.

Supported backbones:
  "dinov2"  — vit_small_patch14_dinov2  (default, background-invariant)
  "dinov3"  — vit_small_patch16_dinov3
  "vitS8"   — vit_small_patch8_224      (original ImageNet ViT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

BACKBONES = {
    "dinov2": ("vit_small_patch14_dinov2", {"img_size": 224}),
    "dinov3": ("vit_small_patch16_dinov3", {"img_size": 224}),
    "vitS8":  ("vit_small_patch8_224",     {}),
}


class VilaEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, backbone: str = "dinov2"):
        super().__init__()

        model_name, extra_kwargs = BACKBONES[backbone]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # returns CLS embedding [B, 384]
            **extra_kwargs,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.backbone.embed_dim   # 384

        self._attn_maps: list[torch.Tensor] = []
        self._hooks: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the raw 384-dim latent vector — the scene fingerprint."""
        with torch.no_grad():
            return self.backbone(x)   # [B, 384]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — encode a scene into its latent vector."""
        return self.forward(x)

    @torch.no_grad()
    def encode_with_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cls [B,384], patch_tokens [B,N,384]).
        Used by encode_foreground to build a background-invariant feature.
        """
        tokens = self.backbone.forward_features(x)  # [B, N+1, 384]
        return tokens[:, 0], tokens[:, 1:]

    # ── Attention rollout ────────────────────────────────────────────────────

    def register_attention_hooks(self):
        """Hook every transformer block to capture softmax attention weights."""
        self.remove_attention_hooks()
        for block in self.backbone.blocks:
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False

        def _make_hook():
            def _h(_, inp, __):
                if inp[0].dim() == 4:
                    self._attn_maps.append(inp[0].detach().cpu())
            return _h

        for block in self.backbone.blocks:
            self._hooks.append(
                block.attn.attn_drop.register_forward_hook(_make_hook())
            )

    def remove_attention_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attn_maps.clear()

    def clear_attention_maps(self):
        self._attn_maps.clear()

    def get_attention_rollout(self) -> torch.Tensor:
        """
        Attention rollout across all layers.
        Returns CLS→patch spatial attention: [B, num_patches].
        """
        if not self._attn_maps:
            raise RuntimeError("No attention maps. Call register_attention_hooks() first.")

        B, _, N, _ = self._attn_maps[0].shape
        rollout = torch.eye(N).unsqueeze(0).expand(B, -1, -1).clone()

        for attn in self._attn_maps:
            a = attn.mean(dim=1)
            a = a + torch.eye(N)
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = torch.bmm(a, rollout)

        return rollout[:, 0, 1:]   # [B, num_patches]

    def get_last_layer_attention(self) -> torch.Tensor:
        """
        CLS→patch attention from the last transformer block only, averaged over heads.
        Sharper and more object-focused than full rollout — preferred for visualization.
        Returns [B, num_patches].
        """
        if not self._attn_maps:
            raise RuntimeError("No attention maps. Call register_attention_hooks() first.")

        last = self._attn_maps[-1]          # [B, heads, N+1, N+1]
        cls_to_patches = last[:, :, 0, 1:]  # [B, heads, N]
        return cls_to_patches.mean(dim=1)   # [B, N]
