from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, act_layer=nn.GELU) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


@dataclass
class DinoV2Config:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4


class DinoVisionTransformer(nn.Module):
    def __init__(self, cfg: DinoV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim))
        num_patches = (cfg.img_size // cfg.patch_size) * (cfg.img_size // cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.embed_dim))
        self.blocks = nn.ModuleList([Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, drop_path=0.0) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def _interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        base_patches = int(self.pos_embed.shape[1] - 1)
        if base_patches == int(h * w):
            return self.pos_embed

        pos = self.pos_embed
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        dim = patch_pos.shape[-1]
        gh = gw = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, gh, gw, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, h, w = self.patch_embed(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        reg_tokens = self.register_tokens.expand(b, -1, -1)
        x = torch.cat([cls_tokens, reg_tokens, x], dim=1)
        pos = self._interpolate_pos_embed(x, h, w).to(dtype=x.dtype, device=x.device)
        x[:, :1] = x[:, :1] + pos[:, :1]
        x[:, 1 + reg_tokens.shape[1] :] = x[:, 1 + reg_tokens.shape[1] :] + pos[:, 1:]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        n_reg = int(reg_tokens.shape[1])
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_regtokens": x[:, 1 : 1 + n_reg],
            "x_norm_patchtokens": x[:, 1 + n_reg :],
            "hw": torch.tensor([h, w], device=x.device),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["x_norm_clstoken"]


def build_dinov2_vits14_reg4() -> DinoVisionTransformer:
    return DinoVisionTransformer(DinoV2Config())


def load_dinov2_vits14_reg4(ckpt_path: str, device: torch.device | None = None) -> DinoVisionTransformer:
    m = build_dinov2_vits14_reg4()
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    if device is not None:
        m.to(device=device)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m
