from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch

from .modules.decoders import MaskDecoder
from .modules.encoders import ImageEncoderViT, PromptEncoder
from .modules.sam import Sam
from .modules.transformer import TwoWayTransformer


def _load_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    ckpt = Path(str(checkpoint))
    if not ckpt.is_file():
        raise FileNotFoundError(str(ckpt))
    with ckpt.open("rb") as f:
        obj = torch.load(f, map_location="cpu")
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(type(obj))
    return obj


def build_sam_vit_h(checkpoint: str | Path | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_sam_vit_l(checkpoint: str | Path | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint: str | Path | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_mobile_sam(checkpoint: str | Path | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=[64, 128, 160, 320],
        encoder_depth=[2, 2, 6, 2],
        encoder_num_heads=[2, 4, 5, 10],
        encoder_global_attn_indexes=None,
        mobile_sam=True,
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint: str | Path | None = None,
    mobile_sam: bool = False,
) -> Sam:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    if mobile_sam:
        from .modules.tiny_encoder import TinyViT

        image_encoder = TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=encoder_embed_dim,
            depths=encoder_depth,
            num_heads=encoder_num_heads,
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
    else:
        image_encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )

    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        state_dict = _load_checkpoint(checkpoint)
        sam.load_state_dict(state_dict, strict=True)

    sam.eval()
    return sam


sam_model_map = {
    "sam_h.pt": build_sam_vit_h,
    "sam_l.pt": build_sam_vit_l,
    "sam_b.pt": build_sam_vit_b,
    "mobile_sam.pt": build_mobile_sam,
}


def build_sam(ckpt: str | Path = "sam_b.pt") -> Sam:
    ckpt_str = str(ckpt)
    model_builder = None
    for k in sam_model_map.keys():
        if ckpt_str.endswith(k):
            model_builder = sam_model_map.get(k)
            break
    if not model_builder:
        raise FileNotFoundError(f"{ckpt_str} is not a supported SAM model. Available models are: {list(sam_model_map.keys())}")
    return model_builder(ckpt)
