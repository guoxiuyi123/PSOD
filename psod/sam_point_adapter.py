from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Optional, Tuple, Union
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F


def _default_checkpoint() -> Path:
    return Path(__file__).resolve().parents[1] / "weights" / "sam_b.pt"


def _ultralytics_assets_url(filename: str) -> str:
    return f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{filename}"


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    part = dst.with_suffix(dst.suffix + ".part")
    resume_pos = int(part.stat().st_size) if part.is_file() else 0
    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        try:
            status = int(getattr(resp, "status", 200))
            if status == 200 and resume_pos > 0:
                part.unlink(missing_ok=True)
                resume_pos = 0
                resp.close()
                resp = urllib.request.urlopen(url, timeout=60)

            mode = "ab" if resume_pos > 0 else "wb"
            with part.open(mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        finally:
            resp.close()
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e

    os.replace(part, dst)


def _ensure_default_checkpoint(checkpoint_path: Path) -> None:
    default_path = _default_checkpoint().resolve()
    if checkpoint_path.resolve() != default_path:
        return
    if checkpoint_path.is_file():
        return
    url = _ultralytics_assets_url(checkpoint_path.name)
    print(f"Downloading {checkpoint_path.name} from {url} -> {checkpoint_path}", file=sys.stderr)
    _download_file(url, checkpoint_path)


@dataclass(frozen=True)
class SamPointResult:
    mask: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]
    score: float


class SamPointAdapter:
    def __init__(
        self,
        checkpoint: Union[str, Path, None] = None,
        device: Optional[str] = None,
    ) -> None:
        from .sam.build import build_sam_vit_b

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if checkpoint is None:
            checkpoint_path = _default_checkpoint()
        else:
            checkpoint_path = Path(str(checkpoint))
            if not checkpoint_path.is_absolute():
                checkpoint_path = (Path.cwd() / checkpoint_path).resolve()

        _ensure_default_checkpoint(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))

        self.model = build_sam_vit_b(checkpoint=checkpoint_path).to(self.device).eval()

        self._im_tensor: Optional[torch.Tensor] = None
        self._features: Optional[torch.Tensor] = None
        self._orig_hw: Optional[Tuple[int, int]] = None
        self._r: Optional[float] = None
        self._new_unpad_hw: Optional[Tuple[int, int]] = None

    def reset_image(self) -> None:
        self._im_tensor = None
        self._features = None
        self._orig_hw = None
        self._r = None
        self._new_unpad_hw = None

    def set_image(self, image: Union[str, Path, np.ndarray]) -> None:
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(str(image)) as im:
                img_rgb = np.array(im.convert("RGB"))
        else:
            if not isinstance(image, np.ndarray):
                raise TypeError(type(image))
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}")
            img_rgb = image

        h, w = img_rgb.shape[:2]
        r = min(1024.0 / h, 1024.0 / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))

        img_resized = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), resample=Image.BILINEAR))
        img_lb = np.zeros((1024, 1024, 3), dtype=img_resized.dtype)
        img_lb[:new_h, :new_w] = img_resized

        im = torch.from_numpy(np.ascontiguousarray(img_lb)).to(self.device)
        im = im.permute(2, 0, 1).unsqueeze(0).float()
        im = (im - self.model.pixel_mean) / self.model.pixel_std

        with torch.no_grad():
            features = self.model.image_encoder(im)

        self._im_tensor = im
        self._features = features
        self._orig_hw = (h, w)
        self._r = r
        self._new_unpad_hw = (new_h, new_w)

    def predict_point(
        self,
        point_xy: Tuple[float, float],
        point_label: int = 1,
        multimask_output: bool = True,
    ) -> SamPointResult:
        from PIL import Image

        if (
            self._im_tensor is None
            or self._features is None
            or self._orig_hw is None
            or self._r is None
            or self._new_unpad_hw is None
        ):
            raise RuntimeError("Call set_image(...) before predict_point(...)")

        x, y = float(point_xy[0]), float(point_xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Invalid point: {point_xy}")

        points = torch.tensor([[x, y]], dtype=torch.float32, device=self.device) * float(self._r)
        labels = torch.tensor([int(point_label)], dtype=torch.int32, device=self.device)
        points_t = (points[:, None, :], labels[:, None])

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points_t, boxes=None, masks=None)
            pred_masks, pred_scores = self.model.mask_decoder(
                image_embeddings=self._features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

        pred_masks = pred_masks[0]
        pred_scores = pred_scores[0]
        best = int(torch.argmax(pred_scores).item())

        mask_logits_256 = pred_masks[best].unsqueeze(0).unsqueeze(0)
        mask_logits_1024 = F.interpolate(mask_logits_256, (1024, 1024), mode="bilinear", align_corners=False)[0, 0]
        mask_1024 = (mask_logits_1024 > float(self.model.mask_threshold)).to(torch.uint8).cpu().numpy()

        new_h, new_w = self._new_unpad_hw
        mask_unpad = mask_1024[:new_h, :new_w]
        h, w = self._orig_hw
        mask = (
            np.array(
                Image.fromarray(mask_unpad.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            > 0
        )

        ys, xs = np.where(mask)
        if ys.size == 0:
            raise ValueError("Empty mask")

        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)

        x1 = max(0.0, min(x1, float(w)))
        y1 = max(0.0, min(y1, float(h)))
        x2 = max(0.0, min(x2, float(w)))
        y2 = max(0.0, min(y2, float(h)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox from mask")

        bbox_xyxy = (x1, y1, x2, y2)
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
        score = float(pred_scores[best].item())
        return SamPointResult(mask=mask, bbox_xyxy=bbox_xyxy, bbox_xywh=bbox_xywh, score=score)

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        point_xy: Tuple[float, float],
        point_label: int = 1,
        multimask_output: bool = True,
    ) -> SamPointResult:
        self.set_image(image)
        try:
            return self.predict_point(point_xy, point_label=point_label, multimask_output=multimask_output)
        finally:
            self.reset_image()

    @staticmethod
    def coco_xywh_to_center_xy(bbox_xywh: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x, y, w, h = bbox_xywh
        return float(x + w / 2.0), float(y + h / 2.0)

    @staticmethod
    def xyxy_to_xywh(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    @staticmethod
    def clamp_xywh(bbox_xywh: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
        x, y, w, h = bbox_xywh
        x = max(0.0, min(float(x), float(width)))
        y = max(0.0, min(float(y), float(height)))
        w = max(0.0, min(float(w), float(width) - x))
        h = max(0.0, min(float(h), float(height) - y))
        return x, y, w, h
