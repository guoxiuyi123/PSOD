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
from scipy.ndimage import binary_fill_holes


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
    mask_area_ratio: float = 0.0
    compactness: float = 0.0


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

    @staticmethod
    def _mask_to_quality_metrics(mask: np.ndarray, img_h: int, img_w: int) -> Tuple[float, float]:
        mask_area = float(mask.sum())
        image_area = float(img_h * img_w)
        area_ratio = mask_area / image_area if image_area > 0 else 0.0

        perimeter = 0
        mask_uint8 = mask.astype(np.uint8)
        diff_h = np.abs(np.diff(mask_uint8, axis=0))
        diff_w = np.abs(np.diff(mask_uint8, axis=1))
        perimeter = float(diff_h.sum() + diff_w.sum())
        compactness = (4.0 * np.pi * mask_area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        return area_ratio, compactness

    @staticmethod
    def _postprocess_mask(
        mask: np.ndarray,
        center_xy: Tuple[float, float],
        morph_kernel: int = 0,
    ) -> np.ndarray:
        mask_filled = binary_fill_holes(mask.astype(np.uint8)).astype(bool)
        return mask_filled

    def _mask_to_bbox(self, mask: np.ndarray, center_xy: Tuple[float, float]) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
        mask_clean = self._postprocess_mask(mask, center_xy)

        ys, xs = np.where(mask_clean)
        if ys.size == 0:
            return None, mask_clean

        h, w = self._orig_hw
        x1 = max(0.0, float(xs.min()))
        y1 = max(0.0, float(ys.min()))
        x2 = min(float(w), float(xs.max() + 1))
        y2 = min(float(h), float(ys.max() + 1))

        if x2 <= x1 or y2 <= y1:
            return None, mask_clean

        return (x1, y1, x2, y2), mask_clean

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
        mask_raw = (
            np.array(
                Image.fromarray(mask_unpad.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            > 0
        )

        bbox_result, mask = self._mask_to_bbox(mask_raw, point_xy)
        if bbox_result is None:
            raise ValueError("Empty mask after postprocessing")

        x1, y1, x2, y2 = bbox_result
        bbox_xyxy = (x1, y1, x2, y2)
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
        score = float(pred_scores[best].item())
        area_ratio, compactness = self._mask_to_quality_metrics(mask, h, w)
        return SamPointResult(
            mask=mask, bbox_xyxy=bbox_xyxy, bbox_xywh=bbox_xywh,
            score=score, mask_area_ratio=area_ratio, compactness=compactness,
        )

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

    def _decode_mask_at_index(self, mask_logits, idx, point_xy):
        from PIL import Image
        h, w = self._orig_hw
        new_h, new_w = self._new_unpad_hw

        m256 = mask_logits[idx].unsqueeze(0).unsqueeze(0)
        m1024 = F.interpolate(m256, (1024, 1024), mode="bilinear", align_corners=False)[0, 0]
        m_bin = (m1024 > float(self.model.mask_threshold)).to(torch.uint8).cpu().numpy()
        m_unpad = m_bin[:new_h, :new_w]
        mask_raw = (
            np.array(
                Image.fromarray(m_unpad.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            > 0
        )
        bbox_result, mask = self._mask_to_bbox(mask_raw, point_xy)
        if bbox_result is None:
            return None
        x1, y1, x2, y2 = bbox_result
        area_ratio, compactness = self._mask_to_quality_metrics(mask, h, w)
        return SamPointResult(
            mask=mask,
            bbox_xyxy=(x1, y1, x2, y2),
            bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
            score=0.0,
            mask_area_ratio=area_ratio,
            compactness=compactness,
        )

    def predict_point_area_guided(
        self,
        point_xy: Tuple[float, float],
        target_area: float,
        point_label: int = 1,
    ) -> SamPointResult:
        if (
            self._im_tensor is None
            or self._features is None
            or self._orig_hw is None
            or self._r is None
            or self._new_unpad_hw is None
        ):
            raise RuntimeError("Call set_image(...) before predict_point_area_guided(...)")

        x, y = float(point_xy[0]), float(point_xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Invalid point: {point_xy}")

        points = torch.tensor([[x, y]], dtype=torch.float32, device=self.device) * float(self._r)
        labels = torch.tensor([int(point_label)], dtype=torch.int32, device=self.device)
        points_t = (points[:, None, :], labels[:, None])

        with torch.no_grad():
            sparse_emb, dense_emb = self.model.prompt_encoder(points=points_t, boxes=None, masks=None)
            pred_masks, pred_scores = self.model.mask_decoder(
                image_embeddings=self._features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

        pred_masks = pred_masks[0]
        pred_scores = pred_scores[0]

        best_result = None
        best_distance = float('inf')
        n_masks = pred_masks.shape[0]

        for i in range(n_masks):
            res = self._decode_mask_at_index(pred_masks, i, point_xy)
            if res is None:
                continue
            _, _, rw, rh = res.bbox_xywh
            mask_area = rw * rh
            score = float(pred_scores[i].item())
            if target_area > 0:
                area_distance = abs(mask_area - target_area) / target_area
            else:
                area_distance = 0.0
            combined = area_distance - score * 0.3
            if combined < best_distance:
                best_distance = combined
                best_result = SamPointResult(
                    mask=res.mask,
                    bbox_xyxy=res.bbox_xyxy,
                    bbox_xywh=res.bbox_xywh,
                    score=score,
                    mask_area_ratio=res.mask_area_ratio,
                    compactness=res.compactness,
                )

        if best_result is None:
            raise ValueError("All mask candidates empty")

        return best_result

    def predict_point_enhanced(
        self,
        point_xy: Tuple[float, float],
        point_label: int = 1,
        num_jitter: int = 5,
        jitter_radius: float = 3.0,
    ) -> SamPointResult:
        if (
            self._im_tensor is None
            or self._features is None
            or self._orig_hw is None
            or self._r is None
            or self._new_unpad_hw is None
        ):
            raise RuntimeError("Call set_image(...) before predict_point_enhanced(...)")

        h, w = self._orig_hw
        best_result = None
        best_quality = -1.0

        candidates = [point_xy]
        cx, cy = point_xy
        for i in range(num_jitter):
            angle = 2.0 * np.pi * i / num_jitter
            jx = cx + jitter_radius * np.cos(angle)
            jy = cy + jitter_radius * np.sin(angle)
            jx = max(0.0, min(float(w), jx))
            jy = max(0.0, min(float(h), jy))
            candidates.append((jx, jy))

        for pt in candidates:
            try:
                res = self.predict_point(pt, point_label=point_label, multimask_output=True)
            except (ValueError, RuntimeError):
                continue

            score = res.score
            area_ratio = res.mask_area_ratio
            compactness = res.compactness

            area_penalty = 0.0
            if area_ratio > 0.5:
                area_penalty = (area_ratio - 0.5) * 2.0
            elif area_ratio < 0.001:
                area_penalty = 0.5

            quality = score * 0.5 + compactness * 0.3 + max(0, 0.2 - area_penalty)

            if quality > best_quality:
                best_quality = quality
                best_result = res

        if best_result is None:
            raise ValueError("All prediction attempts failed")

        return best_result

    def predict_box_refine(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> SamPointResult:
        from PIL import Image

        if (
            self._im_tensor is None
            or self._features is None
            or self._orig_hw is None
            or self._r is None
            or self._new_unpad_hw is None
        ):
            raise RuntimeError("Call set_image(...) before predict_box_refine(...)")

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        if not all(np.isfinite([x1, y1, x2, y2])):
            raise ValueError(f"Invalid box: {bbox_xyxy}")

        h, w = self._orig_hw
        x1 = max(0.0, min(x1, float(w)))
        y1 = max(0.0, min(y1, float(h)))
        x2 = max(0.0, min(x2, float(w)))
        y2 = max(0.0, min(y2, float(h)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box after clamping: ({x1},{y1},{x2},{y2})")

        box_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device) * float(self._r)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=box_tensor, masks=None
            )
            pred_masks, pred_scores = self.model.mask_decoder(
                image_embeddings=self._features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        pred_masks = pred_masks[0]
        pred_scores = pred_scores[0]
        best = int(torch.argmax(pred_scores).item())

        mask_logits_256 = pred_masks[best].unsqueeze(0).unsqueeze(0)
        mask_logits_1024 = F.interpolate(mask_logits_256, (1024, 1024), mode="bilinear", align_corners=False)[0, 0]
        mask_1024 = (mask_logits_1024 > float(self.model.mask_threshold)).to(torch.uint8).cpu().numpy()

        new_h, new_w = self._new_unpad_hw
        mask_unpad = mask_1024[:new_h, :new_w]
        mask_raw = (
            np.array(
                Image.fromarray(mask_unpad.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST),
                dtype=np.uint8,
            )
            > 0
        )

        box_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        bbox_result, mask = self._mask_to_bbox(mask_raw, box_center)
        if bbox_result is None:
            raise ValueError("Empty mask from box-prompt mask after postprocessing")

        bx1, by1, bx2, by2 = bbox_result

        score = float(pred_scores[best].item())
        area_ratio, compactness = self._mask_to_quality_metrics(mask, h, w)
        return SamPointResult(
            mask=mask,
            bbox_xyxy=(bx1, by1, bx2, by2),
            bbox_xywh=(bx1, by1, bx2 - bx1, by2 - by1),
            score=score,
            mask_area_ratio=area_ratio,
            compactness=compactness,
        )

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
