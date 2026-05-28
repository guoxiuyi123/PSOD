from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RefineConfig:
    min_area_ratio: float = 0.4
    max_area_ratio: float = 3.0
    min_box_side: float = 8.0
    dilation_kernel_ratio: float = 0.03
    max_dilation_iter: int = 5
    max_image_coverage: float = 0.3
    center_crop_ratio: float = 1.5


@dataclass
class RefineResult:
    bbox_xywh: Tuple[float, float, float, float]
    score: float
    method: str
    area_ratio_before: float
    area_ratio_after: float


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _mask_to_xywh(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
    return (x1, y1, x2 - x1, y2 - y1)


def _dilate_mask(mask: np.ndarray, kernel_ratio: float, max_iter: int) -> np.ndarray:
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        return mask

    h, w = mask.shape[:2]
    k = max(3, int(min(h, w) * kernel_ratio))
    if k % 2 == 0:
        k += 1
    struct = np.ones((k, k), dtype=bool)
    result = mask.copy().astype(bool)
    for _ in range(max_iter):
        result = binary_dilation(result, structure=struct)
    return result.astype(mask.dtype)


def _crop_mask_to_center_region(
    mask: np.ndarray,
    center_xy: Tuple[float, float],
    target_size: Tuple[float, float],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    cx, cy = center_xy
    tw, th = target_size
    x1 = max(0, int(cx - tw / 2))
    y1 = max(0, int(cy - th / 2))
    x2 = min(img_w, int(cx + tw / 2))
    y2 = min(img_h, int(cy + th / 2))

    cropped = np.zeros_like(mask)
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def refine_pseudo_box(
    mask: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    gt_area: float,
    img_w: int,
    img_h: int,
    score: float,
    center_xy: Tuple[float, float],
    cfg: RefineConfig,
) -> RefineResult:
    x, y, w, h = bbox_xywh
    pseudo_area = max(0.0, w * h)
    area_ratio = pseudo_area / gt_area if gt_area > 0 else 1.0
    image_area = float(img_w * img_h)
    coverage = pseudo_area / image_area if image_area > 0 else 0.0

    if coverage > cfg.max_image_coverage:
        target_w = np.sqrt(gt_area * cfg.center_crop_ratio)
        target_h = target_w
        cropped_mask = _crop_mask_to_center_region(
            mask, center_xy, (target_w, target_h), img_w, img_h
        )
        cropped_bbox = _mask_to_xywh(cropped_mask)
        cw, ch = cropped_bbox[2], cropped_bbox[3]

        if cw >= cfg.min_box_side and ch >= cfg.min_box_side:
            new_ratio = (cw * ch) / gt_area if gt_area > 0 else 1.0
            return RefineResult(
                bbox_xywh=cropped_bbox,
                score=score,
                method="center_crop",
                area_ratio_before=area_ratio,
                area_ratio_after=new_ratio,
            )

    if area_ratio >= cfg.min_area_ratio and area_ratio <= cfg.max_area_ratio:
        return RefineResult(
            bbox_xywh=bbox_xywh,
            score=score,
            method="no_change",
            area_ratio_before=area_ratio,
            area_ratio_after=area_ratio,
        )

    if area_ratio < cfg.min_area_ratio:
        dilated = _dilate_mask(mask, cfg.dilation_kernel_ratio, cfg.max_dilation_iter)
        dil_bbox = _mask_to_xywh(dilated)
        dw, dh = dil_bbox[2], dil_bbox[3]

        if dw >= cfg.min_box_side and dh >= cfg.min_box_side:
            new_ratio = (dw * dh) / gt_area if gt_area > 0 else 1.0
            if new_ratio > area_ratio:
                return RefineResult(
                    bbox_xywh=dil_bbox,
                    score=score,
                    method="dilation_expand",
                    area_ratio_before=area_ratio,
                    area_ratio_after=new_ratio,
                )

        scale = min(2.0, cfg.min_area_ratio / max(area_ratio, 0.01))
        cx, cy = center_xy
        new_w = w * scale
        new_h = h * scale
        nx = max(0.0, cx - new_w / 2.0)
        ny = max(0.0, cy - new_h / 2.0)
        nx2 = min(float(img_w), nx + new_w)
        ny2 = min(float(img_h), ny + new_h)
        expanded = (nx, ny, max(0.0, nx2 - nx), max(0.0, ny2 - ny))
        new_ratio = (expanded[2] * expanded[3]) / gt_area if gt_area > 0 else 1.0
        return RefineResult(
            bbox_xywh=expanded,
            score=score,
            method="bbox_expand",
            area_ratio_before=area_ratio,
            area_ratio_after=new_ratio,
        )

    if area_ratio > cfg.max_area_ratio:
        target_w = np.sqrt(gt_area * cfg.center_crop_ratio)
        target_h = target_w
        cropped_mask = _crop_mask_to_center_region(
            mask, center_xy, (target_w, target_h), img_w, img_h
        )
        cropped_bbox = _mask_to_xywh(cropped_mask)
        cw, ch = cropped_bbox[2], cropped_bbox[3]

        if cw >= cfg.min_box_side and ch >= cfg.min_box_side:
            new_ratio = (cw * ch) / gt_area if gt_area > 0 else 1.0
            if new_ratio < area_ratio:
                return RefineResult(
                    bbox_xywh=cropped_bbox,
                    score=score,
                    method="center_crop",
                    area_ratio_before=area_ratio,
                    area_ratio_after=new_ratio,
                )

    return RefineResult(
        bbox_xywh=bbox_xywh,
        score=score,
        method="no_change",
        area_ratio_before=area_ratio,
        area_ratio_after=area_ratio,
    )
