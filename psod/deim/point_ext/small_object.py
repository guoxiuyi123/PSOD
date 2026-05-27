from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from engine.core import register
from engine.deim.pseudo_box_memory import PseudoBoxConfig, PseudoBoxMemory

from .point_sup_criterion import PointSupDEIMCriterionV2 as PointSupDEIMCriterion


def _unpack_sample(inputs: Tuple[Any, ...]) -> Tuple[Any, Dict[str, Any], Any | None]:
    if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 2:
        inputs = tuple(inputs[0])
    if len(inputs) >= 3 and isinstance(inputs[1], dict):
        return inputs[0], inputs[1], inputs[2]
    if len(inputs) >= 2 and isinstance(inputs[1], dict):
        return inputs[0], inputs[1], None
    if len(inputs) == 1 and isinstance(inputs[0], dict):
        return inputs[0], inputs[0], None
    raise ValueError("Unsupported sample format for Point extensions")


def _pack_sample(image: Any, target: Dict[str, Any], dataset: Any | None) -> Any:
    if dataset is None:
        return image, target
    return image, target, dataset


@register()
class PointGuidedCrop(T.Transform):
    def __init__(
        self,
        p: float = 0.8,
        min_crop_size: int = 128,
        max_crop_size: int = 640,
        object_scale: float = 4.0,
        pick_smallest: bool = True,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.min_crop_size = int(min_crop_size)
        self.max_crop_size = int(max_crop_size)
        self.object_scale = float(object_scale)
        self.pick_smallest = bool(pick_smallest)

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1).item() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        image, target, dataset = _unpack_sample(inputs)
        boxes = target.get("boxes", None)
        if boxes is None:
            return _pack_sample(image, target, dataset)

        if not hasattr(boxes, "format"):
            return _pack_sample(image, target, dataset)

        if getattr(boxes.format, "value", "").lower() not in ("xyxy",):
            return _pack_sample(image, target, dataset)

        if int(getattr(boxes, "shape", [0])[0]) == 0:
            return _pack_sample(image, target, dataset)

        h, w = F.get_size(image)
        b = boxes.to(dtype=torch.float32)
        bw = (b[:, 2] - b[:, 0]).clamp(min=1.0)
        bh = (b[:, 3] - b[:, 1]).clamp(min=1.0)
        area = bw * bh

        if self.pick_smallest:
            idx = int(torch.argmin(area).item())
        else:
            probs = (1.0 / area.clamp(min=1.0)).cpu().numpy()
            s = float(probs.sum())
            if s <= 0:
                idx = int(torch.randint(low=0, high=int(b.shape[0]), size=(1,)).item())
            else:
                probs = probs / s
                idx = int(random.choices(range(int(b.shape[0])), weights=probs, k=1)[0])

        cx = float((b[idx, 0] + b[idx, 2]).mul(0.5).item())
        cy = float((b[idx, 1] + b[idx, 3]).mul(0.5).item())
        obj = float(max(bw[idx].item(), bh[idx].item()))

        crop = int(round(obj * self.object_scale))
        crop = max(self.min_crop_size, min(self.max_crop_size, crop))
        crop = min(crop, int(min(h, w)))
        if crop >= int(h) or crop >= int(w):
            return _pack_sample(image, target, dataset)

        left = int(round(cx - crop / 2))
        top = int(round(cy - crop / 2))
        left = max(0, min(int(w) - crop, left))
        top = max(0, min(int(h) - crop, top))

        image = F.crop(image, top=top, left=left, height=crop, width=crop)

        new_target = dict(target)
        new_target["boxes"] = F.crop(boxes, top=top, left=left, height=crop, width=crop)
        if "masks" in new_target:
            new_target["masks"] = F.crop(new_target["masks"], top=top, left=left, height=crop, width=crop)

        return _pack_sample(image, new_target, dataset)


@dataclass
class ScaleAdaptivePseudoBoxConfig(PseudoBoxConfig):
    adaptive_k: int = 1
    adaptive_scale: float = 0.6


class _ScaleAdaptivePseudoBoxMemory(PseudoBoxMemory):
    cfg: ScaleAdaptivePseudoBoxConfig

    def _init_from_points(self, points: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        pts = points.detach().cpu().float()
        n = int(pts.shape[0])
        if n == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        if n == 1:
            wh = torch.full((1, 2), float(cfg.prior_wh), dtype=torch.float32)
            boxes = torch.cat([pts, wh], dim=-1)
            boxes[:, 2:].clamp_(min=cfg.min_wh, max=cfg.max_wh)
            boxes[:, :2].clamp_(min=0.0, max=1.0)
            return boxes

        d = torch.cdist(pts, pts)
        d.fill_diagonal_(float("inf"))
        k = max(1, min(int(cfg.adaptive_k), n - 1))
        knn = d.topk(k=k, largest=False).values
        r = knn.mean(dim=1).clamp(min=float(cfg.min_wh), max=float(cfg.max_wh))

        wh = (r[:, None].repeat(1, 2) * float(cfg.adaptive_scale)).clamp(min=float(cfg.min_wh), max=float(cfg.max_wh))
        boxes = torch.cat([pts, wh], dim=-1)
        boxes[:, :2].clamp_(min=0.0, max=1.0)
        return boxes


@register()
class PointSupDEIMCriterionScaleAdaptiveInit(PointSupDEIMCriterion):
    def __init__(self, *args, adaptive_k: int = 1, adaptive_scale: float = 0.6, pseudo_box=None, **kwargs):
        super().__init__(*args, pseudo_box=pseudo_box, **kwargs)

        cfg = ScaleAdaptivePseudoBoxConfig()
        if isinstance(pseudo_box, dict):
            for k, v in pseudo_box.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        cfg.adaptive_k = int(getattr(cfg, "adaptive_k", adaptive_k))
        cfg.adaptive_scale = float(getattr(cfg, "adaptive_scale", adaptive_scale))

        self.pseudo_box_cfg = cfg
        self.pseudo_box_memory = _ScaleAdaptivePseudoBoxMemory(cfg)


@register()
class PointSupDEIMCriterionScoreThreshSchedule(PointSupDEIMCriterion):
    def __init__(
        self,
        *args,
        score_thresh_start: float = 0.6,
        score_thresh_end: float = 0.3,
        score_thresh_begin_epoch: int = 0,
        score_thresh_end_epoch: int = 12,
        pseudo_box=None,
        **kwargs,
    ):
        super().__init__(*args, pseudo_box=pseudo_box, **kwargs)
        self._score_thresh_start = float(score_thresh_start)
        self._score_thresh_end = float(score_thresh_end)
        self._score_thresh_begin_epoch = int(score_thresh_begin_epoch)
        self._score_thresh_end_epoch = int(score_thresh_end_epoch)

    def _score_thresh_at(self, epoch: int) -> float:
        b = int(self._score_thresh_begin_epoch)
        e = int(self._score_thresh_end_epoch)
        if e <= b:
            return float(self._score_thresh_end)
        if epoch <= b:
            return float(self._score_thresh_start)
        if epoch >= e:
            return float(self._score_thresh_end)
        t = float(epoch - b) / float(e - b)
        return float(self._score_thresh_start + t * (self._score_thresh_end - self._score_thresh_start))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self, "pseudo_box_memory") and hasattr(self.pseudo_box_memory, "cfg"):
            self.pseudo_box_memory.cfg.score_thresh = float(self._score_thresh_at(int(epoch)))


@register()
class PointSupDEIMCriterionScaleAdaptiveInitScoreThreshSchedule(PointSupDEIMCriterionScaleAdaptiveInit):
    def __init__(
        self,
        *args,
        score_thresh_start: float = 0.6,
        score_thresh_end: float = 0.3,
        score_thresh_begin_epoch: int = 0,
        score_thresh_end_epoch: int = 12,
        center_radius_start: float | None = None,
        center_radius_end: float | None = None,
        center_radius_begin_epoch: int = 0,
        center_radius_end_epoch: int = 12,
        max_scale_up_start: float | None = None,
        max_scale_up_end: float | None = None,
        max_scale_up_begin_epoch: int = 0,
        max_scale_up_end_epoch: int = 12,
        adaptive_k: int = 1,
        adaptive_scale: float = 0.6,
        pseudo_box=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            adaptive_k=adaptive_k,
            adaptive_scale=adaptive_scale,
            pseudo_box=pseudo_box,
            **kwargs,
        )
        self._score_thresh_start = float(score_thresh_start)
        self._score_thresh_end = float(score_thresh_end)
        self._score_thresh_begin_epoch = int(score_thresh_begin_epoch)
        self._score_thresh_end_epoch = int(score_thresh_end_epoch)

        self._center_radius_start = None if center_radius_start is None else float(center_radius_start)
        self._center_radius_end = None if center_radius_end is None else float(center_radius_end)
        self._center_radius_begin_epoch = int(center_radius_begin_epoch)
        self._center_radius_end_epoch = int(center_radius_end_epoch)

        self._max_scale_up_start = None if max_scale_up_start is None else float(max_scale_up_start)
        self._max_scale_up_end = None if max_scale_up_end is None else float(max_scale_up_end)
        self._max_scale_up_begin_epoch = int(max_scale_up_begin_epoch)
        self._max_scale_up_end_epoch = int(max_scale_up_end_epoch)

    def _score_thresh_at(self, epoch: int) -> float:
        b = int(self._score_thresh_begin_epoch)
        e = int(self._score_thresh_end_epoch)
        if e <= b:
            return float(self._score_thresh_end)
        if epoch <= b:
            return float(self._score_thresh_start)
        if epoch >= e:
            return float(self._score_thresh_end)
        t = float(epoch - b) / float(e - b)
        return float(self._score_thresh_start + t * (self._score_thresh_end - self._score_thresh_start))

    def _linear_at(self, epoch: int, start: float, end: float, begin_epoch: int, end_epoch: int) -> float:
        b = int(begin_epoch)
        e = int(end_epoch)
        if e <= b:
            return float(end)
        if epoch <= b:
            return float(start)
        if epoch >= e:
            return float(end)
        t = float(epoch - b) / float(e - b)
        return float(start + t * (end - start))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.pseudo_box_memory.cfg.score_thresh = float(self._score_thresh_at(int(epoch)))
        if self._center_radius_start is not None and self._center_radius_end is not None:
            self.pseudo_box_memory.cfg.center_radius = float(
                self._linear_at(
                    int(epoch),
                    self._center_radius_start,
                    self._center_radius_end,
                    self._center_radius_begin_epoch,
                    self._center_radius_end_epoch,
                )
            )
        if self._max_scale_up_start is not None and self._max_scale_up_end is not None:
            self.pseudo_box_memory.cfg.max_scale_up = float(
                self._linear_at(
                    int(epoch),
                    self._max_scale_up_start,
                    self._max_scale_up_end,
                    self._max_scale_up_begin_epoch,
                    self._max_scale_up_end_epoch,
                )
            )
