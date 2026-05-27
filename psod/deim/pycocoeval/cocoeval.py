from __future__ import annotations

import copy
from typing import Any, List, Optional

import numpy as np
from pycocotools.cocoeval import COCOeval as _COCOeval

__all__ = ["COCOeval"]


class COCOeval(_COCOeval):
    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        iouType: str = "segm",
        using_tiny_metrice: bool = False,
        max_dets: int = 100,
        **kwargs,
    ):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        md = int(max_dets)
        if md <= 0:
            md = 100
        self.params.maxDets = [1, 10, md]
        self.using_tiny_metrice = bool(using_tiny_metrice)

    def summarize_per_class(self):
        if not isinstance(getattr(self, "eval", None), dict) or "precision" not in self.eval:
            raise RuntimeError("COCOeval.summarize_per_class requires accumulate() to be called first")

        precision = self.eval["precision"]
        if precision is None:
            raise RuntimeError("COCOeval.eval['precision'] is empty")

        area_lbls = list(getattr(self.params, "areaRngLbl", ["all", "small", "medium", "large"]))
        max_det_idx = -1

        ious = np.asarray(getattr(self.params, "iouThrs", []), dtype=np.float32)
        iou50_idx = int(np.where(np.isclose(ious, 0.5))[0][0]) if ious.size else 0
        iou75_idx = int(np.where(np.isclose(ious, 0.75))[0][0]) if ious.size else min(5, precision.shape[0] - 1)

        def _area_idx(name: str) -> Optional[int]:
            try:
                return int(area_lbls.index(name))
            except ValueError:
                return None

        a_all = _area_idx("all") if _area_idx("all") is not None else 0
        a_s = _area_idx("small")
        a_m = _area_idx("medium")
        a_l = _area_idx("large")
        a_t = _area_idx("tiny")

        def _mean_valid(x: np.ndarray) -> float:
            x = x.astype(np.float32)
            x = x[x > -1]
            if x.size == 0:
                return -1.0
            v = float(x.mean())
            if not np.isfinite(v):
                return -1.0
            return v

        cat_ids = list(getattr(self.params, "catIds", []))
        rows = []
        for k in range(int(precision.shape[2])):
            cat_id = int(cat_ids[k]) if k < len(cat_ids) else k
            name = str(cat_id)
            coco_gt = getattr(self, "cocoGt", None)
            if coco_gt is not None and hasattr(coco_gt, "cats") and cat_id in coco_gt.cats:
                name = str(coco_gt.cats[cat_id].get("name", name))

            ap = _mean_valid(precision[:, :, k, a_all, max_det_idx])
            ap50 = _mean_valid(precision[iou50_idx, :, k, a_all, max_det_idx])
            ap75 = _mean_valid(precision[iou75_idx, :, k, a_all, max_det_idx])
            aps = _mean_valid(precision[:, :, k, a_s, max_det_idx]) if a_s is not None else -1.0
            apm = _mean_valid(precision[:, :, k, a_m, max_det_idx]) if a_m is not None else -1.0
            apl = _mean_valid(precision[:, :, k, a_l, max_det_idx]) if a_l is not None else -1.0

            if self.using_tiny_metrice:
                apt = _mean_valid(precision[:, :, k, a_t, max_det_idx]) if a_t is not None else -1.0
                rows.append((name, round(ap, 3), round(ap50, 3), round(ap75, 3), round(apt, 3), round(aps, 3), round(apm, 3), round(apl, 3)))
            else:
                rows.append((name, round(ap, 3), round(ap50, 3), round(ap75, 3), round(aps, 3), round(apm, 3), round(apl, 3)))

        return rows

    def evaluate_(self):
        self.evaluate()
        k = len(self.params.catIds) if self.params.useCats else 1
        a = len(self.params.areaRng)
        i = len(self.params.imgIds)
        return np.asarray(self.evalImgs).reshape(k, a, i)

    def create_common_coco_eval(self, img_ids: List[List[int]], eval_imgs: List[np.ndarray]):
        img_ids = [i for p in img_ids for i in p]
        eval_imgs = np.concatenate(eval_imgs, 2) if len(eval_imgs) > 0 else np.zeros((0, 0, 0), dtype=object)
        self.evalImgs = list(eval_imgs.flatten())
        self.params.imgIds = list(img_ids)
        self._paramsEval = copy.deepcopy(self.params)
