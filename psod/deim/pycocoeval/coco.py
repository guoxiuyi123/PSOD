from __future__ import annotations

from typing import Any, List, Optional

from pycocotools.coco import COCO as _COCO

__all__ = ["COCO"]


class COCO(_COCO):
    @classmethod
    def loadRes(cls, cocoGt: "COCO", anns: Optional[List[Any]] = None):
        if cocoGt is None:
            raise ValueError("cocoGt is required")
        return cocoGt.loadRes(anns)
