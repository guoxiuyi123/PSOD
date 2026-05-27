from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _extract_point_from_keypoints(keypoints: List[float]) -> Optional[Tuple[float, float]]:
    if not keypoints:
        return None
    if len(keypoints) % 3 != 0:
        if len(keypoints) >= 2:
            return float(keypoints[0]), float(keypoints[1])
        return None
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:
            return float(x), float(y)
    x, y = keypoints[0], keypoints[1]
    return float(x), float(y)


def _fallback_prior_bbox_xywh(
    ann: Dict[str, Any],
    point_xy: Tuple[float, float],
    image_wh: Tuple[int, int],
    default_box_size: float,
) -> Tuple[float, float, float, float]:
    w_img, h_img = image_wh
    if "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w > 1 and h > 1:
            x = max(0.0, min(x, float(w_img)))
            y = max(0.0, min(y, float(h_img)))
            w = max(0.0, min(w, float(w_img) - x))
            h = max(0.0, min(h, float(h_img) - y))
            return x, y, w, h

    cx, cy = point_xy
    half = float(default_box_size) / 2.0
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(w_img), cx + half)
    y2 = min(float(h_img), cy + half)
    x = x1
    y = y1
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return x, y, w, h


def _group_annotations_by_image_id(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id", None)
        if isinstance(image_id, int):
            groups[image_id].append(ann)
    return groups


def run_points_to_pseudoboxes(
    coco_json: Path,
    image_root: Path,
    out_dir: Path,
    weights: Optional[Path],
    device: Optional[str],
    default_box_size: float,
    output_name: Optional[str],
    max_images: Optional[int],
    trim: bool,
) -> int:
    from ..sam_point_adapter import SamPointAdapter

    coco = _load_json(coco_json)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    image_id_to_info: Dict[int, Dict[str, Any]] = {}
    for im in images:
        if isinstance(im, dict) and isinstance(im.get("id", None), int):
            image_id_to_info[int(im["id"])] = im

    adapter = SamPointAdapter(checkpoint=weights, device=device)

    groups = _group_annotations_by_image_id(annotations)

    total_anns = 0
    failed_anns = 0
    processed_images = 0
    processed_image_ids: set[int] = set()
    for image_id in sorted(groups.keys()):
        if max_images is not None and processed_images >= int(max_images):
            break

        anns = groups[image_id]
        im_info = image_id_to_info.get(image_id, None)
        if not im_info:
            continue

        file_name = im_info.get("file_name", None)
        if not isinstance(file_name, str) or not file_name:
            continue

        image_path = image_root / file_name
        if not image_path.exists():
            continue

        from PIL import Image

        with Image.open(str(image_path)) as im:
            im = im.convert("RGB")
            img_rgb = np.array(im)
            w_img, h_img = im.size

        adapter.set_image(img_rgb)
        for ann in anns:
            total_anns += 1
            point_xy: Optional[Tuple[float, float]] = None
            if isinstance(ann.get("keypoints", None), list):
                point_xy = _extract_point_from_keypoints(ann["keypoints"])
            if point_xy is None and isinstance(ann.get("bbox", None), list) and len(ann["bbox"]) == 4:
                x, y, w, h = [float(v) for v in ann["bbox"]]
                point_xy = float(x + w / 2.0), float(y + h / 2.0)
            if point_xy is None:
                continue

            try:
                res = adapter.predict_point(point_xy)
                bbox_xywh = res.bbox_xywh
            except Exception:
                failed_anns += 1
                bbox_xywh = _fallback_prior_bbox_xywh(
                    ann,
                    point_xy=point_xy,
                    image_wh=(w_img, h_img),
                    default_box_size=float(default_box_size),
                )

            x, y, w, h = bbox_xywh
            ann["bbox"] = [float(x), float(y), float(w), float(h)]
            ann["area"] = float(max(0.0, w) * max(0.0, h))

        adapter.reset_image()
        processed_images += 1
        processed_image_ids.add(int(image_id))

    if trim and processed_image_ids:
        coco["images"] = [im for im in images if isinstance(im, dict) and im.get("id", None) in processed_image_ids]
        coco["annotations"] = [
            ann for ann in annotations if isinstance(ann, dict) and ann.get("image_id", None) in processed_image_ids
        ]

    out_name = output_name
    if out_name is None:
        out_name = coco_json.stem + "_pseudo.json"
    out_path = out_dir / out_name

    _save_json(out_path, coco)
    print(str(out_path))
    print(f"images={processed_images} anns={total_anns} failed={failed_anns}", file=sys.stderr)
    return 0

