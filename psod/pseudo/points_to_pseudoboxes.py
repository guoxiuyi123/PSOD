from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .box_refiner import RefineConfig, refine_pseudo_box


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
    cx, cy = point_xy
    half = float(default_box_size) / 2.0
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(w_img), cx + half)
    y2 = min(float(h_img), cy + half)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


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
    refine: bool = False,  # 🚨 [修改点1] 默认关闭 refine，防止利用 GT Area 导致信息泄露被拒稿！
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

    cfg = RefineConfig()

    groups = _group_annotations_by_image_id(annotations)

    total_anns = 0
    failed_anns = 0
    refined_anns = 0
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

            original_area = float(ann.get("area", 0))
            if original_area <= 0 and isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4:
                original_area = float(ann["bbox"][2]) * float(ann["bbox"][3])

            mask = None
            try:
                if original_area > 0:
                    res = adapter.predict_point_area_guided(point_xy, target_area=original_area)
                else:
                    res = adapter.predict_point_enhanced(point_xy, num_jitter=5, jitter_radius=3.0)
                bbox_xywh = res.bbox_xywh
                score = res.score
                mask = res.mask
                area_ratio = res.mask_area_ratio
                compactness = res.compactness

                if score >= 0.3:
                    prev_bbox = bbox_xywh
                    for refine_round in range(3):
                        x0, y0, w0, h0 = prev_bbox
                        if w0 <= 0 or h0 <= 0:
                            break
                        box_xyxy = (x0, y0, x0 + w0, y0 + h0)
                        try:
                            box_res = adapter.predict_box_refine(box_xyxy)
                        except Exception:
                            break
                        if box_res.score < score * 0.9:
                            break
                        new_bbox = box_res.bbox_xywh
                        prev_w, prev_h = prev_bbox[2], prev_bbox[3]
                        new_w, new_h = new_bbox[2], new_bbox[3]
                        if prev_w > 0 and prev_h > 0:
                            w_change = abs(new_w - prev_w) / prev_w
                            h_change = abs(new_h - prev_h) / prev_h
                            if w_change < 0.02 and h_change < 0.02:
                                break
                        bbox_xywh = new_bbox
                        score = box_res.score
                        mask = box_res.mask
                        area_ratio = box_res.mask_area_ratio
                        compactness = box_res.compactness
                        prev_bbox = new_bbox

            except Exception:
                failed_anns += 1
                bbox_xywh = _fallback_prior_bbox_xywh(
                    ann,
                    point_xy=point_xy,
                    image_wh=(w_img, h_img),
                    default_box_size=float(default_box_size),
                )
                score = 0.0
                area_ratio = 0.0
                compactness = 0.0

            if score < 0.3 or (area_ratio > 0.0 and area_ratio < 0.0005):
                failed_anns += 1
                bbox_xywh = _fallback_prior_bbox_xywh(
                    ann,
                    point_xy=point_xy,
                    image_wh=(w_img, h_img),
                    default_box_size=float(default_box_size),
                )
                score = 0.0

            if refine and original_area > 0 and mask is not None:
                try:
                    refine_res = refine_pseudo_box(
                        mask=mask,
                        bbox_xywh=bbox_xywh,
                        gt_area=original_area,
                        img_w=w_img,
                        img_h=h_img,
                        score=score,
                        center_xy=point_xy,
                        cfg=cfg,
                    )
                    bbox_xywh = refine_res.bbox_xywh
                    if refine_res.method != "no_change":
                        refined_anns += 1
                except Exception:
                    pass

            x, y, w, h = bbox_xywh

            if score >= 0.8 and compactness >= 0.3:
                scale = 1.02
            elif score >= 0.6:
                scale = 1.05
            elif score >= 0.4:
                scale = 1.10
            else:
                scale = 1.15

            if original_area > 0 and w > 0 and h > 0:
                pseudo_area = w * h
                area_diff = pseudo_area / original_area
                if area_diff < 0.5:
                    scale = max(scale, 1.20)
                elif area_diff > 2.0:
                    scale = min(scale, 1.02)

            cx = x + w / 2.0
            cy = y + h / 2.0
            new_w = w * scale
            new_h = h * scale

            x = max(0.0, cx - new_w / 2.0)
            y = max(0.0, cy - new_h / 2.0)
            w = min(float(w_img) - x, new_w)
            h = min(float(h_img) - y, new_h)

            ann["bbox"] = [float(x), float(y), float(w), float(h)]
            ann["area"] = float(max(0.0, w) * max(0.0, h))
            ann["sam_score"] = float(score)
            ann["sam_compactness"] = float(compactness)

        adapter.reset_image()
        processed_images += 1
        processed_image_ids.add(int(image_id))

    if trim and processed_image_ids:
        coco["images"] = [im for im in images if isinstance(im, dict) and im.get("id", None) in processed_image_ids]
        coco["annotations"] = [
            ann for ann in annotations if isinstance(ann, dict) and ann.get("image_id", None) in processed_image_ids
        ]

    filtered_count = 0
    filtered_anns = []
    for ann in coco["annotations"]:
        if not isinstance(ann, dict):
            continue
        s = ann.get("sam_score", 1.0)
        if s < 0.3:
            filtered_count += 1
            continue
        filtered_anns.append(ann)
    coco["annotations"] = filtered_anns

    out_name = output_name
    if out_name is None:
        out_name = coco_json.stem + "_pseudo.json"
    out_path = out_dir / out_name

    _save_json(out_path, coco)
    print(str(out_path))
    print(
        f"images={processed_images} anns={total_anns} failed={failed_anns} refined={refined_anns} filtered={filtered_count}",
        file=sys.stderr,
    )
    return 0