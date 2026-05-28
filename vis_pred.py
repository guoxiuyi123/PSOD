from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _color_from_id(cid: int) -> Tuple[int, int, int]:
    x = (cid * 1103515245 + 12345) & 0x7FFFFFFF
    b = 64 + (x & 127)
    g = 64 + ((x >> 7) & 127)
    r = 64 + ((x >> 14) & 127)
    return int(b), int(g), int(r)


def _iter_image_ids(preds: List[Dict[str, Any]]) -> Iterable[int]:
    seen = set()
    for p in preds:
        image_id = p.get("image_id")
        if isinstance(image_id, int) and image_id not in seen:
            seen.add(image_id)
            yield image_id


def _group_preds(preds: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for p in preds:
        image_id = p.get("image_id")
        if not isinstance(image_id, int):
            continue
        grouped.setdefault(image_id, []).append(p)
    return grouped


def _group_gts(coco: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if not isinstance(image_id, int):
            continue
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def _images_index(coco: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for img in coco.get("images", []):
        image_id = img.get("id")
        if isinstance(image_id, int):
            out[image_id] = img
    return out


def _draw_xywh(img, bbox: List[float], color: Tuple[int, int, int], thickness: int) -> None:
    x, y, w, h = bbox
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p1, p2, color, thickness)


def _draw_point(img, x: float, y: float, color: Tuple[int, int, int], radius: int) -> None:
    cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", type=str, required=True)
    parser.add_argument("--ann-file", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()

    pred_json = Path(args.pred_json).expanduser().resolve()
    ann_file = Path(args.ann_file).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = _load_json(pred_json)
    coco = _load_json(ann_file)

    if not isinstance(preds, list):
        raise ValueError("pred-json 必须是 list[dict] 格式")

    pred_by_img = _group_preds(preds)
    gt_by_img = _group_gts(coco)
    img_index = _images_index(coco)

    written = 0
    for image_id in _iter_image_ids(preds):
        if written >= int(args.max_images):
            break

        info = img_index.get(image_id)
        if not info:
            continue

        file_name = info.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue

        img_path = images_dir / file_name
        if not img_path.exists():
            img_path = images_dir / Path(file_name).name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        for ann in gt_by_img.get(image_id, []):
            bbox = ann.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            _draw_xywh(img, bbox, (0, 220, 0), 2)
            x, y, w, h = bbox
            _draw_point(img, x + w * 0.5, y + h * 0.5, (0, 220, 0), 3)

        dets = pred_by_img.get(image_id, [])
        dets = [d for d in dets if float(d.get("score", 0.0)) >= float(args.score_thr)]
        dets.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        dets = dets[: int(args.topk)]

        for det in dets:
            bbox = det.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            cid = int(det.get("category_id", 0))
            _draw_xywh(img, bbox, _color_from_id(cid), 2)

        safe_name = f"{image_id:012d}_" + Path(file_name).name
        out_path = out_dir / safe_name
        cv2.imwrite(str(out_path), img)
        written += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

