from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _split_ids(ids: List[int], val_ratio: float, seed: int) -> Tuple[Set[int], Set[int]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {val_ratio}")

    rng = random.Random(int(seed))
    ids = list(ids)
    rng.shuffle(ids)

    n_val = int(len(ids) * float(val_ratio))
    n_val = max(1, min(len(ids) - 1, n_val))

    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def _index_by_image_id(annotations: Iterable[Dict[str, Any]], keep_image_ids: Set[int]) -> List[Dict[str, Any]]:
    out = []
    for ann in annotations:
        img_id = ann.get("image_id", None)
        if img_id in keep_image_ids:
            out.append(ann)
    return out


def _filter_images(images: Iterable[Dict[str, Any]], keep_image_ids: Set[int]) -> List[Dict[str, Any]]:
    out = []
    for img in images:
        img_id = img.get("id", None)
        if img_id in keep_image_ids:
            out.append(img)
    return out


def split_coco(coco: Dict[str, Any], ratio: float, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))

    image_ids = [int(x["id"]) for x in images if "id" in x]
    train_ids, val_ids = _split_ids(image_ids, ratio, seed)

    def _base() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in ("info", "licenses", "categories"):
            if k in coco:
                out[k] = coco[k]
        out["images"] = []
        out["annotations"] = []
        return out

    train = _base()
    val = _base()

    train["images"] = _filter_images(images, train_ids)
    val["images"] = _filter_images(images, val_ids)

    train["annotations"] = _index_by_image_id(annotations, train_ids)
    val["annotations"] = _index_by_image_id(annotations, val_ids)

    return train, val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--coco-json", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    coco_json = Path(args.coco_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else coco_json.parent

    coco = _load_json(coco_json)
    train, val = split_coco(coco, ratio=float(args.ratio), seed=int(args.seed))

    train_path = out_dir / "train_gt.json"
    val_path = out_dir / "val_gt.json"
    _dump_json(train, train_path)
    _dump_json(val, val_path)

    print(
        json.dumps(
            {
                "in_images": len(coco.get("images", [])),
                "in_annotations": len(coco.get("annotations", [])),
                "train_images": len(train.get("images", [])),
                "train_annotations": len(train.get("annotations", [])),
                "val_images": len(val.get("images", [])),
                "val_annotations": len(val.get("annotations", [])),
                "train_json": str(train_path),
                "val_json": str(val_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

