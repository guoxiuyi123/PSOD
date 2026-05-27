from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _psod_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_train_config() -> Path:
    return _psod_root() / "configs" / "nwpu_vhr10" / "deim_hgnetv2_n_pseudo_box.yml"


def _run_vendored_deim(argv: Sequence[str]) -> int:
    from .deim.launcher import run

    return int(run(argv))


def _cmd_pseudo(args: argparse.Namespace) -> int:
    from .pseudo.points_to_pseudoboxes import run_points_to_pseudoboxes

    return run_points_to_pseudoboxes(
        coco_json=Path(args.coco_json),
        image_root=Path(args.image_root),
        out_dir=Path(args.out_dir),
        weights=Path(args.weights) if args.weights is not None else None,
        device=args.device,
        default_box_size=float(args.default_box_size),
        output_name=args.output_name,
        max_images=args.max_images,
        trim=bool(args.trim),
    )


def _cmd_train(args: argparse.Namespace, extra: Sequence[str]) -> int:
    cfg = Path(args.config)
    if not cfg.is_file():
        raise FileNotFoundError(str(cfg))
    forwarded = [x for x in extra if x != "--"]
    return _run_vendored_deim(["-c", str(cfg), *forwarded])


def _cmd_eval(args: argparse.Namespace, extra: Sequence[str]) -> int:
    cfg = Path(args.config)
    if not cfg.is_file():
        raise FileNotFoundError(str(cfg))
    if not args.resume:
        raise ValueError("--resume is required for eval")
    forwarded = [x for x in extra if x != "--"]
    return _run_vendored_deim(["-c", str(cfg), "--test-only", "-r", str(args.resume), *forwarded])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psod")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pseudo = sub.add_parser("pseudo", help="从点标注生成伪框（SAM point prompt）")
    p_pseudo.add_argument("--coco-json", type=str, required=True)
    p_pseudo.add_argument("--image-root", type=str, required=True)
    p_pseudo.add_argument("--out-dir", type=str, default=str(_psod_root()))
    p_pseudo.add_argument("--weights", type=str, default=str(_psod_root() / "weights" / "sam_b.pt"))
    p_pseudo.add_argument("--device", type=str, default=None)
    p_pseudo.add_argument("--default-box-size", type=float, default=32.0)
    p_pseudo.add_argument("--output-name", type=str, default=None)
    p_pseudo.add_argument("--max-images", type=int, default=None)
    p_pseudo.add_argument("--trim", action="store_true", default=False)
    p_pseudo.set_defaults(_handler=_cmd_pseudo)

    p_train = sub.add_parser("train", help="训练（vendored DEIM 最小子集）")
    p_train.add_argument("--config", type=str, default=str(_default_train_config()))
    p_train.set_defaults(_handler=_cmd_train)

    p_eval = sub.add_parser("eval", help="评测（vendored DEIM 最小子集，--test-only）")
    p_eval.add_argument("--config", type=str, default=str(_default_train_config()))
    p_eval.add_argument("--resume", type=str, required=True)
    p_eval.set_defaults(_handler=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    if args.command in {"train", "eval"}:
        return int(handler(args, extra))
    return int(handler(args))
