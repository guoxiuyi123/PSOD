from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-t", "--tuning", type=str)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-dir", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)

    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-m", "--mode", type=str, default="det", choices=["det", "mask"])

    parser.add_argument("-u", "--update", nargs="+", action="append")

    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    return parser


def _resolve_config_path(cfg_path: str) -> str:
    p = Path(cfg_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def main() -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfigdir")

    deim_root = Path(__file__).resolve().parent
    point_root = deim_root.parent
    sys.path.insert(0, str(point_root))
    sys.path.insert(0, str(deim_root))

    args = _build_argparser().parse_args()

    from engine.core import YAMLConfig
    from engine.core.yaml_utils import parse_cli
    from engine.misc import dist_utils
    from engine.solver import TASKS

    dist_utils.setup_distributed(print_rank=args.print_rank, print_method=args.print_method, seed=args.seed)

    overrides = {}
    if args.resume is not None:
        overrides["resume"] = args.resume
    if args.tuning is not None:
        overrides["tuning"] = args.tuning
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.use_amp:
        overrides["use_amp"] = True
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.summary_dir is not None:
        overrides["summary_dir"] = args.summary_dir
    if args.path is not None:
        overrides["path"] = args.path
    updates = []
    if args.update:
        for group in args.update:
            updates.extend(group)
    overrides.update(parse_cli(updates))

    cfg_path = _resolve_config_path(args.config)
    cfg = YAMLConfig(cfg_path, **overrides)

    cfg_str = json.dumps(
        {"yaml_cfg": cfg.yaml_cfg, "argv": sys.argv[1:]},
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.test_only:
        if args.path:
            solver.val_onnx_engine(args.mode)
        else:
            solver.val()
    else:
        solver.fit(cfg_str)

    dist_utils.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
