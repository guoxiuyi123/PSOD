import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'psod', 'deim', 'DEIM')) 

from engine.logger_module import get_logger
from engine.extre_module.torch_utils import check_cuda
from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)


def run_deim(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    check_cuda()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', 'mode'] and v is not None}) 

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    cfg_str = json.dumps(cfg.__dict__, indent=4, ensure_ascii=False)
    
    if hasattr(args, 'local_rank') and (args.local_rank == 0 or args.local_rank is None):
        print(GREEN + cfg_str + RESET)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.mode == 'val':
        logger.info("🚀 Launching [Evaluation Mode]")
        if args.path:
            solver.val_onnx_engine(args.onnx_mode)
        else:
            solver.val()
    elif args.mode == 'train':
        logger.info("🚀 Launching [Training Mode]")
        solver.fit(cfg_str)

    dist_utils.cleanup()


def main():
    parser = argparse.ArgumentParser(description="PSOD Unified Entry Point")
    parser.add_argument('mode', type=str, choices=['train', 'val', 'pseudo'], 
                        help='Running mode: train, val, pseudo')

    # DEIM 引擎参数
    parser.add_argument('-c', '--config', type=str, help='YAML config file path')
    parser.add_argument('-r', '--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='Tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='Device (e.g., cuda:0)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='Auto mixed precision training (AMP)')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--summary-dir', type=str, help='Tensorboard summary directory')
    parser.add_argument('-p', '--path', type=str, help='ONNX/Engine model path (Val only)')
    parser.add_argument('--onnx-mode', type=str, default='det', choices=['det', 'mask'])
    parser.add_argument('-u', '--update', nargs='+', help='Dynamically update YAML config')
    parser.add_argument('--print-method', type=str, default='builtin', help='Print method')
    parser.add_argument('--print-rank', type=int, default=0, help='Print rank id')
    parser.add_argument('--local_rank', '--local-rank', type=int, help='Distributed local rank id')
    
    # 核心魔法：使用 parse_known_args。它会自动把不认识的参数（如 --coco-json）打包进 remaining_argv
    args, remaining_argv = parser.parse_known_args()

    if args.mode in ['train', 'val']:
        if not args.config:
            parser.error(f"Mode '{args.mode}' requires -c/--config path!")
        run_deim(args)
        
    elif args.mode == 'pseudo':
        print(YELLOW + "🚀 Launching [Pseudo Label Generation Mode]" + RESET)
        # 调用原始项目的命令行核心
        from psod.cli import main as psod_main
        # 组装命令: ['pseudo', '--coco-json', '...', '--image-root', '...']
        sys.exit(psod_main(['pseudo'] + remaining_argv))

if __name__ == '__main__':
    main()