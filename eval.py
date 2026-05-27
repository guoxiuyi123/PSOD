from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    sys.path.insert(0, str(_project_root()))
    from psod.cli import main as psod_main

    argv = ["eval", *sys.argv[1:]]
    return int(psod_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

