from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Sequence


def run(argv: Sequence[str]) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfigdir")

    point_root = Path(__file__).resolve().parent
    deim_root = (point_root / "DEIM").resolve()
    deim_train = (deim_root / "train.py").resolve()

    sys.path.insert(0, str(point_root))
    sys.path.insert(0, str(deim_root))
    import point_ext

    old_argv = sys.argv
    sys.argv = [str(deim_train), *list(argv)]
    try:
        runpy.run_path(str(deim_train), run_name="__main__")
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        return 1
    finally:
        sys.argv = old_argv

    return 0
