from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _setup_sys_path() -> None:
    root = Path(__file__).resolve().parents[1]
    point_root = (root / "psod" / "deim").resolve()
    deim_root = (point_root / "DEIM").resolve()
    sys.path.insert(0, str(point_root))
    sys.path.insert(0, str(deim_root))


def _build_solver(tmp_path: Path):
    from engine.solver.det_solver import DetSolver

    solver = DetSolver(SimpleNamespace())
    solver.output_dir = tmp_path
    solver.train_dataloader = SimpleNamespace(
        collate_fn=SimpleNamespace(stop_epoch=5, ema_restart_decay=0.9999),
    )
    loaded = []
    solver.load_resume_state = loaded.append
    return solver, loaded


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def main() -> int:
    _setup_sys_path()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        solver, loaded = _build_solver(tmp)
        _touch(tmp / "last.pth")
        _touch(tmp / "best_stg1.pth")
        ok = solver._load_stage2_resume_state(5)
        assert ok is True
        assert loaded[-1].endswith("best_stg1.pth")

        (tmp / "best_stg1.pth").unlink()
        ok = solver._load_stage2_resume_state(5)
        assert ok is True
        assert loaded[-1].endswith("last.pth")

        (tmp / "last.pth").unlink()
        ok = solver._load_stage2_resume_state(5)
        assert ok is False

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
