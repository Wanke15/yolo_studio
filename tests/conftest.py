"""pytest 公共 fixture。

- isolated_storage：把 storage 目录重定向到临时目录，测试之间互不干扰。
- tiny_seg_pt / tiny_detect_pt：真实（未训练）的 YOLO 权重，完全离线生成，不依赖公网。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config, training  # noqa: E402
from helpers import build_tiny_detect_pt, build_tiny_seg_pt  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """所有测试使用独立的临时 storage 目录。"""
    root = tmp_path / "storage"
    monkeypatch.setattr(config, "STORAGE_DIR", root)
    monkeypatch.setattr(config, "DATASETS_DIR", root / "datasets")
    monkeypatch.setattr(config, "MODELS_DIR", root / "models")
    monkeypatch.setattr(config, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(config, "INFER_DIR", root / "inference")
    monkeypatch.setattr(config, "TMP_DIR", root / "tmp")
    config.ensure_dirs()
    training._procs.clear()      # noqa: SLF001 - 清理进程表，避免跨测试污染
    training._log_files.clear()  # noqa: SLF001
    yield root
    for run_id, proc in list(training._procs.items()):  # noqa: SLF001
        try:
            training._kill_process_group({"run_id": run_id, "pid": proc.pid})  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
    training._procs.clear()      # noqa: SLF001
    training._log_files.clear()  # noqa: SLF001


@pytest.fixture(scope="session")
def tiny_seg_pt(tmp_path_factory) -> Path:
    """真实的 YOLO-Seg 权重（nc=1，未训练），约 7 MB。"""
    return build_tiny_seg_pt(tmp_path_factory.mktemp("weights") / "tiny_seg.pt", nc=1)


@pytest.fixture(scope="session")
def tiny_detect_pt(tmp_path_factory) -> Path:
    """真实的 YOLO-Detect 权重（nc=1），用于验证非 segment 模型会被拒绝。"""
    return build_tiny_detect_pt(tmp_path_factory.mktemp("weights") / "tiny_detect.pt", nc=1)