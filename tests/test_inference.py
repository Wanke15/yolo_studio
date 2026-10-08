"""推理测试：输入校验、结果结构与统计、GPU 训练期间禁用 GPU 推理。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import config, datasets, inference, models
from helpers import build_dataset_zip


def _upload_model(tiny_seg_pt) -> dict:
    return models.import_model_file(tiny_seg_pt, name="推理模型")


def _image(tmp_path: Path, name="test.png") -> Path:
    p = tmp_path / name
    Image.new("RGB", (128, 128), (30, 120, 200)).save(p)
    return p


# ---------------------------------------------------------------- 输入校验
def test_rejects_missing_model(tmp_path):
    with pytest.raises(inference.InferenceError, match="有效的模型"):
        inference.run_inference("mdl_missing", _image(tmp_path))


def test_rejects_bad_image(tmp_path, tiny_seg_pt):
    meta = _upload_model(tiny_seg_pt)
    with pytest.raises(inference.InferenceError, match="jpg / jpeg / png"):
        inference.run_inference(meta["id"], tmp_path / "nope.png")
    txt = tmp_path / "file.txt"
    txt.write_text("not an image")
    with pytest.raises(inference.InferenceError, match="jpg / jpeg / png"):
        inference.run_inference(meta["id"], txt)


@pytest.mark.parametrize("conf, expect", [(-0.1, "0~1"), (0.0, "0~1"), (1.5, "0~1"), ("abc", "数字")])
def test_rejects_bad_conf(tmp_path, tiny_seg_pt, conf, expect):
    meta = _upload_model(tiny_seg_pt)
    with pytest.raises(inference.InferenceError, match=expect):
        inference.run_inference(meta["id"], _image(tmp_path), conf=conf, imgsz=640)


def test_rejects_bad_imgsz(tmp_path, tiny_seg_pt):
    meta = _upload_model(tiny_seg_pt)
    with pytest.raises(inference.InferenceError, match="Image Size"):
        inference.run_inference(meta["id"], _image(tmp_path), conf=0.25, imgsz=777)


def test_gpu_inference_blocked_while_gpu_training(tmp_path, tiny_seg_pt):
    """GPU 训练进行中时禁用 GPU 推理（避免显存冲突），CPU 推理仍允许。"""
    meta = _upload_model(tiny_seg_pt)
    run_dir = config.RUNS_DIR / "run_gpu"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.atomic_write_json(
        run_dir / "task.json",
        {"run_id": "run_gpu", "name": "gpu", "status": "RUNNING", "pid": 0,
         "started_at": config.now_iso(), "config": {"device": "cuda:0", "epochs": 1},
         "dataset": {"id": "ds"}, "model": {"id": meta["id"]}},
    )
    with pytest.raises(inference.InferenceError, match="已禁用 GPU 推理"):
        inference.run_inference(meta["id"], _image(tmp_path), device="cuda:0")


# ---------------------------------------------------------------- 真实推理（未训练权重）
def test_real_inference_on_untrained_model(tmp_path, tiny_seg_pt):
    """真实执行一次 predict：验证结果结构、可视化与统计一致性。"""
    meta = _upload_model(tiny_seg_pt)
    img_path = _image(tmp_path)
    result = inference.run_inference(meta["id"], img_path, conf=0.25, imgsz=320, device="cpu")

    assert isinstance(result["original"], Image.Image)
    assert isinstance(result["plotted"], np.ndarray)
    assert result["plotted"].ndim == 3 and result["plotted"].shape[2] == 3
    assert result["total"] == sum(result["counts"].values())  # 按实例数量统计
    assert Path(result["result_path"]).is_file()
    md = inference.stats_markdown(result)
    assert "检测到的实例总数" in md
    if result["total"] == 0:
        # 未检测到目标属于正常结果，不算执行失败
        assert result["message"] == "未检测到目标"
        assert "未检测到目标" in md
    else:
        assert "| 类别 | 实例数 |" in md


def test_result_image_size_matches_original(tmp_path, tiny_seg_pt):
    meta = _upload_model(tiny_seg_pt)
    img_path = _image(tmp_path)
    result = inference.run_inference(meta["id"], img_path, imgsz=320, device="cpu")
    assert result["plotted"].shape[:2] == (128, 128)


def test_inference_on_trained_style_dataset_image(tmp_path, tiny_seg_pt):
    """对数据集中的真实图片（含标签多边形）推理，确保图片读取路径可用。"""
    import zipfile

    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip"), "推理数据集")
    root = config.DATASETS_DIR / ds["id"]
    img = sorted((root / "images" / "train").glob("*.png"))[0]
    meta = _upload_model(tiny_seg_pt)
    result = inference.run_inference(meta["id"], img, conf=0.1, imgsz=320, device="cpu")
    assert Path(result["result_path"]).is_file()
    assert zipfile.is_zipfile(tiny_seg_pt)


# ---------------------------------------------------------------- 目标检测模型推理
def test_detect_model_inference(tmp_path, tiny_detect_pt):
    """detect 模型也能推理：结果结构一致，统计按检测框实例数。"""
    meta = models.import_model_file(tiny_detect_pt, name="检测推理模型")
    assert meta["task"] == "detect"
    img_path = _image(tmp_path)
    result = inference.run_inference(meta["id"], img_path, conf=0.25, imgsz=320, device="cpu")
    assert result["task"] == "detect"
    assert result["total"] == sum(result["counts"].values())
    assert Path(result["result_path"]).is_file()
    md = inference.stats_markdown(result)
    assert "detect（目标检测）" in md and "检测到的实例总数" in md


def test_reject_unsupported_task_model(tmp_path, tiny_seg_pt):
    """非 segment/detect 的模型（如 classify）不参与推理。"""
    meta = _upload_model(tiny_seg_pt)
    # 手工把 meta 改成不支持的任务类型，模拟历史数据/异常状态
    bad = dict(meta, task="classify")
    config.atomic_write_json(config.MODELS_DIR / meta["id"] / "meta.json", bad)
    with pytest.raises(inference.InferenceError, match="只支持"):
        inference.run_inference(meta["id"], _image(tmp_path), device="cpu")