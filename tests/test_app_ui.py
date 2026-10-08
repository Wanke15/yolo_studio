"""界面层测试：Gradio 页面可构建，UI 回调函数在真实数据上返回值正确。

这些测试不启动 HTTP 服务，但会真实调用 app.py 中的事件处理函数。
"""
from __future__ import annotations

import pytest

from core import config, datasets, inference, models, training
from helpers import build_dataset_zip


def _cell(table, row: int, col: int):
    """读取 gr.Dataframe 更新值的单元格。

    Gradio 5 会把 list 值包装成 {'headers':..., 'data':...}（或 pandas.DataFrame）。
    """
    value = table.value
    if isinstance(value, dict):
        value = value.get("data")
    if hasattr(value, "iloc"):
        return value.iloc[row, col]
    return value[row][col]


@pytest.fixture(scope="module")
def app_module():
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    return importlib.import_module("app")


def test_app_builds_with_four_tabs(app_module):
    assert app_module.demo is not None
    # 四个 Tab 均已在 Blocks 中注册
    assert len(app_module.demo.blocks) > 0
    for fn in (
        "ui_import_dataset", "ui_refresh_datasets", "ui_dataset_info", "ui_preview", "ui_delete_dataset",
        "ui_upload_model", "ui_refresh_models", "ui_model_info", "ui_delete_model", "ui_model_download",
        "ui_device_update", "ui_start_training", "ui_stop_training", "ui_poll", "ui_select_history",
        "ui_history_download", "ui_run_inference",
    ):
        assert callable(getattr(app_module, fn)), f"缺少 UI 回调 {fn}"


def test_ui_dataset_flow(app_module, tmp_path):
    zip_path = build_dataset_zip(tmp_path / "d.zip", classes=("shape",))
    msg, table, dropdown = app_module.ui_import_dataset(str(zip_path), "界面测试集")
    assert "导入成功" in msg
    assert _cell(table, 0, 0) == "界面测试集"
    assert dropdown.choices and dropdown.choices[0][1]

    dataset_id = dropdown.value or dropdown.choices[0][1]
    info = app_module.ui_dataset_info(dataset_id)
    assert "界面测试集" in info and "类别数" in info

    gallery, pmsg = app_module.ui_preview(dataset_id, True, 4, "all")
    assert len(gallery) == 4 and "预览" in pmsg

    # 未勾选确认时不允许删除
    msg_no, _, _ = app_module.ui_delete_dataset(dataset_id, False)
    assert "请先勾选" in msg_no
    assert datasets.get_dataset(dataset_id) is not None
    # 勾选后删除成功
    msg_yes, _, _ = app_module.ui_delete_dataset(dataset_id, True)
    assert "已删除" in msg_yes
    assert datasets.get_dataset(dataset_id) is None


def test_ui_dataset_import_error_message(app_module, tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    msg, _, _ = app_module.ui_import_dataset(str(bad), "坏包")
    assert "导入失败" in msg


def test_ui_model_flow(app_module, tiny_seg_pt, tmp_path):
    msg, table, train_dd, infer_dd = app_module.ui_upload_model(str(tiny_seg_pt), "界面模型")
    assert "上传成功" in msg
    assert _cell(table, 0, 0) == "界面模型"
    assert train_dd.value and infer_dd.value

    model_id = train_dd.value
    info = app_module.ui_model_info(model_id)
    assert "界面模型" in info and "segment" in info

    path, dl_msg = app_module.ui_model_download(model_id)
    assert path and path.endswith("model.pt") and "下载链接" in dl_msg

    msg_no, _, _, _ = app_module.ui_delete_model(model_id, False)
    assert "请先勾选" in msg_no
    msg_yes, _, _, _ = app_module.ui_delete_model(model_id, True)
    assert "已删除" in msg_yes


def test_ui_model_reject_detect(app_module, tiny_detect_pt):
    msg, _, _, _ = app_module.ui_upload_model(str(tiny_detect_pt), "检测模型")
    assert "上传失败" in msg and "segment" in msg


def test_ui_inference_requires_inputs(app_module):
    original, plotted, stats, path = app_module.ui_run_inference(None, None, 0.25, 640, "cpu")
    assert original is None and plotted is None and path is None
    assert "请先选择模型" in stats


def test_ui_device_and_status(app_module, tmp_path, tiny_seg_pt):
    dd, note = app_module.ui_device_update()
    assert dd.value in training.available_devices()
    if not training.has_gpu():
        assert "未检测到 CUDA GPU" in note

    # 没有任务时的状态与历史
    status, log, metrics, result, plots, history = app_module.ui_poll()
    assert "没有正在进行的训练任务" in status or "当前任务" in status
    assert isinstance(log, str)

    # 启动参数非法时报错信息
    msg, *_ = app_module.ui_start_training("ds_x", "mdl_x", 1, 2, 320, "cpu", 0, "")
    assert "无法启动训练" in msg

    # 停止不存在的任务
    stop_msg, *_ = app_module.ui_stop_training()
    assert "没有正在运行的任务" in stop_msg


def test_ui_inference_end_to_end(app_module, tmp_path, tiny_seg_pt):
    from PIL import Image

    msg, _, train_dd, infer_dd = app_module.ui_upload_model(str(tiny_seg_pt), "推理界面模型")
    img = tmp_path / "ui_test.png"
    Image.new("RGB", (96, 96), (10, 200, 120)).save(img)
    original, plotted, stats, result_path = app_module.ui_run_inference(
        infer_dd.value, str(img), 0.25, 320, "cpu"
    )
    assert original is not None and plotted is not None
    assert "检测到的实例总数" in stats
    assert result_path and result_path.endswith(".jpg")


def test_ui_history_download_missing(app_module):
    path, msg = app_module.ui_history_download("run_none", "best.pt")
    assert path is None and "请先选择历史任务" in msg