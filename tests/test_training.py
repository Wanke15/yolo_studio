"""训练管理测试：配置校验、单任务模式、子进程状态机、指标读取、端到端 CPU 训练。"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from core import config, datasets, models, training
from helpers import build_dataset_zip, wait_for_status


# ---------------------------------------------------------------- 配置校验
@pytest.mark.parametrize(
    "kwargs, expect",
    [
        ({"epochs": 0}, "Epochs"),
        ({"epochs": "abc"}, "Epochs"),
        ({"epochs": 99999}, "Epochs"),
        ({"batch": 0}, "Batch Size"),
        ({"batch": 4096}, "Batch Size"),
        ({"imgsz": 700}, "Image Size"),
        ({"workers": 99}, "Workers"),
        ({"workers": -1}, "Workers"),
    ],
)
def test_validate_config_rejects_bad_values(kwargs, expect):
    base = {"epochs": 50, "batch": 8, "imgsz": 640, "workers": 2}
    base.update(kwargs)
    with pytest.raises(training.TrainingError, match=expect):
        training.validate_config(**base)


def test_validate_config_ok():
    cfg = training.validate_config(epochs=2, batch=2, imgsz=320, workers=0)
    assert cfg == {"epochs": 2, "batch": 2, "imgsz": 320, "workers": 0}


def test_devices_and_default():
    devices = training.available_devices()
    assert devices[-1] == "cpu"
    assert training.default_device() == devices[0]
    if not training.has_gpu():  # 本机无 GPU
        assert devices == ["cpu"]
        assert not training.gpu_training_active()


def test_validate_device_rejects_unknown():
    with pytest.raises(training.TrainingError, match="不可用|无效"):
        training._validate_device("cuda:9")  # noqa: SLF001
    assert training._validate_device("cpu") == "cpu"  # noqa: SLF001


# ---------------------------------------------------------------- 启动前置校验
def test_start_requires_valid_inputs(tmp_path, tiny_seg_pt):
    with pytest.raises(training.TrainingError, match="数据集"):
        training.start_training("ds_missing", "mdl_missing", epochs=1, batch=2, imgsz=320, device="cpu", workers=0)
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip"), "d")
    with pytest.raises(training.TrainingError, match="模型"):
        training.start_training(ds["id"], "mdl_missing", epochs=1, batch=2, imgsz=320, device="cpu", workers=0)


def test_start_rejects_bad_config(tmp_path, tiny_seg_pt):
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip"), "d")
    mdl = models.import_model_file(tiny_seg_pt, "m")
    with pytest.raises(training.TrainingError, match="Image Size"):
        training.start_training(ds["id"], mdl["id"], epochs=1, batch=2, imgsz=99, device="cpu", workers=0)


# ---------------------------------------------------------------- 任务组合规则
def test_combination_segment_model_with_detect_dataset_rejected(tmp_path, tiny_seg_pt):
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip", label_mode="detection"), "检测数据集")
    mdl = models.import_model_file(tiny_seg_pt, "分割模型")
    assert ds["task"] == "detect"
    with pytest.raises(training.TrainingError, match="需要多边形标签"):
        training.start_training(ds["id"], mdl["id"], epochs=1, batch=2, imgsz=320, device="cpu", workers=0)


@pytest.mark.parametrize(
    "dataset_mode, model_fixture, expect_note",
    [
        ("detection", "tiny_detect_pt", False),   # detect 模型 + 检测数据集
        ("valid", "tiny_detect_pt", True),        # detect 模型 + 分割数据集（自动转外接框）
    ],
)
def test_combination_allowed_and_stopped(tmp_path, request, dataset_mode, model_fixture, expect_note):
    """允许的组合可以启动训练；detect 模型 + 分割数据集会给出自动转框提示。"""
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip", label_mode=dataset_mode), "ds")
    mdl = models.import_model_file(request.getfixturevalue(model_fixture), "模型")
    task = training.start_training(
        ds["id"], mdl["id"], epochs=20, batch=2, imgsz=320, device="cpu", workers=0, name="组合测试"
    )
    try:
        assert task["task"] == "detect"
        assert task["dataset"]["task"] == ("detect" if dataset_mode == "detection" else "segment")
        if expect_note:
            assert "自动转换为外接框" in task["message"]
        # 训练只允许 segment 模型配分割数据集，其余按任务类型放行
        assert training.get_run(task["run_id"])["status"] == "RUNNING"
    finally:
        training.stop_training(task["run_id"])
    assert training.get_run(task["run_id"])["status"] == "STOPPED"


# ---------------------------------------------------------------- 状态机
def test_init_manager_marks_interrupted():
    run_dir = config.RUNS_DIR / "run_stale"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.atomic_write_json(
        run_dir / "task.json",
        {
            "run_id": "run_stale", "name": "旧任务", "status": "RUNNING", "pid": 0,
            "started_at": "2026-01-01 00:00:00", "config": {"device": "cpu", "epochs": 1},
            "dataset": {"id": "ds_x", "name": "x"}, "model": {"id": "mdl_x", "name": "x"},
            "log_path": str(run_dir / "train.log"),
        },
    )
    assert training.current_run() is not None
    messages = training.init_manager()
    assert messages and "INTERRUPTED" in messages[0]
    task = training.get_run("run_stale")
    assert task["status"] == "INTERRUPTED"
    assert "服务重启" in task["message"]
    # 不再占用训练锁
    assert training.current_run() is None


def test_stop_rejects_non_running(tmp_path, tiny_seg_pt):
    with pytest.raises(training.TrainingError, match="不存在"):
        training.stop_training("run_nope")


def test_single_task_mode_and_stop(tmp_path, tiny_seg_pt):
    """真实启动一个训练子进程：验证单任务互斥、资源占用保护、停止后释放。"""
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip"), "d")
    mdl = models.import_model_file(tiny_seg_pt, "m")
    task = training.start_training(
        ds["id"], mdl["id"], epochs=50, batch=2, imgsz=320, device="cpu", workers=0, name="互斥测试"
    )
    try:
        assert training.get_run(task["run_id"])["status"] == "RUNNING"
        assert task["pid"] and task["pid"] > 0
        # 有任务运行时禁止启动新任务
        with pytest.raises(training.TrainingError, match="已有训练任务正在运行"):
            training.start_training(ds["id"], mdl["id"], epochs=1, batch=2, imgsz=320, device="cpu", workers=0)
        # 运行中的数据集/模型不可删除
        assert training.is_resource_in_use("dataset", ds["id"])
        assert training.is_resource_in_use("model", mdl["id"])
        with pytest.raises(datasets.DatasetError, match="正在被训练任务使用"):
            datasets.delete_dataset(ds["id"])
        with pytest.raises(models.ModelError, match="正在被训练任务使用"):
            models.delete_model(mdl["id"])
        # 停止任务
        msg = training.stop_training(task["run_id"])
        assert "已停止" in msg
        stopped = training.get_run(task["run_id"])
        assert stopped["status"] == "STOPPED"
        assert training.current_run() is None
        assert not training._pid_alive(task["pid"])  # noqa: SLF001 - 进程组已终止，无遗留进程
        assert not training.is_resource_in_use("dataset", ds["id"])
        with pytest.raises(training.TrainingError, match="无需停止"):
            training.stop_training(task["run_id"])
    finally:
        if training.get_run(task["run_id"])["status"] == "RUNNING":
            training.stop_training(task["run_id"])


# ---------------------------------------------------------------- 指标读取
def test_metrics_none_without_csv():
    assert training.read_metrics("run_no_such") is None
    assert training.metrics_long_dataframe("run_no_such") is None


def test_metrics_parsing_and_missing_not_zero():
    """列名变体需兼容；缺失的指标保持缺失，不允许伪装成 0。"""
    run_id = "run_metrics"
    out = config.RUNS_DIR / run_id / "output"
    out.mkdir(parents=True, exist_ok=True)
    # 故意使用旧版/变体列名，并缺少 val/seg_loss 列
    cols = ["epoch", "train/box_loss", "train/seg_loss", "val/box_loss",
            "metrics/mAP50(M)", "metrics/mAP50-95(M)"]
    rows = [
        [1, 1.5, 2.5, 1.6, 0.10, 0.05],
        [2, 1.2, 2.0, 1.3, 0.30, 0.12],
        [3, 1.0, 1.8, 1.1, 0.25, 0.20],
    ]
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    config.atomic_write_json(
        config.RUNS_DIR / run_id / "task.json",
        {"run_id": run_id, "name": "metrics", "status": "COMPLETED", "started_at": "2026-01-01 00:00:00",
         "task": "segment", "config": {"device": "cpu", "epochs": 3},
         "dataset": {"id": "d"}, "model": {"id": "m", "task": "segment"},
         "results_csv": str(out / "results.csv")},
    )

    metrics = training.read_metrics(run_id)
    assert metrics is not None
    assert metrics["task"] == "segment"
    assert metrics["labels"]["map50"] == "Mask mAP50"
    assert metrics["epochs"] == [1, 2, 3]
    assert metrics["series"]["train_box_loss"] == [1.5, 1.2, 1.0]
    assert metrics["series"]["train_seg_loss"] == [2.5, 2.0, 1.8]
    # 缺失指标为 None，不是 0
    assert all(v is None for v in metrics["series"]["val_seg_loss"])
    assert metrics["series"]["map50"] == [0.10, 0.30, 0.25]
    # 最优值取最大
    assert metrics["best"]["map50"] == {"epoch": 2, "value": 0.30}
    assert metrics["best"]["map50_95"] == {"epoch": 3, "value": 0.20}
    # 长格式数据只包含有值的点
    df = training.metrics_long_dataframe(run_id)
    assert df is not None and set(df["指标"]) == {
        "Train Box Loss", "Train Seg Loss", "Val Box Loss", "Mask mAP50", "Mask mAP50-95"
    }


def test_metrics_detect_uses_box_columns():
    """detect 任务取 metrics/mAP50(B) 列，展示为 Box mAP；segment 不会误取检测框列。"""
    run_id = "run_detect_metrics"
    out = config.RUNS_DIR / run_id / "output"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    rows = [[1, 1.9, 2.1, 2.0, 2.2, 0.44, 0.21], [2, 1.5, 1.7, 1.6, 1.8, 0.61, 0.35]]
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    config.atomic_write_json(
        config.RUNS_DIR / run_id / "task.json",
        {"run_id": run_id, "name": "detect", "status": "COMPLETED", "started_at": "2026-01-01 00:00:00",
         "task": "detect", "config": {"device": "cpu", "epochs": 2},
         "dataset": {"id": "d"}, "model": {"id": "m", "task": "detect"},
         "results_csv": str(out / "results.csv")},
    )

    metrics = training.read_metrics(run_id)
    assert metrics["task"] == "detect"
    assert metrics["labels"]["map50"] == "Box mAP50"
    assert metrics["series"]["map50"] == [0.44, 0.61]
    assert metrics["best"]["map50_95"] == {"epoch": 2, "value": 0.35}
    # 没有掩码列 → 掩码指标不出现（不伪造）
    assert "mask" not in " ".join(metrics["mapping"].values())
    df = training.metrics_long_dataframe(run_id)
    assert set(df["指标"]) == {"Train Box Loss", "Val Box Loss", "Box mAP50", "Box mAP50-95"}
    # 结果摘要显示 Box mAP
    md = training.result_markdown(run_id)
    assert "最优 Box mAP50" in md and "Mask" not in md
    # 历史记录里带任务类型列
    row = training.run_history_rows([training.get_run(run_id)])[0]
    assert row[2] == "detect" and row[-1] == "0.3500"
    assert "detect（目标检测）" in training.run_config_markdown(run_id)


def test_metrics_handles_truncated_last_line(tmp_path):
    """训练中 results.csv 最后一行可能不完整，解析时需跳过而不是报错。"""
    run_id = "run_trunc"
    out = config.RUNS_DIR / run_id / "output"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,train/seg_loss,val/box_loss,val/seg_loss,metrics/mAP50(M),metrics/mAP50-95(M)\n"
        "1,1.5,2.5,1.6,2.6,0.10,0.05\n"
        "2,1.2,2.0,1.3\n",
        encoding="utf-8",
    )
    config.atomic_write_json(
        config.RUNS_DIR / run_id / "task.json",
        {"run_id": run_id, "name": "t", "status": "RUNNING", "started_at": "2026-01-01 00:00:00",
         "config": {"device": "cpu"}, "dataset": {"id": "d"}, "model": {"id": "m"},
         "results_csv": str(csv_path)},
    )
    metrics = training.read_metrics(run_id)
    assert metrics["epochs"] == [1, 2]
    assert metrics["series"]["map50"] == [0.10, None]


def test_read_log_missing_returns_empty():
    assert training.read_log("run_absent") == ""


def test_run_history_and_config_markdown(tmp_path, tiny_seg_pt):
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip"), "历史数据集")
    mdl = models.import_model_file(tiny_seg_pt, "历史模型")
    run_id = "run_hist"
    d = config.RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    config.atomic_write_json(
        d / "task.json",
        {"run_id": run_id, "name": "历史任务", "status": "COMPLETED", "started_at": "2026-01-01 00:00:00",
         "finished_at": "2026-01-01 00:05:00", "task": "segment",
         "config": {"epochs": 2, "batch": 2, "imgsz": 320, "device": "cpu", "workers": 0},
         "dataset": {"id": ds["id"], "name": ds["name"], "task": "segment", "nc": 1,
                     "train_images": 4, "val_images": 2},
         "model": {"id": mdl["id"], "name": mdl["name"], "task": "segment"},
         "best": {"map50_95": {"epoch": 2, "value": 0.4321}}},
    )
    rows = training.run_history_rows()
    assert rows[0][0] == run_id and rows[0][2] == "segment" and rows[0][-1] == "0.4321"
    md = training.run_config_markdown(run_id)
    assert "历史任务" in md and "历史数据集" in md and "历史模型" in md
    assert "暂无" in training.result_markdown(run_id) or "最优" in training.result_markdown(run_id)
    # 数据集被删除后，历史记录仍保留元信息
    datasets.delete_dataset(ds["id"])
    assert "历史数据集" in training.run_config_markdown(run_id)


# ---------------------------------------------------------------- 端到端（真实 CPU 训练）
@pytest.mark.slow
def test_training_end_to_end_cpu(tmp_path, tiny_seg_pt):
    """真实跑通一次完整训练：1 epoch / batch=2 / imgsz=320 / CPU。

    验证：子进程训练 → 日志 → results.csv → 指标 → best.pt → 自动注册模型。
    """
    ds = datasets.import_dataset_zip(build_dataset_zip(tmp_path / "d.zip", n_train=4, n_val=2), "e2e")
    mdl = models.import_model_file(tiny_seg_pt, "e2e-model")
    task = training.start_training(
        ds["id"], mdl["id"], epochs=1, batch=2, imgsz=320, device="cpu", workers=0, name="端到端测试"
    )
    run_id = task["run_id"]
    task = wait_for_status(run_id, timeout=1800)
    log = training.read_log(run_id, 400)
    assert task["status"] == "COMPLETED", f"训练未成功: {task['status']} / {task.get('message')}\n{log[-3000:]}"

    # 产物文件
    out = Path(task["output_dir"])
    for rel in ("weights/best.pt", "weights/last.pt", "results.csv"):
        assert (out / rel).is_file(), f"缺少训练产物 {rel}"
    assert (out / "results.png").is_file()

    # 指标来自真实输出
    metrics = training.read_metrics(run_id)
    assert metrics is not None
    assert metrics["epochs"], "results.csv 中应有 epoch 数据"
    assert any(v is not None for v in metrics["series"]["train_seg_loss"]), \
        f"未解析到 train/seg_loss，实际列: {metrics['columns']}"
    assert metrics["task"] == "segment" and metrics["labels"]["map50"] == "Mask mAP50"
    assert "map50" in metrics["best"], f"未解析到 mask mAP50，实际列: {metrics['columns']}"

    # best.pt 自动注册为模型
    assert task["registered_model_id"], f"best.pt 未注册: {task.get('message')}"
    registered = models.get_model(task["registered_model_id"])
    assert registered["source"] == "training" and registered["run_id"] == run_id
    assert models.model_path(registered["id"]).is_file()

    # 日志有实际内容
    assert "worker" in log
    # 结果摘要
    result_md = training.result_markdown(run_id)
    assert "最优 Mask mAP50" in result_md

@pytest.mark.slow
def test_training_end_to_end_detect_cpu(tmp_path, tiny_detect_pt):
    """真实跑通一次 detect 训练：1 epoch / batch=2 / imgsz=320 / CPU。

    验证：检测数据集 + detect 模型 → results.csv 里是 Box mAP 列 → 指标面板解析为 Box mAP。
    """
    ds = datasets.import_dataset_zip(
        build_dataset_zip(tmp_path / "d.zip", n_train=4, n_val=2, label_mode="detection"), "e2e-detect"
    )
    assert ds["task"] == "detect"
    mdl = models.import_model_file(tiny_detect_pt, "e2e-detect-model")
    task = training.start_training(
        ds["id"], mdl["id"], epochs=1, batch=2, imgsz=320, device="cpu", workers=0, name="端到端检测测试"
    )
    run_id = task["run_id"]
    task = wait_for_status(run_id, timeout=1800)
    log = training.read_log(run_id, 400)
    assert task["status"] == "COMPLETED", f"训练未成功: {task['status']} / {task.get('message')}\n{log[-3000:]}"
    assert task["task"] == "detect"

    out = Path(task["output_dir"])
    for rel in ("weights/best.pt", "weights/last.pt", "results.csv"):
        assert (out / rel).is_file(), f"缺少训练产物 {rel}"

    metrics = training.read_metrics(run_id)
    assert metrics is not None and metrics["task"] == "detect"
    assert metrics["labels"]["map50"] == "Box mAP50"
    assert any("mAP50(B)" in c for c in metrics["columns"]), f"results.csv 应为检测框指标列: {metrics['columns']}"
    assert "map50" in metrics["best"], f"未解析到 Box mAP50，实际列: {metrics['columns']}"
    assert metrics["best"]["map50"]["value"] >= 0

    # best.pt 自动注册为 detect 模型
    assert task["registered_model_id"], f"best.pt 未注册: {task.get('message')}"
    registered = models.get_model(task["registered_model_id"])
    assert registered["task"] == "detect" and registered["run_id"] == run_id
    assert "最优 Box mAP50" in training.result_markdown(run_id)
    assert "端到端检测测试" in training.run_history_rows([training.get_run(run_id)])[0][1]
