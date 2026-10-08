"""模型管理测试：权重校验、任务类型过滤、索引、下载与删除。"""
from __future__ import annotations

import pytest

from core import config, models, training


def _fake_run(task_kwargs: dict) -> str:
    """写入一个假的 RUNNING 任务，用于验证资源占用保护。"""
    run_id = task_kwargs.get("run_id", "run_fake_model")
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "run_id": run_id, "name": "fake", "status": "RUNNING", "pid": 0,
        "started_at": config.now_iso(), "config": {"device": "cpu"},
        "dataset": {"id": "ds_fake", "name": "fake"},
        "model": {"id": "mdl_fake", "name": "fake"},
    }
    task.update(task_kwargs)
    config.atomic_write_json(run_dir / "task.json", task)
    return run_id


# ---------------------------------------------------------------- 正常流程
def test_upload_real_segment_model(tiny_seg_pt):
    meta = models.import_model_file(tiny_seg_pt, name="测试分割模型")
    assert meta["task"] == "segment"
    assert meta["nc"] == 1
    assert meta["size_mb"] > 0
    assert meta["source"] == "upload"
    assert models.model_path(meta["id"]).is_file()
    # 索引
    listed = models.list_models()
    assert [m["id"] for m in listed] == [meta["id"]]
    assert models.get_model(meta["id"])["name"] == "测试分割模型"
    rows = models.model_table_rows(listed)
    assert rows[0][0] == "测试分割模型" and rows[0][1] == meta["id"]


def test_upload_twice_same_name_keeps_both(tiny_seg_pt):
    a = models.import_model_file(tiny_seg_pt, name="同名模型")
    b = models.import_model_file(tiny_seg_pt, name="同名模型")
    assert a["id"] != b["id"]
    assert len(models.list_models()) == 2
    assert len(models.model_choices(models.list_models())) == 2


def test_register_run_best_idempotent(tiny_seg_pt):
    first = models.register_run_best("run_x1", tiny_seg_pt, name="run_x1_best")
    assert first and first["source"] == "training"
    second = models.register_run_best("run_x1", tiny_seg_pt, name="run_x1_best")
    assert second is None  # 同一任务不重复注册
    assert len(models.list_models()) == 1


def test_register_run_best_missing_file():
    assert models.register_run_best("run_x2", "不存在的路径.pt") is None


# ---------------------------------------------------------------- 异常输入
def test_upload_detect_model_ok(tiny_detect_pt):
    """目标检测模型现在是被支持的。"""
    meta = models.import_model_file(tiny_detect_pt, name="检测模型")
    assert meta["task"] == "detect"
    assert models.model_path(meta["id"]).is_file()
    assert models.list_models(task="detect")[0]["id"] == meta["id"]
    assert models.list_models(task="segment") == []


def test_reject_classify_model(tiny_cls_pt):
    with pytest.raises(models.ModelError, match="classify"):
        models.import_model_file(tiny_cls_pt, name="分类模型")
    assert models.list_models() == []


def test_list_models_returns_all_tasks(tiny_seg_pt, tiny_detect_pt):
    seg = models.import_model_file(tiny_seg_pt, name="分割模型")
    det = models.import_model_file(tiny_detect_pt, name="检测模型")
    ids = {m["id"] for m in models.list_models()}
    assert ids == {seg["id"], det["id"]}
    assert len(models.model_choices(models.list_models())) == 2


def test_reject_wrong_extension(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"junk")
    with pytest.raises(models.ModelError, match=r"只支持 \.pt"):
        models.import_model_file(p)


def test_reject_invalid_pt_content(tmp_path):
    p = tmp_path / "fake.pt"
    p.write_bytes(b"this is definitely not a torch checkpoint")
    with pytest.raises(models.ModelError, match="PyTorch 权重格式|无法加载"):
        models.import_model_file(p)
    assert models.list_models() == []


def test_reject_empty_file(tmp_path):
    p = tmp_path / "empty.pt"
    p.write_bytes(b"")
    with pytest.raises(models.ModelError, match="为空"):
        models.import_model_file(p)


def test_reject_missing_file(tmp_path):
    with pytest.raises(models.ModelError, match="不存在"):
        models.import_model_file(tmp_path / "nope.pt")


def test_reject_zip_that_is_not_torch(tmp_path):
    import zipfile

    p = tmp_path / "notorch.pt"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(models.ModelError, match="无法加载"):
        models.import_model_file(p)


def test_model_path_unsafe_id():
    for bad in ("../x", "a/b"):
        with pytest.raises(models.ModelError, match="非法"):
            models.model_path(bad)


# ---------------------------------------------------------------- 删除
def test_delete_model(tiny_seg_pt):
    meta = models.import_model_file(tiny_seg_pt, name="待删除")
    assert "已删除" in models.delete_model(meta["id"])
    assert models.get_model(meta["id"]) is None
    assert not (config.MODELS_DIR / meta["id"]).exists()


def test_delete_missing_model():
    with pytest.raises(models.ModelError, match="不存在"):
        models.delete_model("mdl_nope")


def test_delete_blocked_while_training(tiny_seg_pt):
    meta = models.import_model_file(tiny_seg_pt, name="使用中")
    _fake_run({"run_id": "run_inuse", "model": {"id": meta["id"], "name": meta["name"]}})
    assert training.is_resource_in_use("model", meta["id"]) is True
    with pytest.raises(models.ModelError, match="正在被训练任务使用"):
        models.delete_model(meta["id"])
    assert (config.MODELS_DIR / meta["id"]).exists()