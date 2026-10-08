"""数据集管理测试：导入校验、路径安全、标签格式、预览与删除。"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from core import config, datasets, training
from helpers import build_dataset_zip


def _import(tmp_path, **kwargs):
    name = kwargs.pop("name", None) or "测试数据集"
    kwargs.setdefault("classes", ("shape",))
    zip_path = build_dataset_zip(tmp_path / "dataset.zip", **kwargs)
    return datasets.import_dataset_zip(zip_path, name)


# ---------------------------------------------------------------- 正常导入
def test_import_valid_dataset(tmp_path):
    meta = _import(tmp_path)
    assert meta["nc"] == 1
    assert meta["train_images"] == 4
    assert meta["val_images"] == 2
    assert meta["names"] == {"0": "shape"}
    assert meta["status"] == "有效"
    # 落盘结构
    ddir = config.DATASETS_DIR / meta["id"]
    assert (ddir / "data.yaml").is_file()
    assert (ddir / "meta.json").is_file()
    assert (ddir / "images" / "train").is_dir()
    # 内部 data.yaml 指向存储目录
    text = (ddir / "data.yaml").read_text(encoding="utf-8")
    assert str(ddir).replace("\\", "/") in text.replace("\\", "/")
    assert datasets.dataset_yaml_path(meta["id"]).is_file()


def test_import_nested_root_dir(tmp_path):
    meta = _import(tmp_path, nested=True)
    assert meta["train_images"] == 4
    assert meta["val_images"] == 2


def test_import_multiple_classes(tmp_path):
    meta = _import(tmp_path, classes=("person", "bag"))
    assert meta["nc"] == 2
    assert meta["names"] == {"0": "person", "1": "bag"}


def test_background_images_allowed(tmp_path):
    """部分图片没有标签文件（背景图）应当允许。"""
    meta = _import(tmp_path, label_mode="background_mixed")
    assert meta["train_images"] == 4
    assert meta["splits"]["train"]["label_files"] == 2
    assert meta["splits"]["train"]["background_images"] == 2


def test_list_and_get_dataset(tmp_path):
    meta = _import(tmp_path)
    rows = datasets.list_datasets()
    assert len(rows) == 1 and rows[0]["id"] == meta["id"]
    assert datasets.get_dataset(meta["id"])["name"] == "测试数据集"
    table = datasets.dataset_table_rows(rows)
    assert table[0][0] == "测试数据集" and table[0][4] == 4
    assert "测试数据集" in datasets.dataset_info_markdown(meta)


def test_preview_images(tmp_path):
    meta = _import(tmp_path)
    images = datasets.preview_images(meta["id"], count=8, overlay=False, split="all")
    assert 0 < len(images) <= 8
    assert all(img.size[0] > 0 for img, _ in images)
    overlay = datasets.preview_images(meta["id"], count=4, overlay=True, split="train")
    assert len(overlay) == 4


# ---------------------------------------------------------------- 结构错误
def test_missing_data_yaml(tmp_path):
    with pytest.raises(datasets.DatasetError, match="data.yaml"):
        _import(tmp_path, with_data_yaml=False)


def test_no_dataset_dir_at_all(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "not a dataset")
    with pytest.raises(datasets.DatasetError, match="data.yaml"):
        datasets.import_dataset_zip(zip_path)


def test_empty_val_split(tmp_path):
    with pytest.raises(datasets.DatasetError, match="val 图片目录不存在或为空"):
        _import(tmp_path, empty_val=True)


def test_all_labels_missing(tmp_path):
    with pytest.raises(datasets.DatasetError, match="未找到对应标签"):
        _import(tmp_path, all_labels_missing=True)


def test_nc_mismatch(tmp_path):
    with pytest.raises(datasets.DatasetError, match="不一致"):
        _import(tmp_path, yaml_overrides={"nc": 5})


def test_names_not_continuous(tmp_path):
    with pytest.raises(datasets.DatasetError, match="连续编号"):
        _import(tmp_path, yaml_overrides={"names": {0: "a", 2: "b"}})


def test_names_empty(tmp_path):
    with pytest.raises(datasets.DatasetError, match="至少需要一个类别"):
        _import(tmp_path, yaml_overrides={"names": {}})


# ---------------------------------------------------------------- 标签格式错误
@pytest.mark.parametrize(
    "mode, expect",
    [
        ("detection", "分割多边形至少需要 3 个点"),
        ("two_points", "分割多边形至少需要 3 个点"),
        ("out_of_range", "超出 0~1 归一化范围"),
        ("negative", "超出 0~1 归一化范围"),
        ("nan", "NaN/Inf"),
        ("odd", "不是偶数"),
        ("bad_class", "不在 names 中"),
        ("float_class", "不是整数"),
    ],
)
def test_invalid_labels_rejected(tmp_path, mode, expect):
    with pytest.raises(datasets.DatasetError, match=expect):
        _import(tmp_path, label_mode=mode)
    # 失败时不得留下不完整数据
    assert list(config.DATASETS_DIR.iterdir()) == []
    assert list(config.TMP_DIR.iterdir()) == []


# ---------------------------------------------------------------- ZIP 安全
def test_path_traversal_rejected(tmp_path):
    with pytest.raises(datasets.DatasetError, match="路径穿越|绝对路径"):
        _import(tmp_path, extra_members=[("../../evil.txt", b"pwned")])
    assert not (tmp_path / "evil.txt").exists()


def test_absolute_path_rejected(tmp_path):
    with pytest.raises(datasets.DatasetError, match="绝对路径"):
        _import(tmp_path, extra_members=[("/etc/evil.txt", b"pwned")])


def test_symlink_rejected(tmp_path):
    with pytest.raises(datasets.DatasetError, match="符号链接"):
        _import(tmp_path, symlink_members=["images/train/link.png"])


def test_zip_bomb_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_ZIP_UNCOMPRESSED_MB", 1)  # 1 MB 上限
    big = b"\x00" * (8 * 1024 * 1024)
    with pytest.raises(datasets.DatasetError, match="解压炸弹|超过上限"):
        _import(tmp_path, extra_members=[("images/train/big.png", big)])


def test_non_zip_rejected(tmp_path):
    p = tmp_path / "not.zip"
    p.write_bytes(b"hello world")
    with pytest.raises(datasets.DatasetError, match="not a zip|不是有效的 ZIP|BadZipFile|压缩包"):
        datasets.import_dataset_zip(p)


def test_unsafe_dataset_id_rejected():
    for bad in ("../etc", "a/b", "a\\b"):
        with pytest.raises(datasets.DatasetError, match="非法"):
            datasets.dataset_yaml_path(bad)


# ---------------------------------------------------------------- 删除
def test_delete_dataset(tmp_path):
    meta = _import(tmp_path)
    msg = datasets.delete_dataset(meta["id"])
    assert "已删除" in msg
    assert datasets.get_dataset(meta["id"]) is None
    assert not (config.DATASETS_DIR / meta["id"]).exists()


def test_delete_missing_dataset(tmp_path):
    with pytest.raises(datasets.DatasetError, match="不存在"):
        datasets.delete_dataset("ds_not_exist")


def test_delete_blocked_while_training(tmp_path):
    """正在被训练任务使用的数据集不允许删除。"""
    meta = _import(tmp_path)
    run_id = "run_fake_0001"
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config.atomic_write_json(
        run_dir / "task.json",
        {
            "run_id": run_id, "name": "fake", "status": "RUNNING", "pid": 0,
            "started_at": config.now_iso(), "config": {"device": "cpu"},
            "dataset": {"id": meta["id"], "name": meta["name"]},
            "model": {"id": "mdl_fake"},
        },
    )
    assert training.is_resource_in_use("dataset", meta["id"]) is True
    with pytest.raises(datasets.DatasetError, match="正在被训练任务使用"):
        datasets.delete_dataset(meta["id"])
    assert (config.DATASETS_DIR / meta["id"]).exists()