"""测试辅助：合成 YOLO-Seg 数据集 ZIP、生成真实（未训练）YOLO-Seg 权重、等待任务状态。"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

IMG_SIZE = 64


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    """生成一张带三角形的合成图片。"""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color)
    draw = ImageDraw.Draw(img)
    draw.polygon([(10, 10), (IMG_SIZE - 12, 20), (30, IMG_SIZE - 12)], fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _label_lines(mode: str, cls: int = 0) -> str:
    """按模式生成标签内容。"""
    poly = "0.10 0.10 0.90 0.20 0.30 0.90"
    if mode == "valid":
        return f"{cls} {poly}\n"
    if mode == "detection":
        return f"{cls} 0.50 0.50 0.20 0.30\n"          # 检测框格式（4 个坐标）
    if mode == "two_points":
        return f"{cls} 0.10 0.10 0.90 0.90\n"           # 只有 2 个点
    if mode == "out_of_range":
        return f"{cls} 0.10 0.10 1.50 0.20 0.30 0.90\n"
    if mode == "negative":
        return f"{cls} -0.10 0.10 0.90 0.20 0.30 0.90\n"
    if mode == "nan":
        return f"{cls} 0.10 0.10 nan 0.20 0.30 0.90\n"
    if mode == "odd":
        return f"{cls} 0.10 0.10 0.90 0.20 0.30 0.90 0.50\n"  # 7 个坐标（奇数，无法组成点对）
    if mode == "bad_class":
        return f"7 {poly}\n"
    if mode == "float_class":
        return f"0.0 {poly}\n"
    raise ValueError(mode)


def build_dataset_zip(
    zip_path: Path,
    *,
    n_train: int = 4,
    n_val: int = 2,
    classes: tuple[str, ...] = ("shape",),
    nested: bool = False,
    label_mode: str = "valid",
    label_mode_train: str | None = None,
    with_data_yaml: bool = True,
    yaml_overrides: dict | None = None,
    all_labels_missing: bool = False,
    empty_val: bool = False,
    extra_members: list[tuple[str, bytes]] | None = None,
    symlink_members: list[str] | None = None,
) -> Path:
    """构造一个 YOLO-Seg 数据集 ZIP。label_mode_train 可单独指定 train 的标签模式。"""
    zip_path = Path(zip_path)
    prefix = "my_dataset/" if nested else ""
    train_mode = label_mode_train or label_mode

    ns = {"nc": len(classes), "names": {i: c for i, c in enumerate(classes)},
          "path": ".", "train": "images/train", "val": "images/val"}
    ns.update(yaml_overrides or {})

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if with_data_yaml:
            zf.writestr(prefix + "data.yaml", yaml.safe_dump(ns, allow_unicode=True))

        for i in range(n_train):
            name = f"train_{i}.png"
            zf.writestr(prefix + f"images/train/{name}", _image_bytes((40 + i * 8, 80, 120)))
            if all_labels_missing:
                continue
            if label_mode == "background_mixed" and i % 2 == 1:
                continue  # 一半图片没有标签文件（背景图）
            mode = "valid" if label_mode == "background_mixed" else train_mode
            zf.writestr(prefix + f"labels/train/{name[:-4]}.txt", _label_lines(mode))

        if n_val:
            if empty_val:
                zf.writestr(prefix + "images/val/.keep", b"")
            for i in range(n_val):
                name = f"val_{i}.png"
                if not empty_val:
                    zf.writestr(prefix + f"images/val/{name}", _image_bytes((120, 60 + i * 8, 60)))
                    if not all_labels_missing:
                        zf.writestr(prefix + f"labels/val/{name[:-4]}.txt", _label_lines("valid"))

        for name, data in extra_members or []:
            zf.writestr(prefix + name, data)

    # 符号链接条目需要手工构造 ZipInfo（writestr 无法直接表达）
    if symlink_members:
        with zipfile.ZipFile(zip_path, "a") as zf:
            for name in symlink_members:
                info = zipfile.ZipInfo(prefix + name)
                info.create_system = 3  # unix
                info.external_attr = (0o120777 << 16)  # S_IFLNK | 0777
                zf.writestr(info, "/etc/passwd")
    return zip_path


def build_tiny_seg_pt(out_path: Path, nc: int = 1) -> Path:
    """用 ultralytics 自带的 yaml 构造一个真实的（未训练）segment 权重，完全离线。"""
    import ultralytics
    from ultralytics import YOLO

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    src = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8-seg.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["nc"] = nc
    # 文件名中包含 yolov8n 以便 ultralytics 识别为 nano 规模（测试更快、权重更小）
    tmp_yaml = out_path.parent / f"yolov8n-seg-nc{nc}.yaml"
    tmp_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = YOLO(str(tmp_yaml))
    model.save(str(out_path))
    return out_path


def build_tiny_detect_pt(out_path: Path, nc: int = 1) -> Path:
    """构造一个真实的 detect 权重（用于验证非 segment 模型会被拒绝）。"""
    import ultralytics
    from ultralytics import YOLO

    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    src = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["nc"] = nc
    tmp_yaml = out_path.parent / f"yolov8n-detect-nc{nc}.yaml"
    tmp_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = YOLO(str(tmp_yaml))
    model.save(str(out_path))
    return out_path


def wait_for_status(run_id: str, timeout: float = 1200.0, interval: float = 2.0) -> dict:
    """轮询训练任务直到离开 RUNNING 状态（内部会调用 training.poll()）。"""
    from core import training

    deadline = time.time() + timeout
    while time.time() < deadline:
        training.poll()
        task = training.get_run(run_id)
        if task and task.get("status") != "RUNNING":
            return task
        time.sleep(interval)
    raise TimeoutError(f"任务 {run_id} 在 {timeout}s 内未结束")