"""模型管理：.pt 权重上传、格式/任务校验、索引、删除、下载。

模型目录结构：
    storage/models/{model_id}/model.pt
    storage/models/{model_id}/meta.json
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from . import config


class ModelError(Exception):
    """模型校验/导入失败，message 直接用于界面提示。"""


def _inspect_pt(path: Path) -> dict:
    """检查 .pt 文件是否为受支持的 YOLO 模型，返回 {'task', 'nc', 'names'}。

    支持 segment（实例分割）与 detect（目标检测）；classify / pose / obb 会被拒绝。

    注意：.pt 的反序列化存在安全风险，本平台仅允许受信任人员上传权重。
    后缀与格式校验只是基本的健全性检查，不构成安全保证。
    """
    if path.suffix.lower() != ".pt":
        raise ModelError("只支持 .pt 模型权重文件")
    if not path.is_file() or path.stat().st_size == 0:
        raise ModelError("模型文件为空或不存在")
    if path.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ModelError(f"模型文件超过上限 {config.MAX_UPLOAD_MB} MB")
    if not zipfile.is_zipfile(path):
        raise ModelError("文件不是有效的 PyTorch 权重格式（.pt 应为 zip 归档），可能已损坏或不是权重文件")

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        task = getattr(model, "task", None)
    except Exception as e:  # noqa: BLE001 - 统一转换为友好的错误提示
        raise ModelError(f"无法加载权重文件，可能不是有效的 YOLO 模型: {e}")

    if task not in config.SUPPORTED_TASKS:
        raise ModelError(
            f"该模型的任务类型为「{task or '未知'}」，本平台只支持 "
            "segment（实例分割）与 detect（目标检测）模型"
        )
    names = getattr(model, "names", {}) or {}
    return {"task": task, "nc": len(names), "names": [str(v) for v in names.values()][:50]}


def import_model_file(
    src_path: str | Path, name: str | None = None, source: str = "upload", run_id: str | None = None
) -> dict:
    """校验并导入模型权重，返回 meta dict。同名模型不会互相覆盖（使用唯一 ID）。"""
    config.ensure_dirs()
    src = Path(src_path)
    if not src.is_file():
        raise ModelError(f"模型文件不存在: {src}")

    info = _inspect_pt(src)

    model_id = config.new_id("mdl")
    dest_dir = config.MODELS_DIR / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "model.pt")

    meta = {
        "id": model_id,
        "name": (name or src.stem).strip() or src.stem,
        "filename": src.name,
        "size_mb": round(src.stat().st_size / 1024 / 1024, 2),
        "task": info["task"],
        "nc": info["nc"],
        "names": info["names"],
        "created_at": config.now_iso(),
        "source": source,          # upload | training
        "run_id": run_id,
    }
    config.atomic_write_json(dest_dir / "meta.json", meta)
    return meta


def register_run_best(run_id: str, best_pt: str | Path, name: str | None = None) -> dict | None:
    """训练完成后把 best.pt 加入模型列表。已注册过则跳过，避免重复。"""
    for m in list_models():
        if m.get("run_id") == run_id and m.get("source") == "training":
            return None
    best_pt = Path(best_pt)
    if not best_pt.is_file():
        return None
    return import_model_file(best_pt, name=name, source="training", run_id=run_id)


def list_models(task: str | None = None) -> list[dict]:
    """列出已导入模型（按导入时间倒序）；task 为空时返回全部任务类型。"""
    config.ensure_dirs()
    metas = []
    for d in config.MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = config.read_json(d / "meta.json")
        if not meta or not (d / "model.pt").is_file():
            continue
        if task and meta.get("task") != task:
            continue
        metas.append(meta)
    return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)


def get_model(model_id: str) -> dict | None:
    if not model_id:
        return None
    meta = config.read_json(config.MODELS_DIR / model_id / "meta.json")
    if meta and (config.MODELS_DIR / model_id / "model.pt").is_file():
        return meta
    return None


def model_path(model_id: str) -> Path:
    """返回模型权重的绝对路径。"""
    if not model_id or "/" in model_id or "\\" in model_id or ".." in model_id:
        raise ModelError("非法的模型 ID")
    path = config.MODELS_DIR / model_id / "model.pt"
    if not path.is_file():
        raise ModelError(f"模型文件不存在: {model_id}")
    return path


def delete_model(model_id: str) -> str:
    """删除模型。正在被训练任务使用时拒绝删除。"""
    from . import training  # 延迟导入，避免循环依赖

    meta = get_model(model_id)
    if not meta:
        raise ModelError(f"模型不存在: {model_id}")
    if training.is_resource_in_use("model", model_id):
        raise ModelError(f"模型「{meta['name']}」正在被训练任务使用，无法删除")
    shutil.rmtree(config.MODELS_DIR / model_id)
    return f"模型「{meta['name']}」已删除"


MODEL_TABLE_HEADERS = ["名称", "模型 ID", "任务", "类别数", "大小(MB)", "来源", "上传时间"]


def model_table_rows(models: list[dict]) -> list[list]:
    source_label = {"upload": "上传", "training": "训练产出"}
    return [
        [
            m["name"],
            m["id"],
            m["task"],
            m.get("nc", "-"),
            m.get("size_mb", "-"),
            source_label.get(m.get("source", "upload"), m.get("source", "-")),
            m["created_at"],
        ]
        for m in models
    ]


def model_choices(models: list[dict]) -> list[tuple[str, str]]:
    """生成 Gradio Dropdown 选项 [(显示文本, model_id)]。"""
    return [
        (f"{m['name']} | {m['id']} | {m.get('size_mb', '?')}MB | {m.get('task', '')}", m["id"])
        for m in models
    ]