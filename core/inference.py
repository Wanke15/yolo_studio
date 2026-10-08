"""在线推理：单张图片分割/检测推理、结果可视化与统计。

直接调用 Ultralytics predict API，不做批量/视频/摄像头推理。
两类模型均使用 results[0].plot() 可视化，按实例数量统计。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, models, training


class InferenceError(Exception):
    """推理失败，message 直接用于界面提示。"""


def run_inference(
    model_id: str,
    image_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
) -> dict:
    """对单张图片做分割推理。

    返回 dict: original(PIL) / plotted(np.ndarray RGB) / counts / total / result_path / message
    未检测到目标不算失败，正常返回结果图与提示。
    """
    meta = models.get_model(model_id)
    if not meta:
        raise InferenceError("请选择有效的模型")
    if meta.get("task") not in config.SUPPORTED_TASKS:
        raise InferenceError(
            f"模型「{meta['name']}」的任务类型为 {meta.get('task')}，"
            "本平台只支持 segment（实例分割）与 detect（目标检测）模型"
        )

    image_path = Path(image_path)
    if not image_path.is_file() or image_path.suffix.lower() not in config.IMAGE_EXTS:
        raise InferenceError("请上传 jpg / jpeg / png 格式的图片")

    try:
        conf = float(conf)
    except (TypeError, ValueError):
        raise InferenceError("置信度阈值必须是数字")
    if not 0.0 < conf <= 1.0:
        raise InferenceError("置信度阈值必须在 0~1 之间")

    imgsz = int(imgsz)
    if imgsz not in config.IMG_SIZE_CHOICES:
        raise InferenceError(f"Image Size 必须是 {config.IMG_SIZE_CHOICES} 之一")

    device = device or training.default_device()
    # GPU 训练期间禁用 GPU 推理，避免显存冲突
    if device.startswith("cuda") and training.gpu_training_active():
        raise InferenceError("当前有训练任务正在使用 GPU，已禁用 GPU 推理。请等待训练结束，或改用 CPU 推理")

    try:
        from ultralytics import YOLO

        model = YOLO(str(models.model_path(model_id)))
        results = model.predict(
            source=str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False
        )
    except InferenceError:
        raise
    except Exception as e:  # noqa: BLE001 - 统一转换为友好提示
        raise InferenceError(f"推理执行失败: {e}")

    result = results[0]
    original = Image.open(image_path).convert("RGB")
    plotted_bgr = result.plot()  # BGR ndarray
    plotted = np.ascontiguousarray(plotted_bgr[:, :, ::-1])  # -> RGB

    # 按实例数量统计（每个检测实例一次，不按像素/多边形点数）
    names = dict(getattr(result, "names", None) or getattr(model, "names", {}) or {})
    counts: Counter[str] = Counter()
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        for cls_id in boxes.cls.tolist():
            counts[names.get(int(cls_id), str(int(cls_id)))] += 1
    total = int(sum(counts.values()))

    # 保存结果图供下载
    out_dir = config.INFER_DIR / config.new_id("inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{image_path.stem}_result.jpg"
    Image.fromarray(plotted).save(result_path, quality=92)

    message = "" if total else "未检测到目标"
    return {
        "original": original,
        "plotted": plotted,
        "counts": counts,
        "total": total,
        "result_path": str(result_path),
        "model_name": meta["name"],
        "task": meta.get("task"),
        "device": device,
        "conf": conf,
        "imgsz": imgsz,
        "message": message,
    }


def stats_markdown(result: dict) -> str:
    """把推理结果统计转换为 Markdown 文本。"""
    lines = [
        f"### 推理结果",
        "",
        f"- **模型**：{result['model_name']}（{config.task_label(result.get('task'))}）",
        f"- **设备**：{result['device']} ｜ Conf：{result['conf']} ｜ imgsz：{result['imgsz']}",
        f"- **检测到的实例总数**：**{result['total']}**",
    ]
    if result["total"]:
        lines.append("")
        lines.append("| 类别 | 实例数 |")
        lines.append("|---|---|")
        for name, n in result["counts"].most_common():
            lines.append(f"| {name} | {n} |")
    else:
        lines.append("")
        lines.append("提示：未检测到目标。可尝试降低 Conf 阈值后重新推理。")
    return "\n".join(lines)