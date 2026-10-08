"""训练子进程入口：加载任务配置并调用 Ultralytics train API。

由 core.training.start_training 通过 subprocess 启动，独立进程运行，
结束后把结果写入 result.json（原子写入），供主进程轮询。

用法：
    python train_worker.py /path/to/runs/{run_id}/task.json
"""
from __future__ import annotations

import json
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import atomic_write_json, now_iso  # noqa: E402


def _write_result(run_dir: Path, status: str, message: str = "", output_dir: str | None = None) -> None:
    atomic_write_json(
        run_dir / "result.json",
        {
            "status": status,
            "message": message,
            "output_dir": output_dir,
            "finished_at": now_iso(),
            "pid": __import__("os").getpid(),
        },
    )


def _on_sigterm(signum, frame):  # noqa: ANN001, ARG001
    raise SystemExit("SIGTERM")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python train_worker.py <task.json>", file=sys.stderr)
        return 2
    task_json = Path(sys.argv[1]).resolve()
    run_dir = task_json.parent
    task = json.loads(task_json.read_text(encoding="utf-8"))
    cfg = task["config"]

    # 只终止当前任务所属进程组时，SIGTERM 会到达这里
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, AttributeError):
        pass

    print(f"[worker] run_id={task['run_id']} name={task['name']}", flush=True)
    print(
        f"[worker] dataset={task['dataset']['yaml_path']} model={task['model']['path']}\n"
        f"[worker] epochs={cfg['epochs']} batch={cfg['batch']} imgsz={cfg['imgsz']} "
        f"device={cfg['device']} workers={cfg['workers']}",
        flush=True,
    )

    try:
        from ultralytics import YOLO

        model = YOLO(task["model"]["path"])
        model_task = getattr(model, "task", None)
        if model_task not in ("segment", "detect"):
            raise RuntimeError(
                f"权重任务类型为 {model_task}，本平台只支持 segment（实例分割）与 detect（目标检测）"
            )
        if model_task == "segment" and (task["dataset"].get("task") or "segment") != "segment":
            raise RuntimeError("segment 模型需要多边形标签，当前数据集为检测框标签")
        # 训练核心：直接使用 Ultralytics 官方 API，参数全部来自任务配置
        # （detect 模型配分割数据集时，Ultralytics 会自动把多边形转换为外接框）
        model.train(
            data=task["dataset"]["yaml_path"],
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            imgsz=cfg["imgsz"],
            device=cfg["device"],
            workers=cfg["workers"],
            project=str(run_dir),
            name="output",
            exist_ok=False,
            plots=True,
            verbose=True,
        )
        save_dir = str(getattr(model.trainer, "save_dir", run_dir / "output"))
        print(f"[worker] 训练完成，输出目录: {save_dir}", flush=True)
        _write_result(run_dir, "COMPLETED", "训练完成", save_dir)
        return 0
    except SystemExit:
        print("[worker] 收到停止信号，训练已终止", flush=True)
        _write_result(run_dir, "STOPPED", "训练进程被外部停止")
        return 130
    except Exception as e:  # noqa: BLE001 - 训练失败需要完整记录原因
        tb = traceback.format_exc()
        print(tb, flush=True)
        text = str(e)
        if "out of memory" in text.lower() or "CUDA out of memory" in tb:
            msg = "训练失败：GPU 显存不足（CUDA OOM）。请减小 Batch Size / Image Size 后重试。"
        elif "CUDA error" in tb or "no kernel image" in tb:
            msg = "训练失败：CUDA 运行时错误。请检查 PyTorch 与 NVIDIA 驱动的兼容性，或改用 CPU。"
        else:
            msg = f"训练失败：{text.splitlines()[0][:300] if text else type(e).__name__}"
        _write_result(run_dir, "FAILED", msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())