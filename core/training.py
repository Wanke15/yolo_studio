"""训练管理：配置校验、独立子进程启动、状态轮询、停止、历史记录、指标读取。

设计要点：
- 单任务模式：同一时间最多一个 RUNNING 任务，状态持久化在 storage/runs/{run_id}/task.json。
- 训练通过 subprocess 启动 train_worker.py，绝不阻塞 Gradio 主进程。
- 停止训练时终止整个进程组，避免遗留 GPU 训练进程。
- 服务重启后，无法确认状态的历史 RUNNING 任务标记为 INTERRUPTED，不自动恢复。
"""
from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config, datasets, models

MAX_LOG_TAIL_BYTES = 256 * 1024  # 日志只读取尾部，避免大文件拖慢 UI


class TrainingError(Exception):
    """训练配置/启动错误，message 直接用于界面提示。"""


# ================================================================ 设备信息
def available_devices() -> list[str]:
    """返回可用设备列表：['cpu'] 或 ['cuda:0', ..., 'cpu']。"""
    devices: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 - 无 torch/CUDA 时退化为 CPU
        pass
    return devices + ["cpu"]


def default_device() -> str:
    """默认设备：第一块 CUDA GPU，无 GPU 时使用 CPU。"""
    devices = available_devices()
    return devices[0] if devices[0] != "cpu" else "cpu"


def has_gpu() -> bool:
    return any(d.startswith("cuda") for d in available_devices())


def gpu_info() -> list[dict]:
    """返回 GPU 名称与显存使用情况，获取失败返回空列表（界面显示“不可用”）。"""
    info: list[dict] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        for i in range(torch.cuda.device_count()):
            entry = {"index": i, "name": torch.cuda.get_device_name(i), "used_mb": None, "total_mb": None}
            try:
                free, total = torch.cuda.mem_get_info(i)
                entry["total_mb"] = round(total / 1024**2)
                entry["used_mb"] = round((total - free) / 1024**2)
            except Exception:  # noqa: BLE001
                pass
            info.append(entry)
    except Exception:  # noqa: BLE001
        pass
    return info


# ================================================================ 内部状态
_lock = threading.RLock()
_procs: dict[str, subprocess.Popen] = {}
_log_files: dict[str, object] = {}


def run_dir(run_id: str) -> Path:
    _check_run_id(run_id)
    return config.RUNS_DIR / run_id


def _check_run_id(run_id: str) -> None:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise TrainingError("非法的训练任务 ID")


def get_run(run_id: str) -> dict | None:
    if not run_id:
        return None
    try:
        return config.read_json(run_dir(run_id) / "task.json")
    except TrainingError:
        return None


def _save_run(task: dict) -> None:
    config.atomic_write_json(config.RUNS_DIR / task["run_id"] / "task.json", task)


def list_runs() -> list[dict]:
    """列出所有训练任务（按开始时间倒序）。"""
    config.ensure_dirs()
    runs = []
    for d in config.RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        task = config.read_json(d / "task.json")
        if task:
            runs.append(task)
    return sorted(runs, key=lambda t: t.get("started_at", ""), reverse=True)


def running_runs() -> list[dict]:
    return [t for t in list_runs() if t.get("status") == config.RUN_STATUS_RUNNING]


def current_run() -> dict | None:
    runs = running_runs()
    return runs[0] if runs else None


def is_resource_in_use(kind: str, resource_id: str) -> bool:
    """数据集/模型是否被正在运行（或已启动）的训练任务占用。"""
    key = "dataset" if kind == "dataset" else "model"
    for task in running_runs():
        if (task.get(key) or {}).get("id") == resource_id:
            return True
    return False


def gpu_training_active() -> bool:
    """是否有训练任务正在占用 GPU（用于禁用 GPU 推理）。"""
    for task in running_runs():
        if str((task.get("config") or {}).get("device", "")).startswith("cuda"):
            return True
    return False


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def init_manager() -> list[str]:
    """服务启动时调用：把无法确认状态的旧 RUNNING 任务标记为 INTERRUPTED。"""
    config.ensure_dirs()
    messages: list[str] = []
    with _lock:
        for task in running_runs():
            pid = int(task.get("pid") or 0)
            alive = _pid_alive(pid)
            task["status"] = config.RUN_STATUS_INTERRUPTED
            task["finished_at"] = config.now_iso()
            task["message"] = (
                f"服务重启，无法确认训练进程状态（原 PID {pid} {'仍存活' if alive else '已退出'}），"
                "已标记为中断。不自动恢复训练。"
            )
            _save_run(task)
            messages.append(f"任务 {task['run_id']} 已标记为 INTERRUPTED")
    return messages


# ================================================================ 配置校验
def validate_config(epochs, batch, imgsz, workers) -> dict:
    def _int(name: str, value, lo: int, hi: int) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise TrainingError(f"{name} 必须是整数")
        if not lo <= v <= hi:
            raise TrainingError(f"{name} 必须在 {lo} ~ {hi} 之间")
        return v

    imgsz = _int("Image Size", imgsz, 32, 4096)
    if imgsz not in config.IMG_SIZE_CHOICES:
        raise TrainingError(f"Image Size 必须是 {config.IMG_SIZE_CHOICES} 之一")
    return {
        "epochs": _int("Epochs", epochs, *config.EPOCHS_RANGE),
        "batch": _int("Batch Size", batch, *config.BATCH_RANGE),
        "imgsz": imgsz,
        "workers": _int("Workers", workers, *config.WORKERS_RANGE),
    }


def _validate_device(device: str) -> str:
    device = (device or "").strip()
    if not device:
        return default_device()
    if device == "cpu":
        return device
    if device.startswith("cuda:") and device[5:].isdigit():
        if device not in available_devices():
            raise TrainingError(f"设备 {device} 不可用，当前可用设备: {', '.join(available_devices())}")
        return device
    raise TrainingError(f"无效的训练设备: {device}")


# ================================================================ 启动 / 停止
def start_training(
    dataset_id: str,
    model_id: str,
    epochs=50,
    batch=8,
    imgsz=640,
    device: str | None = None,
    workers: int = 2,
    name: str | None = None,
) -> dict:
    """启动训练任务，返回 task dict。任何校验失败抛 TrainingError。"""
    config.ensure_dirs()
    with _lock:
        busy = current_run()
        if busy:
            raise TrainingError(
                f"已有训练任务正在运行（{busy['run_id']} / {busy['name']}），请先等待完成或停止它"
            )

        dataset_meta = datasets.get_dataset(dataset_id)
        if not dataset_meta:
            raise TrainingError("请选择有效的数据集")
        model_meta = models.get_model(model_id)
        if not model_meta:
            raise TrainingError("请选择有效的模型")
        if model_meta.get("task") != "segment":
            raise TrainingError("V1 仅支持 segment 分割模型训练")

        cfg = validate_config(epochs, batch, imgsz, workers)
        cfg["device"] = _validate_device(device or "")
        yaml_path = datasets.dataset_yaml_path(dataset_id)
        weight_path = models.model_path(model_id)

        run_id = config.new_id("run")
        default_name = f"{dataset_meta['name']}_{config.new_id('t')[-4:]}"
        task = {
            "run_id": run_id,
            "name": (name or "").strip() or default_name,
            "status": config.RUN_STATUS_RUNNING,
            "pid": None,
            "started_at": config.now_iso(),
            "finished_at": None,
            "exit_code": None,
            "message": "",
            "config": cfg,
            # 数据集/模型元信息快照：即使之后被删除，历史记录仍然完整
            "dataset": {
                "id": dataset_meta["id"],
                "name": dataset_meta["name"],
                "nc": dataset_meta["nc"],
                "train_images": dataset_meta["train_images"],
                "val_images": dataset_meta["val_images"],
                "yaml_path": str(yaml_path),
            },
            "model": {
                "id": model_meta["id"],
                "name": model_meta["name"],
                "task": model_meta["task"],
                "path": str(weight_path),
                "size_mb": model_meta.get("size_mb"),
            },
            "output_dir": None,
            "best_pt": None,
            "last_pt": None,
            "results_csv": None,
            "best": None,
            "registered_model_id": None,
            "log_path": str(config.RUNS_DIR / run_id / "train.log"),
        }
        rd = config.RUNS_DIR / run_id
        rd.mkdir(parents=True, exist_ok=True)
        _save_run(task)

        task_json = rd / "task.json"
        log_file = open(task["log_path"], "ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("YOLO_CONFIG_DIR", str(config.STORAGE_DIR / ".ultralytics"))
        cmd = [sys.executable, "-u", str(config.BASE_DIR / "train_worker.py"), str(task_json)]
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # 独立进程组，便于整组终止
        try:
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                # cwd 设为本次任务目录：Ultralytics 可能产生相对路径的附属目录（如 mlflow），
                # 这样它们都落在该任务的 run 目录内，不会污染项目根目录
                cwd=str(rd), env=env, **kwargs,
            )
        except Exception as e:  # noqa: BLE001
            log_file.close()
            task["status"] = config.RUN_STATUS_FAILED
            task["message"] = f"启动训练子进程失败: {e}"
            task["finished_at"] = config.now_iso()
            _save_run(task)
            raise TrainingError(task["message"])

        _procs[run_id] = proc
        _log_files[run_id] = log_file
        task["pid"] = proc.pid
        _save_run(task)
        return task


def _kill_process_group(task: dict) -> None:
    """仅终止当前任务所属进程组，必要时强制终止。"""
    run_id = task["run_id"]
    proc = _procs.get(run_id)
    pid = int(task.get("pid") or 0)
    if os.name == "nt":
        target = pid or (proc.pid if proc else 0)
        if target:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(target)], capture_output=True, text=True)
        return
    target = pid or (proc.pid if proc else 0)
    if not target:
        return
    try:
        pgid = os.getpgid(target)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGTERM, 8), (signal.SIGKILL, 0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        if wait == 0:
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            if not _pid_alive(target):
                return
            time.sleep(0.3)


def stop_training(run_id: str) -> str:
    """停止指定训练任务（终止其进程组），状态置为 STOPPED。"""
    with _lock:
        task = get_run(run_id)
        if not task:
            raise TrainingError(f"训练任务不存在: {run_id}")
        if task.get("status") != config.RUN_STATUS_RUNNING:
            raise TrainingError(f"任务当前状态为 {task['status']}，无需停止")
        _kill_process_group(task)
        task["status"] = config.RUN_STATUS_STOPPED
        task["finished_at"] = config.now_iso()
        task["message"] = "已由用户手动停止"
        _save_run(task)
        proc = _procs.pop(run_id, None)
        if proc:
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        log_file = _log_files.pop(run_id, None)
        if log_file:
            try:
                log_file.close()
            except Exception:  # noqa: BLE001
                pass
        return f"训练任务 {run_id} 已停止"


# ================================================================ 状态轮询
def poll() -> None:
    """检查正在运行的任务是否结束，更新状态并在训练完成后注册 best.pt。"""
    with _lock:
        for task in running_runs():
            run_id = task["run_id"]
            proc = _procs.get(run_id)
            if proc is None:
                continue  # 非本进程启动（服务重启遗留），已由 init_manager 处理
            rc = proc.poll()
            if rc is None:
                continue  # 仍在运行

            result = config.read_json(config.RUNS_DIR / run_id / "result.json") or {}
            if result.get("status"):
                status = result["status"]
                task["message"] = result.get("message", "")
            else:
                status = config.RUN_STATUS_COMPLETED if rc == 0 else config.RUN_STATUS_FAILED
                task["message"] = "" if rc == 0 else f"训练进程异常退出（exit code {rc}），详见日志"
            if status not in config.RUN_STATUSES:
                status = config.RUN_STATUS_FAILED

            task["status"] = status
            task["exit_code"] = rc
            task["finished_at"] = config.now_iso()
            if result.get("output_dir"):
                task["output_dir"] = result["output_dir"]
            _refresh_output_files(task)
            if status == config.RUN_STATUS_COMPLETED:
                _finalize_completed(task)

            log_file = _log_files.pop(run_id, None)
            if log_file:
                try:
                    log_file.close()
                except Exception:  # noqa: BLE001
                    pass
            _procs.pop(run_id, None)
            _save_run(task)


def _find_output_dir(run_id: str) -> Path | None:
    base = config.RUNS_DIR / run_id
    if (base / "output").is_dir():
        return base / "output"
    candidates = sorted([p for p in base.glob("output*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _refresh_output_files(task: dict) -> None:
    out = Path(task["output_dir"]) if task.get("output_dir") else _find_output_dir(task["run_id"])
    if not out or not out.is_dir():
        return
    task["output_dir"] = str(out)
    weights = out / "weights"
    for key, fname in (("best_pt", "best.pt"), ("last_pt", "last.pt")):
        p = weights / fname
        task[key] = str(p) if p.is_file() else None
    csv_path = out / "results.csv"
    task["results_csv"] = str(csv_path) if csv_path.is_file() else None


def _finalize_completed(task: dict) -> None:
    """训练成功结束：汇总最优指标，并把 best.pt 注册到模型列表。"""
    _refresh_output_files(task)
    metrics = read_metrics(task["run_id"])
    task["best"] = metrics.get("best") if metrics else None
    if task.get("best_pt") and not task.get("registered_model_id"):
        try:
            meta = models.register_run_best(
                task["run_id"], task["best_pt"], name=f"{task['name']}_best"
            )
            if meta:
                task["registered_model_id"] = meta["id"]
                task["message"] = (task.get("message") or "") + (
                    f" 已将 best.pt 注册为模型「{meta['name']}」({meta['id']})"
                )
        except models.ModelError as e:
            task["message"] = (task.get("message") or "") + f" best.pt 注册失败: {e}"


# ================================================================ 日志
def read_log(run_id: str, max_lines: int = 200) -> str:
    """读取训练日志尾部若干行（供 UI 展示）。"""
    task = get_run(run_id)
    path = Path(task["log_path"]) if task and task.get("log_path") else run_dir(run_id) / "train.log"
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > MAX_LOG_TAIL_BYTES:
                f.seek(size - MAX_LOG_TAIL_BYTES)
                f.readline()  # 丢弃可能被截断的首行
            data = f.read()
    except OSError as e:
        return f"日志读取失败: {e}"
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


# ================================================================ 指标
# 不同 Ultralytics 版本的列名可能不同，这里做兼容映射；找不到的指标保持缺失，不填 0
METRIC_ALIASES = {
    "train_box_loss": ["train/box_loss", "train_box_loss"],
    "train_seg_loss": ["train/seg_loss", "train_seg_loss"],
    "val_box_loss": ["val/box_loss", "val_box_loss"],
    "val_seg_loss": ["val/seg_loss", "val_seg_loss"],
    "mask_map50": ["metrics/mAP50(M)", "metrics/mAP50_mask", "metrics/mAP50(Mask)"],
    "mask_map50_95": ["metrics/mAP50-95(M)", "metrics/mAP50-95_mask", "metrics/mAP50-95(Mask)"],
}


def _norm(name: str) -> str:
    """归一化列名：小写并去掉所有非字母数字字符。"""
    import re

    return re.sub(r"[^a-z0-9]", "", name.lower())
METRIC_LABELS = {
    "train_box_loss": "Train Box Loss",
    "train_seg_loss": "Train Seg Loss",
    "val_box_loss": "Val Box Loss",
    "val_seg_loss": "Val Seg Loss",
    "mask_map50": "Mask mAP50",
    "mask_map50_95": "Mask mAP50-95",
}


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """把实际 CSV 列名映射到标准指标键。

    先按别名精确匹配，再退化为“忽略标点/大小写”的等价匹配。
    绝不使用模糊子串匹配，避免把 train/seg_loss 错当成 val/seg_loss。
    """
    normalized = {name.strip(): name for name in fieldnames if name}
    by_normalized = {_norm(name): name for name in normalized}
    mapping: dict[str, str] = {}
    for key, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[key] = normalized[alias]
                break
        else:
            for alias in aliases:
                hit = by_normalized.get(_norm(alias))
                if hit:
                    mapping[key] = hit
                    break
    return mapping


def read_metrics(run_id: str) -> dict | None:
    """读取 results.csv，返回 {'epochs': [...], 'series': {key: [...]}, 'best': {...}}。

    CSV 不存在或没有任何可用指标时返回 None（UI 显示等待状态）。
    """
    task = get_run(run_id)
    csv_path = Path(task["results_csv"]) if task and task.get("results_csv") else None
    if not csv_path or not csv_path.is_file():
        found = _find_output_dir(run_id)
        csv_path = found / "results.csv" if found else None
    if not csv_path or not csv_path.is_file():
        return None

    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = []
            for raw in reader:
                if not raw:
                    continue
                try:
                    epoch = int(float((raw.get("epoch") or raw.get("Epoch") or "").strip()))
                except (TypeError, ValueError):
                    continue  # 训练中最后一行可能不完整
                rows.append((epoch, raw))
    except (OSError, csv.Error) as e:
        return {"error": f"results.csv 读取失败: {e}"}

    if not rows:
        return None
    mapping = _resolve_columns(fieldnames)
    series: dict[str, list] = {key: [] for key in METRIC_ALIASES}
    epochs: list[int] = []
    for epoch, raw in rows:
        epochs.append(epoch)
        for key in METRIC_ALIASES:
            value = None
            col = mapping.get(key)
            if col:
                try:
                    value = float(raw[col])
                except (TypeError, ValueError, KeyError):
                    value = None
            series[key].append(value)

    best: dict[str, dict] = {}
    for key in ("mask_map50", "mask_map50_95"):
        values = series[key]
        pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if pairs:
            e, v = max(pairs, key=lambda p: p[1])
            best[key] = {"epoch": e, "value": v}

    if not any(v is not None for key, vals in series.items() for v in vals):
        return None
    return {
        "epochs": epochs,
        "series": series,
        "best": best,
        "mapping": mapping,
        "columns": [c for c in fieldnames if c],
        "total_epochs_seen": max(epochs) + 1,
    }


def metrics_long_dataframe(run_id: str):
    """转换为 gr.LinePlot 需要的长格式 DataFrame；无数据返回 None。"""
    metrics = read_metrics(run_id)
    if not metrics or metrics.get("error"):
        return None
    import pandas as pd

    rows = []
    for key, label in METRIC_LABELS.items():
        for epoch, value in zip(metrics["epochs"], metrics["series"][key]):
            if value is not None:  # 缺失指标不画点，也不伪造为 0
                rows.append({"epoch": epoch, "指标": label, "数值": value})
    if not rows:
        return None
    return pd.DataFrame(rows)


# ================================================================ UI 辅助
def status_markdown(task: dict | None = None) -> str:
    """训练状态区域 Markdown。"""
    task = task or current_run()
    lines: list[str] = []
    running_gpu = gpu_info()
    if not has_gpu():
        lines.append(
            "> ⚠️ **未检测到可用 CUDA GPU**：当前只能使用 CPU 训练，速度会明显变慢"
            "（可用 GPU 环境请在 GPU 服务器上部署）。"
        )
    if task:
        elapsed = _elapsed_text(task)
        cfg = task.get("config", {})
        metrics = read_metrics(task["run_id"]) or {}
        total_epochs = cfg.get("epochs")
        done = metrics.get("total_epochs_seen")
        lines += [
            f"**当前任务**：`{task['run_id']}` ｜ **名称**：{task['name']}",
            "",
            f"- **状态**：{_status_badge(task['status'])}",
            f"- **开始时间**：{task.get('started_at') or '-'}"
            + (f" ｜ **结束时间**：{task['finished_at']}" if task.get("finished_at") else ""),
            f"- **已运行时间**：{elapsed}",
            f"- **已完成 Epoch / 总 Epoch**：{done if done is not None else '等待中'} / {total_epochs}",
            f"- **数据集**：{task['dataset']['name']} ｜ **基础模型**：{task['model']['name']}",
            f"- **设备**：{cfg.get('device')} ｜ batch={cfg.get('batch')} ｜ imgsz={cfg.get('imgsz')} ｜ "
            f"workers={cfg.get('workers')} ｜ PID={task.get('pid')}",
        ]
        if running_gpu:
            gpu_txt = "；".join(
                f"GPU{g['index']} {g['name']}"
                + (f" 显存 {g['used_mb']}/{g['total_mb']} MB" if g.get("used_mb") is not None else "")
                for g in running_gpu
            )
            lines.append(f"- **GPU 状态**：{gpu_txt}")
        else:
            lines.append("- **GPU 状态**：不可用（未检测到 CUDA 设备）")
        if task.get("message"):
            lines.append(f"- **说明**：{task['message']}")
    else:
        lines.append("**当前没有正在进行的训练任务。**")
    if running_gpu and not task:
        lines.append(
            "**GPU 状态**：" + "；".join(
                f"GPU{g['index']} {g['name']}"
                + (f" 显存 {g['used_mb']}/{g['total_mb']} MB" if g.get("used_mb") is not None else "")
                for g in running_gpu
            )
        )
    return "\n".join(lines)


def _elapsed_text(task: dict) -> str:
    from datetime import datetime

    try:
        start = datetime.fromisoformat(task["started_at"])
    except (KeyError, TypeError, ValueError):
        return "-"
    end = datetime.now()
    if task.get("finished_at"):
        try:
            end = datetime.fromisoformat(task["finished_at"])
        except (TypeError, ValueError):
            pass
    seconds = max(0, int((end - start).total_seconds()))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}小时{m}分{s}秒" if h else f"{m}分{s}秒"


def _status_badge(status: str) -> str:
    emoji = {
        config.RUN_STATUS_RUNNING: "🟢",
        config.RUN_STATUS_COMPLETED: "✅",
        config.RUN_STATUS_FAILED: "❌",
        config.RUN_STATUS_STOPPED: "⏹️",
        config.RUN_STATUS_INTERRUPTED: "⚠️",
    }.get(status, "")
    return f"{emoji} {status}"


def result_markdown(run_id: str) -> str:
    """训练完成后的结果汇总（最优 mAP、产物文件）。"""
    task = get_run(run_id)
    if not task:
        return "任务不存在。"
    metrics = read_metrics(run_id) or {}
    best = task.get("best") or metrics.get("best") or {}
    lines = [f"### 训练结果：{task['name']}", ""]
    if task["status"] != config.RUN_STATUS_COMPLETED:
        lines.append(f"任务状态为 {_status_badge(task['status'])}，以下为当前可用的产物（如有）。")
    for key, label in (("mask_map50", "最优 Mask mAP50"), ("mask_map50_95", "最优 Mask mAP50-95")):
        if key in best:
            lines.append(f"- **{label}**：{best[key]['value']:.4f}（epoch {best[key]['epoch']}）")
        else:
            lines.append(f"- **{label}**：暂无数据")
    lines += ["", "**产物文件**："]
    for key, label in (("best_pt", "best.pt"), ("last_pt", "last.pt"), ("results_csv", "results.csv")):
        path = task.get(key)
        if path and Path(path).is_file():
            size_mb = round(Path(path).stat().st_size / 1024 / 1024, 2)
            lines.append(f"- {label}：`{path}`（{size_mb} MB）")
        else:
            lines.append(f"- {label}：未生成")
    out = task.get("output_dir")
    if out and Path(out).is_dir():
        plots = [p.name for p in sorted(Path(out).glob("*.jpg"))] + \
                [p.name for p in sorted(Path(out).glob("*.png")) if p.name != "results.png"]
        if plots:
            lines.append(f"- 训练结果图：{', '.join(plots[:12])}")
        if (Path(out) / "results.png").is_file():
            lines.append(f"- 指标曲线图：results.png")
    if task.get("registered_model_id"):
        lines.append(f"- 已注册模型 ID：`{task['registered_model_id']}`（可在「模型管理」中下载）")
    if task.get("message"):
        lines.append("")
        lines.append(f"> {task['message']}")
    return "\n".join(lines)


RUN_HISTORY_HEADERS = ["任务 ID", "训练名称", "数据集", "基础模型", "Epochs", "状态", "开始时间", "Mask mAP50-95"]


def run_history_rows(runs: list[dict] | None = None) -> list[list]:
    rows = []
    for t in runs if runs is not None else list_runs():
        best = (t.get("best") or {}).get("mask_map50_95")
        if best is None:
            metrics = read_metrics(t["run_id"]) or {}
            best = (metrics.get("best") or {}).get("mask_map50_95")
        rows.append([
            t["run_id"],
            t["name"],
            (t.get("dataset") or {}).get("name", "-"),
            (t.get("model") or {}).get("name", "-"),
            (t.get("config") or {}).get("epochs", "-"),
            t.get("status", "-"),
            t.get("started_at", "-"),
            f"{best['value']:.4f}" if best else "暂无",
        ])
    return rows


def run_choices(runs: list[dict] | None = None) -> list[tuple[str, str]]:
    return [
        (f"{t['name']} | {t['run_id']} | {t['status']}", t["run_id"])
        for t in (runs if runs is not None else list_runs())
    ]


def run_config_markdown(run_id: str) -> str:
    task = get_run(run_id)
    if not task:
        return "未选择任务。"
    cfg, ds, md = task.get("config", {}), task.get("dataset", {}), task.get("model", {})
    lines = [
        f"### 任务配置：{task['name']}（`{task['run_id']}`）",
        "",
        f"- 状态：{_status_badge(task.get('status', '-'))}"
        + (f" ｜ 说明：{task['message']}" if task.get("message") else ""),
        f"- 开始时间：{task.get('started_at', '-')}"
        + (f" ｜ 结束时间：{task['finished_at']}" if task.get("finished_at") else ""),
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 数据集 | {ds.get('name', '-')} (`{ds.get('id', '-')}`) |",
        f"| 数据集类别数 | {ds.get('nc', '-')} ｜ 训练/验证图片 {ds.get('train_images', '-')}/{ds.get('val_images', '-')} |",
        f"| 基础模型 | {md.get('name', '-')} (`{md.get('id', '-')}`) |",
        f"| Epochs | {cfg.get('epochs', '-')} |",
        f"| Batch Size | {cfg.get('batch', '-')} |",
        f"| Image Size | {cfg.get('imgsz', '-')} |",
        f"| Device | {cfg.get('device', '-')} |",
        f"| Workers | {cfg.get('workers', '-')} |",
    ]
    return "\n".join(lines)


def cleanup_run_output(run_id: str) -> None:
    """（内部使用）删除任务输出目录，测试或重跑时使用。"""
    rd = run_dir(run_id)
    if rd.is_dir():
        shutil.rmtree(rd)