"""全局配置与通用小工具。

所有存储路径集中在此处配置，避免在业务代码中硬编码。
可通过环境变量覆盖，便于部署时把数据目录放到独立磁盘。
"""
from __future__ import annotations

import json
import os
import random
import string
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = Path(os.environ.get("YOLO_STUDIO_STORAGE", BASE_DIR / "storage")).resolve()
DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
RUNS_DIR = STORAGE_DIR / "runs"
INFER_DIR = STORAGE_DIR / "inference"
TMP_DIR = STORAGE_DIR / "tmp"

# ---------------------------------------------------------------- 上传限制
MAX_UPLOAD_MB = int(os.environ.get("YOLO_STUDIO_MAX_UPLOAD_MB", "2048"))          # 单个上传文件上限
MAX_ZIP_UNCOMPRESSED_MB = int(os.environ.get("YOLO_STUDIO_MAX_UNCOMPRESSED_MB", "20480"))  # 解压后总大小上限
MAX_ZIP_FILES = int(os.environ.get("YOLO_STUDIO_MAX_ZIP_FILES", "200000"))        # 解压后文件数上限
MAX_ZIP_RATIO = 500                                                               # 单文件压缩比上限（防解压炸弹）

# ---------------------------------------------------------------- 服务配置
SERVER_NAME = os.environ.get("YOLO_STUDIO_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("YOLO_STUDIO_PORT", "7860"))
SHARE = False  # 默认禁止公网分享
AUTH_USER = os.environ.get("YOLO_STUDIO_USER", "").strip()
AUTH_PASSWORD = os.environ.get("YOLO_STUDIO_PASSWORD", "").strip()

# ---------------------------------------------------------------- 业务常量
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

IMG_SIZE_CHOICES = [320, 480, 640, 960, 1280]
DEFAULT_IMG_SIZE = 640

# 训练参数取值范围（用于校验）
EPOCHS_RANGE = (1, 10000)
BATCH_RANGE = (1, 1024)
WORKERS_RANGE = (0, 32)

# 支持的任务类型（V1.1 起：实例分割 + 目标检测）
SUPPORTED_TASKS = ("segment", "detect")
TASK_LABELS = {
    "segment": "segment（实例分割）",
    "detect": "detect（目标检测）",
    "classify": "classify（图像分类）",
    "pose": "pose（姿态估计）",
    "obb": "obb（旋转框检测）",
}


def task_label(task: str | None) -> str:
    """任务类型的中文可读名称。"""
    return TASK_LABELS.get(str(task), f"{task}（未知任务）")


RUN_STATUS_RUNNING = "RUNNING"
RUN_STATUS_COMPLETED = "COMPLETED"
RUN_STATUS_FAILED = "FAILED"
RUN_STATUS_STOPPED = "STOPPED"
RUN_STATUS_INTERRUPTED = "INTERRUPTED"  # 服务重启后无法确认状态的历史任务
RUN_STATUSES = [
    RUN_STATUS_RUNNING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_STOPPED,
    RUN_STATUS_INTERRUPTED,
]


def ensure_dirs() -> None:
    """创建所有需要的存储目录。"""
    for d in (STORAGE_DIR, DATASETS_DIR, MODELS_DIR, RUNS_DIR, INFER_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 通用工具
def now_iso() -> str:
    """本地时区的 ISO 时间字符串（秒精度）。"""
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def new_id(prefix: str) -> str:
    """生成唯一 ID，例如 ds_20261008_223001_7f3a。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}_{stamp}_{rand}"


def atomic_write_json(path: Path, obj: dict) -> None:
    """原子写入 JSON：先写临时文件再替换，避免读到写了一半的文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path, default=None):
    """读取 JSON，文件不存在或损坏时返回 default。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default