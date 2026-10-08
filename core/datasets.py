"""数据集管理：ZIP 安全解压、结构校验、data.yaml 解析、标签逐行校验、预览、删除。

对外主要接口：
    import_dataset_zip(zip_path, name) -> meta dict
    list_datasets() / get_dataset(dataset_id) / delete_dataset(dataset_id)
    dataset_yaml_path(dataset_id)
    preview_images(dataset_id, count, overlay, split) -> [(PIL.Image, caption)]
"""
from __future__ import annotations

import glob as globlib
import io
import os
import random
import re
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from . import config

MAX_YAML_BYTES = 1 * 1024 * 1024  # data.yaml 大小上限
MAX_ERROR_SAMPLES = 20            # 错误提示最多展示多少条


class DatasetError(Exception):
    """数据集校验/导入失败，message 直接用于界面提示。"""


# ================================================================ ZIP 安全解压
def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(root.resolve())
    except (ValueError, OSError):
        return False


def _check_member(info: zipfile.ZipInfo) -> str | None:
    """检查单个 ZIP 条目，返回错误原因；None 表示安全。"""
    name = info.filename
    if "\x00" in name:
        return f"文件名包含非法字符: {name!r}"
    if "\\" in name:
        return f"ZIP 条目使用了反斜杠路径，已拒绝: {name}"
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return f"ZIP 条目为绝对路径，已拒绝: {name}"
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return f"ZIP 条目包含路径穿越 ('..')，已拒绝: {name}"
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = unix_mode & 0o170000
    if file_type == 0o120000:
        return f"ZIP 中包含符号链接，已拒绝: {name}"
    if file_type not in (0, 0o100000, 0o040000):
        return f"ZIP 中包含特殊文件类型，已拒绝: {name}"
    return None


def _safe_extract(zip_path: Path, dest: Path) -> int:
    """安全解压 ZIP，返回解压文件数。防路径穿越、符号链接、解压炸弹。"""
    max_total = config.MAX_ZIP_UNCOMPRESSED_MB * 1024 * 1024
    total = 0
    try:
        zf_ctx = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as e:
        raise DatasetError(f"文件不是有效的 ZIP 压缩包，或已损坏: {e}")
    with zf_ctx as zf:
        infos = zf.infolist()
        if len(infos) > config.MAX_ZIP_FILES:
            raise DatasetError(f"ZIP 内文件数 {len(infos)} 超过上限 {config.MAX_ZIP_FILES}")

        for info in infos:
            err = _check_member(info)
            if err:
                raise DatasetError(err)
            if info.is_dir():
                continue
            size = info.file_size
            total += size
            if total > max_total:
                raise DatasetError(f"ZIP 解压后总大小超过上限 {config.MAX_ZIP_UNCOMPRESSED_MB} MB，疑似解压炸弹")
            if size > 10 * 1024 * 1024 and size / max(info.compress_size, 1) > config.MAX_ZIP_RATIO:
                raise DatasetError(f"文件压缩比异常（{info.filename}），疑似解压炸弹")

        dest.mkdir(parents=True, exist_ok=True)
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("__MACOSX/") or Path(name).name in (".DS_Store", "Thumbs.db"):
                continue
            target = dest / name
            if not _is_within(target, dest):
                raise DatasetError(f"ZIP 条目目标路径越界，已拒绝: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    if out.tell() > info.file_size + 1024:
                        raise DatasetError(f"文件实际大小超过声明大小: {name}")
    return total


# ================================================================ 结构定位与 yaml 解析
def _find_dataset_root(extract_dir: Path) -> Path:
    """定位数据集根目录，允许 ZIP 内多包一层目录。"""
    if (extract_dir / "data.yaml").is_file():
        return extract_dir
    yaml_files = [p for p in extract_dir.glob("*/data.yaml")]
    if len(yaml_files) == 1:
        return yaml_files[0].parent
    if not yaml_files:
        raise DatasetError("未找到 data.yaml，目录结构不符合 YOLO 数据集格式")
    dirs = ", ".join(sorted(p.parent.name for p in yaml_files))
    raise DatasetError(f"发现多个 data.yaml，无法确定数据集根目录: {dirs}")


def _parse_names(raw, nc) -> dict[int, str]:
    """解析 names 字段（支持 dict / list），校验 ID 从 0 连续编号。"""
    if isinstance(raw, dict):
        names: dict[int, str] = {}
        for k, v in raw.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                raise DatasetError(f"names 中的类别 ID 不是整数: {k!r}")
            names[idx] = v
        keys = sorted(names)
    elif isinstance(raw, (list, tuple)):
        names = {i: v for i, v in enumerate(raw)}
        keys = list(range(len(names)))
    else:
        raise DatasetError("data.yaml 缺少有效的 names 字段")

    if not names:
        raise DatasetError("names 为空，至少需要一个类别")
    if len(names) > 10000:
        raise DatasetError("类别数量异常（>10000）")
    if keys != list(range(len(keys))):
        raise DatasetError(f"类别 ID 必须从 0 开始连续编号，当前为: {keys[:20]}")
    for idx, name in names.items():
        if not isinstance(name, str) or not name.strip():
            raise DatasetError(f"类别 {idx} 的名称无效: {name!r}")
        names[idx] = name.strip()
    if nc is not None:
        try:
            nc_int = int(nc)
        except (TypeError, ValueError):
            raise DatasetError(f"data.yaml 中的 nc 不是整数: {nc!r}")
        if nc_int != len(names):
            raise DatasetError(f"nc={nc_int} 与 names 数量 {len(names)} 不一致")
    return names


def _load_data_yaml(root: Path) -> dict:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise DatasetError("未找到 data.yaml")
    if yaml_path.stat().st_size > MAX_YAML_BYTES:
        raise DatasetError("data.yaml 文件过大，已拒绝解析")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DatasetError(f"data.yaml 解析失败: {e}")
    if not isinstance(data, dict):
        raise DatasetError("data.yaml 内容不是有效的 YAML 映射")
    return data


def _split_entries(data: dict, key: str) -> list[str]:
    """读取 train/val 字段，支持字符串或列表，拆分空格/逗号分隔的多路径。"""
    raw = data.get(key)
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    entries: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise DatasetError(f"data.yaml 中 {key} 字段类型无效: {item!r}")
        entries.extend(p for p in re.split(r"[,\s]+", item.strip()) if p)
    return entries


def _resolve_split_images(root: Path, entries: list[str], split: str) -> list[Path]:
    """把 train/val 条目解析为图片文件列表。支持目录、glob、txt 列表文件。"""
    files: list[Path] = []
    for entry in entries:
        rel = entry.replace("\\", "/").lstrip("./")
        target = (root / rel).resolve()
        if not _is_within(target, root):
            raise DatasetError(f"{split} 路径越出数据集目录: {entry}")
        if any(ch in rel for ch in "*?["):
            files += [Path(p) for p in globlib.glob(str(root / rel), recursive=True) if Path(p).is_file()]
        elif target.is_dir():
            files += [p for p in sorted(target.rglob("*")) if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS]
        elif target.is_file() and target.suffix.lower() == ".txt":
            try:
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as e:
                raise DatasetError(f"无法读取 {split} 列表文件 {entry}: {e}")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (target.parent / line).resolve() if line.startswith("./") else (root / line).resolve()
                if not _is_within(p, root):
                    raise DatasetError(f"{split} 列表中的路径越出数据集目录: {line}")
                files.append(p)
        elif target.is_file() and target.suffix.lower() in config.IMAGE_EXTS:
            files.append(target)
        else:
            raise DatasetError(f"{split} 路径不存在或不是有效目录/文件: {entry}")

    # 去重 + 排序，过滤非图片后缀
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in files:
        if p.suffix.lower() not in config.IMAGE_EXTS:
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return sorted(uniq)


def _img_to_label_path(img: Path, root: Path) -> Path:
    """按 YOLO 约定由图片路径推导标签路径：images/ -> labels/，后缀改 .txt。"""
    rel_parts = list(img.relative_to(root).parts)
    stem_txt = Path(rel_parts[-1]).with_suffix(".txt").name
    for i in range(len(rel_parts) - 1, -1, -1):
        if rel_parts[i] == "images":
            rel_parts[i] = "labels"
            rel_parts[-1] = stem_txt
            return root.joinpath(*rel_parts)
    return root / "labels" / stem_txt


# ================================================================ 标签逐行校验
def _validate_label_file(label_path: Path, names: dict[int, str]) -> tuple[int, list[str]]:
    """校验单个标签文件，返回 (目标实例数, 错误信息列表)。空文件允许（背景图）。"""
    errors: list[str] = []
    try:
        text = label_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return 0, [f"无法读取标签文件: {e}"]
    count = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            cls = int(tokens[0])
        except (ValueError, IndexError):
            errors.append(f"{label_path.name}:{lineno} 类别 ID 不是整数: {tokens[0]!r}")
            continue
        if cls not in names:
            errors.append(f"{label_path.name}:{lineno} 类别 ID {cls} 不在 names 中")
            continue
        coords = tokens[1:]
        if len(coords) < 6:
            kind = "检测框标签" if len(coords) == 4 else "坐标点数量"
            errors.append(
                f"{label_path.name}:{lineno} 分割多边形至少需要 3 个点（6 个坐标），"
                f"当前 {len(coords)} 个坐标（疑似{kind}格式，不被支持）"
            )
            continue
        if len(coords) % 2 != 0:
            errors.append(f"{label_path.name}:{lineno} 坐标数量 {len(coords)} 不是偶数，无法组成点对")
            continue
        try:
            vals = [float(v) for v in coords]
        except ValueError:
            errors.append(f"{label_path.name}:{lineno} 坐标包含非数值")
            continue
        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
            errors.append(f"{label_path.name}:{lineno} 坐标包含 NaN/Inf")
            continue
        if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
            errors.append(f"{label_path.name}:{lineno} 坐标超出 0~1 归一化范围")
            continue
        count += 1
    return count, errors


def _validate_split(
    root: Path, images: list[Path], names: dict[int, str], split: str
) -> tuple[dict, list[str], list[str]]:
    """校验一个 split 的所有标签，返回 (统计, 错误, 警告)。"""
    stats = {"images": len(images), "label_files": 0, "instances": 0, "background_images": 0}
    errors: list[str] = []
    warnings: list[str] = []
    error_total = 0
    for img in images:
        label_path = _img_to_label_path(img, root)
        if not label_path.is_file():
            stats["background_images"] += 1  # 允许背景图没有标签
            continue
        stats["label_files"] += 1
        n_inst, errs = _validate_label_file(label_path, names)
        stats["instances"] += n_inst
        if errs:
            error_total += len(errs)
            if len(errors) < MAX_ERROR_SAMPLES:
                errors.extend(errs[: MAX_ERROR_SAMPLES - len(errors)])
    if error_total > len(errors):
        errors.append(f"... 共发现 {error_total} 处标签格式错误，仅展示前 {len(errors) - 1} 条")
    if stats["images"] > 0 and stats["label_files"] == 0:
        errors.append(
            f"{split} 的 {stats['images']} 张图片均未找到对应标签（应位于 labels/ 目录），"
            "请检查目录结构是否符合 YOLO-Seg 格式"
        )
    if stats["instances"] == 0 and stats["label_files"] > 0:
        warnings.append(f"{split} 中所有标签文件都为空（全部为背景图）")
    return stats, errors, warnings


# ================================================================ 导入
def _build_split_yaml_value(root: Path, entries: list[str], files: list[Path], split: str) -> str:
    """生成写入内部 data.yaml 的 train/val 值。

    常见情况（单个 root 内的目录）保持原样，便于人工查看；
    其余情况（glob、列表文件、绝对路径等）改写为 splits/{split}.txt 列表文件。
    """
    if len(entries) == 1:
        raw = entries[0].replace("\\", "/")
        if not any(ch in raw for ch in "*?[") and not raw.lower().endswith(".txt"):
            p = (root / raw).resolve()
            if _is_within(p, root) and p.is_dir():
                rel = os.path.relpath(p, root).replace("\\", "/")
                return "." if rel == "." else rel
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    list_path = splits_dir / f"{split}.txt"
    rel_files = ["./" + os.path.relpath(f, root).replace("\\", "/") for f in files]
    list_path.write_text("\n".join(rel_files) + "\n", encoding="utf-8")
    return f"splits/{split}.txt"


def _write_internal_yaml(root: Path, names: dict[int, str], split_values: dict[str, str]) -> Path:
    data = {"path": str(root), **split_values, "nc": len(names), "names": {int(k): v for k, v in names.items()}}
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return yaml_path


def import_dataset_zip(zip_path: str | Path, name: str | None = None) -> dict:
    """导入 YOLO-Seg 数据集 ZIP：校验通过后保存到 storage/datasets/{dataset_id}。

    任何校验失败都会抛出 DatasetError，且不保留不完整数据。
    """
    config.ensure_dirs()
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise DatasetError(f"ZIP 文件不存在: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise DatasetError("只支持 .zip 格式的数据集压缩包")
    if zip_path.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DatasetError(f"数据集 ZIP 超过上限 {config.MAX_UPLOAD_MB} MB")

    dataset_id = config.new_id("ds")
    tmp_dir = config.TMP_DIR / dataset_id
    final_dir = config.DATASETS_DIR / dataset_id
    try:
        _safe_extract(zip_path, tmp_dir)
        root = _find_dataset_root(tmp_dir)
        data = _load_data_yaml(root)
        names = _parse_names(data.get("names"), data.get("nc"))

        split_values: dict[str, str] = {}
        split_stats: dict[str, dict] = {}
        all_errors: list[str] = []
        all_warnings: list[str] = []
        for split in ("train", "val"):
            entries = _split_entries(data, split)
            if not entries:
                raise DatasetError(f"data.yaml 缺少 {split} 字段")
            images = _resolve_split_images(root, entries, split)
            if not images:
                raise DatasetError(f"{split} 图片目录不存在或为空（仅支持 jpg/jpeg/png）")
            stats, errors, warnings = _validate_split(root, images, names, split)
            split_stats[split] = stats
            all_errors += errors
            all_warnings += warnings
            split_values[split] = _build_split_yaml_value(root, entries, images, split)

        if all_errors:
            detail = "\n".join(f"- {e}" for e in all_errors)
            raise DatasetError(f"标签校验失败，共 {len(all_errors)} 处问题：\n{detail}")

        # 校验通过：落盘
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(root), str(final_dir))
        yaml_path = _write_internal_yaml(final_dir, names, split_values)

        meta = {
            "id": dataset_id,
            "name": (name or zip_path.stem).strip() or zip_path.stem,
            "created_at": config.now_iso(),
            "nc": len(names),
            "names": {str(k): v for k, v in names.items()},
            "train_images": split_stats["train"]["images"],
            "val_images": split_stats["val"]["images"],
            "splits": split_stats,
            "source_zip": zip_path.name,
            "zip_size_mb": round(zip_path.stat().st_size / 1024 / 1024, 2),
            "yaml_path": str(yaml_path),
            "status": "有效",
            "warnings": all_warnings,
        }
        config.atomic_write_json(final_dir / "meta.json", meta)
        return meta
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ================================================================ 查询 / 删除
def list_datasets() -> list[dict]:
    """列出所有已导入数据集（按导入时间倒序）。"""
    config.ensure_dirs()
    metas = []
    for d in config.DATASETS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = config.read_json(d / "meta.json")
        if meta:
            metas.append(meta)
    return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)


def get_dataset(dataset_id: str) -> dict | None:
    if not dataset_id:
        return None
    meta = config.read_json(config.DATASETS_DIR / dataset_id / "meta.json")
    if meta and (config.DATASETS_DIR / dataset_id / "data.yaml").is_file():
        return meta
    return None


def dataset_yaml_path(dataset_id: str) -> Path:
    """返回可供 Ultralytics 使用的 data.yaml 绝对路径。"""
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or ".." in dataset_id:
        raise DatasetError("非法的数据集 ID")
    yaml_path = config.DATASETS_DIR / dataset_id / "data.yaml"
    if not yaml_path.is_file():
        raise DatasetError(f"数据集 {dataset_id} 的 data.yaml 不存在")
    return yaml_path


def delete_dataset(dataset_id: str) -> str:
    """删除数据集目录。正在被训练任务使用时拒绝删除。"""
    from . import training  # 延迟导入，避免循环依赖

    meta = get_dataset(dataset_id)
    if not meta:
        raise DatasetError(f"数据集不存在: {dataset_id}")
    if training.is_resource_in_use("dataset", dataset_id):
        raise DatasetError(f"数据集「{meta['name']}」正在被训练任务使用，无法删除")
    shutil.rmtree(config.DATASETS_DIR / dataset_id)
    return f"数据集「{meta['name']}」已删除"


# ================================================================ 预览
PALETTE = [
    (230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143), (233, 196, 106),
    (244, 162, 97), (231, 111, 81), (114, 9, 183), (76, 201, 240), (6, 214, 160),
]


def _draw_overlay(img: Image.Image, label_path: Path, names: dict[int, str]) -> Image.Image:
    """在图片上绘制分割多边形（Pillow 实现，非标注编辑器）。"""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    try:
        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return img
    for line in lines:
        tokens = line.split()
        if len(tokens) < 7:
            continue
        try:
            cls = int(tokens[0])
            vals = [float(v) for v in tokens[1:]]
        except ValueError:
            continue
        pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, len(vals) - 1, 2)]
        if len(pts) < 3:
            continue
        color = PALETTE[cls % len(PALETTE)]
        draw.polygon(pts, outline=color + (255,), fill=color + (60,))
        draw.line(pts + [pts[0]], fill=color + (255,), width=3)
        label = names.get(cls, str(cls))
        tx, ty = pts[0]
        tw = draw.textlength(label)
        draw.rectangle([tx, ty, tx + tw + 6, ty + 14], fill=color + (200,))
        draw.text((tx + 3, ty + 1), label, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def preview_images(dataset_id: str, count: int = 8, overlay: bool = False, split: str = "all"):
    """随机抽取最多 count 张图片用于预览，返回 [(PIL.Image, caption)]。"""
    meta = get_dataset(dataset_id)
    if not meta:
        raise DatasetError(f"数据集不存在: {dataset_id}")
    root = config.DATASETS_DIR / dataset_id
    names = {int(k): v for k, v in meta["names"].items()}

    candidates: list[tuple[Path, str]] = []
    for sp in ("train", "val"):
        if split not in ("all", sp):
            continue
        for img in sorted((root / "images" / sp).rglob("*")) if (root / "images" / sp).is_dir() else []:
            if img.suffix.lower() in config.IMAGE_EXTS:
                candidates.append((img, sp))
    if not candidates:  # 非标准结构：从 data.yaml 解析
        data = _load_data_yaml(root)
        for sp in ("train", "val"):
            if split not in ("all", sp):
                continue
            for img in _resolve_split_images(root, _split_entries(data, sp), sp):
                candidates.append((img, sp))

    random.shuffle(candidates)
    count = max(1, min(int(count), 8))
    results = []
    for img_path, sp in candidates[:count]:
        try:
            img = Image.open(img_path)
            img.load()
            img = img.convert("RGB")
        except Exception as e:  # noqa: BLE001 - 单张图片损坏不影响整体预览
            continue
        caption = f"{sp}/{img_path.name}"
        if overlay:
            img = _draw_overlay(img, _img_to_label_path(img_path, root), names)
            caption += " (标签叠加)"
        results.append((img, caption))
    return results


def dataset_table_rows(datasets: list[dict]) -> list[list]:
    """转换为 UI 表格行。"""
    rows = []
    for m in datasets:
        rows.append([
            m["name"],
            m["id"],
            m["nc"],
            ", ".join(list(m["names"].values())[:8]) + ("..." if m["nc"] > 8 else ""),
            m["train_images"],
            m["val_images"],
            m.get("status", "有效"),
            m["created_at"],
        ])
    return rows


DATASET_TABLE_HEADERS = ["名称", "数据集 ID", "类别数", "类别列表", "训练图片", "验证图片", "状态", "导入时间"]


def dataset_info_markdown(meta: dict) -> str:
    if not meta:
        return "未选择数据集。"
    splits = meta.get("splits", {})
    train, val = splits.get("train", {}), splits.get("val", {})
    lines = [
        f"### 数据集：{meta['name']}",
        "",
        f"- **数据集 ID**：`{meta['id']}`",
        f"- **类别数 (nc)**：{meta['nc']}",
        f"- **类别列表**：{', '.join(f'{k}:{v}' for k, v in meta['names'].items())}",
        f"- **训练集**：{meta['train_images']} 张图片 / {train.get('label_files', 0)} 个标签文件 / "
        f"{train.get('instances', 0)} 个实例",
        f"- **验证集**：{meta['val_images']} 张图片 / {val.get('label_files', 0)} 个标签文件 / "
        f"{val.get('instances', 0)} 个实例",
        f"- **状态**：{meta.get('status', '有效')}",
        f"- **导入时间**：{meta['created_at']}",
        f"- **来源**：{meta.get('source_zip', '-')} ({meta.get('zip_size_mb', 0)} MB)",
        f"- **data.yaml**：`{meta.get('yaml_path', '-')}`",
    ]
    if meta.get("warnings"):
        lines.append("")
        lines.append("**提示**：" + "；".join(meta["warnings"]))
    return "\n".join(lines)