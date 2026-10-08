"""YOLO Studio Lite —— Gradio Web UI 入口。

运行：
    python app.py
默认访问 http://<服务器IP>:7860

页面包含四个 Tab：数据集管理 / 模型管理 / 模型训练 / 在线推理。
长耗时训练在独立子进程中执行，界面通过 gr.Timer 轮询状态、日志和指标。
"""
from __future__ import annotations

import os
from pathlib import Path

# Ultralytics 的配置文件放在 storage 下，便于随数据目录一起持久化（须在导入 ultralytics 前设置）
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "storage" / ".ultralytics"))

import gradio as gr  # noqa: E402

from core import config, datasets, inference, models, training  # noqa: E402

config.ensure_dirs()


# ================================================================ 通用工具
def _dataset_choices() -> list[tuple[str, str]]:
    return [
        (f"{m['name']} | {m.get('task', 'segment')} | 类别 {m['nc']} | "
         f"训练/验证 {m['train_images']}/{m['val_images']}", m["id"])
        for m in datasets.list_datasets()
    ]


def _model_choices() -> list[tuple[str, str]]:
    return [
        (f"{m['name']} | {m['task']} | {m.get('size_mb', '?')}MB | {m['source']}", m["id"])
        for m in models.list_models()
    ]


def _ds_table() -> gr.Dataframe:
    return gr.Dataframe(value=datasets.dataset_table_rows(datasets.list_datasets()))


def _ds_dropdown(value=None) -> gr.Dropdown:
    return gr.Dropdown(choices=_dataset_choices(), value=value)


def _model_table() -> gr.Dataframe:
    return gr.Dataframe(value=models.model_table_rows(models.list_models()))


def _model_dropdowns(value_train=None, value_infer=None):
    choices = _model_choices()
    return (
        gr.Dropdown(choices=choices, value=value_train),
        gr.Dropdown(choices=choices, value=value_infer),
    )


# ================================================================ Tab 1 数据集管理
def ui_import_dataset(zip_path, name):
    if not zip_path:
        return "请先选择数据集 ZIP 文件。", _ds_table(), _ds_dropdown()
    try:
        meta = datasets.import_dataset_zip(zip_path, name)
    except datasets.DatasetError as e:
        return f"❌ **导入失败**：{e}", _ds_table(), _ds_dropdown()
    except Exception as e:  # noqa: BLE001
        return f"❌ **导入异常**：{type(e).__name__}: {e}", _ds_table(), _ds_dropdown()
    msg = (
        f"✅ 数据集「{meta['name']}」导入成功：类别数 {meta['nc']}，"
        f"训练集 {meta['train_images']} 张，验证集 {meta['val_images']} 张。\n\n"
        f"数据集 ID：`{meta['id']}`"
    )
    if meta.get("warnings"):
        msg += "\n\n提示：" + "；".join(meta["warnings"])
    return msg, _ds_table(), _ds_dropdown(meta["id"])


def ui_refresh_datasets():
    return _ds_table(), _ds_dropdown()


def ui_dataset_info(dataset_id):
    meta = datasets.get_dataset(dataset_id)
    return datasets.dataset_info_markdown(meta)


def ui_preview(dataset_id, overlay, count, split):
    if not dataset_id:
        return [], "请先选择数据集。"
    try:
        images = datasets.preview_images(dataset_id, int(count), bool(overlay), split or "all")
    except datasets.DatasetError as e:
        return [], f"❌ 预览失败：{e}"
    if not images:
        return [], "该数据集中没有可预览的图片。"
    return images, f"已生成 {len(images)} 张预览图（随机抽样）。"


def ui_delete_dataset(dataset_id, confirmed):
    if not dataset_id:
        return "请先选择要删除的数据集。", _ds_table(), _ds_dropdown()
    if not confirmed:
        return "⚠️ 请先勾选「我确认要删除该数据集」。", _ds_table(), _ds_dropdown()
    try:
        msg = datasets.delete_dataset(dataset_id)
    except datasets.DatasetError as e:
        return f"❌ **删除失败**：{e}", _ds_table(), _ds_dropdown(dataset_id)
    return f"✅ {msg}", _ds_table(), _ds_dropdown()


# ================================================================ Tab 2 模型管理
def ui_upload_model(file_path, name):
    if not file_path:
        return "请先选择 .pt 模型文件。", _model_table(), *_model_dropdowns()
    try:
        meta = models.import_model_file(file_path, name)
    except models.ModelError as e:
        return f"❌ **上传失败**：{e}", _model_table(), *_model_dropdowns()
    except Exception as e:  # noqa: BLE001
        return f"❌ **上传异常**：{type(e).__name__}: {e}", _model_table(), *_model_dropdowns()
    msg = (
        f"✅ 模型「{meta['name']}」上传成功：任务类型 {meta['task']}，类别数 {meta['nc']}，"
        f"大小 {meta['size_mb']} MB。\n\n模型 ID：`{meta['id']}`"
    )
    return msg, _model_table(), *_model_dropdowns(value_train=meta["id"], value_infer=meta["id"])


def ui_refresh_models():
    return _model_table(), *_model_dropdowns()


def ui_model_info(model_id):
    meta = models.get_model(model_id)
    if not meta:
        return "未选择模型。"
    names = meta.get("names") or []
    source = {"upload": "用户上传", "training": "训练产出"}.get(meta.get("source", ""), meta.get("source", "-"))
    lines = [
        f"### 模型：{meta['name']}",
        "",
        f"- **模型 ID**：`{meta['id']}`",
        f"- **任务类型**：{config.task_label(meta.get('task'))}",
        f"- **类别数**：{meta.get('nc', '-')}",
        f"- **类别列表**：{', '.join(names) if names else '-'}",
        f"- **文件大小**：{meta.get('size_mb', '-')} MB",
        f"- **来源**：{source}" + (f"（训练任务 `{meta['run_id']}`）" if meta.get("run_id") else ""),
        f"- **上传/生成时间**：{meta['created_at']}",
        f"- **文件路径**：`{models.model_path(meta['id'])}`",
    ]
    if meta.get("source") == "training":
        lines.append("")
        lines.append("> 该模型由平台训练生成，可直接用于「模型训练」（继续微调）或「在线推理」。")
    return "\n".join(lines)


def ui_delete_model(model_id, confirmed):
    if not model_id:
        return "请先选择要删除的模型。", _model_table(), *_model_dropdowns()
    if not confirmed:
        return "⚠️ 请先勾选「我确认要删除该模型」。", _model_table(), *_model_dropdowns()
    try:
        msg = models.delete_model(model_id)
    except models.ModelError as e:
        return f"❌ **删除失败**：{e}", _model_table(), *_model_dropdowns()
    return f"✅ {msg}", _model_table(), *_model_dropdowns()


def ui_model_download(model_id):
    if not model_id:
        return None, "请先选择模型。"
    try:
        return str(models.model_path(model_id)), f"已生成下载链接：{model_id}"
    except models.ModelError as e:
        return None, f"❌ {e}"


# ================================================================ Tab 3 模型训练
def ui_device_update():
    devices = training.available_devices()
    default = training.default_device()
    if not training.has_gpu():
        note = "> ⚠️ **未检测到 CUDA GPU**，当前仅可用 CPU 训练：速度会明显变慢，请使用较小的 Epochs / Batch / Image Size 进行验证。"
    else:
        note = f"> 已检测到 GPU：{', '.join(d for d in devices if d != 'cpu')}"
    return gr.Dropdown(choices=devices, value=default), note


def ui_start_training(dataset_id, model_id, epochs, batch, imgsz, device, workers, run_name):
    try:
        task = training.start_training(
            dataset_id=dataset_id,
            model_id=model_id,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            name=run_name,
        )
    except training.TrainingError as e:
        return f"❌ **无法启动训练**：{e}", training.status_markdown(), "", None, "", gr.Dataframe(value=training.run_history_rows()), gr.Dropdown(choices=training.run_choices())
    except Exception as e:  # noqa: BLE001
        return (
            f"❌ **启动异常**：{type(e).__name__}: {e}",
            training.status_markdown(), "", None, "",
            gr.Dataframe(value=training.run_history_rows()), gr.Dropdown(choices=training.run_choices()),
        )
    msg = f"✅ 训练任务已启动：`{task['run_id']}`（{task['name']}），PID={task['pid']}。"
    if task["config"]["device"] == "cpu":
        msg += " 当前使用 CPU 训练，速度较慢，请耐心等待。"
    for note in task.get("notes") or []:
        msg += "\n\n> 提示：" + note
    return (
        msg, training.status_markdown(task), "", None, training.result_markdown(task["run_id"]),
        gr.Dataframe(value=training.run_history_rows()), gr.Dropdown(choices=training.run_choices(), value=task["run_id"]),
    )


def ui_stop_training():
    task = training.current_run()
    if not task:
        return "当前没有正在运行的任务。", training.status_markdown(), gr.Dataframe(value=training.run_history_rows())
    try:
        msg = training.stop_training(task["run_id"])
    except training.TrainingError as e:
        return f"❌ {e}", training.status_markdown(), gr.Dataframe(value=training.run_history_rows())
    return f"⏹️ {msg}", training.status_markdown(), gr.Dataframe(value=training.run_history_rows())


def ui_poll():
    """定时轮询：刷新状态、日志、指标、结果图与历史记录。"""
    training.poll()
    task = training.current_run()
    if task is None:
        # 没有正在运行的任务时，展示最近一次任务的状态，避免训练结束瞬间状态区“空掉”
        runs = training.list_runs()
        task = runs[0] if runs else None
    active_id = task["run_id"] if task else None

    status = training.status_markdown(task)
    if task and task.get("status") != config.RUN_STATUS_RUNNING:
        status = "> 当前没有正在进行的训练任务，以下为最近一次任务的状态。\n\n" + status
    log = training.read_log(active_id, 200) if active_id else ""
    metrics_df = training.metrics_long_dataframe(active_id) if active_id else None
    result = training.result_markdown(active_id) if active_id else "暂无训练结果。"
    plots = _run_plots(active_id) if active_id else []
    history = gr.Dataframe(value=training.run_history_rows())
    return status, log, metrics_df, result, plots, history


def _run_plots(run_id: str):
    task = training.get_run(run_id) or {}
    out = task.get("output_dir")
    if not out or not Path(out).is_dir():
        found = training._find_output_dir(run_id)  # noqa: SLF001 - 内部辅助，UI 直接复用
        if not found:
            return []
        out = str(found)
    paths: list[tuple[str, str]] = []
    for name in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        p = Path(out) / name
        if p.is_file():
            paths.append((str(p), name))
    for p in sorted(Path(out).glob("val_batch*_pred.jpg"))[:4]:
        paths.append((str(p), p.name))
    for p in sorted(Path(out).glob("train_batch*.jpg"))[:2]:
        paths.append((str(p), p.name))
    return paths


def ui_select_history(run_id):
    if not run_id:
        return "未选择任务。", "", None, "", None, None
    log = training.read_log(run_id, 200)
    metrics_df = training.metrics_long_dataframe(run_id)
    task = training.get_run(run_id) or {}
    return (
        training.run_config_markdown(run_id),
        log,
        metrics_df,
        training.result_markdown(run_id),
        str(task["best_pt"]) if task.get("best_pt") and Path(task["best_pt"]).is_file() else None,
        str(task["results_csv"]) if task.get("results_csv") and Path(task["results_csv"]).is_file() else None,
    )


def ui_history_download(run_id, what):
    task = training.get_run(run_id) if run_id else None
    if not task:
        return None, "请先选择历史任务。"
    key = {"best.pt": "best_pt", "results.csv": "results_csv", "last.pt": "last_pt"}[what]
    path = task.get(key)
    if not path or not Path(path).is_file():
        return None, f"该任务没有可下载的 {what}（任务状态：{task.get('status')}）。"
    return str(path), f"已生成下载链接：{what}"


def ui_history_download_best(run_id):
    """下载历史任务的 best.pt（独立函数以便暴露为具名 API）。"""
    return ui_history_download(run_id, "best.pt")


def ui_history_download_csv(run_id):
    """下载历史任务的 results.csv（独立函数以便暴露为具名 API）。"""
    return ui_history_download(run_id, "results.csv")


def ui_history_refresh():
    """刷新历史记录下拉与表格。"""
    return gr.Dataframe(value=training.run_history_rows()), gr.Dropdown(choices=training.run_choices())


def ui_initial_refresh():
    """页面加载时刷新所有动态下拉选项。"""
    return (
        gr.Dropdown(choices=_dataset_choices()),
        gr.Dropdown(choices=_model_choices()),
        gr.Dropdown(choices=_model_choices()),
        gr.Dropdown(choices=_model_choices()),
        gr.Dropdown(choices=training.run_choices()),
    )


def ui_initial_tables():
    """页面加载时刷新所有表格。"""
    return (
        gr.Dataframe(value=datasets.dataset_table_rows(datasets.list_datasets())),
        gr.Dataframe(value=models.model_table_rows(models.list_models())),
        gr.Dataframe(value=training.run_history_rows()),
    )


# ================================================================ Tab 4 在线推理
def ui_run_inference(model_id, image_path, conf, imgsz, device):
    if not model_id:
        return None, None, "⚠️ 请先选择模型。", None
    if not image_path:
        return None, None, "⚠️ 请先上传一张图片。", None
    try:
        result = inference.run_inference(model_id, image_path, conf=conf, imgsz=imgsz, device=device)
    except inference.InferenceError as e:
        return None, None, f"❌ **推理失败**：{e}", None
    except Exception as e:  # noqa: BLE001
        return None, None, f"❌ **推理异常**：{type(e).__name__}: {e}", None
    return result["original"], result["plotted"], inference.stats_markdown(result), str(result["result_path"])


# ================================================================ 页面
CSS = """
.gradio-container {max-width: 1500px !important;}
footer {display: none !important;}
"""

with gr.Blocks(title="YOLO Studio Lite", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        """
# YOLO Studio Lite
轻量级 YOLO **实例分割（segment）/ 目标检测（detect）** 管理平台：数据集导入 → 模型管理 → 训练与监控 → 在线推理。全部数据保存在服务器本地 `storage/` 目录。
        """
    )

    with gr.Tabs():
        # ---------------------------------------------------- Tab 1
        with gr.Tab("① 数据集管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 导入数据集（YOLO 格式 ZIP：分割 / 检测）")
                    ds_zip = gr.File(label="数据集 ZIP", file_types=[".zip"], type="filepath")
                    ds_name = gr.Textbox(label="数据集名称", placeholder="留空则使用 ZIP 文件名")
                    ds_import_btn = gr.Button("导入数据集", variant="primary")
                    ds_import_msg = gr.Markdown()
                    gr.Markdown(
                        """
**ZIP 结构要求**
```
dataset.zip
├── data.yaml
├── images/{train,val}/
└── labels/{train,val}/
```
- 允许 ZIP 内额外包一层数据集根目录
- 图片支持 jpg / jpeg / png
- 分割标签：`class_id x1 y1 x2 y2 ...`（多边形 ≥ 3 个点，坐标 0~1）
- 检测标签：`class_id cx cy w h`（中心点 + 宽高，均为 0~1）
- 任务类型按标签格式自动识别；同一数据集混用两种格式会被拒绝
- 背景图可无标签文件
                        """
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        ds_refresh_btn = gr.Button("🔄 刷新数据集列表")
                        ds_selector = gr.Dropdown(label="选择数据集", choices=[], interactive=True, allow_custom_value=True)
                    ds_table = gr.Dataframe(
                        headers=datasets.DATASET_TABLE_HEADERS, value=[], interactive=False, wrap=True
                    )
                    ds_info = gr.Markdown("未选择数据集。")
            with gr.Row():
                preview_split = gr.Radio(choices=["all", "train", "val"], value="all", label="预览来源", scale=0)
                preview_count = gr.Slider(1, 8, value=8, step=1, label="预览数量", scale=0)
                preview_overlay = gr.Checkbox(label="显示标签多边形叠加", value=False, scale=0)
                preview_btn = gr.Button("生成预览", scale=0)
            preview_msg = gr.Markdown()
            ds_gallery = gr.Gallery(label="图片预览（随机抽样，最多 8 张）", columns=4, height="auto")
            with gr.Row():
                ds_del_confirm = gr.Checkbox(label="我确认要删除该数据集（同时删除已解压文件）", value=False)
                ds_del_btn = gr.Button("删除数据集", variant="stop")
            ds_del_msg = gr.Markdown()

        # ---------------------------------------------------- Tab 2
        with gr.Tab("② 模型管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 上传模型权重（.pt）")
                    model_file = gr.File(label="YOLO-Seg 模型权重", file_types=[".pt"], type="filepath")
                    model_name = gr.Textbox(label="模型名称", placeholder="留空则使用文件名")
                    model_upload_btn = gr.Button("上传模型", variant="primary")
                    model_upload_msg = gr.Markdown()
                    gr.Markdown(
                        """
**说明**
- 接受 Ultralytics **segment（实例分割）** 与 **detect（目标检测）** 模型；classify / pose 模型会被拒绝
- 训练完成后产生的 `best.pt` 会自动出现在下方列表
- 平台不会联网下载预训练权重，请上传本地 `.pt` 文件
- ⚠️ `.pt` 反序列化存在安全风险，仅允许上传受信任的权重文件
                        """
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        model_refresh_btn = gr.Button("🔄 刷新模型列表")
                        model_selector = gr.Dropdown(label="选择模型", choices=[], interactive=True, allow_custom_value=True)
                    model_table = gr.Dataframe(
                        headers=models.MODEL_TABLE_HEADERS, value=[], interactive=False, wrap=True
                    )
                    model_info = gr.Markdown("未选择模型。")
                    with gr.Row():
                        model_dl_btn = gr.Button("生成下载链接")
                        model_dl_file = gr.File(label="模型下载", interactive=False)
                    model_dl_msg = gr.Markdown()
            with gr.Row():
                model_del_confirm = gr.Checkbox(label="我确认要删除该模型", value=False)
                model_del_btn = gr.Button("删除模型", variant="stop")
            model_del_msg = gr.Markdown()

        # ---------------------------------------------------- Tab 3
        with gr.Tab("③ 模型训练"):
            device_note = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 训练配置")
                    train_dataset = gr.Dropdown(label="数据集", choices=[], interactive=True, allow_custom_value=True)
                    train_model = gr.Dropdown(label="模型（segment / detect）", choices=[], interactive=True, allow_custom_value=True)
                    with gr.Row():
                        train_epochs = gr.Number(value=50, precision=0, label="Epochs")
                        train_batch = gr.Number(value=8, precision=0, label="Batch Size")
                    with gr.Row():
                        train_imgsz = gr.Dropdown(
                            choices=config.IMG_SIZE_CHOICES, value=config.DEFAULT_IMG_SIZE, label="Image Size"
                        )
                        train_workers = gr.Number(value=2, precision=0, label="Workers")
                    train_device = gr.Dropdown(label="Device", choices=["cpu"], value="cpu", interactive=True, allow_custom_value=True)
                    train_name = gr.Textbox(label="训练名称", placeholder="留空自动生成")
                    with gr.Row():
                        train_start_btn = gr.Button("🚀 开始训练", variant="primary")
                        train_stop_btn = gr.Button("⏹️ 停止训练", variant="stop")
                    train_action_msg = gr.Markdown()
                    gr.Markdown(
                        """
**提示**
- 单任务模式：同一时间只允许一个训练任务
- 训练在独立子进程中执行，可随时在页面查看日志与指标
- 停止训练只会终止本次任务所属的进程组
- 任务组合：segment 模型 ↔ 分割数据集；detect 模型 ↔ 检测数据集；
  detect 模型 + 分割数据集会把多边形自动转成外接框（框精度看 Box mAP）
                        """
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### 训练状态")
                    train_status = gr.Markdown("当前没有正在进行的训练任务。")
                    gr.Markdown("### 训练日志（每 3 秒刷新，显示最近 200 行）")
                    train_log = gr.Textbox(lines=18, max_lines=18, autoscroll=True, interactive=False, label="")
                    gr.Markdown("### 训练指标（来自 results.csv，缺失指标不会以 0 填充）")
                    train_metrics_plot = gr.LinePlot(
                        value=None, x="epoch", y="数值", color="指标",
                        title="损失与 Mask mAP", x_title="Epoch", y_title="值",
                        height=380, tooltip=["epoch", "指标", "数值"],
                    )
                    gr.Markdown("### 训练结果")
                    train_result = gr.Markdown("暂无训练结果。")
                    train_plots = gr.Gallery(label="训练结果图", columns=4)

            gr.Markdown("---")
            gr.Markdown("### 历史训练记录（选择任务后可查看配置 / 日志 / 指标并下载产物）")
            with gr.Row():
                history_refresh_btn = gr.Button("🔄 刷新历史记录")
                history_selector = gr.Dropdown(label="选择历史任务", choices=[], interactive=True, allow_custom_value=True)
            history_table = gr.Dataframe(headers=training.RUN_HISTORY_HEADERS, value=[], interactive=False, wrap=True)
            with gr.Row():
                with gr.Column(scale=1):
                    history_config = gr.Markdown("未选择任务。")
                    history_result = gr.Markdown("")
                    with gr.Row():
                        history_dl_best = gr.Button("下载 best.pt")
                        history_dl_csv_btn = gr.Button("下载 results.csv")
                    history_dl_best_file = gr.File(label="best.pt", interactive=False)
                    history_dl_csv_file = gr.File(label="results.csv", interactive=False)
                    history_dl_msg = gr.Markdown()
                with gr.Column(scale=1):
                    history_log = gr.Textbox(lines=10, max_lines=10, autoscroll=True, interactive=False, label="日志")
                with gr.Column(scale=1):
                    history_metrics_plot = gr.LinePlot(
                        value=None, x="epoch", y="数值", color="指标",
                        title="历史任务指标", x_title="Epoch", y_title="值", height=300,
                        tooltip=["epoch", "指标", "数值"],
                    )
                    history_plots = gr.Gallery(label="结果图", columns=3)

        # ---------------------------------------------------- Tab 4
        with gr.Tab("④ 在线推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 推理配置")
                    infer_model = gr.Dropdown(label="模型（segment / detect）", choices=[], interactive=True, allow_custom_value=True)
                    infer_image = gr.Image(label="上传图片（jpg / jpeg / png）", type="filepath", height=260)
                    infer_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Conf Threshold")
                    infer_imgsz = gr.Dropdown(
                        choices=config.IMG_SIZE_CHOICES, value=config.DEFAULT_IMG_SIZE, label="Image Size"
                    )
                    infer_device = gr.Dropdown(label="推理设备", choices=["cpu"], value="cpu", interactive=True, allow_custom_value=True)
                    infer_btn = gr.Button("🔍 开始推理", variant="primary")
                    infer_dl_file = gr.File(label="下载推理结果图", interactive=False)
                    gr.Markdown(
                        """
**说明**
- 单张图片推理（分割 / 检测均可），使用 `results[0].plot()` 生成可视化结果
- GPU 训练进行中时禁用 GPU 推理，避免显存冲突
- 未检测到目标属于正常结果，不是执行失败
                        """
                    )
                with gr.Column(scale=2):
                    infer_stats = gr.Markdown("等待推理…")
                    with gr.Row():
                        infer_original = gr.Image(label="原始图片", height=380)
                        infer_result_img = gr.Image(label="分割结果", height=380)

    # ---------------------------------------------------- 事件绑定
    ds_refresh_btn.click(ui_refresh_datasets, outputs=[ds_table, ds_selector])
    ds_import_btn.click(
        ui_import_dataset, inputs=[ds_zip, ds_name], outputs=[ds_import_msg, ds_table, ds_selector]
    )
    ds_selector.change(ui_dataset_info, inputs=ds_selector, outputs=ds_info)
    preview_btn.click(
        ui_preview, inputs=[ds_selector, preview_overlay, preview_count, preview_split],
        outputs=[ds_gallery, preview_msg],
    )
    ds_del_btn.click(
        ui_delete_dataset, inputs=[ds_selector, ds_del_confirm],
        outputs=[ds_del_msg, ds_table, ds_selector],
    )

    model_refresh_btn.click(ui_refresh_models, outputs=[model_table, train_model, infer_model])
    model_upload_btn.click(
        ui_upload_model, inputs=[model_file, model_name],
        outputs=[model_upload_msg, model_table, train_model, infer_model],
    )
    model_selector.change(ui_model_info, inputs=model_selector, outputs=model_info)
    model_dl_btn.click(ui_model_download, inputs=model_selector, outputs=[model_dl_file, model_dl_msg])
    model_del_btn.click(
        ui_delete_model, inputs=[model_selector, model_del_confirm],
        outputs=[model_del_msg, model_table, train_model, infer_model],
    )

    demo.load(ui_device_update, outputs=[train_device, device_note])
    demo.load(ui_initial_refresh,
              outputs=[train_dataset, train_model, infer_model, model_selector, history_selector])
    demo.load(ui_initial_tables, outputs=[ds_table, model_table, history_table])

    train_start_btn.click(
        ui_start_training,
        inputs=[train_dataset, train_model, train_epochs, train_batch, train_imgsz, train_device,
                train_workers, train_name],
        outputs=[train_action_msg, train_status, train_log, train_metrics_plot, train_result, history_table,
                 history_selector],
    )
    train_stop_btn.click(ui_stop_training, outputs=[train_action_msg, train_status, history_table])
    history_refresh_btn.click(ui_history_refresh, outputs=[history_table, history_selector])
    history_selector.change(
        ui_select_history, inputs=history_selector,
        outputs=[history_config, history_log, history_metrics_plot, history_result, history_dl_best_file,
                 history_dl_csv_file],
    )
    history_dl_best.click(
        ui_history_download_best, inputs=history_selector,
        outputs=[history_dl_best_file, history_dl_msg],
    )
    history_dl_csv_btn.click(
        ui_history_download_csv, inputs=history_selector,
        outputs=[history_dl_csv_file, history_dl_msg],
    )

    infer_btn.click(
        ui_run_inference, inputs=[infer_model, infer_image, infer_conf, infer_imgsz, infer_device],
        outputs=[infer_original, infer_result_img, infer_stats, infer_dl_file],
    )

    timer = gr.Timer(3.0)
    timer.tick(
        ui_poll, outputs=[train_status, train_log, train_metrics_plot, train_result, train_plots, history_table]
    )


def main() -> None:
    for msg in training.init_manager():
        print(f"[startup] {msg}")
    if not training.has_gpu():
        print("[startup] 未检测到 CUDA GPU：训练/推理将使用 CPU（速度较慢）。")
    auth = (config.AUTH_USER, config.AUTH_PASSWORD) if config.AUTH_USER and config.AUTH_PASSWORD else None
    if auth is None:
        print("[startup] 未设置访问认证（可通过 YOLO_STUDIO_USER / YOLO_STUDIO_PASSWORD 环境变量开启）。")
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        share=config.SHARE,
        auth=auth,
        show_error=True,
        allowed_paths=[str(config.STORAGE_DIR)],
    )


if __name__ == "__main__":
    main()