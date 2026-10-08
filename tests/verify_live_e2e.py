"""对**运行中的** YOLO Studio Lite 执行一次完整的验收流程（通过 Gradio HTTP 接口，等价于浏览器操作）。

用法：
    # 终端 1
    python app.py
    # 终端 2
    python tests/verify_live_e2e.py http://127.0.0.1:7860

流程（对应 Spec §10 验收标准）：
    1. 连接服务，确认页面可访问
    2. 上传 YOLO-Seg 数据集 ZIP → 校验类别数 / 训练・验证图片数
    3. 上传 segment 模型 .pt
    4. 启动训练（epochs=1, batch=2, imgsz=320, CPU）
    5. 轮询状态 / 日志 / 指标，直到训练结束
    6. 下载 best.pt 与 results.csv
    7. 用训练产出的模型对一张图片做在线推理，检查结果图与统计

脚本自行生成数据集 ZIP 和权重文件（完全离线），不会下载任何外部资源。
注意：该脚本会在目标实例的 storage 中留下真实数据（数据集/模型/训练记录）。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/符号报 UnicodeEncodeError
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import build_dataset_zip, build_tiny_seg_pt  # noqa: E402


def _field(result, key):
    """从 Gradio 返回的 dict/list 结构中取值。"""
    if isinstance(result, dict):
        return result.get(key)
    return None


def _file_path(result) -> Path | None:
    """gr.File 的返回值可能是文件路径字符串，也可能是 {'path': ...} / {'value': ...}。"""
    if isinstance(result, dict):
        for key in ("path", "value"):
            if result.get(key):
                return Path(result[key])
        return None
    return Path(result) if result else None


TERMINAL_STATUSES = ("COMPLETED", "FAILED", "STOPPED", "INTERRUPTED")


def _status_from_history(history, run_id: str) -> str:
    """从 ui_poll 返回的历史表格里读取该任务的状态（比解析状态文本更可靠）。"""
    if not isinstance(history, dict):
        return ""
    data = history.get("data") or []
    headers = history.get("headers") or []
    try:
        id_idx = headers.index("任务 ID")
        status_idx = headers.index("状态")
    except ValueError:
        id_idx, status_idx = 0, 5
    for row in data:
        if len(row) > max(id_idx, status_idx) and str(row[id_idx]) == run_id:
            return str(row[status_idx])
    return ""


def main(url: str) -> int:
    from gradio_client import Client, handle_file

    work = Path(tempfile.mkdtemp(prefix="yolo_verify_"))
    print(f"[1/7] 连接服务 {url} ...")
    client = Client(url, verbose=False)
    # 预检：确认连到的是 YOLO Studio Lite，而不是本机其它 Gradio 应用
    info = client.view_api(return_format="dict")
    names = set(info.get("named_endpoints", {}).keys())
    required = {"/ui_import_dataset", "/ui_upload_model", "/ui_start_training", "/ui_poll",
                "/ui_select_history", "/ui_history_download_best", "/ui_history_download_csv",
                "/ui_run_inference"}
    missing = required - names
    if missing:
        raise RuntimeError(
            f"{url} 不是 YOLO Studio Lite 实例（缺少接口 {sorted(missing)}）。"
            "请确认 app.py 已在该端口启动。"
        )
    print(f"      ✓ 已连接（{len(names)} 个接口，服务身份校验通过）")

    print("[2/7] 生成并上传数据集 ZIP ...")
    zip_path = build_dataset_zip(work / "verify_dataset.zip", n_train=4, n_val=2, classes=("shape",))
    msg, table, dropdown = client.predict(handle_file(zip_path), "验收数据集", api_name="/ui_import_dataset")
    assert "导入成功" in msg, f"数据集导入失败: {msg}"
    dataset_id = _field(dropdown, "value")
    print(f"      ✓ {msg.splitlines()[0]}\n      dataset_id={dataset_id}")

    print("[3/7] 生成并上传 segment 权重 ...")
    pt_path = build_tiny_seg_pt(work / "verify_seg.pt", nc=1)
    msg, table, train_dd, infer_dd = client.predict(handle_file(pt_path), "验收模型", api_name="/ui_upload_model")
    assert "上传成功" in msg, f"模型上传失败: {msg}"
    model_id = _field(train_dd, "value")
    print(f"      ✓ {msg.splitlines()[0]}\n      model_id={model_id}")

    print("[4/7] 启动训练（epochs=1, batch=2, imgsz=320, device=cpu）...")
    out = client.predict(dataset_id, model_id, 1, 2, 320, "cpu", 0, "验收训练",
                         api_name="/ui_start_training")
    start_msg, status, log, _plot, _result, _hist, hist_dd = out
    assert "训练任务已启动" in start_msg, f"训练启动失败: {start_msg}"
    run_id = _field(hist_dd, "value")
    print(f"      ✓ {start_msg}\n      run_id={run_id}")

    print("[5/7] 轮询训练状态（最多等待 20 分钟）...")
    deadline = time.time() + 1200
    final_status = ""
    while time.time() < deadline:
        status, log, _metrics, _result, _plots, history = client.predict(api_name="/ui_poll")
        final_status = _status_from_history(history, run_id) or next(
            (s for s in TERMINAL_STATUSES if s in status), ""
        )
        print(f"      · 状态={final_status or 'RUNNING'} ｜ 日志行数={len(log.splitlines())}")
        if final_status:
            break
        time.sleep(5)
    print(f"      最终状态: {final_status}")
    assert final_status == "COMPLETED", f"训练未成功完成: {final_status}\n日志尾部:\n{log[-2000:]}"
    assert "已运行时间" in status

    print("[6/7] 读取历史任务并下载产物 ...")
    config_md, hist_log, metrics_df, result_md, best_file, csv_file = client.predict(
        run_id, api_name="/ui_select_history"
    )
    assert "最优 Mask mAP50" in result_md, f"结果摘要异常: {result_md[:500]}"
    assert "Mask mAP50-95" in result_md
    print(f"      ✓ 结果摘要: {result_md.splitlines()[2]}")
    n_metric_rows = len(metrics_df.get("data", [])) if isinstance(metrics_df, dict) else len(metrics_df or [])
    print(f"      ✓ 指标数据点数: {n_metric_rows}")

    dl_best, dl_msg = client.predict(run_id, api_name="/ui_history_download_best")
    dl_csv, _ = client.predict(run_id, api_name="/ui_history_download_csv")
    best_path, csv_path = _file_path(dl_best), _file_path(dl_csv)
    assert best_path and best_path.is_file() and best_path.stat().st_size > 1000, f"best.pt 下载失败: {dl_best}"
    assert csv_path and csv_path.is_file(), f"results.csv 下载失败: {dl_csv}"
    print(f"      ✓ 下载 best.pt: {best_path} ({best_path.stat().st_size // 1024} KB)")
    print(f"      ✓ 下载 results.csv: {csv_path}")

    print("[7/7] 用训练模型做在线推理 ...")
    from PIL import Image

    img_path = work / "verify_input.png"
    Image.new("RGB", (160, 120), (20, 140, 90)).save(img_path)
    original, plotted, stats, result_file = client.predict(
        model_id, handle_file(img_path), 0.25, 320, "cpu", api_name="/ui_run_inference"
    )
    assert "检测到的实例总数" in stats, f"推理统计异常: {stats}"
    res = _file_path(result_file)
    print(f"      ✓ {stats.splitlines()[2]}")
    print(f"      ✓ 推理结果图: {res} 存在={res.is_file() if res else False}")

    print("\n=========== 验收流程全部通过 ===========")
    print(f"数据集: {dataset_id}\n模型: {model_id}\n训练: {run_id} ({final_status})")
    print(f"临时文件目录: {work}")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
    code = 0
    try:
        code = main(target)
    except AssertionError as e:
        print(f"\n✗ 验收失败: {e}")
        code = 1
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\n✗ 验收异常: {type(e).__name__}: {e}")
        code = 2
    sys.stdout.flush()
    # gradio_client 会留下非守护线程，正常 sys.exit 可能卡在解释器退出阶段，这里强制退出
    os._exit(code)