# YOLO Studio Lite

轻量级 YOLO **实例分割（segment）** Web 管理平台，用于公司内网部署。所有操作在浏览器完成，
数据保存在服务器本地文件系统，不依赖数据库、Redis、消息队列、云平台或任何公网服务。

- 技术栈：Python 3.11+ / Gradio Blocks / Ultralytics / PyTorch
- 四个页面：**① 数据集管理 ② 模型管理 ③ 模型训练 ④ 在线推理**
- 训练在独立子进程中执行，界面可实时查看日志、状态与指标

> V1 范围：仅支持 Ultralytics YOLO **实例分割（segment）**。不支持图片标注、检测/分类/姿态任务、
> 多 GPU、超参搜索、用户权限体系、ONNX/TensorRT 导出。

---

## 1. 目录结构

```
yolo_studio/
├── app.py                 # Gradio 页面与事件绑定（唯一入口）
├── train_worker.py        # 训练子进程入口（独立进程调用 Ultralytics）
├── core/
│   ├── config.py          # 路径与全局配置（全部可通过环境变量覆盖）
│   ├── datasets.py        # ZIP 安全解压、结构校验、标签逐行校验、预览、删除
│   ├── models.py          # .pt 校验（任务类型）、模型索引、删除
│   ├── training.py        # 训练配置校验、子进程启动/监控/停止、指标读取、历史记录
│   └── inference.py       # 单张图片推理、可视化、按实例统计
├── storage/               # 运行时数据（数据集 / 模型 / 训练记录），可挂载宿主机目录
│   ├── datasets/{dataset_id}/  (data.yaml, images/, labels/, meta.json)
│   ├── models/{model_id}/      (model.pt, meta.json)
│   └── runs/{run_id}/          (task.json, train.log, result.json, output/)
├── tests/                 # pytest 自动化测试（含真实 CPU 端到端训练测试）
│   └── verify_live_e2e.py # 对运行中的服务执行完整验收流程（HTTP，等价浏览器操作）
├── requirements.txt
└── pytest.ini
```

---

## 2. 环境依赖

| 组件 | 版本（本机实测） | 说明 |
|---|---|---|
| Python | 3.11 ~ 3.12 | Spec 目标 3.11，开发环境实测 3.12.12 |
| PyTorch | 2.9.0 | CPU 版实测；GPU 部署需与服务器驱动匹配的 CUDA 版 |
| torchvision | 0.24.0 | 与 torch 2.9.0 配套 |
| Ultralytics | 8.4.80 | 训练/推理均使用官方 API |
| Gradio | 5.50.0（实测 5.23.0 也可用） | Web UI，5.23 ~ 5.50 均已验证 |
| 其它 | PyYAML / Pillow / numpy / pandas | 见 `requirements.txt` |

**PyTorch 与 CUDA 兼容性**：`torch` 的 CUDA 版本必须是服务器驱动支持的版本（不确定就用 CPU 版先跑通流程）。
在本机安装 CUDA 版示例：

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
```

内网无公网时，把 `--index-url` 换成内部 pip 镜像地址。

---

## 3. 方式一：Python 直接运行

```bash
# 1) 安装依赖（建议先单独装好与 CUDA 匹配的 torch/torchvision）
pip install -r requirements.txt

# 2)（GPU 机器）确认 PyTorch 能看到显卡
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# 3) 启动
python app.py
```

访问：`http://<服务器IP>:7860`（默认监听 `0.0.0.0:7860`，`share=False`，不对外网开放）

常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `YOLO_STUDIO_HOST` / `YOLO_STUDIO_PORT` | `0.0.0.0` / `7860` | 监听地址与端口 |
| `YOLO_STUDIO_STORAGE` | `./storage` | 数据根目录（数据集/模型/训练记录） |
| `YOLO_STUDIO_USER` / `YOLO_STUDIO_PASSWORD` | 空 | 两者都设置时启用简单访问认证 |
| `YOLO_STUDIO_MAX_UPLOAD_MB` | `2048` | 单个上传文件大小上限 |
| `YOLO_STUDIO_MAX_UNCOMPRESSED_MB` | `20480` | 数据集 ZIP 解压后总大小上限（防解压炸弹） |

> ⚠️ **无 GPU 时**：界面顶部会明确提示"未检测到 CUDA GPU"，Device 只剩 `cpu`，训练可用但很慢。
> 用 `epochs=1、batch=2、imgsz=320` 的小数据集可以快速验证流程。

---

## 4. 方式二：Linux 服务器后台常驻运行

纯 Python 项目，不需要 Docker。推荐用 systemd 让它开机自启、崩溃自动重启：

```ini
# /etc/systemd/system/yolo-studio.service
[Unit]
Description=YOLO Studio Lite
After=network.target

[Service]
Type=simple
User=yolo
WorkingDirectory=/opt/yolo_studio
Environment=YOLO_STUDIO_HOST=0.0.0.0
Environment=YOLO_STUDIO_PORT=7860
Environment=YOLO_STUDIO_STORAGE=/data/yolo_studio/storage
Environment=YOLO_STUDIO_USER=yolo
Environment=YOLO_STUDIO_PASSWORD=<改成强口令>
ExecStart=/opt/yolo_studio/.venv/bin/python /opt/yolo_studio/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-studio
sudo systemctl status yolo-studio
journalctl -u yolo-studio -f        # 查看服务日志（训练日志仍在 storage/runs/{run_id}/train.log）
```

不用 systemd 时也可以直接后台运行：

```bash
cd /opt/yolo_studio
nohup python app.py > app.log 2>&1 &
```

部署要点：
- 数据目录用 `YOLO_STUDIO_STORAGE` 指到独立的持久化磁盘（例如 `/data/yolo_studio/storage`），
  备份/迁移只需要搬这个目录
- 端口默认 `7860`，`share=False`，不会生成公网链接
- 服务启动时不会再联网下载任何模型权重（预训练权重请手动上传）
- 建议给服务单独建一个 Python 虚拟环境（`.venv`），避免污染系统环境

---

## 5. 使用流程

### 5.1 数据集 ZIP 格式（Tab ①）

```
dataset.zip
├── data.yaml
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
└── labels/
    ├── train/*.txt
    └── val/*.txt
```

`data.yaml` 示例：

```yaml
path: .
train: images/train
val: images/val

names:
  0: person
  1: bag
```

导入规则（全部在服务端强制校验，失败会给出具体原因且不保存任何数据）：

- 允许 ZIP 内额外包一层数据集根目录（自动识别唯一的 `data.yaml` 所在根目录）
- 必须存在 `data.yaml`；`train` / `val` 图片目录必须存在且非空
- 图片仅支持 `jpg / jpeg / png`；允许背景图没有对应标签文件
- `names` 必须有效，类别 ID 从 0 连续编号，`nc` 与 `names` 数量一致
- 标签必须是 YOLO-Seg 格式：`class_id x1 y1 x2 y2 ...`（多边形 ≥ 3 个点、坐标 0~1、
  数值有限、`class_id` 必须存在于 `names`）；**不接受检测框（4 坐标）标签**
- ZIP 解压防护：拒绝路径穿越（`..`）、绝对路径、符号链接、特殊文件、超限解压（解压炸弹）
- 导入成功后重新生成内部 `data.yaml`，`path` 指向存储目录，`train/val` 保持原结构

导入后可以：查看类别数与训练/验证图片数量 → 随机预览最多 8 张图片（可叠加分割多边形）
→ 勾选确认后删除数据集（正在被训练任务使用的数据集禁止删除）。

### 5.2 模型上传（Tab ②）

- 只接受 Ultralytics **segment** 模型的 `.pt`（detect/classify/pose 会被拒绝并提示实际任务类型）
- 上传时会实际加载权重文件校验：损坏文件、非 PyTorch 归档、空文件都会给出明确错误
- 同名模型不会互相覆盖（内部使用唯一 ID）
- 训练完成后产生的 `best.pt` **自动加入模型列表**，可直接用于继续微调或推理
- 平台不会联网下载预训练权重，请把本地 `.pt` 上传上来
- ⚠️ **安全提示**：`.pt` 反序列化存在代码执行风险，本平台只允许受信任人员上传权重；
  后缀与格式校验只是健全性检查，不构成安全保证

### 5.3 训练（Tab ③）

1. 选择数据集与 segment 模型
2. 设置 `Epochs / Batch Size / Image Size / Device / Workers / 训练名称`
   （Image Size 可选 320 / 480 / 640 / 960 / 1280，Device 自动列出可用 CUDA 设备与 CPU）
3. 点击「🚀 开始训练」→ 服务端生成唯一 `run_id`，**以独立子进程**启动训练
4. 页面每 3 秒刷新：状态（任务 ID、状态、开始/结束时间、已运行时间、已完成 Epoch/总 Epoch、GPU 显存）、
   日志尾部（最多 200 行，自动滚动）、指标曲线（Train/Val Box Loss、Train/Val Seg Loss、Mask mAP50、Mask mAP50-95）
5. 训练结束后展示：最优 Mask mAP50 / Mask mAP50-95、best.pt、last.pt、results.csv、训练结果图，
   并自动把 best.pt 注册为模型

规则：
- **单任务模式**：同一时间只允许一个训练任务，重复点击不会产生多个训练进程
- 「⏹️ 停止训练」只终止本次任务所属进程组（POSIX 用 `killpg`，Windows 用 `taskkill /T /F`），必要时强制终止
- 状态：`RUNNING / COMPLETED / FAILED / STOPPED / INTERRUPTED`
- 服务重启后历史记录保留；无法确认状态的旧任务标记为 `INTERRUPTED`，**不自动恢复**，也不占用训练锁
- 指标全部来自 `results.csv` 的真实输出；不同 Ultralytics 版本的列名差异做了兼容映射，
  **缺失的指标显示为缺失，不会用 0 填充**

历史记录（Tab ③ 下方）：任务 ID / 名称 / 数据集 / 基础模型 / Epochs / 状态 / 开始时间 / Mask mAP50-95，
选择后可查看配置、日志、指标与结果摘要，并下载 `best.pt`、`results.csv`
（即使数据集后来被删除，历史任务仍保留原始元信息）。

### 5.4 在线推理（Tab ④）

1. 选择一个模型（上传的或训练产出的）
2. 上传一张 jpg/jpeg/png 图片，设置 Conf Threshold 与 Image Size
3. 点击「🔍 开始推理」→ 展示原图、分割结果图、检测到的**实例总数**与各类别实例数，并可下载结果图

- 统计按**实例数量**计算（不是像素或多边形点数）
- 未检测到目标属于正常结果，页面提示"未检测到目标"，不视为失败
- **GPU 训练进行中时禁用 GPU 推理**，避免显存冲突（可改用 CPU 推理）
- V1 不做批量推理、视频推理、实时摄像头

---

## 6. 自动化测试

```bash
python -m pytest                    # 全量（含真实 CPU 端到端训练，本机约 1~2 分钟）
python -m pytest -m "not slow"      # 跳过端到端训练测试
```

覆盖范围：

| 测试文件 | 内容 |
|---|---|
| `tests/test_datasets.py` | 正常导入（含嵌套根目录、多类别、背景图）、缺 data.yaml、空 val、nc/names 异常、标签 8 类格式错误、路径穿越、绝对路径、符号链接、解压炸弹、非 ZIP、删除与占用保护、预览 |
| `tests/test_models.py` | 真实 segment 权重上传、detect 模型拒绝、假 .pt/空文件/非 torch 归档拒绝、同名不覆盖、best.pt 幂等注册、删除与占用保护 |
| `tests/test_training.py` | 参数范围校验、设备校验、启动前置校验、单任务互斥、停止（进程组终止）、服务重启标记 INTERRUPTED、指标列名兼容与**缺失不填 0**、截断 CSV 容错、历史记录、**真实 1-epoch CPU 端到端训练**（产物 + 指标 + 自动注册模型） |
| `tests/test_inference.py` | 输入校验（模型/图片/conf/imgsz）、GPU 训练时禁止 GPU 推理、真实推理结果结构、未检测到目标路径 |
| `tests/test_app_ui.py` | Gradio 页面可构建、四个 Tab 与全部回调存在、数据集/模型/推理/历史记录 UI 回调真实数据流、错误提示文案 |

测试用的数据集与权重**全部离线生成**（`tests/helpers.py` 用 ultralytics 自带 yaml 构造真实
segment/detect 权重，不下载任何外部资源），可反复运行。

### 对运行中的服务做完整验收

```bash
# 终端 1
python app.py
# 终端 2（等价于浏览器点击全流程：上传→训练→下载→推理）
python tests/verify_live_e2e.py http://127.0.0.1:7860
```

---

## 7. 常见问题

**Q：页面提示"未检测到 CUDA GPU"？**
无 GPU 或 PyTorch 装成了 CPU 版。执行 `python -c "import torch;print(torch.cuda.is_available())"` 确认；
为 False 时重新安装与服务器驱动匹配的 CUDA 版 PyTorch。

**Q：训练报 `CUDA out of memory`？**
页面会明确提示。减小 `Batch Size` / `Image Size`，或关闭其他占用显存的进程后重试。

**Q：训练很慢？**
CPU 训练慢属正常。GPU 训练时可适当提高 `Workers`；`Batch Size` 越大显存占用越高。

**Q：`Value is not in the list of choices`？**
已修复（所有动态下拉框均设置了 `allow_custom_value`）。若仍出现，请刷新页面重新选择。

**Q：训练时报 `MLflow: results logged to runs/mlflow`？**
本机安装了 `mlflow` 包时，Ultralytics 会自动启用 MLflow 记录。平台的训练子进程已把工作目录
设为该任务的 run 目录，不会污染项目根目录；如需彻底关闭可执行 `yolo settings mlflow=False`。

**Q：能上传 detect 模型吗？**
不能。V1 只支持 segment，上传 detect 模型会提示实际任务类型并被拒绝。

**Q：训练产物在哪？**
`storage/runs/{run_id}/output/`（`weights/best.pt`、`weights/last.pt`、`results.csv`、结果图），
也可以在页面直接下载 `best.pt` / `results.csv`。

**Q：数据会丢吗？**
不会。所有数据都在 `YOLO_STUDIO_STORAGE`（默认 `./storage`）下的数据集/模型/训练目录里，
服务重启、换端口、重新部署都不影响；备份时直接拷贝该目录即可。

**Q：可以多个人同时用吗？**
可以多人访问，但训练是**单任务模式**（同时只允许一个训练任务），避免显存与资源冲突。
如需访问控制，设置 `YOLO_STUDIO_USER` / `YOLO_STUDIO_PASSWORD` 启用简单认证（无 RBAC/SSO）。

---

## 8. 安全说明（内网部署）

- 仅部署在可信公司内网；默认 `share=False`，不生成公网分享链接
- 支持环境变量配置简单访问认证
- 上传文件大小、ZIP 结构、解压体积均有上限校验；不执行 ZIP 中的任何脚本
- 所有子进程调用都不使用 `shell=True`，不允许用户提交任意 Python 代码或命令
- 数据集/模型使用唯一 ID，UI 无法访问 `storage/` 之外的任意服务器文件；
  JSON 状态文件采用原子写入
- 正在被训练任务使用的数据集/模型禁止删除
- ⚠️ **PyTorch 权重反序列化存在固有安全风险**：仅允许受信任人员上传 `.pt` 权重，
  不要把任意来源的权重文件上传到平台

---

## 9. 实际验证结果摘要

验证环境：**Windows 11 + Python 3.12.12 + torch 2.9.0+cpu（无 GPU）+ ultralytics 8.4.80 + gradio 5.50.0**
部署环境：**内网 tp001（172.28.40.170）+ Python 3.10.4 + torch 2.6.0（无 GPU）+ ultralytics 8.4.80 + gradio 5.23.0**

### 9.0 内网测试机部署记录

| 项 | 值 |
|---|---|
| 主机 | `172.28.40.170`（tp001），无 GPU，80 核 / 251G 内存 |
| 代码目录 | `/data/wangke/vibe_coding/yolo_studio`（与 git commit 内容一致，部署的 commit 记录在同目录 `DEPLOY_COMMIT`） |
| Python 环境 | `/data/anaconda3/envs/llm`（共享环境，仅新增 `ultralytics / ultralytics-thop / nvidia-ml-py`，未改动 torch/numpy/gradio） |
| 数据目录 | `/data/wangke/vibe_coding/yolo_studio/storage` |
| 访问地址 | **http://172.28.40.170:7870**（7860 已被机器的 `es_gradio.py` 占用，故使用 7870） |
| 服务管理 | systemd：`yolo-studio.service`（`enable` 开机自启，`Restart=always`） |

```bash
systemctl status yolo-studio        # 查看状态
systemctl restart yolo-studio       # 重启
journalctl -u yolo-studio -f        # 查看服务日志（训练日志在 storage/runs/{run_id}/train.log）
```

启用访问认证（可选）：编辑 `/etc/systemd/system/yolo-studio.service`，加入
`Environment=YOLO_STUDIO_USER=xxx` 与 `Environment=YOLO_STUDIO_PASSWORD=xxx`，然后
`systemctl daemon-reload && systemctl restart yolo-studio`。

更新部署：本地改完提交后，`git archive --format=tar HEAD | ssh root@172.28.40.170 "tar xzf - -C /data/wangke/vibe_coding/yolo_studio"`
（该机器在内网，不能直接 `git pull` GitHub）。

### 9.1 自动化测试

```
python -m pytest   →   86 passed（含真实训练端到端测试，约 1~2 分钟）
```

其中端到端测试 `test_training_end_to_end_cpu` 真实执行了 1 个 epoch 的 CPU 训练并断言：
子进程启动 → 日志写入 → `results.csv` 生成 → 指标解析出真实 Loss/mAP → `best.pt`/`last.pt`/`results.png`
存在 → `best.pt` 自动注册为模型。**未使用任何模拟数据。**

### 9.2 运行中服务的完整验收（HTTP，等价浏览器操作）

`python tests/verify_live_e2e.py http://127.0.0.1:<port>` → **7/7 全部通过**：

| 步骤 | 结果 |
|---|---|
| 1 连接服务 + 服务身份校验 | ✓ 21 个接口 |
| 2 上传 YOLO-Seg 数据集 ZIP | ✓ 识别类别数 1，训练 4 张 / 验证 2 张 |
| 3 上传 segment `.pt` 权重 | ✓ 任务类型 segment，6.43 MB |
| 4 启动训练（epochs=1, batch=2, imgsz=320, CPU） | ✓ 独立子进程 PID=36676 |
| 5 轮询状态/日志/指标 | ✓ 日志 0→81 行，RUNNING → COMPLETED |
| 6 结果摘要 + 下载产物 | ✓ 最优 Mask mAP50 0.0125(epoch 1)，best.pt 6575 KB、results.csv 下载成功 |
| 7 在线推理 | ✓ 分割结果图生成成功 |

另有一次**用户真实数据**的训练（本机浏览器操作）：数据集 `coco8-seg`（nc=80，训练 4 / 验证 4）+
模型 `yolo11n-seg`，50 epochs / batch 8 / imgsz 640 / CPU，3 分 22 秒完成，
最优 **Mask mAP50 = 0.847 / Mask mAP50-95 = 0.577**，`best.pt` 自动注册为可用模型。

### 9.2.1 内网测试机（tp001）上的同样验收

在同一份代码部署到 `172.28.40.170:7870`、使用该机 Python 3.10 + gradio 5.23.0 + torch 2.6.0 后，
用本地机器跨网络执行 `python tests/verify_live_e2e.py http://172.28.40.170:7870` → **7/7 全部通过**：
数据集导入（类别数 1 / 训练 4 / 验证 2）→ 权重上传（6.43 MB）→ 训练启动并完成（日志 3→80 行）→
最优 Mask mAP50 0.0000（未训练权重 1 epoch，符合预期）→ 下载 best.pt 6574 KB 与 results.csv →
在线推理结果图生成成功。

### 9.3 异常的输入处理

无效 ZIP、路径穿越、符号链接、解压炸弹、缺 `data.yaml`、非法分割坐标（含检测框标签）、
非 segment 模型、无效 `.pt`、重复启动训练、停止训练、训练进程异常退出等场景
均已由自动化测试覆盖并通过。

### 9.4 尚未验证的事项（如实标注）

- **GPU 训练 / GPU 推理：未执行**。本机 `torch.cuda.is_available() == False`（CPU-only 构建），
  无 NVIDIA 设备，因此无法验证 CUDA 路径下的显存监控、GPU 训练速度与 OOM 提示。
  相关逻辑（设备列表、默认选第一块 GPU、GPU 训练时禁用 GPU 推理、OOM 错误提示）已实现并有 CPU 侧测试，
  但**必须在 GPU 服务器上按 Spec §10 复测**。
- **Docker：不在本方案内**。本平台按纯 Python 项目交付，不需要 Docker 或容器运行时。
- 多 GPU 分布式训练不在 V1 范围内。
- 其余验证均在上述环境中真实执行，无模拟数据。