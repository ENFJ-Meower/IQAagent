# IQA Agent：开源多模态模型零训练图像质量评估

本项目默认使用公开权重的 `Qwen/Qwen2.5-VL-7B-Instruct`，在不使用
KonIQ-10k 验证集或 SPAQ 测试集训练、微调及在线校准的前提下，通过
Skill Prompt、固定信号处理证据和受限 Router/Decision 输出图像质量分数。

## 当前版本的重要变化

- 在线预测统一输出 `0-100`，不再把 SPAQ 的 `0-100` MOS 与 `1-5`
  预测直接计算 MAE。
- 删除验证集偏移、Isotonic/Logistic 校准、KonIQ MOS 分段映射及
  Train-Augmented 模式。
- 默认 Backbone 为公开权重 `Qwen/Qwen2.5-VL-7B-Instruct`。
- 每张送入 VLM 的图片默认限制为 `512×512` 等面积素预算，适配
  `max-model-len=4096`；原生细节由独立的五区域 contact sheet 保留。
- Skill 链固定为两次 VLM 调用：技术检查 + 最终分维度评分。
- 首张图片执行 VLM 预检；模型输出无效时立即停止，不再把 Router fallback
  记为有效模型结果。
- 结果 CSV 记录 API 错误、finish reason、token 使用量和回复摘要。
- Router 只负责规则选择、证据冲突裁决和解释信息组织。
- 图像缓存按文件内容 SHA-256 查找，不使用文件名决定在线结果。
- 噪声图和梯度图使用固定尺度，不再对每张图片单独 Min-Max 拉伸。
- BRISQUE 不可用时返回 `None` 并重分配 Router 权重，不会再被当作满分。
- 错误样本标记为 `error`；模型失效但 Router 可计算时标记为 `fallback`，
  两者都不会进入正式 SRCC/MAE。

## 环境要求

- Python 3.10 或更高版本
- 推荐 NVIDIA GPU；显存不足时可使用 Qwen2.5-VL-7B 的量化部署
- Windows 本地运行推荐使用 Conda/venv；vLLM 服务推荐使用 WSL2 或 Linux

### 1. 创建环境

Windows PowerShell：

```powershell
cd IQAagent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果使用本地 Qwen2.5-VL：

```powershell
pip install -r requirements-local.txt
```

如需指定 CUDA 版本，建议先按照 PyTorch 官网命令安装匹配的
`torch/torchvision`，再安装 `requirements-local.txt`。

`piq`/BRISQUE 是可选证据。未安装或执行失败时系统仍可运行，但会在结果
`warnings` 中记录并使用其余证据。

## 数据目录

推荐目录：

```text
IQAagent/
├─ data/
│  ├─ koniq/
│  │  ├─ images/
│  │  └─ koniq10k_val.csv
│  └─ spaq/
│     ├─ images/
│     └─ spaq_test.csv
├─ cache/
└─ results/
```

默认 CSV 字段：

| 数据集 | 图片列 | MOS 列 | 原生范围 |
|---|---|---|---|
| KonIQ-10k Val | `img_id` | `img_mos` | 1-5 |
| SPAQ Test | `image_id` | `MOS` | 0-100 |

字段或路径不一致时使用命令行参数覆盖，不需要修改代码。

## 运行方式

### 方式 A：本地 Transformers

先用 3 张图片检查流程：

```powershell
python evaluation\run_eval.py --dataset spaq --vlm local --limit 3
```

正式全量评测时删除 `--limit`：

```powershell
python evaluation\run_eval.py --dataset koniq --vlm local
python evaluation\run_eval.py --dataset spaq --vlm local
```

### 方式 B：OpenAI 兼容的本地模型服务

Linux/WSL2 中启动公开权重模型：

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":4}' \
  --mm-processor-kwargs '{"max_pixels":262144}'
```

评测：

```powershell
$env:VLM_BASE_URL="http://127.0.0.1:8000/v1"
$env:VLM_API_KEY="EMPTY"
python evaluation\run_eval.py `
  --dataset spaq `
  --vlm server `
  --workers 2 `
  --limit 3 `
  --max-image-pixels 262144
```

也可以显式传入：

```powershell
python evaluation\run_eval.py `
  --dataset spaq `
  --vlm server `
  --base-url http://127.0.0.1:8000/v1 `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --limit 3
```

先确认三行结果均满足 `status=ok`、`vlm_valid=true`，且
`vlm_direct_score`/`vlm_dimension_score` 非空，再删除 `--limit 3`
执行全量。默认预检会在第一张退化为 fallback 时中止；只有专门调试时才使用
`--no-preflight`。

项目不再内置任何商业 API 地址。若使用第三方托管服务，应确认其模型与
公开权重版本完全一致，并记录模型名称、版本和服务参数。

## 预计算缓存

预计算只读取图像像素，不读取 MOS：

```powershell
python precompute.py --dataset koniq
python precompute.py --dataset spaq
```

评测时启用：

```powershell
python evaluation\run_eval.py `
  --dataset koniq `
  --vlm local `
  --cache-dir cache\koniq
```

缓存采用图片内容哈希作为键，`tool_scores.json` 不保存图片文件名或 MOS。

## 自定义路径和字段

```powershell
python evaluation\run_eval.py `
  --dataset spaq `
  --images-dir D:\datasets\SPAQ\images `
  --labels-csv D:\datasets\SPAQ\spaqTest.csv `
  --img-col image_id `
  --mos-col MOS `
  --label-min 0 `
  --label-max 100 `
  --vlm local
```

## 输出指标

模型始终生成 `predicted_score_0_100`。`status=ok` 才进入指标计算；
`fallback` 只用于故障诊断。标签缩放只发生在离线
`evaluation/metrics.py`：

- `SRCC`：预测排名与原生 MOS 排名的 Spearman 相关系数；
- `MAE_100`：两套数据统一到 0-100 后的 MAE；
- `MAE_native`：映射回各数据集原生尺度后的 MAE。

数据集名、文件名、MOS、Split 等信息不会进入 Pipeline、Skill 或 Router。

## 断点续跑

```powershell
python evaluation\run_eval.py `
  --dataset koniq `
  --vlm server `
  --resume results\eval_koniq_已有结果.csv `
  --output results\eval_koniq_full.csv
```

只有已有 CSV 中 `status=ok` 的 `image_id` 用于离线调度；`fallback` 和
`error` 会在恢复运行时重试。该字段不会传给在线评分流程。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
```

## 合规边界

允许：

- 用 MOS 做离线指标计算和离线误差分析；
- 使用不依赖目标评测数据训练的固定信号处理工具；
- Router 根据当前图片证据选择规则并裁决冲突。

禁止：

- 用 KonIQ Val 或 SPAQ Test 训练、微调、拟合偏移或校准函数；
- 把 MOS 或 MOS 衍生的映射参数送入在线 Pipeline；
- 把数据集名、图片 ID、文件名、MOS、失真标签、Split 送入 Skill、
  Router 或 VLM；
- 根据目标集结果反复调在线评分参数后再报告为零样本成绩。

## 结果说明

旧版本报告中的 50 张图片结果仅为历史探索，不是本版本的正式成绩。
正式结果必须使用当前代码重新运行 KonIQ-10k Val 全量和 SPAQ Test 全量，
并同时保存命令、模型版本、结果 CSV、失败样本数和运行环境。
