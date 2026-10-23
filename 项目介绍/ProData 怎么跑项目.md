# ProData 运行指南

## 环境要求

### 硬件

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | 1× A100 80G | 8× A100/H100（多机 16 节点） |
| 显存 | 40G（7B 推理） | 80G×8（14B RL 训练） |
| RAM | 64G | 512G（多机数据加载） |
| Disk | 200G | 1T+（模型 + 数据集 + checkpoint） |

### 软件

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 | RL 用 3.10，LongDS 用 3.12（3.11+ 有 Ray 兼容问题） |
| CUDA | 12.6 | 训练推荐 12.6 |
| verl | v0.4.0 | RL 主干，需改造 |
| LLaMA-Factory | 0.9.4.dev0 | SFT |
| ms-swift | 3.11.0.dev0 | DataPRM SFT |
| vLLM | 0.13.0 | DataPRM 部署 |
| SGLang | — | RL rollout 引擎 |

---

## 一、环境配置

### 1.1 创建环境

```bash
conda create -n prodata python=3.10
conda activate prodata
```

### 1.2 安装 SFT 框架

```bash
cd train/sft/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 1.3 安装 RL 框架

```bash
cd train/rl/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install -e .[vllm]
pip install -e .[sglang]
apt install sqlite3
```

### 1.4 安装评测依赖

```bash
cd eval/ProData
pip install -r requirements.txt
apt install sqlite3
```

### 1.5 验证安装

```bash
python -c "import verl; print(verl.__version__)"
python -c "import sglang; print('sglang ok')"
```

---

## 二、编译检查（无需 GPU）

```bash
cd train/rl/verl
python -m compileall agent/ recipe/dapo/  # 语法检查
```

---

## 三、模型准备

基座模型：`Qwen/Qwen2.5-Coder-7B-Instruct` 或 `14B-Instruct`，从 HuggingFace 下载（可配镜像）。

---

## 四、数据准备

ProData-12K 在 HuggingFace（`prodata-12k`），下载后放到：

```text
train/SFT/LLaMA-Factory/data/prodata/prodata_12k.json
```

RL 数据（`ProData-Data`）的 `rl` 目录：
- `train.parquet`：RL 训练数据
- `test.parquet`：RL 评测数据
- `gold_csv_results/`：SQL 数据的 golden CSV 答案
- `train_files/`：csv/excel/sqlite 文件
- `db_schema.json`：SQL 的数据库 schema

---

## 五、启动训练

### SFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_full/prodata_12k_full_sft.yaml
```

### RL

改 `multi.sh` 里的路径和密钥（WANDB/SWANLAB/OPENAI），然后：

```bash
bash multi.sh
```

`multi.sh` 会启动 Ray，跑 `recipe.dapo.main_dapo`。注意 `async_interpreter.py` 里的 conda 路径要改成自己的，代码执行依赖独立的 Python 3.10 环境（`run_code_requirements.txt`）。

---

## 六、核心参数说明

### 数据和模型

| 参数 | 值 | 说明 |
|------|-----|------|
| max_prompt_length | 2048 | prompt 最大长度 |
| max_response_length | 8192 | response 最大长度 |
| max_start_length | 2048 | 起始长度 |
| max_obs_length | 2048 | observation 截断长度 |

### RL 算法

| 参数 | 值 | 说明 |
|------|-----|------|
| adv_estimator | grpo | advantage 估计方法 |
| rollout_n | 4 | 每条 prompt 采样数 |
| n_sample | 1 | GRPO 组大小 |
| sft_loss_type | mu | SFT+RL 混合损失 |
| mu_peak/valley/decay_steps | 0.9/0.05/350 | mu 衰减曲线 |

### Agent 控制

| 参数 | 值 | 说明 |
|------|-----|------|
| max_turns | 10 | 最大交互轮数 |
| temperature | 0.7 | 采样温度 |
| top_p | 1 | 核采样 |

### 显存和性能

| 参数 | 值 | 说明 |
|------|-----|------|
| gpu_memory_utilization | 0.4 | rollout 显存占比（给训练留空间） |
| sp_size | 4 | ulysses 序列并行度 |
| enable_chunked_prefill | True | 分块 prefill |

### 训练控制

| 参数 | 值 | 说明 |
|------|-----|------|
| lr | 1e-6 | RL 学习率 |
| clip_ratio_low/high | 0.2/0.28 | PPO clip 范围 |
| grad_clip | 1.0 | 梯度裁剪 |

---

## 七、多步任务特殊注意事项

- 代码执行依赖独立 conda 环境 `darl-run`，需提前建好并装 `run_code_requirements.txt` 的包
- `async_interpreter.py` 里 conda 路径 `/root/anaconda3/bin/conda` 要改成自己的
- SQL 评测需要 sqlite3 已安装，BIRD 评测要先准备 `dev_sqlite_files`
- rollout 的 workspace 目录要可写，代码执行会在里面跑

---

## 八、Trace 诊断

RL 训练用 SwanLab/WandB 看 reward、loss、entropy 曲线。评测侧用 `compute_pass3.py` 算 pass@3，`agreement_verifier.py` 的 summary.json 看 self-consistency。

无 GPU 冒烟测试：把 `max_turns` 调到 1、`rollout_n` 调到 1、数据切小，跑一轮验证 pipeline 通不通。

---

## 九、常见问题排查

| 症状 | 诊断 | 修复 |
|------|------|------|
| rollout 时 conda 报错 | `async_interpreter.py` 的 conda 路径不对 | 改成自己的 conda 绝对路径 |
| 代码执行 `No module named X` | 缺包 | 执行器会自动 pip 装，或预装进 darl-run 环境 |
| 训练 reward 全 0 | 代码执行崩或 reward 函数 bug | 查 rollout 日志，看执行器 stderr |
| advantage 全 0 | 组内 reward 全零 | DAPO filter_groups 已丢，检查执行器是否正常 |
| 显存 OOM | gpu_memory_utilization 太高 | 调到 0.3-0.4，或开 offload |
| SQL 报 "Only SELECT" | 模型生成非 SELECT | 正常，这是保护机制，模型看到错误会重写 |
| 评测超时 | 单题 max_turns 内没答完 | 调大 max_turns 或查执行器超时 |
