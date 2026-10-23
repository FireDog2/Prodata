# ProData 项目复现与训练心得

## 前言

这份文档按时间顺序记录我复现 ProData 完整训练管线踩过的坑：环境搭建 → 数据准备 → SFT → RL → 评测。所有坑都来自实际代码和配置（`async_interpreter.py`、`multi.sh`、`eval_python.py`），不是从论文博客抄的。写下来是为了下次少踩，也方便面试时讲清楚每个坑的根因和解法。

---

## 一、硬件和训练时间

| 任务 | GPU | 模型 | 显存 | 步数 | 墙钟时间 |
|------|-----|------|------|------|----------|
| SFT | 4× A100 | Qwen2.5-Coder-7B | 80G | 3 epoch | ~几小时 |
| RL | 8× A100 | 7B/14B | 80G×8 | — | 天级 |

时间去向大头在 rollout 的代码执行：每条 trajectory 最多 10 轮，每轮跑代码是串行阻塞的，这部分占 rollout 总时间的比重远高于纯推理。训练更新（FSDP 反传）反而是相对固定的一部分。

---

## 二、环境搭建

### 2.1 Python 版本

RL 用 Python 3.10。3.11+ 有 Ray 兼容问题（`multi.sh` 里 Ray 是核心调度），不要用 3.12。LongDS 的 runner 反而要 3.12，两者是独立环境，别混。

### 2.2 框架版本锁定

verl 锁 v0.4.0。我在上面改了多轮 rollout 和损失目标，升级 verl 会把改动冲掉，得重新移植。LLaMA-Factory 锁 0.9.4.dev0，ms-swift 锁 3.11.0.dev0。

### 2.3 代码执行环境隔离

这里有个坑——RL 训练和代码执行要用**两个不同的 conda 环境**。训练环境跑 verl，代码执行环境（`darl-run`）跑模型生成的代码。为什么分开？因为模型生成的代码会 `pip install` 各种包，如果和训练环境共用，依赖会互相污染甚至把训练环境搞崩。

### 2.4 CUDA/PyTorch 对齐

CUDA 12.6。vLLM/SGLang 对 CUDA 版本敏感，装之前先确认 PyTorch 的 CUDA 版本和 vLLM/SGLang 预编译轮子对得上，不然推理引擎起不来。

### 2.5 模型下载

Qwen2.5-Coder-7B/14B-Instruct 从 HuggingFace 下，国内要配镜像。

---

## 三、数据准备

ProData-12K 是 parquet/json 混合，RL 数据有 train.parquet/test.parquet 加 gold_csv_results（SQL 的 golden 答案）和 db_schema.json（数据库 schema）。

这里有个坑——**代码执行器跑代码的目录**要和数据文件所在目录对齐。`multi.sh` 里 `csv_folder` 指向 train_files，`working_path` 是代码执行的工作目录，两者配错会导致"模型能生成代码但代码读不到数据文件"，reward 全是 0。

---

## 四、训练中踩的坑

### 4.1 conda 路径写死

**症状**：rollout 阶段代码执行一直报 `conda` 找不到。

**根因**：`async_interpreter.py` 里 conda 路径写死成 `/root/anaconda3/bin/conda`（第 41 行），换机器就失效。

**修复**：改成自己的 conda 绝对路径。README 里专门提了这条，说明这是高频坑。

**为什么这么设计**：代码执行要精确控制用哪个环境跑，写死绝对路径比依赖 PATH 环境变量更可靠，代价是换环境要改。

### 4.2 代码执行内存失控

**症状**：训练跑到一半，rollout worker 突然 OOM，整条 trajectory 废了，连带这一批的 reward 全 0。

**根因**：模型生成了吃内存的代码（大数组、无界循环）。

**修复**：`run_task` 里 `RLIMIT_AS` 卡 1GB。超了 OOM 报错当 observation 返回，模型看到错误换写法。

**为什么 1GB**：够常规 pandas 分析，但挡得住失控代码。

### 4.3 缺包导致浪费轮次

**症状**：模型 import 一个环境没装的库，代码报 `No module named X`，这轮白跑。

**修复**：`execute` 捕获缺包错误，自动 `pip install` 后重跑。sklearn 特判成 scikit-learn。

**为什么自动装**：数据分析任务依赖的库很杂，预装不全，自动装能把"缺包"从浪费一轮变成自动补齐。

### 4.4 SQL 误删库

**症状**：评测 SQL 任务时，模型生成 `DROP TABLE` 之类的破坏性语句。

**修复**：`is_not_select_query` 用 sqlparse 只放行 SELECT，其他返回 "Only SELECT statements are allowed"。

**为什么必须拦**：评测库是共享的真实数据，删了后面全废。这条是硬保护，不能省。

### 4.5 reward 全零导致 advantage 全零

**症状**：训练 reward 曲线长时间不涨，排查发现大量组内 reward 全是 0。

**根因**：代码执行崩导致一批 trajectory 全拿不到有效 reward，GRPO 组内归一化后 advantage 全是 0，白费算力。

**修复**：DAPO `enable_filter_groups=True`，按 seq_final_reward 把全零组丢掉，只让有效组参与归一化。

### 4.6 RL 冷启动格式崩坏

**症状**：RL 训练早期，模型很快忘了 `<code>`/`<answer>` 格式，输出一堆解析不了的东西。

**根因**：纯 RL 冷启动 reward 信号稀疏，模型没有格式约束就放飞。

**修复**：`sft_loss_type=mu` 混合损失，mu 从 0.9 衰减到 0.05，早期 SFT 占主导稳格式，后期放开探索。

### 4.7 多轮 prompt 膨胀

**症状**：多轮交互里 prompt 越来越长，显存和推理时间都涨。

**修复**：`max_obs_length=2048` 截断 observation，`max_prompt_length=2048` 限制 prompt 总长，防止无限膨胀。

### 4.8 vLLM/SGLang 和训练抢显存

**症状**：rollout 推理和训练更新在同一批 GPU 上抢显存，OOM。

**修复**：`gpu_memory_utilization=0.4` 给 rollout 只留 40% 显存，剩下给训练侧的 actor/ref。

---

## 五、超参数调优

| 参数 | 试过的值 | 最终 | 理由 |
|------|----------|------|------|
| lr | 5e-6 / 1e-6 | 1e-6 | RL 对 lr 敏感，1e-6 稳 |
| rollout_n | 2 / 4 / 8 | 4 | 成本 vs advantage 方差折中 |
| mu_peak | 0.5 / 0.9 | 0.9 | 早期要强格式约束 |
| mu_decay_steps | 100 / 350 | 350 | 太短格式没稳住 |
| gpu_memory_utilization | 0.5 / 0.4 | 0.4 | 给训练留空间 |
| clip_ratio | 0.2 | 0.2/0.28 不对称 | 允许正向多走一点 |

---

## 六、SFT 微调

### 6.1 为什么先 SFT 再 RL

RL 冷启动容易崩（见 4.6），SFT 先教会模型 `<code>`/`<answer>` 格式和基本分析流程，RL 再在这个基础上探索。

### 6.2 SFT 数据怎么构造

ProData-12K 是数据合成管线产出的轨迹（见数据合成模块），包含问题 + 完整的多步解题轨迹。

### 6.3 SFT 训到什么程度转 RL

看格式正确率（`<code>`/`<answer>` 都能正常生成）和基础任务准确率。格式崩了别转 RL，先补 SFT。

---

## 七、Reward 函数设计

数据分析任务用 rule-based 的 template_reward + 模型 judge 的 answer_reward，不用学习式 reward model。

**为什么 rule-based 而非学习式**：数据分析的格式错误很明确（没 `<answer>`、代码没跑成功），rule 就能判；答案正确性用 gpt-4o-mini judge 判断，数值题 3% 容差、选择题精确匹配。

**边缘情况**：预测为空直接 0 分；judge 返回格式异常（没 `<score>` 标签）也判 0，避免误给分。

---

## 八、RL 训练阶段

### 8.1 SFT→RL 衔接

从 SFT checkpoint 起训，RL 的 mu 损失继续用 SFT 数据做约束，防止早期格式崩坏。

### 8.2 response_mask 的作用

RL 只对 response 部分算 loss，prompt 部分（包括 observation）不参与梯度。这样模型只被训练"生成更好的 action"，不被训练"复述 observation"。

### 8.3 KL penalty 实际效果

主训练里 KL 关掉了（kl_coef=0），因为 GRPO 靠组内归一化约束，加 KL 会压制探索。需要稳定性的场景才开 KL。

### 8.4 GAE vs GRPO 实际差异

多步 agent 里 GAE 要 value bootstrap，但中间状态（代码执行结果）的 value 不好估。GRPO 组内归一化绕开 value 估计，更适合这个场景。

---

## 九、训练监控

### 9.1 看哪些指标

reward 均值/分布、格式正确率（template_reward）、entropy、loss。reward 看爬坡，格式看稳定，entropy 看坍缩。

### 9.2 怎么看训练正不正常

reward 涨 + entropy 稳定下降 = 正常；reward 不涨 + entropy 掉 = 坍缩；reward 全 0 = 执行器或 reward 函数有问题。

---

## 十、整体总结

### 10.1 显存管理优先级（从轻到重）

调 gpu_memory_utilization → 调 max_prompt/response_length → 开 offload → 换更大的卡。

### 10.2 GPU 数量选择

| GPU 数 | 可行性 | 配置调整 |
|--------|--------|----------|
| 1 | 仅推理/评测 | 关训练 |
| 4 | 7B SFT | ZeRO-3 |
| 8 | 7B/14B RL | rollout_n 可到 4 |
| 16 节点 | 14B 大规模 RL | 多机 Ray |

### 10.3 换个任务/算法要注意什么

换任务：改数据加载 + reward 函数 + 评测。换算法：改 advantage 计算 + trainer，成本最高。

### 10.4 模块之间故障定位

reward 全 0 → 先查执行器（stderr）→ 再查 reward 函数（对已知答案测试）→ 最后查数据路径（csv_folder 对齐）。这个顺序从"最可能崩"到"最不可能崩"。

### 10.5 推荐默认参数组合

冒烟测试：max_turns=1、rollout_n=1、小数据切分。完整训练：multi.sh 里的默认参数（rollout_n=4、mu 0.9→0.05、max_turns=10）。

---

## 十一、最后

### 11.1 这个项目让我理解的几件事

GRPO 说到底就是把"这组里谁更好"用 reward 的相对排序表达出来，不用训 Critic。听起来高大上，看了代码就是组内归一化加广播。真正难的不是算法，是让多轮代码 rollout 在训练热路径里稳定——执行器崩一次、reward 全零一次，训练就卡死，这些才是 agent RL 区别于纯 LLM RL 的地方。

### 11.2 数据分析 agent RL 与普通 RL 的区别

普通 RL（数学/代码）是静态的：一个 prompt 一个 response，reward 由规则判。数据分析 agent 是动态的：模型在环境里多轮交互，每步 action 是代码，执行结果影响后续，还有静默错误这种 outcome reward 抓不到的失败。所以多了两个普通 RL 没有的复杂度：执行环境的稳定性、和过程级 credit assignment。

---

## 附录：训练各阶段核心超参数速查

### SFT

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-Coder-7B/14B-Instruct |
| 框架 | LLaMA-Factory 0.9.4.dev0 |
| 微调类型 | 全参数 + ZeRO-3 |
| cutoff_len | 8192 |
| lr | 1e-5 |
| epochs | 3 |
| scheduler | cosine |

### RL

| 参数 | 值 |
|------|-----|
| 框架 | verl v0.4.0 |
| 算法 | DAPO + GRPO |
| adv_estimator | grpo |
| rollout_n | 4 |
| max_turns | 10 |
| sft_loss_type | mu（0.9→0.05，350 步） |
| lr | 1e-6 |
| gpu_memory_utilization | 0.4 |
| sp_size | 4 |
