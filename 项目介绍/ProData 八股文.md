# ProData 项目八股文 —— 核心技术主题详解

> 每个主题包含：核心概念、本项目中的应用、代码示例、面试追问要点。
> 基于项目实际代码编写，覆盖 RL 算法、分布式训练、Agent 工具调用、代码执行、自进化机制、过程奖励等领域（共 19 个主题）。
> 全文所有文件路径都指向 `ProData/` 目录下的真实源码，配置值均来自 `multi.sh` / `config.yaml` / 各 yaml 文件。

---

## 0. 项目总览：四块拼图如何拼成一个完整故事

面试开场大概率会问"你这个项目整体在做什么"。先建立一张全景图，后面 18 个主题都是这张图上的局部放大。

**一句话定位**：ProData 是一套「**数据分析 Agent 的训练配方 + 数据合成流水线**」，解决的是——真实数据分析对 Agent 有高要求（多格式、大规模数据文件、长程多步推理），但现有数据、训练策略、多轮 rollout 稳定性都不够。

**四个子项目，各司其职**：

| 子项目 | 角色 | 关键产出 / 数字 |
|--------|------|-----------------|
| **ProData**（核心） | 数据合成 + 训练配方 | ProData-12K 轨迹集；ProData-14B 平均 71.16%、ProData-7B 68.10% |
| **DataPRM** | 过程奖励模型（PRM） | 环境感知生成式 PRM（4B）；Best-of-N 提升 ScienceAgentBench +7.21%、DABStep +11.28% |
| **DataCOPE** | 自进化 Skill 闭环 | da_agent → verifier → skill_manager 三角色迭代产出 SKILL.md |
| **LongDS** | 长程评测基准 | 68 任务 / 2225 轮（Lite 24 任务 / 777 轮），6 种状态演化模式 |

**ProData 主训练的四件事**（README 原文的四点，是对"训练配方"最精炼的概括）：

1. **细粒度任务分类法 + 递归的 easy-to-hard 任务组合机制** → 提升合成 query 的多样性和难度；
2. **知识增强的轨迹采样 + 模型过滤 + 规则过滤** → 保证轨迹质量；
3. **动态可调的 SFT + RL 混合训练目标** → 对应主题 3 的 `mu` 衰减；
4. **省显存且稳定的代码多轮 rollout 框架** → 对应主题 6、13、14。

**训练管线顺序**：`ProData-12K 数据` → `SFT（LLaMA-Factory + ZeRO-3）` → `RL（verl + DAPO/GRPO）` → 评测（DABench / TableBench / BIRD）。DataPRM 和 DataCOPE 是围绕主训练的"增强组件"，LongDS 是"更难的长程评测"。

**面试时的一句话版本**：我做了数据分析 Agent 从数据、训练到评测的完整闭环——主训练用 GRPO/DAPO 做 RL，冷启动用 SFT+RL 混合损失稳格式，多轮代码执行用独立沙箱隔离，另外做了环境感知的 PRM 去抓"静默错误"，还用自进化 Skill 闭环让 Agent 从自己的轨迹里提炼经验。

---

## 1. GRPO（组内相对奖励）

**核心概念**：GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的 policy 优化方法，核心是**省掉 Critic 网络**。PPO 需要训一个 value 网络来估计 advantage `A(s,a) = r + γV(s') - V(s)`，这多了一整套网络 + 优化器 + 显存。GRPO 的替代方案是：对同一个 prompt 采样一组 N 条 response，用「**组内归一化的 reward**」直接当 advantage：

```
A_i = (r_i - mean(r_1..r_N)) / std(r_1..r_N)
```

**直觉**：advantage 本来回答的是"这条 response 比平均好多少"。PPO 用 value 网络估这个"平均"，GRPO 直接拿同一 prompt 的 N 条采样算个经验平均。好处是不用训 value 网络，省一半显存和训练复杂度；代价是 advantage 方差比 GAE 大（只依赖 N 条采样，N 小时估计很抖），对 rollout 数量敏感。

**本项目**：`multi.sh` 里 `adv_estimator=grpo`、`rollout_n=4`、`n_sample=1`（组大小）。这里有个关键点——多轮 agent 场景下，reward 是按 **trajectory** 算的标量，所以 GRPO 的组内归一化必须发生在 **trajectory 维度**，而不是 token 维度。这也是主题 5「trajectory 展平 + uid 聚合」的由来：先把组内每条 trajectory 的标量 advantage 算出来，再广播回这条 trajectory 的每个 token。

另一个和 GRPO 绑定的配置是 `loss_agg_mode="token-mean"`——policy loss 在 token 维度怎么聚合。多轮场景下每条 trajectory 长度不同，如果按 sequence-mean 会偏向短序列；token-mean 保证每个有效 token 贡献一致，配合 response_mask 屏蔽 padding。

**追问**：
- **为什么 rollout_n=4 而不是更大？** → 每条 trajectory 是多轮代码执行，成本远高于单轮问答，rollout_n 翻倍意味着代码执行次数翻倍；4 是显存和方差之间的折中。
- **GRPO 为什么通常要配 KL 惩罚？这里为什么 `use_kl_in_reward=False`？** → GRPO 原文会从 reward 里扣掉 `β·KL(π||π_ref)` 防止 policy 跑飞。但 DAPO 的实验发现 KL 项其实会**压制性能**（过度约束探索），所以这条线（DAPO 变体）干脆关掉 KL（`kl_coef=0.0`、`use_kl_loss=False`），靠 clip 和 filter_groups 来约束更新幅度。
- **GRPO 和 PPO 显存差在哪？** → 省的是 value 网络（参数、梯度、优化器状态）和它的一次前向+反向。RL 里 actor/ref/rollout 三份显存已经压得很紧（见主题 14），砍掉 critic 是实打实的显存收益。

---

## 2. DAPO（filter groups + 动态采样）

**核心概念**：DAPO 在 GRPO 基础上加了几处改进，最关键的是「**dynamic sampling 过滤**」——把组内所有 response 的 reward 都为零的组直接丢掉，不参与梯度。为什么需要它：如果一个组里 reward 全零，组内归一化后 advantage 全 0，这组样本既不算有效信号、又白占算力，还会因为 `(ri - mean)/std` 的 std=0 造成数值问题。丢掉的代价是有效 batch 变小，所以 DAPO 会「多采样」来补：**生成更多 batch，过滤后凑够训练 batch**。

**本项目**：`multi.sh` 里：

```bash
enable_filter_groups=True
filter_groups_metric=seq_final_reward
max_num_gen_batches=10

train_prompt_bsz=16
gen_prompt_bsz=$((train_prompt_bsz * 2))   # 32，生成 batch 是训练 batch 的 2 倍
```

注意 `gen_prompt_bsz = train_prompt_bsz * 2` 这条——**生成量翻倍就是"动态采样"的落地**：先多生成，`filter_groups` 把全零组丢光，剩下的再凑 `train_prompt_bsz` 进梯度。`max_num_gen_batches=10` 是"最多生成 10 个 batch 去找足够多有效组"的上限，防止某个数据段质量太差导致无限重采样、卡死训练。

`filter_groups_metric=seq_final_reward` 表示按「轨迹最终 reward」判断整组是否有效。这直接对应训练里踩过的坑——代码执行一崩，一批 trajectory 全拿不到 reward，组内归一化后 advantage 全是 0，不丢这些组训练就卡死。

**追问**：
- **为什么要按 seq_final_reward 而不是 seq_reward（过程 reward）过滤？** → 过程 reward 是稠密信号，全零的概率低，用它过滤几乎不起作用；最终 reward 全零才是"这组废了"的信号。
- **gen_prompt_bsz 翻倍会不会浪费？** → 表面多算了一半，但被过滤的组本来就不该进梯度；不翻倍的话过滤后 batch 变小，每次 update 的有效样本不足，反而浪费训练步数。这是"用冗余换稳定"。
- **DAPO 和 GRPO 还有哪些区别？** → 除了 filter groups，还有 clip-higher（主题 4 的 `clip_ratio_high=0.28`）、去掉 KL 项、以及这个动态采样。四点是成套的。

---

## 3. SFT + RL 混合损失（sft_loss_type=mu）

**核心概念**：纯 RL 冷启动容易格式崩坏——reward 信号稀疏，模型很快忘掉 `<code>`/`<answer>` 这些格式 token，输出没法解析。混合损失把 SFT loss（模仿参考轨迹）和 RL loss（policy 梯度）按系数 `mu` 加权：

```
L = (1 - mu) · L_rl + mu · L_sft
```

`mu` 从高到低衰减：训练早期靠 SFT 稳格式、保语法，后期靠 RL 探索。这是"**先扶着走，再放手跑**"的思路——数据分析场景尤其需要，因为 reward 要能解析出 `<answer>` 才有意义，格式一旦崩了 reward 就全废。

**本项目**：`multi.sh` 里：

```bash
sft_loss_type=mu
mu_warmup_steps=0
mu_decay_steps=350
mu_peak=0.9
mu_valley=0.05
```

`mu` 峰值 0.9（早期 SFT 占 90%、RL 只占 10%）衰减 350 步到 0.05（后期 SFT 只占 5%）。`mu_warmup_steps=0` 表示不预热，直接从 0.9 开始线性衰减。350 步是我调的——太短格式没稳住，太长抑制探索。

**追问**：
- **mu 直接固定 0.5 不行吗？** → 固定值意味着训练全程都在"既要模仿又要探索"，两个目标互相拉扯，既没稳格式也没放开探索，收敛更慢。衰减的本质是让训练早期有一个"确定性锚点"。
- **SFT loss 的参考轨迹从哪来？** → 来自 ProData-12K 里的高质量轨迹（知识增强采样 + 过滤后），RL 阶段每条 rollout 都有一条对应的 SFT 参考响应，用它算交叉熵作为 L_sft。
- **mu_decay_steps 怎么选？** → 经验上对齐"格式正确率稳定"的时间点：观察 SFT 侧的格式 loss 什么时候收敛、格式正确率什么时候稳住，太短格式没稳就衰减会崩，太长探索被压制、reward 上不去。

---

## 4. PPO 的 clip ratio（双 clip / clip-higher）

**核心概念**：PPO 用 clip 限制 policy 更新的幅度，防止一步更新把 policy 冲太远。ratio `r = π_new / π_old`，clip 到 `[1-ε, 1+ε]`，loss 取 `min(r·A, clip(r)·A)`。ε 太小更新慢，太大不稳定。

DAPO 的改进叫「**clip-higher**」：上下界**不对称**。标准 PPO 对称 clip（`[1-ε, 1+ε]`）会**过度惩罚正向 advantage 的提升**——当某条 rollout 真的好（A>0、r 很大），对称 clip 会把 r 压回来，抑制模型往好方向走。所以 DAPO 把上界放宽（`clip_ratio_high > clip_ratio_low`），让模型敢于在好方向上多走一点。

另外还有一个「**dual-clip PPO**」（论文 arXiv:1912.09729）的 `clip_ratio_c` 常数：对**负 advantage**（A<0，惩罚侧）额外设一个常数上界。

**本项目**：`multi.sh` 里 `clip_ratio_low=0.2`、`clip_ratio_high=0.28`、`clip_ratio_c=10.0`。真正的 loss 在 `verl/trainer/ppo/core_algos.py` 的 `compute_policy_loss`：

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

pg_losses3 = -advantages * clip_ratio_c          # 负 advantage 侧的常数上界
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

解读：`clip_ratio_c=10.0` 不是 reward 归一化，而是**负 advantage 时对 loss 的一个常数上界**（`-A · 10.0`）。它防止某条 rollout 的 ratio 爆炸、产生巨大的负梯度把 policy 一巴掌打飞——`min(pg_losses3, clip_pg_losses1)` 保证惩罚侧 loss 不超过 `-A·10`。

**追问**：
- **为什么低 0.2 高 0.28 不对称？** → 对称 clip 会过度惩罚正向 advantage 提升，DAPO 用更宽的上界让模型敢于在好方向上多走一点（clip-higher）。
- **clip_ratio_c 到底在约束什么？** → 负 advantage 侧。`A<0` 时 policy 要"远离"这条坏 rollout，但远离的幅度不该无限大，`-A·10` 是个硬上限。选 10 是因为它足够大（基本不挡正常更新），只在极端 ratio 时才生效。
- **为什么还要 `grad_clip=1.0`？** → clip ratio 管的是"policy 更新的目标函数"层面，grad_clip 管的是"梯度范数"层面，两者是不同层级的保险，一个防目标爆炸、一个防梯度爆炸。

---

## 5. 多轮 trajectory 展平 + uid 聚合

**核心概念**：多轮 agent 每条 trajectory 长度不同，要拼成规整 batch 就得展平；但 GRPO 的 advantage 是**按 trajectory 算的标量**，得广播到该 trajectory 的每个 token 上。所以需要记录每个 token 属于哪条 trajectory（uid），展平后按 uid 广播 advantage。这还牵涉到「有效长度裁剪」——拼接后的 batch 里混着 padding，得先算出真正的有效长度再裁剪。

**本项目**：`agent/tensor_helper.py` 的 `TensorHelper` 类集中处理这些。核心方法：

```python
def cut_to_effective_len(self, tensor_dict, keys, cut_left=True):
    # attention_mask 求和 = 每条序列的有效 token 数，取 batch 内 max 作为统一裁剪长度
    effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
    result = tensor_dict.copy()
    for key in keys:
        if cut_left:
            result[key] = tensor_dict[key][:, -effective_len:]   # 左填充 → 裁右边
        else:
            result[key] = tensor_dict[key][:, :effective_len]
    return result

def concatenate_with_padding(self, tensors, pad_to_left=True):
    concatenated = torch.cat(tensors, dim=1)
    padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
    return padded_tensor

def _example_level_pad(self, responses, responses_str, active_mask):
    # 把非 active 的 example 整体 pad 掉（对齐 filter_groups 丢组后的 batch）
    padded_responses = torch.full((batch_size, seq_len), self.config.pad_token_id, ...)
    padded_responses[active_mask] = responses
    ...
```

`_example_level_pad` 是配合 DAPO filter_groups 的关键：过滤丢组后，batch 里有些 example 是"空壳"，需要用 active_mask 把它们整条 pad 掉，同时保证 `active_mask.sum() == responses.shape[0]`（有效 sample 数和响应数对齐）。

**追问**：
- **为什么不直接按 trajectory 训练，省掉展平这一步？** → 每条 trajectory 长度不同，不展平就得 padding 到最大长度，GPU 算一堆 padding 浪费；展平后 batch 里全是有效 token，只靠 attention_mask 标记边界。
- **`cut_to_effective_len` 为什么取 batch 内 max 而不是每条各自的长度？** → batch 必须是规整矩形张量，同一 batch 内所有序列得对齐到统一长度，取 max 保证不丢 token，靠 attention_mask 让短的序列忽略多余部分。
- **uid 在展平里起什么作用？** → 展平后 token 序列失去了"边界"，uid 告诉 loss 计算"这个 token 属于哪条 trajectory、该用哪个标量 advantage"，否则 advantage 广播会错位。

---

## 6. 代码执行沙箱隔离

**核心概念**：训练时模型生成的代码不可信——可能死循环、吃满内存、import 有毒包、执行破坏性 SQL。隔离手段包括：独立进程/环境、内存上限、超时、只读权限、SQL 白名单。

**本项目**：`agent/async_interpreter.py` 的 `run_task` 是最核心的隔离层：

```python
def run_task(code, timeout):
    running_code = "\n".join(code)

    def limit_memory():
        import resource
        mem = 1024 * 1024 * 1024  # 1GB
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(running_code)
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(
            ["/root/anaconda3/bin/conda", "run", "-n", "darl-run", "python", tmp_file_path],
            capture_output=True, preexec_fn=limit_memory, timeout=timeout
        )
        ...
    except subprocess.TimeoutExpired:
        return "", "TimeoutError: 'Timed Out'"
    finally:
        os.remove(tmp_file_path)  # 清理临时文件
```

四层隔离，层层递进：

1. **独立 conda 环境**：`conda run -n darl-run` 跑在单独的 Python 3.10 环境，和训练进程隔离，即使代码装乱七八糟的包也不污染训练环境；
2. **内存上限**：`preexec_fn=limit_memory` 在子进程 fork 后、exec 前设置 `RLIMIT_AS` 1GB；
3. **超时**：`subprocess.run(timeout=...)` 到点抛 `TimeoutExpired`；
4. **临时文件**：代码写进 `NamedTemporaryFile`，`finally` 里删掉。

还有两个容易被忽略的细节：`execute()` 里检测到 `No module named` 会**自动 pip install 缺失的包然后重跑一次**（带 `check_module_in_env` 判断避免装了还装）；`filtered_stderr` 把 `ERROR conda.cli.main_run` 这类 conda 噪音过滤掉，避免污染 observation。

SQL 侧只放行 SELECT（`is_not_select_query`）：

```python
def is_not_select_query(self, sql):
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False
    first_statement = parsed[0]
    return first_statement.get_type() in ("CREATE", "INSERT", "UPDATE", "DELETE")
```

SQL 执行还用了**独立进程 + 硬超时**（`exec_on_db` 里 `multiprocessing.Process` + `p.join(timeout)`，超时就 `terminate()`），因为 SQLite 里一个长查询可能卡住主进程，`subprocess` 的超时不一定管用。

**追问**：
- **内存上限用 RLIMIT_AS 有什么副作用？** → RLIMIT_AS 限制的是虚拟地址空间，不是物理内存，某些库（如 numpy）会预留大虚拟空间，可能误杀。所以 1GB 是经验值，够跑常规分析但能挡住失控代码。
- **为什么 SQL 要额外用 multiprocessing 而不是复用 subprocess 超时？** → SQLite 是进程内执行的 C 库，一个恶意/超长查询可能在 C 层卡住，Python 的 subprocess 超时只能管到"启动"，管不到"执行中"；独立进程 + terminate 是硬兜底。
- **auto pip install 会不会被模型利用注入恶意包？** → 有风险，但它只装"缺失的 top-level 模块名"，且装在隔离的 darl-run 环境、有内存/超时兜底。严格来说生产环境应该改成白名单，这里是训练态的一个折中。

---

## 7. 自进化 Skill 闭环（DataCOPE）

**核心概念**：自进化 agent 的核心是「**从自己跑过的轨迹里提炼经验**」。三个角色：`da_agent` 解题、`verifier` 挑出有效轨迹、`skill_manager` 把有效轨迹抽象成可复用的 `SKILL.md`。下一轮 `da_agent` 带着 skill 再跑，形成正反馈——这是把"经验"从一次性轨迹里蒸馏成"可迁移的 prompt 资产"的过程。

**本项目**：`core/loop.py` 的 `run()` 主循环。关键逻辑：

```python
for iteration in range(iterations + 1):
    # 上一轮的 skill 目录（iteration 0 时没有）
    previous_skill_output_dir = (None if iteration == 0
                                 else skill_manager_output_dir_prefix / f"iter_{iteration-1}" / "skills")
    stage_use_skill = iteration > 0          # 第 0 轮不带 skill，第 1 轮起带
    stage_skills_base_dir = (None if previous_skill_output_dir is None
                             else str(previous_skill_output_dir))

    run_da_stage(...)                         # da_agent 用 skill 解题
    verifier_results = run_verifier_stage(...) # verifier.init_run / iterate_run
    run_skill_manager_stage(...,
        mode="create" if iteration == 0 else "modify", ...)  # 第0轮创建，之后修改
```

一个容易被忽略的细节是 `mode="modify"` 时，先 `shutil.copytree(previous_skill_output_dir, skill_manager_output_dir, ignore=...prediction_*.json)` 把上一轮的 skill **复制过来再改**，而不是从零生成——这保证了 skill 是**增量演化**，不会每次重写丢掉之前的好经验。

还有个工程点：`stage_success_status.json` 记录了每个 stage 的成功状态，配合 `--resume-from-iteration` / `--resume-from-stage` 支持**断点续跑**——大任务跑到一半崩了，不用从头来。

三种 verifier 里，`agreement`（自洽）和 `supervised`（有 ground truth）都实现了 `init_run` / `iterate_run` 两个接口：第 0 轮 `init_run`（只看当前轮），第 1 轮起 `iterate_run`（对比 prev 和 curr 两轮，找出"加了 skill 之后答案发生了显著变化"的轨迹）。

**追问**：
- **为什么 verifier 要用 self-consistency 而不是只看 ground truth？** → 因为真实场景没有 ground truth。agreement 模式按答案聚类算 self-consistency，虽然"自洽 ≠ 正确"，但它能提供一个无需标签的轨迹质量信号，这对自进化是可扩展的。
- **create 和 modify 的区别为什么要显式分开？** → create 是从零写一份 skill，modify 是基于已有 skill 做增量修订。prompt 不同：modify 的 prompt 强调"提供的 SKILL 不保证全对，发现和当前数据/轨迹冲突要修正"，避免盲信上一轮的 skill。
- **为什么 skill 要复制再改而不是原地改？** → 保留历史版本、可回滚、可追溯；`stage_success_status` + 复制机制让每次迭代的输出独立成目录，方便对比"第 N 轮 skill 比第 N-1 轮好在哪"。

---

## 8. Self-Consistency 验证

**核心概念**：对同一问题采样多条答案，按答案聚类，出现次数多的答案更可能是对的。但注意——self-consistency **不保证正确性**（可能大家都错在同一个地方），它只是"可靠性"的代理指标。

**本项目**：`verifier/instances/agreement_verifier.py` 的 `_group_answers`：

```python
def _group_answers(self, preds):
    groups = []
    repr_answers = []
    for pred in preds:
        placed = False
        for i, rep in enumerate(repr_answers):
            if answer is not None and rep is not None and question_scorer(str(answer), str(rep)):
                groups[i].append(pred); placed = True; break
        if not placed:
            groups.append([pred]); repr_answers.append(answer)
    groups.sort(key=lambda g: -len(g))
    return groups
```

聚类的"相同答案"判断用 `question_scorer`（归一化 + 模糊匹配，处理数值等价/大小写/格式差异）。聚类后每组的 self-consistency = `组大小 / 总样本数`，`_write_answer_dirs` 里写成 `summary.json`，每组只保留**最有代表性的那条轨迹**（`_best_file` 选 `turns` 最少、`file_path` 最小的那条，也就是"最省步数到达该答案"的轨迹）。

`iterate_run`（第 1 轮起）多了一个关键过滤：如果某个 query 在 prev 和 curr 两轮里**最高自洽组的 SC 都是 1.0 且答案相同**，直接跳过——说明这个题"稳定一致"，没有值得提炼的新信息，避免 skill_manager 被无变化样本淹没。

**追问**：
- **self-consistency 和 correctness 为什么不相关？** → 模型可能系统性犯同一个错（比如都把某个 join 条件写错），这样自洽度 100% 但全错。所以 verifier 的 prompt 里反复强调"不要假设自洽=正确，要回到数据文件里验证"。
- **为什么每组只保留一条代表性轨迹，而不是全部？** → skill_manager 要读的是"这个答案是怎么解出来的"，一组内同答案的轨迹高度冗余，保留 turns 最少的那条能拿到"最高效的解法"，同时控制输入长度。
- **`question_scorer` 为什么不能简单用字符串相等？** → 数值答案"36, 51, 53"和"51, 36, 53"本质相同，字符串相等会误判成不同答案、把自洽度算低。归一化匹配才能真实反映"答案聚类"。

---

## 9. 过程奖励模型（PRM）

**核心概念**：outcome reward 只给最终答案打分，PRM 给中间每一步打分，能做更细粒度的 credit assignment。难点是 PRM 训练数据的标注——怎么知道中间某一步对不对？

**本项目**：DataPRM 是「**环境感知的生成式 PRM**」，会主动写代码去执行环境验证中间步骤（不是被动读文本，而是跑代码验证）。`evaluate/prompt.py` 的 `JUDGE_SYS_PROMPT` 定义了完整流程和三档打分：

```python
JUDGE_SYS_PROMPT = """...
- 1 表示该段落正确合理，分析和代码准确，有助于解题
- 0.5 表示整体正确但有细节遗漏或小错误，可在后续步骤修正
- 0 表示存在致命分析错误或代码错误，导致结果错误或误导
...
# Verification Format
1. ... '<reasoning></reasoning>' 先分析正确性，'<code></code>' 再写代码验证
4. System will execute the code and return results in <interpreter>...</interpreter>
   ... 若执行结果不足以判断，继续深挖；若足以判断，给出 <score></score> 和 <summary></summary>
"""
```

PRM 的验证是**多轮代码交互**：`reasoning`（分析怀疑点）→ `code`（写验证代码）→ `<interpreter>`（系统执行返回结果）→ 反复直到能下结论 → `score`（1/0.5/0）→ `summary`（一两句理由）。这就是"环境感知"的含义——PRM 不是靠语言先验猜，而是真的跑数据文件验证中间变量。

**追问**：
- **为什么用三档（1/0.5/0）而不是二档（对/错）？** → 0.5 是给"方向对、细节错"的 grounding 错误留的，这类错误下一步能救回来；如果只用二档，这些"还能救"的步骤会被粗暴当成错，误导 policy 不敢试错（这正好是 DataPRM 要解决的两大问题之一，见主题 10）。
- **为什么叫"生成式"PRM？** → 传统 PRM 输出一个标量分数（判别式），DataPRM 输出的是 `<reasoning>` + `<code>` + `<score>` 的完整生成序列，中间还主动写代码，所以是"生成式"的。
- **PRM 的 4B 参数为什么能赢过更大的模型？** → 因为数据质量而非模型规模：8K+ 训练实例通过"多样性驱动轨迹生成 + 知识增强的 step 级标注"构造，标注信号本身就把"静默错误"和"可纠正错误"区分开了。

---

## 10. 静默错误（Silent Error）

**核心概念**：代码不报错、输出结果也像模像样，但逻辑是错的。这是数据分析 agent 最隐蔽的失败模式——解释器只能抓语法/运行时错误，抓不到"你算的是 A 却回答了 B"。典型例子：join 条件写错但结果非空、单位没换算（元 vs 万元）、过滤条件漏了一个、聚合列选错——这些都不会抛异常，但结论全错。

**本项目**：这是 DataPRM 的**立身之本**。README 里明确写：通用领域 PRM 抓不到静默错误，还会把必要的试错误判成失败。DataPRM 针对这两点做了两件事：

1. **主动写代码验证中间状态**（抓静默错误）：不是被动看文本，而是真的跑一段验证代码，检查中间变量是否符合预期；
2. **reflection-aware 三元奖励**（不误伤试错）：用 0.5 这个档位区分"可纠正的 grounding 错误"和"不可挽回的错误"，避免把正常的 trial-and-error 当成失败惩罚。

这两个改进背后是一个实验观察：通用 PRM（比如数学领域的）迁移到数据分析场景会**双重失效**——既漏报静默错误（因为数学 PRM 靠文本推理、不碰真实数据），又误报试错行为（数据分析天然要"先写代码跑一下看看，再改"）。

**追问**：
- **静默错误怎么在训练里体现？** → 它导致 outcome reward 高但过程是错的——模型"蒙对"了。PRM 能在中间步骤就发现逻辑错误，给负的中间反馈，防止 policy 学到"碰运气"的策略。
- **为什么"试错"在数据分析里是必要的？** → 真实分析里 agent 无法一次就知道数据 schema、缺失值、单位，必须先跑几段探索代码。如果 PRM 把"探索失败"全判 0，policy 会学到"不敢试"，反而降低正确率。
- **静默错误和普通 bug 的边界在哪？** → 普通 bug 解释器会报错、能抓到；静默错误是"执行成功但语义错"。这也是为什么不能用 execution reward 替代 PRM——execution reward 只看"跑没跑通"，静默错误跑通了也拿高分。

---

## 11. ReAct 与工具调用格式

**核心概念**：ReAct 让模型交替生成"思考/行动"和"观察"，行动通过特定格式（工具名 + 参数）触发工具调用，观察是工具返回结果。格式协议是关键——解析器要能稳定地从模型输出里抽出 action。

**本项目**：`eval_python.py` 用 `<code>`/`<answer>` 标签协议：

```python
def postprocess_prediction(self, prediction):
    pattern = r'<(code|answer)>(.*?)</\1>'
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, ''
```

`execute_code` 里根据 action 分三路：

```python
if cur_action == 'code':
    exec_result = self._execute(generation_code)
    next_ob = f'<interpreter>\n{exec_result.strip()}\n</interpreter>'
    done = 0
elif cur_action == 'answer':
    next_ob = ''; done = 1
else:  # 没匹配到标签
    next_ob = 'Your previous action is invalid. ... Please try again.'
    done = 0
```

完整的 ReAct 循环在 `run_llm_loop`：生成 → 解析 action → 执行/结束/提示重试 → 把 observation append 回对话 → 下一轮。有个工程细节：`_postprocess_responses` 会把输出**截断到第一个 `</code>` 或 `</answer>`** 之后的部分丢弃，防止模型一次吐多个标签导致解析错乱。

**追问**：
- **为什么不用 JSON 格式的工具调用？** → 训练数据里模型输出自然语言 + 代码块，标签协议更贴合"代码生成"场景，解析也更鲁棒（正则就行，不用 JSON parser 处理格式错误）。
- **invalid action 为什么不直接算失败，而是给提示重试？** → 多轮 RL 里"格式错误"是可恢复的，给一次提示让模型纠正，既保留信号又教会它格式。直接算失败会把"格式崩了"和"答案错了"混为一谈。
- **`<think>` 标签在这套格式里是什么角色？** → Qwen 系列的思考标签，`reward_function` 的 `check()` 会把它剥掉再检查格式，保证"思考内容不干扰格式校验"。

---

## 12. Agent Harness 抽象 + Registry 模式

**核心概念**：要支持多种 agent 后端（React 自研、Codex、Claude Code）就得抽象统一接口，再用注册表按名字加载。好处是加新 agent 不用改调用方。

**本项目**：`da_agent/da_agent.py` 里 `AgentRegistry.load(agent_type, ...)` 按名字加载；三种 harness 分别对应 `react`、`codex`、`claude_code`，都实现 `solve_task`。注册机制：

```python
from src.runtimes.registry import register_agent
from src.runtimes.base_agent import BaseAgent

@register_agent("my_agent")
class MyAgent(BaseAgent):
    ...
```

`da_agent.py` 的调用侧完全屏蔽差异：`DaAgent.generate` 里对每个 task 调 `agent.solve_task(prompt, system=system_prompt)`，只有 React 需要额外传 `extra_info`，其余一致。system prompt 也按 agent 类型分派（`_get_system_prompt` 返回 `REACT_SYSTEM_PROMPT` / `CODEX_SYSTEM_PROMPT` / `CLAUDE_CODE_SYSTEM_PROMPT`）。

**追问**：
- **三种 harness 的差异在哪？** → React 是自研的文本循环（自己控制 max_turns、temperature），Codex/Claude Code 是走它们原生的 CLI/SDK。抽象层把这种差异藏起来，上层 da_agent 只调 `solve_task`。
- **Registry 相比 if-else 的好处？** → 开放封闭原则：加新 agent 只需 `@register_agent` 注册 + import，不改 `DaAgent.generate`。而且 `import src.runtimes.agents` 触发所有 decorator，注册是**导入时自动完成**的，不会漏注册。
- **为什么 React 要单独传 extra_info？** → React 自研循环需要一些运行时上下文（比如 trajectory id、index）写进 prediction 文件，而 Codex/Claude Code 的 SDK 内部处理，不需要外部传入。

---

## 13. 左填充与 attention_mask（TensorHelper）

**核心概念**：语言模型通常右填充（padding 在序列末尾），但某些推理场景要左填充（padding 在开头，让生成的 token 对齐到序列末尾）。attention_mask 标记哪些是有效 token，position_ids 从 mask 推导。

**本项目**：`agent/tensor_helper.py` 里几个关键方法：

```python
def create_position_ids(self, attention_mask):
    # cumsum 后减 1，再乘 mask 把 padding 位置归零
    return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

def convert_pad_structure(self, tensor, pad_to_left=True):
    # 用 argsort 稳定地把 padding token 排到一边
    mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
    sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
    return tensor.gather(1, sorted_indices), sorted_indices
```

`convert_pad_structure` 的 argsort 技巧很巧妙：把"是否 padding"转成 0/1 掩码，`argsort` 把 0（padding）排前面、1（有效）排后面，就实现了左填充的**重排**，同时返回 `sorted_indices` 方便反向映射。`create_position_ids` 用 cumsum 保证 position id 只对有效 token 递增、padding 位置归 0。

**追问**：
- **为什么 RL 多轮 rollout 要左填充？** → 多轮里 prompt 会越来越长（不断 append response + observation），左填充保证有效 token 对齐到序列右端，方便按最大有效长度裁剪，也匹配 SGLang 的 KV cache 布局（`data.truncation='left'` 也对应这个）。
- **position_ids 为什么不能直接用 `arange`？** → 左填充下，序列开头是 padding，`arange` 会把 padding 也算进位置编码，导致有效 token 的 position 偏移。cumsum 方案让 padding 位置归零、有效 token 从 0 连续编号。
- **`cut_to_effective_len` 和 `convert_pad_structure` 怎么配合？** → 先重排（padding 靠左），再按有效长度裁掉左边的 padding，得到紧凑的有效 token 序列，节省 KV cache 和计算。

---

## 14. 分布式训练（Ray + FSDP + Sequence Parallel）

**核心概念**：大模型 RL 训练要跨多卡多机。Ray 做任务调度（rollout worker、trainer worker），FSDP 做参数分片，sequence parallel（如 ulysses）把长序列切到多卡降低单卡激活显存。

**本项目**：`multi.sh` 里：

```bash
sp_size=4                                    # ulysses 序列并行度 4
fsdp_size=-1                                 # -1 表示全分片（每卡一份）
actor_ppo_max_token_len=$((2048 + 8192))     # 10240
gpu_memory_utilization=0.4                   # rollout 只给 40% 显存
use_remove_padding=True                       # 去掉 padding 计算
enable_gradient_checkpointing=True            # 重计算换显存
use_dynamic_bsz=True                          # 动态 batch size
rollout_mode=sync                             # 同步 rollout
rollout_engine_name=sglang                    # 推理引擎
```

关键约束在 `max_start_length=2048` / `max_obs_length=2048` / `max_prompt_length=2048` / `max_response_length=8192`——多轮场景下 prompt（历史）和 observation（执行结果）都要单独截断，避免长程对话把序列撑爆。

**追问**：
- **为什么 RL 训练比 SFT 显存压力大这么多？** → RL 同时要训 actor（前向+反向+优化器状态）+ ref（前向）+ rollout（推理 KV cache），三份显存叠加，所以 `gpu_memory_utilization` 只给 0.4，还给 rollout 单独留空间。
- **`fsdp_size=-1` 和 `sp_size=4` 怎么分工？** → FSDP 把**参数/优化器状态**分片到所有卡（省参数显存），ulysses SP 把**单条长序列的激活**切到 4 卡（省激活显存）。一个解决"参数多"，一个解决"序列长"，正交互补。
- **`use_remove_padding` 为什么能省显存？** → 常规 batch 里有大量 padding token 也被算进 attention 和 FFN，remove_padding 把 batch 里所有有效 token 紧凑排布、只算有效部分，省的是"padding 白算"的激活显存。

---

## 15. RLHF 冷启动（SFT → RL）

**核心概念**：直接 RL 冷启动（从基座模型开始 RL）容易崩，因为 reward 稀疏、policy 没学过目标格式。先 SFT 让模型学会基本格式和任务分布，再 RL 微调。数据分析场景尤其需要——SFT 先教会模型 `<code>`/`<answer>` 格式和基本的分析流程。

**本项目**：SFT 用 `prodata_12k_full_sft.yaml`（Qwen2.5-Coder-7B/14B，cutoff_len=8192、lr=1e-5、3 epoch、ZeRO-3），RL 再从 SFT checkpoint 起训。注意 SFT 和 RL 的学习率差了 10 倍：SFT `lr=1e-5`，RL `lr=1e-6`（`actor_rollout_ref.actor.optim.lr=1e-6`、`lr_warmup_steps=20`）——RL 阶段必须更小步长，否则 policy 被 reward 信号冲飞。

**追问**：
- **SFT 训到什么程度转 RL 合适？** → 看 SFT 的格式正确率（`<code>`/`<answer>` 都能正常生成）和基础任务准确率。格式崩了别转 RL，先补 SFT。
- **为什么 RL 的 lr 要比 SFT 小一个数量级？** → SFT 是最大似然、目标函数平滑，可以用大 lr；RL 的 policy 梯度方差大、reward 信号有噪声，lr 大了 policy 会剧烈震荡甚至崩溃，所以要 1e-6 这种保守值 + warmup。
- **冷启动崩的典型表现是什么？** → 模型输出不再带 `<answer>` 标签（格式崩坏）、重复无意义 token、或者只输出一种 action 的坍缩。这些都是 reward 稀疏 + 没学过格式的直接后果，SFT 预热就是专门治这个。

---

## 16. Outcome Reward vs Process Reward

**核心概念**：outcome reward 只在轨迹末尾给一个 0/1，credit assignment 粗糙，中间步骤的好/坏都被平均掉。process reward 给每一步打分，能定位错误发生在哪一步，训练信号更稠密、更精确。

**本项目**：ProData 主训练用 outcome reward（`use_template_reward=true`，`answer_ratio=0.9` + `template_ratio=0.1`），DataPRM 提供 process reward 作为补充。`eval_python.py` 的 `reward_function`：

```python
def reward_function(trajectory, prediction, ground_truth):
    template_reward = 1.0 if check() else 0.0   # 格式对不对
    answer_reward = 0.0                          # 答案对不对（judge model 判）
    ...
    return template_reward * 0.1 + answer_reward * 0.9, template_reward, answer_reward
```

细节拆解：
- **template_reward**：`check()` 校验轨迹格式——最后一条是 assistant 且带 `<answer>`，每个 assistant 轮次剥掉 `<think>`/`<code>`/`<answer>` 后**内容为空**（即"除了标签和思考，没有多余废话"）；
- **answer_reward**：用 judge model（`gpt-4o-mini`）对比预测答案和 ground truth，数值题 3% 容差内算对，选择题需精确匹配；
- 最终 reward = `0.1·template + 0.9·answer`，对应 `answer_ratio=0.9`、`template_ratio=0.1`。

还有个 `use_execution_reward=False`、`execution_ratio=0`——这里**关掉了 execution reward**（只按"代码跑没跑通"打分），因为 execution reward 太粗、和 answer 高度相关，而且会被静默错误骗（代码跑通了但答案错）。

**追问**：
- **为什么主训练不用 process reward 而单独做 DataPRM？** → process reward 需要 PRM 在 rollout 的每一步都推理，训练时延和成本都高；outcome reward 快但粗。所以主训练用 outcome reward 保效率，DataPRM 在需要细粒度 credit 的场景（Best-of-N、RL 增强）单独用。
- **answer_ratio 为什么给 0.9 这么高，template 只 0.1？** → 最终目标是"答案对"，格式只是前提不是目的。格式权重太高会训练出一个"格式完美但答案乱编"的模型。0.1 足够在早期拉格式，0.9 保证主信号是正确性。
- **为什么关掉 execution reward？** → execution reward 和 answer reward 高度相关（答案对基本意味着代码跑通了），加它没有新信息；而且它测不出静默错误——代码跑通但结果错的情况它给满分，反而误导。

---

## 17. Best-of-N 与 Test-Time Scaling

**核心概念**：Best-of-N 是推理时采样 N 条候选，用 reward 模型打分挑最好的。它**不做训练**，纯靠推理时的计算换性能。PRM 能在中间就筛掉差的候选，比 outcome reward 的 Best-of-N 更高效。

**本项目**：DataPRM 的 Best-of-N 推理把 ScienceAgentBench 提升 7.21%、DABStep 提升 11.28%——这就是用 4B PRM 在推理时做细粒度筛选的效果。更关键的是把 DataPRM 接进 RL 后，DABench 到 78.73%、TableBench 到 64.84%，验证了 process reward 做训练信号的价值。

**追问**：
- **Best-of-N 的收益瓶颈在哪？** → 如果 N 条候选都错在同一个地方（系统性错误），Best-of-N 挑不出好的。PRM 的价值在于它能在中间步骤区分好坏，而不是只在末尾赌最终答案。
- **PRM 的 Best-of-N 为什么比 outcome 的 Best-of-N 高效？** → outcome 的 Best-of-N 要么用最终 reward 挑（信息太少），要么全采样到末尾再比（浪费）。PRM 能在中间步骤提前剪枝，把算力集中在有希望的候选上。
- **Best-of-N 和 RL 是什么关系？** → 两者互补：Best-of-N 是"推理时"用 PRM 挑，不改变模型权重；RL 是"训练时"用 PRM 做过程监督改权重。前者即插即用、后者一劳永逸但成本高。

---

## 18. 长程状态管理（LongDS）

**核心概念**：真实数据分析是连续多轮，前面的 filter、指标定义、中间表会影响后面。agent 要能"保持、更新、复用"分析状态。LongDS 定义了 6 种状态演化模式：初始状态构建、状态继承、状态更新、反事实扰动、回滚、多状态组合。

**本项目**：LongDS 基准 68 任务 / 2225 轮（Lite 24 任务 / 777 轮），覆盖 Business、Community、Education、Geoscience、Social Good、Sports 六领域，从 Kaggle notebook 构造。Lite 版是精心选的 24 个"经验上可解且对 agent 有区分度"的任务。支持 Codex、Claude Code、Kimi Code、Qoder、DSGym 五种 runner，native CLI 模式下每任务独立 Docker 容器、跨轮保持同一 session。

**追问**：
- **为什么长程评测比单轮难这么多？** → 单轮错了就错了，长程里第 3 轮的状态错误会传导到第 15 轮，误差累积。评测要设计"状态依赖"的任务，让 agent 无法靠单轮的局部正确蒙混过关。
- **为什么 6 种模式要分开设计？** → 不同模式考的是不同的能力：状态继承考"记住上一轮"，反事实扰动考"在改动假设后重新推导"，回滚考"撤销错误状态"。混在一起测不出具体短板。
- **Lite 和 Full 怎么选？** → Lite 是"可解 + 有区分度"的子集，跑起来快、便宜，适合快速迭代；Full 更全但更贵。汇报分数必须标注版本和 split，不同版本分数不可直接比。

---

## 19. Overlong Buffer（超长响应惩罚）

**核心概念**：多轮 agent 容易生成**又臭又长但不对**的 trajectory——一直写代码、一直跑、就是不给 `<answer>`，把 rollout 的序列长度和显存都耗光。Overlong Buffer 给"超过长度上限还没出答案"的响应一个惩罚，逼模型学会在该收敛的时候收敛。

**本项目**：`multi.sh` 里：

```bash
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))     # 4096，超过这个长度算"超长"
overlong_penalty_factor=1.0           # 惩罚系数
reward_model.reward_manager=dapo
```

配合 `max_response_length=8192`：模型最多能生成 8192 token，但如果它超过 4096 还没给出有效 `<answer>`（`overlong_buffer_len`），就开始扣 reward。`overlong_penalty_factor=1.0` 是惩罚强度，可以调成 0.5 软化、1.5 加重。

**追问**：
- **为什么不直接截断，而是用 penalty？** → 直接截断会丢信息、且截断点可能正好是答案开头；penalty 是软约束，让模型自己学会"及时给答案"，而不是被硬砍。
- **overlong_buffer_len 和 max_response_length 是什么关系？** → max_response_length 是硬上限（超出就截断/丢弃），overlong_buffer_len 是软阈值（超过就扣分）。软阈值通常在硬上限的一半左右，给模型一个"该收手"的信号。
- **为什么数据分析场景尤其需要这个？** → 数据分析里"再跑一段代码看看"总是有借口，模型容易陷入无限试错循环。overlong penalty 给了一个"探索有成本"的信号，平衡探索和收敛。

---

## 快速检索表

| 主题 | 核心概念 | 文件 | 面试频率 |
|------|----------|------|----------|
| 总览 | 四块拼图 + 训练管线 | README | ⭐⭐⭐⭐⭐ |
| GRPO | 组内归一化 advantage | multi.sh | ⭐⭐⭐⭐⭐ |
| DAPO filter_groups | 丢全零 reward 组 + 动态采样 | multi.sh | ⭐⭐⭐⭐ |
| SFT+RL 混合损失 | mu 衰减 | multi.sh | ⭐⭐⭐⭐ |
| PPO clip ratio | 高低不对称 clip + dual-clip | core_algos.py | ⭐⭐⭐ |
| trajectory 展平 | attention_mask 裁剪 + uid | tensor_helper.py | ⭐⭐⭐⭐ |
| 代码执行沙箱 | RLIMIT_AS + conda 隔离 | async_interpreter.py | ⭐⭐⭐⭐⭐ |
| 自进化闭环 | 三角色迭代 | loop.py | ⭐⭐⭐⭐⭐ |
| self-consistency | 答案聚类 | agreement_verifier.py | ⭐⭐⭐⭐ |
| PRM | 1/0.5/0 三档 + 代码验证 | prompt.py | ⭐⭐⭐⭐⭐ |
| 静默错误 | 不报错但逻辑错 | — | ⭐⭐⭐⭐⭐ |
| ReAct 格式 | code/answer 标签 | eval_python.py | ⭐⭐⭐⭐ |
| Harness 抽象 | registry 模式 | da_agent.py | ⭐⭐⭐ |
| 左填充 | position_ids + argsort | tensor_helper.py | ⭐⭐⭐ |
| 分布式训练 | Ray+FSDP+SP | multi.sh | ⭐⭐⭐⭐ |
| 冷启动 | SFT→RL | prodata yaml | ⭐⭐⭐ |
| Outcome vs Process | credit assignment | eval_python.py | ⭐⭐⭐⭐ |
| Best-of-N | 推理时采样 | run_eval.py | ⭐⭐⭐ |
| 长程状态 | 6 种演化模式 | LongDS | ⭐⭐⭐⭐ |
| Overlong Buffer | 超长响应惩罚 | multi.sh | ⭐⭐⭐ |
