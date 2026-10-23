# ProData 核心流程代码追踪

> 精确到文件和行号的代码级别追踪。覆盖训练流程和自进化流程两条主线。

## 概述

两条子流程 + 核心文件索引：

| 文件 | 职责 |
|------|------|
| `multi.sh` | RL 训练入口 + 全部超参 |
| `recipe/dapo/main_dapo.py` | DAPO+GRPO 主程序 |
| `agent/async_interpreter.py` | 代码执行沙箱 |
| `agent/tensor_helper.py` | 张量裁剪/填充 |
| `core/loop.py` | 自进化主循环 |
| `da_agent/da_agent.py` | 解题 agent |
| `verifier/instances/agreement_verifier.py` | self-consistency 验证 |
| `skill_manager/skill_manager.py` | skill 生成 |

---

## 子流程 1: RL 训练（rollout → reward → advantage → update）

### 1.1 入口: multi.sh

**文件**: `train/rl/verl/multi.sh:1-229`

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${n_gpus} ...
python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    actor_rollout_ref.rollout.n=${rollout_n} \
    algorithm.adv_estimator=${adv_estimator} \
    ... 2>&1 | tee $exp_name.log
```

启动 Ray 后，把所有超参传给 `main_dapo`。关键参数：rollout_n=4、adv_estimator=grpo、sft_loss_type=mu、max_turns=10。

### 1.2 数据加载

**文件**: `recipe/dapo/main_dapo.py`

读 parquet → 转 DataProto → 分 batch。字段 prompt / data_source / reward_model.ground_truth。

### 1.3 多轮 rollout

**文件**: `examples/sglang_multiturn/`（SGLang 多轮引擎）

模型生成 response → 解析 `<code>`/`<answer>` → 代码送执行器 → observation 拼回 prompt → 循环到 done 或 max_turns=10。

### 1.4 代码执行

**文件**: `agent/async_interpreter.py:27-59`（run_task）

```python
def run_task(code, timeout):
    def limit_memory():
        import resource
        mem = 1024 * 1024 * 1024  # 1GB
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    result = subprocess.run(
        ["/root/anaconda3/bin/conda", "run", "-n", "darl-run", "python", tmp_file_path],
        capture_output=True, preexec_fn=limit_memory, timeout=timeout
    )
```

**文件**: `agent/async_interpreter.py:90-144`（CSVInterpreter.execute）

抽代码块 → 跑 → 缺包自动 pip 装 → 重跑 → 返回 result/report。

### 1.5 Reward 计算

**文件**: `eval/python/eval_python.py:591-694`（reward_function）

template_reward（格式）+ answer_reward（答案，judge 模型打分），加权 0.1/0.9。

### 1.6 filter_groups

**文件**: `recipe/dapo/main_dapo.py`（DAPO 过滤逻辑）

按 `filter_groups_metric=seq_final_reward`，reward 全零的组丢掉，max_num_gen_batches=10。

### 1.7 GRPO advantage + 反传

**文件**: `recipe/dapo/main_dapo.py`（advantage 计算 + update）

组内归一化 `(ri - mean) / std` → advantage 按 uid 广播到 token → clip loss → 反传（FSDP）。

---

## 子流程 2: 自进化（da_agent → verifier → skill_manager）

### 2.1 入口: loop.py

**文件**: `datacope/general/src/core/loop.py:271-482`（run）

```python
for iteration in range(iterations + 1):
    stage_use_skill = iteration > 0
    stage_skills_base_dir = (None if previous_skill_output_dir is None
                             else str(previous_skill_output_dir))
    run_da_stage(...)       # 解题
    run_verifier_stage(...) # 分组
    run_skill_manager_stage(..., mode="create" if iteration == 0 else "modify")
```

### 2.2 da_agent 解题

**文件**: `da_agent/da_agent.py:40-76`（generate）

`generate()` → `_evaluate_parallel/sequential` → `_evaluate_single_sample` → `agent.solve_task(prompt, system)`。

### 2.3 verifier 分组

**文件**: `verifier/instances/agreement_verifier.py:143-168`（_group_answers）

按答案聚类，`question_scorer` 判断等价，组按大小排序。

**文件**: `agreement_verifier.py:237-261`（init_run）

每个 query 聚类后 `_write_answer_dirs` 写 Answer1/Answer2/... 目录 + summary.json（含 self_consistency）。

### 2.4 skill_manager 提炼

**文件**: `skill_manager/skill_manager.py:58-76`（create_skill）

`CATEGORY_SKILL_GENERATE_PROMPT.format(...)` → `self.agent.solve_task(prompt, system=SYSTEM_PROMPT)` 产出 SKILL.md。

---

## 完整调用链速查

```
RL 训练:
multi.sh
└─ ray start --head
└─ python -m recipe.dapo.main_dapo
    ├─ 读 parquet → DataProto
    ├─ SGLang 多轮 rollout
    │   └─ async_interpreter.run_code()
    │       └─ CSVInterpreter.execute() → run_task() → subprocess(conda darl-run)
    ├─ reward_function（template 0.1 + answer 0.9）
    ├─ filter_groups（丢全零组）
    ├─ GRPO advantage（组内归一化 + uid 广播）
    └─ FSDP 反传

自进化:
start.sh
└─ python src/core/loop.py → main()
    └─ run()
        ├─ run_da_stage → DaAgent.generate → solve_task
        ├─ run_verifier_stage → VerifierRegistry.load → init_run/iterate_run
        │   └─ _group_answers → 聚类 + summary.json
        └─ run_skill_manager_stage → SkillManager.run_batch
            └─ create_skill/modify_skill → claude_code → SKILL.md
```
