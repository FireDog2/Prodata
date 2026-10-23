# ProData 核心流程完整分析

## 1. 概述

ProData 的核心是一条从"数据合成"到"训练"到"评测"到"自进化"的完整生命周期。两条主流程：**训练流程**（数据合成 → SFT → RL）和**自进化流程**（da_agent → verifier → skill_manager）。训练流程产出能解题的 agent，自进化流程让 agent 从轨迹里持续变强。

| 流程 | 输入 | 输出 | 核心组件 |
|------|------|------|----------|
| 数据合成 | 种子任务 + 数据文件 | ProData-12K 轨迹 | 任务分类 + 递归合成 + 双过滤 |
| SFT | ProData-12K | SFT checkpoint | LLaMA-Factory + Qwen2.5-Coder |
| RL | SFT checkpoint + RL 数据 | 强化后的 policy | verl + DAPO + GRPO |
| 自进化 | 任务集 | SKILL.md | da_agent + verifier + skill_manager |

## 2. 数据加载机制

RL 训练数据是 parquet 格式，字段：

| 字段 | 类型 | 用途 |
|------|------|------|
| prompt | list[dict] | 多轮对话（question + 文件信息） |
| data_source | str | 数据来源（dabench/tablebench/bird） |
| extra_info | dict | 附加信息（qsubtype、index） |
| reward_model.ground_truth | str | 标准答案 |

数据流：parquet → `main_dapo` 读入 → DataProto → rollout 采样 → trajectory 展平成 batch。

## 3. 多轮 rollout 逐轮详解

状态机：**生成 → 解析 → 执行 → 观察 → 循环**

```
[初始] prompt = question + 文件信息
   ↓
[生成] 模型生成 response（含 <code> 或 <answer>）
   ↓
[解析] postprocess_prediction 抽 action + content
   ↓
[执行] action==code → 跑代码返回 observation
       action==answer → done=1
       action==None → invalid 提示
   ↓
[观察] observation 拼进 prompt（user 角色）
   ↓
[循环] 未 done 且 < max_turns → 回 [生成]
```

每轮模型看到的是：完整历史（之前的 response + observation）。执行结果通过 `<interpreter>` 标签包裹返回。

## 4. 关键数据结构

### trajectory（多轮对话列表）

| 字段 | 类型 | 说明 |
|------|------|------|
| role | str | assistant / user |
| content | str | 生成内容 / 观察 |

### DataProto（训练 batch）

| 字段 | 类型 | 说明 |
|------|------|------|
| input_ids | Tensor | token 序列 |
| attention_mask | Tensor | 有效 token 标记 |
| trajectory_uid | Tensor | token 归属的 trajectory id |

生命周期：trajectory（字符串）→ tokenize → 展平 → DataProto → advantage 广播 → 反传。

### prediction_*.json（评测/自进化输出）

| 字段 | 说明 |
|------|------|
| prediction | 提取的答案 |
| ground_truth | 标准答案 |
| conversation | 完整轨迹 |
| turns | 交互轮数 |
| token_usage | token 统计 |

## 5. 完整调用链

```
multi.sh
└─ ray start --head
└─ python -m recipe.dapo.main_dapo
    ├─ load parquet → DataProto
    ├─ rollout（SGLang 多轮）
    │   └─ async_interpreter.run_code()
    │       └─ CSVInterpreter.execute()
    │           └─ run_task() → subprocess.run(conda run darl-run python)
    ├─ reward 计算（template + answer）
    ├─ filter_groups（丢全零 reward 组）
    ├─ GRPO advantage（组内归一化）
    └─ 反传更新（FSDP）

datacope start.sh
└─ python src/core/loop.py → main()
    └─ run()
        ├─ run_da_stage → DaAgent.generate → solve_task
        ├─ run_verifier_stage → VerifierRegistry.load → init_run/iterate_run
        └─ run_skill_manager_stage → SkillManager.run_batch → create/modify
```

## 6. 设计模式

| 模式 | 位置 | 用途 |
|------|------|------|
| 注册表模式 | registry.py | agent/verifier/dataset 按名加载 |
| 工厂模式 | AgentRegistry.load | 三种 harness 统一创建 |
| 策略模式 | verifier（supervised/agreement） | 验证策略可替换 |
| 模板方法 | BaseAgent/BaseVerifier | 定义骨架，子类实现细节 |
| 迭代器模式 | loop.py 的 iteration | 自进化闭环 |
| 装饰器模式 | @register_agent | 注册机制 |

## 7. 故障排查指南

| 症状 | 可能原因 | 诊断步骤 |
|------|----------|----------|
| reward 全 0 | 代码执行崩 / reward bug | 查 rollout 日志 stderr |
| advantage 全 0 | 组内 reward 全零 | 查 filter_groups 过滤比例 |
| 训练 loss 飙高 | mu 衰减太快 / lr 太高 | 查格式正确率、reward 分布 |
| entropy 坍缩 | 探索消失 / 卡局部最优 | 查 entropy 曲线 + reward 是否还在涨 |
| rollout 超时 | 代码死循环 / 依赖缺失 | 查执行器 timeout + stderr |
| 评测分低 | 模型没学会格式 / 数据偏差 | 查 template_reward 和 answer_reward 分开看 |
| SQL 报错 | 非 SELECT 语句 | 正常保护，看模型是否重写 |
| 显存 OOM | gpu_mem 太高 / 序列太长 | 降 gpu_memory_utilization / 开 offload |
