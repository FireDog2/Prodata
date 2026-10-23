# ProData 架构设计详解

## 设计目标

整个项目把"数据分析智能体的训练"当成一条流水线，优化目标是让每一层的产出能直接喂给下一层，而不是四套独立系统。稳定不变的是 agent 的抽象接口（solve_task）和代码执行的沙箱边界；经常变的是具体 agent 后端（react/codex/claude_code）、验证策略（supervised/agreement）、以及 RL 算法。这个"稳定接口 + 可插拔实现"的分层，让加新任务、新 agent、新验证策略都不动核心循环。

## 分层架构

```
┌───────────────────────────────────────────────┐
│  评测 / 应用层  (eval_python.py, run_eval.py) │  数据源 → 打分 → pass@3
├───────────────────────────────────────────────┤
│  自进化编排层  (loop.py, main.py)              │  da_agent → verifier → skill_manager
├───────────────────────────────────────────────┤
│  Agent 抽象层  (da_agent, base_agent)         │  solve_task 统一接口
│       ┌──────────┬──────────┬──────────┐      │
│       │  react   │  codex   │ claude   │      │  AgentRegistry 按名加载
├───────┴──────────┴──────────┴──────────┴──────┤
│  代码执行层  (async_interpreter.py)            │  conda 隔离 + 1GB + 仅 SELECT
├───────────────────────────────────────────────┤
│  训练层  (verl: dapo + grpo + mu loss)        │  Ray + FSDP + SGLang
├───────────────────────────────────────────────┤
│  数据合成层  (ProData-12K)                    │  任务分类 + 递归合成 + 双过滤
└───────────────────────────────────────────────┘
```

## 主要抽象类

**BaseAgent / BaseDaAgent**：定义 agent 的统一接口 `solve_task(prompt, system) -> AgentResult`。它只关心"输入 prompt、输出答案"，不管后端是自研 React 循环还是 Codex CLI。这个抽象是三种 harness 能共存的前提。

**AgentRegistry**：注册表，`@register_agent` 装饰器 + 按名字 `load(agent_type)`。加新 agent 只写一个类 + 一行注册，调用方无感。

**BaseVerifier**：验证器抽象，`init_run`（首次）和 `iterate_run`（迭代）两个入口。supervised 模式按 ground truth 分 success/fail，agreement 模式按 self-consistency 分组。输出统一的 `{traj_dir, prompt}` 给 skill_manager。

**BaseSkillManager**：skill 生成器抽象，`create`（第 0 轮）和 `modify`（后续轮）两个动作。它把"从轨迹提炼 SKILL.md"这件事对上层封装成一个方法。

## 数据流设计

```
用户任务 (question + data_files)
    │
    ▼
DaAgent.generate() ──► solve_task(prompt)
    │                    │
    │                    ▼
    │              生成 response
    │                    │ 解析 <code>/<answer>
    │                    ▼
    │              async_interpreter.run_code()
    │                    │ 返回 observation
    │                    └──► 循环直到 <answer> 或 max_turns
    ▼
prediction_*.json (答案 + 轨迹 + turns + token_usage)
    │
    ▼
Verifier.init_run/iterate_run() ──► 分组 + summary.json
    │
    ▼
SkillManager.run_batch() ──► SKILL.md
    │
    ▼
下一轮 DaAgent 注入 SKILL.md
```

数据在每层之间的载体：agent 输出 `prediction_*.json`（答案+完整轨迹），verifier 把它聚类成分组目录 + `summary.json`，skill_manager 读轨迹 + 摘要产出 `SKILL.md`。RL 训练侧的数据载体是 DataProto（展平的 token batch + uid）。

## 为什么选择「多 agent 角色闭环」+「registry 注册」？

自进化要闭环，就必须把"解题"和"提炼经验"拆成两个角色，否则 agent 一边解题一边改自己的 prompt 会发散。加第三个"验证者"是因为提炼经验需要先知道哪些轨迹是好的——这就是 verifier。registry 注册是工程选择：三种 harness 和两种验证策略如果写死 if-else，每加一个都要改核心循环；注册表让扩展变成"加文件"，核心循环 `loop.py` 只调用注册表。

## 为什么代码执行层要独立？

代码执行是唯一和"外部世界"打交道的层，也是唯一可能把整个训练机搞崩的层。独立出来有两个好处：一是边界清晰——沙箱（内存、超时、只读）全部在这一层，上层 agent 不用关心；二是可替换——RL 训练、评测、DataPRM 验证三个场景都要执行代码，共用 `async_interpreter.py` 而不是各写一套。如果执行器崩了，异常被这一层捕获转成 observation，不会往上冒泡成训练中断。

## 关键取舍

| 取舍 | 选择 | 原因 |
|------|------|------|
| 代码执行位置 | 训练热路径内置（非外挂 sandbox） | 外挂崩了整条 trajectory 废，内置可恢复 |
| advantage 估计 | GRPO 组内归一化（非 PPO/Critic） | 多步 agent 的 value 不好估，GRPO 省显存 |
| 损失目标 | SFT+RL 混合（mu 衰减，非纯 RL） | 纯 RL 冷启动格式崩坏 |
| 过程奖励 | 单独 DataPRM（非主训练内置） | 主训练要效率，PRM 要细粒度，分开各取所需 |
| 验证策略 | self-consistency（agreement，非仅 ground truth） | 真实场景无标签，自洽是可扩展的替代信号 |
| agent 后端 | registry 多 harness（非单一自研） | 同一任务能对比自研/Codex/Claude Code |
| SQL 执行 | 仅 SELECT（非任意 SQL） | 防止破坏性语句删库 |
| 内存隔离 | 1GB RLIMIT_AS（非无限） | 挡失控代码，够常规分析 |
