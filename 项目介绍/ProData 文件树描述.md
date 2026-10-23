# ProData 文件树

## 完整目录树

```
ProData/
├── prodata/                          ⭐⭐⭐ 主训练与评测（数据合成→SFT→RL→评测）
│   ├── README.md                     项目总说明
│   ├── train/
│   │   ├── sft/LLaMA-Factory/       ⭐⭐ SFT（Qwen2.5-Coder，ZeRO-3）
│   │   │   └── examples/train_full/
│   │   │       ├── prodata_12k_full_sft.yaml       SFT 主配置
│   │   │       └── prodata_da_dataset_full_sft.yaml
│   │   └── rl/verl/                 ⭐⭐⭐ RL（verl v0.4.0 改造）
│   │       ├── multi.sh              训练入口脚本（所有超参）
│   │       ├── agent/
│   │       │   ├── async_interpreter.py  ⭐⭐⭐ 代码执行沙箱
│   │       │   └── tensor_helper.py      ⭐⭐ 张量裁剪/填充
│   │       └── recipe/dapo/main_dapo.py  ⭐⭐⭐ DAPO 主程序
│   └── eval/
│       ├── python/eval_python.py     ⭐⭐⭐ Python 评测（DABench/TableBench）
│       ├── python/interpreter.py     ⭐⭐ 评测侧执行器
│       ├── sql/eval_bird.py          ⭐⭐ SQL 评测（BIRD）
│       └── compute_pass3.py          ⭐⭐ pass@3 计算
├── dataprm/                          ⭐⭐⭐ 过程奖励模型
│   ├── evaluate/
│   │   ├── run_eval.py               ⭐⭐⭐ DataPRM 评测主程序
│   │   ├── prompt.py                 ⭐⭐ 三档打分 prompt
│   │   ├── llm.py                    ⭐⭐ LLM 封装（执行环境交互）
│   │   └── tools/                    ⭐ 文档/图像查询工具
│   └── ms-swift/                     ⭐⭐ DataPRM SFT 训练
├── longds/                           ⭐⭐⭐ 长程评测基准
│   ├── README.md                     LongDS 说明（68任务/2225轮）
│   └── runners/                      ⭐⭐ 各 agent 原生 runner
│       ├── codex/ claude_code/ kimi_code/ qoder_cli/
│       ├── DSGym/                    ⭐⭐ DSGym executor
│       └── src/judge.py              ⭐⭐ 评测打分
├── datacope/                         ⭐⭐⭐ 自进化框架
│   ├── general/                      ⭐⭐⭐ DataCOPE 主框架
│   │   ├── config.yaml               运行配置
│   │   ├── start.sh                  启动脚本
│   │   └── src/
│   │       ├── core/loop.py          ⭐⭐⭐ 主循环（三角色编排）
│   │       ├── core/config.py        ⭐⭐ CLI + 配置加载
│   │       ├── core/schema.py        ⭐⭐ 数据结构定义
│   │       ├── da_agent/da_agent.py  ⭐⭐⭐ 解题 agent
│   │       ├── da_agent/prompts/     ⭐⭐ 三种 harness 的 system prompt
│   │       ├── verifier/instances/   ⭐⭐⭐ 验证器（agreement/supervised）
│   │       ├── skill_manager/        ⭐⭐⭐ skill 生成器
│   │       ├── runtimes/agents/      ⭐⭐ 三种 harness 实现
│   │       └── runtimes/executors/   ⭐⭐ 容器执行（Docker/manager）
│   ├── reason_task/                  ⭐⭐ 推理类任务（DABStep）
│   └── report_task/                  ⭐⭐ 报告类任务（DDR_Bench）
└── prodata-analysis/                 ⭐ 分析实验（经验洞察）
```

## 模块关系图

```
                    ┌─────────────────────────┐
                    │  prodata（训练+评测）      │
                    │  multi.sh → main_dapo    │
                    │    ├─ async_interpreter  │
                    │    └─ tensor_helper      │
                    └──────────┬──────────────┘
                               │ 提供训练出的 agent + 评测方法
                               ▼
     ┌─────────────────────────────────────────────────┐
     │  dataprm（过程奖励）── 接入 RL / Best-of-N        │
     │  datacope（自进化）── 用 agent 轨迹提炼 skill      │
     │  longds（长程评测）── 评测 agent 状态管理          │
     └─────────────────────────────────────────────────┘
```

四者的依赖关系：prodata 是主干（训练 + 评测），dataprm 给训练提供过程奖励信号，datacope 用 agent 轨迹做自进化（skill 提炼），longds 是更难的评测（长程状态）。

## 关键文件详解

### async_interpreter.py（⭐⭐⭐）

- `run_task(code, timeout)`：subprocess 跑代码，RLIMIT_AS 1GB 内存限制
- `CSVInterpreter`：Python 执行，含 `extra_code`（抽代码块）、`execute`（跑 + 缺包自动装）、`update_code`（状态复用）
- `SQLInterpreter`：SQL 执行，`is_not_select_query`（仅 SELECT）、`replace_cur_year`（YEAR(CURDATE())→2025）、`exec_on_db`（多进程 + 超时）

### loop.py（⭐⭐⭐）

- `run()`：主循环，驱动 iterations 次 da_agent → verifier → skill_manager
- `run_da_stage()`：调用 DaAgent 解题，支持 sample_nums 多样本
- `run_verifier_stage()`：加载验证器（supervised/agreement）
- `run_skill_manager_stage()`：加载 skill_manager，create/modify
- `main()`：CLI 入口，解析 config.yaml + 命令行

### eval_python.py（⭐⭐⭐）

- `LLM.run_llm_loop()`：多轮主循环（max_turns）
- `execute_code()`：解析 `<code>`/`<answer>` 并执行
- `reward_function()`：template_reward(0.1) + answer_reward(0.9)
- `get_final_answer()`：单题求解入口

### main_dapo.py（⭐⭐⭐）

- DAPO + GRPO 训练编排：rollout → reward → filter_groups → advantage → 更新

## 代码调用链追踪

```
multi.sh
  └─ ray start --head
  └─ python3 -m recipe.dapo.main_dapo (主程序)
       └─ rollout（SGLang 多轮）
            └─ async_interpreter.run_code()      # 执行 <code>
                 └─ CSVInterpreter.execute()
                      └─ run_task() → subprocess.run(conda run darl-run python)
       └─ reward 计算（template + answer）
       └─ filter_groups（丢全零 reward 组）
       └─ GRPO advantage（组内归一化）
       └─ 反传更新（FSDP）

datacope/general/start.sh
  └─ python src/core/loop.py → main()
       └─ run()
            ├─ run_da_stage → DaAgent.generate → solve_task
            ├─ run_verifier_stage → VerifierRegistry.load → init_run/iterate_run
            └─ run_skill_manager_stage → SkillManager.run_batch → create/modify
```

## 扩展指南

- **添加新的数据领域任务**：在 `task_processor/loaders/` 加 `@register_dataset` 的类，配好数据文件即可
- **添加新的 agent harness**：在 `runtimes/agents/` 加 `@register_agent` 的类，实现 `solve_task`
- **添加新的验证策略**：在 `verifier/instances/` 加 `@register_verifier` 的类
- **添加新的 RL 算法**：改 advantage 计算 + trainer（需动核心，成本最高）
- **添加新的评测基准**：在 eval 侧加评测脚本，复用 `reward_function` 和 `interpreter`
