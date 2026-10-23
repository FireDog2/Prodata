[2024.09 – 2025.06]

项目名称：**ProData 通用数据分析智能体**

项目背景：数据分析智能体的落地难在四件事——数据格式杂（csv/excel/sqlite/json）、要长程多步推理、代码执行不稳定、outcome reward 抓不到"代码不报错但逻辑错"的静默错误。当时 veRL、OpenRLHF 的 PPO 管线都是为单轮问答设计的，把工具调用的多轮交互压成一个 token 序列后，**GRPO** 这类按 trajectory 算 advantage 的算法根本对不上；只给最终答案打 reward 又分不清"蒙对"和"真对"。我从数据、训练、奖励、评测四个断点下手，做了一套全栈配方。

主要职责：

**数据合成（ProData-12K）：** 先按分析动作拆细粒度任务分类，再用递归"由易到难"组合生成新 query——模型先做单步子任务，成功的子任务当素材往上叠复合任务。生成后过模型判断 + 规则过滤两道筛，规则过滤当硬门槛（格式、可执行性、数值校验），最终筛出 **12K 条**轨迹，覆盖多领域、多任务类型、多文件格式。

**RL 训练管线改造：** 在 **verl v0.4.0** 上改出多轮代码 rollout，做三件事——把 SFT loss 和 RL loss 揉进一个目标（**sft_loss_type=mu**，峰值 0.9、谷值 0.05、350 步衰减），让训练早期稳格式、后期放开探索；加 **DAPO filter_groups**（按 seq_final_reward 过滤、最多 10 个 gen batch）丢掉全零 reward 组；重写代码执行器，独立 conda 环境 `darl-run` + **1GB RLIMIT_AS** 内存隔离 + SQL 仅 SELECT + 缺包自动 pip 装。配置 rollout_n=4、max_turns=10、adv_estimator=grpo。

**过程奖励模型（DataPRM）：** 让 **4B** 参数的 PRM 当"主动验证者"——自己写 Python 代码去执行环境验证中间步骤，用 **1/0.5/0 三档**打分区分"可修正的 grounding 错误"和"没救的致命错误"。训练数据用多样性驱动轨迹生成 + 知识增强逐步标注做 **8K+ 条**。结果 Best-of-N 下 ScienceAgentBench **+7.21%**、DABStep **+11.28%**，接入 RL 后 DABench **78.73%**、TableBench **64.84%**。

**自进化多智能体框架（DataCOPE）：** 三角色闭环——da_agent（react/codex/claude_code 三种 harness）解题，verifier 用 self-consistency 或 ground truth 把轨迹分组，skill_manager（claude_code）从有效轨迹提炼可复用 **SKILL.md**，下一轮 da_agent 带着上一轮 skill 再跑。让智能体从自己跑过的轨迹里长经验，而不是靠人写 prompt。

整体成果：ProData-14B 在多个数据分析基准平均 **71.16%**，ProData-7B 达 **68.10%**；配套 LongDS 长程评测基准 **68 任务 / 2225 轮**，覆盖 6 领域、6 种状态演化模式。整条流水线从"没数据"到"有稳定的训练主干"到"有过程奖励"到"能自进化"，每个组件都是被上一个断点逼出来的。
