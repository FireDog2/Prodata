# ProData 项目简历

## 项目简介

ProData 是一套面向"通用数据分析智能体"的全栈训练与评测配方，解决从数据、训练、奖励建模到评测的四个断点。我作为核心开发者，负责其中数据合成、RL 训练管线改造和自进化多智能体框架三块。整个项目用 Qwen2.5-Coder-7B/14B 做基座，LLaMA-Factory 做 SFT、verl v0.4.0 改造成 RL 主干，训练出的 ProData-14B 在多个数据分析基准上平均 71.16%，ProData-7B 也有 68.10%。配套产出的 DataPRM（4B 过程奖励模型）把 ScienceAgentBench 提升 7.21%、DABStep 提升 11.28%，以及 LongDS 长程评测基准（68 个任务 / 2225 轮）。

---

## 功能清单

- **数据合成管线（ProData-12K）**：我做了细粒度任务分类法和"由易到难"的递归任务组合，把合成 query 的多样性和难度都拉起来，最终筛出 12K 条高质量轨迹。合成不是随机生成，是先生成再按"模型判断 + 规则过滤"两道筛。
- **RL 训练主干**：在 verl v0.4.0 上改出多轮代码 rollout 框架，配 DAPO + GRPO，adv_estimator 用 grpo、rollout_n=4、max_turns=10，让单轮问答和工具调用多步共用一套训练代码。
- **SFT+RL 动态混合损失**：用 sft_loss_type=mu 把 SFT loss 和 RL loss 揉进一个训练目标，mu 峰值 0.9、谷值 0.05、350 步衰减，训练后期逐渐从"模仿轨迹"过渡到"自己探索"。
- **代码执行沙箱（async_interpreter）**：多轮 rollout 里模型的每一步 action 都是 Python/SQL 代码，我用独立 conda 环境 `darl-run` 做隔离，1GB 内存上限（RLIMIT_AS），SQL 只放行 SELECT，缺包自动 pip 装。这套把代码执行从"跑一下就崩"变成训练热路径里能稳定跑的东西。
- **DataPRM 过程奖励模型**：4B 参数的生成式 PRM，会主动写代码去执行环境里验证中间步骤，用 1/0.5/0 三档打分区分"可修正的 grounding 错误"和"没救的致命错误"，最终在 DABench 上把 RL 拉到 78.73%、TableBench 64.84%。
- **DataCOPE 自进化多智能体框架**：da_agent 解题 → verifier 用 self-consistency 或 ground truth 挑出有效轨迹 → skill_manager 从轨迹里提炼可复用 SKILL.md，下一轮带着 skill 再跑。三个 agent 角色闭环迭代，让智能体从自己跑过的轨迹里"长"出经验。
- **LongDS 长程评测基准**：68 个任务 / 2225 轮（Lite 版 24 任务 / 777 轮），覆盖 6 个领域、6 种状态演化模式，专门测 agent 在多轮里能不能保持、更新、复用分析状态。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 训练框架 | verl v0.4.0（RL，改造）、LLaMA-Factory 0.9.4.dev0（SFT）、ms-swift 3.11.0.dev0（DataPRM SFT） |
| 基座模型 | Qwen2.5-Coder-7B-Instruct / 14B-Instruct |
| 推理引擎 | SGLang（RL rollout）、vLLM 0.13.0（DataPRM 部署、评测服务） |
| RL 算法 | DAPO（filter_groups + GRPO adv_estimator）、sft_loss_type=mu 混合损失 |
| Agent 抽象 | DaAgent / BaseAgent / AgentRegistry（react、codex、claude_code 三种 harness） |
| 工具格式 | `<code>`/`<answer>` 标签协议、ReAct 循环、SKILL.md |
| 代码执行 | subprocess + conda 隔离（darl-run）、1GB RLIMIT_AS、func_timeout、sqlite3（仅 SELECT） |
| 数据结构 | DataProto、trajectory/uid 展平、attention_mask 驱动的左填充、TensorHelper |
| 评测 | DABench、TableBench、BIRD（SQL）、DABStep、ScienceAgentBench、LongDS |
| 分布式 | Ray、FSDP、DeepSpeed ZeRO-3、ulysses sequence parallel(sp=4) |
| 诊断/监控 | SwanLab / WandB、pass@3、self-consistency 分组 |

---

## 项目简历内容

### 模板 A：AI Agent / 多智能体系统工程师

**[2024.09 – 2025.06] | ProData 通用数据分析智能体 | 核心开发者**

**项目背景：**

数据分析是 LLM 智能体落地最硬的一块：数据文件格式杂（csv/excel/sqlite/json）、要长程多步推理、代码执行不稳定。但当时的训练框架（veRL、OpenRLHF 的 PPO 管线）都是为单轮问答设计的——一个 sample 对应一次 prompt→response→reward。工具调用 agent 完全不同：模型要在环境里连续多轮观察、行动、拿反馈，每步 action 的质量决定后面能不能走到对的 done；而且 GRPO/RLOO 这类要按 trajectory 聚 reward 算 advantage 的算法，如果把多轮压成一个 token 序列就根本对不上。另一个问题是 reward：只看最终答案的 outcome reward 无法区分"中间步骤错但运气好蒙对"和"过程全对只是最后格式错"。我就从这几个断点下手，做了数据合成、训练管线改造、过程奖励模型和自进化框架四件事。

**主要负责：**

**一、数据合成管线（ProData-12K）**

做 RL 第一步就卡在没数据上——公开的数据分析指令数据要么太简单（单步查数），要么同质化严重。我的做法是把任务先按分析动作拆成细粒度分类（聚合、join、分组、窗口、跨文件关联……），再用"由易到难"的递归组合生成新 query：先让模型做单步子任务，把成功的子任务当素材往上叠更难的复合任务。这样难度是可控递进的，不是随机抽。生成之后不是直接进训练集，而是过两道筛：模型判断答案正确性 + 规则过滤（格式、可执行性、答案数值校验）。最后筛出 12K 条轨迹，覆盖多个领域和文件格式。这里有个容易踩的坑——纯靠模型生成会产出大量"看起来对、算起来错"的轨迹，必须把规则过滤当成硬门槛而不是参考。

**二、RL 训练管线改造（verl + 多轮代码 rollout）**

这部分是技术最深的一块。标准 verl 的多轮 rollout 是把每轮 response 都重新拼 prompt 再喂给引擎，中间状态靠字符串拼，代码执行靠外挂 sandbox，一轮崩了整条 trajectory 就废。我做了三件事：一是把 SFT loss 和 RL loss 揉进一个目标（sft_loss_type=mu，峰值 0.9 谷值 0.05、350 步衰减），让训练早期先稳住格式、后期再放开探索，避免 RL 一开始就 reward 崩；二是 DAPO 的 filter_groups（按 seq_final_reward 过滤、最多 10 个 gen batch），把全零 reward 的组直接丢掉，防止 GRPO 的 advantage 全是 0；三是重写了代码执行器（`agent/async_interpreter.py`），代码跑在独立 conda 环境里、设 1GB 内存上限、SQL 只放行 SELECT、缺包自动 pip 装，把"执行器崩"从训练中断的常态降成了可恢复的异常。关键区别是：别人的 rollout 把执行当外挂工具，我把执行做进了训练热路径的稳定组件。

**三、过程奖励模型（DataPRM）**

outcome reward 只能给最终答案打分，过程奖励模型（PRM）能给每一步打分，但通用领域的 PRM 在数据分析上基本失效——它抓不到"静默错误"（逻辑错了但代码不报错，跑出来结果还像模像样），还会把必要的试错当成失败乱扣分。我的解法是让 PRM 变成"主动验证者"：不是被动看文本，而是自己写 Python 代码去执行环境里跑一遍中间状态，用 1/0.5/0 三档打分，0.5 专门留给"可修正的 grounding 错误"（方向对、细节错了），把"还能救"和"没救"区分开。训练数据也是我用多样性驱动的轨迹生成 + 知识增强的逐步标注做出来的，8K+ 条。结果是用 4B 参数就把 Best-of-N 推理下 ScienceAgentBench 提了 7.21%、DABStep 提了 11.28%，把它接进 RL 后 DABench 到 78.73%、TableBench 到 64.84%。

**四、自进化多智能体框架（DataCOPE）**

这块最贴合"多智能体系统"。核心是一个三角色闭环：da_agent（react/codex/claude_code 三种 harness）在任务集上解题，verifier 用 self-consistency 或 ground truth 把轨迹分成"有效/无效"两组（agreement 模式按答案聚类算 self-consistency，supervised 模式按 ground truth 分 success/fail），skill_manager（claude_code）从有效轨迹里提炼可复用的 SKILL.md，下一轮迭代 da_agent 带着上一轮的 skill 再跑。整个循环由 `core/loop.py` 的 `run()` 驱动，iterations 参数控制迭代次数，每次迭代产出新 skill 供下一轮用。这样 agent 不是靠人写 prompt 变强，而是从自己跑过的轨迹里"长"出经验。

**项目成果：**

- 任务覆盖：ProData-14B 在多个数据分析基准上平均得分 71.16%，ProData-7B 达 68.10%
- 数据规模：合成并筛出 ProData-12K（12K 条高质量轨迹），覆盖多领域、多任务类型、多文件格式
- 奖励建模：DataPRM 4B 参数，Best-of-N 下 ScienceAgentBench +7.21%、DABStep +11.28%；接入 RL 后 DABench 78.73%、TableBench 64.84%
- 训练稳定性：代码执行从"外挂工具"改造为训练热路径稳定组件——1GB 内存隔离、SQL 仅 SELECT、缺包自动安装，多轮 rollout 不再因执行器崩溃中断
- 算法支持：DAPO（filter_groups）+ GRPO + SFT/RL 混合损失（mu），让单轮问答和工具调用多步共用一套训练主干
- 自进化机制：da_agent → verifier → skill_manager 三角色闭环，agent 从轨迹中自动提炼 SKILL.md，无需人工写 prompt
- 评测体系：LongDS 长程基准 68 任务 / 2225 轮，覆盖 6 领域、6 种状态演化模式

**技术架构：**

```
用户任务(question + 数据文件)
        │
        ▼
┌─────────────────────────────────────────────┐
│  DaAgent (da_agent/da_agent.py)             │  ← 三种 harness: react / codex / claude_code
│  生成 response → 解析 <code>/<answer>        │
└──────────────────┬──────────────────────────┘
                   │ action(code/answer)
                   ▼
┌─────────────────────────────────────────────┐
│  代码执行器 (async_interpreter.py)           │  ← 独立 conda 环境 darl-run
│  Python: 1GB RLIMIT_AS + 缺包自动pip        │     SQL: sqlite3 仅 SELECT
│  返回 <interpreter> 观察                    │
└──────────────────┬──────────────────────────┘
                   │ observation
                   ▼
┌─────────────────────────────────────────────┐
│  Verifier (verifier/instances/*.py)         │  ← agreement(self-consistency) / supervised(ground truth)
│  轨迹分组 → 有效/无效                       │
└──────────────────┬──────────────────────────┘
                   │ 有效轨迹 + 分组摘要
                   ▼
┌─────────────────────────────────────────────┐
│  SkillManager (skill_manager/*.py)          │  ← claude_code 提炼 SKILL.md
│  create(iter0) / modify(iter1+)             │
└──────────────────┬──────────────────────────┘
                   │ SKILL.md（下一轮注入 da_agent）
                   ▼
            下一轮迭代（loop.py run()）
```

---

## STAR 法则完整示例

### S（背景，15%）

标准 verl 的 PPO 训练管线是为单轮 RLHF 设计的——一个 sample 对应一次 prompt→response→reward。但数据分析 agent 完全不同：模型要在工具/环境里连续多轮观察、行动、拿反馈，每步 action 的质量决定后面能不能走到对的 done。而且数据分析任务里大量存在"静默错误"——代码不报错、结果看着也对，但逻辑是错的，outcome reward 根本发现不了。如果把多轮压成一个 token 序列丢给 trainer，GRPO 这类需要按 trajectory 做组内对比的算法就对不上 advantage；如果只给最终答案打 reward，又分不清"蒙对"和"真对"。

### T（任务，15%）

1. 让单轮问答和工具调用多步共用一套训练代码，N 类任务共用同一套训练主干
2. 把 SFT 和 RL 揉进一个训练目标，训练早期稳格式、后期放开探索
3. 把代码执行做成训练热路径里的稳定组件，而不是外挂沙箱
4. 给数据分析做一个能抓静默错误、能区分"可救/没救"的过程奖励模型
5. 让智能体能从自己跑过的轨迹里自进化，而不是靠人写 prompt

### A（行动，50%）

**数据合成。** 我先按分析动作拆细粒度任务分类，再用递归"由易到难"组合生成新 query——先让模型做单步子任务，成功的子任务当素材往上叠复合任务。生成后过两道筛：模型判断 + 规则过滤，规则过滤当硬门槛（格式、可执行性、数值校验），不是参考。筛出 12K 条。

**RL 训练管线。** 我试过直接套标准 verl 多轮 rollout，结果代码执行一崩整条 trajectory 就废，GRPO 的 advantage 经常全 0。最后我做了三处改动：sft_loss_type=mu 混合损失（峰值 0.9 谷值 0.05、350 步衰减）；DAPO filter_groups（按 seq_final_reward 过滤、最多 10 个 gen batch，丢掉全零 reward 组）；重写 `agent/async_interpreter.py` 把代码执行放进独立 conda 环境，1GB 内存上限、SQL 只放行 SELECT、缺包自动 pip 装。在 `agent/tensor_helper.py` 里用 attention_mask 驱动有效长度裁剪和左填充，处理多轮 trajectory 展平成 batch 后的 padding。

**过程奖励模型。** 我让 DataPRM 主动写 Python 代码去执行环境里验证每一步，用 1/0.5/0 三档打分，0.5 专门留给可修正的 grounding 错误。训练数据用多样性驱动轨迹生成 + 知识增强逐步标注做 8K+ 条。

**自进化框架。** 用 `core/loop.py` 的 `run()` 驱动三角色闭环：da_agent 解题 → verifier 分组 → skill_manager 提炼 SKILL.md，iterations 参数控制迭代，下一轮 da_agent 带着上一轮 skill 再跑。

### R（结果，20%）

- 任务覆盖：ProData-14B 平均 71.16%、ProData-7B 平均 68.10%
- 数据规模：ProData-12K（12K 条高质量轨迹）
- 奖励建模：DataPRM 4B，ScienceAgentBench +7.21%、DABStep +11.28%；接入 RL 后 DABench 78.73%、TableBench 64.84%
- 训练稳定性：代码执行改造后多轮 rollout 不再因执行器崩溃中断
- 算法支持：DAPO + GRPO + SFT/RL 混合损失，单轮和多步共用一套主干
- 自进化机制：三角色闭环从轨迹提炼 SKILL.md，无需人工写 prompt
- 评测体系：LongDS 68 任务 / 2225 轮，6 领域 6 种状态演化模式

---

## 简历要点（可复制）

1. 设计并实现 ProData 数据分析智能体全栈训练配方（数据合成 + SFT + RL），ProData-14B 在多个数据分析基准平均得分 71.16%，ProData-7B 达 68.10%
2. 在 verl v0.4.0 上实现多轮代码 rollout 框架，通过 sft_loss_type=mu 混合损失（SFT+RL）+ DAPO filter_groups + GRPO，让单轮问答与工具调用多步共用一套训练主干，max_turns=10、rollout_n=4
3. 重写代码执行沙箱（async_interpreter），独立 conda 环境 + 1GB 内存隔离 + SQL 仅 SELECT + 缺包自动安装，将执行器崩溃从训练中断常态降为可恢复异常
4. 设计 DataPRM 环境感知过程奖励模型（4B），主动写代码验证中间步骤 + 1/0.5/0 三档打分，Best-of-N 下 ScienceAgentBench +7.21%、DABStep +11.28%，接入 RL 后 DABench 78.73%
5. 构建 DataCOPE 自进化多智能体框架（da_agent + verifier + skill_manager 三角色闭环），agent 从自身轨迹自动提炼可复用 SKILL.md，无需人工写 prompt
6. 构建 LongDS 长程评测基准，68 任务 / 2225 轮，覆盖 6 领域、6 种状态演化模式

---

## 面试常见追问与回答策略

**Q1: 这个项目为什么要分这么多组件？数据合成、训练、PRM、框架、评测，是不是各做各的？**

**策略**：面试官在探你有没有全局视角，能不能把四个组件串成一条因果链。别回答"分工不同"，要回答"每个组件解决流水线里一个具体断点"。

**A**：这四个不是并列，是一条流水线上的四个断点。最开始卡在没数据，所以做数据合成；有数据了发现标准 RL 管线跑不了工具调用多步，所以改训练主干；训练出来发现 outcome reward 抓不到静默错误，所以做过程奖励模型 DataPRM；最后发现怎么验证 agent 在长程里能不能保持状态，所以做 LongDS。每一块都是被上一块逼出来的。比如我一开始只想用 outcome reward，结果发现数学题能行、数据分析不行——因为分析题有大量"代码不报错但逻辑错"的静默错误，PRM 能写代码去验证中间状态，这是 outcome reward 做不到的。

**Q2: 为什么 trajectory 要展平成 batch 再按 uid 聚合，不能直接按 trajectory 训练吗？**

**策略**：考你对 GRPO/DAPO advantage 计算的理解。核心是 token 级 loss 和 trajectory 级 reward 的对应关系。

**A**：GRPO 的 advantage 是按 trajectory（组）算的——同一个 prompt 采样 rollout_n 条，组内归一化后每条拿到一个标量 advantage，这个标量要广播到这条 trajectory 的每个 token 上算 token-level loss。如果直接按 trajectory 训练，每条 trajectory 长度不同、没法拼成规整 batch，GPU 算起来全是 padding 浪费。展平 + 记录 uid 的做法是把所有 trajectory 拼成一个大 batch，同时保留每条 token 属于哪条 trajectory 的映射，advantage 按 uid 广播。`tensor_helper.py` 里 attention_mask 驱动的有效长度裁剪和左填充就是干这个的。

**Q3: sft_loss_type=mu 这个混合损失，为什么要让 mu 从 0.9 衰减到 0.05？直接固定一个比例不行吗？**

**策略**：考你是否理解 RL 冷启动的稳定性问题。核心是训练早期的格式崩坏。

**A**：RL 早期如果完全放开探索，reward 信号很稀疏，模型很快就把 `<code>`/`<answer>` 格式忘掉，输出一堆没法解析的东西，reward 直接崩。mu 高的时候 SFT loss 占主导，逼模型维持模仿轨迹的格式；训练推进后 mu 降到 0.05，让模型逐渐从"模仿"过渡到"自己探索"。这个 350 步的衰减是我调的——太短没稳住格式，太长又抑制了 RL 的探索收益。

**Q4: 代码执行为什么用独立 conda 环境 + 1GB 内存上限，而不是直接 subprocess 跑？**

**策略**：考你工程上的隔离和稳定性设计。核心是"恶意/失控代码不能拖垮训练"。

**A**：训练时模型生成的代码是不可信的，可能死循环、吃满内存、import 一堆乱七八糟的包污染环境。我用 `resource.setrlimit(resource.RLIMIT_AS, (1GB, 1GB))` 把单次执行的内存上限卡死，超了直接 OOM 报错返回，不拖垮训练进程；独立 conda 环境 `darl-run` 隔离依赖，避免模型代码装的包污染训练环境；SQL 只放行 SELECT，防止模型执行 DROP/UPDATE 把评测库删了。缺包自动 pip 装是处理"模型 import 了没装的库"这个高频情况，装完重跑一遍。

**Q5: 最大的技术难点是什么？（debug story）**

**策略**：不要讲"改 bug 改了三天"，要讲一个需要权衡取舍的架构决策，最好有"我试了 X 不行最后用 Y"的过程。

**A**：最难的是让多轮代码 rollout 在训练热路径里稳定。我最开始直接套标准 verl 的多轮 rollout，把代码执行当外挂 sandbox，结果一轮代码崩了整条 trajectory 就废，GRPO 的 advantage 经常全 0，训练卡着不动。我一开始以为是 reward 函数的问题，加了各种 logging 排查，最后发现是执行器崩导致大量 trajectory 拿不到有效 reward，组内归一化后 advantage 全是 0。解决是把执行做成训练热路径里的稳定组件——独立环境 + 内存隔离 + 只 SELECT + 缺包自动装，再加上 DAPO filter_groups 把全零 reward 组直接丢掉。这两条一起才把训练跑稳。

**Q6: 如果要支持第 N+1 类任务，需要改哪些地方？**

**策略**：考你对扩展点的理解。要说出具体的注册机制和抽象边界。

**A**：看是加什么。加数据格式或领域任务：DataCOPE 那边新增一个 dataset loader（`task_processor/loaders/` 里加 `@register_dataset` 的类）就行，agent 和 verifier 都不用动；加新的 agent harness：`runtimes/agents/` 里加 `@register_agent` 的类，实现 `solve_task` 接口；加新的验证策略：`verifier/instances/` 里加 `@register_verifier`。这些都是注册表模式，`registry.py` 里已经有 React、Codex、Claude Code、supervised、agreement 这些实例，照着加就行。真正要动核心的，是加一个新的 RL 算法——那要改 advantage 计算和 trainer，是另一层的事。

---

## 简历避坑指南

### 错误示范 vs 正确示范

| 错误示范 | 正确示范 | 问题 |
|----------|----------|------|
| "使用了先进的多智能体协作技术" | "三角色闭环：da_agent 解题 → verifier 用 self-consistency 分组 → skill_manager 提炼 SKILL.md" | 没有具体角色名和机制 |
| "负责训练相关工作，显著提升了模型性能" | "在 verl v0.4.0 上改出多轮代码 rollout，ProData-14B 平均 71.16%" | 动词弱，没有量化成果 |
| "我们团队开发了一套数据合成管线" | "我做了细粒度任务分类 + 递归由易到难组合，筛出 12K 条轨迹" | 面试官想知道你个人做了什么 |
| "底层采用先进的过程奖励模型" | "DataPRM 4B 主动写代码验证中间步骤，DABStep +11.28%" | 没有具体算法和数字 |
| "系统优化了代码执行稳定性" | "1GB RLIMIT_AS 内存隔离 + SQL 仅 SELECT + 缺包自动安装" | "优化了"是空话，要写具体机制 |

### 四条原则

1. **动词开头**：设计、实现、改造、构建、重写（不用"参与"、"负责"、"协助"）
2. **数字优先**：71.16%、+11.28%、12K 条、4B 参数（不用"显著"、"大幅"等模糊词）
3. **技术具体**：写"GRPO 组内归一化"、"sft_loss_type=mu"、"RLIMIT_AS"、"self-consistency"；不写"先进算法"、"智能优化"
4. **突出个人**：用"我设计"、"我实现"、"我重写"；不用"我们"、"团队"、"项目完成了"

---

## 不同岗位的简历侧重

| 岗位方向 | 重点突出 | 关键词 | 弱化 |
|----------|----------|--------|------|
| AI Agent / 多智能体系统 | DataCOPE 三角色闭环、自进化机制、agent harness 抽象、工具调用格式 | agent loop、verifier、skill、self-consistency、registry | RL 的 loss 细节、advantage 公式 |
| LLM 训练 / RL 工程师 | verl 改造、DAPO+GRPO、sft_loss_type=mu 混合损失、advantage 计算、多轮 rollout | verl、DAPO、GRPO、advantage、rollout、FSDP | 评测基准的领域细节 |
| 后端 / 平台工程 | 代码执行沙箱、隔离、超时、并发、可观测性 | subprocess、RLIMIT_AS、sqlite 只读、ThreadPoolExecutor | 数据合成方法论 |
| 数据分析 / MLE | ProData-12K 数据合成、reward 建模、benchmark | 任务分类、递归合成、静默错误、pass@3 | 多智能体框架的抽象层 |
