# ProData 项目技术面试题集

> 基于项目实际代码编写，每题附参考答案。覆盖数据合成、RL 训练、代码执行、过程奖励、自进化框架、Agent Harness、测评、RAG、MCP 九大模块。
> 所有配置值来自 `multi.sh` / `config.yaml`，代码片段来自真实源码（`tensor_helper.py` / `async_interpreter.py` / `loop.py` / `registry.py` / `query_document.py` 等）。

---

## 一、项目基础问题

### Q1. 介绍一下这个项目？

ProData 是一套面向**通用数据分析智能体**的全栈训练配方，解决四个断点：数据、训练、奖励、评测。整个项目是四块拼图：

1. **ProData**（核心）：数据合成 + 训练配方，产出 ProData-12K 轨迹集，ProData-14B 平均 71.16%、ProData-7B 68.10%；
2. **DataPRM**：环境感知的生成式过程奖励模型（4B），Best-of-N 把 DABStep 提 11.28%、ScienceAgentBench 提 7.21%；
3. **DataCOPE**：自进化 Skill 闭环（da_agent → verifier → skill_manager）；
4. **LongDS**：长程多轮评测基准（68 任务 / 2225 轮）。

基座是 Qwen2.5-Coder-7B/14B，SFT 用 LLaMA-Factory（ZeRO-3），RL 在 verl v0.4.0 上改（GRPO/DAPO + 多轮代码 rollout）。一句话定位：**把 RL 用到"动态、工具交互、状态演化"的数据分析场景**，而不是数学/代码的静态推理。

### Q2. 你在项目里具体做了什么？

四块，每块都超出一句"复现"：

- **数据合成**：细粒度任务分类 + 递归由易到难组合 + 模型/规则双过滤，筛出 12K 条高质量轨迹；
- **RL 管线**：`sft_loss_type=mu` 混合损失、DAPO filter_groups、重写代码执行器（`async_interpreter.py`）让多轮 rollout 稳定；
- **过程奖励**：DataPRM 主动写代码验证中间步骤、三档打分（1/0.5/0）抓静默错误；
- **自进化**：da_agent + verifier + skill_manager 三角色闭环，把有效轨迹抽象成 SKILL.md。

### Q3. 为什么要在 verl 上改，而不是直接用？

verl 的 PPO 管线是为**单轮 RLHF** 设计的，多轮工具调用场景有两个对不上：

1. **advantage 是 trajectory 级**：单轮 RLHF 里 advantage 按 token 算，多轮 agent 里 reward 是按整条 trajectory 算的标量，需要组内归一化后再广播回 token；
2. **代码执行的外挂式不稳定**：原版 verl 没有"训练热路径里跑不可信代码"的能力，需要把执行器做成稳定组件（内存隔离 + 超时 + 只读 + 缺包自动装）。

我改了损失目标（`sft_loss_type=mu`）、加了 DAPO 过滤（`filter_groups`）、重写了执行器，让多轮 rollout 能稳定跑。

### Q4. 核心循环是怎么跑的？

以 RL 为例：`multi.sh` 启动 Ray（`ray start --head`），跑 `recipe.dapo.main_dapo`。SGLang 做 rollout 引擎，模型每轮生成 `<code>`（执行代码）或 `<answer>`（给答案），`<code>` 送 `async_interpreter.py` 在独立 conda 环境执行、返回 `<interpreter>` 观察，最多 `max_turns=10` 轮。最后按 trajectory 算 reward（`template 0.1 + answer 0.9`），GRPO 组内归一化算 advantage，DAPO filter_groups 丢全零组，反传更新。评测侧 `eval_python.py` 是同一套 ReAct 循环的简化版，`max_turns=9`。

### Q5. 关键数据结构是什么？

**trajectory 展平后的 batch + uid 映射**。每条 trajectory 长度不同，展平成大 batch；uid 记录每个 token 归属哪条 trajectory，advantage 按 uid 从 trajectory 级标量广播到 token。`tensor_helper.py` 处理三件事：`cut_to_effective_len`（attention_mask 求和找有效长度再裁剪）、`convert_pad_structure`（argsort 实现左填充重排）、`create_position_ids`（cumsum 从 mask 推导 position id）。配合 DAPO 丢组后的 `_example_level_pad` 把非 active example 整条 pad 掉。

### Q6. 诊断系统怎么设计的？

两个层面：

- **训练侧**：SwanLab/WandB 看 reward、loss、entropy 三条曲线。entropy 掉 0 = 坍缩（探索消失），entropy 掉 + reward 涨 = 正常收敛，entropy 掉 + reward 不涨 = 卡局部最优；
- **验证侧**：pass@3（同一题跑 3 次取最好，`eval_python.py` 里 `for index in range(3)`）+ self-consistency（同题多样本答案聚类）。`agreement_verifier.py` 把轨迹按答案聚类写 `summary.json`，skill_manager 根据这个判断哪些轨迹值得学。

### Q7. 项目的技术栈全景是什么？

- **训练**：LLaMA-Factory 0.9.4（SFT，ZeRO-3）+ verl v0.4.0（RL，DAPO/GRPO）+ Ray（调度）+ FSDP（参数分片）+ Ulysses SP（序列并行）+ SGLang（rollout）；
- **执行**：独立 conda 环境（Python 3.10）+ RLIMIT_AS 1GB + subprocess timeout + SQLite 只读 SELECT；
- **推理**：vLLM 0.13.0（DataPRM 部署）+ SGLang（训练 rollout）；
- **数据**：ProData-12K（Parquet/CSV/Excel/SQLite 多格式），`db_schema.json` 存 SQL schema；
- **评测**：DABench / TableBench（Python）、BIRD（SQL）、LongDS（长程）。

---

## 二、核心技术问题

### 2.1 RL 基础

**Q: GRPO 和 PPO 的核心区别？**

PPO 需要 Critic（value 网络）估 advantage `A = r + γV(s') - V(s)`，GRPO 用组内归一化替代 Critic：`A_i = (r_i - mean(r_1..r_N)) / std(r_1..r_N)`。省掉 value 网络（参数 + 梯度 + 优化器 + 一次前向反向），省一半显存。代价是 advantage 方差大、依赖 `rollout_n` 够大。多轮 agent 场景 rollout 贵（每条 trajectory 要多轮代码执行），所以 `rollout_n=4` 是折中。另外 `loss_agg_mode=token-mean` 保证不同长度 trajectory 的每个有效 token 贡献一致。

**Q: GAE 在 agent 多步场景怎么用？**

GAE 要 value bootstrap，多步里每步的 value 不好估（中间状态是代码执行结果/工具返回，不是自然语言，语义边界不清晰）。所以 ProData 主训练用 GRPO 而不是 GAE/PPO，避开 value 估计。真要给中间步做 credit，走的是另一条路——PRM 给中间步打分（DataPRM），而不是 value 网络 bootstrap。

**Q: 训练时 entropy 掉到 0 说明什么？**

policy 坍缩了——所有 rollout 输出趋同、探索消失。排查顺序：①reward 信号是不是太稀疏（全 0 组太多）→ 查执行器和 reward 函数；②`mu` 衰减太快导致格式崩了 → 看格式正确率曲线；③lr 太高把 policy 冲崩。healthy 的 entropy 应该稳定下降但不到 0。注：`multi.sh` 里 `entropy_coeff=0`，entropy 只做监控指标、不进 loss。

**Q: SFT 转 RL 时 loss 突然飙高怎么排查？**

先看格式正确率——`<code>`/`<answer>` 生成率骤降是 `mu` 衰减太快或 KL 失效（本线 KL 全关，靠 clip 和 filter_groups 兜）。再看 reward 分布，是不是某批 trajectory 全 0 reward 导致 advantage 异常（`(r-mean)/std` 里 std 趋 0）。最后看 lr（RL 用 1e-6，比 SFT 的 1e-5 小一个数量级）。

**Q: reward 全零的一组 trajectory，训练里会发生什么？**

组内归一化后 advantage 全 0，这组不贡献梯度，还占显存和算力（甚至 std=0 有数值问题）。DAPO filter_groups 就是把这些组丢掉（`filter_groups_metric=seq_final_reward`）。但丢组会让有效 batch 变小，所以 `gen_prompt_bsz = train_prompt_bsz * 2` 多生成补足。如果 reward 全零比例很高，说明执行器或 reward 函数本身有问题，先修那里。

**Q: 为什么主训练关掉 KL 惩罚？**

DAPO 的实验发现 KL 项会**压制性能**（过度约束探索）。GRPO 本身靠组内归一化 + clip 约束更新幅度，再加 KL 是双重约束。所以 `use_kl_in_reward=False`、`kl_coef=0.0`、`use_kl_loss=False` 全关。DataPRM 接进 RL 这种需要额外稳定性的场景才考虑开 KL。

### 2.2 工具调用与训练策略

**Q: 多轮交互里，怎么把 observation 塞进训练序列？**

每轮 response 后面 append 一个 user 角色（observation），形成 assistant→user→assistant→user 交替。`eval_python.py` 的 `_update_rolling_state` 就是 `prompt.append(assistant_prompt)` + `prompt.append(observation_prompt)`。observation 被截断到 `max_obs_length=2048`，历史 prompt 截断到 `max_start_length=2048`（左截断 `data.truncation='left'`），防止长程对话撑爆序列。

**Q: 代码执行失败（报错）是负 reward 吗？**

不直接是。报错本身只是 observation，模型看到报错可以继续改。reward 只看最终答案对不对（`template 0.1 + answer 0.9`）。报错会消耗 turns，最终没答出来就是 0 reward。这也是为什么不直接给 execution reward（`use_execution_reward=False`）——"跑通了"不等于"答案对"，而且会被静默错误骗。

**Q: 数据过滤会不会误杀有用的 trajectory？**

会。所以过滤分**模型判断（软）+ 规则过滤（硬）**两层。规则过滤只卡硬条件（格式、可执行性、数值校验），**不卡"答案错了"**——因为答案错也可能是好样本（负样本对 RL 有用）。模型过滤负责语义层面的质量判断，但模型也可能误判，所以它是软过滤、可容错。

**Q: invalid action（没匹配到 code/answer 标签）怎么处理？**

`postprocess_prediction` 用正则 `<(code|answer)>(.*?)</\1>` 抽 action，匹配不到就返回 `None`。`execute_code` 里 `None` 走 else 分支，给一段提示"你上一轮 action 无效，请把代码放 `<code>` 里、答案放 `<answer>` 里，重试"，`done=0` 继续循环。这是软纠正——不直接判负，给模型一次救回来的机会。

### 2.3 分布式训练

**Q: Ray 在这个项目里管什么？**

管 rollout worker（SGLang 推理）和 trainer worker（FSDP 训练）的调度，以及它们之间的数据传输。`multi.sh` 里 `ray start --head --num-gpus 8` 启动，`recipe.dapo.main_dapo` 是 Ray 驱动的主程序。RL 的训练范式天然是"生产数据（rollout）→ 消费数据（训练）"的流水线，Ray 正好把这两类异构 worker 组织起来。

**Q: FSDP 和 vLLM/SGLang 怎么协调？**

训练用 FSDP 分片参数，rollout 用 SGLang 推理，**两套显存**。所以 `gpu_memory_utilization` 只给 0.4，给训练侧（actor + ref + 优化器）留空间。rollout 和训练可能在不同 worker 上，通过 Ray 传数据。RL 比 SFT 显存压力大的根本原因：actor（前向+反向+优化器）+ ref（前向）+ rollout（KV cache）三份叠加。

**Q: 多机多卡怎么扩展？**

`multi.sh` 里 `NNODES=16`、`n_gpus=8`，Ray 多机集群。扩展瓶颈不在训练和推理（都能加卡），而在 **rollout 的代码执行**——每轮都要跑代码，16 机时代码执行的并发和文件系统 IO 是新的瓶颈。这是 agent RL 特有的，纯 LLM RL 没有这个约束。

**Q: sp_size 和 fsdp_size 怎么分工？**

`fsdp_size=-1`（全分片）把**参数/优化器状态**切到所有卡，省参数显存；`sp_size=4`（Ulysses）把**单条长序列的激活**切到 4 卡，省激活显存。一个解决"参数多"，一个解决"序列长"（`actor_ppo_max_token_len=10240`），正交互补。

### 2.4 框架对比

**Q: 为什么用 verl 而不是 OpenRLHF / TRL？**

verl 对多轮、工具调用、RLHF 的支持最成熟，trainer 结构清晰好改（`recipe/dapo/main_dapo`、`core_algos.py` 的 loss 都是可插拔的）。OpenRLHF 偏单轮，TRL 的 GRPO 对多步 agent 支持有限。而且 verl 的 DAPO recipe 天然支持 filter_groups，和我要做的多轮代码 rollout 方向一致。

**Q: 为什么 SGLang 做 rollout 而不是 vLLM？**

SGLang 的多轮、structured output、KV cache 布局对多轮 rollout 更友好，`multi.sh` 里 `rollout_engine_name=sglang`。vLLM 用在 DataPRM 的部署和评测服务上（vLLM 0.13.0）。两者定位不同：训练态 rollout 要的是多轮稳定性，部署态要的是吞吐和易用。

### 2.5 工程实践

**Q: 代码执行器里缺包自动 pip install 会不会有安全风险？**

有。pip 装包是执行模型生成的 import 触发的（`execute()` 检测 `No module named` 后 `pip install`），理论上模型可以诱导装恶意包。但在训练沙箱里，环境是隔离的 conda 环境（`darl-run`），单次执行有 1GB 内存（RLIMIT_AS）和超时限制，风险可控。生产环境会去掉自动安装、改成预装白名单。而且 `check_module_in_env` 会先判断模块是否已装，避免重复安装。

**Q: SQL 执行为什么要限制只 SELECT？**

防止模型生成 DROP/UPDATE/DELETE 把评测数据库删了。`is_not_select_query` 用 sqlparse 解析第一条语句，CREATE/INSERT/UPDATE/DELETE 直接拒绝返回 "Only SELECT statements are allowed"。SQL 侧还有额外保护：`exec_on_db` 用 `multiprocessing.Process` + `p.join(timeout)` 硬超时，超时 `terminate()`，因为 SQLite 长查询可能在 C 层卡住，Python 的 subprocess timeout 管不到。

**Q: rollout 超时怎么处理？**

`run_task` 用 subprocess 的 `timeout` 参数，超时返回 "TimeoutError: 'Timed Out'" 作为 observation，让模型知道超时了可以换写法。SQL 侧用 multiprocessing 单独进程跑，`p.join(timeout)` 超时 terminate。还有个 overlong penalty（`overlong_buffer_len=4096`、`overlong_penalty_factor=1.0`）——模型超过 4096 token 还没给 `<answer>` 就扣 reward，逼它及时收敛。

**Q: RLIMIT_AS 限制内存有什么副作用？**

`RLIMIT_AS` 限制的是**虚拟地址空间**，不是物理内存。某些库（numpy）会预留大虚拟空间，可能被误杀。所以 1GB 是经验值——够跑常规分析，但能挡住失控代码（死循环吃内存、恶意分配）。严格来说应该用 cgroup 或容器级内存限制，RLIMIT_AS 是训练态的轻量折中。

---

## 三、Agent Harness 与 Registry 抽象

### Q: 为什么要做 Agent Harness 抽象？不直接写死一种 agent？

因为 DataCOPE 要支持三种 agent 后端：**React（自研文本循环）、Codex、Claude Code**。如果调用方写死 if-else，每加一种 agent 就要改 `da_agent` 的 generate 逻辑。抽象成统一接口后，上层只调 `solve_task`，加新 agent 只需注册 + import，符合开闭原则。

### Q: 这个抽象的具体结构是什么？

三层：

1. **`BaseAgent`（抽象基类）**：`base_agent.py` 定义 `solve_task(prompt, system, **kwargs) -> AgentResult` 抽象方法，`AgentResult` 统一返回 response/conversation/turns/metadata；
2. **`AgentRegistry`（注册表）**：`registry.py` 里 `register_agent(name)` 装饰器 + `AgentRegistry.load(name, **kwargs)` 按名字加载；
3. **三种具体实现**：`react_ds_agent.py` / `codex_agent.py` / `claude_code_agent.py` 都继承 `BaseAgent` 实现 `solve_task`。

```python
@register_agent("my_agent")
class MyAgent(BaseAgent):
    def solve_task(self, prompt, system, **kwargs) -> AgentResult:
        ...
```

调用侧 `da_agent.py` 的 `DaAgent.generate` 对每个 task 调 `agent.solve_task(prompt, system=system_prompt)`，只有 React 需要额外传 `extra_info`（写 trajectory id/index 进文件），其余一致。system prompt 也按 agent 类型分派：`_get_system_prompt` 返回 `REACT_SYSTEM_PROMPT` / `CODEX_SYSTEM_PROMPT` / `CLAUDE_CODE_SYSTEM_PROMPT`。

### Q: 三种 harness 的差异在哪？

- **React**：自研文本循环，自己控制 `max_turns`、`temperature`、`top_p`、`max_tokens`，走 openai backend，代码执行走本地容器 manager（`manager_url`）；
- **Codex / Claude Code**：走各自原生 CLI/SDK，多轮循环和代码执行由它们内部处理，配置项少（只需 model/api_key/base_url）。

抽象层把"循环控制、代码执行、状态管理"这些差异藏起来，上层只关心"给 prompt 拿答案"。

### Q: Registry 相比 if-else 的具体好处？

三个：①**开放封闭**——加新 agent 不改调用方，只 `@register_agent` + import；②**自动发现**——`import src.runtimes.agents` 触发所有 decorator，注册是导入时自动完成的，不会漏；③**统一命名**——`AgentRegistry.load(name)` 按名字字符串加载，配置里写 `agent_type: codex` 就能换，不用改代码。`list_agents()` / `get_agent_class()` 还支持运行期枚举和反射。

### Q: 加一个新 agent（比如 Kimi Code）要改哪些地方？

1. 在 `src/runtimes/agents/` 新建 `kimi_code_agent.py`，继承 `BaseAgent` 实现 `solve_task`，加 `@register_agent("kimi_code")`；
2. 在 `agents/__init__.py` import 它（触发注册）；
3. 加一个 `kimi_code_system_prompt.py` 到 prompts 目录；
4. `da_agent.py` 的 `_get_system_prompt` 加一个分支。

调用侧 `DaAgent.generate` 不用改，`config.yaml` 里 `agent_type: kimi_code` 即可。

---

## 四、Skill 自进化闭环（DataCOPE）

### Q: 自进化 Skill 闭环的整体流程是什么？

三个角色迭代：

1. **da_agent** 解题（第 0 轮不带 skill，第 1 轮起带上一轮的 skill）；
2. **verifier** 挑有效轨迹（`agreement` 按答案聚类算 self-consistency，或 `supervised` 用 ground truth）；
3. **skill_manager** 把有效轨迹抽象成 `SKILL.md`（第 0 轮 `mode="create"`，之后 `mode="modify"`）。

`loop.py` 的主循环：`for iteration in range(iterations+1)`，第 0 轮 `init_run` + create，第 1 轮起 `iterate_run` + modify。skill 从第 i 轮传给第 i+1 轮的 da_agent，形成正反馈——把"一次性轨迹里的经验"蒸馏成"可迁移的 prompt 资产"。

### Q: skill_manager 怎么保证 skill 质量？prompt 里有什么约束？

`skill_manager/prompt.py` 的 `SYSTEM_PROMPT` 有几条硬约束：

1. 必须先读 `skills/skill-creator/SKILL.md`（skill 的格式规范）；
2. skill 要**简洁、500 行内**，去掉冗余/通用/不重要的信息——"每一块内容都要拷问：LLM 真的需要这个吗？"；
3. 不评估、不打包 skill，只创建/修改文件；
4. 生成的 skill 要**通用**，避免私有路径、私有信息、过度个性化描述。

modify 模式的 prompt 更强调"提供的 SKILL 不保证全对，发现和当前数据/轨迹冲突要修正"，避免盲信上一轮 skill。

### Q: SKILL.md 的结构是什么？为什么这样设计？

```yaml
---
name: csv-analysis
description: "Use this skill for CSV data analysis tasks that require..."
---
# CSV Analysis Skill
## Required Workflow
1. Locate the CSV file ...
2. Read the full CSV file ...
...
## Output Expectations
...
```

`name` + `description` 是 YAML 元数据，`description` 决定"什么时候该触发这个 skill"（自动触发靠它）。正文是工作流步骤 + 输出期望。这样设计是因为：skill 不是"知识库"，是**可执行的工作流规范**——它告诉 agent"遇到这类任务该怎么一步步做、怎么验证"，而不是塞一堆事实。

### Q: skill 怎么被 agent 真正用起来？和 RAG 有什么区别？

skill 是**项目级能力注入**：把生成的 `SKILL.md` 放到 `.codex/skills/` 或 `.claude/skills/`，agent 启动时加载，按 `description` 匹配任务自动触发（`project-level-skills-guide.md` 描述了这个机制）。和 RAG 的关键区别：

- **RAG** 是"检索一段相关文本塞进 prompt"（内容检索）；
- **skill** 是"注入一套可复用的流程规范"（能力注入），而且是**自进化生成的**——skill 本身来自 agent 自己跑过的有效轨迹。

两者都算"上下文增强"，但 skill 更强调整套工作流 + 验证步骤的可迁移，且是闭环自我迭代的。

### Q: 断点续跑和增量演化怎么实现？

两个机制：①`stage_success_status.json` 记录每个 stage 的成功状态，配合 `--resume-from-iteration` / `--resume-from-stage` 跳过已完成阶段；②modify 模式先 `shutil.copytree(上一轮 skills → 本轮 skills, ignore=prediction_*.json)` 复制上一轮再改，保证 skill 是**增量演化**、不丢之前的好经验。这样大任务跑崩了不用从头来，skill 也不会每次从零重写。

### Q: verifier 为什么分 init_run 和 iterate_run？

第 0 轮只有当前轮的轨迹，`init_run` 只做"答案聚类 + 写 summary"；第 1 轮起有两轮轨迹，`iterate_run` 对比 prev 和 curr，找出"加了 skill 之后答案发生显著变化"的轨迹——这些变化才是 skill 真正起作用的证据。还有个过滤：prev 和 curr 两轮最高自洽组 SC 都是 1.0 且答案相同的 query 直接跳过，因为"稳定一致"的题没有可提炼的新信息。

---

## 五、数据处理与合成（ProData-12K）

### Q: 数据是怎么合成出来的？为什么不直接用现成数据集？

现成数据分析数据集太少、难度单一、格式单一。ProData 的合成配方（README 四点里的数据部分）：

1. **细粒度任务分类法**：把数据分析任务按类型（聚合、过滤、join、多步推导…）细分类，保证覆盖全面；
2. **递归的 easy-to-hard 任务组合**：用简单任务组合出复杂任务（比如"先过滤再聚合再对比"），提升难度和多样性；
3. **知识增强的轨迹采样**：采样时注入领域知识（schema、业务规则）引导轨迹生成；
4. **模型过滤 + 规则过滤**：模型判断语义质量（软），规则卡硬条件（格式、可执行、数值校验）（硬）。

产出 ProData-12K，覆盖多领域、多任务类别、多文件格式（CSV/Excel/SQLite/JSON）。

### Q: 数据处理里"知识增强"具体指什么？

指在生成轨迹时注入**结构化知识**，而不是让模型盲猜。比如 SQL 任务注入 `db_schema.json`（表结构 + DDL），让轨迹生成时知道有哪些表、字段、外键关系；文档类任务注入 manual.md 的规则说明。DataPRM 的 8K 训练实例也是"知识增强的 step 级标注"——标注每一步对错时让标注者/模型知道数据文件的真实结构，避免标注者自己猜错。

### Q: 规则过滤和模型过滤怎么分工？为什么答案错了不删？

- **规则过滤（硬）**：格式对不对（有没有 `<code>`/`<answer>`）、代码能不能执行、数值校验（数值题在容差内）——这些是客观可判的；
- **模型过滤（软）**：语义层面的质量判断，比如"这条轨迹的推理是否合理"，模型打分，可容错。

答案错了不删是因为：对 RL 来说，**负样本也有价值**（告诉 policy 什么不该做）。删掉所有"答案错"的轨迹会让 RL 只见过正确答案，学不到"避开错误路径"的能力。规则过滤只删"没有训练价值的"（格式崩、不可执行），不删"答案错的"。

### Q: DataCOPE 的任务数据结构长什么样？

`config.yaml` 里的 task 是一个 JSON 列表，每项：

```json
{
  "task_id": "1741",
  "query": "What are the applicable fee IDs for ... in 2023?",
  "data_files": ["acquirer_countries.csv", "fees.json", "manual.md", ...],
  "ground_truth": "36, 51, 53, ...",
  "metadata": {"level": "hard", "category": "Applicable_Fee_IDs"}
}
```

关键字段：`query`（问题）、`data_files`（解题需要的文件，从 `data/data/{task}/` 加载）、`ground_truth`（可选，supervised verifier 用）、`metadata.category`（任务类别，DataCOPE 可**按类别**生成独立 skill）。`category_list` 让 da_agent 分门别类跑，skill_manager 按类别生成技能。

---

## 六、测评体系（Benchmark 与指标）

### Q: 这个项目怎么评测？用了哪些 benchmark？

三类：

1. **Python 分析**：DABench、TableBench（`eval_python.py`），用 pass@3（跑 3 次取最好）+ 模板/答案双 reward；
2. **SQL 分析**：BIRD（`eval_bird.py`），用 `compute_pass3.py` 算 pass@3，gold CSV 结果对比；
3. **长程评测**：LongDS（68 任务 / 2225 轮，Lite 24 / 777 轮），6 领域 6 种状态演化模式，测多轮状态管理。

DataPRM 侧用 DABStep、ScienceAgentBench，指标是 Best-of-N 提升幅度。

### Q: pass@3 是什么？为什么用 pass@3 而不是单次？

pass@3 = 同一题采样 3 次，只要有一次答对就算过（`eval_python.py` 里 `for index in range(3)` 跑三次）。数据分析任务有随机性（采样、代码实现方式），单次评估方差大、会低估模型真实能力。pass@3 降低采样方差，更接近"模型到底会不会"的真实水平。代价是 3 倍推理成本。

### Q: reward/评分函数具体怎么算？

`eval_python.py` 的 `reward_function`：

```python
template_reward = 1.0 if check() else 0.0   # 格式：标签 + 无多余内容
answer_reward = ...                          # judge 模型判答案对错
return template_reward * 0.1 + answer_reward * 0.9
```

- **template_reward**：`check()` 校验格式——最后一条是 assistant 且带 `<answer>`，每轮剥掉 `<think>`/`<code>`/`<answer>` 后内容为空（不能有标签外废话）；
- **answer_reward**：judge 模型（gpt-4o-mini）对比预测答案和 ground truth，数值题 3% 容差、选择题精确匹配。

### Q: 为什么数值题用 3% 容差而不是精确匹配？

数据分析的数值答案（金额、费率、比例）往往有浮点误差、舍入方式差异、单位换算。精确匹配会把这些"本质上对"的答案判错，低估模型。3% 是经验阈值——既容忍合理的数值误差，又不会把"明显算错"的放过。judge prompt 里明确写"abs(Predicted)/abs(True) 与 3% 比较"。

### Q: LongDS 为什么专门做长程评测？和单轮 benchmark 差在哪？

真实数据分析是**连续多轮**，前面的 filter、指标定义、中间表影响后面。单轮 benchmark 每问独立、错了一问不影响下一问；LongDS 设计成"状态依赖"——第 3 轮的状态错误会传导到第 15 轮，误差累积。6 种演化模式（初始状态构建、状态继承、状态更新、反事实扰动、回滚、多状态组合）分别考不同能力：状态继承考"记住上一轮"，反事实扰动考"改假设后重新推导"，回滚考"撤销错误状态"。这样 agent 无法靠单轮的局部正确蒙混过关。

---

## 七、RAG 与知识增强

### Q: 这个项目里用到 RAG 吗？

严格说**没有完整的 RAG 系统**（没有 embedding + 向量索引 + 检索重排那套），但有**知识增强的检索式工具**，思想相通：

- DataPRM 的 `query_document.py` / `query_image.py`：给 agent 提供"读文档/看图片回答问题"的能力，本质是**检索 + 生成**的简化版——读文件内容，让另一个 LLM/VLM 基于内容回答；
- DataCOPE 的 SKILL.md 检索：把自进化生成的 skill 按 `description` 匹配任务注入，是"能力检索"而非"内容检索"；
- SQL 任务的 `db_schema.json`：预注入 schema，避免模型盲猜表结构。

### Q: query_document 工具具体怎么实现的？

```python
def query_document(query: str, file_path: str) -> str:
    content = read_document(file_path)   # utf-8 失败回退 gbk
    # .json/.csv 超长截断到 10000 字符
    prompt = f"基于文档内容回答问题...\n文档：{content}\n问题：{query}"
    response = client.chat.completions.create(model=MODEL_NAME, ...)
    return response.choices[0].message.content
```

读文件（utf-8/gbk 双编码兜底）→ 拼 prompt → 另一个 LLM 回答。`query_image.py` 类似，把图片 base64 编码后走 VLM（默认 Qwen3-VL-235B）。配置通过环境变量 `DOC_OPENAI_API_KEY` / `DOC_MODEL_NAME` / `IMG_*` 注入。这是 DataPRM"主动验证"能力的来源——PRM 不只读文本，还能查文档、看图来验证中间步骤。

### Q: 这种"读文件喂 LLM"和真正的 RAG 差在哪？

真正的 RAG 有 **embedding 检索**（把文档切 chunk、embedding 化、存向量库，query 时做相似度检索只取相关 chunk）+ **重排**（rerank）+ **引用**（citation）。`query_document` 是**全文读入**（小文件直接全塞 prompt，大文件截断），没有向量检索——适合"文件不大、全读比检索更准"的场景。如果要 scale 到超大文档集，就得换成真正的 RAG（分 chunk + 向量检索），否则 prompt 装不下、成本也高。

### Q: 如果要给这个项目加 RAG，加在哪？

三个合理的接入点：

1. **schema/manual 检索**：SQL 任务里 `db_schema.json` 和文档类任务里 manual.md 如果很大，可以 embedding 化做检索，而不是全文注入；
2. **skill 检索升级**：现在 skill 是"按 category 全量注入"，可以改成"按 query 检索最相关的 skill"，减少无关注入、降低 token 成本；
3. **历史轨迹检索**：把有效轨迹向量化，新题来了检索"历史上最相似的题怎么解的"，类似 few-shot 的检索式增强。

---

## 八、MCP 与工具抽象

### Q: 这个项目用 MCP 了吗？和 MCP 什么关系？

**没有直接用 MCP**，但工具抽象的思路和 MCP 一致。项目里的工具（代码执行、SQL 查询、query_document/query_image）是通过**函数 schema 注入 prompt** 的方式暴露给模型的：

```python
# tool_prompt.py
QUERY_DOCUMENT_PROMPT = """
def query_document(query: str, file_path: str) -> str:
    '''Ask a question about the document
    Parameters:
      - query: The question to ask (string)
      - file_path: The path to the document file (...)
    Returns: The query result as a string
    Example:
      result = query_document("What is the calculation formula?", "manual.md")
    '''
"""
```

这个 schema 被 `JUDGE_SYS_PROMPT.format(tools=...)` 塞进 system prompt，模型照着函数签名写代码调用，`interpreter.py` 的 `tool_functions` 参数把真实 Python 函数传进去执行（`StatementRemover` 还会把 `print`/`query_document` 这类"工具调用"和普通代码区分开）。

### Q: MCP（Model Context Protocol）是什么？和项目里的工具抽象有什么异同？

MCP 是 Anthropic 提的**工具/资源标准化协议**——定义一个统一的 client-server 接口，让 LLM 通过标准协议调用外部工具/资源/数据源，而不是每个工具都写死一套。和项目工具抽象的对比：

| 维度 | 项目现状 | MCP |
|------|---------|-----|
| 工具定义 | 函数 schema 字符串注入 prompt | 标准 JSON Schema（tools/list） |
| 调用方式 | 模型写代码、interpreter 执行 | 协议化 tool call |
| 跨模型复用 | 每种 harness 自己适配 | 一次实现，多个 client 通用 |
| 资源/数据源 | 手动读文件/预注入 | 标准 resources 接口 |

**相同点**：都是"把能力抽象成可被模型调用的接口"。**不同点**：项目是"函数签名进 prompt + in-process 执行"的轻量方案，MCP 是"标准化协议 + 独立 server 进程"的工程化方案。

### Q: 如果要把项目迁移到 MCP，怎么迁？

把三个能力包装成 MCP server：

1. **代码执行器**（`async_interpreter.py`）→ 一个 `execute_code` 工具，隔离逻辑（内存/超时/只读）保留在 server 内部；
2. **SQL 查询** → 一个 `query_sql` 工具，只读 SELECT 校验保留；
3. **query_document/query_image** → 两个资源查询工具。

迁移收益：①工具定义标准化，React/Codex/Claude Code 三种 harness 都能直接通过 MCP 调，不用各自适配；②工具可独立部署/升级，不侵入训练进程。代价：多一层进程通信开销，训练热路径上要评估延迟影响。

### Q: 为什么训练热路径上没用 MCP，而是 in-process 执行？

性能 + 稳定性。训练 rollout 是**高频热路径**——每条 trajectory 最多 10 轮、每轮跑代码，batch 并发时上千次调用。MCP 的 client-server 进程通信（JSON-RPC over stdio/SSE）每次调用有额外开销，在训练这种吞吐敏感场景会放大延迟。in-process 的 `subprocess` 直接执行更可控、更快。MCP 更适合**产品化部署**（工具生态、跨 agent 复用），训练态优先考虑吞吐。

---

## 九、综合开放问题

**Q: 被质疑"只是复现跑了一下框架"怎么回答？**

我改了四层东西，不是复现。一是**数据**——自己做数据合成管线筛出 12K 条，不是直接用现成数据；二是**损失**——加了 `sft_loss_type=mu` 混合损失和 DAPO filter_groups，这是 verl 原版没有的；三是**执行**——重写了 `async_interpreter`，把代码执行做成训练热路径的稳定组件（内存隔离/只读/自动装包三层）；四是**奖励**——单独做了 DataPRM 过程奖励模型。任何一层都超出一句"复现"。

**Q: 测完接口性能后，思考过怎么优化吗？**

想过。最大的瓶颈是 rollout 的**代码执行延迟**——每条 trajectory 最多 10 轮、每轮跑代码，代码执行是串行阻塞的。优化方向：①**批量执行**（batch 内多条 trajectory 的代码并行跑，减少进程启动开销）；②**缓存常见的库加载**（pandas/numpy 每次 subprocess 都要重新 import，进程池复用能省掉）；③**进程池复用**而不是每轮 subprocess 起新进程。这是"能跑稳"之后的下一个台阶。

**Q: 最大的技术风险点是什么？**

代码执行的安全性和稳定性。模型生成的代码不可控，即使有内存隔离（RLIMIT_AS 1GB）和超时，极端情况（fork bomb、写爆磁盘）还是可能拖垮训练机。所以我加了 SQL 只读、内存上限、独立 conda 环境三层，但这只是缓解不是根治。生产环境需要更重的**容器级隔离**（每个 rollout 一个容器，cgroup 限制 CPU/内存/IO）。

**Q: 你觉得哪里还可以改进？**

当前 step 之间的 value 没有 bootstrap（GRPO 只到 trajectory 级），中间步骤的好坏被平均。实际可以用 **terminal value function** 或 **PRM 的中间分数**做更细粒度的 credit assignment，让"中间步骤错、答案蒙对"的轨迹被正确惩罚。这是 PRM 接进主训练（不只是 Best-of-N 推理）的下一步。

---

## 十、大厂高频追问

### 10.1 系统设计与架构

**Q: 模型从 4B scale 到 70B，全栈要改什么？**

训练侧：显存翻 N 倍，FSDP 分片粒度、`sp_size`、offload 都要调，可能要换 Megatron（`USE_MEGATRON`）。推理侧：rollout 的 KV cache 和并发重配。执行侧：代码执行不受影响，但 rollout 吞吐成为新瓶颈，多机调度和文件系统要扛住。数据侧：小模型能训的 12K 数据对大模型可能欠拟合，要扩数据。

**Q: 训练中途 worker 崩了怎么恢复？**

`multi.sh` 里 `trainer.resume_mode=auto`，从 checkpoint 恢复。Ray 会重启 worker，但要注意 rollout 和训练的显存状态对齐。崩在 rollout 阶段比崩在训练更新阶段好恢复——因为没改模型权重。DataCOPE 侧用 `stage_success_status.json` + `--resume-from-iteration` 做断点续跑。

**Q: 架构的扩展瓶颈最先在哪？**

rollout 的代码执行。训练和推理都能 scale（加卡），但代码执行是每个 trajectory 都要串行跑代码，rollout 并发上来后代码执行的 CPU/IO 先打满。这是 agent RL 特有的瓶颈，纯 LLM RL 没有。

### 10.2 算法深度

**Q: 多步 agent 场景下 GRPO 和 PPO 的 credit assignment 差异？**

PPO 用 Critic 给每个状态估 value，能按步 credit；GRPO 组内归一化只到 trajectory 级，中间步骤好坏被平均。所以 GRPO 在"最终答案对"的简单任务够用，但"中间步骤错、答案蒙对"的任务需要 PRM 补细粒度 credit。

**Q: 为什么不按 token 序列直接训，要拆成 step？**

agent 的中间状态（代码执行结果、工具返回）不是自然语言，压成 token 序列会破坏语义边界。拆成 step 让每步 observation 独立、可解析，GRPO 的 trajectory 级 reward 才能正确对齐。而且拆 step 后能插入 PRM 的中间奖励。

**Q: KL penalty 放在 loss 里还是 reward 里？**

reward 里（`use_kl_in_reward`）直接扣 reward，loss 里（`kl_loss`）加正则项。ProData 主训练两者都关（`kl_coef=0`、`use_kl_loss=False`），因为 GRPO 靠组内归一化约束，加 KL 压制探索。DataPRM 和需要稳定性的场景才开 KL。

**Q: entropy 掉 50% 是正常还是报警？**

分阶段看。训练早期 entropy 快速下降正常（收敛到好策略）；后期还掉 50% 可能是坍缩——探索没了，卡局部最优。结合 reward 曲线：entropy 掉 + reward 不涨 = 坍缩；entropy 掉 + reward 涨 = 正常收敛。

### 10.3 工程落地

**Q: 训练和在线推理行为不一致（train-test skew）怎么解决？**

多轮 rollout 里，训练时模型能拿到的 observation 和在线时不同（训练环境缺某个库、在线有），会导致模型学到依赖训练环境的错误策略。解决：让训练执行器和在线执行器用同一套环境配置，`async_interpreter.py` 的独立 conda 环境就是为了一致性。

**Q: reward 函数有 bug 的症状和排查？**

症状：reward 分布异常（全 0、全 1、或和答案明显不相关）。排查顺序：①reward 函数对已知答案的样本输出对不对；②格式解析（template_reward）有没有误判；③judge 模型返回格式（`<score>` 标签）有没有解析错。`eval_python.py` 的 `reward_function` 里有大量 log，就是为这个。

### 10.4 对比与前沿

**Q: 和 OpenAI 的 Agent / Anthropic Computer Use 比？**

他们做的是通用 agent + 工具调用的产品化，我做的是数据分析垂直领域的训练配方。共同点：都要处理多轮工具调用和不稳定执行。差异：我在训练侧（RL、PRM、数据合成）下功夫，他们在产品侧（工具生态、安全护栏）下功夫。

**Q: 和 DeepSeek-R1 的关系？**

R1 证明纯 RL 能让推理能力涌现，但它针对数学/代码的静态推理。我的工作是把 RL 用在"动态、工具交互、状态演化"的数据分析场景——多了执行环境、多轮状态、静默错误这些 R1 没碰的复杂度。GRPO 的算法思想来自 R1，但落地工程挑战完全不同。

**Q: 这个方向的前沿在哪？**

三个：①更细粒度的过程奖励（PRM 到 token 级）；②自进化闭环更自动（skill 发现和复用，DataCOPE 已在走）；③多 agent 协作（规划者 + 执行者 + 验证者分工）。

---

## 十一、面试话术与策略

### 被问"这个项目最大的技术难点是什么"

不要讲"改 bug 改了三天"，讲架构级取舍：多轮代码 rollout 的稳定性。两个可选方案——外挂 sandbox（简单但崩了整条废）vs 训练热路径内置稳定执行器（复杂但可控），我选了后者并加内存隔离/只读/自动装包三层。这个选择背后是"执行器稳定性决定 RL 能不能训得动"的判断。

### 被问"你觉得哪里还可以改进"

不要答"都挺好"，讲深度改进方向：当前 step 之间 value 没有 bootstrap（GRPO 只到 trajectory 级），可以用 terminal value function 或 PRM 中间分数做更好的 credit assignment。这是 PRM 接进主训练的下一步。

### 被问"你懂 RAG / MCP 吗"

要能接到项目上（见第七、八节）：项目里没有完整 RAG 但有知识增强工具（query_document/query_image + SKILL 检索），没直接用 MCP 但工具抽象思想一致。回答框架：先讲清楚概念 → 讲项目里的对应物 → 讲如果要上会怎么迁、收益和代价各是什么。这样既显得懂，又不脱离项目。

### 简历要点速查

- ProData-14B 平均 71.16%，ProData-7B 68.10%，ProData-12K 数据
- verl v0.4.0 + DAPO + GRPO + sft_loss_type=mu，max_turns=10、rollout_n=4、gen_bsz=32/train_bsz=16
- 代码执行：1GB RLIMIT_AS + conda 隔离 + SQL 仅 SELECT + 缺包自动装
- DataPRM 4B：主动验证 + 三档打分（1/0.5/0），DABStep +11.28%、DABench 78.73%
- DataCOPE：da_agent → verifier → skill_manager 三角色闭环，SKILL.md 项目级注入
- LongDS：68 任务 / 2225 轮，6 领域 6 状态演化模式
- Harness：BaseAgent + Registry + @register_agent，支持 react/codex/claude_code
- 知识增强：query_document / query_image 工具，db_schema 注入
