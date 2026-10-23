# ProData 代码讲解 —— 按执行流顺序

> 面向初学者，中文讲解。按代码执行流的顺序，从数据到训练到评测，拆开 ProData 的核心模块。每个模块说清楚三件事：这文件是干嘛的、关键代码长什么样、面试容易被问到哪。
> 所有路径都在 `ProData/` 目录下，代码片段取自真实源码。

**目录**

1. 数据合成（ProData-12K）
2. DataCOPE 任务数据结构
3. SFT 训练配置（LLaMA-Factory）
4. RL 训练入口（multi.sh）
5. DAPO + GRPO 主程序与损失（main_dapo / core_algos.py）
6. 多轮 rollout（sglang multiturn）
7. 代码执行器（async_interpreter.py）
8. 张量辅助（tensor_helper.py）
9. 评测主循环（eval_python.py）
10. Reward 函数（eval_python.py）
11. pass@3 与评测汇总（compute_pass3.py）
12. LongDS 评测（runners）
13. DataPRM 打分 prompt（prompt.py）
14. DataPRM 代码解释器（interpreter.py）
15. 知识增强工具（query_document / query_image）
16. 工具声明与 MCP（tool_prompt.py）
17. DataCOPE 主循环（loop.py）
18. Agent Harness 基类与注册表（base_agent.py + registry.py）
19. React harness（react_ds_agent.py）
20. 验证器（agreement_verifier.py）
21. Skill 管理器（skill_manager.py）

---

## 1. 数据合成（ProData-12K）

**文件**: `prodata/` 数据管线（README 描述，代码在 train 数据目录）

数据分析这个领域缺数据，现成数据集又少、难度又单一。ProData 的做法是合成：先按任务类型做一个细粒度的分类法（聚合、过滤、join、多步推导……分得很细），然后「由易到难」递归组合——简单子任务跑通了，就把它当素材往上叠，拼出更复杂的任务。这样难度是可控递进的，而不是随机生成一堆「看着对、一算就错」的样本。

合成出来的轨迹再走两道过滤：规则过滤是硬门槛（格式对不对、代码能不能执行、数值在不在容差内），模型判断是软门槛（推理合不合理）。最后筛出 12K 条，就是 ProData-12K，RL 的原料。

面试问「为什么规则过滤当硬门槛」——因为纯模型生成会产出大量格式对但答案错的样本，规则过滤只卡可执行的客观条件，不卡「答案错了」。答案错对 RL 反而是有用的负样本。

---

## 2. DataCOPE 任务数据结构

**文件**: `datacope/general/config.yaml` 里 `task`/`dataset` 指向的 JSON

DataCOPE 跑的任务就是一个 JSON 列表，每条长这样：

```json
{
  "task_id": "1741",
  "query": "What are the applicable fee IDs for Belles_cookbook_store in 2023?",
  "data_files": ["acquirer_countries.csv", "fees.json", "manual.md", "payments.csv"],
  "ground_truth": "36, 51, 53, ...",
  "metadata": {"level": "hard", "category": "Applicable_Fee_IDs"}
}
```

`query` 是问题，`data_files` 是解题需要的文件（从 `data/data/{task}/` 加载），`ground_truth` 可选（supervised verifier 才用），`metadata.category` 是任务类别。有 category 的时候 DataCOPE 可以按类别各生成一份 skill，`config.yaml` 的 `category_list` 就是控制这个的。

面试点在于 `ground_truth` 为什么可选——因为真实场景没有标签，agreement verifier 靠答案聚类算 self-consistency，不依赖 ground truth。这个设计让自进化是可扩展的。

---

## 3. SFT 训练配置（LLaMA-Factory）

**文件**: `prodata/train/SFT/LLaMA-Factory/examples/train_full/prodata_12k_full_sft.yaml`

```yaml
model_name_or_path: Qwen/Qwen2.5-Coder-7B-Instruct
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
dataset: prodata_12k
cutoff_len: 8192
learning_rate: 1.0e-5
num_train_epochs: 3.0
bf16: true
```

全参数微调，不是 LoRA。理由是数据分析 agent 要学的行为变化太大——格式（`<code>`/`<answer>` 协议）、工具调用、多步推理，这些都是 LoRA 低秩约束塞不下的。ZeRO-3 把优化器状态和梯度分片，7B/14B 全参才训得动。

SFT 的 lr 是 1e-5，后面 RL 是 1e-6，差了一个数量级，这个对比面试里常被拿来问「为什么 RL 要更小」。

---

## 4. RL 训练入口（multi.sh）

**文件**: `prodata/train/rl/verl/multi.sh`

这个脚本干一件事：把所有超参数集中定义好，启动 Ray，然后跑 `python3 -m recipe.dapo.main_dapo`，参数通过命令行传进去。关键几行：

```bash
adv_estimator=grpo
rollout_n=4
n_sample=1                 # GRPO 组大小
max_turns=10

sft_loss_type=mu
mu_peak=0.9
mu_valley=0.05
mu_decay_steps=350

enable_filter_groups=True
filter_groups_metric=seq_final_reward
gen_prompt_bsz=$((train_prompt_bsz * 2))   # 生成量翻倍，喂给 filter_groups

sp_size=4
fsdp_size=-1
gpu_memory_utilization=0.4
```

几个数字背后是权衡：`rollout_n=4` 是显存和方差的折中（多轮代码执行太贵，翻倍就是翻倍成本）；`mu` 三个值定义了 SFT 损失占比从 0.9 衰减到 0.05 的曲线；`gen_prompt_bsz` 翻倍是因为 filter_groups 会丢全零组，得多生成一些才能凑够训练 batch。

面试大概率问 `mu_peak=0.9` 到 `mu_valley=0.05` 为什么——早期 90% 靠 SFT 稳格式，后期 5% 放开探索，中间线性衰减。

---

## 5. DAPO + GRPO 主程序与损失

**文件**: `prodata/train/rl/verl/recipe/dapo/main_dapo.py`（编排）+ `verl/trainer/ppo/core_algos.py`（损失）

main_dapo 是编排层：读数据 → rollout 采样 → 算 reward → filter_groups 过滤 → GRPO 算 advantage → 反传更新。真正的损失在 `core_algos.py` 的 `compute_policy_loss`：

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

pg_losses3 = -advantages * clip_ratio_c            # 负 advantage 侧的常数上界
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

这里对应 `multi.sh` 里的 `clip_ratio_low=0.2`、`clip_ratio_high=0.28`、`clip_ratio_c=10.0`。三个 clip 各管各的：低 0.2 高 0.28 是「clip-higher」，让模型敢在好方向上多走一点；`clip_ratio_c=10.0` 是负 advantage 侧（惩罚侧）的一个硬上限，防止某条 rollout 的 ratio 爆炸、一巴掌把 policy 打飞。

filter_groups 和 GRPO 归一化的先后顺序是常考点：先过滤再归一化。全零组要是不先丢掉，会拉低 mean/std，污染其他组的 advantage。

---

## 6. 多轮 rollout（sglang multiturn）

**文件**: `prodata/train/rl/verl/examples/sglang_multiturn/`

多轮交互由 SGLang 引擎驱动，`multi.sh` 里对应的开关是：

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns}
+actor_rollout_ref.rollout.multi_turn.tool_config_path=".../sandbox_fusion_tool_config.yaml"
```

每轮循环是：模型生成 → 解析 action（`<code>` 还是 `<answer>`）→ 执行 → 返回 observation → 再生成，直到出 `<answer>` 或者撞到 max_turns。

序列结构的关键在 observation 怎么塞：每轮 response 后面 append 一个 user 角色当 observation，形成 assistant → user → assistant → user 交替。这就是多轮 agent 和单轮问答在数据结构上的本质区别。observation 用 `max_obs_length=2048` 截断，历史 prompt 用 `max_start_length=2048` 左截断（`data.truncation='left'`），防止对话越长越膨胀。

---

## 7. 代码执行器（async_interpreter.py）

**文件**: `prodata/train/rl/verl/agent/async_interpreter.py`

训练时模型生成的代码不可信，所以要隔离。`run_task` 是最核心的一段：

```python
def run_task(code, timeout):
    def limit_memory():
        import resource
        mem = 1024 * 1024 * 1024  # 1GB
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(running_code)
        tmp_file_path = tmp_file.name

    result = subprocess.run(
        ["/root/anaconda3/bin/conda", "run", "-n", "darl-run", "python", tmp_file_path],
        capture_output=True, preexec_fn=limit_memory, timeout=timeout
    )
```

四层隔离：独立 conda 环境（`darl-run`，Python 3.10）、内存上限（RLIMIT_AS 1GB）、超时（subprocess timeout）、临时文件（finally 里删）。`execute()` 里还做了一件事——检测到 `No module named` 就自动 pip install 缺的包再跑一次，这样训练不因为环境缺包就整个卡住。

SQL 侧是另一个执行器 `SQLInterpreter`，`is_not_select_query` 用 sqlparse 只放行 SELECT，CREATE/INSERT/UPDATE/DELETE 直接拒。SQL 执行还用 `multiprocessing.Process` + `p.join(timeout)` 做硬超时，因为 SQLite 长查询可能卡在 C 层，Python 的 subprocess timeout 管不到。

这里有个容易被忽略的细节：`update_code` 会把成功执行的代码里 print 语句换成 pass、保留变量定义。这样下一轮代码能接着用上一轮的变量（状态跨轮延续），又不会重复打印上一轮的输出。

---

## 8. 张量辅助（tensor_helper.py）

**文件**: `prodata/train/rl/verl/agent/tensor_helper.py`

多轮 trajectory 长度不一，要拼成规整 batch，就得做展平和裁剪。这个文件集中处理这些脏活：

```python
def cut_to_effective_len(self, tensor_dict, keys, cut_left=True):
    effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
    for key in keys:
        result[key] = tensor_dict[key][:, -effective_len:]
    return result

def create_position_ids(self, attention_mask):
    return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
```

`cut_to_effective_len` 用 attention_mask 求和找有效长度，统一裁到那个长度；`create_position_ids` 用 cumsum 从 mask 推导 position id，保证 padding 位置归零、有效 token 连续编号。

还有一个 `convert_pad_structure` 用了 argsort 的技巧：把「是否 padding」转成 0/1 掩码，`argsort(stable=True)` 把 padding 排前面、有效 token 排后面，就实现了左填充的重排，同时返回 indices 方便反向映射。

面试问「为什么按 max 而不是每条独立长度」——batch 得是规整矩形张量，统一到最大有效长度，短的靠 attention_mask 忽略多余的 padding。

---

## 9. 评测主循环（eval_python.py）

**文件**: `prodata/eval/python/eval_python.py`

评测走的是同一套 ReAct 循环，只是简化了。核心在 `run_llm_loop`：

```python
for step in range(self.config.max_turns):   # max_turns=9
    gen_output = self.respond(inputs, temperature, top_p)
    response_str = self._postprocess_responses(gen_output)
    next_ob, done = self.execute_code(response_str)
    if done: break
    inputs = self._update_rolling_state(inputs, response_str, next_ob)
```

温度 0.7、top_p 0.95。`execute_code` 里：action 是 `code` 就执行并返回 `<interpreter>` 观察，是 `answer` 就 done=1 结束，其他都算 invalid 给一段「请重试」的提示。`_postprocess_responses` 会截断到第一个 `</code>` 或 `</answer>`，防止模型一次吐多个标签。

面试问「评测温度为什么是 0.7 不是 0」——pass@3 要多样性，跑三次取最好，温度 0 三次输出一样，pass@3 就没意义了。

---

## 10. Reward 函数（eval_python.py）

**文件**: `prodata/eval/python/eval_python.py`

reward 拆成两块：

```python
template_reward = 1.0 if check() else 0.0   # 格式对不对
answer_reward = ...                          # 答案对不对（judge 模型判）
return template_reward * 0.1 + answer_reward * 0.9, ...
```

`check()` 校验的是格式：最后一条得是 assistant 且带 `<answer>`，每个 assistant 轮次把 `<think>`/`<code>`/`<answer>` 剥掉后内容必须为空——也就是「除了标签和思考，不能有多余废话」。`answer_reward` 用 judge 模型（gpt-4o-mini）对比预测和 ground truth，数值题 3% 容差内算对，选择题精确匹配。

权重 0.1 和 0.9 对应 `multi.sh` 里的 `answer_ratio=0.9`、`template_ratio=0.1`。template_reward 单独给分是为了让模型始终守住 `<code>`/`<answer>` 协议，格式崩了 reward 就全废了。

还有个点：`use_execution_reward=False`、`execution_ratio=0`——故意不开 execution reward。因为「代码跑通了」不等于「答案对」，而且会被静默错误骗。

---

## 11. pass@3 与评测汇总（compute_pass3.py）

**文件**: `prodata/eval/python/compute_pass3.py`

pass@3 就是同一题跑三次、三次里只要对一次就算过。实现很直白：

```python
def compute_pass3(eval_results_list):
    correct_id_set = set()
    for eval_results in eval_results_list:      # 3 份结果
        for item in eval_results:
            if item["answer_score"] == 1.0:
                correct_id_set.add(item["id"])
    return len(correct_id_set) / len(eval_results_list[0])
```

三次结果的正确 id 取并集，除以总题数。`compute_avg` 是另一个口径——所有三次的总正确数除以 (3 × 总题数)，算的是平均单次准确率。两个口径分开报，面试时能区分「模型到底会不会」和「单次抽样准不准」。

数据分析任务有采样随机性（代码写法、数据顺序），单次评估方差大、会低估真实能力，pass@3 把这个方差压下去，代价是三倍推理成本。

---

## 12. LongDS 评测（runners）

**文件**: `longds/runners/{codex,claude_code,kimi_code,qoder_cli,DSGym}/`

LongDS 是长程多轮评测基准，68 个任务 2225 轮（Lite 版 24 任务 777 轮），六个领域、六种状态演化模式。runner 就是跑评测的驱动：

```bash
python runners/codex/run_codex_longds.py --use-docker --run-parallel 4 --judge
```

每个任务一个独立 Docker 容器，跨轮保持同一个 agent session（不重置），四任务并发跑，每轮默认超时 3600 秒。最后 `summary.json` 里 `task_avg_score` 是任务均分（0-1 区间），judge 用独立端点打分。

面试点在于为什么长程比单轮难：单轮错了就错了，长程里第 3 轮的状态错误会传导到第 15 轮，误差累积。所以任务要设计成「状态依赖」的，让 agent 没法靠单轮的局部正确蒙混过关。

---

## 13. DataPRM 打分 prompt（prompt.py）

**文件**: `dataprm/evaluate/prompt.py`

DataPRM 是「环境感知的生成式 PRM」，prompt 定义了三档打分和完整的验证流程：

```python
JUDGE_SYS_PROMPT = """...
- 1 表示该段落正确合理，分析和代码准确，有助于解题
- 0.5 表示整体正确但有细节遗漏或小错误，可在后续步骤修正
- 0 表示存在致命分析错误或代码错误，导致结果错误或误导
...
# Verification Format
1. '<reasoning></reasoning>' 先分析正确性，'<code></code>' 再写代码验证
4. System will execute the code and return results in <interpreter>...</interpreter>
   ... 若足以判断，给出 <score></score> 和 <summary></summary>
"""
```

流程是 `reasoning`（分析怀疑点）→ `code`（写验证代码）→ `<interpreter>`（系统执行返回结果）→ 反复直到能下结论 → `score` → `summary`。这就是「主动验证」的含义：PRM 不靠语言先验猜，而是真的跑数据文件验证中间变量。

三档（1/0.5/0）而不是二档，是为了给「方向对、细节错」的 grounding 错误留个 0.5 的位置。这类错误下一步能救回来，如果粗暴判 0，会让 policy 不敢试错——这正好是 DataPRM 要解决的两大问题之一。

---

## 14. DataPRM 代码解释器（interpreter.py）

**文件**: `dataprm/evaluate/interpreter.py`

这个解释器和训练侧的执行器不同，它多了「工具函数注入」和「AST 清理」两件事。`run_task` 会把工具函数代码拼在用户代码前面一起执行：

```python
def run_task(code, workspace, timeout, tool_functions=None):
    if tool_functions is not None:
        running_code = "\n".join(tool_functions) + "\n\n" + "\n".join(code)
    else:
        running_code = "\n".join(code)
    ...
    result = subprocess.run([os.sys.executable, tmp_file_path], cwd=workspace, ...)
```

`StatementRemover` 用 ast 把 `print`、`query_document` 这类「工具调用/展示语句」从成功执行后的代码里摘掉，只保留变量定义，实现跨步状态复用：

```python
class StatementRemover(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id in self.functions_to_remove:   # ['print', 'query_document']
                return ast.Pass()
        return self.generic_visit(node)
```

用 AST 而不是正则，是为了精确地删掉「独立成句的 print 调用」，又不动 `df.print_something()` 这种带 receiver 的调用，也不动字符串里的 print。

---

## 15. 知识增强工具（query_document / query_image）

**文件**: `dataprm/evaluate/tools/query_document.py` + `query_image.py`

这俩是 DataPRM「主动验证」的能力来源——PRM 不只读文本，还能查文档、看图片来验证中间步骤。`query_document` 的核心：

```python
def query_document(query: str, file_path: str) -> str:
    content = read_document(file_path)   # utf-8 失败回退 gbk
    # .json/.csv 超长截断到 10000 字符
    prompt = f"基于文档回答问题...\n文档：{content}\n问题：{query}"
    response = client.chat.completions.create(model=MODEL_NAME, ...)
    return response.choices[0].message.content
```

读文件 → 拼 prompt → 另一个 LLM 回答。`query_image` 类似，图片 base64 编码后走 VLM（默认 Qwen3-VL-235B）。配置靠环境变量 `DOC_OPENAI_API_KEY` / `DOC_MODEL_NAME` / `IMG_*` 注入。

严格说这不是完整 RAG——没有 embedding + 向量索引，是小文件全文读入。面试可能问「这和 RAG 差在哪」：RAG 有 chunk 切分 + 向量检索 + 重排，适合超大文档集；这里是全文塞 prompt，适合「文件不大、全读比检索更准」的场景。

---

## 16. 工具声明与 MCP（tool_prompt.py）

**文件**: `dataprm/evaluate/tools/tool_prompt.py`

项目里的工具是通过「函数签名字符串注入 prompt」的方式暴露给模型的：

```python
QUERY_DOCUMENT_PROMPT = """
def query_document(query: str, file_path: str) -> str:
    '''
    Ask a question about the document
    Parameters:
      - query: The question to ask (string)
      - file_path: The path to the document file (...)
    Returns: The query result as a string
    Example:
      result = query_document("What is the calculation formula?", "manual.md")
    '''
"""
```

这个 schema 被 `JUDGE_SYS_PROMPT.format(tools=...)` 塞进 system prompt，模型照着函数签名写代码调用，真实 Python 函数由 `interpreter.py` 的 `tool_functions` 参数传进去执行。

这套东西和 MCP 的思路一样（把能力抽象成可调用的接口），但不是 MCP——MCP 是标准化的 client-server 协议（JSON Schema 定义工具、独立 server 进程、跨模型复用），这里只是「函数签名进 prompt + in-process 执行」。面试如果问「用没用 MCP」，可以这样答：没直接用，但工具抽象思想一致；训练热路径上选 in-process 是为了吞吐，MCP 的进程通信开销在这种高频场景放大了。

---

## 17. DataCOPE 主循环（loop.py）

**文件**: `datacope/general/src/core/loop.py`

自进化闭环的驱动，三个 stage 串起来：

```python
for iteration in range(iterations + 1):
    stage_use_skill = iteration > 0          # 第 0 轮不带 skill，第 1 轮起带
    previous_skill_output_dir = (None if iteration == 0
                                 else skill_manager_output_dir_prefix / f"iter_{iteration-1}" / "skills")

    run_da_stage(...)                        # da_agent 解题
    verifier_results = run_verifier_stage(...)  # verifier 分组（init_run / iterate_run）
    run_skill_manager_stage(...,
        mode="create" if iteration == 0 else "modify", ...)
```

第 0 轮是 create（从零生成 skill），之后是 modify（在已有 skill 上改）。modify 时先 `shutil.copytree` 把上一轮的 skill 复制过来再改（`ignore=prediction_*.json`），保证增量演化、不丢已学到的经验。

还有断点续跑：`stage_success_status.json` 记录每个 stage 的成功状态，`--resume-from-iteration` / `--resume-from-stage` 跳过已完成阶段。大任务跑崩了不用从头来。

面试点在于为什么 iteration 0 用 create、之后用 modify——首次从零生成，后续在已有 skill 上增量改，避免重新生成丢掉学到的经验。

---

## 18. Agent Harness 基类与注册表

**文件**: `datacope/general/src/runtimes/base_agent.py` + `registry.py`

要支持多种 agent 后端（React 自研、Codex、Claude Code），得抽象统一接口。`base_agent.py` 定义抽象基类：

```python
class BaseAgent(ABC):
    @abstractmethod
    def solve_task(self, prompt, system, **kwargs) -> AgentResult:
        """Solve a given task..."""
        pass
```

`registry.py` 用装饰器 + 类字典做注册：

```python
class AgentRegistry:
    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def load(cls, name: str, **kwargs) -> BaseAgent:
        if name.lower() not in cls._agents:
            raise ValueError(f"Agent '{name}' not found. Available: {list(cls._agents.keys())}")
        return cls._agents[name.lower()](**kwargs)

def register_agent(name: str):
    def decorator(cls):
        AgentRegistry.register(name, cls)
        return cls
    return decorator
```

上层 `da_agent.py` 的 `DaAgent.generate` 只调 `agent.solve_task(...)`，具体是哪种 agent 由 `AgentRegistry.load(agent_type, ...)` 决定。加新 agent 就三步：继承 BaseAgent 实现 solve_task、加 `@register_agent("xxx")`、在 `__init__.py` import 触发注册。调用方一行不用改。

面试问「Registry 相比 if-else 的好处」——开放封闭原则，加新 agent 不改调用方；而且 `import src.runtimes.agents` 触发所有 decorator，注册是导入时自动完成的，不会漏。

---

## 19. React harness（react_ds_agent.py）

**文件**: `datacope/general/src/runtimes/agents/react_ds_agent.py`

三种 harness 里自研的那个，ReAct 循环在这里实现。`solve_task` 的核心：

```python
@register_agent("react")
class ReActDSAgent(BaseAgent):
    def solve_task(self, prompt, system="", **kwargs):
        conversation = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]
        env = AllocatedCodeEnv(manager_url=self.manager_url, ...)
        conversation, _ = env.init(conversation, **extras)

        for turn in range(self.max_turns):
            response, reasoning_response, token_usage = self.backend_instance.generate(conversation, **self.generate_kwargs)
            step_output = env.step(response)                    # 解析 + 执行 + 返回 observation
            conversation.append({"role": "assistant", "content": step_output['postprocessed_action']})
            if step_output['observations']:
                conversation.extend(step_output['observations'])
            if step_output['done']:
                final_answer = step_output['metadata'].get('final_answer', response)
                break
```

几个点：`AllocatedCodeEnv` 是从容器 manager 分配的执行环境，`backend_instance.generate` 走 litellm/vllm/sglang 后端，`env.step` 负责解析 action、跑代码、返回 observation。异常处理里还有个恢复机制——某轮挂了会 append 一条 `Error: ... Please try a different approach.` 让模型换思路，而不是直接放弃。

和 Codex/Claude Code 的区别：React 自己控制 max_turns、temperature、top_p、代码执行走本地容器 manager；Codex/Claude Code 走原生 CLI/SDK，循环和执行都在它们内部。

---

## 20. 验证器（agreement_verifier.py）

**文件**: `datacope/general/src/verifier/instances/agreement_verifier.py`

verifier 的作用是「挑出值得学的轨迹」。agreement 模式按答案聚类：

```python
def _group_answers(self, preds):
    groups = []
    repr_answers = []
    for pred in preds:
        placed = False
        for i, rep in enumerate(repr_answers):
            if question_scorer(str(pred["answer"]), str(rep)):
                groups[i].append(pred); placed = True; break
        if not placed:
            groups.append([pred]); repr_answers.append(pred["answer"])
    groups.sort(key=lambda g: -len(g))
    return groups
```

聚类后 self-consistency = 组大小 / 总样本数，写进 `summary.json`。每组只留一条代表轨迹（`_best_file` 选 turns 最少的那条，也就是「最省步数到答案」的解法）。

`iterate_run` 对比前后两轮，多一个过滤：如果某个 query 两轮最高自洽组的 SC 都是 1.0 且答案相同，直接跳过——「稳定一致」的题没有可提炼的新信息。

面试点：为什么选 turns 最少的代表轨迹——更短的轨迹更高效、更接近最优解，作为 skill 素材质量更高。以及 self-consistency 为什么不等于正确性（模型可能系统性犯同一个错，自洽度 100% 但全错）。

---

## 21. Skill 管理器（skill_manager.py）

**文件**: `datacope/general/src/skill_manager/skill_manager.py`

最后一个 stage，把有效轨迹提炼成 SKILL.md。`create_skill` 和 `modify_skill` 结构一样，只是 prompt 不同：

```python
def create_skill(self, category, traj_dir, verifier_prompt=""):
    prompt = CATEGORY_SKILL_GENERATE_PROMPT.format(
        category=category, task=self.task, traj_dir=traj_dir,
        skill_dir=skill_dir, data_dir=self.data_dir, verifier_prompt=verifier_prompt)
    return self.agent.solve_task(prompt, system=SYSTEM_PROMPT)
```

skill_manager 自己也是一个 agent（`AgentRegistry.load(agent_type, ...)`），默认用 Claude Code 当 backend。`run_batch` 按 category 并行处理（`ThreadPoolExecutor`，`parallel_workers` 控制并发）。

面试点：为什么 skill_manager 用 claude_code 而不是 react——提炼经验需要强代码理解和长上下文，Claude Code 的 agent 能力比自研 React 循环强，这里没必要自己造轮子。还有 prompt 里的约束：skill 要简洁、500 行内，「每一块内容都要拷问 LLM 真的需要这个吗」，且要通用（避免私有路径、过度个性化）。
