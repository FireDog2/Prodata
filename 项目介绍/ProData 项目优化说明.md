# ProData 项目优化说明

## 第一部分：工程优化

### 1. 代码执行内存隔离

**问题**: 训练时模型生成的代码不可控——一次 `np.zeros((10**9, 10**9))` 就能把训练机内存吃满，整个 rollout worker OOM，连带训练中断。这不是假设，是 RL 训练里高频出现的真实情况。

**方案**: `async_interpreter.py` 的 `run_task` 用 `resource.setrlimit(resource.RLIMIT_AS, (1GB, 1GB))` 把单次执行的虚拟地址空间卡在 1GB，超了直接 OOM 报错返回，作为 observation 让模型知道"内存爆了，换个写法"。

**效果**: 单次执行内存有硬上限，失控代码只影响自己这条 trajectory，不拖垮训练进程。

**为什么 1GB 而不是 512MB 或 2GB**: 512MB 太小，常规 pandas 分析（读几个 csv 做 join）就 OOM 了；2GB 太大，起不到隔离作用。1GB 是"够跑常规分析、但挡得住失控代码"的平衡点。

### 2. SQL 只读保护

**问题**: 评测用真实 sqlite 数据库，模型如果生成 `DROP TABLE` 或 `DELETE FROM`，评测库就没了，后续所有评测全废。

**方案**: `SQLInterpreter.is_not_select_query` 用 sqlparse 解析，CREATE/INSERT/UPDATE/DELETE 直接拒绝，返回 "Only SELECT statements are allowed"。

**效果**: 数据库永不被动修改，模型看到错误会重写成 SELECT。

### 3. 缺包自动安装

**问题**: 模型会 `import` 环境里没装的库，导致代码报 `No module named X`，白白浪费一轮交互。

**方案**: `CSVInterpreter.execute` 捕获 "No module named" 错误，自动 `pip install` 后重跑一遍。特殊处理 sklearn（包名是 scikit-learn）。

**效果**: 缺包从"浪费一轮"变成"自动补齐 + 重跑"，rollout 效率提升。

**为什么 sklearn 要特判**: pip 包名是 `scikit-learn` 但 import 名是 `sklearn`，不特判会 `pip install sklearn` 装到错误包。

### 4. 跨轮状态复用

**问题**: 多轮分析里，第 2 轮的代码要引用第 1 轮算出来的变量。如果每轮都从零跑，之前的状态全丢，模型得反复重算。

**方案**: `CSVInterpreter.update_code` 把成功执行的代码里 print 语句换成 pass，保留变量定义。下一轮代码拼上这些定义，变量就能接着用。

**效果**: 跨轮变量状态延续，模型不用重算中间结果，节省轮次。

---

## 第二部分：算法改进

# 数据分析 RL 算法改进

ProData 基于开源项目 [verl](https://github.com/volcengine/verl)（v0.4.0，面向 LLM 的 RLHF 训练库）进行二次开发，在其分布式训练架构上叠加了针对数据分析 agent 的改进。

**ProData 多轮代码 rollout 与开源 verl 的区别**：开源 verl 的多轮 rollout 主要面向对话/工具调用，代码执行靠外挂 sandbox，执行失败会污染整条 trajectory；ProData 在此基础上把代码执行内建为训练热路径的稳定组件——独立 conda 环境 `darl-run` 隔离依赖、1GB RLIMIT_AS 内存上限、SQL 仅 SELECT、缺包自动安装。这些改动在 `agent/async_interpreter.py`，让"执行器崩"从训练中断的常态变成可恢复的异常。

**SFT+RL 混合损失与开源 verl 的区别**：开源 verl 的 PPO/GRPO 是纯 RL 目标，冷启动容易格式崩坏。ProData 引入 `sft_loss_type=mu`（峰值 0.9、谷值 0.05、350 步衰减），把 SFT loss 和 RL loss 揉进一个目标，训练早期稳格式、后期放开探索。这在 `multi.sh` 的 mu 参数里配置，是数据分析 agent 能稳定 RL 训练的关键。

**DAPO filter_groups 与开源 verl 的区别**：开源 verl 的 GRPO 对 reward 全零的组照常归一化，导致 advantage 全 0、白费算力。ProData 开启 `enable_filter_groups=True`、`filter_groups_metric=seq_final_reward`，在算 advantage 之前丢掉全零 reward 组，让组内归一化只发生在有效组上。

## 第三部分：面试怎么讲

**全链路经验。** 这个项目覆盖了数据合成 → SFT → RL → 过程奖励 → 评测的完整链路。能跑通整条链路，说明你不是只会调某个框架的 API，而是理解从数据到模型的每个环节，知道哪里会断、怎么接上。

**每个改动都讲得出 tradeoff。** 比如 mu 为什么峰值 0.9 谷值 0.05（早期稳格式 vs 后期探索）；代码执行为什么 1GB（够用 vs 隔离）；rollout_n 为什么 4（显存/成本 vs advantage 方差）；SQL 为什么仅 SELECT（安全 vs 灵活性）。每个都不是拍脑袋，是权衡后的选择。

**能判断训练正不正常，不只看 loss。** 被问到"怎么判断训练怎么样"，可以答：先看 reward 曲线有没有爬坡、格式正确率（`<code>`/`<answer>` 生成率）有没有在早期就稳定、entropy 有没有在训练后期降到合理区间但没坍缩到 0；如果 reward 全 0 的比例高，先查执行器而不是调超参。
