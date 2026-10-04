# Ref2Dex Autonomous Research Agent

本仓库采用“自主探索、少量人工决策、正式验证后再形成结论”的科研工作方式。

核心原则：

> Optimize for research progress, not procedural completeness.

AI 的目标不是把每个可能的问题都研究完整，而是在资源和时间有限的情况下，持续获得能够改变研究决策的信息，并向 `MISSION.md` 中定义的最终目标推进。

---

## 1. 默认工作模式：自主探索

除非触发本文件定义的 Decision Checkpoint，否则默认允许 AI 自主：

* 阅读代码与研究记录；
* 修改当前仓库中的代码、配置和测试；
* 创建和切换 `agent/*` 分支；
* 编写诊断脚本；
* 执行 smoke test；
* 运行小规模训练、评估和离线分析；
* 根据结果修复 bug、调整实现并继续尝试；
* 放弃明显失败的局部方案；
* 设计下一轮最小实验。

不需要因为普通研究变量、网络结构、reward、Cm 接法或实现细节变化而逐次请求用户批准。

研究探索本身允许失败。

默认行为应是：

> 在安全且可逆的情况下继续推进，而不是因为存在不确定性而停止。

---

## 2. 启动时只读取最小上下文

每次开始新任务时默认读取：

1. `AGENTS.md`
2. `docs/MISSION.md`
3. `docs/CAMPAIGN.md`
4. `docs/README.md`
5. `docs/STATE.md`
6. `docs/experiments/INDEX.md`
7. 与当前问题直接相关的代码和活跃实验记录

研究路线迁移或规范审计需要读取 `docs/ref.md`（如果存在）；它是设计输入，不替代本文件
和当前状态。`docs/research/README.md` 是目录说明；历史 plan、指导、activity、decision、log
和 handoff 只有在需要核对证据时才追溯，不属于默认上下文。


---

## 3. 每个实验必须服务于一个决策

在运行一个非纯工程 smoke 的实验前，AI 必须回答：

1. 这个实验想区分什么假设？
2. 实验结果会改变哪个下一步决策？
3. 获取该信息的最便宜方法是什么？

然后将实验分类为：

### Blocker

不解决就无法继续主路线。

处理方式：立即解决。

### Decision

结果 A 与结果 B 会导致不同的下一步路线。

处理方式：优先执行最小可判别实验。

### Evidence

无论结果如何，当前近期路线都不会改变，但未来论文、复现或完整论证需要。

处理方式：写入对应 experiment card 的 limitations/future evidence，当前不做。

### Curiosity

只是增加理解，并不会明显影响路线或最终 claim。

处理方式：默认不做。

“实验科学上有意义”不是运行它的充分条件。

---

## 4. Probe 优先，Validation 延后

探索阶段默认运行 Probe。

Probe 的目的不是证明结论，而是判断：

> 这个方向是否值得继续投入？

Probe 应尽量满足：

* 单 seed 或极少 seed；
* 少量环境；
* 小数据；
* 短训练；
* 小规模离线分析；
* 尽可能低的 GPU 和时间成本。

只有满足以下条件之一时才进入 Validation：

* Probe 出现明显正向信号；
* 即将基于某结果改变核心架构；
* 即将形成正式科研结论；
* 结果需要进入论文或对外汇报；
* 两条路线需要正式比较后才能选定。

Validation 才要求 matched control、多 seed、固定 metric、预先定义的判定条件和完整证据。

不要把 Validation 的严谨程度施加到每一个探索想法上。

### 4.1 默认优先使用 GPU

神经网络训练、微调、重复模型推理和批量模型评估默认优先使用 GPU；支持 GPU
加速的仿真采集也优先使用 GPU。离线实验同样适用，不因“Probe”“小规模”或
历史 CPU-only 阶段而默认改用 CPU。

设备选择须针对当前任务重新判断。纯文件处理、标签审计、统计计算，以及 GPU
启动成本明显超过收益的极小 smoke 可以使用 CPU；模型计算改用 CPU 时，在
实验卡或运行记录中写明具体原因。具体选择和资源边界遵循
[`CAMPAIGN.md`](docs/CAMPAIGN.md) 的 GPU execution policy 与 GPU budget。

---

## 5. Decision Checkpoint

以下情况必须在继续前，在对应 experiment card 或 `docs/STATE.md` 中记录简短 Decision Note，
并重新核对证据、资源和安全边界：

1. 准备改变 `MISSION.md` 中的核心研究问题；
2. 准备放弃一个核心研究假设；
3. 准备改变最终需要证明的 claim；
4. 两条路线都合理，而任一路线后续成本都明显较高；
5. 需要突破 `CAMPAIGN.md` 中的 GPU、时间、磁盘或权限限制；
6. 操作不可逆或可能影响当前仓库之外的数据、checkpoint、进程或环境；
7. 连续三个有效 Probe 都没有使 North-star scoreboard 获得进展，也没有排除重要路线；
8. 准备把探索性观察升级为正式科学结论。

Decision Note 必须简短，只包含：

* 当前需要决定的问题；
* 当前最关键证据；
* root 选择的行动及其理由；
* 预计成本、成功/失败后的下一步和停止条件；
* 仍需外部授权的边界（如果存在）。

在 `MISSION.md`、`CAMPAIGN.md` 和当前用户授权允许的范围内，root 自主选择并执行
记录过的行动，不要求用户在预设的 Option A/Option B 之间插入选择。只有改变核心研究
问题或 claim、突破资源/权限边界、执行不可逆外部操作，或需要新的长期身份时，才暂停
并向用户请求相应授权。

不要提交长篇计划让用户审批。

---


---

## 7. Git 规则

`main` 保存已经值得长期保留的状态。

新研究路线默认使用：

`agent/<short-description>`

例如：

`agent/cmv2-online-adapt`

`agent/cm-action-ranking`

分支代表“实现路线”，不是单个超参数实验。

不要因为：

* seed 改变；
* reward coefficient 改变；
* epoch 改变；
* 一次评估；

而新建分支。

同一实现路线的 Probe 可以在同一分支完成。

当一条路线产生值得保留的代码、工具或证据后再合并回 `main`。

除非用户明确要求，否则不要向远程仓库 push。

push的时候可以尝试设置代理

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897



不要 reset、覆盖或删除用户已有修改。

---

## 8. 身份与版本

代码版本、实验和研究阶段必须使用不同的身份，不要用同一个连续版本号同时表示三者。

### 8.1 Code Identity

具体实现由 Git 管理。

使用：

* branch：表示一条实现或研究路线；
* git commit：表示具体代码状态。

例如：

`agent/cm-action-ranking`

以及：

`git commit: <sha>`

不要为了记录一次实验而人为创建新的代码版本号。

---

### 8.2 Experiment Identity

实验使用独立的 experiment ID。

探索性 Probe：

`P-<YYYYMMDD>-<short-topic>`

例如：

`P-20260923-cm-action-ranking`

正式 Validation：

`VAL-<YYYYMMDD>-<short-topic>`

例如：

`VAL-20260924-cm-ranking-online-ab`

不同 seed、repeat、checkpoint 或运行实例通常不创建新的 experiment ID，而由 experiment 内部的 `run_id` 区分。

因此：

* `experiment_id` 表示“研究问题/实验设计”；
* `run_id` 表示“该实验的一次具体执行”。


---

## 9. 科研结论边界

工程 smoke 只能证明：

* 代码能运行；
* tensor 有限；
* checkpoint 能加载；
* wiring 正确。

Probe 只能形成：

* `PROMISING`
* `UNPROMISING`
* `UNCLEAR`

正式 Validation 才允许形成：

* `SUPPORTED`
* `REFUTED`
* `INCONCLUSIVE`
* `INVALID_IMPLEMENTATION`

不要把：

* 单 seed；
* 单 rollout；
* 训练 loss；
* 单次成功视频；
* 工程 smoke；

描述成正式科学结论。

---

## 10. 保护边界

必须始终遵守 `docs/CAMPAIGN.md` 中的机器和资源约束。

默认禁止：

* sudo；
* 系统级修改；
* 杀死未知或他人的进程；
* 修改授权工作区之外的项目；
* 覆盖已有 checkpoint；
* 删除无法确认归属的数据；
* 无界生成 cache、视频和 checkpoint。

外部数据和代码如果被标记为 read-only，只能读取或软链接。

---


## 11. 延后证据

对于：

* 未来论文需要；
* 当前不改变决策；
* 但最终不能永久遗漏；

的实验，记录在对应 experiment card 的 limitations/future evidence 中。

延后证据记录意味着：

> 这个实验没有被忘记，只是当前不值得占用探索预算。

不要为了“严谨”而立即执行所有延后实验。

---

## 12. 关于tmp

不要把任何输出放到/tmp，如果有必要，放在项目目录下的tmp，如果没有这个目录就新建。

---


## 13. Task 与研究路线目录

新的研究方向必须在 `src/task/<TaskName>/` 下建立稳定的 Task 目录。Task 表示一条长期
可复用的研究问题或代码边界，不按 seed、epoch、checkpoint 或单个超参数创建新 Task。
Git branch 仍表示具体实现路线，例如 `agent/cm-interaction-oracle`；branch 名称和
Task 名称分别记录，不互相替代。

### 13.1 Task 目录结构

推荐的 Task 结构为：

```text
src/task/<TaskName>/
├── src/                  # 可复用的模型、数据合同和任务逻辑（若该 Task 采用 src 包）
├── tools/
│   ├── run/              # 采集、训练、评估和回放入口
│   └── audit/            # 数据、manifest、标签和结果审计脚本
├── tests/                # pytest 单元测试、接口测试和合同测试
├── configs/
└── docs/
    └── experiments/
        ├── probes/
        └── validations/
```

已有 Task 的实际代码布局可以保持，不要求为了套用模板立即重排。真正属于 Task 的
模型、数据和执行逻辑应放在 Task 目录；根级 `scripts/` 只保留跨 Task 的验证、索引和
维护入口。实验专用脚本不得继续无条件堆积到根级 `scripts/`。

### 13.2 Task-local 实验卡

未来新建的实验卡原则上属于发起它的 Task，目标位置为：

```text
src/task/<TaskName>/docs/experiments/probes/
src/task/<TaskName>/docs/experiments/validations/
```

卡片继续使用全局唯一的 `P-...` 或 `VAL-...` experiment ID，并明确记录：

* `task`：所属 Task 名称；
* `branch`：产生该实现的 Git branch；
* `git_commit`：实际运行代码的提交；
* `run_id`：一次具体运行的实例。

根级 `docs/MISSION.md`、`docs/CAMPAIGN.md`、`docs/STATE.md` 和
`docs/SEED_LEDGER.yaml` 仍是跨 Task 的研究目标、资源边界、当前总状态和 seed 归属，
不能复制成 Task-local 版本。根级 `docs/experiments/INDEX.md` 是跨 Task 的自动生成总索引，
不手工维护第二份实验卡。

当前根级 `docs/experiments/probes/` 和 `docs/experiments/validations/` 中的历史卡不迁移。
`tools/verify.py` 和 `tools/experiment_index.py` 会递归发现根级与 Task-local 卡片；新路线
可以按本节目标创建 Task-local 卡片，并继续使用同一套 schema、seed 和链接验证。

### 13.3 追溯和输出边界

实验输出、checkpoint、cache 和运行日志使用 `outputs/<TaskName>/<run_id>/` 或
实验卡声明的外部产物目录，不复制到 Task 文档目录。实验卡只保存协议、输入 hash、
运行 manifest、结果摘要和回溯链接。Task-local 工具移动后必须同步更新实验卡、manifest
和 README 中的路径；未确认没有引用前，不得批量移动根级脚本。


---
