# Ref2Dex Autonomous Research Agent

本仓库采用“自主探索、少量人工决策、正式验证后再形成结论”的科研工作方式。

核心原则：

> Optimize for research progress, not procedural completeness.

AI 的目标不是把每个可能的问题都研究完整，而是在资源和时间有限的情况下，持续获得能够改变研究决策的信息，并向 `MISSION.md` 中定义的最终目标推进。

---

## 1. 默认工作模式：自主探索

不需要因为普通研究变量、网络结构、reward、或实现细节变化而逐次请求用户批准。

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
5. `docs/context/INDEX.md`，然后只读它标成必读的文件。归档不在默认上下文里
6. `docs/experiments/INDEX.md`
7. 与当前问题直接相关的代码和活跃实验记录

`docs/STATE.md` 是全库历史。当前战役的上下文以 `docs/context/` 为准，不把 `STATE.md` 整份读入。

进入具体 Task 时，再读取所属 Task 已有的研究文档；不遍历无关 Task。
`docs/research/README.md` 只是根级研究文档的组织说明；只有当前问题涉及跨
Task 综合材料时才读取它。历史 plan、指导、activity、decision、log、handoff 和已归档
research 只有在需要核对证据时才追溯，不属于默认上下文。


---

## 3. 每个实验必须服务于一个决策

在运行一个非纯工程 smoke 的实验前，AI 必须回答：

1. 这个实验想区分什么假设？
2. 实验结果会改变哪个下一步决策？
3. 获取该信息的最便宜方法是什么？但是注意有些实验就是要在大规模训练下才能起效果。

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


### 4.1 默认优先使用 GPU

神经网络训练、微调、重复模型推理和批量模型评估默认优先使用 GPU；支持 GPU
加速的仿真采集也优先使用 GPU。离线实验同样适用，不因“Probe”“小规模”或
历史 CPU-only 阶段而默认改用 CPU。具体选择和资源边界遵循
[`CAMPAIGN.md`](docs/CAMPAIGN.md) 的 GPU execution policy 与 GPU budget。

### 4.2 实验的时候需要记录运行环境

例如conda环境等等

---

## 5. Git 规则

默认在`main` 开发，及时提交代码

提交之后可以自行向远端push

push的时候可以尝试设置代理

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897



不要 reset、覆盖或删除用户已有修改。

**默认：提交完就推送。** 每完成一步工作就 `git add`（只加你这轮改的文件）→ `git commit` → `git push origin main`，并在回答里报出 commit sha 与推送结果；只有机主当轮明确说"先不要推送"才停。推送后用 `git ls-remote origin main` 核对远端 sha 与自己的 `git rev-parse HEAD` 一致，两个 sha 都写进回答。

**推送凭据（2026-10-22 由 Hermes 建好，机主授权）：**
- 服务器上**默认没有任何 credential.helper**，`~/.netrc` 里只有 wandb 的 token，VS Code 的 GitHub 凭据在本机客户端、不在服务器 —— 所以裸 SSH 里直接 push 会报 `fatal: could not read Username for 'https://github.com': terminal prompts disabled`（不是网络问题：`curl https://github.com` 返回 200）。
- 本仓库已配好**仓库级**凭据助手：
  - 脚本：`/home2/wyy/.local/bin/gh-git-credential.sh`（权限 700；内部现取现用 `~/.config/gh/hosts.yml` 的 oauth_token，**不复制、不打印**）
  - 配置：`git -C <repo> config credential.helper '/home2/wyy/.local/bin/gh-git-credential.sh'`
  - 用法：直接 `git push origin main`，非交互也 OK。
- 兜底（换机器/换 home/助手丢了）：临时 askpass 再推，用完立即删除：
  ```sh
  printf '#!/bin/sh\ncase "$1" in\n  Username*) printf "hitsz-oyx\\n" ;;\n  Password*) grep -m1 "oauth_token:" ~/.config/gh/hosts.yml | sed "s/.*oauth_token: *//" | tr -d " \\r" ;;\nesac\n' > /tmp/gh_askpass.sh
  chmod 700 /tmp/gh_askpass.sh
  GIT_ASKPASS=/tmp/gh_askpass.sh GIT_TERMINAL_PROMPT=0 git push origin main
  rm -f /tmp/gh_askpass.sh
  ```
- 代理（旧建议，网络受限时再试）：`export http_proxy=http://127.0.0.1:7897` / `https_proxy` / `HTTP_PROXY` / `HTTPS_PROXY`。
- **禁止**：打印/复制/提交任何 token 或凭据文件；`push --force`；把与本轮无关的改动一起 commit；推送失败时假称成功（如实报错并说明卡在哪一步）。

---

## 6. 身份与版本

代码版本、实验和研究阶段必须使用不同的身份，不要用同一个连续版本号同时表示三者。


### 6.1 Experiment Identity

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

---

## 7. 保护边界

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


---

## 8. 关于tmp

不要把任何输出放到/tmp，如果有必要，放在项目目录下的tmp，如果没有这个目录就新建。

---


## 9. Task 与研究路线目录

新的研究方向在 `src/task/<TaskName>/` 下建立稳定的 Task 目录，由用户建立

### 10.1 Task 目录结构

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

### 10.2 Task-local 实验卡

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
其中研究文档可按第 13.4 节在 Task 内细化；`SEED_LEDGER.yaml` 保持全局唯一。
根级 `docs/experiments/INDEX.md` 是跨 Task 的自动生成总索引，不手工维护第二份实验卡。

当前根级 `docs/experiments/probes/` 和 `docs/experiments/validations/` 中的历史卡不迁移。
`tools/verify.py` 和 `tools/experiment_index.py` 会递归发现根级与 Task-local 卡片；新路线
可以按本节目标创建 Task-local 卡片，并继续使用同一套 schema、seed 和链接验证。

### 10.3 追溯和输出边界

实验输出、checkpoint、cache 和运行日志使用 `outputs/<TaskName>/<run_id>/` 或
实验卡声明的外部产物目录，不复制到 Task 文档目录。实验卡只保存协议、
运行 manifest、结果摘要和回溯链接。Task-local 工具移动后必须同步更新实验卡、manifest
和 README 中的路径；未确认没有引用前，不得批量移动根级脚本。

### 10.4 研究文档的分层组织与继承

`MISSION.md`、`CAMPAIGN.md`、`STATE.md` 和 `README.md` 可以按需放在
`src/task/<TaskName>/docs/` 下，分别记录该 Task 的子目标、执行约束、当前状态和文档入口。
已有 Task 根目录的 `README.md` 可以继续作为入口，不要求再建一份。

`src/task/<TaskName>/docs/research/` 是可选目录，不是每个 Task 的必备结构。只有需要保存
较长的方法说明、文献边界或跨多个实验的综合分析时才创建；单个实验的协议和结果仍写在
实验卡中，数据与复现检查仍放在 `tools/audit/`。Task 没有这类材料时，不要创建空目录。

读取时按“仓库根级 → 当前 Task → 当前实验”的顺序逐层补充上下文。若相关子目录已有
更局部的研究文档，沿当前任务所属目录继续读取；不递归加载其他 Task 或历史归档。
局部文件不存在时继承上层约定，不为目录完整性创建空文件，也不整份复制上层文档。

* 根级 `MISSION.md` 定义最终研究目标和 claim；Task 的 `MISSION.md` 说明本路线解决
  哪个子问题、与总目标的关系，以及本路线的完成条件。
* 根级 `CAMPAIGN.md` 定义共享资源和权限边界；Task 的 `CAMPAIGN.md` 只补充本路线的
  预算、设备安排和停止条件。各 Task 的预算合计仍受全局上限约束，不能各自重复领取上限。
* 根级 `STATE.md` 只汇总影响跨 Task 决策的重要进展；Task 的 `STATE.md` 如有需要，
  只维护本路线的当前判断、blocker 和下一步。历史过程与证据仍放在实验卡和 Git 中。

根级 `docs/research/` 只用于真正跨 Task 的综合材料；当前历史材料已归档到
`docs/archive/2026-10-04-root-research/`。局部约定可以细化目标或收紧限制，不能自行改写全局 claim、放宽资源限制或扩大权限；
需要改变这些边界时遵循第 5 节。研究结论的适用范围须明确，Task 内的局部结果不能自动
升级为全局结论。这里只规定文档的组织与读取方式，不要求迁移已有文档或新增一套记录流程。

---


## 11. git提交

ref文档的变更默认可以提交,ai不得私自往ref目录下添加新文件，新建分支需要用户同意



## 12. 训练时的注意事项

开始训练之后需要观察GPU利用率和显存占用情况，并预估完成时间，有不合理的情况及时告诉用户

## 13. 数据集下载注意事项

如果要在hugging face上下载，优先尝试在 https://modelscope.cn 网址查看是否有对应的数据集，如果没有再尝试访问hf的国内镜像站，如果还是没有，再用代理下载

## 14. 关于解决问题

遇到棘手和困难的问题，一方面查找相关文献，看看是否是别人解决过的问题，可以把仓库克隆到third_party,一方面可以查看各个task下的实验卡，看看是不是已经有过类似的实验，但是只能借鉴，不能盲目相信，需要仔细查看是否正确。

查找相关文献时可以把文档记录在仓库的paper目录下，子目录以论文名命名，可以用缩写，例如dexplore，schema信息标注好arxiv的链接，防止之后重复索引。可以把浏览的思考结果写成笔记放在子目录下，方便以后查阅。自己在查找文献的时候也可以先看看paper目录下的整理好的笔记，防止重复检索。有源码的话也可以记录在schema里面，方便查阅。

## 15. 有关用户参与

遇到一些视觉上的需求的时候，可以让用户参与，查看可视化效果以验证功能的正确性。例如数据处理等等

## 16. 关于哈希

不需要记录和核对非常严格的哈希，如果版本控制在掌控范围内，不需要那么严谨

## 17. 有关机器环境

可以先参考一下docs/MACHINE_MEMORY.md文档