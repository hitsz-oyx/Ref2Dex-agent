# Consequence evaluator

用户方案：[ref1](docs/user/ref/ref1.md)。实现分支：`consequence-evaluator`。
当前工作只建立离线 oracle headroom 检验，不改变根级 Mission 的最终 Cm 策略收益要求。

第一阶段固定 `K=24`、`K_exec=8`，H 沿用采集策略当前观测/历史的原始合同。
先采连续六专家 rollout，单次24步平滑扰动后让专家继续到 episode 结束；不 fork。
比较 `E0(H,A)` 与 `Eoracle(H,A,Z_GT)`，PointWorld 适配与在线 proposal 延后。
主要衡量同任务、同当前阶段、跨 episode 的局部偏好排序；它不是同状态候选控制收益。

## 已实现的最小合同与训练入口

- [窗口准备](tools/run/prepare_windows.py)：episode/seed group 先 split，再切完整24步窗口。
- [数据合同](src/consequence_evaluator/data.py)：严格输入白名单、完整时钟、刚体效应、标签 mask 与 split 检查。
- [两臂模型](src/consequence_evaluator/model.py)：同容量两层128维 Transformer，24步 progress 分布与独立标量分支评分。
- [训练](tools/run/train_matched.py)：相同初始化、抽样、优化器和更新预算；只在训练集拟合归一化，用 val 保存最佳权重。
- [资产预检](tools/audit/preflight.py)：核对六专家、motion、旧 rollout 和当前 GPU 占用。
- [研究原文](docs/research/PRIMARY_SOURCES.md)与[代码/数据复用核对](docs/DATA_REUSE.md)。

两臂区别只在 future slot 内容：E0 在归一化之后使用固定零向量，Eoracle 读取真实物理量。
两臂均保留同一个 future module。模型输入只有 H、真实执行的18维 native action、物体刚体 effect；
不接收未来 reward、success、drop/contact flag、episode quality、扰动身份或时间到成功。
第一次只评价单个刚体 anchor，Z为 `T_t^-1 @ T_(t+1:t+24)`；这与现有 PointWorld
对 anchor 输出的 translation/rotation 可互换。多物体表示后续单独扩展。
当前 A 仍是 native control，PointWorld 的手部 point-flow action 适配尚未实现；不得宣称两者已完整打通。

Robometer 原版使用联合两轨迹 preference head；独立 `s` 的 Bradley–Terry loss 是明确改编。
Progress 用10-bin soft CE，只监督可靠 `expert_success` 的绝对 episode 进度，失败/次优 mask。
不把每个窗口重新标成0→1，也不把 episode 最终失败直接继承为全部局部窗口失败。
首版要求显式局部偏好 annotation；不自动产生旧Y或从窗口外结局伪造局部排序。

## 连续 episode 输入

源目录的 `manifest.json`：

```json
{
  "schema": "ref2dex.consequence-evaluator.episodes.v1",
  "rollout_kind": "continuous", "training_allowed": true,
  "fps": 30, "units": "m",
  "history_contract": "采集策略的观测/历史 schema、维度与来源 SHA256",
  "episodes": [{
    "episode": "unique_episode_id", "split_group": "source_seed_group",
    "split": "train", "task": "airplane", "quality": "expert_success",
    "path": "unique_episode_id.npz", "sha256": "实际文件 SHA256"
  }]
}
```

每个 episode NPZ 使用 `allow_pickle=False`，仅含以下字段：

| 字段 | Shape / 含义 |
| --- | --- |
| history | `[T+1, ...]`，策略原始观测/历史，不新定义history length |
| action | `[T,18]`，真实执行的[-1,1] native control |
| object_pose | `[T+1,4,4]`，测得的 object-to-stationary-world，米 |
| timestamps | `[T+1]`，真实连续30Hz |
| phase | `[T]`，当前状态决定的采集阶段，不作为额外模型输入 |
| progress | `[T+1]`，可靠成功示范的绝对进度；未知可NaN |
| progress_mask | `[T+1]`，bool；失败/次优全部false |

其中 `action[t]` 把 `object_pose[t]` 变成 `object_pose[t+1]`。
episode 文件必须完整，不能把 reset 后的另一条轨迹拼进来。尚未具备真实、已核验的采集导出器。
已有 collector 代码可复用，但新24步平滑扰动/阶段采样和上述源格式仍需接入、实测。

偏好文件单独记录 `scope: local_window`、`label_provenance`，以及 `pairs`。
每对包含 `chosen` / `rejected` 的 `{episode, tick}` 与 `annotation`。
标签来源必须核对窗口内的具体事件；无明确局部顺序的比较不标。
程序验证 split/task/phase 相同、episode不同、annotation非空，但无法替代语义审查。

准备后运行的模板（真实源、偏好审计、空闲GPU和注册seed就绪后）：

```bash
python src/task/consequence-evaluator/tools/run/prepare_windows.py \
  --source <continuous-episode-directory> --preferences <local-preferences.json> \
  --output outputs/consequence-evaluator/<prepared-run>
python src/task/consequence-evaluator/tools/run/train_matched.py \
  --data outputs/consequence-evaluator/<prepared-run> \
  --output outputs/consequence-evaluator/<fit-run> \
  --gpu <idle-gpu> --seed <registered-probe-seed> --updates 1000 --batch 32 --seconds 1800
```

训练不读取 test pairs；当前只报告 val开发指标。独立、冻结后 heldout 评价及分任务/阶段汇总
仍待实现，不能用 val 最佳分数冒充正式测试结果。保存两臂 initial/latest/best、训练/验证JSONL、
输入/源码hash、显存、吞吐与ETA。当前入口尚未在真实GPU数据上验证，CPU小模型检查只证明工程合同。

## 当前判断与下一步

2026-10-07：旧 oracle 原始 outputs 与六专家checkpoint未在当前仓库找到；外部只读项目中
存在仿真资产，但扫描外部旧 DExplore outputs 未找到这六个固定自训练权重。
当前GPU0/1/2仍运行PointWorld，其他卡均有任务，因此没有启动新采集或真实拟合。
等待备份路径，先核对连续数据可用性、局部标签、阶段覆盖，再冻结小规模实验卡与seed。

Recorded reactive future A 已受未来反馈影响；Eoracle−E0 只度量在这种A条件下的额外信息。
正向才值得进行EWM与后续在线规划；负向先查数据/监督/拟合，不能直接判世界模型无用。
