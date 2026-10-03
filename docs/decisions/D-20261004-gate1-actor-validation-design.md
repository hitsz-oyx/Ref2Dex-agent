# Gate 1 multi-actor Validation design

**当前需要决定的问题。** qfix 后 e260 h32 仍有稳定的 actor-local signal，但 e420
没有稳定复现；是否应该把下一轮预算用于预注册的多 actor/outcome Validation，而不是
继续在单一 actor 上堆 interaction feature。

**root 选择的行动及理由。** 冻结当前实现合同，先形成可执行的 Validation 配方；
在配方冻结、独立 raw collection 完成前不启动 Cm。这样能区分“表示在某一 actor 上
有效”和“表示跨 actor 保留长期价值信息”。

**冻结合同。**

- code commit 使用包含 `fcfde1c`、`9c87acd`、`99061d8` 的 assembler/fit 版本；
  缺失 `physical_timing` 默认失败，不能静默 legacy shift。
- `history_length=10`，`horizon=32`；`H` 是连续过去 history，`E/I` 是未来
  contemporaneous object-frame 序列；quaternion 使用确定性首符号和时间 continuity。
- 只保留完整 future window；target 是 `gamma=0.99` 的 exact Monte-Carlo return。
- primary arms 固定为 `V_H` 与 `V_HEI`；matched diagnostic control 固定为
  `V_HF` 与 `V_HFEI`；`V_HAEI` 作为 action-inclusive sensitivity arm。
- split 以 `(actor/source_run, episode_id)` 为单位；所有 arms 使用相同 split、model
  seed、epoch 和 episode-balanced MAE。actor/source-run cluster bootstrap 是主要
  不确定性审计单位，不能用逐 window bootstrap 替代。

**预注册数据要求。** 至少四个独立 actor/checkpoint namespaces，每个 actor 产生
完整的普通失败、stable-success 和 drop-after-success 覆盖；每个 episode 只出现一次，
不得因结果筛选或后验 horizon 选择加入数据。所有 raw shard 必须通过 state/next-state、
timing、quaternion、finite-value、row-key 和 augmentation audit；任何失败都将该
actor 标记为 `INVALID_IMPLEMENTATION`，不以剩余 actor 补救。

**判定。** `V_HEI` 相对 `V_H` 的 episode-balanced MAE reduction 必须至少 10%，且
episode 与 actor-cluster 的 95% bootstrap CI 都排除零；`V_HAEI` 不得显示同等量级的
action-only 替代增益。满足时只能将 Gate 1 标为 `PROMISING` 并进入 Gate 2，不满足时
按证据分为 `UNCLEAR`/`UNPROMISING`，不启动 Cm。任何单 actor、单 seed、单 rollout
或单一成功 episode 都不能满足 Gate 1。

**停止条件和成本边界。** 若四个独立 actor 中有两个以上未通过合同或 outcome 覆盖，
停止继续扩展该配方并记录 Research Debt；若资源超过四 GPU、单次超过 Validation
预算或需要修改外部项目，重新写 Decision Memo。当前这份设计不启动训练、不写外部
checkpoint，仅冻结下一轮采集和离线 fit 的判定接口。
