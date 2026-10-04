# Gate 1 horizon sweep decision

**当前需要决定的问题。** 在已修复 history、contemporaneous interaction 和 temporal
encoder 后，哪个 future horizon 值得保留为下一轮 Gate 1 Validation 的主配置。

**实验设计。** 使用同一组 8 个 e260 pre-step raw runs（224 episodes）、history
length 10、qfix assembler、GRU bridge、exact Monte-Carlo return 和
`(source_run, episode_id)` split。h3/5/10/16 各先跑一个 30-epoch seed；h5 再补到
五个 seeds。该实验只筛选 horizon，不升级 Gate 结论。

**结果。** 单 seed 的 `V_HEI` 相对 `V_H` episode-balanced MAE reduction 为：

| horizon | reduction | episode bootstrap CI | 解释 |
| ---: | ---: | ---: | --- |
| 3 | +9.6% | [0.139, 4.585] | 低成本正向筛选信号 |
| 5 | +9.2% | [0.700, 3.713] | 需多 seed 复核 |
| 10 | −2.1% | [−7.484, 3.753] | 未显示正向 |
| 16 | −10.5% | [−15.891, 5.781] | 未显示正向 |

h5 五个 seed 的 reduction 为 `+9.2%, −3.9%, +4.4%, +16.0%, −7.2%`，只有 2/5
episode CI 排除零。`V_HAEI` 也没有稳定达到同量级的预先门槛。此前 qfix h32
五 seed direct 结果为 `+7.7%, +24.7%, +34.9%, +45.9%, +35.6%`，因此 h32 仍是
当前最有证据的候选，但 e420 跨 actor 仍不稳定。

**root 选择的行动及理由。** 不把 h5 的单 actor 信号升级为 Gate 结论，也不因 h10/h16
单轮负值放弃路线。保留 h32 作为下一轮 Validation 的冻结 horizon，同时优先收集
至少两个额外 checkpoint namespaces，并确保普通失败、stable-success 和
drop-after-success 都有覆盖；之后再做正式 actor-cluster bootstrap。

**停止条件。** 若新增 namespace 仍无法形成 outcome coverage，或 h32 在满足覆盖的
跨 actor 数据上不能达到预注册 reduction/CI 门，则维持 `UNCLEAR/UNPROMISING`，不启动
Cm。当前没有在线训练或外部 worktree 写入。
