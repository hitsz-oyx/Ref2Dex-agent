# Gate 1 corrected horizon probe

**当前需要决定的问题。** 8-run corrected e260 数据上的 h16 I+ 信号是否只是
单一 horizon 的偶然现象，还是在其他短期 consequence horizon 也有方向。

**关键证据。** corrected h16 direct I+ 五个 split 全部为正，4/5 episode CI
排除零；但旧 horizon sweep 使用了 history-action 清零的旧 assembler，不能直接
代表修复后合同。

**root 选择的行动及理由。** 对同一 8-run corrected raw shards 离线组装
`H={3,5,10,32}`，固定 history length 10、I+ interaction 119 维、GRU bridge、
一个预先固定 seed `20261221`，各 horizon 只跑一次 direct six-arm fit。该 probe
只回答是否值得为某 horizon 投入多 seed；h16 的五 seed 结果不被重算或替换。

**停止条件和下一步。** 若其他 horizon 没有正向方向，固定 h16 进入正式
actor/outcome Validation；若某个 horizon 有方向，再对该 horizon 追加 matched
control 和多 seed。无论结果如何不启动 Cm，因为本 probe 不是 Validation。

**资源边界。** 四张空闲 GPU 并行，输出仅写项目 `tmp/`；只读取已有 raw shards，
不覆盖既有 artifacts、checkpoint 或外部项目。

**结果。** 单 seed 初筛中 h3/h5/h32 direct 分别为 `+11.7%`、`+15.0%`、
`+16.4%`（CI 均为正），h10 direct 为 `+5.1%`，但 h10 `V_HAEI` 为 `+21.6%`。
按预定规则追加 h3/h5/h32 direct 多 seed：h3 为 `+11.7,-5.2,+9.2,+7.9%`
（只保留方向性），h5 为 `+15.0,+10.6,-6.6,+19.0%`（不稳定）。加入确定性
quaternion-sign 修复后的 h32 为 `+7.7,+24.7,+34.9,+45.9,+35.6%`，episode
CI 有 4/5、8-source-run cluster CI 有 4/5 排除零。对应 matched future-action
control 为 `+21.2,+13.8,+17.4,+35.3,+33.1%`，五个 episode 和 cluster CI 均为正。
由此将后续正式
Validation horizon 固定为 h32；h3/h5 不再追加预算，h10 保留为 action-inclusive
诊断，不启动 Cm。

旧版 future-action 和未做确定性 quaternion-sign 的 h32 数字已标记为
`INVALID_IMPLEMENTATION`，修复记录见
[`D-20261004-gate1-future-action-index-repair`](D-20261004-gate1-future-action-index-repair.md)
和 [`D-20261004-gate1-quaternion-sign-repair`](D-20261004-gate1-quaternion-sign-repair.md)。
