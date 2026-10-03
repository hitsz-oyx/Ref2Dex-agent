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
