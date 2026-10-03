# Gate 1 history-action contract repair

**当前需要决定的问题。** Gate 1 v2 的 history 是否真的包含
`{s_{t-L+1:t}, a_{t-L:t-1}}` 中的 preceding actions。

**关键证据。** 两个 assembler 都曾把每个窗口的最老 history action 强制置零。
在四个 e260 raw run 中，这个位置约 99.8% 的窗口有非零 factual
`previous_action`；因此这不是只影响 episode 起点的 padding，而是系统性丢失历史。

**root 选择的行动及理由。** 删除无条件清零，保留 raw collector 的 factual
preceding action；episode 初始动作仍由 collector 的真实值表示。同步修订 metadata，
然后重新组装、audit、augmentation 和 h16 direct probe。保留旧数据与旧结果作为
历史证据，不覆盖任何 artifact。

**预计成本、成功/失败后的下一步和停止条件。** 组装和离线 audit 为 CPU 文件处理；
5 个 h16 direct fit 使用 4 张空闲 GPU，约数分钟。若修复后 split 敏感性消失，进入
matched future-action control；若仍敏感，按 outcome coverage 诊断，不再把旧结果作为
修复后合同的结论。该 probe 不启动 Cm，也不改变 Gate 1 的研究问题。

**外部授权边界。** 只读取已有 raw shards；不修改外部项目、checkpoint 或在线进程。
