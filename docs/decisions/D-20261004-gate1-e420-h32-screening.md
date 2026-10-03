# Gate 1 corrected e420 h32 screening

**当前需要决定的问题。** corrected e260 的 h32 I+ 是否能在独立 e420 actor/data
上复现，还是 h32 也只是 e260-local signal。

**关键证据。** corrected e260 h32 direct 和 future-action control 都在五 seed、
8 source-run cluster CI 下为正；corrected e420 h16 direct/control 则高度混合。

**root 选择的行动及理由。** 复用已经存在的 corrected e420 s286/s287 raw shards，
离线组装 h32、执行 dataset/augmentation audit，并用固定 seed `20261221` 做一次
direct six-arm screening。此 probe 不改 raw 数据、不重采集、不启动 Cm；结果只决定
是否值得为 e420 h32 追加多 seed。

**停止条件和下一步。** 若 h32 仍混合，停止 e420 horizon 扩展，保留 e260-local
`PROMISING`，进入预注册多 actor Validation 设计；若 h32 正向，再追加 matched
control 和多 seed。无论结果如何不把 single-seed screening 写成 Gate closeout。

**结果。** corrected e420 h32 dataset 为 44 episodes、21,940 windows，base 与
augmentation audit 通过。五个 unique model seeds 的 direct I+ 改善为
`+4.0%`、`+22.6%`、`+30.1%`、`+3.9%`、`+2.2%`；seed 1、3 的 episode CI
排除零（2/5）；e420 只有 2 source-run clusters，cluster CI 仅 seed 1、3、4
为正。HAEI screening 为 `+2.9%`。因此 h32 也没有稳定跨 actor 复现，按停止条件
停止 e420 horizon 扩展；h32 只保留为 e260-local `PROMISING` candidate，不启动 Cm。
