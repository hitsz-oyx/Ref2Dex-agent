# V1.40: 自训练 s3 checkpoint 的互补性与固定整段路由

- experiment_id: `EXP-20260923-V140-S3-EXPERT-ROUTE`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

## 假设与冻结候选

V1.39 的单个最佳 s3 策略在未见 seeds82–84 为 40/44/42，
尚不稳定。但 seed81 上四枚**自训练** checkpoint 的成功
互补：源 e180 35/64、标准继续 e260 30/64、回启课程
e240 36/64、回启课程 e260 45/64；同环境成功并集为
62/64。测试能否根据 episode 开始帧，在整段只选一次策略，
而非逐步切换，以提高跨 seed 的严格抓取。

冻结四枚 checkpoint，不再训练 actor：

| 专家名 | run / epoch | SHA256 |
| --- | --- | --- |
| `source` | V1.35 Cm-off e180 | `863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c` |
| `standard` | V1.39 标准继续 e260 | `3212bcc195d374a4d1cb31b21051e4a0a93f63fc345a0fe987279a562092dccb` |
| `back240` | V1.39 回启课程 e240 | `a88924a590966c964b029e067985fea41465f230d459454b8048899cbe8670c2` |
| `back260` | V1.39 回启课程 e260 | `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f` |

发现集固定 seeds81–84，每个64 env、首个完整 episode、
提前终止关闭；已完成的结果只读复用，缺失的 source/
standard/back240 seeds82–84 各评估一次。首先计算每 seed
四专家同环境严格成功并集。若任一 seed 的并集不足
58/64（90.6%），判定这些专家不足以支持90%路由，
停止后续路由开发与 heldout 评估。

若并集均过门，再仅用发现集的 `start_frame` 拟合至多
4段、整段不变的专家路由；每段选择一专家，边界只取
发现集出现过的帧。优化目标为成功 episode 数减去
每额外一段2个成功的复杂度罚项；平局先选更少段，
再选较早边界及固定专家名顺序。路由不能用实际接触、
随机 seed 或未来结果作运行时输入。冻结映射后在全新
seeds85–89 各评估64 episode；稳定门槛为每 seed≥90%。
发现集是模型选择数据，不能算入未见泛化证据。

并集仅是不可实现的 oracle 上界，不是抓取成绩。路由
结果也不证明 Cm 有用；此试验只推进 s3 稳定性。
最多两张 GPU5/6 并发，输出远低于300GB；来源 SHA、
输入与评估代码由各 run manifest 锁定。
