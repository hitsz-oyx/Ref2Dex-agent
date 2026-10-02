# P-20261003-cm-learned-macro-policy

## 研究问题

固定 Cm selector 在 native 上失败，但 raw 3 tick macro 的局部 score 为正。测试一个
没有固定 action prior 的二元策略，能否利用 Cm consequence 特征学习状态依赖的
macro/Cup 选择，并在共同接触支持 reward 下保持安全。

## 固定对照

- `Cm-on` 与 `Cm-off` 使用同一二元 categorical PPO 网络、初始化、候选 expert bank、
  optimizer、rollout 数、环境 budget 和 evaluation seeds。
- on 输入冻结 Cm 的物理 score/std/retention/release 以及 raw-top action difference；
  off 对应通道置零。两者都保留相同 history/state 输入。
- option 0：Cm raw-top candidate 3 tick + fixed Cup 7 tick；option 1：fixed Cup 10 tick。
- reward 只由真实模拟器的接触支持高度进展、contact 和 mesh clearance 组成；Cm 预测
  不进入 reward label。
- 训练后 deterministic argmax 与初始化 argmax、Cm-on/off 独立 evaluation 比较。

## 判定

训练必须产生实际 option-ID 改变；Cm-on stable hold 不低于 off，且 contact-loss 不比
off 增加超过 2 个百分点。单个 Probe 不形成正式 Cm utility 结论；失败关闭该二元
learned-macro 配方，不追加 epochs/rollouts 或 reward/seed 扫描。

## Terminal result

工程 smoke 的 Cm-on/off 均完成 96-env 完整 episode、真实 PPO 更新、实际 PD target
校验和冻结 fingerprint 校验。第一次 on smoke 的根目录启动和第二次子 batch 校验错误
均保留在独立失败 manifest 中，修复后 smoke 才计入工程通过。

科学训练使用同一 seed `636`、assignment seed `8103` 和 4 个 rollout/arm；on/off
分别完成 `207187` native steps，参数均发生更新。两个新 evaluation seed `637,638`
使用训练后 checkpoint、确定性 argmax、无 optimizer update。

实际二元选择确实改变：eval 中 Cm-on 的 option1 argmax 为 `18/163`（11.04%），
Cm-off 为 `212/272`（77.94%），相对无先验初始化的全 option0 均有变化。可是共同的
stable success（45 tick 且无后续 drop）为 Cm-on `7/192=3.65%`，Cm-off
`11/192=5.73%`，stable-hold 门失败。以 acquisition 后 release 作为 contact-loss
审计，Cm-on `66/76=86.84%`、Cm-off `73/84=86.90%`，差 `-0.06pp`，风险门通过。

因此本 Probe 判定 **UNPROMISING**：Cm consequence 通道确实改变了策略分布并进入了
实际宏动作，但没有达到预设 stable-hold 门。关闭该配方，不追加 epochs、rollouts、
reward 或 seed 扫描；这不是 Cm 没有一步物理信息的证明，也不形成正式 utility 结论。
完整机器结果见 [results](P-20261003-cm-learned-macro-policy-results.json)。
