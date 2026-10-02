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
