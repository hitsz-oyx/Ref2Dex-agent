# Probe: Cm critic 显著性在第二困难训练 seed 的复制

probe_id: `P-20260923-CM-CRITIC-SECOND-SEED`

date: 2026-09-23

branch: `agent/cm-critic-salience`

status: PLANNED

## Question and decision

critic-salience 在已知困难训练 seed74 相对 Cm-off +23.44pp，
但只是一 seed Probe。现在固定完全相同的方法，在另一此前
effect-rank 失败的训练 seed73 上测试是否也有明显正信号。
若过门，启动新训练 seed 的正式 Cm-on/off/置乱显著性验证；
否则不因 seed74 的高结果形成 Cm utility claim。

## Minimal protocol and gate

同自训练 s3 e260、冻结 CmLite V1.37、corrected_r2 输入；
只改训练 seed 为73，e300、critic salience coefficient1.0、
minibatch 均值归一化不变。评估 seed133/134、64env 严格完整
首回合。已知同训练 seed73 off 为38/64、33/64。

预定推进门：critic 两次合计至少82/128（比 off 71/128
高≥11/128，即≥8pp），且每次评估均高于 off。
这些对照结果已知，仍只能标 Probe。

## Budget and stop

1 GPU、预计 <30min、新产物 <2GB。代码 commit、输入 SHA、
唯一 run_id 固定；GPU/输入冲突、非有限值、合同失败则停止。

## Result

待运行。
