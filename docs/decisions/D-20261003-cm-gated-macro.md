# Decision Memo：在 3 tick Cm macro 上复用既有安全门

raw-top 3 tick macro 的原生 score 已有正向证据（`+13.55 mm`，90% 组区间
`[+4.06,+21.85]`），但末段 contact 下界为 `-0.075`，不能进入策略训练。使用
同一批 A/B 记录按预先存在的 2-sigma、OOD、retention 和 release 规则做离线重放时，
风险约束后的推荐仍有 `+9.30 mm` 点估计，score 下界 `+1.98 mm`，contact 下界
`-0.049`，没有改阈值或扫描参数。

选择一次新的 native A/B：在新的 contact states 上固定该安全规则；规则通过时执行
raw-top candidate 3 tick + fixed Cup 7 tick，规则不通过时整段回退 fixed Cup；与
fixed Cup 以 `p=.5` 随机分配。Cm 仍只负责物理 consequence 预测。两侧支持至少 48
窗口、20 episode，score 和三项风险门同时通过才进入小规模 categorical policy
training；否则关闭 gated macro，不再追 sigma/宏长度/seed。
