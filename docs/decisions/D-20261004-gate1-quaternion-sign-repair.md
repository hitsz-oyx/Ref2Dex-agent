# Gate 1 quaternion sign repair

**当前需要决定的问题。** v2 assembler 已经在每个窗口内做 quaternion continuity，
但窗口首样本仍继承 raw `q/-q` 符号；重叠窗口可能因此把同一姿态编码成相反符号。
这会给 GRU 增加无意义的表示差异，必须在正式 Validation 前修复。

**root 选择的行动及理由。** commit `9c87acd` 将每段 quaternion 的首样本按最大
绝对分量确定符号，再沿时间传播 continuity；effect relative quaternion 也采用同一
规则。该变换只选择等价 quaternion 的代表，不改变物理旋转、target、split 或 row
order。

**验证。** slow/reference 与 fast/vectorized assembler 的 pre-step/legacy 交错 episode
回归测试，以及等价 `q/-q` 首样本测试共 3 项通过。重新组装的 e260/e420 h32 base
和 interaction augmentation audits 均为 `errors=[]`，维度仍为 80→119。

**当前实现一致性 Probe。** qfix h32 五 seed fit（同时报告 direct 与 future-action
control）得到：

| 数据 | direct `V_HEI`/`V_H` | direct episode / cluster CI 排除零 | control `V_HFEI`/`V_HF` | control episode / cluster CI 排除零 |
| --- | --- | ---: | --- | ---: |
| e260 | `+7.7,+24.7,+34.9,+45.9,+35.6%` | 4/5 / 4/5 | `+21.2,+13.8,+17.4,+35.3,+33.1%` | 5/5 / 5/5 |
| e420 | `+4.2,+25.8,+29.8,+3.9,+8.0%` | 3/5 / 4/5 | `+4.3,+27.3,+9.4,+4.5,+19.1%` | 4/5 / 4/5 |

因此 qfix 后 e260 h32 仍是 actor-local `PROMISING`，e420 仍未达到稳定跨 actor
复现。qfix artifacts 位于 `tmp/e260_all8_h32_histfix_qfix*`、
`tmp/e420_pre_s286287_h32_histfix_qfix*` 和对应 `tmp/gate1_split_rng/*qfix*`；这些
结果取代此前未做确定性首符号的 h32 fit。它们仍不是正式 Validation，不启动 Cm。
