可以。我现在反而建议 **WAMv4 第一版就只保留“目标物体 + 手”**，把桌面、旁边物体、背景全部去掉。对于我们当前只做抓取，这会让问题定义干净很多：

\[
\boxed{
\text{Target Object}+\text{Hand}+G_{\text{local}}
\rightarrow
(\tau,\ O_{\text{future}},\ C)
}
\]

这样做的代价也明确：WAM 本身不再负责场景避障、桌面碰撞和其他物体约束。这些可以留给 executor / motion planner。唯一我稍微担心的是“从桌底穿过去”这种明显受支撑面影响的情况，但第一版完全可以先不管，后面把 **support plane 作为 1 个额外 token** 做消融，而不是重新塞几千 scene points。

---

## 1. 输入我建议先这么定

目标物体：

\[
N_O=512
\]

直接复用现在数据集已经有的 canonical surface + normal。

手：

\[
2\times11
\]

先继续用现有 semantic hand points，不急着扩大。因为现有 WAM 已经有独立 explicit-hand token 路径，手并不会因为物体有 512 点就被平均掉。

Goal / object motion 仍然用：

\[
N_{\text{flow}}=32
\]

也就是说：

```text
512 object surface ──→ PTv3 geometry
22 hand points     ──→ PTv3 + explicit hand tokens

32 object queries  ──→ object future / local goal
128 interaction queries ──→ C
```

这四个数字不用强行相等。

---

# 2. C 我会改成“固定 object query 上的 interaction field”

这里比我上一版说的“直接预测 16 个 learned C tokens”更容易训练。

先从 512 个 target-object surface points 中固定拿：

\[
\boxed{128\text{ 个 interaction queries}}
\]

这 128 个点在同一物体上有稳定 identity。

然后预测：

\[
C\in\mathbb R^{24\times128\times D_C}
\]

注意：**不再是**

\[
24\times22\times128
\]

的完整 hand-object Cartesian graph。

手的信息已经通过 shared WAM / cross-attention 进入每个 object query。

---

## 3. 这样标签反而非常简单

如果 object query 本身已经给定了位置和 normal，那么以下东西根本不用预测：

\[
p^{obj},n^{obj}
\]

它们就是 query 自己的属性。

所以当前 v4 的 9D：

\[
[c,p_{xyz},n_{xyz},v_n,v_t]
\]

我会砍掉很多。

第一版 \(C\) 可以只预测：

\[
\boxed{
C_{t,j}
=
[
c_{t,j},
P_{t,j}(\text{part}),
r_{t,j}
]
}
\]

其中：

### \(c_{t,j}\)

这个 object surface patch 是否正在/即将与手交互。

### \(P(\text{part})\)

谁在和它交互，例如：

\[
[
palm,\ thumb,\ index,\ middle,\ ring,\ little
]
\]

6 类。

### \(r_{t,j}\)

interaction 状态，可以先用三类：

\[
\text{approach},\quad
\text{hold/contact},\quad
\text{release}
\]

所以总共大概：

\[
1+6+3=10D
\]

甚至第一版可以只：

\[
1+6=7D
\]

先把 contact field 做扎实。

---

# 4. GT 怎么生成？不需要 KNN

这正好可以统一到我们的 voxel/grid 思路。

对未来每一帧 \(t+k\)：

### object

把 128 个 object queries 放到 1cm grid：

\[
V_O
\]

### hand

把 11 个 semantic points 放进去。

但 11 点直接用于 contact GT 确实太稀，所以**标签生成阶段**可以从 11 点确定性构造一个 semantic scaffold。

例如：

```text
thumb_mcp  ───── thumb_tip
index_mcp  ───── index_tip
middle_mcp ───── middle_tip
...
```

沿 segment rasterize 到 voxel grid。

Palm 用 wrist + MCP 构造几条固定线/面：

```text
        index_mcp
           |
thumb ─ wrist ─ middle_mcp
           |
       ring/little
```

这些 voxel 每个都带：

```text
part_id = thumb/index/.../palm
```

整个过程没有：

- `torch.cdist`
- KNN
- nearest neighbor

就是：

\[
\boxed{\text{semantic skeleton rasterization}}
\]

---

# 5. Contact GT 就是 voxel overlap

例如对 object query \(j\)：

如果它所在 voxel 或邻近一层 voxel 被 hand scaffold 占据：

\[
c_{t,j}=1
\]

否则：

\[
c_{t,j}=0
\]

可以用一层 dilation：

\[
V_H^+=Dilate(V_H,1)
\]

然后：

\[
\boxed{
c_{t,j}
=
\mathbf 1[
v(o_{t,j})\in V_H^+
]
}
\]

这实际上对应大约 1cm 左右 interaction band。

如果想稍微 soft 一点，也不用距离：

- 同 voxel：1.0
- 一层邻 voxel：0.5
- 更远：0

就行。

---

# 6. hand-part GT 也自然出来了

因为 hand voxel 本身带 part ID。

如果 object query \(j\) 命中 index finger 的 voxel：

\[
P_{t,j}(\text{index})=1
\]

如果附近同时有 thumb/index：

可以按命中的 voxel 数量归一化：

\[
P_{t,j}(p)
=
\frac{N_{p,j}}{\sum_qN_{q,j}}
\]

所以不需要“找最近的 finger”。

---

# 7. approach / contact / release 也不用算速度或距离

直接看 contact occupancy 时间变化。

比如：

\[
c_{t-1,j}=0,\quad c_{t,j}=0,\quad c_{t+1,j}=1
\]

就是：

\[
\text{approach}
\]

连续：

\[
1\rightarrow1\rightarrow1
\]

就是：

\[
\text{hold/contact}
\]

而：

\[
1\rightarrow0
\]

就是：

\[
\text{release}
\]

所以动态 interaction 完全可以从时序 voxel occupancy 得到。

这其实比现在 v4 的：

\[
v_n,\ v_t
\]

更稳定，因为那些量建立在 32 点 radial normal 近似上，本身也不是很可靠。

---

# 8. 模型端怎么预测 C

可以是：

\[
Z_O^{128}
=
\text{readout}(Z_{\rm PTv3},Q_O^{128})
\]

然后手 token：

\[
Z_H^{22}
\]

通过一个 cross-attention：

\[
\tilde Z_O
=
CrossAttn(Z_O,Z_H,Z_H)
\]

这一步实际上是在回答：

> 对这个 object patch，目前哪些 hand semantics 与它相关？

然后时间展开：

\[
\tilde Z_O
+
E_t
\rightarrow
C_{1:24}
\]

最终：

\[
24\times128\times7/10
\]

这比当前：

\[
24\times22\times32\times9
\]

更规整，而且提高 object spatial resolution 后显存不会暴涨得那么厉害。

---

# 9. 只有单一物体，会不会又学成“物体不动”？

不会因为“只有一个物体”自动发生，但**数据采样必须改**。

真正危险的是大量：

\[
\Delta O\approx0
\]

的 approach/pre-contact 窗口。

所以训练应该显式保留三种：

\[
\text{approach / no object motion}
\]

\[
\text{contact / static object}
\]

\[
\text{contact / moving object}
\]

尤其第二类非常重要：

\[
\boxed{
O\text{不动，但}C\text{已经明显变化}
}
\]

这样模型不能靠“object static → 全部输出零”偷懒。

而且我们已有的 object future loss 可以继续对 moving/static 分层，不需要因为去掉 scene 就改变。

---

# 10. 参数实验后面很好做

我同意你说的先定一套，后面消融。

第一版我会直接固定：

\[
\boxed{
N_{\rm object}=512
}
\]

\[
\boxed{
voxel=1cm
}
\]

\[
\boxed{
N_{\rm flow}=32
}
\]

\[
\boxed{
N_C=128
}
\]

然后以后只做几个干净的参数对照：

\[
N_C=64/128/256
\]

\[
N_O=256/512
\]

\[
voxel=5mm/10mm
\]

不用现在就全部展开。

---

所以如果现在让我把 WAMv4 收敛成一个明确版本，我会选：

\[
\boxed{
\text{512-point target object}
+
\text{22 semantic hand points}
+
G_{\rm local}
}
\]

没有其他 scene points。

C 改成：

\[
\boxed{
24\times128\text{ object-centric interaction field}
}
\]

GT 用：

\[
\boxed{
\text{future semantic-hand voxel occupancy}
}
\]

来生成 contact + hand-part + interaction-state 标签，**模型和标签生产都不需要 KNN**。

我觉得这一版已经足够干净，可以作为真正值得训练的 WAMv4 基线。