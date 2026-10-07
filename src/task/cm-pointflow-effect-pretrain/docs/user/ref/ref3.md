可以直接采用 **PointWorld 的核心架构**，但不要照搬它默认的 `base`/`large` 规模。对我们这个任务，**用 PointWorld-small 再做一次任务裁剪更合理**。

PointWorld 官方目前有 `small / base / large` 三档。默认是 `base + predictor_dim=256`；而 small 的 PTv3 编码深度是：

\[
[2,2,2,6,2]
\]

通道大致是：

\[
128\rightarrow128\rightarrow128\rightarrow256\rightarrow512
\]

decoder 也只有 4 个 stage。它明显比 base/large 收缩。更重要的是，我们当前自己的模型已经有 **37.2M 参数**，其实本身也不算小，所以“换 PointWorld 就一定更大”并不成立。

我建议我们现在不是完全复刻 PointWorld，而是定一个 **PointWorld-small-WM24**：

- `PTv3 size = small`
- `predictor_dim = 128`
- `grid = 1 cm`
- `patch size = 64 或 128`，先用 **128**
- 当前 scene 最多 **4096 points**
- future hand action 固定 \(24\times22=528\) points
- 不要 RGB / DINO / image feature
- 不要完整 PointWorld 的视觉分支
- 第一版也可以先不加 uncertainty head

于是输入只有几千点：

\[
\underbrace{P_t^{scene}}_{\le4096}
+
\underbrace{Q_{t+1:t+24}^{hand}}_{528}
\]

而不是 PointWorld 原本面对的最高约万级 scene points。

---

### 我更建议连融合方式也直接学 PointWorld

我们当前结构是：

\[
H\rightarrow Z_H
\]

\[
A\rightarrow Z_A
\]

最后才：

\[
Z_H+Z_A\rightarrow E
\]

现在已经看到 action 几乎被忽略。

PointWorld则更直接：

\[
\boxed{
[\text{当前 scene points};
\text{未来 robot/hand trajectory points}]
\rightarrow PTv3
}
\]

也就是 action 本身就在 3D 空间里和 scene 一起做 attention。

对我们：

```text
当前 object/scene points
             +
24步 × 双手11 keypoints
             ↓
     Unified PTv3-small
             ↓
     scene point features
             +
 hand global summary / FiLM
             ↓
       Dynamics head
             ↓
   future object effect
```

这个我认为比当前“两个 Transformer 分开编码再 cross-attention”更符合我们的 action 定义。

因为我们的 \(A\) 本来就是：

\[
\boxed{\text{3D hand trajectory}}
\]

没有必要先把它抽象成另一套 token 空间，再期待 dynamics decoder 自己理解空间接触关系。

---

### 但输出端我不建议完全照抄 PointWorld

PointWorld直接预测每个 scene point 的未来 flow：

\[
P_t^{scene}\rightarrow
P_{t+1:t+T}^{scene}
\]

我们 OakInk2 有一个明显优势：

> 每个物体的 rigid SE(3) GT 是精确已知的。

所以可以保留我们的：

\[
\boxed{PTv3\ backbone + rigid object effect head}
\]

即对每个 object 聚合其 point feature：

\[
z_m=
\operatorname{Pool}
\{f_i:i\in object_m\}
\]

然后输出：

\[
z_m\rightarrow
\{\Delta t_\tau,\Delta R_\tau\}_{\tau=1}^{24}
\]

再解析得到 512 点的未来 trajectory。

这样既用了 PointWorld 的强项：

> **scene/action 统一3D建模**

又保留我们数据最干净的部分：

> **rigid object dynamics supervision**

没必要强行把一个刚体物体拆成512个独立点自由预测。

---

### 训练方法反而应该比模型大小更认真地照 PointWorld

这个我现在认为比“到底128还是256维”更重要。

我们 V1 至少应该拿过来：

\[
\boxed{\text{per-horizon target normalization}}
\]

例如对每个 \(\tau\)：

\[
\tilde E_\tau=
\frac{E_\tau-\mu_\tau}{\sigma_\tau}
\]

以及 PointWorld 那种：

\[
\boxed{\text{motion-aware soft weighting}}
\]

而不是现在所有 object × timestep 等权。

这样直接处理现在的：

\[
\hat E\approx0
\]

问题。

Loss 可以先用 normalized Huber，而不是 raw-meter L1。

---

## 所以我建议 V1 定成下面这个

```text
                     H
         current scene/object points
                       │
                       │
A: future 24-step hand keypoint trajectory
                       │
                       ▼
              unified PTv3-small
           dim 128 / grid 1 cm
                       │
             ┌─────────┴──────────┐
             │                    │
       scene local features    hand summary
             │                    │
             └──── FiLM/fusion ───┘
                       │
                object pooling
                       │
             object × 24 queries
                       │
                 SE(3) head
                       │
          translation + rotation
                       │
          analytical point effect
```

训练：

\[
H+A\rightarrow E
\]

仍然不变。

对照仍然是：

\[
H\rightarrow E
\]

\[
H+A\rightarrow E
\]

\[
H+\operatorname{shuffle}(A)\rightarrow E
\]

---

### 规模我会先控制在这个程度

| 项目 | 当前模型 | 建议 V1 |
|---|---:|---:|
| hidden | 384 | **128** |
| scene backbone | SparseConv + 8L Transformer | **PTv3-small** |
| action encoder | 独立 4L Transformer | **取消** |
| dynamics | 6L decoder | **取消/大幅简化** |
| scene points | 512×局部物体数 | ≤4096 |
| action points | 528 | 528 |
| RGB | 无 | 无 |
| KNN | 无 | 无 |
| 输出 | rigid SE(3) | rigid SE(3) |

所以虽然 PTv3 本身结构比我们现在的 sparse stem 更复杂，但我们同时**删除了独立 action Transformer 和 6层 dynamics decoder**。

因此整个网络未必比现在 37M 大，甚至很可能更合理、更高效。真正需要在实现后跑一次 `sum(p.numel())` 才能给精确参数量，我不建议现在凭结构猜一个数字。

我的选择会是：

\[
\boxed{\text{直接采用 PointWorld-small 的空间动力学骨架，而不是 PointWorld-base}}
\]

并且**先忠实照它的 normalization、motion weighting、action/scene unified PTv3 融合**。这次不要再自己凭论文重新设计一套 3D backbone。我们只保留真正属于我们任务的两个改动：**24步双手关键点 action** 和 **rigid multi-object SE(3) effect head**。