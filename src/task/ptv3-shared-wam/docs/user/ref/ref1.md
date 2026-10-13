可以，而且我认为这条线值得和 Residual PPO **并行**。我会把它定义成一个新的 baseline，而不是继续修改现在的 `GoalPointWAM`：

\[
\boxed{\text{PTv3-Shared-WAM}}
\]

目标是借 PointWAM 最核心的思想——**scene future 和 hand future 来自同一个交互表征**——但保留我们已有的 PTv3、三域数据、object-goal 条件和机器人 Chunk-R。

PointWAM 官方描述的核心正是“一套 transformer 读取 scene/hand/task 条件，然后两个 head 分别预测 scene 与 hand trajectory；scene forecast 主要监督共享 trunk，hand forecast 再进入 retargeter”。[Chrockey](https://chrockey.github.io/PointWAM/?utm_source=chatgpt.com)

---

# 1. 我建议的新整体架构

先给完整图：

```text
                    observed scene H4
                           │
                           ▼
                    Scene Projector
                           │
                           ▼
                         PTv3
                    （结构完全不改）
                           │
                  scene point features
                           │
                     scene tokens
                           │
                           │
Current hand H4 ──→ Hand Token Encoder
                           │
                     22 hand tokens
                           │
             ┌─────────────┴──────────────┐
             │                            │
             ▼                            ▼
       scene tokens                  hand tokens
             │                            │
             └──────────┬─────────────────┘
                        ▼
            Shared Interaction Transformer
                        │
              Z_scene + Z_hand
                  ┌─────┴─────┐
                  │           │
                  ▼           ▼
          Scene Future      Hand Future
              Head             Head
                  │           │
                  ▼           ▼
               U_hat        V_hat = τ_hat
              [24,...]      [24,2,11,3]
                              │
                              │ + Object Goal
                              │
                              ▼
                          Chunk-R
                       + robot state
                              │
                              ▼
                         action chunk
                              │
                              ▼
                            robot
```

这里最关键的是：

\[
\boxed{
\hat U,\hat V
\text{ 都来自同一个 }Z
}
\]

而不是现在：

\[
\hat\tau=A(H,G),
\]

然后：

\[
E=W(H,\hat\tau).
\]

也就是说，新 baseline **彻底去掉 \(V\rightarrow U\) 这根显式箭头**。

---

# 2. PTv3 保持不变，但 hand 的地位明显提高

我不建议直接把 22 个 hand points 拼进当前 PTv3 voxel 输入。

原因很实际。

我们的 PTv3 使用：

\[
1\text{ cm voxel}
\]

而抓取接触时 fingertip 和 object surface 很可能进入同一个 voxel。

当前代码里面又会根据相同 voxel key 聚合 feature。

如果直接把：

```text
scene point
hand point
```

扔进去，很可能变成：

\[
\text{mean}(f_{\rm scene},f_{\rm hand}),
\]

这反而把 hand identity 弄弱了。

所以我建议：

\[
\boxed{\text{PTv3 专门负责 scene spatial representation}}
\]

而 hand points 用**一等 token**存在。

---

# 3. Hand Token Encoder

当前我们已经有很好的 hand observation：

每个 hand point：

\[
x_{h,k}
=
[
p_{h,k},
v_{h,k},
a_{h,k},
\Delta p_{h,k}
].
\]

大约就是现在 raw-hand 的：

\[
3+9=12D.
\]

新版本不要再把它仅仅作为：

> trajectory query 上加一个 residual。

而是：

\[
e_{h,k}
=
MLP(x_{h,k})
+
E_{\rm handID}
+
E_{\rm keypointID}
+
E_{\rm type}.
\]

得到：

\[
Z_H^0\in\mathbb R^{22\times128}.
\]

其中明确保存：

\[
\text{left/right hand ID}
\]

以及：

\[
\text{wrist / MCP / fingertip semantic ID}.
\]

这样 fingertip 从一开始就是一个独立 token。

这就是我说的：

\[
\boxed{\text{提高 hand point 的地位}}
\]

不是简单把 hand loss 权重调大。

而是从 architecture 上让：

\[
\boxed{
\text{scene token 与 hand token 是同一级的信息载体}
}
\]

---

# 4. PTv3 输出怎么进入共享 backbone

现有 `SceneHistoryEncoder` 已经能得到：

\[
Z_S^0\in\mathbb R^{256\times128}.
\]

所以新模型实际上可以直接复用：

\[
256\ scene\ tokens
+
22\ hand\ tokens.
\]

总共：

\[
278
\]

个 tokens。

然后增加一个很小的 Shared Interaction Transformer：

\[
F_{\rm int}
\]

比如第一版：

\[
4\ layers,
\quad
d=128,
\quad
4\ heads,
\quad
FFN=512.
\]

计算：

\[
[Z_S,Z_H]
=
F_{\rm int}
(
[Z_S^0,Z_H^0]
).
\]

这层才是真正负责学习：

\[
\boxed{\text{hand ↔ scene interaction}}
\]

的地方。

注意：

**我们没有手工规定哪个 fingertip 对哪个 object。**

没有：

\[
d_{\rm finger-object}
\]

输入，

没有：

\[
\tau\rightarrow E
\]

硬结构，

也不提前告诉网络：

> 这个点一定和这个物体有关。

它自己通过 self-attention 学：

\[
\operatorname{Attention}(hand_i,scene_j).
\]

这正是你说的：

> 不再强行把手和物关联起来，让共同 backbone 自己学习。

我非常赞同这个方向。

---

# 5. Scene Future Head：真正改成 PointWAM-style

这里不要再用现有：

\[
W(H,\tau)\rightarrow E.
\]

新 head 是：

\[
\boxed{
\hat U=W_{\rm head}(Z_S,Z_H)
}
\]

注意没有：

\[
\tau
\]

作为条件。

---

## 输出什么？

我建议第一版不要一下子预测所有 background dense points。

我们三域原本就有稳定的 object point / rigid effect supervision，所以可以把每个物体上的固定点转换成 point trajectory：

\[
u_{i,h}
=
p_{i,t+h}-p_{i,t}.
\]

最终：

\[
\hat U
\in
\mathbb R^{B\times O\times24\times N_o\times3}.
\]

例如每个 tracked object：

\[
N_o=32.
\]

静止物体：

\[
u_{i,h}\approx0.
\]

运动物体：

\[
u_{i,h}\neq0.
\]

这已经非常接近 PointWAM 的：

\[
\text{scene-point trajectory forecasting}.
\]

而且我们的 OakInk2 / GRAB / ARCTIC 现有 rigid pose/effect 标签完全可以转换成这种监督，不需要重新做一套数据。

以后有 EgoDex dense tracks 后再扩成真正 dense scene trajectory。

---

# 6. Hand Future Head

这一版我也建议改变当前设计。

目前是：

\[
24\times2\times11
\]

个 learnable queries 去 cross-attend scene memory。

新版本可以直接从：

\[
Z_H
\]

开始。

每个 hand token 自己负责预测自己的未来：

\[
z_{h,k}
+
E_{\rm time}(j)
\rightarrow
\Delta p_{h,k,j}.
\]

即：

\[
\hat V
=
F_H(Z_H)
\]

输出：

\[
\boxed{
\hat V\in
\mathbb R^{24\times2\times11\times3}
}
\]

这样：

\[
\text{finger token}
\]

不是 decoder 最后才凭空出现的 query。

而是：

\[
\boxed{
\text{从输入到输出都始终保留自己的身份}
}
\]

我认为这比当前 raw-hand residual 注入更合理。

---

# 7. Object Goal 放在哪里？

这是我们和 PointWAM 最大的特殊点之一。

PointWAM 用的是 language instruction，因此 language 不会直接泄露未来 object coordinates。它可以让 scene/hand 两个 head 都读 language。[Chrockey](https://chrockey.github.io/PointWAM/?utm_source=chatgpt.com)

但我们不能照抄。

因为我们的：

\[
G
=
P^{obj}_{t+24}
\]

实际上就是**未来物体位置**。

如果我们让 Scene Future Head 直接看到这个：

\[
G=P^{obj}_{future},
\]

同时又要求：

\[
\hat U\approx U^{GT},
\]

那 World task 很容易退化成：

> 把 goal 抄到 output。

这会构成非常强的 supervision leakage。

---

## 所以我建议一个非常重要的设计

Shared backbone 只吃：

\[
\boxed{
\text{observed scene}
+
\text{observed hand}
}
\]

先得到：

\[
Z.
\]

然后：

### Scene head

只用：

\[
Z.
\]

\[
\boxed{
\hat U=W(Z)
}
\]

完全 goal-blind。

---

### Hand head

则允许：

\[
Z+G
\]

预测：

\[
\boxed{
\hat V=A(Z,G).
}
\]

具体可以让 object goal tokens cross-attend 到 hand tokens：

\[
\tilde Z_H
=
CrossAttn(
Z_H,G
).
\]

再：

\[
\hat V=F_H(\tilde Z_H).
\]

所以完整图变成：

```text
                   scene observations
                          │
                          ▼
                        PTv3
                          │
hand observations ────────┤
                          ▼
               Shared Interaction Trunk
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
             scene Z           hand Z
                 │                 │
                 │            Object Goal
                 │                 │
                 │            Cross Attention
                 │                 │
                 ▼                 ▼
             U_hat             V_hat / τ
```

我认为这是**最适合我们项目的 PointWAM adaptation**。

---

# 8. 为什么我不建议让 goal 进入 shared trunk？

因为这可以保证：

\[
\hat U
\]

是真的：

> 根据当前 interaction state 预测未来世界，

而不是：

> 根据用户希望物体去哪预测物体去哪。

也就是说它仍然保持：

\[
\boxed{\text{forecast}}
\]

而不是：

\[
\boxed{\text{desired outcome decoder}}.
\]

这对后面 World Model/RL 非常重要。

---

# 9. Human 三域预训练的 loss

第一阶段完全拿掉：

\[
W(H,\tau)
\]

这套 action-conditioned World。

只训练：

\[
\boxed{
L_{\rm pretrain}
=
\lambda_H L_{\rm hand}
+
\lambda_S L_{\rm scene}.
}
\]

Hand：

\[
L_{\rm hand}
=
\frac{1}{N_H}
\sum
m^H_{k,h}
\|
\hat V_{k,h}-V^*_{k,h}
\|^2.
\]

Scene：

\[
L_{\rm scene}
=
\frac{1}{N_S}
\sum
w_i m^S_{i,h}
\|
\hat U_{i,h}-U^*_{i,h}
\|^2.
\]

moving/static 权重继续保留。

---

## Hand 不允许被 scene 点数量淹没

这一点非常重要。

不要：

\[
L=
\frac{
\sum_{\rm scene}L+
\sum_{\rm hand}L
}{
N_{\rm scene}+N_{\rm hand}
}.
\]

因为 scene point 数量太大。

而应该分别归一化：

\[
\bar L_H
\]

和：

\[
\bar L_S.
\]

然后：

\[
L=\bar L_H+\lambda_S\bar L_S.
\]

第一版我甚至建议：

\[
\lambda_H=\lambda_S=1
\]

就够了。

因为这样 22 个 hand points 作为一个完整 task，和整个 scene task 获得同一级梯度权重。

如果之后确实 hand 学得慢，再做：

\[
2:1
\]

对照。

不要一上来拍脑袋把 fingertip loss 乘 10。

---

# 10. PTv3 怎么训练

我不建议再冻结 PTv3。

但也不应该随机重训。

直接加载我们当前 PointWorld/PTv3 权重：

\[
\theta_{\rm PTv3}^{PW}.
\]

训练时使用 discriminative LR：

\[
\eta_{\rm PTv3}\approx10^{-5}
\]

新模块：

\[
\eta_{\rm new}\approx10^{-4}.
\]

新模块包括：

\[
\text{HandTokenEncoder}
\]

\[
\text{InteractionTransformer}
\]

\[
\text{HandHead}
\]

\[
\text{SceneHead}
\]

\[
\text{GoalConditioner}.
\]

这样 PTv3 可以继续学：

\[
\text{哪些 scene geometry 对 hand interaction 有用}.
\]

而不是像现在 C 一样：

\[
\boxed{PTv3\ frozen}.
\]

---

# 11. 三域数据完全可以继续用

输入还是：

\[
OakInk2+GRAB+ARCTIC.
\]

甚至当前 `MixedWindows`、采样比例、train/val 切分基本都可以复用。

需要变的主要是 label assembly。

现有：

\[
\text{hand future}
\]

直接继续作为：

\[
V^*.
\]

现有 object SE(3)/effect：

\[
T_{obj,t+h}
\]

作用在 canonical object points 上：

\[
p_{i,t+h}
=
T_{obj,t+h}p_i^0
\]

即可生成：

\[
U^*.
\]

所以不需要重新处理 4.38M windows。

---

# 12. Robot fine-tuning 才是这版真正关键的地方

Human pretraining 后不要再走现在这种：

```text
freeze backbone
→ adapter
→ decoder
→ repair R
→ realign actor
```

新路线一次做成一个完整 pair。

机器人 demonstration 提供：

\[
(s_t,V^*,U^*,a^*).
\]

其中：

\[
V^*
\]

由 Inspire FK 得到，

\[
U^*
\]

由真实 object pose 得到，

\[
a^*
\]

是实际执行 command。

---

## Robot loss

我建议：

\[
\boxed{
L_{\rm robot}
=
L_H
+
\lambda_S L_S
+
\lambda_A L_{\rm action}^{pred}
+
\lambda_R L_{\rm action}^{GT}
}
\]

其中：

### predicted-\(\tau\) action loss

\[
L_{\rm action}^{pred}
=
\|
R(s,\hat V)-a^*
\|.
\]

这保持 actor/R end-to-end alignment。

---

### GT-\(\tau\) R anchor

同时：

\[
L_{\rm action}^{GT}
=
\|
R(s,V^*)-a^*
\|.
\]

这个是我们相对于 PointWAM **必须额外加入的保险**。

因为上一轮 joint training 已经明确出现：

\[
R(V^*)
:
10.6\rightarrow44.0\text{ mm}.
\]

说明只训练：

\[
R(s,\hat V)
\]

会让 R 和 actor 一起漂。

所以：

\[
\boxed{
L_R^{GT}
}
\]

必须把 R 钉在“真实物理 hand trajectory → action”的语义上。

---

# 13. Robot stage 哪些模块更新？

这次不要只更新 decoder。

我建议：

| 模块 | Robot finetune |
|---|---|
| PTv3 | ✅ 小 LR |
| Hand Token Encoder | ✅ |
| Shared Interaction Trunk | ✅ |
| Hand trajectory head | ✅ |
| Scene trajectory head | ✅ |
| Goal conditioner | ✅ |
| Chunk-R | ✅ |
| 旧 action-conditioned World | ❌ 不参与 |

也就是说真正做到：

\[
\boxed{
\text{full shared forecaster + R finetune}
}
\]

这才是真正接近 PointWAM robot stage。

---

# 14. Robot proprioception 第一版反而不要塞回 forecaster

这点我会和我们现在的 ref2_8 做不同设计。

第一版保持：

\[
A:
(scene,hand,goal)
\rightarrow V.
\]

而：

\[
q,\dot q,a_{t-1}
\]

只进入：

\[
R(s,V).
\]

为什么？

因为我们现在是在做一个**干净 PointWAM-style baseline**。

否则：

\[
\text{shared forecaster}
\]

又立即 robot-specific。

先测试：

\[
\boxed{
\text{embodiment-independent hand future}
+
\text{embodiment-specific retargeter}
}
\]

是否能工作。

如果不够，再做：

\[
+robot\ proprio
\]

作为明确 ablation。

---

# 15. Action execution 也建议顺便恢复 chunk 语义

Chunk-R 仍可以保持现在：

\[
8\times12.
\]

但不要默认：

\[
K_{\rm exec}=1.
\]

至少正式比较：

\[
K_{\rm exec}=1
\]

和：

\[
K_{\rm exec}=4
\]

甚至：

\[
K_{\rm exec}=8.
\]

因为 PointWAM 本身是 action-buffer/chunk deployment，而不是每 tick 把计划完全丢掉。官方也明确把 retargeter 描述成输出 action chunk。[Chrockey](https://chrockey.github.io/PointWAM/?utm_source=chatgpt.com)

这很可能对：

\[
closure\rightarrow preload\rightarrow lift
\]

这种 contact commitment 很重要。

---

# 16. 那我们原来的 action-conditioned World 怎么办？

**不要删。**

但把它从这版 baseline 中完全解耦。

形成两个模型：

### Shared WAM

\[
F(H)
\rightarrow
(U,V).
\]

负责：

\[
\boxed{\text{policy representation + supervised baseline}}
\]

---

### Counterfactual World

继续保留：

\[
W_{\rm cf}(H,\tau)
\rightarrow E.
\]

负责：

\[
\boxed{\text{counterfactual prediction}}
\]

以后可以做：

\[
W(H,\tau_1),
W(H,\tau_2),...
\]

用于：

- evaluator；
- MPC；
- RL critic；
- candidate ranking。

但第一阶段：

\[
\boxed{
\nabla_{\tau}L_{W_{\rm cf}}=0
}
\]

不要再让它直接拉 actor。

这样整个研究逻辑会非常漂亮：

```text
Shared WAM
   │
   ├── produces strong motion prior
   │
   ▼
robot policy
   │
   ├────────────────────┐
   ▼                    ▼
Residual RL        Counterfactual World
                        │
                        ▼
                critic / planning
```

---

# 17. 我会怎么实际推进

我建议按下面这一条流水线做，不再横向同时改十个变量：

1. **Shared-WAM V0 工程版**：保留 PTv3，新增 22 hand tokens + 4-layer interaction trunk + hand/scene 双头；验证两个 loss 都能进入 PTv3，且不存在 \(V\rightarrow U\) 路径。
2. **三域 pretrain**：直接从当前 PTv3 checkpoint 初始化；先跑短规模，看 hand first8/all24 和 moving/static scene EPE 是否都正常，再扩到正式规模。
3. **Airplane overfit sanity**：只用完整成功 airplane robot demos，full forecaster + R 联合微调，加入 \(R(s,V^*)\to a^*\) anchor；暂时不要 human replay，不要旧 World loss，先证明单任务能抓。
4. **Robot mixed finetune**：airplane overfit 成功后，再加入三域 replay，检验 human prior 保留和 robot success。
5. **Chunk execution A/B**：固定同一个 checkpoint，只比较 \(K_{\rm exec}=1\) 与 4/8。
6. **和旧架构对比**：旧 GoalPointWAM vs Shared-WAM，在相同 airplane data/updates 上比较 hand EPE、GT-\(\tau\) R error、pred-\(\tau\) action error、held/lift。
7. **Residual PPO 并行**：当前旧 WAM+old R 继续跑 RL，不等待 Shared-WAM 完成；之后再把 RL base 替换成 Shared-WAM，看 sample efficiency 是否进一步改善。

---

# 18. 这个方案真正的新点在哪里

它不是简单“复制 PointWAM”。

PointWAM 是：

\[
(scene,hand,language)
\rightarrow
(U,V).
\]

我们的会是：

\[
\boxed{
Z=F_{\rm shared}(scene,hand)
}
\]

\[
\boxed{
U=F_U(Z)
}
\]

\[
\boxed{
V=F_V(Z,G_{\rm object})
}
\]

其中：

\[
G_{\rm object}
\]

只告诉 hand policy：

> 希望物体怎么运动。

World branch 仍然只预测：

> 根据当前真实 interaction 状态，世界实际上将怎么运动。

这其实很符合我们的最终研究目标：

\[
\boxed{
\text{goal-conditioned action}
+
\text{goal-blind world understanding}
}
\]

而且没有强迫：

\[
V\rightarrow U.
\]

---

## 我最推荐的核心定义

如果最后要把这个方法写成一句公式，我会定成：

\[
\boxed{
Z_t
=
F_{\theta}
(
S_{\le t},
H_{\le t}
)
}
\]

世界预测：

\[
\boxed{
\hat U_{t:t+K}
=
D_U(Z_t)
}
\]

动作/手预测：

\[
\boxed{
\hat V_{t:t+K}
=
D_V(Z_t,G_t)
}
\]

机器人：

\[
\boxed{
\hat A_{t:t+K_a}
=
R_\phi(s_t,\hat V_{t:t+K})
}
\]

训练：

\[
\boxed{
L
=
L_U
+
L_V
+
L_{A,\hat V}
+
L_{A,V^*}.
}
\]

而我们现有 counterfactual World：

\[
W_{\rm cf}(H,V)\rightarrow E
\]

独立保留，**不参与这版 supervised baseline 的 actor gradient**。

我认为这会是目前最干净、最值得真正做一版的架构：既保留 PointWAM 已验证有效的“shared scene-hand future representation”，又保留我们后续 World Model + RL 的研究空间，而且第一次真正让 22 个 hand points 从头到尾成为网络里的一级实体，而不是 decoder 的附加条件。