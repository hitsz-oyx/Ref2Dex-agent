关键点先说：**PointWAM 并不是把 115 万段视频直接当 RGB 视频丢进模型训练。**  
它先用一套离线 preprocessing，把人类视频统一转换成：

\[
\boxed{
\text{当前3D场景}
+
\text{当前手}
+
\text{语言}
\rightarrow
\text{未来3D场景轨迹}
+
\text{未来3D手轨迹}
}
\]

然后才做大规模预训练。也就是说，它把“视频学习”变成了一个非常规整的 **3D trajectory prediction problem**。[arXiv](https://arxiv.org/abs/2610.02840)

可以把整个 pipeline 画成：

```text
EgoDex / VITRA 人类视频
        │
        ├───────────── Hand branch ─────────────┐
        │                                       │
        │   3D human hand annotations           │
        │            ↓                          │
        │   semantic hand keypoints             │
        │            ↓                          │
        │      future hand τ_hand               │
        │                                       │
        └──────────── Scene branch ─────────────┤
                                                │
 RGB frames                                     │
     │                                          │
     ├─ VGGT-Ω → monocular depth                │
     │                                          │
     ├─ CoTracker3 → dense 2D point tracks      │
     │                                          │
     └─ camera pose + intrinsics                 │
              ↓                                 │
        lift 2D tracks to 3D                    │
              ↓                                 │
       future scene τ_scene                     │
                                                ▼
                      ┌──────────────────────────────┐
                      │ PointWAM trajectory forecaster│
                      │                              │
                      │ P0 + Q0 + language           │
                      │          ↓                   │
                      │ shared 3D Transformer        │
                      │        ↙       ↘             │
                      │ τ_scene      τ_hand           │
                      └──────────────────────────────┘
                                   │
                       Human pretraining
                                   │
                                   ▼
                       Robot demonstrations
                                   │
                         + robot action A
                                   ↓
                       joint fine-tuning
                                   ↓
                    τ_hand + state → action
```

---

## 1. 第一件事：先把长视频切成统一的小窗口

他们并不是拿整段几分钟的视频训练。

对 EgoDex / VITRA：

- 视频先按 **10 Hz** 处理；
- 每个训练 clip 长 **1 秒**；
- 即原始：

\[
11\text{ frames}
\]

- 窗口每 **0.5 秒**滑一次；
- 如果这一秒内 wrist 总位移小于 **2 cm**，就认为基本没操作，丢掉；
- 再把这 11 帧**线性插值成 31 帧**：

\[
t_0,t_1,\ldots,t_{30}.
\]

于是：

\[
T=30.
\]

这样就和 DexJoCo 机器人数据的 30-step trajectory horizon 对齐了。人类的 30 步覆盖约 1 秒，DexJoCo 的 30 步覆盖 0.6 秒；他们并没有要求真实时间尺度完全一样，只统一 trajectory tensor 的长度。[arXiv](https://arxiv.org/abs/2610.02840)

这里有一个数据量上的细节：

- EgoDex 原始约 **324K episodes**；
- 切窗以后能得到约 **5.2M clips**；
- VITRA 提供约 **824K 个较短 episode**；
- 论文用于预训练的数据源总计约 **1.148M episodes**。[arXiv](https://arxiv.org/abs/2610.02840)

所以它的大规模并不是“115 万个 MP4 逐帧端到端训”，而是先离线转换，训练时采这些标准化窗口。

---

# 2. Hand branch：视频里的人手怎么变成机器人可以学的 \(\tau_{\rm hand}\)

这是最省事的一条。

EgoDex 已经有比较强的 3D hand annotation：

\[
25\text{ joints / hand}
\]

并且每个 joint 有 SE(3) 信息。

VITRA 则会重建：

\[
21\text{ joints / visible hand}.
\]

PointWAM 并不直接拿所有人体关节，而是把它们统一映射成少量具有语义对应关系的 **hand keypoints**。于是无论：

- human hand；
- Allegro；
- Inspire；

最后都表示为类似：

\[
Q_t=
\{q_t^1,\ldots,q_t^K\},
\qquad q\in\mathbb R^3.
\]

因此一条人类视频天然就能产生：

\[
V^*=
\{
q_{t+\tau}^{k}-q_t^{k}
\}.
\]

也就是：

\[
\boxed{
\tau_{\rm hand}
\in
\mathbb R^{T\times K\times3}.
}
\]

论文明确说，人类视频 hand trajectory 就来自 EgoDex 的 3D hand poses 和 VITRA 重建的 hand poses，然后映射到机器人定义的 semantic keypoints。[arXiv](https://arxiv.org/abs/2610.02840)

这其实是为什么他们能把 human data 和 robot data 混到一个任务里的核心。

---

# 3. Scene branch 才是最有意思的部分

人类视频没有：

\[
\text{object 6DoF GT}
\]

也没有：

\[
\text{simulator rigid body state}.
\]

所以作者**根本不去识别“这是不是杯子、香蕉、锤子”**。

而是直接做：

\[
\boxed{\text{dense point tracking}}
\]

这点非常关键。

---

## 3.1 用 CoTracker3 得到 2D point tracks

对每个 1 秒 clip：

先 resize 到：

\[
640\times360.
\]

然后在**第一帧每隔 2 个 pixel 放一个 query point**。

所以第一帧有非常密集的：

\[
(x_i,y_i).
\]

CoTracker3 从第一帧开始往后跟踪：

\[
(x_i^0,y_i^0)
\rightarrow
(x_i^1,y_i^1)
\rightarrow\cdots
\rightarrow
(x_i^{30},y_i^{30}).
\]

作者专门说不需要 backward tracking，因为所有 query 都从第一帧开始。[arXiv](https://arxiv.org/abs/2610.02840)

这件事情非常巧：

他们根本不需要：

- SAM 把物体分出来；
- 给物体类别；
- 给 object ID；
- 知道哪个点属于哪个刚体。

只需要：

> “这个第一帧里的 pixel 后面去了哪里？”

---

# 4. 但 2D tracking 不够，所以还需要 depth

PointWAM 用：

\[
\boxed{\text{VGGT-}\Omega}
\]

给每一帧预测 monocular depth。

问题是 monocular depth 只有：

\[
\text{depth up to scale}.
\]

比如模型预测：

\[
\hat z=1.2
\]

你不知道实际是：

- 0.6 m，
- 1.2 m，
- 2.4 m。

所以他们用了一个非常实用的 trick：

## 用已经知道尺度的 3D 手来校准 depth scale

因为 EgoDex / VITRA 本身有 metric 3D hand。

把 hand joints 投影到图像：

\[
q_{3D}\rightarrow(x,y).
\]

然后比较：

\[
z_{\rm hand}^{GT}
\]

和 VGGT：

\[
z_{\rm VGGT}.
\]

计算比例：

\[
r_i=
\frac{z_i^{GT}}
{z_i^{pred}}.
\]

异常值去掉以后，取 median：

\[
s=
\operatorname{median}(r_i).
\]

然后：

\[
z^{metric}
=
s\,z^{pred}.
\]

于是 monocular depth 就被校准成真实米制深度了。[arXiv](https://arxiv.org/abs/2610.02840)

这个设计其实对我们特别有参考价值。

---

# 5. 2D trajectory + depth → 3D trajectory

现在每个 CoTracker point 都有：

\[
(x_t,y_t)
\]

和：

\[
z_t.
\]

再加 camera intrinsics：

\[
K
\]

就可以 unproject：

\[
p_t^{cam}
=
z_t
K^{-1}
\begin{bmatrix}
x_t\\
y_t\\
1
\end{bmatrix}.
\]

然后用 camera pose：

\[
T_{world\leftarrow cam}
\]

转换到统一 world frame：

\[
p_t^{world}
=
T_{world\leftarrow cam}
p_t^{cam}.
\]

因此一个 pixel track：

\[
(x_0,y_0)\rightarrow\cdots\rightarrow(x_{30},y_{30})
\]

最终变成：

\[
p_0^{3D}
\rightarrow
p_1^{3D}
\rightarrow
\cdots
\rightarrow
p_{30}^{3D}.
\]

论文明确就是用 **scaled depth + camera intrinsics + camera poses** 把 CoTracker tracks lift 到 world frame。[arXiv](https://arxiv.org/abs/2610.02840)

---

# 6. 然后不预测绝对位置，而预测 displacement

假设第一帧有一个 point：

\[
p_t^i.
\]

未来轨迹 target 是：

\[
u_{t,\tau}^{i*}
=
p_{t+\tau}^i-p_t^i.
\]

于是所有 scene point 的 label：

\[
U^*
=
\{
u_{t,\tau}^{i*}
\}.
\]

同理 hand：

\[
v_{t,\tau}^{k*}
=
q_{t+\tau}^k-q_t^k.
\]

所以一个 human training sample 最终就是：

\[
\boxed{
(P_t,Q_t,L,\;U^*,V^*)
}
\]

其中：

- \(P_t\)：当前 scene point cloud；
- \(Q_t\)：当前 hand keypoints；
- \(L\)：语言 instruction；
- \(U^*\)：未来 scene point displacement；
- \(V^*\)：未来 hand displacement。

注意：

\[
\boxed{\text{这里完全没有 robot action。}}
\]

---

# 7. 所以 Human Pretraining 实际训练什么？

只训练：

\[
(P_t,Q_t,L)
\rightarrow
(\hat U,\hat V).
\]

loss 大致是：

\[
L_{\rm human}
=
L_{\rm scene}
+
L_{\rm hand}.
\]

其中：

\[
L_{\rm hand}
=
\frac{1}{KT}
\sum_{k,\tau}
\left\|
v_{k,\tau}-v^*_{k,\tau}
\right\|_2^2
\]

scene 也是 point trajectory MSE，只不过用了加权：

\[
L_{\rm scene}
=
\sum_{i,\tau}
w_{i,\tau}
\|
u_{i,\tau}-u^*_{i,\tau}
\|^2.
\]

\(w\) 会更重视 moving points，但不会把 static points 完全删掉。[arXiv](https://arxiv.org/abs/2610.02840)

这点和我们之前 PointWorld 的一些讨论其实很像：

> **静止点不能全丢，否则模型不知道“什么应该保持不动”。**

---

# 8. 网络到底看到什么？

它不是给 Transformer 喂 12,000×30 个 future point。

模型输入只看 **当前帧**：

\[
P_t+Q_t+L.
\]

当前 scene 最多大约：

\[
12,000\text{ points}.
\]

然后：

```text
current RGB-D / reconstructed point cloud
                 ↓
           Mosaic3D encoder
                 ↓
          scene point features

current semantic hand keypoints
                 ↓
               MLP
                 ↓
           hand features

language instruction
                 ↓
        frozen text encoder

scene + hand
       ↓
1 cm voxelization
       ↓
3D Transformer / Volt
       ↓
shared spatiotemporal representation
       │
       ├── Scene Head
       │       ↓
       │   30-step scene flow
       │
       └── Hand Head
               ↓
           30-step hand flow
```

主模型训练时，Mosaic3D 和 text encoder 是冻结的；PointWAM 的 trajectory backbone 从头训练。主版本是 Volt-H，大约 **651M trainable parameters** 在 human pretraining 阶段。[arXiv](https://arxiv.org/abs/2610.02840)

---

# 9. 大规模训练本身其实没那么神秘

正式配置：

\[
1.148M\text{ human episodes}
\]

预训练：

\[
60,000\text{ iterations}.
\]

global batch：

\[
192.
\]

硬件：

\[
4\times \text{NVIDIA B200}.
\]

优化器：

\[
AdamW,
\quad lr=10^{-4},
\quad wd=0.01.
\]

bfloat16 mixed precision。

Human pretraining：

\[
\boxed{34\text{ hours}}
\]

左右。[arXiv](https://arxiv.org/abs/2610.02840)

所以严格来讲，它也不是把五百万个 clip 全完整过很多 epoch。

60k × 192：

\[
\approx11.5M
\]

个 sample exposure。

它更像从一个巨大的预处理 human trajectory pool 里不断采样。

---

# 10. 到这里还没有机器人

这是特别重要的。

Human stage：

\[
\boxed{
\text{Video}
\rightarrow
3D~scene/hand~trajectories
\rightarrow
PointWAM
}
\]

完全没有：

\[
A.
\]

所以 human video 并没有教它：

> Allegro 应该输出哪个 joint target。

它教的是：

> **“人在这种场景、这种任务下，手和环境未来通常怎样共同变化。”**

也就是学习一个强大的：

\[
\boxed{
(P,Q,L)\rightarrow
(\tau_{\rm world},\tau_{\rm hand})
}
\]

prior。

---

# 11. 第二阶段才用 DexJoCo

Human pretraining 完成以后：

```text
Human checkpoint
      ↓
DexJoCo robot demonstrations
```

DexJoCo 有：

\[
s_t,\quad A_t.
\]

而且 simulator 可以得到未来真实 rigid-body motion。

于是机器人数据也被转成和 human **完全一样的 trajectory representation**：

### hand

robot q：

\[
q_t
\xrightarrow{FK}
Q_t
\]

从而得到：

\[
V^*.
\]

### scene

replay simulation：

\[
\text{object rigid body SE(3)}
\]

直接移动第一帧 scene points，从而得到：

\[
U^*.
\]

所以 Human 和 Robot 共享完全同一个 supervision：

\[
\boxed{
U^*,V^*
}
\]

区别只是 Robot 额外多一个：

\[
A^*.
\] :chatgpt-content-reference{index="10"}


---

# 12. 然后才加 Action Retargeter

第二阶段完整结构：

\[
P,Q,L
\rightarrow
\hat V,\hat U
\]

同时：

\[
(\hat V,s_t)
\rightarrow
\hat A.
\]

loss：

\[
\boxed{
L=
L_{\rm scene}
+
L_{\rm hand}
+
L_{\rm action}.
}
\]

Human stage 没有 action head supervision；

Robot fine-tuning 时 retargeter **从头训练**。

他们在 DexJoCo 上 fine-tune：

\[
60K\text{ iterations}
\]

也是 4×B200，大约：

\[
19h.
\] :chatgpt-content-reference{index="11"}


---

# 13. 为什么这么做能吃大规模视频？

这里才是 PointWAM 真正聪明的地方。

如果你直接用人类视频训练：

\[
RGB\rightarrow robot~action
\]

不成立，因为视频里根本没有 robot action。

如果训练：

\[
RGB\rightarrow human~joint~angle
\]

又 embodiment-specific。

PointWAM 找了一个中间空间：

\[
\boxed{
\text{3D hand trajectory}
+
\text{3D scene trajectory}
}
\]

这个空间同时适用于：

```text
Human
  ↓
3D hand points + scene points

Allegro
  ↓
3D hand points + scene points

Inspire
  ↓
3D hand points + scene points
```

所以真正的数据桥梁是：

\[
\boxed{
\text{Human video}
\rightarrow
\text{embodiment-light 3D representation}
\leftarrow
\text{Robot demonstration}
}
\]

而不是把人类动作硬 retarget 成 Allegro action。

这也是为什么论文里 human pretraining 收益特别夸张：

只 Robot 训练：

\[
12.1\%
\]

EgoDex 预训练：

\[
60.1\%
\]

EgoDex + VITRA：

\[
69.0\%.
\]

数据从 324K → 736K → 1.15M episodes 时，性能也继续涨。[arXiv](https://arxiv.org/abs/2610.02840)

---

## 对我们现在最重要的一点

你可以发现，它和我们之前尝试 EgoDex 的方式其实有一个巨大区别。

我们之前很容易想成：

\[
\text{EgoDex}
\rightarrow
\tau_{\rm hand}
\rightarrow
\text{想办法变 Inspire action}.
\]

PointWAM **根本没有在 human stage 做这个事**。

它是：

\[
\boxed{
\text{EgoDex}
\rightarrow
(H_{\rm 3D},\tau_{\rm hand},\tau_{\rm scene})
\rightarrow
\text{训练 representation}
}
\]

一直等到机器人数据阶段，才解决：

\[
\tau_{\rm hand}\rightarrow A_{\rm robot}.
\]

所以人类视频的作用不是给 robot policy 提供动作标签，而是提供：

> **“手怎么运动时，环境会怎么响应”这一百万级别的几何先验。**

这也是我觉得我们之前最值得重新审视的地方：**EgoDex 对我们不一定应该承担 \(A\) 的监督，它更适合作为 PointWorld / PointWAM backbone 的大规模 \(H\to(\tau_{\rm hand},\tau_{\rm object/scene})\) 预训练数据。** 

如果把 PointWAM 的 video pipeline 压缩成一句最核心的公式，就是：

\[
\boxed{
\text{RGB video}
\overset{\text{VGGT + CoTracker + hand pose}}{\longrightarrow}
(P_t,Q_t,U^*,V^*)
\overset{\text{pretrain}}{\longrightarrow}
f_\theta
}
\]

然后：

\[
\boxed{
f_\theta
+\text{robot demo}
\rightarrow
(\hat V,s)\rightarrow A.
}
\]

这就是完整的“视频规模化 → 机器人控制”桥梁。