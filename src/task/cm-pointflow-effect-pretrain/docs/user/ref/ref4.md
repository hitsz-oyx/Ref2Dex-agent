我仔细按代码链路看了一遍。结论不是“工程没问题，继续等 40k”这么简单。

**目前没有看到坐标系、GT 泄漏、SE(3) 计算这类会让结果完全无效的硬错误，但有两个比较严重的设计/实现错配，其中第一个我认为足以影响 H+A 是否能学到 action。**

## 1. 最大问题：24 步 action 在进入 PTv3 时被 1 cm voxel 合并了

现在 `pointworld.py` 是：

```python
coord = torch.cat((xyz, a[..., :3].reshape(B, 528, 3)), 1)
...
grid = torch.floor((coords-coords.amin(0))/.01).long()

unique, inverse = torch.unique(keys, dim=0, return_inverse=True)

point = self.backbone(
    coord=mean_groups(coords, inverse, len(unique)),
    feat=mean_groups(inputs, inverse, len(unique)),
    ...
)
```

这意味着：

> **同一个 1 cm voxel 里的所有 scene/action points 在进入 PTv3 前直接求平均。**

问题是 action 是：

\[
24\times22
\]

个时空点。

例如某个 fingertip：

\[
q_1,q_2,q_3,q_4,q_5
\]

如果 5 帧都在同一个 1 cm voxel：

```text
q1  t=1
q2  t=2
q3  t=3
q4  t=4
q5  t=5
       ↓
    一个 voxel
       ↓
xyz 平均
feature 平均
time embedding 也平均
       ↓
一个 PTv3 token
```

所以：

\[
\boxed{\text{trajectory 被压成 occupancy-like representation}}
\]

而不是保留完整的：

\[
A_{1:24}
\]

---

### 这个问题在我们的数据里不会罕见

之前 OakInk2 100-sequence audit 已经统计：

120 Hz 下 hand adjacent displacement：

\[
p99=6.49\text{ mm}
\]

现在下采样到30 Hz，当然位移会变大，但大量：

- palm anchors；
- 正在稳定抓持的 fingertips；
- 缓慢 manipulation；

仍然很容易连续多个 timestep 落在同一个 1 cm voxel。

而且我们每只手里还有 6 个 palm/MCP anchor，本来就比 fingertip 动得慢。

所以528个 action points真正进入 PTv3 后可能远少于528个独立 action spatial tokens。

仓库自己做 interface probe 时已经看到：

\[
7244\ raw\ points
\rightarrow1788\ unique\ voxels
\]

虽然这包含 scene points，但压缩程度已经很明显。

---

## 更严重的是：time embedding 也被平均

action feature 本来是：

\[
f(q_\tau)+e_\tau+e_{\rm hand}+e_{\rm keypoint}
\]

但是 voxel pooling 之后：

\[
\bar f=
\frac1n\sum_{\tau\in voxel}
(f(q_\tau)+e_\tau+\cdots)
\]

网络只看到一个“平均时间”。

例如：

```text
轨迹 A:
t1→t2→t3

轨迹 B:
t3→t2→t1
```

如果都经过同一组 voxel，pooling 后可能非常接近。

这恰好会削弱我们最关心的：

\[
\boxed{\text{action chunk 的方向和时序}}
\]

所以目前 step=1000：

\[
H+A\not>H
\]

我不会马上解释成“人手 action 没信息”。

**当前 action representation 在 backbone 入口就已经损失了一部分时序信息。**

---

# 2. PointWorld 的 5 mm motion selector 被机械搬到了 30 Hz OakInk2

现在 loss：

```python
delta = diff(actual)
weight = sigmoid(
    5/.005 * (norm(delta) - .005)
)
```

也就是：

\[
w=
\sigma
\left[
1000(\|\Delta p_t\|-0.005)
\right]
\]

看看几个数：

\[
\Delta=0:
w\approx0.0067
\]

\[
\Delta=2mm:
w\approx0.047
\]

\[
\Delta=5mm:
w=0.5
\]

\[
\Delta=8mm:
w\approx0.953
\]

所以它实际上强烈强调：

\[
\boxed{\text{单帧运动}>5mm}
\]

在30 Hz下相当于速度约：

\[
0.005\times30=0.15m/s
\]

也就是 **15 cm/s**。

对于很多：

- 稳定抓持；
- 慢速旋转；
- 夹持；
- 开盖；
- 精细 manipulation；

物体每33 ms根本不一定移动5 mm。

---

### 这和我们 window sampling 的定义还不一致

训练 window 被判为 moving 的条件是：

\[
\exists\tau:
\|p_\tau-p_0\|>2mm
\]

或者：

\[
\theta>0.02rad
\]

这是**累计运动**。

但真正 loss weighting 用的是：

\[
\|p_\tau-p_{\tau-1}\|>5mm
\]

这是**单帧增量运动**。

所以完全可能：

```text
这个 window：
0.8s 内物体移动 40 mm
→ 明显 moving sample

但是每帧：
40mm / 24 ≈ 1.67mm

PointWorld weight:
≈ sigmoid(-3.33)
≈ 0.034
```

也就是说：

> sampler 费劲把它当“重要 moving sample”抽进来，loss 又把它乘成约3%的权重。

这是明显的 protocol mismatch。

---

# 3. 一个好消息：坐标合同我没有发现错误

这部分实现是对的。

当前 anchor frame：

\[
C=T_{\rm anchor,t}^{-1}
\]

当前 object：

\[
T^c_{m,t}=CT_{m,t}
\]

future：

\[
T^c_{m,t+\tau}=CT_{m,t+\tau}
\]

effect：

\[
E_{m,\tau}
=
T^c_{m,t+\tau}
(T^c_{m,t})^{-1}
\]

然后当前 object points：

\[
P_{m,t}=T^c_{m,t}P_m
\]

应用：

\[
E_{m,\tau}P_{m,t}
=
T^c_{m,t+\tau}P_m
\]

数学上完全成立。

没有发现“未来每帧重新 object-centric 导致 effect 被消掉”的问题。

---

# 4. Action 坐标也基本正确

future hand：

\[
Q^c_{t+\tau}=C Q^{world}_{t+\tau}
\]

使用的也是**固定当前 anchor frame**。

然后 action feature：

\[
[
Q^c_{t+\tau},
Q^c_{t+\tau}-Q^c_t,
Q^c_{t+\tau}-Q^c_{t+\tau-1}
]
\]

这是合理的。

而且 shuffle 后代码明确**不重新相对 recipient H 计算 action**。

这一点是对的，因为 shuffled action 就应该作为完整 donor trajectory。

---

# 5. Shuffle control 本身也基本正确

batch 内先按：

```text
right-only
left-only
both
```

做 derangement。

singleton 找不到 donor 时，再从整个 dataset 随机找：

- 相同 hand presence；
- 不同 sample_id。

所以不是那种：

> batch size=2导致shuffle经常等于自己

的假 shuffle。

这一块我没有发现明显 bug。

---

# 6. 但 H+A 的 action 全局 summary 又丢了一次时序

PTv3之后：

```python
hand = packed[:, N:].masked_fill(...).amax(1)
summary = hand
```

也就是对528个 action point：

\[
z_A=\max_{\tau,h,k}f_{\tau,h,k}
\]

然后：

\[
local_{scene}
\leftarrow
local_{scene}+\operatorname{FiLM}(z_A)
\]

这进一步把：

- timestep；
- hand；
- keypoint；

全部 global max 掉。

当然 action 和 scene 已经在 PTv3 内交互，所以不是完全靠这个 summary。

但如果 PTv3前面又做了1 cm时空混合，这个 global max 就更加不能补回来。

---

# 7. 还有一个实验设计问题：当前 best checkpoint 选择方式偏向单一指标

现在：

```python
score =
model/anchor/cat0/h24/point_epe
```

只用：

\[
\boxed{\text{moving anchor h24 EPE}}
\]

选 best。

这本身没错，因为是主指标。

但是以后真正判断 action sensitivity 时，不能只比较三个 `best.pt`。

因为：

```text
History best可能在 step 13000
Action best可能在 step 22000
Shuffle best可能在 step 9000
```

这样不再是 matched training state。

仓库目前 final-screen 已经规定：

> equal-update final checkpoint 是 primary。

这点是正确的。**正式结论必须用 final/equal-step，不要用各自 best 做三臂主比较。**

---

# 8. normalization 我认为基本正确

这部分反而做得不错。

它按 horizon × axis 统计：

\[
\mu_{\tau xyz},\sigma_{\tau xyz}
\]

而不是一个全局 scale。

h24 flow std：

\[
21.41/19.90/20.72mm
\]

translation std：

\[
25.81/24.07/25.56mm
\]

这已经解决了上一版 raw-meter loss 的严重尺度问题。

而且 stats：

- 只来自 train；
- 4096 balanced windows；
- hash freeze；
- checkpoint带 stats identity。

没有看到 leakage。

---

# 9. 一个较小的问题：rotation loss 也继承了 motion selector

现在：

```python
ow = weight.mean(-1)
...
rotation_loss *= ow
```

而 `weight` 完全根据：

\[
\text{point translational displacement between adjacent frames}
\]

计算。

对于**原地旋转**：

\[
\Delta t\approx0,\qquad
\Delta R\neq0
\]

如果物体尺寸较小，表面逐帧 displacement 仍可能低于5 mm。

于是 rotation supervision也会被削弱。

这对 OakInk2 里的：

- 转瓶盖；
- 旋钮；
- 工具旋转；

尤其不理想。

---

# 我的判断

现在工程不是“有一个代码写错导致结果全假”。

更准确是：

### 硬合同

这些目前看是正确的：

- 30 Hz / K24；
- current anchor frame；
- future effect；
- hand keypoint顺序 right/left；
- mask；
- SE(3)；
- sequence split；
- normalization；
- shuffle donor；
- 无 label leakage。

### 真正可疑

有两个：

\[
\boxed{\textbf{A. 1cm voxel 把 action 的时间维混进空间维}}
\]

以及：

\[
\boxed{\textbf{B. PointWorld 5mm/frame selector 与30Hz OakInk2不匹配}}
\]

其中 A 更像**架构接口问题**，B 更像**训练 recipe domain mismatch**。

---

## 我建议现在怎么办

**不要停当前三臂。**

它已经跑了，而且作为“严格 PointWorld recipe adaptation”的 V1 很有价值。

但是我建议马上并行做两个**不需要长训练的数据诊断**。

第一个统计：

\[
\frac{\#\text{unique action voxels}}
{\#\text{valid action points}}
\]

并按：

- fingertip / palm；
- timestep；
- moving/static；
- single/bimanual；

统计。

如果528个 action points最后平均只有比如：

\[
150\sim250
\]

个 unique action voxels，那这个问题就不能忽略。

更关键的是统计：

> **同一个 `(hand,keypoint)` 的24步里，有多少 timestep发生 voxel collision。**

第二个直接统计当前训练集的 motion selector 权重：

\[
w_{\tau,m,p}
\]

报告：

\[
p10,p50,p90
\]

以及：

- moving-anchor 平均 weight；
- program window平均 weight；
- rotation-dominant window平均 weight。

我怀疑 moving-anchor 的 median weight 会明显偏低。

---

如果这两个诊断证实，我下一版不会换掉 PTv3，而是只修两个地方：

**action voxel key 加时间维：**

\[
(batch,\tau,x,y,z)
\]

而不是：

\[
(batch,x,y,z)
\]

至少让不同 timestep 不会被 voxel pooling 合并；scene 的 \(\tau=0\)。

然后 motion weighting 改成更符合我们任务的**累计 effect magnitude**，或者把阈值按30 Hz重新校准，而不是机械复制 PointWorld 的5 mm/frame。

这两个修改比现在继续讨论“50M是不是太大”重要得多。因为目前最大风险不是模型容量，而是**我们把 PointWorld 的实现搬过来时，把它的数据时间尺度假设也一起搬过来了。**