我查完之后，结论比较明确：**我们的 \(C\) 不应该自己从零定义。最合适的是把 ContactGen、GenDexGrasp、D(R,O) Grasp 和 ContactExplorer 的共同部分抽出来，再按你们“11 个 canonical hand points + 完整 3D object surface”做简化。**

现有工作大概形成了三种交互表示：

| 工作 | 交互表示核心 | 对我们的价值 |
|---|---|---|
| ContactGen | contact location + hand part + contact direction | **最适合定义 \(C\) 的语义** |
| GenDexGrasp | object xyz + normal + contact value | **最简单、跨手型** |
| D(R,O) Grasp | robot-object 点对点距离矩阵 | **连续关系，不局限于二值 contact** |
| ContactExplorer | 哪个 hand keypoint 接触哪个 object region | **和我们 11 hand points 最像** |

ContactGen 明确把交互拆成三部分：**contact map、hand-part map、direction map**；而且源码里就是 `contacts_object / partition_object / uv_object` 三路输出。[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_ContactGen_Generative_Contact_Modeling_for_Grasp_Generation_ICCV_2023_paper.html?utm_source=chatgpt.com) GenDexGrasp 则证明只用 object-centric contact map 就能作为不同机器人手之间的中间表示；源码 `CMapAdam.py` 里的 `contact_map_goal` 实际就是 `xyz(3) + normal(3) + contact_value(1)`。[GitHub](https://github.com/tengyu-liu/GenDexGrasp?utm_source=chatgpt.com) D(R,O) 更进一步，直接把 robot point cloud 和 object point cloud 的点对点距离矩阵作为交互关系，因此避免了“接触/不接触”这种过硬的离散化。[GitHub](https://github.com/Jzzzi/DRO-Grasp?utm_source=chatgpt.com) ContactExplorer 则直接用 hand keypoints 和 object surface points，维护“哪些 finger/keypoint 接触哪些 object regions”，并用 surface distance + contact force 得到 contact mask，这和我们的数据结构非常接近。[GitHub](https://github.com/ruoyiqiao/ContactExplorer?utm_source=chatgpt.com)

---

## 我建议我们的 \(C\) 不要定义成一个抽象 latent

而是定义成一个**可解释的 object-centric hand-object interaction field**。

我们已经有 canonical 11 个 hand points，所以 ContactGen 里的 `hand part map` 其实天然有了：

\[
i\in\{1,\dots,11\}
\]

本身就表示：

> wrist / thumb / index / middle ... 哪一个 canonical point。

因此没必要额外预测一个 hand-part 分类。

对每个未来时刻 \(k\)、每个 canonical hand point \(i\)，我建议定义：

\[
\boxed{
C_{k,i}
=
[
c_{k,i},
p^{obj}_{k,i},
n^{obj}_{k,i},
v^n_{k,i},
v^t_{k,i}
]
}
\]

这几个量分别对应：

\[
c_{k,i}\in[0,1]
\]

**contact confidence**；

\[
p^{obj}_{k,i}\in\mathbb R^3
\]

这个 hand point 在 object surface 上对应的目标接触位置；

\[
n^{obj}_{k,i}\in\mathbb R^3
\]

该位置的 object surface normal；

\[
v^n_{k,i}
\]

手相对物体沿法向的 approach / separation velocity；

\[
v^t_{k,i}
\]

切向 slip velocity magnitude。

也就是说每个 hand point 大约：

\[
1+3+3+1+1=9D
\]

---

# 为什么是这五种？

它们基本对应已有工作的交集。

### ① Contact location

这是所有 contact-based grasp 工作都有的。

ContactGen：

\[
\text{contact map}
\]

GenDexGrasp：

\[
\text{contact value on object surface}
\]

ContactExplorer：

\[
\text{hand keypoint}\leftrightarrow\text{object region}
\]

所以这一项必须有。

---

### ② Hand part

ContactGen 专门预测：

\[
\text{part map}
\]

因为“杯子这个位置被哪个手指接触”很重要。[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_ContactGen_Generative_Contact_Modeling_for_Grasp_Generation_ICCV_2023_paper.html?utm_source=chatgpt.com)

但我们已经有：

\[
i=1,\dots,11
\]

所以：

\[
\boxed{\text{hand-part identity 不用再预测}}
\]

直接由 keypoint ID 给。

这其实是我们比 ContactGen 更简单的地方。

---

### ③ Surface normal / direction

ContactGen 还有一个 direction map。

因为只有：

> 食指接触杯子右边

仍然不够。

不同方向贴上去，形成的 grasp 完全不同。

而 GenDexGrasp 的 contact map 本身就显式带 object normal；源码也是：

```text
xyz | normal | contact_value
 3  |   3    |      1
```

所以我觉得：

\[
n^{obj}
\]

必须留下。[GitHub](https://github.com/tengyu-liu/GenDexGrasp?utm_source=chatgpt.com)

---

### ④ Distance / contact confidence

这里我不建议一开始用硬：

\[
c=\mathbf 1[d<10mm]
\]

而是受 D(R,O) 启发，用连续距离。

D(R,O) 的核心甚至就是：

\[
D_{ij}
=
\|h_i-o_j\|
\]

整个 hand-object relation 用距离表示。它的源码 `multilateration.py` 明确就是 point-wise relative distance matrix。[GitHub](https://github.com/zhenyuwei2003/DRO-Grasp?utm_source=chatgpt.com)

所以我们的 contact label 可以从距离生成 soft target：

\[
\boxed{
c_{k,i}
=
\exp
\left(
-\frac{d_{k,i}^2}{2\sigma^2}
\right)
}
\]

而不是 hard 0/1。

这样：

- 2 mm；
- 5 mm；
- 15 mm；

不会全被硬切成两类。

对 WAM 更友好。

---

# ⑤ 我认为我们应该比静态 grasp 工作多保留一个“运动方向”

这是我觉得我们真正能超过 ContactGen / GenDexGrasp 的地方。

因为它们主要研究：

\[
\text{static grasp}
\]

而我们有完整 trajectory。

所以可以计算：

\[
v^{rel}_{k,i}
=
v^{hand}_{k,i}
-
v^{object}_{k,p}
\]

再按 object normal 分：

\[
v^n
=
-v^{rel}\cdot n
\]

和：

\[
v^t
=
\|
v^{rel}
-
(v^{rel}\cdot n)n
\|
\]

物理意义非常明确：

- \(v^n>0\)：手指正在向物体压近；
- \(v^n\approx0\)：稳定贴住；
- \(v^n<0\)：正在离开；
- \(v^t\) 大：可能发生 slip。

所以：

\[
\boxed{
\text{contact location}
+
\text{surface normal}
+
\text{approach/slip}
}
\]

比单纯 contact map 更适合“交互过程”。

---

# 但第一版我不会预测 24×11×9

没必要一下把 C 做得那么重。

我建议只预测三个 anchor：

\[
\boxed{
k=8,\ 16,\ 24
}
\]

也就是约：

\[
0.27s,\quad0.53s,\quad0.8s
\]

所以：

\[
C=
\{
C^8,C^{16},C^{24}
\}
\]

输出规模只有：

\[
3\times11\times9
=
297
\]

非常小。

它大概描述：

```text
0.27s：正在靠近哪里
0.53s：哪些手指开始建立接触
0.80s：最后形成什么抓法
```

这比只预测一个最终 `C_grasp` 更适合 WAM。

---

# C 的 GT 怎么自动生成？

这一点你们的数据非常占优势。

千万不要直接用 WAM 当前那 32 个 object points 做 nearest neighbor，因为太稀。

应该用原始高分辨率 object surface。

对于每个：

\[
h_{k,i}
\]

先转换到 object frame：

\[
h^{obj}_{k,i}
=
T^{-1}_{obj,k}h_{k,i}
\]

然后在 canonical object mesh / dense point cloud 上找：

\[
p^*_{k,i}
=
\arg\min_{p\in O}
\|h^{obj}_{k,i}-p\|
\]

得到：

\[
d_{k,i}
=
\|h^{obj}_{k,i}-p^*_{k,i}\|
\]

以及 surface normal：

\[
n_{k,i}=n(p^*_{k,i})
\]

于是：

\[
c_{k,i}
=
\exp(-d^2/2\sigma^2)
\]

再用相邻帧求 \(v^n,v^t\)。

这整个 label generation：

\[
\boxed{\text{不需要人工标注}}
\]

---

# 那真实 contact force 呢？

机器人仿真里我们确实还有 force。

ContactExplorer 就是：

\[
\text{near surface}
\land
\text{contact force > threshold}
\]

来定义 contact mask。它源码里 `keypoint_contact_mask` 就这么构造。[GitHub](https://github.com/ruoyiqiao/ContactExplorer?utm_source=chatgpt.com)

但我**不建议把 force 放进核心 C**。

因为：

- GRAB/OakInk/Ego 数据没有真实 contact force；
- robot sim 有；
- 一放进去就产生 domain mismatch。

比较好的办法是：

\[
C_{\rm core}
=
\text{pure geometry}
\]

所有数据都能监督。

另外加一个 robot-only auxiliary：

\[
C_{\rm force}
=
[f_n,\ contact_{\rm phys}]
\]

只在 simulator 数据：

\[
mask_{\rm robot}=1
\]

时训练。

这样不会污染大规模 human pretraining。

---

# 那 object contact map 要不要保留？

我建议**不用同时预测一套 object dense map + 一套 hand point map**。

因为会重复。

我们有 11 个 canonical points，所以：

\[
C_{i}
=
\text{第 i 个 hand point 对应 object surface 的 interaction}
\]

已经足够直观。

如果以后发现：

> 11 个 keypoints 太稀，掌心 contact 表达不出来，

再扩成：

\[
11\rightarrow16/21
\]

或者添加 palm patches。

这比第一版直接输出：

\[
32\times11
\]

interaction matrix 简单得多。

---

# 不过我会额外算一个 D(R,O)-style consistency

虽然 C head 不需要输出整个：

\[
11\times32
\]

distance matrix，但训练时可以计算：

\[
D_{ij}
=
\|\hat h_i-\hat o_j\|
\]

作为 consistency。

于是三个 head：

\[
\hat\tau,\quad
\hat O,\quad
\hat C
\]

之间不是完全独立。

比如 C 说：

\[
c_{i}=0.95
\]

但是：

\[
\min_j
\|\hat h_i-\hat o_j\|
=
8cm
\]

显然不合理。

所以加：

\[
L_{cons}
=
c_i
\cdot
\max(d_i-d_{\rm contact},0)
\]

这正好借鉴 D(R,O) 对连续 hand-object distance 的强调。[GitHub](https://github.com/Jzzzi/DRO-Grasp?utm_source=chatgpt.com)

---

# 对应到我们 WAM，我最终推荐这样的 C

\[
\boxed{
C_{k,i}
=
[
p_c,\;
p_x^{obj},p_y^{obj},p_z^{obj},
n_x,n_y,n_z,
v_n,v_t
]
}
\]

其中：

\[
k\in\{8,16,24\}
\]

\[
i=1,\dots,11
\]

也就是：

\[
\boxed{3\times11\times9}
\]

的 structured interaction head。

而不是一个：

\[
C\in\mathbb R^{256}
\]

看不懂的 latent。

---

## 对现有工作的对应关系很漂亮

\[
p_c
\]

来自 ContactGen / GenDexGrasp 的 **contact map**；

\[
i
\]

就是 ContactGen 的 **part map**，但我们由 canonical point ID 免费得到；

\[
p^{obj},n
\]

对应 object-centric contact location / direction；

\[
d
\]

借鉴 D(R,O) 的 continuous geometric relation；

\[
v_n,v_t
\]

则是利用我们独有的 clean 3D temporal data 扩展成真正的 **dynamic interaction representation**。

所以它不是拍脑袋定义。

---

## 源码我建议直接看这四处

- ContactGen：`contactgen/hand_object.py` 和 `contactgen/model.py`，里面直接生成/预测 `contacts_object`、`partition_object`、`uv_object`。[ContactGen GitHub](https://github.com/stevenlsw/contactgen?utm_source=chatgpt.com)
- GenDexGrasp：`utils_model/CMapAdam.py`，最值得抄它的 object `xyz + normal + contact value` 数据合同。[GenDexGrasp GitHub](https://github.com/tengyu-liu/GenDexGrasp?utm_source=chatgpt.com)
- D(R,O) Grasp：`utils/multilateration.py` 和 `utils/func_utils.py`，看 point-wise distance relation 怎么定义。[D(R,O) Grasp GitHub](https://github.com/zhenyuwei2003/DRO-Grasp?utm_source=chatgpt.com)
- ContactExplorer：`src/tasks/allegro_singulation.py`，直接看 keypoint-to-surface distance + force contact mask。[ContactExplorer GitHub](https://github.com/ruoyiqiao/ContactExplorer?utm_source=chatgpt.com)

CEDex 也值得作为后续 cross-embodiment 参考：它就是从 human-like contact representation 出发，再对齐不同 robot hand，目前 grasp generation 和 ContactGen-based CVAE 训练代码都已经公开。[GitHub](https://github.com/GeorgeWuzy/CEDex-Grasp?utm_source=chatgpt.com)

**所以我当前最建议：先实现这个 297D 的 `C{8,16,24}` GT label generator，先不碰 WAM。** 把 GRAB / robot rollout 随便抽几十条可视化出来：每个 hand point 画它的 target surface point、normal 和 contact confidence。如果这些图肉眼上确实能把“靠近→接触→稳定抓取”表示清楚，再给 WAMv3 加 C head。这样风险最小。


我更推荐**“共享主干 + 三头并行，但 C 头显式读取 hand/object 的关系特征”**，而不是纯级联：

\[
\hat\tau,\hat O \rightarrow C
\]

原因是纯级联会把 trajectory prediction 的误差全部传给 \(C\)。你们现在 \(\tau\) 本身还不够准，一旦：

\[
\hat\tau=\tau^{GT}+\epsilon_\tau
\]

那接触预测又多一层：

\[
\epsilon_\tau\rightarrow\epsilon_C
\]

很容易变成最脆弱的一条链。

更合适的是：

\[
Z_{\text{WAM}}
\rightarrow
\begin{cases}
\hat\tau^{hand}\\
\hat O^{future}\\
\hat C
\end{cases}
\]

但 \(C\) 头内部不要只是对 shared token 做 MLP，而是专门构造：

\[
Z^{HO}_{ij}
=
f(
Z_i^{hand},
Z_j^{object},
p_i-p_j
)
\]

再通过 cross-attention / relation MLP 输出 contact。

也就是：

```text
              Shared WAM trunk
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   hand head    object head   interaction head
      τ             O              C
                                  ▲
                                  │
                         hand-object relation
                         from shared tokens
```

### 但再加一个 consistency constraint

虽然不让：

\[
C=f(\hat\tau,\hat O)
\]

成为唯一计算路径，但应该约束三者不能互相矛盾。

例如预测：

\[
\hat C_{i,j}=1
\]

时，预测的 hand/object future 应满足：

\[
\|\hat h_i-\hat o_j\|
\]

足够小。

所以训练：

\[
L=
L_\tau+L_O+\lambda_C L_C
+\lambda_{\text{cons}}L_{\text{cons}}
\]

这样既避免级联误差，又让三个 head 物理一致。

还有一个更关键的判断：如果你第一版的 \(C\) 只是

\[
[x,y,z,p_{contact}]
\]

这种**完全可以由 hand/object 几何确定的量**，那甚至不必让网络独立预测全部内容。可以把几何部分直接算：

\[
C_{\text{geo}}
=
g(\hat\tau,\hat O)
\]

而 C head 只预测几何不能决定的部分，比如：

- contact probability；
- grasp role；
- contact phase；
- approach direction；
- preload / interaction confidence。

所以最终我最推荐的是：

\[
\boxed{
\text{并行预测为主，几何关系耦合为辅}
}
\]

不是纯三头互不相干，也不是 `τ/O → C` 的硬级联。

对你们 WAMv3 来说，这个改动也最小：**保留现有 hand/scene heads，只新增一个 hand-object relational head，再加 consistency loss。**