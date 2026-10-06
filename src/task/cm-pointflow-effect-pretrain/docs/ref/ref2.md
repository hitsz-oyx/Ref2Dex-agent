可以。**先别下载 OakInk2 的完整 2.13 TB。** 对我们现在要做的

\[
H_t+\text{hand keypoint trajectory}\rightarrow \text{object effect}
\]

第一版只需要 **3D 标注 + 物体模型 + task metadata**，RGB 多视角视频暂时完全不需要。

官方 Hugging Face 数据库整个是 2.13 TB，但 `anno_preview` 只有约 **36.7 GB**；其中已经包含每帧的 `raw_mano` 和 `obj_transf`。`object_raw.tar` 约 15.6 MB、`object_repair.tar` 约 140 MB、`program.tar` 约 3.6 MB。[Hugging Face](https://huggingface.co/datasets/kelvin34501/OakInk-v2/tree/main)

### 1. 下载 OakInk2 工具代码

```bash
git clone https://github.com/oakink/OakInk2.git
cd OakInk2

conda create -n oakink2_cm python=3.10 -y
conda activate oakink2_cm

pip install -e .
pip install huggingface_hub
```

官方仓库和数据入口分别在这里：

[OakInk2 官方仓库](https://github.com/oakink/OakInk2?utm_source=chatgpt.com)  
[OakInk2 Hugging Face 数据](https://huggingface.co/datasets/kelvin34501/OakInk-v2?utm_source=chatgpt.com)

### 2. 只下载我们真正需要的数据

我建议不要直接运行官方 `script/download.py`，因为它的 `snapshot_download()` 默认会把整个仓库往下拉，完整仓库有 **2.13 TB**。[Hugging Face](https://huggingface.co/datasets/kelvin34501/OakInk-v2/tree/main)

在仓库根目录新建：

```bash
mkdir -p data_download
```

然后执行：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="kelvin34501/OakInk-v2",
    repo_type="dataset",
    local_dir="data_download",
    allow_patterns=[
        "anno_preview/*.pkl",
        "object_raw.tar",
        "object_repair.tar",
        "object_affordance.tar",
        "program.tar",
        "program_extension.tar",
    ],
)
PY
```

这样预计下载量大概 **37 GB 左右**，而不是 2 TB。

其中真正核心的是：

```text
anno_preview/          ~36.7 GB   必须
object_raw.tar         ~15.6 MB   必须
object_repair.tar      ~140 MB    建议
program.tar            ~3.6 MB    建议
object_affordance.tar  ~170 MB    可选但很小
program_extension.tar  ~19 MB     可选
```

### 3. 解压小文件

```bash
mkdir -p data/oakink2

tar -xf data_download/object_raw.tar \
    -C data/oakink2

tar -xf data_download/object_repair.tar \
    -C data/oakink2

tar -xf data_download/object_affordance.tar \
    -C data/oakink2

tar -xf data_download/program.tar \
    -C data/oakink2

tar -xf data_download/program_extension.tar \
    -C data/oakink2
```

然后把标注目录链接过去即可，不需要复制 37 GB：

```bash
ln -s "$(realpath data_download/anno_preview)" \
      data/oakink2/anno_preview
```

最后结构大概：

```text
data/oakink2/
├── anno_preview/
│   ├── scene_...pkl
│   ├── scene_...pkl
│   └── ...
├── object_raw/
├── object_repair/
├── object_affordance/
└── program/
```

---

### 4. 为什么 `anno_preview` 已经够我们训练？

一个 `.pkl` 内就已经有：

```python
{
    "mocap_frame_id_list": ...,
    "obj_list": ...,

    "obj_transf": {
        object_id: {
            frame_id: 4x4_transform
        }
    },

    "raw_mano": {
        frame_id: {
            "rh__pose_coeffs": ...,
            "lh__pose_coeffs": ...,
            "rh__tsl": ...,
            "lh__tsl": ...,
            "rh__betas": ...,
            "lh__betas": ...
        }
    }
}
```

也就是说我们直接能得到：

\[
\text{MANO}_t
\]

和：

\[
T^{obj}_t
\]

随时间的完整轨迹。官方格式就是这样定义的。[GitHub](https://github.com/oakink/OakInk2?utm_source=chatgpt.com)

因此我们完全不需要 RGB。

---

## 5. 还需要下载 MANO 模型

因为 OakInk2 给的是 MANO 参数，如果我们要生成：

- 21 hand joints；
- 10/11 semantic keypoints；
- 或 778 vertices；

需要 MANO v1.2 模型文件。

这个因为许可证原因不能随 OakInk2 一起分发，需要你自己在 MANO 官方网站注册下载：

[MANO 官方下载页](https://mano.is.tue.mpg.de/?utm_source=chatgpt.com)

放成：

```text
OakInk2/
└── asset/
    └── mano_v1_2/
        └── models/
            ├── MANO_LEFT.pkl
            └── MANO_RIGHT.pkl
```

官方 OakInk2 也是要求这个目录结构。[GitHub](https://github.com/oakink/OakInk2?utm_source=chatgpt.com)

---

# 我们下载完后第一版数据不要做复杂

我建议第一版数据直接定义成：

\[
H_t=
\{
K^{hand}_t,
P^{obj}_t
\}
\]

动作：

\[
a=
K^{hand}_{t+1:t+8}-K^{hand}_t
\]

监督：

\[
E=
T^{obj}_{t+1:t+8}
\left(T^{obj}_t\right)^{-1}
\]

即：

```text
当前 hand 关键点
当前 object 几何
       +
未来 8 帧 hand keypoint trajectory
       ↓
模型
       ↓
未来 8 帧 object SE(3)
```

### 手先用多少点？

第一版我建议 **不要 MANO 778 点，也不要 4096 点**。

直接先取大约：

> **10–15 个语义关键点。**

甚至可以先用 MANO 原生 21 joints，然后后面消融：

```text
5 fingertips + palm
10/11 semantic points
21 MANO joints
```

完全没有 KNN。

### object 怎么表示？

物体 mesh 只采一次，例如：

```text
256 或 512 canonical object points
```

然后每帧根据 `obj_transf` 得到：

\[
P^{obj}_t=T_tP^{obj}_{canonical}
\]

这样 object correspondence 是**解析精确的**，比 PointWAM 从视频跟踪出来的 pseudo trajectory 干净得多。

---

## 6. 我建议下载完成后不要马上大规模训练

先做一个 **100 条 sequence 数据审计**。

必须确认：

- MANO keypoints 连续；
- 左右手坐标一致；
- object transform 连续；
- 单位是米；
- object mesh 与 transform 对得上；
- 8-frame hand displacement 分布合理；
- object effect 分布不是 95% 静止；
- interaction window 比例足够。

尤其最后一点很重要。

如果整条 OakInk2 全部滑窗，很可能很多 window 是：

```text
手还没碰到物体
人在走
换工具
静止
```

第一版应该优先根据 `program` 标注或者手物距离筛出真正 manipulation 段。

---

### 我建议正式路线就定成

```text
现在
│
├─ 下载 OakInk2 annotation-only ~37GB
│
├─ MANO → 10/21 keypoints
│
├─ object mesh → 256/512 points
│
├─ 构造 8-frame dynamics windows
│
├─ 先统计 100 sequences
│
└─ 再决定正式训练规模
```

**先不要碰 SPIDER，也不要一次性合并 CHORD 的七个数据集。**

OakInk2 单数据源先把 `human keypoint trajectory → object effect` 跑通，这一步成功以后再把 TACO/ARCTIC/HOT3D 加进来。这样变量最少。


是的。**PointWAM 本身就是同时支持单手和双手的，而且它的处理方式其实很适合我们照搬。**

它没有为了统一数据而把双手样本拆成两个单手样本，也没有强制所有样本必须双手齐全。它定义的是：

\[
\mathcal Q_t=\{q_t^{(k)}\}_{k=1}^{K}
\]

其中**每只手固定 10 个 3D keypoints**。因此：

- 单手任务：\(K=10\)
- 双手任务：\(K=20\)

论文的训练设置也明确写了：人类预训练使用 **20 个 hand keypoints**，机器人 fine-tuning 根据任务使用 **10 或 20 个**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE\&utm_source=chatgpt.com)

### 人类数据怎么处理

EgoDex/VITRA 里如果两只手都能看到，就构造：

```text
right hand: 10 points
left hand:  10 points
--------------------
total:      20 points
```

每只手的 10 点都是：

- 6 个 palm anchor
- thumb / index / middle / ring 4 个 fingertip

如果某一只手在该视频里缺失：

> **直接把这只手的 keypoints 标成 invalid，不做插值、不伪造。**

这是论文附录明确写的。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE\&utm_source=chatgpt.com)

所以模型实际上天然支持：

```text
样本 A：
Right ✓
Left  ✗

样本 B：
Right ✓
Left  ✓

样本 C：
Right ✗
Left  ✓
```

靠 validity mask 统一 batch。

---

## 到机器人数据也是同一个合同

DexJoCo 既有单臂任务，也有双臂任务。

双手时，PointWAM 明确规定：

```text
[右手10点] [左手10点]
```

右手排在前面，左手排后面，并且**每只手都有自己的 hand ID**。

同时左右手不是把一只镜像过去，而是：

- 分别走自己的 FK chain；
- 分别计算 10 个 keypoints；
- 都转到同一个 scene coordinate frame。

论文特别说明，因为左右 Allegro hand 的 joint order 不同，所以使用独立 kinematic chains。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE\&utm_source=chatgpt.com)

所以本质上是：

\[
\boxed{
(\text{xyz},\text{keypoint ID},\text{hand ID},\text{valid mask})
}
\]

---

# 这个设计其实正好解决了我们刚才的单双手问题

我上一条说 V0 可以先筛单手，是为了保证 attribution 干净。

但如果我们现在决定**尽量复刻 PointWAM representation**，其实有一个更好的方案：

> **不需要扔掉 OakInk2 双手数据。**

直接固定最大两只手：

\[
Q_t\in \mathbb R^{2\times K\times3}
\]

例如我们每只手取 10/11 点。

再有：

\[
M_{\rm hand}\in\{0,1\}^2
\]

以及：

\[
ID_{\rm hand}\in\{\text{left},\text{right}\}
\]

那么单手：

```text
Right:
10 keypoints   valid=1

Left:
10 empty slots valid=0
```

双手：

```text
Right:
10 keypoints   valid=1

Left:
10 keypoints   valid=1
```

网络结构完全不用变。

---

## 对我们的 \(H+a\rightarrow E\) 甚至比 PointWAM 更自然

我们不是要预测 hand trajectory，而是把它作为条件。

所以双手样本直接变成：

\[
H_t+
F_{\rm right}^{1:K}
+
F_{\rm left}^{1:K}
\rightarrow
E_{\rm object}^{1:K}
\]

单手样本则：

\[
H_t+
F_{\rm right}^{1:K}
+
\text{MASK}
\rightarrow
E_{\rm object}^{1:K}
\]

这样之前我担心的：

> 左手扶着瓶子，右手拧瓶盖，却只把右手作为 action

的问题直接消失。

因为模型会看到：

\[
F_{\rm left}+F_{\rm right}
\]

两只手共同构成完整 intervention。

这其实比筛掉双手数据更合理。

---

# 所以 OakInk2 可以直接这样处理

例如固定每只手 11 点：

```text
每只手：
5 fingertips
+
6 palm / knuckle anchors
= 11
```

则最大：

\[
22\text{ hand points}
\]

一个 8-step chunk 也不过：

\[
22\times8\times3=528
\]

个连续数值。

相比之前：

\[
4096\times3
\]

完全是小数据量。

数据 tensor 可以直接设计成：

```text
hand_xyz       [B, 2, 11, 3]
hand_future    [B, 2, 11, K, 3]
hand_valid     [B, 2]
hand_id        [B, 2]
keypoint_id    [11]

object_points  [B, N, 3]

object_future  [B, N, K, 3]
或
object_SE3     [B, K, 6]
```

然后把两只手和 object 放到统一 3D backbone。

---

## 我现在会修改之前的建议

如果我们的目标已经变成：

> **先尽量复刻 PointWAM 的 keypoint representation，然后改成 action-conditioned effect prediction**

那我**不建议再只筛 `rh_main`**。

更合理的是：

\[
\boxed{\text{OakInk2 的单手 + 双手全部保留}}
\]

只要求每个 training window：

1. 至少一只手有效；
2. target object pose 有效；
3. 当前/未来 hand keypoints 连续；
4. 对缺失手进行 mask；
5. 双手 interaction 时两只手 future trajectory 都作为条件。

这样我们反而能充分利用 OakInk2 最大的优势：**它本来就是高质量双手复杂操作数据集。**

PointWAM 已经证明了这个 representation contract 可以把**单手、双手、人手、机器人手**放到同一个训练框架里。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE\&utm_source=chatgpt.com)

这条做法我认为比“先删掉所有双手样本”更适合作为我们现在正式训练的数据设计。