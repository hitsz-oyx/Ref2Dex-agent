# UniDex 多源配比与 loss：官方代码专项核验

核验日期：2026-10-15。只读核验 [UniDex 论文](https://arxiv.org/abs/2603.22264) 与官方公开源码，
版本 `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`；稀疏副本为 `third_party/UniDex-source-audit/`。
未运行作者训练、未取得作者训练 manifest，以下比例公式描述公开实现，不能等同于论文运行的实测比例。

## 结论

- **来源配比：公开入口按来源窗口数决定，不是四源等比例，也没有给出固定百分比。**
- **训练 loss：归一化 FAAS 上的 flow-matching 全元素 MSE；未看到来源、左右手、机器人有效 actuator、缺手或插值帧的监督 mask/专属 loss weight。**
- **未使用的机器人维度零填充后仍参与 loss。** `action_mask` 是 attention mask，不能解释成有效动作维度 mask。
- **公开代码本身存在运行合同缺口。** 例如 TACO 引用未提供的 normalizer/full，模型配置引用缺失的 src.openmodel 命名空间，train.py 第一行标为 development。因此应严格说“公开实现显示”，不能断言论文训练使用完全相同配方。

## 1. 四源的实际抽样概率

[`config/dataset/mixture.yaml` L1–8](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/mixture.yaml#L1-L8)
仅列 H2O、HOI4D、HOT3D、TACO；无 weights。
[`MixtureDataset` L7–43](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/mixture.py#L7-L43)
保存各 `len(ds)`、拼接 index 范围，总长度为其和；无 WeightedRandomSampler 或来源抽签。
[`train.py` L182–187](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/train.py#L182-L187)
全体窗口 random_split 后用 shuffle=True。

因此正常取样时，训练 epoch 中来源 s 的概率为：

```text
P(source=s) = N_train_windows(s) / sum_s N_train_windows(s)
```

这是代码推导；不是原始帧数比例，不是轨迹数比例，不是每源 25%。窗口数还受分段、stride、chunk_size、
滑窗间距、已存在 retarget HDF5 和机器人手数量影响。
四份来源配置都允许八种机器人手，但没有单独的 hand sampling weight；配置允许并不证明每源实际具有等量八手产物。
[H2O hands](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/H2o.yaml#L18-L26)、
[HOI4D hands](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/HOI4D.yaml#L18-L26)、
[HOT3D hands](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/Hot3D.yaml#L19-L27)、
[TACO hands](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/Taco.yaml#L18-L26)

补充：[`BaseDataset.__getitem__` L642–645](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base.py#L642-L645)
加载异常时，会在同一个 source dataset 中随机重取 index。因此若有坏样本，源内实际曝光分布还会偏向可成功加载窗口；
公开文件没有各源成功窗口计数，不能给出可靠数值百分比。

## 2. loss 的完整关键链路

模型配置 [`unidex.yaml` L4–14](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/model/unidex.yaml#L4-L14)
使用 30-step、82D action/proprio。
训练类 [`PointCloudUniDexTrain.forward` L883–914](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/unidex/unidex.py#L883-L914)
只取 prompt、pointcloud、action、state，`t` 从 uniform `[0,1)` 抽样，原样返回基类 loss。
没有传 source id、hand validity、actuator validity 或 action_is_pad 给 loss。

基类 [`PointCloudUniDex.forward` L757–811](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/unidex/unidex.py#L757-L811)
对全部 action 元素添加 Gaussian noise，预测 velocity，代码等价于：

```text
sigma_min = 0.001                  # constructor 默认值
x0 ~ Normal(0, I), x1 = normalized_actions
t ~ Uniform(0, 1)
psi_t = (1 - (1-sigma_min)*t)*x0 + t*x1
v_target = x1 - (1-sigma_min)*x0
loss = mean_{batch,horizon,all_82_dims} (v_pred - v_target)^2
```

`sigma_min` 默认值见 [constructor L25–35](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/unidex/unidex.py#L25-L35)，
路径表达式见 [`psi_t` L746–755](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/unidex/unidex.py#L746-L755)。

[`train.py` L93–106](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/train.py#L93-L106)
训练 wrapper 没有二次加权；validation 为 inference 输出与 target 的默认 MSE，也无 source-wise weighting。
没有额外 contact、penetration、smoothness、velocity-magnitude、左右手平衡、wrist/joints 分项系数或按 source 重标定的监督项。
这是对上述公开主训练路径的核验，不能排除未公开运行版本中的改动。

## 3. mask 与 padding 的真实含义

### FAAS 未使用维度

[`BaseDataset._apply_action_map` L537–579](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base.py#L537-L579)
保留前 18D 双腕，将每手 joints 映射到各自 32D 零数组，未使用 slots 保持零，然后拼成 82D。
不同机器人用同一个语义布局，不需要在 loss 里路由不同输出头；但代码没有返回 actuator mask。

零目标仍进入加噪和 MSE：即使某维 `x1=0`，其 flow target 仍包含 `-(1-sigma_min)*x0`。
所以“padding 为零”不能理解成“该维不产生 loss”，也不能把总 loss 当成仅在真实可控 DoF 上归一化的误差。

### 缺手与补齐

[`base_retarget.py` L430–495](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base_retarget.py#L430-L495)
缺腕姿态换 UNSEEN_POSE、NaN joints 换零；这些会继续生成相对腕目标和 joint targets。
[`mixture.py` L32–40](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/mixture.py#L32-L40)
在 shape 不等时补零，也没有产生 pad mask。
这与我们的 metric point displacement + 显式有效性 mask 不同，不建议照搬。

### attention mask 不是 loss mask

[`build_causal_mask_and_position_ids` L250–285](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/unidex/unidex.py#L250-L285)
屏蔽未使用 text tokens，并规定 pcd/text、proprio、action tokens 的互相注意力范围。
它没有 action dimension 轴，不能屏蔽某手/某机器人未使用 actuator 的 loss。
`train.py` 虽保留了 `action_is_pad` 的 dtype 特判，但上述 model forward 没有读取它；不能仅见字段名就认定监督已屏蔽。
[device helper L33–42](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/train.py#L33-L42)

## 4. normalization 是隐式量纲平衡，不是来源配比

执行顺序为原 robot values → hand/side-specific affine scale/shift → FAAS mapping → normalization。
[scale/shift L493–507](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base.py#L493-L507)、
[affine 实现 L526–534](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base.py#L526-L534)、
[map 与 normalize L632–638](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/dataset/base.py#L632-L638)

base config 为 action/state minmax，pointcloud identity。
[base config L10–14](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/normalizer/base.yaml#L10-L14)
`MixtureDataset` 合并各 normalizer 后赋回全部来源；minmax 合并对每维取全体下界最小值/上界最大值。
[merge L38–77](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/utils/normalizers.py#L38-L77)

公式为 `2*(x-mid)/(max-min+1e-6)`，并非各源单独 whitening。
[minmax L93–99](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/src/utils/normalizers.py#L93-L99)
因此 normalized-space MSE 对不同物理维度有不同尺度，不能拿其数值直接对应米制 hand/object displacement loss。

## 5. 论文与公开代码必须分开陈述

[论文 Appendix A.1–A.2](https://arxiv.org/html/2603.22264v1#A1)
给 conditional flow matching 目标、8 H800、batch128 和训练预算；未给四源固定 sampling 百分比或 source-specific loss 系数。
论文描述 82D FAAS，但没有给出“所有缺失/unused维都采用有效性mask”的声明。

上述公开代码入口另有缺口：

- [`train.py` L1](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/train.py#L1) 自标开发中。
- [TACO config L1–2](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/Taco.yaml#L1-L2) 引用 `normalizer: full`；该 commit 的 [normalizer 目录](https://github.com/unidex-ai/UniDex/tree/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/dataset/normalizer) 只有 base.yaml。
- [model config projector/joint](https://github.com/unidex-ai/UniDex/blob/97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d/config/model/unidex.yaml#L63-L75) 引用 `src.openmodel`，该 commit 无此 package。

所以可确认设计和公开路径中的采样/loss语义；**无法从这些文件精确还原作者9M帧训练的四源数值配比**。

## 6. 对我们最实用的借鉴

这是本项目建议，不是 UniDex 实验结论：

1. 来源比例与 loss 比例应分别明确记录；source-balanced sampler 不能替代手/物体分项和 validity mask。
2. 当前人手/物体目标共用米制几何语义，优先保留已有 TRAIN-only stats 和 hand/scene mask合同，再对新增源重新拟合统计；不套用机器人 FAAS 的 zero padding 监督。
3. 设定全局采样概率后记录实际成功加载/有效监督曝光数量；否则长轨迹、多机器人复制和坏样本重取都会使名义样本数误导。
4. 无必要一开始添加 source-specific loss 系数；先统一目标、单位和可用标签，再报告每源有效 hand/scene loss，发现真实梯度或尺度失衡后做最小干预。
