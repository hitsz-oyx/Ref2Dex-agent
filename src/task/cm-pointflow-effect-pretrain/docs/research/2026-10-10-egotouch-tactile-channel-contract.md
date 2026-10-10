# EgoTouch tactile channel 合同：原始通道、grid重建与未来归一化

日期：2026-10-10。Task：`cm-pointflow-effect-pretrain`。这是两样本只读复算与主源调研，不是训练或全库验证。
承接 [视频/触觉主源报告](2026-10-10-video-tactile-primary-sources.md)，本次只写此 note，未下载新数据、模型或视频。

## 结论与身份

**两份现有样本的217个有限grid格可以逐值重建；得到明确的候选归一化分组和右手坏列修复，但不能据此把分组命名当成已验证的硬件 bend/contact 语义。**
下一步预测任务应读取raw通道，以固定255或TRAIN-only统计缩放；不要将记录未来最大值用于history输入。
`manual_contact=false` 的覆盖与语义是UNKNOWN，不能用作整段无接触、触觉全零或contact negative标签。

固定身份：

- raw：`zhouzhoujy/EgoTouch@cfdbb0ac31cc2af4247943820aa250575e7e6637`。
- code：`Jianyi2004/TouchAnything@d74f9ef5c189a957b7ff72781a0c998e41b45a56`。
- 原label本地根：`outputs/cm-pointflow-effect-pretrain/egotouch-label-schema-20261010-r1/raw/`。
- 本地RGB/帧号结果：`outputs/cm-pointflow-effect-pretrain/egotouch-rgb-clock-20261010-r3/manifest.json`；两样本43/47帧，`synchronized_qualified=false`，索引一致不保证瞬时传感器同步。

样本U=`Home/plug_unplug_usb_cable/20260320_135743_682`，M=`Office/slide_mouse/20260314_161602_832`。
文件来自固定原始release的 [U目录](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/tree/cfdbb0ac31cc2af4247943820aa250575e7e6637/Home/plug_unplug_usb_cable/20260320_135743_682)、[M目录](https://huggingface.co/datasets/zhouzhoujy/EgoTouch/tree/cfdbb0ac31cc2af4247943820aa250575e7e6637/Office/slide_mouse/20260314_161602_832)。

| 文件 | SHA256 |
| --- | --- |
| U `jq_pressure.json` | `5beb068aed40ef8dcd44c058a012812dd4dbd230e101b3c043c6e144a76ac4c1` |
| U `pressure_grids.npz` | `ccac54a0ba9d4babd3a4e465a8761ff3b6ff220e58ca825798c633cd7d8420e1` |
| U `manual_contact_annotation.json` | `eec901e6ee5c8f17e6891051869dc09dfdd7edd4d61afae0ec6fb19da02176d7` |
| M `jq_pressure.json` | `52415b8c73fd48ccf2f49fe6bb8e2d16a46d5d133d9379a3a9ebc88b9fa4cb6e` |
| M `pressure_grids.npz` | `4d25e4c0198c0802de71b1ff54a245c66ccf931a4670cb12c279cf9b1b73366e` |
| M `manual_contact_annotation.json` | `318327e0b705288b940b9b4a03b2ad81fbae8adb3fb8272e1952a1f102da27c7` |

## 官方保证到哪里

论文说明raw256通道映射到NaN掩码的21×21手形图，右手水平镜像；首帧baseline subtraction是可选的，只有首帧被判定无接触才做，判断来源为人工annotation或低信号fallback。
论文还说明右手坏列通过邻列插值修复，tactile与bending分开归一化。[TouchAnything Appendix 8.1–8.2](https://arxiv.org/html/2605.13083v1#A1)

**UNKNOWN：** 已读pinned公开Git tree没有找到实际生成 `pressure_grids.npz` 的raw预处理实现；HDF5 converter仅加载NPZ。
论文没有逐raw index的硬件分类表、明确坏列编号或max计算范围。因此不能声称官方已保证下面的精确通道划分适用于全库。
`src/utils/pressure_map.py` 是由手弯曲生成pseudo-pressure的另一流程，不能当作raw实测通道的分类/标定证据。
[pinned converter](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py#L591)、[pinned pseudo-pressure模块](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/src/utils/pressure_map.py)

## 217格的精确mapping与候选归一化分组

以下所有index和grid坐标都是**0-based**。原始256通道不等于256个独立压力taxel。
左右mapping各217项，但分别只引用138/147个unique raw index；复制到多个格造成79/70个重复项。
[pinned left mapping](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/configs/pressure_position_mapping_left.json)、[pinned right mapping](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/configs/pressure_position_mapping_right.json)

| 候选分组（处理语义） | left raw indices | right raw indices | 每手grid格数 / unique raw数 |
| --- | --- | --- | --- |
| 用 `bend_max` 的分组 | `208..222` | `33..47` | 66格；左右各15 unique indices |
| 用 `tactile_max` 的分组 | `0..14,16..30,138..158,160..174,176..190,195..206,224..238,240..254` | `1..15,17..31,49..60,65..79,81..95,97..111,113..127,225..239,241..255` | 151格；left123/right132 unique indices |

任一grid格归属由官方mapping的raw index集合成员关系精确定义；未出现在mapping中的raw通道当前不受NPZ重建验证。
66格的候选bend组中，12个index各复制5格，3个各复制2格；不是66个独立测量。
这些是**候选 normalization groups**，不是已核实的66格硬件bend和151格外部contact。
两样本有全零或无正增量通道，两种分母都会给零；它们的分组不能从这两样本单独辨识。

## 两样本完整重建与右手修复

两样本NPZ均有 `baseline_corrected_left/right=True`、`separate_normalization=True`；以下重建只应用于这些flag为真的样本。
设 `D_h(t,i)=max(sensor_h(t,i)-sensor_h(0,i),0)`，按候选分组除以该记录的 `tactile_max/bend_max` 并clip到[0,1]。
left使用mapping `(r,c)`；right使用 `(r,20-c)`，然后将display列 **6和10** 的每个有限格替换为其左右邻格均值。
这等价于在未镜像坐标修复列14和10再镜像；实际producer执行顺序尚未找到，不能声称唯一识别了实现顺序。
本次重建选择“subtract→normalize/clip→mirror→spatial repair”，全格吻合；并未发现需要时间插值才能复现这些NPZ。

| 样本 | T | left max absolute error | right max absolute error | NaN mask |
| --- | ---: | ---: | ---: | --- |
| U | 43 | `1.1920928911e-8` | `2.7815500903e-8` | 左右均逐格完全一致 |
| M | 47 | `9.5367431729e-9` | `2.9140048596e-8` | 左右均逐格完全一致 |

right两列共覆盖28个grid格，重建后仍然finite；**finite不能等价于直接测量有效**。
这28格原先对应的20个raw indices不再直接进入输出：`8,11,24,27,40,43,56,59,72,75,88,91,104,107,120,123,232,235,248,251`。
因此在该候选修复规则下，right最多保留127个直接raw index来源（114个候选tactile、13个候选bend），另28格是空间合成。
这个数是来源上界，不是独立物理sensor数量或样本数；left也不能按217格等权重复计作217独立taxels。

**baseline条件边界：** 数值吻合说明这两样本发布grid确实可由首帧扣除重建，不说明首帧真实unloaded。
原始通道已有非零首帧；扣除和负值clip会删除基线接触与释放信息。
不可默认其他episode全部subtract，更不能据 `manual_contact=false` 判定subtract安全；保留raw及baseline flag，missing annotation不自动当无接触。

## 全record max与未来泄漏风险

两样本只保存 **共享 `tactile_max` / `bend_max`**，没有左右专属max键。
在上面候选partition/空间修复下，储存max恰好等于**左右双手全record**扣除首帧后的分组最大值。

| 样本/分组 | stored max = 全record修复后max | 仅首4帧max | 首次全record峰值frame |
| --- | ---: | ---: | ---: |
| U tactile | 20 | 19 | 42 |
| U bend | 30 | 4 | 42 |
| M tactile | 50 | 10 | 32 |
| M bend | 45 | 3 | 26 |

**事实：** 现有grid的早期非零值确实使用这些stored分母；表中的峰值出现在首4帧history之后。
**推论：** 这些max很可能来自全record且修复后的统计，而非固定硬件量程；两记录不同max与精确峰值相符支持该解释。
**UNKNOWN：** 没有producer代码，不能唯一证明max选择算法；手动设置或其他统计巧合仍不能完全排除。
即便不用max字段作为feature，未来峰值决定早期grid缩放也可能泄漏记录未来信息。
因此下一步凡把pressure放进history，须重新从raw计算，使用固定255或仅TRAIN拟合/冻结统计；evaluation不得每条记录看未来重新归一化。
target也宜同一固定尺度，不能逐record强行将峰值变成1；若只做既有processed标签的benchmark，需要明确其offline归一化限制。

另外，pinned converter只读取 `tactile_max_left/right`、`bend_max_left/right`，缺字段默认0。
这两NPZ的全局键将被忽略，转换后四个max全变0，虽grid值不变却丢失真实分母provenance。
不能拿当前代码默认值当原始release合同；后续adapter应显式识别global/shared与per-hand两种schema。
[固定converter L619–624](https://github.com/Jianyi2004/TouchAnything/blob/d74f9ef5c189a957b7ff72781a0c998e41b45a56/scripts/core/convert_to_hdf5.py#L619)

## EgoTac新线索：可借鉴表示，不能移植其通道语义

EgoTac代码HEAD经只读 `git ls-remote` 固定为 `87ba7059304bd581050315f3a9af3029710ed690`。
其论文Appendix B描述自采 **EgoTac-SC/Aether** 的132 force channels（60 finger+72 palm）和5个被排除的bend channels；并非EgoTouch256raw/21×21采集协议。
论文没有把这套partition标为EgoTouch兼容。[EgoTac Appendix B](https://arxiv.org/html/2608.15060v1#A2)

代码的MANO mapper要求输入shape `(132,)`，经patch/bilinear weights生成778-vertex场，并有contact threshold/smoothing。
这是EgoTac作者的加工表示，对EgoTouch属于第三方候选；相同的“132”也不能证明right候选tactile通道可按其顺序输入。
官方guide还明示raw HDF5/SVO转换与glove-to-MANO mapping不在source release中。
没有EgoTouch raw-index→EgoTac canonical sensor order、面积/Pa/N标定和硬件几何桥接，当前兼容性为 **UNKNOWN/未建立**。
[固定mapper](https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/data_processing/common/tactile_to_mano_mapper_patch.py#L364)、[固定Aether guide](https://github.com/Mr-Zwkid/EgoTac/blob/87ba7059304bd581050315f3a9af3029710ed690/docs/process_data_aether.md)

## 可复用的最低成本复算

从当前工作树根运行以下代码，只读取已下载两样本；代码中的candidate集合不构成硬件语义认证。
它按flags决定baseline，保留NaN并核对所有grid，不运行模型、不创建产物。

```python
import json
from pathlib import Path
import numpy as np

raw_root = Path("outputs/cm-pointflow-effect-pretrain/egotouch-label-schema-20261010-r1/raw")
mapping_root = Path("tmp/video-tactile-primary-sources")
for grid_path in sorted(raw_root.rglob("pressure_grids.npz")):
    rows = [json.loads(s) for s in (grid_path.parent / "jq_pressure.json").read_text().splitlines()]
    with np.load(grid_path, allow_pickle=False) as z:
        for side in ("left", "right"):
            mapping = json.loads((mapping_root / f"TouchAnything-configs__pressure_position_mapping_{side}.json").read_text())
            raw = np.array([r[f"sensor_{side}"] for r in rows], dtype=np.float64)
            assert raw.shape == (len(rows), 256)
            baseline = bool(z[f"baseline_corrected_{side}"])
            delta = np.maximum(raw - raw[0], 0) if baseline else raw.copy()
            candidate_bend = set(range(208, 223) if side == "left" else range(33, 48))
            rebuilt = np.full((len(rows), 21, 21), np.nan)
            for key, index in mapping.items():
                row, col = map(int, key.split(","))
                group = "bend" if index in candidate_bend else "tactile"
                scale = float(z[f"{group}_max"])
                assert scale > 0
                col = 20 - col if side == "right" else col
                rebuilt[:, row, col] = np.clip(delta[:, index] / scale, 0, 1)
            if side == "right":
                for col in (6, 10):
                    valid_rows = np.isfinite(rebuilt[0, :, col])
                    rebuilt[:, valid_rows, col] = (
                        rebuilt[:, valid_rows, col - 1] + rebuilt[:, valid_rows, col + 1]
                    ) / 2
            published = z[f"{side}_pressure_grid"]
            assert np.array_equal(np.isnan(rebuilt), np.isnan(published))
            error = float(np.nanmax(np.abs(rebuilt - published)))
            assert error < 1e-6, (grid_path, side, error)
            print(grid_path.parent.name, side, baseline, error)
```

**下一项最便宜Decision：** 在现有两样本做raw/255的固定尺度候选表示，显式标记重复来源、右手28个imputed格及候选normalization分组；先以原始通道预测为工程目标，暂不叫contact/force预测。
其后用少量已有不同任务label复算检查partition和修复是否稳定，记录不匹配，而非扩大下载或训练。
若要把candidate升为触觉/contact监督，需作者提供硬件channel type/坏列/producer合同，或取得独立校准证据；没有这些证据可继续原始sensor prediction，但不能形成物理force或外部contact claim。
