# brief41：τ checkpoint 固定面板 updates 曲线

## 结果解读表（预注册，逐字保留）

| 观测结果（在同一固定面板、同一 init-A 参考上） | 含义 | 会改变哪个决定 |
|---|---|---|
| step128 与 step1223 的三项 EPE **配对差 CI 全部含 0**，且 \|差\| < (init→1223 改善量) 的 5% | τ 预测在 ≈128 updates 内已饱和，1223 预算的其余部分没有带来预测收益 ⇒ **配方/监督信号受限**，不是预算受限 | 不再给同配方加预算；τ 的下一步转向监督信号与条件语义（是否引入可用 goal/event），而非加 GPU 时长 |
| step128 三项均明显优于 step1223（相对差 ≥10% 且 CI 排除 0） | 现有 `best_selection` 口径是对的，末态选点才是错的 | 必须把 τ 的选点/早停改成**面板口径**；报告里 step1223 的数字必须标注为"末态选点"，不能当"最佳选点" |
| 曲线在 step1000→1223 仍有 CI 排除 0 的改善（单调下降） | 尚未饱和 | 才允许把"给同配方补预算"提上议程（仍需先与网页 GPT 对齐，不得自行决定） |
| 三项 EPE 相对 init 不降，或 OOM / 盘 <20GiB / 非有限值 | 故障门 | 立即停手、保留证据、**不自动重启**、不删除历史产物 |

以上为 Hermes brief41 的事前判据，不随结果更改。本轮只读旧产物并对尚未评估的既有checkpoint运行一次冻结面板，不训练、不改配方、不增加数据、不运行rollout或物理仿真。若checkpoint缺失，记录卡点，不生成替代checkpoint。

## 运行前核对及计算方向（读取新面板数值前登记）

`panel_gpu456.py` SHA256实测与指定值一致：`d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51`，不复制、不修改。GPU4初检2MiB占用、24252MiB空闲、利用率0%，磁盘约130GiB；每个面板前再次检查free≥8GiB、磁盘≥20GiB并确认GPU4无显著占用。唯一新增推理为step20/128/1000各一次；stdout与命令/资源记录待该面板结束后写进其新输出目录，不预创建推理输出目录。

**卡点**：`tmp/t38-r2/training/step000500.pt` 不存在。实际保存文件仅step000000、000001、000020、000128、001000、001223；有dev-step000500.json不代表保存了该步模型。不生成替代checkpoint，不补训，不把dev数字填进面板曲线；step500行保留缺失。

配对差使用冻结面板脚本同一`cluster_ci`函数、124母序列/4000次重采样，读取已保存sums/counts/mothers，不增加前向推理。曲线表差为当前step−init；饱和/选点比较为step128−step1223，10%相对差以step1223为分母；末段比较为step1223−step1000，负值表示改善。单调性只可检查现有模型点，无法检验缺失的step500；三个指标分别列出，不用平均抵消。逐条检验原表条件；若均不满足或彼此冲突，明确结论为预注册归因未成立，不自行补出新判据或追加预算。

## 单一结论与预注册检验

**命中第三行：在本固定TRAIN面板上，1000→1223仍未饱和。只允许把同配方补预算提上议程，先与网页GPT对齐；本轮没有启动训练，也没有决定补多少预算。** 这不是“128已饱和/配方受限”，也不是“128在本面板最好”。DEV选点与TRAIN面板相反是不同评价口径的真实结果，不作为bug掩盖，不能据此改配方。尚未建立DEV泛化随预算改善或执行收益。

| 原表条件 | 实际检验 | 命中 |
|---|---|---|
| 128≈1223，三项差CI含0且绝对差<改善量5% | 三项差CI均严格>0；绝对差2.6315/18.7270/2.2214mm，大于阈值0.2940/1.3731/0.2374mm | 否 |
| 128三项优于1223≥10%、CI排除0 | 方向相反：128比末态高25.58%/58.96%/34.15%（分母为1223） | 否 |
| 1000→1223三项继续改善且CI排除0，曲线下降 | 三项均改善；所有可观测点0→20→128→1000→1223逐项下降 | **是** |
| 故障门 | 三个新增点三项均低于init；无OOM/非有限值/磁盘门 | 否 |

step500模型缺失，因此不能声称完整六点单调，更不能插值当作step500实测；它不阻断已独立保存的1000→1223配对检验。

## 冻结面板曲线：mean与相对init的配对差

每格为 `mean mm; Δ(current−init) [cluster95%CI]`。所有CI按124母序列重采样4000次，与brief38同函数、同seed170。init-A数值16.1694/59.2219/11.2523逐字引用既有报告；不重跑、不从NPZ重新估计其基线均值。step0的Δ=0为自身比较恒等式。

| updates | first8_hand | fixed_contact_hand | first8_object |
|---:|---|---|---|
| 0 | 16.1694; Δ+0.0000 [+0.0000, +0.0000] | 59.2219; Δ+0.0000 [+0.0000, +0.0000] | 11.2523; Δ+0.0000 [+0.0000, +0.0000] |
| 20 | 15.3142; Δ-0.8552 [-1.0394, -0.6774] | 58.6423; Δ-0.5796 [-1.1056, -0.0741] | 10.6319; Δ-0.6204 [-0.9981, -0.2556] |
| 128 | 12.9199; Δ-3.2494 [-3.7750, -2.7479] | 50.4868; Δ-8.7351 [-11.4560, -6.0957] | 8.7257; Δ-2.5266 [-3.3336, -1.7095] |
| 500 | **缺失checkpoint，未推理** | **缺失** | **缺失** |
| 1000 | 10.5711; Δ-5.5983 [-6.1953, -5.0087] | 34.2418; Δ-24.9801 [-28.9130, -21.3106] | 6.6985; Δ-4.5538 [-5.5337, -3.6284] |
| 1223 | 10.2884; Δ-5.8809 [-6.5041, -5.2801] | 31.7598; Δ-27.4621 [-31.9789, -23.3049] | 6.5043; Δ-4.7480 [-5.7424, -3.8141] |

![固定TRAIN面板updates曲线](../../../../../tmp/t38-r2/panel-step000020-A/updates-curve.png)

阴影为既有/新增点的绝对mean cluster CI；连线仅辅助读图，不是中间step实测。可导出的PDF、CSV与完整精度JSON均在panel-step000020-A目录，产物不入库。

## 直接跨checkpoint的配对差

| 比较（左−右，mm） | first8_hand Δ [CI95] | fixed_contact_hand Δ [CI95] | first8_object Δ [CI95] |
|---|---|---|---|
| 128−1223 | +2.6315 [+2.2762, +3.0007] | +18.7270 [+15.5631, +21.9997] | +2.2214 [+1.7291, +2.7968] |
| 1223−1000 | -0.2827 [-0.3883, -0.1793] | -2.4820 [-3.6373, -1.3622] | -0.1942 [-0.3329, -0.0761] |

末段1000→1223三项分别进一步下降2.67%/7.25%/2.90%；区间全部严格小于0，不能用绝对mean CI互相重叠来否定配对改善。128→1223三项下降，亦不能由另一DEV指标的best128推翻这张面板。step1223仍标注为**末态选点**；它是目前可观测固定面板点中均值最低者，不声称未保存的step或DEV上最优。

## 三种口径必须分开

1. **本轮面板**：固定TRAIN 512锚点/124母序列，A合同，goal/event=UNKNOWN。指标为前8帧hand、GT固定contact窗hand、前8帧object；按有效点累计求EPE，母序列cluster配对CI。本轮可观测三项均在1223最低。
2. **DEV hand macro**：既有dev-step*.json，训练入口对DEV分布子集的8个source分别计算有效hand EPE，再做source等权macro；代码包含goal/event条件字段和conditioned/unknown_event两个分支，但**本配方两分支实际都无有效goal/event**。这是全有效未来hand评价，不是固定TRAIN面板的first8/contact/object三指标。128最低，500/1000略回升，如实保留。
3. **best_selection/trajectory**：`best_selection.score=1.7842713233903924`是DEV source-macro objective，**不是mm**；同step128的`best_selection.trajectory.score=27.54176624612927mm`是`v5.tau-short-root-relative-contact.macro-mm.v1`，三项为short/root_relative/contact_phase，分别8.08155795/28.93192344/45.61181735mm；不是本轮手/接触窗/物体三项。两种DEV选择记录都在128最好；本轮不更改任何选择器或早停代码。

| DEV updates | hand macro mm（8项均值） | source_macro_objective | trajectory macro mm | conditioned−unknown hand macro mm（原精度） |
|---:|---:|---:|---:|---:|
| 0 | 56.152612 | 2.144058053 | 未记录 | -5.07763056046e-07 |
| 20 | 54.907403 | 2.048133373 | 29.660991 | +8.91403042935e-07 |
| 128 | 50.117074 | 1.784271323 | 27.541766 | -9.12582349688e-07 |
| 500 | 50.536784 | 1.787589699 | 28.014544 | +8.14851603081e-07 |
| 1000 | 50.831964 | 1.830530119 | 28.114541 | -3.88891055358e-07 |

DEV各source原始hand EPE（mm），禁止拿它们替代缺失的面板step500：

| DEV updates | source0 | source1 | source2 | source4 | source5 | source6 | source9 | source10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.203691 | 93.601688 | 83.311143 | 38.796989 | 49.355105 | 58.791634 | 29.209368 | 52.951282 |
| 20 | 40.472556 | 90.878909 | 83.129296 | 36.665644 | 48.051320 | 58.216669 | 31.669396 | 50.175434 |
| 128 | 38.533003 | 82.201619 | 78.046875 | 35.993432 | 41.891827 | 56.186225 | 29.332729 | 38.750884 |
| 500 | 39.025236 | 79.491814 | 77.703878 | 37.944886 | 40.480623 | 59.320554 | 32.473178 | 37.854102 |
| 1000 | 39.428237 | 78.919109 | 80.049299 | 36.765034 | 39.384447 | 61.298066 | 32.743760 | 38.067760 |

**条件路径的既有实测事实与精度说明**：step000000和step001000的conditioned−unknown均约0.0mm（四位小数为0.0000），JSON原值分别−5.07763056045718e−7与−3.8889105535844237e−7mm，而非字面精确0。本配方下有效goal条件对hand预测没有贡献：A合同本就无有效goal，属预期；当前checkpoint的**有效goal/event条件信息路径未被训练使用**。既有64条训练日志的event_goal_gradient_norm_after_clip min=max=0.0也与此一致。这里的“未使用”指有效条件信息，不把它扩张成所有参数逐位未更新。该字段实现上比较conditioned与drop_event；不是有效goal的独立A/B消融，不能外推“可用goal/event也没有价值”。上述原值与dev非单调均不是bug，本轮不因此改配方。

## 路径、实际命令、身份与资源

原脚本实测SHA256始终为 `d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51`；未修改脚本，也未建脚本副本。固定锚点SHA仍为 `440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706`。模型仍SharedWAMV6；相较brief38交付cb0d200f，model/data/events/goal_policy/flow_matching代码无diff。所有新增前向使用物理GPU4、batch4；Python3.8.20/Torch2.4.1+cu121/CUDA12.1。配方/数据/seed不变，没有训练或物理仿真。

| updates | metrics.json路径 | 来源 |
|---:|---|---|
| 0 | `src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A/metrics.json` | REUSED |
| 20 | `tmp/t38-r2/panel-step000020-A/metrics.json` | NEW_ONCE |
| 128 | `tmp/t38-r2/panel-step000128-A/metrics.json` | NEW_ONCE |
| 500 | 无（checkpoint不存在） | 卡点 |
| 1000 | `tmp/t38-r2/panel-step001000-A/metrics.json` | NEW_ONCE |
| 1223 | `tmp/t38-r2/panel-trained-A/metrics.json` | REUSED |

三个实际调用（每个仅一次；既有init/1223不重跑）：

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/training/step000020.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel-step000020-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/training/step000128.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel-step000128-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/training/step001000.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel-step001000-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
```

每个新目录的invocation.json保留确切argv、脚本SHA、起止epoch、checkpoint字节数及资源预检；panel-launch.log保留原stdout/stderr。对应推理elapsed_s分别24.972/24.431/24.800s。checkpoint三份各1217082017bytes，只读；step500不在brief38已交付的保存列表，配置checkpoint_interval=1000且DEV500不是new best，与现有保存策略一致，本轮未删除任何文件。

跨点CI仅从既有/本轮paired-panel.npz读取sums/counts/mothers，调用冻结脚本cluster_ci(...,repeats=4000)；128/1223与1000/1223的mothers/counts逐项相等。完整精度结果及检验布尔值：`tmp/t38-r2/panel-step000020-A/curve-analysis.json`。无故障门触发、无自动重启。

## 新产物字节数与Git交付

以下为记录时各新目录普通文件逻辑占用（不计符号链接），包含该目录的日志/JSON，step20还包含CSV/图及离线分析。旧checkpoint与旧init/1223面板只读，不计新产物；最终Git收据写入后有少量JSON字节增加。

| updates | 输出目录 | 文件字节快照 | 推理秒数 |
|---:|---|---:|---:|

| 20 | `tmp/t38-r2/panel-step000020-A/` | 11300610 | 24.972 |
| 128 | `tmp/t38-r2/panel-step000128-A/` | 10954311 | 24.431 |
| 1000 | `tmp/t38-r2/panel-step001000-A/` | 10954316 | 24.800 |

资源记录：`tmp/t38-r2/panel-step000020-A/resources-and-scope.json`。三次前检GPU4均2MiB占用、24252MiB空闲；磁盘起始分别为139264524288/139249111040/139237875712bytes，后检138905059328bytes，均≥20GiB。新增面板仅3次，训练更新0、rollout0，未修改原脚本、模型、配方或执行器；未删除历史数据。

只提交本新文档，tmp/outputs不入库。最终核对命令为`git log --oneline -1`、`git rev-parse HEAD`、`git ls-remote origin main`；三者实际最终结果保存于`tmp/t38-r2/panel-step000020-A/git-final-verification.json`并在交付回答逐项报出，避免文档嵌入自身commit SHA的循环。

提交前 `git pull --rebase origin main` 实测被他人未暂存修改拒绝：`cannot pull with rebase: You have unstaged changes`。未stash、未reset、未修改这些文件；按派单只显式提交本报告，再普通push，最终三项核对记录到上述Git收据。
