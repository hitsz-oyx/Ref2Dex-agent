# Source 控制门控离线分诊（2026-10-23）

本报告只读取已完成的 R3/R4/R5 轨迹、注册 reference 和保存的 A0 action；没有启动仿真、模型或 GPU。审计工具：`tools/audit/audit_source_gate_triage.py`。

## 结论

引用几何本身不自洽：R3/R4 40 例的 reference frame 0→首次有效双指力帧，在物体坐标系中的位置差中位数为 1.1225 m（p10 0.9801，p90 1.2495），旋转差中位数 2.0256 rad（p10 1.5380，p90 2.8760）；相对每例门限的中位数倍率分别为 83.23× 和 20.77×。按每例 `geometry.wrist_gap_m` 和 `angle_tolerance_rad`，位置/旋转双门命中为 0/40，位置和旋转各自也都是 0/40。这个结果支持“录制引用的 approach→grasp 过远”这一实现/接口缺陷假设。

但这不是 R3/R4 runtime intent 的唯一解释：A0 走 `grasp_intent_from_retargeter(..., plan=None)` 时，代码把 `grasp_pose_object` 直接复制为当拍 `approach_pose_object`，所以 intent 内部两 pose 差为零（R3/R4 所有 A0 轨迹 `tau_valid` 也均为 false）。R3/R4 阶段门实际还要求 `near_fingers >= 2`，或腕位置差≤`wrist_gap_m`且指尖最小距离≤`tip_gap_m`；历史 R3/R4 代码没有把 `angle_tolerance_rad` 用进该门。

R4 最后一个控制时刻的 40 例中，31 例仍为 `approach`、9 例已经进入 `close/lift`。在仍为 approach 的 31 例里，20 例的重建腕目标已满足位置+旋转静态门，但指尖接近门不满足；另 11 例腕目标本身仍未满足静态门。这解释了“接近但不进入 close”与“进入 close 但不认证 contact”两个层次。R4 的 5 个 `contact_without_lift` 属于后者之后的 live coupling/lift 失败，不是限位主因。

R5 的 `raw_force_contact_frames`/`object_contact_frames` 是 source 录制证据，不是 live success；`source_contact_is_live_success` 在生产 `SourceGraspReference.manifest()` 中明确固定为 false。R5 40 例中 39 例有 source object-contact 帧、25 例运行时曾有 measured contact、13 例曾进入 lift/hold、21 例阶段曾达到 contact；因此 71/69 帧与 live success=false 并不矛盾，缺的是运行时 `FeedbackObserver` 的运动耦合持续条件及后续 lift/goal 条件。

## 表 1：引用 approach→grasp 物体坐标几何

R3/R4 的 approach frame 定义为 reference frame 0，grasp frame 为首次 `contact_force_valid` 且至少两指力≥0.1 N 的帧；R5 使用 `source-keys.json` 的 `grasp_frame`。

| run | position p10 / median / p90 (m) | rotation p10 / median / p90 (rad) | position hit | rotation hit | both hit | position >5× | rotation >5× |
|---|---:|---:|---:|---:|---:|---:|---:|
| r3 | 0.9801 / 1.1225 / 1.2495 | 1.5380 / 2.0256 / 2.8760 | 0/40 | 0/40 | 0/40 | 40/40 | 40/40 |
| r4 | 0.9801 / 1.1225 / 1.2495 | 1.5380 / 2.0256 / 2.8760 | 0/40 | 0/40 | 0/40 | 40/40 | 40/40 |
| r5 | 0.9902 / 1.1416 / 1.2450 | 1.2677 / 1.7532 / 2.6560 | 0/40 | 0/40 | 0/40 | 40/40 | 40/40 |

字面上的 runtime `GraspIntent` 字段另有一个重要结果：R3/R4 A0 因 `plan=None`，R5 因 `SourceGraspReference.intent()`，三版都把 `grasp_pose_object` 设成与 `approach_pose_object` 相同；因此 intent 内部位置/旋转差为 0、40/40 自洽。表 1 检查的是录制 reference 的实际 approach frame→grasp/contact frame，不把这两个层次混算。

逐例明细已写入审计 JSON；R4 逐例门控表如下。

## 表 2：R4 最后控制时刻的腕误差与实际阶段门

| id | object | phase_at_last_control | palm_object_origin_distance_m | wrist_position_error_m | wrist_gap_m | wrist_rotation_error_rad | angle_tolerance_rad | near_fingers | tip_gap_m | tip_gap_tolerance_m | final_goal_error_m | wrist_static_gate | code_gate | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p000-A0 | bowl | approach | 0.17342846423731528 | 0.13917618739064466 | 0.0191373152634874 | 0.36671338081936483 | 0.10297479007780296 | 0 | 0.02087769053686183 | 0.0191373152634874 | 0.3201255542913973 | False | False | wrist_not_at_target |
| p001-A0 | camera | approach | 1.1266127655141072 | 3.935912961184649e-06 | 0.011751516946479423 | 1.3228088838991654e-05 | 0.07257555031347399 | 0 | 1.1341887027777258 | 0.011751516946479423 | 0.08261721136361798 | True | False | wrist_at_target_finger_gate_false |
| p002-A0 | cylinderlarge | approach | 0.37895428426582933 | 0.11471815834692459 | 0.029855780943017086 | 0.07496928916010066 | 0.1409944786436261 | 0 | 0.13403819481133983 | 0.029855780943017086 | 0.08116203685904065 | False | False | wrist_not_at_target |
| p003-A0 | eyeglasses | close | 0.20518886141984483 | 0.5025656192591573 | 0.010858930957587451 | 0.6301975959167769 | 0.04786729542862004 | 1 | 0.003872579527724398 | 0.010858930957587451 | 0.07709219764285488 | False | False | phase_gate_reached_contact_or_lift_missing |
| p004-A0 | fryingpan | close | 0.2628892889734557 | 0.284861656558967 | 0.00825036042344925 | 0.24883442523233112 | 0.02812751112535778 | 0 | 0.02626353204947757 | 0.00825036042344925 | 0.08150314825688615 | False | False | phase_gate_reached_contact_or_lift_missing |
| p005-A0 | hand | approach | 1.1894623305371879 | 1.1137547521018157e-05 | 0.014628433488508789 | 2.685458573715174e-05 | 0.08002671160709375 | 0 | 1.0535741630570896 | 0.014628433488508789 | 0.07553223530083535 | True | False | wrist_at_target_finger_gate_false |
| p006-A0 | headphones | approach | 0.8894592533441231 | 0.3476612680961631 | 0.01359830387089989 | 2.0705696359882895 | 0.07573740570741042 | 0 | 0.896506131906927 | 0.01359830387089989 | 0.07646235126029365 | False | False | wrist_not_at_target |
| p007-A0 | knife | lift | 0.16528974450116424 | 0.04696784001759526 | 0.003711708438447022 | 0.05122673343664691 | 0.01405518915702597 | 4 | 0.007195896087932096 | 0.003711708438447022 | 0.016516675091994126 | False | True | phase_gate_reached_contact_or_lift_missing |
| p008-A0 | lightbulb | approach | 1.078916499335187 | 1.2153781947891439e-05 | 0.0152001100685762 | 2.279476193953935e-05 | 0.11055410807881523 | 0 | 1.0322100720436842 | 0.0152001100685762 | 0.07833112314888671 | True | False | wrist_at_target_finger_gate_false |
| p009-A0 | mouse | approach | 1.0371865418450799 | 1.2365917652184728e-05 | 0.009755645069724744 | 4.833630354502118e-05 | 0.06500615990716252 | 0 | 0.9301650386405255 | 0.009755645069724744 | 0.07600646370766911 | True | False | wrist_at_target_finger_gate_false |
| p010-A0 | phone | approach | 1.0181580691983503 | 1.2055675355102875e-05 | 0.005699437497611447 | 4.2240594893956086e-05 | 0.039026748121311884 | 0 | 0.8522458051466704 | 0.005699437497611447 | 0.07654577940766609 | True | False | wrist_at_target_finger_gate_false |
| p011-A0 | piggybank | approach | 1.4965405225946398 | 1.073773097796188e-05 | 0.030000593334251008 | 3.057236363925504e-05 | 0.12659289272598806 | 0 | 1.4225805157702565 | 0.030000593334251008 | 0.08129204257332537 | True | False | wrist_at_target_finger_gate_false |
| p012-A0 | pyramidlarge | approach | 0.39745937689339295 | 0.15083690312177359 | 0.022295384464663042 | 0.009620634445730003 | 0.13178295438296908 | 0 | 0.16353221707668922 | 0.022295384464663042 | 0.08202526768192975 | False | False | wrist_not_at_target |
| p013-A0 | pyramidmedium | approach | 0.9857962206297934 | 6.258615817085519e-06 | 0.01778733302677124 | 1.8139328037527677e-05 | 0.12809826079851597 | 0 | 0.8909867139213175 | 0.01778733302677124 | 0.07602500802580611 | True | False | wrist_at_target_finger_gate_false |
| p014-A0 | pyramidsmall | approach | 0.8451930592602845 | 1.1754042122348198e-05 | 0.008893665771015536 | 4.513007986956082e-05 | 0.12809824407866136 | 0 | 0.7451570883277542 | 0.008893665771015536 | 0.07750425143686224 | True | False | wrist_at_target_finger_gate_false |
| p015-A0 | scissors | approach | 1.099702249105501 | 1.2312740108134741e-05 | 0.0040491688000370875 | 1.976896236157661e-05 | 0.01613291596432642 | 0 | 0.9298257125802625 | 0.0040491688000370875 | 0.07442198527068788 | True | False | wrist_at_target_finger_gate_false |
| p016-A0 | spherelarge | approach | 1.2115633222901885 | 1.2361605424547186e-05 | 0.02620143722742796 | 2.0637517352564183e-05 | 0.18079028605728756 | 0 | 1.0180052160169435 | 0.02620143722742796 | 0.07565257763504561 | True | False | wrist_at_target_finger_gate_false |
| p017-A0 | spheremedium | close | 0.1869548462304716 | 0.4188889379437213 | 0.018151842099007956 | 0.4451431551789323 | 0.18143819009584108 | 4 | 0.0019315831254566262 | 0.018151842099007956 | 0.15570656059974372 | False | True | phase_gate_reached_contact_or_lift_missing |
| p018-A0 | spheresmall | approach | 0.1824692364871405 | 0.32603997938762674 | 0.00913851283335761 | 0.45910942978769986 | 0.1814381712463284 | 1 | 0.013933098871490723 | 0.00913851283335761 | 0.07669329164678067 | False | False | wrist_not_at_target |
| p019-A0 | stamp | approach | 0.9211817595813014 | 1.2248963218277502e-05 | 0.01590384536667647 | 3.0032623999689925e-05 | 0.11442269026143033 | 0 | 0.8998028719623713 | 0.01590384536667647 | 0.07618831266056758 | True | False | wrist_at_target_finger_gate_false |
| p020-A0 | bowl | approach | 0.16064983285023585 | 0.1933716653518859 | 0.0191373152634874 | 0.17524692424561153 | 0.10297479007780296 | 0 | 0.020974848661378387 | 0.0191373152634874 | 0.0696812744513383 | False | False | wrist_not_at_target |
| p021-A0 | camera | close | 0.19408854779703213 | 0.0433965871553198 | 0.011751516946479423 | 0.01368834178830023 | 0.07257555031347399 | 2 | 0.00933784426592499 | 0.011751516946479423 | 0.08502312955932534 | False | True | phase_gate_reached_contact_or_lift_missing |
| p022-A0 | cylinderlarge | approach | 0.34631278260304355 | 0.11754802636128855 | 0.029855780943017086 | 0.13267826108779843 | 0.1409944786436261 | 0 | 0.1392943719347004 | 0.029855780943017086 | 0.08181835628120018 | False | False | wrist_not_at_target |
| p023-A0 | eyeglasses | approach | 0.82246948109775 | 0.1254484760016079 | 0.010858930957587451 | 0.007763193282275017 | 0.04786729542862004 | 0 | 0.5512894313615198 | 0.010858930957587451 | 0.07692338648193393 | False | False | wrist_not_at_target |
| p024-A0 | fryingpan | approach | 1.1320641098228776 | 1.8361131575039171e-06 | 0.00825036042344925 | 1.6303111084831576e-05 | 0.02812751112535778 | 0 | 0.9765407271886554 | 0.00825036042344925 | 0.07515972610483555 | True | False | wrist_at_target_finger_gate_false |
| p025-A0 | hand | approach | 1.189913851880869 | 1.0012040467666492e-05 | 0.014628433488508789 | 2.3693065580684802e-05 | 0.08002671160709375 | 0 | 1.0157538683501537 | 0.014628433488508789 | 0.07564419596145647 | True | False | wrist_at_target_finger_gate_false |
| p026-A0 | headphones | approach | 1.2338851300433438 | 7.0620598258357835e-06 | 0.01359830387089989 | 2.395113536805981e-05 | 0.07573740570741042 | 0 | 1.0663831722279464 | 0.01359830387089989 | 0.07922851003027745 | True | False | wrist_at_target_finger_gate_false |
| p027-A0 | knife | approach | 0.19850502693647945 | 0.09763361111835771 | 0.003711708438447022 | 0.014447800732487862 | 0.01405518915702597 | 0 | 0.03376051827658333 | 0.003711708438447022 | 0.0755294003724161 | False | False | wrist_not_at_target |
| p028-A0 | lightbulb | approach | 1.2806383071123484 | 1.0876271060883397e-05 | 0.0152001100685762 | 2.667565533631465e-05 | 0.11055410807881523 | 0 | 1.0872956726059018 | 0.0152001100685762 | 0.07856568012769959 | True | False | wrist_at_target_finger_gate_false |
| p029-A0 | mouse | approach | 0.19289179340410512 | 0.12219733752028768 | 0.009755645069724744 | 0.05949147635336045 | 0.06500615990716252 | 0 | 0.06455642436982718 | 0.009755645069724744 | 0.076836974652278 | False | False | wrist_not_at_target |
| p030-A0 | phone | approach | 1.2684097089685917 | 1.2500565616715468e-05 | 0.005699437497611447 | 3.4210576884663757e-05 | 0.039026748121311884 | 0 | 1.2546949631691378 | 0.005699437497611447 | 0.08389754541201734 | True | False | wrist_at_target_finger_gate_false |
| p031-A0 | piggybank | approach | 1.227103519550322 | 9.016885698351786e-06 | 0.030000593334251008 | 8.147057609558996e-06 | 0.12659289272598806 | 0 | 1.2257940241980654 | 0.030000593334251008 | 0.08111245610517973 | True | False | wrist_at_target_finger_gate_false |
| p032-A0 | pyramidlarge | close | 0.18701836839507796 | 0.41768803322117287 | 0.022295384464663042 | 0.5797649816702669 | 0.13178295438296908 | 5 | 0.004042275749556287 | 0.022295384464663042 | 0.08209237798196933 | False | True | phase_gate_reached_contact_or_lift_missing |
| p033-A0 | pyramidmedium | close | 0.17391178769225535 | 0.40656599944033656 | 0.01778733302677124 | 0.34170520168879004 | 0.12809826079851597 | 4 | 0.003925938868510763 | 0.01778733302677124 | 0.07638679787248193 | False | True | phase_gate_reached_contact_or_lift_missing |
| p034-A0 | pyramidsmall | approach | 1.1827177916644112 | 7.309851475608639e-06 | 0.008893665771015536 | 1.8028725627628518e-05 | 0.12809824407866136 | 0 | 1.010323240921323 | 0.008893665771015536 | 0.07746328763629286 | True | False | wrist_at_target_finger_gate_false |
| p035-A0 | scissors | approach | 1.237561000869003 | 1.1126156897160511e-05 | 0.0040491688000370875 | 1.077835060597926e-05 | 0.01613291596432642 | 0 | 0.9916474398699741 | 0.0040491688000370875 | 0.07995286414264535 | True | False | wrist_at_target_finger_gate_false |
| p036-A0 | spherelarge | close | 0.21918923878655566 | 0.34874920504889284 | 0.02620143722742796 | 0.600340584588484 | 0.18079028605728756 | 4 | 0.003650950958729764 | 0.02620143722742796 | 0.07089655789430778 | False | True | phase_gate_reached_contact_or_lift_missing |
| p037-A0 | spheremedium | approach | 0.9878274009274546 | 2.0001599082659922e-06 | 0.018151842099007956 | 3.520404652516698e-06 | 0.18143819009584108 | 0 | 0.9307007467854713 | 0.018151842099007956 | 0.07640984547174386 | True | False | wrist_at_target_finger_gate_false |
| p038-A0 | spheresmall | close | 0.17347694682045225 | 0.2213661000719463 | 0.00913851283335761 | 1.0931960154792422 | 0.1814381712463284 | 2 | 0.004450813252606746 | 0.00913851283335761 | 0.09329115305277419 | False | True | phase_gate_reached_contact_or_lift_missing |
| p039-A0 | stamp | approach | 0.5803775100944547 | 0.14541114506016029 | 0.01590384536667647 | 0.11628898647204076 | 0.11442269026143033 | 0 | 0.3892538134566185 | 0.01590384536667647 | 0.07580353473732492 | False | False | wrist_not_at_target |

R4 分类计数：`phase_at_last_control=approach` 31/40；其中腕静态位置+旋转门已满足但指尖门不满足 20/40，腕静态门未满足 11/40；已经进入 close/lift 9/40。p000-A0 的实测 palm→object 原点距离为 0.1734 m、腕目标位置误差 0.1392 m、腕目标旋转误差 0.3667 rad，而 `final_goal_error_m` 为 0.3201 m；后者是物体当前位姿到任务 post-grasp goal 的误差，不能替代腕到 grasp target 的误差。

## 表 3：contact 认证路径与 R5 实测

| 路径 | 全部条件 | R4/R5 证据 |
|---|---|---|
| `FeedbackObserver.contact_observed` | `force_valid`; 至少 2 个力向量范数≥0.1 N；`InteractionObserver` 的 `near_fingers`≥2（指尖到表面≤18 mm） | R5 source 证据不是该 live 标志；R5 40 例中 25 例运行时曾有 `measured_contact` |
| `FeedbackObserver.grasp_candidate` | `contact_observed` 持续至少 0.12 s | R4/R5 consumer 未把 source force 帧直接转成 candidate；R4 仅 9/40 曾离开 approach |
| `FeedbackObserver.grasp_established` | `InteractionObserver.coupled`：near≥2、相对速度<0.04 m/s、物体速度>0.004 m/s、相对速度≤0.001+0.35×物体速度，并持续0.12 s；若 force_valid 还需至少2指力 | R5 仅 13/40 曾进入 lift/hold；source `raw/object` 帧不满足该 live coupling 条件 |
| close→lift | `contact_since` 持续≥0.12 s 且 `feedback.grasp_established` | R4 5 例 `contact_without_lift`；R5 仍需按 live coupling/lift 解释 |
| source key manifest | `raw_force_contact_frames`/`object_contact_frames` 只描述 reference；`source_contact_is_live_success=False` 是显式合同 | R5：raw 总计4724（中位109），object 总计4466（中位102），39/40 object-contact>0；p006 为 raw71/object69、live=false、goal_within_geometry=false |

实现路径（当前代码行号）：`task_feedback.py:47-57` 统计力、near-finger 和 0.12 s candidate；`interaction_state.py:64-87` 定义 18 mm proximity 与 motion coupling；`pose_consumer.py:219-231` 定义 approach→close 与 close→lift；`source_ceiling.py:76-128` 把 source 接触计数与 live success 分开，并显式写入 `source_contact_is_live_success=False`；`summarize_source_ceiling.py:18-23` 把未达 stage contact 标为 `stage_contact_not_certified`。

## 决策与最小修复建议

本轮不启动 R6 全 40。应先在 3–5 个已有失败例上修正引用/门控接口：要么用引用中同一物体坐标系的可达 approach→grasp 几何导出腕门限/分段判据，要么明确把 source 的实测关键 grasp pose 作为 grasp intent，并保留 live contact/coupling 认证；不能把 source reference force contact 当成 live success。修复后只做小样本回放，确认 approach→close 可达、close→lift 仍由 live coupling 控制，再决定全批。
