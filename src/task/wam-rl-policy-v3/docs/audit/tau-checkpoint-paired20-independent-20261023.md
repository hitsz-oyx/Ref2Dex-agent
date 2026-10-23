| 结果解读（20对、40回合全部完成） | 实测 | 决定 |
|---|---|---|
| 净成功不变，死因改变 | 旧2/20、新2/20；新独成功1、旧独成功1 | **不满足新τ主线替换门；不触发新权重成分分解** |
| 安全中止减少，但未形成更多抓取阶段达成 | 安全中止14→9；contact4→4，lift/hold2→2；接触未认证超时3→8 | 变化局限接近/闭合安全与获取；不称整体执行收益 |
| EPE改进已真实进入执行，收益存在正负互换 | lightbulb成功→失败；spheremedium失败→成功 | τ可被执行器消费，但当前净收益未达门槛 |

本报告是同一 A/B 的独立原始轨迹审计，**没有新增仿真、模型推理、训练或基线**。
实验身份与协议见[正式实验卡](../experiments/probes/P-20261023-tau-execution-ab.md)。
运行 `outputs/wam-rl-policy-v3/tau-checkpoint-ab-20261023-r1/`；运行提交
`98d115327215a5922028861c25174b579ac4b55b`，branch `main`。固定v11原顺序p000–p019，20物体各1条；没有独立命名的既有20例文件，因此取已冻结40例的固定前缀，不按结果选样。

旧τ：`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt`；
新τ：`tmp/t38-r2/training/step001223.pt`。两臂同当前OnlineV6Runtime、WAM goal=None/event-unknown、冻结v11 R/执行器/相位机/评分器；任务目标本身仍保留。历史旧runner传scheduler goal/event，因此两臂均重跑，没有混用历史成功率。

| 指标 | 旧step11000 | 新step1223 |
|---|---:|---:|
| 严格600拍末态成功 | 2 | 2 |
| 安全中止 | 14 | 9 |
| joint abort | 8 | 6 |
| table penetration代理门 | 6 | 3 |
| contact not certified（跑满600拍） | 3 | 8 |
| 有认证contact但未lift（跑满600拍） | 1 | 1 |
| 进入consumer close | 11 | 4 |
| 实际controls | 6237 | 8538 |

死因互斥分类先按物理中止，再按末态阶段；“接触未认证”不是新即时中止门。table penetration是未修改的CAD最低点/桌面footprint代理，不是已证实三角面压穿。

| stage_achieved（累计回合数，可重叠） | 旧 | 新 |
|---|---:|---:|
| approach | 6 | 4 |
| contact | 4 | 4 |
| lift | 2 | 2 |
| hold | 2 | 2 |
| release | 0 | 0 |

最高阶段：旧none14/approach2/contact2/hold2；新none16/contact2/hold2。release不适用。

| final_goal_error_m分位数 | 旧 | 新 |
|---|---:|---:|
| min | 0.001215155 | 0.000608216 |
| p25 | 0.077264633 | 0.076353015 |
| median | 0.229886296 | 0.080863513 |
| p75 | 0.882375230 | 0.238272756 |
| p90 | 1.619204415 | 0.949820137 |
| max | 2.985535697 | 1.782619819 |

中止回合采用最后观测状态，非补造600拍结果。配对新−旧误差中位数约0（`-1.024e-10m`）；两个边际中位数之差不能解释为典型配对改善。

| 首次触限关节（同tick并列全部保留） | DOF | 旧回合事件数 | 新回合事件数 |
|---|---:|---:|---:|
| middle_proximal_joint | 8 | 1 | 0 |
| pinky_proximal_joint | 10 | 3 | 0 |
| ring_proximal_joint | 12 | 2 | 0 |
| thumb_proximal_pitch_joint | 15 | 2 | 2 |
| thumb_intermediate_joint | 16 | 3 | 5 |

旧首触限集合：15×2、8/10/12×1、10×1、16×3、10/12×1，共8回合；新15/16×1、16×4、15×1，共6回合。事件数因并列可大于回合数。

实际消费链：前8帧sparse11位移进入686维R观测；完整24帧经刚性基点Kabsch生成腕姿序列，取预测接触前末帧（无预测接触则第24帧）作为grasp keypose。仅approach/close更新τ，lift/hold使用实测后段servo。

两个成功互换案例的首次τ查询均在tick3；此前四帧q/dq/object/hand逐元素相同。两例τ/R动作/实发动作都在tick3首次分化，q/hand从tick4分化，说明权重差异已传至实际执行。

- `p008` / `s10_lightbulb_pass_1`：成功 True→False；首次实测contact 1.1666666666666667→1.1666666666666667s；lift 1.9→Nones；reach filter 6→456controls。
- `p017` / `s10_spheremedium_inspect_1`：成功 False→True；首次实测contact None→1.7666666666666666s；lift None→3.566666666666667s；reach filter 78→9controls。

这不是组件因果分解：尚未分离前8帧/R与完整24帧/keypose各自贡献；位移范数不是EPE。新τ在lightbulb进入接触后未获得lift/hold，在spheremedium则从关节中止变为成功，不能归为单向“接近更好”。

验证：20/20配对九字段初态逐元素相同，且20/20与原v11保存初态相同；三worker全部Python运行文件一致，14个关键执行/评分/PD/host模块还与原v11冻结文件逐字节一致；40/40原始轨迹重算成功、阶段、误差与实发PD通过，故意翻转成功标签被拒；两套独立汇总的逐例终点及主要统计一致。真实model.encode入口goal/event六字段始终零/invalid。隐藏PhysX缓存未认证。

环境：客户端graspenv Python3.8.20/Torch2.4.1+cu121/CUDA12.1；native固定venv Python3.8.10/Torch2.0.1+cu118。仅物理GPU4/5/6；14775controls，0updates；最长worker1458.82s；GPU峰值5322/5320/5314MiB，执行进程已退出。运行产物约150MiB，远低于4GiB预算；额外审计仍使总量<200MiB，不复制checkpoint/视频。

证据：`independent-pair-audit.json`、`independent-consumption-final.json`、`independent-runtime-qualification.json`、`independent-g{4,5,6}-rescore.json`；主汇总`summary.json`，均位于运行目录。独立工具：
[配对与首触限审计](../../tools/audit/summarize_tau_execution_pairs.py)、[消费入口检查](../../tools/audit/inspect_tau_execution_consumption.py)。

局限：已暴露固定20面板、每臂一次，没有训练未见或重复稳健性结论；安全中止使实际时域不等长，不能把全程累计相位controls当作同等暴露率。保留旧主线τ；EPE只支持预测层判断，当前不能据此通过执行验收。

仓库文档门已运行，但另一Task既有`P-20261023-tau-fixed-panel-metric.md`缺4项schema检查而FAIL；本次原始执行/评分审计通过，不把它写成全仓verify通过。
