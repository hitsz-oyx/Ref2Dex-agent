# S2b：τ 合同、完整参考时域与冻结校准迁移预检

本轮仅修改 `src/task/ptv3-shared-wam-v6/**`；读取 v3 的合同/原产物，未修改其代码。
代码基点 `418e4c0a08fef5ee54221e7896d32ae37509c0f5` 加本次提交的 diff；没有重新训练 R、
v6 或 PPO。GPU4 只采集 204 controls；C 仅 CPU 接口 dry-run。正式 RL 继续 HOLD。

所有命令在仓库根运行。运行环境：CPU/client 使用
`/home2/wyy/miniconda3/envs/graspenv/bin/python`（Python3.8.20、NumPy1.24.4，C 的
Torch2.4.1+cu121）；物理使用固定 `dexplore_v120_train/bin/python`、Torch2.0.1+cu118。
为遵守本会话严格写入范围，机器产物保存在 Task 内忽略的 `outputs/` 和 `tmp/`，没有
向 docs 复制 NPZ/checkpoint。三个小 JSON 是可审查的结果摘要。

## A：覆盖完整 RELEASE 声明终点的参考

**Blocker/最小工程验证：** 区分原参考的过早截断与完整参考时域；决定后续能否在完整
参考上审计 executor。最便宜方法是固定同一 source-PD target 流连续重放，不拟合模型。
原脚本在首次窄任务成功时停止，81 状态不是完整任务参考。

Diff：新增 `tools/audit/collect_complete_tau_reference.py`、`complete_reference_host.py`、
`audit_complete_reference_cpu.py`，新增纯 CPU `src/ptv3_shared_wam_v6/tau_contract.py`。
host wrapper 只在内存调整输出/scratch 和短 socket 路径；保存原文与三处路径 diff，
没有改生产 v3 文件，也没有改物理、reset、PD 或任务成功条件。

| 参考 | 状态 | controls | 有效 GT-τ controls | 有效 tick | 完整终点后状态 |
|---|---:|---:|---:|---|---:|
| 历史窄参考 | 81 | 80 | 70（87.5%） | 3–72 | 未覆盖终点166 |
| 本轮连续参考 | 205 | 204 | 194（95.098%） | 3–196 | 38（1.2667s） |

原声明不变：CONTACT31、RELEASE114、完整终点166。终点后 30 个 settle tick 加 8 个
lookahead tick；在终点166至196的整整1秒，GT-τ 始终有完整8未来。0–2的 H4 启动和
197–203最后7个控制无完整前视，保留 invalid；最后状态204没有后续控制。
终点至196为31个采样状态、跨度1秒，不能按30个状态宣称1秒。

产物为 `outputs/s2b-reference-r2/reference.npz`（路径相对本 Task），由真实物理连续
采集，**未拼接旧 run、未重复终帧造状态**。与历史81状态的 q/dq/hand/object/physical/
tick/table/两个force-valid数组最大差均0，原80动作最大差0；这次确实恢复了同一前缀。
PD target 误差2.3842e-7、clipping0、安全abort0，native tick/episode错配0，正常退出。
实耗19.43s；完成后GPU4/5/6均6MiB、0%，无自有进程残留。r1因 native 改cwd导致socket
路径错误，0 controls后失败退出；保留该记录，r2修路径后成功，不把失败尝试隐去。

**新假设明确存在：** 原166个命令之后持续保持最后PD target；这不是原teacher观察到的
动作后缀，但输出状态全部由仿真实测。新物理进程按同target/seed/reset重采；前缀本次
完全相等，不承诺跨机器/未来运行逐位确定。原raw RELEASE仅作为声明意图保留。

**边界：覆盖通过，完整 RELEASE 阳性见证仍未补成。** 第204帧 scorer仍ACQUIRE、
path_complete=false、task_success=false；不能把“时间覆盖完整终态声明”称为“执行到
成功释放终态”。没有降低阈值或把失败后缀写成成功。若A要求的是物理成功终态，仍有此
blocker；延长时域本身不能解决它。也未用此新参考再次跑 calibrated-R，因此不声称延长
参考修好了旧失败。详见 [覆盖摘要](s2b-reference-coverage-20261022.json)。

复现（物理命令写入全新目录，最多204 controls、180s，使用前检查所选4/5/6空闲）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/collect_complete_tau_reference.py --output src/task/ptv3-shared-wam-v6/outputs/s2b-reference-repro --dry-run
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/collect_complete_tau_reference.py --output src/task/ptv3-shared-wam-v6/outputs/s2b-reference-repro
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/audit_complete_reference_cpu.py --run src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2
```

期望：dry-run `DRY_RUN_PASS` / 205、204、194；采集 `COMPLETED` 指时域覆盖；现有run的
CPU审计 `COVERAGE_PASS_RELEASE_NOT_QUALIFIED`，精确前缀误差0。重采如物理前缀漂移，
CPU exact-prefix断言应失败，不能悄悄改成“同一参考”。审核新run时替换最后命令的run路径。

## B：τ 通道与失持归因 CPU 合同

Diff：`tests/test_tau_channel_contract.py` 调用真实 `world_tau_to_executor` 和
`OnlineV6Runtime.predict`→`online`→`collate` 条件入口；WAM输出用spy截获，不使用随机
模型“恰巧不敏感”来证明条件缺失。`audit_tau_channels_cpu.py` 对历史NPZ逐帧重算并检查
实际τ数组/valid mask，而不是仅凭 t<参考长度猜测有效。

- GT 的 `τ[0]=h_ref[t+1]-h_ref[t]`，不是当前时刻；插值才补独立数学原点τ(0)=0。
  非线性序列验证30Hz原8行不变、60Hz首行=0.5×原首行，后续按相邻future行插值。
- Inspire8/30=0.26667s，Shadow8/60=0.13333s；不允许把相同“8步”当相同时间窗。
- GT只按参考tick推进，定义不消费 live hand/goal/event。WAM每次调用重新消费当前H4、
  G、E；更改这些输入时真实model batch相应改变。断言不声称任何已训练权重必然使用每个条件。
- 历史GT有效tick精确为3–72，共70 controls。encoder不可用时即使有未来也invalid，τ归零。
- “64/66帧首次失持”的措辞存在边界问题：校准repeat0的窄持握区间47–66，首个坏状态67；
  repeat1区间50–64，首个坏状态65。**64/66是最后正状态，65/67才是首个负状态**。
  两种边界GT均有效；首个坏状态逐数组等于固定GT reference未来−当前，排除参考过期解释。
  这是窄持握合取条件失效；observer的首次 `grasp_lost` 更晚，为104/110，不能混称。

复现：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider src/task/ptv3-shared-wam-v6/tests/test_tau_channel_contract.py
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/audit_tau_channels_cpu.py
```

期望：`7 passed`；真实run审计 `PASS`，last-held `[66,64]`、first-loss `[67,65]`、
`actual_tau_valid=true`、`coverage_expiry_explains_first_loss=false`。
[逐帧归因摘要](s2b-tau-loss-evidence-20261022.json) 保存具体字段。

## C：过拟合迁移检查仅 dry-run

Diff：新增 `tools/audit/dry_run_calibration_transfer.py`，没有训练/物理执行选项；严格加载
同一base checkpoint和历史 `calibrated.pt`，各对同一真实GT观测做一次CPU前向。
已验证686→12及两者1024/1024/512/512架构合同。

迁移退化结论为 **NOT_EVALUATED**。最小Decision screen预先固定为：

1. 见过轨迹原初态；
2. 同轨迹hand-root沿world-x平移+5mm（必须在native中一致重建q/dq/刚体/隐藏缓存及
   上次动作，并核对实测；当前reset尚未执行认证，不能只改686向量冒充新物理初态）；
3. 名称/源路径与校准不同的DEV轨迹。优先同物体；当前不存在另一同物体DEV，固定按字典序
   取有界`s10_airplane_pass_1`，173状态、原raw RELEASE；初态hand最大差0.1691m。
   物体与轨迹共同变化是局限，不能把退化精确归因于其中一个因素。

每组source-PD可行性见证、原R+GT、同一校准R+GT各一次：**9 rollouts、最多5400controls、
900s、1GiB、0updates、0WAM calls**。600/control上限是每条硬界；实际可在完成该条参考
终点+30+8后更早停止。只有source阳性的cell才解释R间比较；相同窄任务输入，完整RELEASE
另评。报告ever/final、elapsed持握、首失效时实际τ mask、动作RMSE、clip/safety/domain
与fallback。新reference不得默默替换历史训练尾部mask，更不得重选checkpoint。

预先决策：source-positive的偏移/异轨迹cell若原R成功、校准R失败，则标记观测到的退化；
持握elapsed下降≥0.2s另记预警。若原R本身不成功，binary结果不能认证不退化；最小检查
通过也仅表示此screen未发现退化，不是总体泛化或统计非劣结论。只在结果改变路线时考虑
失败cell再加两次repeat，不在本轮执行。祖先/WAM终身未见未经认证。

复现：

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/dry_run_calibration_transfer.py --dry-run
```

期望 `DRY_RUN_PASS`、`transfer_verdict=NOT_EVALUATED`，training/simulator/GPU/optimizer
均false；模型接口单点action差异不作迁移误差或性能指标。
[预检与固定协议](s2b-transfer-dry-run-20261022.json) 保存全部预算/选择/停止条件。

## 验证与 diff 查看

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider src/task/ptv3-shared-wam-v6/tests/test_tau_channel_contract.py src/task/ptv3-shared-wam-v6/tests/test_v6_executor_wiring.py src/task/ptv3-shared-wam-v6/tests/test_v6_calibration_scaffold.py
git show --stat HEAD -- src/task/ptv3-shared-wam-v6
git show HEAD -- src/task/ptv3-shared-wam-v6/tools/audit src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/tau_contract.py src/task/ptv3-shared-wam-v6/tests/test_tau_channel_contract.py
```

本轮CPU回归 **31 passed / 2.04s**；diff仅为本Task文件。Git操作按S2b要求提交、push前pull --rebase，
再核对远端SHA；实际提交/远端SHA在会话最终交付中报告。
