# v6 runtime 修复与 executor 合同复现

范围：仅本 Task。未修改 `wam-rl-policy-v3/**`；读取其 executor/clock/PPO 源码作为合同依据。
没有启动训练或仿真，没有分配 GPU。环境：graspenv绝对路径 Python3.8.20，Torch2.4.1+cu121；
CPU测试，`CUDA_VISIBLE_DEVICES=''`。结论仅为接口工程验收，官方 PPO 仍未接 v6。

## A. 根因、明确行为与 diff

Hermes 报告的路径可复现。测试 `object.__new__(OnlineV6Runtime)` 绕过真实checkpoint加载，
而其 `test_v5_contract.config()` 是小模型配置，没有 voxel 两字段。生产两份v6 JSON及
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt`
实际均为 `voxel_origin_m=[-2,-2,-2]`、`voxel_m=.005`，fps30/history4/horizon24。
因此这是**测试未提供完整runtime配置**，不是已检查的生产checkpoint丢字段。

开始工作时测试夹具已由现存未提交改动补上字段，原文件单跑为3 passed；移除这两字段后
独立复现同一 `KeyError: 'voxel_origin_m'`。新增回归测试修前2 failed，修后通过。
现有夹具修补保留在工作树，本轮不代为提交其他会话已写的测试或EncodingUnavailable改动；新的独立测试
文件包含完整的真实predict回归，因此本轮补丁本身也覆盖该调用。

明确策略：**没有隐式默认值**。缺字段、非有限三维origin、非正标量voxel均抛出带字段名的
`ValueError`；配置错误不能被当作物理不可行或普通tau-domain失效。构造模型前校验，
`predict/workspace_valid` 也校验，覆盖测试绕过构造器的情形。

关键 diff（完整补丁用下方Git命令读取）：

```diff
+def voxel_config(config):
+    missing = [k for k in ('voxel_origin_m', 'voxel_m') if k not in config]
+    if missing:
+        raise ValueError('v6 runtime config missing required key(s): '+', '.join(missing))
+    # require finite origin[3] and finite positive scalar size
+    ...
@@ OnlineV6Runtime.__init__
 self.config=packet["config"]; self.step=packet["step"]
+voxel_config(self.config)
@@ predict / workspace_valid
+origin, size = voxel_config(self.config)
-grid=np.floor((xyz-np.asarray(self.config["voxel_origin_m"]))/self.config["voxel_m"])
+grid=np.floor((xyz-origin)/size)
@@ predict
-chunks.append(self._trajectory(batch)[:,:,0][:,:,ids].float().cpu().numpy())
+chunks.append(full21_to_sparse11(self._trajectory(batch))[:,:,0].float().cpu().numpy())
```

`executor_bridge.full21_to_sparse11` 只作点索引，保持24×2×21×3→24×2×11×3和两个slot；
runtime随后选择right slot0并旋转到world，返回B×24×11×3。新增非空batch/goal/event/code
数量校验，避免少传code时静默漏环境。新增 `_synchronize` 对CPU不调用CUDA同步，使独立
合同测试可在CPU运行；CUDA路径仍在计时边界同步。

## B. 已运行复现命令及期望

从仓库根运行（所有pytest临时输出显式位于本Task，不使用系统/tmp）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_ppo_model_identity.py \
  --basetemp=src/task/ptv3-shared-wam-v6/.pytest-tmp

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_v6_executor_wiring.py \
  src/task/ptv3-shared-wam-v6/tests/test_v6_calibration_scaffold.py \
  --basetemp=src/task/ptv3-shared-wam-v6/.pytest-tmp
```

本工作树期望依次 **3 passed**、**24 passed**。合并上述三个文件为一次调用为27 passed。
其中原文件第三个测试是开始时已有的未提交代码；仅本轮commit的干净树原文件可能仍只有
两个测试。新增两文件的24项测试不依赖该第三测试。

| 合同 | 断言/范围 |
|---|---|
| full21→sparse11 | `(0,1,5,9,13,17,4,8,12,16,20)`；独立哨兵数据同时检查左右slot，防止双手错误相互掩盖 |
| runtime | 真实小型SharedWAMV6和`predict`，两环境分chunk；每个full输出24×2×21×3，最终24×11×3；非单位旋转，world位移不加平移 |
| event/goal | hook实际EventGoalResidual输入；RELEASE@8帧和CONTACT@40帧仍保留各自类型/时间；goal为相对当前物体的8cm终点 |
| mask | 只有右手11 observed joints有效；左手和skin无效，模型左手输出0；executor无效τ整块归零 |
| 时间 | τ首帧是未来1/30，Inspire前8帧不变；60Hz从隐式τ(0)=0插值；valid event_hz=60明确拒绝 |
| executor | 实际ExecutorLayout.pack/packet：411+10+264+1=686，canonical与Inspire原生11点顺序显式区分 |
| 校准输入/预算 | 跨episode、错误时钟/坐标、startup/末端mask、非法动作/来源、超过2000updates/1800秒均拒绝 |

测试用小型geometry替身，不验证真实PTv3/已训练v6权重/物理PD链。event residual初始为零，
hook证明输入实际进入模型，不据此宣称随机模型已学会事件语义。`world_tau_to_executor`是
显式可复用adapter，不能据其存在宣称官方PPO已完成接线。

**官方 PPO 未接线的可复现判据**：解析真实入口AST，必须导入并调用
`wam_rl_policy_v3.wam_v4_runtime.V4Runtime`，无v6 import或`OnlineV6Runtime`调用。
这是锁定当前源码事实的断言；以后真正改接v6时应更新该测试的预期。

```bash
CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_v6_executor_wiring.py \
  -k official_ppo --basetemp=src/task/ptv3-shared-wam-v6/.pytest-tmp
```

期望 `1 passed, 10 deselected`。此断言通过表示“确认仍未接线”，不是v6 PPO运行通过。

## C/D. 重训与校准

见 [最小重训清单及校准合同](v6-retrain-calibration-20261022.md)。脚手架dry-run已真实读取
现存Scheduled686 checkpoint并复制同架构模型，0updates/无optimizer/无仿真/无GPU。
真实单轨迹微拟合未运行，等待Hermes确认；完整RL继续HOLD。

## 完整diff

使用本次提交SHA（最终交付消息给出）读取，避免把其他会话的工作树改动混入审查：

```bash
git show <this-session-commit> -- src/task/ptv3-shared-wam-v6
```

本轮只提交自己的runtime校验/映射补丁、新bridge、新测试、新脚手架及两个v6审计文档；
保留开始前已有改动。未改动共享根文档或v3范围。
