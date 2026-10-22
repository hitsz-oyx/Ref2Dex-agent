# 结果解读表

| 可能结果 | 本轮实测 / 含义 | 决定 |
|---|---|---|
| legal 比例 ≥90% 且 legal 子集 τ 与全集差 < 噪声带 | 未触发；已认证 legal 为0/6,000，不满足覆盖前提 | 不放行训练 |
| legal 比例 <90% 但 legal 子集 τ 稳定 | 0/6,000，但子集为空，τ 稳定性未定义；不能以0%覆盖触发本行放行 | 不放行；报告补链范围 |
| legal 子集 τ 显著偏移（>噪声带） | 无法检验：legal 锚点0/512、有效帧0；差值与噪声带均为null | 不声称已证明数值偏移或对非法条件的依赖 |
| 无法在不改语义前提下判定 | 六字段各0个可认证窗口，6,000个来源未证；source-only A0物理门与512行WAM预测面板也不是同一测量 | **HOLD：不启动唯一一次训练，不改任何生产代码或config status** |

本表在分析前先写入可能结果，分析后填入实测。goal修复、event未统一、合法scheduler
重放0/6,000的已确认结论直接继承；没有重跑既有审计、模型推理或物理实验。

## 裁决与证据范围

**唯一一次训练继续HOLD。** 全量curated集合6,000个窗口的六字段合法来源，当前可认证
数均为0；联合legal-only集合精确为 `[]`，不是“找到了0%的有效事件”。原缓存事件值
本身仍然存在；缺的是将它们按原语义重建成部署事件的来源绑定与因果状态证明。

读取与代码证据冻结HEAD：`cce0abd69aa84bd3e8b532aedfaa1dae24238dd3`，branch=`main`。
这是证据代码的版本；本次文档提交SHA由最终Git回执另报。新文件仅本报告及
[逐窗口普查manifest](brief31-deployability-census-20261023.json)，均位于本轮允许的
`ptv3-shared-wam-v6/docs/audit/` 新文件范围；未创建新的outputs目录或脚本。

实际只做了新问题所需的manifest全量关联和冻结代码可达性检查，复用了：

- r7 `curated-manifest.json` 的6,000个admitted TRAIN窗口。
- r9 `scheduler-cache-identity-receipt.json` 的完整6,000个排除项及1,067组来源字段记录。
- r1 `event-source-audit.json` 的既有事件计数。
- r1 `paired-frame-records.npz` 的持久化误差和mask，以及固定anchor的identity。

没有重新检查goal修复是否通过，没有再运行事件生成/反例、scheduler重放、bootstrap
或旧面板推理。冻结代码用 `git show <HEAD>:<path>` 读取，不以另一线程后来可能改动的
工作区内容替代它。

## 1. 全量逐字段合法性普查

窗口元组固定为 **`(source_id, sequence_id, object_slot, tick_t)`**；
`sample_id=(sequence_id, object_slot, tick_t)`，H4起点为 `tick_t-3`。
元组清单是全量6,000条，非抽样；与r9来源记录逐条集合相等，没有遗漏、重复或借用DEV。
共有1,064个母序列、1,067个sequence/slot组。

这里的 **legal** 指：有可追溯的部署前指令来源，加上足以恢复对应时刻实际状态机的
≤t状态，使**原字段语义和值**得到认证。未知来源记 `UNPROVEN`；既不填false，也不
仅因与某个常量碰巧相等就记legal。“0个可认证”不等于“已证明6,000个真值全错”。

| 字段 / 上下文 | legal / 总数 | 来源未证窗口 | 恢复原语义缺少的绑定 | 部署代码证据 |
|---|---:|---:|---|---|
| type | **0 / 6,000** | 6,000 | TaskSpec.events、event_index、phase/recovering | `task_scheduler.py:82–88` |
| delta30 | **0 / 6,000** | 6,000 | planned source_frame/source_fps、source cursor、phase_since、acquire_source_origin | `task_scheduler.py:89–95` |
| valid | **0 / 6,000** | 6,000 | 已知TaskSpec.events的可用性/来源、event_index | `task_scheduler.py:84–95` |
| known-NONE | **0 / 6,000** | 6,000 | 完整已知计划及其是否已耗尽；不能把删失当NONE | `task_scheduler.py:84–87` |
| recovery | **0 / 6,000** | 6,000 | FeedbackObserver reset/state或同源因果feedback、scheduler phase/recovering、指令绑定 | `task_scheduler.py:82–83,148–153` |
| release | **0 / 6,000** | 6,000 | 有序release指令/位姿、event_index/phase/path状态、released/arrival反馈 | `task_scheduler.py:97–118,133–141` |

表内代码路径前缀均为 `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/`。
前四项是event packet及其语义；recovery/release是决定该packet的部署上下文，**不是
额外两列模型张量**。不能拿 `event_type==RELEASE` 代替已授权release，也不能把
proximity phase标签当作已记录的recovery状态。

完整来源关联依据是r9逐组库存：所有1,067组的 `discovered_intent_fields=[]`；旧事件
sidecar的来源已由上一轮确认为未来teacher。本轮只是把这些既有来源事实关联到每个
curated元组和六项认证要求，未重新遍历原始数据来重复同一审计。

| source_id | 本次覆盖窗口 | 六字段联合legal |
|---|---:|---:|
| 0 | 96 | 0 |
| 1 | 5,632 | 0 |
| 2 | 96 | 0 |
| 4 | 30 | 0 |
| 5 | 18 | 0 |
| 6 | 66 | 0 |
| 8 | 12 | 0 |
| 9 | 4 | 0 |
| 10 | 46 | 0 |
| 合计 | **6,000** | **0** |

**legal-only可用集合：`L = ∅`，绝对数0。** JSON的
`legal_only_window_tuples=[]`，`unproven_windows`列出全部6,000个原始元组、六字段
状态和未准入原因；字段级 `proven_illegal=0`、`unproven=6000`，保持证据边界。

既有curated事件计数直接引用：raw-valid5,983、raw-invalid17，valid known-NONE986、
CONTACT1,458、RELEASE3,539。其中17条raw-invalid的数值packet可能等于online默认
unknown，但原掩码的来源是teacher可见性；没有同一部署时刻的TaskSpec/状态机绑定，
不能以“填一个unknown恰好相等”认证其完整语义。它们不自动成为17条legal训练样本。
把全部6,000条统一unknown则会改变原5,983条有效事件的条件，是本轮禁止的合同变更。

## 2. τ天花板管线到底消费什么

### source-only A0物理门

冻结代码显示它是source PD执行链资格门，**不是WAM τ EPE预测器**：

| file:line | 直接证据 |
|---|---|
| `tools/run/run_source_ceiling40.py:45–60` | registration仅`arms=['A0']`，每job仅`order=['A0']`，0updates；scope明确source-only、not tau comparison |
| 同文件`:74,99` | supervisor验证40个job都仅A0，运行冻结worker |
| `tools/run/run_source_ceiling_worker.py:68` | worker拒绝非A0、PPO或residual checkpoint |
| 同文件`:117` | `wam=SimpleNamespace(step=None,seconds=0.)`；没有实例化OnlineV6Runtime |
| 同文件`:198–206` | tau初始化为0、valid=False；唯一`wam.predict(...next_event.wam_condition()...)`位于`arm in ('A1','A2')`分支，A0下不可达 |
| 同文件`:217–228` | 实际intent来自SourceGraspReference；consumer收到pose/q/dq/hand/palm、forces、measured feedback、几何与intent，不收E_next字段 |
| `src/wam_rl_policy_v3/source_ceiling.py:76–127,140–163` | 先验source手腕/手指/力/位姿提取抓取意图；manifest明确`consumes_R=False`，不是GT-τ/R资格 |
| 同文件`:234–250` | approach/close消费live q/dq/palm/object_pose/forces/feedback/intent，后抓取继承既有servo |
| `src/wam_rl_policy_v3/executor_contract.py:12–20` | task10仅goal局部translation/rotation、path_progress、当前phase one-hot；无event type/delta/valid |
| `tools/run/run_source_ceiling_worker.py:239–243` | 结果是physical hold/goal/safety成功、wam_calls与source_R_consumed_controls；并非预测EPE |

表内路径前缀为 `src/task/wam-rl-policy-v3/`。虽然worker import了WAM类型，另有不可达
事件调用，也计算/记录actor observation，**import、日志字段和死分支不等于A0实际消费**。
其TaskController产生测量feedback和任务状态，但不会把训练集E_next张量送入source consumer。
source参考轨迹自己的事前接触/抓取意图也不能等同于curated训练事件packet。

按本brief指定source-only管线的范围：**「事件合同只影响 v6 训练时条件，不影响 τ 面板测量」**。
这句话限定为上述A0/source执行器资格面板“不消费v6的E_next packet”，不是对所有
使用v6的评估都作出的断言。A0日志中的tau=0/valid=False也不表示预测误差为0。

### 512固定TRAIN预测面板

它是另一条管线，实际消费v6事件。冻结的v6
`tools/run/score_tau_panel_trivial_baselines.py:124–127,166–169` 使用V6Windows和
collate后直接`model(batch)`；`src/ptv3_shared_wam_v6/model.py:138–139` 实际加上
`self.event_goal(batch)`。这里沿用已确认事实，不重新执行入口审计。

因此不能用source-only A0没有event调用，推出512行WAM预测误差对event惰性；也不能
用512行持久化误差重算source-only物理40初态成功率。这两种“天花板/面板”不是同一
总体、接口或指标。本轮不更改任何一条管线来强行建立对应关系。

## 3. legal-only子集上的离线重算

已在持久化NPZ上完成元组过滤：512条记录与固定anchor逐条匹配；其与6,000条curated
集合交集为 **480条**，不是512条全部属于新训练池。与 `L` 的交集为 **0条**。
无新推理，无GPU使用，无旧预测/标签/面板覆盖。

| 既有口径 | 全512缓存有效帧 | 全512 B1−B0a配对帧中位差mm | legal-only有效帧 | legal-only统计 / 与全集差 |
|---|---:|---:|---:|---|
| overall | 12,216 | -42.213406 | **0** | **null / null** |
| first8 | 4,091 | -17.135709 | **0** | **null / null** |
| contact-window | 7,416 | -31.158743 | **0** | **null / null** |

状态统一为 `UNDEFINED_EMPTY_LEGAL_SET`；CI与噪声带也记null，不能填0或NaN后参与
比较。这里复用的是原配对帧误差口径，未将帧均值重新当成有效点加权mean EPE。

能量化的影响是**当前合同下可认证训练窗口损失6,000/6,000、legal面板可评估记录0**。
无法由空集量化数值τ偏移，既不能声称“差异小于噪声带”，也不能声称已经证明偏移
显著或旧上界依赖非法条件。信息缺口就是没有可比的合法条件/样本，而不是模型吞吐。
额外GPU4推理不能补出缺失的指令和状态链，故未申请/使用GPU4。

## 4. 事件敏感性对照裁决

**不做事件扰动对照。** brief第2步指定的source-only A0不消费event packet，故第4步
条件未触发。没有为了凑实验在不可达分支注入事件，也没有伪造一次“敏感性为0”。

512行WAM面板虽然消费event，但当前没有可绑定、可重建的legal事件比较态；用常量
NONE、全部unknown、GT倒计时或假定recovery=False替换，都会改变本轮要求保留的
合同语义。不能借此绕过“无法不改语义判定则停止”的规则。无合法对照，因而也不
执行无助于裁决的新重复推理来估噪声带。

## 5. 补链范围与训练启动裁决

最低需要解决的绑定粒度为 **1,064个母序列、1,067个对象流**。按现有选中tick，若从
已知episode起点精确重放到各组最晚选中t，共涉及：

- 母序列时刻记录260,526条（同母多对象共享时钟计一次）。
- 对象流时刻记录261,720条；按30Hz归一化为8,724对象流秒，约2.423对象流小时。
- 每个对象流需要可认证的事前任务意图与因果feedback/observer/scheduler reset或状态链。

这是现有窗口的**覆盖工作量**，不是新增训练预算、实测机器人采集耗时或可直接认领的
补录成本。人类示教不天然带机器人力/feedback；能否追回原始独立TaskSpec、是否需要
新录制、是否有合法snapshot都未确认，不能编造一个工时数字。已有合法snapshot可
替代前缀重放，但新造window reset、未来回填snapshot或伪造力信号均不能代替补链。

**启动裁决：HOLD，不放行唯一一次训练。** 原表前两行的放行均需要非空legal集合及
可比较τ结果；本次两者都没有。goal代码修复已确认通过，不因本结论被重新判坏。
数据侧合法指令/事件绑定覆盖为0，是独立于goal实现修复的启动条件缺口。
只要原语义无法认证，本轮就按第四行停止，不修改生产代码、训练config或他人任务。

## 验证、复现命令与写盘字节数

**12/12项只读合同/完整性检查PASS，失败0；pytest新运行0、新增测试代码0。** 检查
包括全量键唯一性、旧来源记录精确对应、1,067组覆盖、无已绑定意图、A0注册、WAM
不实例化、唯一predict在不可达A1/A2内、consumer/task10无event packet字段、legal
集合为空、缓存anchor精确对应、480/0交集、空集结果为null而非0。清单见JSON的
`verification_checks`。本轮无生产代码变更，不重复既有154项或事件审计测试。

环境：Python3.8.20、NumPy1.24.4，CPU manifest关联与缓存记录过滤；未导入模型运行时。
可在仓库根执行以下只读复核，不产生文件或启动worker：

```bash
git show cce0abd69aa84bd3e8b532aedfaa1dae24238dd3:src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py | sed -n '45,60p'
git show cce0abd69aa84bd3e8b532aedfaa1dae24238dd3:src/task/wam-rl-policy-v3/tools/run/run_source_ceiling_worker.py | sed -n '65,68p;117p;198,206p;217,228p'
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python - <<'PY'
import json, pathlib, numpy as np
root = pathlib.Path('.')
base = root/'outputs/ptv3-shared-wam-v6'
c = json.load(open(base/'tau-curated-20261023-r7/curated-manifest.json'))
old = json.load(open(base/'tau-curated-20261023-r9/scheduler-cache-identity-receipt.json'))
r = json.load(open(root/'src/task/ptv3-shared-wam-v6/docs/audit/brief31-deployability-census-20261023.json'))
key = lambda x: (int(x['source_id']), int(x['sequence_id']), int(x['slot']), int(x['tick']))
keys = {key(x) for x in c['selection']['train_window_rows'] if x.get('admitted') is True}
assert len(keys) == 6000
assert keys == {(int(x['source_id']), *map(int, x['sample_id'])) for x in old['excluded_windows']}
assert {tuple(x['window_tuple']) for x in r['unproven_windows']} == keys
assert r['legal_only_window_tuples'] == []
assert all(v['legal'] == 0 and v['total'] == 6000 and v['unproven'] == 6000 for v in r['fields'].values())
assert r['passed'] == 12 and r['failed'] == 0
with np.load(base/'tau-contract-20261023-r1/paired-frame-records.npz') as z:
    print('cached rows:', len(z['sequence_id']), 'legal rows:', r['cached_panel']['legal_intersection_rows'])
    print('full valid frames:', int(z['frame_valid'].sum()))
print('field census:', r['fields'])
print('legal-only tuples:', r['legal_only_window_tuples'])
print('legal metrics:', r['cached_panel']['metrics'])
print('supplementation:', r['supplementation'])
PY
```

写盘前预估<16MiB，硬上限2GiB；`/home2`当时可用148,142,641,152 bytes。

本轮2个产物逻辑大小合计 **1,367,442 bytes**；训练 **0 updates**、checkpoint新增 **0 bytes**、config_status不变。
