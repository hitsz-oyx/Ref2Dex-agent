# 结果解读表

| 可能结果 | 实测与含义 | 决定 |
|---|---|---|
| 信号齐、小样本legal高、GT偏差小 | 未发生：完整链可准入0/1,064母序列 | 不能建议直接全量重导或放行训练 |
| 信号齐、legal高、GT偏差大 | 未发生：没有可比较的legal输出，GT偏差未定义 | 不编造一致率或错配数值 |
| 信号齐但legal低于50% | 本次不是“信号齐后重放失败”：30母序列/409窗口在输入门即全部拒绝 | 不把0%预检准入率当成状态机性能；unknown-mask仅列机主选择项 |
| 关键信号缺失 | **命中：机器人contact_forces、force_valid、原生object velocity、support/workspace及完整TaskSpec可用绑定均0/1,064** | 保留完整现部署事件链就需新录rollout；不能只重导。不训练，交机主选择是否改条件 |
| 原型接口或语义失败 | 未触发实际录制重放；独立合成完整输入的同状态机工程测试通过 | 真实输入不足时停止，不改生产代码 |

本表在普查前写入可能结果，普查后填入数字。brief31结论、source-only A0不消费event
以及既有补录信封直接继承，不重复运行。**训练0updates、GPU使用0、新rollout0、
生产语义和config_status变更0。**

## 裁决：完整现部署链需要新录；低成本路线需要明确换条件

这次追到了所有curated母序列的上游录制，而非再次查看旧overlay说“没存路径”。
共清点 **1,064母序列、6,000窗口、4,232个去重录制层，全部READABLE**。
原始列中有pose、hand、clock，但没有可绑定到当前机器人接口的力反馈、力有效性、
support/workspace与原生对象速度，独立TaskSpec绑定也不足。不能从人类几何轨迹
重导出当时没有测量的机器人力或原生机器人状态。

**如保留当前force/计划/反馈驱动的完整事件语义，需要新录robot rollout。**
这不表示“任何无力的event估计都不能运行”；生产代码确实有force-free路径，但它
不确认同样的release，也没有当前measured support语义。采用它或观测差分速度属于
另一个录制/条件合同，不能悄悄当作本次保真重导。

如首要目标是低成本保留原6,000个训练窗口，则建议机主审议下文的**统一
unknown-event mask配方变更**；本轮只设计，未应用。既有goal准入和训练放行约束
仍遵守，不凭本报告自动启动唯一一次训练。

## 冻结身份与全量清点边界

背景冻结HEAD：`349ff9466f65fc18ad9428e9c792eabe06ff35f5`；审计工具与库存实现
checkpoint：`8d3bd9e95b108615f7170541442aad6a676a09d0`，branch=`main`。
库存执行期间共享HEAD前移到e2d6e347，原JSON如实记录；未修改该线程的source-ceiling
或任何执行器代码。新增工具、tests、研究记录均在本轮允许范围。

[完整库存](brief32-raw-signal-inventory-20261023.json)逐母序列记录：

1. r7对应parent recipe里的v5手/物体overlay。
2. `local-data-20261021-r2/data/plan.json` 直接绑定的processed序列目录或native NPZ。
3. 既有inventory绑定的hand packet。
4. plan明确给出的原始raw文件，例如GRAB原始NPZ、ARCTIC MANO原始NPY。

每个目录枚举所有NPY头及JSON metadata，每个NPZ枚举所有NPY成员头；不按source抽一条
然后推广。只读取头和metadata，不执行pickle；**1,043个层含opaque object dtype**，
其内未被解包的human标签不能当作已验证机器人力。结论范围是这些逐序列可追溯的
录制/导出接口，不声称全服务器任意未绑定档案都不存在信号，也不借另一线程rollout补洞。

| source_id | 母序列 | 机器人力可用 | 原生对象六维速度可用 | 显式记录goal位置 |
|---|---:|---:|---:|---:|
| 0 | 3 | 0 | 0 | 0 |
| 1 / GRAB | 1,040 | 0 | 0 | 0 |
| 2 | 3 | 0 | 0 | 0 |
| 4 / Shadow | 3 | 0 | 0 | **3** |
| 5 | 3 | 0 | 0 | 0 |
| 6 | 3 | 0 | 0 | 0 |
| 8 | 3 | 0 | 0 | 0 |
| 9 | 3 | 0 | 0 | 0 |
| 10 | 3 | 0 | 0 | 0 |
| 合计 | **1,064** | **0** | **0** | **3** |

“母序列”沿brief31的sequence_id口径；字段存在与逐tick观测全部有效是不同检查。
这次未重复旧mask/未来标签审计，也未把已知84个前缀缺口用插值补齐。

## 1. 逐信号可得性表

下表消费者路径前缀为 `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/`。
“直接覆盖”只表示原始接口列可读；“TaskSpec绑定覆盖”要求它在episode前已知并且
绑定到此执行，不能用观测列改名充当计划。

| 信号 | 部署消费者 file:line | 现有录制字段名 | 覆盖母序列数/总 | 缺失 / 可重导判定 |
|---|---|---|---:|---|
| ≤t object pose | `task_feedback.py:39–45`；`task_scheduler.py:124–126` | overlay `object_poses`，native `poses/object` | **1,064/1,064**列存在 | 有观测；不能当独立object_path指令。source10插入帧不是新的原生观测 |
| ≤t hand11 | `task_feedback.py:45` → `interaction_state.py:46–60` | `joints/hand`、validity、hand packet | **1,064/1,064**列存在 | 有几何；手槽、点序与mask仍需对应部署，不等于有机器人传感器 |
| object linear/angular velocity6 | `task_feedback.py:43–44,68` | 无原生对象velocity列 | **0/1,064**；缺1,064 | pose差分可定义另一因果速度，但不是已记录的原生state13，不能近似填坑 |
| time | `task_feedback.py:40–41`；`task_scheduler.py:126–131` | `time/timestamps/frame_ids`与WAM30Hz网格 | **1,064/1,064** | 观测时钟可得；不等于TaskSpec指令时钟已绑定 |
| source cursor | `task_scheduler.py:71,91–94,184–187` | 无 `source_cursor_frame/plan_cursor_frame` | **0/1,064**已录 | 内部派生状态；完整计划、reset和因果输入齐时可重放，不应拿frame_id替代 |
| contact_forces5×3，N | `task_feedback.py:47–52` | 无机器人 `contact_forces` | **0/1,064**；缺1,064 | human `contact/near/program` 不合格；Shadow原始NPZ也没有该列 |
| force_valid | `task_feedback.py:47,64–70` | 无 `contact_force_valid/force_valid` | **0/1,064**；缺1,064 | 不能因文件缺失就填“已观测且为0力”，也不能声称false等价于原生反馈 |
| supported | `task_feedback.py:69`；`task_scheduler.py:205–206` | 无已测 `supported/support` | **0/1,064** | 生产support不能仅由“物体看起来近桌面”冒充 |
| support原始object force / table pose | `measured_execution.py:48–61` | 无 `object_contact_force/table_pose`原生接口列 | **0/1,064**完整接口 | GRAB的human table记录不是已绑定的机器人力+table pose7+审计bounds组合 |
| workspace_ok | `task_scheduler.py:143–145`；`physical_safety.py:9–24` | 无已测workspace/abort；`q/dq`仅Shadow3条 | **0/1,064**已测；q/dq候选3/1,064 | Shadow q22不是Inspire q18；pose_valid或voxel-valid也不是physical workspace_ok |
| TaskSpec event kind | `task_scheduler.py:82–88,136–141` | 有GT事件sidecar / human program标签；无独立计划绑定 | **0/1,064** | 标签不冒充事前计划事件 |
| TaskSpec event source_frame | 同上；`task_spec.py:29–35,67–69` | 无计划事件frame绑定 | **0/1,064** | 观测frame_id不是planned onset |
| TaskSpec source_fps | `task_scheduler.py:62,91–94` | 有观测timebase、若干source_clock_profile/framerate | **0/1,064** TaskSpec绑定 | 观测fps可追溯，不等于有独立指令时间线；不能说所有时钟都缺 |
| TaskSpec object_path | `task_scheduler.py:62–64` | 观测pose全有；Shadow有goal位置3条 | **0/1,064**完整计划绑定 | goal位置3条是正证据，但只有position target，不包含完整事件/终态声明 |
| TaskSpec source_frames | `task_scheduler.py:61–62,187` | 有观测frame_ids；无独立路径frame映射 | **0/1,064**绑定 | 不把整条未来观测轨迹转成指令 |
| TaskSpec terminal_mode | `task_spec.py:52–56,71–72` | Shadow的success/reset/terminal_time不等于HOLD/RELEASE声明 | **0/1,064**绑定 | 不用示教终点推断声明；UNKNOWN也不能悄悄换HOLD |
| TaskSpec source_role | `task_spec.py:48,65` | 数据source/profile可得；无TaskSpec来源角色绑定 | **0/1,064**绑定 | human/Shadow profile不是`explicit_task_instruction`的证明 |
| reset / observer / scheduler snapshot | `task_runtime.py:11–20`；`task_feedback.py:34–37` | 无已绑定snapshot；只有数据序列边界 | **0/1,064**已录 | 可通过原生reset+完整输入重建；不允许从每个窗口自设reset |

TaskSpec还需要source_length、goal_type、require_support及annotation状态；现有观测
长度/对象类型可提供候选元数据，但不足以在不改语义前提下完成部署声明。

两处容易误认的字段已单独核查：

- Shadow3条的原始`object`均为`(100,7)`，`goal=(100,3)`、`q/dq=(100,22)`；有目标位置，
  **没有object13里的native velocity，也没有接触力/force-valid**。`dq`是关节速度。
- processed的`program`只是布尔mask，不是TaskSpec。GRAB/ARCTIC metadata声明
  `program_available=False`；对应producer `prepare_native_full.py:232–234` 写的是
  `program=np.zeros_like(near)`。不能因文件名“program”认定它存了任务计划。

## 2. 30母序列重导原型：输入门阻塞，未伪造输出

[审计工具](../../tools/audit/audit_event_raw_signals.py)包含原型
`replay_complete_record`：只对完整、显式before-reset TaskSpec与原生观测记录调用
**部署同一个**TaskScheduler和FeedbackObserver；每次仅索引当前frame，输入无效则
报错，不跨缺口；不补force、不差分velocity、不拿GT造TaskSpec。

按sequence_id稳定选择30母序列：每个少数source现有3条全部纳入，GRAB6条，
共 **409个curated窗口**，不是挑只看起来能成功的片段。预检结果：

| 项目 | 实测 |
|---|---:|
| 完整输入准入母序列 | **0/30** |
| 准入窗口 | **0/409（0%）** |
| 实际状态机调用 / 真实事件导出 | **0 / 0** |
| type一致率 | null |
| delta30差中位数 / 分位数 | null / null |
| valid不一致率 | null |
| 母序列聚类95% CI | null（无可比较母序列，不能bootstrap出证据） |

这是brief第4条“关键信号缺失则直接报告”的停止分支；不声称完成了30条合法重放，
也不把预检拒绝率解释成状态机拿不到等价语义的实验结果。GT标签保持不变，无逐tick
合法输出就不执行无意义的GT差异统计。

8项新测试通过：NPY头读取不碰payload/pickle、坏头拒绝、human标签不冒充机器人
信号、dq不冒充object velocity、缺输入先于状态机构造报错、独立完整合成输入用同一
状态机因果重放且future-read guard通过、无效前缀拒绝，以及force-free release反证。
合成测试仅验证工具接口，**不计入30母序列原型或6,000窗口legal准入**。

force-free分支为何不能冒充保真重导：`task_feedback.py:57` 允许无力时依靠运动
coupling判断established，但`:64–67` 只有force_valid时才累积release；
`measured_execution.py:50–59` 的support也显式要求对象接触力与table pose。
测试中同样的远离观测，有已声明有效的零力测量时可确认release，无力模式始终不确认。
因此“API允许省略force”与“能复现当前完整事件链”是两回事。

## 3. 成本：当前可测部分与新录信封

[成本JSON](brief32-cost-envelope-20261023.json)把实测与假设分开：

- 全量4,232层头/metadata扫描实测 **142.075s墙钟**，峰值观察内存约210MiB，无GPU。
  这是I/O和库存成本，不是事件重放CPU时间。
- 原型真实状态机调用0次，因此**完整6k重导CPU小时与合法导出字节均未测、记null**。
  不能把约15μs的拒绝检查时间线性外推，伪称整套重导接近免费。
- 若以后补齐输入，状态机离线重导不需要GPU。逐tick重放实测为`c`毫秒后，按既有
  261,720对象tick的外推公式为 `261720*c/3,600,000` CPU小时；本轮没有给c编造数值。

保留完整事件语义的新录规模，沿用brief31的等规模信封，不重新推导旧结论：

| 项目 | 规模 / 成本量级 |
|---|---|
| 需要绑定的母序列 / 对象流 | **1,064 / 1,067**，约10³次母序列级录制 |
| 母序列tick / 对象tick | **260,526 / 261,720**，约10⁵条原生状态记录 |
| 30Hz累计母流 / 对象流时长 | **2.412 / 2.423小时**，不是运行墙钟保证 |
| 原始最小事件链状态按0.5–2KiB/对象tick | **134,000,640–536,002,560 bytes**（约128–511MiB），另加指令、snapshot、geometry元数据 |
| 最终事件四字段紧凑7bytes/tick | **1,832,040 bytes**；按16bytes对齐则4,187,520bytes，不含文件头/指令绑定 |
| 条件墙钟预算示例 | 若实时倍率10、reset2s，则串行约**0.83h**；倍率1、reset10s则约**5.37h** |

最后一行是显式假设的成本情景，不是已测吞吐或GPU小时承诺；未含失败重试、
任务/资产转换、排队、控制器资格验证，也不保证robot能按human时钟完成或产出6k准入
窗口。新录仿真可能需要GPU，但本轮没有启动它。现有仅做CPU头扫描/预检。

新录必须在reset前冻结TaskSpec，记录实际robot pose/hand/velocity、接触力及其
validity、support/workspace所需原始状态与边界，以及确切时钟/reset/状态机版本。
新条件只属于对应新执行；**不能把新机器人rollout的事件贴回旧human窗口，声称修好了
旧执行的条件链**。旧示教若被用作新执行的事前任务指令，需要如实绑定该新来源。

## 4. 给机主的低成本替代：统一unknown-event mask（未实施）

若不购买上述新录成本，可以把“学习event-conditioned WAM”改成明确的unknown-event
训练/推理合同。保持模型架构，条件张量定义为：

| 张量 / 行为 | 提议变化 |
|---|---|
| `event_type` | 全部int64零占位 |
| `event_delta_steps` | 全部float32零占位 |
| `event_valid` | 全部bool False，表示unknown；**不是known NONE** |
| `event_hz` | 保持30 |
| 模型event features | 全部`[0,0,0,0,0]`，EventGoalResidual被valid mask屏蔽 |
| 旧raw-invalid/dropout诊断 | 不伪写成“观测证据变了”；旧标签与来源留作只读诊断 |
| 其余数据/张量 | 原窗口、H4/K24、轨迹/接触targets及masks、goal输入、normalizer、模型结构不因该event方案改变 |
| 部署接口 | 若获批，WAM模型入口也必须统一unknown；TaskScheduler仍可保留自己的计划/反馈控制逻辑，不能删除计划假装同一控制器 |

这是**条件配方变更**，需新recipe/data identity、明确训练与推理adapter协议以及机主
裁决；不是补录成功、不是数据组成/训练预算缩减，也不自动认证其它训练门槛。
source-only A0不消费event packet的既定事实不受此方案影响；512行WAM预测可能受
影响，不能用旧持久化预测量化新mask效果或声称τ质量不变。

**建议裁决路径**：坚持完整计划/力反馈事件，就承认需要新数据录制；优先保留原6k
human数据并控制成本，就评审统一unknown配方。当前证据不支持“只重导就放行训练”。
本轮没有替机主选择、没有修改任何生产语义或config status。

## 复现命令、测试与写盘字节数

环境：`dexplore_repro_py38_torch222_cu121`，Python3.8.20、NumPy1.24.4；CPU only。
完整库存只需首次执行；已有JSON可直接检查，不必重复扫描。工具拒绝覆盖输出。

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/audit_event_raw_signals.py \
  --curated outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json \
  --output src/task/ptv3-shared-wam-v6/docs/audit/brief32-reproduced-signal-inventory-20261023.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src \
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_event_raw_signal_audit.py
```

**测试8 passed，0.23s**；另有库存覆盖、JSON完整性和文档链接检查。写盘前估计<32MiB，
本轮上限2GiB；当时`/home2`可用147,897,876,480 bytes。不删历史outputs，不写新
rollout或checkpoint；
本轮5个新文件（3份研究记录+工具+tests）逻辑大小合计 **7,626,382 bytes**；合法事件轨迹新增 **0 bytes**，checkpoint新增 **0 bytes**。
