# τ loader修复后同配方预算重跑

## 结果解读表（启动前预注册）

| 观测结果 | 含义 | 决定 |
|---|---|---|
| 到 step2500 三项仍改善且 CI 排除 0（≥2% 相对） | 配方仍**预算受限** | 只将进一步预算提上议程，不自行开长跑 |
| 有改善但边际 < 前段一半 | **边际递减** | 按斜率外推达标所需预算，不直接长跑 |
| 三项都无 CI 排除 0 的改善 | **配方/监督信号受限**（A 合同无 goal/no-event 的预测上限已到） | 停训转条件审计 |
| OOM/非有限 loss/loader/盘<20GiB/本 run 产物>32GiB | 故障门 | 立即停、保留证据、**不自动重启** |

来源：Hermes 2026-10-23 11:0x 指令，承接brief43。Blocker为修复loader；Decision为同冻结面板区分预算/边际/监督信号限制。最便宜方案是纯CPU确定性采样器复现与容量检查，再唯一一次新run训练及1500/2000/2500冻结面板；不做rollout/物理/执行器操作，不扩源、不改模型。

## 最终实测判读

**工程结论：修复后真实运行到1768（比旧故障1570多198次更新），loader_failure=null；预算目标2500未完成。**

**单一结论：原硬停前未达2500：预算/边际/监督信号三者的完整判定未完成，不追加预算。** 本轮所有结果限同一固定TRAIN面板，未跑rollout，不声称执行或heldout泛化收益。

实际optimizer updates=1768，last_step=1768，状态=STOPPED，last_loss=0.6461899280548096，elapsed_s=8241.210221954621，returncode=0；automatic_restart=false。阶段监控未触发平台/故障；原supervisor停止原因=HARD_STOP_SAVE_MARGIN_30_SECONDS。启动PID=843895，supervisor=843884，启动UTC=2026-10-23T03:45:59.218624+00:00，退出UTC=2026-10-23T06:03:43.191518+00:00。

| 面板step | first8_hand mm | fixed_contact_hand mm | first8_object mm | 状态 |
|---:|---:|---:|---:|---|
| init（仅引用） | 16.1694 | 59.2219 | 11.2523 | 既有 |
| 1223（仅引用） | 10.2884 | 31.7598 | 6.5043 | 既有末态 |
| 1500 | 10.0343 | 30.0755 | 6.3679 | 本run唯一一次 |
| 2000 | — | — | — | 未运行：未达到该点 |
| 2500 | — | — | — | 未运行：未达到该点 |

相对step1223的配对Δ和cluster95%CI（new−1223；负值为改善；124母序列/4000次、同锚点与init-A参考）：

| step | first8_hand Δ [CI95] | contact hand Δ [CI95] | first8_object Δ [CI95] | 相对变化% |
|---:|---|---|---|---|
| 1500 | -0.2542 [-0.3388, -0.1727] | -1.6843 [-2.4681, -0.9238] | -0.1364 [-0.2480, -0.0344] | -2.47 / -5.30 / -2.10 |

每1000 updates边际斜率（mm/1000，正值改善；旧r2前段仅引用，新r4后段同配方seed170）：

| 段 | first8_hand | contact hand | first8_object |
|---|---:|---:|---:|
| 1000→1223 | +1.2675 | +11.1300 | +0.8709 |
| 1223→1500 | +0.9175 | +6.0806 | +0.4925 |

逐项斜率比及触发：`[{"segment": [1223, 1500], "versus": [1000, 1223], "ratios": [0.7238883340594238, 0.5463215594396342, 0.5655072614523317], "less_than_half": [false, false, false]}]`。边际递减优先规则为启动前登记，不因结果更改。

## 启动、身份、资源与保留核对

启动前pgrep原始stdout保存在`tmp/t38-r4/pgrep-before-launch.txt`（空，returncode1）；resource-before.json、explicit-preflight-initial.json、explicit-preflight.json记录资源与身份。资源等待器先排队，三卡空闲后原supervisor只启动一次；实际launch.json、train-launch.log、training/progress.json/result.json及stop-receipt.json提供进程/updates/日志/退出码证据。

训练实际非停止配置逐字段与旧配方相同=True；source config SHA=8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51，condition_recipe SHA=36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d，数据身份/6000/seed170/global288/world3见原登记，未改。原脚本d10bc847逐字复用；本run复用原冻结面板SHA=d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51，计算语句逐字相同。保存钩子SHA=99a9d39776e2845a37119628cddd542b8da7679f4e21fcd5bf49c72fb3f7c51c，阶段watcher SHA=8929b36220b4399b4b6cb700cc8a835d4411689cbfffd33a2beee85a4bcf72f7。

| step | 面板目录 | GPU | 秒数 | peak allocated bytes | 可用盘bytes（后） | 总产物bytes（后） |
|---:|---|---:|---:|---:|---:|---:|
| 1500 | `tmp/t38-r4/panel-step001500-A/` | 4 | 25.537 | 641889792 | 65352990720 | 6519375915 |

本run保留checkpoint：

```json
[
  {
    "file": "step000000.pt",
    "bytes": 405716522
  },
  {
    "file": "step000128.pt",
    "bytes": 1217082017
  },
  {
    "file": "step001500.pt",
    "bytes": 1217082017
  },
  {
    "file": "step001768.pt",
    "bytes": 1217082017
  }
]
```

删除本run中途checkpoint共3个，合计3651245411bytes；仅本run有实际清理时生成pruning.jsonl，所有删除仅限tmp/t38-r4/training，未删除任何历史。普通文件占用快照4085349480bytes；详细完整精度分析final-analysis.json。所有面板argv/资源在各点invocation.json与comparison-vs1223.json，原始log位于本run根目录。

终态git log/HEAD/origin/main三项核对收据在tmp/t38-r4/git-final-verification.json，交付回答也逐项报出；tmp产物不入库。

## 越过1570的实测证明与资源归因

`passed-fault-proof-step1580.json`和`passed-fault-proof-step1700.json`在训练仍RUNNING时捕获三rank进程、updates/有限loss、FD/主存/实际socket与source累计计数。1580 loss=0.5924485084，1700 loss=0.5823579246；逐步重放同seed/step/rank的**分配计数**与实际训练source_draws逐项完全相等，证实真实训练使用了1579处修复分配，不是只跑离线采样器。未声称原故障后的所有远期步都已通过：1783/source4那一处由CPU完整draw回放验证，本轮真实训练只到1768。

1580时rank2/1/0的FD为123/127/231，对应soft上限1024/1024/1048576；VmRSS约3774704/3772800/3896476kB，MemAvailable319867600kB；实际监听socket相对地址47bytes（<108）。workers仍为4/rank，未加timeout、未减少batch、未换seed、未提高每序列8的上限。终态source/runtime/config-before三个文件SHA完全一致，均8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51；包括停止字段的完整JSON也逐字段相等。

原配方/模型/损失/三个lr/global288/world3/6000窗口/manifest/ordered tuple/seed170均未改。sampler只扩展了原本报错的分配域；统计的“3个名额重分配/L1差6”指source计数，不表示只改变3条window draw（非法批次后续RNG流也会变）。合法draw的逐行等价不等于CUDA训练权重bit-exact，本轮没有重做旧A合同/模型边界审计。

`terminal-evidence.json`确认三自有rank退出，`pgrep`返回1/stdout为空；GPU4/5/6为6/4/4MiB、util0。`train-launch.log`中FAILED_DATA_LOADER、exceeds sequence cap、AF_UNIX path too long、CUDA out of memory、nonfinite loss、is non-finite匹配均0。末态result.last_loss=0.6461899281来自1768的rank0训练loss；progress最后常规loss=0.5620046076来自1760的日志，不将它当作1768末次loss。

正常退出码0，STOPPED，loader_failure=null，checkpoint_written=true，automatic_restart=false；收据UTC06:03:43.191518（本地14:03:43），早于1792735439（14:03:59）硬停，原因HARD_STOP_SAVE_MARGIN_30_SECONDS。末态`tmp/t38-r4/training/step001768.pt`为1217082017bytes、完整ZIP中央目录2487项/data.pkl331907bytes；保留0/128/1500/1768，按既定策略只清理本r4中途1/20/1000三个文件3651245411bytes，r3/r2/r1和历史outputs未删除。退出收据时（清理前）本run7736585134bytes，清理后终态普通文件快照4085358152bytes，可用盘70896054272bytes；后续少量JSON/日志使快照字节变化，均远低于32GiB。completion-result.json的training_started=false仅指离线报告生成器不启动训练，训练状态以native result/receipt为准。

## 为什么没达到2500：原预算与实测速率

GPU4此前被外部物理任务占用，原训练实际在11:45:59启动；另有GPU5间歇占用，均未干预。旧run steady=0.32381168 updates/s；本轮最终steady=0.21329448，elapsed=8241.210s，forward_loss/backward/optimizer平均1643.461/2983.177/16.840ms。运行中`performance-after-start.txt`驱动实测GPU5 `SW Thermal Slowdown: Active`、约92°C；1580现场GPU5 SM=555MHz。热降频与降速同时出现是资源证据，本轮未做控制实验分摊全部性能差，未修改风扇/功率/设备或配方。共享盘也曾快速下降，最低可见快照仍大于20GiB，未触发磁盘故障门。

三个rank在1500保存后暂停并empty_cache：GPU4/5/6空闲14596/15352/14264MiB，均满足8GiB门。原面板在GPU4执行；脚本25.536626s，整个子进程38.112360s，peak allocated641889792bytes；加配对CI计算/调度后整次暂停约60s。未使用GPU7。面板命令/资源原始值在invocation.json，metrics.json与comparison-vs1223.json各保留全精度。

本轮严格保留config SHA和原硬停；运行中已报告降速与完训ETA，并请求是否允许只改停止条件，未收到明确放宽，因此没有修改配置或继续训练。当前只到1768，2000/2500/4000都未达到，未补造或重跑旧面板，也未另评1768。1500处三项下降≥2%且CI排除0，斜率比0.7239/0.5463/0.5655均未小于一半，平台门/明显衰减门均未触发；但原第一行要求**到2500**，不能用1500替代。因此单一判读是“完整预算判定未完成”，不能写成配方受限、路线被否证，也不能据此批准11k。

1500在第一次容量修复分配1579之前，不能把其EPE改善归因于loader修复；它只是同冻结TRAIN面板的新曲线点。工程修复效力由真实1700运行/抽样计数与1768持久checkpoint证明。预测执行收益仍未测；没有rollout/物理/执行器修改。判据/边界来自Hermes；本轮无新增外部审计裁决；仍缺的证据为2500评测点。

## 面板与训练复现命令（本轮实际执行）

```sh
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r2/panel_gpu456.py --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r4/training/step001500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r4/panel-step001500-A --reference /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/panel-init-A --gpu 4 --batch-size 4
```

指标路径：`tmp/t38-r4/panel-step001500-A/metrics.json`；配对比较：同目录`comparison-vs1223.json`。原冻结脚本d10bc847原SHA不变。真实训练CLI（展开原launcher的GPU/运行时环境，仅用于取证；执行仍须通过原launcher的全局互斥）为：

```sh
CUDA_VISIBLE_DEVICES=4,5,6 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD/tmp/t38-r4/bootstrap${PYTHONPATH:+:$PYTHONPATH}" REF2DEX_B43_RUN_ROOT="$PWD/tmp/t38-r4" REF2DEX_RUN_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/b38-843884 TMPDIR=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/b38-843884/tmp /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py --config /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/configs/tau_no_goal_no_event_train_v1.json --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/t38-r4/training --initialize-condition-from outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt --workers 4 --git-commit 37958ad3ff7467ec9ae242fe72c85546c4c73f77 --preserve-all-checkpoints --checkpoint-budget-gib 32
```

run-local保存/同步钩子由上文launcher环境注入，见bootstrap/sitecustomize.py与identity.json；面板、阶段watcher、独立STOP看门狗的原argv与日志均在本run中。旧init-A/1223严格只引用。

复现CPU故障/修复用前文工具命令，回归结果8 passed；若重跑测试请显式使用新的项目内`--basetemp`目录。上述训练/面板是取证用**实际历史命令**，原run不允许覆盖、ONE_SHOT不允许复用，原hard_stop_epoch现已过去。本轮未创建新的训练或改停止预算；后续要补足缺少的732更新，需要另行明确停止条件/新SHA授权并保留1768的optimizer/RNG状态，不能原命令无声重启。按本轮steady粗估732更新约3432s（57.2min）再加评测/恢复开销，只是成本估计，不是新预算分配。

## 启动前固定协议

run_id=`t38-r4`，新输出`tmp/t38-r4`；旧r3/r2/r1均只读、不删除。按原launcher重新weights-only初始化parent step11000，新optimizer、seed170，非恢复旧optimizer。目标2500，沿用旧config的updates2500/max_seconds10000/hard_stop_epoch1792735439（本地14:03:59），配置字节SHA保持8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51。4000为可选上限愿望，但本轮优先遵守配置身份不变与原硬停，不自行改上限。

同512锚点/124母序列/4000 cluster bootstrap，anchor SHA=440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706；init-A与step1223只引用不重跑。新run面板1500/2000/2500各一次，Δ=new−1223（负值改善），CI用旧冻结cluster_ci与保存NPZ配对。每个点若三项都无严格改善（Δ<0且CI上界<0）即STOP；否则继续至2500。故障优先，继承brief43重叠行处理：逐项斜率比较，任一项边际<前一段一半或非正时，保守按边际递减优先于累计≥2%行。前段使用旧1000→1223，后段1223→1500→2000→2500；跨run段不能当作同一优化器连续轨迹，CI不包含训练重复方差。若结果混合不满足分支，不强贴标签、不追加预算。

只使用GPU4/5/6。每个面板点在原生checkpoint save的DDP barrier后，三个自有rank同步暂停、empty_cache释放未使用缓存；独立watcher确认三个rank就绪、GPU空闲显存≥8GiB和盘≥20GiB后在GPU4推理，再release三个rank。只改变保存/资源调度，不改变数据、模型、损失、lr、global_batch或RNG。原冻结panel_gpu456.py逐字复用（d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51），不用GPU7，不放宽资源门。若缓存释放后仍无法满足8GiB门，则STOP并报告，不抢占进程。

原launcher保持按recipe SHA全局互斥/短socket根/32GiB看门狗；原始一次启动，不自动重启。保留0/128/1500/2000/2500及末态checkpoint；只允许本run自有中途文件在正常终态收据后清理，故障保留全部。看门狗记录进度/loss/硬停倒计时/磁盘；面板记录脚本秒数、wall秒数、峰值显存与写盘字节。

## 根因、可反驳预测与修复

旧rank2原始日志（r3/train-launch.log:148）为`ValueError('batch 96 exceeds sequence cap for source 8: need=25, capacity=24')`。真实6000清单上，单进程CPU `draw(96,step=1579,rank=2)`复现同字串；4 workers/rank、默认预取使1579的sampler错误在完成1570更新时暴露，但日志未单独记录每个预取槽，差9只作为与预取一致的解释。去掉torch/GPU/worker/IPC仍可复现，因此容量检查是充分原因。

排序假设与反驳条件：①source有放回分配先忽略容量，预测纯CPU相同seed/step/rank必失败；已证实。②unix路径/句柄耗尽，预测依赖IPC或FD压力；单进程纯numpy复现且旧日志无AF_UNIX/too-many-open-files，不支持本次归因。③worker并发/主存/GPU不足，预测worker0不能复现或有OOM证据；worker0可复现、旧reserved22.914GiB/allocated16.466GiB且无OOM，不支持作为本次必要原因。新启动前资源在resource-before.json（不是冒充故障瞬间主存）：GPU4/5/6各4MiB、util0，MemAvailable329108248kB，FD4/soft1024；disk153874509824bytes，pgrep返回1。旧故障瞬间RSS/FD未采样，不虚构这些历史值。

通用修复：保留原source随机抽样；只在某source超过`序列数×max_per_sequence`时截去超额，等概率分给仍有容量的source。所有原本合法批次不额外消耗RNG，原source/sequence/phase/row选择逐项不变；总容量不足仍显式拒绝。保持max_per_sequence=8、所有6000窗口、seed170与均匀source/sequence/phase意图，不按source8、step1579或物体特化。承认修复会改变原本报错的非法分配：它是故障域上的扩展，不冒称这些批次bit-exact，也不声称source分布数学上完全不变。

修前测试4失败/1通过（原故障及一般容量边界），修后8通过；另有一次误写不存在test路径导致0收集，随后正确命令8通过，不把它计作通过。真实清单4008步×3rank共12024个分配发现原故障2处：1579/rank2/source8为25>24，1783/rank2/source4为26>24；共3个名额重分配、总source计数L1差6。修后所有分配满足总96与每序列容量；两处完整draw与确定性重复通过；4个原合法完整draw逐行一致。该CPU证据不替代真实训练；本轮随后已用live1700进程/更新/抽样计数和末态1768 checkpoint完成越过1570的验证，见前部终态证据。

复现工具：`tools/audit/reproduce_tau_sampler_capacity.py`；原source副本、修前/后测试、sampler-proof.json在新run根目录。仅审查采样容量，没有重复A合同或6000窗口身份取证。


## 逐行diff与身份登记

```diff
--- curated_sampling.py before
+++ curated_sampling.py fixed
@@ -66,6 +66,28 @@
             by_source[source][sequence][phase].append(row)
         self.by_source = by_source
         self.sources = tuple(sorted(by_source))
+        self.source_capacities = {
+            source: len(by_source[source]) * self.max_per_sequence
+            for source in self.sources
+        }
+
+    def _draw_source_counts(self, count: int, rng) -> dict[int, int]:
+        if count > sum(self.source_capacities.values()):
+            raise ValueError(f"batch {count} exceeds sequence cap across all sources")
+        chosen = rng.choice(self.sources, size=count, replace=True)
+        per_source = {int(s): int((chosen == s).sum()) for s in self.sources}
+        # Preserve every previously feasible draw and its RNG stream exactly.
+        # Only impossible overflow is reassigned, uniformly among sources with
+        # remaining sequence capacity. No source/seed/step-specific exception.
+        excess = 0
+        for source, capacity in self.source_capacities.items():
+            excess += max(0, per_source[source] - capacity)
+            per_source[source] = min(per_source[source], capacity)
+        for _ in range(excess):
+            available = [s for s in self.sources
+                         if per_source[s] < self.source_capacities[s]]
+            per_source[int(rng.choice(available))] += 1
+        return per_source

     def draw(self, count: int, *, step: int = 0, rank: int = 0) -> list[dict]:
         """Draw one deterministic batch and enforce the sequence cap."""
@@ -73,16 +95,11 @@
             raise ValueError("count must be positive")
         rng = np.random.default_rng(np.random.SeedSequence(
             [self.seed, int(step), int(rank), 20261023]))
-        chosen = rng.choice(self.sources, size=count, replace=True)
-        per_source = {int(s): int((chosen == s).sum()) for s in self.sources}
+        per_source = self._draw_source_counts(count, rng)
         out: list[dict] = []
         for source in self.sources:
             need = per_source[source]
             sequences = tuple(sorted(self.by_source[source]))
-            if need > len(sequences) * self.max_per_sequence:
-                raise ValueError(
-                    f"batch {count} exceeds sequence cap for source {source}: "
-                    f"need={need}, capacity={len(sequences) * self.max_per_sequence}")
             counts = defaultdict(int)
             for _ in range(need):
                 available = tuple(s for s in sequences
```

```json
{
  "config_before_sha256": "8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51",
  "config_after_sha256": "8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51",
  "stopping_fields": {},
  "all_other_config_fields_identical": true,
  "condition_recipe_sha256": "36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d",
  "data_identity": {
    "manifest_sha256": "5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f",
    "ordered_window_tuple_sha256": "d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7",
    "windows": 6000
  },
  "seed": 170,
  "world_size": 3,
  "global_batch": 288,
  "panel_original_sha256": "d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51",
  "panel_gpu47_sha256": "d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51",
  "code_sha256": {
    "src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py": "89688641486e6937c7db101b7b0c83ac8afb7c1bf0100d49d8a1e83e366f08f9",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/model.py": "a9f415b871678d62f852d8baa14dbe9647ba8a97a1fd88a59dbfe57ae1ef37a9",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/data.py": "3726d214e34cd3769a49f8a5e24242f82f919cdb61fa448584fb165d16c60535",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/goal_policy.py": "a1b452f419190549b293a190de6f207ca8e6b211aaa2a497389359b6efb5d9fc",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/events.py": "b5b70b628c0e572b5f2b80db10798ddb2ac34bccfa8962b5b81fcbd17d7aabcc",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/flow_matching.py": "ca558c9d4adec6f3b67a6b0f407f6c0ae1fc830772b6830daacd87d8e9479b53",
    "src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/curated_sampling.py": "23df387c799220cf9f04a4ceeb688fd0d989a8c6304670b4d84d21368158c4ec"
  },
  "checkpoint_hook_sha256": "99a9d39776e2845a37119628cddd542b8da7679f4e21fcd5bf49c72fb3f7c51c",
  "stage_watch_sha256": "8929b36220b4399b4b6cb700cc8a835d4411689cbfffd33a2beee85a4bcf72f7"
}
```

实际启动将执行`/usr/bin/python3 tmp/t38-r4/start_once.py`，其中唯一训练命令是原launcher，环境只加入run-local保存/同步钩子：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tmp/t38-r4/bootstrap${PYTHONPATH:+:$PYTHONPATH}" REF2DEX_B43_RUN_ROOT="$PWD/tmp/t38-r4" /usr/bin/python3 src/task/ptv3-shared-wam-v6/tools/run/launch_tau_no_goal_no_event.py --checks src/task/ptv3-shared-wam-v6/docs/audit/brief37-no-goal-no-event-20261023-r1/checks.json --output tmp/t38-r4 --queue
```

修复验证命令（无训练/GPU）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/task/ptv3-shared-wam-v6/src /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider src/task/ptv3-shared-wam-v6/tests/test_curated_sampling.py src/task/ptv3-shared-wam-v6/tests/test_curated_sampling_capacity.py
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python src/task/ptv3-shared-wam-v6/tools/audit/reproduce_tau_sampler_capacity.py --manifest outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json --before-source tmp/t38-r4/sampler-before.py --horizon 4008 --output <新的JSON路径>
```

原source由本轮编辑前保存，SHA0c2ebed9034a57b8b0d488f3b3413ed63a7a4f2de25f92ea08ac8f21b7ab34d5，可用`git show caa14194:src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/curated_sampling.py`核对。保存/同步钩子的CPU mock测试通过（checkpoint-hook-test.json），不是训练启动证据。全部tmp产物不入库。

## Git记录

启动前`git pull --rebase origin main`被未提交工作树拒绝（exit128）；保留其他会话修改、不stash、不强推，显式路径提交本轮sampler修复/回归测试/复现工具/预注册报告后普通push。终态再核对HEAD/origin/main/ls-remote。

## 启动前资源变化

最终预检UTC03:23:24（本地11:23）发现GPU4被另一会话PID805270的run_object_tracking占用1969MiB/util35，GPU5/6空闲；无train_pretrain进程。explicit-preflight仅GPUs_free失败，尚未启动训练。向机主报告协调空卡，未终止/干预该PID。资源准入监控只排队、无训练，三卡全空闲且pgrep为空才调用原launcher一次；最晚准入为硬停前8400秒（本地11:43:59），避免不足完整2500步时间窗时盲目启动。若未获三卡则NOT_STARTED收据，无自动重启；原config及其SHA仍不变。真实启动以launch.json为准。

## 准入重新核实（无训练重启）

本地11:44:44复查GPU4原占用进程已退出，三卡全空闲。原8400秒准入余量是本轮自行设置的保守工程估计，并非机主硬停；剩余约8315秒仍大于旧实测2500/.3238≈7721秒，加模型载入与三个暂停面板后约7900秒。因而将准入余量收紧为8100秒（不改config、不改14:03:59硬停）。此前NOT_STARTED资源记录与自动生成的零更新报告完整归档到tmp/t38-r4/resource-admission-before-launch/，不覆盖证据；本轮从未启动过训练，不是训练失败后自动重启。科学四行判据保持原文，继续唯一一次原launcher训练。

## 真实唯一启动证据（阶段结果待实测）

本地11:45:59（UTC03:45:59.218624）原launcher启动torchrun PID843895、supervisor843884，物理GPU4/5/6；训练源码提交37958ad3ff7467ec9ae242fe72c85546c4c73f77。training-start-proof.json保存原pgrep进程、launch argv、三真实rank hook、GPU利用率、updates与checkpoint。采样时真实step20、loss1.473211126有限、world3/global288/seed170不变，GPU4/5/6为22563MiB/99%、20297MiB/100%、24211MiB/98%。已写更新后checkpoint，日志socket路径错误0。不能把这些早期更新或CPU replay当作已稳定越过1570；后续必须给>1570持续更新和日志/进程/持久checkpoint证据。

此前资源监控并未调用任何训练，归档的0更新状态只是训练前准入记录。原ONE_SHOT/global recipe lock在这次唯一原launcher启动后生效；本轮没有自动重启任何训练。

## 终态提交与复现入口

终态`git pull --rebase origin main`再次返回128（unstaged changes）；未stash、未强推，未碰他人脏文件。只显式提交本报告，普通push；实际commit/push/HEAD/origin/main/ls-remote结果写入`tmp/t38-r4/git-final-verification.json`，交付回答同步列出。运行产物与checkpoint均不入库。

唯一启动入口为本轮已执行的`/usr/bin/python3 tmp/t38-r4/start_once.py`，该原目录现因ONE_SHOT显式拒绝再次启动；完整launcher环境/参数与原生展开命令见上文。CPU故障复现与修复检查可在新JSON输出路径执行：

```sh
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python src/task/ptv3-shared-wam-v6/tools/audit/reproduce_tau_sampler_capacity.py --manifest outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json --before-source tmp/t38-r4/sampler-before.py --horizon 4008 --output tmp/t38-r4/repro-check-NEW.json
```

该命令只检查采样器，不启动训练、不生成模型读数；若NEW路径已存在须换新路径，不覆盖产物。本轮完成loader工程修复及真实越点验证，未完成2500预算判定。
