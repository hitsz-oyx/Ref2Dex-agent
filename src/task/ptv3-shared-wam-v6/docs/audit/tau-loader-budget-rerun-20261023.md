# τ loader修复后同配方预算重跑

## 结果解读表（启动前预注册）

| 观测结果 | 含义 | 决定 |
|---|---|---|
| 到 step2500 三项仍改善且 CI 排除 0（≥2% 相对） | 配方仍**预算受限** | 只将进一步预算提上议程，不自行开长跑 |
| 有改善但边际 < 前段一半 | **边际递减** | 按斜率外推达标所需预算，不直接长跑 |
| 三项都无 CI 排除 0 的改善 | **配方/监督信号受限**（A 合同无 goal/no-event 的预测上限已到） | 停训转条件审计 |
| OOM/非有限 loss/loader/盘<20GiB/本 run 产物>32GiB | 故障门 | 立即停、保留证据、**不自动重启** |

来源：Hermes 2026-10-23 11:0x 指令，承接brief43。Blocker为修复loader；Decision为同冻结面板区分预算/边际/监督信号限制。最便宜方案是纯CPU确定性采样器复现与容量检查，再唯一一次新run训练及1500/2000/2500冻结面板；不做rollout/物理/执行器操作，不扩源、不改模型。

## 启动前固定协议

run_id=`t38-r4`，新输出`tmp/t38-r4`；旧r3/r2/r1均只读、不删除。按原launcher重新weights-only初始化parent step11000，新optimizer、seed170，非恢复旧optimizer。目标2500，沿用旧config的updates2500/max_seconds10000/hard_stop_epoch1792735439（本地14:03:59），配置字节SHA保持8c157ad4c903b0ed720d45057803fbc73212d1de48d0724df3171287b8578b51。4000为可选上限愿望，但本轮优先遵守配置身份不变与原硬停，不自行改上限。

同512锚点/124母序列/4000 cluster bootstrap，anchor SHA=440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706；init-A与step1223只引用不重跑。新run面板1500/2000/2500各一次，Δ=new−1223（负值改善），CI用旧冻结cluster_ci与保存NPZ配对。每个点若三项都无严格改善（Δ<0且CI上界<0）即STOP；否则继续至2500。故障优先，继承brief43重叠行处理：逐项斜率比较，任一项边际<前一段一半或非正时，保守按边际递减优先于累计≥2%行。前段使用旧1000→1223，后段1223→1500→2000→2500；跨run段不能当作同一优化器连续轨迹，CI不包含训练重复方差。若结果混合不满足分支，不强贴标签、不追加预算。

只使用GPU4/5/6。每个面板点在原生checkpoint save的DDP barrier后，三个自有rank同步暂停、empty_cache释放未使用缓存；独立watcher确认三个rank就绪、GPU空闲显存≥8GiB和盘≥20GiB后在GPU4推理，再release三个rank。只改变保存/资源调度，不改变数据、模型、损失、lr、global_batch或RNG。原冻结panel_gpu456.py逐字复用（d10bc84776a852d3ff8308c06976180dbfc23dadc1ed58b7c197c506b7bdde51），不用GPU7，不放宽资源门。若缓存释放后仍无法满足8GiB门，则STOP并报告，不抢占进程。

原launcher保持按recipe SHA全局互斥/短socket根/32GiB看门狗；原始一次启动，不自动重启。保留0/128/1500/2000/2500及末态checkpoint；只允许本run自有中途文件在正常终态收据后清理，故障保留全部。看门狗记录进度/loss/硬停倒计时/磁盘；面板记录脚本秒数、wall秒数、峰值显存与写盘字节。

## 根因、可反驳预测与修复

旧rank2原始日志（r3/train-launch.log:148）为`ValueError('batch 96 exceeds sequence cap for source 8: need=25, capacity=24')`。真实6000清单上，单进程CPU `draw(96,step=1579,rank=2)`复现同字串；4 workers/rank、默认预取使1579的sampler错误在完成1570更新时暴露，但日志未单独记录每个预取槽，差9只作为与预取一致的解释。去掉torch/GPU/worker/IPC仍可复现，因此容量检查是充分原因。

排序假设与反驳条件：①source有放回分配先忽略容量，预测纯CPU相同seed/step/rank必失败；已证实。②unix路径/句柄耗尽，预测依赖IPC或FD压力；单进程纯numpy复现且旧日志无AF_UNIX/too-many-open-files，不支持本次归因。③worker并发/主存/GPU不足，预测worker0不能复现或有OOM证据；worker0可复现、旧reserved22.914GiB/allocated16.466GiB且无OOM，不支持作为本次必要原因。新启动前资源在resource-before.json（不是冒充故障瞬间主存）：GPU4/5/6各4MiB、util0，MemAvailable329108248kB，FD4/soft1024；disk153874509824bytes，pgrep返回1。旧故障瞬间RSS/FD未采样，不虚构这些历史值。

通用修复：保留原source随机抽样；只在某source超过`序列数×max_per_sequence`时截去超额，等概率分给仍有容量的source。所有原本合法批次不额外消耗RNG，原source/sequence/phase/row选择逐项不变；总容量不足仍显式拒绝。保持max_per_sequence=8、所有6000窗口、seed170与均匀source/sequence/phase意图，不按source8、step1579或物体特化。承认修复会改变原本报错的非法分配：它是故障域上的扩展，不冒称这些批次bit-exact，也不声称source分布数学上完全不变。

修前测试4失败/1通过（原故障及一般容量边界），修后8通过；另有一次误写不存在test路径导致0收集，随后正确命令8通过，不把它计作通过。真实清单4008步×3rank共12024个分配发现原故障2处：1579/rank2/source8为25>24，1783/rank2/source4为26>24；共3个名额重分配、总source计数L1差6。修后所有分配满足总96与每序列容量；两处完整draw与确定性重复通过；4个原合法完整draw逐行一致。该CPU证据不是“真实训练稳定越过1570”，后者待新run updates/进程/日志/checkpoint实测填入。

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
