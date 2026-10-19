# WAM v5 — current engineering state (2026-10-19)

用户授权先准备远端数据、同步实现v5；需要GPU时可暂停已确认远端WAM训练并立即恢复。
参考[ref1](user/ref1.md)和[ref1_1](user/ref1_1.md)，先解决数据正确性与局部C读出，
再决定teacher/loss。用户现明确要求停scale、启动四卡v5；完整缓存已完成，四rank启动检查通过。
本轮从头训练，不加入C→PPO、重定向或CEDex抓取数据。

## 当前四卡训练协议

[训练Probe](experiments/probes/P-20261019-wam-v5-mixed-pretrain.md)使用冻结395ab290代码，
1024宽/12层、每卡128/全局512、8 workers/rank、bf16/TF32、两组LR1e-4、500步warmup、
80k更新上限。沿用v4九源概率/运动分层及seed170，DEV固定seed171/256窗口每源，
第20步首次验证、之后每1000步；首步/里程碑/最优/停止末步保存。

四rank真实混合数据8步检查通过，参数描述性签名相同，峰值rank0约19.79GiB；22项测试通过，
含实际optimizer/DEV/非周期STOP保存及CPU逐位resume一致性。CUDA逐位resume仍未验证。
当前已停止scale并保留39000完整checkpoint；旧launcher因own-STOP记FAILED是用户停止语义。
新全量recipe已冻结并准入，TRAIN3218740/DEV545459窗口全部保留；21点归一化重算1152个TRAIN窗口。
`pretrain-nine-v5-20261019-r1`已实际运行，supervisor3028892/torchrun3029034，
启动报告已到280更新，首轮DEV和step1/20 checkpoint落盘；四卡100%、约24–30GiB/卡。
最近稳定窗口约1.15s/update，80k含验证/存盘初估27–30h；仍在500步warmup，无效果结论。

## 当前实现与已确认修复

- 21点τ/K24/30Hz；保留TRAIN-only逐horizon24×3归一化，手/物体分别统计。
- 778手表面输入、2048物体/接触查询、32物体flow点、256+42+32共享tokens，PTv3候选5mm。
- C为邻近置信度+16部位；现已读取同ID的2048个PTv3逐点特征，无KNN或空间搜索。
  新编码`object-centric-surface-contact16-local-id.v5`拒绝旧全局读出配置。
- 远端Trimesh4.12.2胶囊已居中，旧代码重复减半长确有表面偏移。改为bounds居中并验证尺寸。
  匹配s4-00000的100帧：关节点变化0，表面样本平均变化10.43mm，60.15%变化>1mm；
  14.81%的C查询变化>0.1。这是一条工程序列的对照，不是全量或真实接触准确率。
- freeze与reader共享H4有效性交集谓词；不再准入四帧轮换手但没有持续有效关节点的窗口。
- 新cache绑定协议、依赖与上游；Shadow不继承旧缓存，人手显式继承。新数组发布时计算
  payload身份，recipe/reader核对contract和序列metadata；不在每batch重哈希数组。
  不声称检测不可变目录下绕过metadata的手工payload篡改。PARTIAL几何registry拒绝准入。
- DEV增加根点/根点相对、旧11/新增10、指尖、接触阶段、各horizon手/物体误差、C近邻/核心带
  MAE和16部位混淆。首步/里程碑checkpoint保留；hand_in_ptv3=false实际排除骨架与表面。

## 验证与边界

22项v5测试通过；局部Python/结构化数据/文档链接/seed检查通过。全仓verify仍被六张既有
trajectory-policy实验卡缺少v2元数据阻塞，未修改无关历史卡。

历史单卡短测（非当前长训）：修复后九源工程recipe共9334 TRAIN窗口，DEV为空，采用原运动分层；
batch128/4 workers完成8步AdamW，九源均出现，loss/梯度有限。GPU前向+loss312.90ms、
反向748.83ms、optimizer42.21ms，合计1.104s/步；峰值allocated19.16GiB。
训练loop含读取/首次启动共18.64s（约2.33s/步）。不视为完整数据/DDP稳态吞吐或效果证据。
该轮主WAM暂停33.99s并恢复；后按用户切换请求已停止，不再保持scale运行。

## 当前数据与后台进程

远端根目录：`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/`。
数据修复与GPU短测代码快照`code-repair-6e4bf1f5/`；窗口曝光审计使用`code-audit-14cf12bd/`。

| 阶段 | 当前入口 |
| --- | --- |
| 冻结输入身份 | `data/plan/plan.json`：39526序列、3764199个原准入TRAIN/DEV窗口 |
| CAD | `data/manifest.json`：2747物体完成；NPZ在`data/geometry/`，--geometry传`data` |
| 修复手表面 | `data/hands-r2/progress.json`，39526完成；6471人手继承，33055 Shadow重建 |
| 修复接触 | `data/contacts-r2/progress.json`，39526完成；parallel3025527已收尾 |
| 训练准入冻结 | `outputs/finalize-training-20261019-r1/progress.json`，COMPLETED；recipe/统计已准入 |
| 四卡长训 | `outputs/ptv3-shared-wam-v5/pretrain-nine-v5-20261019-r1/process.json`及`progress.json` |
| 全量比例审计 | `outputs/contact-audit-20261019-r2/prevalence-r3/summary.json`，PID3021191 |
| 修复后法线小审计 | `outputs/contact-audit-20261019-r2/normals/summary.json`及六组PNG/NPZ |
| GPU/修复证据 | `outputs/surface-repair-20261019-r2/`，gpu-smoke及matched-shadow-validation.json |

手缓存39526/39526、隔离0；contact39526/39526，16250070有效物体帧，16 workers补全耗时879.79s。
进度以JSON为准；并行补全沿用相同teacher/contract，原serial STOP保留。训练准入仅发布完整数据。
完整产物保留；比例审计5小时/128MiB，空闲NAS<20GiB或STOP停止。
原`data/hands`中的Shadow、`data/contacts`和旧finalizer保留作诊断证据；旧contact已停止在4547条。
初始工程9测试/18.94GiB记录对应旧全局C读出及旧Shadow表面，不能替代当前验证。
新长训启动证据在远端`outputs/pretrain-nine-v5-20261019-r1-startup-report.json`。

## Contact决策

[当前Probe](experiments/probes/P-20261019-v5-contact-definition-audit.md)的修复后法线比较已完成324帧/物体对，
重算baseline与缓存完全一致。候选`d_eff=d+4*max(-dot(h-o,n)-1mm,0)`使各source的1.46–23.49%
查询置信度下降>0.1；这不是可忽略差异，但多数source仅1序列，panel半数contact-enriched，
不能当全量比例或法线向外正确性的证据。已导出图供查看，暂不启用候选teacher。

全量P(c>0/.25/.5)统计持续跟随cache；TRAIN/DEV曝光应用修复H4准入与K24重复次数，
但尚未应用source概率/main3运动分层权重。覆盖未完成前不据局部比例改balanced BCE。
普通masked BCE、c>.25部位CE、3mm/10mm/33mm邻近定义均保持。

连续三角面/解析primitive距离、法线朝向认证、双手分别的C、多类hand voxel分离、v4权重
迁移，以及BC/RL的21点接口留作独立后续决策。四GPU逐位resume等价尚未验证。
