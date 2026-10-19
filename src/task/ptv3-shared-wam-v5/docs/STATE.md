# WAM v5 — current engineering state (2026-10-19)

用户授权先准备远端数据、同步实现v5；需要GPU时可暂停已确认远端WAM训练并立即恢复。
参考[ref1](user/ref1.md)和[ref1_1](user/ref1_1.md)，先解决数据正确性与局部C读出，
再决定teacher/loss。用户现明确要求停scale、启动四卡v5；完整缓存已完成，四rank启动检查通过。
原768运行从头训练，现按用户授权修复工程合同后从10049末步续训；不加入C→PPO、重定向或CEDex抓取数据。

## 本次审查（真实checkpoint2000，训练继续）

[batch/训练合同审计](experiments/probes/P-20261019-wam-v5-batch-audit.md)确认动态Hilbert深度
的结构batch依赖、NCCL后fork风险、总目标best单一保留，以及CLI覆盖batch不进入resume合同。
实际入口CPU微型续训复现了config batch1→实际batch2仍通过；不是当前正式global512接错。
远端同Torch2.0.1/4090、固定2000步checkpoint，九源真实目标单独/8/128推理均通过：
batch128相对单样本τ位移宏平均0.118mm、单点最大1.826mm、C最大差0.003906；
重复单样本自身已有0.064mm平均波动，不能全归因于depth。真实depth9→10使首目标
1726/1726个初层voxel的Hilbert patch成员改变；小早期面板不提供全量或后期误差上界。
审计6.8s、allocated3.33GiB，独立GPU1进程限14GiB；主训练没有暂停或热改，随后仍推进到3000步。
DEV20/1000/2000手EPE59.69/47.89/42.87mm，根点相对19.95/19.67/19.27mm；
整体τ在改善，手型相对误差改善尚小。2000步总目标best与宏手EPE best重合，完整审计副本独立保留。
结论：不据此停训。下次启动/恢复前应验证spawn/forkserver、绑定有效local/global batch，
并另保留预定义τ候选；当时尚未实施，现已完成下面的恢复修复与四卡检查。后续有意义checkpoint再复测batch敏感性。

## 当前四卡训练协议

[训练Probe](experiments/probes/P-20261019-wam-v5-mixed-pretrain.md)当前冻结ac951370代码，
768宽/12头/12层、每卡128/全局512、8 workers/rank、bf16/TF32、两组LR1e-4、500步warmup、
80k更新上限。沿用v4九源概率/运动分层及seed170，DEV固定seed171/256窗口每源，
第20步首次验证、之后每1000步；首步/里程碑/最优/停止末步保存。
loader改spawn/120s读取超时；实际local/global batch写入checkpoint并在恢复时核对。
objective-best保持原定义，新增独立trajectory-best：短期h1–4、根点相对、接触阶段三个
source-macro毫米误差等权平均；两者均受轮转保护。只从仍存在的父运行checkpoint初始化τ候选。

33项测试通过，含legacy实际batch来源绑定、CPU完整恢复、读取失败末步保存、τ候选保留，
以及实际CUDA fused AdamW恢复和事件零残差迁移。四GPU逐位重放等价仍未认证。
首次四卡恢复及旧代码回退发现Torch2.0.1 CPU映射后fused step设备错误，均无完成更新；
ac951370只迁移fused/capturable step到参数GPU，保留数值/动量。单参数CUDA下一步与不中断训练误差0。
修复后`recovery-contract-probe-20261019-r2`从10049完成8更新至10057，DEV/保存通过、
四rank参数签名一致、564份fused step恢复，末步optimizer计数10057且4份RNG齐全。
当前已停止scale并保留39000完整checkpoint；旧launcher因own-STOP记FAILED是用户停止语义。
新全量recipe已冻结并准入，TRAIN3218740/DEV545459窗口全部保留；21点归一化重算1152个TRAIN窗口。
原1024运行`pretrain-nine-v5-20261019-r1`已通过own STOP正常结束于869步，
完整`step000869.pt`（1993176330 bytes）、首步/20步及best产物保留。
原`pretrain-nine-v5-width768-20261019-r1`已正常STOP于10049，完整末步与历史选择保留。
当前`pretrain-nine-v5-width768-contracts-20261019-r2`从该末步续训，不消费短测8次更新；
保留四卡/global512、完整recipe/统计、optimizer/RNG、LR和累计7日/80000步上限。
恢复启动报告到10140步、九源均出现、四卡100%，约0.90s/update、23–28GiB/卡；
剩余含验证/存盘初估18–20h。LR1e-4，未重启warmup。
原768历史短窗0.889s/update，较1024约少23%；新运行稳态以progress/日志为准。

## 接触／脱手事件条件

用户确认需要WAM新增下一CONTACT/RELEASE的类型、相对帧数及有效性。已阅读分享会话末尾
并向前核对，接口与边界见[event设计](event-conditioning.md)。仓库77045e4f已有可选
零初始化goal残差分支，不增加共享token/点数/KNN；33测试中的4项验证完整WAM迁移输出
逐位保持、分支有梯度、未知/远期事件和单位合同。当前生产ac951370仍无事件输入。
事件侧表、近邻标签转事件的语义/删失覆盖审计、按名迁移新增optimizer参数及条件续训待完成；
现有ckpt可用于后续扩展，但不能把weights-only扩展称为optimizer完整恢复。

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
| 四卡长训 | `outputs/ptv3-shared-wam-v5/pretrain-nine-v5-width768-contracts-20261019-r2/process.json`及`progress.json` |
| 恢复检查 | `outputs/contract-transition-20261019-r2/startup-report.json`及`recovery-contract-probe-20261019-r2/` |
| 全量比例审计 | `outputs/contact-audit-20261019-r2/prevalence-r3/summary.json`，PID3021191 |
| 修复后法线小审计 | `outputs/contact-audit-20261019-r2/normals/summary.json`及六组PNG/NPZ |
| GPU/修复证据 | `outputs/surface-repair-20261019-r2/`，gpu-smoke及matched-shadow-validation.json |

手缓存39526/39526、隔离0；contact39526/39526，16250070有效物体帧，16 workers补全耗时879.79s。
进度以JSON为准；并行补全沿用相同teacher/contract，原serial STOP保留。训练准入仅发布完整数据。
完整产物保留；比例审计5小时/128MiB，空闲NAS<20GiB或STOP停止。
原`data/hands`中的Shadow、`data/contacts`和旧finalizer保留作诊断证据；旧contact已停止在4547条。
初始工程9测试/18.94GiB记录对应旧全局C读出及旧Shadow表面，不能替代当前验证。
旧1024启动证据在远端`outputs/pretrain-nine-v5-20261019-r1-startup-report.json`；
切换意图在旧运行`user-stop-width768.json`，新运行入口记录在
`outputs/pretrain-nine-v5-width768-20261019-r1-launch.json`。
768实际启动/吞吐证据：`outputs/pretrain-nine-v5-width768-20261019-r1-startup-report.json`。

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
