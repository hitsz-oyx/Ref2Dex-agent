# PTv3-Shared-WAM v2 当前状态

## 2026-10-15 q 三维合同与远端 WAM 适配

RL660 reference-q raw replay 与 nearest-live branch follow-up 均通过独立
command/PD/clock 审计，但 q_ref 直接执行仍为 UNPROMISING 信号（全程手/物体
误差大、held=0；整体 audit 因 native 短回放门保持 UNCLEAR），因此没有把 q_ref
改成监督标签。新 v2 adapter 位于 `src/task/ptv3-shared-wam-v2`，每个窗口显式
保存 `q_ref_next`、`q_pd`、`q_real/q_real_next`，动作损失只使用 q_pd 的
continuous-lift 编码；本地 checkpoint 的 4-step engineering smoke 已通过，
13 项 v2 合同测试通过。远端 v2 80k checkpoint 尚未能读取：反向 SSH
`127.0.0.1:22023` connection refused，直连 `100.64.0.23:22` timeout；因此
两卡科学微调尚未启动，不将本地 smoke 冒充远端结果。GPU4/6 保留给远端父权重
到达后的两卡启动，GPU5/7 既有 geometry-lift 已自然完成 step19000/19000。
适配器实现 commit 为 `d4b011a4`，新增的
`tools/run/launch_robot_finetune.py` 只监督和终止自己创建的 torchrun 进程组，
每60秒记录 progress/GPU/磁盘，8GB本run产物预算与20GiB可用空间安全线均独立于
已由用户解除的旧400GB总上限。远端状态缓存仍停在 step33520，不能据此伪造
最新 parent；需等 SSH 恢复后按 registry、稳定 hash 和 checkpoint 元数据验收。
04:07 CST 复核显示 `tailscale whois 100.64.0.23` 为 `peer not found`，本机
Tailscale netcheck 亦无可用 UDP/DERP；因此当前阻塞是远端网络/Windows 转发状态，
不是本地微调代码或 GPU 资源问题。

2026-10-14：本地1000步Probe已完成，局部PROMISING。固定三域dev的hand_all24
从29.72/123.10/63.58降到23.83/87.16/55.14mm；scene在GRAB/ARCTIC改善，
OakInk2略差。best/latest为step1000，不能据此形成抓取或Cm增益结论。
模型98,591,142参数，geometry11,092,000、shared85,054,464。

远端GPU0/2已启动80k长训：`pretrain-main3-remote-80k-20261014-r1`，代码1650ffce。
每卡128/global256，从seed136重新初始化，不迁移工程权重或静默扩展旧1000步配置。
最多48h、run8GB，384窗固定dev（每域128）每1000步评估；每2000步恢复点，
保护best/latest与20k里程碑。监督PID1718152、torchrun1718160、worker1718169/1718170。
目录 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`；graspenv为Torch2.0.1/cu118。
复用只读NAS输入与原horizon统计；已通过双卡四步及完整模型step4→5恢复更新。
启动固定step460快照约0.682秒/update、375.5样本/s，纯训练剩余约15.1h；dev/保存另计。
30秒GPU0/2平均利用率96.8%/96.3%，物理显存最大29107/23733MiB，尚无训练后dev。
本地预算watcher3838855在393GB主动停止本run，初始合计344.48GB，共享硬限400GB。

长训/恢复须使用修复代码d0288294，并在启动shell先执行 `ulimit -n 65536`。
这是进程软上限；hard1048576与系统配置不变。默认1024会在mmap缓存约1018FD时失败。
修复还处理Torch2.0 fused AdamW恢复的CPU step标量；远端最小GPU回归及完整模型
step4→5恢复更新均通过，模型/optimizer/RNG恢复、有限梯度和实际参数更新已核对。
本地Torch2.4 GPU回归与13项v2合同测试通过；原有61项新旧测试记录保持。
未改模型、数据、优化配方或旧Task的5/7微调，不操作远端1/3上的其他任务。

预检报告：`outputs/ptv3-shared-wam-v2/remote-preflight-20261014-r1/summary.json`。
短工程检查不保证长期吞吐/利用率；正式长训仍需固定预算、完整配置与持续资源观察。
详见[实验卡](experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。
