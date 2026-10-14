# PTv3-Shared-WAM v2 当前状态

2026-10-14：本地1000步Probe已完成，局部PROMISING。固定三域dev的hand_all24
从29.72/123.10/63.58降到23.83/87.16/55.14mm；scene在GRAB/ARCTIC改善，
OakInk2略差。best/latest为step1000，不能据此形成抓取或Cm增益结论。
模型98,591,142参数，geometry11,092,000、shared85,054,464。

远端GPU0/2工程预检READY，尚未启动大规模长训。
目录 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`；graspenv为Torch2.0.1/cu118。
复用只读NAS三域输入，统计只迁移data路径，数值/划分/采样权重不变。
`engineering-remote-ddp2-batch128-20261014-r2`在代码7c256e8a上完成4步DDP与保存，
每卡128/global256约0.79秒/update、324样本/s，物理显存0/2约20483/21891MiB。

长训/恢复须使用修复代码d0288294，并在启动shell先执行 `ulimit -n 65536`。
这是进程软上限；hard1048576与系统配置不变。默认1024会在mmap缓存约1018FD时失败。
修复还处理Torch2.0 fused AdamW恢复的CPU step标量；远端最小GPU回归及完整模型
step4→5恢复更新均通过，模型/optimizer/RNG恢复、有限梯度和实际参数更新已核对。
本地Torch2.4 GPU回归与13项v2合同测试通过；原有61项新旧测试记录保持。
未改模型、数据、优化配方或旧Task的5/7微调，不操作远端1/3上的其他任务。

预检报告：`outputs/ptv3-shared-wam-v2/remote-preflight-20261014-r1/summary.json`。
短工程检查不保证长期吞吐/利用率；正式长训仍需固定预算、完整配置与持续资源观察。
详见[实验卡](experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。
