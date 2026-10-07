# PointWorld 训练吞吐优化（2026-10-07）

目标是降低每16个窗口的更新耗时，保留当前三臂训练的数据、监督语义和研究结论边界。
当前运行`pointworld-temporal-wm24-20261007`的源码、配置和进程保持冻结。本轮只在
新模块、独立入口和工程输出中验证性能，不迁移或重启当前生产训练。

## 测量与选择

原配置microbatch2、accumulation8。GPU1/2的12次采样平均利用率约26%/25%，
每卡约2GiB显存；主训练进程约占一个CPU核。一次更新的CUDA trace有41459个
真实kernel，其中37239个小于10微秒，同时有2282次流同步。提前缓存数据后
仍需约1.4s/更新，CPU调度、小算子和同步比磁盘读取更值得优先处理。
trace统计排除了GPU user annotation范围，不能将这些范围重复算成kernel。

采用两步方案：先进行保持整数排序与训练配方的Hilbert融合；较大的实际batch
作为下一轮配方候选。DDP不是当前单卡小算子调度瓶颈的直接解法。

## 已实现的融合入口

`src/oakink_wm/pointworld_performance.py`逐点用一个Triton kernel执行上游Hilbert
编码的相同比特操作。保留z/z-trans/hilbert/hilbert-trans四种序列及其方向，
未删除Hilbert、改变时间key、池化、模型参数、loss或RNG调用。仅替换当前进程
中上游编码函数，vendor文件和正在运行的Python进程不会受影响。

count是运行时参数，避免按每批不同点数重复编译；depth1..16是有限编译特化。
非连续int32/int64输入会先连续化，CPU仍使用上游实现。实际GPU路径依赖当前
graspenv已有的Triton，不安装或改动已有环境。缓存须显式放到项目产物目录。

独立训练入口为`tools/run/train_oakink2_pointworld_fast.py`。它复用原训练循环，
保留microbatch2/accumulation8、原损失的逐pair归一化、采样、shuffle donor、
优化器与调度。checkpoint身份额外记录融合模块及新入口SHA256；每160次批次
搬运核查这些实现文件及原实现文件。原独立evaluator能核查并读取扩展后的身份，
其默认上游编码仍可评价融合checkpoint。

从原训练导入必须显式使用`--resume ... --import-reference-checkpoint`及新输出
目录。只允许原implementation_sources完全匹配；数据、统计、配置、arm和vendor
仍由原训练恢复合同检查。导入不变更模型、optimizer、step或RNG张量。
融合checkpoint后续直接使用`--resume`，严格检查包含融合源码的身份。

以下是后续恢复的入口示例，需先完成旧worker的保存停止、核对GPU归属，并对
三臂同步安排；本轮没有执行这个生产恢复命令，也不延长原绝对deadline：

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR="$PWD/tmp" \
TRITON_CACHE_DIR="$PWD/outputs/cm-pointflow-effect-pretrain/pointworld-fast-new/triton-cache" \
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/cm-pointflow-effect-pretrain/tools/run/train_oakink2_pointworld_fast.py \
  --data outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006 \
  --stats outputs/cm-pointflow-effect-pretrain/pointworld-temporal-wm24-20261007/norm_stats.json \
  --output outputs/cm-pointflow-effect-pretrain/pointworld-fast-new/train-action \
  --arm action --deadline 1791424717.7631629 \
  --resume outputs/cm-pointflow-effect-pretrain/pointworld-temporal-wm24-20261007/train-action/latest.pt \
  --import-reference-checkpoint
```

## 工程结果及边界

产物根目录`outputs/cm-pointflow-effect-pretrain/pointworld-perf-20261007/`。
早期trace短暂共享本任务自己的GPU0/history worker；当时GPU0仅有该已核验进程。
融合测试短暂共享自己的GPU1/action worker，未使用GPU0后来出现的其他会话进程、
GPU3或其他用户的任务。因此耗时是小规模工程估计，不是独占设备基准或生产ETA。
所有输入为train/val，没有打开TEST；没有修改原三臂源码或数据。

整数测试覆盖depth1..16、低depth穷举、随机坐标、边界、重复点、batch编码、
四种排序、singleton、非连续int32和4096点输入；编码和argsort逐位一致。
`fused-tests-entry.log`还验证导入只允许完全匹配原身份，且不复制/变更模型和
optimizer对象；源码漂移或误把融合checkpoint作为原checkpoint导入会被拒绝。

`fused-benchmark-r4/verification.json`记录三臂BF16前向、loss、全部梯度和
RNG的比较。CPU/CUDA RNG严格一致；最大translation元素差5.10e-6m，rotation
元素差6.29e-5，loss差1.67e-6，梯度相对L2差<=3.08e-4。工程界限为translation
5e-5m、rotation元素3e-4、loss1e-5、梯度相对L2<1e-3。
浮点结果不声称bitwise一致。原实现重复执行也有scatter/BF16噪声。
早期r1/r3使用过严的微小浮点差阈值而未通过，失败日志保留；r2补入原实现重复
对照，r4使用已有BF16物理量smoke量级并补梯度相对范数，不将失败日志丢弃。

在相同缓存训练窗口上，原2×8中位数1.406s，融合2×8中位数1.137s，吞吐约
1.24倍；另一短测约1.32倍。保持2×8时，融合改动的收益约24%–32%。
下面是融合版本独立batch sweep，不是配方等价性或预测质量比较：

| 实际microbatch | 累积次数 | 更新耗时中位数 | 本进程PyTorch峰值allocated显存 |
| ---: | ---: | ---: | ---: |
| 2 | 8 | 1.112s | 1328MiB |
| 4 | 4 | 0.654s | 1744MiB |
| 8 | 2 | 0.457s | 2565MiB |
| 16 | 1 | 0.364s | 4053MiB |

每行都处理16个窗口，但较大batch改变原逐pair loss归一化、批次共同体素原点及
随机操作布局。因此8×2/16×1只能作为明确记录的新配方候选，需三臂共同采用并
重新核对loss/sampling与验证，不能静默接续为原2×8完全相同的实验。
显存值不是整卡总使用，也不能保证所有训练窗口都同样低。

`checkpoint-import-verification.json`核验原3-step工程checkpoint导入后的
模型、全部optimizer状态、CPU/CUDA RNG严格一致，保持step3且未额外更新。
`checkpoint-import/`与`native-resume/`保存加载和原入口val-only smoke。
原正式checkpoint与全部失败诊断保留。

当前建议：本轮如切换执行后端，先用已验证的融合入口保留2×8；下一轮再以
8×2或16×1降低重复前后向开销。吞吐优化不证明模型精度或策略收益，也不通过
显存占用或GPU busy百分比代替每秒窗口数和最终预测评价。
