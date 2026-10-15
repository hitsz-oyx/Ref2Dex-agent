# 九源完整序列质量审阅

入口为 `tools/audit/review_nine_sources.py`，参考旧 Task 的
`src/task/cm-pointflow-effect-pretrain/tools/visualize/visualize_multisource.py`。
它从冻结配方读取全量 TRAIN/DEV 序列目录，按需加载完整序列，不局限于一个抽样面板。
TRAIN共4,806,144个准入窗口，DEV共821,635个；包含这些窗口所属的39,528条完整序列。
不读取保留TEST标签。源ID8 DexYCB没有DEV，界面明确显示为空。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v3/tools/audit/review_nine_sources.py \
  --recipe outputs/ptv3-shared-wam-v3/nine-source-recipe-20261015-r1 \
  --output outputs/ptv3-shared-wam-v3/nine-source-review-20261015-r1 \
  --host 127.0.0.1 --port 8769
```

在VS Code的“端口”中转发8769，然后打开 `http://127.0.0.1:8769`。
服务默认存续24小时；在审阅输出目录创建 `STOP` 或终止该viewer自己的PID即可停止。
`server.json` 保存PID和地址，不启动模型或占用计算GPU。

## 使用

- 选择数据源和TRAIN/DEV；按序列名、row、batch搜索。每页200条，可翻页或上一条/下一条，
  所有匹配记录都可选择，没有把总体缩成200条。
- 帧滑条从0到原序列末帧。完整序列播放可看到准入窗口之外的原始帧；界面单独标明
  当前检查物体/anchor是否属于冻结准入索引。
- “当前H4+24窗口”播放以启动时检查帧为中心，播放前3帧至后24帧；未来对照偏移可选0–24。
- 默认世界坐标；“固定检查帧目标坐标”只在切换时确定一次anchor，不跟随播放移动，
  不会把飞离物体重新锁回原点。相机可对准近景、当前+未来、整条轨迹。
- 橙色为当前检查物体，其他物体用其他颜色；右手蓝、左手绿，未来点为黄/紫。
  物体为同一CAD表面采样，手为原生11个点，不补造缺失点或手部网格。
  白线长度固定0.1米，三维坐标不按样本缩放。
- 速度与加速度使用处理后序列的实际时间戳、CAD表面质心，跨缺失/segment不连线。
  通常为30Hz；独立报告中的27.90m/s峰值来自60Hz raw记录，处理后30Hz峰值约21.3m/s。
  HOI4D逐帧标明实测/插值。旧三源native包使用30Hz timestamps；ARCTIC原相机frame_id
  不被错误当作120Hz时钟。
- 标记“可疑/可用/待定”，可选择当前H4+24窗口或整条序列并填写说明。
  每条标记写入独立不可覆盖的 `annotations/<id>.json`，保留源、划分、row、帧区间、
  原始payload、测量与配方；不会修改数据、冻结窗口或训练准入。

## 已知异常的定位

默认打开 `UniDexGrasp++ · 仿真采集`，TRAIN，row14909、frame75，即
`batch028 / episode-00 / env105`。也可点击“打开7.4m异常样本”。
这是本地官方state policy仿真记录，不是UniDex多源人类训练数据集。

物体为约3.45×11.35×3.45cm的罐子。0.8秒内query位移7.387m；60Hz原始记录中
一次16.7ms位移0.465m，差分速度27.90m/s。raw→compact时间/位姿/手逐项一致，
世界坐标标签重建误差2.38e-7m。存在物理质量疑点，尚未定位具体仿真动力学原因。
默认相机包含当前与未来；可切近景，播放到2.6秒附近检查飞离，再看速度/加速度曲线。

## 验证

`--check-only` 无需启动viser即可读取九源各一个完整序列并报告总窗口数。
实际Viser服务控件已遍历九源、两种坐标系、空DEV、异常跳转、窗口播放和独立标记保存。
回归测试覆盖ARCTIC时钟、缺失/跨段不连接、固定anchor不掩盖运动。


2026-10-15 motion清洗：服务切换到`nine-source-motion-recipe-20261015-r1`；仍使用8769。序列列表标明整段/尾段隔离，整段隔离的10条记录也能选择回看；详情显示首次疑点帧和规则。原始完整轨迹仍可播放，是否进入训练看当前anchor的冻结索引准入提示。新标记输出到`nine-source-motion-review-20261015-r1/annotations`，旧标记保留。
