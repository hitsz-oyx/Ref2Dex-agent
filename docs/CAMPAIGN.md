# Current Execution Campaign

本文件定义当前机器、资源与操作边界。

这些限制与科研 Mission 分开。

---

## Workspace

允许修改：

`/home2/wyy/oyx_ws/ai_ws`

当前 Ref2Dex-agent 应在该工作区内独立开发。

---

## External Project

外部项目：

`/home2/wyy/oyx_ws/Ref2Dex`

默认：

READ ONLY

允许：

* 读取代码；
* 读取数据；
* 读取 checkpoint；
* 创建指向其中内容的软链接。

禁止：

* 修改；
* 删除；
* 覆盖；
* 在其中生成 cache 或输出。

---

## System operations

禁止：

* sudo；
* apt/system package 修改；
* 修改系统服务；
* kill 未确认属于当前任务的进程；
* 影响其他用户 GPU 任务。

可以创建本项目自己的 conda environment。

---

## Network proxy

需要代理时：

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

---

## GPU execution policy

默认优先使用 GPU，按当前任务选择设备：

* 神经网络训练、微调、重复模型推理和批量模型评估优先使用 GPU，包括离线
  V/Cm 拟合与评估；支持 GPU 加速的仿真采集也优先使用 GPU。
* 默认先用一张空闲 GPU，按实际吞吐和显存需求选择 batch；不要仅因任务属于
  Probe、数据量较小或离线分析而默认使用 CPU。
* 纯文件处理、标签审计和统计计算可使用 CPU。极小 smoke 若 GPU 启动成本
  明显超过收益，或当前存在明确的设备/实现限制，也可使用 CPU；模型计算
  选择 CPU 时，在实验卡或运行记录中说明具体原因。
* 每次任务进入训练或批量模型计算阶段时重新判断设备；历史实验的 CPU-only
  设置不自动延续为新任务限制。当前用户明确限定 CPU-only 的任务遵守该限定。

本策略不改变下面的 GPU、时间和存储上限；使用 GPU 前仍须检查显存和进程。

---

## GPU budget

硬上限：

最多同时使用 4 张 GPU。

使用 GPU 前检查显存和当前进程。

不得停止、抢占或干扰未知进程。


---

## Storage

当前工作产物总上限：

300 GB。

需要主动清理：

* 无用中间视频；
* 可重新生成的大型 debug artifact；
* 重复 cache；
* 无长期价值的临时 checkpoint。

不得删除仍可能作为研究证据的正式运行产物。

---

## Git

默认本地开发。

分支前缀：

`agent/`

不要自动 push 到远程。

每条主要研究路线应存在明确 Git checkpoint。

