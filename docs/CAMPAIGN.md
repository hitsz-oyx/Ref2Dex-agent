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

## GPU budget

硬上限：

最多同时使用 4 张 GPU。

使用 GPU 前检查显存和当前进程。

不得停止、抢占或干扰未知进程。

---

## Probe budget

默认 Probe：

* 优先单 GPU；
* 优先少环境、小数据和短运行；
* 单个 Probe 默认目标 wall time <= 60 min；
* 能用离线分析回答的问题，不先启动完整训练。

超过该范围仍可执行，但必须有明确理由。

---

## Validation budget

正式 Validation 可以使用：

最多 4 GPU。

如果预计：

* 单一实验 wall time > 6 h；
* 额外生成数据 > 50 GB；
* 需要多组长训练；

则触发 Decision Checkpoint。

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

---

## Deadline

2026.9.25 23:59

---
