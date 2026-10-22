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
* 按实际吞吐和显存需求选择 batch
* 每次任务进入训练或批量模型计算阶段时重新判断设备；历史实验的 CPU-only
  设置不自动延续为新任务限制。当前用户明确限定 CPU-only 的任务遵守该限定。

本策略不改变下面的 GPU、时间和存储上限；使用 GPU 前仍须检查显存和进程。

---

## GPU budget

保证本地服务器有空闲的两张卡就行，远端可以全占

目前monica用户的进程，如果利用率和显存允许的情况下，允许共享GPU

---

## Storage

远端服务器上存放在nas，不需要遵守这个上限。
继续监控实际可用空间；新数据任务在所在文件系统可用空间低于20GiB时停止新增写入，
保留已完成产物和可恢复下载。既有训练自己的时间/输出预算不自动改写，这个上限限制是针对本地服务器。

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




## 2026-10-22 owner-authorized WAM flow-matching exception

仅适用于 `P-20261022-wam-v6-flowmatching`：机主经飞书明确要求先试
flow matching，短试验后直接大规模、延长训练，沿用最近 GT2048 object contact
loss。此运行可用物理 GPU4/5/6 三卡，0–3/7 禁止触碰；覆盖旧运行4h上限，
设为新目标累计43200s（12h）或30000 updates，以先到为准，非无限运行。
不读写凭据，不影响其他作业，supervisor及训练以独立session脱离TUI。
单阶段checkpoint预算8GiB，预计整轮新增<24GiB；可用空间低于20GiB保存停止。
短试验包括真实数据梯度、容量、三卡保存/严格恢复及128步后ODE采样检查；
技术检查通过即可按机主指令续训，不把可运行视为泛化或滚动性能提高。
此例外不扩大其他实验预算，不使用其他GPU。

## 2026-10-22 12:29 owner stop supersedes the flow-matching continuation

机主明确停止flowmatching-overnight-20261022-r2；SIGINT后目标进程已退、GPU4/5/6释放。
上方12h/30000更新的续训例外不再授权自动恢复/重启。P1仍HOLD；本轮只修确认缺陷与CPU
审计，不将释放的GPU4–7改派新方法实验。后续以机主新指令为准。
