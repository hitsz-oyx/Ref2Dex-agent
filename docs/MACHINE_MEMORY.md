# 机器与运行环境备忘

本文档记录 Ref2Dex 在本地服务器和远程服务器上的机器信息、路径、Conda 环境以及
运行前检查方法。机器状态会变化；需要以命令的即时输出为准。本文档最后核验于
2026-10-18。

## 使用原则

- 训练、批量推理和仿真采集默认优先使用 GPU；启动前必须检查 GPU 进程、显存和预计
  完成时间。
- 不要杀死无法确认属于当前任务的进程，也不要覆盖已有 checkpoint。
- 运行产物、缓存和日志放在项目规定的 `outputs/<TaskName>/<run_id>/` 或实验卡声明的
  外部目录，不要写入 `/tmp`。
- 外部项目 `/home2/wyy/oyx_ws/Ref2Dex` 默认只读；只能读取或建立软链接。
- 研究运行前记录机器、GPU、Conda 环境、Git commit 和实际输出目录；远程 checkout
  可能不是本地当前 commit，不能只根据远程目录判断代码版本。

## 本地服务器：`server`

### 硬件与资源

- 主机名：`server`。
- GPU：8 张 NVIDIA GeForce RTX 3090，每张 24,576 MiB。
- 当前核验时 GPU 0–7 均基本空闲；这只是一次快照，使用前仍需重新运行
  `nvidia-smi`。
- 项目根目录：`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent`。
- `/home2` 当前可用空间约 160 GiB；新增数据前检查空间，低于 20 GiB 时停止新增写入，
  并优先清理可重新生成的缓存、视频和 debug 产物。

### Conda

本机 Conda 安装在 `/home2/wyy/miniconda3`。非交互 shell 默认可能找不到 `conda`，
运行前先执行：

```bash
source /home2/wyy/miniconda3/etc/profile.d/conda.sh
conda env list
conda activate <环境名>
```

已核验的常用环境如下。环境的具体依赖可能继续变化，使用前以 `conda list` 和实际
导入结果为准。

| 环境 | 用途或备注 | Python / PyTorch（本次核验） |
| --- | --- | --- |
| `graspenv` | 抓取、DExplore 相关运行 | Python 3.8.20 / PyTorch 2.4.1+cu121 |
| `fastwam` | WAM 相关运行 | Python 3.10.20 / PyTorch 2.7.1+cu128 |
| `dexplore_repro_py38_torch222_cu121` | DExplore 复现环境 | Python 3.8.20 / PyTorch 2.4.1+cu121 |
| `imitation-juicer`、`pusht`、`robo3d` 等 | 其他任务专用环境 | 使用前单独核验 |

系统 `/usr/bin/python` 是 Python 2.7，不要将其当作项目运行时。建议在每次运行日志中
保存以下信息：

```bash
hostname
git rev-parse --short HEAD
which python
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvidia-smi
```

### 本地 GPU 运行前检查

```bash
nvidia-smi
df -h /home2
```

本地资源策略是至少保留两张空闲卡；根据实际显存和吞吐选择 `CUDA_VISIBLE_DEVICES`，
不要默认占用全部八张卡。若要运行支持 GPU 的训练或批量评估，应使用对应 Conda 环境，
而不是系统 Python。

## 远程服务器：4×RTX 4090 节点

### 连接与硬件

- SSH：用户 `wbcd`，地址 `127.0.0.1`，端口 `22023`，密钥
  `/home2/wyy/.ssh/id_ed25519`。
- 本仓库专用 known-hosts 文件：`tmp/ssh-4-4090/known_hosts`。
- OpenSSH 使用小写 `-p 22023`；`scp` 使用大写 `-P 22023`。
- 远端主机名：`deepcybo-4-4090-48`。
- GPU：4 张 NVIDIA GeForce RTX 4090，每张 49,140 MiB；远端按当前 campaign 可以占用
  四张卡，但仍需检查是否有其他任务和显存占用。
- 当前 WAM 训练代码目录：
  `/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/training-code`。
  不要假设远端存在本地路径 `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent`。
- NAS 当前可用空间约 48 TiB；实验输出和缓存放在远端训练树或其声明的 NAS 输出目录。

推荐的只读检查命令：

```bash
ssh -o UserKnownHostsFile=tmp/ssh-4-4090/known_hosts \
  -i /home2/wyy/.ssh/id_ed25519 -p 22023 wbcd@127.0.0.1 \
  'hostname; nvidia-smi; df -h /mnt/ugreen_nas'
```

### 远端 Conda

远端 Conda 安装在 `/home/wbcd/miniconda3`。非交互 SSH shell 可能不会自动加载 Conda，
因此使用前显式初始化：

```bash
source /home/wbcd/miniconda3/etc/profile.d/conda.sh
conda env list
conda activate graspenv
```

当前远端 WAM/DExplore 运行使用的 Python 是：

```text
/home/wbcd/miniconda3/envs/graspenv/bin/python
Python 3.8.20
PyTorch 2.0.1+cu118，CUDA 11.8
```

远端还存在 `dexplore-4090`、`dexplore-data`、`ref2dex-*` 等多个专用环境。除非实验卡
明确指定，不要因为环境名称相近就替换 `graspenv`；运行前记录 `which python`、
`python --version`、PyTorch/CUDA 版本和 `conda env export --from-history` 的结果。

### Isaac Gym 与 DExplore

`graspenv` 本身不会自动导入 `isaacgym`。远端 Isaac Gym Python 目录为：

```text
/home/wbcd/workspace/dex/third_party/isaacgym/python
```

运行 Isaac Gym 或 DExplore 前，将该目录加入 `PYTHONPATH` 或在程序中加入 `sys.path`，
例如：

```bash
export PYTHONPATH=/home/wbcd/workspace/dex/third_party/isaacgym/python:${PYTHONPATH}
```

远端 DExplore checkout：

```text
/home/wbcd/workspace/dex/retarget/dexplore
```

官方 Inspire checkpoint：

```text
/home/wbcd/workspace/dex/retarget/dexplore/checkpoint/inspire.pth
```

其 SHA256 为
`8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
共享训练树中也有 `third_party/DExplore`；运行器明确要求校验 hash 时，应使用通过校验
的 checkpoint。可用于新鲜 DExplore 检查的动作文件为：

```text
/home/wbcd/workspace/dex/retarget/dexplore/data/s1_airplane_fly_1/interaction_hand_inspire.pt
```

## 远程运行与文件传输

远程运行前先在本地确认任务目录、Git commit 和环境，再在远端检查 GPU。远端 checkout
可能处于旧的 detached commit；应把实际使用的 task-local runner 同步到远端，并记录
本地 Git commit。

示例：

```bash
ssh -o UserKnownHostsFile=tmp/ssh-4-4090/known_hosts \
  -i /home2/wyy/.ssh/id_ed25519 -p 22023 wbcd@127.0.0.1 \
  'source /home/wbcd/miniconda3/etc/profile.d/conda.sh && conda activate graspenv && nvidia-smi'
```

通过转发 SSH 端点传输大 checkpoint 时，单条长命令可能造成截断。每次传输后都检查文件
大小和 SHA256；必要时将文件分成项目目录中的小块，远端重组后再校验。不要把传输临时
文件放入 `/tmp`。

## 代理与故障排查

需要网络代理时使用当前 campaign 规定的代理：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
```

常见问题的检查顺序：先确认 `hostname` 和实际工作目录，再确认 `which python`、Conda
环境和 PyTorch/CUDA；然后检查 `nvidia-smi`、磁盘空间和输出目录。Isaac Gym 导入失败
时优先检查 `PYTHONPATH`，不要直接重装环境或覆盖已有 checkpoint。
