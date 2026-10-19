# 机器与运行环境备忘

本文档记录 Ref2Dex 在本地服务器和远程服务器上的机器信息、路径、Conda 环境以及
运行前检查方法。机器状态会变化；需要以命令的即时输出为准。


## 本地服务器：`server`

### 硬件与资源

- 主机名：`server`。
- GPU：8 张 NVIDIA GeForce RTX 3090，每张 24,576 MiB。

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
| `dexplore_repro_py38_torch222_cu121` | DExplore 复现环境 | Python 3.8.20 / PyTorch 2.4.1+cu121 |

系统 `/usr/bin/python` 是 Python 2.7，不要将其当作项目运行时


## 远程服务器：4×RTX 4090 节点

### 连接与硬件

- SSH：用户 `wbcd`，地址 `127.0.0.1`，端口 `22023`，密钥
  `/home2/wyy/.ssh/id_ed25519`。
- 本仓库专用 known-hosts 文件：`tmp/ssh-4-4090/known_hosts`。
- 远端主机名：`deepcybo-4-4090-48`。
- GPU：4 张 NVIDIA GeForce RTX 4090，每张 49,140 MiB；远端按当前 campaign 可以占用
  四张卡，但仍需检查是否有其他任务和显存占用。
- 当前 WAM 训练代码目录：
  `/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/training-code`。
  不要假设远端存在本地路径 `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent`。
- NAS 当前可用空间约 48 TiB；实验输出和缓存放在远端训练树或其声明的 NAS 输出目录。


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

## 代理与故障排查

需要网络代理时使用当前 campaign 规定的代理：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
```
远端服务器的代理端口不是7897是27897
