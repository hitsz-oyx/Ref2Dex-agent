可以直接下官方已经发布的 **SPIDER `retarget_full`**。目前完整数据约 **27.8 GB**，包含 **7,876 条成功轨迹**，其中有 Inspire。[Hugging Face](https://huggingface.co/datasets/retarget/retarget_full)

最推荐用他们仓库自带的 downloader，因为会自动做 checksum 校验：

```bash
git clone https://github.com/facebookresearch/spider.git
cd spider

# 如果没有 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync --frozen

# 下载完整 retarget_full
uv run -m examples.lifted_bench.download_release \
    --output-dir example_datasets/retarget_full
```

这是官方 README 给出的完整数据下载方式。下载器会检查压缩包和解压后文件的 checksum。[GitHub](https://github.com/facebookresearch/spider)

下载完可以直接浏览：

```bash
uv run examples/inspect_dataset.py \
    --dataset-dir example_datasets/retarget_full
```

然后在本机打开 `http://localhost:8080`，可以选轨迹、拖动时间轴、播放。这个浏览过程**不需要 GPU，也不需要重新跑物理优化**。[Hugging Face](https://huggingface.co/datasets/retarget/retarget_full)

数据大致会是：

```text
example_datasets/retarget_full/
└── processed/
    ├── dexycb_lifted/
    │   ├── allegro/
    │   ├── inspire/
    │   ├── xhand/
    │   └── sharpa/
    ├── hot3d_v2/
    │   └── inspire/
    ├── hrdexdb_24f/
    │   └── inspire/
    └── oakink_lifted/
        └── inspire/
```

单个任务下面类似：

```text
processed/<dataset>/<robot>/right/<task>/0/
├── trajectory_mjwp.npz
├── trajectory_kinematic.npz
├── config.yaml
└── metrics.json
```

其中我们最应该先看的就是：

```text
trajectory_mjwp.npz
```

这是 physics-retarget 后保存的轨迹。数据里仿真频率是 **100 Hz**，reference 是 **12.5 Hz**；对单物体场景，保存 qpos 的最后 7 维是物体 `XYZ + quaternion(wxyz)`。[Hugging Face](https://huggingface.co/datasets/retarget/retarget_full)

### 如果你只想先试，不想下 28 GB

官方还有一个小的 example dataset：

```bash
sudo apt install git-lfs
git lfs install

git clone \
  https://huggingface.co/datasets/retarget/retarget_example \
  example_datasets
```

适合先确认环境和文件格式。[GitHub](https://github.com/facebookresearch/spider)

### 对我们来说，我建议先别下载全部

我们现在首先需要回答的是：

> **SPIDER 的 Inspire 轨迹能不能在 Ref2Dex 的 Isaac Gym 中做 control replay。**

所以理想情况其实是**只取 Inspire 子集**。官方整包是 27.8 GB，包含 Allegro/XHand/Inspire/Sharpa 四种手，而且四个数据源总计 7,876 条；没必要第一步全用。[Hugging Face](https://huggingface.co/datasets/retarget/retarget_full)

SPIDER 完整数据页面在这里：

[SPIDER retarget_full 数据集](https://huggingface.co/datasets/retarget/retarget_full?utm_source=chatgpt.com)

仓库：

[SPIDER GitHub](https://github.com/facebookresearch/spider?utm_source=chatgpt.com)
