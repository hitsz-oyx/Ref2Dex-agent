# pointwam-repro 编码器资产

论文没有点名 Mosaic3D 的 checkpoint。这里准备使用 Mosaic3D 自己用 Recap-CLIP
训练的全数据 PPT 权重：

- 仓库 `junhalee/Mosaic3D`
- 文件 `sc+ar+sc+++ma+st.ckpt`
- 点特征维 768，等于该配置里的 `text_dim`

文本塔使用 `UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP`。送入 Volt 的是宽度 1024 的
token 状态，再由可训练线性层投到 1280。权重文件不进 Git。当前前向实验还没有
加载它们。
