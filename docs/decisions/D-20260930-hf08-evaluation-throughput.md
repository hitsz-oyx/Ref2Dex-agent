# HF08 评价吞吐与运行交接

问题：固定48个96环境actor-only评价能否在60min墙钟预算完成。

证据：相同配置、source actor、96环境的工程测量为97.789s，串行预计约78min；主预训练仍正常，未改科研变量。

root选择保留原seed、checkpoint、96episode及48个评价点，使用已授权的两GPU并行。每个training-seed/epoch/eval-seed面板的三臂同GPU，两个eval-seed的GPU分配在training-seed间互换，避免GPU与训练臂混杂。预计评价墙钟约40min，GPU计算时长单独记录；总墙钟6h、20GiB、同时2GPU上限不变。

仅暂停已核对属于本任务的launcher（PID3202150），当前native fit（PID3240830）继续完整运行。待其完成、hash/原生结果复核后，终止已暂停launcher并记录调度交接，继续既定六臂训练，再执行并行评价。失败则保留完整证据并修复工程合同，不缩减矩阵或改gate。

无新的外部授权边界，不改变核心研究问题或claim。运行状态与科学结论分别记录。
