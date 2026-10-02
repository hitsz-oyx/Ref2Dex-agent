# Engineering correction：初始化 native player 批量维度

排除seed669首次原生运行在首次专家推理失败，(1,96,1442)被展平为(1,138432)，
尚未产生科学后果。parent/native均exit1，parent墙钟42.010009秒，完整失败目录
`P-20261002-contact-relative-feedback-native-engineering-r1`保留、不覆盖。

rl_games真实PpoPlayerContinuous.get_action最小接口重放RED捕获(1,96,1442)，
只调用BasePlayer.get_batch_size后GREEN收到(96,1442)；3.577秒接口诊断用CPU，
仅零tensor与shape断言、没有真实模型训练/重复推理，GPU启动不划算。按5秒计费。
遗漏的player初始化是工程bug，已有所有原生collector均先调用该方法。

只补批量初始化，gain/候选/seed/RNG/接触触发/预算不变。唯一新r2目录重跑，
明确核对r1所有PID已terminal，失败时间和磁盘并入600秒/128MiB，不重置预算。
仍为排除科学/训练的工程，完成后才固定独立候选机会Probe。C3保持OPEN。
