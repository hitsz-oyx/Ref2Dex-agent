# 九源复审：HOI4D插值边界漏筛修复

用户要求再次审计九源。CPU全量40161条，原始/索引证据见
`outputs/ptv3-shared-wam-v3/nine-source-reaudit-20261015-r1/`，独立real-reader验证和root逐窗复算均确认：
旧strict与新expanded各有8个TRAIN窗口的未来23→24帧含超阈值插值半段，5.65–8.11m/s，DEV0。
例旧row36531/query283，native15Hz153→154对应30Hz306→308，窗口终点307仍含306→307异常半段；
原候选context要求原生转换两端均在retained coverage，导致未选择该序列。
37条其他疑点只触碰一个原生端点，未发现坏跳变本身进入标签，应与8条实质遗漏区分。
8/55343个HOI4D TRAIN窗约0.0145%，不能因此解释整个模型效果。

## Decision Note

问题：修复候选预选完整性，并保持已训练数据身份可追溯。证据：raw数值、有效label mask、
reader blocked=false及sampling pool均一致。行动：按full_capture超阈值选候选，之后仍执行
原定inclusive异常转换与H4/K24交叠隔离；这也会更保守移除单端点触碰的少量窗口。
从原manual父recipe重建新strict snapshot，保留所有原始文件和旧配方，不在运行中换输入。

资源：CPU单进程≤10min、派生metadata≤200MB、空闲≥20GiB；没有GPU或远端修改。
停止条件：原索引身份漂移、缺失旧排除集合、source counts减法不符或原manual序列身份变化。
新增保护拒绝已有quality_admission的recipe作为重建输入，避免Catalog已过滤索引与base descriptor
错位或重复扣数；手工row37130操作核验稳定source sequence身份。
正/负：修复通过则派生版本可供后续数据切换；当前run保留其原8窗暴露记录，不能称其已经修复。
无额外外部授权需求；本轮不停止现有训练或改写远端checkpoint。

4项回归通过：原生两端未全覆盖但插值半段监督的候选选择、inclusive边界，以及冻结索引身份
/重复已有quality descriptor拒绝。生成后检查旧排除集合为新集合子集、来源数量逐项相符。
