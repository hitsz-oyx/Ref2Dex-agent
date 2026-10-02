# Decision Memo：让候选根据当前手物相对运动改变控制

问题：固定专家凸混合缺乏可靠增益后，当前接触反馈程序是否有新的可执行机会。
HF23 88实际H10窗口中Cm34/340步改变指令，但接触loss相对direct+1.385pp、
支持高度-4.417mm，区间跨0；当前风险约束配方关闭，不调门、不追加训练。

选择保留强rotation-Cup的逐步XYZ/手指反馈和片段开始旋转锚点，只添加两个
固定增益0.5的XYZ反馈：速度项dt*(object_velocity-wrist_translation_velocity)，
位置项(current_object_minus_wrist-initial_object_minus_wrist)。转换为native PD
raw command再裁剪[-1,1]。对象的世界位置与虚拟腕平移的常数原点差在锚点相减
时消失。URDF三个零origin世界轴平移位于旋转之前；native reset根旋转恒等。
不冻结XYZ，不改手指，不将旋转掌心COM当虚拟腕原点。

先工程：排除旧623/590观察轨迹检查独立PD与时间因果，再排除新seed669原生
短执行，保存完整actual pre/post、专家bank、rawforce、mesh几何并独立复算。
工程问题是能否正确实现反馈，不判断候选收益；预算600秒/128MiB/单空闲GPU5，
准备/静态/原生/审计合计，失败保留，不能从工程结果选择gain或科学门。
工程通过后另固定真实随机候选机会Probe与重复base/Cup噪声门；位置候选第一步
与Cup相同不作为排除理由，programme责任不同。实际机会过门才拟合对应Cm，
旧固定权重H10模型不直接用于新程序；之后直接执行/再规划，最后策略训练验收。

速度/位置反馈可能降低抬升或接触：后续机会需保留lift/contact/drop并列条件。
失败停止本实现、返回控制职责，不在固定gain周围扫描。Mission/claim/权限/资源
不变，C3仍OPEN；当前会话直接执行，不使用子代理。
