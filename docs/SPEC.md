# ResponseLab v0.1 实现规格

> 状态：当前代码的真实行为，更新于 2026-07-22。

## 1. 产品目标

ResponseLab 是一个 macOS 桌面工具，用于比较参考拟合脉冲与待补偿拟合脉冲的复频响，
按 `H_ref / H_dut` 构造补偿响应，并通过 `FFT → 频域相乘 → IFFT` 应用到 CSV 或
Keysight Infiniium 自描述 BIN 信号。

## 2. 输入

- 两份拟合脉冲 CSV 与待补偿 CSV 共用三种明确合同：
  1. Keysight Infiniium `File Format, WaveformXYValues` v1/v2；必须具有唯一的
     `Points`、`X Units, Second`、`Y Units, Volt`、`Data`，v2 的 `Data` 后必须有
     两列 `float`/`double` 精度声明，实际数据行数必须等于 `Points`；
  2. Keysight 官方 Python 示例的 `Time (s),<source> (V)` 两列表头；
  3. 兼容无表头 `time,value`，GUI 固定按秒和伏特解释。
  三类格式都从经过均匀性验证的时间差推导采样率，也不要求两份脉冲具有相同采样率或
  长度；界面不要求手填采样率。
- `WaveformXYValues` 允许保存不等间隔点，但本工具的 FFT 合同要求均匀采样；默认拒绝
  非均匀时间轴，并提示在 Infiniium 保存时启用 `Linearly Interpolate`。
- `File Format, DatabaseCsv`、协议/测量结果表、Y-only 文件和旧式 `Revision` 多源或
  分段 CSV 不属于当前时域单波形合同，不能模糊猜测。
- 待补偿 BIN 只接受 Keysight Infiniium `AG10` 容器。`Fs` 由秒单位 `X Increment`
  的倒数得到，时间原点取 `X Origin`，幅值单位必须为 Volt；不提供手工采样率、裸流
  dtype/字节序/偏移、通道布局、缩放或偏置输入。
- GUI 只接受恰好一个 waveform 的文件；可加载 waveform 限于 Normal/Average，且必须
  恰好包含一个与 `Points` 一致的 little-endian normal `float32` buffer。未知版本、
  多 waveform、Peak Detect/复合 buffer、非 Second/Volt、尾随未知字节和普通裸 BIN
  均 fail-closed。底层显式索引 API 不改变 GUI 的单 waveform 合同。
- 时间轴必须严格递增且等间隔，最少 8 点。
- BIN 导出为单 waveform Normal AG10，保存 `X Increment`、`X Origin` 和 Volt 单位；
  它只承诺被本工具按上述子集重读，不承诺无损保留输入文件的其他 waveform、分段元数据
  或任意 Keysight/第三方私有格式。
- 主“数据补偿”不按来源设置固定点数门限。CSV 在 `np.loadtxt` 前按文件体积、物理行数
  和实际选择列估算解析峰值；通用无表头格式通过 `usecols` 只建立时间列与所选值列，
  Keysight/显式两列表头格式则解析完整两列以发现任一数据行的多余列。BIN 根据头部
  `Points` 按 `40 B/点 + 16 MiB` 在 payload 映射和时间轴分配前预检。两者与后续 DSP
  使用相同动态预算。CSV/BIN 形成同一 `TimeSeries` 后，`run_compensation` 必须在响应分析、CZT、
  镜像延拓和目标 FFT 前执行共享内存预检。估算包含目标点数/通道、`E=3N-2`、带内
  DFT bins、脉冲 CZT 卷积长度和分析网格；安全预算为 `min(1.5 GiB, 50% 当前可用内存,
  当前可用内存 - 512 MiB)`，系统可用内存不可探测时使用 768 MiB 回退值。

## 3. 频响分析

每份脉冲分别计算 `dt · rfft(h[n])`，然后插值到从 0 到两者较小 Nyquist 的公共物理
频率轴。幅度差为：

```text
ΔA_dB(f) = 20 log10(|H_ref(f)| / |H_dut(f)|)
```

相位只在连续、数值可解析的频段内展开。常规复响应比较不把数值零点当成有效复比；
影响频段纯幅度模式对参考零点的数学极限例外见第 7 节。
响应表中的两路幅度保留 `20 log10(|dt · RFFT(h)|)` 的原始输入标度，再按对数幅度插值
到公共网格；它们不以任一路峰值为 0 dB。自动推荐 -20 dB 频带时才分别按各自峰值
归一化，这个内部选带尺度不写回响应图或响应 CSV。

## 4. 补偿

- 支持仅幅频、仅相频、幅频和相频三种模式。
- 默认从两份归一化脉冲谱共同不低于 -20 dB 的最长连续区间自动推荐谱宽候选；补偿边界取
  该区间的 2% 与 95%。线性相位拟合频带首次用该区间的 8% 与 90% 作为初值并固定，后续自动
  补偿频带不再改写它。数据补偿时补偿上限还受目标信号 Nyquist
  限制，推荐与分析使用同一公共频率轴；若目标上限在该轴上少于 16 点则明确拒绝。
  -20 dB 不是噪声或相位可信度判据，输入需已去基线，用户可以关闭自动模式并手动覆盖。
- 用户设置补偿起点与终点；补偿频带外为单位响应。
- 原始分析差异完整保留用于诊断；实际应用默认最大增益 20 dB，并允许用户修改或显式
  关闭。发生限制时必须按目标实际 DFT bins 报告原始所需峰值，设置写入 manifest。
- 频带外严格使用单位响应；频带内每侧默认用带宽 10% 的 raised-cosine 平滑进入核心。
  运行诊断与应用均以目标 Nyquist 判断物理边缘；相位肩部分别连续展开，不能在 ±π
  分支处制造非物理跳变。
- 若所选频带包含无法数值解析的响应比，分析会拒绝并提示缩小或移动频带。

## 5. 相位拟合与可选去斜

含相位的模式在用户选择的线性相位拟合频带内拟合相位差公共斜率：

```text
estimated_relative_delay_s = slope / (2π)
phase_used(f) = Δφ(f) - slope · f   # 开关打开
phase_used(f) = Δφ(f)               # 开关关闭
```

各连续可信相位岛使用独立干扰截距，按归一化频谱幅度平方加权，只汇总岛内相位变化来
估计公共斜率。零权重频点不参与拟合。结果中的 `phase_trend_rad` 表示拟合得到的
`slope · f`，而不是把各岛截距平均成一条没有物理意义的全局直线；它只在开关打开时
从补偿相位中移除。

界面始终显示带符号相对时延，正值表示 DUT 比参考更晚，负值表示 DUT 比参考更早。
“去除线性相位（保持目标时间位置）”默认打开；关闭时仍拟合并报告时延，但实际补偿
保留完整相位差。常数相位和非线性相位差在两种设置下都保留。仅幅频模式不应用相位补偿。

## 6. 目标数据处理

目标信号先自动做首尾镜像延拓。工具通过 CZT 在延拓记录的每个带内 DFT 频点直接计算
两份拟合脉冲的有限记录频响，再按模式构造 `H_ref/H_dut`；显示分析网格不参与实际应用
响应的插值：

```text
Y[k] = rfft(x[n]) · C(f_k)
y[n] = irfft(Y[k])
```

反变换后取回中间的原记录，避免循环回卷；输出与输入保持相同点数、时间轴和通道数。
若目标记录的 DFT 分辨率导致补偿带内没有频点，则拒绝运行。接近谱零点时使用与脉冲
记录长度和系数 L1 范数有关的 Horner 舍入误差界判断数值可解析性；低于误差界的 DUT
响应不可安全反演。实值 RFFT 的 DC/Nyquist 端点只能接受数值误差范围内的实补偿；若
所需 Nyquist 补偿含真实复相位，则拒绝运行并要求调整频带。

## 7. 影响频段归因

- 一次只按 Vpp、眼高或眼宽中的一个指标排名。三类指标都使用两份等长、等采样率的
  单通道拟合脉冲；Vpp 不再要求或读取参考/DUT 原始采集波形。眼高/眼宽还要求
  `样点数`能被 `M` 整除、自动推导的 `Np ≥ 2`、`M ≥ 3`，并按用户选择的 NRZ 或
  PAM4 固定符号序列回放；页面不再要求重复填写 Np。
- Vpp 提供两种互斥方法：`LFP 峰峰值` 和 `频域 RMS 误差`。二者共享理想码型来源、
  `M`、峰前保留 UI 和峰后保留 UI。`M ≥ 1`；峰前/峰后允许为 0，界面范围为
  0..4096 UI，默认各 8 UI。
- 理想码型可选界面项 `内置 PRBS13Q Gray（8191）` 或单列外部文件。内置码型固定为 8191 symbols，
  Gray 符号码映射为归一化 PAM4 电平 `{-1,-1/3,+1/3,+1}`。外部文件必须由用户显式
  声明为整数符号码 `0..3` 或无量纲幅度系数；算法不按数值范围猜测。无量纲系数
  保留数值尺度/偏置且至少含两个不同电平，拟合脉冲承担每单位符号响应的电压量纲。
  文本在 `np.loadtxt` 前先检查 32 MiB 硬上限和动态内存预算，并冻结初始大小快照；
  有效 symbol 行数先进入周期 FFT 预算。并发追加不能扩大解析输入，超限时提示提供
  “一列单周期 symbol，而不是采集波形”。
- 参考和 DUT 各自使用首次 `argmax(abs(h))` 作为 pmax，窗口包含
  `[-pre_ui·M, +post_ui·M]` 的全部样点。窗口越过任一拟合脉冲边界时拒绝，不截短、
  不补零。窗口 tap 按模折叠到 `pattern_length·M` 个样点的周期核，再与每 UI 起点放置
  一个理想 symbol 的稀疏激励做圆周卷积；这样周期首尾仍包含前后周期拖尾/ISI。
- Vpp 明确使用 `Rs=Fs/M`、`UI=M/Fs`，摘要必须同时显示 Fs、Rs、M 与 UI 时长。
  周期模型以各自 pmax 为零延迟；完整脉冲补偿相位先减去两条脉冲绝对峰值时刻差
  `t_peak,dut-t_peak,ref`（含 CSV `t0` 与索引差），再执行可选的剩余线性去斜。
- LFP 指标是一个完整周期上的精确 `max(y)-min(y)`，单位 V；不用分位数、分块中位数或
  “稳健 Vpp”替代。频域方法令 `D[k]=Y_candidate[k]-Y_reference[k]`，排除 DC，并按
  rFFT 的 DC/Nyquist/共轭频点权重用 Parseval 直接计算 AC 误差，单位 Vrms；参考对自身
  的指标为 0。该量保留复数相位误差，不是总波形 RMS，也不是“等效 Vpp”。
- 局部候选的幅度、相位与幅相补偿仍由两份**完整拟合脉冲**在周期 DFT 网格上的
  `H_ref/H_dut` 得到。pmax 窗口只决定 Vpp 指标模型保留多少脉冲拖尾，不裁短补偿比，
  也不改变主“数据补偿”路径。用户应增加前后窗口直到指标和频段排名收敛。
- Vpp 的 LFP 扫描为每个有效候选恢复一次完整周期时域波形；频域 RMS 扫描直接在复频谱
  计算，不为每个候选执行 IFFT，只有点选/展示波形时才恢复一次时域结果。
- 眼指标先把固定符号写成间隔 `M` 点的冲激列，再与拟合脉冲做完整卷积；每条轨迹围绕
  最大绝对峰值确定并冻结的主光标截取 `-1 UI ~ +1 UI`，共 `2M+1` 点，不需要 Nb。
  默认眼高在固定 0 UI 使用相邻发送轨道的 1% 内侧经验边界；眼宽复用本地眼图库的
  41 条固定电压水平切片、每侧最靠内 crossing、线性插值和 1% 经验边界；1% 分位每侧
  至少要求 100 个 crossing。结果只限制
  为非负，不额外截断到 1 UI。1% 只描述固定轨迹数据库，不代表 BER/SER。参考主光标
  归一化为 1，DUT 与全部候选使用同一幅值尺度。任一眼缺少足够左右 crossing 时眼宽
  标为不可测，不参与数值排名；绘图位置按 DUT 补偿前的电平和中心幅度分位从完整记录
  选取最多 600 个，并由参考/补偿前/补偿后三图严格复用，不只取
  开头。DUT 候选还沿用补偿前冻结的主光标，不允许重新寻优或
  重定位。
- 候选标签表示用户所设的**满权核心**，界面可用 Hz、kHz、MHz 或 GHz 设置，默认
  100 MHz。每侧再接核心一半宽度的半余弦肩部；相邻核心的公共边界权重为 1，不留
  Tukey 零端点扫描缝。规则核心按所设宽度步进；若扫描跨度不能整除，最后追加一个完整
  核心并把其高端锚定到 `scan_high`。当有限
  脉冲或 Vpp 周期模型的物理分辨率更差时，核心向上扩为所设宽度的整数倍，肩宽
  同步保持为实际核心宽度的一半，并给出告警。
- 局部幅度在 `log(|H_ref|/|H_dut|)` 域按权重组合，相位使用展开并可选去斜后的连续
  相位差。`H_ref=0` 且 DUT 响应可逆时，纯幅度比的极限为零，允许生成零幅度补偿；该点
  没有可辨识相位，因此相位与幅相候选保守标为不可解析。DUT 分母为零时三种模式均不
  做反演猜测。
- 全频幅度、相位和幅相回放作为模型闭环诊断保留，但不作为局部候选的硬门禁。推荐在
  全部有效、正改善的局部候选中按原始改善量锁定全局最佳频段。只有这个最佳候选原本
  为“幅相”时，才在**同一频段**内比较幅度与相位；若单模式与联合模式之差不超过
  原始指标差距 1% 和浮点容差中的较大者，就改用改善更大的单模式标签。该简化不会切换
  到另一个频段。
- 无法解析的候选在结果协议中保存 `NaN + valid_mask=False`，影响曲线以断点显示，不能
  用 0 伪装成“可解析但无影响”。
- 后台启动前用真实候选生成器估算成本：候选核心超过 2000 个或包含眼图
  `symbol_count × M`、Vpp `pattern_length × M` 周期频谱/波形在内的预计峰值内存
  超过同一动态安全预算时拒绝运行。Vpp 不沿用旧原始波形的 192 B/点：独立进程实测 RMS
  周期缓存约 448 B/周期点，门禁取 576 B/点；LFP 还要恢复完整候选周期，取 640 B/点。
  局部频带只改变全周期频谱中的非零权重，并不缩短缓存数组，不能按 active bins 折减。
  目标
  样点数乘评估次数超过 5000 万时先提示长任务。取消在文件加载之间、成本估算后、
  工作区准备后、每个候选评估之间以及点选回放的前后检查；已进入的单次 FFT/IFFT 或
  卷积不强制终止，而是在该数值步骤完成后安全丢弃过期结果。

## 8. UI

- 标题说明固定为“频响分析与补偿”。
- 深色三栏工作台：左侧输入，中间六个页签，右侧参数。
- 参数只呈现输入解析所需字段、补偿模式、自动/手动补偿频带、线性相位拟合频带和可见
  的“去除线性相位（保持目标时间位置）”开关。Keysight CSV/BIN 区只说明自动读取
  时间、采样率和电压单位，不显示高级解析或手工 Fs。
- “拟合脉冲比较”只要求两份脉冲；“数据补偿”才要求第三份目标信号并启用导出。
- 绘图工具栏显式提供框选放大、拖动和恢复推荐范围，滚轮缩放始终可用；恢复后的 x/y
  范围关闭自动量程，后续切换标签不会再次改写。
- 频率图的推荐 y 范围按当前可见 x 区间内的实际曲线计算。单条响应相位在显示层按连续
  相位岛统一平移整数圈，不逐点 wrap；关闭去斜时显示未去斜相位。蓝色阴影标记分析/
  补偿频带，橙色虚线标记线性相位拟合频带边界；图例声明实际开关状态。
- 六个页签为：拟合脉冲、频率响应、频响差异比较、频响补偿、输出预览、影响频段。
- “影响频段”一次只比较 Vpp、眼高或眼宽中的一个指标，并分别显示幅度、相位、幅相
  三条改善曲线；眼指标显示参考、补偿前、补偿后三幅共轴的半透明 2 UI 轨迹叠加图，
  中心线固定在 0 UI；Vpp 显示由理想码型与窗口脉冲得到的稳态模型对比。LFP 改善纵轴
  使用 V，频域误差改善纵轴使用 Vrms。
- 只做脉冲比较时禁用“输出预览”，避免把候选补偿响应误解为已经应用的结果。
- 任意输入或参数变化会使当前预览失效并禁用导出。
- 影响页参数变化只使影响结果失效，不禁用已完成的数据补偿导出；两类任务在主窗口内
  互斥运行，并在参数变化或安全关闭时请求后台中断。

## 9. 导出

一次导出生成主输出、响应 CSV 和 JSON manifest。源为 BIN 时，主输出是一个带自动采样率
和时间原点元数据的单 waveform Keysight AG10 文件。响应 CSV 中：两路幅度都是保留输入
标度的 `20 log10(|dt · RFFT(h)|)`；`phase_difference_deg` 是分岛展开相位差；
`fitted_linear_phase_trend_deg` 是 `slope·f` 且不含各岛截距；
`phase_after_optional_detrend_deg` 是开关作用后的实际带内
相位源；`correction_phase_deg` 是包裹后的补偿相位。

`response-lab-manifest/v3` 记录输入来源、有效设置、线性相位斜率、带符号相对时延及其
约定、原始频响 dB 定义与输入标度、直接频域应用信息、输出统计和哈希。源文件在导出前后与
分析时快照核对，三个输出作为可回滚批次提交。

## 10. 验证

```bash
python3 main.py --self-test
QT_QPA_PLATFORM=offscreen .venv/bin/python main.py --gui-smoke-test
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q main.py src tests examples
```

## 11. 依据与证据边界

- Keysight CSV 文件合同依据官方
  [Waveform XY CSV 格式](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_xy_files.htm)、
  [HEADer 命令](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-PG/Content/Topics/Commands/DISK/WAVeform_HEADer.htm)
  和[Python 浮点波形示例](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-PG/Content/Topics/Python/Scripts/waveform-data-float-format.htm)。
- Keysight BIN 文件合同依据官方
  [Waveform BIN 文件说明](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_BIN_files.htm)、
  [BIN File Format 字段表](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/BIN_File_Format.htm)
  和[官方 Python 示例](https://helpfiles.keysight.com/csg/d9300a/Help/examples/XR8/binary-to-csv.py)。
- 内置码型的 8191-symbol PRBS13Q Gray 语义参考 Keysight 的
  [Provided PAM4 Pattern Files](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Signals/Provided_PAM4_Pattern_Files.htm)
  与 IEEE 802.3 公开材料；它是确定性模型激励，不把本工具变成标准合规测量仪。
- 当前 CSV 自动化证据覆盖依据官方文档独立构造的 `WaveformXYValues` v1/v2 夹具、官方
  Python 两列表头夹具和无表头数值夹具；尚无可再分发的实机导出 CSV，因此不表述为
  “已通过真实 Keysight CSV 验证”。旧式 `Revision` CSV 要在取得真实样本后另建 reader。
- 当前 BIN 格式证据只覆盖上述 AG10 时域模拟子集。对其他 Keysight 产品线的 IQ/频域/
  数字 BIN、第三方同名 `.bin`、厂商私有多通道或分段布局均不作兼容声明，必须先获得
  对应官方格式或 adapter 与真实文件闭环证据。
