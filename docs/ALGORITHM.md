# ResponseLab 算法说明

内部统一使用秒、Hz、rad 和幅度定义的 dB（`20·log10`）。

## 1. 补偿方向

参考拟合脉冲的响应为 `H_ref(f)`，待补偿拟合脉冲的响应为 `H_dut(f)`。补偿方向固定为：

```text
C(f) = H_ref(f) / H_dut(f)
```

交换两份脉冲会得到相反的补偿方向。

## 2. 脉冲频响

CSV 时间列固定使用秒。每份脉冲分别通过时间间隔得到采样率，并计算：

```text
dt = median(time[n+1] - time[n])
fs = 1 / dt
H(f) = dt · rfft(pulse)
```

`dt` 标度避免两份脉冲采样率不同时产生虚假的幅度差。两份响应随后插值到从 0 到较小
Nyquist 的公共物理频率轴。相位只在连续可解析区段内展开，不跨越数值为零的频谱缺口。

界面默认自动推荐补偿频带。先把两份幅度谱分别按各自峰值归一化，寻找两者共同不低于
`-20 dB` 的最长连续区间 `[f_a, f_b]`，补偿频带取：

```text
补偿频带 = [f_a + 0.02·(f_b-f_a), f_a + 0.95·(f_b-f_a)]
```

首次分析且用户尚未输入线性相位拟合频带时，用区间的 8% 与 90% 作为初值并固定。后续
自动补偿频带只更新补偿边界，无论用户是否继续手动调整，都不会改写相位拟合频带。

这是确定性的幅度谱候选，不是“标准规定带宽”，也不是噪声底或相位可信度估计。输入
脉冲应已去除直流基线，用户必须检查拟合带内相位质量，并可切换为手动设置。目标数据
Nyquist 只在与实际分析相同的公共频率轴上裁剪候选范围；可用点不足时直接拒绝。

## 3. 幅频差

```text
delta_magnitude_db(f) = 20·log10(|H_ref(f)| / |H_dut(f)|)
```

在补偿频带核心内，幅度补偿就是精确的 `|H_ref| / |H_dut|`，不做最大增益、最大衰减
或正则化裁剪。例如 `|H_ref|=1`、`|H_dut|=0.01` 时，补偿为 100 倍，也就是 40 dB。

所选补偿频带内完整使用该响应比，频带外严格使用单位响应。若待补偿脉冲在所选频带内
存在数值零点，响应比发散，分析会明确报错并要求用户缩小或移动补偿频带。仅幅频模式
下若参考响应为零，则补偿幅度就是零。

## 4. 相频差与去斜

```text
delta_phase(f) = phase_ref(f) - phase_dut(f)
```

若 DUT 相对参考晚 `tau` 秒，则相位差包含 `+2πf·tau`。工具在用户设置的线性相位拟合
频带内拟合公共斜率：

```text
delta_phase_i(f) ≈ slope·f + intercept_i
estimated_relative_delay_s = slope / (2π)
phase_used(f) = delta_phase(f) - slope·f   # 开关打开
phase_used(f) = delta_phase(f)             # 开关关闭
```

带符号相对时延的约定为：正值表示 DUT 比参考更晚，负值表示 DUT 比参考更早。界面始终
显示该估计；“去除线性相位（保持目标时间位置）”开关只决定是否从实际补偿相位中扣除
`slope·f`，不会关闭拟合或时延报告。仅幅频模式不应用相位补偿。

拟合不是把所有频点直接交给一条普通直线。频谱零点可能把可信相位分成多个不连续区段，
而各区段独立展开后可能相差整数个 `2π`。工具让每个连续区段拥有独立截距，先在区段内
按频谱置信度加权并中心化，再汇总各区段的斜率分子与分母。因此斜率拟合只使用区段内
的相位变化，不会把区段之间的整数圈偏移误判为线性趋势。权重为零的频点不参与拟合，
非零权重保持原始相对比例，不会被机器精度阈值抬平成同权。

诊断数组 `phase_trend_rad` 记录拟合的 `slope·f`；只有开关打开时该趋势才从补偿相位中
移除。各区段的 `intercept_i` 只是拟合斜率时消去的干扰参数，不会被平均成一条虚假的
全局截距线。

开关打开时，“频率响应”绘图分别扣除参考与待补偿脉冲各自的线性相位，再给每个连续
相位岛统一减去整数个 `2π`；关闭时显示未去斜相位。“频响差异比较”仅在打开时同时显示
去斜前与去斜后曲线，关闭时明确显示“未去斜，实际补偿”。图例声明实际开关状态。

## 5. 三种模式

```text
仅幅频       C(f) = |H_ref| / |H_dut|
仅相频       C(f) = exp(j·phase_used)
幅频 + 相频  C(f) = (|H_ref| / |H_dut|) · exp(j·phase_used)
```

所选频带外三种模式都严格回到 `1+0j`。

## 6. 导出字段契约

响应 CSV 的两路幅度均以公共分析网格上的
`max(max|H_ref|, max|H_dut|)` 为 0 dB 参考；`phase_difference_deg` 是分岛展开的原始
相位差，`fitted_linear_phase_trend_deg` 是 `slope·f` 且不含各岛截距，
`phase_after_optional_detrend_deg` 是开关作用后的实际带内相位源，
`correction_phase_deg` 是复补偿响应经 `angle()` 得到的包裹相位。

`response-lab-manifest/v3` 同时记录幅度共同峰的线性值与 dB 定义、带符号相对时延、符号
约定、有效设置、直接频域应用信息和文件哈希。共同峰的线性单位继承输入幅度单位乘秒；
若输入幅度未标定，manifest 明确记为未指定，而不把它伪称为绝对物理单位。

## 7. 应用到待补偿数据

待补偿数据先在首尾各镜像延拓一份记录。工具通过 CZT 在延拓记录自己的每个带内 DFT
频点直接计算两份拟合脉冲的有限记录频响，并在这些实际频点构造补偿响应，然后执行：

```text
X[k] = rfft(x[n])
Y[k] = X[k] · C(f_k)
y[n] = irfft(Y[k])
```

显示分析网格只用于绘图和诊断，不作为实际补偿响应的插值源；因此其两点之间的窄陷波
若恰好落在目标 DFT 频点上，仍会被识别。接近谱零点的 CZT 结果使用直接多项式求值复核，
并用与记录长度、系数 L1 范数有关的 Horner 舍入误差界判断可解析性；低于误差界的 DUT
响应会拒绝运行，可解析的有限小响应仍按用户选择原样应用。目标记录过短、补偿带内没有
实际 DFT 频点时也会拒绝。实值 RFFT 的 DC/Nyquist 端点只能乘实数；若目标 Nyquist 需要
超出数值误差的复相位补偿，会拒绝运行而不是静默投影。没有额外的滤波器设计或二次逼近。
反变换后取回中间原记录，避免循环回卷；输出与输入的点数、时间轴和通道数保持一致。

## 8. Vpp 的周期稳态码型模型

### 8.1 理想码型与 pmax 窗口

Vpp 不直接测两份任意长度的采集波形，而是让参考和 DUT 拟合脉冲接受**同一个理想周期
码型**。设码型有 `K` 个 symbols，每 UI 有 `M` 个样点，则一个周期有：

```text
P = K · M  samples
Rs = Fs / M  [baud]
UI = M / Fs  [s]
s[n] = a[k],  n = k·M
s[n] = 0,     其他样点
```

内置码型是 8191-symbol PRBS13Q Gray，四个符号码映射到
`{-1, -1/3, +1/3, +1}`。外部 CSV/TXT 必须一行一个值，并由用户明确选择 `0..3`
符号码或无量纲幅度系数；工具不自动猜测。系数保留数值尺度与偏置，电压量纲来自
拟合脉冲的每单位符号响应。外部文本在 `np.loadtxt` 前受 32 MiB 硬上限和动态预算
保护，并先冻结初始大小快照、用有效行数门禁周期 FFT；并发追加不会扩大解析输入。
超限表示很可能误选了采集波形，应改为只提供一列单周期 symbols。

参考和 DUT 分别取首次出现的绝对峰值：

```text
pmax = first argmax(abs(h[n]))
pre_samples  = pre_ui  · M
post_samples = post_ui · M
```

保留窗口包含 `pmax-pre_samples ... pmax+post_samples`。任一端越过完整拟合脉冲边界时
直接拒绝，不静默裁剪或补零。窗口只定义指标模型包含多少前游、后游拖尾；用来生成局部
补偿的 `H_ref/H_dut` 仍由完整拟合脉冲计算。

两份周期核都把各自 pmax 当作 lag=0，因此完整脉冲频响比的相位也先减去
`t_peak,dut - t_peak,ref`，其中 `t_peak=t0+pmax/Fs`；之后才应用用户选择的剩余线性
相位去斜。这样 CSV 时间原点差或脉冲数组索引差不会被重复施加到已对齐的周期模型。

### 8.2 圆周卷积与拖尾

以 pmax 为零延迟，把窗口 tap 按模 `P` 折叠为周期核：

```text
g[q] = sum h[pmax + l]，其中 l mod P = q
Y[k] = rfft(s)[k] · rfft(g)[k]
y[n] = irfft(Y)[n], n = 0 ... P-1
```

这是一个完整周期的稳态圆周卷积。窗口中来自 pmax 之前的 tap 会作用到周期前端，来自
pmax 之后的 tap 会跨到下一个周期；因此不会把第一个 symbol 当成“前面全是零”的启动
瞬态。若增加峰前/峰后 UI 后结果仍明显变化，说明原窗口尚未覆盖足够拖尾，不能把当前
排名解释成已经收敛的 ISI 结论。

### 8.3 LFP 峰峰值

LFP 方法对参考、DUT 和每个候选都使用一个完整稳态周期：

```text
LFP_Vpp(y) = max(y[n]) - min(y[n])    [V]
```

这是确定性码型模型上的精确极差，不是 `Q99.9%-Q0.1%`、分块中位数或随机采集估计。
每个候选需要一次 `irfft` 才能知道全周期最大值与最小值。

### 8.4 复频谱 AC RMS 误差

频域方法比较候选和参考的**复频谱**，因而幅度误差和相位误差都会进入结果：

```text
D[k] = Y_candidate[k] - Y_reference[k]
D[0] = 0                            # 排除 DC
```

设周期长度为 `P`。当 `P` 为偶数时：

```text
error_rms = sqrt(
    2·sum(|D[k]|², k=1...P/2-1) + |D[P/2]|²
) / P                               [Vrms]
```

当 `P` 为奇数时没有独立 Nyquist 点，所有非 DC rFFT bin 都乘 2。这个式子来自未归一化
前向 FFT 的 Parseval 关系，DC 和 Nyquist 不能误乘双边权重。参考相对于自身的指标定义为
`0 Vrms`；候选越小表示越接近参考。它不是候选波形本身的 RMS，也不是把 RMS 乘某个
经验系数得到的“等效 Vpp”。扫描时可以直接在频域计算，不为每个候选执行 IFFT；只有
用户点选一个候选并需要展示波形时才恢复该周期。

### 8.5 频段评分与物理分辨率

对候选频段 `b`，先由完整脉冲产生局部补偿 `C_b[k]`，再得到：

```text
Y_b[k] = Y_dut[k] · C_b[k]
score_b = |metric_dut - metric_ref| - |metric_b - metric_ref|
```

LFP 的 metric 单位为 V；频域误差的 metric 单位为 Vrms，且 `metric_ref=0`。两者不能放在
同一数轴或互相换算。周期模型的独立频率尺度约为 `fs/P`，仍需和两份有限拟合脉冲的
`fs/Npulse` 一起约束候选核心宽度；更密的显示频点不等于更高的物理定位能力。

## 9. Keysight Infiniium BIN 合同

当前 `.bin` 路径不是无头样本流解析器。加载器先读取 Keysight `AG10` 文件头、waveform
header 和 data header，再从 `X Increment` 得到：

```text
Fs = 1 / XIncrement                 [Hz]
t[n] = XOrigin + n·XIncrement       [s]
```

只有 X 单位为 Second、Y 单位为 Volt、Normal/Average waveform 且恰好一个与 `Points`
一致的 little-endian normal float32 buffer 才进入算法。GUI 要求文件恰好一个 waveform；
未知版本、多 waveform、Peak Detect、非秒/伏特单位、截断/尾随字节和随机裸 BIN 都明确
拒绝。删除手工 Fs 与“高级解析”是为了避免把错误元数据伪装成可计算时间轴。
主数据补偿不再给 BIN 单设固定点数硬上限。CSV/BIN 进入同一 `TimeSeries` 后，
`run_compensation` 在响应分析、CZT、镜像延拓和目标 FFT 前估算新增峰值：模型显式使用
`N`、通道数、`E=3N-2`、带内 DFT bins、`next_fast_len(Npulse+B-1)` CZT 长度和分析网格。
预算同时受 1.5 GiB、当前可用内存 50% 和至少 512 MiB 系统余量约束；可用内存探测失败
时采用 768 MiB 保守回退。BIN 加载器仍按文件头独立检查解码工作集。
具体地，BIN 在 payload memmap 和全长时间轴前按 `112 B/点 + 32 MiB` 估算；CSV 在
`np.loadtxt` 前按文本字节 3 倍、物理行数和实际选择列估算，并用 `usecols` 避免读取
无关宽列。加载器、直接补偿和影响频段三条路径共用同一系统可用内存快照规则。

导出写成单 waveform Normal AG10，保存 `XIncrement` 与 `XOrigin`，数据量化为
little-endian float32。该 writer 只承诺本工具支持子集的可重读性，不会保留输入文件中
未建模的其他 waveform、复杂 buffer 或厂商私有元数据。官方格式依据：

- [Keysight Waveform BIN files](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_BIN_files.htm)
- [Keysight BIN File Format](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/BIN_File_Format.htm)
- [Keysight binary-to-csv.py](https://helpfiles.keysight.com/csg/d9300a/Help/examples/XR8/binary-to-csv.py)

## 10. 结果检查

1. 只比较时选择两份脉冲并点击“拟合脉冲比较”；需要生成补偿数据时再选择目标信号并
   点击“数据补偿”。
2. 在“拟合脉冲”确认两份数据角色正确。
3. 在“频率响应”确认所选补偿带内响应有效。
4. 在“频响差异比较”检查幅度差、相位差和去斜结果。
5. 在“频响补偿”检查候选补偿幅度和相位。
6. 执行“数据补偿”后，在“输出预览”检查补偿前后的波形与频谱。
7. 分析 Vpp 时确认码型来源、M 和 pmax 前后窗口；分别核对 LFP 的 V 与频域误差的
   Vrms 标签，逐步增加窗口确认排名收敛。
8. 导入 BIN 时确认页面显示自动元数据解析；若工具要求用户猜 dtype、偏移或 Fs，说明
   运行的不是本合同对应版本。
