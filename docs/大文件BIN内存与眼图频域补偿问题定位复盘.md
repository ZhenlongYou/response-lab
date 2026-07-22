# ResponseLab 大文件 BIN、频域补偿与眼图问题定位复盘

> 目的：这不是一份“改了哪些代码”的流水账，而是一份可以迁移到其他大数据 DSP 工具的排障方法教材。它重点回答：为什么一个并不大的 BIN 会触发很高的内存峰值，如何判断问题来自数据所有权、FFT 工作集还是算法定义，以及怎样用最小反例、独立基准和真实入口把修复闭环。

## 1. 最终结论先行

本次审查发现的问题并非单一的“BIN 读取太占内存”，而是三类问题叠加：

1. **数据表示和数组生命周期造成了不必要的常驻内存。** 原始 BIN 是 `float32`，每点只有 4 字节；进入算法后却会生成 `float64` 时间轴、`float64` 波形、反射扩展、`complex128` 频谱、全频轴、全频段校正数组、逆 FFT 输出等。只看文件大小会严重低估峰值工作集。
2. **频域补偿存在边界语义和安全性缺口。** 包括窄频段未严格遵循实际 RFFT 频点、相位在 `±π` 附近可能错误插值、补偿增益缺少稳健上限、频段硬切换导致时域振铃，以及展示诊断域与目标可应用频域可能被混用等。
3. **眼图指标和展示可能产生“有数但证据不足”或视觉偏差。** 1% 眼宽不能由少量 crossing 可靠估计；只画最前面的轨迹会形成时间顺序偏差；三张眼图若选择不同轨迹，则肉眼比较失去同源性；候选频段只复查前三名还会漏掉“第四名在额外种子成为最佳”所揭示的推荐不稳定性。

修复后的代表性实测结果如下。数值为独立子进程中的 RSS 峰值增量，用于比较同一机器、同一入口下的相对变化：

| 场景 | 修复前 | 修复后 | 说明 |
|---|---:|---:|---|
| 100 万点 Keysight BIN 加载 | 约 72 MiB（旧记录约 75.8 MB） | 22,740,992–27,787,264 B，约 21.7–26.5 MiB | 两次 fresh-process 快照；避免全长差分和理想轴 |
| 100 万点、50–200 MHz 窄带补偿 | 约 601 MiB（旧记录约 630 MB） | 151,011,328–157,581,312 B，约 144.0–150.3 MiB | 两次快照；只保留带内频点并缩短临时量寿命 |
| 50 万点、全奈奎斯特补偿 | — | 160,022,528–167,821,312 B，约 152.6–160.1 MiB | 两次快照；全带宽仍有不可避免的频谱工作集 |
| 4000 万点稀疏 Keysight BIN | 可能进入 payload 后才失败 | payload 前拒绝；guard 测试锁定失败时机 | 该步骤曾观测 RSS 增量 0，但不单独以此作证明 |

这里的“低内存”不等于“零内存”。精确全记录 FFT 的复杂度仍是 `O(N log N)`，其频谱和工作区仍需要 `O(N)` 内存。真正的改进是：只保留数学上必需的数据，避免同一信息以多种全长形式同时存活，并在已知无法安全执行时于读取 payload 前失败。

---

## 2. 审查范围与证据边界

审查对象是 RinysProject 中的规范实现，而不是 `/Users/mac/Documents/频响补偿工具` 下可能用于演示或交付的数据副本。规范工程位于：

```text
/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator
```

本次修复在隔离工作树中完成：

```text
/Users/mac/PycharmProjects/RinysProject/.codex-worktrees/responselab-memory-eye-safety/
  codex_projects/frequency_response_compensator
```

对应分支为 `codex/responselab-memory-eye-safety`，核心修复提交为 `53560ec`。

审查范围包括：

- Keysight Infiniium 自描述 BIN 的头部解析、payload 加载和采样率合同；
- 通用波形加载后的 `TimeSeries` 表示；
- 反射扩展、RFFT、频域校正、IRFFT 和中央裁剪；
- 补偿频段、幅度增益、相位和奈奎斯特边界；
- 虚拟眼高、眼宽、轨迹选择和多种子候选复查；
- GUI 实际入口、内存预检、报错路径和回归测试。

没有把虚拟眼图包装成以下能力：

- 真实接收机 CDR；
- 随机抖动与确定性抖动分解；
- 噪声卷积、浴盆曲线和 BER 外推；
- PCIe、OIF 或其他规范的合规判决。

当前眼图是有限确定性激励下的工程代理指标，适合做“补偿前后相对比较”，不应被解释成仪器级 BER 或合规眼图。

### 2.1 首次阅读需要的术语

| 术语 | 本文含义 |
|---|---|
| payload | 文件头之后真正保存采样值的主体字节；Keysight normal float buffer 为每点 4 字节的小端 `float32` |
| `TimeSeries` | 工具内部的等间隔时域模型，至少保存 `float64` 时间轴、`float64` 电压和采样率 |
| RSS | Resident Set Size，进程已经驻留在物理内存中的页面；不是 Python 数组字节数，也不是累计申请量 |
| RFFT bin | 实信号离散傅里叶变换的一个实际频点，频率由记录长度和采样率共同决定 |
| ULP | Unit in the Last Place，某个浮点数附近相邻可表示数之间的最小间隔 |
| TOCTOU | Time-of-check to time-of-use；检查文件后、真正读取前，同一路径可能已被替换 |
| oracle | 不复用被测实现的独立预期值来源，例如手算公式或 NumPy 直接 FFT |
| RED / GREEN | 修复前测试应失败为 RED；修复后同一测试通过为 GREEN |
| rail | NRZ/PAM4 中由已知发送符号形成的一条电平轨迹分布 |
| crossing | 眼图轨迹穿过某个水平阈值的位置，通常通过相邻采样点线性插值得到 |
| 相位岛 | 被不可信谱零点隔开的、连续且可信的一段相位频点 |

---

## 3. 为什么一个大文件 BIN 会需要这么多内存

### 3.1 第一个关键认知：磁盘大小不等于算法工作集

假设 BIN 中有 `N` 个 `float32` 电压样点：

\[
S_{file} = 4N\ \text{bytes}
\]

100 万点 payload 只有约 4 MB。但进入 Python/NumPy/SciPy 后，常见数组是：

| 数据 | 典型类型 | 每元素字节数 | 大致元素数 |
|---|---:|---:|---:|
| 原始 payload（文件/映射） | `float32` | 4 | `N` |
| 电压数组 | `float64` | 8 | `N` |
| 时间轴 | `float64` | 8 | `N` |
| 反射扩展波形 | `float64` | 8 | `E = 3N - 2` |
| RFFT 频谱 | `complex128` | 16 | `K = floor(E/2) + 1` |
| IRFFT 输出 | `float64` | 8 | `E` |

要先分开两个阶段。**加载阶段**可能同时存在已触页的 `float32` 映射和正式 `float64` 时间、电压，逻辑量级至多约：

\[
4N + 8N + 8N = 20N\ \text{bytes}
\]

其中 `4N` 映射只有实际触页部分进入 RSS；旧路径还会叠加完整 `diff`、统计副本等。加载函数返回后，原始 memmap 的局部引用结束，DSP 从 `TimeSeries` 的 `16N` 开始。**补偿阶段**在仅一通道、还没有算校正系数时，基础工作集粗略为：

\[
16N + 8E + 16K + 8E
\]

因为 `E ≈ 3N`、`K ≈ 1.5N`，代入得到约：

\[
16N + 24N + 24N + 24N = 88N\ \text{bytes}
\]

100 万点就是约 88 MB（十进制），即约 83.9 MiB。这还没算：

- DUT 与目标脉冲可能同时存在；
- 全长频率轴；
- 全长复数校正系数；
- 插值、相位展开和幅度限制的中间数组；
- FFT 后端工作区；
- Python 对象、GUI 绘图缓存和内存分配器未立即归还的页。

因此“4 MB 文件为什么能占几百 MB”的答案不是 Python 神秘地把文件放大了，而是**算法同时表示了多个长度为 `N`、`1.5N` 或 `3N` 的实数与复数数组**。

上面的 `88N` 是解释量级的**基础工作集账本**，不是对旧实现约 601 MiB 峰值的精确分解。旧路径还曾产生以下全长或近全长对象：

| 生命周期中的对象 | 旧路径量级 | 新路径处理 |
|---|---:|---|
| `TimeSeries.time + values` | `16N` | 保留，这是当前模型合同 |
| 反射扩展 | `24N` | 保留，但推迟到带内校正求完后再分配 |
| 全 RFFT 频率轴 | 约 `12N` | 窄带时只建 `B` 个带内频点 |
| RFFT 频谱 | 约 `24N` | 保留，最后使用后立即释放 |
| 全频复数 correction | 约 `24N` | 改为 `16B` 字节的带内 correction |
| `spectrum * correction` 结果 | 约 `24N` | 改为频谱带内切片原位相乘 |
| IRFFT 全长输出 | `24N` | 中央裁剪复制完成后立即释放 |
| 返回切片的底层 base | 可继续钉住 `24N` | 返回紧凑自有的 `8N` 结果 |

这里 `B` 是实际补偿频带内的 RFFT bin 数。表中各项的生命周期部分重叠、部分串行，不能机械相加当成 RSS；SciPy FFT/CZT 工作区、文件映射驻留页、GUI 已有对象和分配器保留页也不在 `nbytes` 表中。旧测量没有保存逐时刻原生分配 trace，因此约 601 MiB 只能作为历史高水位基线，不能假装已经逐字节归因。可复现的结论是：新路径删除了哪些确定的 `O(N)` 副本，以及当前独立进程峰值是多少。

### 3.2 反射扩展为什么特别贵

为减小记录边界的突变，补偿不是直接对原波形做 FFT，而是进行反射扩展：

\[
x_{ext} = [x_{N-1},\ldots,x_1,\ x_0,\ldots,x_{N-1},\ x_{N-2},\ldots,x_0]
\]

其长度为：

\[
E = 3N - 2
\]

这个设计本身有信号处理意义，但代价是后续 RFFT 与 IRFFT 都按约三倍长度工作。如果输入先有完整 `time[N]` 和 `value[N]`，再构造 `extended[E]`，峰值自然远高于 BIN payload。

### 3.3 视图不一定节省常驻内存

NumPy 切片通常是视图。例如：

```python
output = inverse_extended[padding : padding + N]
```

`output.nbytes` 看起来只有 `8N`，但它的 `.base` 仍指向长度 `E` 的 `inverse_extended`。只要 `output` 活着，整个 `8E` 底层缓冲就不能释放。这种问题很隐蔽，因为日志若只打印 `output.nbytes` 会以为内存已经降下来了。

定位时必须同时检查：

- `array.nbytes`：当前视图覆盖多少字节；
- `array.base`：是否仍引用更大的底层数组；
- `array.flags.owndata`：结果是否拥有独立紧凑缓冲；
- 变量最后一次使用的位置：大数组何时才真正不可达。

本次修复让中央裁剪结果成为紧凑、独立拥有的数组，然后立即释放三倍长度的逆 FFT 缓冲。这是一次看似多做 `N` 点复制、实际显著降低长期驻留内存的取舍。

### 3.4 `mmap` 不是“无需内存”的同义词

内存映射解决的是“不要先把整个文件复制到 Python 字节串”，但被访问并驻留的映射页会计入进程 RSS，同时相应文件页也可由操作系统页缓存管理；两者是不同观察角度，不是互斥去向。如果随后执行 `astype(float64)`，又会分配一份完整数组。因此，只把 `np.fromfile` 换成 `memmap` 并不能自动消除：

- `float32 → float64` 的完整转换；
- 全长时间轴；
- FFT 输入与输出；
- 补偿中间数组；
- 结果对象对大底层缓冲的引用。

正确的问题不是“是否用了 mmap”，而是“在峰值时有多少字节的数组同时活着，它们各自为什么必须存在”。

### 3.5 RSS、数组字节数和累计分配不是一回事

本次测量使用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 记录进程 RSS 高水位。macOS 返回字节，常见 Linux 返回 KiB，因此测量辅助函数在非 macOS 上乘以 1024。单项峰值增量定义为：

\[
\Delta RSS_{peak}=\max(0,\ ru\_maxrss_{after}-ru\_maxrss_{before})
\]

这不是某一时刻“逻辑仍存活数组”的精确总和：它还可能包含分配器未归还页面、已触页的文件映射和原生库工作区；也不包含未驻留的虚拟地址。若前序步骤已经创造更高峰值，后续操作即使分配很多内存，差值仍可能为 0，所以关键内存场景要放在全新子进程中执行。

选择这种测量方式是因为：

- `array.nbytes` 只统计可见数组，不包含 FFT 工作区和解释器开销；
- `tracemalloc` 主要跟踪 Python 分配，可能看不到 NumPy/SciPy 原生缓冲；
- 同一长寿命进程的内存分配器可能保留已释放页，造成第二次测试失真；
- RSS 峰值表示某一瞬间的存活工作集，不是程序一生累计申请过的总量。

可比较的性能实验应满足：同一机器、同一依赖版本、同一输入、同一入口、全新子进程。否则“优化前后”数字没有可解释性。

---

## 4. 定位过程：从现象到可证伪假设

### 4.1 先确认规范源码和干净基线

第一步没有立刻改算法，而是确认：

1. 用户实际运行的是哪个 `main.py`；
2. Documents 目录是源码、演示数据还是交付副本；
3. 当前仓库是否有用户尚未提交的改动；
4. 项目从工程目录和仓库根目录启动时是否表现一致。

这一步很重要。若在错误副本里修复，即使测试全绿，用户实际入口仍没有变化。若直接在脏工作树上改，后续也无法区分原有改动和本次修复。因此本次使用隔离工作树冻结审查基线。

### 4.2 把“很占内存”拆成可以测量的问题

模糊问题：

> 为什么大 BIN 要这么多内存？

拆成四个可回答问题：

1. **加载峰值**：仅把 Keysight BIN 变成 `TimeSeries`，RSS 增加多少？
2. **DSP 峰值**：给定 `N`、通道数和频段宽度，补偿峰值增加多少？
3. **所有权**：函数返回后，是否仍由小视图持有大缓冲？
4. **失败时机**：明显超预算的文件，是在映射 payload 前拒绝，还是分配到一半才由系统报错？

拆开后才能判断修复应落在加载器、数据模型、DSP 内核还是 UI 门禁，而不是笼统地“优化 NumPy”。

### 4.3 建立内存账本

对每个大数组记录五个字段：

| 字段 | 要回答的问题 |
|---|---|
| shape | 元素数量由 `N`、`E`、`K` 还是带内点数 `B` 决定？ |
| dtype | 每点 4、8 还是 16 字节？ |
| owner | 谁拥有底层缓冲？是否只是视图？ |
| birth | 在哪一步分配？ |
| last use | 最后一次使用后能否立即释放或复用？ |

这张账本暴露了几个主要浪费源：

- 加载均匀波形时，为验证采样间隔构造了全长 `np.diff(time)`；
- 为验证理想时间轴又构造了一个全长 `origin + arange(N) * dt`；
- 窄带补偿仍构造了全频率轴和全频谱校正数组；
- 频谱乘法产生第二份完整复数频谱；
- IRFFT 中央裁剪是视图，返回后仍持有三倍长度底层缓冲；
- 结果模型再次复制了已是规范类型和只读语义的数组。

### 4.4 为每个怀疑点构造最小反例

性能问题需要数字，算法问题需要反例。以下反例能以很小输入暴露真实错误：

| 怀疑点 | 最小反例 | 错误表现 | 正确不变量 |
|---|---|---|---|
| 均匀时间轴验证 | 巨大 `origin` 加很小 `dt` | 浮点加法把相邻时间压成重复值 | 重复或非单调时间必须拒绝 |
| 只抽查端点 | `dt` 约为 `origin` 的 1.1 ULP | 端点似乎合理，中间局部不均匀 | 分块验证覆盖全部相邻间隔 |
| 频段边界 | `E=28`、`Fs=1 GHz`、低端等于理论第 2 bin | 手工 `Fs/E` 与 `rfftfreq` 差 1 ULP | 选择必须遵循实际公开频轴 |
| 窄陷波 | 目标差异只位于一个真实 DFT bin | 展示网格插值漏掉差异 | 补偿在实际 DFT bins 上求值 |
| 相位插值 | 左右相位分别接近 `+π`、`-π` | 线性插值错误地穿过 0 | 先连续展开再插值 |
| 增益换算 | 20 dB 上限 | 若用 `/10` 会得到 100 倍幅度 | 电压增益为 `10^(dB/20)=10` |
| 1% 眼宽 | 每侧只有 20 个 crossing | 1% 分位由一个事件任意支配 | 每侧至少约 `ceil(1/p)=100` 个 crossing |
| 候选稳定性 | A/B/C 初始前三，D 在额外种子胜出 | 只复查前三会漏选 D | 所有候选、所有启用模式都参与复查 |
| 眼图比较 | 三组各自选“好看”的轨迹 | 视觉差异来自抽样，不一定来自补偿 | 三张图共享 DUT-before 选出的索引 |

反例的价值在于：它不是“我觉得这里有风险”，而是能让旧实现失败、修复后通过，并明确告诉测试究竟证明了什么。

### 4.5 先写 RED，再做最小语义修复

每个高风险问题都先固化为失败测试：

1. 用独立公式或逐元素实现给出期望值；
2. 确认旧实现确实失败，避免写出“永远绿”的同源测试；
3. 修复只改变对应语义；
4. 再运行局部、模块、全套和 GUI 入口验证。

例如，RFFT 频段边界测试直接使用 `np.fft.rfftfreq` 和 `np.fft.rfft/irfft` 构造独立预期，不调用生产代码内部的频轴辅助函数。这避免了“测试和实现共享同一个错误公式”。

---

## 5. 大文件 BIN 的具体修复思路

### 5.1 均匀采样序列不再先物化完整时间轴验证副本

Keysight BIN 自描述头部给出 `XOrigin` 与 `XIncrement`。对于已经验证格式和时间单位的波形，加载器现在通过 [`TimeSeries.from_uniform_samples`](../src/response_lab/models.py) 建立统一采样序列。

核心原则是：

- 时间轴仍可作为模型对外合同存在；
- 验证不再额外构造 `diff[N-1]`、中位数工作副本和理想轴 `[N]`；
- 使用最大 131,072 点的受控分块覆盖全部相邻间隔；
- 检查有限性、严格单调、重复点、局部均匀性以及元数据采样率一致性；
- 分块只限制验证临时内存，不改变验证覆盖率。

这点很容易被误解：**分块验证不是只抽样检查**。它仍逐段覆盖整个序列，只是不让临时数组随 `N` 无限增长。

### 5.2 先读头部、算预算，再碰 payload

Keysight BIN 是自描述格式，头部包含波形数量、缓冲类型、点数、每点字节数、`XIncrement`、`XOrigin` 和单位等信息。加载流程现在是：

```text
打开文件
  → 严格解析并验证所有头部边界
  → 验证只能选择单一 normal float buffer
  → 验证 XUnits 是时间且 XIncrement 有效
  → 根据点数估算加载和后续模型内存
  → 超预算则在 payload 前拒绝
  → 映射/读取指定 payload
  → 转换并构建 TimeSeries
  → 再次确认源文件身份和内容未变化
```

加载预算采用保守模型：

\[
S_{load,est} = 40N + 16\ \text{MiB}
\]

它不是声称每次恰好使用 `40N` 字节，而是将时间轴、电压转换、payload 页、模型和固定开销包进当前受测环境的保守门禁估算。它在 macOS arm64、Python 3.12、NumPy 2.5、SciPy 1.18 的代表性测试中包络实测值；不同操作系统、FFT 后端和分配器仍由额外固定余量与系统安全预算保护。预算门禁的目标是“不要启动已知注定危险的任务”，不是跨平台预测到最后一个字节，也不是数学意义上的绝对内存上界。

4000 万点稀疏夹具的逻辑 payload 约 152.6 MiB。验证确认它在高层入口、映射 payload 和构造全长时间轴之前被拒绝；单次流程曾观测该步骤 RSS 高水位增量为 0，但真正锁定失败时机的是“payload 映射或时间轴构造一旦被调用就使测试失败”的 guard 测试。这样的提前拒绝比等待系统在中途抛出 `MemoryError` 更可控，也能给用户明确的所需内存与当前安全预算。

### 5.3 文件一致性检查与内存优化不能互相牺牲

加载器仍需要防止“检查的是旧文件、读取的是同路径新文件”的 TOCTOU 问题。当前实现对文件身份和内容进行前后确认，因此 payload 可能被顺序读取约三次：预哈希、转换、后哈希。

这会增加 I/O 时间，但不会同时保留三份 payload。它是**时间成本**，不是三倍常驻内存。若以后进一步优化，应先证明文件一致性合同仍成立，不能简单删掉安全检查。

### 5.4 不用“降低估算值”伪装成内存优化

只把 UI 的预计内存数字改小，会让更多危险任务进入执行，却没有减少一个数组。本次顺序是：

1. 用数组账本和子进程 RSS 找到真实峰值；
2. 修改数据结构和生命周期；
3. 重新实测；
4. 让估算器包络实测值，并保留 FFT 后端工作区余量；
5. 用测试保证估算不会低于代表性实测。

估算是安全门，不是宣传数字。

---

## 6. 频域补偿的算法审查与修复

### 6.1 补偿目标的数学定义

设目标脉冲和 DUT 脉冲在真实 RFFT 频点上的频谱分别为：

\[
H_{ref}[k],\quad H_{dut}[k]
\]

理想复数校正为：

\[
C_{ideal}[k] = \frac{H_{ref}[k]}{H_{dut}[k]}
\]

应用到输入波形 `x[n]` 时：

\[
Y[k] = X[k]\,C[k],\qquad y[n]=\operatorname{IRFFT}\{Y[k]\}
\]

但工程实现不能直接无条件使用这个比值：当 `|H_dut|` 很小时，比值会放大噪声和数值误差；频段边缘硬切换会产生时域振铃；相位若不连续展开，会产生错误群时延。因此 `C[k]` 必须是带有限定条件的安全校正，而不是裸除法。

### 6.2 必须在真实 DFT bins 上计算，不依赖展示网格

分析页面可以把频响插值到公共展示网格，但真正补偿的频点由扩展长度 `E` 和采样率 `Fs` 决定：

\[
f_k = \operatorname{rfftfreq}(E, 1/F_s)
\]

如果先在稀疏展示网格求校正、再插值到 DFT bins，一个只落在单个真实 bin 的窄陷波可能完全消失。因此补偿改为直接在目标应用频点求 DUT 和参考脉冲的复数响应。

频段包含关系也必须使用同一个 `rfftfreq` 数组，不能另写 `kF_s/E` 再期待浮点端点逐位一致。一个可手算的边界例子：

- `Fs = 1 GHz`
- `E = 28`
- `low = 2 × Fs/E`

数学实数上它对应第 2 个 bin，但二进制浮点中 `rfftfreq(E, 1/Fs)[2]` 可能比单独计算的 `2×Fs/E` 小 1 ULP。实现和测试统一遵循 NumPy/SciPy 的实际频轴合同，避免边界选择在不同路径中不一致。

### 6.3 DUT 近零频点必须失败关闭

若：

\[
|H_{dut}[k]| \approx 0
\]

则理想比值趋于无穷。正确行为不是让 `NaN/Inf` 传播到整条波形，也不是静默当作 1，而是：

- 判断该频点是否在实际应用频段内；
- 在带内无法定义时给出明确错误；
- 在带外不让无效比值污染结果；
- 对接近阈值但仍可计算的频点应用增益上限并记录告警。

这里的“近零”不是随意写死一个绝对电压。设脉冲系数为 `h[n]`、脉冲长度为 `L`、响应峰值为 `H_peak`，当前有效门限为：

\[
T_{zero}=\max\left(64\epsilon H_{peak},\
\gamma_{8(L-1)}\frac{\lVert h\rVert_1}{F_s}\right),\qquad
\gamma_m=\frac{m\epsilon}{1-m\epsilon}
\]

第一项约束相对机器精度，第二项是复数 Horner 直接求值的保守前向误差界。CZT 初算结果低于约 `H_peak·sqrt(ε)` 时，会再用本机 `longdouble` Horner 求值复核；最终只有 `|H_dut| ≥ T_zero` 才视为数值可逆。对应实现是 [`_pulse_response_on_uniform_frequencies`](../src/response_lab/dsp.py)，反例是 `tests/test_end_to_end.py::test_long_delay_dut_zero_respects_horner_error_bound`。

### 6.4 增益上限必须按幅度 dB 换算

幅度增益的 dB 定义为：

\[
G_{dB}=20\log_{10}|C|
\]

因此 20 dB 的线性幅度上限为：

\[
|C|_{max}=10^{20/20}=10
\]

不能使用功率换算 `10^(dB/10)`，否则 20 dB 会错误地允许 100 倍电压增益。本次默认最大增益为 20 dB；实际发生裁剪时，结果中保留告警，避免用户把受限补偿误解为理想完全反演。

### 6.5 频段边缘使用 raised-cosine 肩部

若校正系数在频段边界从 `1` 突然跳到 `C`，频域不连续对应长时域振铃。边缘权重使用 raised-cosine 平滑过渡：

\[
w(u)=\frac{1-\cos(\pi u)}{2},\quad 0\le u\le1
\]

默认每侧过渡宽度为整个选定频带宽度的 10%，即 `0.10 × (band_high-band_low)`；若频带从 DC 开始则不造左肩，若到达目标应用 Nyquist 则不造右肩。幅度按 `1+w(|C|-1)` 从 1 平滑到目标幅度，相位沿肩部连续分支按 `w·φ` 过渡。这样保留带内目标，同时降低硬矩形窗引起的旁瓣能量。

### 6.6 相位必须先处理 `±π` 分支切换

相位 `+179°` 和 `-179°` 实际只相差 2°，但直接线性插值会认为相差 358°，中点错误地穿过 0°。这里有两个相关但不同的相位处理，不能混为一谈：

1. **显示分析与线性延时拟合**：谱零点会把可信频点分成多个相位岛。每个岛独立 `unwrap`，锚点先取响应置信度最高的频点；若并列，再取最靠近岛中心者，以免一个岛的整数圈偏置污染另一个岛。
2. **实际补偿的 raised-cosine 肩部**：核心区使用 `exp(jφ)`，主值相位相差整周并不改变复数校正；只有肩部从单位响应过渡到目标响应时分支才重要。左肩从外边界向带内展开，右肩反向后展开再翻回，保证各自走连续的短路径。

完成各自的连续化后才做延时拟合或边缘加权。

这类错误通常不会让程序崩溃，却会在时域表现为异常振铃、长尾或延时跳变，因此必须用接近 `±π` 的专门反例验证。

### 6.7 可应用范围与肩部边界必须遵循目标奈奎斯特

当目标波形和 DUT 的采样率不同，可应用的最高频率由目标应用网格决定。分析页面仍可把两份脉冲在公共奈奎斯特范围内的高频差异显示出来，作为诊断信息；但目标波形奈奎斯特以上没有可应用的 DFT bins，不能把“图上可见”误写成“可以补偿”。

修复后，自动频段建议、实际校正频点和 raised-cosine 右肩是否存在，都以目标应用奈奎斯特为边界；显示分析网格不因此被强制裁掉。若用户选择的频段窄到没有覆盖任何实际 DFT bin，工具会明确提示，而不是生成“运行成功但实际上没改任何频点”的结果。

### 6.8 DC 与偶数长度 Nyquist bin 只有实数自由度

对实信号 RFFT：

- DC bin 必须是实数；
- 偶数长度记录的 Nyquist bin 也必须是实数；
- `irfft` 的时域结果没有这些端点的独立虚部自由度。

实现不会简单依赖 `irfft` 静默丢弃虚部。若补偿频带包含 DC，或扩展长度 `E` 为偶数且频带包含最后一个 Nyquist bin，则检查端点校正 `C`：当

\[
|\operatorname{Im}C| > \max(|C|,\ tiny)\sqrt{\epsilon}
\]

时，说明校正确实要求实值时域信号无法表达的复相位，必须失败关闭；只有不超过该相对容差的数值噪声才投影到具有相同符号和幅度的实数。频域误差也按实际 `irfft` 自由度计算，不能把不会出现在时域中的端点虚部计入误差或改善量。

---

## 7. 频域补偿如何降低峰值内存

主要改动位于 [`dsp.py`](../src/response_lab/dsp.py)：

1. 使用 SciPy 的实数 FFT，并在安全条件下允许覆盖输入缓冲；
2. 只为选中频带构造频率、幅度、相位和校正数组，不再为窄带任务创建全频带辅助数组；
3. 对完整频谱执行原位乘法，不创建第二份同尺寸 `complex128` 结果；
4. 每个大数组在最后一次使用后立即释放引用；
5. IRFFT 后只复制中央 `N` 点到紧凑拥有的结果，然后释放 `E` 点缓冲；
6. 数据模型可以接纳已满足 dtype、连续性和只读合同的数组，避免再复制一次；
7. 峰值估算按宽带与窄带分别计算，并为后端工作区保留安全余量。

为什么没有直接把 FFT 按文件块分块？因为当前补偿是全记录反射扩展后的精确频域乘法。普通分块会改变：

- 频率分辨率；
- 记录边界条件；
- 长脉冲/长时延响应的卷积结果；
- 中央裁剪与全记录 FFT 的等价性。

若要真正流式化，需要明确设计 overlap-save/overlap-add、有限脉冲响应截断、块长、重叠长度和误差上界，并建立与全记录结果的容差合同。它不是“把循环切成几块”的无语义优化。本次选择保留精确语义，降低常数工作集，并对超预算任务提前拒绝。

---

## 8. 眼图与眼高眼宽的审查

### 8.1 先明确虚拟眼图的生成逻辑

工具使用已知符号序列激励脉冲响应，先完成完整线性卷积，再按单位间隔 UI 折叠成约 2 UI 的轨迹窗口。这里要区分：

- **2 UI 是绘图/测量窗口宽度**；
- **不是只截取 2 UI 的脉冲响应再卷积**。

若先截断脉冲响应，会漏掉长前游标或后游标 ISI，虚拟眼图会被人为变得更开。

### 8.2 眼高口径

在固定采样相位（0 UI）按已知发送电平分组。对相邻两条 rail，使用稳健分位数估计垂直开口：

\[
EH_i = Q_{low}(V_{i+1}) - Q_{high}(V_i)
\]

其中上下分位数与设置的概率口径一致。眼高为负时表示分布已重叠；缺少足够电平样本时应标记不可用，而不是伪造为 0。

### 8.3 眼宽不能在 crossing 证据不足时给出精确数字

眼宽通过多个水平切片上、每条轨迹最靠近 0 UI 的左右 crossing 估计。设尾部概率为 `p`，要让该分位至少约有一个真实事件支撑，每侧 crossing 数至少应为：

\[
N_{cross,min} \ge \left\lceil\frac{1}{p}\right\rceil
\]

当 `p=1%` 时，至少需要约 100 个左 crossing 和 100 个右 crossing。修复后的实际门限还取以下最大值：

\[
\max\left(5,\left\lceil\frac{1}{p}\right\rceil,
\left\lceil0.01N_{trace}\right\rceil\right)
\]

证据不足时返回 `NaN/unavailable`，不是 0。两者含义完全不同：

- `0 UI`：已经测量并确认眼闭合；
- `NaN`：没有足够 crossing 支持这个概率口径。

完整眼宽估计如下。对相邻 rail 的 0 UI 中位数 `c_low<c_high`，在两者间 5%–95% 高度均匀取 41 个阈值 `t`。每条轨迹在每个 `t` 上分别找最靠近 0 UI 的左、右 crossing；只有左右事件数都满足上面的门限，该阈值才有效。对有效阈值：

\[
L(t)=Q_{1-p}(x_{left}),\qquad
R(t)=Q_p(x_{right}),\qquad
W=\max\left(0,\max_t[R(t)-L(t)]\right)
\]

例如 `p=1%` 时，若某个切片有 100 个左 crossing 都在 `-0.45 UI`、100 个右 crossing 都在 `+0.40 UI`，该切片眼宽为 `0.85 UI`；若每侧只有 20 个 crossing，则不是 `0 UI`，而是因证据不足得到 `NaN`；若有足够事件但所有内侧 crossing 都落在 0 UI，才得到已经测量的 `0 UI`。实现不会把大于 1 UI 的有限记录结果强行截成 1 UI。

### 8.4 crossing 选择必须保留“最靠内”语义

一条轨迹可能与同一水平切片相交多次。对左侧应选择小于 0 UI 且最接近 0 的 crossing，对右侧选择大于 0 UI 且最接近 0 的 crossing。等于阈值的样点和线性穿越必须统一处理，避免同一 crossing 因符号位或零值规则被算到错误一侧。

实现采用受控批次同时处理 41 个水平切片，降低 Python 循环和线程争用，但仍用逐行独立基准验证数值一致性。优化的是执行方式，不是指标定义。

### 8.5 不能只画最前 600 条轨迹

符号序列有时间结构，最前 600 条可能集中在某些电平、码型或瞬态阶段。固定取头部会造成展示偏差。现在按 DUT-before 的发送标签和中心幅度分布进行覆盖式选择，让图中轨迹覆盖不同 rail 与分布位置。

更重要的是，参考、补偿前和补偿后三张图复用同一组 DUT-before 索引。这样肉眼看到的变化主要来自波形处理，而不是三次抽样恰好选了不同轨迹。

### 8.6 多种子稳定性不能只复查初始前三名

一个典型反例：主种子排名为 A、B、C、D，D 位列第四；额外种子中 D 成为最佳。如果只复查前三名，就看不到主推荐其实不稳定，仍会错误保留 A。

修复后的真实流程不是把三个种子的分数聚合后重新排名，而是：主种子先产生一项推荐；两个额外种子各自全量复扫：

- 所有候选频段；
- 幅度、相位和联合三种比较模式；
- 每个额外种子重新选出自己的最佳频段与模式。

若额外种子的最佳结果与主推荐在频段容差或模式上不一致，工具取消主推荐并标记其不稳健；它不会把第四名 D 自动晋升为新的最终推荐。这个门禁证明的是“不要输出不稳定推荐”，不是多种子统计最优性。

相应地，工作量和内存估算也按真实评估次数更新，避免 UI 显示的预计耗时仍停留在旧逻辑。

眼图测量与多种子复查的核心代码分别见 [`virtual_eye_metrics.py`](../src/response_lab/virtual_eye_metrics.py) 和 [`attribution.py`](../src/response_lab/attribution.py)；工作量门禁位于 [`influence_controller.py`](../src/response_lab/influence_controller.py)。

---

## 9. 测试为什么有效：从“测试数量”转向“证据矩阵”

338 个测试通过本身不是正确性的证明。更重要的是，每项用户可见要求是否有独立、能失败的证据。

| 要求 | RED 反例 | 独立基准 | GREEN 证据 | 仍然不证明什么 |
|---|---|---|---|---|
| 窄带只修改真实频点 | 单 bin 陷波、ULP 边界 | 直接 `np.fft.rfftfreq/rfft/irfft` | 端到端输出逐点一致 | 不证明任意模拟前端都可逆 |
| 20 dB 增益限制正确 | 近零 DUT 响应 | `10^(20/20)` 手算 | 幅度不超过 10，且有告警 | 不证明 20 dB 是所有系统最佳值 |
| 相位跨 `±π` 连续 | `+179°/-179°` | 连续相位手算 | 无 358° 错绕 | 不证明测得相位本身无噪声 |
| 频段边缘减振铃 | 同一校正的硬边和余弦边 | 时域边缘能量比较 | raised-cosine 能量更低 | 不等于所有波形都无振铃 |
| 1% 眼宽证据充分 | 每侧 20 crossings | `ceil(1/p)` | 返回 unavailable | 不等于 100 事件已达到 BER 置信度 |
| 图间可直接比较 | 三组不同分布 | 索引逐项相等 | 三组复用 before 索引 | 不证明抽样图等于完整密度图 |
| 大 BIN 安全拒绝 | 4000 万点稀疏文件 | 文件头声明点数 | payload 前失败、RSS 增量 0 | 不证明任务能在当前内存运行 |
| 内存估算不乐观 | 实测子进程峰值 | OS RSS | 估算值包络实测 | 不保证所有 FFT 后端完全相同 |

### 9.1 验证阶梯

本次采用由小到大的验证阶梯：

1. **手算级**：dB 换算、DFT bin、crossing 数门限；
2. **合成解析信号**：纯延时、单 bin 陷波、已知 rail 和 crossing；
3. **边界反例**：`±π`、Nyquist、巨大时间原点、1 ULP；
4. **局部单元测试**：验证单个函数合同；
5. **端到端测试**：加载、分析、补偿、模型与报告串联；
6. **独立子进程 RSS**：避免内存分配器历史污染；
7. **GUI 无界面烟雾测试**：验证用户实际入口和线程工作流；
8. **大文件稀疏夹具**：验证 payload 前门禁，而不为测试写满磁盘；
9. **独立审查者复核**：寻找实现者没有想到的反例。

这种阶梯比“只跑一次真实大文件”更容易定位失败层级，也比“只有单元测试”更能发现入口集成问题。

---

## 10. 被否决的捷径，以及为什么否决

### 10.1 “换成 mmap 就好了”

否决原因：mmap 只改变文件访问方式，不消除 float64 转换、时间轴、FFT、频谱和模型复制。若不做数组生命周期账本，RSS 可能几乎不变。

### 10.2 “把内存估算调低，让用户先跑”

否决原因：这是取消安全门，不是优化。真实分配不变，失败只会更晚、更难解释。

### 10.3 “把整段 FFT 随便切块”

否决原因：会改变频率分辨率、反射边界和长响应卷积。没有 overlap-save/overlap-add 设计与误差合同之前，不能声称结果等价。

### 10.4 “在展示频率网格算补偿就够了”

否决原因：窄陷波可能只落在真实 DFT bin，展示网格会漏掉它。画图网格和运算网格职责不同。

### 10.5 “裁剪过的增益不提示用户”

否决原因：结果看起来正常，但已经不再是理想反演。安全限制必须进入可见告警和报告。

### 10.6 “眼图取前 600 条最快”

否决原因：快但有顺序偏差，且三张图各取自己的 600 条会破坏对比公平性。

### 10.7 “初始前三名已经足够稳定”

否决原因：排名第四的候选可能在额外种子中最稳健。稳定性审查必须覆盖完整候选集。

### 10.8 “测试都绿了，所以完成了”

否决原因：同源测试可能复现同一错误；测试也可能没有覆盖真实 GUI 入口、内存峰值或大文件失败时机。必须审查测试是否能在错误实现上变红。

---

## 11. 可复用的问题解决方法

以后遇到“大数据算法内存高、速度慢、结果可疑”时，可按以下顺序处理。

### 第 1 步：写清合同和单位

- 输入是什么格式、dtype、单位？
- `Fs`、`Rs`、每 UI 样点数 `M` 是否区分清楚？
- 输出是精确值、近似值还是展示值？
- 边界包含还是不包含？
- 证据不足时是 0、NaN 还是报错？

### 第 2 步：确认真实入口和规范源码

- 用户从哪里启动？
- 是否有 demo、副本、打包目录或旧工作树？
- 当前改动是否隔离，能否回到可重复基线？

### 第 3 步：冻结最小复现和基线数字

- 固定输入、版本、机器和命令；
- 分开测加载、计算、绘图；
- 用独立子进程测 RSS 峰值；
- 记录运行时间、结果误差和失败阶段。

### 第 4 步：画内存生命周期账本

对所有 `O(N)` 数组写出：

```text
名称 / shape / dtype / nbytes / owner / birth / last-use
```

重点查：隐式 dtype 提升、广播临时量、花式索引复制、切片保留大 base、模型二次复制、FFT 输入输出重叠时机。

### 第 5 步：为算法疑点构造最小反例

优先选择：

- 单一 DFT bin；
- DC/Nyquist；
- `±π` 相位；
- 1 ULP 边界；
- 零分母；
- 样本数刚好低于统计门限；
- 排名刚好在 Top-K 之外。

### 第 6 步：先证明旧实现会错

测试若不能让旧实现失败，就无法证明修复覆盖了真实缺陷。期望值应尽量来自手算、标准库直接公式或更慢但清晰的独立实现。

### 第 7 步：做最小语义修复

优化优先级通常是：

1. 删除重复信息；
2. 缩短大数组生命周期；
3. 只计算用户选中的子域；
4. 在安全条件下原位复用；
5. 分块临时计算但保持全覆盖；
6. 最后才考虑改变算法的近似或流式版本。

### 第 8 步：检查所有权，而不只看局部 `nbytes`

- 返回数组是否拥有数据？
- 小切片是否挂着三倍大底层缓冲？
- 只读包装是否又复制了一遍？
- GUI 或结果历史是否保留旧工作区？

### 第 9 步：走真实路径验收

- 工程目录启动；
- 仓库根目录启动；
- GUI 线程和进度路径；
- 正常小文件；
- 可执行但接近预算的大文件；
- 明显超预算、应提前拒绝的大文件。

### 第 10 步：写明剩余边界

完成不等于没有限制。必须告诉后续使用者：

- 哪些成本是算法不可避免的；
- 哪些是有意保留的安全 I/O；
- 哪些指标只是工程代理；
- 若要进一步流式化，需要重新定义什么语义。

---

## 12. 下次排查时可直接复制的工作表

```markdown
# 问题名称

## 现象
- 用户看到什么？
- 输入规模、格式、单位是什么？
- 在哪个入口、哪个版本复现？

## 合同
- 正确结果的数学定义：
- 边界/异常/证据不足的语义：
- 允许的近似与误差：

## 基线
- 加载 RSS 峰值：
- 计算 RSS 峰值：
- 运行时间：
- 输出数值：

## 内存账本
| 数组 | shape | dtype | 字节数 | owner | 产生位置 | 最后使用 |
|---|---:|---:|---:|---|---|---|

## 可证伪假设
1. 如果 ______ 是根因，那么构造 ______ 后应观察到 ______。
2. 如果 ______ 是根因，那么移除/隔离 ______ 后应观察到 ______。

## 最小反例
- 输入：
- 手算或独立基准：
- 旧实现输出：
- 正确不变量：

## 修复
- 改变的语义：
- 不改变的语义：
- 内存/时间复杂度变化：

## 验证矩阵
| 要求 | RED | 独立 oracle | GREEN | 限制 |
|---|---|---|---|---|

## 剩余风险
- （填写仍未消除的风险或适用边界）
```

---

## 13. 本案已填工作表与复现入口

### 13.1 受测环境

当前验证环境为 macOS arm64，Python 3.12.1、NumPy 2.5.1、SciPy 1.18.0、PySide6 6.11.1。下面命令使用规范工程已有虚拟环境运行隔离工作树源码：

```bash
RESP_PROJECT=/Users/mac/PycharmProjects/RinysProject/.codex-worktrees/responselab-memory-eye-safety/codex_projects/frequency_response_compensator
RESP_PYTHON=/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator/.venv/bin/python
cd "$RESP_PROJECT"
```

路径若在其他机器不同，只需修改这两个任务专用变量。不要混用系统 Python，否则依赖版本变化会让性能数字不可比。

### 13.2 案例 A：100 万点 BIN 加载

| 工作表字段 | 本案填写 |
|---|---|
| 现象 | `float32` payload 约 4 MB，旧路径 RSS 峰值记录约 72 MiB |
| 假设 | 全长时间轴之外又建立 `diff`、中位数工作副本或理想时间轴，且 `float32→float64` 转换与映射页重叠 |
| RED | 巨大 `XOrigin` 加极小 `XIncrement`、内部重复、严格递增但局部不均匀三个反例 |
| 修复 | `from_uniform_samples` 只保留一条正式时间轴，使用最大 131,072 点临时块覆盖全部间隔验证 |
| 当前实测 | 两次独立进程高水位增量为 22,740,992 B 与 27,787,264 B，即约 21.7–26.5 MiB |
| 当前门禁 | `40N+16 MiB = 56,777,216 B`，约 54.1 MiB |
| 证据边界 | 修复前的精确命令和逐分配 trace 当时未归档，旧值仅作方向性基线；本次 fresh-process 输入哈希与原始高水位已单独归档 |

对应可执行测试：

```bash
PYTHONPATH=src "$RESP_PYTHON" -m pytest -q \
  tests/test_io.py::test_bin_loader_estimate_tracks_optimized_child_process_rss \
  tests/test_io.py::test_uniform_bin_axis_rejects_increment_below_float64_origin_resolution \
  tests/test_io.py::test_uniform_bin_axis_rejects_internal_duplicate_rounding \
  tests/test_io.py::test_uniform_bin_axis_rejects_strict_but_nonuniform_rounding
```

第一个测试验证“当前门禁估算包络已归档实测”，后三个测试验证“降低临时内存没有变成抽样漏检”。它们不会重新生成某个固定 RSS 数字；若要比较新机器性能，应按 3.5 节的 `ru_maxrss` 方法在全新子进程重新测量，并把环境、输入哈希和基线高水位一起记录。

### 13.3 案例 B：100 万点窄带补偿

| 工作表字段 | 本案填写 |
|---|---|
| 输入 | `N=1,000,000`、单通道、`Fs=2 GHz`、参考/DUT 脉冲各 1024 点 |
| 频带 | 50–200 MHz，实际估算包含 225,002 个 RFFT bins |
| 现象 | 旧路径历史峰值约 601 MiB，明显高于基础 `88N` 账本 |
| 假设 | 全频轴、全频 correction、频谱乘法副本、IFFT 大 base 和模型二次复制在不同阶段重叠 |
| 修复 | 带内 CZT/校正、原位频谱相乘、显式最后使用、紧凑裁剪、结果模型接管已有数组 |
| 当前实测 | 两次 fresh-process 快照为 151,011,328 B 与 157,581,312 B，即约 144.0–150.3 MiB |
| 当前估算 | 262,430,752 B，约 250.3 MiB，包络实测并保留后端余量 |
| 证据边界 | 历史 601 MiB 没有原生逐分配 trace，不能声称已完全归因；本次 fresh-process 输入/输出哈希和峰值已归档 |

回归入口：

```bash
PYTHONPATH=src "$RESP_PYTHON" -m pytest -q \
  tests/test_compensation_memory.py::test_memory_estimate_envelopes_two_independent_rss_measurements \
  tests/test_compensation_memory.py::test_run_adopts_owned_application_output_without_a_second_full_copy \
  tests/test_end_to_end.py::test_frequency_application_returns_a_compact_owned_array
```

50 万点全奈奎斯特的两次快照为 160,022,528 B 与 167,821,312 B（约 152.6–160.1 MiB），估算为 250,632,352 B（约 239.0 MiB）。宽带点数虽少一半，但 `B` 变为 750,000，所以峰值与 100 万点窄带接近；这正说明“样点数”不是唯一尺度，带内频点数和 CZT 工作长度也必须进入预算。

### 13.4 案例 C：4000 万点稀疏 BIN 门禁

验证脚本用独立 `struct` 写入器生成格式完整的 AG10 头部和稀疏零 payload；逻辑文件真实为约 152.6 MiB，但文件系统不必实际写满所有零块。运行：

```bash
RESP_TMP_ROOT="$(mktemp -d /tmp/responselab-validation.XXXXXX)"
PYTHONPATH=src "$RESP_PYTHON" examples/validate_vpp_keysight_pipeline.py \
  --output-dir "$RESP_TMP_ROOT/result" \
  --large-samples 40000000
```

成功时终端打印 `PASS`，并生成：

```text
$RESP_TMP_ROOT/result/validation_report.json
$RESP_TMP_ROOT/result/validation_report.md
```

报告保存 Python、平台、NumPy、Git 状态、逻辑/分配文件大小、各阶段墙钟时间和 `ru_maxrss`。注意：脚本在高层拒绝前已经做过其他步骤，因此该步骤的高水位增量可能为 0；**0 本身不是“未触碰 payload”的充分证据**。失败时机还由以下测试独立保证：它们把 payload 映射、时间轴构造或响应分析替换为“一旦调用就失败”的 guard，证明预算门禁先执行。

```bash
PYTHONPATH=src "$RESP_PYTHON" -m pytest -q \
  tests/test_io.py::test_bin_sample_budget_stops_before_payload_mapping \
  tests/test_io.py::test_bin_dynamic_budget_stops_before_payload_mapping_and_time_axis \
  tests/test_compensation_memory.py::test_run_compensation_rejects_csv_and_bin_before_analysis_allocation
```

### 13.5 算法结论到具体证据的索引

| 结论 | 可搜索的测试 node id |
|---|---|
| RFFT 端点遵循真实频轴 | `tests/test_end_to_end.py::test_band_endpoint_selection_matches_rfftfreq_rounding_contract` |
| 窄带告警使用实际目标 bin | `tests/test_end_to_end.py::test_gain_limit_warning_uses_actual_target_fft_bins_for_a_narrow_band` |
| 默认 20 dB 限制 | `tests/test_end_to_end.py::test_default_gain_limit_caps_an_unstable_inverse_at_twenty_db` |
| raised-cosine 降低边界振铃 | `tests/test_end_to_end.py::test_raised_cosine_band_edges_reduce_impulse_ringing_energy`；比较脉冲响应中央保护区之外的平方能量 |
| `±π` 肩部连续 | `tests/test_dsp.py::test_phase_edge_taper_uses_a_continuous_branch_across_pi` |
| off-grid DUT 零点失败关闭 | `tests/test_end_to_end.py::test_off_grid_dut_zero_on_application_bin_is_rejected` |
| DC/Nyquist 实自由度 | `tests/test_end_to_end.py::test_nyquist_negative_real_correction_is_preserved` 与 `test_unrepresentable_complex_target_nyquist_correction_is_rejected` |
| crossing 缺失不是 0 眼宽 | `tests/test_virtual_eye_metrics.py::test_missing_crossings_are_unavailable_instead_of_zero_width` |
| 1% 眼宽至少百事件 | `tests/test_virtual_eye_metrics.py::test_one_percent_eye_width_requires_at_least_one_hundred_crossings` |
| crossing 向量化等价于逐行 oracle | `tests/test_virtual_eye_metrics.py::test_vectorized_crossing_selector_matches_manual_row_oracle` |
| 绘图覆盖全记录且三图共享索引 | `tests/test_attribution.py::test_eye_metrics_use_all_traces_and_plot_stratifies_the_full_record`，以及同文件对 `plot_trace_indices` 的配对断言 |
| 第四候选不能被 Top-3 隐藏 | `tests/test_attribution.py::test_multiseed_review_cannot_hide_a_fourth_band_that_becomes_best` |

这个最后的测试期望是发现不稳定后**取消主推荐**，不是把第四候选自动晋升。

### 13.6 原始 fresh-process 记录

本次重新测量的环境、基线/结束高水位、输入与输出 SHA-256、估算值和夹具公式保存在 [`内存实测记录_2026-07-23.json`](./内存实测记录_2026-07-23.json)。记录同时保留较早快照，并明确标注哪些修复前数字没有原始 JSON，避免把历史近似值写成可逐字节复现的基准。

完整回归、实际入口和 GUI 烟雾测试：

```bash
PYTHONPATH=src "$RESP_PYTHON" -m pytest -q
PYTHONPATH=src "$RESP_PYTHON" main.py --self-test
QT_QPA_PLATFORM=offscreen PYTHONPATH=src "$RESP_PYTHON" main.py --gui-smoke-test
```

## 14. 本次交付的验证记录

完成修复后执行并通过：

```text
pytest：338 passed
Ruff：src、tests、main.py、验证脚本全部通过
工程目录 main.py --self-test：通过
仓库根目录入口自检：通过
QT_QPA_PLATFORM=offscreen GUI smoke test：通过
git diff --check：通过
4000 万点稀疏 BIN 高层门禁：payload 前拒绝；guard 测试证明失败时机
```

文档另经技术准确性和首次读者可复现性两类独立审查；审查提出的多种子语义、Nyquist 展示边界、RSS 证据链和眼宽公式问题均已回写到本文。审查结论不是算法证据本身，最终仍以可执行测试、原始记录和明确限制为准。

主要实现与测试入口：

- [`models.py`](../src/response_lab/models.py)：均匀时间序列、数组所有权和模型合同；
- [`keysight_bin.py`](../src/response_lab/keysight_bin.py)：Keysight 头部与 payload 解析；
- [`io.py`](../src/response_lab/io.py)：加载预算、源文件一致性与通用入口；
- [`dsp.py`](../src/response_lab/dsp.py)：频域补偿、内存估算和频点安全；
- [`virtual_eye_metrics.py`](../src/response_lab/virtual_eye_metrics.py)：眼高、眼宽与 crossing；
- [`attribution.py`](../src/response_lab/attribution.py)：虚拟眼构造、多种子全候选复查和推荐稳定性；
- [`influence_controller.py`](../src/response_lab/influence_controller.py)：工作量门禁与 GUI 数据；
- [`test_compensation_memory.py`](../tests/test_compensation_memory.py)：内存峰值与预检；
- [`test_dsp.py`](../tests/test_dsp.py)、[`test_end_to_end.py`](../tests/test_end_to_end.py)：频域数学与端到端反例；
- [`test_virtual_eye_metrics.py`](../tests/test_virtual_eye_metrics.py)：眼高眼宽统计合同；
- [`validate_vpp_keysight_pipeline.py`](../examples/validate_vpp_keysight_pipeline.py)：大文件与完整链路验证。

### 当前仍需知道的限制

1. 为保证同路径文件在加载中没有被替换，payload 可能被顺序读取约三次；这主要影响 I/O 时间，不是三份常驻副本。
2. 全记录精确 FFT 仍需要 `O(N)` 峰值内存。超过安全预算时会提前拒绝，而不是冒险执行。
3. 20 dB 增益上限和 10% raised-cosine 肩部是安全默认值，不是所有链路的唯一最优配置；裁剪会显式告警。
4. 虚拟眼指标适合相对工程比较，不等价于带 CDR、噪声、抖动和 BER 外推的仪器合规结果。
5. 提交已保存在本地分支；当时 GitHub 凭据失效，因此尚未同步到远端。修复本身和本地验证不受此影响。

---

## 15. 最值得记住的五句话

1. **大文件内存问题首先是“同时存活的数组问题”，不是文件字节数问题。**
2. **优化前先写 shape × dtype × 生命周期账本，尤其检查小视图是否持有大底层缓冲。**
3. **算法审查要用能让旧实现失败的最小反例，而不是只跑正常样例。**
4. **展示网格、运算网格、统计置信度和 GUI 抽样各有自己的合同，不能混为一谈。**
5. **测试通过只是起点；真实入口、独立 oracle、子进程 RSS、大文件失败时机和剩余边界共同构成交付证据。**
