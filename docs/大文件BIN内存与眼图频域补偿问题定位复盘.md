# ResponseLab 大文件 BIN、频域补偿与眼图问题定位复盘

> 目的：这不是一份“改了哪些代码”的流水账，而是一份可以迁移到其他大数据 DSP 工具的排障方法教材。它重点回答：为什么一个并不大的 BIN 会触发很高的内存峰值，如何判断问题来自数据所有权、FFT 工作集还是算法定义，以及怎样用最小反例、独立基准和真实入口把修复闭环。

## 1. 最终结论先行

本次审查发现的问题并非单一的“BIN 读取太占内存”，而是三类问题叠加：

1. **数据表示和数组生命周期造成了不必要的常驻内存。** 原始 BIN 是 `float32`，每点只有 4 字节；旧路径进入算法后却会生成 `float64` 时间轴、`float64` 波形、反射扩展、`complex128` 频谱、全频轴、全频段校正数组、逆 FFT 输出等。第二阶段修复已让 BIN 波形、分块 FFT 和大记录输出保持 `float32/complex64`，只把频率轴、脉冲响应比和安全判定保留在 `float64/complex128`。
2. **频域补偿存在边界语义和安全性缺口。** 包括窄频段未严格遵循实际 RFFT 频点、相位在 `±π` 附近可能错误插值、补偿增益缺少稳健上限、频段硬切换导致时域振铃，以及展示诊断域与目标可应用频域可能被混用等。
3. **眼图指标和展示可能产生“有数但证据不足”或视觉偏差。** 1% 眼宽不能由少量 crossing 可靠估计；只画最前面的轨迹会形成时间顺序偏差；三张眼图若选择不同轨迹，则肉眼比较失去同源性；候选频段只复查前三名还会漏掉“第四名在额外种子成为最佳”所揭示的推荐不稳定性。

修复后的代表性实测结果如下。数值为独立子进程中的 RSS 峰值增量，用于比较同一机器、同一入口下的相对变化：

| 场景 | 修复前 | 修复后 | 说明 |
|---|---:|---:|---|
| 100 万点 Keysight BIN 加载 | 约 72 MiB（旧记录约 75.8 MB） | 22,740,992–27,787,264 B，约 21.7–26.5 MiB | 两次 fresh-process 快照；避免全长差分和理想轴 |
| 100 万点、50–200 MHz 窄带补偿 | 约 601 MiB（旧记录约 630 MB） | 151,011,328–157,581,312 B，约 144.0–150.3 MiB | 两次快照；只保留带内频点并缩短临时量寿命 |
| 50 万点、全奈奎斯特补偿 | — | 160,022,528–167,821,312 B，约 152.6–160.1 MiB | 两次快照；全带宽仍有不可避免的频谱工作集 |
| 100 万点 Keysight BIN 加载（float32 保留路径） | 22,740,992–27,787,264 B | 20,889,600–21,807,104 B，约 19.9–20.8 MiB | 第二阶段两次 fresh-process 快照 |
| 3000 万点、0～Nyquist、全频 2× 补偿 | 精确 `3N-2` 路径估算 12,995,072,032 B | 分块路径估算 543,883,464 B；最终补偿新增高水位 211,451,904 B | 真实 120,000,164 B；输入输出均为 float32 |
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

对应分支为 `codex/responselab-memory-eye-safety`。第一阶段内存/眼图修复提交为
`53560ec`，第二阶段有限边界分块与混合精度提交为 `5240790`，证据门禁与最终内存
包络提交依次为 `48ccd8e`、`eb31e84`；多通道后端余量修复提交为 `f03f256`，绝对预算
上限提升到 8 GiB 的提交为 `35bc081`。

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
| `TimeSeries` | 工具内部的等间隔时域模型，保存 `float64` 时间轴、`float32` 或 `float64` 电压和采样率；BIN 原生幅值不再无条件扩成 float64 |
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
| 电压数组（旧路径 / 当前 BIN） | `float64` / `float32` | 8 / 4 | `N` |
| 时间轴 | `float64` | 8 | `N` |
| 反射扩展波形 | `float64` | 8 | `E = 3N - 2` |
| RFFT 频谱 | `complex128` | 16 | `K = floor(E/2) + 1` |
| IRFFT 输出 | `float64` | 8 | `E` |

要先分开两个阶段。旧版**加载阶段**可能同时存在已触页的 `float32` 映射和正式
`float64` 时间、电压，逻辑量级至多约：

\[
4N + 8N + 8N = 20N\ \text{bytes}
\]

其中 `4N` 映射只有实际触页部分进入 RSS；旧路径还会叠加完整 `diff`、统计副本等。
当前 BIN 保留 float32，加载峰值的主要逻辑量变为映射 `4N`、幅值副本 `4N` 和时间轴
`8N`；加载返回后 `TimeSeries` 常驻约 `12N`。下面的 `88N` 推导描述的是旧的
float64、`3N-2` **精确补偿路径**，不是新分块路径：

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
pytest：356 passed
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
- [`validate_vpp_keysight_pipeline.py`](../examples/validate_vpp_keysight_pipeline.py)：大文件与完整链路验证；
- [`validate_large_bin_streaming.py`](../examples/validate_large_bin_streaming.py)：真实 BIN、自动路由、分块补偿和 fresh-process RSS 复现。

### 当前仍需知道的限制

1. 为保证同路径文件在加载中没有被替换，payload 可能被顺序读取约三次；这主要影响 I/O 时间，不是三份常驻副本。
2. 小记录仍优先使用 `3N-2` 全记录精确 FFT；当该路径超过安全预算时自动切换到有显式“float32 量化 + 冲激响应截尾”联合误差界的有限边界分块路径。若所需上下文太长、量化已经耗尽预算或分块路径也超过内存预算，仍会失败关闭。
3. 20 dB 增益上限和 10% raised-cosine 肩部是安全默认值，不是所有链路的唯一最优配置；裁剪会显式告警。
4. 虚拟眼指标适合相对工程比较，不等价于带 CDR、噪声、抖动和 BER 外推的仪器合规结果。
5. 提交已保存在本地分支；当时 GitHub 凭据失效，因此尚未同步到远端。修复本身和本地验证不受此影响。

---

## 15. 第二阶段：为什么不再要求大记录镜像到 `3N-2`

这一阶段来自三个连续追问：30M 点会不会爆内存、主数据能否统一使用 `float32`、镜像是否一定要达到 `3N-2`。回答这些问题时，不能只把 dtype 改小，因为那只能把同一个高复杂度工作集缩小一半；真正需要改变的是**边界条件如何进入 FFT**。

### 15.1 先分清“精确整段路径”和“数学必需条件”

旧路径对长度为 `N` 的记录执行：

```text
首侧 reflect(N-1) + 原记录(N) + 尾侧 reflect(N-1)
```

因此：

\[
E=(N-1)+N+(N-1)=3N-2
\]

这个长度来自“把完整记录各镜像一遍”的工程选择，不是 FFT、RFFT 或频域补偿定理要求的固定长度。它的优点是语义直接：在一条很长的偶对称延拓记录上做一次循环卷积，再裁出中间原记录。缺点同样直接：当 `N=30,000,000` 时，`E=89,999,998`，频谱有 `45,000,000` 个复数 bin，波形、频谱、CZT 和逆变换临时量会同时进入十 GiB 量级。

真正的数学需求是：对某个输出样点，滤波器有效冲激响应范围内所需的输入邻域必须正确。它并不要求每个块都镜像整条记录。

### 15.2 新路径的核心语义

新路径采用以下规则：

1. **记录内部不镜像。** 分块边界左、右侧的上下文直接来自原记录真实相邻样点。
2. **只在整条记录的两个物理边界镜像。** 当索引小于 0 或大于等于 `N` 时，才使用偶对称反射。
3. **镜像长度由补偿滤波器的有效冲激响应决定。** 不再固定为 `N-1`。
4. **每个块只保留没有循环回卷污染的中央有效区。** 这与 overlap-save 的基本思想一致，但同时支持双边、零相位或非因果校正。

全局反射索引用周期 `2(N-1)` 折叠。设待访问的整数索引为 `q`：

\[
r=q\bmod 2(N-1)
\]

\[
i(q)=
\begin{cases}
r, & r<N\\
2(N-1)-r, & r\ge N
\end{cases}
\]

这样，块内部跨接缝时 `i(q)=q`，取到的是真实邻点；只有超出完整记录边界时才反射。

### 15.3 上下文长度不是拍脑袋设置

分块 FFT 点数记为 `M`，默认是 `1,048,576`。先在这组真实 RFFT bin 上以 `complex128` 构造安全处理后的目标补偿响应 `C64[k]`，再得到一周期的理想冲激响应：

\[
h_{64}[m]=\operatorname{IRFFT}\{C_{64}[k]\}
\]

实际分块 FFT 使用 `float32/complex64`，因此算法先量化

\[
h_{32}=\operatorname{float32}(h_{64})
\]

并计算完整的量化 L1 误差 `||h64-h32||1`。只有量化误差尚未耗尽总预算时，才在 `h32` 上寻找最小对称上下文 `P`。`h[0],h[1],...` 对应一侧时延，数组尾部 `h[M-1],h[M-2],...` 对应另一侧时延。保留：

```text
h[0:P+1] 和 h[M-P:M]
```

总合同不是只检查尾部，而是同时约束量化和被显式置零的尾部：

\[
\frac{
\lVert h_{64}-h_{32}\rVert_1+
\sum_{m=P+1}^{M-P-1}|h_{32}[m]|
}{\lVert h_{64}\rVert_1}
\le \epsilon_{approx}
\]

默认：

\[
\epsilon_{approx}=128\epsilon_{float32}
=1.52587890625\times10^{-5}
\]

选择 L1 界的原因是它能给出固定 `M` 上“理想周期冲激响应”到“实际应用的量化、截尾冲激响应”的最坏逐点卷积误差上界。若输入满足 `|x[n]| <= Xmax`，这两项近似造成的误差满足：

\[
|e[n]|\le X_{max}\left(
\lVert h_{64}-h_{32}\rVert_1+
\lVert h_{32,discarded}\rVert_1
\right)
\]

被计入第二项的 `h32` 中部在应用前会真正置零，再重新 RFFT 得到最终 `complex64` 校正；若只“报告尾部但仍应用完整响应”，块接缝会读取错误的循环回绕样本，报告界就不成立。这个界不等同于实际 RMS 误差，也不覆盖“固定 `M` 网格与旧 `3N-2` 网格不同”或 FFT 运算自身的舍入；后两者分别由路径标识、闭式时域反例和数值容差约束。

每块中央有效点数为：

\[
B=M-2P
\]

如果 `B < max(256, M/16)`，工具不会继续用一个几乎全是上下文的低效块，也不会静默截短冲激响应，而是明确报错，要求增大 `M` 或由用户显式放宽尾部容差。

### 15.4 一块数据到底怎样流动

对中央有效区起点 `s`：

```text
需要的全局索引：s-P ... s+B+P-1，共 M 点
记录内部索引：直接读取真实数据
记录外部索引：按完整记录边界偶对称反射
块 RFFT：float32 -> complex64
频域相乘：complex64 *= complex64 correction
块 IRFFT：complex64 -> float32
写入输出：只取 P ... P+B-1
```

因此当前实现不是“不做镜像”，而是：

> 不再把完整记录左右各镜像 `N-1` 点；只为每个块提供由冲激响应尾界决定的有限上下文，并且只在完整记录的两个端点发生镜像。

### 15.5 为什么采用混合精度，而不是所有变量都强制 float32

主数据通路改为：

| 对象 | 类型 | 原因 |
|---|---|---|
| Keysight BIN 幅值 | `float32` | 文件本身就是四字节电压；避免无意义扩大 |
| 大记录分块输入/输出 | `float32` | 常驻量随 `N` 增长，是内存优化重点 |
| 块 RFFT 频谱 | `complex64` | SciPy 对 float32 RFFT 原生返回 complex64 |
| 时间轴、采样率、频率 bin | `float64` | 防止长记录时间累计和频带端点失去分辨率 |
| 脉冲 CZT、`H_ref/H_dut`、增益与零点判定 | `complex128/float64` | 深陷波、相消、相位和近零响应是数值安全关键 |
| 目标冲激响应与误差审计 | `float64` 目标 + `float32` 应用副本 | 量化误差先占用总 L1 预算，剩余预算才允许截尾 |

这叫“按误差敏感度分配精度”。如果连频率轴和除法安全判定也改为 float32，节省的只是约百万点块级临时量，却会显著增加 Nyquist、窄带端点、深陷波和相位符号误判风险。

### 15.6 自动路由如何保证旧结果不被无声改变

`application_strategy` 有三种合同：

| 值 | 行为 |
|---|---|
| `auto` | 先估算并尝试旧的整段精确路径；只有它超过动态安全预算时，才估算并选择分块路径 |
| `exact` | 强制 `3N-2` 整段路径；超过预算直接拒绝 |
| `streaming` | 强制有限边界分块路径，适合验证与大记录批处理 |

因此小记录已有结果继续使用原来的 `float64/complex128` 整段算法和严格回归容差；大记录才进入有明确联合误差界的 `float32/complex64` 路径。`response-lab-manifest/v4` 把 exact 与 streaming 作为带方法标识的两种 application 合同，并记录 FFT 点数、每侧上下文、每块有效点数、量化相对 L1、截尾相对 L1、联合上界、输出 dtype 和预估峰值，不能把近似分块结果伪装成精确整段结果。

### 15.7 30M 全频案例的内存账本

测试条件：单通道 `N=30,000,000`、`Fs=2 GHz`、补偿 `0～1 GHz`、参考脉冲为幅度 1 冲激、DUT 为幅度 0.5 冲激，因此闭式输出是 `y=2x`。

旧精确路径：

| 项目 | 数值 |
|---|---:|
| 扩展长度 `E=3N-2` | 89,999,998 |
| RFFT bin | 45,000,000 |
| 保守新增峰值估算 | 12,995,072,032 B，约 12.10 GiB |

新分块路径：

| 项目 | 数值 |
|---|---:|
| 块 FFT 点数 | 1,048,576 |
| 块 RFFT bin | 524,289 |
| float32 完整输出 | 120,000,000 B |
| 保守新增峰值估算 | 543,883,464 B，约 518.69 MiB |
| 本例上下文 | 每侧 0 点；常数 2× 校正的冲激响应只有 `h[0]=2` |

> **版本边界（2026-07-23 最终安全审计）：**上表和早期 JSON 中的两个估算值是
> `N_FFT→2N_FFT` 加密网格门禁加入前的历史快照，不是当前准入数字。相同 30M、
> 全带、`N_FFT=1,048,576` 几何在当前代码中，精确路径估算为
> `18,755,432,832 B`，分块路径估算为 `1,173,579,336 B`，其中加密网格审计计费
> `629,335,072 B`。当前 fresh worker 重跑的加载后补偿高水位增量为
> `379,437,056 B`，从空进程基线到最高点的总增量为 `861,011,968 B`，估算包络本次
> 实测。原始历史 JSON 保持不变以保留证据链；引用它们时必须同时标明对应的旧版
> 算法和提交。

真实 120,000,164 B Keysight BIN 的 fresh-process 实测：

| 阶段 | RSS 高水位/增量 |
|---|---:|
| 子进程基线 | 98,254,848 B，约 93.70 MiB |
| 加载后高水位 | 581,599,232 B，约 554.66 MiB |
| 加载新增峰值 | 483,344,384 B，约 460.95 MiB |
| 补偿后高水位 | 793,051,136 B，约 756.31 MiB |
| 已加载基础上的补偿新增峰值 | 211,451,904 B，约 201.66 MiB |
| 相对空进程的总峰值增量 | 694,796,288 B，约 662.61 MiB |

加载耗时约 0.195 s，补偿耗时约 0.617 s；相对独立闭式答案 `2*x` 的最大绝对误差为：

\[
1.1920928955078125\times 10^{-6}
\]

它约为 `float32` 在幅值 2 附近的几个 ULP，符合百万点 FFT 往返和频域乘法的预期。该恒定增益夹具主要验证容量、dtype、自动路由和 FFT 数值；非零上下文、块接缝和双边边界另由三抽头闭式卷积测试验证。

夹具输入是 `[-1,1] V` 线性序列，闭式输出峰值是 `2 V`。CLI 的绝对误差验收合同为：

\[
E_{accept}=64\epsilon_{dtype}\times 2\ \mathrm{V}
\]

其中 64 是给一次块 RFFT、频域乘法和 IRFFT 往返保留的工程舍入倍率，不是把联合 L1
截尾合同重复放宽。float32 阈值为 `1.52587890625e-5 V`；若强制 exact，则同一公式使用
float64 epsilon。脚本和入口回归测试调用同一个公开公式，避免文档、测试与 CLI 各写一份数字。

尾部修复后曾有一次独立 high-water 快照：补偿阶段新增 `322,797,568 B`，高于当时旧
估算 `309,002,440 B`。这促使量化与上下文审计工作数组改成按 `32 B/M` 线性计费。
随后证据 CLI 增加“实测必须被估算包络”的 PASS 门禁，又在 clean commit `48ccd8e`
捕获第二个反例：实测 `447,987,712 B`，高于 `342,556,872 B`。因此 FFT 后端计划、
内部工作区与分配器保留页再按 `192 B/M` 计费；当前估算比最高反例高
`95,895,752 B`，约 21.4%。第一份历史 JSON 未改写地保存在
[`30M_BIN分块补偿高水位校准_2026-07-23.json`](./30M_BIN分块补偿高水位校准_2026-07-23.json)，
SHA-256 为 `6dbbbb8bef89940349fe1939e1f61bc7a63259b3c5b421a957dbb1cbdb379b79`。
它由旧版脚本产生：虽然沿用了 `v1` 字样，却还没有现代报告的 `source`、`invocation`、
status 和 acceptance 字段，无法绑定精确提交，故只作为方向性历史样本，不进入当前
自动验收或估算回归。
第二份由门禁判为 FAIL 的原始反例保存在
[`30M_BIN分块补偿估算反例_2026-07-23.json`](./30M_BIN分块补偿估算反例_2026-07-23.json)，
SHA-256 为 `f2d8706ef232e9343467adb4bee7d9a121607567df0d9e0c9d668a26d6835800`。
它带 clean HEAD、命令、脚本哈希和失败判据，是当前估算回归使用的正式反例；后续
clean commit 上的 PASS 报告才是最终功能验收。

这张表只表示提交 `35bc081` 上一次 fresh worker 的 `ru_maxrss` 高水位样本；父进程生成
夹具、GUI 预览和导出不在本次 worker 测量内，重复运行也会受 FFT 计划、文件页与分配器
状态影响。原始报告保存在
[`30M_BIN分块补偿实测_2026-07-23.json`](./30M_BIN分块补偿实测_2026-07-23.json)：
源码运行时 `git status` 为空，`git diff` SHA-256 是空内容哈希，夹具与脚本 SHA-256、
worker 命令/退出码/stdout/stderr、验收判据和动态预算均在其中。该最终报告自身的
SHA-256 为 `b4bd2e01c51baecdda1714185afedfbcf638392cd15d59341b0ffe5750cb96a0`。
注意 8 GiB 是绝对上限而非预留内存；该次运行加载后可用内存约 5.00 GiB，因此动态预算
按 50% 规则实际为 2,686,820,352 B（约 2.50 GiB）。

### 15.8 30M 实测反而发现了一个新的功能错误

第一次 30M 运行没有 OOM，而是在入口报“补偿频带超过待补偿信号 Nyquist”。继续保留 stderr 后发现：BIN 写入的是 `XIncrement=1/Fs`，读取时再算 `Fs=1/XIncrement`。两次除法往返使恢复的 Nyquist 比 1 GHz 小约一个 ULP，而旧代码使用严格的：

```python
band_high_hz > nyquist_hz
```

于是把数学上相同的全频端点误判成越界。修复不是删除 Nyquist 检查，而是使用：

\[
tolerance=32\epsilon_{float64}\max(|f_1|,|f_2|,1)
\]

容差内视为同一物理端点，并把应用域归一到同一端点；真正超过该范围仍失败关闭。回归测试显式构造 `XIncrement` 相邻可表示值，证明一个 ULP 往返可以通过，而已有“真实越过 Nyquist”和“Nyquist 需要不可表示复相位”的测试继续通过。

这次定位说明一个重要方法：

> 子进程退出、`SIGKILL` 或 UI 报错只是观察，不是根因。必须保留真实异常、退出码、RSS 阶段值和输入合同，才能区分 OOM、门禁拒绝和浮点边界错误。

### 15.9 为什么算法改完后还要检查报告、导出和 GUI

若只改 DSP，内存仍可能在后处理阶段重新爆掉。第二阶段又找到三处线性临时量：

1. manifest 原来把完整 float32 输出转成 float64，再计算 `output**2`，会连续产生两份大数组；现在 SHA-256、min/max/RMS 都按有界块处理，并保留输出真实 dtype。
2. BIN 导出原来无条件把输出转成 float64；CSV 导出还会 `column_stack` 完整时间和值。现在 BIN 保持 float32，CSV 以 131,072 点块写出。
3. GUI 原来把完整 30M 波形交给绘图库，并对完整输入和输出各做一次 FFT。现在时域预览最多 200,000 点，频谱只取中间连续 1,048,576 点；连续窗口避免先抽取再 FFT 的混叠，完整导出数据不受预览裁剪影响。

“算法峰值降低”只有在加载、运行、预览、统计和导出全链路都保持有界时才成立。

## 16. 第二阶段的问题定位流程与可复现入口

### 16.1 实际采用的定位顺序

1. **冻结旧语义。** 先保留小文件精确路径的端点、相位、增益和裁剪测试，不把性能重构混成数学重写。
2. **建立 shape × dtype × 生命周期账本。** 分开加载常驻量、算法新增工作区和导出临时量；不把文件大小当成 RSS。
3. **先写最小闭式 RED。** 全频常数 2× 补偿锁定 dtype、长度和跨多块处理；旧设置模型因没有 streaming 策略而失败。
4. **再写结构反例。** 三抽头 `0.25,1,0.25` 的时域闭式卷积同时检查块接缝、左右上下文和全局反射；正负一拍移位检查相位符号。
5. **加入失败关闭反例。** 构造 200 点远端抽头，使 512 点块没有足够中央有效区；期望是报错，不是给出貌似成功的截尾结果。
6. **把实际入口纳入。** `auto` 在同一 70 MiB 预算下拒绝约 73 MiB 的精确路径、接受约 33 MiB 的分块路径。
7. **跑真实 30M BIN。** 记录 header、payload、dtype、动态预算、阶段 RSS、耗时、输出方法和闭式误差；第一次失败保留完整 stderr，进而发现 ULP Nyquist 错误。
8. **做变异检查。** 人为把上下文强制为 0，三抽头测试会在块接缝和边界产生明显误差，证明测试不是只验证“函数能运行”。
9. **检查下游消费者。** manifest、CSV/BIN 导出和 GUI 预览必须有界，否则总体任务仍未完成。

### 16.2 关键测试入口

```bash
RESP_PYTHON="/Users/mac/PycharmProjects/RinysProject/codex_projects/\
frequency_response_compensator/.venv/bin/python"

PYTHONPATH=src "$RESP_PYTHON" -m pytest -q \
  tests/test_streaming_compensation.py \
  tests/test_large_bin_validation_cli.py \
  tests/test_compensation_memory.py::test_thirty_million_full_band_switches_from_unsafe_exact_to_bounded_streaming \
  tests/test_ui_plot_labels.py::test_thirty_million_output_preview_is_bounded_before_plot_or_fft \
  tests/test_ui_plot_labels.py::test_output_focus_converts_only_three_time_points_for_large_record

# 真实生成 30M 点 Keysight BIN，在 fresh child process 中加载、全频补偿并记录 RSS。
PYTHONPATH=src "$RESP_PYTHON" examples/validate_large_bin_streaming.py \
  --samples 30000000 \
  --strategy auto \
  --output-json results/large_bin_streaming_30m.json
```

验证脚本把夹具 SHA-256、Python/NumPy/SciPy/平台、精确与分块估算、动态预算、
三个 RSS 高水位、墙钟时间、实际应用方法和相对 `2*x` 的最大误差写入 JSON。PASS 还要求：
实际补偿高水位增量不超过所选估算、有效 Git HEAD 和 tracked diff/untracked runtime
文件都已绑定。若补偿后 `ru_maxrss` 没有高于加载后高水位，则该次 RSS 观察记为
`INCONCLUSIVE`，不能假装证明估算包络。脚本默认
用 `--samples 1000000 --strategy streaming` 强制快检真实分块入口；30M 自动路由证据必须
像上面一样同时显式传入 `--samples 30000000 --strategy auto`。

测试与判据的映射：

| 风险 | 独立判据 |
|---|---|
| 常数全频增益错误 | `test_forced_streaming_full_band_constant_gain_matches_closed_form`，闭式 `2*x` |
| 块接缝使用了镜像而不是真实邻点 | `test_streaming_uses_real_neighbors_at_seams_and_reflects_only_global_edges`，三抽头直接卷积 |
| 相位正负号颠倒 | `test_streaming_phase_sign_matches_closed_form_sample_shift`，正负一拍时域移位 |
| 长尾被静默截断 | `test_streaming_rejects_filter_tail_that_leaves_no_safe_block_core` |
| 只审计尾部、却仍应用完整循环响应 | `test_streaming_explicitly_truncates_tail_instead_of_wrapping_it_at_seam` |
| 脉冲延迟恰好落在整块而在网格上别名 | `test_streaming_rejects_block_grid_alias_instead_of_returning_wrong_output` |
| 短脉冲产生未收敛的长逆响应 | `test_streaming_rejects_short_pulse_inverse_when_block_grid_has_not_converged` |
| float32 量化未计入报告界 | `test_nonbinary_three_tap_reports_quantization_plus_truncation_bound` |
| auto 仍走高内存整段路径 | `test_auto_strategy_falls_back_only_when_exact_path_exceeds_budget` |
| 验证脚本算法完成后又在序列化失败 | `test_large_bin_validation_cli_reports_streaming_pass` |
| worker 失败原因被父进程吞掉 | `test_large_bin_validation_cli_preserves_worker_failure_stderr` |
| RSS 高水位未增长却被当成估算证明 | `test_worker_marks_equal_rss_high_water_as_inconclusive` |
| BIN ULP 往返误判 Nyquist | `test_full_nyquist_accepts_one_ulp_bin_increment_round_trip` |
| manifest 把 float32 再扩大 | `test_streaming_manifest_preserves_float32_evidence_and_application_contract` |
| GUI 又对 30M 做完整 FFT 或完整时间轴转换 | `test_thirty_million_output_preview_is_bounded_before_plot_or_fft` 与 `test_output_focus_converts_only_three_time_points_for_large_record` |

### 16.3 如何解释“30M 会不会爆”

不能给一个脱离机器状态的绝对承诺。正确答案由四层组成：

1. **算法可扩展性：** 分块路径的 FFT/CZT 工作集由 `M` 和脉冲长度控制；随 `N` 线性增长的主要新增量是 float32 输出 `4N×channels`。
2. **动态预算：** 绝对上限提高到 8 GiB，仍取当前可用内存的 50% 并保留 512 MiB 系统余量；取三者最小值。
3. **滤波器长尾：** 30M 记录本身能装下，不代表任意补偿响应都能用默认 `M`。长尾若不能满足 L1 合同会明确拒绝。
4. **实测边界：** 文中的 RSS 是“夹具生成完成后启动的 fresh worker 本轮高水位样本”，不是跨运行稳定常数，也没有覆盖父进程造夹具、GUI 绘图和导出。多通道、极长脉冲、复杂长尾、FFT 后端和同时运行的其他程序都会改变结果，必须重新看门禁报告。

因此，当前版本对“30M 单通道 BIN 全频补偿”的回答是：**不会再因为固定 `3N-2` 镜像而天然进入约 12.1 GiB 的工作集；本机真实 30M 案例已经在 fresh worker 中完成且估算包络本轮补偿增量。但这不是对任意机器、通道数和补偿响应的无条件不爆内存承诺，工具仍会根据当时可用内存和实际联合 L1 误差合同决定运行或失败关闭。**

### 16.4 仍然保留的边界

1. 分块路径是有两道独立门禁的工程近似：`N_FFT→2N_FFT` 离散加密网格上的相对 L∞ 收敛检查，以及已通过该检查的块网格上 float32 量化与截尾联合相对 L1 界。前者不是连续频率证明，后者也不宣称与无限精度、无限长卷积或旧 `3N-2` 网格逐位相同。
2. 默认 `M=1,048,576` 适合大量普通校正；极长群时延或很窄、很硬的频域结构可能要求更大的块。
3. 当前输出仍在内存中保留一份完整 float32 数组，尚未改为 memmap 直接落盘；因此多通道的常驻输出仍按 `4N×channels` 增长。
4. `TimeSeries` 仍保存完整 float64 时间轴，30M 单通道约占 240 MB。未来若需要上亿点，可进一步引入“原点 + 增量 + 点数”的隐式时间轴领域模型，但这会影响 GUI、导出和多个算法，不能只做局部替换。
5. GUI 频谱是中间连续窗口的预览，不是完整 30M 记录的全分辨率频谱；manifest 和主输出保留完整数据合同。

## 17. 最值得记住的五句话

1. **大文件内存问题首先是“同时存活的数组问题”，不是文件字节数问题。**
2. **优化前先写 shape × dtype × 生命周期账本，尤其检查小视图是否持有大底层缓冲。**
3. **算法审查要用能让旧实现失败的最小反例，而不是只跑正常样例。**
4. **展示网格、运算网格、统计置信度和 GUI 抽样各有自己的合同，不能混为一谈。**
5. **测试通过只是起点；真实入口、独立 oracle、子进程 RSS、大文件失败时机和剩余边界共同构成交付证据。**
