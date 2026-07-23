# ResponseLab：Windows EXE 与 Keysight CSV/BIN 验收交接

> 本文描述当前代码的真实边界。Windows 构建必须绑定交付 commit，并在干净 Windows
> x64 机器上完成。不要把 `.bin` 扩展名、裸 float32 或第三方厂商私有容器当成
> Keysight Infiniium AG10。

## 0. 当前目标

在 Windows 10/11 x64 上生成 `dist\ResponseLab\ResponseLab.exe`，并验证：

1. Keysight `WaveformXYValues` v1/v2、官方 Python 两列表头和无表头 CSV 在拟合脉冲、
   目标信号与影响频段路径保持可用，并从时间列自动取得采样率；
2. 受支持的 Keysight Infiniium AG10 时域 waveform 自动恢复采样率、时间原点和电压；
3. 导出的 AG10 BIN 能由本工具重新导入，并与 CSV 路径得到一致的补偿结果；
4. 影响频段 Vpp 的 PRBS13Q/外部码型、LFP 和频域 RMS 误差在 EXE 中可运行；
5. 不受支持或过大的 BIN 在读取 payload 或启动大型 FFT 前明确拒绝。

## 1. 当前 BIN 合同

### 1.1 支持的输入子集

- File Header：Cookie 必须是 `AG`，Version 必须是 `10`。非零 File Size 必须等于
  实际文件长度；Infiniium 2026 可写 0，此时以实际 EOF 作为边界。
- GUI 要求文件恰好包含一个 waveform。底层接口可以显式传 `waveform_index`，但 UI
  不在多个 waveform 中猜选。
- Waveform Type 只接受 Normal 或 Average；X Units 必须为 Second(2)，Y Units 必须为
  Volt(1)，`X Increment` 与 `X Origin` 必须是有限物理量。
- 每个可加载 waveform 必须恰好有一个 buffer，且是与 `Points` 完全一致的 normal
  little-endian float32 数据。
- 时间轴与采样率自动得到：

```text
Fs = 1 / XIncrement
t[n] = XOrigin + n · XIncrement
```

界面只显示“自动读取采样率、时间原点和电压单位”，没有手工 Fs，也没有 dtype、字节序、
偏移、通道布局、缩放与偏置等“高级裸 BIN”参数。

主“数据补偿”不再给 BIN 单设固定点数门限。BIN 加载层按文件头检查解码工作集；随后
CSV/BIN 共用的 `run_compensation` 会在响应分析、CZT、镜像延拓和目标 FFT 前，按点数、
通道、带内 bins 与 CZT 长度执行动态内存预检。预算取 8 GiB、当前可用内存 50% 和
保留 512 MiB 系统余量三者最小值；Windows 通过 `GlobalMemoryStatusEx` 读取可用物理
内存。只读 memmap 只降低初始 payload 复制，不取消后续频域工作区成本。
加载器本身也执行同一动态预算：CSV 在 `np.loadtxt` 前按文件大小、物理行与选择列估算；
通用无表头格式用 `usecols` 读取需要的列，自描述格式完整验证两列及各行列数。BIN 在
payload 映射和 `np.arange` 时间轴前按 `24 B/点 + 16 MiB` 估算。影响频段的 Vpp/眼图
门禁同样不再只依赖固定 1.5 GB。

### 1.2 明确不支持

- 无文件头的 raw float32/int16/int32 样本流；
- AG 未知版本、多 waveform 的 GUI 自动选择、Peak Detect 双 buffer；
- 数字/逻辑/直方图/复数/IQ/频域 waveform；
- 非 Second X 轴、非 Volt Y 轴；
- 其他 Keysight 产品线或第三方厂商同名 `.bin`；
- 对输入容器的无损原位编辑，或保留未建模 waveform、buffer、分段与私有元数据。

这些文件必须先取得对应产品/版本的官方格式、真实脱敏样例与独立读回证据，再建立
显式 adapter。不能恢复“尝试若干 dtype/偏移/Fs，直到曲线看起来合理”的路径。

### 1.3 导出子集

源数据为单通道、有限、等间隔时间序列时，导出器写：

- 单 waveform Normal AG10；
- X Units=Second、Y Units=Volt；
- `X Increment=1/Fs`，`X Origin=time_s[0]`；
- normal little-endian float32 payload。

writer 按块检查/量化并使用同目录临时文件原子替换。NaN/Inf、float32 溢出、非均匀时间
轴、多通道或少于 8 点会拒绝。该输出承诺 ResponseLab 自己可重读，不把它描述为任意
Keysight/第三方格式的无损写回。

## 2. 源码入口

- 启动：`main.py`
- GUI 路由与大 BIN 样点门禁：`src/response_lab/ui.py`
- CSV/Keysight BIN 的 TimeSeries 入口：`src/response_lab/io.py`
- Keysight CSV 有界表头识别：`src/response_lab/keysight_csv.py`
- AG10 header 扫描、只读 memmap 和 writer：`src/response_lab/keysight_bin.py`
- Vpp 周期码型模型：`src/response_lab/vpp_analysis.py`
- 导出批次：`src/response_lab/reporting.py`
- Windows 构建入口：`build_window.bat`

`build_window.bat` 不需要厂商 DLL 或额外 reader/writer；Keysight AG10 解析器随
ResponseLab 源码一起构建。其他 Keysight 产品线或第三方同名 `.bin` 仍不在支持范围内。

## 3. Windows x64 构建

### 3.1 环境

- Windows 10/11 x64；64 位 Python 3.11 或更新版本；
- 在干净目录检出已确认的交付 commit；不要复制 macOS `.venv`；
- 构建机器能安装 `pyproject.toml` 中声明的依赖。

```powershell
git clone https://github.com/ZhenlongYou/codex.git
cd codex\codex_projects\frequency_response_compensator
git checkout <交付 commit 或 release 分支>
build_window.bat
```

脚本建立 Windows `.venv`，运行测试、Ruff、compileall、self-test、GUI smoke test，再以
PyInstaller `--onedir` 生成 EXE。交付时必须保留整个 `dist\ResponseLab\`，不能只复制
`ResponseLab.exe`。

若需要手工重现打包命令：

```powershell
.\.venv\Scripts\python.exe -m PyInstaller `
  --noconfirm --clean --windowed --onedir `
  --name ResponseLab `
  --paths src `
  --add-data "src\response_lab\assets;response_lab\assets" `
  --collect-all pyqtgraph `
  --collect-all scipy `
  main.py
```

## 4. 必须执行的验证

### 4.1 源码验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q main.py src tests examples
.\.venv\Scripts\python.exe main.py --self-test
.\.venv\Scripts\python.exe main.py --gui-smoke-test
```

绿色测试只证明已覆盖断言通过。Keysight 格式还要使用与生产 writer 独立构造或由真实
Infiniium 导出的文件进行读入验证，避免 writer 与 reader 共享同一错误布局仍然自洽。

### 4.2 Keysight CSV 验收矩阵

| 验收项 | 必须观察的结果 |
| --- | --- |
| WXY v1/v2 | v1 直接读取数据；v2 跳过并校验精度行；Points、Second、Volt 全部严格匹配。 |
| 官方 Python 表头 | `Time (s),<source> (V)` 自动按秒/伏特读取并保留 source 名称。 |
| 兼容无表头 | 首行数值不能丢失；仍按明确的 time(s),value(V) 合同读取。 |
| 严格拒绝 | DatabaseCsv、错误版本/单位/精度、重复/缺失字段、Points 不符均失败。 |
| 均匀性 | 非均匀 WXY 不产生伪 Fs，并提示在仪器保存时启用 Linearly Interpolate。 |
| 大文件 | 动态内存门禁必须在 `np.loadtxt` 建表前生效，窗口不会冻结或撑爆内存。 |

上述 CSV 自动化夹具来自官方示例的独立编码；没有实机文件时不能写成“已通过真实
Keysight CSV 验证”。

### 4.3 Keysight BIN 验收矩阵

| 验收项 | 必须观察的结果 |
| --- | --- |
| 元数据 | 已知 `XIncrement`、`XOrigin`、Points、Volt payload 全部与独立真值一致。 |
| CSV/BIN 等价 | 同一波形的 CSV 与 AG10 BIN 进入补偿后，时间轴、Fs、输出在 float32 量化容差内一致。 |
| 身份补偿 | `H_ref=H_dut` 时输出与输入一致；不能只断言“没有崩溃”。 |
| 导出重读 | 输出 AG10 重新导入后，Points、Fs、X Origin 和补偿值正确。 |
| 严格拒绝 | 裸 BIN、未知版本、截断、错误单位、多 waveform GUI 输入、Peak Detect 均失败。 |
| 大文件 | 头部扫描不复制 payload；超过当前样点/内存门禁时，在 payload 映射和 FFT 前失败。 |
| Vpp | 可见项为“内置 PRBS13Q Gray（8191）”；外部单列码型在解析前受 32 MiB 上限保护；LFP 显示 V，频域误差显示 Vrms。 |

### 4.4 干净机器验收

在一台没有源码、没有开发 venv 的 Windows 机器上：

1. 启动 `dist\ResponseLab\ResponseLab.exe`，确认图标、六个页签和绘图组件可见；
2. 导入两份拟合脉冲和一个真实/独立 AG10 文件，确认没有手工采样率或高级解析控件；
3. 完成数据补偿并导出 BIN，用 EXE 再导入，检查 Fs、时间原点和波形；
4. 用同一信号的 CSV 对照补偿结果；
5. 运行 Vpp LFP 与频域 RMS 两种模式，检查码型、M、pmax 前后窗口和单位；
6. 导入一个随机裸 BIN 和一个已知不支持变体，确认错误信息明确；
7. 断网后重复启动，确认 EXE 不依赖构建机路径或网络。

只有这些闭环通过，才能交付“支持 Keysight Infiniium AG10 子集的 Windows EXE”。没有
真实 Infiniium 或独立格式夹具证据时，应写“按官方格式实现并通过独立合成夹具”，不能
扩大成“兼容所有 Keysight BIN”。

## 5. 未来第三方 BIN adapter

若后续仍需接入朗视或其他厂商格式，应新增独立 adapter，并至少取得 reader、writer、
采样率/单位元数据、脱敏样例、许可与可再分发依赖。没有 writer 时只允许导出 CSV；不能
把 ResponseLab AG10 writer 的输出标成第三方格式。第三方格式选择必须显式出现在 UI，
不能只按 `.bin` 后缀路由。

## 6. 官方依据

- [Keysight Waveform BIN files](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_BIN_files.htm)
- [Keysight BIN File Format](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/BIN_File_Format.htm)
- [Keysight 官方 Python 转 CSV 示例](https://helpfiles.keysight.com/csg/d9300a/Help/examples/XR8/binary-to-csv.py)
- [Keysight Waveform XY CSV](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_xy_files.htm)
- [Keysight Waveform HEADer](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-PG/Content/Topics/Commands/DISK/WAVeform_HEADer.htm)
- [Keysight Python 浮点波形示例](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-PG/Content/Topics/Python/Scripts/waveform-data-float-format.htm)
