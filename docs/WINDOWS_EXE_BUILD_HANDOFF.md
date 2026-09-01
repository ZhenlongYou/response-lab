# ResponseLab：Windows EXE 打包 Agent 执行合同

> 本文描述当前代码的真实边界。Windows 构建必须绑定交付 commit，并在干净 Windows
> x64 机器上完成。不要把 `.bin` 扩展名、裸 float32 或第三方厂商私有容器当成
> Keysight Infiniium AG10。

## 给 Windows Agent 的硬性指令

本文件是 ResponseLab Windows 打包任务的权威入口，真正执行构建的是项目根目录的
`build_window.bat`。`.md` 负责定义步骤、证据和停止条件，`.bat` 负责执行可自动化的
源码检查、PyInstaller 构建和打包后入口检查；两者不能互相替代。

收到打包任务后，Agent 必须先通读本文件和当前代码，再按以下规则执行：

1. 向任务发起者取得一个**完整 Git commit OID**，只从该提交构建。没有精确提交时停止，
   不得自行选择 `main`、最新分支或本地工作区；
2. 使用干净的 Windows x64 检出目录。`git status --porcelain` 必须为空，不能把 macOS
   `.venv`、旧 `dist`、旧 `build` 或旧 `.spec` 文件带入候选包；
3. 发布候选只能由 `build_window.bat` 产生。不得跳过失败步骤、直接拼接另一套 PyInstaller
   参数、删测试、加 skip、放宽容差，或用“EXE 文件存在”代替功能验证；
4. 自动化门禁通过后，继续完成第 4.4 节的干净机器真实工作流。任何适用项为 `FAIL`、
   `INCONCLUSIVE` 或 `NOT_RUN` 时，只能报告对应状态，不能写“完美打包”“没有功能错误”
   或“可正式发布”；
5. 只验收一次构建得到的同一份字节。签名、重新压缩、替换 DLL 或重新构建都会改变交付
   身份，必须重新记录哈希并重跑适用验收；
6. 交付物是完整 `ResponseLab` onedir 的 ZIP，不是单独的 `ResponseLab.exe`。

若构建或验收失败，Agent 可以诊断和修复代码，但任何修改都必须形成新 commit，并从本
文件第一步重新开始。不得在已构建的 `dist\ResponseLab` 中手工替换文件后继续沿用旧证据。

## 0. 当前目标

在 Windows 10/11 x64 上生成 `dist\ResponseLab\ResponseLab.exe`，并验证：

1. Keysight `WaveformXYValues` v1/v2、官方 Python 两列表头和无表头 CSV 在拟合脉冲、
   目标信号与影响频段路径保持可用，并从时间列自动取得采样率；
2. 受支持的 Keysight Infiniium AG10 时域 waveform 自动恢复采样率、时间原点和电压；
3. 导出的 AG10 BIN 能由本工具重新导入，并与 CSV 路径得到一致的补偿结果；
4. 影响频段 Vpp 的 PRBS13Q/外部码型、LFP 和频域 RMS 误差在 EXE 中可运行；
5. 不受支持或过大的 BIN 在读取 payload 或启动大型 FFT 前明确拒绝；
6. 影响频段眼高、眼宽和 Vpp 对两份拟合脉冲的采样率差异采用同一 `1000 ppm` 闭区间
   合同：500 ppm 与恰好 1000 ppm 可运行，2000 ppm 拒绝；门限内不重采样，并保留提示；
7. 影响频段有限记录边界、最大补偿增益和边缘过渡设置在 EXE 中与源码合同一致。

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
GUI 的通用无表头与自描述格式都完整验证恰好两列及各行列数，只有底层程序化 API 未启用
严格列数合同时才可用 `usecols` 显式选择宽表列。BIN 在
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

- Windows 10/11 x64；发布矩阵使用 Windows x64 Python 3.11 和 3.13，内部试用优先
  Python 3.11。`build_window.bat` 会拒绝低于 3.11、x86 和 ARM64，但这不代表所有未来
  Python 版本都已获支持；
- 在干净目录检出已确认的完整交付 commit；不要复制 macOS `.venv`；
- 构建机器能安装 `pyproject.toml` 中声明的依赖。

```powershell
git clone https://github.com/ZhenlongYou/response-lab.git
cd response-lab
git checkout --detach <完整交付 commit OID>
git status --porcelain
git rev-parse HEAD
.\build_window.bat
```

`git rev-parse HEAD` 必须与任务指定 OID 完全相同，`git status --porcelain` 必须没有输出。
若本次发布要求包含 `1000 ppm` 新合同，还要在构建前确认当前提交中的
`src/response_lab/influence_policy.py` 定义
`PULSE_SAMPLE_RATE_TOLERANCE_PPM = 1000.0`，并且第 4.0 节列出的边界测试实际执行。

脚本建立 Windows `.venv`，运行测试、Ruff、compileall、源码 self-test、源码 GUI smoke
test，再以 PyInstaller `--onedir` 生成 EXE。打包完成后还会直接运行
`ResponseLab.exe --self-test` 和 `ResponseLab.exe --gui-smoke-test`；任一失败都会终止构建。
交付时必须保留整个 `dist\ResponseLab\`，不能只复制 `ResponseLab.exe`。

### 3.2 GitHub Actions 原生 Windows 构建

仓库的 `.github/workflows/responselab-windows.yml` 会在 `windows-latest` 上分别把 x64
Python 3.11 与 3.13 放入 PATH，再从没有项目 `.venv` 的检出目录调用同一个
`build_window.bat`，由脚本创建 venv，并把完整 `dist\ResponseLab\` 上传为 Actions
artifact；另一个短作业确认已有 x86 venv 会在安装依赖前被拒绝。该 workflow 支持手动
触发，并在 ResponseLab 源码或自身配置发生 push / pull request 变化时运行。

CI 绿色只能证明 Windows runner 上的源码门禁、PyInstaller onedir 构建，以及打包后 EXE
的算法自检和 offscreen GUI smoke test 完成，不能替代 4.4 节的干净机器可见窗口启动、
真实 Keysight 文件与导出重读验收。下载 artifact 后仍必须交付整个 `ResponseLab` 目录。

下面的命令只用于诊断 `build_window.bat` 内部的 PyInstaller 阶段，不能作为发布入口。
手工运行所得目录必须丢弃，发布候选仍要从干净检出重新运行完整批处理：

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

### 4.0 本轮关键合同

除完整测试外，Agent 必须在构建日志中确认以下回归被收集且通过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_attribution.py::test_eye_attribution_accepts_exact_1000_ppm_cross_pulse_difference `
  tests/test_vpp_analysis.py::test_vpp_window_validation_accepts_exact_1000_ppm_at_non_round_rate `
  tests/test_influence_workflow.py::test_exact_1000_ppm_pulses_run_from_real_influence_button `
  tests/test_influence_workflow.py::test_zero_boundary_repair_runs_from_real_influence_button `
  tests/test_influence_workflow.py::test_gain_limit_repair_runs_from_real_influence_button
```

该命令预期产生 6 个通过案例，其中真实按钮测试的 1000 ppm 用例按眼图和 Vpp 参数化为
两项。它是发布前的重点复核，不能替代 `build_window.bat` 已执行的完整测试。

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
| Vpp | 可见项为“PRBS13Q”（固定 8191-symbol Gray 映射周期）；外部单列码型在解析前受 32 MiB 上限保护；LFP 显示 V，频域误差显示 Vrms。 |

### 4.4 干净机器验收

先按第 5 节从本次唯一的 `dist\ResponseLab` 创建候选 ZIP，再把**同一个 ZIP**复制到一台
没有源码、没有开发 venv 的 Windows 机器上。示例输入不会打入产品 onedir；Agent 还要
从同一源码 commit 单独复制 `examples\ResponseLab_角色清晰导入示例\` 为“验收输入”，
记录这些输入的 SHA-256，但不得把它们追加进已哈希的候选 ZIP：

1. 把候选 ZIP 解压到包含空格和中文的全新目录，例如
   `C:\Users\Public\ResponseLab 验收\`，从该目录启动 `ResponseLab.exe`，确认图标、六个
   页签和绘图组件可见；
2. 使用单独“验收输入”目录中的参考拟合脉冲、DUT 拟合脉冲、目标 CSV、目标 AG10 BIN
   和 PRBS13Q 外部码型完成可见操作；若任务另提供真实/独立
   AG10 文件，也必须纳入且单独记录来源与 SHA-256；
3. 完成数据补偿并导出 BIN，用 EXE 再导入，检查 Fs、时间原点和波形；
4. 用同一信号的 CSV 对照补偿结果；
5. 在“影响频段”页分别运行眼高或眼宽、Vpp LFP 与 Vpp 频域 RMS，检查码型、M、pmax
   前后窗口、单位、最大增益和边缘过渡；
6. 导入一个随机裸 BIN 和一个已知不支持变体，确认错误信息明确；
7. 关闭网络后退出并重新启动，重复读取示例和一次补偿，确认 EXE 不依赖源码、构建机
   路径、开发 venv 或网络；
8. 检查 SmartScreen、杀毒软件和公司应用白名单。被安全策略阻止时交给 IT 处理签名或
   白名单，不能通过关闭终端安全策略把阻断伪装成通过。

只有这些闭环通过，才能交付“支持 Keysight Infiniium AG10 子集的 Windows EXE”。没有
真实 Infiniium 或独立格式夹具证据时，应写“按官方格式实现并通过独立合成夹具”，不能
扩大成“兼容所有 Keysight BIN”。

## 5. 候选包、哈希与交付证据

自动化通过后立即从本次唯一的 `dist\ResponseLab` 创建运输包；干净机器验收和后续上传
必须消费这个 ZIP，验收后不得重新压缩。`<版本>` 和 `<短提交>` 由本次任务确定：

```powershell
$Commit = git rev-parse HEAD
$ShortCommit = $Commit.Substring(0, 12)
$Version = "<版本>"
$Zip = "ResponseLab-Windows-x64-$Version-$ShortCommit.zip"

if (Test-Path -LiteralPath $Zip) { throw "候选 ZIP 已存在，禁止覆盖" }
Compress-Archive -LiteralPath .\dist\ResponseLab -DestinationPath $Zip
Get-FileHash -Algorithm SHA256 -LiteralPath $Zip
(Get-ChildItem -LiteralPath .\dist\ResponseLab -Recurse -File | Measure-Object).Count
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze
```

Agent 的最终交接必须给出：

| 字段 | 必须记录的内容 |
| --- | --- |
| 源码身份 | 完整 commit OID、分支或 tag、`git status --porcelain` 为空 |
| 构建环境 | Windows 版本、x64、Python 完整版本、`pip freeze` |
| 自动化结果 | `build_window.bat` 退出码、完整测试数量与跳过项、源码和 EXE 两类 self/GUI smoke |
| 真实工作流 | 第 4.4 节逐项 `PASS/FAIL/INCONCLUSIVE/NOT_RUN`，输入文件来源与哈希 |
| 交付身份 | ZIP 文件名、字节数、SHA-256、onedir 文件数 |
| 安全状态 | 是否签名、SmartScreen/杀软/白名单结果；未执行必须写 `NOT_RUN` |
| 未闭环项 | 所有失败、跳过、平台缺口和人工检查缺口，不得省略 |

ZIP 生成后不得再修改 `dist\ResponseLab` 并声称 ZIP 仍代表已验收目录。上传、复制或发布后，
必须重新计算接收端 ZIP 的 SHA-256，并与上表完全一致。只有全部适用项为 `PASS`，才能称
“该精确 Windows x64 候选包已完成所列范围验收”；这仍不等于兼容所有 Windows 环境、
所有 Keysight BIN 或不存在未知缺陷。

## 6. 未来第三方 BIN adapter

若后续仍需接入朗视或其他厂商格式，应新增独立 adapter，并至少取得 reader、writer、
采样率/单位元数据、脱敏样例、许可与可再分发依赖。没有 writer 时只允许导出 CSV；不能
把 ResponseLab AG10 writer 的输出标成第三方格式。第三方格式选择必须显式出现在 UI，
不能只按 `.bin` 后缀路由。

## 7. 官方依据

- [Keysight Waveform BIN files](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_BIN_files.htm)
- [Keysight BIN File Format](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/BIN_File_Format.htm)
- [Keysight 官方 Python 转 CSV 示例](https://helpfiles.keysight.com/csg/d9300a/Help/examples/XR8/binary-to-csv.py)
- [Keysight Waveform XY CSV](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Files/waveform_xy_files.htm)
- [Keysight Waveform HEADer](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-PG/Content/Topics/Commands/DISK/WAVeform_HEADer.htm)
- [Keysight Python 浮点波形示例](https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-PG/Content/Topics/Python/Scripts/waveform-data-float-format.htm)

## 8. 维护规则

后续 Agent 开始 Windows 打包时，先读本文件，再核对 `build_window.bat`、
`.github/workflows/responselab-windows.yml`、`pyproject.toml` 和 `handoff.md`。以下任一项变化
都必须同步更新本合同：Python/Windows 支持矩阵、PyInstaller 参数、资源目录、EXE 名称、
测试节点、示例输入、Keysight 格式边界、影响频段合同、ZIP 命名或干净机器验收项目。

修改本文件后至少执行一次所列测试节点的 `pytest --collect-only`，确认路径和测试名没有
失效；涉及构建脚本或程序代码时，还要在 Windows x64 上重新运行完整 `build_window.bat`。
不要把某一次 release 的 commit、ZIP 哈希或测试数量写成永久事实；这些值属于每次打包的
交付证据，应由当次 Agent 按第 5 节重新生成。
