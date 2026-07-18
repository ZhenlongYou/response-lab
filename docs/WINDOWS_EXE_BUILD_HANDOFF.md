# ResponseLab：Windows EXE 与朗视 BIN 接入交接

> 给下一位 Windows 执行者。先完成“朗视 BIN 接入和验证”，再打 EXE；不要把当前通用
> 裸 BIN 路径当成朗视 BIN 支持。

## 0. 目标与禁止事项

目标：在 Windows x64 上产出 `ResponseLab.exe`。用户选中朗视 BIN 后，工具调用用户提供的
`read_longsight_bin_data(path)[0]["ch1_data"]` 读取；补偿结果可由朗视软件或同一读函数
再次正确读取。

禁止事项：

- 不能用 `numpy.fromfile`、猜测文件头偏移或把文件扩展名改成 `.bin` 来冒充朗视格式。
- 不能把当前 `save_bin_float32()` 的输出标为“朗视 BIN”。它只会写裸 little-endian float32。
- 没有得到朗视写入函数或完整格式说明前，朗视输入只能导出 CSV；不能提供“写回朗视 BIN”。
- 不要在 macOS 打包 Windows EXE；必须在 Windows 上构建和验收。

## 1. 当前源码事实

- 入口：`main.py`
- GUI：`src/response_lab/ui.py`
- 当前通用 CSV/裸 BIN I/O：`src/response_lab/io.py`
- 导出分发：`src/response_lab/reporting.py`
- 依赖与 SVG 资源声明：`pyproject.toml`

当前 `.bin` 分支是“通用裸样本流”：按用户给出的 dtype、字节序、通道、偏移等读取；导出
`.bin` 时调用 `save_bin_float32()`。这与朗视专有文件格式不同，因此当前版本**不能读取或
写出用户的朗视 BIN**。

## 2. 开始前必须向用户拿到的内容

不要只拿到这一行调用代码。必须拿到：

```python
data = read_longsight_bin_data(path)[0]["ch1_data"]
```

以及以下资料：

1. `read_longsight_bin_data` 的完整源码、所在模块路径或可安装包；包括全部依赖、DLL、许可文件。
2. 对应的**写入函数**完整源码/API。例如能以“原文件 + 替换 ch1 数组 + 输出路径”的方式写出。
3. 一个可分享的脱敏 BIN、其采样率、通道数，以及朗视软件读回的预期 ch1 样点数和单位。
4. 返回对象中是否还含采样率、垂直刻度、单位、触发时间、其他通道、文件头/元数据。
5. 写回需求：只替换 `ch1_data`，还是保留所有通道、原文件头和全部元数据。
6. 第三方包/DLL 是否允许随 EXE 再分发；若不允许，必须改为安装说明或 CSV 导出。

若用户没有写函数，先完成“朗视 BIN 导入 + CSV 导出”，并把“朗视 BIN 写回”显式禁用。

## 3. 推荐的适配边界

新建 `src/response_lab/longsight_io.py`，不要把朗视细节塞进 GUI 或 DSP：

```python
def load_longsight_timeseries(path: str | Path, sample_rate_hz: float) -> TimeSeries:
    """用厂商 reader 读取 record[0]['ch1_data']，返回单位明确的一维 TimeSeries。"""

def save_longsight_compensated(
    source_path: str | Path,
    destination_path: str | Path,
    compensated_values: np.ndarray,
) -> Path:
    """调用厂商 writer；保留文件头、元数据和未替换通道。"""
```

要求：

- `load_*` 验证 `ch1_data` 为有限的一维数值数组、样点数不少于 8；明确数据是 V、mV 还是 ADC
  counts。补偿前后必须保持同一物理单位。
- 使用用户输入或文件元数据中的真实 `Fs`，单位为 Hz；不要把波特率当作采样率。
- `save_*` 需要处理厂商的量化/缩放规则，并保留其他通道与元数据。若 writer 要求整数码值，
  必须定义饱和、舍入和刻度，而不是直接 `astype(int16)`。
- 写出后立即用独立的 `read_longsight_bin_data` 再读一次，验证 ch1、样点数、采样率和元数据。

GUI 建议增加 BIN 格式选择：`朗视 BIN` 与 `原始 BIN（高级）`。朗视模式只显示采样率（若文件
本身没有可靠采样率）；原始 BIN 模式保留现有“高级 BIN 解析参数”。不要再仅按 `.bin` 扩展名
选择解析器。

导出也必须以“输入/目标格式”路由：

- 朗视 BIN + 有 writer：`save_longsight_compensated`。
- 朗视 BIN + 无 writer：仅允许 CSV。
- 原始 BIN：沿用明确标注的 raw float32 BIN 导出。

需修改的主要调用点：

- `src/response_lab/ui.py`：格式选择、读取请求和导出格式限制。
- `src/response_lab/influence_controller.py`：Vpp 的原始数据加载也需使用同一格式路由。
- `src/response_lab/reporting.py`：不能再只以 `.bin` 后缀选择 `save_bin_float32()`。
- `src/response_lab/io.py`：保留为通用裸 BIN，不要把朗视 reader 混入其中。

## 4. 朗视接入的最低测试

新增测试时可把厂商 reader/writer 封在适配模块边界，以 fake adapter 测路由；真实脱敏样例则
作为手工验收，不要把保密原始数据提交到仓库。

| 验收项 | 必须证明的结果 |
| --- | --- |
| 读入 | 朗视 reader 返回的 `ch1_data` 与工具输入数组一致；样点数、单位和 Fs 正确。 |
| 身份补偿 | `H_ref = H_dut` 时输出与输入一致（容差须由浮点/量化格式说明）。 |
| 写回 | 用朗视 reader 重读输出；补偿 ch1 正确，其他通道和关键元数据保持。 |
| CSV 回退 | 无朗视 writer 时，GUI 不提供朗视 BIN 导出，只能导出 CSV。 |
| 错误 | 缺 reader、缺 writer、缺 ch1、NaN/Inf、长度不符、未知 Fs 都给出明确报错。 |
| 影响频段 | Vpp 读取朗视原始波形时与主补偿路径使用相同的朗视 adapter。 |

完成接入后，至少运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q main.py src tests examples
.\.venv\Scripts\python.exe main.py --self-test
.\.venv\Scripts\python.exe main.py --gui-smoke-test
```

最后必须用用户给的脱敏朗视文件，在 GUI 完成一次真实导入、补偿、写回、厂商 reader 重读的
手工闭环；纯 mock 测试不足以证明格式兼容。

## 5. Windows x64 打包步骤

### 环境

- Windows 10/11 x64；建议 Python 3.11 x64（与 `pyproject.toml` 的 `>=3.11` 要求一致）。
- 在干净目录克隆本仓库的目标分支；不要把 macOS `.venv` 复制到 Windows。
- 使用 Windows 虚拟环境；PyInstaller 只在 Windows 构建 Windows EXE。

```powershell
git clone https://github.com/ZhenlongYou/codex.git
cd codex\codex_projects\frequency_response_compensator
git checkout codex/serdes-workspace-architecture
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install "pyinstaller>=6.0"
```

先完成第 3、4 节的朗视 adapter 接入和验证，再执行以下命令。优先用 `--onedir`；PySide6、SciPy
和厂商 DLL 在单文件解压模式下更难排错。

仓库根目录的 `build_window.bat` 已把环境创建、依赖安装、测试和以下打包命令固定下来。下一位
agent 应先检查并填好该文件中的 `PYINSTALLER_VENDOR_ARGS`，再双击或在 cmd 中执行：

```bat
build_window.bat
```

该变量必须按实际朗视 reader/writer 的包、DLL 与许可证填写；空值只适用于未依赖厂商二进制的
构建，不能据此声称朗视支持可用。

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

若朗视 reader 依赖包、DLL 或数据文件，不能猜测；按其真实安装位置补充：

```powershell
# 示例：实际名称和路径必须按用户提供的 reader 决定。
# --collect-all longsight_reader
# --add-binary "C:\path\to\vendor.dll;."
# --add-data "C:\path\to\license.dat;."
```

交付目录是 `dist\ResponseLab\`，交付时保留整个目录，不要只复制 `ResponseLab.exe`。

## 6. EXE 的最终验收

在一台没有源码、没有开发 venv 的干净 Windows 机器上：

1. 启动 `dist\ResponseLab\ResponseLab.exe`，确认界面、SVG 图标和所有页签可见。
2. 选择两份拟合脉冲和脱敏朗视 BIN；确认朗视模式只要求 Fs，且 ch1 波形合理。
3. 完成一次补偿；用朗视 reader/软件重读输出，检查 ch1、其他通道、元数据与量化行为。
4. 检查 CSV 导出、影响频段 Vpp、眼高、眼宽和导出 manifest。
5. 临时断开网络再重复启动；EXE 不应依赖构建机网络或源码路径。

只有以上五项通过，才能交付“支持朗视 BIN 的 Windows EXE”。若 writer 或再读验证缺失，只能
交付“支持朗视 BIN 导入、CSV 导出的 Windows EXE”。

## 7. 建议下一位 agent 使用的技能

- `python-gui-venv`：Windows Qt/PySide6 环境和启动问题。
- `test-effectiveness-gate`：厂商格式读写的真实样例与读回验证。
- `github-code-handoff`：只提交朗视 adapter、测试、打包说明；不要混入其他未提交文件。
