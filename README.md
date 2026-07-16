# ResponseLab

ResponseLab 是一款频响分析与补偿桌面工具。它从两份拟合脉冲计算
`参考响应 / 待补偿响应`，在用户选择的频带内生成补偿响应，再通过
`FFT → 频域相乘 → IFFT` 把差异直接应用到 CSV 或原始 BIN 信号。界面采用深色三栏
工作台，包含拟合脉冲、频率响应、频响差异比较、频响补偿和输出预览五组交互图。

## 直接运行

首次使用先在终端完成一次项目环境初始化：

```bash
cd /Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
python3 main.py
```

以后在 PyCharm 中直接运行项目根目录的 `main.py`，或执行 `python3 main.py` 即可。
`main.py` 会自动切换到本项目 `.venv`，不会修改系统 Python 或 Conda 环境；若尚未安装
依赖，也会直接显示上述初始化命令。

## 输入约定

### 拟合脉冲 CSV

- **无表头**；第 1 列是时间，第 2 列是幅值。
- 两份脉冲分别从时间列的中位时间间隔推导采样率，允许点数和采样率不同。
- 时间列固定使用秒，非均匀时间轴会被拒绝。

示例（没有列名）：

```text
0.000000000,0.000012
0.000000001,0.014120
0.000000002,0.652300
```

### 待补偿信号 CSV

- 同样是无表头的“时间 + 幅值”两列格式。
- 采样率同样由第 1 列时间间隔推导，不需要手动输入。

### 待补偿信号 BIN

- BIN 是无文件头的原始样本流，**必须手动输入采样率**。
- 默认按 little-endian `float32`、单通道、0 字节文件头解析。
- 界面还可设置 `float64 / int16 / int32`、字节序、文件头偏移、通道数、目标通道、
  interleaved/planar 排列、幅值缩放和偏置。

## 补偿方向和相位策略

补偿方向固定为：

```text
C(f) = H_reference(f) / H_dut(f)
```

模式支持“仅幅频”“仅相频”“幅频 + 相频”。用户选择补偿频带，频带两端自动平滑
回到单位响应。
在含相位的模式中，工具会在“去斜观察频带”内拟合相位差的线性斜率并显示估计相对
时延；该线性时延从实际补偿相位中移除，因此拟合脉冲在文件中的相对位置不会平移待
补偿信号。常数相位和非线性相位差仍会保留。“仅幅频”不执行时延估计，但仍显示原始
相位差供观察。

补偿频带核心内不做最大增益、最大衰减或正则化裁剪：分析得到多少幅相差异，就应用
多少。所选频带外严格使用单位响应。若所选频带包含待补偿响应为零、无法计算响应比的
频点，工具会要求缩小或移动补偿频带。

## 导出

“导出补偿结果”会同时生成：

- `<原名>_compensated.csv` 或 `.bin`：等长补偿信号；CSV 仍然无表头，BIN 为
  little-endian `float32`。
- `<原名>_compensated_response.csv`：绘图所用的频率响应与差异诊断表。
- `<输出文件>.response-lab.json`：输入文件哈希、有效参数、估计时延、频域应用信息、输出
  统计和输出文件哈希。

待补偿信号自动做首尾镜像延拓，并按自身采样率建立 DFT 频点；补偿响应插值到这些频点
后直接相乘并反变换，再取回原记录。输出与输入保持相同的点数、时间轴和通道数。

三份源文件在导出前后都会与分析时的大小和 SHA-256 快照核对；若文件被外部修改，
必须重新分析。三个输出先完整写入同一文件系统的暂存目录，再作为一个可回滚批次提交，
避免普通写入异常留下相互不匹配的半套结果。

## 校验

```bash
python3 main.py --self-test
QT_QPA_PLATFORM=offscreen .venv/bin/python main.py --gui-smoke-test
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q main.py src tests examples
```

算法解释和可手算例子见 [docs/ALGORITHM.md](docs/ALGORITHM.md)，完整设计边界与后续增强
见 [docs/SPEC.md](docs/SPEC.md)。
