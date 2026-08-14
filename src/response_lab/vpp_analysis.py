"""基于周期稳态码型的 LFP Vpp 与频域 RMS 误差分析。"""

# 延迟解析类型标注，便于公开数据类引用自身而不引入运行时分支。
from __future__ import annotations

# fstat 把外部码型的体积检查和文本解析绑定到同一个已打开文件。
import os

# Callable 允许控制器把周期 FFT 的工作量门禁插入同一文件描述符的解析前阶段。
from collections.abc import Callable, Iterator

# 冻结数据类让同一份分析设置可安全复用于多个补偿候选。
from dataclasses import dataclass

# Path 明确外部码型来源，避免数值核心依赖当前工作目录字符串。
from io import StringIO
from pathlib import Path

# Literal 把界面允许的分析模式和文件值类型收窄为冻结合同。
from typing import Literal, TextIO

# NumPy 用于可复现地生成 PRBS13Q 位流并保存紧凑的 symbol code。
import numpy as np

# 类型别名明确公开数组的元素类型，方便调用层区分码型与电压。
from numpy.typing import NDArray

# SciPy FFT 提供生产级实数圆周卷积，并允许测试精确监视 IFFT 次数。
from scipy import fft as scipy_fft

# Cooperative cancellation is shared with CSV/BIN loaders and GUI workers.
from response_lab.cancellation import CancellationCheck, raise_if_cancelled

# 复用领域模型已验证的等间隔时间轴、采样率和通道形状约束。
from response_lab.memory_budget import current_memory_budget
from response_lab.models import TimeSeries

# PAM4 symbol code 固定使用 0..3 的无符号字节表示。
UInt8Array = NDArray[np.uint8]
# 卷积输入和指标统一使用 float64，避免码型整数运算发生溢出。
FloatArray = NDArray[np.float64]
# 频域周期模型统一使用 complex128，保留补偿相位信息。
ComplexArray = NDArray[np.complex128]
# 窗口 lag 使用平台无关的 64 位整数，支持长记录和长码型。
IntArray = NDArray[np.int64]

# 外部理想码型是控制记录，限制文本体积可在 np.loadtxt 分配前拦截误选的大采集文件。
_MAX_PATTERN_FILE_BYTES = 32 * 1024 * 1024
# 最短合法文本约为每行一个数字和换行；解析表、取整/unique、映射和最终只读副本
# 的峰值实测路径可能同时保留多份 float64 数组；32 MiB 最短行冻结快照实测约
# 838 MiB，因此按源文本 32 倍并加固定开销，保留约 26% 的峰值余量。
_PATTERN_TEXT_MEMORY_MULTIPLIER = 32
_PATTERN_LOADER_FIXED_OVERHEAD_BYTES = 32 * 1024 * 1024


# 冻结 UI 到数值核心之间的全部 Vpp 分析参数，禁止运行中漂移。
@dataclass(frozen=True)
class VppAnalysisSettings:
    """LFP 或频域 RMS Vpp 分析所需的完整显式设置。"""

    # 指标方法在时域 exact Vpp 与频域 AC RMS 误差之间二选一。
    method: Literal["lfp", "frequency_rms_error"]
    # 码型来源显式区分内置标准资产和用户文件。
    pattern_source: Literal["builtin_prbs13q_gray", "file"]
    # 每 UI 样点数负责把游标 UI 精确换算为离散样点。
    samples_per_ui: int
    # pmax 之前保留的完整 UI 数量。
    pre_cursor_ui: int
    # pmax 之后保留的完整 UI 数量。
    post_cursor_ui: int
    # 外部模式保存单列码型路径，内置模式固定为空。
    pattern_path: Path | None
    # 文件数值语义由用户声明；amplitude_values 是无量纲符号系数，禁止自动猜测。
    file_value_kind: Literal["symbol_codes", "amplitude_values"]

    # 创建时拒绝含糊或不能精确换算为整数样点的设置。
    def __post_init__(self) -> None:
        """验证枚举、UI 整数和码型路径之间的一致性。"""

        # 两种方法分别对应时域 exact Vpp 和频域 AC 误差。
        if self.method not in {"lfp", "frequency_rms_error"}:
            # 未知方法不能静默回退到任一种数值定义。
            raise ValueError("method 必须是 lfp 或 frequency_rms_error")
        # 码型来源必须显式选择内置标准周期或单列外部文件。
        if self.pattern_source not in {"builtin_prbs13q_gray", "file"}:
            # 拒绝自动猜测可避免同一文件在不同入口得到不同结果。
            raise ValueError("pattern_source 必须是 builtin_prbs13q_gray 或 file")
        # 文件值含义必须由用户声明，不能根据数值范围自动猜测。
        if self.file_value_kind not in {"symbol_codes", "amplitude_values"}:
            # 限定为两种冻结语义，便于载入器做 fail-closed 验证。
            raise ValueError("file_value_kind 必须是 symbol_codes 或 amplitude_values")
        # 三个 UI 参数都必须是真正整数，布尔值不能冒充 0 或 1。
        integer_fields = {
            "samples_per_ui": self.samples_per_ui,
            "pre_cursor_ui": self.pre_cursor_ui,
            "post_cursor_ui": self.post_cursor_ui,
        }
        # 逐字段检查可在异常中指出具体的错误输入来源。
        for field_name, field_value in integer_fields.items():
            # NumPy 整数可由界面控件传入，但布尔类型必须拒绝。
            if isinstance(field_value, (bool, np.bool_)) or not isinstance(
                field_value, (int, np.integer)
            ):
                # 非整数 UI 无法无损换算为拟合脉冲窗口样点。
                raise ValueError(f"{field_name} 必须是整数")
            # 统一保存为 Python int，避免序列化时携带 NumPy 标量。
            object.__setattr__(self, field_name, int(field_value))
        # 每 UI 至少一个样点，否则无法构造离散码型激励。
        if self.samples_per_ui < 1:
            # 零或负采样密度没有物理意义。
            raise ValueError("samples_per_ui 必须至少为 1")
        # 前后游标长度允许为零，但不能反向截取。
        if self.pre_cursor_ui < 0 or self.post_cursor_ui < 0:
            # 负 UI 会破坏围绕 pmax 的包含式窗口合同。
            raise ValueError("pre_cursor_ui 和 post_cursor_ui 不能为负数")
        # 外部模式必须给出路径，内置模式则拒绝悄悄忽略多余路径。
        if self.pattern_source == "file" and self.pattern_path is None:
            # 缺少路径时无法加载用户选择的理想码型。
            raise ValueError("file 码型来源必须提供 pattern_path")
        # 内置码型不能同时携带外部路径，避免界面状态与实际来源不一致。
        if self.pattern_source == "builtin_prbs13q_gray" and self.pattern_path is not None:
            # fail-closed 比静默忽略路径更容易发现上层连线错误。
            raise ValueError("内置码型来源的 pattern_path 必须为 None")
        # 把文件路径规范化为 Path，供载入阶段稳定处理。
        if self.pattern_path is not None:
            # 冻结数据类通过 object.__setattr__ 完成一次性规范化。
            object.__setattr__(self, "pattern_path", Path(self.pattern_path))


# 一份模型同时保留脉冲窗口证据和可复用的周期稳态频谱。
@dataclass(frozen=True)
class VppPeriodicModel:
    """参考或 DUT 的窗口脉冲及其周期稳态输出模型。"""

    # 首次绝对峰值在完整脉冲记录中的样点索引。
    peak_index: int
    # 拟合窗口在完整脉冲中的包含式起点。
    window_start: int
    # 拟合窗口在完整脉冲中的排除式终点。
    window_stop: int
    # 每个拟合 tap 相对于 pmax 的整数样点 lag。
    lag_samples: IntArray
    # 窗口内供展示和卷积的参考或 DUT 拟合脉冲。
    pulse_window_v: FloatArray
    # 把窗口 lag 折叠到一个码型周期后的圆周卷积核。
    circular_kernel_v: FloatArray
    # 理想码型与窗口脉冲卷积后的单边复频谱。
    spectrum_v: ComplexArray
    # 完整周期的稳态时域波形，供 LFP exact max-min 使用。
    waveform_v: FloatArray


# 准备结果缓存码型 FFT 和两份模型，供多个补偿候选重复测量。
@dataclass(frozen=True)
class VppAnalysisCache:
    """一次码型/脉冲准备后可复用的 Vpp 分析缓存。"""

    # 保存本缓存适用的完整冻结设置。
    settings: VppAnalysisSettings
    # 参考与 DUT 共享的真实采样率，单位 Hz。
    sample_rate_hz: float
    # 由 Fs/M 推导的符号率，单位 baud，必须与用户输入 M 一起展示。
    symbol_rate_hz: float
    # 一个 UI 的物理时长 M/Fs，单位秒。
    ui_duration_s: float
    # 一个完整理想码型周期所含的离散样点数。
    period_samples: int
    # 与所有 rFFT 数组逐点对齐的非负频率轴，单位 Hz。
    frequency_hz: FloatArray
    # 已完成显式 code 映射或幅度载入的理想 symbol 周期。
    pattern_levels: FloatArray
    # UI 起点插零激励的可复用单边频谱。
    excitation_spectrum: ComplexArray
    # 参考拟合脉冲及其周期稳态模型。
    reference_model: VppPeriodicModel
    # DUT 拟合脉冲及其周期稳态模型。
    dut_model: VppPeriodicModel
    # 参考基线的所选指标，单位 V。
    reference_metric_v: float
    # DUT 基线的所选指标，单位 V。
    dut_metric_v: float


# 候选结果同时提供指标和补偿后模型，便于上层完整填充结果视图。
@dataclass(frozen=True)
class VppCandidateMeasurement:
    """一次 DUT 频谱补偿候选的 Vpp 测量结果。"""

    # 候选的 LFP Vpp 或频域 AC 误差 Vrms，单位 V。
    value_v: float
    # DUT 模型频谱逐点乘候选 correction 后的只读复频谱。
    corrected_spectrum_v: ComplexArray
    # LFP 返回完整周期波形；RMS 用 None 表示没有执行 IFFT。
    waveform_v: FloatArray | None


# 内部数组均复制并设为只读，避免冻结数据类中仍残留可变 NumPy 缓存。
def _readonly_array(values: object, dtype: np.dtype | type) -> np.ndarray:
    """复制为指定 dtype 的一维有限只读数组。"""

    # 独立副本切断对调用方输入和 FFT 工作区的可写别名。
    array = np.array(values, dtype=dtype, copy=True)
    # 周期模型的所有公开数组都必须是一维，便于频点和样点严格对齐。
    if array.ndim != 1:
        # 拒绝隐式展平可暴露错误的通道或矩阵输入。
        raise ValueError("Vpp 分析数组必须是一维")
    # 实数和复数都要求全部有限，防止 NaN/Inf 污染指标。
    if not np.all(np.isfinite(array)):
        # 在缓存创建处 fail-closed，比候选扫描后才出现 NaN 更可诊断。
        raise ValueError("Vpp 分析数组必须全部为有限值")
    # 关闭写权限，使缓存能够安全跨候选复用。
    array.setflags(write=False)
    # 返回拥有独立内存的只读一维数组。
    return array


# 独立发生器把 IEEE PRBS13Q 的种子、抽头、位序与 Gray 映射冻结在一个入口。
def generate_prbs13q_gray_symbols() -> UInt8Array:
    """返回 IEEE PRBS13Q Gray 映射的一个完整 8191-symbol 周期。"""

    # IEEE 公布的 S0..S12 初始状态是 0000010101011。
    state = np.array([int(bit) for bit in "0000010101011"], dtype=np.uint8)
    # 每个 PAM4 symbol 消耗两个连续 PRBS 位，因此生成两个完整二进制周期。
    bits = np.empty(2 * 8191, dtype=np.uint8)
    # 严格按 S0 xor S1 xor S11 xor S12 生成新位，再向高编号寄存器移位。
    for bit_index in range(bits.size):
        # 首先计算反馈位，避免原地移位覆盖仍需参与异或的寄存器。
        new_bit = state[0] ^ state[1] ^ state[11] ^ state[12]
        # 本拍输出定义为反馈后进入 S0 的新位，与 IEEE 示例前缀一致。
        bits[bit_index] = new_bit
        # 从尾端向前复制可保持所有旧状态，随后再写入新的 S0。
        state[1:] = state[:-1]
        # 新反馈位进入 S0，完成一次 LFSR 更新。
        state[0] = new_bit
    # 两位索引 00、01、10、11 分别映射到 Gray symbol 0、1、3、2。
    gray_lookup = np.array([0, 1, 3, 2], dtype=np.uint8)
    # 偶数位为 MSB、奇数位为 LSB，组成 0..3 的查表索引。
    pair_indices = 2 * bits[0::2] + bits[1::2]
    # 复制查表结果，防止返回值与局部查找表共享可写内存。
    symbols = np.array(gray_lookup[pair_indices], dtype=np.uint8, copy=True)
    # 内置标准资产只读，避免调用层无意修改后污染同一次分析。
    symbols.setflags(write=False)
    # 返回完整的 8191-symbol 周期。
    return symbols


def _estimate_pattern_loader_peak_bytes(file_size_bytes: int) -> int:
    """保守估算文本解析、语义校验和只读副本同时存在时的峰值内存。"""

    return int(
        _PATTERN_LOADER_FIXED_OVERHEAD_BYTES
        + int(file_size_bytes) * _PATTERN_TEXT_MEMORY_MULTIPLIER
    )


def _preflight_pattern_loader_memory(file_size_bytes: int) -> None:
    """在扫描文本行或调用 NumPy 解析器之前应用共享动态内存预算。"""

    estimate_bytes = _estimate_pattern_loader_peak_bytes(file_size_bytes)
    memory_budget = current_memory_budget()
    if estimate_bytes <= memory_budget.budget_bytes:
        return
    available_text = (
        f"系统当前可用约 {memory_budget.available_bytes / (1024.0**2):.0f} MiB"
        if memory_budget.available_bytes is not None
        else "系统可用内存不可探测，使用 768 MiB 回退预算"
    )
    raise MemoryError(
        "外部理想码型动态内存预检拒绝：预计解析峰值约 "
        f"{estimate_bytes / (1024.0**2):.0f} MiB，安全预算约 "
        f"{memory_budget.budget_bytes / (1024.0**2):.0f} MiB（{available_text}）；"
        "已在 NumPy 文本解析前停止"
    )


def _count_pattern_data_rows(
    pattern_stream: TextIO,
    *,
    cancelled: CancellationCheck | None = None,
) -> int:
    """在同一文本描述符上统计 np.loadtxt 会读取的非空、非注释数据行。"""

    # TextIOWrapper 提供 seek/迭代；保持辅助函数局部可避免公开额外文件接口。
    pattern_stream.seek(0)
    data_rows = 0
    for line_number, line in enumerate(pattern_stream):
        if line_number % 1024 == 0:
            raise_if_cancelled(cancelled, message="影响频段分析已取消")
        stripped = line.lstrip("\ufeff").strip()
        if stripped and not stripped.startswith("#"):
            data_rows += 1
    pattern_stream.seek(0)
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    return data_rows


class _CancellablePatternLines:
    """Yield frozen pattern text while polling a GUI cancellation callback."""

    def __init__(self, stream: StringIO, cancelled: CancellationCheck | None) -> None:
        self._stream = stream
        self._cancelled = cancelled

    def __iter__(self) -> Iterator[str]:
        self._stream.seek(0)
        for line_number, line in enumerate(self._stream):
            if line_number % 1024 == 0:
                raise_if_cancelled(
                    self._cancelled,
                    message="影响频段分析已取消",
                )
            yield line
        raise_if_cancelled(self._cancelled, message="影响频段分析已取消")

    def getvalue(self) -> str:
        """Expose the frozen snapshot for compatible file-like inspection."""

        return self._stream.getvalue()


# 统一码型入口始终返回可直接参与卷积的归一化浮点电平。
def load_pattern_levels(
    settings: VppAnalysisSettings,
    *,
    symbol_count_preflight: Callable[[int], None] | None = None,
    cancelled: CancellationCheck | None = None,
) -> FloatArray:
    """按显式来源和数值类型载入一个周期的理想 PAM4 电平。"""

    # 内置来源使用冻结的 IEEE PRBS13Q Gray 资产。
    if settings.pattern_source == "builtin_prbs13q_gray":
        raise_if_cancelled(cancelled, message="影响频段分析已取消")
        # 内置周期长度固定，控制器仍应在生成数组前应用同一周期 FFT 门禁。
        if symbol_count_preflight is not None:
            symbol_count_preflight(8191)
        # 转为 float64 后再运算，避免 uint8 的减法下溢。
        symbol_codes = generate_prbs13q_gray_symbols().astype(np.float64)
        raise_if_cancelled(cancelled, message="影响频段分析已取消")
        # 线性映射 0、1、2、3 到 -1、-1/3、1/3、1。
        levels = (2.0 * symbol_codes - 3.0) / 3.0
    # 外部来源进入严格的单列文件解析和显式值语义分支。
    else:
        # 设置验证已保证文件来源具有非空 Path，这里保留断言帮助静态收窄类型。
        assert settings.pattern_path is not None
        # 同一已打开文件完成体积检查和解析，路径替换不能把门限与内容拆开。
        try:
            pattern_file = settings.pattern_path.open("rb")
        except OSError as error:
            raise ValueError(f"无法读取外部码型文件：{error}") from error
        with pattern_file:
            before = os.fstat(pattern_file.fileno())
            if before.st_size > _MAX_PATTERN_FILE_BYTES:
                raise ValueError(
                    "外部码型文件超过 32 MiB 安全上限；"
                    "请提供一列单周期 symbol，而不是采集波形"
                )
            # 文件大小与预算检查绑定当前描述符，低内存下不先扫描长行或申请解析表。
            _preflight_pattern_loader_memory(before.st_size)
            # 只读取初始 fstat 大小再多一个字节；并发追加不能扩大 NumPy 的解析输入。
            frozen_bytes = bytearray()
            remaining = before.st_size + 1
            while remaining > 0:
                raise_if_cancelled(cancelled, message="影响频段分析已取消")
                block = pattern_file.read(min(1024 * 1024, remaining))
                if not block:
                    break
                frozen_bytes.extend(block)
                remaining -= len(block)
            raise_if_cancelled(cancelled, message="影响频段分析已取消")
            after_snapshot = os.fstat(pattern_file.fileno())
            if len(frozen_bytes) != before.st_size or (
                before.st_dev != after_snapshot.st_dev
                or before.st_ino != after_snapshot.st_ino
                or before.st_size != after_snapshot.st_size
                or before.st_mtime_ns != after_snapshot.st_mtime_ns
                or before.st_ctime_ns != after_snapshot.st_ctime_ns
            ):
                raise ValueError("外部码型文件在加载期间发生变化，请重新选择")
            try:
                frozen_text = frozen_bytes.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ValueError(f"无法读取外部码型文件：{error}") from error
            pattern_stream = StringIO(frozen_text)
            # 有效 symbol 行数决定周期 FFT 长度；先交给控制器做完整工作量门禁。
            expected_symbol_count = _count_pattern_data_rows(
                pattern_stream,
                cancelled=cancelled,
            )
            if expected_symbol_count < 1:
                raise ValueError("外部码型文件至少需要一行 symbol 数据")
            if symbol_count_preflight is not None:
                symbol_count_preflight(expected_symbol_count)
            # 逗号分隔解析同时支持只有换行的单列 CSV，并固定至少二维以识别横向多列。
            try:
                # float64 先保存原值，随后才按用户声明检查 code 或无量纲系数语义。
                file_values_2d = np.loadtxt(
                    _CancellablePatternLines(pattern_stream, cancelled),
                    dtype=np.float64,
                    delimiter=",",
                    ndmin=2,
                )
            # 文本格式错误或数值转换失败统一转为领域错误。
            except (OSError, ValueError) as error:
                # 对上层统一报告领域错误，同时保留原异常供日志追踪。
                raise ValueError(f"无法读取外部码型文件：{error}") from error
            raise_if_cancelled(cancelled, message="影响频段分析已取消")
            after = os.fstat(pattern_file.fileno())
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns
            ):
                raise ValueError("外部码型文件在加载期间发生变化，请重新选择")
            try:
                path_after = settings.pattern_path.stat()
            except OSError as error:
                raise ValueError("外部码型文件在加载期间发生变化，请重新选择") from error
            if before.st_dev != path_after.st_dev or before.st_ino != path_after.st_ino:
                raise ValueError("外部码型路径在加载期间被替换，请重新选择")
        # 冻结合同要求一行一个 symbol，因此一行多列或空文件都不能展平容忍。
        if file_values_2d.shape[0] < 1 or file_values_2d.shape[1] != 1:
            # fail-closed 可防止把横向 CSV 的列含义误当成连续码型。
            raise ValueError("外部码型文件必须是一列且每行一个 symbol")
        if file_values_2d.shape[0] != expected_symbol_count:
            raise ValueError("外部码型数据行数在预检与解析之间不一致")
        # 取唯一数据列，保持文件中的逐行顺序。
        file_values = file_values_2d[:, 0]
        # NaN 或 Inf 不能进入 FFT 和 Vpp 指标。
        if not np.all(np.isfinite(file_values)):
            # 明确指出有限性约束，避免后续只得到不可解释的 NaN。
            raise ValueError("外部码型必须全部为有限值")
        # symbol_codes 模式只接受精确整数 0..3，绝不根据范围自动猜测。
        if settings.file_value_kind == "symbol_codes":
            # 四舍五入只用于验证整数性，非整数不能被悄悄量化。
            rounded_codes = np.rint(file_values)
            # 精确比较可拒绝 0.1 等模拟电平，同时允许文本形式 1.0。
            if not np.array_equal(file_values, rounded_codes):
                # 非整数 code 表示用户选择的 file_value_kind 与文件不一致。
                raise ValueError("symbol_codes 文件只能包含整数 0、1、2、3")
            # 所有码值都必须落在 PAM4 的四个合法 symbol 中。
            if np.any((rounded_codes < 0.0) | (rounded_codes > 3.0)):
                # 超范围值不裁剪，避免产生隐藏的数据失真。
                raise ValueError("symbol_codes 文件只能包含整数 0、1、2、3")
            # 单一重复 code 没有 transition，不能用于评估拖尾造成的 ISI。
            if np.unique(rounded_codes).size < 2:
                raise ValueError("symbol_codes 文件至少需要两个不同 symbol")
            # 使用同一线性公式把外部 code 映射到归一化 PAM4 电平。
            levels = (2.0 * rounded_codes - 3.0) / 3.0
        # 用户声明无量纲幅度系数时只验证有限性和至少两个电平。
        else:
            # 幅度系数至少需要两个不同电平，常量激励无法形成 Vpp 或辨识 ISI。
            if np.unique(file_values).size < 2:
                # 常量文件通常表示选错列或导出失败，因此直接拒绝而非返回零指标。
                raise ValueError("无量纲幅度系数文件至少需要两个不同电平")
            # 保留无量纲系数的数值尺度、偏置和顺序；电压量纲只来自拟合脉冲。
            levels = file_values
    # 复制为连续 float64，确保返回缓存不与临时数组共享可写状态。
    readonly_levels = np.array(levels, dtype=np.float64, copy=True)
    # 调用层只能读取周期码型，不能在候选扫描之间修改它。
    readonly_levels.setflags(write=False)
    # 返回可复用于参考、DUT 和所有候选的同一周期电平。
    return readonly_levels


# 脉冲公共验证集中拒绝多通道或采样率不一致输入，避免两模型频轴错配。
def _validate_pulses(reference_pulse: TimeSeries, dut_pulse: TimeSeries) -> None:
    """验证参考和 DUT 脉冲能共享同一个离散频率轴。"""

    # 公共入口只接受已经通过 TimeSeries 领域校验的时域记录。
    if not isinstance(reference_pulse, TimeSeries) or not isinstance(dut_pulse, TimeSeries):
        # 裸数组缺少采样率和时间轴，不能可靠换算 UI 或频率。
        raise TypeError("reference_pulse 和 dut_pulse 必须是 TimeSeries")
    # 当前 Vpp 模型一次只处理一个脉冲通道，禁止默默选取第一列。
    if reference_pulse.channels != 1 or dut_pulse.channels != 1:
        # 上层应先明确选择需要分析的通道。
        raise ValueError("Vpp 分析仅接受单通道脉冲")
    # 两个脉冲必须共享采样率，才能使用同一个 UI 样点数和候选频谱。
    if not np.isclose(
        reference_pulse.sample_rate_hz,
        dut_pulse.sample_rate_hz,
        rtol=1.0e-12,
        atol=0.0,
    ):
        # 不在这里重采样，避免未经用户确认改变脉冲及其峰值位置。
        raise ValueError("参考脉冲与 DUT 脉冲的采样率必须一致")


def _pulse_window_geometry(
    pulse: TimeSeries,
    pre_samples: int,
    post_samples: int,
) -> tuple[int, int, int]:
    """Return validated pmax/start/stop before any pattern-sized allocation."""

    pulse_values = pulse.values[:, 0]
    peak_index = int(np.argmax(np.abs(pulse_values)))
    if float(abs(pulse_values[peak_index])) == 0.0:
        raise ValueError("拟合脉冲的 pmax 幅度必须非零")
    window_start = peak_index - pre_samples
    window_stop = peak_index + post_samples + 1
    if window_start < 0 or window_stop > pulse_values.size:
        raise ValueError(
            "拟合脉冲窗口越界："
            f"pmax={peak_index}, 请求 [{window_start}, {window_stop}), "
            f"记录长度={pulse_values.size}"
        )
    return peak_index, window_start, window_stop


def _validated_pulse_window_geometries(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: VppAnalysisSettings,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """验证公共采样合同，并返回两份脉冲各自以 pmax 为中心的窗口几何。"""

    _validate_pulses(reference_pulse, dut_pulse)
    pre_samples = settings.pre_cursor_ui * settings.samples_per_ui
    post_samples = settings.post_cursor_ui * settings.samples_per_ui
    return (
        _pulse_window_geometry(reference_pulse, pre_samples, post_samples),
        _pulse_window_geometry(dut_pulse, pre_samples, post_samples),
    )


def validate_vpp_pulse_windows(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: VppAnalysisSettings,
) -> None:
    """在理想码型文件读取或周期数组分配前验证两份 pmax 窗口。"""

    _validated_pulse_window_geometries(reference_pulse, dut_pulse, settings)


# 单模型构建器执行 pmax 窗口、周期折叠和 rFFT 圆周卷积。
def _build_periodic_model(
    pulse: TimeSeries,
    pre_samples: int,
    post_samples: int,
    period_samples: int,
    excitation_spectrum: ComplexArray,
    window_geometry: tuple[int, int, int],
    *,
    cancelled: CancellationCheck | None = None,
) -> VppPeriodicModel:
    """从一条完整脉冲构建指定码型的周期稳态模型。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 单通道值取出为一维视图，输入 TimeSeries 本身保持只读。
    pulse_values = pulse.values[:, 0]
    # 几何已经在码型载入和 FFT 分配前验证，此处只复用冻结索引。
    peak_index, window_start, window_stop = window_geometry
    # 复制包含式窗口，作为界面展示和数值复核的拟合脉冲证据。
    pulse_window = _readonly_array(
        pulse_values[window_start:window_stop],
        np.float64,
    )
    # 每个窗口 tap 的 lag 以 pmax 为零，严格覆盖 -pre 到 +post。
    lag_samples = _readonly_array(
        np.arange(-pre_samples, post_samples + 1, dtype=np.int64),
        np.int64,
    )
    # 周期卷积核长度与理想码型插零后的一个周期完全一致。
    circular_kernel = np.zeros(period_samples, dtype=np.float64)
    # 相同模周期位置可能由超长窗口多次命中，因此必须累加而非赋值。
    np.add.at(circular_kernel, np.mod(lag_samples, period_samples), pulse_window)
    # 实数核只计算非负频率，降低时间和内存占用。
    kernel_spectrum = scipy_fft.rfft(circular_kernel)
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 周期卷积定理把码型激励频谱与脉冲核频谱逐点相乘。
    model_spectrum = excitation_spectrum * kernel_spectrum
    # LFP 需要一个完整稳态周期，指定 n 防止奇数周期长度被误恢复。
    model_waveform = scipy_fft.irfft(model_spectrum, n=period_samples)
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 公开字段全部转为独立只读数组，供界面与候选分析安全复用。
    return VppPeriodicModel(
        peak_index=peak_index,
        window_start=window_start,
        window_stop=window_stop,
        lag_samples=lag_samples,
        pulse_window_v=pulse_window,
        circular_kernel_v=_readonly_array(circular_kernel, np.float64),
        spectrum_v=_readonly_array(model_spectrum, np.complex128),
        waveform_v=_readonly_array(model_waveform, np.float64),
    )


# 单边 rFFT Parseval 直接计算复频谱 AC 误差，保留幅度和相位差。
def _frequency_ac_rms_error(
    candidate_spectrum: ComplexArray,
    reference_spectrum: ComplexArray,
    period_samples: int,
) -> float:
    """返回候选与参考周期波形去 DC 后的频域误差 Vrms。"""

    # rFFT 长度必须与完整周期样点数严格对应。
    expected_bins = period_samples // 2 + 1
    # 两份频谱必须一维、同长并覆盖同一周期的所有非负频点。
    if (
        candidate_spectrum.ndim != 1
        or reference_spectrum.ndim != 1
        or candidate_spectrum.shape != reference_spectrum.shape
        or candidate_spectrum.size != expected_bins
    ):
        # 形状不一致时频点无法一一作复数误差，必须拒绝。
        raise ValueError("候选与参考频谱形状必须匹配周期 rFFT 长度")
    # 任何非有限频点都会让能量和 Vrms 失去意义。
    if not np.all(np.isfinite(candidate_spectrum)) or not np.all(
        np.isfinite(reference_spectrum)
    ):
        # 在平方前拒绝异常值，避免溢出或静默 NaN。
        raise ValueError("候选与参考频谱必须全部为有限值")
    # 复数差同时保留幅度和相位误差，不能先取 abs 再相减。
    difference = candidate_spectrum - reference_spectrum
    # 只有一个样点时 rFFT 仅含 DC；按 D[0]=0 定义 AC 误差为零。
    if period_samples == 1:
        # 直接返回精确零，避免对空频点集合做特殊 NumPy 运算。
        return 0.0
    # 偶数长度的最后一个 rFFT bin 是唯一的 Nyquist 实频点，只计一次。
    if period_samples % 2 == 0:
        # 1..-2 为拥有负频共轭伙伴的内部频点，能量需要乘二。
        paired_power = 2.0 * float(np.sum(np.abs(difference[1:-1]) ** 2))
        # Nyquist 不存在独立负频伙伴，因此其能量只计一次。
        nyquist_power = float(np.abs(difference[-1]) ** 2)
        # D[0] 被直接排除，得到完整双边 AC 误差频谱的平方和。
        spectral_power = paired_power + nyquist_power
    # 奇数周期没有独立 Nyquist 端点，所有非 DC bin 都按共轭对计权。
    else:
        # 奇数长度没有 Nyquist，所有非 DC rFFT bin 都有负频共轭伙伴。
        spectral_power = 2.0 * float(np.sum(np.abs(difference[1:]) ** 2))
    # SciPy 的未归一化前向 FFT 满足 mean(|d|^2)=sum(|D|^2)/N^2。
    return float(np.sqrt(spectral_power) / period_samples)


# 公开准备入口只做一次码型 FFT，并建立参考/DUT 两份可复用模型。
def prepare_vpp_analysis(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: VppAnalysisSettings,
    *,
    prepared_pattern_levels: object | None = None,
    cancelled: CancellationCheck | None = None,
) -> VppAnalysisCache:
    """准备周期稳态参考/DUT 模型及其基线 Vpp 指标。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 前游标长度按 UI 整数乘每 UI 样点数精确换算，不做四舍五入。
    pre_samples = settings.pre_cursor_ui * settings.samples_per_ui
    # 后游标使用同一整数换算，保证 GUI 显示和窗口索引完全一致。
    post_samples = settings.post_cursor_ui * settings.samples_per_ui
    # 先验证两份 pmax 窗口；错误窗口不得先申请码型周期级 FFT 工作区。
    reference_geometry, dut_geometry = _validated_pulse_window_geometries(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 控制器可把预检时加载的外部码型冻结后传入，避免估算与正式计算二次读文件。
    if prepared_pattern_levels is None:
        pattern_levels = load_pattern_levels(settings, cancelled=cancelled)
    else:
        pattern_levels = _readonly_array(prepared_pattern_levels, np.float64)
        if pattern_levels.size < 1 or np.unique(pattern_levels).size < 2:
            raise ValueError("预加载理想码型至少需要两个不同电平")
    # 一个周期样点数是 symbol 数乘每 UI 样点数，保留 UI 内插零位置。
    period_samples = int(pattern_levels.size * settings.samples_per_ui)
    # 构造稀疏的理想 symbol 激励；每 UI 只有起始样点非零。
    excitation = np.zeros(period_samples, dtype=np.float64)
    # 按码型顺序把各 symbol 电平放到对应 UI 起点。
    excitation[:: settings.samples_per_ui] = pattern_levels
    # 激励 rFFT 在参考、DUT 以及所有补偿候选间复用。
    excitation_spectrum = _readonly_array(scipy_fft.rfft(excitation), np.complex128)
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 从完整参考脉冲按其首次 pmax 独立截取窗口并构建模型。
    reference_model = _build_periodic_model(
        reference_pulse,
        pre_samples,
        post_samples,
        period_samples,
        excitation_spectrum,
        reference_geometry,
        cancelled=cancelled,
    )
    # DUT 使用自身首次 pmax，不能借用参考峰值索引。
    dut_model = _build_periodic_model(
        dut_pulse,
        pre_samples,
        post_samples,
        period_samples,
        excitation_spectrum,
        dut_geometry,
        cancelled=cancelled,
    )
    # LFP 对两份完整稳态周期分别计算 exact max-min。
    if settings.method == "lfp":
        # 参考指标不使用分位数近似或分块中位数。
        reference_metric_v = float(
            np.max(reference_model.waveform_v) - np.min(reference_model.waveform_v)
        )
        # DUT 基线指标采用完全相同的 exact max-min 定义。
        dut_metric_v = float(np.max(dut_model.waveform_v) - np.min(dut_model.waveform_v))
    # 频域 RMS 分支直接比较 DUT 与参考的复频谱 AC 误差。
    else:
        # RMS 方法把参考相对于自身的误差定义为精确零。
        reference_metric_v = 0.0
        # DUT 指标直接在复频谱上按 Parseval 计算 AC 误差，不依赖时域分位数。
        dut_metric_v = _frequency_ac_rms_error(
            dut_model.spectrum_v,
            reference_model.spectrum_v,
            period_samples,
        )
    # rfftfreq 用真实采样率生成与模型频谱一一对应的 Hz 频轴。
    frequency_hz = scipy_fft.rfftfreq(
        period_samples,
        d=1.0 / float(reference_pulse.sample_rate_hz),
    )
    # 返回只读缓存，上层可以不重复拟合脉冲地扫描任意补偿候选。
    return VppAnalysisCache(
        settings=settings,
        sample_rate_hz=float(reference_pulse.sample_rate_hz),
        symbol_rate_hz=float(
            reference_pulse.sample_rate_hz / settings.samples_per_ui
        ),
        ui_duration_s=float(
            settings.samples_per_ui / reference_pulse.sample_rate_hz
        ),
        period_samples=period_samples,
        frequency_hz=_readonly_array(frequency_hz, np.float64),
        pattern_levels=pattern_levels,
        excitation_spectrum=excitation_spectrum,
        reference_model=reference_model,
        dut_model=dut_model,
        reference_metric_v=reference_metric_v,
        dut_metric_v=dut_metric_v,
    )


# 候选入口只对已准备的 DUT 模型乘补偿频谱，不重复拟合或重建码型。
def measure_candidate(
    cache: VppAnalysisCache,
    correction_spectrum: object,
    *,
    cancelled: CancellationCheck | None = None,
) -> VppCandidateMeasurement:
    """测量 ``DUT模型频谱 × correction_spectrum`` 的所选 Vpp 指标。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 复制并检查候选补偿为一维有限 complex128，隔离调用方后续修改。
    correction = _readonly_array(correction_spectrum, np.complex128)
    # 补偿频点必须与缓存 DUT rFFT 频点一一对应，禁止插值或截断猜测。
    if correction.shape != cache.dut_model.spectrum_v.shape:
        # 上层应先在 cache.frequency_hz 上构造候选补偿频谱。
        raise ValueError("correction_spectrum 必须与 DUT 模型 rFFT 形状一致")
    # 复数逐点相乘同时施加幅度和相位补偿，并保留可规范化的独立副本。
    corrected_values = np.array(
        cache.dut_model.spectrum_v * correction,
        dtype=np.complex128,
        copy=True,
    )
    # 实信号 rFFT 的 DC bin 只有实数自由度，虚部不会出现在 irfft 波形中。
    corrected_values[0] = complex(float(corrected_values[0].real), 0.0)
    # 偶数周期的最后一个 bin 是 Nyquist，也必须投影到真实余弦自由度。
    if cache.period_samples % 2 == 0:
        # 与 SciPy/NumPy irfft 一致地丢弃无物理意义的 Nyquist 虚部。
        corrected_values[-1] = complex(float(corrected_values[-1].real), 0.0)
    # 规范化后再冻结，确保 LFP 与 RMS 对任意 complex correction 具有相同端点语义。
    corrected_spectrum = _readonly_array(corrected_values, np.complex128)
    # LFP 必须恢复一个完整稳态周期后才能得到 exact max-min。
    if cache.settings.method == "lfp":
        # 指定 n 保证奇数周期长度不会按默认规则多恢复一个样点。
        waveform = _readonly_array(
            scipy_fft.irfft(corrected_spectrum, n=cache.period_samples),
            np.float64,
        )
        raise_if_cancelled(cancelled, message="影响频段分析已取消")
        # 候选 LFP 严格使用一次 IFFT 结果的最大值减最小值。
        value_v = float(np.max(waveform) - np.min(waveform))
        # LFP 返回波形供界面展示和独立复核。
        candidate_waveform: FloatArray | None = waveform
    # 频域 RMS 候选使用 Parseval，不恢复任何候选时域波形。
    else:
        # RMS 候选直接比较复频谱 AC 误差，不构造时域数组。
        value_v = _frequency_ac_rms_error(
            corrected_spectrum,
            cache.reference_model.spectrum_v,
            cache.period_samples,
        )
        # None 明确表示频域方法没有执行 IFFT，而不是数据遗漏。
        candidate_waveform = None
    # 返回只读补偿频谱、指标以及仅 LFP 存在的周期波形。
    return VppCandidateMeasurement(
        value_v=value_v,
        corrected_spectrum_v=corrected_spectrum,
        waveform_v=candidate_waveform,
    )
