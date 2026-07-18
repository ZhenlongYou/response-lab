"""主要影响频段归因所需的轻量标量指标。

本模块只负责从原始单通道波形估计 Vpp，不执行频响补偿，也不比较波形样点。
参考记录和 DUT 记录可以具有不同的采样率与长度；公共接口只返回可供归因层使用的
标量结果及分块证据。
"""

# 延迟解析类型标注，避免公共模型在导入阶段产生额外依赖。
from __future__ import annotations

# 冻结数据类用于传递不可变的指标结果和分块证据。
from dataclasses import dataclass

# NumPy 提供固定分位数、块中位数和有限值校验。
import numpy as np

# TimeSeries 是携带采样率和时间轴校验的唯一输入边界。
from .models import TimeSeries

# 下分位固定为 0.1%，用于削弱孤立负向毛刺对 Vpp 的支配。
_LOWER_QUANTILE = 0.001
# 上分位固定为 99.9%，与下分位共同定义本模块唯一的 Vpp 口径。
_UPPER_QUANTILE = 0.999
# 分块至少保留 TimeSeries 允许的最小八点记录，避免空洞分位数结果。
_MIN_BLOCK_SAMPLES = 8


# 单条波形的结果同时保留最终标量和可审计的块级明细。
@dataclass(frozen=True)
class VppEstimate:
    """一条记录在公共时间窗口下得到的稳健 Vpp 与极值诊断。"""

    # value 是所有完整块 Vpp 的中位数，单位与输入幅度一致。
    value: float
    # block_values 按记录中的时间顺序保存每个完整块结果。
    block_values: tuple[float, ...]
    # extrema_value 是完整块极值 Vpp 的中位数，只用于诊断而不参与扫描排名。
    extrema_value: float
    # extrema_block_values 与 block_values 共用块边界，保存逐块 max-min 结果。
    extrema_block_values: tuple[float, ...]
    # block_samples 记录当前采样率下一个公共时间窗口包含的点数。
    block_samples: int
    # used_samples 只统计进入完整块的样本。
    used_samples: int
    # discarded_samples 记录不足一块且未参与指标的尾部点数。
    discarded_samples: int


# 比较结果保留两侧独立证据，不引入任何样点级对齐关系。
@dataclass(frozen=True)
class VppComparison:
    """参考记录与 DUT 记录的两个独立 Vpp 标量。"""

    # 参考侧使用自身采样率下的公共时间窗口分块。
    reference: VppEstimate
    # DUT 侧独立分块，允许与参考具有不同点数和采样率。
    dut: VppEstimate
    # common_duration_s 是两条记录较短的半开有效时长，单位为秒。
    common_duration_s: float

    # 扫描器通常只需要扁平标量，因此提供不泄漏块结构的只读属性。
    @property
    def reference_vpp(self) -> float:
        """返回供扫描器直接使用的参考 Vpp 标量。"""

        # 扁平属性隐藏块证据结构，使扫描器只依赖稳定的标量接口。
        return self.reference.value

    # DUT 扁平属性与参考属性保持对称，便于直接计算指标差距。
    @property
    def dut_vpp(self) -> float:
        """返回供扫描器直接使用的 DUT Vpp 标量。"""

        # DUT 标量与参考标量使用完全相同的固定分位数定义。
        return self.dut.value

    # 极值诊断与稳健排名值分开暴露，防止调用方误把毛刺敏感量用于扫描评分。
    @property
    def reference_extrema_vpp(self) -> float:
        """返回参考记录按相同完整块聚合的极值 Vpp 诊断。"""

        # 极值值来自参考侧块级 max-min 的中位数，单位与输入幅度一致。
        return self.reference.extrema_value

    # DUT 极值属性保持与参考侧对称，方便报告并列展示两侧诊断。
    @property
    def dut_extrema_vpp(self) -> float:
        """返回 DUT 记录按相同完整块聚合的极值 Vpp 诊断。"""

        # 该属性不改变 dut_vpp 的稳健分位数语义，只补充毛刺敏感证据。
        return self.dut.extrema_value

    # 兼容早期草案中的命名，后续新代码应优先使用 common_duration_s。
    @property
    def comparison_window_s(self) -> float:
        """兼容早期调用方使用的公共窗口属性名。"""

        # 旧名称与新扫描器字段指向同一个物理时长，不复制状态。
        return self.common_duration_s


# 核心数值 oracle 只做固定分位差，不包含分块或跨记录比较策略。
def _robust_vpp(values: np.ndarray) -> float:
    """用固定线性分位数计算 Q99.9% - Q0.1%。"""

    # 显式锁定线性插值方法，避免 NumPy 默认策略变化造成历史结果漂移。
    lower, upper = np.quantile(
        values,
        (_LOWER_QUANTILE, _UPPER_QUANTILE),
        method="linear",
    )
    # 两个分位数使用相同的输入幅度单位，因此差值仍保持原始幅度单位。
    return float(upper - lower)


# 极值诊断保留传统 max-min 口径，便于观察毛刺对示波器 Vpp 差异的贡献。
def _extrema_vpp(values: np.ndarray) -> float:
    """计算单个完整块的最大值减最小值。"""

    # 最大值和最小值取自同一个完整块，因此差值仍保持输入的幅度单位。
    return float(np.max(values) - np.min(values))


# 一个块同时计算稳健值和极值，确保两个证据严格共用同一批样本。
def _measure_block(values: np.ndarray) -> tuple[float, float]:
    """返回单块的（稳健 Vpp，极值 Vpp）。"""

    # 元组第一项维持扫描器现有的分位数排名口径，第二项仅供诊断。
    return _robust_vpp(values), _extrema_vpp(values)


# 较短记录以整条波形为唯一块，避免人为裁掉其有效数据。
def _whole_record_estimate(series: TimeSeries) -> VppEstimate:
    """把整条单通道记录作为一个 Vpp 比较块。"""

    # 当前切片只读取唯一通道；多通道选择由后续显式验证负责。
    values = np.asarray(series.values[:, 0], dtype=np.float64)
    # 单块结果同时作为最终值和可审计的块级证据。
    block_value, extrema_block_value = _measure_block(values)
    # 整条记录均被使用，因此不存在被丢弃的尾部样本。
    return VppEstimate(
        value=block_value,
        block_values=(block_value,),
        extrema_value=extrema_block_value,
        extrema_block_values=(extrema_block_value,),
        block_samples=series.samples,
        used_samples=series.samples,
        discarded_samples=0,
    )


# 较长记录按公共物理秒数换算自己的块点数，再只保留完整块。
def _windowed_estimate(
    series: TimeSeries,
    comparison_window_s: float,
) -> VppEstimate:
    """把较长记录切成不超过公共物理时长的完整等样本块。"""

    # 将公共秒数换算到当前记录自己的采样网格，不能借用另一条记录的点数。
    exact_block_samples = comparison_window_s * series.sample_rate_hz
    # 只补偿浮点乘法在整数边界附近的舍入误差，避免意外多取一个真实样本。
    rounding_tolerance = (
        32.0
        * np.finfo(np.float64).eps
        * max(abs(exact_block_samples), 1.0)
    )
    # 向下取整保证每个离散块的半开时长不会超过公共比较窗口。
    block_samples = int(np.floor(exact_block_samples + rounding_tolerance))
    # 过少样本无法提供与 TimeSeries 最小记录约束一致的稳健分位数证据。
    if block_samples < _MIN_BLOCK_SAMPLES:
        # 少于八点会违反项目对有效时域记录的最低证据要求。
        raise ValueError("公共比较窗口在当前采样率下不足 8 个样本")
    # 只保留能够完整容纳公共窗口的块，不足一块的尾部不参与统计。
    complete_blocks = series.samples // block_samples
    # 理论上较长记录至少有一块；显式检查可防止异常采样率静默产生空中位数。
    if complete_blocks < 1:
        # 没有完整块时不能用残缺尾部冒充公共时长测量。
        raise ValueError("记录长度不足一个完整公共比较窗口")
    # 每个完整块同时计算稳健值和极值，两个诊断不会采用不同样本边界。
    block_measurements = tuple(
        _measure_block(
            np.asarray(
                series.values[
                    block_index * block_samples : (block_index + 1) * block_samples,
                    0,
                ],
                dtype=np.float64,
            )
        )
        for block_index in range(complete_blocks)
    )
    # 第一列保留原有 Q99.9%-Q0.1% 块值，继续作为扫描排名的唯一 Vpp。
    block_values = tuple(measurement[0] for measurement in block_measurements)
    # 第二列保留 max-min 块值，只向报告提供毛刺敏感的诊断证据。
    extrema_block_values = tuple(
        measurement[1] for measurement in block_measurements
    )
    # 取块级中位数，降低较长记录中偶发极端时间段对最终标量的支配。
    value = float(np.median(np.asarray(block_values, dtype=np.float64)))
    # 极值诊断也取相同完整块集合的中位数，保持与稳健值的聚合可比性。
    extrema_value = float(
        np.median(np.asarray(extrema_block_values, dtype=np.float64))
    )
    # 已使用样本只包含整数个完整块，余数单独记录供报告审计。
    used_samples = complete_blocks * block_samples
    # 返回最终标量和所有块级证据，归因层无需重新理解分块规则。
    return VppEstimate(
        value=value,
        block_values=block_values,
        extrema_value=extrema_value,
        extrema_block_values=extrema_block_values,
        block_samples=block_samples,
        used_samples=used_samples,
        discarded_samples=series.samples - used_samples,
    )


# 这一分派函数确保较短记录整条使用、较长记录才执行分块中位数。
def _estimate_for_common_window(
    series: TimeSeries,
    comparison_window_s: float,
) -> VppEstimate:
    """较短记录整条使用，较长记录按公共窗口分块。"""

    # 半开有效时长 N / Fs 与公共窗口同单位，适合跨采样率判断长短。
    effective_duration_s = series.samples / series.sample_rate_hz
    # 容忍时长计算末位误差，防止本应等长的记录被错误切成一块再丢一个样本。
    durations_match = np.isclose(
        effective_duration_s,
        comparison_window_s,
        rtol=32.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    # 较短记录或等时长记录自身就是公共比较块。
    if durations_match:
        # 等时长记录不再二次分块，防止浮点误差丢掉末尾样本。
        return _whole_record_estimate(series)
    # 只有真正较长的记录才进入完整块中位数路径。
    return _windowed_estimate(series, comparison_window_s)


# 输入边界明确拒绝缺少物理时间语义或通道选择歧义的数据。
def _validate_single_channel(series: object, label: str) -> None:
    """拒绝缺少时间语义或通道选择含糊的输入记录。"""

    # 普通数组没有采样率与时间轴，不能在接口内部猜测其物理采样语义。
    if not isinstance(series, TimeSeries):
        # 普通数组没有采样率，无法建立跨记录公共时间窗口。
        raise TypeError(f"{label}必须是 TimeSeries")
    # Vpp 归因没有隐式通道选择规则，因此多通道输入必须由调用方先明确拆分。
    if series.channels != 1:
        # 调用方必须先显式选通道，模块不会静默读取第零列。
        raise ValueError(f"{label}必须是单通道 TimeSeries")


# 单波形入口供扫描器重复测量补偿后的 DUT 波形。
def measure_waveform_vpp(
    series: TimeSeries,
    *,
    block_duration_s: float | None = None,
) -> float:
    """测量单条波形的稳健 Vpp，可按给定物理时长取完整块中位数。"""

    # 单条测量同样执行类型和通道检查，避免绕过比较入口的边界约束。
    _validate_single_channel(series, "待测记录")
    # 未指定分块时，整条记录自身作为唯一统计块。
    if block_duration_s is None:
        # 未指定物理块时长时，整个记录就是唯一统计总体。
        return _whole_record_estimate(series).value
    # 布尔值虽然是 Python 整数子类，但不能表达有意义的物理秒数。
    if (
        isinstance(block_duration_s, (bool, np.bool_))
        or not np.isfinite(block_duration_s)
        or block_duration_s <= 0.0
    ):
        # 非正、非有限时长无法换算成有意义的样本块。
        raise ValueError("Vpp 分块时长必须是正的有限秒数")
    # 显式时长只使用完整块，块级结果再取中位数。
    return _windowed_estimate(series, float(block_duration_s)).value


# 双波形入口负责确定公共时长，但仍分别测量两侧幅度分布。
def compare_waveform_vpp(reference: TimeSeries, dut: TimeSeries) -> VppComparison:
    """独立估计参考与 DUT 的稳健 Vpp，不做逐点对齐或重采样。"""

    # 先验证参考侧，避免后续代码静默读取二维数组的第一个通道。
    _validate_single_channel(reference, "参考记录")
    # DUT 侧使用完全相同的通道约束，保证比较两端语义对称。
    _validate_single_channel(dut, "DUT 记录")
    # N / Fs 表示 N 个等间隔样本覆盖的半开有效采集窗口。
    reference_duration_s = reference.samples / reference.sample_rate_hz
    # DUT 使用自己的采样率计算时长，不能假设两条记录共用采样网格。
    dut_duration_s = dut.samples / dut.sample_rate_hz
    # 公共比较窗口由较短记录决定，较长记录后续只能使用该长度的完整块。
    comparison_window_s = min(reference_duration_s, dut_duration_s)
    # 较短参考记录整条使用，较长参考记录则按公共窗口切块。
    reference_estimate = _estimate_for_common_window(reference, comparison_window_s)
    # DUT 独立遵循同一物理时长规则，不进行任何样点对齐或重采样。
    dut_estimate = _estimate_for_common_window(dut, comparison_window_s)
    # 返回两个标量及公共物理时长，供归因层计算指标差距。
    return VppComparison(
        reference=reference_estimate,
        dut=dut_estimate,
        common_duration_s=float(comparison_window_s),
    )


# 保留短名称兼容早期调用；实现始终转发到正式公共入口。
def compare_vpp(reference: TimeSeries, dut: TimeSeries) -> VppComparison:
    """兼容早期短名称，并转发到扫描器使用的明确公共入口。"""

    # 兼容入口不复制算法，确保两个名称永远遵循同一分块与分位数契约。
    return compare_waveform_vpp(reference, dut)


# 仅导出结果模型、正式扫描器接口和兼容别名。
__all__ = [
    "VppComparison",
    "VppEstimate",
    "compare_vpp",
    "compare_waveform_vpp",
    "measure_waveform_vpp",
]
