"""频响比较、相位去斜与直接频域补偿。

约定：真实时域信号使用 ``rfft/irfft``；频率使用 Hz，相位内部使用 rad，幅度 dB
使用电压/幅度定义 ``20*log10``。两份脉冲允许采样率和点数不同，它们分别变换后
插值到公共物理频率轴，不会通过数组下标强行对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft

from .memory_budget import (
    parse_macos_vm_stat,
    safe_memory_budget_bytes,
    system_available_memory_bytes,
)
from .models import (
    CompensationRun,
    CompensationSettings,
    PulseComparison,
    ResponseAnalysis,
    TimeSeries,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_AUTOMATIC_BAND_FLOOR_DB = -20.0
_AUTOMATIC_BAND_MINIMUM_POINTS = 16
_AUTOMATIC_BAND_SIGNIFICANT_DIGITS = 2
_NUMERIC_RESPONSE_FLOOR_RATIO = 64.0 * np.finfo(np.float64).eps
_DIRECT_RESPONSE_REFINEMENT_RATIO = np.sqrt(np.finfo(np.float64).eps)

# 2026-07-23 的数组生命周期优化后，两组全新独立子进程实测为：
# N=1,000,000、50–200 MHz 新增 RSS 峰值 157,581,312 B；
# N=500,000、0–Nyquist 新增 RSS 峰值 160,022,528 B。
# 下列分项模型分别给出约 67% 和 57% 余量，并随通道数、带内 bin 和长脉冲
# CZT 卷积长度增长。固定余量覆盖 FFT 计划、分配器碎片和小型分析对象。
_COMPENSATION_BASE_BYTES_PER_SAMPLE = 192
_COMPENSATION_EXTRA_CHANNEL_BYTES_PER_SAMPLE = 128
_COMPENSATION_CZT_BYTES_PER_WORKING_SAMPLE = 160
_COMPENSATION_ANALYSIS_BYTES_PER_POINT = 96
_COMPENSATION_PULSE_FFT_BYTES_PER_BIN = 64
_COMPENSATION_FIXED_OVERHEAD_BYTES = 32 * 1024**2


@dataclass(frozen=True)
class CompensationMemoryEstimate:
    """一次直接频域补偿的保守新增峰值工作区估算。"""

    target_samples: int
    target_channels: int
    extended_samples: int
    rfft_bins: int
    active_band_bins: int
    czt_working_samples: int
    estimated_peak_bytes: int

    @property
    def estimated_bytes_per_target_sample(self) -> float:
        """把固定和带内开销折算为本次输入的每点字节数，便于诊断。"""

        return self.estimated_peak_bytes / self.target_samples


def _compensation_memory_estimate_from_shape(
    *,
    target_samples: int,
    target_channels: int,
    sample_rate_hz: float,
    reference_samples: int,
    dut_samples: int,
    settings: CompensationSettings,
) -> CompensationMemoryEstimate:
    """只按形状估算工作区，不构造时间轴、掩码或 FFT 数组。"""

    integer_fields = {
        "目标样点数": target_samples,
        "目标通道数": target_channels,
        "参考脉冲样点数": reference_samples,
        "DUT 脉冲样点数": dut_samples,
    }
    for label, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label}必须是正整数")
        if int(value) <= 0:
            raise ValueError(f"{label}必须是正整数")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("待补偿信号采样率必须是正的有限值")

    samples = int(target_samples)
    channels = int(target_channels)
    extended_samples = 3 * samples - 2
    rfft_bins = extended_samples // 2 + 1

    # 比真实 band_mask 各向外多取最多一个 bin；浮点端点舍入不会低估 CZT 长度。
    low_index = max(
        0,
        int(np.floor(settings.band_low_hz * extended_samples / sample_rate_hz)),
    )
    high_index = min(
        rfft_bins - 1,
        int(np.ceil(settings.band_high_hz * extended_samples / sample_rate_hz)),
    )
    active_band_bins = max(0, high_index - low_index + 1)

    # SciPy CZT 使用长度约为 pulse_samples + m - 1 的 Bluestein 卷积；按下一快速
    # FFT 长度预估比只按目标 N 或带宽百分比更接近实际临时量。
    longest_pulse = max(int(reference_samples), int(dut_samples))
    czt_working_samples = next_fast_len(longest_pulse + max(active_band_bins, 1) - 1)

    # 显示频响分析另有脉冲 FFT 与 analysis_points 多数组常驻，不能被目标带宽掩盖。
    pulse_fft_length = next_fast_len(
        max(2 * longest_pulse, 2 * (int(settings.analysis_points) - 1))
    )
    pulse_fft_bins = pulse_fft_length // 2 + 1

    estimated_peak_bytes = (
        samples * _COMPENSATION_BASE_BYTES_PER_SAMPLE
        + samples
        * max(channels - 1, 0)
        * _COMPENSATION_EXTRA_CHANNEL_BYTES_PER_SAMPLE
        + czt_working_samples * _COMPENSATION_CZT_BYTES_PER_WORKING_SAMPLE
        + int(settings.analysis_points) * _COMPENSATION_ANALYSIS_BYTES_PER_POINT
        + pulse_fft_bins * _COMPENSATION_PULSE_FFT_BYTES_PER_BIN
        + _COMPENSATION_FIXED_OVERHEAD_BYTES
    )
    return CompensationMemoryEstimate(
        target_samples=samples,
        target_channels=channels,
        extended_samples=extended_samples,
        rfft_bins=rfft_bins,
        active_band_bins=active_band_bins,
        czt_working_samples=czt_working_samples,
        estimated_peak_bytes=int(estimated_peak_bytes),
    )


def _parse_macos_vm_stat(output: str) -> int | None:
    """兼容旧测试入口；真实解析实现在中立资源模块。"""

    return parse_macos_vm_stat(output)


def _system_available_memory_bytes() -> int | None:
    """兼容旧测试/补丁入口；系统探测实现在中立资源模块。"""

    return system_available_memory_bytes()


def _safe_compensation_memory_budget_bytes(available_memory_bytes: int | None) -> int:
    """兼容旧测试入口；预算规则由中立资源模块统一维护。"""

    return safe_memory_budget_bytes(available_memory_bytes)


def _preflight_compensation_memory(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    input_signal: TimeSeries,
    settings: CompensationSettings,
) -> CompensationMemoryEstimate:
    """在响应分析、CZT、镜像延拓和目标 FFT 之前执行共享内存门禁。"""

    estimate = _compensation_memory_estimate_from_shape(
        target_samples=input_signal.samples,
        target_channels=input_signal.channels,
        sample_rate_hz=input_signal.sample_rate_hz,
        reference_samples=reference_pulse.samples,
        dut_samples=dut_pulse.samples,
        settings=settings,
    )
    available = _system_available_memory_bytes()
    budget = _safe_compensation_memory_budget_bytes(available)
    if estimate.estimated_peak_bytes > budget:
        available_text = (
            f"，系统当前可用约 {available / (1024.0**2):.0f} MiB"
            if available is not None
            else "，系统可用内存无法探测，已采用保守回退预算"
        )
        raise MemoryError(
            "补偿内存预检（CSV/BIN 共用）拒绝启动：预计新增峰值工作区约 "
            f"{estimate.estimated_peak_bytes / (1024.0**2):.0f} MiB，"
            f"本次安全预算约 {budget / (1024.0**2):.0f} MiB{available_text}。"
            f"估算已计入 {estimate.extended_samples} 点镜像记录、"
            f"{estimate.active_band_bins} 个带内频点和 CZT 临时量；"
            "请缩短目标记录、减少通道数或缩小补偿频带。"
        )
    return estimate


def _contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """返回布尔掩码中每个 True 区间的半开索引 ``[start, stop)``。"""

    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in changes.reshape(-1, 2)]


def _segmented_unwrap(
    wrapped_phase_rad: FloatArray,
    reliable_mask: NDArray[np.bool_],
    confidence: FloatArray,
) -> FloatArray:
    """只在连续可信区间内展开相位，并用确定性锚点选择整数圈。

    无效缺口保持 NaN，因此深陷波的随机相位不会把错误分支传播到另一段频带。
    每段锚点先选择最高置信度，再选择最靠近区段中心的候选点。
    """

    output = np.full(wrapped_phase_rad.shape, np.nan, dtype=np.float64)
    for start, stop in _contiguous_runs(reliable_mask):
        if stop - start < 2:
            continue
        segment = np.unwrap(wrapped_phase_rad[start:stop])
        segment_confidence = confidence[start:stop]
        maximum = float(np.max(segment_confidence))
        candidates = np.flatnonzero(np.isclose(segment_confidence, maximum, rtol=1e-12, atol=0.0))
        center = 0.5 * (stop - start - 1)
        anchor = int(candidates[np.argmin(np.abs(candidates - center))])
        segment -= 2.0 * np.pi * np.round(segment[anchor] / (2.0 * np.pi))
        output[start:stop] = segment
    return output


def _interpolate_segments(
    source_frequency_hz: FloatArray,
    source_values: FloatArray,
    target_frequency_hz: FloatArray,
) -> FloatArray:
    """逐个有限区段线性插值，绝不跨 NaN 缺口连接相位。"""

    finite = np.isfinite(source_values)
    output = np.full(target_frequency_hz.shape, np.nan, dtype=np.float64)
    for start, stop in _contiguous_runs(finite):
        if stop - start < 2:
            continue
        source_x = source_frequency_hz[start:stop]
        source_y = source_values[start:stop]
        spacing_hz = float(np.median(np.diff(source_x)))
        endpoint_tolerance_hz = max(
            spacing_hz * 1.0e-12,
            64.0 * np.finfo(np.float64).eps * max(abs(source_x[0]), abs(source_x[-1]), 1.0),
        )
        target_mask = (
            (target_frequency_hz >= source_x[0] - endpoint_tolerance_hz)
            & (target_frequency_hz <= source_x[-1] + endpoint_tolerance_hz)
        )
        clipped_target = np.clip(
            target_frequency_hz[target_mask],
            source_x[0],
            source_x[-1],
        )
        output[target_mask] = np.interp(clipped_target, source_x, source_y)
    return output


def _pulse_spectrum(
    pulse: TimeSeries, settings: CompensationSettings
) -> tuple[FloatArray, ComplexArray, FloatArray, NDArray[np.bool_]]:
    """计算带连续时间 ``dt`` 标度、但暂不含 ``t0`` 的单边脉冲谱。

    FFT 至少补零到原始脉冲长度的两倍。这样即使脉冲能量靠近记录末端，可信
    频点间由记录内时移造成的相位步进也小于 pi，不会让 ``unwrap`` 猜错圈数。
    若数值上仍发现接近 pi 的可信相邻步进，会继续细化网格；到资源上限仍有
    歧义则明确拒绝，而不是静默返回一个可能错误的时延。
    """

    values = np.asarray(pulse.values[:, 0], dtype=np.float64)
    if settings.taper_alpha > 0.0:
        values = values * signal.windows.tukey(values.size, alpha=settings.taper_alpha)
    requested_fft = max(2 * values.size, 2 * (settings.analysis_points - 1))
    fft_length = next_fast_len(requested_fft)
    dt_s = 1.0 / pulse.sample_rate_hz
    maximum_fft_length = 2**22
    phase_step_limit_rad = np.pi * (1.0 - 64.0 * np.finfo(np.float64).eps)

    while True:
        response = dt_s * np.fft.rfft(values, n=fft_length)
        frequency_hz = np.fft.rfftfreq(fft_length, d=dt_s)
        magnitude = np.abs(response)
        peak = float(np.max(magnitude))
        if peak <= np.finfo(np.float64).tiny:
            raise ValueError("拟合脉冲为全零或频谱能量低于浮点可解析范围")
        # 这里只排除浮点数值上无法辨认的零响应，不设置工程增益或可信度门限。
        # 补偿范围由用户选择的频带决定，频带内的可解析差异原样保留。
        threshold = max(
            np.finfo(np.float64).tiny,
            peak * _NUMERIC_RESPONSE_FLOOR_RATIO,
        )
        reliable = magnitude > threshold

        adjacent_reliable = reliable[:-1] & reliable[1:]
        phase_step_rad = np.abs(np.angle(response[1:] * np.conj(response[:-1])))
        ambiguous = adjacent_reliable & (phase_step_rad >= phase_step_limit_rad)
        if not np.any(ambiguous):
            return frequency_hz, response, magnitude, reliable

        refined_fft_length = next_fast_len(2 * fft_length)
        if refined_fft_length > maximum_fft_length:
            maximum_step_rad = float(np.max(phase_step_rad[ambiguous]))
            raise ValueError(
                "拟合脉冲可信频点的相位步进仍接近 pi，无法可靠展开相位"
                f"（最大 {maximum_step_rad:.6g} rad）。请缩短记录前的空白、提高采样率，"
                "或减少脉冲记录长度后重试。"
            )
        fft_length = refined_fft_length


def suggest_frequency_settings(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    template: CompensationSettings,
    *,
    maximum_frequency_hz: float | None = None,
    suggest_phase_fit_band: bool = False,
) -> CompensationSettings:
    """根据两份脉冲的共同幅度谱候选区间建议补偿频带。

    自动补偿频带只选择两份归一化幅度都不低于 -20 dB 的最长连续区间，并在区间
    两端留出裕量。自动生成的边界按 Hz 物理量取两位有效数字；若取整会破坏边界
    顺序或越过可信区间，则安全回退到未取整值。这样默认设置会随输入采样率和
    脉冲带宽缩放，也不会把公共 Nyquist 误当成工程有效带宽。第三份待补偿数据可
    通过 ``maximum_frequency_hz`` 进一步限制上限。-20 dB 是确定性的幅度启发式，
    不是噪声或相位可信度估计；输入脉冲应已去除直流基线，用户仍需检查拟合带内
    的相位质量。

    默认保留模板中的手动线性相位拟合频带。只有首次分析尚无用户输入时，调用方才应
    显式传入 ``suggest_phase_fit_band=True`` 取得一个可运行的拟合频带初值。
    """

    common_limit_hz = min(reference_pulse.nyquist_hz, dut_pulse.nyquist_hz)
    selection_limit_hz = common_limit_hz
    if maximum_frequency_hz is not None:
        if not np.isfinite(maximum_frequency_hz) or maximum_frequency_hz <= 0.0:
            raise ValueError("自动频带的额外频率上限必须是正的有限值")
        selection_limit_hz = min(common_limit_hz, float(maximum_frequency_hz))

    ref_f, _, ref_mag, _ = _pulse_spectrum(reference_pulse, template)
    dut_f, _, dut_mag, _ = _pulse_spectrum(dut_pulse, template)
    frequency_hz = np.linspace(0.0, common_limit_hz, template.analysis_points)
    tiny = np.finfo(np.float64).tiny

    def normalized_db(source_f: FloatArray, magnitude: FloatArray) -> FloatArray:
        log_magnitude = np.log(np.maximum(magnitude, tiny))
        interpolated = np.interp(frequency_hz, source_f, log_magnitude)
        return 20.0 / np.log(10.0) * (interpolated - float(np.max(log_magnitude)))

    joint_db = np.minimum(
        normalized_db(ref_f, ref_mag),
        normalized_db(dut_f, dut_mag),
    )
    eligible = frequency_hz <= selection_limit_hz
    candidates = [
        (start, stop)
        for start, stop in _contiguous_runs(
            eligible & (joint_db >= _AUTOMATIC_BAND_FLOOR_DB)
        )
        if stop - start >= _AUTOMATIC_BAND_MINIMUM_POINTS
    ]
    if not candidates:
        if maximum_frequency_hz is not None and np.count_nonzero(eligible) < (
            _AUTOMATIC_BAND_MINIMUM_POINTS
        ):
            raise ValueError(
                "目标信号 Nyquist 相对拟合脉冲过低，当前公共分析网格无法自动选择频带；"
                "请检查三份数据的采样率是否匹配"
            )
        raise ValueError(
            "无法从两份拟合脉冲中找到足够宽的共同 -20 dB 连续频带；"
            "请关闭自动频带并手动设置，或检查脉冲数据"
        )
    start, stop = max(candidates, key=lambda bounds: bounds[1] - bounds[0])
    usable_low_hz = float(frequency_hz[start])
    usable_high_hz = float(frequency_hz[stop - 1])
    usable_span_hz = usable_high_hz - usable_low_hz
    grid_step_hz = float(frequency_hz[1] - frequency_hz[0])
    if usable_span_hz <= 12.0 * grid_step_hz:
        raise ValueError("自动识别的共同有效频带过窄，请改用手动频带")

    unrounded_updates = {
        "band_low_hz": usable_low_hz + 0.02 * usable_span_hz,
        "band_high_hz": usable_low_hz + 0.95 * usable_span_hz,
    }
    if suggest_phase_fit_band:
        unrounded_updates.update(
            phase_fit_low_hz=usable_low_hz + 0.08 * usable_span_hz,
            phase_fit_high_hz=usable_low_hz + 0.90 * usable_span_hz,
        )
    rounded_updates = {
        name: float(f"{value_hz:.{_AUTOMATIC_BAND_SIGNIFICANT_DIGITS}g}")
        for name, value_hz in unrounded_updates.items()
    }
    rounded_bounds_are_safe = (
        usable_low_hz <= rounded_updates["band_low_hz"]
        < rounded_updates["band_high_hz"] <= usable_high_hz
    )
    if suggest_phase_fit_band:
        rounded_bounds_are_safe = rounded_bounds_are_safe and (
            rounded_updates["band_low_hz"]
            < rounded_updates["phase_fit_low_hz"]
            < rounded_updates["phase_fit_high_hz"]
            < rounded_updates["band_high_hz"]
        )
    updates = rounded_updates if rounded_bounds_are_safe else unrounded_updates
    return replace(
        template,
        **updates,
    )


def fit_linear_phase_slope(
    frequency_hz: FloatArray,
    phase_rad: FloatArray,
    weights: FloatArray,
    fit_mask: NDArray[np.bool_],
) -> float:
    """用每个可信岛独立截距的联合模型估计公共相位斜率。

    每个相位岛由独立展开和锚定得到，岛与岛之间可能相差任意整数圈。直接用单个
    截距做全带 WLS 会把这些圈差误认为斜率。这里先在每个岛内分别中心化，再聚合
    numerator/denominator；因此只使用岛内随频率的变化量估计公共斜率。各岛截距
    是干扰参数，不返回一个并不存在的“全局截距”。
    """

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    phase_rad = np.asarray(phase_rad, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (
        frequency_hz.ndim == phase_rad.ndim == weights.ndim == fit_mask.ndim == 1
        and frequency_hz.shape == phase_rad.shape == weights.shape == fit_mask.shape
    ):
        raise ValueError("线性相位拟合的频率、相位、权重和掩码必须是一维等长数组")
    if (
        np.any(~np.isfinite(frequency_hz))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(phase_rad[fit_mask]))
    ):
        raise ValueError("线性相位拟合的有效频点必须包含有限值和非负权重")
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("相位去斜观察频段内没有足够的连续可信频点")
    numerator = 0.0
    denominator = 0.0
    usable_points = 0
    for start, stop in _contiguous_runs(fit_mask):
        positive_weight = weights[start:stop] > 0.0
        if np.count_nonzero(positive_weight) < 2:
            continue
        x = frequency_hz[start:stop][positive_weight]
        y = phase_rad[start:stop][positive_weight]
        w = weights[start:stop][positive_weight]
        x_center = float(np.average(x, weights=w))
        y_center = float(np.average(y, weights=w))
        numerator += float(np.sum(w * (x - x_center) * (y - y_center)))
        denominator += float(np.sum(w * (x - x_center) ** 2))
        usable_points += x.size

    if usable_points < 3:
        raise ValueError("相位去斜观察频段内没有足够的连续可信频点")
    if denominator <= 0.0:
        raise ValueError("相位去斜频段过窄，无法拟合线性斜率")
    return numerator / denominator


def _anchor_phase_islands(
    phase_rad: FloatArray,
    reliable_mask: NDArray[np.bool_],
    confidence: FloatArray,
) -> FloatArray:
    """对每个可信岛独立减去整数个 2*pi，保持物理等价但避免带权伪跳变。"""

    output = np.full(phase_rad.shape, np.nan, dtype=np.float64)
    for start, stop in _contiguous_runs(reliable_mask):
        segment = phase_rad[start:stop].copy()
        if segment.size == 0 or not np.all(np.isfinite(segment)):
            continue
        segment_confidence = confidence[start:stop]
        maximum = float(np.max(segment_confidence))
        candidates = np.flatnonzero(np.isclose(segment_confidence, maximum, rtol=1e-12, atol=0.0))
        center = 0.5 * (segment.size - 1)
        anchor = int(candidates[np.argmin(np.abs(candidates - center))])
        segment -= 2.0 * np.pi * np.round(segment[anchor] / (2.0 * np.pi))
        output[start:stop] = segment
    return output


def _apply_compensation_safety(
    correction: ComplexArray,
    frequency_hz: FloatArray,
    settings: CompensationSettings,
    *,
    domain_high_hz: float,
) -> ComplexArray:
    """对带内响应应用显式增益上限和 raised-cosine 边缘过渡。"""

    applied = np.array(correction, dtype=np.complex128, copy=True)
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    if applied.ndim != 1 or frequencies.shape != applied.shape:
        raise ValueError("补偿安全处理的频率轴与响应必须是一维等长数组")

    if settings.maximum_gain_db is not None:
        maximum_finite_gain_db = 20.0 * np.log10(np.finfo(np.float64).max)
        maximum_gain = (
            float("inf")
            if settings.maximum_gain_db >= maximum_finite_gain_db
            else 10.0 ** (settings.maximum_gain_db / 20.0)
        )
        magnitude = np.abs(applied)
        excessive = magnitude > maximum_gain
        if np.any(excessive):
            applied[excessive] *= maximum_gain / magnitude[excessive]

    transition_fraction = settings.edge_transition_fraction
    if transition_fraction == 0.0 or frequencies.size == 0:
        return applied
    transition_hz = transition_fraction * (
        settings.band_high_hz - settings.band_low_hz
    )
    if transition_hz <= 0.0:
        return applied

    weight = np.ones(frequencies.size, dtype=np.float64)
    left_stop = 0
    if settings.band_low_hz > 0.0:
        left_stop = int(
            np.searchsorted(
                frequencies,
                settings.band_low_hz + transition_hz,
                side="left",
            )
        )
        left_frequency_hz = frequencies[:left_stop]
        left_position = np.clip(
            (left_frequency_hz - settings.band_low_hz) / transition_hz,
            0.0,
            1.0,
        )
        weight[:left_stop] = 0.5 - 0.5 * np.cos(np.pi * left_position)
    nyquist_tolerance_hz = (
        max(domain_high_hz, 1.0) * 32.0 * np.finfo(np.float64).eps
    )
    right_start = frequencies.size
    if settings.band_high_hz < domain_high_hz - nyquist_tolerance_hz:
        right_start = int(
            np.searchsorted(
                frequencies,
                settings.band_high_hz - transition_hz,
                side="right",
            )
        )
        right_frequency_hz = frequencies[right_start:]
        right_position = np.clip(
            (settings.band_high_hz - right_frequency_hz) / transition_hz,
            0.0,
            1.0,
        )
        right_weight = 0.5 - 0.5 * np.cos(np.pi * right_position)
        weight[right_start:] = np.minimum(weight[right_start:], right_weight)

    if np.all(weight == 1.0):
        return applied
    magnitude = np.abs(applied)
    phase = np.angle(applied)
    # 不能直接缩放 principal angle：肩部若跨越 +pi/-pi，会把纯粹的相位分支
    # 切换变成接近 2 rad 的真实跳变。左右肩分别从各自外边界向带内 unwrap，
    # 使从单位响应到目标响应走连续且最短的相位路径；核心区 weight=1，分支等价。
    if left_stop > 1:
        phase[:left_stop] = np.unwrap(phase[:left_stop])
    if frequencies.size - right_start > 1:
        phase[right_start:] = np.unwrap(phase[right_start:][::-1])[::-1]
    blended_magnitude = 1.0 + weight * (magnitude - 1.0)
    return blended_magnitude * np.exp(1j * weight * phase)


_GAIN_LIMIT_WARNING_PREFIX = "原始响应需要最高"


def _record_gain_limit_warning(
    warnings: list[str],
    requested_peak_db: float,
    maximum_gain_db: float,
) -> None:
    """以最后一次实际求值结果更新限幅告警，避免显示网格误报或漏报。"""

    warnings[:] = [
        warning
        for warning in warnings
        if not warning.startswith(_GAIN_LIMIT_WARNING_PREFIX)
    ]
    if requested_peak_db > maximum_gain_db + 1.0e-9:
        warnings.append(
            f"{_GAIN_LIMIT_WARNING_PREFIX} {requested_peak_db:.3g} dB 增益；"
            f"实际补偿已限制为 {maximum_gain_db:.3g} dB"
        )


def analyze_responses(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: CompensationSettings,
    *,
    application_domain_high_hz: float | None = None,
) -> ResponseAnalysis:
    """比较两份拟合脉冲并构造 ``H_ref / H_dut`` 补偿响应。

    补偿方向固定为 ``reference / dut``。含相位模式在用户拟合带内估计线性相位
    斜率；是否从实际补偿相位中减去该趋势由 ``detrend_phase`` 明确控制。
    """

    common_nyquist_hz = min(reference_pulse.nyquist_hz, dut_pulse.nyquist_hz)
    if settings.band_high_hz > common_nyquist_hz:
        raise ValueError(
            f"补偿上限 {settings.band_high_hz:g} Hz 超过两份脉冲公共 Nyquist "
            f"{common_nyquist_hz:g} Hz"
        )
    if settings.mode != "magnitude" and settings.phase_fit_high_hz > common_nyquist_hz:
        raise ValueError("线性相位拟合频带超过两份脉冲公共 Nyquist")
    if application_domain_high_hz is None:
        application_domain_high_hz = common_nyquist_hz
    elif (
        not np.isfinite(application_domain_high_hz)
        or application_domain_high_hz <= 0.0
        or application_domain_high_hz < settings.band_high_hz
    ):
        raise ValueError("实际应用频域上限必须是覆盖补偿频带的正有限值")

    ref_f, ref_h0, ref_mag, ref_reliable = _pulse_spectrum(reference_pulse, settings)
    dut_f, dut_h0, dut_mag, dut_reliable = _pulse_spectrum(dut_pulse, settings)
    frequency_hz = np.linspace(0.0, common_nyquist_hz, settings.analysis_points)

    tiny = np.finfo(np.float64).tiny
    ref_log_mag = np.log(np.maximum(ref_mag, tiny))
    dut_log_mag = np.log(np.maximum(dut_mag, tiny))
    ref_log_common = np.interp(frequency_hz, ref_f, ref_log_mag)
    dut_log_common = np.interp(frequency_hz, dut_f, dut_log_mag)
    ref_mag_common = np.exp(ref_log_common)
    dut_mag_common = np.exp(dut_log_common)

    ref_confidence = ref_mag / max(float(np.max(ref_mag)), tiny)
    dut_confidence = dut_mag / max(float(np.max(dut_mag)), tiny)
    ref_phase0 = _segmented_unwrap(np.angle(ref_h0), ref_reliable, ref_confidence)
    dut_phase0 = _segmented_unwrap(np.angle(dut_h0), dut_reliable, dut_confidence)
    ref_phase_common = _interpolate_segments(ref_f, ref_phase0, frequency_hz)
    dut_phase_common = _interpolate_segments(dut_f, dut_phase0, frequency_hz)

    numeric_ratio_floor = _NUMERIC_RESPONSE_FLOOR_RATIO
    ref_threshold = max(float(np.max(ref_mag_common)), tiny) * numeric_ratio_floor
    dut_threshold = max(float(np.max(dut_mag_common)), tiny) * numeric_ratio_floor
    ref_valid = ref_mag_common >= ref_threshold
    dut_valid = dut_mag_common >= dut_threshold
    reliable = (
        np.isfinite(ref_phase_common)
        & np.isfinite(dut_phase_common)
        & ref_valid
        & dut_valid
    )
    band_mask = (
        (frequency_hz >= settings.band_low_hz)
        & (frequency_hz <= settings.band_high_hz)
    )
    if settings.mode in {"magnitude", "both"} and np.any(band_mask & ~dut_valid):
        raise ValueError(
            "补偿频带内的待补偿脉冲响应为零，响应比无法计算；请缩小或移动补偿频带"
        )
    if settings.mode in {"phase", "both"} and np.any(band_mask & ~reliable):
        raise ValueError(
            "补偿频带内存在无法解析相位的频点；请缩小或移动补偿频带"
        )

    delta_t0_s = float(dut_pulse.time_s[0] - reference_pulse.time_s[0])
    reference_phase = ref_phase_common - 2.0 * np.pi * frequency_hz * reference_pulse.time_s[0]
    dut_phase = dut_phase_common - 2.0 * np.pi * frequency_hz * dut_pulse.time_s[0]
    phase_difference = ref_phase_common - dut_phase_common + 2.0 * np.pi * frequency_hz * delta_t0_s
    phase_difference[~reliable] = np.nan

    normalized_ref = ref_mag_common / max(float(np.max(ref_mag_common)), tiny)
    normalized_dut = dut_mag_common / max(float(np.max(dut_mag_common)), tiny)
    fit_weights = np.minimum(normalized_ref, normalized_dut) ** 2
    fit_mask = (
        reliable
        & (frequency_hz >= settings.phase_fit_low_hz)
        & (frequency_hz <= settings.phase_fit_high_hz)
    )
    if settings.mode == "magnitude":
        slope = 0.0
    else:
        slope = fit_linear_phase_slope(
            frequency_hz, phase_difference, fit_weights, fit_mask
        )
    # 这里记录拟合的 slope*f；只有 detrend_phase 打开时才从补偿相位中去除。
    # 每个相位岛的独立截距只用于消去干扰，不会被伪造成单一趋势线。
    phase_trend = slope * frequency_hz
    if settings.detrend_phase:
        phase_used_unanchored = phase_difference - slope * frequency_hz
    else:
        phase_used_unanchored = phase_difference.copy()
    phase_used = _anchor_phase_islands(phase_used_unanchored, reliable, fit_weights)

    magnitude_difference_db = 20.0 / np.log(10.0) * (ref_log_common - dut_log_common)

    log_amplitude = np.zeros_like(frequency_hz)
    applied_phase = np.zeros_like(frequency_hz)
    if settings.mode in {"magnitude", "both"}:
        log_amplitude[band_mask] = (
            ref_log_common[band_mask] - dut_log_common[band_mask]
        )
    if settings.mode in {"phase", "both"}:
        applied_phase[band_mask] = phase_used[band_mask]
    with np.errstate(over="ignore", invalid="ignore"):
        correction = np.exp(log_amplitude + 1j * applied_phase)
    if not np.all(np.isfinite(correction)):
        raise ValueError("补偿频带内的响应比超出浮点数值范围，请缩小或移动补偿频带")
    if settings.mode == "magnitude":
        correction[band_mask & ~ref_valid] = 0.0 + 0.0j
    correction[band_mask] = _apply_compensation_safety(
        correction[band_mask],
        frequency_hz[band_mask],
        settings,
        domain_high_hz=float(application_domain_high_hz),
    )
    correction[0] = np.copysign(abs(correction[0]), correction[0].real or 1.0) + 0.0j
    correction[-1] = np.copysign(abs(correction[-1]), correction[-1].real or 1.0) + 0.0j

    return ResponseAnalysis(
        frequency_hz=frequency_hz,
        reference_magnitude_db=20.0 * np.log10(np.maximum(ref_mag_common, tiny)),
        dut_magnitude_db=20.0 * np.log10(np.maximum(dut_mag_common, tiny)),
        reference_phase_rad=reference_phase,
        dut_phase_rad=dut_phase,
        magnitude_difference_db=magnitude_difference_db,
        phase_difference_rad=phase_difference,
        phase_trend_rad=phase_trend,
        phase_after_optional_detrend_rad=phase_used,
        reliable_mask=reliable,
        correction_ideal=correction,
        phase_detrend_slope_rad_per_hz=slope,
        estimated_relative_delay_s=slope / (2.0 * np.pi),
        settings=settings,
    )


def _pulse_response_on_uniform_frequencies(
    pulse: TimeSeries,
    frequency_hz: FloatArray,
    settings: CompensationSettings,
    *,
    reference_peak: float,
) -> tuple[ComplexArray, FloatArray, NDArray[np.bool_]]:
    """在目标 DFT 的连续频点上直接计算一份拟合脉冲响应。

    显示分析网格可以比目标数据的 DFT 网格更粗，因此不能把显示曲线再次插值后当作
    实际补偿响应。这里使用 CZT 在目标频点直接求有限记录的 DTFT；接近谱零点时再用
    多项式 Horner 求值复核，并以保守前向误差界避免把解析零点误判成可逆小量。
    """

    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
        raise ValueError("实际补偿频率轴必须是一维有限非空数组")
    values = np.asarray(pulse.values[:, 0], dtype=np.float64)
    if settings.taper_alpha > 0.0:
        values = values * signal.windows.tukey(values.size, alpha=settings.taper_alpha)
    # 一次复数 Horner 迭代最多包含 4 次实乘、2 次实加和复系数累加；用 8 次
    # 舍入作保守上界。gamma_k*l1 给出直接求值的绝对前向误差界，使长延迟相消
    # 不会因为固定 eps 门限过窄而被误判为可逆。
    floating_epsilon = np.finfo(np.float64).eps
    rounding_operations = 8 * max(values.size - 1, 1)
    accumulated_roundoff = rounding_operations * floating_epsilon
    if accumulated_roundoff >= 1.0:
        raise ValueError("拟合脉冲记录过长，无法建立稳定的频响求值误差界")
    horner_gamma = accumulated_roundoff / (1.0 - accumulated_roundoff)
    coefficient_l1 = float(np.sum(np.abs(values), dtype=np.longdouble))
    direct_evaluation_error = (
        horner_gamma * coefficient_l1 / pulse.sample_rate_hz
    )

    def direct_dtft(selected_frequency_hz: FloatArray) -> ComplexArray:
        """用本机 ``longdouble`` 可提供的最高精度复核相消敏感频点。"""

        normalized_frequency = np.asarray(
            selected_frequency_hz / pulse.sample_rate_hz,
            dtype=np.longdouble,
        )
        z = np.exp(-2j * np.longdouble(np.pi) * normalized_frequency)
        direct = np.polynomial.polynomial.polyval(
            z,
            values.astype(np.longdouble, copy=False),
        )
        return np.asarray(direct, dtype=np.complex128) / pulse.sample_rate_hz

    if frequencies.size == 1:
        response = direct_dtft(frequencies)
    else:
        spacing_hz = float(frequencies[1] - frequencies[0])
        if spacing_hz <= 0.0 or not np.allclose(
            np.diff(frequencies),
            spacing_hz,
            rtol=1.0e-10,
            atol=max(abs(spacing_hz) * 1.0e-12, np.finfo(np.float64).tiny),
        ):
            raise ValueError("实际补偿频率轴必须严格递增且等间隔")
        start_hz = float(frequencies[0])
        response = signal.czt(
            values,
            m=frequencies.size,
            w=np.exp(-2j * np.pi * spacing_hz / pulse.sample_rate_hz),
            a=np.exp(2j * np.pi * start_hz / pulse.sample_rate_hz),
        )

    # 与显示分析一致，连续时间频响包含 dt=1/fs 标度；后续的峰值、零点门限和
    # 直接复核必须在同一物理量纲内比较。
    if frequencies.size > 1:
        response = np.asarray(response, dtype=np.complex128) / pulse.sample_rate_hz

    if frequencies.size > 1:
        # CZT 对普通频点足够精确，但在两个大数相消的谱零点附近会留下约 1e-12
        # 量级的旋转误差。只重算低幅度候选点，既保留速度，又能区分严格零点和
        # 用户明确允许的有限小响应（例如 1e-9）。
        provisional_magnitude = np.abs(response)
        scale = max(float(reference_peak), float(np.max(provisional_magnitude)))
        refine_mask = (
            provisional_magnitude <= scale * _DIRECT_RESPONSE_REFINEMENT_RATIO
        )
        if np.any(refine_mask):
            response[refine_mask] = direct_dtft(frequencies[refine_mask])

    magnitude = np.abs(response)
    peak = max(float(reference_peak), float(np.max(magnitude)), np.finfo(np.float64).tiny)
    numeric_threshold = max(
        peak * _NUMERIC_RESPONSE_FLOOR_RATIO,
        direct_evaluation_error,
    )
    return response, magnitude, magnitude >= numeric_threshold


def apply_frequency_correction(
    values: FloatArray,
    sample_rate_hz: float,
    analysis: ResponseAnalysis,
    *,
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    _warnings: list[str] | None = None,
) -> FloatArray:
    """对待补偿数据执行 ``FFT → 乘补偿响应 → IFFT``。

    信号先在首尾各镜像延拓一份记录，再在延拓记录的每个带内 DFT 频点直接计算
    ``H_ref/H_dut``，而不是从较粗的显示分析网格插值。最后取回中间原记录，避免
    把末端循环回卷到开头。
    """

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("待补偿信号采样率必须是正的有限值")
    array = np.asarray(values, dtype=np.float64)
    one_dimensional = array.ndim == 1
    if one_dimensional:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 8 or not np.all(np.isfinite(array)):
        raise ValueError("待补偿信号必须是至少 8 点的有限一维或二维数组")
    if analysis.settings.band_high_hz > 0.5 * sample_rate_hz:
        raise ValueError("补偿频带超过待补偿信号 Nyquist")

    samples = array.shape[0]
    padding = samples - 1
    extended_samples = 3 * samples - 2
    rfft_bins = extended_samples // 2 + 1
    # 与 ``np.fft.rfftfreq(E, d=1/Fs)`` 保持相同的运算顺序。直接写 Fs/E
    # 在端点刚好等于某个 bin 时可能差 1 ULP，从而错误多纳入或漏掉一个频点。
    frequency_step_hz = 1.0 / (extended_samples * (1.0 / sample_rate_hz))

    # 与 np.fft.rfftfreq 的 ``arange * step`` 定义保持一致，但只计算带内切片。
    # 初始 ceil/floor 后再用真实浮点频率修正边界，避免端点恰落 bin 时的舍入偏差。
    low_index = max(
        0,
        int(np.ceil(analysis.settings.band_low_hz / frequency_step_hz)),
    )
    high_index = min(
        rfft_bins - 1,
        int(np.floor(analysis.settings.band_high_hz / frequency_step_hz)),
    )
    while low_index > 0 and (low_index - 1) * frequency_step_hz >= analysis.settings.band_low_hz:
        low_index -= 1
    while low_index < rfft_bins and low_index * frequency_step_hz < analysis.settings.band_low_hz:
        low_index += 1
    while (
        high_index + 1 < rfft_bins
        and (high_index + 1) * frequency_step_hz
        <= analysis.settings.band_high_hz
    ):
        high_index += 1
    while high_index >= 0 and high_index * frequency_step_hz > analysis.settings.band_high_hz:
        high_index -= 1
    if low_index > high_index:
        raise ValueError(
            "待补偿记录的 DFT 频率分辨率不足，补偿频带内没有可应用的频点；"
            "请加长记录或扩大补偿频带"
        )
    band_slice = slice(low_index, high_index + 1)
    band_frequency_hz = (
        np.arange(low_index, high_index + 1, dtype=np.float64) * frequency_step_hz
    )
    reference_peak = 10.0 ** (
        float(np.max(analysis.reference_magnitude_db)) / 20.0
    )
    dut_peak = 10.0 ** (float(np.max(analysis.dut_magnitude_db)) / 20.0)
    ref_response, ref_magnitude, ref_valid = _pulse_response_on_uniform_frequencies(
        reference_pulse,
        band_frequency_hz,
        analysis.settings,
        reference_peak=reference_peak,
    )
    if analysis.settings.mode == "magnitude":
        del ref_response
    dut_response, dut_magnitude, dut_valid = _pulse_response_on_uniform_frequencies(
        dut_pulse,
        band_frequency_hz,
        analysis.settings,
        reference_peak=dut_peak,
    )
    if analysis.settings.mode == "magnitude":
        del dut_response
    if analysis.settings.mode in {"magnitude", "both"} and np.any(~dut_valid):
        raise ValueError(
            "补偿频带内的待补偿脉冲响应为零，响应比无法计算；请缩小或移动补偿频带"
        )
    if analysis.settings.mode in {"phase", "both"} and np.any(
        ~(ref_valid & dut_valid)
    ):
        raise ValueError(
            "补偿频带内存在无法解析相位的频点；请缩小或移动补偿频带"
        )

    band_correction = np.ones(band_frequency_hz.size, dtype=np.complex128)
    if analysis.settings.mode in {"magnitude", "both"}:
        band_correction *= ref_magnitude / dut_magnitude
        if analysis.settings.mode == "magnitude":
            band_correction[~ref_valid] = 0.0 + 0.0j
    if analysis.settings.mode in {"phase", "both"}:
        delta_t0_s = float(dut_pulse.time_s[0] - reference_pulse.time_s[0])
        phase_rad = np.angle(ref_response * np.conj(dut_response))
        phase_rad += 2.0 * np.pi * band_frequency_hz * delta_t0_s
        if analysis.settings.detrend_phase:
            phase_rad -= (
                analysis.phase_detrend_slope_rad_per_hz * band_frequency_hz
            )
        band_correction *= np.exp(1j * phase_rad)
        del phase_rad, ref_response, dut_response
    if not np.all(np.isfinite(band_correction)):
        raise ValueError("补偿频带内的响应比超出浮点数值范围，请缩小或移动补偿频带")
    if (
        _warnings is not None
        and analysis.settings.maximum_gain_db is not None
        and analysis.settings.mode in {"magnitude", "both"}
    ):
        requested_peak_db = 20.0 * np.log10(
            max(float(np.max(np.abs(band_correction))), np.finfo(np.float64).tiny)
        )
        _record_gain_limit_warning(
            _warnings,
            requested_peak_db,
            analysis.settings.maximum_gain_db,
        )
    band_correction = _apply_compensation_safety(
        band_correction,
        band_frequency_hz,
        analysis.settings,
        domain_high_hz=0.5 * sample_rate_hz,
    )

    def project_real_endpoint(
        correction: ComplexArray,
        index: int,
        label: str,
    ) -> None:
        """只投影数值噪声；拒绝实值 RFFT 端点无法表达的真实复相位。"""

        value = correction[index]
        representability_tolerance = (
            max(abs(value), np.finfo(np.float64).tiny)
            * _DIRECT_RESPONSE_REFINEMENT_RATIO
        )
        if abs(value.imag) > representability_tolerance:
            raise ValueError(
                f"目标 {label} 频点需要非实补偿，实值时域数据无法表示该相位；"
                "请调整补偿频带以排除该端点"
            )
        correction[index] = (
            np.copysign(abs(value), value.real or 1.0) + 0.0j
        )

    if low_index == 0:
        project_real_endpoint(band_correction, 0, "DC")
    if extended_samples % 2 == 0 and high_index == rfft_bins - 1:
        project_real_endpoint(band_correction, -1, "Nyquist")

    # 大数组直到带内 CZT 完成后才分配；频谱只改带内切片，避免全频 correction
    # 和 ``spectrum * correction`` 两个复数临时数组同时常驻。
    del ref_magnitude, ref_valid, dut_magnitude, dut_valid, band_frequency_hz
    extended = np.pad(array, ((padding, padding), (0, 0)), mode="reflect")
    spectrum = rfft(extended, axis=0, overwrite_x=True)
    del extended
    spectrum[band_slice] *= band_correction[:, None]
    del band_correction
    filtered = irfft(
        spectrum,
        n=extended_samples,
        axis=0,
        overwrite_x=True,
    )
    del spectrum
    # 必须复制裁剪结果；直接返回视图会让 3N-2 长的 IFFT 缓冲一直存活。
    if one_dimensional:
        return np.array(filtered[padding : padding + samples, 0], copy=True)
    return np.array(filtered[padding : padding + samples], copy=True, order="C")


def compare_pulses(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: CompensationSettings,
    *,
    application_domain_high_hz: float | None = None,
) -> PulseComparison:
    """比较两份拟合脉冲，不要求也不处理待补偿数据。"""

    analysis = analyze_responses(
        reference_pulse,
        dut_pulse,
        settings,
        application_domain_high_hz=application_domain_high_hz,
    )
    warnings: list[str] = []
    if settings.maximum_gain_db is not None and settings.mode in {"magnitude", "both"}:
        band = (
            (analysis.frequency_hz >= settings.band_low_hz)
            & (analysis.frequency_hz <= settings.band_high_hz)
        )
        requested_peak_db = (
            float(np.max(analysis.magnitude_difference_db[band]))
            if np.any(band)
            else float("-inf")
        )
        if requested_peak_db > settings.maximum_gain_db + 1.0e-9:
            _record_gain_limit_warning(
                warnings,
                requested_peak_db,
                settings.maximum_gain_db,
            )
    return PulseComparison(
        reference_pulse=reference_pulse,
        dut_pulse=dut_pulse,
        analysis=analysis,
        warnings=tuple(warnings),
    )


def run_compensation(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    input_signal: TimeSeries,
    settings: CompensationSettings,
) -> CompensationRun:
    """执行频响分析并在目标信号频域中直接应用补偿。"""

    if settings.band_high_hz > input_signal.nyquist_hz:
        raise ValueError("补偿频带超过待补偿信号 Nyquist")
    # 来源格式不参与预算：CSV 与 Keysight BIN 都已归一化为同一 TimeSeries，必须在
    # compare_pulses/CZT 和目标镜像数组分配之前通过同一内存门禁。
    _preflight_compensation_memory(reference_pulse, dut_pulse, input_signal, settings)
    comparison = compare_pulses(
        reference_pulse,
        dut_pulse,
        settings,
        application_domain_high_hz=input_signal.nyquist_hz,
    )
    analysis = comparison.analysis
    warnings = list(comparison.warnings)
    output = apply_frequency_correction(
        input_signal.values,
        input_signal.sample_rate_hz,
        analysis,
        reference_pulse=reference_pulse,
        dut_pulse=dut_pulse,
        _warnings=warnings,
    )
    return CompensationRun.from_owned_output(
        reference_pulse=reference_pulse,
        dut_pulse=dut_pulse,
        input_signal=input_signal,
        output_values=output,
        analysis=analysis,
        warnings=tuple(warnings),
    )
