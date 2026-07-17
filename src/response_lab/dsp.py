"""频响比较、相位去斜与直接频域补偿。

约定：真实时域信号使用 ``rfft/irfft``；频率使用 Hz，相位内部使用 rad，幅度 dB
使用电压/幅度定义 ``20*log10``。两份脉冲允许采样率和点数不同，它们分别变换后
插值到公共物理频率轴，不会通过数组下标强行对齐。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.fft import next_fast_len

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
        threshold = max(np.finfo(np.float64).tiny, peak * 64.0 * np.finfo(np.float64).eps)
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
) -> CompensationSettings:
    """根据两份脉冲的共同幅度谱候选区间生成可直接运行的频带设置。

    自动频带只选择两份归一化幅度都不低于 -20 dB 的最长连续区间，并在区间
    两端留出裕量。这样默认设置会随输入采样率和脉冲带宽缩放，也不会把公共
    Nyquist 误当成工程有效带宽。第三份待补偿数据可通过 ``maximum_frequency_hz``
    进一步限制上限。-20 dB 是确定性的幅度启发式，不是噪声或相位可信度估计；
    输入脉冲应已去除直流基线，用户仍需检查拟合带内的相位质量。
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

    return replace(
        template,
        band_low_hz=usable_low_hz + 0.02 * usable_span_hz,
        band_high_hz=usable_low_hz + 0.95 * usable_span_hz,
        phase_fit_low_hz=usable_low_hz + 0.08 * usable_span_hz,
        phase_fit_high_hz=usable_low_hz + 0.90 * usable_span_hz,
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
        raise ValueError("相位观察频段过窄，无法估计线性斜率")
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


def analyze_responses(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: CompensationSettings,
) -> ResponseAnalysis:
    """比较两份拟合脉冲并构造 ``H_ref / H_dut`` 补偿响应。

    补偿方向固定为 ``reference / dut``。默认先在用户观察带内拟合相位斜率，再从
    实际补偿相位中减去该斜率，因此 CSV 起点或脉冲相对时延不会移动最终信号。
    """

    common_nyquist_hz = min(reference_pulse.nyquist_hz, dut_pulse.nyquist_hz)
    if settings.band_high_hz > common_nyquist_hz:
        raise ValueError(
            f"补偿上限 {settings.band_high_hz:g} Hz 超过两份脉冲公共 Nyquist "
            f"{common_nyquist_hz:g} Hz"
        )
    if settings.mode != "magnitude" and settings.phase_fit_high_hz > common_nyquist_hz:
        raise ValueError("相位观察频带超过两份脉冲公共 Nyquist")

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

    numeric_ratio_floor = 64.0 * np.finfo(np.float64).eps
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
    # 实际去除的只有线性时延项 slope*f；每个相位岛的独立截距不是单一趋势线。
    phase_trend = slope * frequency_hz
    if settings.remove_relative_delay:
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
        delay_removed_phase_rad=phase_used,
        reliable_mask=reliable,
        correction_ideal=correction,
        estimated_dut_delay_s=slope / (2.0 * np.pi),
        settings=settings,
    )


def apply_frequency_correction(
    values: FloatArray,
    sample_rate_hz: float,
    analysis: ResponseAnalysis,
) -> FloatArray:
    """对待补偿数据执行 ``FFT → 乘补偿响应 → IFFT``。

    信号先在首尾各镜像延拓一份记录，再把分析差异插值到延拓记录的 DFT 频点，
    直接相乘并反变换。最后取回中间原记录，避免把末端循环回卷到开头。
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
    extended = np.pad(array, ((padding, padding), (0, 0)), mode="reflect")
    frequency_hz = np.fft.rfftfreq(extended.shape[0], d=1.0 / sample_rate_hz)
    band_mask = (
        (frequency_hz >= analysis.settings.band_low_hz)
        & (frequency_hz <= analysis.settings.band_high_hz)
    )
    correction = np.ones(frequency_hz.size, dtype=np.complex128)
    if analysis.settings.mode in {"magnitude", "both"}:
        source_log_amplitude = (
            analysis.magnitude_difference_db * np.log(10.0) / 20.0
        )
        log_amplitude = np.interp(
            frequency_hz,
            analysis.frequency_hz,
            source_log_amplitude,
        )
        correction[band_mask] *= np.exp(log_amplitude[band_mask])
    if analysis.settings.mode in {"phase", "both"}:
        phase_rad = _interpolate_segments(
            analysis.frequency_hz,
            analysis.delay_removed_phase_rad,
            frequency_hz,
        )
        if np.any(~np.isfinite(phase_rad[band_mask])):
            raise ValueError("补偿频带内存在无法插值的相位频点；请缩小或移动补偿频带")
        correction[band_mask] *= np.exp(1j * phase_rad[band_mask])
    correction[0] = (
        np.copysign(abs(correction[0]), correction[0].real or 1.0) + 0.0j
    )
    if extended.shape[0] % 2 == 0:
        correction[-1] = (
            np.copysign(abs(correction[-1]), correction[-1].real or 1.0) + 0.0j
        )

    spectrum = np.fft.rfft(extended, axis=0)
    filtered = np.fft.irfft(
        spectrum * correction[:, None],
        n=extended.shape[0],
        axis=0,
    )
    output = filtered[padding : padding + samples]
    return output[:, 0] if one_dimensional else output


def compare_pulses(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: CompensationSettings,
) -> PulseComparison:
    """比较两份拟合脉冲，不要求也不处理待补偿数据。"""

    return PulseComparison(
        reference_pulse=reference_pulse,
        dut_pulse=dut_pulse,
        analysis=analyze_responses(reference_pulse, dut_pulse, settings),
        warnings=(),
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
    comparison = compare_pulses(reference_pulse, dut_pulse, settings)
    analysis = comparison.analysis
    output = apply_frequency_correction(
        input_signal.values,
        input_signal.sample_rate_hz,
        analysis,
    )
    return CompensationRun(
        reference_pulse=reference_pulse,
        dut_pulse=dut_pulse,
        input_signal=input_signal,
        output_values=output,
        analysis=analysis,
        warnings=(),
    )
