"""ResponseLab 的无界面领域模型。

所有内部时间使用秒、频率使用 Hz、相位使用 rad、幅度比使用电压/幅度定义的 dB。
模型在创建时完成形状与单位验证，让 GUI、测试和批处理入口共享同一组约束。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
WaveformArray = NDArray[np.float32] | NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
CompensationMode = Literal["magnitude", "phase", "both"]
ApplicationStrategy = Literal["auto", "exact", "streaming"]
BoundaryMode = Literal["zero", "reflect"]
PULSE_SAMPLE_RATE_TOLERANCE_PPM = 100.0
_UNIFORM_TIME_RTOL = 1.0e-5
_MAX_TIME_AXIS_PHASE_ERROR_RAD = np.deg2rad(1.0)
_UNIFORM_TIME_VALIDATION_CHUNK_SAMPLES = 131_072


def validate_cross_pulse_sample_rates(
    reference_rate_hz: float,
    dut_rate_hz: float,
    *,
    subject: str,
) -> None:
    """允许小幅导出舍入差异，但拒绝会改变离散频率网格的采样率偏差。"""

    mismatch_ppm = (
        abs(float(dut_rate_hz) - float(reference_rate_hz))
        / float(reference_rate_hz)
        * 1.0e6
    )
    if mismatch_ppm <= PULSE_SAMPLE_RATE_TOLERANCE_PPM:
        return
    raise ValueError(
        f"{subject}采样率差异 {mismatch_ppm:.3f} ppm，"
        f"超过允许的 {PULSE_SAMPLE_RATE_TOLERANCE_PPM:g} ppm；"
        "工具不会静默重采样"
    )


def _all_finite_in_chunks(values: np.ndarray) -> bool:
    """验证任意形状数值数组，临时布尔量不随大记录长度增长。"""

    flat = values.reshape(-1)
    chunk_elements = 1_048_576
    return all(
        np.all(np.isfinite(flat[start : start + chunk_elements]))
        for start in range(0, flat.size, chunk_elements)
    )


def _readonly_float_array(values: object, *, dimensions: int | None = None) -> FloatArray:
    """复制为有限的 float64 只读数组，防止缓存结果被界面意外修改。"""

    array = np.array(values, dtype=np.float64, copy=True)
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"数组必须是 {dimensions} 维，实际为 {array.ndim} 维")
    if not _all_finite_in_chunks(array):
        raise ValueError("数据包含 NaN 或 Inf")
    array.setflags(write=False)
    return array


def _readonly_waveform_array(
    values: object,
    *,
    dimensions: int | None = None,
) -> WaveformArray:
    """波形只保留 float32/float64，BIN 不再被无条件扩大为 float64。"""

    source = np.asarray(values)
    dtype = np.float32 if source.dtype == np.dtype(np.float32) else np.float64
    array = np.array(source, dtype=dtype, copy=True, order="C")
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"数组必须是 {dimensions} 维，实际为 {array.ndim} 维")
    if not _all_finite_in_chunks(array):
        raise ValueError("数据包含 NaN 或 Inf")
    array.setflags(write=False)
    return array


def _readonly_complex_array(values: object) -> ComplexArray:
    array = np.array(values, dtype=np.complex128, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("复数频响必须是一维有限数组")
    array.setflags(write=False)
    return array


def _readonly_float_array_allow_nan(values: object) -> FloatArray:
    """复制为只读一维数组，允许用 NaN 标记不可信相位。"""

    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or np.any(np.isinf(array)):
        raise ValueError("相位数组必须是一维且不能包含 Inf")
    array.setflags(write=False)
    return array


def _readonly_bool_array(values: object) -> BoolArray:
    array = np.array(values, dtype=np.bool_, copy=True)
    if array.ndim != 1:
        raise ValueError("可信掩码必须是一维")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TimeSeries:
    """等间隔真实时域数据。

    ``values`` 总是 ``(samples, channels)``；即使只有一个通道也保留第二维。
    ``time_scale_to_s`` 用于导出时恢复用户选择的原始时间单位。
    """

    time_s: FloatArray
    values: WaveformArray
    sample_rate_hz: float
    source_path: Path | None = None
    source_format: Literal["csv", "bin", "memory"] = "memory"
    time_unit: str = "s"
    time_scale_to_s: float = 1.0
    value_columns: tuple[int, ...] = (1,)
    source_metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_uniform_samples(
        cls,
        *,
        values: object,
        sample_rate_hz: float,
        time_origin_s: float,
        time_increment_s: float,
        source_path: Path | None = None,
        source_format: Literal["csv", "bin", "memory"] = "memory",
        time_unit: str = "s",
        time_scale_to_s: float = 1.0,
        value_columns: tuple[int, ...] = (1,),
        source_metadata: Mapping[str, object] | None = None,
    ) -> Self:
        """从格式元数据定义的等间隔采样构造模型，避免重复 O(N) 大数组。

        该入口仍验证采样值、时间原点、实际相邻间隔及采样率的一致性；它以有界分块
        替代完整 ``diff``、``median`` 和第二条理想时间轴。
        """

        if isinstance(time_origin_s, (bool, np.bool_)) or not np.isfinite(time_origin_s):
            raise ValueError("时间原点必须是有限秒数")
        if (
            isinstance(time_increment_s, (bool, np.bool_))
            or not np.isfinite(time_increment_s)
            or time_increment_s <= 0.0
        ):
            raise ValueError("时间间隔必须是正的有限秒数")
        if (
            isinstance(sample_rate_hz, (bool, np.bool_))
            or not np.isfinite(sample_rate_hz)
            or sample_rate_hz <= 0.0
        ):
            raise ValueError("采样率必须是正的有限值")
        inferred_rate_hz = 1.0 / float(time_increment_s)
        if not np.isclose(
            float(sample_rate_hz),
            inferred_rate_hz,
            rtol=_UNIFORM_TIME_RTOL,
            atol=0.0,
        ):
            raise ValueError("采样率与时间间隔不一致")
        if not np.isfinite(time_scale_to_s) or time_scale_to_s <= 0.0:
            raise ValueError("时间单位换算必须为正数")

        value_array = _readonly_waveform_array(values)
        if value_array.ndim == 1:
            value_array = value_array[:, None]
            value_array.setflags(write=False)
        if value_array.ndim != 2 or value_array.shape[0] < 8:
            raise ValueError("values 必须是至少 8 点的 (样本数, 通道数) 数组")

        sample_count = int(value_array.shape[0])
        last_time_s = float(time_origin_s) + (sample_count - 1) * float(time_increment_s)
        if not np.isfinite(last_time_s):
            raise ValueError("时间轴终点超出 float64 有限范围")
        time_array = np.arange(sample_count, dtype=np.float64)
        time_array *= float(time_increment_s)
        time_array += float(time_origin_s)
        # 正增量且有限终点保证中间理论值有限；但 float64 舍入仍可能只在轴内
        # 部分位置制造重复或非等间隔。分块验证所有实际相邻间隔，将 float64
        # 临时量限制在约 1 MiB，避免通用构造路径的完整 diff 和理想时间轴。
        for start in range(
            0,
            sample_count - 1,
            _UNIFORM_TIME_VALIDATION_CHUNK_SAMPLES,
        ):
            stop = min(
                sample_count - 1,
                start + _UNIFORM_TIME_VALIDATION_CHUNK_SAMPLES,
            )
            intervals_s = time_array[start + 1 : stop + 1] - time_array[start:stop]
            if np.any(intervals_s <= 0.0):
                raise ValueError("float64 时间轴无法保持严格递增")
            intervals_s -= float(time_increment_s)
            np.abs(intervals_s, out=intervals_s)
            if float(np.max(intervals_s)) > (_UNIFORM_TIME_RTOL * float(time_increment_s)):
                raise ValueError("float64 时间轴无法保持与采样率一致的等间隔")
        time_array.setflags(write=False)

        instance = object.__new__(cls)
        object.__setattr__(instance, "time_s", time_array)
        object.__setattr__(instance, "values", value_array)
        object.__setattr__(instance, "sample_rate_hz", float(sample_rate_hz))
        object.__setattr__(instance, "source_path", Path(source_path) if source_path else None)
        object.__setattr__(instance, "source_format", source_format)
        object.__setattr__(instance, "time_unit", time_unit)
        object.__setattr__(instance, "time_scale_to_s", float(time_scale_to_s))
        object.__setattr__(instance, "value_columns", tuple(value_columns))
        object.__setattr__(
            instance,
            "source_metadata",
            MappingProxyType(dict(source_metadata or {})),
        )
        return instance

    def __post_init__(self) -> None:
        time_s = _readonly_float_array(self.time_s, dimensions=1)
        values = _readonly_waveform_array(self.values)
        if values.ndim == 1:
            values = values[:, None]
            values.setflags(write=False)
        if values.ndim != 2 or values.shape[0] != time_s.size:
            raise ValueError("values 必须是 (样本数, 通道数)，并与时间长度一致")
        if time_s.size < 8:
            raise ValueError("至少需要 8 个样本")
        intervals_s = np.diff(time_s)
        if np.any(intervals_s <= 0.0):
            raise ValueError("时间列必须严格递增")
        median_interval_s = float(np.median(intervals_s))
        relative_deviation = float(
            np.max(np.abs(intervals_s - median_interval_s)) / median_interval_s
        )
        if relative_deviation > _UNIFORM_TIME_RTOL:
            raise ValueError("时域数据必须等间隔；请先在加载阶段重采样")
        ideal_time_s = time_s[0] + np.arange(time_s.size, dtype=np.float64) * median_interval_s
        maximum_time_residual_s = float(np.max(np.abs(time_s - ideal_time_s)))
        nyquist_from_time_hz = 0.5 / median_interval_s
        time_axis_phase_error_rad = 2.0 * np.pi * nyquist_from_time_hz * maximum_time_residual_s
        if time_axis_phase_error_rad > _MAX_TIME_AXIS_PHASE_ERROR_RAD:
            raise ValueError("时间轴累计残差在 Nyquist 处超过 1° 相位误差；请显式重采样")
        if (
            isinstance(self.sample_rate_hz, (bool, np.bool_))
            or not np.isfinite(self.sample_rate_hz)
            or self.sample_rate_hz <= 0.0
        ):
            raise ValueError("采样率必须是正的有限值")
        inferred_rate_hz = 1.0 / median_interval_s
        if not np.isclose(
            float(self.sample_rate_hz),
            inferred_rate_hz,
            rtol=_UNIFORM_TIME_RTOL,
            atol=0.0,
        ):
            raise ValueError("采样率与时间列中位间隔不一致")
        if not np.isfinite(self.time_scale_to_s) or self.time_scale_to_s <= 0.0:
            raise ValueError("时间单位换算必须为正数")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_metadata", MappingProxyType(dict(self.source_metadata)))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def samples(self) -> int:
        return int(self.time_s.size)

    @property
    def channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def nyquist_hz(self) -> float:
        return 0.5 * float(self.sample_rate_hz)


@dataclass(frozen=True)
class CompensationSettings:
    """一次频响分析与直接频域补偿的完整参数。"""

    mode: CompensationMode = "both"
    band_low_hz: float = 0.0
    band_high_hz: float = 1.0
    phase_fit_low_hz: float = 0.0
    phase_fit_high_hz: float = 1.0
    detrend_phase: bool = True
    taper_alpha: float = 0.0
    maximum_gain_db: float | None = 20.0
    edge_transition_fraction: float = 0.10
    analysis_points: int = 16385
    application_strategy: ApplicationStrategy = "auto"
    boundary_mode: BoundaryMode = "zero"
    streaming_fft_samples: int = 1_048_576
    streaming_tail_relative_tolerance: float = float(128.0 * np.finfo(np.float32).eps)

    def __post_init__(self) -> None:
        if self.mode not in {"magnitude", "phase", "both"}:
            raise ValueError("补偿模式必须是 magnitude、phase 或 both")
        numeric = (
            self.band_low_hz,
            self.band_high_hz,
            self.phase_fit_low_hz,
            self.phase_fit_high_hz,
            self.taper_alpha,
            self.edge_transition_fraction,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("补偿参数必须是有限值")
        if not 0.0 <= self.band_low_hz < self.band_high_hz:
            raise ValueError("补偿频带必须满足 0 <= low < high")
        if self.mode != "magnitude" and not (0.0 <= self.phase_fit_low_hz < self.phase_fit_high_hz):
            raise ValueError("线性相位拟合频带必须满足 0 <= low < high")
        if not 0.0 <= self.taper_alpha <= 1.0:
            raise ValueError("Tukey alpha 必须位于 0 到 1")
        if self.maximum_gain_db is not None and (
            isinstance(self.maximum_gain_db, (bool, np.bool_))
            or not np.isfinite(self.maximum_gain_db)
            or self.maximum_gain_db < 0.0
        ):
            raise ValueError("最大补偿增益必须是非负有限 dB，或设为 None")
        if not 0.0 <= self.edge_transition_fraction <= 0.5:
            raise ValueError("补偿边缘过渡比例必须位于 0 到 0.5")
        if isinstance(self.analysis_points, (bool, np.bool_)) or not isinstance(
            self.analysis_points, (int, np.integer)
        ):
            raise ValueError("分析频点数必须是整数")
        if self.analysis_points < 257:
            raise ValueError("分析频点至少为 257")
        if self.application_strategy not in {"auto", "exact", "streaming"}:
            raise ValueError("应用策略必须是 auto、exact 或 streaming")
        if self.boundary_mode not in {"zero", "reflect"}:
            raise ValueError("记录边界模式必须是 zero 或 reflect")
        if (
            isinstance(self.streaming_fft_samples, (bool, np.bool_))
            or not isinstance(self.streaming_fft_samples, (int, np.integer))
            or self.streaming_fft_samples < 512
        ):
            raise ValueError("分块 FFT 点数必须是至少 512 的整数")
        if (
            isinstance(self.streaming_tail_relative_tolerance, (bool, np.bool_))
            or not np.isfinite(self.streaming_tail_relative_tolerance)
            or not 0.0 < self.streaming_tail_relative_tolerance <= 1.0e-2
        ):
            raise ValueError("分块尾部相对容差必须位于 0 到 1e-2 之间")


@dataclass(frozen=True)
class ResponseAnalysis:
    """两份拟合脉冲比较后得到的频响差异与补偿响应。

    两个 ``*_magnitude_db`` 保存未经峰值归一化的原始频谱幅度 dB。
    ``estimated_relative_delay_s`` 的正值表示 DUT 比参考脉冲更晚。
    """

    frequency_hz: FloatArray
    reference_magnitude_db: FloatArray
    dut_magnitude_db: FloatArray
    reference_phase_rad: FloatArray
    dut_phase_rad: FloatArray
    magnitude_difference_db: FloatArray
    phase_difference_rad: FloatArray
    phase_trend_rad: FloatArray
    phase_after_optional_detrend_rad: FloatArray
    reliable_mask: BoolArray
    correction_ideal: ComplexArray
    phase_detrend_slope_rad_per_hz: float
    estimated_relative_delay_s: float
    settings: CompensationSettings

    def __post_init__(self) -> None:
        frequency_hz = _readonly_float_array(self.frequency_hz, dimensions=1)
        finite_arrays = {
            "reference_magnitude_db": self.reference_magnitude_db,
            "dut_magnitude_db": self.dut_magnitude_db,
            "magnitude_difference_db": self.magnitude_difference_db,
            "phase_trend_rad": self.phase_trend_rad,
        }
        nullable_arrays = {
            "reference_phase_rad": self.reference_phase_rad,
            "dut_phase_rad": self.dut_phase_rad,
            "phase_difference_rad": self.phase_difference_rad,
            "phase_after_optional_detrend_rad": self.phase_after_optional_detrend_rad,
        }
        converted: dict[str, FloatArray] = {}
        for name, values in finite_arrays.items():
            converted[name] = _readonly_float_array(values, dimensions=1)
        for name, values in nullable_arrays.items():
            converted[name] = _readonly_float_array_allow_nan(values)
        reliable_mask = _readonly_bool_array(self.reliable_mask)
        correction_ideal = _readonly_complex_array(self.correction_ideal)
        expected = frequency_hz.size
        if expected < 257 or np.any(np.diff(frequency_hz) <= 0.0) or frequency_hz[0] < 0.0:
            raise ValueError("分析频率轴必须是至少 257 点的严格递增非负轴")
        if any(array.size != expected for array in converted.values()):
            raise ValueError("所有响应数组必须与频率轴等长")
        if reliable_mask.size != expected or correction_ideal.size != expected:
            raise ValueError("可信掩码和理想补偿必须与频率轴等长")
        for name in nullable_arrays:
            if np.any(~np.isfinite(converted[name][reliable_mask])):
                raise ValueError(f"可信频点上的 {name} 不能是 NaN")
        if not np.isfinite(self.phase_detrend_slope_rad_per_hz):
            raise ValueError("线性相位拟合斜率必须是有限值")
        if not np.isfinite(self.estimated_relative_delay_s):
            raise ValueError("拟合相对时延必须是有限值")
        expected_delay_s = self.phase_detrend_slope_rad_per_hz / (2.0 * np.pi)
        if not np.isclose(
            self.estimated_relative_delay_s,
            expected_delay_s,
            rtol=1.0e-12,
            atol=1.0e-24,
        ):
            raise ValueError("拟合相对时延必须等于去斜斜率除以 2*pi")
        if not isinstance(self.settings, CompensationSettings):
            raise ValueError("分析结果必须绑定 CompensationSettings")
        object.__setattr__(self, "frequency_hz", frequency_hz)
        for name, array in converted.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "reliable_mask", reliable_mask)
        object.__setattr__(self, "correction_ideal", correction_ideal)


@dataclass(frozen=True)
class PulseComparison:
    """只依赖两份拟合脉冲的频响比较结果。"""

    reference_pulse: TimeSeries
    dut_pulse: TimeSeries
    analysis: ResponseAnalysis
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_pulse, TimeSeries) or not isinstance(
            self.dut_pulse, TimeSeries
        ):
            raise ValueError("拟合脉冲比较必须包含两份有效的 TimeSeries")
        if not isinstance(self.analysis, ResponseAnalysis):
            raise ValueError("拟合脉冲比较必须包含有效的频响分析")
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))


@dataclass(frozen=True)
class CompensationRun:
    """可绘图、导出并生成 manifest 的完整一次运行。"""

    reference_pulse: TimeSeries
    dut_pulse: TimeSeries
    input_signal: TimeSeries
    output_values: WaveformArray
    analysis: ResponseAnalysis
    warnings: tuple[str, ...] = field(default_factory=tuple)
    application_method: str = ""
    application_metadata: Mapping[str, object] = field(default_factory=dict)

    @staticmethod
    def _normalized_application_contract(
        analysis: ResponseAnalysis,
        application_method: object,
        application_metadata: Mapping[str, object] | None,
    ) -> tuple[str, Mapping[str, object]]:
        """Derive and validate one boundary-consistent application contract."""

        boundary_mode = analysis.settings.boundary_mode
        exact_method = (
            f"{boundary_mode}_extend_czt_pulse_ratio_rfft_multiply_irfft_crop"
        )
        streaming_method = (
            f"finite_{boundary_mode}_overlap_save_rfft_multiply_irfft"
        )
        method = str(application_method).strip() or exact_method
        if method not in {exact_method, streaming_method}:
            raise ValueError("应用方法必须与补偿设置的记录边界模式一致")
        metadata = dict(application_metadata or {})
        metadata_boundary = metadata.get("boundary_mode")
        if metadata_boundary is not None and metadata_boundary != boundary_mode:
            raise ValueError("应用元数据的记录边界模式必须与补偿设置一致")
        return method, MappingProxyType(metadata)

    @classmethod
    def from_owned_output(
        cls,
        *,
        reference_pulse: TimeSeries,
        dut_pulse: TimeSeries,
        input_signal: TimeSeries,
        output_values: WaveformArray,
        analysis: ResponseAnalysis,
        warnings: tuple[str, ...] = (),
        application_method: str = "",
        application_metadata: Mapping[str, object] | None = None,
    ) -> Self:
        """校验并接管 DSP 新分配的紧凑输出，避免再复制一份大数组。"""

        if not all(
            isinstance(series, TimeSeries) for series in (reference_pulse, dut_pulse, input_signal)
        ):
            raise ValueError("补偿运行的三份输入必须是 TimeSeries")
        if not isinstance(analysis, ResponseAnalysis):
            raise ValueError("补偿运行必须包含有效的频响分析")
        output_array = np.asarray(output_values)
        if (
            output_array.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}
            or output_array.ndim != 2
            or output_array.shape != input_signal.values.shape
            or not output_array.flags.owndata
            or not output_array.flags.c_contiguous
            or not _all_finite_in_chunks(output_array)
        ):
            raise ValueError("DSP 输出必须是自有、连续、有限且与输入同形的 float32/float64 数组")
        output_array.setflags(write=False)
        instance = object.__new__(cls)
        object.__setattr__(instance, "reference_pulse", reference_pulse)
        object.__setattr__(instance, "dut_pulse", dut_pulse)
        object.__setattr__(instance, "input_signal", input_signal)
        object.__setattr__(instance, "output_values", output_array)
        object.__setattr__(instance, "analysis", analysis)
        object.__setattr__(instance, "warnings", tuple(str(item) for item in warnings))
        method, metadata = cls._normalized_application_contract(
            analysis,
            application_method,
            application_metadata,
        )
        object.__setattr__(instance, "application_method", method)
        object.__setattr__(
            instance,
            "application_metadata",
            metadata,
        )
        return instance

    def __post_init__(self) -> None:
        if not all(
            isinstance(series, TimeSeries)
            for series in (self.reference_pulse, self.dut_pulse, self.input_signal)
        ):
            raise ValueError("补偿运行的三份输入必须是 TimeSeries")
        if not isinstance(self.analysis, ResponseAnalysis):
            raise ValueError("补偿运行必须包含有效的频响分析")
        output_values = _readonly_waveform_array(self.output_values, dimensions=2)
        if output_values.shape != self.input_signal.values.shape:
            raise ValueError("补偿输出必须与输入信号形状一致")
        warnings = tuple(str(warning) for warning in self.warnings)
        method, metadata = self._normalized_application_contract(
            self.analysis,
            self.application_method,
            self.application_metadata,
        )
        object.__setattr__(self, "output_values", output_values)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "application_method", method)
        object.__setattr__(
            self,
            "application_metadata",
            metadata,
        )
