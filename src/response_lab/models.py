"""ResponseLab 的无界面领域模型。

所有内部时间使用秒、频率使用 Hz、相位使用 rad、幅度比使用电压/幅度定义的 dB。
模型在创建时完成形状与单位验证，让 GUI、测试和批处理入口共享同一组约束。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
CompensationMode = Literal["magnitude", "phase", "both"]
_UNIFORM_TIME_RTOL = 1.0e-5
_MAX_TIME_AXIS_PHASE_ERROR_RAD = np.deg2rad(1.0)


def _readonly_float_array(values: object, *, dimensions: int | None = None) -> FloatArray:
    """复制为有限的 float64 只读数组，防止缓存结果被界面意外修改。"""

    array = np.array(values, dtype=np.float64, copy=True)
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"数组必须是 {dimensions} 维，实际为 {array.ndim} 维")
    if not np.all(np.isfinite(array)):
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
    values: FloatArray
    sample_rate_hz: float
    source_path: Path | None = None
    source_format: Literal["csv", "bin", "memory"] = "memory"
    time_unit: str = "s"
    time_scale_to_s: float = 1.0
    value_columns: tuple[int, ...] = (1,)
    source_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time_s = _readonly_float_array(self.time_s, dimensions=1)
        values = _readonly_float_array(self.values)
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
        time_axis_phase_error_rad = (
            2.0 * np.pi * nyquist_from_time_hz * maximum_time_residual_s
        )
        if time_axis_phase_error_rad > _MAX_TIME_AXIS_PHASE_ERROR_RAD:
            raise ValueError(
                "时间轴累计残差在 Nyquist 处超过 1° 相位误差；请显式重采样"
            )
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
class BinConfig:
    """无自描述 BIN 的显式解码约定。"""

    sample_rate_hz: float
    dtype: Literal["float32", "float64", "int16", "int32"] = "float32"
    byte_order: Literal["little", "big"] = "little"
    offset_bytes: int = 0
    channels: int = 1
    channel_index: int = 0
    layout: Literal["interleaved", "planar"] = "interleaved"
    scale: float = 1.0
    value_offset: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, (bool, np.bool_))
            or not np.isfinite(self.sample_rate_hz)
            or self.sample_rate_hz <= 0.0
        ):
            raise ValueError("BIN 采样率必须手动输入为正数")
        integer_fields = {
            "offset_bytes": self.offset_bytes,
            "channels": self.channels,
            "channel_index": self.channel_index,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"BIN {name} 必须是整数")
            object.__setattr__(self, name, int(value))
        if self.dtype not in {"float32", "float64", "int16", "int32"}:
            raise ValueError("BIN dtype 必须是 float32、float64、int16 或 int32")
        if self.byte_order not in {"little", "big"}:
            raise ValueError("BIN 字节序必须是 little 或 big")
        if self.layout not in {"interleaved", "planar"}:
            raise ValueError("BIN 布局必须是 interleaved 或 planar")
        if self.offset_bytes < 0:
            raise ValueError("文件头偏移不能为负数")
        if self.channels < 1 or not 0 <= self.channel_index < self.channels:
            raise ValueError("BIN 通道索引超出范围")
        if not np.isfinite(self.scale) or self.scale == 0.0:
            raise ValueError("BIN scale 必须是非零有限值")
        if not np.isfinite(self.value_offset):
            raise ValueError("BIN offset 必须是有限值")


@dataclass(frozen=True)
class CompensationSettings:
    """一次频响分析与直接频域补偿的完整参数。"""

    mode: CompensationMode = "both"
    band_low_hz: float = 0.0
    band_high_hz: float = 1.0
    phase_fit_low_hz: float = 0.0
    phase_fit_high_hz: float = 1.0
    remove_relative_delay: bool = True
    taper_alpha: float = 0.0
    analysis_points: int = 16385

    def __post_init__(self) -> None:
        if self.mode not in {"magnitude", "phase", "both"}:
            raise ValueError("补偿模式必须是 magnitude、phase 或 both")
        numeric = (
            self.band_low_hz,
            self.band_high_hz,
            self.phase_fit_low_hz,
            self.phase_fit_high_hz,
            self.taper_alpha,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("补偿参数必须是有限值")
        if not 0.0 <= self.band_low_hz < self.band_high_hz:
            raise ValueError("补偿频带必须满足 0 <= low < high")
        if self.mode != "magnitude" and not (
            0.0 <= self.phase_fit_low_hz < self.phase_fit_high_hz
        ):
            raise ValueError("相位观察频带必须满足 0 <= low < high")
        if not 0.0 <= self.taper_alpha <= 1.0:
            raise ValueError("Tukey alpha 必须位于 0 到 1")
        if isinstance(self.analysis_points, (bool, np.bool_)) or not isinstance(
            self.analysis_points, (int, np.integer)
        ):
            raise ValueError("分析频点数必须是整数")
        if self.analysis_points < 257:
            raise ValueError("分析频点至少为 257")


@dataclass(frozen=True)
class ResponseAnalysis:
    """两份拟合脉冲比较后得到的频响差异与补偿响应。"""

    frequency_hz: FloatArray
    reference_magnitude_db: FloatArray
    dut_magnitude_db: FloatArray
    reference_phase_rad: FloatArray
    dut_phase_rad: FloatArray
    magnitude_difference_db: FloatArray
    phase_difference_rad: FloatArray
    phase_trend_rad: FloatArray
    delay_removed_phase_rad: FloatArray
    reliable_mask: BoolArray
    correction_ideal: ComplexArray
    estimated_dut_delay_s: float
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
            "delay_removed_phase_rad": self.delay_removed_phase_rad,
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
        if not np.isfinite(self.estimated_dut_delay_s):
            raise ValueError("估计时延必须是有限值")
        if not isinstance(self.settings, CompensationSettings):
            raise ValueError("分析结果必须绑定 CompensationSettings")
        object.__setattr__(self, "frequency_hz", frequency_hz)
        for name, array in converted.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "reliable_mask", reliable_mask)
        object.__setattr__(self, "correction_ideal", correction_ideal)


@dataclass(frozen=True)
class CompensationRun:
    """可绘图、导出并生成 manifest 的完整一次运行。"""

    reference_pulse: TimeSeries
    dut_pulse: TimeSeries
    input_signal: TimeSeries
    output_values: FloatArray
    analysis: ResponseAnalysis
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not all(
            isinstance(series, TimeSeries)
            for series in (self.reference_pulse, self.dut_pulse, self.input_signal)
        ):
            raise ValueError("补偿运行的三份输入必须是 TimeSeries")
        if not isinstance(self.analysis, ResponseAnalysis):
            raise ValueError("补偿运行必须包含有效的频响分析")
        output_values = _readonly_float_array(self.output_values, dimensions=2)
        if output_values.shape != self.input_signal.values.shape:
            raise ValueError("补偿输出必须与输入信号形状一致")
        warnings = tuple(str(warning) for warning in self.warnings)
        object.__setattr__(self, "output_values", output_values)
        object.__setattr__(self, "warnings", warnings)
