"""Headerless CSV and raw BIN import/export for ResponseLab."""

from __future__ import annotations

import hashlib
import operator
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .models import BinConfig, TimeSeries

_TIME_SCALE_TO_S = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "µs": 1.0e-6,
    "μs": 1.0e-6,
    "ns": 1.0e-9,
    "ps": 1.0e-12,
}
_DEFAULT_UNIFORMITY_RTOL = 1.0e-6
_BIN_DTYPE_CODES = {
    "float32": "f4",
    "float64": "f8",
    "int16": "i2",
    "int32": "i4",
}
_SOURCE_HASH_BLOCK_BYTES = 1024 * 1024


def _snapshot_source_file(path: Path) -> tuple[int, str]:
    """Return a stable size/SHA-256 snapshot, rejecting changes during hashing."""

    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(_SOURCE_HASH_BLOCK_BYTES), b""):
            digest.update(block)
            bytes_read += len(block)
        after = os.fstat(handle.fileno())
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or bytes_read != after.st_size
    ):
        raise OSError("源文件在加载快照期间发生变化，请重新选择后再试")
    return bytes_read, digest.hexdigest()


def _confirm_source_snapshot(path: Path, expected: tuple[int, str]) -> None:
    """Ensure parsing and the recorded source snapshot refer to the same bytes."""

    if _snapshot_source_file(path) != expected:
        raise OSError("源文件在加载期间发生变化，已拒绝使用不一致的数据")


def _detect_delimiter(path: Path) -> str | None:
    """Return the most likely supported delimiter; ``None`` means whitespace."""

    with path.open("r", encoding="utf-8-sig") as stream:
        first_line = next((line for line in stream if line.strip()), "")
    if not first_line:
        raise ValueError("CSV 文件为空")
    counts = {delimiter: first_line.count(delimiter) for delimiter in (",", ";", "\t")}
    delimiter = max(counts, key=counts.get)
    return delimiter if counts[delimiter] else None


def _as_column_index(value: object, *, label: str) -> int:
    try:
        index = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label}必须是整数列索引") from error
    if isinstance(value, bool) or index < 0:
        raise ValueError(f"{label}必须是非负整数列索引")
    return index


def _normalize_value_columns(value_columns: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value_columns, (int, np.integer)) and not isinstance(value_columns, bool):
        columns = (_as_column_index(value_columns, label="数值列"),)
    else:
        try:
            columns = tuple(_as_column_index(column, label="数值列") for column in value_columns)
        except TypeError as error:
            raise ValueError("数值列必须是一个索引或索引序列") from error
    if not columns:
        raise ValueError("至少选择一个数值列")
    if len(set(columns)) != len(columns):
        raise ValueError("数值列不能重复选择")
    return columns


def load_csv_timeseries(
    path: str | Path,
    *,
    time_unit: str = "s",
    time_column: int = 0,
    value_columns: int | Sequence[int] = (1,),
    resample_nonuniform: bool = False,
    uniformity_rtol: float = _DEFAULT_UNIFORMITY_RTOL,
    max_resample_relative_deviation: float = 0.05,
) -> TimeSeries:
    """Load a numeric, headerless time-series CSV."""

    source_path = Path(path).resolve()
    source_snapshot = _snapshot_source_file(source_path)
    try:
        scale_to_s = _TIME_SCALE_TO_S[time_unit]
    except (KeyError, TypeError) as error:
        supported = "s、ms、us、µs、ns、ps"
        raise ValueError(f"不支持的 CSV 时间单位；可选 {supported}") from error
    time_column = _as_column_index(time_column, label="时间列")
    selected_columns = _normalize_value_columns(value_columns)
    if time_column in selected_columns:
        raise ValueError("时间列不能同时作为数值列")
    if not np.isfinite(uniformity_rtol) or uniformity_rtol < 0.0:
        raise ValueError("均匀采样相对容差必须是非负有限值")
    if (
        not np.isfinite(max_resample_relative_deviation)
        or max_resample_relative_deviation <= uniformity_rtol
    ):
        raise ValueError("轻微非均匀重采样阈值必须大于均匀采样容差")

    delimiter = _detect_delimiter(source_path)
    table = np.loadtxt(
        source_path,
        delimiter=delimiter,
        ndmin=2,
        encoding="utf-8-sig",
    )
    if table.shape[0] < 8:
        raise ValueError("CSV 至少需要 8 个样本")
    all_columns = (time_column, *selected_columns)
    if max(all_columns) >= table.shape[1]:
        raise ValueError(f"CSV 只有 {table.shape[1]} 列，所选列索引超出范围")
    time_s = np.asarray(table[:, time_column], dtype=np.float64) * scale_to_s
    values = np.asarray(table[:, selected_columns], dtype=np.float64)
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(values)):
        raise ValueError("CSV 所选时间列或数值列包含 NaN 或 Inf")
    intervals_s = np.diff(time_s)
    if np.any(intervals_s <= 0.0):
        raise ValueError("CSV 时间列必须严格递增")
    median_interval_s = float(np.median(intervals_s))
    relative_deviation = float(np.max(np.abs(intervals_s - median_interval_s)) / median_interval_s)
    ideal_original_time_s = (
        time_s[0] + np.arange(time_s.size, dtype=np.float64) * median_interval_s
    )
    maximum_cumulative_time_residual_s = float(
        np.max(np.abs(time_s - ideal_original_time_s))
    )
    nyquist_phase_error_rad = float(
        np.pi * maximum_cumulative_time_residual_s / median_interval_s
    )
    nyquist_phase_error_deg = float(np.degrees(nyquist_phase_error_rad))
    original_samples = int(time_s.size)
    resampled = False
    exceeds_cumulative_phase_limit = nyquist_phase_error_deg > 1.0
    if relative_deviation > uniformity_rtol or exceeds_cumulative_phase_limit:
        if not resample_nonuniform:
            if exceeds_cumulative_phase_limit:
                raise ValueError(
                    "CSV 时间轴累计残差在 Nyquist 处超过 1°；"
                    "请先将数据重采样为均匀时间间隔后再导入"
                )
            raise ValueError("CSV 时间间隔非均匀；请先重采样为均匀时间间隔后再导入")
        if relative_deviation > max_resample_relative_deviation:
            raise ValueError("CSV 时间间隔偏差过大，不能作为轻微非均匀采样重采样")
        candidate_time_s = time_s[0] + np.arange(time_s.size, dtype=np.float64) * (
            median_interval_s
        )
        magnitude = max(abs(float(time_s[0])), abs(float(time_s[-1])), median_interval_s)
        endpoint_tolerance_s = min(
            8.0 * abs(float(np.spacing(magnitude))),
            median_interval_s * 1.0e-6,
        )
        uniform_time_s = candidate_time_s[candidate_time_s <= time_s[-1] + endpoint_tolerance_s]
        if uniform_time_s.size < 8:
            raise ValueError("PCHIP 重采样后的有效范围少于 8 个样本")
        if uniform_time_s[-1] > time_s[-1]:
            uniform_time_s[-1] = time_s[-1]
        values = np.asarray(
            PchipInterpolator(time_s, values, axis=0, extrapolate=False)(uniform_time_s),
            dtype=np.float64,
        )
        time_s = uniform_time_s
        resampled = True
    _confirm_source_snapshot(source_path, source_snapshot)
    sample_rate_hz = 1.0 / median_interval_s
    return TimeSeries(
        time_s=time_s,
        values=values,
        sample_rate_hz=sample_rate_hz,
        source_path=source_path,
        source_format="csv",
        time_unit=time_unit,
        time_scale_to_s=scale_to_s,
        value_columns=selected_columns,
        source_metadata={
            "headerless": True,
            "delimiter": delimiter if delimiter is not None else "whitespace",
            "original_samples": original_samples,
            "maximum_relative_interval_deviation": relative_deviation,
            "maximum_cumulative_time_residual_s": maximum_cumulative_time_residual_s,
            "nyquist_phase_error_rad": nyquist_phase_error_rad,
            "nyquist_phase_error_deg": nyquist_phase_error_deg,
            "resampled_with_pchip": resampled,
            "source_size_bytes": source_snapshot[0],
            "source_sha256": source_snapshot[1],
        },
    )


def load_bin_timeseries(path: str | Path, config: BinConfig) -> TimeSeries:
    """Decode one configured channel from a raw, headerless BIN file."""

    if config.dtype not in _BIN_DTYPE_CODES:
        raise ValueError("BIN dtype 必须是 float32、float64、int16 或 int32")
    if config.byte_order not in {"little", "big"}:
        raise ValueError("BIN 字节序必须是 little 或 big")
    if config.layout not in {"interleaved", "planar"}:
        raise ValueError("BIN 布局必须是 interleaved 或 planar")

    source_path = Path(path).resolve()
    source_snapshot = _snapshot_source_file(source_path)
    file_size = source_snapshot[0]
    if config.offset_bytes > file_size:
        raise ValueError("BIN 文件头偏移超过文件长度")

    byte_prefix = "<" if config.byte_order == "little" else ">"
    dtype = np.dtype(byte_prefix + _BIN_DTYPE_CODES[config.dtype])
    payload_bytes = file_size - config.offset_bytes
    if payload_bytes % dtype.itemsize:
        raise ValueError("BIN 数据末尾包含不足一个样本的残字节")

    scalar_count = payload_bytes // dtype.itemsize
    if scalar_count % config.channels:
        raise ValueError("BIN 数据长度不能被通道数整除")
    samples = scalar_count // config.channels
    if samples < 8:
        raise ValueError("BIN 每通道至少需要 8 个样本")

    raw = np.fromfile(source_path, dtype=dtype, count=scalar_count, offset=config.offset_bytes)
    if config.layout == "interleaved":
        selected = raw.reshape(samples, config.channels)[:, config.channel_index]
    else:
        selected = raw.reshape(config.channels, samples)[config.channel_index]
    with np.errstate(over="ignore", invalid="ignore"):
        physical = selected.astype(np.float64) * config.scale + config.value_offset
    _confirm_source_snapshot(source_path, source_snapshot)
    time_s = np.arange(samples, dtype=np.float64) / config.sample_rate_hz
    return TimeSeries(
        time_s=time_s,
        values=physical[:, None],
        sample_rate_hz=config.sample_rate_hz,
        source_path=source_path,
        source_format="bin",
        time_unit="s",
        time_scale_to_s=1.0,
        value_columns=(config.channel_index,),
        source_metadata={
            "sample_rate_entered_manually": True,
            "dtype": config.dtype,
            "byte_order": config.byte_order,
            "offset_bytes": config.offset_bytes,
            "channels": config.channels,
            "channel_index": config.channel_index,
            "layout": config.layout,
            "scale": config.scale,
            "value_offset": config.value_offset,
            "source_size_bytes": source_snapshot[0],
            "source_sha256": source_snapshot[1],
        },
    )


def save_csv_timeseries(
    path: str | Path,
    time_s: object,
    values: object,
    time_scale_to_s: float = 1.0,
) -> Path:
    """Write time plus one or more value columns without a header row."""

    if not np.isfinite(time_scale_to_s) or time_scale_to_s <= 0.0:
        raise ValueError("CSV 导出时间单位换算必须是正的有限值")
    if np.iscomplexobj(time_s) or np.iscomplexobj(values):
        raise ValueError("CSV 导出不支持复数时间或复数信号")
    time_array = np.asarray(time_s, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if time_array.ndim != 1:
        raise ValueError("CSV 导出时间必须是一维数组")
    if time_array.size == 0:
        raise ValueError("CSV 导出时间和值必须是非空数组")
    if value_array.ndim == 1:
        value_array = value_array[:, None]
    if value_array.ndim != 2 or value_array.shape[0] != time_array.size:
        raise ValueError("CSV 导出值必须与时间长度一致")
    if value_array.shape[1] == 0:
        raise ValueError("CSV 导出至少需要一个数值列")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError("CSV 导出数据包含 NaN 或 Inf")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((time_array / time_scale_to_s, value_array))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            np.savetxt(stream, table, delimiter=",", fmt="%.17g")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path


def save_bin_float32(path: str | Path, values: object) -> Path:
    """Write values as raw little-endian float32 in C/interleaved order."""

    if np.iscomplexobj(values):
        raise ValueError("BIN float32 导出不支持复数信号")
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim not in {1, 2} or value_array.size == 0:
        raise ValueError("BIN 导出值必须是一维或二维非空数组")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("BIN 导出数据包含 NaN 或 Inf")
    with np.errstate(over="ignore"):
        encoded = value_array.astype("<f4")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("BIN 导出数据超出 float32 有限范围")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded.tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path
