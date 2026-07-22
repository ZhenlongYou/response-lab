"""Generic/Keysight CSV and self-describing Keysight Infiniium BIN I/O."""

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

from .keysight_bin import (
    _inspect_keysight_bin_from_open_file,
    _load_keysight_waveform_from_open_file,
    write_keysight_bin,
)
from .keysight_csv import _inspect_keysight_csv_from_open_file
from .memory_budget import safe_memory_budget_bytes, system_available_memory_bytes
from .models import TimeSeries

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
_SOURCE_HASH_BLOCK_BYTES = 1024 * 1024

# CSV 文本解析除最终 float64 表外还有字符解码、数值转换和 TimeSeries 校验副本。
# 按文件字节 3 倍、每物理行 256 B 基础量及每个实际选择列 64 B 估算；显式 PCHIP
# 重采样再增加每行 128 B + 每列 32 B。固定 32 MiB 覆盖解析器与分配器开销。
_CSV_TEXT_BYTES_MULTIPLIER = 3
_CSV_BASE_BYTES_PER_ROW = 256
_CSV_BYTES_PER_SELECTED_COLUMN_ROW = 64
_CSV_RESAMPLE_BASE_BYTES_PER_ROW = 128
_CSV_RESAMPLE_BYTES_PER_SELECTED_COLUMN_ROW = 32
_CSV_FIXED_OVERHEAD_BYTES = 32 * 1024**2

# BIN 的 memmap 本身较轻，但随后会同时出现 float64 时间轴、幅值副本、差分、理想
# 时间轴和 TimeSeries 验证临时量；112 B/点加 32 MiB 固定量保守覆盖这一阶段。
_BIN_LOADER_BYTES_PER_SAMPLE = 112
_BIN_LOADER_FIXED_OVERHEAD_BYTES = 32 * 1024**2


def _snapshot_open_file(handle: object) -> tuple[int, str]:
    """Hash one already-open file while preserving its descriptor and position."""

    original_position = handle.tell()
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        before = os.fstat(handle.fileno())
        handle.seek(0)
        for block in iter(lambda: handle.read(_SOURCE_HASH_BLOCK_BYTES), b""):
            digest.update(block)
            bytes_read += len(block)
        after = os.fstat(handle.fileno())
    finally:
        handle.seek(original_position)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or bytes_read != after.st_size
    ):
        raise OSError("源文件在加载快照期间发生变化，请重新选择后再试")
    return bytes_read, digest.hexdigest()


def _snapshot_source_file(path: Path) -> tuple[int, str]:
    """Return a stable size/SHA-256 snapshot, rejecting changes during hashing."""

    with path.open("rb") as handle:
        return _snapshot_open_file(handle)


def _confirm_source_snapshot(path: Path, expected: tuple[int, str]) -> None:
    """Ensure parsing and the recorded source snapshot refer to the same bytes."""

    if _snapshot_source_file(path) != expected:
        raise OSError("源文件在加载期间发生变化，已拒绝使用不一致的数据")


def _confirm_open_source_identity(path: Path, handle: object) -> None:
    """Reject a path replacement while a descriptor-backed import is running."""

    opened = os.fstat(handle.fileno())
    try:
        current = path.stat()
    except OSError as error:
        raise OSError("源文件在加载期间被移动或删除，请重新选择后再试") from error
    if opened.st_dev != current.st_dev or opened.st_ino != current.st_ino:
        raise OSError("源文件在加载期间被替换，已拒绝使用不一致的数据")


def _inspect_csv_layout_from_open_file(handle: object) -> tuple[str | None, int]:
    """Read only the first numeric row to infer delimiter and source column count."""

    original_position = handle.tell()
    try:
        handle.seek(0)
        first_line = ""
        for raw_line in handle:
            try:
                candidate = raw_line.decode("utf-8-sig").strip()
            except UnicodeDecodeError as error:
                raise ValueError("CSV 必须是 UTF-8 编码文本") from error
            if candidate and not candidate.lstrip().startswith("#"):
                first_line = candidate
                break
    finally:
        handle.seek(original_position)
    if not first_line:
        raise ValueError("CSV 文件为空")
    counts = {delimiter: first_line.count(delimiter) for delimiter in (",", ";", "\t")}
    delimiter = max(counts, key=counts.get)
    selected_delimiter = delimiter if counts[delimiter] else None
    columns = (
        len(first_line.split(selected_delimiter))
        if selected_delimiter is not None
        else len(first_line.split())
    )
    if columns == 0:
        raise ValueError("CSV 首个数据行没有可解析列")
    return selected_delimiter, columns


def _count_physical_rows_from_open_file(handle: object) -> int:
    """Count physical lines with bounded blocks; comments/blanks intentionally overestimate rows."""

    original_position = handle.tell()
    line_breaks = 0
    bytes_read = 0
    last_byte = b""
    try:
        handle.seek(0)
        for block in iter(lambda: handle.read(_SOURCE_HASH_BLOCK_BYTES), b""):
            line_breaks += block.count(b"\n")
            bytes_read += len(block)
            last_byte = block[-1:]
    finally:
        handle.seek(original_position)
    return line_breaks + int(bytes_read > 0 and last_byte != b"\n")


def _estimate_csv_loader_peak_bytes(
    *,
    file_size_bytes: int,
    physical_rows: int,
    selected_columns: int,
    resample_nonuniform: bool,
) -> int:
    """Estimate text parser, selected numeric table, and validation/resampling peak memory."""

    per_row_bytes = (
        _CSV_BASE_BYTES_PER_ROW
        + selected_columns * _CSV_BYTES_PER_SELECTED_COLUMN_ROW
    )
    if resample_nonuniform:
        per_row_bytes += (
            _CSV_RESAMPLE_BASE_BYTES_PER_ROW
            + selected_columns * _CSV_RESAMPLE_BYTES_PER_SELECTED_COLUMN_ROW
        )
    return int(
        _CSV_FIXED_OVERHEAD_BYTES
        + int(file_size_bytes) * _CSV_TEXT_BYTES_MULTIPLIER
        + int(physical_rows) * per_row_bytes
    )


def _raise_if_loader_memory_exceeds_budget(
    *,
    label: str,
    estimated_bytes: int,
    stage: str,
    explicit_limit_bytes: int | None = None,
) -> int:
    """Apply the shared live-memory budget before a loader allocates its large arrays."""

    available = system_available_memory_bytes()
    dynamic_budget = safe_memory_budget_bytes(available)
    effective_budget = (
        min(dynamic_budget, int(explicit_limit_bytes))
        if explicit_limit_bytes is not None
        else dynamic_budget
    )
    if estimated_bytes > effective_budget:
        available_text = (
            f"系统当前可用约 {available / (1024.0**2):.0f} MiB"
            if available is not None
            else "系统可用内存不可探测，使用 768 MiB 回退预算"
        )
        raise MemoryError(
            f"{label} 动态内存预检拒绝：预计峰值约 "
            f"{estimated_bytes / (1024.0**2):.0f} MiB，安全预算约 "
            f"{effective_budget / (1024.0**2):.0f} MiB（{available_text}）；"
            f"已在{stage}停止"
        )
    return effective_budget


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
    """Load a Keysight WaveformXYValues or generic headerless time-series CSV."""

    source_path = Path(path).resolve()
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
    all_columns = (time_column, *selected_columns)

    # 文件布局检查、动态预算、文本解析和最终快照共用一个描述符，避免路径替换把
    # “小文件预检”与“另一份大文件解析”拼接成一次加载。
    with source_path.open("rb") as opened_file:
        source_snapshot = _snapshot_open_file(opened_file)
        keysight_layout = _inspect_keysight_csv_from_open_file(
            source_path,
            opened_file,
        )
        if keysight_layout is None:
            delimiter, source_column_count = _inspect_csv_layout_from_open_file(
                opened_file
            )
            if max(all_columns) >= source_column_count:
                raise ValueError(
                    f"CSV 首个数据行只有 {source_column_count} 列，所选列索引超出范围"
                )
            parse_columns = all_columns
            loadtxt_usecols: tuple[int, ...] | None = parse_columns
            data_offset = 0
        else:
            if time_unit != "s" or time_column != 0 or selected_columns != (1,):
                raise ValueError(
                    "Keysight XY CSV 已由表头固定为第 1 列秒、第 2 列伏特，"
                    "不能覆盖时间单位或列映射"
                )
            delimiter = ","
            parse_columns = (0, 1)
            loadtxt_usecols = None
            data_offset = keysight_layout.data_offset
        physical_rows = _count_physical_rows_from_open_file(opened_file)
        estimated_loader_bytes = _estimate_csv_loader_peak_bytes(
            file_size_bytes=source_snapshot[0],
            physical_rows=physical_rows,
            selected_columns=len(parse_columns),
            resample_nonuniform=resample_nonuniform,
        )
        _raise_if_loader_memory_exceeds_budget(
            label="CSV",
            estimated_bytes=estimated_loader_bytes,
            stage=" NumPy 文本解析前",
        )
        opened_file.seek(data_offset)
        try:
            table = np.loadtxt(
                opened_file,
                delimiter=delimiter,
                usecols=loadtxt_usecols,
                ndmin=2,
                encoding="utf-8-sig",
            )
        except ValueError as error:
            if keysight_layout is not None:
                raise ValueError(
                    "Keysight CSV 数据区必须恰好包含可按数值解析的 "
                    f"time, voltage 两列，且每行列数一致：{error}"
                ) from error
            raise ValueError(
                "CSV 所选列无法按数值解析、或某个数据行缺少所选列："
                f"{error}"
            ) from error
        if _snapshot_open_file(opened_file) != source_snapshot:
            raise OSError("源文件在加载期间发生变化，已拒绝使用不一致的数据")
        _confirm_open_source_identity(source_path, opened_file)

    if table.shape[0] < 8:
        raise ValueError("CSV 至少需要 8 个样本")
    if (
        keysight_layout is not None
        and keysight_layout.points is not None
        and table.shape[0] != keysight_layout.points
    ):
        raise ValueError(
            "Keysight CSV Points 与实际数据行数不一致："
            f"声明 {keysight_layout.points}，实际 {table.shape[0]}"
        )
    # usecols 已按 (time, selected values...) 的显式顺序构造紧凑表，不再按源列号索引。
    if keysight_layout is not None:
        scale_to_s = 1.0
        time_unit = "s"
        selected_columns = (1,)
    time_s = np.asarray(table[:, 0], dtype=np.float64) * scale_to_s
    values = np.asarray(table[:, 1:], dtype=np.float64)
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
            keysight_interpolation_hint = (
                "；Keysight Infiniium 导出时请启用 Linearly Interpolate"
                if keysight_layout is not None
                and keysight_layout.container == "keysight_infiniium_waveform_xy"
                else ""
            )
            if exceeds_cumulative_phase_limit:
                raise ValueError(
                    "CSV 时间轴累计残差在 Nyquist 处超过 1°；"
                    "请先将数据重采样为均匀时间间隔后再导入"
                    f"{keysight_interpolation_hint}"
                )
            raise ValueError(
                "CSV 时间间隔非均匀；请先重采样为均匀时间间隔后再导入"
                f"{keysight_interpolation_hint}"
            )
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
    sample_rate_hz = 1.0 / median_interval_s
    source_metadata = {
        "headerless": keysight_layout is None,
        "delimiter": delimiter if delimiter is not None else "whitespace",
        "original_samples": original_samples,
        "maximum_relative_interval_deviation": relative_deviation,
        "maximum_cumulative_time_residual_s": maximum_cumulative_time_residual_s,
        "nyquist_phase_error_rad": nyquist_phase_error_rad,
        "nyquist_phase_error_deg": nyquist_phase_error_deg,
        "resampled_with_pchip": resampled,
        "physical_rows_preflight": physical_rows,
        "estimated_loader_bytes": estimated_loader_bytes,
        "source_size_bytes": source_snapshot[0],
        "source_sha256": source_snapshot[1],
    }
    if keysight_layout is not None:
        source_metadata.update(
            {
                "container": keysight_layout.container,
                "format_reference": keysight_layout.format_reference,
                "instrument": keysight_layout.instrument,
                "software_version": keysight_layout.software_version,
                "serial_number": keysight_layout.serial_number,
                "acquisition_date": keysight_layout.acquisition_date,
                "signal_type": keysight_layout.signal_type,
                "source_name": keysight_layout.source_name,
                "channel_noise": keysight_layout.channel_noise,
                "intrinsic_jitter": keysight_layout.intrinsic_jitter,
                "interpolation_factor": keysight_layout.interpolation_factor,
                "bandwidth": keysight_layout.bandwidth,
                "x_units": keysight_layout.x_units,
                "y_units": keysight_layout.y_units,
                "x_precision": keysight_layout.x_precision,
                "y_precision": keysight_layout.y_precision,
                "sample_rate_source": (
                    "keysight_xy_time_column"
                    if keysight_layout.container == "keysight_infiniium_waveform_xy"
                    else "csv_time_column"
                ),
            }
        )
        if keysight_layout.format_version is not None:
            source_metadata["keysight_format_version"] = keysight_layout.format_version
        if keysight_layout.points is not None:
            source_metadata["declared_points"] = keysight_layout.points
    return TimeSeries(
        time_s=time_s,
        values=values,
        sample_rate_hz=sample_rate_hz,
        source_path=source_path,
        source_format="csv",
        time_unit=time_unit,
        time_scale_to_s=scale_to_s,
        value_columns=selected_columns,
        source_metadata=source_metadata,
    )


def load_bin_timeseries(
    path: str | Path,
    *,
    waveform_index: int | None = None,
    max_decoded_bytes: int = 1_500_000_000,
    max_samples: int | None = None,
) -> TimeSeries:
    """Load one supported waveform from a self-describing Infiniium AG BIN."""

    source_path = Path(path).expanduser().resolve()
    # Header scan, budget check, hash and payload mapping share one descriptor.
    # This prevents a directory-entry replacement from bypassing the checked header.
    with source_path.open("rb") as opened_file:
        info = _inspect_keysight_bin_from_open_file(source_path, opened_file)
        if waveform_index is None:
            if len(info.waveforms) != 1:
                raise ValueError(
                    "Keysight BIN 包含多个 waveform；当前入口不会猜测目标记录"
                )
            selected_index = 0
        else:
            selected_index = operator.index(waveform_index)
            if isinstance(waveform_index, (bool, np.bool_)):
                raise ValueError("waveform_index 必须是整数且不能为布尔值")
        if not 0 <= selected_index < len(info.waveforms):
            raise ValueError("waveform_index 超出 Keysight BIN 的 waveform 范围")
        waveform_info = info.waveforms[selected_index]
        if waveform_info.points < 8:
            raise ValueError("Keysight BIN waveform 至少需要 8 个样本")
        if max_samples is not None:
            if (
                isinstance(max_samples, (bool, np.bool_))
                or not isinstance(max_samples, (int, np.integer))
                or max_samples < 8
            ):
                raise ValueError("BIN 样点安全上限必须是至少 8 的整数")
            if waveform_info.points > int(max_samples):
                raise MemoryError(
                    f"Keysight BIN 含 {waveform_info.points} 点，超过本流程安全上限 "
                    f"{int(max_samples)} 点；已在读取 payload 前停止"
                )
        if (
            isinstance(max_decoded_bytes, (bool, np.bool_))
            or not isinstance(max_decoded_bytes, (int, np.integer))
            or max_decoded_bytes <= 0
        ):
            raise ValueError("BIN 解码内存上限必须是正整数")
        # 同时计入 memmap 后的 float64 时间/幅值、TimeSeries 验证副本和固定分配开销。
        estimated_decoded_bytes = (
            int(waveform_info.points) * _BIN_LOADER_BYTES_PER_SAMPLE
            + _BIN_LOADER_FIXED_OVERHEAD_BYTES
        )
        loader_memory_budget_bytes = _raise_if_loader_memory_exceeds_budget(
            label="BIN",
            estimated_bytes=estimated_decoded_bytes,
            explicit_limit_bytes=int(max_decoded_bytes),
            stage=" payload 前",
        )

        source_snapshot = _snapshot_open_file(opened_file)
        waveform = _load_keysight_waveform_from_open_file(
            source_path,
            opened_file,
            selected_index,
        )
        time_s = waveform.x_origin_s + np.arange(
            waveform.values.size,
            dtype=np.float64,
        ) * waveform.x_increment_s
        series = TimeSeries(
            time_s=time_s,
            values=waveform.values,
            sample_rate_hz=waveform.sample_rate_hz,
            source_path=source_path,
            source_format="bin",
            time_unit="s",
            time_scale_to_s=1.0,
            value_columns=(selected_index,),
            source_metadata={
                "container": "keysight_infiniium_ag",
                "keysight_version": info.version,
                "waveform_index": selected_index,
                "waveform_label": waveform.label,
                "segment_index": waveform.segment_index,
                "sample_rate_source": "keysight_x_increment",
                "x_increment_s": waveform.x_increment_s,
                "x_origin_s": waveform.x_origin_s,
                "estimated_decoded_bytes": estimated_decoded_bytes,
                "loader_memory_budget_bytes": loader_memory_budget_bytes,
                "source_size_bytes": source_snapshot[0],
                "source_sha256": source_snapshot[1],
            },
        )
        if _snapshot_open_file(opened_file) != source_snapshot:
            raise OSError("源文件在加载期间发生变化，已拒绝使用不一致的数据")
        _confirm_open_source_identity(source_path, opened_file)
        return series


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


def save_bin_timeseries(
    path: str | Path,
    time_s: object,
    values: object,
    *,
    label: str = "ResponseLab",
) -> Path:
    """Atomically write one re-importable Keysight Infiniium AG waveform."""

    if np.iscomplexobj(time_s) or np.iscomplexobj(values):
        raise ValueError("BIN 导出不支持复数时间或复数信号")
    time_array = np.asarray(time_s, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("BIN 导出时间和值必须是一维非空数组")
    if value_array.ndim == 2 and value_array.shape[1] == 1:
        value_array = value_array[:, 0]
    if value_array.ndim != 1 or value_array.size != time_array.size:
        raise ValueError("BIN 导出当前只支持与时间等长的单通道值")
    if time_array.size < 8:
        raise ValueError("BIN 导出至少需要 8 个样本")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(value_array)):
        raise ValueError("BIN 导出数据包含 NaN 或 Inf")
    intervals_s = np.diff(time_array)
    if np.any(intervals_s <= 0.0):
        raise ValueError("BIN 导出时间必须严格递增且等间隔")
    interval_s = float(np.median(intervals_s))
    if not np.allclose(intervals_s, interval_s, rtol=1.0e-9, atol=0.0):
        raise ValueError("BIN 导出时间必须等间隔")
    sample_rate_hz = 1.0 / interval_s

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        write_keysight_bin(
            temporary_name,
            value_array,
            sample_rate_hz,
            x_origin_s=float(time_array[0]),
            label=label,
        )
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path
