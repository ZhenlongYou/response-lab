"""Generic/Keysight CSV and self-describing Keysight Infiniium BIN I/O."""

from __future__ import annotations

import csv
import hashlib
import operator
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from .cancellation import CancellationCheck, raise_if_cancelled
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
_EXPORT_CHUNK_SAMPLES = 131_072
_CANCEL_CHECK_ROWS = 4096
_LOAD_CANCEL_MESSAGE = "加载已取消"

# CSV 文本解析除最终 float64 表外还有字符解码、数值转换和 TimeSeries 校验副本。
# 按文件字节 3 倍、每物理行 256 B 基础量及每个实际选择列 64 B 估算；显式 PCHIP
# 重采样再增加每行 128 B + 每列 32 B。固定 32 MiB 覆盖解析器与分配器开销。
_CSV_TEXT_BYTES_MULTIPLIER = 3
_CSV_BASE_BYTES_PER_ROW = 256
_CSV_BYTES_PER_SELECTED_COLUMN_ROW = 64
_CSV_RESAMPLE_BASE_BYTES_PER_ROW = 128
_CSV_RESAMPLE_BYTES_PER_SELECTED_COLUMN_ROW = 32
_CSV_FIXED_OVERHEAD_BYTES = 32 * 1024**2

# 自描述 BIN 保留一份 float64 时间轴和一份原生 float32 幅值副本；XIncrement/XOrigin
# 已在头部校验后分块验证实际间隔，不再创建全长 diff、median 和第二条理想时间轴。
# 2026-07-23 两个全新 macOS arm64 子进程加载 100 万点的新增 RSS 分别为
# 21,807,104 B 与 20,889,600 B；24 B/点加 16 MiB 固定量约为实测的 1.9 倍。
_BIN_LOADER_BYTES_PER_SAMPLE = 24
_BIN_LOADER_FIXED_OVERHEAD_BYTES = 16 * 1024**2
_BIN_RESIDENT_BYTES_PER_SAMPLE = 12


class SourceFileFingerprint(NamedTuple):
    """Content evidence plus immutable-within-load descriptor identity."""

    size_bytes: int
    sha256: str
    device: int
    inode: int
    modified_time_ns: int
    changed_time_ns: int


@dataclass(frozen=True)
class BinPayloadLayout:
    """Header-only BIN geometry passed to a full-pipeline preflight callback."""

    samples: int
    channels: int
    sample_rate_hz: float
    estimated_resident_bytes: int
    estimated_loader_peak_bytes: int


@dataclass(frozen=True)
class _GenericCsvLayout:
    """One supported headerless CSV layout discovered before numeric allocation."""

    delimiter: str | None
    source_column_count: int
    packed_pair_separator: str | None = None


def _snapshot_open_file(
    handle: object,
    *,
    cancelled: CancellationCheck | None = None,
) -> SourceFileFingerprint:
    """Hash one already-open file while preserving its descriptor and position."""

    original_position = handle.tell()
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        before = os.fstat(handle.fileno())
        handle.seek(0)
        while True:
            raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
            block = handle.read(_SOURCE_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            bytes_read += len(block)
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
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
    return SourceFileFingerprint(
        size_bytes=bytes_read,
        sha256=digest.hexdigest(),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        modified_time_ns=int(after.st_mtime_ns),
        changed_time_ns=int(after.st_ctime_ns),
    )


def snapshot_source_file(
    path: str | Path,
    *,
    cancelled: CancellationCheck | None = None,
) -> SourceFileFingerprint:
    """Return stable content and identity evidence, rejecting changes while hashing."""

    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as handle:
        return _snapshot_open_file(handle, cancelled=cancelled)


def _confirm_open_source_identity(path: Path, handle: object) -> None:
    """Reject a path replacement while a descriptor-backed import is running."""

    opened = os.fstat(handle.fileno())
    try:
        current = path.stat()
    except OSError as error:
        raise OSError("源文件在加载期间被移动或删除，请重新选择后再试") from error
    if opened.st_dev != current.st_dev or opened.st_ino != current.st_ino:
        raise OSError("源文件在加载期间被替换，已拒绝使用不一致的数据")


def _parse_generic_csv_record(text: str, *, line_number: int) -> tuple[str, ...]:
    """Parse one generic CSV record without treating malformed quoting as numeric data."""

    try:
        return tuple(
            cell.strip() for cell in next(csv.reader([text], skipinitialspace=True, strict=True))
        )
    except csv.Error as error:
        raise ValueError(f"CSV 第 {line_number} 行的引号或字段格式无效：{error}") from error


def _packed_pair_separator(
    record: tuple[str, ...],
    *,
    source_line: str,
) -> str | None:
    """Identify a single CSV field that itself holds exactly ``time, amplitude``."""

    if len(record) != 1 or not source_line.lstrip().startswith('"'):
        return None
    cell = record[0].strip()
    comma_fields = tuple(field.strip() for field in cell.split(","))
    if len(comma_fields) == 2 and all(comma_fields):
        return ","
    whitespace_fields = tuple(cell.split())
    if len(whitespace_fields) == 2:
        return "whitespace"
    return None


def _inspect_csv_layout_from_open_file(
    handle: object,
    *,
    cancelled: CancellationCheck | None = None,
) -> _GenericCsvLayout:
    """Read one headerless data record and classify its physical and logical columns."""

    original_position = handle.tell()
    try:
        handle.seek(0)
        first_line = ""
        first_line_number = 0
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number == 1 or line_number % _CANCEL_CHECK_ROWS == 0:
                raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
            try:
                candidate = raw_line.decode("utf-8-sig").strip()
            except UnicodeDecodeError as error:
                raise ValueError("CSV 必须是 UTF-8 编码文本") from error
            if candidate and not candidate.lstrip().startswith("#"):
                first_line = candidate
                first_line_number = line_number
                break
    finally:
        handle.seek(original_position)
    if not first_line:
        raise ValueError("CSV 文件为空")
    first_record = _parse_generic_csv_record(
        first_line,
        line_number=first_line_number,
    )
    packed_separator = _packed_pair_separator(
        first_record,
        source_line=first_line,
    )
    if packed_separator is not None:
        return _GenericCsvLayout(
            delimiter=None,
            source_column_count=2,
            packed_pair_separator=packed_separator,
        )
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
    return _GenericCsvLayout(
        delimiter=selected_delimiter,
        source_column_count=columns,
    )


def _load_packed_pair_csv_from_open_file(
    handle: object,
    *,
    separator: str,
    physical_rows: int,
    cancelled: CancellationCheck | None = None,
) -> np.ndarray:
    """Load one quoted CSV field per line as its logical time/amplitude pair.

    The destination is allocated only after the shared dynamic-memory preflight.
    It intentionally accepts exactly two embedded numeric tokens per data row, so
    mixed layouts and accidental extra columns cannot silently shift the signal.
    """

    original_position = handle.tell()
    table = np.empty((physical_rows, 2), dtype=np.float64)
    parsed_rows = 0
    try:
        handle.seek(0)
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number == 1 or line_number % _CANCEL_CHECK_ROWS == 0:
                raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
            try:
                candidate = raw_line.decode("utf-8-sig").strip()
            except UnicodeDecodeError as error:
                raise ValueError("CSV 必须是 UTF-8 编码文本") from error
            if not candidate or candidate.lstrip().startswith("#"):
                continue
            if not candidate.lstrip().startswith('"'):
                raise ValueError(
                    "CSV 单字段时间/幅度格式要求每行只含一个双引号包裹的 CSV 字段"
                    f"（第 {line_number} 行无效）"
                )
            record = _parse_generic_csv_record(candidate, line_number=line_number)
            if len(record) != 1:
                raise ValueError(
                    "CSV 单字段时间/幅度格式要求每行只含一个 CSV 字段，"
                    f"但第 {line_number} 行包含 {len(record)} 个字段"
                )
            cell = record[0].strip()
            fields = (
                tuple(field.strip() for field in cell.split(","))
                if separator == ","
                else tuple(cell.split())
            )
            if len(fields) != 2 or not all(fields):
                expected = "逗号" if separator == "," else "空白"
                raise ValueError(
                    "CSV 单字段时间/幅度格式要求每行在同一字段内恰好包含 "
                    f"两个由{expected}分隔的数值（第 {line_number} 行无效）"
                )
            try:
                table[parsed_rows, 0] = float(fields[0])
                table[parsed_rows, 1] = float(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"CSV 单字段时间/幅度格式的时间和幅度必须都是数值（第 {line_number} 行）"
                ) from error
            parsed_rows += 1
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    finally:
        handle.seek(original_position)
    return table[:parsed_rows]


def _count_physical_rows_from_open_file(
    handle: object,
    *,
    cancelled: CancellationCheck | None = None,
) -> int:
    """Count physical lines with bounded blocks; comments/blanks intentionally overestimate rows."""

    original_position = handle.tell()
    line_breaks = 0
    bytes_read = 0
    last_byte = b""
    try:
        handle.seek(0)
        while True:
            raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
            block = handle.read(_SOURCE_HASH_BLOCK_BYTES)
            if not block:
                break
            line_breaks += block.count(b"\n")
            bytes_read += len(block)
            last_byte = block[-1:]
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    finally:
        handle.seek(original_position)
    return line_breaks + int(bytes_read > 0 and last_byte != b"\n")


class _CancellableCsvLines:
    """Iterable CSV view that preserves the source name for diagnostics/tests."""

    def __init__(
        self,
        handle: object,
        cancelled: CancellationCheck | None,
    ) -> None:
        self._handle = handle
        self._cancelled = cancelled
        self.name = getattr(handle, "name", "<open CSV>")

    def __iter__(self) -> Iterator[bytes]:
        for line_number, raw_line in enumerate(self._handle, start=1):
            if line_number == 1 or line_number % _CANCEL_CHECK_ROWS == 0:
                raise_if_cancelled(self._cancelled, message=_LOAD_CANCEL_MESSAGE)
            yield raw_line
        raise_if_cancelled(self._cancelled, message=_LOAD_CANCEL_MESSAGE)


def _estimate_csv_loader_peak_bytes(
    *,
    file_size_bytes: int,
    physical_rows: int,
    selected_columns: int,
    resample_nonuniform: bool,
) -> int:
    """Estimate text parser, selected numeric table, and validation/resampling peak memory."""

    per_row_bytes = _CSV_BASE_BYTES_PER_ROW + selected_columns * _CSV_BYTES_PER_SELECTED_COLUMN_ROW
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


def _normalize_expected_columns(expected_columns: int | None) -> int | None:
    """Validate an optional exact logical-column contract for user-facing loaders."""

    if expected_columns is None:
        return None
    try:
        normalized = operator.index(expected_columns)
    except TypeError as error:
        raise ValueError("CSV 期望列数必须是正整数") from error
    if isinstance(expected_columns, bool) or normalized <= 0:
        raise ValueError("CSV 期望列数必须是正整数")
    return normalized


def load_csv_timeseries(
    path: str | Path,
    *,
    time_unit: str = "s",
    time_column: int = 0,
    value_columns: int | Sequence[int] = (1,),
    expected_columns: int | None = None,
    resample_nonuniform: bool = False,
    uniformity_rtol: float = _DEFAULT_UNIFORMITY_RTOL,
    max_resample_relative_deviation: float = 0.05,
    cancelled: CancellationCheck | None = None,
) -> TimeSeries:
    """Load a Keysight WaveformXYValues or generic headerless time-series CSV.

    ``expected_columns`` is an opt-in exact logical-column contract for fixed GUI
    workflows.  Its default keeps programmatic extraction from wider tables; a
    packed single-field time/amplitude record counts as two logical columns.
    """

    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    source_path = Path(path).resolve()
    try:
        scale_to_s = _TIME_SCALE_TO_S[time_unit]
    except (KeyError, TypeError) as error:
        supported = "s、ms、us、µs、ns、ps"
        raise ValueError(f"不支持的 CSV 时间单位；可选 {supported}") from error
    time_column = _as_column_index(time_column, label="时间列")
    selected_columns = _normalize_value_columns(value_columns)
    expected_columns = _normalize_expected_columns(expected_columns)
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
        source_snapshot = _snapshot_open_file(opened_file, cancelled=cancelled)
        # First-record inspection uses the binary iterator and could otherwise
        # materialize one malformed, file-sized line before the detailed row/
        # column estimate.  The byte-scaled portion alone is a safe lower bound,
        # so reject that geometry before any line-oriented read.
        _raise_if_loader_memory_exceeds_budget(
            label="CSV",
            estimated_bytes=_estimate_csv_loader_peak_bytes(
                file_size_bytes=source_snapshot.size_bytes,
                physical_rows=1,
                selected_columns=len(all_columns),
                resample_nonuniform=resample_nonuniform,
            ),
            stage=" NumPy 文本解析前",
        )
        keysight_layout = _inspect_keysight_csv_from_open_file(
            source_path,
            opened_file,
        )
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        if keysight_layout is None:
            generic_layout = _inspect_csv_layout_from_open_file(
                opened_file,
                cancelled=cancelled,
            )
            delimiter = generic_layout.delimiter
            packed_pair_separator = generic_layout.packed_pair_separator
            if (
                expected_columns is not None
                and generic_layout.source_column_count != expected_columns
            ):
                raise ValueError(
                    "普通无表头 CSV 必须每行恰好 "
                    f"{expected_columns} 列，检测到 "
                    f"{generic_layout.source_column_count} 列"
                )
            if max(all_columns) >= generic_layout.source_column_count:
                raise ValueError(
                    "CSV 首个数据行只有 "
                    f"{generic_layout.source_column_count} 列，所选列索引超出范围"
                )
            parse_columns = all_columns
            # 严格 GUI 合同读入整行，让 NumPy 同时校验后续行没有多/少列。
            loadtxt_usecols: tuple[int, ...] | None = (
                None if expected_columns is not None else parse_columns
            )
            data_offset = 0
        else:
            if expected_columns is not None and expected_columns != 2:
                raise ValueError("Keysight XY CSV 由表头固定为恰好 2 列")
            if time_unit != "s" or time_column != 0 or selected_columns != (1,):
                raise ValueError(
                    "Keysight XY CSV 已由表头固定为第 1 列秒、第 2 列伏特，不能覆盖时间单位或列映射"
                )
            delimiter = ","
            packed_pair_separator = None
            parse_columns = (0, 1)
            loadtxt_usecols = None
            data_offset = keysight_layout.data_offset
        physical_rows = _count_physical_rows_from_open_file(
            opened_file,
            cancelled=cancelled,
        )
        estimated_loader_bytes = _estimate_csv_loader_peak_bytes(
            file_size_bytes=source_snapshot.size_bytes,
            physical_rows=physical_rows,
            selected_columns=(
                expected_columns
                if keysight_layout is None and expected_columns is not None
                else len(parse_columns)
            ),
            resample_nonuniform=resample_nonuniform,
        )
        _raise_if_loader_memory_exceeds_budget(
            label="CSV",
            estimated_bytes=estimated_loader_bytes,
            stage=" NumPy 文本解析前",
        )
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        opened_file.seek(data_offset)
        if keysight_layout is None and packed_pair_separator is not None:
            packed_table = _load_packed_pair_csv_from_open_file(
                opened_file,
                separator=packed_pair_separator,
                physical_rows=physical_rows,
                cancelled=cancelled,
            )
            table = packed_table if parse_columns == (0, 1) else packed_table[:, parse_columns]
        else:
            try:
                table = np.loadtxt(
                    _CancellableCsvLines(
                        opened_file,
                        cancelled,
                    ),
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
                if expected_columns is not None:
                    raise ValueError(
                        "普通无表头 CSV 必须每行恰好 "
                        f"{expected_columns} 列，且每行列数一致：{error}"
                    ) from error
                raise ValueError(
                    f"CSV 所选列无法按数值解析、或某个数据行缺少所选列：{error}"
                ) from error
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        if (
            keysight_layout is None
            and packed_pair_separator is None
            and expected_columns is not None
        ):
            if table.shape[1] != expected_columns:
                raise ValueError(
                    f"普通无表头 CSV 必须每行恰好 {expected_columns} 列，检测到 {table.shape[1]} 列"
                )
            # 严格整行校验后恢复公共 API 约定的（时间，选中数值）紧凑顺序。
            table = table[:, parse_columns]
        if _snapshot_open_file(opened_file, cancelled=cancelled) != source_snapshot:
            raise OSError("源文件在加载期间发生变化，已拒绝使用不一致的数据")
        _confirm_open_source_identity(source_path, opened_file)

    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
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
    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    intervals_s = np.diff(time_s)
    if np.any(intervals_s <= 0.0):
        raise ValueError("CSV 时间列必须严格递增")
    median_interval_s = float(np.median(intervals_s))
    relative_deviation = float(np.max(np.abs(intervals_s - median_interval_s)) / median_interval_s)
    ideal_original_time_s = time_s[0] + np.arange(time_s.size, dtype=np.float64) * median_interval_s
    maximum_cumulative_time_residual_s = float(np.max(np.abs(time_s - ideal_original_time_s)))
    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    nyquist_phase_error_rad = float(np.pi * maximum_cumulative_time_residual_s / median_interval_s)
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
                f"CSV 时间间隔非均匀；请先重采样为均匀时间间隔后再导入{keysight_interpolation_hint}"
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
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    sample_rate_hz = 1.0 / median_interval_s
    source_metadata = {
        "headerless": keysight_layout is None,
        "delimiter": (
            "packed-comma"
            if packed_pair_separator == ","
            else "packed-whitespace"
            if packed_pair_separator is not None
            else delimiter
            if delimiter is not None
            else "whitespace"
        ),
        "original_samples": original_samples,
        "maximum_relative_interval_deviation": relative_deviation,
        "maximum_cumulative_time_residual_s": maximum_cumulative_time_residual_s,
        "nyquist_phase_error_rad": nyquist_phase_error_rad,
        "nyquist_phase_error_deg": nyquist_phase_error_deg,
        "resampled_with_pchip": resampled,
        "physical_rows_preflight": physical_rows,
        "estimated_loader_bytes": estimated_loader_bytes,
        "source_size_bytes": source_snapshot.size_bytes,
        "source_sha256": source_snapshot.sha256,
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
    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    series = TimeSeries(
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
    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    return series


def load_bin_timeseries(
    path: str | Path,
    *,
    waveform_index: int | None = None,
    max_decoded_bytes: int = 1_500_000_000,
    max_samples: int | None = None,
    payload_preflight: Callable[[BinPayloadLayout], None] | None = None,
    cancelled: CancellationCheck | None = None,
) -> TimeSeries:
    """Load one supported waveform from a self-describing Infiniium AG BIN."""

    raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
    source_path = Path(path).expanduser().resolve()
    # Header scan, budget check, hash and payload mapping share one descriptor.
    # This prevents a directory-entry replacement from bypassing the checked header.
    with source_path.open("rb") as opened_file:
        descriptor_before = os.fstat(opened_file.fileno())
        info = _inspect_keysight_bin_from_open_file(
            source_path,
            opened_file,
            # GUI 不传索引，必须在文件头早拒绝歧义；显式 API 选择仍保留兼容能力。
            require_single_waveform=waveform_index is None,
            cancelled=cancelled,
        )
        if waveform_index is None:
            if len(info.waveforms) != 1:
                raise ValueError("Keysight BIN 包含多个 waveform；当前入口不会猜测目标记录")
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
        # 计入 memmap、一次 float64 时间/幅值构造和固定分配开销。
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
        if waveform_info.unsupported_reason is not None:
            raise ValueError(
                f"Waveform {selected_index} 无法加载：{waveform_info.unsupported_reason}"
            )
        sample_rate_hz = waveform_info.sample_rate_hz
        if sample_rate_hz is None:
            raise ValueError("Keysight BIN 无法从 X Increment 推导采样率")
        if payload_preflight is not None:
            payload_preflight(
                BinPayloadLayout(
                    samples=int(waveform_info.points),
                    channels=1,
                    sample_rate_hz=sample_rate_hz,
                    estimated_resident_bytes=(
                        int(waveform_info.points) * _BIN_RESIDENT_BYTES_PER_SAMPLE
                    ),
                    estimated_loader_peak_bytes=estimated_decoded_bytes,
                )
            )
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        descriptor_after_preflight = os.fstat(opened_file.fileno())
        if (
            descriptor_before.st_dev != descriptor_after_preflight.st_dev
            or descriptor_before.st_ino != descriptor_after_preflight.st_ino
            or descriptor_before.st_size != descriptor_after_preflight.st_size
            or descriptor_before.st_mtime_ns != descriptor_after_preflight.st_mtime_ns
            or descriptor_before.st_ctime_ns != descriptor_after_preflight.st_ctime_ns
        ):
            raise OSError("源文件在 BIN 头部预检期间发生变化，请重新选择后再试")

        source_snapshot = _snapshot_open_file(opened_file, cancelled=cancelled)
        try:
            waveform = _load_keysight_waveform_from_open_file(
                source_path,
                opened_file,
                selected_index,
                prevalidated_info=info,
                cancelled=cancelled,
            )
        except OSError as error:
            raise OSError(
                "源文件在 BIN payload 加载期间发生变化或被操作系统阻止，已拒绝继续"
            ) from error
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        series = TimeSeries.from_uniform_samples(
            values=waveform.values,
            sample_rate_hz=waveform.sample_rate_hz,
            time_origin_s=waveform.x_origin_s,
            time_increment_s=waveform.x_increment_s,
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
                "estimated_resident_bytes": (
                    int(waveform_info.points) * _BIN_RESIDENT_BYTES_PER_SAMPLE
                ),
                "loader_memory_budget_bytes": loader_memory_budget_bytes,
                "source_size_bytes": source_snapshot.size_bytes,
                "source_sha256": source_snapshot.sha256,
            },
        )
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        if _snapshot_open_file(opened_file, cancelled=cancelled) != source_snapshot:
            raise OSError("源文件在加载期间发生变化，已拒绝使用不一致的数据")
        _confirm_open_source_identity(source_path, opened_file)
        raise_if_cancelled(cancelled, message=_LOAD_CANCEL_MESSAGE)
        return series


def save_csv_timeseries(
    path: str | Path,
    time_s: object,
    values: object,
    time_scale_to_s: float = 1.0,
    *,
    cancelled: CancellationCheck | None = None,
) -> Path:
    """Write time plus one or more value columns without a header row."""

    if not np.isfinite(time_scale_to_s) or time_scale_to_s <= 0.0:
        raise ValueError("CSV 导出时间单位换算必须是正的有限值")
    if np.iscomplexobj(time_s) or np.iscomplexobj(values):
        raise ValueError("CSV 导出不支持复数时间或复数信号")
    time_array = np.asarray(time_s, dtype=np.float64)
    value_array = np.asarray(values)
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
    for start in range(0, time_array.size, _EXPORT_CHUNK_SAMPLES):
        raise_if_cancelled(cancelled, message="导出已取消")
        stop = min(time_array.size, start + _EXPORT_CHUNK_SAMPLES)
        if not np.all(np.isfinite(time_array[start:stop])) or not np.all(
            np.isfinite(value_array[start:stop])
        ):
            raise ValueError("CSV 导出数据包含 NaN 或 Inf")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        raise_if_cancelled(cancelled, message="导出已取消")
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            for start in range(0, time_array.size, _EXPORT_CHUNK_SAMPLES):
                raise_if_cancelled(cancelled, message="导出已取消")
                stop = min(time_array.size, start + _EXPORT_CHUNK_SAMPLES)
                table = np.empty((stop - start, value_array.shape[1] + 1), dtype=np.float64)
                table[:, 0] = time_array[start:stop] / time_scale_to_s
                table[:, 1:] = value_array[start:stop]
                np.savetxt(stream, table, delimiter=",", fmt="%.17g")
            stream.flush()
            os.fsync(stream.fileno())
        raise_if_cancelled(cancelled, message="导出已取消")
        os.replace(temporary_name, output_path)
    except Exception:
        # 取消可能发生在 os.fdopen 接管 mkstemp 描述符之前；POSIX 允许删除
        # 仍打开的文件而会掩盖泄漏，Windows 则会以共享冲突拒绝删除。
        with suppress(OSError):
            os.close(descriptor)
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
    cancelled: CancellationCheck | None = None,
) -> Path:
    """Atomically write one re-importable Keysight Infiniium AG waveform."""

    if np.iscomplexobj(time_s) or np.iscomplexobj(values):
        raise ValueError("BIN 导出不支持复数时间或复数信号")
    time_array = np.asarray(time_s, dtype=np.float64)
    value_array = np.asarray(values)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("BIN 导出时间和值必须是一维非空数组")
    if value_array.ndim == 2 and value_array.shape[1] == 1:
        value_array = value_array[:, 0]
    if value_array.ndim != 1 or value_array.size != time_array.size:
        raise ValueError("BIN 导出当前只支持与时间等长的单通道值")
    if time_array.size < 8:
        raise ValueError("BIN 导出至少需要 8 个样本")
    # 从完整记录跨度恢复标称间隔，避免有限时间原点下第一对 float64 时间戳的
    # 舍入误差直接写进 XIncrement。后续仍逐块检查每个实际间隔，不能用端点
    # 平均掩盖真正的非均匀时间轴。
    interval_s = float(
        (time_array[-1] - time_array[0]) / float(time_array.size - 1)
    )
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("BIN 导出时间必须严格递增且等间隔")
    for start in range(0, time_array.size, _EXPORT_CHUNK_SAMPLES):
        raise_if_cancelled(cancelled, message="导出已取消")
        stop = min(time_array.size, start + _EXPORT_CHUNK_SAMPLES)
        if not np.all(np.isfinite(time_array[start:stop])) or not np.all(
            np.isfinite(value_array[start:stop])
        ):
            raise ValueError("BIN 导出数据包含 NaN 或 Inf")
        interval_stop = min(time_array.size - 1, stop)
        if start < interval_stop:
            intervals_s = (
                time_array[start + 1 : interval_stop + 1] - time_array[start:interval_stop]
            )
            if np.any(intervals_s <= 0.0):
                raise ValueError("BIN 导出时间必须严格递增且等间隔")
            if not np.allclose(
                intervals_s,
                interval_s,
                rtol=_DEFAULT_UNIFORMITY_RTOL,
                atol=0.0,
            ):
                raise ValueError("BIN 导出时间必须等间隔")
    sample_rate_hz = 1.0 / interval_s

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        raise_if_cancelled(cancelled, message="导出已取消")
        write_keysight_bin(
            temporary_name,
            value_array,
            sample_rate_hz,
            x_origin_s=float(time_array[0]),
            label=label,
            cancelled=cancelled,
        )
        # Windows 的 os.fsync/_commit 要求底层句柄具有写权限；只读句柄会报
        # ``OSError: [Errno 9] Bad file descriptor``。writer 已在原子提交前完成
        # flush + fsync，这里保留二次落盘确认时也必须用可写二进制句柄。
        with open(temporary_name, "r+b") as stream:
            os.fsync(stream.fileno())
        raise_if_cancelled(cancelled, message="导出已取消")
        os.replace(temporary_name, output_path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return output_path
