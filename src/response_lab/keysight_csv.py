"""Strict layouts for Infiniium WaveformXYValues and compatible XY CSV headers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_MAX_HEADER_BYTES = 1024 * 1024
_MAX_HEADER_LINES = 256
_MAX_HEADER_LINE_BYTES = 64 * 1024
_SUPPORTED_FORMAT_VERSIONS = {1, 2}
_REQUIRED_FIELDS = {
    "file format",
    "format version",
    "points",
    "x units",
    "y units",
}


@dataclass(frozen=True)
class KeysightCsvLayout:
    """Validated location and metadata of one Infiniium XY waveform table."""

    path: Path
    container: str
    format_reference: str
    data_offset: int
    format_version: int | None
    points: int | None
    x_precision: str
    y_precision: str
    instrument: str | None
    software_version: str | None
    serial_number: str | None
    acquisition_date: str | None
    signal_type: str | None
    source_name: str | None
    channel_noise: str | None
    intrinsic_jitter: str | None
    interpolation_factor: str | None
    bandwidth: str | None
    x_units: str
    y_units: str


def _decode_header_line(raw_line: bytes) -> str:
    try:
        return raw_line.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Keysight CSV 表头必须是 UTF-8/ASCII 文本") from error


def _parse_csv_record(raw_line: bytes) -> tuple[str, ...]:
    text = _decode_header_line(raw_line)
    try:
        row = next(csv.reader([text], skipinitialspace=True, strict=True))
    except csv.Error as error:
        raise ValueError(f"Keysight CSV 表头记录无法解析：{error}") from error
    return tuple(cell.strip() for cell in row)


def _read_header_line(
    handle: BinaryIO,
    *,
    line_number: int,
    total_bytes: int,
) -> tuple[bytes, int]:
    if line_number >= _MAX_HEADER_LINES:
        raise ValueError(f"Keysight CSV 在 {_MAX_HEADER_LINES} 行内未找到完整 Data 表头")
    raw_line = handle.readline(_MAX_HEADER_LINE_BYTES + 1)
    if len(raw_line) > _MAX_HEADER_LINE_BYTES:
        raise ValueError("Keysight CSV 表头单行超过 64 KiB 安全上限")
    total_bytes += len(raw_line)
    if total_bytes > _MAX_HEADER_BYTES:
        raise ValueError("Keysight CSV 表头超过 1 MiB 安全上限")
    return raw_line, total_bytes


def _record_value(
    record: tuple[str, ...],
    *,
    field_name: str,
    required: bool = True,
) -> str:
    if len(record) != 2:
        raise ValueError(f"Keysight CSV 字段 {field_name!r} 必须恰好包含名称和值两列")
    if required and not record[1]:
        raise ValueError(f"Keysight CSV 字段 {field_name!r} 缺少值")
    return record[1]


def _validate_first_data_record(handle: BinaryIO, data_offset: int) -> None:
    """Reject an empty or non-XY data table before NumPy allocates its output."""

    original_position = handle.tell()
    try:
        handle.seek(data_offset)
        while True:
            raw_line = handle.readline(_MAX_HEADER_LINE_BYTES + 1)
            if not raw_line:
                raise ValueError("Keysight CSV 数据区为空")
            if len(raw_line) > _MAX_HEADER_LINE_BYTES:
                raise ValueError("Keysight CSV 数据行超过 64 KiB 安全上限")
            record = _parse_csv_record(raw_line)
            if not any(record):
                continue
            if len(record) != 2 or not all(record):
                raise ValueError("Keysight CSV 数据区必须恰好包含 time, voltage 两列")
            return
    finally:
        handle.seek(original_position)


def _inspect_keysight_csv_from_open_file(
    path: Path,
    handle: BinaryIO,
) -> KeysightCsvLayout | None:
    """Inspect a shared descriptor, returning ``None`` for generic headerless CSV."""

    original_position = handle.tell()
    line_number = 0
    total_bytes = 0
    try:
        handle.seek(0)
        first_record: tuple[str, ...] | None = None
        while first_record is None:
            raw_line, total_bytes = _read_header_line(
                handle,
                line_number=line_number,
                total_bytes=total_bytes,
            )
            if not raw_line:
                return None
            line_number += 1
            record = _parse_csv_record(raw_line)
            if any(record):
                first_record = record

        if first_record[0] == "Time (s)":
            if (
                len(first_record) != 2
                or not first_record[1].endswith(" (V)")
                or not first_record[1][:-4].strip()
            ):
                raise ValueError(
                    "Keysight 官方 Python XY CSV 必须是 'Time (s),<source> (V)' 两列表头"
                )
            data_offset = handle.tell()
            _validate_first_data_record(handle, data_offset)
            return KeysightCsvLayout(
                path=path,
                container="time_voltage_header_csv",
                format_reference="keysight_infiniium_python_example",
                data_offset=data_offset,
                format_version=None,
                points=None,
                x_precision="unknown",
                y_precision="unknown",
                instrument=None,
                software_version=None,
                serial_number=None,
                acquisition_date=None,
                signal_type=None,
                source_name=first_record[1][:-4].strip(),
                channel_noise=None,
                intrinsic_jitter=None,
                interpolation_factor=None,
                bandwidth=None,
                x_units="Second",
                y_units="Volt",
            )
        if first_record[0] == "Revision":
            raise ValueError(
                "检测到旧式 Keysight Revision CSV；当前缺少可验证的真实多源/分段样例，"
                "因此明确不支持该格式"
            )
        if first_record[0] != "File Format":
            return None
        file_format = _record_value(first_record, field_name="File Format")
        if file_format != "WaveformXYValues":
            raise ValueError(
                "不支持的 Keysight CSV File Format："
                f"{file_format!r}；ResponseLab 仅接受 WaveformXYValues 时域波形"
            )

        fields: dict[str, str] = {"file format": file_format}
        data_seen = False
        while not data_seen:
            raw_line, total_bytes = _read_header_line(
                handle,
                line_number=line_number,
                total_bytes=total_bytes,
            )
            if not raw_line:
                raise ValueError("Keysight CSV 缺少 Data 数据区标记")
            line_number += 1
            record = _parse_csv_record(raw_line)
            if not any(record):
                continue
            field_name = record[0]
            if field_name == "Data":
                if record != ("Data", ""):
                    raise ValueError("Keysight CSV Data 标记必须恰好写成 'Data,'")
                data_seen = True
                break
            normalized_name = field_name.casefold()
            if normalized_name in fields:
                raise ValueError(f"Keysight CSV 字段 {field_name!r} 重复")
            fields[normalized_name] = _record_value(
                record,
                field_name=field_name,
                required=normalized_name in _REQUIRED_FIELDS,
            )

        missing = sorted(_REQUIRED_FIELDS.difference(fields))
        if missing:
            raise ValueError(f"Keysight CSV 缺少必需字段：{', '.join(missing)}")
        try:
            format_version = int(fields["format version"])
        except ValueError as error:
            raise ValueError("Keysight CSV Format Version 必须是整数") from error
        if format_version not in _SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"不支持的 Keysight CSV Format Version：{format_version}；当前仅支持 1 和 2"
            )
        try:
            points = int(fields["points"])
        except ValueError as error:
            raise ValueError("Keysight CSV Points 必须是整数") from error
        if points <= 0:
            raise ValueError("Keysight CSV Points 必须为正整数")

        x_units = fields["x units"]
        y_units = fields["y units"]
        if x_units.casefold() != "second":
            raise ValueError(f"Keysight CSV X Units 必须是 Second，实际为 {x_units!r}")
        if y_units.casefold() != "volt":
            raise ValueError(f"Keysight CSV Y Units 必须是 Volt，实际为 {y_units!r}")

        x_precision = "double"
        y_precision = "double"
        if format_version == 2:
            precision_record: tuple[str, ...] | None = None
            while precision_record is None:
                raw_line, total_bytes = _read_header_line(
                    handle,
                    line_number=line_number,
                    total_bytes=total_bytes,
                )
                if not raw_line:
                    raise ValueError("Keysight CSV v2 缺少 Data 后的精度声明行")
                line_number += 1
                candidate = _parse_csv_record(raw_line)
                if any(candidate):
                    precision_record = candidate
            if len(precision_record) != 2:
                raise ValueError("Keysight CSV v2 精度声明必须恰好包含 X/Y 两列")
            x_precision, y_precision = (cell.casefold() for cell in precision_record)
            if x_precision not in {"float", "double"} or y_precision not in {
                "float",
                "double",
            }:
                raise ValueError("Keysight CSV v2 精度声明仅支持 float 或 double")

        data_offset = handle.tell()
        _validate_first_data_record(handle, data_offset)
        return KeysightCsvLayout(
            path=path,
            container="keysight_infiniium_waveform_xy",
            format_reference="keysight_infiniium_waveform_xy_v1_v2",
            data_offset=data_offset,
            format_version=format_version,
            points=points,
            x_precision=x_precision,
            y_precision=y_precision,
            instrument=fields.get("instrument") or None,
            software_version=fields.get("swversion") or None,
            serial_number=fields.get("serialnumber") or None,
            acquisition_date=fields.get("date") or None,
            signal_type=fields.get("signal type") or None,
            source_name=fields.get("source name") or None,
            channel_noise=fields.get("channel noise") or None,
            intrinsic_jitter=fields.get("intrinsic jitter") or None,
            interpolation_factor=fields.get("interpolation factor") or None,
            bandwidth=fields.get("bandwidth") or None,
            x_units=x_units,
            y_units=y_units,
        )
    finally:
        handle.seek(original_position)


def inspect_keysight_csv(path: str | Path) -> KeysightCsvLayout:
    """Validate and describe one self-identifying Infiniium WaveformXYValues CSV."""

    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as handle:
        layout = _inspect_keysight_csv_from_open_file(source_path, handle)
    if layout is None or layout.container != "keysight_infiniium_waveform_xy":
        raise ValueError(
            "CSV 没有 Keysight WaveformXYValues 表头；其他受支持的 CSV "
            "仍可通过 load_csv_timeseries 读取"
        )
    return layout
