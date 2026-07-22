"""Generic/Keysight CSV and Keysight Infiniium BIN public I/O behavior tests."""

from __future__ import annotations

import hashlib
import os
import struct

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import response_lab.io as io_module
from response_lab.io import (
    load_bin_timeseries,
    load_csv_timeseries,
    save_bin_timeseries,
    save_csv_timeseries,
)

_FILE_HEADER = struct.Struct("<2s2sii")
_WAVE_HEADER = struct.Struct("<iiiiifdddii16s16s24s16sdI")
_DATA_HEADER = struct.Struct("<ihhi")


def _write_infiniium_fixture(
    path,
    values,
    *,
    x_increment_s=0.25e-9,
    x_origin_s=-1.0e-9,
):
    """Use an independent struct encoder so the public loader is not its own oracle."""

    encoded = np.asarray(values, dtype="<f4")
    payload = encoded.tobytes()
    wave_header = _WAVE_HEADER.pack(
        _WAVE_HEADER.size,
        1,
        1,
        encoded.size,
        0,
        encoded.size * x_increment_s,
        x_origin_s,
        x_increment_s,
        x_origin_s,
        2,
        1,
        b"22 JUL 2026".ljust(16, b"\0"),
        b"12:00:00".ljust(16, b"\0"),
        b"D9300A:TEST".ljust(24, b"\0"),
        b"Channel 1".ljust(16, b"\0"),
        0.0,
        0,
    )
    data_header = _DATA_HEADER.pack(_DATA_HEADER.size, 1, 4, len(payload))
    file_size = _FILE_HEADER.size + len(wave_header) + len(data_header) + len(payload)
    path.write_bytes(
        _FILE_HEADER.pack(b"AG", b"10", file_size, 1)
        + wave_header
        + data_header
        + payload
    )
    return path


def _write_keysight_xy_fixture(
    path,
    *,
    points=8,
    version=2,
    newline="\n",
    time_s=None,
):
    """Freeze the documented Infiniium WaveformXYValues layout independently."""

    times = (
        np.arange(points, dtype=np.float64) * 2.5e-10
        if time_s is None
        else np.asarray(time_s, dtype=np.float64)
    )
    rows = [f"{time_value:.16E}, {0.125 * index:.7g}" for index, time_value in enumerate(times)]
    precision_row = ["double, float"] if version == 2 else []
    path.write_text(
        newline.join(
            [
                "File Format, WaveformXYValues",
                f"Format Version, {version}",
                "Instrument, D9300A",
                "SwVersion, P.26.01.214",
                "SerialNumber, MXL3441NC2",
                "Date, 3/30/2026 13:33:48 GMT-06:00",
                "",
                f"Points, {points}",
                "Signal Type, Unspecified",
                "Source Name, Channel 1",
                "Channel Noise, 0.002",
                "Intrinsic Jitter, 1E-14",
                "Interpolation Factor, 0",
                "Bandwidth, 128000000000",
                "X Units, Second",
                "Y Units, Volt",
                "Data,",
                *precision_row,
                *rows,
            ]
        )
        + newline,
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("time_unit", "scale_to_s"),
    [
        ("s", 1.0),
        ("ms", 1.0e-3),
        ("us", 1.0e-6),
        ("µs", 1.0e-6),
        ("ns", 1.0e-9),
        ("ps", 1.0e-12),
    ],
)
def test_csv_supports_explicit_time_units(tmp_path, time_unit, scale_to_s) -> None:
    path = tmp_path / f"pulse-{time_unit}.csv"
    expected_time_s = np.arange(8, dtype=np.float64) * 2.0e-9
    raw_time = expected_time_s / scale_to_s
    np.savetxt(path, np.column_stack((raw_time, np.arange(8))), delimiter=",")

    series = load_csv_timeseries(path, time_unit=time_unit)

    assert series.sample_rate_hz == pytest.approx(5.0e8)
    np.testing.assert_allclose(series.time_s, expected_time_s)


def test_keysight_waveform_xy_v2_uses_documented_header_and_data_contract(tmp_path) -> None:
    expected_time_s = -1.0e-9 + np.arange(8, dtype=np.float64) * 0.25e-9
    path = _write_keysight_xy_fixture(
        tmp_path / "channel1.csv",
        time_s=expected_time_s,
    )

    series = load_csv_timeseries(path)

    assert series.samples == 8
    assert series.sample_rate_hz == pytest.approx(4.0e9)
    np.testing.assert_allclose(series.time_s, expected_time_s, rtol=0.0, atol=1.0e-24)
    np.testing.assert_allclose(series.values[:, 0], 0.125 * np.arange(8))
    assert series.source_metadata["container"] == "keysight_infiniium_waveform_xy"
    assert series.source_metadata["keysight_format_version"] == 2
    assert series.source_metadata["instrument"] == "D9300A"
    assert series.source_metadata["source_name"] == "Channel 1"
    assert series.source_metadata["acquisition_date"] == "3/30/2026 13:33:48 GMT-06:00"
    assert series.source_metadata["interpolation_factor"] == "0"
    assert series.source_metadata["bandwidth"] == "128000000000"
    assert series.source_metadata["x_units"] == "Second"
    assert series.source_metadata["y_units"] == "Volt"
    assert series.source_metadata["headerless"] is False
    assert series.source_metadata["sample_rate_source"] == "keysight_xy_time_column"


def test_keysight_waveform_xy_v1_has_no_precision_row(tmp_path) -> None:
    path = _write_keysight_xy_fixture(
        tmp_path / "channel1-v1.csv",
        version=1,
        newline="\r\n",
    )

    series = load_csv_timeseries(path)

    assert series.samples == 8
    assert series.source_metadata["keysight_format_version"] == 1
    assert series.source_metadata["x_precision"] == "double"
    assert series.source_metadata["y_precision"] == "double"


def test_keysight_official_python_xy_header_uses_seconds_and_volts(tmp_path) -> None:
    path = tmp_path / "my_data_CHANnel1.csv"
    time_s = -1.0e-9 + np.arange(8, dtype=np.float64) * 0.5e-9
    values = np.linspace(-0.4, 0.4, 8)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("Time (s),CHANnel1 (V)\n")
        np.savetxt(stream, np.column_stack((time_s, values)), delimiter=",")

    series = load_csv_timeseries(path)

    assert series.sample_rate_hz == pytest.approx(2.0e9)
    np.testing.assert_allclose(series.time_s, time_s)
    np.testing.assert_allclose(series.values[:, 0], values)
    assert series.source_metadata["container"] == "time_voltage_header_csv"
    assert series.source_metadata["format_reference"] == "keysight_infiniium_python_example"
    assert series.source_metadata["source_name"] == "CHANnel1"
    assert series.source_metadata["x_units"] == "Second"
    assert series.source_metadata["y_units"] == "Volt"
    assert series.source_metadata["x_precision"] == "unknown"
    assert series.source_metadata["y_precision"] == "unknown"
    assert series.source_metadata["sample_rate_source"] == "csv_time_column"
    assert series.source_metadata["headerless"] is False


def test_keysight_python_xy_header_rejects_non_voltage_data(tmp_path) -> None:
    path = tmp_path / "current.csv"
    rows = np.column_stack((np.arange(8, dtype=np.float64), np.arange(8)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("Time (s),CHANnel1 (A)\n")
        np.savetxt(stream, rows, delimiter=",")

    with pytest.raises(ValueError, match=r"Keysight.*\(V\).*"):
        load_csv_timeseries(path)


def test_legacy_keysight_revision_csv_fails_with_explicit_boundary(tmp_path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "Revision, 0\nType, Voltage\nData,\n0, 0\n1e-9, 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="旧式 Keysight.*Revision.*不支持"):
        load_csv_timeseries(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("File Format, WaveformXYValues", 'File Format, "WaveformXYValues'),
        ("X Units, Second", 'X Units, "Second'),
        ("Y Units, Volt", 'Y Units, "Volt'),
    ],
)
def test_keysight_waveform_xy_rejects_unclosed_header_quotes(
    tmp_path,
    original,
    replacement,
) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "broken-quote.csv")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="表头记录无法解析"):
        load_csv_timeseries(path)


def test_keysight_waveform_xy_allows_empty_optional_metadata(tmp_path) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "empty-optional.csv")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "Signal Type, Unspecified",
            "Signal Type,",
            1,
        ),
        encoding="utf-8",
    )

    series = load_csv_timeseries(path)

    assert series.source_metadata["signal_type"] is None


@pytest.mark.parametrize("replacement", ["Data", "Data,,,"])
def test_keysight_waveform_xy_requires_exact_data_marker(tmp_path, replacement) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "bad-data-marker.csv")
    path.write_text(
        path.read_text(encoding="utf-8").replace("Data,", replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Data 标记"):
        load_csv_timeseries(path)


def test_keysight_waveform_xy_rejects_declared_points_mismatch(tmp_path) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "wrong-points.csv", points=9)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Points.*声明 9，实际 8"):
        load_csv_timeseries(path)


@pytest.mark.parametrize("row_offset", [0, 4])
def test_keysight_waveform_xy_rejects_extra_data_column(tmp_path, row_offset) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "three-columns.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    data_index = lines.index("double, float") + 1 + row_offset
    lines[data_index] += ", 99"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="恰好.*两列|列数"):
        load_csv_timeseries(path)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("File Format, WaveformXYValues", "File Format, DatabaseCsv", "DatabaseCsv"),
        ("Format Version, 2", "Format Version, 3", "仅支持 1 和 2"),
        ("X Units, Second", "X Units, Hertz", "X Units.*Second"),
        ("Y Units, Volt", "Y Units, Watt", "Y Units.*Volt"),
        ("double, float", "double, decimal", "精度声明"),
    ],
)
def test_keysight_waveform_xy_rejects_wrong_family_version_units_and_precision(
    tmp_path,
    original,
    replacement,
    message,
) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "invalid.csv")
    text = path.read_text(encoding="utf-8").replace(original, replacement, 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_csv_timeseries(path)


def test_keysight_nonuniform_xy_points_recommend_instrument_linear_interpolation(
    tmp_path,
) -> None:
    intervals_s = 0.25e-9 * np.array([1.0, 1.0, 1.001, 0.999, 1.0, 1.0, 1.0])
    time_s = np.concatenate(([0.0], np.cumsum(intervals_s)))
    path = _write_keysight_xy_fixture(tmp_path / "nonuniform.csv", time_s=time_s)

    with pytest.raises(ValueError, match="Linearly Interpolate"):
        load_csv_timeseries(path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_keysight_waveform_xy_rejects_missing_or_duplicate_required_fields(
    tmp_path,
    mutation,
) -> None:
    path = _write_keysight_xy_fixture(tmp_path / f"{mutation}.csv")
    text = path.read_text(encoding="utf-8")
    if mutation == "missing":
        text = text.replace("Points, 8\n", "", 1)
        message = "缺少必需字段.*points"
    else:
        text = text.replace("X Units, Second\n", "X Units, Second\nX Units, Second\n", 1)
        message = "X Units.*重复"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_csv_timeseries(path)


def test_keysight_waveform_xy_allows_unknown_future_metadata(tmp_path) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "future-field.csv")
    text = path.read_text(encoding="utf-8").replace(
        "Data,\n",
        "Future Metadata, preserved by instrument\nData,\n",
        1,
    )
    path.write_text(text, encoding="utf-8")

    series = load_csv_timeseries(path)

    assert series.samples == 8
    assert series.source_metadata["keysight_format_version"] == 2


def test_keysight_xy_header_rejects_manual_unit_or_column_override(tmp_path) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "fixed-columns.csv")

    with pytest.raises(ValueError, match="表头固定.*不能覆盖"):
        load_csv_timeseries(path, time_unit="ns")


def test_keysight_csv_dynamic_budget_rejects_before_loadtxt(tmp_path, monkeypatch) -> None:
    path = _write_keysight_xy_fixture(tmp_path / "budget.csv")
    monkeypatch.setattr(
        io_module,
        "system_available_memory_bytes",
        lambda: 96 * 1024**2,
        raising=False,
    )

    def forbidden_loadtxt(*_args, **_kwargs):
        raise AssertionError("Keysight CSV budget must run before np.loadtxt")

    monkeypatch.setattr(io_module.np, "loadtxt", forbidden_loadtxt)

    with pytest.raises(MemoryError, match="CSV.*动态内存.*解析前"):
        load_csv_timeseries(path)


def test_keysight_csv_dynamic_budget_scales_with_file_rows_and_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    small_path = _write_keysight_xy_fixture(tmp_path / "small.csv", points=8)
    large_path = _write_keysight_xy_fixture(tmp_path / "large.csv", points=2_000)

    def estimated_bytes(path) -> int:
        payload = path.read_bytes()
        return io_module._estimate_csv_loader_peak_bytes(
            file_size_bytes=len(payload),
            physical_rows=payload.count(b"\n"),
            selected_columns=2,
            resample_nonuniform=False,
        )

    small_estimate = estimated_bytes(small_path)
    large_estimate = estimated_bytes(large_path)
    assert large_estimate > small_estimate
    budget = (small_estimate + large_estimate) // 2
    monkeypatch.setattr(io_module, "system_available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(io_module, "safe_memory_budget_bytes", lambda _available: budget)
    real_loadtxt = io_module.np.loadtxt
    parsed_paths: list[str] = []

    def recording_loadtxt(handle, *args, **kwargs):
        parsed_paths.append(str(handle.name))
        if str(handle.name) == str(large_path):
            raise AssertionError("large Keysight CSV must be rejected before np.loadtxt")
        return real_loadtxt(handle, *args, **kwargs)

    monkeypatch.setattr(io_module.np, "loadtxt", recording_loadtxt)

    assert load_csv_timeseries(small_path).samples == 8
    with pytest.raises(MemoryError, match="动态内存预检拒绝"):
        load_csv_timeseries(large_path)
    assert parsed_paths == [str(small_path)]


def test_csv_rejects_unknown_time_unit(tmp_path) -> None:
    path = tmp_path / "pulse.csv"
    np.savetxt(path, np.column_stack((np.arange(8), np.arange(8))), delimiter=",")

    with pytest.raises(ValueError, match="时间单位"):
        load_csv_timeseries(path, time_unit="minute")


def test_headerless_csv_keeps_first_row_and_derives_sample_rate(tmp_path) -> None:
    path = tmp_path / "pulse.csv"
    time_ns = np.arange(8, dtype=np.float64) * 0.5
    values = np.array([123.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    np.savetxt(path, np.column_stack((time_ns, values)), delimiter=",")

    series = load_csv_timeseries(path, time_unit="ns")

    assert series.samples == 8
    assert series.values[0, 0] == pytest.approx(123.0)
    assert series.sample_rate_hz == pytest.approx(2.0e9)
    np.testing.assert_allclose(series.time_s, time_ns * 1.0e-9)
    assert series.source_metadata["headerless"] is True
    assert series.source_metadata["resampled_with_pchip"] is False
    assert series.source_metadata["source_size_bytes"] == path.stat().st_size
    assert series.source_metadata["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("delimiter", [",", ";", "\t", " "])
def test_csv_detects_delimiter_and_honors_selected_columns(tmp_path, delimiter) -> None:
    path = tmp_path / "selected-columns.txt"
    sample_index = np.arange(8, dtype=np.float64)
    time_us = sample_index * 0.25
    channel_a = 10.0 + sample_index
    channel_b = 20.0 + 2.0 * sample_index
    np.savetxt(
        path,
        np.column_stack((sample_index, time_us, channel_a, channel_b)),
        delimiter=delimiter,
    )

    series = load_csv_timeseries(
        path,
        time_unit="us",
        time_column=1,
        value_columns=(3, 2),
    )

    assert series.sample_rate_hz == pytest.approx(4.0e6)
    np.testing.assert_allclose(series.values[:, 0], channel_b)
    np.testing.assert_allclose(series.values[:, 1], channel_a)


def test_csv_uses_only_selected_columns_and_ignores_unselected_text(
    tmp_path,
    monkeypatch,
) -> None:
    """多余列不能扩大数值表，也不能因未选择的文本内容导致假失败。"""

    path = tmp_path / "wide-selected.csv"
    rows = [f"{index},unused-{index},{10.0 + index},ignored" for index in range(8)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    real_loadtxt = io_module.np.loadtxt
    observed_usecols: list[tuple[int, ...] | None] = []

    def recording_loadtxt(*args, **kwargs):
        observed_usecols.append(kwargs.get("usecols"))
        return real_loadtxt(*args, **kwargs)

    monkeypatch.setattr(io_module.np, "loadtxt", recording_loadtxt)
    series = load_csv_timeseries(path, time_column=0, value_columns=(2,))

    assert observed_usecols == [(0, 2)]
    np.testing.assert_allclose(series.values[:, 0], 10.0 + np.arange(8))


def test_csv_dynamic_budget_rejects_before_loadtxt(tmp_path, monkeypatch) -> None:
    """低可用内存时文本解析必须在 NumPy 建表前停止。"""

    path = tmp_path / "budget.csv"
    np.savetxt(path, np.column_stack((np.arange(8), np.arange(8))), delimiter=",")
    monkeypatch.setattr(
        io_module,
        "system_available_memory_bytes",
        lambda: 96 * 1024**2,
        raising=False,
    )

    def forbidden_loadtxt(*_args, **_kwargs):
        raise AssertionError("CSV budget must run before np.loadtxt")

    monkeypatch.setattr(io_module.np, "loadtxt", forbidden_loadtxt)

    with pytest.raises(MemoryError, match="CSV.*动态内存.*解析前"):
        load_csv_timeseries(path)


@pytest.mark.parametrize(
    "column_options",
    [
        {"time_column": -1},
        {"value_columns": ()},
        {"value_columns": (0,)},
        {"value_columns": (2,)},
        {"value_columns": (1, 1)},
    ],
)
def test_csv_rejects_invalid_column_mapping(tmp_path, column_options) -> None:
    path = tmp_path / "two-columns.csv"
    np.savetxt(path, np.column_stack((np.arange(8), np.arange(8))), delimiter=",")

    with pytest.raises(ValueError, match="列"):
        load_csv_timeseries(path, **column_options)


def test_csv_rejects_nonfinite_nonincreasing_and_short_inputs(tmp_path) -> None:
    valid = np.column_stack((np.arange(8, dtype=np.float64), np.arange(8)))

    nonfinite = valid.copy()
    nonfinite[3, 1] = np.nan
    nonfinite_path = tmp_path / "nonfinite.csv"
    np.savetxt(nonfinite_path, nonfinite, delimiter=",")
    with pytest.raises(ValueError, match="NaN|Inf|有限"):
        load_csv_timeseries(nonfinite_path)

    nonincreasing = valid.copy()
    nonincreasing[4, 0] = nonincreasing[3, 0]
    nonincreasing_path = tmp_path / "nonincreasing.csv"
    np.savetxt(nonincreasing_path, nonincreasing, delimiter=",")
    with pytest.raises(ValueError, match="严格递增"):
        load_csv_timeseries(nonincreasing_path)

    short_path = tmp_path / "short.csv"
    np.savetxt(short_path, valid[:7], delimiter=",")
    with pytest.raises(ValueError, match="至少.*8"):
        load_csv_timeseries(short_path)


def test_csv_does_not_silently_resample_nonuniform_time(tmp_path) -> None:
    path = tmp_path / "jittered.csv"
    intervals_s = 1.0e-6 * np.array([1.0, 1.0, 1.001, 0.999, 1.0, 1.0, 1.0])
    time_s = np.concatenate(([0.0], np.cumsum(intervals_s)))
    np.savetxt(path, np.column_stack((time_s, np.sin(np.arange(8)))), delimiter=",")

    with pytest.raises(ValueError, match="非均匀"):
        load_csv_timeseries(path)


def test_csv_can_explicitly_pchip_resample_light_jitter(tmp_path) -> None:
    path = tmp_path / "jittered.csv"
    intervals_s = 1.0e-6 * np.array([1.0, 1.0, 1.001, 0.999, 1.0, 1.0, 1.0])
    time_s = np.concatenate(([0.0], np.cumsum(intervals_s)))
    values = (time_s / 1.0e-6) ** 2
    np.savetxt(path, np.column_stack((time_s, values)), delimiter=",")

    series = load_csv_timeseries(path, resample_nonuniform=True)

    expected_time_s = np.arange(8, dtype=np.float64) * 1.0e-6
    expected_values = PchipInterpolator(time_s, values)(expected_time_s)
    assert series.sample_rate_hz == pytest.approx(1.0e6)
    np.testing.assert_allclose(series.time_s, expected_time_s, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(series.values[:, 0], expected_values)


def test_csv_rejects_accumulated_phase_error_despite_small_local_jitter(tmp_path) -> None:
    path = tmp_path / "cumulative-drift.csv"
    base_interval_s = 1.0e-6
    fractional_shift = 0.8e-6
    intervals_s = base_interval_s * np.concatenate(
        (
            np.full(8_000, 1.0 + fractional_shift),
            np.full(8_000, 1.0 - fractional_shift),
        )
    )
    time_s = np.concatenate(([0.0], np.cumsum(intervals_s)))
    np.savetxt(path, np.column_stack((time_s, np.sin(time_s))), delimiter=",")

    with pytest.raises(ValueError, match="累计残差.*1°"):
        load_csv_timeseries(path)

    series = load_csv_timeseries(path, resample_nonuniform=True)

    assert series.source_metadata["maximum_relative_interval_deviation"] <= 1.0e-6
    assert series.source_metadata["maximum_cumulative_time_residual_s"] > 0.0
    assert series.source_metadata["nyquist_phase_error_deg"] > 1.0
    assert series.source_metadata["resampled_with_pchip"] is True
    uniform_interval_s = float(np.median(np.diff(series.time_s)))
    expected_time_s = (
        series.time_s[0]
        + np.arange(series.samples, dtype=np.float64) * uniform_interval_s
    )
    resulting_phase_error_deg = np.degrees(
        np.pi * np.max(np.abs(series.time_s - expected_time_s)) / uniform_interval_s
    )
    assert resulting_phase_error_deg <= 1.0


def test_csv_refuses_to_resample_large_time_jitter(tmp_path) -> None:
    path = tmp_path / "large-jitter.csv"
    intervals_s = 1.0e-6 * np.array([1.0, 1.0, 1.10, 0.90, 1.0, 1.0, 1.0])
    time_s = np.concatenate(([0.0], np.cumsum(intervals_s)))
    np.savetxt(path, np.column_stack((time_s, np.arange(8))), delimiter=",")

    with pytest.raises(ValueError, match="偏差过大"):
        load_csv_timeseries(path, resample_nonuniform=True)


def test_csv_pchip_resampling_does_not_extrapolate_past_input_time(tmp_path) -> None:
    path = tmp_path / "bounded-resampling.csv"
    intervals_s = 1.0e-6 * np.array([1.02] * 5 + [0.98] * 4)
    time_s = np.concatenate(([0.0], np.cumsum(intervals_s)))
    values = np.cos(time_s / 1.0e-6)
    np.savetxt(path, np.column_stack((time_s, values)), delimiter=",")

    series = load_csv_timeseries(path, resample_nonuniform=True)

    assert series.samples == 9
    assert series.time_s[-1] <= time_s[-1]


def test_bin_automatically_reads_sample_rate_origin_and_values(tmp_path) -> None:
    path = tmp_path / "capture.bin"
    expected = np.linspace(-0.75, 0.75, 16, dtype=np.float32)
    _write_infiniium_fixture(path, expected)

    series = load_bin_timeseries(path)

    assert series.sample_rate_hz == pytest.approx(4.0e9)
    assert series.time_s[0] == pytest.approx(-1.0e-9)
    np.testing.assert_allclose(series.values[:, 0], expected, rtol=0.0, atol=0.0)
    assert series.source_metadata["sample_rate_source"] == "keysight_x_increment"
    assert series.source_metadata["keysight_version"] == "10"
    assert series.source_metadata["source_size_bytes"] == path.stat().st_size
    assert series.source_metadata["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_bin_builds_model_through_validated_uniform_axis_path(
    tmp_path,
    monkeypatch,
) -> None:
    """自描述 BIN 不应走会复制并逐点重验完整时间列的通用构造路径。"""

    path = tmp_path / "uniform-fast-path.bin"
    expected = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    _write_infiniium_fixture(path, expected)
    real_time_series = io_module.TimeSeries
    recorded: dict[str, object] = {}

    class UniformPathOnly:
        def __new__(cls, *_args, **_kwargs):
            raise AssertionError("BIN loader must not call the generic TimeSeries constructor")

        @classmethod
        def from_uniform_samples(cls, **kwargs):
            recorded.update(kwargs)
            return real_time_series.from_uniform_samples(**kwargs)

    monkeypatch.setattr(io_module, "TimeSeries", UniformPathOnly)

    series = load_bin_timeseries(path)

    assert recorded["time_increment_s"] == pytest.approx(0.25e-9)
    assert recorded["time_origin_s"] == pytest.approx(-1.0e-9)
    np.testing.assert_array_equal(series.values[:, 0], expected)


def test_bin_loader_estimate_tracks_optimized_child_process_rss() -> None:
    """预检应覆盖新加载路径实测，但不能继续按旧的多重时间轴分配误报。"""

    estimated = (
        1_000_000 * io_module._BIN_LOADER_BYTES_PER_SAMPLE  # noqa: SLF001
        + io_module._BIN_LOADER_FIXED_OVERHEAD_BYTES  # noqa: SLF001
    )

    # 独立 macOS arm64 进程实测新增峰值 27,787,264 B；上界防止退回旧 145 MB
    # 估算。固定余量仍需覆盖分块间隔、文件映射页、哈希缓冲和分配器差异。
    assert 27_787_264 <= estimated < 80 * 1024**2


def test_uniform_bin_axis_rejects_increment_below_float64_origin_resolution(
    tmp_path,
) -> None:
    """正的 XIncrement 若在巨大 XOrigin 下无法形成递增 float64 时间轴也必须拒绝。"""

    path = tmp_path / "unrepresentable-time-axis.bin"
    _write_infiniium_fixture(
        path,
        np.arange(16, dtype=np.float32),
        x_increment_s=0.25e-9,
        x_origin_s=1.0e16,
    )

    with pytest.raises(ValueError, match="float64.*严格递增"):
        load_bin_timeseries(path)


def test_uniform_bin_axis_rejects_internal_duplicate_rounding(tmp_path) -> None:
    """首尾间隔正常但中间发生舍入重复时，也不能绕过时间轴不变量。"""

    path = tmp_path / "internally-repeated-time-axis.bin"
    _write_infiniium_fixture(
        path,
        np.arange(8, dtype=np.float32),
        x_increment_s=np.spacing(1.0) * 0.75,
        x_origin_s=1.0,
    )

    with pytest.raises(ValueError, match="float64.*严格递增"):
        load_bin_timeseries(path)


def test_uniform_bin_axis_rejects_strict_but_nonuniform_rounding(tmp_path) -> None:
    """舍入后即使递增，实际间隔偏离元数据/Fs 也必须拒绝。"""

    path = tmp_path / "nonuniform-rounded-time-axis.bin"
    _write_infiniium_fixture(
        path,
        np.arange(32, dtype=np.float32),
        x_increment_s=np.spacing(1.0) * 1.1,
        x_origin_s=1.0,
    )

    with pytest.raises(ValueError, match="float64.*等间隔"):
        load_bin_timeseries(path)


def test_bin_rejects_legacy_raw_bytes_instead_of_guessing(tmp_path) -> None:
    path = tmp_path / "legacy-raw.bin"
    path.write_bytes(np.arange(16, dtype="<f4").tobytes())

    with pytest.raises(ValueError, match="Keysight Infiniium|AG"):
        load_bin_timeseries(path)


def test_bin_sample_budget_stops_before_payload_mapping(tmp_path, monkeypatch) -> None:
    path = tmp_path / "over-budget.bin"
    _write_infiniium_fixture(path, np.arange(16, dtype=np.float32))

    def forbidden_payload_load(*_args, **_kwargs):
        raise AssertionError("payload should not be mapped after the sample preflight fails")

    monkeypatch.setattr(
        "response_lab.io._load_keysight_waveform_from_open_file",
        forbidden_payload_load,
    )
    with pytest.raises(MemoryError, match="读取 payload 前"):
        load_bin_timeseries(path, max_samples=8)


def test_bin_dynamic_budget_stops_before_payload_mapping_and_time_axis(
    tmp_path,
    monkeypatch,
) -> None:
    """动态预算必须位于 memmap 与全长 arange 两种分配之前。"""

    path = tmp_path / "dynamic-over-budget.bin"
    _write_infiniium_fixture(path, np.arange(16, dtype=np.float32))
    monkeypatch.setattr(
        io_module,
        "system_available_memory_bytes",
        lambda: 96 * 1024**2,
        raising=False,
    )

    def forbidden_payload_load(*_args, **_kwargs):
        raise AssertionError("BIN budget must run before payload mapping")

    def forbidden_arange(*_args, **_kwargs):
        raise AssertionError("BIN budget must run before full time-axis allocation")

    monkeypatch.setattr(io_module, "_load_keysight_waveform_from_open_file", forbidden_payload_load)
    monkeypatch.setattr(io_module.np, "arange", forbidden_arange)

    with pytest.raises(MemoryError, match="BIN.*动态内存.*payload 前"):
        load_bin_timeseries(path)


def test_bin_high_level_rejects_path_replacement_after_sample_preflight(
    tmp_path,
    monkeypatch,
) -> None:
    """Header budget and payload must belong to one stable opened file identity."""

    path = tmp_path / "selected.bin"
    replacement = tmp_path / "replacement.bin"
    _write_infiniium_fixture(path, np.arange(8, dtype=np.float32))
    _write_infiniium_fixture(
        replacement,
        np.arange(16, dtype=np.float32),
        x_increment_s=0.5e-9,
    )
    original_snapshot = io_module._snapshot_open_file
    snapshot_calls = 0

    def replace_before_first_snapshot(handle):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            os.replace(replacement, path)
        return original_snapshot(handle)

    monkeypatch.setattr(io_module, "_snapshot_open_file", replace_before_first_snapshot)

    with pytest.raises(OSError, match="替换|不一致"):
        load_bin_timeseries(path, max_samples=8)


def test_csv_and_bin_loaders_agree_for_the_same_waveform(tmp_path) -> None:
    values = np.sin(np.arange(32, dtype=np.float64) * 0.3)
    time_s = -1.0e-9 + np.arange(values.size, dtype=np.float64) * 0.25e-9
    csv_path = tmp_path / "capture.csv"
    bin_path = tmp_path / "capture.bin"
    np.savetxt(csv_path, np.column_stack((time_s, values)), delimiter=",")
    _write_infiniium_fixture(bin_path, values, x_origin_s=time_s[0])

    csv_series = load_csv_timeseries(csv_path)
    bin_series = load_bin_timeseries(bin_path)

    assert bin_series.sample_rate_hz == pytest.approx(csv_series.sample_rate_hz)
    np.testing.assert_allclose(bin_series.time_s, csv_series.time_s, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(bin_series.values, csv_series.values, rtol=1.0e-6, atol=1.0e-7)


def test_save_csv_writes_headerless_time_and_all_value_columns(tmp_path) -> None:
    path = tmp_path / "output.csv"
    time_s = np.arange(8, dtype=np.float64) * 2.0e-9
    values = np.column_stack((np.arange(8), -np.arange(8)))

    returned_path = save_csv_timeseries(
        path,
        time_s,
        values,
        1.0e-9,
    )

    exported = np.loadtxt(path, delimiter=",", ndmin=2)
    assert returned_path == path
    assert exported.shape == (8, 3)
    np.testing.assert_allclose(exported[:, 0], time_s / 1.0e-9)
    np.testing.assert_allclose(exported[:, 1:], values)


def test_save_csv_rejects_invalid_shape_scale_and_nonfinite_values(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    time_s = np.arange(8, dtype=np.float64)

    with pytest.raises(ValueError, match="换算"):
        save_csv_timeseries(path, time_s, np.arange(8), time_scale_to_s=0.0)
    with pytest.raises(ValueError, match="非空"):
        save_csv_timeseries(path, np.array([]), np.array([]))
    with pytest.raises(ValueError, match="长度"):
        save_csv_timeseries(path, time_s, np.arange(7))
    with pytest.raises(ValueError, match="值"):
        save_csv_timeseries(path, time_s, np.empty((8, 0)))
    with pytest.raises(ValueError, match="NaN|Inf"):
        save_csv_timeseries(path, time_s, np.full(8, np.nan))
    with pytest.raises(ValueError, match="复数"):
        save_csv_timeseries(path, time_s, np.full(8, 1.0 + 2.0j))


def test_save_bin_timeseries_writes_reimportable_self_describing_file(tmp_path) -> None:
    path = tmp_path / "output.bin"
    time_s = -2.0e-9 + np.arange(8, dtype=np.float64) * 0.5e-9
    values = np.linspace(-1.25, 1.25, 8, dtype=np.float64)

    returned_path = save_bin_timeseries(path, time_s, values)

    assert returned_path == path
    assert path.read_bytes()[:2] == b"AG"
    reloaded = load_bin_timeseries(path)
    assert reloaded.sample_rate_hz == pytest.approx(2.0e9)
    np.testing.assert_allclose(reloaded.time_s, time_s, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(reloaded.values[:, 0], values, rtol=1.0e-6)


def test_save_bin_timeseries_rejects_invalid_time_and_values(tmp_path) -> None:
    path = tmp_path / "invalid.bin"
    time_s = np.arange(8, dtype=np.float64)

    with pytest.raises(ValueError, match="非空"):
        save_bin_timeseries(path, np.array([]), np.array([]))
    with pytest.raises(ValueError, match="NaN|Inf"):
        save_bin_timeseries(path, time_s, np.full(8, np.inf))
    with pytest.raises(ValueError, match="复数"):
        save_bin_timeseries(path, time_s, np.full(8, 1.0 + 2.0j))
    with pytest.raises(ValueError, match="float32"):
        save_bin_timeseries(path, time_s, np.full(8, np.finfo(np.float64).max))
    with pytest.raises(ValueError, match="等间隔|时间"):
        invalid_time_s = time_s.copy()
        invalid_time_s[-1] += 0.25
        save_bin_timeseries(path, invalid_time_s, np.arange(8))
