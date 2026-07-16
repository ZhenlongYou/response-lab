"""Headerless CSV and raw BIN public I/O behavior tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from response_lab.io import (
    load_bin_timeseries,
    load_csv_timeseries,
    save_bin_float32,
    save_csv_timeseries,
)
from response_lab.models import BinConfig


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


def test_bin_decodes_interleaved_channel_with_manual_configuration(tmp_path) -> None:
    path = tmp_path / "capture.bin"
    raw = np.column_stack(
        (
            np.arange(8, dtype=np.int16),
            np.arange(100, 108, dtype=np.int16),
        )
    )
    path.write_bytes(b"HEAD" + raw.astype(">i2").tobytes())
    config = BinConfig(
        sample_rate_hz=2.5e6,
        dtype="int16",
        byte_order="big",
        offset_bytes=4,
        channels=2,
        channel_index=1,
        layout="interleaved",
        scale=0.25,
        value_offset=-1.0,
    )

    series = load_bin_timeseries(path, config)

    np.testing.assert_allclose(series.values[:, 0], raw[:, 1] * 0.25 - 1.0)
    np.testing.assert_allclose(series.time_s, np.arange(8) / 2.5e6)
    assert series.sample_rate_hz == pytest.approx(2.5e6)
    assert series.source_metadata["sample_rate_entered_manually"] is True
    assert series.source_metadata["dtype"] == "int16"
    assert series.source_metadata["byte_order"] == "big"
    assert series.source_metadata["source_size_bytes"] == path.stat().st_size
    assert series.source_metadata["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize(
    ("dtype_name", "dtype_code", "raw_values"),
    [
        ("float32", "f4", np.linspace(-1.0, 1.0, 8)),
        ("float64", "f8", np.linspace(-2.0, 2.0, 8)),
        ("int16", "i2", np.arange(-4, 4)),
        ("int32", "i4", np.arange(100_000, 100_008)),
    ],
)
def test_bin_supports_all_dtypes_and_byte_orders(
    tmp_path,
    byte_order,
    dtype_name,
    dtype_code,
    raw_values,
) -> None:
    path = tmp_path / f"{dtype_name}-{byte_order}.bin"
    prefix = "<" if byte_order == "little" else ">"
    path.write_bytes(np.asarray(raw_values, dtype=prefix + dtype_code).tobytes())

    series = load_bin_timeseries(
        path,
        BinConfig(sample_rate_hz=8.0e6, dtype=dtype_name, byte_order=byte_order),
    )

    np.testing.assert_allclose(series.values[:, 0], raw_values, rtol=1.0e-6)


def test_bin_decodes_selected_planar_channel(tmp_path) -> None:
    path = tmp_path / "planar.bin"
    raw_by_channel = np.vstack(
        (
            np.arange(8, dtype=np.float64),
            10.0 + np.arange(8, dtype=np.float64),
            20.0 + np.arange(8, dtype=np.float64),
        )
    )
    path.write_bytes(raw_by_channel.astype("<f8").tobytes())
    config = BinConfig(
        sample_rate_hz=1.0e9,
        dtype="float64",
        byte_order="little",
        channels=3,
        channel_index=2,
        layout="planar",
    )

    series = load_bin_timeseries(path, config)

    np.testing.assert_allclose(series.values[:, 0], raw_by_channel[2])


def test_bin_rejects_residual_bytes_and_incomplete_channel_frames(tmp_path) -> None:
    residual_path = tmp_path / "residual.bin"
    residual_path.write_bytes(np.arange(8, dtype="<i2").tobytes() + b"\x00")
    with pytest.raises(ValueError, match="残字节"):
        load_bin_timeseries(
            residual_path,
            BinConfig(sample_rate_hz=1.0e6, dtype="int16"),
        )

    incomplete_path = tmp_path / "incomplete-frame.bin"
    incomplete_path.write_bytes(np.arange(25, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="通道数整除"):
        load_bin_timeseries(
            incomplete_path,
            BinConfig(sample_rate_hz=1.0e6, dtype="int16", channels=3),
        )


def test_bin_rejects_offset_past_end_and_too_few_samples(tmp_path) -> None:
    path = tmp_path / "short.bin"
    path.write_bytes(np.arange(7, dtype="<f4").tobytes())

    with pytest.raises(ValueError, match="偏移超过"):
        load_bin_timeseries(
            path,
            BinConfig(sample_rate_hz=1.0e6, offset_bytes=path.stat().st_size + 1),
        )
    with pytest.raises(ValueError, match="至少.*8"):
        load_bin_timeseries(path, BinConfig(sample_rate_hz=1.0e6))


@pytest.mark.parametrize(
    "invalid_option",
    [
        {"dtype": "uint8"},
        {"byte_order": "native"},
        {"layout": "blocked"},
    ],
)
def test_bin_rejects_unsupported_decode_options(tmp_path, invalid_option) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(np.arange(8, dtype="<f4").tobytes())

    with pytest.raises(ValueError, match="BIN"):
        config = BinConfig(sample_rate_hz=1.0e6, **invalid_option)
        load_bin_timeseries(path, config)


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


def test_save_bin_float32_writes_little_endian_c_order(tmp_path) -> None:
    path = tmp_path / "output.bin"
    values = np.array([[1.25, -2.5], [3.75, -4.0]], dtype=np.float64)

    returned_path = save_bin_float32(path, values)

    assert returned_path == path
    assert path.read_bytes() == values.astype("<f4").tobytes(order="C")


def test_save_bin_float32_rejects_empty_nonfinite_and_overflow_values(tmp_path) -> None:
    path = tmp_path / "invalid.bin"

    with pytest.raises(ValueError, match="非空"):
        save_bin_float32(path, np.array([]))
    with pytest.raises(ValueError, match="NaN|Inf"):
        save_bin_float32(path, np.array([np.inf]))
    with pytest.raises(ValueError, match="复数"):
        save_bin_float32(path, np.array([1.0 + 2.0j]))
    with pytest.raises(ValueError, match="float32"):
        save_bin_float32(path, np.array([np.finfo(np.float64).max]))
