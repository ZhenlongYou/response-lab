"""直接频域补偿的共享内存预检回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import response_lab.dsp as dsp_module
from response_lab.dsp import (
    _compensation_memory_estimate_from_shape,
    _parse_macos_vm_stat,
    _safe_compensation_memory_budget_bytes,
    _streaming_memory_estimate_from_shape,
    run_compensation,
)
from response_lab.models import CompensationSettings, TimeSeries


def _settings(*, band_low_hz: float, band_high_hz: float) -> CompensationSettings:
    return CompensationSettings(
        mode="magnitude",
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        analysis_points=4097,
    )


def _series(source_format: str) -> TimeSeries:
    sample_rate_hz = 2.0e9
    time_s = np.arange(32, dtype=np.float64) / sample_rate_hz
    return TimeSeries(
        time_s,
        np.ones(32, dtype=np.float64),
        sample_rate_hz,
        source_path=Path(f"target.{source_format}"),
        source_format=source_format,
    )


def test_memory_estimate_envelopes_two_independent_rss_measurements() -> None:
    """估算包住优化后实测，同时不得继续误报旧实现的巨大峰值。"""

    # 独立 macOS arm64 子进程（Python 3.12 / NumPy 2.5 / SciPy 1.18）：
    # N=1,000,000、50–200 MHz 时新增峰值 157,581,312 B。
    narrow = _compensation_memory_estimate_from_shape(
        target_samples=1_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=_settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
    )
    assert narrow.active_band_bins >= 225_000
    assert 157_581_312 <= narrow.estimated_peak_bytes < 320 * 1024**2

    # N=500,000、0–Nyquist 的独立进程峰值为 160,022,528 B。
    full_band = _compensation_memory_estimate_from_shape(
        target_samples=500_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=_settings(band_low_hz=0.0, band_high_hz=1.0e9),
    )
    assert full_band.active_band_bins >= 750_000
    assert 160_022_528 <= full_band.estimated_peak_bytes < 320 * 1024**2
    assert full_band.estimated_bytes_per_target_sample > narrow.estimated_bytes_per_target_sample


def test_memory_estimate_counts_extra_channels_and_long_pulse_czt() -> None:
    """多通道 FFT 和长脉冲 CZT 都必须提高预算。"""

    settings = _settings(band_low_hz=50.0e6, band_high_hz=200.0e6)
    baseline = _compensation_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    multi_channel = _compensation_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=4,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    long_pulse = _compensation_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=300_000,
        dut_samples=300_000,
        settings=settings,
    )

    assert multi_channel.estimated_peak_bytes > baseline.estimated_peak_bytes
    assert long_pulse.czt_working_samples > baseline.czt_working_samples
    assert long_pulse.estimated_peak_bytes > baseline.estimated_peak_bytes


def test_thirty_million_full_band_switches_from_unsafe_exact_to_bounded_streaming() -> None:
    """30M 分块估算须包住归档峰值并落在 1.5 GiB 门禁内。"""

    settings = _settings(band_low_hz=0.0, band_high_hz=1.0e9)
    exact = _compensation_memory_estimate_from_shape(
        target_samples=30_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    streaming = _streaming_memory_estimate_from_shape(
        target_samples=30_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    multi_channel_streaming = _streaming_memory_estimate_from_shape(
        target_samples=30_000_000,
        target_channels=4,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )

    assert exact.estimated_peak_bytes > int(1.5 * 1024**3)
    assert streaming.estimated_peak_bytes < int(1.5 * 1024**3)
    evidence_folder = Path(__file__).resolve().parents[1] / "docs"
    calibration_paths = (
        evidence_folder / "30M_BIN分块补偿估算反例_2026-07-23.json",
    )
    observations = [
        json.loads(path.read_text(encoding="utf-8"))["measurement"]
        for path in calibration_paths
    ]
    # 带 clean HEAD、命令和验收字段的旧估算曾被同一夹具/算法的 fresh-worker
    # 高水位反例击穿；当前模型必须按 M 线性计入 FFT 后端余量并包住原始记录。
    for observation in observations:
        assert observation["streaming_memory_estimate"]["estimated_peak_bytes"] < (
            observation["post_load_compensation_peak_delta_bytes"]
        )
    assert streaming.estimated_peak_bytes >= max(
        observation["post_load_compensation_peak_delta_bytes"]
        for observation in observations
    )
    assert streaming.fft_samples == settings.streaming_fft_samples
    assert streaming.estimated_peak_bytes >= 30_000_000 * np.dtype(np.float32).itemsize
    assert multi_channel_streaming.estimated_peak_bytes > (
        streaming.estimated_peak_bytes
        + 3
        * settings.streaming_fft_samples
        * dsp_module._STREAMING_BACKEND_RESERVE_BYTES_PER_FFT_SAMPLE
    )


def test_dynamic_budget_keeps_half_available_and_512_mib_headroom() -> None:
    """动态预算同时受绝对上限、50% 比例和系统余量约束。"""

    gib = 1024**3
    assert _safe_compensation_memory_budget_bytes(8 * gib) == int(1.5 * gib)
    assert _safe_compensation_memory_budget_bytes(2 * gib) == 1 * gib
    assert _safe_compensation_memory_budget_bytes(768 * 1024**2) == 256 * 1024**2


def test_macos_available_memory_parser_counts_only_distinct_reclaimable_pages() -> None:
    """inactive 已含 purgeable，解析器不得因重复相加而抬高安全预算。"""

    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             999.
Pages inactive:                           200.
Pages speculative:                         50.
Pages purgeable:                           500.
"""

    assert _parse_macos_vm_stat(output) == (100 + 200 + 50) * 16384


@pytest.mark.parametrize("source_format", ["csv", "bin"])
def test_run_compensation_rejects_csv_and_bin_before_analysis_allocation(
    monkeypatch: pytest.MonkeyPatch,
    source_format: str,
) -> None:
    """共享入口应在 compare/CZT/FFT 之前对两种文件来源执行同一门禁。"""

    pulse = _series("memory")
    target = _series(source_format)
    monkeypatch.setattr(dsp_module, "_system_available_memory_bytes", lambda: 96 * 1024**2)

    def forbidden_compare(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("memory preflight must run before compare_pulses")

    monkeypatch.setattr(dsp_module, "compare_pulses", forbidden_compare)

    with pytest.raises(MemoryError, match="补偿内存预检.*CSV/BIN 共用"):
        run_compensation(
            pulse,
            pulse,
            target,
            _settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
        )


def test_run_adopts_owned_application_output_without_a_second_full_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FFT 已返回自有紧凑数组时，运行模型应校验后直接接管并设为只读。"""

    pulse = _series("memory")
    target = _series("bin")
    owned_output = np.arange(32, dtype=np.float64).reshape(32, 1).copy()

    def return_owned_output(*_args: object, **_kwargs: object) -> np.ndarray:
        return owned_output

    monkeypatch.setattr(dsp_module, "apply_frequency_correction", return_owned_output)

    run = run_compensation(
        pulse,
        pulse,
        target,
        _settings(band_low_hz=1.0e6, band_high_hz=10.0e6),
    )

    assert np.shares_memory(run.output_values, owned_output)
    assert not run.output_values.flags.writeable
