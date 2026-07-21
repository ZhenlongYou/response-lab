"""直接频域补偿的共享内存预检回归测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import response_lab.dsp as dsp_module
from response_lab.dsp import (
    _compensation_memory_estimate_from_shape,
    _parse_macos_vm_stat,
    _safe_compensation_memory_budget_bytes,
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
    """估算必须包住审阅进程实测，而不是沿用已证伪的 192 B/点。"""

    # 独立 macOS 子进程：N=1,000,000、50–200 MHz 时新增峰值 626,688,000 B。
    narrow = _compensation_memory_estimate_from_shape(
        target_samples=1_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=_settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
    )
    assert narrow.active_band_bins >= 225_000
    assert narrow.estimated_peak_bytes >= 626_688_000

    # 独立 macOS 子进程：N=500,000、0–Nyquist 时新增峰值 451,919,872 B。
    full_band = _compensation_memory_estimate_from_shape(
        target_samples=500_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=_settings(band_low_hz=0.0, band_high_hz=1.0e9),
    )
    assert full_band.active_band_bins >= 750_000
    assert full_band.estimated_peak_bytes >= 451_919_872
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
