"""自动频带输出使用简洁且可直接执行的两位有效数字。"""

from __future__ import annotations

import numpy as np

from response_lab.dsp import suggest_frequency_settings
from response_lab.models import CompensationSettings, TimeSeries


def _high_rate_pulses() -> tuple[TimeSeries, TimeSeries]:
    sample_rate_hz = 3.4e12
    samples = 12_800
    index = np.arange(samples, dtype=np.float64)
    time_s = index / sample_rate_hz
    reference = np.exp(-0.5 * ((index - 200.0) / 25.0) ** 2)
    dut = 0.82 * np.exp(-0.5 * ((index - 202.0) / 27.0) ** 2)
    return (
        TimeSeries(time_s, reference[:, None], sample_rate_hz),
        TimeSeries(time_s, dut[:, None], sample_rate_hz),
    )


def test_auto_frequency_and_phase_bounds_use_two_significant_digits() -> None:
    reference, dut = _high_rate_pulses()
    template = CompensationSettings(
        mode="both",
        band_low_hz=10.0e6,
        band_high_hz=300.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        analysis_points=16_385,
    )

    suggested = suggest_frequency_settings(
        reference,
        dut,
        template,
        suggest_phase_fit_band=True,
    )

    assert suggested.band_low_hz == 0.86e9
    assert suggested.band_high_hz == 41.0e9
    assert suggested.phase_fit_low_hz == 3.4e9
    assert suggested.phase_fit_high_hz == 39.0e9


def test_auto_band_rounding_does_not_modify_manual_phase_fit_bounds() -> None:
    reference, dut = _high_rate_pulses()
    template = CompensationSettings(
        mode="both",
        band_low_hz=10.0e6,
        band_high_hz=300.0e6,
        phase_fit_low_hz=23.456789e6,
        phase_fit_high_hz=245.6789e6,
        analysis_points=16_385,
    )

    suggested = suggest_frequency_settings(reference, dut, template)

    assert suggested.band_low_hz == 0.86e9
    assert suggested.band_high_hz == 41.0e9
    assert suggested.phase_fit_low_hz == template.phase_fit_low_hz
    assert suggested.phase_fit_high_hz == template.phase_fit_high_hz
