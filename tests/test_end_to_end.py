"""从拟合脉冲差异到直接频域补偿的闭环测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from response_lab.dsp import apply_frequency_correction, run_compensation
from response_lab.models import CompensationSettings, TimeSeries

FS_HZ = 1.0e9


def _series(values: np.ndarray) -> TimeSeries:
    time_s = np.arange(values.shape[0], dtype=np.float64) / FS_HZ
    if values.ndim == 1:
        values = values[:, None]
    return TimeSeries(time_s, values, FS_HZ)


def _pulse(scale: float = 1.0) -> TimeSeries:
    index = np.arange(1024, dtype=np.float64)
    values = scale * np.exp(-0.5 * ((index - 240.0) / 2.0) ** 2)
    return _series(values)


def _settings() -> CompensationSettings:
    return CompensationSettings(
        mode="magnitude",
        band_low_hz=10.0e6,
        band_high_hz=300.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        analysis_points=4097,
    )


def test_identity_run_preserves_all_channels_and_length() -> None:
    rng = np.random.default_rng(42)
    input_values = rng.normal(size=(4096, 2))

    run = run_compensation(_pulse(), _pulse(), _series(input_values), _settings())

    assert run.output_values.shape == input_values.shape
    np.testing.assert_allclose(run.output_values, input_values, atol=2e-12)
    assert not hasattr(run, "fir")


@pytest.mark.parametrize(("dut_scale", "expected_gain"), [(0.5, 2.0), (0.01, 100.0)])
def test_analyzed_magnitude_difference_is_applied_without_gain_clipping(
    dut_scale: float,
    expected_gain: float,
) -> None:
    samples = 10_000
    time_s = np.arange(samples, dtype=np.float64) / FS_HZ
    input_values = np.sin(2.0 * np.pi * 100.0e6 * time_s)

    run = run_compensation(_pulse(), _pulse(dut_scale), _series(input_values), _settings())
    input_rms = np.sqrt(np.mean(input_values**2))
    output_rms = np.sqrt(np.mean(run.output_values[:, 0] ** 2))

    assert output_rms / input_rms == pytest.approx(expected_gain, rel=5e-5)


def test_pure_pulse_delay_is_reported_but_does_not_shift_target_signal() -> None:
    reference = _pulse()
    delay_samples = 9
    dut_values = np.zeros(reference.samples, dtype=np.float64)
    dut_values[delay_samples:] = reference.values[:-delay_samples, 0]
    rng = np.random.default_rng(7)
    target = rng.normal(size=4096)
    settings = replace(_settings(), mode="both", remove_relative_delay=True)

    run = run_compensation(reference, _series(dut_values), _series(target), settings)

    assert run.analysis.estimated_dut_delay_s == pytest.approx(
        delay_samples / FS_HZ,
        abs=0.05 / FS_HZ,
    )
    np.testing.assert_allclose(run.output_values[:, 0], target, atol=2e-10)


def test_constant_pi_phase_difference_preserves_negative_real_dft_bins() -> None:
    target = np.ones(4096, dtype=np.float64)
    settings = replace(
        _settings(),
        mode="both",
        band_low_hz=0.0,
        remove_relative_delay=True,
    )

    run = run_compensation(_pulse(), _pulse(-1.0), _series(target), settings)

    np.testing.assert_allclose(run.output_values[:, 0], -target, atol=2e-10)


def test_nyquist_negative_real_correction_is_preserved() -> None:
    reference = np.zeros(1024, dtype=np.float64)
    reference[100] = 1.0
    target = (-1.0) ** np.arange(4096, dtype=np.float64)
    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=400.0e6,
        remove_relative_delay=True,
        analysis_points=4097,
    )

    run = run_compensation(
        _series(reference),
        _series(-reference),
        _series(target),
        settings,
    )

    np.testing.assert_allclose(run.output_values[:, 0], -target, atol=2e-10)


def test_direct_frequency_application_avoids_large_circular_wrap() -> None:
    values = np.zeros(2048, dtype=np.float64)
    values[-10] = 1.0

    run = run_compensation(_pulse(), _pulse(0.01), _series(values), _settings())

    assert np.max(np.abs(run.output_values[:64, 0])) < 0.1
    assert int(np.argmax(np.abs(run.output_values[:, 0]))) > 1900


def test_nontrivial_phase_response_is_applied_in_frequency_domain() -> None:
    reference = np.zeros(1024, dtype=np.float64)
    reference[100] = 1.0
    dut = reference.copy()
    dut[105] = 0.2
    samples = 10_000
    index = np.arange(samples, dtype=np.float64)
    expected = np.sin(2.0 * np.pi * 0.1 * index)
    target = expected + 0.2 * np.sin(2.0 * np.pi * 0.1 * (index - 5.0))
    settings = replace(_settings(), mode="both", remove_relative_delay=False)

    run = run_compensation(_series(reference), _series(dut), _series(target), settings)

    error = run.output_values[1000:-1000, 0] - expected[1000:-1000]
    assert np.sqrt(np.mean(error**2)) < 1.0e-4


def test_reference_spectral_zero_suppresses_target_tone() -> None:
    reference = np.zeros(1024, dtype=np.float64)
    reference[100] = 1.0
    reference[104] = -1.0
    dut = np.zeros_like(reference)
    dut[100] = 1.0
    samples = 10_000
    index = np.arange(samples, dtype=np.float64)
    target = np.sin(2.0 * np.pi * 0.25 * index)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=240.0e6,
        band_high_hz=260.0e6,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0,
        analysis_points=4097,
    )

    run = run_compensation(
        _series(reference),
        _series(dut),
        _series(target),
        settings,
    )

    interior = run.output_values[1000:-1000, 0]
    assert np.sqrt(np.mean(interior**2)) < 2.0e-3


def test_frequency_application_is_deterministic() -> None:
    rng = np.random.default_rng(13)
    values = rng.normal(size=(2048, 2))
    first = run_compensation(_pulse(), _pulse(0.8), _series(values), _settings())
    second = apply_frequency_correction(values, FS_HZ, first.analysis)

    np.testing.assert_array_equal(first.output_values, second)
