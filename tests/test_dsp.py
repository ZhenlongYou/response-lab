"""频响差异、相位去斜和补偿模式的行为测试。"""

from __future__ import annotations

import numpy as np
import pytest

from response_lab.dsp import analyze_responses, fit_linear_phase_slope
from response_lab.models import CompensationSettings, TimeSeries

SAMPLE_RATE_HZ = 1.0e9


def _pulse(values: np.ndarray, *, t0_s: float = 0.0) -> TimeSeries:
    time_s = t0_s + np.arange(values.size, dtype=np.float64) / SAMPLE_RATE_HZ
    return TimeSeries(time_s, values[:, None], SAMPLE_RATE_HZ)


def _pulse_at_rate(values: np.ndarray, sample_rate_hz: float) -> TimeSeries:
    time_s = np.arange(values.size, dtype=np.float64) / sample_rate_hz
    return TimeSeries(time_s, values[:, None], sample_rate_hz)


def _gaussian(samples: int = 1024, center: float = 260.0, width: float = 2.0) -> np.ndarray:
    index = np.arange(samples, dtype=np.float64)
    return np.exp(-0.5 * ((index - center) / width) ** 2)


def _settings(mode: str = "both", *, remove_delay: bool = True) -> CompensationSettings:
    return CompensationSettings(
        mode=mode,
        band_low_hz=5.0e6,
        band_high_hz=350.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        remove_relative_delay=remove_delay,
        analysis_points=4097,
    )


def test_linear_phase_detrend_ignores_independent_integer_cycle_island_offsets() -> None:
    frequency_hz = np.linspace(0.0, 40.0e9, 801)
    expected_delay_s = 0.83e-12
    expected_slope = 2.0 * np.pi * expected_delay_s
    phase_rad = expected_slope * frequency_hz + 0.37
    fit_mask = np.zeros(frequency_hz.size, dtype=bool)
    fit_mask[40:220] = True
    fit_mask[310:510] = True
    fit_mask[590:760] = True
    phase_rad[310:510] += 8.0 * np.pi
    phase_rad[590:760] -= 6.0 * np.pi
    weights = np.linspace(0.2, 1.0, frequency_hz.size) ** 2

    slope = fit_linear_phase_slope(frequency_hz, phase_rad, weights, fit_mask)

    assert slope == pytest.approx(expected_slope, rel=1.0e-12)
    assert slope / (2.0 * np.pi) == pytest.approx(expected_delay_s, rel=1.0e-12)


def test_linear_phase_detrend_preserves_relative_low_confidence_weights() -> None:
    frequency_hz = np.arange(400, dtype=np.float64) * 1.0e6
    first_slope = 2.0e-9
    second_slope = -9.0e-9
    phase_rad = np.zeros(frequency_hz.size, dtype=np.float64)
    fit_mask = np.zeros(frequency_hz.size, dtype=bool)
    first = slice(20, 120)
    second = slice(250, 350)
    fit_mask[first] = True
    fit_mask[second] = True
    phase_rad[first] = first_slope * frequency_hz[first] + 0.3
    phase_rad[second] = second_slope * frequency_hz[second] + 8.0 * np.pi
    weights = np.zeros(frequency_hz.size, dtype=np.float64)
    first_weight = 2.1e-28
    second_weight = 1.0e-20
    weights[first] = first_weight
    weights[second] = second_weight
    expected_slope = (
        first_weight * first_slope + second_weight * second_slope
    ) / (first_weight + second_weight)

    slope = fit_linear_phase_slope(frequency_hz, phase_rad, weights, fit_mask)

    assert slope == pytest.approx(expected_slope, rel=1.0e-12)


def test_linear_phase_detrend_rejects_all_zero_fit_weights() -> None:
    frequency_hz = np.linspace(0.0, 10.0e9, 101)
    phase_rad = 3.0e-12 * frequency_hz
    fit_mask = np.ones(frequency_hz.size, dtype=bool)

    with pytest.raises(ValueError, match="足够的连续可信频点"):
        fit_linear_phase_slope(
            frequency_hz,
            phase_rad,
            np.zeros(frequency_hz.size, dtype=np.float64),
            fit_mask,
        )


def test_identical_pulses_produce_identity_correction() -> None:
    pulse = _pulse(_gaussian())

    analysis = analyze_responses(pulse, pulse, _settings())

    reliable = analysis.reliable_mask
    assert reliable.any()
    assert np.max(np.abs(analysis.magnitude_difference_db[reliable])) < 1e-9
    assert np.max(np.abs(analysis.phase_difference_rad[reliable])) < 1e-9
    np.testing.assert_allclose(analysis.correction_ideal, 1.0 + 0.0j, atol=1e-10)


def test_zero_energy_pulse_is_rejected_instead_of_silently_returning_identity() -> None:
    zeros = np.zeros(1024, dtype=np.float64)

    with pytest.raises(ValueError, match="全零|频谱能量"):
        analyze_responses(_pulse(zeros), _pulse(zeros), _settings("magnitude"))


def test_half_amplitude_dut_requests_plus_six_db_in_core_band() -> None:
    reference_values = _gaussian()
    analysis = analyze_responses(
        _pulse(reference_values),
        _pulse(0.5 * reference_values),
        _settings("magnitude"),
    )
    core = (
        (analysis.frequency_hz >= 60.0e6)
        & (analysis.frequency_hz <= 250.0e6)
        & analysis.reliable_mask
    )

    assert np.count_nonzero(core) > 20
    assert np.median(20.0 * np.log10(np.abs(analysis.correction_ideal[core]))) == pytest.approx(
        6.0206, abs=0.08
    )
    assert np.max(np.abs(np.angle(analysis.correction_ideal[core]))) < 1e-9


def test_magnitude_difference_is_not_clipped_in_core_band() -> None:
    reference_values = _gaussian()
    analysis = analyze_responses(
        _pulse(reference_values),
        _pulse(0.01 * reference_values),
        _settings("magnitude"),
    )
    selected_band = (
        (analysis.frequency_hz >= analysis.settings.band_low_hz)
        & (analysis.frequency_hz <= analysis.settings.band_high_hz)
        & analysis.reliable_mask
    )

    correction_db = 20.0 * np.log10(np.abs(analysis.correction_ideal[selected_band]))
    outside = (analysis.frequency_hz < analysis.settings.band_low_hz) | (
        analysis.frequency_hz > analysis.settings.band_high_hz
    )
    assert np.count_nonzero(selected_band) > 20
    np.testing.assert_allclose(correction_db, 40.0, atol=1e-9)
    np.testing.assert_allclose(analysis.correction_ideal[outside], 1.0 + 0.0j)


def test_magnitude_zero_semantics_are_explicit() -> None:
    reference_notch = np.zeros(1024, dtype=np.float64)
    reference_notch[100] = 1.0
    reference_notch[104] = -1.0
    flat = np.zeros_like(reference_notch)
    flat[100] = 1.0
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=240.0e6,
        band_high_hz=260.0e6,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0,
        analysis_points=4097,
    )

    analysis = analyze_responses(_pulse(reference_notch), _pulse(flat), settings)
    notch_index = int(np.argmin(np.abs(analysis.frequency_hz - 250.0e6)))
    assert analysis.frequency_hz[notch_index] == pytest.approx(250.0e6)
    assert analysis.correction_ideal[notch_index] == 0.0 + 0.0j

    with pytest.raises(ValueError, match="待补偿脉冲响应为零"):
        analyze_responses(_pulse(flat), _pulse(reference_notch), settings)


def test_magnitude_only_ignores_invalid_phase_observation_band() -> None:
    sample_rate_hz = 100.0e6
    values = _gaussian(samples=512, center=120.0, width=2.0)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=1.0e6,
        band_high_hz=30.0e6,
        phase_fit_low_hz=250.0e6,
        phase_fit_high_hz=20.0e6,
        analysis_points=2049,
    )

    analysis = analyze_responses(
        _pulse_at_rate(values, sample_rate_hz),
        _pulse_at_rate(0.8 * values, sample_rate_hz),
        settings,
    )

    assert analysis.estimated_dut_delay_s == 0.0
    assert np.max(np.abs(np.angle(analysis.correction_ideal))) < 1.0e-12


def test_relative_delay_is_measured_but_removed_from_applied_phase() -> None:
    reference_values = _gaussian()
    delay_samples = 7
    dut_values = np.zeros_like(reference_values)
    dut_values[delay_samples:] = reference_values[:-delay_samples]

    analysis = analyze_responses(
        _pulse(reference_values), _pulse(dut_values), _settings("phase", remove_delay=True)
    )
    fit = (
        (analysis.frequency_hz >= 30.0e6)
        & (analysis.frequency_hz <= 220.0e6)
        & analysis.reliable_mask
    )
    residual_slope = np.polyfit(
        analysis.frequency_hz[fit], analysis.delay_removed_phase_rad[fit], 1
    )[0]

    assert analysis.estimated_dut_delay_s == pytest.approx(
        delay_samples / SAMPLE_RATE_HZ, abs=0.15 / SAMPLE_RATE_HZ
    )
    assert abs(residual_slope) < 2.0e-11
    assert np.max(np.abs(np.angle(analysis.correction_ideal[fit]))) < 0.03


def test_csv_time_origin_offset_is_measured_but_does_not_change_correction() -> None:
    values = _gaussian()
    origin_offset_s = 37.0 / SAMPLE_RATE_HZ

    analysis = analyze_responses(
        _pulse(values),
        _pulse(values, t0_s=origin_offset_s),
        _settings("phase", remove_delay=True),
    )

    assert analysis.estimated_dut_delay_s == pytest.approx(
        origin_offset_s,
        abs=0.02 / SAMPLE_RATE_HZ,
    )
    np.testing.assert_allclose(analysis.correction_ideal, 1.0 + 0.0j, atol=1e-10)


def test_phase_only_mode_preserves_unit_magnitude_through_transitions() -> None:
    reference_values = _gaussian()
    dut_values = np.zeros_like(reference_values)
    dut_values[11:] = reference_values[:-11]

    analysis = analyze_responses(
        _pulse(reference_values), _pulse(dut_values), _settings("phase", remove_delay=False)
    )

    np.testing.assert_allclose(np.abs(analysis.correction_ideal), 1.0, atol=1e-12)
    outside = (analysis.frequency_hz < 5.0e6) | (analysis.frequency_hz > 350.0e6)
    np.testing.assert_allclose(analysis.correction_ideal[outside], 1.0 + 0.0j, atol=1e-12)


def test_swapping_roles_inverts_trusted_core_response() -> None:
    reference_values = _gaussian()
    first = analyze_responses(
        _pulse(reference_values), _pulse(0.7 * reference_values), _settings("both")
    )
    second = analyze_responses(
        _pulse(0.7 * reference_values), _pulse(reference_values), _settings("both")
    )
    core = (
        (first.frequency_hz >= 60.0e6)
        & (first.frequency_hz <= 150.0e6)
        & first.reliable_mask
        & second.reliable_mask
    )

    np.testing.assert_allclose(
        first.correction_ideal[core] * second.correction_ideal[core],
        1.0 + 0.0j,
        atol=2e-3,
    )


def test_long_delay_near_record_end_is_unwrapped_without_aliasing() -> None:
    samples = 1024
    index = np.arange(samples, dtype=np.float64)
    reference_values = np.exp(-0.5 * ((index - 120.0) / 2.0) ** 2)
    delay_samples = 700
    dut_values = np.zeros_like(reference_values)
    dut_values[delay_samples:] = reference_values[:-delay_samples]
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=5.0e6,
        band_high_hz=350.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        remove_relative_delay=True,
        analysis_points=257,
    )

    analysis = analyze_responses(_pulse(reference_values), _pulse(dut_values), settings)

    assert analysis.estimated_dut_delay_s == pytest.approx(
        delay_samples / SAMPLE_RATE_HZ,
        abs=0.05 / SAMPLE_RATE_HZ,
    )
    trusted = analysis.reliable_mask & (analysis.frequency_hz <= 250.0e6)
    assert np.nanmax(np.abs(analysis.delay_removed_phase_rad[trusted])) < 1e-9


def _comb_notch_delay_analysis(*, remove_delay: bool, avoid_notches: bool = True):
    samples = 2048
    reference_values = np.zeros(samples, dtype=np.float64)
    reference_values[200] = 1.0
    reference_values[264] = -1.0
    delay_samples = 17
    dut_values = np.zeros_like(reference_values)
    dut_values[delay_samples:] = reference_values[:-delay_samples]
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=5.0e6,
        band_high_hz=14.0e6 if avoid_notches else 350.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=300.0e6,
        remove_relative_delay=remove_delay,
        analysis_points=4097,
    )
    return analyze_responses(_pulse(reference_values), _pulse(dut_values), settings)


def test_comb_notches_use_independent_island_intercepts_for_delay_fit() -> None:
    analysis = _comb_notch_delay_analysis(remove_delay=True)
    fit_mask = (
        analysis.reliable_mask
        & (analysis.frequency_hz >= 20.0e6)
        & (analysis.frequency_hz <= 300.0e6)
    )
    island_starts = fit_mask & ~np.r_[False, fit_mask[:-1]]

    assert np.count_nonzero(island_starts) >= 10
    assert analysis.estimated_dut_delay_s == pytest.approx(
        17.0 / SAMPLE_RATE_HZ,
        abs=0.02 / SAMPLE_RATE_HZ,
    )


def test_reported_phase_trend_is_the_removed_linear_delay_component() -> None:
    analysis = _comb_notch_delay_analysis(remove_delay=True)
    expected_trend = (
        2.0
        * np.pi
        * analysis.estimated_dut_delay_s
        * analysis.frequency_hz
    )

    np.testing.assert_allclose(analysis.phase_trend_rad, expected_trend, atol=1.0e-12)


def test_phase_band_containing_spectral_notches_is_rejected() -> None:
    with pytest.raises(ValueError, match="无法解析相位.*缩小或移动补偿频带"):
        _comb_notch_delay_analysis(remove_delay=False, avoid_notches=False)


def test_continuous_time_fft_scaling_matches_different_sample_rates() -> None:
    first_rate_hz = 1.0e9
    second_rate_hz = 1.25e9
    first_time_s = np.arange(1024, dtype=np.float64) / first_rate_hz
    second_time_s = np.arange(1280, dtype=np.float64) / second_rate_hz
    first_values = np.exp(-0.5 * ((first_time_s - 300.0e-9) / 5.0e-9) ** 2)
    second_values = np.exp(-0.5 * ((second_time_s - 300.0e-9) / 5.0e-9) ** 2)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=5.0e6,
        band_high_hz=250.0e6,
        phase_fit_low_hz=5.0e6,
        phase_fit_high_hz=80.0e6,
        analysis_points=4097,
    )

    analysis = analyze_responses(
        _pulse_at_rate(first_values, first_rate_hz),
        _pulse_at_rate(second_values, second_rate_hz),
        settings,
    )
    comparison_band = (
        analysis.reliable_mask
        & (analysis.frequency_hz >= 5.0e6)
        & (analysis.frequency_hz <= 80.0e6)
    )

    assert np.count_nonzero(comparison_band) > 100
    assert np.max(np.abs(analysis.magnitude_difference_db[comparison_band])) < 1e-3
