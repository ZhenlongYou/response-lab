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


def test_band_endpoint_selection_matches_rfftfreq_rounding_contract() -> None:
    """算术切片必须逐位复现公开 RFFT 频率轴的端点归属。"""

    samples = 10
    extended_samples = 3 * samples - 2
    band_low_hz = 2.0 * FS_HZ / extended_samples
    band_high_hz = 5.0 * FS_HZ / extended_samples
    target = np.arange(samples, dtype=np.float64)
    settings = replace(
        _settings(),
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        maximum_gain_db=None,
        edge_transition_fraction=0.0,
    )

    run = run_compensation(_pulse(), _pulse(0.5), _series(target), settings)

    padding = samples - 1
    extended = np.pad(target, (padding, padding), mode="reflect")
    frequency_hz = np.fft.rfftfreq(extended_samples, d=1.0 / FS_HZ)
    spectrum = np.fft.rfft(extended)
    band = (frequency_hz >= band_low_hz) & (frequency_hz <= band_high_hz)
    spectrum[band] *= 2.0
    expected = np.fft.irfft(spectrum, n=extended_samples)[padding : padding + samples]

    np.testing.assert_allclose(run.output_values[:, 0], expected, atol=2.0e-12)


@pytest.mark.parametrize(("dut_scale", "expected_gain"), [(0.5, 2.0), (0.1, 10.0)])
def test_default_gain_limit_preserves_corrections_up_to_twenty_db(
    dut_scale: float,
    expected_gain: float,
) -> None:
    samples = 10_000
    time_s = np.arange(samples, dtype=np.float64) / FS_HZ
    input_values = np.sin(2.0 * np.pi * 100.0e6 * time_s)

    run = run_compensation(_pulse(), _pulse(dut_scale), _series(input_values), _settings())
    input_rms = np.sqrt(np.mean(input_values**2))
    output_rms = np.sqrt(np.mean(run.output_values[:, 0] ** 2))

    assert output_rms / input_rms == pytest.approx(expected_gain, rel=1e-4)


def test_default_gain_limit_caps_an_unstable_inverse_at_twenty_db() -> None:
    samples = 10_000
    time_s = np.arange(samples, dtype=np.float64) / FS_HZ
    input_values = np.sin(2.0 * np.pi * 100.0e6 * time_s)

    run = run_compensation(_pulse(), _pulse(1.0e-6), _series(input_values), _settings())
    input_rms = np.sqrt(np.mean(input_values**2))
    output_rms = np.sqrt(np.mean(run.output_values[:, 0] ** 2))

    assert output_rms / input_rms == pytest.approx(10.0, rel=1e-4)
    assert any("120" in warning and "20" in warning for warning in run.warnings)


def test_gain_limit_warning_uses_actual_target_fft_bins_for_a_narrow_band() -> None:
    """显示网格未落入极窄频带时，真实应用频点的限幅仍必须可审计。"""

    target_samples = 10_004
    time_s = np.arange(target_samples, dtype=np.float64) / FS_HZ
    input_values = np.sin(2.0 * np.pi * 100.0e6 * time_s)
    settings = replace(
        _settings(),
        band_low_hz=99.999e6,
        band_high_hz=100.001e6,
        edge_transition_fraction=0.0,
    )

    run = run_compensation(
        _pulse(),
        _pulse(1.0e-6),
        _series(input_values),
        settings,
    )

    display_band = (
        (run.analysis.frequency_hz >= settings.band_low_hz)
        & (run.analysis.frequency_hz <= settings.band_high_hz)
    )
    assert not np.any(display_band)
    assert any("120" in warning and "20" in warning for warning in run.warnings)


def test_unlimited_gain_remains_an_explicit_auditable_option() -> None:
    samples = 10_000
    time_s = np.arange(samples, dtype=np.float64) / FS_HZ
    input_values = np.sin(2.0 * np.pi * 100.0e6 * time_s)
    settings = replace(
        _settings(),
        maximum_gain_db=None,
        edge_transition_fraction=0.0,
    )

    run = run_compensation(_pulse(), _pulse(0.01), _series(input_values), settings)
    input_rms = np.sqrt(np.mean(input_values**2))
    output_rms = np.sqrt(np.mean(run.output_values[:, 0] ** 2))

    assert output_rms / input_rms == pytest.approx(100.0, rel=5e-5)


def test_raised_cosine_band_edges_reduce_impulse_ringing_energy() -> None:
    samples = 4096
    impulse = np.zeros(samples, dtype=np.float64)
    center = samples // 2
    impulse[center] = 1.0
    safe_settings = replace(
        _settings(),
        band_low_hz=100.0e6,
        band_high_hz=200.0e6,
    )
    hard_settings = replace(safe_settings, edge_transition_fraction=0.0)

    safe = run_compensation(
        _pulse(), _pulse(0.5), _series(impulse), safe_settings
    ).output_values[:, 0]
    hard = run_compensation(
        _pulse(), _pulse(0.5), _series(impulse), hard_settings
    ).output_values[:, 0]
    safe_off_center_energy = float(np.sum(safe**2) - safe[center] ** 2)
    hard_off_center_energy = float(np.sum(hard**2) - hard[center] ** 2)

    assert safe_off_center_energy < 0.9 * hard_off_center_energy


def test_pure_pulse_delay_is_reported_but_does_not_shift_target_signal() -> None:
    reference = _pulse()
    delay_samples = 9
    dut_values = np.zeros(reference.samples, dtype=np.float64)
    dut_values[delay_samples:] = reference.values[:-delay_samples, 0]
    rng = np.random.default_rng(7)
    target = rng.normal(size=4096)
    settings = replace(_settings(), mode="both", detrend_phase=True)

    run = run_compensation(reference, _series(dut_values), _series(target), settings)

    assert run.analysis.phase_detrend_slope_rad_per_hz / (2.0 * np.pi) == pytest.approx(
        delay_samples / FS_HZ,
        abs=0.05 / FS_HZ,
    )
    assert run.analysis.estimated_relative_delay_s == pytest.approx(
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
        detrend_phase=True,
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
        detrend_phase=True,
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
    settings = replace(_settings(), mode="both", detrend_phase=False)

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


def test_off_grid_dut_zero_on_application_bin_is_rejected() -> None:
    """实际应用频点上的 DUT 零点不能被较粗的显示网格漏掉。"""

    reference = np.zeros(1024, dtype=np.float64)
    reference[100] = 1.0
    dut = reference.copy()
    dut[105] = 1.0
    target_samples = 10_004
    target = np.zeros(target_samples, dtype=np.float64)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=95.0e6,
        band_high_hz=105.0e6,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0,
        maximum_gain_db=None,
        edge_transition_fraction=0.0,
        analysis_points=4097,
    )

    # H_dut(f)=exp(-j*2*pi*f*100/fs) * (1+exp(-j*2*pi*f*5/fs))，
    # 因此 100 MHz 是解析零点。镜像延拓长度 3*N-2=30010，恰好包含该频点；
    # 但 4097 点显示网格不包含它，旧实现会插值越过零点并错误地继续运行。
    with pytest.raises(ValueError, match="待补偿脉冲响应为零"):
        run_compensation(
            _series(reference),
            _series(dut),
            _series(target),
            settings,
        )


def test_long_delay_dut_zero_respects_horner_error_bound() -> None:
    """长延迟相消的解析零点也必须按求值误差界判为不可逆。"""

    reference = np.zeros(1200, dtype=np.float64)
    reference[100] = 1.0
    dut = reference.copy()
    dut[1100] = 1.0
    target_samples = 1334
    target = np.zeros(target_samples, dtype=np.float64)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.49e6,
        band_high_hz=0.51e6,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0,
        analysis_points=4097,
    )

    # 双抽头间隔 1000 点，所以 0.5 MHz 是解析零点；目标镜像延拓后为
    # 4000 点，0.5 MHz 恰好是实际 DFT bin。固定 64*eps 门限不足以覆盖
    # 长 Horner 链的累计舍入误差，旧实现会产生约 1e13 倍伪增益。
    with pytest.raises(ValueError, match="待补偿脉冲响应为零"):
        run_compensation(
            _series(reference),
            _series(dut),
            _series(target),
            settings,
        )


def test_off_grid_finite_notch_matches_closed_form_application_response() -> None:
    """有限深陷波必须按实际 DFT 频点的解析响应补偿，不能按显示网格抹平。"""

    reference = np.zeros(1024, dtype=np.float64)
    reference[100] = 1.0
    dut = reference.copy()
    dut[105] = 1.0 - 1.0e-4
    target_samples = 10_004
    rng = np.random.default_rng(20260718)
    target = rng.normal(size=target_samples)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=95.0e6,
        band_high_hz=105.0e6,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0,
        maximum_gain_db=None,
        edge_transition_fraction=0.0,
        analysis_points=4097,
    )

    run = run_compensation(
        _series(reference),
        _series(dut),
        _series(target),
        settings,
    )

    padding = target_samples - 1
    extended = np.pad(target, (padding, padding), mode="reflect")
    frequency_hz = np.fft.rfftfreq(extended.size, d=1.0 / FS_HZ)
    band = (
        (frequency_hz >= settings.band_low_hz)
        & (frequency_hz <= settings.band_high_hz)
    )
    # 两个脉冲的公共时移在幅度比中抵消，独立闭式解只剩五采样间隔双抽头。
    dut_magnitude = np.abs(
        1.0
        + (1.0 - 1.0e-4)
        * np.exp(-2j * np.pi * frequency_hz * 5.0 / FS_HZ)
    )
    correction = np.ones(frequency_hz.size, dtype=np.float64)
    correction[band] = 1.0 / dut_magnitude[band]
    expected_extended = np.fft.irfft(
        np.fft.rfft(extended) * correction,
        n=extended.size,
    )
    expected = expected_extended[padding : padding + target_samples]

    np.testing.assert_allclose(run.output_values[:, 0], expected, rtol=1.0e-9, atol=1.0e-8)


def test_target_record_without_an_in_band_dft_bin_is_rejected() -> None:
    """过短记录不能静默返回一份看似已补偿、实际完全未处理的输出。"""

    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=10.0e6,
        band_high_hz=20.0e6,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0,
        analysis_points=4097,
    )

    # 8 点目标经镜像延拓后只有 22 点，DFT 间隔约 45.45 MHz；10~20 MHz
    # 频带内没有任何频点。旧实现会保持单位响应并把未处理结果当成成功。
    with pytest.raises(ValueError, match="DFT 频率分辨率不足"):
        run_compensation(
            _pulse(),
            _pulse(0.5),
            _series(np.ones(8, dtype=np.float64)),
            settings,
        )


def test_unrepresentable_complex_target_nyquist_correction_is_rejected() -> None:
    """实信号 Nyquist bin 只能乘实数，不能静默丢弃所需的复相位。"""

    pulse_rate_hz = 2.0e9
    reference = np.zeros(1024, dtype=np.float64)
    reference[100] = 1.0
    dut = np.zeros_like(reference)
    dut[101] = 1.0
    pulse_time_s = np.arange(reference.size, dtype=np.float64) / pulse_rate_hz
    reference_pulse = TimeSeries(
        pulse_time_s,
        reference[:, None],
        pulse_rate_hz,
    )
    dut_pulse = TimeSeries(pulse_time_s, dut[:, None], pulse_rate_hz)
    target = (-1.0) ** np.arange(64, dtype=np.float64)
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=400.0e6,
        band_high_hz=500.0e6,
        phase_fit_low_hz=100.0e6,
        phase_fit_high_hz=300.0e6,
        detrend_phase=False,
        analysis_points=4097,
    )

    # DUT 晚一个 2 GHz 脉冲采样点（0.5 ns），因此 1 GHz 目标的
    # Nyquist=500 MHz 需要 exp(+j*pi/2)=+j；实值 RFFT 的 Nyquist bin
    # 无法承载该相位，旧实现却静默投影为 +1。
    with pytest.raises(ValueError, match="Nyquist.*非实补偿"):
        run_compensation(
            reference_pulse,
            dut_pulse,
            _series(target),
            settings,
        )


def test_frequency_application_is_deterministic() -> None:
    rng = np.random.default_rng(13)
    values = rng.normal(size=(2048, 2))
    first = run_compensation(_pulse(), _pulse(0.8), _series(values), _settings())
    second = apply_frequency_correction(
        values,
        FS_HZ,
        first.analysis,
        reference_pulse=first.reference_pulse,
        dut_pulse=first.dut_pulse,
    )

    np.testing.assert_array_equal(first.output_values, second)


@pytest.mark.parametrize("channels", [1, 2])
def test_frequency_application_returns_a_compact_owned_array(channels: int) -> None:
    """补偿结果不得用切片继续占住三倍长的 IFFT 底层缓冲。"""

    rng = np.random.default_rng(20260723)
    values = rng.normal(size=(4096, channels))
    if channels == 1:
        values = values[:, 0]
    comparison = run_compensation(_pulse(), _pulse(0.8), _series(values), _settings())

    output = apply_frequency_correction(
        values,
        FS_HZ,
        comparison.analysis,
        reference_pulse=comparison.reference_pulse,
        dut_pulse=comparison.dut_pulse,
    )

    assert output.flags.owndata
    assert output.base is None
    assert output.nbytes == np.asarray(values).size * np.dtype(np.float64).itemsize
