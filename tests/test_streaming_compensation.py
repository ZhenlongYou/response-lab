"""大记录有限边界分块补偿的独立数值合同。"""

from __future__ import annotations

import numpy as np
import pytest

import response_lab.dsp as dsp_module
from response_lab.dsp import run_compensation
from response_lab.models import CompensationSettings, TimeSeries
from response_lab.reporting import build_manifest, sha256_array

FS_HZ = 1.0e9


def _series(values: np.ndarray, *, time_origin_s: float = 0.0) -> TimeSeries:
    return TimeSeries.from_uniform_samples(
        values=values,
        sample_rate_hz=FS_HZ,
        time_origin_s=time_origin_s,
        time_increment_s=1.0 / FS_HZ,
    )


def _impulse(scale: float) -> TimeSeries:
    values = np.zeros(64, dtype=np.float64)
    values[16] = scale
    return _series(values)


def _symmetric_three_tap_reference() -> TimeSeries:
    values = np.zeros(64, dtype=np.float64)
    values[15:18] = (0.25, 1.0, 0.25)
    return _series(values)


def _shifted_impulse(index: int) -> TimeSeries:
    values = np.zeros(64, dtype=np.float64)
    values[index] = 1.0
    return _series(values)


def test_forced_streaming_full_band_constant_gain_matches_closed_form() -> None:
    """全频常数补偿的闭式答案是逐点乘 2，且必须跨越多个处理块。"""

    rng = np.random.default_rng(20260723)
    input_values = rng.normal(size=4097).astype(np.float32)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=1024,
    )

    run = run_compensation(
        _impulse(1.0),
        _impulse(0.5),
        _series(input_values),
        settings,
    )

    assert run.application_method == "finite_reflect_overlap_save_rfft_multiply_irfft"
    assert run.output_values.dtype == np.dtype(np.float32)
    assert run.output_values.shape == (input_values.size, 1)
    tolerance = 32.0 * np.finfo(np.float32).eps
    np.testing.assert_allclose(
        run.output_values[:, 0],
        2.0 * input_values,
        rtol=tolerance,
        atol=tolerance,
    )
    assert any("有限边界" in warning and "float32" in warning for warning in run.warnings)


def test_streaming_uses_real_neighbors_at_seams_and_reflects_only_global_edges() -> None:
    """三抽头闭式卷积同时检查块接缝和整条记录的两端边界。"""

    rng = np.random.default_rng(91)
    input_values = rng.normal(size=4097).astype(np.float32)
    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=1024,
    )

    run = run_compensation(
        _symmetric_three_tap_reference(),
        _impulse(1.0),
        _series(input_values),
        settings,
    )

    reflected = np.pad(input_values.astype(np.float64), (1, 1), mode="reflect")
    expected = (
        0.25 * reflected[:-2]
        + reflected[1:-1]
        + 0.25 * reflected[2:]
    )
    tolerance = 96.0 * np.finfo(np.float32).eps
    np.testing.assert_allclose(
        run.output_values[:, 0],
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    assert run.application_metadata["context_samples_each_side"] == 1
    assert run.input_signal.values.dtype == np.dtype(np.float32)


@pytest.mark.parametrize("delay_samples", [-1, 1])
def test_streaming_phase_sign_matches_closed_form_sample_shift(delay_samples: int) -> None:
    """正负一拍相位斜率的输出方向由独立时域移位定义，不依赖另一套 FFT。"""

    input_values = np.linspace(-1.0, 1.0, 4097, dtype=np.float32)
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=1024,
    )

    run = run_compensation(
        _shifted_impulse(16 + delay_samples),
        _shifted_impulse(16),
        _series(input_values),
        settings,
    )

    if delay_samples > 0:
        expected = np.pad(input_values, (1, 0), mode="reflect")[:-1]
    else:
        expected = np.pad(input_values, (0, 1), mode="reflect")[1:]
    tolerance = 96.0 * np.finfo(np.float32).eps
    np.testing.assert_allclose(
        run.output_values[:, 0],
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_streaming_rejects_filter_tail_that_leaves_no_safe_block_core() -> None:
    """不允许为追求低内存而静默截掉长延迟抽头。"""

    reference_values = np.zeros(512, dtype=np.float64)
    reference_values[32] = 1.0
    reference_values[232] = 0.5
    dut_values = np.zeros(512, dtype=np.float64)
    dut_values[32] = 1.0
    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=512,
    )

    with pytest.raises(ValueError, match="上下文过长.*有效数据不足"):
        run_compensation(
            _series(reference_values),
            _series(dut_values),
            _series(np.ones(4097, dtype=np.float32)),
            settings,
        )


def test_streaming_rejects_block_grid_alias_instead_of_returning_wrong_output() -> None:
    """脉冲抽头相隔整块时，分块网格不能把非单位响应误判成单位响应。"""

    fft_samples = 512
    reference_values = np.zeros(fft_samples + 1, dtype=np.float64)
    reference_values[0] = 1.0
    dut_values = np.zeros_like(reference_values)
    dut_values[0] = 1.25
    dut_values[-1] = -0.25
    rng = np.random.default_rng(20260724)
    input_values = rng.normal(size=4097).astype(np.float32)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=fft_samples,
    )

    # 独立 oracle 直接在整条有限镜像记录的实际 RFFT bins 上应用解析响应。
    # 在 512 点分块网格上，exp(-j*2*pi*k*512/512)=1，错误实现会看到
    # H_dut=1 并原样返回输入；整段网格并不满足这个别名关系。
    padding = input_values.size - 1
    extended = np.pad(input_values.astype(np.float64), (padding, padding), mode="reflect")
    frequency_hz = np.fft.rfftfreq(extended.size, d=1.0 / FS_HZ)
    dut_response = 1.25 - 0.25 * np.exp(
        -2j * np.pi * frequency_hz * fft_samples / FS_HZ
    )
    exact_extended = np.fft.irfft(
        np.fft.rfft(extended) / np.abs(dut_response),
        n=extended.size,
    )
    exact_output = exact_extended[padding : padding + input_values.size]
    exact_rms_delta = float(
        np.sqrt(np.mean((exact_output - input_values.astype(np.float64)) ** 2))
    )
    assert exact_rms_delta > 0.1

    with pytest.raises(ValueError, match="分块.*网格.*混叠"):
        run_compensation(
            _series(reference_values),
            _series(dut_values),
            _series(input_values),
            settings,
        )


def test_streaming_rejects_time_origin_delay_that_aliases_on_both_audit_grids() -> None:
    """真实时间原点相差 2N_FFT 时，N 与 2N 网格不能共同把时移伪装成单位响应。"""

    fft_samples = 512
    pulse_values = np.zeros(64, dtype=np.float64)
    pulse_values[16] = 1.0
    delay_samples = 2 * fft_samples
    rng = np.random.default_rng(20260725)
    input_values = rng.normal(size=4097).astype(np.float32)
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=fft_samples,
    )

    # 物理补偿包含 +delay_samples 的时移。按完整记录两端偶反射得到的闭式索引
    # 与原输入显著不同；若忽略 time_s[0]，N 和 2N 网格都会看到单位响应并假绿。
    shifted_indices = np.arange(input_values.size, dtype=np.int64) + delay_samples
    reflection_period = 2 * (input_values.size - 1)
    folded = np.mod(shifted_indices, reflection_period)
    reflected_indices = np.where(
        folded < input_values.size,
        folded,
        reflection_period - folded,
    )
    expected_shifted = input_values[reflected_indices]
    assert float(
        np.sqrt(
            np.mean(
                (expected_shifted.astype(np.float64) - input_values.astype(np.float64))
                ** 2
            )
        )
    ) > 1.0

    with pytest.raises(ValueError, match="拟合脉冲.*有效支持长度.*混叠"):
        run_compensation(
            _series(pulse_values),
            _series(
                pulse_values,
                time_origin_s=delay_samples / FS_HZ,
            ),
            _series(input_values),
            settings,
        )


def test_streaming_magnitude_only_does_not_treat_time_origin_as_filter_support() -> None:
    """纯幅度补偿与脉冲时间原点无关，安全门禁不能据此误拒绝。"""

    fft_samples = 512
    pulse_values = np.zeros(64, dtype=np.float64)
    pulse_values[16] = 1.0
    input_values = np.linspace(-1.0, 1.0, 1025, dtype=np.float32)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=fft_samples,
    )

    run = run_compensation(
        _series(pulse_values),
        _series(pulse_values, time_origin_s=4 * fft_samples / FS_HZ),
        _series(input_values),
        settings,
    )

    np.testing.assert_allclose(
        run.output_values[:, 0],
        input_values,
        rtol=32.0 * np.finfo(np.float32).eps,
        atol=32.0 * np.finfo(np.float32).eps,
    )


def test_auto_fallback_applies_the_same_block_grid_alias_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto 因预算转入分块路径时，不能绕过 forced-streaming 的安全门禁。"""

    fft_samples = 512
    reference_values = np.zeros(fft_samples + 1, dtype=np.float64)
    reference_values[0] = 1.0
    dut_values = np.zeros_like(reference_values)
    dut_values[0] = 1.25
    dut_values[-1] = -0.25
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="auto",
        streaming_fft_samples=fft_samples,
    )
    input_signal = _series(np.ones(4097, dtype=np.float32))
    estimate_arguments = {
        "target_samples": input_signal.samples,
        "target_channels": input_signal.channels,
        "sample_rate_hz": input_signal.sample_rate_hz,
        "reference_samples": reference_values.size,
        "dut_samples": dut_values.size,
        "settings": settings,
    }
    exact_estimate = dsp_module._compensation_memory_estimate_from_shape(  # noqa: SLF001
        **estimate_arguments
    )
    streaming_estimate = dsp_module._streaming_memory_estimate_from_shape(  # noqa: SLF001
        **estimate_arguments
    )
    assert streaming_estimate.estimated_peak_bytes < exact_estimate.estimated_peak_bytes
    fallback_budget = (
        streaming_estimate.estimated_peak_bytes + exact_estimate.estimated_peak_bytes
    ) // 2
    monkeypatch.setattr(dsp_module, "_system_available_memory_bytes", lambda: None)
    monkeypatch.setattr(
        dsp_module,
        "_safe_compensation_memory_budget_bytes",
        lambda _available: fallback_budget,
    )

    with pytest.raises(ValueError, match="分块.*网格.*混叠"):
        run_compensation(
            _series(reference_values),
            _series(dut_values),
            input_signal,
            settings,
        )


def test_streaming_rejects_short_pulse_inverse_when_block_grid_has_not_converged() -> None:
    """短 DUT 也可能产生长逆响应；支持长度门禁不能替代网格收敛认证。"""

    fft_samples = 1024
    delay_samples = 191
    reference_values = np.zeros(256, dtype=np.float64)
    reference_values[0] = 0.1
    dut_values = np.zeros_like(reference_values)
    dut_values[0] = 1.0
    dut_values[delay_samples] = 0.1
    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=fft_samples,
        streaming_tail_relative_tolerance=1.0e-3,
    )

    # 独立解析响应在 N_FFT 与 2N_FFT 网格之间并未收敛。先由 N_FFT 点样值构造
    # 循环冲激响应，再把正负时延嵌入 2N_FFT 网格；这不调用生产补偿或其 CZT helper。
    block_frequency_hz = np.fft.rfftfreq(fft_samples, d=1.0 / FS_HZ)
    block_correction = 0.1 / (
        1.0
        + 0.1
        * np.exp(
            -2j
            * np.pi
            * block_frequency_hz
            * delay_samples
            / FS_HZ
        )
    )
    circular_impulse = np.fft.irfft(block_correction, n=fft_samples)
    refined_impulse = np.zeros(2 * fft_samples, dtype=np.float64)
    refined_impulse[: fft_samples // 2 + 1] = circular_impulse[
        : fft_samples // 2 + 1
    ]
    refined_impulse[-(fft_samples // 2 - 1) :] = circular_impulse[
        fft_samples // 2 + 1 :
    ]
    block_response_on_refined_grid = np.fft.rfft(refined_impulse)
    refined_frequency_hz = np.fft.rfftfreq(2 * fft_samples, d=1.0 / FS_HZ)
    exact_refined_correction = 0.1 / (
        1.0
        + 0.1
        * np.exp(
            -2j
            * np.pi
            * refined_frequency_hz
            * delay_samples
            / FS_HZ
        )
    )
    relative_grid_mismatch = float(
        np.max(
            np.abs(block_response_on_refined_grid - exact_refined_correction)
        )
        / np.max(np.abs(exact_refined_correction))
    )
    assert relative_grid_mismatch > 1.5e-3

    with pytest.raises(ValueError, match="分块.*网格未收敛"):
        run_compensation(
            _series(reference_values),
            _series(dut_values),
            _series(np.ones(4097, dtype=np.float32)),
            settings,
        )


def test_streaming_explicitly_truncates_tail_instead_of_wrapping_it_at_seam() -> None:
    """已计入误差界的远端抽头不能继续通过块循环卷积形成双倍接缝误差。"""

    reference_values = np.zeros(128, dtype=np.float64)
    reference_values[0] = 1.0
    reference_values[1] = 0.2
    reference_values[100] = 0.005
    dut_values = np.zeros(128, dtype=np.float64)
    dut_values[0] = 1.0
    input_values = np.zeros(1200, dtype=np.float32)
    input_values[410] = -1.0
    input_values[922] = 1.0
    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=512,
        streaming_tail_relative_tolerance=0.006,
    )

    run = run_compensation(
        _series(reference_values),
        _series(dut_values),
        _series(input_values),
        settings,
    )

    indices = np.arange(input_values.size, dtype=np.int64)
    lag_one = np.where(indices >= 1, indices - 1, 1 - indices)
    lag_hundred_raw = indices - 100
    period = 2 * (input_values.size - 1)
    folded = np.mod(lag_hundred_raw, period)
    lag_hundred = np.where(
        folded < input_values.size,
        folded,
        period - folded,
    )
    full_expected = (
        input_values.astype(np.float64)
        + 0.2 * input_values[lag_one]
        + 0.005 * input_values[lag_hundred]
    )
    maximum_error = float(
        np.max(np.abs(run.output_values[:, 0] - full_expected))
    )

    assert run.application_metadata["context_samples_each_side"] == 1
    assert maximum_error <= 0.005 + 128.0 * np.finfo(np.float32).eps


def test_nonbinary_three_tap_reports_quantization_plus_truncation_bound() -> None:
    """metadata 必须约束 float32 冲激响应量化，而不只审计 complex128 理想值。"""

    reference_values = np.zeros(64, dtype=np.float64)
    reference_values[:3] = (0.3, 0.7, 0.2)
    dut_values = np.zeros(64, dtype=np.float64)
    dut_values[0] = 1.0
    input_values = np.random.default_rng(18).normal(size=4097).astype(np.float32)
    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        detrend_phase=False,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=4096,
    )

    run = run_compensation(
        _series(reference_values),
        _series(dut_values),
        _series(input_values),
        settings,
    )

    reflected = np.pad(input_values.astype(np.float64), (2, 0), mode="reflect")
    expected = (
        0.3 * reflected[2:]
        + 0.7 * reflected[1:-1]
        + 0.2 * reflected[:-2]
    )
    tolerance = 192.0 * np.finfo(np.float32).eps
    np.testing.assert_allclose(
        run.output_values[:, 0],
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    assert run.application_metadata["context_samples_each_side"] == 2
    ideal_impulse = np.asarray((0.3, 0.7, 0.2), dtype=np.float64)
    quantized_impulse = ideal_impulse.astype(np.float32).astype(np.float64)
    expected_quantization_relative_l1 = float(
        np.sum(np.abs(ideal_impulse - quantized_impulse), dtype=np.longdouble)
        / np.sum(np.abs(ideal_impulse), dtype=np.longdouble)
    )
    assert run.application_metadata[
        "float32_impulse_quantization_relative_l1"
    ] == pytest.approx(expected_quantization_relative_l1, abs=5.0e-13)
    assert run.application_metadata[
        "float32_impulse_quantization_relative_l1"
    ] > 0.0
    assert run.application_metadata[
        "impulse_approximation_relative_l1_bound"
    ] == pytest.approx(
        run.application_metadata["discarded_tail_relative_l1"]
        + run.application_metadata["float32_impulse_quantization_relative_l1"],
        abs=1.0e-18,
    )
    assert (
        run.application_metadata["impulse_approximation_relative_l1_bound"]
        <= settings.streaming_tail_relative_tolerance
    )
    grid_error = run.application_metadata["block_grid_refinement_relative_linf"]
    assert grid_error >= 0.0
    assert run.application_metadata["impulse_approximation_bound_scope"] == (
        "float32_quantization_and_tail_on_validated_block_grid_only"
    )
    assert grid_error <= settings.streaming_tail_relative_tolerance
    assert run.application_metadata[
        "block_grid_refinement_relative_linf_tolerance"
    ] == pytest.approx(settings.streaming_tail_relative_tolerance)
    assert run.application_metadata["refined_grid_error_bound_scope"] == (
        "NFFT_to_2NFFT_sampled_frequency_grid_not_continuous_frequency"
    )


def test_streaming_manifest_preserves_float32_evidence_and_application_contract(
    tmp_path,
) -> None:
    input_values = np.linspace(-1.0, 1.0, 4097, dtype=np.float32)
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=1024,
    )
    run = run_compensation(
        _impulse(1.0),
        _impulse(0.5),
        _series(input_values),
        settings,
    )

    manifest = build_manifest(run, tmp_path / "not-written.bin")

    assert manifest["schema"] == "response-lab-manifest/v4"
    assert manifest["application"]["method"] == run.application_method
    assert manifest["application"]["fft_samples"] == 1024
    assert "extended_samples" not in manifest["application"]
    assert manifest["output"]["values_sha256"] == sha256_array(run.output_values)
    assert manifest["application"]["output_dtype"] == "float32"


def test_auto_strategy_falls_back_only_when_exact_path_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一预算下整段路径约 73 MiB、分块约 33 MiB，应自动选择后者。"""

    monkeypatch.setattr(
        dsp_module,
        "_system_available_memory_bytes",
        lambda: 582 * 1024**2,
    )
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * FS_HZ,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="auto",
        streaming_fft_samples=1024,
    )

    run = run_compensation(
        _impulse(1.0),
        _impulse(0.5),
        _series(np.ones(100_000, dtype=np.float32)),
        settings,
    )

    assert run.application_metadata["strategy"] == "streaming"
    assert run.output_values.dtype == np.dtype(np.float32)


def test_full_nyquist_accepts_one_ulp_bin_increment_round_trip() -> None:
    """BIN 的 XIncrement 往返误差不能把数学上相同的全频端点判为越界。"""

    time_increment_s = np.nextafter(0.5e-9, np.inf)
    recovered_rate_hz = 1.0 / time_increment_s
    target = TimeSeries.from_uniform_samples(
        values=np.linspace(-1.0, 1.0, 4097, dtype=np.float32),
        sample_rate_hz=recovered_rate_hz,
        time_origin_s=0.0,
        time_increment_s=time_increment_s,
    )
    assert target.nyquist_hz < 1.0e9
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=1.0e9,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy="streaming",
        streaming_fft_samples=1024,
    )
    reference_values = np.zeros(64, dtype=np.float64)
    reference_values[16] = 1.0
    dut_values = 0.5 * reference_values

    def pulse(values: np.ndarray) -> TimeSeries:
        return TimeSeries.from_uniform_samples(
            values=values,
            sample_rate_hz=2.0e9,
            time_origin_s=0.0,
            time_increment_s=0.5e-9,
        )

    run = run_compensation(pulse(reference_values), pulse(dut_values), target, settings)

    tolerance = 32.0 * np.finfo(np.float32).eps
    np.testing.assert_allclose(
        run.output_values[:, 0],
        2.0 * target.values[:, 0],
        rtol=tolerance,
        atol=tolerance,
    )
