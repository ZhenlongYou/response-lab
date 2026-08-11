"""验证大文件演示数据的通道差异处于可观察但温和的范围。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from response_lab.dsp import run_compensation
from response_lab.models import CompensationSettings, TimeSeries

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "examples" / "generate_demo_data.py"


def _load_generator_module():
    """加载演示生成器，不执行其生成大文件的命令行入口。"""

    module_spec = importlib.util.spec_from_file_location(
        "response_lab_large_demo_generator",
        GENERATOR_PATH,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_large_demo_pulses_are_mild_but_observable_channel_difference() -> None:
    """大文件示例保留可补偿 ISI，而非夸张的严重通道失真。"""

    generator = _load_generator_module()
    reference, dut = generator._build_pulses()
    main = generator.PEAK_INDEX

    np.testing.assert_allclose(reference[main], 1.0, atol=1e-12)
    # 高斯参考脉冲的尾部会有极小重叠，所以允许 50 µV 以内的相邻抽头叠加。
    np.testing.assert_allclose(dut[main], 0.985, atol=5e-5)
    np.testing.assert_allclose(dut[main - generator.M], 0.009, atol=5e-5)
    np.testing.assert_allclose(dut[main + generator.M], 0.009, atol=5e-5)
    np.testing.assert_allclose(dut[main - 2 * generator.M], -0.003, atol=5e-5)
    np.testing.assert_allclose(dut[main + 2 * generator.M], -0.003, atol=5e-5)

    relative_difference = np.linalg.norm(dut - reference) / np.linalg.norm(reference)
    assert 0.015 < relative_difference < 0.04

    frequency_hz = np.fft.rfftfreq(
        generator.PULSE_SAMPLES,
        d=1.0 / generator.SAMPLE_RATE_HZ,
    )
    reference_spectrum = np.fft.rfft(reference)
    dut_spectrum = np.fft.rfft(dut)
    expected_relative_channel = (
        0.985
        + 2.0 * 0.009 * np.cos(2.0 * np.pi * frequency_hz * generator.M / generator.SAMPLE_RATE_HZ)
        - 2.0 * 0.003
        * np.cos(2.0 * np.pi * frequency_hz * (2 * generator.M) / generator.SAMPLE_RATE_HZ)
    )
    # 默认频响图约显示到 165 GHz；整个可视范围内都应保持零相位差。
    display_band = frequency_hz <= 165.0e9
    np.testing.assert_allclose(
        dut_spectrum[display_band] / reference_spectrum[display_band],
        expected_relative_channel[display_band],
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.max(
        np.abs(np.angle(dut_spectrum[display_band] / reference_spectrum[display_band]))
    ) < 1e-10
    symbol_nyquist_hz = generator.SAMPLE_RATE_HZ / (2.0 * generator.M)
    symbol_nyquist_index = int(np.argmin(np.abs(frequency_hz - symbol_nyquist_hz)))

    correction_at_dc_db = -20.0 * np.log10(
        np.abs(dut_spectrum[0] / reference_spectrum[0])
    )
    correction_at_symbol_nyquist_db = -20.0 * np.log10(
        np.abs(
            dut_spectrum[symbol_nyquist_index]
            / reference_spectrum[symbol_nyquist_index]
        )
    )

    assert 0.0 < correction_at_dc_db < 0.05
    assert 0.3 < correction_at_symbol_nyquist_db < 0.4


def test_large_demo_target_uses_the_same_relative_channel() -> None:
    """目标只经过 H_dut/H_ref，符合补偿响应会抵消的相对通道模型。"""

    generator = _load_generator_module()
    samples = 2_048
    actual = generator._build_target(samples)

    rng = np.random.default_rng(generator.RANDOM_SEED)
    levels = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float64)
    symbols = rng.choice(levels, size=(samples + generator.M - 1) // generator.M + 2)
    ideal = np.repeat(symbols, generator.M)[:samples]
    taps = np.zeros(4 * generator.M + 1, dtype=np.float64)
    center = 2 * generator.M
    taps[center] = 0.985
    taps[center - generator.M] = taps[center + generator.M] = 0.009
    taps[center - 2 * generator.M] = taps[center + 2 * generator.M] = -0.003
    expected = np.convolve(ideal, taps, mode="full")[center : center + samples]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert "相对通道" in generator.README_TEXT
    assert "H_ref / H_dut = 1 / T" in generator.README_TEXT
    assert "零相位" in generator.README_TEXT


def test_large_demo_data_compensation_keeps_visible_phase_at_zero() -> None:
    """实际补偿路径在频响图可视范围内不应重新引入相位差。"""

    generator = _load_generator_module()
    reference_values, dut_values = generator._build_pulses()
    pulse_time_s = np.arange(generator.PULSE_SAMPLES) / generator.SAMPLE_RATE_HZ
    reference = TimeSeries(
        pulse_time_s,
        reference_values[:, None],
        generator.SAMPLE_RATE_HZ,
    )
    dut = TimeSeries(
        pulse_time_s,
        dut_values[:, None],
        generator.SAMPLE_RATE_HZ,
    )
    target_values = generator._build_target(4_096)
    target = TimeSeries.from_uniform_samples(
        values=target_values,
        sample_rate_hz=generator.SAMPLE_RATE_HZ,
        time_origin_s=0.0,
        time_increment_s=1.0 / generator.SAMPLE_RATE_HZ,
    )
    settings = CompensationSettings(
        mode="both",
        band_low_hz=1.0e9,
        band_high_hz=60.0e9,
        phase_fit_low_hz=5.0e9,
        phase_fit_high_hz=55.0e9,
        detrend_phase=True,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=4097,
        application_strategy="exact",
    )

    run = run_compensation(reference, dut, target, settings)
    analysis = run.analysis
    visible = analysis.reliable_mask & (analysis.frequency_hz <= 165.0e9)
    applied = (
        (analysis.frequency_hz >= analysis.settings.band_low_hz)
        & (analysis.frequency_hz <= analysis.settings.band_high_hz)
    )

    assert np.max(np.abs(analysis.phase_difference_rad[visible])) < 1e-8
    assert np.max(np.abs(np.angle(analysis.correction_ideal[applied]))) < 1e-8
    assert np.all(np.isfinite(run.output_values))
