"""验证大文件演示数据的通道差异处于可观察但温和的范围。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

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
    np.testing.assert_allclose(dut[main], 0.92, atol=5e-5)
    np.testing.assert_allclose(dut[main + generator.M], 0.07, atol=5e-5)
    np.testing.assert_allclose(dut[main + 2 * generator.M], -0.025, atol=5e-5)

    relative_difference = np.linalg.norm(dut - reference) / np.linalg.norm(reference)
    assert 0.08 < relative_difference < 0.15

    frequency_hz = np.fft.rfftfreq(
        generator.PULSE_SAMPLES,
        d=1.0 / generator.SAMPLE_RATE_HZ,
    )
    reference_spectrum = np.fft.rfft(reference)
    dut_spectrum = np.fft.rfft(dut)
    expected_relative_channel = (
        generator.DEMO_MAIN_TAP
        + generator.DEMO_POSTCURSOR_1_TAP
        * np.exp(-2j * np.pi * frequency_hz * generator.M / generator.SAMPLE_RATE_HZ)
        + generator.DEMO_POSTCURSOR_2_TAP
        * np.exp(
            -2j
            * np.pi
            * frequency_hz
            * (2 * generator.M)
            / generator.SAMPLE_RATE_HZ
        )
    )
    usable_band = frequency_hz <= 60.0e9
    np.testing.assert_allclose(
        dut_spectrum[usable_band] / reference_spectrum[usable_band],
        expected_relative_channel[usable_band],
        rtol=1e-10,
        atol=1e-10,
    )
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

    assert 0.2 < correction_at_dc_db < 0.4
    assert 1.6 < correction_at_symbol_nyquist_db < 1.75


def test_large_demo_target_uses_the_same_relative_channel() -> None:
    """目标只经过 H_dut/H_ref，符合补偿响应会抵消的相对通道模型。"""

    generator = _load_generator_module()
    samples = 2_048
    actual = generator._build_target(samples)

    rng = np.random.default_rng(generator.RANDOM_SEED)
    levels = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float64)
    symbols = rng.choice(levels, size=(samples + generator.M - 1) // generator.M + 2)
    ideal = np.repeat(symbols, generator.M)[:samples]
    taps = np.zeros(2 * generator.M + 1, dtype=np.float64)
    taps[0] = generator.DEMO_MAIN_TAP
    taps[generator.M] = generator.DEMO_POSTCURSOR_1_TAP
    taps[2 * generator.M] = generator.DEMO_POSTCURSOR_2_TAP
    expected = np.convolve(ideal, taps, mode="full")[:samples]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert "相对通道" in generator.README_TEXT
    assert "H_ref / H_dut = 1 / T" in generator.README_TEXT
