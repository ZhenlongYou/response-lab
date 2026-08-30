"""LFP 周期稳态 Vpp 与频域 RMS 误差分析的独立数值验收。"""

# 延迟求值类型标注，便于测试辅助函数表达 NumPy 数组而不增加运行时依赖。
from __future__ import annotations

# 哈希用于把完整 8191-symbol 标准向量冻结为独立回归证据。
import hashlib

# Path 为 pytest 临时目录下的外部码型文件提供明确类型。
from pathlib import Path

# NumPy 构造黄金码型并执行精确数组断言。
import numpy as np

# pytest 提供偶/奇周期参数化和 fail-closed 异常断言。
import pytest

# 模块别名供 monkeypatch 精确监视生产 IFFT 路径，而不影响 NumPy oracle。
import response_lab.vpp_analysis as vpp_analysis
from response_lab.cancellation import OperationCancelledError

# MemoryBudget 构造确定的低内存快照，验证文本解析前门禁。
from response_lab.memory_budget import MemoryBudget

# TimeSeries 复用应用层已有的采样率、时间轴和通道不变量。
from response_lab.models import TimeSeries

# 公开入口共同冻结设置合同、码型生成和电平载入行为。
from response_lab.vpp_analysis import (
    VppAnalysisSettings,
    generate_prbs13q_gray_symbols,
    load_pattern_levels,
    measure_candidate,
    prepare_vpp_analysis,
    validate_vpp_pulse_windows,
)


# 测试辅助函数用严格等间隔时间轴构造单通道脉冲，避免绕过领域模型校验。
def _series(values_v: list[float], sample_rate_hz: float = 8.0e9) -> TimeSeries:
    """把至少八点的电压序列包装为单通道 TimeSeries。"""

    # float64 数组保证测试 oracle 与生产 FFT 使用相同数值精度。
    values = np.asarray(values_v, dtype=np.float64)
    # 样点 n 的时间严格等于 n/Fs，满足 TimeSeries 的均匀采样合同。
    time_s = np.arange(values.size, dtype=np.float64) / sample_rate_hz
    # 返回单通道序列；TimeSeries 会把一维 values 规范化为二维列向量。
    return TimeSeries(time_s=time_s, values=values, sample_rate_hz=sample_rate_hz)


def _window_validation_settings() -> VppAnalysisSettings:
    """构造无需读取外部码型即可验证脉冲窗口的最小设置。"""

    return VppAnalysisSettings(
        method="lfp",
        pattern_source="builtin_prbs13q_gray",
        samples_per_ui=4,
        pre_cursor_ui=2,
        post_cursor_ui=5,
        pattern_path=None,
        file_value_kind="symbol_codes",
    )


def _centered_validation_pulse(sample_rate_hz: float) -> TimeSeries:
    """构造前后窗口均有充足余量的单主光标脉冲。"""

    values = np.zeros(64, dtype=np.float64)
    values[32] = 1.0
    return _series(values.tolist(), sample_rate_hz)


def test_vpp_window_validation_accepts_minor_sample_rate_difference() -> None:
    """50 ppm 的跨文件采样率舍入差异应共享同一 Vpp 离散网格。"""

    reference_rate_hz = 8.0e9
    validate_vpp_pulse_windows(
        _centered_validation_pulse(reference_rate_hz),
        _centered_validation_pulse(reference_rate_hz * (1.0 + 50.0e-6)),
        _window_validation_settings(),
    )


def test_vpp_window_validation_rejects_material_sample_rate_difference() -> None:
    """200 ppm 差异必须保守拒绝，并报告实际差异和门限。"""

    reference_rate_hz = 8.0e9
    with pytest.raises(ValueError, match=r"200\.000 ppm.*100 ppm"):
        validate_vpp_pulse_windows(
            _centered_validation_pulse(reference_rate_hz),
            _centered_validation_pulse(reference_rate_hz * (1.0 + 200.0e-6)),
            _window_validation_settings(),
        )


# 独立 oracle 不使用 FFT 或生产折叠内核，只展开三周期并直接执行卷积定义。
def _three_period_direct_oracle(
    levels: np.ndarray,
    samples_per_ui: int,
    pulse_taps: np.ndarray,
    lag_samples: np.ndarray,
) -> np.ndarray:
    """以双循环直接计算三周期展开后的中间稳态周期。"""

    # 一个周期的离散长度等于 symbol 数乘每 UI 样点数。
    period_samples = int(levels.size * samples_per_ui)
    # 理想 symbol 仅在 UI 起点激励，UI 内其余样点为零。
    excitation = np.zeros(period_samples, dtype=np.float64)
    # 每隔 samples_per_ui 放置一个用户定义电平。
    excitation[::samples_per_ui] = levels
    # 展开三个相同周期，让中间周期同时具有真实的前后周期上下文。
    triple_excitation = np.tile(excitation, 3)
    # 预分配中间周期的直接卷积输出。
    output = np.zeros(period_samples, dtype=np.float64)
    # 外层逐点遍历中间周期，避免借用任何批量卷积实现。
    for phase_index in range(period_samples):
        # 中间周期的绝对索引从一个完整周期之后开始。
        absolute_index = period_samples + phase_index
        # 内层严格实现 y[n] = sum h[lag] * x[n-lag]。
        for tap_value, lag_value in zip(pulse_taps, lag_samples, strict=True):
            # 逐 tap 累加保留 lag 正负号和跨周期索引的独立证据。
            output[phase_index] += tap_value * triple_excitation[
                absolute_index - int(lag_value)
            ]
    # 返回不依赖生产 FFT 路径的稳态周期黄金数组。
    return output


# IEEE 示例前缀与完整周期哈希共同杀死位序、抽头和 Gray 映射错误。
def test_builtin_prbs13q_matches_ieee_prefix_histogram_and_hash() -> None:
    """内置码型必须是冻结的 8191-symbol PRBS13Q Gray 周期。"""

    # 公开发生器返回标准 symbol code，而不是已经缩放的模拟电压。
    symbols = generate_prbs13q_gray_symbols()

    # IEEE PRBS13Q 的完整 PAM4 周期固定为 8191 symbols。
    assert symbols.shape == (8191,)
    # uint8 足以无损保存 0..3，同时避免内置资产占用无谓内存。
    assert symbols.dtype == np.uint8
    # 标准种子 S0..S12 的公开前缀可检测 LFSR 位序和首次输出时机。
    assert "".join(str(int(value)) for value in symbols[:46]) == (
        "1031320220111130103121231210012102121023131112"
    )
    # 完整 symbol 直方图可检测只生成一遍奇数 PRBS 后错误配对的实现。
    np.testing.assert_array_equal(np.bincount(symbols, minlength=4), [2047, 2048, 2048, 2048])
    # 固定字节哈希让前缀之后的任何单点漂移都使测试变红。
    assert hashlib.sha256(symbols.tobytes()).hexdigest() == (
        "13c35313f944a67a294d841d43c621c0232fbd3723a635d405c6e52224becfc9"
    )


# 设置对象与电平载入器共同保证 UI 参数不会在进入数值核心后被隐式改写。
def test_builtin_pattern_maps_symbol_codes_to_readonly_normalized_pam4_levels() -> None:
    """内置 symbol code 必须显式映射为 -1、-1/3、1/3、1。"""

    # 上层完整填写所有冻结字段，内置模式不携带外部文件路径。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="builtin_prbs13q_gray",
        samples_per_ui=4,
        pre_cursor_ui=2,
        post_cursor_ui=5,
        pattern_path=None,
        file_value_kind="symbol_codes",
    )
    # 通过公开载入器取得用于卷积的归一化电压电平。
    levels = load_pattern_levels(settings)
    # 独立从标准 code 按线性公式构造期望值，避免复用生产映射表。
    expected = (2.0 * generate_prbs13q_gray_symbols().astype(np.float64) - 3.0) / 3.0

    # 整个 8191-symbol 周期必须逐点等于独立映射结果。
    np.testing.assert_array_equal(levels, expected)
    # 分析缓存对调用层只读，防止一次修改悄悄污染后续候选扫描。
    assert not levels.flags.writeable


# 完整内置周期必须真正进入模型 FFT，而不是只让独立发生器测试通过。
def test_builtin_pattern_prepares_full_8191_symbol_lfp_model() -> None:
    """内置 PRBS13Q 应按 M 插零并生成完整周期的可复用模型。"""

    # M=4 使周期长度足以暴露把 symbol 数误当样点数的实现。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="builtin_prbs13q_gray",
        samples_per_ui=4,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=None,
        file_value_kind="symbol_codes",
    )
    # 单位参考 tap 保留内置归一化码型的完整电平范围。
    reference_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # 半增益 DUT 提供可手验的 exact Vpp 缩放关系。
    dut_pulse = _series([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    # 生产准备入口加载标准码型、插零并构建两份周期模型。
    cache = prepare_vpp_analysis(reference_pulse, dut_pulse, settings)

    # 完整周期样点数必须严格等于 8191 symbols × 4 samples/UI。
    assert cache.period_samples == 8191 * 4
    # Fs=8 GSa/s、M=4 应明确推导出 2 GBd，不能把 Fs 与 Rs 混用。
    assert cache.symbol_rate_hz == pytest.approx(2.0e9)
    # 一个 UI 的物理时长是 M/Fs=0.5 ns。
    assert cache.ui_duration_s == pytest.approx(0.5e-9)
    # 单边 rFFT 频点数遵守 floor(N/2)+1。
    assert cache.frequency_hz.size == cache.period_samples // 2 + 1
    # 归一化 PRBS13Q 同时含 -1 和 +1，单位 tap 的 exact Vpp 应为 2 V。
    np.testing.assert_allclose(cache.reference_metric_v, 2.0, atol=1.0e-12)
    # 半增益 DUT 的 exact Vpp 应线性缩放为 1 V。
    np.testing.assert_allclose(cache.dut_metric_v, 1.0, atol=1.0e-12)


# 外部 symbol code 必须逐行读取并复用与内置资产相同的 PAM4 映射。
def test_external_symbol_code_file_loads_one_symbol_per_row(tmp_path: Path) -> None:
    """单列 0..3 code 文件必须按用户声明映射，不做类型猜测。"""

    # 写入故意非单调的短周期，使行序或 Gray 重排错误能被测试发现。
    pattern_path = tmp_path / "pattern_codes.csv"
    # 单列格式每一行只包含一个 symbol，符合冻结的外部文件合同。
    pattern_path.write_text("3\n0\n2\n1\n3\n", encoding="utf-8")
    # 用户明确声明文件内容是 symbol code，而不是无量纲幅度系数。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=2,
        pre_cursor_ui=0,
        post_cursor_ui=1,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )

    # 公开载入器执行单列验证和显式 code-to-level 映射。
    levels = load_pattern_levels(settings)

    # 逐点期望值可检测把 code 当作幅度或错误排序的实现。
    np.testing.assert_allclose(levels, [1.0, -1.0, 1.0 / 3.0, -1.0 / 3.0, 1.0])


# amplitude_values 模式必须原样保留无量纲系数，而不能再次执行 PAM4 code 映射。
def test_external_amplitude_file_preserves_finite_multilevel_values(tmp_path: Path) -> None:
    """显式无量纲幅度系数文件应保留数值尺度、偏置和每行顺序。"""

    # 使用不对称且带偏置的四电平数据，能杀死隐式归一化和去均值实现。
    pattern_path = tmp_path / "pattern_amplitudes.csv"
    # 每行一个无量纲系数，重复首电平以形成有意义的短周期边界。
    pattern_path.write_text("-0.8\n-0.2\n0.4\n1.1\n-0.8\n", encoding="utf-8")
    # 用户明确声明这些数值已经是所需的模拟电平。
    settings = VppAnalysisSettings(
        method="frequency_rms_error",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=1,
        post_cursor_ui=1,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )

    # 载入器只做格式和物理有限性验证，不改变用户定义的电平。
    levels = load_pattern_levels(settings)

    # 精确数组断言确保尺度、偏置、次序和重复值全部保留。
    np.testing.assert_array_equal(levels, [-0.8, -0.2, 0.4, 1.1, -0.8])


def test_preloaded_pattern_freezes_preflight_and_model_to_one_content(
    tmp_path: Path,
) -> None:
    """A controller-preloaded period must be reused without reopening its path."""

    pattern_path = tmp_path / "frozen_pattern.csv"
    pattern_path.write_text("-1.0\n0.5\n1.0\n-0.5\n", encoding="utf-8")
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=2,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    frozen_levels = load_pattern_levels(settings)
    # Replacing the path with an invalid constant file must not alter the frozen run.
    pattern_path.write_text("0.25\n0.25\n", encoding="utf-8")
    unit_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    cache = prepare_vpp_analysis(
        unit_pulse,
        unit_pulse,
        settings,
        prepared_pattern_levels=frozen_levels,
    )

    np.testing.assert_array_equal(cache.pattern_levels, frozen_levels)
    assert cache.period_samples == frozen_levels.size * settings.samples_per_ui


# 生产 rFFT 圆周卷积必须与完全独立的三周期直接求和在参考和 DUT 上同时一致。
def test_periodic_models_match_three_period_direct_convolution_oracle(tmp_path: Path) -> None:
    """短周期稳态模型必须正确处理插零、负 lag 与周期边界。"""

    # 五个不对称模拟电平避免特殊对称性掩盖周期移位错误。
    pattern_path = tmp_path / "short_amplitude_pattern.csv"
    # 单列周期故意让首尾不同，从而显式检验圆周折返。
    pattern_path.write_text("-1.0\n0.25\n0.75\n-0.5\n1.0\n", encoding="utf-8")
    # 每 UI 两点且前后各一 UI，窗口应精确覆盖 pmax 两侧各两个样点。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=2,
        pre_cursor_ui=1,
        post_cursor_ui=1,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 参考脉冲的窗口 tap 对应 lag -2、-1、0、1、2。
    reference_pulse = _series([0.0, 0.0, -0.2, 0.1, 1.0, 0.3, -0.1, 0.0, 0.0])
    # DUT 使用不同的前后游标和主峰，确保两份模型各自独立拟合。
    dut_pulse = _series([0.0, 0.0, 0.15, -0.05, 0.8, 0.25, 0.08, 0.0, 0.0])

    # 生产入口一次准备参考、DUT 和可复用频域缓存。
    cache = prepare_vpp_analysis(reference_pulse, dut_pulse, settings)
    # 载入原始幅度周期供完全独立的直接卷积 oracle 使用。
    levels = np.array([-1.0, 0.25, 0.75, -0.5, 1.0], dtype=np.float64)
    # 包含式窗口的离散 lag 必须恰为 -2 到 +2。
    lag_samples = np.arange(-2, 3, dtype=np.int64)
    # 直接计算参考稳态周期，不调用生产模块的任何卷积辅助函数。
    expected_reference = _three_period_direct_oracle(
        levels,
        2,
        np.array([-0.2, 0.1, 1.0, 0.3, -0.1]),
        lag_samples,
    )
    # 对 DUT 重复独立计算，防止生产端错误复用参考脉冲窗口。
    expected_dut = _three_period_direct_oracle(
        levels,
        2,
        np.array([0.15, -0.05, 0.8, 0.25, 0.08]),
        lag_samples,
    )

    # rFFT 圆周卷积应在浮点容差内逐点等于三周期直接求和。
    np.testing.assert_allclose(cache.reference_model.waveform_v, expected_reference, atol=1.0e-12)
    # DUT 模型也必须逐点一致，不能只让参考路径正确。
    np.testing.assert_allclose(cache.dut_model.waveform_v, expected_dut, atol=1.0e-12)
    # pmax 为索引 4，前后各两点的包含式窗口应精确落在 [2, 7)。
    assert (cache.reference_model.peak_index, cache.reference_model.window_start) == (4, 2)
    # stop 为 7 才能同时包含 lag +2，防止常见的右端少一个样点错误。
    assert cache.reference_model.window_stop == 7
    # 公开 lag 数组让界面和测试能明确复核 UI 到样点的精确换算。
    np.testing.assert_array_equal(cache.reference_model.lag_samples, [-2, -1, 0, 1, 2])
    # LFP 指标定义为完整稳态周期的 exact max-min，而不是分位数近似。
    np.testing.assert_allclose(cache.reference_metric_v, np.ptp(expected_reference), atol=1.0e-12)
    # DUT 指标遵循相同定义，供上层计算补偿前后的变化。
    np.testing.assert_allclose(cache.dut_metric_v, np.ptp(expected_dut), atol=1.0e-12)


# 偶数周期验证 Nyquist 单计权，奇数周期验证最后一个 rFFT 频点仍需双计权。
@pytest.mark.parametrize(
    "levels",
    [
        [-0.7, 0.2, 1.1, -0.1, 0.8, 0.4],
        [-0.7, 0.2, 1.1, -0.1, 0.8],
    ],
    ids=["even_with_nyquist", "odd_last_bin_doubled"],
)
def test_frequency_rms_error_matches_time_domain_ac_parseval(
    tmp_path: Path,
    levels: list[float],
) -> None:
    """频域误差 Vrms 必须等于时域误差去 DC 后的 RMS。"""

    # 每个参数化场景写成严格单列幅度文件，避免引入 code 映射因素。
    pattern_path = tmp_path / f"parseval_{len(levels)}.csv"
    # 保留非零均值以确认生产定义确实排除 D[0]，而非计算总 RMS。
    pattern_path.write_text("\n".join(str(value) for value in levels), encoding="utf-8")
    # M=1 且窗口只含 pmax，使独立时域期望可直接按 tap 比例手算。
    settings = VppAnalysisSettings(
        method="frequency_rms_error",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 参考单 tap 增益为 1，放在记录中央以满足任何边界检查。
    reference_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # DUT 单 tap 增益为 0.35，因此误差时域应为 -0.65 倍码型。
    dut_pulse = _series([0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0])

    # 生产入口应在频域计算参考与 DUT 的 AC 误差指标。
    cache = prepare_vpp_analysis(reference_pulse, dut_pulse, settings)
    # 独立时域误差不复用生产频谱或 Parseval 辅助函数。
    direct_error = -0.65 * np.asarray(levels, dtype=np.float64)
    # D[0]=0 等价于从周期误差中移除其平均值。
    direct_ac_error = direct_error - np.mean(direct_error)
    # 直接均方根是单边频谱计权公式的独立 oracle。
    expected_vrms = float(np.sqrt(np.mean(np.square(direct_ac_error))))

    # 参考相对于自身的频域误差定义为严格零。
    assert cache.reference_metric_v == 0.0
    # DUT Vrms 必须同时满足偶/奇长度的独立时域 Parseval 等价关系。
    np.testing.assert_allclose(cache.dut_metric_v, expected_vrms, rtol=1.0e-13, atol=1.0e-13)


# 候选补偿应直接复用 DUT 周期频谱，LFP 只允许为最终 exact Vpp 做一次 IFFT。
def test_lfp_candidate_correction_recovers_reference_with_one_ifft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全带 2 倍补偿必须把 0.5 倍 DUT 恢复为参考且只 IFFT 一次。"""

    # 非对称短码型确保输出包含多个不同峰谷，exact Vpp 不是退化常数。
    pattern_path = tmp_path / "candidate_lfp.csv"
    # 使用单列无量纲系数避免 symbol code 映射掩盖候选缩放错误。
    pattern_path.write_text("-1.0\n0.2\n0.8\n-0.4\n", encoding="utf-8")
    # 零游标窗口把脉冲模型约化为可手验的单 tap 增益。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=2,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 参考脉冲主 tap 为 1。
    reference_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # DUT 主 tap 为 0.5，完整频谱乘二后应逐点恢复参考。
    dut_pulse = _series([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    # 准备阶段的 IFFT 不计入候选测量次数，因此先完成缓存。
    cache = prepare_vpp_analysis(reference_pulse, dut_pulse, settings)
    # rFFT 每个频点统一乘二，端点保持真实，表示无相位旋转的全带补偿。
    correction = np.full(cache.frequency_hz.shape, 2.0 + 0.0j, dtype=np.complex128)
    # 保存真实 SciPy IFFT，计数包装器仍调用原实现得到候选波形。
    original_irfft = vpp_analysis.scipy_fft.irfft
    # 可变列表让嵌套包装器在 Python 作用域内累加调用次数。
    irfft_calls = [0]

    # 包装器接受任意 SciPy 参数，以免测试耦合到实现的调用风格。
    def counting_irfft(*args: object, **kwargs: object) -> np.ndarray:
        """记录生产候选路径的 IFFT 次数并转发真实实现。"""

        # 每进入一次包装器就记录一次候选时域恢复。
        irfft_calls[0] += 1
        # 转发原函数并返回其 float64 周期波形。
        return original_irfft(*args, **kwargs)

    # 只替换被测模块持有的 SciPy FFT 命名空间成员。
    monkeypatch.setattr(vpp_analysis.scipy_fft, "irfft", counting_irfft)

    # 公开候选入口以 DUT 模型频谱乘 correction 后执行 LFP 测量。
    measurement = measure_candidate(cache, correction)

    # LFP 候选路径严格只允许一次完整周期 IFFT。
    assert irfft_calls[0] == 1
    # 2 倍补偿后的候选波形应逐点恢复参考稳态波形。
    np.testing.assert_allclose(
        measurement.waveform_v,
        cache.reference_model.waveform_v,
        atol=1.0e-12,
    )
    # 候选指标也应恢复参考 exact max-min。
    np.testing.assert_allclose(measurement.value_v, cache.reference_metric_v, atol=1.0e-12)


# 纯相位候选的幅度谱不变，只有比较复频谱才能发现其时域误差。
def test_rms_candidate_is_phase_sensitive_and_never_calls_ifft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RMS 候选必须检测纯相位误差，并完全留在频域。"""

    # 七点奇数周期同时继续覆盖无 Nyquist 条件下的相位误差。
    pattern_path = tmp_path / "phase_sensitive_rms.csv"
    # 非对称、非零均值码型让多个内部频点具有可观能量。
    pattern_path.write_text("-1.0\n-0.2\n0.9\n0.3\n-0.6\n1.1\n0.4\n", encoding="utf-8")
    # 单 tap、M=1 让参考与 DUT 基线完全一致，候选误差只来自 correction 相位。
    settings = VppAnalysisSettings(
        method="frequency_rms_error",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 参考与 DUT 使用同一单位脉冲，因此准备后的基线 RMS 应为零。
    unit_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # 在 monkeypatch 之前完成模型准备，因为准备阶段允许为展示构造基线波形。
    cache = prepare_vpp_analysis(unit_pulse, unit_pulse, settings)
    # 从单位补偿开始，只旋转两个拥有负频共轭伙伴的内部 bin。
    correction = np.ones(cache.frequency_hz.shape, dtype=np.complex128)
    # 第一内部频点施加明显正相位，幅度严格保持一。
    correction[1] = np.exp(1.0j * 0.70)
    # 第二内部频点施加不同负相位，防止偶然的纯循环移位关系。
    correction[2] = np.exp(-1.0j * 0.35)
    # 独立使用 NumPy IFFT 构造时间域 oracle，不触碰被测 SciPy 路径。
    expected_candidate = np.fft.irfft(
        cache.dut_model.spectrum_v * correction,
        n=cache.period_samples,
    )
    # 参考稳态周期同样用 NumPy 由缓存频谱恢复。
    expected_reference = np.fft.irfft(
        cache.reference_model.spectrum_v,
        n=cache.period_samples,
    )
    # AC 误差先去均值，再按时间域定义直接计算 RMS。
    direct_error = expected_candidate - expected_reference
    # 该独立 oracle 等价于生产端显式令 D[0]=0。
    direct_ac_error = direct_error - np.mean(direct_error)
    # 保存预期纯相位误差 Vrms，供频域结果比较。
    expected_vrms = float(np.sqrt(np.mean(np.square(direct_ac_error))))

    # 若 RMS 候选错误调用 SciPy IFFT，测试应立即以明确消息失败。
    def forbidden_irfft(*args: object, **kwargs: object) -> np.ndarray:
        """禁止频域 RMS 候选退回时间域。"""

        # 任何调用都违反冻结的 RMS 无 IFFT 性能合同。
        raise AssertionError("frequency_rms_error candidate must not call irfft")

    # 只在候选测量阶段替换生产模块使用的 IFFT。
    monkeypatch.setattr(vpp_analysis.scipy_fft, "irfft", forbidden_irfft)

    # 候选测量应仅做频谱乘法和 Parseval 加权。
    measurement = measure_candidate(cache, correction)

    # 纯相位补偿不改变幅度谱，证明非零结果不能来自幅度差。
    np.testing.assert_allclose(
        np.abs(measurement.corrected_spectrum_v),
        np.abs(cache.dut_model.spectrum_v),
        atol=1.0e-12,
    )
    # 复频谱相位误差必须产生严格正的 Vrms；只比较幅度谱会错误返回零。
    assert measurement.value_v > 0.0
    # 生产 Parseval 结果应等于独立 NumPy 时间域 AC RMS。
    np.testing.assert_allclose(measurement.value_v, expected_vrms, rtol=1.0e-13, atol=1.0e-13)
    # None 是“未执行 IFFT”的公开结果语义。
    assert measurement.waveform_v is None


# pmax 和窗口边界是拟合脉冲语义的根基，必须以明确索引 fail-closed。
def test_first_absolute_peak_is_used_and_out_of_record_window_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同绝对峰值取首次，任一窗口越界都不能补零或截短。"""

    # 三 symbol 幅度周期足够构建短模型，同时包含至少两个电平。
    pattern_path = tmp_path / "peak_boundary.csv"
    # 单列幅度文件避免额外的 symbol 映射因素。
    pattern_path.write_text("-1.0\n0.25\n1.0\n", encoding="utf-8")
    # 零游标先隔离首次峰值选择行为，M=2 留待越界场景精确换算。
    zero_window_settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=2,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 参考在索引 3 和 6 具有相同绝对峰值，首次应固定为索引 3。
    reference_pulse = _series([0.0, 0.1, 0.2, 1.0, 0.3, 0.2, -1.0, 0.0, 0.0])
    # DUT 的独立峰值位于索引 5，证明实现不能复用参考 peak_index。
    dut_pulse = _series([0.0, 0.1, 0.2, 0.3, 0.4, -0.9, 0.2, 0.0, 0.0])

    # 先准备合法零窗口模型，直接检查两条独立峰值路径。
    cache = prepare_vpp_analysis(reference_pulse, dut_pulse, zero_window_settings)

    # np.argmax(abs) 的首次命中合同应选索引 3，而不是后面的 -1。
    assert cache.reference_model.peak_index == 3
    # DUT 必须使用自己的索引 5。
    assert cache.dut_model.peak_index == 5
    # 零 UI 窗口仍包含 pmax 本身，窗口长度必须恰为一个样点。
    assert cache.reference_model.window_stop - cache.reference_model.window_start == 1

    # 前游标两 UI 在 M=2 时精确等于四样点，会把参考 start 推到 -1。
    overflowing_settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=2,
        pre_cursor_ui=2,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 让任何码型 FFT 都直接失败，证明窗口门禁发生在周期级分配之前。
    def forbidden_rfft(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("invalid pulse window must fail before pattern FFT")

    monkeypatch.setattr(vpp_analysis.scipy_fft, "rfft", forbidden_rfft)
    # 越界必须抛出领域错误，不能通过零填充或截短继续计算一个看似有效的 Vpp。
    with pytest.raises(ValueError, match="拟合脉冲窗口越界"):
        # 公开入口应在构建参考模型时立即发现 start=-1。
        prepare_vpp_analysis(reference_pulse, dut_pulse, overflowing_settings)


@pytest.mark.parametrize("zero_role", ["reference", "dut"])
def test_zero_fitted_pulse_fails_before_pattern_fft(
    zero_role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-zero pulse has no pmax origin and must fail before period allocation."""

    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="builtin_prbs13q_gray",
        samples_per_ui=64,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=None,
        file_value_kind="symbol_codes",
    )
    zero_pulse = _series([0.0] * 9)
    unit_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    reference_pulse = zero_pulse if zero_role == "reference" else unit_pulse
    dut_pulse = zero_pulse if zero_role == "dut" else unit_pulse

    def forbidden_rfft(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("zero pmax must fail before pattern FFT")

    monkeypatch.setattr(vpp_analysis.scipy_fft, "rfft", forbidden_rfft)

    with pytest.raises(ValueError, match="pmax.*非零"):
        prepare_vpp_analysis(reference_pulse, dut_pulse, settings)


# 冻结设置对象应在任何文件或 FFT 工作前拒绝无效枚举、布尔整数和路径矛盾。
@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "percentile"},
        {"pattern_source": "auto"},
        {"file_value_kind": "guess"},
        {"samples_per_ui": 0},
        {"samples_per_ui": True},
        {"pre_cursor_ui": -1},
        {"post_cursor_ui": 0.5},
        {"pattern_source": "file", "pattern_path": None},
        {"pattern_path": Path("unexpected.csv")},
    ],
    ids=[
        "unknown_method",
        "unknown_source",
        "guessed_value_kind",
        "zero_samples_per_ui",
        "boolean_samples_per_ui",
        "negative_pre_cursor",
        "fractional_post_cursor",
        "missing_file_path",
        "builtin_with_file_path",
    ],
)
def test_settings_fail_closed_on_invalid_frozen_contract(overrides: dict[str, object]) -> None:
    """无效设置必须在构造阶段抛错，不能隐式修正或猜测。"""

    # 基础字典是一份完整合法的内置 LFP 设置。
    base_settings: dict[str, object] = {
        "method": "lfp",
        "pattern_source": "builtin_prbs13q_gray",
        "samples_per_ui": 2,
        "pre_cursor_ui": 1,
        "post_cursor_ui": 1,
        "pattern_path": None,
        "file_value_kind": "symbol_codes",
    }
    # 每个参数化场景仅覆盖一个合同维度，便于定位失败原因。
    invalid_settings = {**base_settings, **overrides}

    # 构造阶段必须统一以 ValueError 拒绝，不让错误流入数值核心。
    with pytest.raises(ValueError):
        # 动态字典刻意模拟 GUI 表单组装后的运行时输入。
        VppAnalysisSettings(**invalid_settings)  # type: ignore[arg-type]


# 外部文件必须严格遵守“一列/一 symbol”和用户声明的值语义。
@pytest.mark.parametrize(
    ("value_kind", "contents", "message"),
    [
        ("symbol_codes", "0,1,2\n", "一列"),
        ("symbol_codes", "0\n1.5\n2\n", "整数 0、1、2、3"),
        ("symbol_codes", "0\n4\n", "整数 0、1、2、3"),
        ("symbol_codes", "0\n0\n0\n", "至少需要两个不同 symbol"),
        ("amplitude_values", "0.25\n0.25\n", "至少需要两个不同电平"),
        ("amplitude_values", "0.0\nnan\n", "有限值"),
    ],
    ids=[
        "multiple_columns",
        "fractional_symbol_code",
        "out_of_range_symbol_code",
        "constant_symbol_code",
        "constant_amplitude",
        "nonfinite_amplitude",
    ],
)
def test_external_pattern_rejects_malformed_or_semantically_invalid_values(
    tmp_path: Path,
    value_kind: str,
    contents: str,
    message: str,
) -> None:
    """外部码型不得展平多列、量化非法 code 或接受无效幅度。"""

    # 每个场景写入独立临时文件，确保解析只受当前内容影响。
    pattern_path = tmp_path / "invalid_pattern.csv"
    # UTF-8 文本保持与正常外部码型入口相同的真实文件路径。
    pattern_path.write_text(contents, encoding="utf-8")
    # 值语义由参数显式指定，生产载入器不得自行切换类型。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind=value_kind,  # type: ignore[arg-type]
    )

    # 错误消息应指出被违反的格式或数值约束。
    with pytest.raises(ValueError, match=message):
        # 任何非法输入都必须在生成 FFT 前停止。
        load_pattern_levels(settings)


# 误把大型采集 CSV 选为理想码型时必须在文本解析和 FFT 分配之前停止。
def test_external_pattern_file_size_is_bounded_before_parsing(tmp_path: Path) -> None:
    """Oversized pattern text should fail before np.loadtxt can allocate arrays."""

    pattern_path = tmp_path / "mistaken_capture.csv"
    # 稀疏截断快速构造超过 32 MiB 的文件，不把大测试数据真正写入仓库。
    with pattern_path.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )

    with pytest.raises(ValueError, match="32 MiB|采集波形"):
        load_pattern_levels(settings)


def test_external_pattern_dynamic_budget_rejects_before_numpy_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """码型文本的最坏临时数组必须在 np.loadtxt 前进入动态内存预算。"""

    pattern_path = tmp_path / "memory_guard_pattern.csv"
    pattern_path.write_text("0\n1\n2\n3\n", encoding="utf-8")
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )
    monkeypatch.setattr(
        vpp_analysis,
        "current_memory_budget",
        lambda: MemoryBudget(available_bytes=64 * 1024**2, budget_bytes=8 * 1024**2),
        raising=False,
    )

    def forbidden_loadtxt(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("pattern memory guard must run before np.loadtxt")

    monkeypatch.setattr(vpp_analysis.np, "loadtxt", forbidden_loadtxt)

    with pytest.raises(MemoryError, match="理想码型.*动态内存预检"):
        load_pattern_levels(settings)


def test_pattern_memory_estimate_envelopes_maximum_short_line_snapshot_benchmark() -> None:
    """32 MiB 最短行冻结快照的估算须覆盖独立 maxRSS 实测并保留至少 20% 余量。"""

    measured_peak_bytes = 878_362_624
    estimated_peak_bytes = vpp_analysis._estimate_pattern_loader_peak_bytes(
        32 * 1024**2
    )

    assert estimated_peak_bytes >= int(np.ceil(1.20 * measured_peak_bytes))


def test_pattern_symbol_count_preflight_runs_before_numpy_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一文件描述符统计的有效 symbol 数必须在数组解析前交给工作量门禁。"""

    pattern_path = tmp_path / "preflight_pattern.csv"
    pattern_path.write_text("# comment\n0\n\n1\n2 # inline\n3\n", encoding="utf-8")
    settings = VppAnalysisSettings(
        method="frequency_rms_error",
        pattern_source="file",
        samples_per_ui=4,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )
    observed_counts: list[int] = []
    original_loadtxt = np.loadtxt

    def guarded_loadtxt(*args: object, **kwargs: object) -> np.ndarray:
        assert observed_counts == [4]
        return original_loadtxt(*args, **kwargs)

    monkeypatch.setattr(vpp_analysis.np, "loadtxt", guarded_loadtxt)

    levels = load_pattern_levels(
        settings,
        symbol_count_preflight=observed_counts.append,
    )

    assert observed_counts == [4]
    np.testing.assert_allclose(levels, [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])


def test_pattern_growth_during_preflight_cannot_expand_numpy_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并发追加只能使冻结快照失效，不能让解析器读取初始大小以外的文本。"""

    pattern_path = tmp_path / "growing_pattern.csv"
    initial_text = "0\n1\n"
    pattern_path.write_text(initial_text, encoding="utf-8")
    initial_snapshot_text = pattern_path.read_bytes().decode("utf-8")
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )
    original_loadtxt = np.loadtxt

    def append_after_count(_symbol_count: int) -> None:
        with pattern_path.open("a", encoding="utf-8") as stream:
            stream.write("2\n3\n")

    def inspect_frozen_source(source: object, *args: object, **kwargs: object) -> np.ndarray:
        assert hasattr(source, "getvalue")
        assert source.getvalue() == initial_snapshot_text
        return original_loadtxt(source, *args, **kwargs)

    monkeypatch.setattr(vpp_analysis.np, "loadtxt", inspect_frozen_source)

    with pytest.raises(ValueError, match="加载期间发生变化"):
        load_pattern_levels(
            settings,
            symbol_count_preflight=append_after_count,
        )


def test_external_pattern_load_cancels_before_numpy_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close request during chunked input must stop before NumPy parsing."""

    pattern_path = tmp_path / "large_pattern.csv"
    pattern_path.write_bytes(b"0\n" * 700_000)
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )
    cancellation_checks = 0

    def cancelled() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks >= 2

    def forbidden_loadtxt(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("cancelled pattern must not reach NumPy parsing")

    monkeypatch.setattr(vpp_analysis.np, "loadtxt", forbidden_loadtxt)

    with pytest.raises(OperationCancelledError, match="已取消"):
        load_pattern_levels(settings, cancelled=cancelled)

    assert cancellation_checks == 2


def test_prepare_vpp_analysis_forwards_cancellation_to_external_pattern_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公共准备入口自行载入外部码型时也必须贯穿取消回调。"""

    pattern_path = tmp_path / "pattern.csv"
    pattern_path.write_text("0\n1\n", encoding="utf-8")
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="symbol_codes",
    )
    unit_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def observe_loader(_settings, *, cancelled=None, **_kwargs):
        assert cancelled is not None
        raise OperationCancelledError("影响频段分析已取消")

    monkeypatch.setattr(vpp_analysis, "load_pattern_levels", observe_loader)

    with pytest.raises(OperationCancelledError, match="已取消"):
        prepare_vpp_analysis(
            unit_pulse,
            unit_pulse,
            settings,
            cancelled=lambda: False,
        )


# 候选频谱必须完整覆盖缓存频轴且全部有限，防止候选扫描静默错位。
def test_candidate_correction_rejects_wrong_shape_and_nonfinite_values(tmp_path: Path) -> None:
    """错误长度、二维或 NaN correction 都必须 fail-closed。"""

    # 三点短码型构造最小但非退化的 LFP 缓存。
    pattern_path = tmp_path / "candidate_validation.csv"
    # 单列无量纲系数具有三个不同电平。
    pattern_path.write_text("-1.0\n0.0\n1.0\n", encoding="utf-8")
    # 单 tap 模型让本测试只关注 correction 输入验证。
    settings = VppAnalysisSettings(
        method="lfp",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 参考与 DUT 相同即可得到合法缓存和明确的 rFFT 形状。
    unit_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # 准备一次后复用到三个非法候选场景。
    cache = prepare_vpp_analysis(unit_pulse, unit_pulse, settings)

    # 少一个频点不能靠截断或广播补齐。
    with pytest.raises(ValueError, match="形状一致"):
        # 一维但长度错误专门验证频轴匹配分支。
        measure_candidate(cache, np.ones(cache.frequency_hz.size - 1, dtype=np.complex128))
    # 二维数组即使元素总数相同也不能隐式展平。
    with pytest.raises(ValueError, match="必须是一维"):
        # 列向量可杀死粗暴 reshape(-1) 的实现。
        measure_candidate(cache, np.ones((cache.frequency_hz.size, 1), dtype=np.complex128))
    # 非有限频点不能进入复数乘法和能量平方。
    with pytest.raises(ValueError, match="有限值"):
        # 全 NaN 候选验证有限性检查发生在指标计算前。
        measure_candidate(
            cache,
            np.full(cache.frequency_hz.shape, np.nan + 0.0j, dtype=np.complex128),
        )


# 隔离 DC 与 Nyquist 可避免宽带随机例子偶然掩盖单边频谱端点计权错误。
def test_rms_candidate_excludes_dc_and_counts_even_nyquist_once(tmp_path: Path) -> None:
    """DC-only 变化应为零，偶数周期 Nyquist 误差能量只能计一次。"""

    # 六点偶数周期确保 rFFT 最后一个频点是真实 Nyquist bin。
    pattern_path = tmp_path / "dc_nyquist.csv"
    # 交替分量非零，使 Nyquist-only 候选产生可测误差。
    pattern_path.write_text("-1.0\n0.2\n0.8\n-0.4\n1.1\n0.3\n", encoding="utf-8")
    # 单 tap 参考/DUT 完全相同，候选差异仅由指定端点 correction 产生。
    settings = VppAnalysisSettings(
        method="frequency_rms_error",
        pattern_source="file",
        samples_per_ui=1,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        pattern_path=pattern_path,
        file_value_kind="amplitude_values",
    )
    # 单位脉冲让模型频谱就是码型频谱，便于端点独立推导。
    unit_pulse = _series([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # 准备共享参考与 DUT 频谱的 RMS 缓存。
    cache = prepare_vpp_analysis(unit_pulse, unit_pulse, settings)

    # 只把 DC 增益改成七倍，其余频点保持单位补偿。
    dc_only_correction = np.ones(cache.frequency_hz.shape, dtype=np.complex128)
    # D[0] 按合同必须被排除，因此该变化不贡献 Vrms。
    dc_only_correction[0] = 7.0 + 0.0j
    # 频域候选无需恢复波形即可判断 DC-only 误差。
    dc_measurement = measure_candidate(cache, dc_only_correction)

    # 严格零可杀死把 DC 也纳入总 RMS 的错误实现。
    assert dc_measurement.value_v == 0.0

    # 偶数长度 Nyquist 只有实数余弦自由度，irfft 会忽略该 bin 的虚部。
    imaginary_nyquist_correction = np.ones(
        cache.frequency_hz.shape,
        dtype=np.complex128,
    )
    # 保持实部为单位增益，只添加一个无法对应真实波形的虚部。
    imaginary_nyquist_correction[-1] = 1.0 + 5.0j
    # 频域指标必须与真实 irfft 语义一致，不能把无物理自由度的虚部计作误差。
    imaginary_nyquist_measurement = measure_candidate(
        cache,
        imaginary_nyquist_correction,
    )

    # 规范化 rFFT 端点后候选仍与参考完全相同，因此 AC 误差应为零。
    assert imaginary_nyquist_measurement.value_v == 0.0

    # 只反转 Nyquist 符号，内部成对频点和 DC 全部保持不变。
    nyquist_correction = np.ones(cache.frequency_hz.shape, dtype=np.complex128)
    # 偶数长度最后一个 rFFT bin 是无负频伙伴的唯一 Nyquist 分量。
    nyquist_correction[-1] = -1.0 + 0.0j
    # 使用 NumPy IFFT 独立恢复候选和参考，构造时间域 RMS oracle。
    direct_candidate = np.fft.irfft(
        cache.dut_model.spectrum_v * nyquist_correction,
        n=cache.period_samples,
    )
    # 缓存参考频谱恢复为独立参考周期。
    direct_reference = np.fft.irfft(
        cache.reference_model.spectrum_v,
        n=cache.period_samples,
    )
    # Nyquist 误差自身均值为零，但仍显式去均值以遵守统一 AC 定义。
    direct_error = direct_candidate - direct_reference
    # 直接时间域 oracle 不依赖生产 Parseval 端点计权。
    expected_vrms = float(
        np.sqrt(np.mean(np.square(direct_error - np.mean(direct_error))))
    )
    # 生产候选应在单边频谱中只计一次 Nyquist 能量。
    nyquist_measurement = measure_candidate(cache, nyquist_correction)

    # 非零断言保证测试码型确实激励了 Nyquist，而不是退化通过。
    assert expected_vrms > 0.0
    # 若错误地把 Nyquist 乘二，结果会放大 sqrt(2) 并使该断言失败。
    np.testing.assert_allclose(
        nyquist_measurement.value_v,
        expected_vrms,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
