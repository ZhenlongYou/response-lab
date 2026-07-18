"""影响频段引擎的独立物理谕示与定向回归测试。"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本测试仅关闭对应格式告警。
# ruff: noqa: I001

# 延迟解析测试辅助函数的类型标注，保持 Python 3.11 下的导入行为稳定。
from __future__ import annotations

# NumPy 用于构造有手算结果的离散脉冲和时间轴。
import numpy as np
# pytest 提供显式数值容差，让浮点误差不会被宽范围掩盖。
import pytest
# SciPy FIR 设计构造严格线性相位、仅幅度受损的独立合成 oracle。
from scipy import signal

# 模块别名仅用于窃听内部缓存构造次数，数值验证仍走公共接口。
import response_lab.attribution as attribution_module
# 归因公共接口是测试对用户可见行为的唯一依赖。
from response_lab.attribution import (
    AttributionSettings,
    BandAttribution,
    FrequencyBand,
    VirtualEyeSettings,
    build_virtual_eye,
    build_eye_comparison,
    candidate_frequency_bands,
    compose_frequency_correction,
    cosine_core_band_weights,
    evaluate_attribution_band,
    prepare_frequency_attribution,
    scan_frequency_attribution,
    tukey_band_weights,
)
# TimeSeries 同时校验采样率与时间轴的物理一致性。
from response_lab.models import TimeSeries


# 构造可手算的单 UI 矩形响应，作为 NRZ/PAM4 2 UI 轨迹几何的公共谕示。
def _ideal_one_ui_pulse(*, np_ui: int, samples_per_ui: int) -> TimeSeries:
    """构造无 ISI 的单 UI 矩形脉冲，NRZ 眼高应为 2、眼宽应为 1 UI。"""

    # 8 GHz 和 M=8 给出 1 ns UI，方便手算并避免单位歧义。
    sample_rate_hz = 8.0e9
    # 总样点数必须严格等于 Np*M，保护用户已确认的输入合同。
    samples = np_ui * samples_per_ui
    # 均匀时间轴让 TimeSeries 可独立反算出同一采样率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 脉冲默认为零，不向其他 UI 注入前游或后游 ISI。
    values = np.zeros(samples, dtype=np.float64)
    # 把单位矩形放在记录中部，避免首尾裁剪影响手算结果。
    pulse_start = (np_ui // 2) * samples_per_ui
    # 持续整数一个 UI 的单位响应使每个采样相位都完全开眼。
    values[pulse_start : pulse_start + samples_per_ui] = 1.0
    # 单通道电压保留二维形状，与真实 CSV 加载合同一致。
    return TimeSeries(time_s, values[:, None], sample_rate_hz)


# 生成固定一 UI 回波输入，供缓存、扫描与候选回放测试共享同一物理基线。
def _echo_eye_inputs() -> tuple[
    TimeSeries,
    TimeSeries,
    AttributionSettings,
]:
    """构造一组频响非零且适合缓存回归的短脉冲。"""

    # 8 GSa/s 与 M=8 给出 1 GBd，Np=20 使物理分辨率为 50 MHz。
    sample_rate_hz = 8.0e9
    # 每 UI 八个样点同时保持眼宽有足够相位粒度。
    samples_per_ui = 8
    # 20 UI 记录容纳主冲激和一个短延时回波。
    pulse_length_ui = 20
    # 样点数严格由 Np*M 得到。
    samples = pulse_length_ui * samples_per_ui
    # 均匀时间轴使 TimeSeries 可独立验证采样率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 参考使用记录内部的单位冲激。
    reference_values = np.zeros(samples, dtype=np.float64)
    # 主冲激远离边界，不让延拓方式主导测试。
    reference_values[64] = 1.0
    # DUT 保留主冲激并加入一 UI 后的浅回波。
    dut_values = reference_values.copy()
    # 0.18 的回波不会产生频响零点，所有候选均可稳定测量。
    dut_values[72] = 0.18
    # 两份脉冲共享完全相同的时间几何。
    reference_pulse = TimeSeries(
        time_s,
        reference_values[:, None],
        sample_rate_hz,
    )
    # DUT 仅在电压响应上不同。
    dut_pulse = TimeSeries(time_s, dut_values[:, None], sample_rate_hz)
    # 0–2 GHz 全频联合补偿能改善回波眼，因此可继续观察全部局部扫描。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.0,
        scan_high_hz=2.0e9,
        eye=VirtualEyeSettings(
            modulation="pam4",
            pulse_length_ui=pulse_length_ui,
            samples_per_ui=samples_per_ui,
            symbol_count=700,
            random_seed=20260718,
        ),
    )
    # 返回的三元组可在多个缓存测试中保持同一数值输入。
    return reference_pulse, dut_pulse, settings


# 锁定理想 NRZ 的归一化眼高 2 和眼宽 1 UI，防止电平或轨迹定义漂移。
def test_ideal_nrz_pulse_has_hand_calculable_open_eye() -> None:
    """理想 NRZ 相邻电平 -1/+1 的间距为 2，且全 UI 无 ISI。"""

    # 选择较短的 8 UI 记录使 oracle 简单，同时仍覆盖首尾稳态裁剪。
    pulse = _ideal_one_ui_pulse(np_ui=8, samples_per_ui=8)
    # 固定种子与足够符号数使两个 NRZ 电平都有确定条件分布。
    settings = VirtualEyeSettings(
        modulation="nrz",
        pulse_length_ui=8,
        samples_per_ui=8,
        symbol_count=2048,
        random_seed=20260718,
    )

    # 通过公共接口生成眼图与指标，不读取内部中间量。
    result = build_virtual_eye(pulse, settings)

    # 理想 NRZ 上下电平为 +1/-1，所以唯一眼的手算高度为 2 V。
    assert result.eye_heights_v == pytest.approx((2.0,), abs=1.0e-12)
    # 所有 M 个相位均保持正开口，因此手算宽度为整数 1 UI。
    assert result.eye_widths_ui == pytest.approx((1.0,), abs=1.0e-12)
    # 横轴必须严格按附件从 -1 UI 到 +1 UI，并包含 2*M+1 个样点。
    np.testing.assert_array_equal(
        result.plot_time_ui,
        np.linspace(-1.0, 1.0, 17, dtype=np.float64),
    )
    # 每行轨迹长度必须是 2*M+1，且绘图最多保留 600 条确定性轨迹。
    assert result.plot_traces_v.shape == (600, 17)
    # 绘图轨迹必须真实包含两个电平，防止界面只显示占位图。
    assert np.min(result.plot_traces_v) == pytest.approx(-1.0, abs=1.0e-12)
    # 上轨同样必须到达 +1 V，与眼高 oracle 交叉约束。
    assert np.max(result.plot_traces_v) == pytest.approx(1.0, abs=1.0e-12)
    # 第 M 列就是 0 UI 主光标，不能被错误放在轨迹首点或末点。
    assert set(np.unique(result.plot_traces_v[:, 8]).round(12)) == {-1.0, 1.0}


# 眼宽只能计入包含 0 UI 的连续开口，远端孤岛不得被求和到同一只眼。
def test_eye_width_ignores_disconnected_open_islands() -> None:
    """中心连续区之外的正开口样点不应增加眼宽。"""

    # 九个相位点对应 M=4 的 -1 UI 到 +1 UI 轨迹。
    open_mask = np.array(
        [True, True, False, True, True, True, False, True, True],
        dtype=np.bool_,
    )
    # 中心索引 4 所在连续区只有 3 个相位箱，即 3/4 UI。
    assert attribution_module._centered_open_width(  # noqa: SLF001 - 独立几何谕示
        open_mask,
        center=4,
        samples_per_ui=4,
    ) == pytest.approx(0.75, abs=1.0e-12)
    # 中心闭合时，即使两侧都有孤立开口，围绕 0 UI 的眼宽仍必须为零。
    center_closed = open_mask.copy()
    # 显式关闭 0 UI 相位箱。
    center_closed[4] = False
    # 远端 True 不得被错误求和成非零眼宽。
    assert attribution_module._centered_open_width(  # noqa: SLF001 - 独立几何谕示
        center_closed,
        center=4,
        samples_per_ui=4,
    ) == 0.0


# 用附件的直接 np.convolve 与逐符号切片建立独立 oracle，锁定完整轨迹生成路径。
def test_virtual_eye_traces_match_attachment_direct_convolution_oracle() -> None:
    """FFT 优化结果必须逐点等价于附件的冲激列卷积和 2 UI 提取。"""

    # 4 GSa/s 与 M=4 给出 1 GBd，并让手工轨迹只有九列便于审查。
    sample_rate_hz = 4.0e9
    # 十 UI 脉冲长度同时决定附件首尾各舍弃十个符号。
    pulse_length_ui = 10
    # 每 UI 四点使每条轨迹长度严格为 2*4+1=9。
    samples_per_ui = 4
    # 总样点严格由 Np*M 得到。
    samples = pulse_length_ui * samples_per_ui
    # 均匀时间轴提供真实采样率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 非对称多抽头脉冲能同时暴露主光标偏移、窗口端点和卷积方向错误。
    pulse_values = np.zeros(samples, dtype=np.float64)
    # 唯一最大主光标放在第 18 点，故意不与 UI 整数边界对齐。
    main_index = 18
    # 主光标原始幅度为 2，standalone 必须先把它归一化到一。
    pulse_values[main_index] = 2.0
    # 前游标检验卷积方向和 -1 UI 左半窗。
    pulse_values[main_index - 5] = -0.3
    # 非符号间隔后游标检验每采样点轨迹内容。
    pulse_values[main_index + 3] = 0.45
    # 更远后游标检验卷积首尾稳态裁剪。
    pulse_values[main_index + 8] = -0.2
    # 包装单通道拟合脉冲。
    pulse = TimeSeries(
        time_s,
        pulse_values[:, None],
        sample_rate_hz,
    )
    # 800 个固定 NRZ 符号保证每个电平的 1% 分位至少有 100 个稳态样本。
    settings = VirtualEyeSettings(
        modulation="nrz",
        pulse_length_ui=pulse_length_ui,
        samples_per_ui=samples_per_ui,
        symbol_count=800,
        random_seed=42,
    )

    # 产品路径使用缓存冲激 RFFT 与滑窗视图生成轨迹。
    result = build_virtual_eye(pulse, settings)
    # 独立 oracle 只读取固定符号，不复用产品路径的卷积或切片实现。
    cache = attribution_module._prepare_virtual_eye_cache(settings)
    # 附件做法是在每个 UI 起点放置一个符号冲激。
    impulse_train = np.zeros(
        settings.symbol_count * samples_per_ui,
        dtype=np.float64,
    )
    # 固定符号逐 M 点写入冲激列。
    impulse_train[::samples_per_ui] = cache.symbols
    # standalone 按自身主光标 2 V 归一化脉冲。
    normalized_pulse = pulse_values / pulse_values[main_index]
    # 直接 np.convolve(mode="full") 与附件实现完全相同且独立于 SciPy FFT。
    direct_waveform = np.convolve(
        impulse_train,
        normalized_pulse,
        mode="full",
    )
    # 列表逐符号保存附件定义的 -1 UI 到 +1 UI 轨迹。
    direct_traces: list[np.ndarray] = []
    # 首尾各舍弃 Np 个符号，循环边界与附件 pulse_span_ui 定义一致。
    for symbol_index in range(
        pulse_length_ui,
        settings.symbol_count - pulse_length_ui,
    ):
        # 第 k 个符号主光标输出位置等于 k*M 加脉冲主光标索引。
        center = symbol_index * samples_per_ui + main_index
        # 左边界位于主光标前一 UI。
        start = center - samples_per_ui
        # 半开右边界越过主光标后一 UI 一个样点，以包含 +1 UI 端点。
        stop = center + samples_per_ui + 1
        # 直接复制该符号的完整九点轨迹。
        direct_traces.append(direct_waveform[start:stop])
    # 附件只把前 600 条确定性轨迹交给绘图。
    expected_plot_traces = np.asarray(
        direct_traces[:600],
        dtype=np.float64,
    )

    # 产品横轴必须与附件 linspace(-1,+1,2*M+1) 逐点一致。
    np.testing.assert_array_equal(
        result.plot_time_ui,
        np.linspace(-1.0, 1.0, 9, dtype=np.float64),
    )
    # FFT/滑窗优化只允许浮点舍入差，不得改变任何一条轨迹的中心或方向。
    np.testing.assert_allclose(
        result.plot_traces_v,
        expected_plot_traces,
        rtol=0.0,
        atol=1.0e-12,
    )


# 用 Np 不等于 M 的输入证明 PAM4 三眼几何只把 M 当作每 UI 样点数。
def test_pam4_geometry_uses_m_and_attachment_normalized_levels() -> None:
    """Np 与 M 故意不同，防止把脉冲 UI 数误当成每 UI 样点数。"""

    # Np=7、M=8 使长度合同和 UI 几何不能通过参数互换侥幸成立。
    pulse = _ideal_one_ui_pulse(np_ui=7, samples_per_ui=8)
    # PAM4 使用附件确认的 -1/-1/3/+1/3/+1 峰值归一化电平。
    settings = VirtualEyeSettings(
        modulation="pam4",
        pulse_length_ui=7,
        samples_per_ui=8,
        symbol_count=4096,
        random_seed=20260718,
    )

    # 公共接口应从时间轴得到 8 GHz，再仅用 M 推导 1 GBd。
    result = build_virtual_eye(pulse, settings)

    # 三对相邻 PAM4 电平的手算间距都等于 2/3 V。
    assert result.eye_heights_v == pytest.approx(
        (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
        abs=1.0e-12,
    )
    # 理想单 UI 矩形在所有八个相位都开眼，三只眼宽均为 1 UI。
    assert result.eye_widths_ui == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-12)
    # 波特率必须是 Fs/M=1 GHz，若误用 Np 会得到不同结果。
    assert result.baud_rate_hz == pytest.approx(1.0e9, abs=1.0e-6)
    # 绘图轨迹保留附件 PAM4 外轨，界面纵轴必须能看到 ±1 V。
    assert (
        np.min(result.plot_traces_v),
        np.max(result.plot_traces_v),
    ) == pytest.approx(
        (-1.0, 1.0),
        abs=1.0e-12,
    )


# 锁定 standalone 与设备比较的两种归一化语义，防止 DUT 相对增益被各自归一化抹掉。
def test_device_comparison_uses_reference_cursor_as_common_amplitude_scale() -> None:
    """单独查看各自归一化；参考/DUT 比较必须共同除以参考主光标。"""

    # 单位矩形提供主光标位置和无 ISI 轨迹几何。
    unit_pulse = _ideal_one_ui_pulse(np_ui=8, samples_per_ui=8)
    # 参考设备主光标和整个响应放大为 2 V。
    reference_values = np.asarray(unit_pulse.values * 2.0, dtype=np.float64)
    # DUT 保持 1 V，因此真实相对增益只有参考的一半。
    dut_values = np.asarray(unit_pulse.values, dtype=np.float64)
    # 两台设备使用同一时间轴，隔离本测试关注的幅度标尺。
    reference_pulse = TimeSeries(
        unit_pulse.time_s,
        reference_values,
        unit_pulse.sample_rate_hz,
    )
    # DUT 拟合脉冲同样保持 Np*M 合同。
    dut_pulse = TimeSeries(
        unit_pulse.time_s,
        dut_values,
        unit_pulse.sample_rate_hz,
    )
    # 固定一份 NRZ 符号源供 standalone 与设备工作区交叉比较。
    eye_settings = VirtualEyeSettings(
        modulation="nrz",
        pulse_length_ui=8,
        samples_per_ui=8,
        symbol_count=2048,
        random_seed=20260718,
    )

    # standalone 参考按自身 2 V 主光标归一化，理想眼高仍为 2。
    standalone_reference = build_virtual_eye(reference_pulse, eye_settings)
    # standalone DUT 也按自身 1 V 主光标归一化，用于单脉冲形状观察。
    standalone_dut = build_virtual_eye(dut_pulse, eye_settings)

    # 两份 standalone 结果都只表达各自归一化后的脉冲形状。
    assert standalone_reference.eye_heights_v == pytest.approx((2.0,))
    # DUT 单独观察时同样得到单位归一化眼高。
    assert standalone_dut.eye_heights_v == pytest.approx((2.0,))

    # 实际设备比较改用 eye_height，并在零点前的低频范围建立合法候选。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.0,
        scan_high_hz=0.5e9,
        eye=eye_settings,
    )
    # 准备阶段必须以参考 2 V 主光标作为三组轨迹共同幅度标尺。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )

    # 共同幅度基准应保存原始参考主光标 2 V，而不是绝对值归一化后的 1。
    assert workspace.eye_amplitude_normalizer_v == pytest.approx(2.0)
    # 参考归一化后仍为 2 的 NRZ 电平间距。
    assert workspace.reference_metric == pytest.approx(2.0, abs=1.0e-12)
    # DUT 相对参考只有一半增益，所以共同标尺下眼高必须保留为 1。
    assert workspace.before_metric == pytest.approx(1.0, abs=1.0e-12)
    # DUT 轨迹中心外轨同样只到 ±0.5，不能在绘图阶段被重新各自归一化。
    assert workspace.before_eye is not None
    # 共同标尺的二维轨迹提供对指标的独立可视化证据。
    assert np.max(np.abs(workspace.before_eye.plot_traces_v)) == pytest.approx(
        0.5,
        abs=1.0e-12,
    )


# 让前 600 条轨迹与全部稳态轨迹的分位结果不同，锁定绘图截断不污染指标。
def test_eye_metrics_use_all_traces_while_plot_keeps_only_first_600() -> None:
    """眼高统计必须覆盖全部有效轨迹，600 条上限只服务绘图。"""

    # 8 GSa/s 与 M=8 给出一组易于手工放置符号间隔抽头的时轴。
    sample_rate_hz = 8.0e9
    # 二十 UI 记录容纳六个前后游标并留下充足边界。
    pulse_length_ui = 20
    # 每 UI 八点使所有人工抽头都精确落在符号时刻。
    samples_per_ui = 8
    # 总样点仍严格满足 Np*M。
    samples = pulse_length_ui * samples_per_ui
    # 均匀时间轴提供采样率合同。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 主光标和多个符号间隔 ISI 抽头让条件分位数依赖整个固定序列。
    pulse_values = np.zeros(samples, dtype=np.float64)
    # 主光标放在第 64 点并保持单位幅度。
    main_index = 64
    # 单位主光标同时是 standalone 的幅度归一化基准。
    pulse_values[main_index] = 1.0
    # 六个不同权重的符号间隔抽头产生足够丰富的离散 ISI 分布。
    for offset, weight in (
        (-16, 0.12),
        (-8, -0.23),
        (8, 0.31),
        (16, -0.17),
        (24, 0.09),
        (32, -0.05),
    ):
        # 每个抽头按相对主光标样点写入，不引入非整数 UI 插值。
        pulse_values[main_index + offset] = weight
    # 包装单通道拟合脉冲。
    pulse = TimeSeries(
        time_s,
        pulse_values[:, None],
        sample_rate_hz,
    )
    # 2%/98% 分位使前 600 条与全部 4056 条稳态轨迹形成确定性差异。
    settings = VirtualEyeSettings(
        modulation="nrz",
        pulse_length_ui=pulse_length_ui,
        samples_per_ui=samples_per_ui,
        symbol_count=4096,
        random_seed=20260718,
        rail_quantile=0.02,
    )

    # 公共结果应保存全部轨迹计算的眼高，同时仅返回绘图子集。
    result = build_virtual_eye(pulse, settings)
    # 读取同一固定种子缓存，只用于给前 600 条轨迹恢复对应条件符号标签。
    cache = attribution_module._prepare_virtual_eye_cache(settings)

    # 绘图数据严格截到附件允许的 600 条。
    assert result.plot_traces_v.shape == (600, 17)
    # 取前 600 个稳态符号，与返回轨迹逐行对应。
    plot_symbols = cache.stable_symbols[: result.plot_traces_v.shape[0]]
    # 低电平条件轨迹用于取其靠上 98% 边界。
    lower_level_traces = result.plot_traces_v[plot_symbols == -1.0]
    # 高电平条件轨迹用于取其靠下 2% 边界。
    upper_level_traces = result.plot_traces_v[plot_symbols == 1.0]
    # 仅用绘图子集计算 0 UI 的错误眼高，故意构造反例对照。
    plot_only_height_v = float(
        np.quantile(upper_level_traces[:, samples_per_ui], 0.02)
        - np.quantile(lower_level_traces[:, samples_per_ui], 0.98)
    )

    # 全部稳态轨迹的确定性归一化眼高为 0.26。
    assert result.eye_heights_v == pytest.approx((0.26,), abs=1.0e-12)
    # 前 600 条只能得到 0.16 V，证明实现没有用绘图上限替代完整统计。
    assert plot_only_height_v == pytest.approx(0.16, abs=1.0e-12)
    # 两个 oracle 显著分离，能杀死先截 600 条再算指标的实现。
    assert result.eye_heights_v[0] - plot_only_height_v > 0.09


# 防止样本不足的 PAM4 轨道仍输出看似精确的 1%/99% 分位眼高。
def test_pam4_rejects_underpopulated_rail_quantiles() -> None:
    """任一 PAM4 电平样本不足时不得输出缺少统计支撑的 1% 轨道。"""

    # 合法的 7 UI、每 UI 八点脉冲隔离本测试关注的条件样本数量。
    pulse = _ideal_one_ui_pulse(np_ui=7, samples_per_ui=8)
    # 稳态区总共少于 4*100 个符号，因此至少一个电平必然达不到 1% 分位门槛。
    settings = VirtualEyeSettings(
        modulation="pam4",
        pulse_length_ui=7,
        samples_per_ui=8,
        symbol_count=100,
        random_seed=20260718,
    )

    # 明确失败比由单个尾部样点生成一张过度乐观的眼图更可靠。
    with pytest.raises(ValueError, match="每个调制电平至少需要 100"):
        # 失败发生在固定激励准备阶段，不应继续执行卷积与眼图折叠。
        build_virtual_eye(pulse, settings)


# 固定 Np*M 长度合同，避免算法静默裁剪或补零后生成错位眼图。
def test_virtual_eye_rejects_samples_not_equal_to_np_times_m() -> None:
    """长度不匹配必须明确失败，不能静默裁剪成一张似是而非的眼图。"""

    # 构造 56 点有效脉冲，但故意声明 Np*M=48 点。
    pulse = _ideal_one_ui_pulse(np_ui=7, samples_per_ui=8)
    # Np=6 与真实七 UI 记录冲突，M 仍保持正确以隔离长度错误。
    settings = VirtualEyeSettings(
        modulation="nrz",
        pulse_length_ui=6,
        samples_per_ui=8,
        symbol_count=2048,
    )

    # 错误消息包含 Np*M，帮助用户直接定位两个入参。
    with pytest.raises(ValueError, match=r"Np\*M=48"):
        # 调用不得补零、裁剪或自动猜测 Np。
        build_virtual_eye(pulse, settings)


# 验证整个扫描工作区只生成一次固定符号激励，避免候选间统计口径变化。
def test_prepared_eye_workspace_constructs_fixed_stimulus_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参考、DUT 和全部候选必须复用同一份固定激励缓存。"""

    # 短回波输入使三种模式在整个扫描带内均可计算。
    reference_pulse, dut_pulse, settings = _echo_eye_inputs()
    # 保留真实构造函数，窃听器只计数而不改变返回内容。
    original_prepare_cache = attribution_module._prepare_virtual_eye_cache
    # 可变列表允许嵌套窃听函数记录每次调用的设置对象。
    cache_settings: list[VirtualEyeSettings] = []

    # 代理缓存构造入口，仅计数而不改变真实眼图数值。
    def counted_prepare_cache(
        eye_settings: VirtualEyeSettings,
    ) -> object:
        """记录固定激励构造次数后调用真实实现。"""

        # 保留设置身份，可检查没有暗中改随机种子。
        cache_settings.append(eye_settings)
        # 真实构造仍会生成符号、稳态索引和冲激 FFT。
        return original_prepare_cache(eye_settings)

    # 替换模块内部名称，使公共 prepare 的全局查找经过窃听器。
    monkeypatch.setattr(
        attribution_module,
        "_prepare_virtual_eye_cache",
        counted_prepare_cache,
    )
    # 准备阶段同时生成参考眼和补偿前眼。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 两幅基线眼只允许触发一次固定激励构造。
    assert cache_settings == [settings.eye]
    # 扫描全频与所有局部候选。
    scan_frequency_attribution(workspace)
    # 扫描不得重新生成符号序列或冲激 FFT。
    assert cache_settings == [settings.eye]
    # 工作区必须真实持有眼图缓存，而不是仅靠全局状态侥幸复用。
    assert workspace.eye_cache is not None
    # 参考和补偿前 2 UI 横轴共享缓存中同一只读数组。
    assert workspace.reference_eye is not None
    # 补偿前眼同样在眼模式必须存在。
    assert workspace.before_eye is not None
    # 对象身份相同直接证明 -1 UI 到 +1 UI 横轴没有被重复分配。
    assert (
        workspace.reference_eye.plot_time_ui
        is workspace.before_eye.plot_time_ui
    )


# 比较缓存路径和独立完整计算，确保性能优化没有改变眼指标。
def test_cached_eye_results_match_fresh_full_evaluations() -> None:
    """工作区缓存不得改变基线或点选候选的完整数值结果。"""

    # 使用 PAM4 同时覆盖三只眼、固定主光标和 2 UI 轨迹。
    reference_pulse, dut_pulse, settings = _echo_eye_inputs()
    # 工作区基线使用共享缓存构造。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 公共 build_virtual_eye 每次使用一份新鲜局部缓存，可作为非共享对照。
    assert settings.eye is not None
    # 参考对照自行选择主光标并固定在 0 UI。
    fresh_reference = build_virtual_eye(reference_pulse, settings.eye)
    # 工作区应保留完整参考眼。
    assert workspace.reference_eye is not None
    # 三只眼的高度必须与新鲜构造完全一致。
    assert workspace.reference_eye.eye_heights_v == pytest.approx(
        fresh_reference.eye_heights_v,
        abs=1.0e-12,
    )
    # 三只眼的宽度也不得被缓存路径改变。
    assert workspace.reference_eye.eye_widths_ui == pytest.approx(
        fresh_reference.eye_widths_ui,
        abs=1.0e-12,
    )
    # 公共完整轨迹横轴与工作区结果逐点相同。
    np.testing.assert_array_equal(
        workspace.reference_eye.plot_time_ui,
        fresh_reference.plot_time_ui,
    )
    # 频域卷积只允许浮点舍入量级的二维轨迹差异。
    np.testing.assert_allclose(
        workspace.reference_eye.plot_traces_v,
        fresh_reference.plot_traces_v,
        rtol=0.0,
        atol=1.0e-12,
    )
    # 补偿前非共享对照必须使用参考冻结的同一离散相位。
    assert workspace.sampling_phase_index is not None
    # 单独构造 DUT 眼来交叉检查基线缓存。
    fresh_before = build_virtual_eye(
        dut_pulse,
        settings.eye,
        sampling_phase_index=workspace.sampling_phase_index,
    )
    # 工作区中补偿前眼不能为空。
    assert workspace.before_eye is not None
    # 基线高度逐眼对齐。
    assert workspace.before_eye.eye_heights_v == pytest.approx(
        fresh_before.eye_heights_v,
        abs=1.0e-12,
    )
    # 基线宽度逐眼对齐。
    assert workspace.before_eye.eye_widths_ui == pytest.approx(
        fresh_before.eye_widths_ui,
        abs=1.0e-12,
    )
    # 取中间候选避免边界频点，点选公共评估应返回完整数据。
    selected_band = workspace.candidates[len(workspace.candidates) // 2]
    # 联合模式同时覆盖幅度与相位缓存。
    evaluation = evaluate_attribution_band(workspace, selected_band, "both")
    # 点选评估默认必须保留补偿后脉冲。
    assert evaluation.corrected_values is not None
    # 有效眼候选默认必须保留完整眼图。
    assert evaluation.eye_after is not None
    # 用点选波形构造全新 TimeSeries，不复用工作区眼图缓存。
    corrected_pulse = TimeSeries(
        dut_pulse.time_s,
        evaluation.corrected_values,
        dut_pulse.sample_rate_hz,
        source_format="memory",
    )
    # 独立缓存调用显式复用工作区的 DUT 原点与参考幅度基准，形成公平对照。
    assert workspace.eye_cache is not None
    # 共同幅度标尺在眼模式必须存在。
    assert workspace.eye_amplitude_normalizer_v is not None
    # 调用底层确定性内核避免 standalone 的“各自归一化”语义改变设备相对增益。
    fresh_after = attribution_module._build_virtual_eye_from_cache(
        corrected_pulse,
        workspace.eye_cache,
        sampling_phase_index=workspace.sampling_phase_index,
        main_index=workspace.dut_eye_main_index,
        amplitude_normalizer_v=workspace.eye_amplitude_normalizer_v,
        include_plot=True,
    )
    # 点选缓存路径的眼高与独立对照一致。
    assert evaluation.eye_after.eye_heights_v == pytest.approx(
        fresh_after.eye_heights_v,
        abs=1.0e-12,
    )
    # 眼宽也保持相同离散 UI 契约。
    assert evaluation.eye_after.eye_widths_ui == pytest.approx(
        fresh_after.eye_widths_ui,
        abs=1.0e-12,
    )
    # 点选路径仍生成真实非空二维轨迹。
    assert (
        evaluation.eye_after.plot_traces_v.size
        == fresh_after.plot_traces_v.size
    )
    # 二维轨迹数值同样只允许浮点舍入差异。
    np.testing.assert_allclose(
        evaluation.eye_after.plot_traces_v,
        fresh_after.plot_traces_v,
        rtol=0.0,
        atol=1.0e-12,
    )


# 构造峰值移动一采样点的候选，证明补偿后不能偷偷重新选择时序原点。
def test_candidate_reuses_dut_origin_when_peak_moves_one_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """补偿后最大样点移动一格时，候选仍必须使用 DUT 基线原点。"""

    # DUT 基线主冲激在第 64 点，准备阶段应将它冻结。
    reference_pulse, dut_pulse, settings = _echo_eye_inputs()
    # 工作区同时冻结参考相位和 DUT 主脉冲原点。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 反例波形在原点保留 0.9，但相邻一格变成更大的 1.0。
    moved_values = np.zeros_like(dut_pulse.values)
    # 原 DUT 原点仍有主要能量，固定对齐应在此采样。
    moved_values[64, 0] = 0.9
    # 新最大值在第 65 点，若错误重新 argmax 就会得到更好的眼高。
    moved_values[65, 0] = 1.0
    # 只读保护使伪补偿输出与真实内核的返回合同一致。
    moved_values.setflags(write=False)

    # 用确定性移峰波形替换频域回放，隔离本测试关注的原点冻结规则。
    def return_peak_shifted_values(
        prepared: object,
        correction: np.ndarray,
    ) -> np.ndarray:
        """用一样点峰值移动反例替代候选 IFFT 输出。"""

        # 窃听器必须只服务当前测试工作区。
        assert prepared is workspace
        # 补偿数组仍应是有限的一维 RFFT 权重。
        assert np.all(np.isfinite(correction))
        # 返回预构造的单样点移动波形。
        return moved_values

    # 替换候选时域输出，其余频带校验和眼图度量保持真实。
    monkeypatch.setattr(
        attribution_module,
        "_apply_cached_correction",
        return_peak_shifted_values,
    )
    # 中间频带的幅度模式足以进入补偿后眼图测量。
    selected_band = workspace.candidates[len(workspace.candidates) // 2]
    # 公共点选评估必须传递冻结 DUT 原点。
    evaluation = evaluate_attribution_band(workspace, selected_band, "magnitude")
    # 评估应当有完整眼图供绘制。
    assert evaluation.eye_after is not None
    # 冻结原点必须仍是基线的第 64 点。
    assert workspace.dut_eye_main_index == 64
    # 工作区必须持有固定符号缓存以构造独立对照。
    assert workspace.eye_cache is not None
    # 把反例数组包装成拟合脉冲。
    moved_pulse = TimeSeries(
        dut_pulse.time_s,
        moved_values,
        dut_pulse.sample_rate_hz,
        source_format="memory",
    )
    # 固定对照显式使用 DUT 基线原点。
    fixed_origin_eye = attribution_module._build_virtual_eye_from_cache(
        moved_pulse,
        workspace.eye_cache,
        sampling_phase_index=workspace.sampling_phase_index,
        main_index=workspace.dut_eye_main_index,
        amplitude_normalizer_v=workspace.eye_amplitude_normalizer_v,
        include_plot=True,
    )
    # 动态对照使用公共默认 argmax，会被移到第 65 点。
    assert settings.eye is not None
    # 错误的重定位对照仅用于证明反例具有区分力。
    dynamic_origin_eye = build_virtual_eye(
        moved_pulse,
        settings.eye,
        sampling_phase_index=workspace.sampling_phase_index,
    )
    # 实际候选指标必须与固定原点对照相同。
    assert evaluation.attribution.metric_after == pytest.approx(
        min(fixed_origin_eye.eye_heights_v),
        abs=1.0e-12,
    )
    # 峰值归一化 PAM4 下动态重定位会把限制眼高从 0.60 虚增到 2/3。
    assert (
        min(dynamic_origin_eye.eye_heights_v)
        - evaluation.attribution.metric_after
        > 0.06
    )


# 构造中心眼高相同而眼宽不同的反例，确保频段回放真正读取 eye_width 分支。
def test_eye_width_damage_is_recovered_without_substituting_eye_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候选恢复窄脉冲的开口宽度时，中心眼高不应冒充眼宽指标。"""

    # 8 GSa/s 与 M=8 给出 1 GBd，并让眼宽分辨率为 0.125 UI。
    sample_rate_hz = 8.0e9
    # 20 UI 记录提供 50 MHz 原生频率分辨率和充足卷积稳态区。
    pulse_length_ui = 20
    # 每 UI 八点与前述眼宽分辨率一致。
    samples_per_ui = 8
    # Np*M 严格决定拟合脉冲样点数。
    samples = pulse_length_ui * samples_per_ui
    # 均匀时间轴供 TimeSeries 验证实际采样率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 参考矩形持续完整 1 UI，中心眼高为 2、眼宽为 1 UI。
    reference_values = np.zeros(samples, dtype=np.float64)
    # 主光标远离记录边界，便于提取两侧各 1 UI 的完整轨迹。
    main_index = 64
    # 完整八点单位矩形形成 1 UI 正开口。
    reference_values[main_index : main_index + samples_per_ui] = 1.0
    # DUT 主光标幅度仍为一，但矩形只持续半 UI。
    dut_values = np.zeros(samples, dtype=np.float64)
    # 四点单位矩形保持 0 UI 眼高不变，同时把眼宽压缩到 0.5 UI。
    dut_values[main_index : main_index + samples_per_ui // 2] = 1.0
    # 包装参考拟合脉冲。
    reference_pulse = TimeSeries(
        time_s,
        reference_values[:, None],
        sample_rate_hz,
    )
    # 包装仅眼宽受损的 DUT 拟合脉冲。
    dut_pulse = TimeSeries(
        time_s,
        dut_values[:, None],
        sample_rate_hz,
    )
    # 只比较眼宽，使错误的 eye_width->eye_height 替换会把基线差距变成零。
    settings = AttributionSettings(
        metric="eye_width",
        scan_low_hz=0.1e9,
        scan_high_hz=0.4e9,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=pulse_length_ui,
            samples_per_ui=samples_per_ui,
            symbol_count=2048,
            random_seed=20260718,
        ),
    )
    # 工作区用真实 2 UI 条件轨迹建立参考和补偿前指标。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )

    # 两份脉冲在 0 UI 主光标处具有完全相同的 NRZ 眼高。
    assert workspace.reference_eye is not None
    # DUT 基线眼同样必须存在。
    assert workspace.before_eye is not None
    # 中心眼高相等能杀死把 eye_width 偷换成 eye_height 的实现。
    assert workspace.reference_eye.eye_heights_v == pytest.approx(
        workspace.before_eye.eye_heights_v,
        abs=1.0e-12,
    )
    # 正确参考眼宽为完整 1 UI。
    assert workspace.reference_metric == pytest.approx(1.0, abs=1.0e-12)
    # DUT 只保留半 UI 开口。
    assert workspace.before_metric == pytest.approx(0.5, abs=1.0e-12)

    # 用“该候选频段已恢复参考脉冲”的确定性回放隔离指标分支，不依赖另一个滤波器 oracle。
    def return_reference_pulse(
        prepared: object,
        correction: np.ndarray,
    ) -> np.ndarray:
        """把当前候选的补偿后输出固定为已知参考脉冲。"""

        # 回放必须服务本测试工作区，防止替身吞掉错误对象。
        assert prepared is workspace
        # 真实候选仍应先构造有限复频域补偿。
        assert np.all(np.isfinite(correction))
        # 返回只读二维参考值，与真实 IFFT 输出形状一致。
        return reference_pulse.values

    # 只替换候选频域应用结果，其余频带有效性和 2 UI 轨迹测量保持真实。
    monkeypatch.setattr(
        attribution_module,
        "_apply_cached_correction",
        return_reference_pulse,
    )
    # 选择一个不跨矩形频响零点的低频候选。
    selected_band = workspace.candidates[0]
    # 幅度模式触发真实候选校验和补偿后眼图测量。
    evaluation = evaluate_attribution_band(
        workspace,
        selected_band,
        "magnitude",
    )

    # 恢复后的指标必须回到参考 1 UI，而不是仍报告中心眼高 2。
    assert evaluation.attribution.metric_after == pytest.approx(1.0, abs=1.0e-12)
    # 基线 0.5 UI 差距被完整消除，所以改善量必须为 0.5 UI。
    assert evaluation.attribution.improvement == pytest.approx(0.5, abs=1.0e-12)
    # 恢复率为一说明频段回放确实沿眼宽分支闭环。
    assert evaluation.attribution.recovery_ratio == pytest.approx(1.0, abs=1.0e-12)


# 锁定扫描只计算标量且三种模式复用同一频带权重，防止内存和耗时回退。
def test_scan_reuses_band_weights_and_never_builds_eye_plot_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量扫描只保留标量，且同一频带三模式共享一份 Tukey 权重。"""

    # 先完成准备，窃听记录中只包含真正的扫描路径。
    reference_pulse, dut_pulse, settings = _echo_eye_inputs()
    # 工作区已经拥有完整参考与补偿前眼。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 保留真实眼图内核，窃听器仅记录 include_plot 和返回数组大小。
    original_build_from_cache = attribution_module._build_virtual_eye_from_cache
    # 每次眼图测量保存（是否要绘图、横轴大小、纵轴大小）。
    eye_calls: list[tuple[bool, int, int]] = []

    # 记录每次眼图构造是否关闭绘图数组，同时保留真实标量计算。
    def tracked_build_from_cache(
        pulse: TimeSeries,
        cache: object,
        *,
        sampling_phase_index: int | None = None,
        main_index: int | None = None,
        amplitude_normalizer_v: float | None = None,
        include_plot: bool = True,
    ) -> object:
        """调用真实内核后记录本次是否分配了绘图轨迹。"""

        # 真实数值路径保持不变，只插入观测点。
        eye = original_build_from_cache(
            pulse,
            cache,
            sampling_phase_index=sampling_phase_index,
            main_index=main_index,
            amplitude_normalizer_v=amplitude_normalizer_v,
            include_plot=include_plot,
        )
        # 记录绘图开关、固定时间轴大小和二维轨迹实际元素数。
        eye_calls.append(
            (include_plot, eye.plot_time_ui.size, eye.plot_traces_v.size)
        )
        # 原样返回结果供扫描计算标量。
        return eye

    # 保留真实满权核心函数并记录每次的可见物理边界。
    original_core_weights = attribution_module.cosine_core_band_weights
    # 二元组列表用于判定每个频带只构造一次权重。
    weight_calls: list[tuple[float, float]] = []

    # 统计核心权重构造次数，验证同一频段不会为三种模式重复建窗。
    def tracked_core_weights(
        frequency_hz: np.ndarray,
        *,
        core_low_hz: float,
        core_high_hz: float,
        shoulder_hz: float,
        domain_low_hz: float | None = None,
        domain_high_hz: float | None = None,
    ) -> np.ndarray:
        """记录可见核心边界后调用真实余弦肩部权重。"""

        # 每次调用记录一对物理 Hz 边界。
        weight_calls.append((core_low_hz, core_high_hz))
        # 真实核心和肩部值不做任何替换或缓存伪造。
        return original_core_weights(
            frequency_hz,
            core_low_hz=core_low_hz,
            core_high_hz=core_high_hz,
            shoulder_hz=shoulder_hz,
            domain_low_hz=domain_low_hz,
            domain_high_hz=domain_high_hz,
        )

    # 窃听眼图内核以验证扫描传入 include_plot=False。
    monkeypatch.setattr(
        attribution_module,
        "_build_virtual_eye_from_cache",
        tracked_build_from_cache,
    )
    # 窃听满权核心函数以验证三模式复用。
    monkeypatch.setattr(
        attribution_module,
        "cosine_core_band_weights",
        tracked_core_weights,
    )
    # 执行全频闭环和所有局部频段的三模式扫描。
    result = scan_frequency_attribution(workspace)
    # 每个全频或局部频带只应构造一份权重，而不是三份。
    assert len(weight_calls) == 1 + len(workspace.candidates)
    # 该合成输入下每个频带的三种模式都可进入眼图标量测量。
    assert len(eye_calls) == 3 * (1 + len(workspace.candidates))
    # 扫描调用全部关闭轨迹保留；小型 2*M+1 时间轴仍可共享。
    assert all(call == (False, 17, 0) for call in eye_calls)
    # 最终扫描结果仅积累三模式的轻量 BandAttribution。
    assert len(result.candidates) == 3 * len(workspace.candidates)
    # 标量摘要不暴露或持有眼图大数组字段。
    assert all(not hasattr(candidate, "eye_after") for candidate in result.candidates)


# 确认每条 PAM4 轨道用一次分位调用同时取得上下边界。
def test_virtual_eye_combines_lower_and_upper_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PAM4 每条轨道应以一次 quantile 同时取得上下边界。"""

    # 理想 PAM4 脉冲使分位调用数与电平数一一对应。
    pulse = _ideal_one_ui_pulse(np_ui=7, samples_per_ui=8)
    # 四电平设置应产生四次合并分位调用。
    settings = VirtualEyeSettings(
        modulation="pam4",
        pulse_length_ui=7,
        samples_per_ui=8,
        symbol_count=700,
        random_seed=20260718,
    )
    # 保留真实 NumPy quantile 函数以维持独立数值结果。
    original_quantile = attribution_module.np.quantile
    # 列表保存每次请求的 q 数组。
    requested_quantiles: list[np.ndarray] = []

    # 窃听分位参数而不修改 NumPy 的真实排序和插值结果。
    def tracked_quantile(
        values: np.ndarray,
        q: object,
        *,
        axis: int,
    ) -> np.ndarray:
        """记录 q 后调用真实 NumPy 分位实现。"""

        # 复制 q 防止后续调用修改观测记录。
        requested_quantiles.append(np.asarray(q, dtype=np.float64).copy())
        # 保持原数据、轴和分位数语义。
        return original_quantile(values, q, axis=axis)

    # 替换模块引用的 NumPy quantile 观察实际调用。
    monkeypatch.setattr(attribution_module.np, "quantile", tracked_quantile)
    # 公共调用仍应生成完整 PAM4 眼图。
    result = build_virtual_eye(pulse, settings)
    # 四条电平轨道各调用一次，不再分别计算上下分位。
    assert len(requested_quantiles) == 4
    # 每次都同时请求 q 和 1-q 两个边界。
    assert all(
        quantiles == pytest.approx(
            np.array([settings.rail_quantile, 1.0 - settings.rail_quantile]),
            abs=0.0,
        )
        for quantiles in requested_quantiles
    )
    # 三只眼的指标仍完整存在。
    assert len(result.eye_heights_v) == 3


# 保留底层通用 Tukey 权重的闭式谕示，避免其它调用方行为被归因窗改动破坏。
def test_tukey_half_band_has_hand_calculable_weights() -> None:
    """alpha=0.5 的九点平滑窗应有独立闭式 oracle。"""

    # 九个等间隔频点覆盖 0 到 800 MHz，便于逐点手算余弦边沿。
    frequency_hz = np.arange(9, dtype=np.float64) * 100.0e6

    # 完整频带宽 800 MHz，其中每侧 200 MHz 用于半余弦过渡。
    weights = tukey_band_weights(
        frequency_hz,
        low_hz=0.0,
        high_hz=800.0e6,
        alpha=0.5,
    )

    # 标准 Tukey(alpha=0.5) 的九点闭式结果可直接手算。
    assert weights == pytest.approx(
        np.array([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]),
        abs=1.0e-12,
    )


# 用多组物理分辨率验证核心宽度只向上取整，不宣称零填充带来的伪精度。
@pytest.mark.parametrize(
    ("physical_resolution_hz", "expected_width_hz", "expects_warning"),
    [
        # 80 MHz 原生分辨率足以保留用户请求的 100 MHz 窗宽。
        (80.0e6, 100.0e6, False),
        # 265.625 MHz 原生分辨率必须向上取整到 300 MHz，而非靠零填充伪装。
        (265.625e6, 300.0e6, True),
    ],
)
def test_candidate_bands_respect_physical_resolution(
    physical_resolution_hz: float,
    expected_width_hz: float,
    expects_warning: bool,
) -> None:
    """100 MHz 是扫描步进；不可分辨时只扩大窗宽并明确告警。"""

    # 扫描 0–1 GHz，步进和请求窗宽都锁定为用户确认的 100 MHz。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=7,
            samples_per_ui=8,
            symbol_count=1024,
        ),
    )

    # 候选生成只依赖物理分辨率，不允许被显示 FFT 点数左右。
    bands, effective_width_hz, warnings = candidate_frequency_bands(
        settings,
        physical_resolution_hz=physical_resolution_hz,
    )

    # 有效窗宽必须等于独立手算结果。
    assert effective_width_hz == pytest.approx(expected_width_hz, abs=1.0e-6)
    # 所有相邻中心仍保持 100 MHz，扩大的是可解释窗宽而非扫描粒度。
    assert np.diff([band.center_hz for band in bands]) == pytest.approx(
        100.0e6,
        abs=1.0e-6,
    )
    # 只有物理分辨率不足时才出现降级告警。
    assert bool(warnings) is expects_warning


# 验证纯幅度与纯相位干预正交，联合模式才同时应用两项频响差。
def test_complex_intervention_keeps_magnitude_and_phase_orthogonal() -> None:
    """半权重闭式值能杀死幅相分支互换、相位符号和线性插幅错误。"""

    # Href/Hdut 的幅度为 2，所以对数幅度差为 ln(2)。
    log_magnitude_ratio = np.array([np.log(2.0)], dtype=np.float64)
    # Href 相位 0.3、Hdut 相位 -0.2，正确相位差为 +0.5 rad。
    phase_ratio_rad = np.array([0.5], dtype=np.float64)
    # 频段平滑边沿取半权重，期望在复对数域中插值。
    weights = np.array([0.5], dtype=np.float64)

    # 纯幅度只能改变模长，不能偷偷修正 DUT 的相位。
    magnitude = compose_frequency_correction(
        log_magnitude_ratio,
        phase_ratio_rad,
        weights,
        mode="magnitude",
    )
    # 纯相位只能旋转 +0.25 rad，模长必须严格保持一。
    phase = compose_frequency_correction(
        log_magnitude_ratio,
        phase_ratio_rad,
        weights,
        mode="phase",
    )
    # 幅相共同补偿应同时取得 sqrt(2) 模长和 +0.25 rad 相位。
    both = compose_frequency_correction(
        log_magnitude_ratio,
        phase_ratio_rad,
        weights,
        mode="both",
    )

    # 对数域半插值得到 sqrt(2)，不是错误的线性幅度 1.5。
    assert magnitude == pytest.approx(
        np.array([np.sqrt(2.0) + 0.0j]),
        abs=1.0e-12,
    )
    # 相位分支的解析期望为 exp(+j*0.25)。
    assert phase == pytest.approx(np.exp(0.25j)[None], abs=1.0e-12)
    # 联合分支是前两者逐频点相乘。
    assert both == pytest.approx(
        np.sqrt(2.0) * np.exp(0.25j)[None],
        abs=1.0e-12,
    )


# 注入已知幅度缺口，锁定频段定位、模式识别和补偿方向。
def test_known_amplitude_notch_localizes_band_and_mode() -> None:
    """线性相位浅陷波只应由幅度补偿恢复，推荐频段必须覆盖注入真值。"""

    # 8 GSa/s、M=8 对应 1 GBd；800 点拟合脉冲给出 10 MHz 物理分辨率。
    sample_rate_hz = 8.0e9
    # M 决定 UI 几何，并与 Np=100 形成 800 点长度合同。
    samples_per_ui = 8
    # 足够长的脉冲记录容纳 401 抽头对称 FIR，避免截断破坏线性相位。
    np_ui = 100
    # 总样点严格等于 Np*M。
    samples = np_ui * samples_per_ui
    # 时间轴由算法反算采样率，不额外传入符号率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 401 抽头带通核只定义 1.0–1.2 GHz 的局部幅度分量。
    bandpass = signal.firwin(
        401,
        [1.0e9, 1.2e9],
        pass_zero=False,
        fs=sample_rate_hz,
        window=("kaiser", 8.0),
    )
    # 参考使用与 FIR 群时延相同的单位冲激，确保二者相位趋势完全相同。
    reference_values = np.zeros(samples, dtype=np.float64)
    # 401 抽头对称 FIR 的群时延为 200 个样点。
    reference_values[200] = 1.0
    # DUT 从单位冲激中减去半幅带通核，频带内保持正的浅幅度缺口。
    dut_values = reference_values.copy()
    # 浅缺口不会穿过零点，因此不引入 pi 相位翻转。
    dut_values[:401] -= 0.5 * bandpass
    # 两份拟合脉冲共享同一时轴和采样率。
    reference_pulse = TimeSeries(
        time_s,
        reference_values[:, None],
        sample_rate_hz,
    )
    # DUT 仅幅度频响不同。
    dut_pulse = TimeSeries(time_s, dut_values[:, None], sample_rate_hz)
    # 扫描范围在缺口两侧各保留足够背景候选。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.5e9,
        scan_high_hz=1.6e9,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=np_ui,
            samples_per_ui=samples_per_ui,
            symbol_count=1000,
            random_seed=20260718,
        ),
    )

    # 准备阶段只计算一次目标 DFT 网格频响与固定符号基线。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 扫描三种模式和全部 100 MHz 候选。
    result = scan_frequency_attribution(workspace)

    # 局部直接回放存在有效正改善，因此必须给出推荐。
    assert result.status == "ok"
    # 推荐对象不应为空。
    assert result.recommendation is not None
    # 纯幅度 oracle 的模式必须是 magnitude；幅相数值并列时优先简单模式。
    assert result.recommendation.mode == "magnitude"
    # 推荐 100 MHz 窗必须与真实 1.0–1.2 GHz 缺口相交。
    assert result.recommendation.band.high_hz > 1.0e9
    # 推荐下边界同样必须低于缺口上限。
    assert result.recommendation.band.low_hz < 1.2e9
    # 幅度模式应产生远大于浮点误差的正改善。
    assert result.recommendation.improvement > 1.0e-3
    # 所有纯相位候选都应停留在独立数值 oracle 的近零范围。
    phase_scores = [
        candidate.improvement
        for candidate in result.candidates
        if candidate.valid and candidate.mode == "phase"
    ]
    # 该上限远小于真实幅度改善，能杀死模式分支互换。
    assert max(abs(score) for score in phase_scores) < 1.0e-10
    # 按推荐频段重算一次大型输出，供三幅眼图展示。
    evaluation = evaluate_attribution_band(
        workspace,
        result.recommendation.band,
        result.recommendation.mode,
    )
    # 三联轨迹必须来自真实候选结果而非占位数组。
    comparison = build_eye_comparison(workspace, evaluation)
    # 三幅图共用完全相同的 2*M+1 轨迹长度。
    assert (
        comparison.reference_traces_v.shape
        == comparison.before_traces_v.shape
        == comparison.after_traces_v.shape
    )
    # 公共横轴必须覆盖附件定义的 -1 UI 到 +1 UI。
    assert comparison.time_ui[[0, -1]] == pytest.approx((-1.0, 1.0))
    # 共同纵轴下限必须覆盖三组轨迹的实际最小值。
    assert comparison.amplitude_range_v[0] < min(
        float(np.min(comparison.reference_traces_v)),
        float(np.min(comparison.before_traces_v)),
        float(np.min(comparison.after_traces_v)),
    )
    # 共同纵轴上限同样必须覆盖三组轨迹的实际最大值。
    assert comparison.amplitude_range_v[1] > max(
        float(np.max(comparison.reference_traces_v)),
        float(np.max(comparison.before_traces_v)),
        float(np.max(comparison.after_traces_v)),
    )


# 相同脉冲必须在任何候选 IFFT 前早退，不能从浮点噪声中强行推荐。
def test_identical_pulses_do_not_force_a_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """指标差距只在机器误差内时应保守返回无差距。"""

    # Np=7、M=8 保持参数不同并得到 56 点短基准。
    pulse = _ideal_one_ui_pulse(np_ui=7, samples_per_ui=8)
    # 扫描范围避开 Nyquist，同时让物理分辨率扩大候选窗宽。
    settings = AttributionSettings(
        metric="eye_width",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=7,
            samples_per_ui=8,
            symbol_count=600,
        ),
    )

    # 参考与 DUT 传入同一理想脉冲，真实指标差距严格为零。
    workspace = prepare_frequency_attribution(pulse, pulse, settings)

    # 若零差距路径误入候选回放，此替身立即让测试失败。
    def reject_unnecessary_ifft(
        prepared: object,
        correction: np.ndarray,
    ) -> np.ndarray:
        """若无差距扫描仍进入候选 IFFT，立即使测试失败。"""

        # 参数仅为符合内部函数签名，早退正确时永远不应读取。
        del prepared, correction
        # 明确指出误触发的是无差距下的频域应用。
        raise AssertionError("无差距时不应执行任何候选 IFFT")

    # 在扫描前替换 IFFT 入口，用来证明早退位于所有反事实之前。
    monkeypatch.setattr(
        attribution_module,
        "_apply_cached_correction",
        reject_unnecessary_ifft,
    )
    # 即使扫描器能计算每个候选，也不能从舍入噪声中强挑第一名。
    result = scan_frequency_attribution(workspace)

    # 状态明确区分“没有差距”和“模型无法闭环”。
    assert result.status == "no_difference"
    # 无差距时推荐必须为空。
    assert result.recommendation is None
    # 早退不伪造全频证据。
    assert result.full_band_results == ()
    # 局部候选同样为空，表明没有执行任何 IFFT。
    assert result.candidates == ()


# 全频结果不能推出局部结果，三种全频不改善时仍须直接评估各局部候选。
def test_scan_continues_when_full_band_model_does_not_improve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全频补偿不改善时仍应保留完整局部影响曲线。"""

    # 回波脉冲在 0.2–1.0 GHz 局部带内的全频反事实已知会进一步恶化眼高。
    reference_pulse, dut_pulse, broad_settings = _echo_eye_inputs()
    # 眼图几何和固定种子保持不变，只缩小扫描带。
    assert broad_settings.eye is not None
    # 窄带设置使频响模型无法闭环完整的时域回波。
    unsupported_settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.2e9,
        scan_high_hz=1.0e9,
        eye=broad_settings.eye,
    )
    # 准备后基线仍有明显差距，因此不会命中无差距早退。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        unsupported_settings,
    )
    # 保留真实时域补偿函数，窃听器只统计 IFFT 次数。
    original_apply = attribution_module._apply_cached_correction
    # 每次频域应用记录其复补偿长度。
    correction_sizes: list[int] = []

    # 计数频域应用次数，用于证明早退发生在全部局部候选之前。
    def counted_apply(
        prepared: object,
        correction: np.ndarray,
    ) -> np.ndarray:
        """记录一次 IFFT 候选后调用真实实现。"""

        # 记录补偿频点数，便于确认正确进入了频域路径。
        correction_sizes.append(int(correction.size))
        # 真实 IFFT 保持全频诊断和局部候选的数值路径不变。
        return original_apply(prepared, correction)

    # 替换时域应用入口以区分三次全频与后续局部调用。
    monkeypatch.setattr(
        attribution_module,
        "_apply_cached_correction",
        counted_apply,
    )
    # 扫描必须继续到全部局部频道，不能用全频非线性指标替代局部反事实。
    result = scan_frequency_attribution(workspace)
    # 幅度、相位、幅相三份全频证据都必须保留。
    assert len(result.full_band_results) == 3
    # 每个核心都必须产生幅度、相位、幅相三份直接回放证据。
    assert len(result.candidates) == 3 * len(workspace.candidates)
    # 三次全频加全部局部评估构成完整 IFFT 次数。
    assert len(correction_sizes) == 3 * (1 + len(workspace.candidates))
    # 每份补偿都应覆盖工作区的完整 RFFT 轴。
    assert correction_sizes == [workspace.frequency_hz.size] * len(correction_sizes)


# 用不等长、不等采样率原始波形验证 Vpp 扫描仍能找到边界单音频段。
def test_vpp_scan_uses_unequal_raw_waveforms_and_finds_tone_band() -> None:
    """Vpp 必须来自两条原始波形，且不同采样率、长度不妨碍频段定位。"""

    # 拟合脉冲使用 8 GSa/s、80 点，对应恰好 100 MHz 物理分辨率。
    pulse_rate_hz = 8.0e9
    # 两份拟合脉冲等长等采样率，满足扫描器共通输入合同。
    pulse_samples = 80
    # 均匀脉冲时间轴由 TimeSeries 独立验证。
    pulse_time_s = np.arange(pulse_samples, dtype=np.float64) / pulse_rate_hz
    # 参考冲激幅度为 2，形成全频 Href/Hdut=2 的可手算幅度比。
    reference_pulse_values = np.zeros(pulse_samples, dtype=np.float64)
    # 冲激远离边界，镜像补偿不依赖首点特殊情况。
    reference_pulse_values[40] = 2.0
    # DUT 冲激同位置、半幅度，因此没有相位差。
    dut_pulse_values = np.zeros(pulse_samples, dtype=np.float64)
    # Href/Hdut 的幅度精确为 2。
    dut_pulse_values[40] = 1.0
    # 包装参考拟合脉冲。
    reference_pulse = TimeSeries(
        pulse_time_s,
        reference_pulse_values[:, None],
        pulse_rate_hz,
    )
    # 包装 DUT 拟合脉冲。
    dut_pulse = TimeSeries(
        pulse_time_s,
        dut_pulse_values[:, None],
        pulse_rate_hz,
    )
    # 参考原始波形采用 8 GSa/s、1 us，共 8000 点。
    reference_rate_hz = 8.0e9
    # 有效时长由样点数/Fs决定，为精确 1 us。
    reference_samples = 8000
    # 参考时轴无需和 DUT 逐点对齐。
    reference_time_s = np.arange(reference_samples, dtype=np.float64) / reference_rate_hz
    # 单音故意位于两个 100 MHz 核心的公共 0.6 GHz 边界。
    tone_frequency_hz = 0.6e9
    # 参考单音幅度为 2。
    reference_values = 2.0 * np.sin(
        2.0 * np.pi * tone_frequency_hz * reference_time_s
    )
    # DUT 原始波形故意改为 10 GSa/s、2 us，共 20000 点。
    dut_rate_hz = 10.0e9
    # 两倍时长会在 Vpp 比较器中切成两个公共时长块并取中位数。
    dut_samples = 20000
    # DUT 使用自身独立采样网格。
    dut_time_s = np.arange(dut_samples, dtype=np.float64) / dut_rate_hz
    # DUT 单音幅度为 1，稳健 Vpp 约为参考一半。
    dut_values = np.sin(2.0 * np.pi * tone_frequency_hz * dut_time_s)
    # 参考原始数据与拟合脉冲完全分离。
    reference_waveform = TimeSeries(
        reference_time_s,
        reference_values[:, None],
        reference_rate_hz,
    )
    # DUT 原始数据长度和采样率均与参考不同。
    dut_waveform = TimeSeries(
        dut_time_s,
        dut_values[:, None],
        dut_rate_hz,
    )
    # 扫描 0.2–0.9 GHz，0.6 GHz 正好是两个相邻候选的公共边界。
    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.2e9,
        scan_high_hz=0.9e9,
    )

    # Vpp 准备阶段接受不同原始波形网格，但仍要求拟合脉冲严格匹配。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
        reference_waveform=reference_waveform,
        dut_waveform=dut_waveform,
    )
    # 扫描使用补偿前后原始 DUT 波形计算 Vpp，不从脉冲峰峰值替代。
    result = scan_frequency_attribution(workspace)

    # 原始参考幅度为 DUT 两倍，所以参考稳健 Vpp 必须更大。
    assert result.reference_metric > 1.9 * result.before_metric
    # 单频幅度差能够由局部频段解释，必须得到推荐。
    assert result.status == "ok"
    # 推荐存在且模式为纯幅度。
    assert result.recommendation is not None
    # 相位严格相同，容差内并列应优先 magnitude。
    assert result.recommendation.mode == "magnitude"
    # 推荐核心必须包含 0.6 GHz 单音；边界包含关系同样算覆盖。
    assert result.recommendation.band.low_hz <= tone_frequency_hz
    # 上边界也允许恰好等于公共边界。
    assert result.recommendation.band.high_hz >= tone_frequency_hz
    # 局部补偿应显著缩小 Vpp 差距。
    assert result.recommendation.recovery_ratio > 0.9


# 注入已知相位隆起，验证相位符号、去斜和纯相位模式定位。
def test_known_phase_bump_localizes_band_and_phase_mode() -> None:
    """单位幅度的局部相位隆起应由相位模式定位，防止模式行交换。"""

    # 与幅度 oracle 使用相同 8 GSa/s、800 点几何，保持 10 MHz 物理分辨率。
    sample_rate_hz = 8.0e9
    # M=8 决定 2 UI 轨迹的离散相位分辨率。
    samples_per_ui = 8
    # Np=100 与 M 形成 800 点脉冲合同。
    np_ui = 100
    # 总点数由两个用户入参直接决定。
    samples = np_ui * samples_per_ui
    # 时间轴提供采样率物理语义。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 原始 N 点 DFT 网格只用于构造独立合成脉冲，扫描仍在目标网格直接求 DTFT。
    frequency_hz = np.fft.rfftfreq(samples, d=1.0 / sample_rate_hz)
    # 相位默认为零，对应单位幅度单位相位响应。
    phase_bump_rad = np.zeros(frequency_hz.shape, dtype=np.float64)
    # 1.0–1.1 GHz 线性上升到 0.8 rad。
    rising = (frequency_hz >= 1.0e9) & (frequency_hz <= 1.1e9)
    # 三角隆起的左半边有闭式线性表达。
    phase_bump_rad[rising] = (
        0.8 * (frequency_hz[rising] - 1.0e9) / 0.1e9
    )
    # 1.1–1.2 GHz 对称下降回零。
    falling = (frequency_hz > 1.1e9) & (frequency_hz <= 1.2e9)
    # 右半边保持连续且端点相位为零。
    phase_bump_rad[falling] = (
        0.8 * (1.2e9 - frequency_hz[falling]) / 0.1e9
    )
    # DUT 频响模长严格为一，仅施加负相位隆起。
    zero_delay_dut = np.fft.irfft(np.exp(-1j * phase_bump_rad), n=samples)
    # 将周期冲激响应整体移到记录内部，参考使用同一整体时延。
    center_index = 300
    # DUT 有限记录保留完整 N 点周期冲激响应。
    dut_values = np.roll(zero_delay_dut, center_index)
    # 参考是同位置单位冲激。
    reference_values = np.zeros(samples, dtype=np.float64)
    # 共同整体时延会由相位去斜消除，不应被定位为局部损伤。
    reference_values[center_index] = 1.0
    # 构造参考拟合脉冲。
    reference_pulse = TimeSeries(
        time_s,
        reference_values[:, None],
        sample_rate_hz,
    )
    # 构造 DUT 拟合脉冲。
    dut_pulse = TimeSeries(time_s, dut_values[:, None], sample_rate_hz)
    # 扫描范围包围真实 1.0–1.2 GHz 相位隆起。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.5e9,
        scan_high_hz=1.6e9,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=np_ui,
            samples_per_ui=samples_per_ui,
            symbol_count=1000,
            random_seed=20260718,
        ),
    )

    # 准备连续相位分支并移除共同整体时延。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 执行三模式局部反事实扫描。
    result = scan_frequency_attribution(workspace)

    # 局部相位补偿可闭环，必须给出推荐。
    assert result.status == "ok"
    # 推荐对象存在。
    assert result.recommendation is not None
    # 纯相位 oracle 必须识别为 phase，而非 magnitude 或 both。
    assert result.recommendation.mode == "phase"
    # 推荐窗与真实相位隆起相交。
    assert result.recommendation.band.high_hz > 1.0e9
    # 推荐窗下边界必须低于 1.2 GHz 真值上限。
    assert result.recommendation.band.low_hz < 1.2e9
    # 相位候选产生可观的眼高改善。
    assert result.recommendation.improvement > 1.0e-2
    # 纯幅度分支不应接近相位分支的真实改善量级。
    magnitude_scores = [
        candidate.improvement
        for candidate in result.candidates
        if candidate.valid and candidate.mode == "magnitude"
    ]
    # 离散构造会有微小离网格幅度纹波，但最大影响应保持低于 1e-3。
    assert max(abs(score) for score in magnitude_scores) < 1.0e-3


# 用闭式频点锁定 100 MHz 满权核心及两侧 50 MHz 半余弦肩部。
def test_cosine_shoulders_leave_the_visible_100_mhz_core_at_full_weight() -> None:
    """100 MHz 可见核心必须满权，两侧 50 MHz 肩部平滑降到零。"""

    # 七个闭式采样点覆盖支撑端点、肩部中点和完整核心。
    frequency_hz = np.array(
        [-50.0, -25.0, 0.0, 50.0, 100.0, 125.0, 150.0],
        dtype=np.float64,
    ) * 1.0e6
    # 可见核心为 0–100 MHz，平滑肩部各为 50 MHz。
    weights = cosine_core_band_weights(
        frequency_hz,
        core_low_hz=0.0,
        core_high_hz=100.0e6,
        shoulder_hz=50.0e6,
    )
    # 半余弦肩部中点为 0.5，整个核心包含两端都为 1。
    np.testing.assert_allclose(
        weights,
        np.array([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0]),
        atol=1.0e-15,
    )


# 覆盖全部公共边界和非整数尾段，防止频道之间或 scan_high 前出现盲区。
def test_candidate_cores_cover_boundaries_and_anchor_noninteger_scan_tail() -> None:
    """相邻 100 MHz 核心无扫描缝，非整数尾部也有完整频段覆盖。"""

    # 0–1.05 GHz 故意留出 50 MHz 尾部，能暴露简单 floor 候选数的盲区。
    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=1.05e9,
        detrend_phase=False,
    )
    # 80 MHz 物理分辨率不需要扩大用户要求的 100 MHz 核心。
    bands, effective_width_hz, _warnings = candidate_frequency_bands(
        settings,
        physical_resolution_hz=80.0e6,
    )
    # 页面列表显示的每个核心仍严格为 100 MHz。
    assert all(
        band.high_hz - band.low_hz == pytest.approx(100.0e6)
        for band in bands
    )
    # 结果模型记录核心宽度，不把两侧肩部误算进来。
    assert effective_width_hz == pytest.approx(100.0e6)
    # 最后一个核心必须锚定到真实扫描上限。
    assert bands[-1].high_hz == pytest.approx(1.05e9)
    # 密集频率轴包含所有 100 MHz 交界点和 1.05 GHz 尾点。
    dense_frequency_hz = np.linspace(0.0, 1.05e9, 1051, dtype=np.float64)
    # 各候选核心加余弦肩部后建立一个权重银行。
    weight_bank = np.stack(
        [
            cosine_core_band_weights(
                dense_frequency_hz,
                core_low_hz=band.low_hz,
                core_high_hz=band.high_hz,
                shoulder_hz=50.0e6,
                domain_low_hz=settings.scan_low_hz,
                domain_high_hz=settings.scan_high_hz,
            )
            for band in bands
        ],
        axis=0,
    )
    # 任意扫描频率都至少落在一个满权核心内，不会只吃到趋零肩部。
    np.testing.assert_allclose(np.max(weight_bank, axis=0), 1.0, atol=1.0e-15)


# 区分参考零点的幅度极限与不可辨识相位，防止三种模式共用错误掩码。
def test_reference_spectral_zero_is_valid_for_magnitude_but_not_phase() -> None:
    """Href=0、Hdut!=0 是合法零幅度比，但不具有可解析相位。"""

    # 8 GSa/s 下间隔两点的同号双抽头在 2 GHz 有精确零点。
    sample_rate_hz = 8.0e9
    # 80 点拟合脉冲提供恰好 100 MHz 物理分辨率。
    pulse_samples = 80
    # 两份拟合脉冲共享严格相同的时间轴。
    pulse_time_s = np.arange(pulse_samples, dtype=np.float64) / sample_rate_hz
    # 参考的两个同号抽头相差两个样点。
    reference_values = np.zeros(pulse_samples, dtype=np.float64)
    # 第一抽头放在记录内部。
    reference_values[30] = 1.0
    # 第二抽头使 2 GHz 处恰好相消。
    reference_values[32] = 1.0
    # DUT 使用单冲激，在 2 GHz 的分母严格非零。
    dut_values = np.zeros(pulse_samples, dtype=np.float64)
    # 与参考首抽头共享时间原点。
    dut_values[30] = 1.0
    # 包装参考拟合脉冲。
    reference_pulse = TimeSeries(
        pulse_time_s,
        reference_values[:, None],
        sample_rate_hz,
    )
    # DUT 与参考长度、采样率相同。
    dut_pulse = TimeSeries(
        pulse_time_s,
        dut_values[:, None],
        sample_rate_hz,
    )
    # 402 点目标经镜像延拓后为 1204 点，2 GHz 恰在 RFFT 网格。
    waveform_samples = 402
    # 原始波形时间轴继续使用 8 GSa/s。
    waveform_time_s = np.arange(waveform_samples, dtype=np.float64) / sample_rate_hz
    # 一条有限余弦足以检查零幅度补偿后的 IFFT 有限性。
    waveform_values = np.cos(2.0 * np.pi * 1.0e9 * waveform_time_s)
    # 参考和 DUT 原始波形在本反例中只提供 Vpp 度量合同。
    waveform = TimeSeries(
        waveform_time_s,
        waveform_values[:, None],
        sample_rate_hz,
    )
    # 单个 100 MHz 核心把 2 GHz 零点放在正中。
    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=1.95e9,
        scan_high_hz=2.05e9,
        detrend_phase=False,
    )
    # 准备时纯幅度只应要求 DUT 分母可逆。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
        reference_waveform=waveform,
        dut_waveform=waveform,
    )
    # 唯一候选核心包含精确参考零点。
    band = workspace.candidates[0]
    # 纯幅度可把该频点抑制为零。
    magnitude = evaluate_attribution_band(workspace, band, "magnitude")
    # 合法零比不能被标为不可解析。
    assert magnitude.attribution.valid
    # 时域回放必须存在且保持有限。
    assert magnitude.corrected_values is not None
    # 零幅度补偿不能在 IFFT 中生成 NaN/Inf。
    assert np.all(np.isfinite(magnitude.corrected_values))
    # 参考零点相位无定义，纯相位应保守无效。
    assert not evaluate_attribution_band(workspace, band, "phase").attribution.valid
    # 幅相联合同样不能绕过相位不可解析边界。
    assert not evaluate_attribution_band(workspace, band, "both").attribution.valid


# 手动 Vpp 频带也必须受参考原始波形 Nyquist 限制，而非只检查 DUT。
def test_vpp_scan_uses_reference_waveform_nyquist_in_manual_mode() -> None:
    """手动 Vpp 频带也必须受参考原始波形的 Nyquist 限制。"""

    # 拟合脉冲使用 8 GSa/s，本身不是限制方。
    pulse_rate_hz = 8.0e9
    # 80 点拟合脉冲满足 100 MHz 分辨率。
    pulse_time_s = np.arange(80, dtype=np.float64) / pulse_rate_hz
    # 单冲激避免频响零点干扰 Nyquist 反例。
    pulse_values = np.zeros(80, dtype=np.float64)
    # 冲激放在记录内部。
    pulse_values[40] = 1.0
    # 参考和 DUT 拟合脉冲可完全相同。
    pulse = TimeSeries(pulse_time_s, pulse_values[:, None], pulse_rate_hz)
    # 参考原始波形只有 1 GSa/s，Nyquist 为 0.5 GHz。
    reference_rate_hz = 1.0e9
    # 至少 64 点保证时间轴和 Vpp 有效。
    reference_time_s = np.arange(64, dtype=np.float64) / reference_rate_hz
    # 简单正弦值不影响本测试的频带边界。
    reference_waveform = TimeSeries(
        reference_time_s,
        np.sin(2.0 * np.pi * 0.1e9 * reference_time_s)[:, None],
        reference_rate_hz,
    )
    # DUT 原始波形使用 4 GSa/s，不会触发 0.8 GHz 上限。
    dut_rate_hz = 4.0e9
    # DUT 时间轴长度独立。
    dut_time_s = np.arange(128, dtype=np.float64) / dut_rate_hz
    # DUT 波形也是有限单通道记录。
    dut_waveform = TimeSeries(
        dut_time_s,
        np.sin(2.0 * np.pi * 0.1e9 * dut_time_s)[:, None],
        dut_rate_hz,
    )
    # 0.6–0.8 GHz 对脉冲和 DUT 合法，却越过参考原始波形 Nyquist。
    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.6e9,
        scan_high_hz=0.8e9,
        detrend_phase=False,
    )
    # 准备阶段必须在做任何 FFT 前拒绝超限手动频带。
    with pytest.raises(ValueError, match="公共 Nyquist"):
        # 两份原始数据的较低 Nyquist 才是 Vpp 公共上限。
        prepare_frequency_attribution(
            pulse,
            pulse,
            settings,
            reference_waveform=reference_waveform,
            dut_waveform=dut_waveform,
        )


# 短 DUT 原始记录不能借用长拟合脉冲的分辨率宣称 100 MHz 定位能力。
def test_vpp_target_record_resolution_expands_candidate_core() -> None:
    """Vpp 候选核心还必须受实际被补偿原始记录的 Fs/N 限制。"""

    # 八十点、8 GSa/s 的两份拟合脉冲具有 100 MHz 独立分辨率。
    pulse_rate_hz = 8.0e9
    # 时间轴严格等间隔并覆盖十纳秒。
    pulse_time_s = np.arange(80, dtype=np.float64) / pulse_rate_hz
    # 单冲激避免频响零点影响本测试的窗口几何。
    pulse_values = np.zeros(80, dtype=np.float64)
    # 冲激位于记录内部。
    pulse_values[40] = 1.0
    # 参考与 DUT 脉冲相同，只关注原始目标分辨率。
    pulse = TimeSeries(pulse_time_s, pulse_values[:, None], pulse_rate_hz)
    # 八点、8 GSa/s 的 DUT 原始记录只有 1 GHz 独立频率分辨率。
    waveform_time_s = np.arange(8, dtype=np.float64) / pulse_rate_hz
    # 非常数波形提供有限且非零的稳健 Vpp。
    waveform_values = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    # 参考和 DUT 原始波形共享本反例的短记录长度。
    waveform = TimeSeries(
        waveform_time_s,
        waveform_values[:, None],
        pulse_rate_hz,
    )
    # 用户请求 0–1 GHz；若只看拟合脉冲会错误生成十个 100 MHz 核心。
    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        detrend_phase=False,
    )

    # 准备阶段应把真实目标时长纳入物理定位能力。
    workspace = prepare_frequency_attribution(
        pulse,
        pulse,
        settings,
        reference_waveform=waveform,
        dut_waveform=waveform,
    )

    # 物理分辨率按 max(Fs/N) 取到短原始记录的 1 GHz。
    assert workspace.physical_resolution_hz == pytest.approx(1.0e9)
    # 核心向上扩大到 1 GHz，而不是显示十个没有独立频点支撑的 100 MHz 频道。
    assert workspace.effective_window_width_hz == pytest.approx(1.0e9)
    # 扫描跨度正好容纳一个完整核心。
    assert workspace.candidates == (
        FrequencyBand(low_hz=0.0, high_hz=1.0e9, center_hz=0.5e9),
    )
    # 用户必须收到分辨率不足的明确告警。
    assert any("物理频率分辨率" in warning for warning in workspace.warnings)


# 锁定 1% 简化只改同一频段内的模式标签，不能把推荐换到别的频道。
def test_mode_simplification_never_moves_the_raw_best_frequency_band() -> None:
    """1% 规则只简化同一频段模式，不能换成较早频段。"""

    # 低频核心的幅度改善只比全局最大值低 0.9%。
    low_band = FrequencyBand(0.0, 100.0e6, 50.0e6)
    # 高频核心才是原始最优幅相频段。
    high_band = FrequencyBand(100.0e6, 200.0e6, 150.0e6)
    # 低频幅度候选重现旧实现会误选的原顺序首项。
    candidates = (
        BandAttribution(low_band, "magnitude", 0.0, 0.991, 0.991, True),
        BandAttribution(low_band, "phase", 0.0, 0.2, 0.2, True),
        BandAttribution(low_band, "both", 0.0, 0.3, 0.3, True),
        BandAttribution(high_band, "magnitude", 0.0, 0.5, 0.5, True),
        BandAttribution(high_band, "phase", 0.0, 0.4, 0.4, True),
        BandAttribution(high_band, "both", 0.0, 1.0, 1.0, True),
    )
    # 排名助手隔离频段选择与 FFT 数值误差。
    selected = attribution_module._select_recommendation(
        candidates,
        baseline_gap=1.0,
        baseline_tolerance=1.0e-12,
        mode_materiality_fraction=0.01,
    )
    # 推荐必须保留 150 MHz 原始最优频段。
    assert selected is not None
    # 低频 0.991 不得因为更早出现而被 1% 规则选中。
    assert selected.band == high_band
    # 同频段单模式不接近 1.0，因此仍应保留幅相。
    assert selected.mode == "both"


# 局部时域回放本身就是直接证据，不应被无关的全频结果否决。
def test_recommendation_uses_direct_local_counterfactual() -> None:
    """有效局部相位改善应按自身分数参与排名。"""

    # 单一候选核心使测试只比较模式支持。
    band = FrequencyBand(0.0, 100.0e6, 50.0e6)
    # 局部相位回放把指标差距缩小最多，应直接成为该频段解释。
    candidates = (
        BandAttribution(band, "magnitude", 0.0, 0.5, 0.5, True),
        BandAttribution(band, "phase", 0.0, 0.9, 0.9, True),
        BandAttribution(band, "both", 0.0, 0.7, 0.7, True),
    )
    # 排名只接收已经完成直接时域回放的局部候选。
    selected = attribution_module._select_recommendation(
        candidates,
        baseline_gap=1.0,
        baseline_tolerance=1.0e-12,
        mode_materiality_fraction=0.01,
    )
    # 最大正改善候选必须存在。
    assert selected is not None
    # 相位改善 0.9 高于幅相 0.7 和幅度 0.5，应直接推荐相位。
    assert selected.mode == "phase"
