"""“影响频段”展示适配器的证据保真测试。"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本测试仅关闭对应格式告警。
# ruff: noqa: I001

# SimpleNamespace 构造只包含适配器真正读取字段的轻量运行结果。
from types import SimpleNamespace

# Path 用于构造真实 InfluenceRequest，测试不触发文件读取。
from pathlib import Path

# NumPy 用于核对 NaN 断点、布尔掩码和真实零值。
import numpy as np

# 领域数据类确保测试候选与真实扫描结果使用相同字段合同。
from response_lab.attribution import (
    AttributionSettings,
    BandAttribution,
    FrequencyBand,
    VirtualEyeSettings,
)
# 被测适配器负责请求设置、眼图轨迹和影响曲线协议。
from response_lab.influence_controller import (
    InfluenceRequest,
    _build_attribution_settings,
    _estimate_workload,
    eye_payload,
    influence_curve_payload,
)
# 模型类用于验证控制器不篡改 PAM4 请求，时间轴仍自推采样率。
from response_lab.models import CompensationSettings, TimeSeries


# 无效候选必须形成 NaN 断点，不能覆盖同频轴上的真实零改善。
def test_curve_payload_preserves_zero_and_marks_invalid_candidate_as_nan() -> None:
    """有效零分应保留 0，无效项应输出 NaN、False 和简短诊断。"""

    # 两个 100 MHz 频段构成最小可观察曲线。
    first_band = FrequencyBand(low_hz=0.0, high_hz=100.0e6, center_hz=50.0e6)
    # 第二频段用于放置一个不可解析的纯幅度候选。
    second_band = FrequencyBand(
        low_hz=100.0e6,
        high_hz=200.0e6,
        center_hz=150.0e6,
    )
    # 第一个纯幅度候选有效，但改善量恰好是真实零。
    valid_zero = BandAttribution(
        band=first_band,
        mode="magnitude",
        metric_after=1.0,
        improvement=0.0,
        recovery_ratio=0.0,
        valid=True,
    )
    # 第二个纯幅度候选因谱零点不可解析。
    invalid_magnitude = BandAttribution(
        band=second_band,
        mode="magnitude",
        metric_after=np.nan,
        improvement=np.nan,
        recovery_ratio=np.nan,
        valid=False,
        invalid_reason="候选频段响应不可解析",
    )
    # 两个相位候选都有效，避免缺失网格被误计为本测试的无效项。
    valid_phase_first = BandAttribution(
        band=first_band,
        mode="phase",
        metric_after=1.1,
        improvement=0.1,
        recovery_ratio=0.1,
        valid=True,
    )
    # 第二个相位点使用不同改善量验证中心映射顺序。
    valid_phase_second = BandAttribution(
        band=second_band,
        mode="phase",
        metric_after=1.2,
        improvement=0.2,
        recovery_ratio=0.2,
        valid=True,
    )
    # 联合模式第一个点同样保持有限。
    valid_both_first = BandAttribution(
        band=first_band,
        mode="both",
        metric_after=1.3,
        improvement=0.3,
        recovery_ratio=0.3,
        valid=True,
    )
    # 联合模式第二个点完成完整的两频段三模式矩阵。
    valid_both_second = BandAttribution(
        band=second_band,
        mode="both",
        metric_after=1.4,
        improvement=0.4,
        recovery_ratio=0.4,
        valid=True,
    )
    # 工作区只需提供适配器用于建立频率轴的候选频段。
    workspace = SimpleNamespace(candidates=(first_band, second_band))
    # 结果按真实扫描的“逐频段、逐模式”顺序保存六个候选。
    result = SimpleNamespace(
        candidates=(
            valid_zero,
            valid_phase_first,
            valid_both_first,
            invalid_magnitude,
            valid_phase_second,
            valid_both_second,
        ),
        recommendation=valid_zero,
    )
    # 可点击列表只含有效候选，证明无效项不会通过曲线修复进入列表。
    run = SimpleNamespace(
        workspace=workspace,
        result=result,
        displayed_candidates=(valid_zero,),
    )

    # 执行领域结果到页面协议的转换。
    payload = influence_curve_payload(run)

    # 频率轴保持两个真实中心。
    np.testing.assert_array_equal(
        payload["frequency_hz"],
        np.array([50.0e6, 150.0e6]),
    )
    # 有效零改善严格保留为 0.0。
    assert payload["scores"]["magnitude"][0] == 0.0
    # 无效幅度候选以 NaN 表示断点，而不是伪造的零分。
    assert np.isnan(payload["scores"]["magnitude"][1])
    # 布尔掩码与两个幅度点的有效性逐项对应。
    np.testing.assert_array_equal(
        payload["valid_masks"]["magnitude"],
        np.array([True, False]),
    )
    # 只有一个模式-频段点不可解析。
    assert payload["invalid_count"] == 1
    # 诊断明确说明断点，不把失败解释成零影响。
    assert payload["diagnostic"] == "1 个候选不可解析，曲线以断点表示"
    # 列表仍只包含有效零分推荐。
    assert len(payload["candidates"]) == 1
    # 推荐文字中的改善值证明真实零候选没有被过滤。
    assert "改善 0" in payload["candidates"][0]


# 眼图适配器必须保留共同时轴和参考/补偿前/补偿后三角色数值。
def test_eye_payload_preserves_trace_roles_without_density_conversion() -> None:
    """页面协议应直接携带三组轨迹，不得串位或量化成像素。"""

    # M=2 时每条轨迹严格含 2*M+1=5 个点。
    time_ui = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    # 三角色使用不同偏置，任意错位都会被数组断言捕获。
    reference = np.arange(10, dtype=np.float64).reshape(2, 5) + 10.0
    # 补偿前另用 20 V 偏置的测试数据。
    before = np.arange(10, dtype=np.float64).reshape(2, 5) + 20.0
    # 补偿后使用 30 V 偏置。
    after = np.arange(10, dtype=np.float64).reshape(2, 5) + 30.0
    # SimpleNamespace 精确模拟 eye_payload 读取的新领域字段。
    comparison = SimpleNamespace(
        time_ui=time_ui,
        reference_traces_v=reference,
        before_traces_v=before,
        after_traces_v=after,
        amplitude_range_v=(-4.0, 4.0),
    )

    # 转换为页面轨迹协议。
    payload = eye_payload(comparison)

    # 公共 UI 时轴不重建为边缘或分箱网格。
    np.testing.assert_array_equal(payload["time_ui"], time_ui)
    # 共同幅值范围保持输入定义。
    assert payload["amplitude_range_v"] == (-4.0, 4.0)
    # 参考角色必须是第一组轨迹。
    np.testing.assert_array_equal(payload["reference"]["traces_v"], reference)
    # 补偿前不能误用补偿后数据。
    np.testing.assert_array_equal(payload["before"]["traces_v"], before)
    # 补偿后保留第三组独立数值。
    np.testing.assert_array_equal(payload["after"]["traces_v"], after)


# 用户在页面选择 PAM4 后，控制器不能在构造归因设置时退回默认 NRZ。
def test_controller_preserves_pam4_modulation_request() -> None:
    """PAM4 请求应原样进入 VirtualEyeSettings。"""

    # 16 点均匀时间轴足以构造有效 TimeSeries，本测试不执行眼图卷积。
    sample_rate_hz = 8.0e9
    # 时间轴保留秒单位合同。
    time_s = np.arange(16, dtype=np.float64) / sample_rate_hz
    # 单通道脉冲数值只用于模型形状验证。
    pulse_values = np.zeros(16, dtype=np.float64)
    # 主光标设为有限非零值。
    pulse_values[8] = 1.0
    # 参考脉冲通过真实模型保留时间轴采样率。
    reference_pulse = TimeSeries(
        time_s=time_s,
        values=pulse_values,
        sample_rate_hz=sample_rate_hz,
    )
    # DUT 对本测试可使用相同有效脉冲。
    dut_pulse = TimeSeries(
        time_s=time_s,
        values=pulse_values,
        sample_rate_hz=sample_rate_hz,
    )
    # 手动频带避免调用自动带宽建议器，隔离本测试变量。
    frequency_settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=1.0e9,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0e9,
        detrend_phase=False,
    )
    # 使用完整冻结请求覆盖正式主窗调度边界。
    request = InfluenceRequest(
        reference_pulse_path=Path("reference.csv"),
        dut_pulse_path=Path("dut.csv"),
        metric="eye_height",
        modulation="pam4",
        pulse_length_ui=4,
        samples_per_ui=4,
        reference_data_path=None,
        dut_data_path=None,
        band_width_hz=250.0e6,
        frequency_settings=frequency_settings,
        auto_frequency_bands=False,
        bin_config=None,
        version=1,
    )

    # 只调用请求到纯算法设置的转换层。
    settings = _build_attribution_settings(
        request,
        reference_pulse,
        dut_pulse,
        None,
        None,
    )

    # 眼图设置必须存在。
    assert settings.eye is not None
    # 最终调制仍是用户选择的 PAM4，不是默认 NRZ。
    assert settings.eye.modulation == "pam4"
    # 一个页面频宽同时定义相邻核心中心间距。
    assert settings.frequency_step_hz == 250.0e6
    # 满权核心宽度必须与用户输入完全一致。
    assert settings.requested_window_hz == 250.0e6


# 频段宽度直接决定候选数量和 FFT 补偿掩码，所有指标都必须拒绝非物理数值。
def test_request_rejects_invalid_band_width_for_every_metric() -> None:
    """Vpp、眼高和眼宽都只接受正有限 Hz 频段宽度。"""

    # 使用有效手动扫描范围，确保失败只由新增频宽字段触发。
    frequency_settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=1.0e9,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0e9,
        detrend_phase=False,
    )
    # 三种指标覆盖 Vpp 不需要眼参数及两个眼指标需要完整参数的分支。
    metric_inputs = (
        ("vpp", None, None, None),
        ("eye_height", "nrz", 4, 4),
        ("eye_width", "pam4", 4, 4),
    )
    # 零、负数和两个方向的非有限值都不能进入候选生成器。
    invalid_widths_hz = (0.0, -100.0e6, np.nan, np.inf, -np.inf)

    # 逐指标验证同一个公开请求合同。
    for metric, modulation, pulse_length_ui, samples_per_ui in metric_inputs:
        # 每个无效分区都应产生相同的领域错误，而非后续数组异常。
        for band_width_hz in invalid_widths_hz:
            # 错误信息明确指向用户可修改的控件。
            with np.testing.assert_raises_regex(ValueError, "频段宽度"):
                # 构造请求本身即完成轻量校验，无需启动后台线程。
                InfluenceRequest(
                    reference_pulse_path=Path("reference.csv"),
                    dut_pulse_path=Path("dut.csv"),
                    metric=metric,
                    modulation=modulation,
                    pulse_length_ui=pulse_length_ui,
                    samples_per_ui=samples_per_ui,
                    reference_data_path=None,
                    dut_data_path=None,
                    band_width_hz=band_width_hz,
                    frequency_settings=frequency_settings,
                    auto_frequency_bands=False,
                    bin_config=None,
                    version=1,
                )

# 工作量估算必须在大型镜像数组分配前限制候选数量和峰值内存。
def test_workload_estimate_reports_long_scan_and_rejects_unbounded_inputs() -> None:
    """正常长任务给提示，单位错误和危险内存请求在 prepare 前失败。"""

    # 0–50 GHz、100 MHz 核心得到 500 个候选，是实际高速接口的合理宽扫描。
    normal_settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=50.0e9,
        detrend_phase=False,
    )
    # 十万点原始记录足以触发长任务提示，但不应被安全门禁拒绝。
    candidate_count, evaluation_count, notice = _estimate_workload(
        normal_settings,
        physical_resolution_hz=50.0e6,
        target_samples=100_000,
        other_input_samples=101_000,
    )
    # 核心步进为 100 MHz，所以候选数可手算为 500。
    assert candidate_count == 500
    # 三模式还各包含一次全频闭环。
    assert evaluation_count == 3 * 501
    # 提示应同时说明候选数和可安全取消。
    assert "500 个频段" in notice
    # 300 GHz 范围会产生 3000 个候选，超过防单位错误上限。
    excessive_candidates = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=300.0e9,
        detrend_phase=False,
    )
    # 过量候选必须在任何目标频谱分配前失败。
    with np.testing.assert_raises_regex(ValueError, "2000"):
        # 小记录不能掩盖候选数量本身的不合理。
        _estimate_workload(
            excessive_candidates,
            physical_resolution_hz=50.0e6,
            target_samples=100,
            other_input_samples=100,
        )
    # 一千万点目标即使只扫少量频段，保守峰值内存也超过 1.5 GiB。
    memory_heavy = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        detrend_phase=False,
    )
    # 内存门禁给出可操作的峰值说明。
    with np.testing.assert_raises_regex(ValueError, "峰值内存"):
        # 不真实分配数组，只向纯估算器提供样点数。
        _estimate_workload(
            memory_heavy,
            physical_resolution_hz=50.0e6,
            target_samples=10_000_000,
            other_input_samples=10_000_000,
        )


# 大 M 眼图的固定符号卷积远长于拟合脉冲，必须单独进入内存门禁。
def test_workload_estimate_counts_virtual_eye_excitation_memory() -> None:
    """眼图缓存超过安全预算时应在构造 symbol_count*M 冲激前失败。"""

    # Np=2、M=100000 的拟合脉冲只有二十万点，但固定激励超过两亿点。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=2,
            samples_per_ui=100_000,
            symbol_count=2048,
        ),
        detrend_phase=False,
    )

    # 门禁只做整数估算，不应真实分配约 2.05 亿点的冲激和复频谱。
    with np.testing.assert_raises_regex(ValueError, "峰值内存"):
        # 两份脉冲本身不超限，失败必须来自新增的眼图卷积缓存估算。
        _estimate_workload(
            settings,
            physical_resolution_hz=100.0e6,
            target_samples=200_000,
            other_input_samples=400_000,
        )
