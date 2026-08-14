"""“影响频段”展示适配器的证据保真测试。"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本测试仅关闭对应格式告警。
# ruff: noqa: I001

# SimpleNamespace 构造只包含适配器真正读取字段的轻量运行结果。
from types import SimpleNamespace
# replace 只改冻结请求中的 M，便于覆盖类型和下限边界。
from dataclasses import replace

# Path 用于构造真实 InfluenceRequest，测试不触发文件读取。
from pathlib import Path

# NumPy 用于核对 NaN 断点、布尔掩码和真实零值。
import numpy as np
# pytest 参数化覆盖手工相位拟合带的部分越界与完全不相交分区。
import pytest

import response_lab.influence_controller as influence_controller_module

# 取消异常验证 worker 不会把正常关窗路径当成分析失败。
from response_lab.cancellation import OperationCancelledError
# 领域数据类确保测试候选与真实扫描结果使用相同字段合同。
from response_lab.attribution import (
    AttributionSettings,
    BandAttribution,
    FrequencyBand,
    VirtualEyeSettings,
)
# 被测适配器负责请求设置、眼图轨迹和影响曲线协议。
from response_lab.influence_controller import (
    InfluenceAnalysisThread,
    InfluenceRequest,
    _build_attribution_settings,
    _estimate_influence_peak_memory_bytes,
    _estimate_workload,
    eye_payload,
    influence_curve_payload,
)
# 模型类用于验证控制器不篡改 PAM4 请求，时间轴仍自推采样率。
from response_lab.models import CompensationSettings, TimeSeries
from response_lab.vpp_analysis import VppAnalysisSettings


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
    workspace = SimpleNamespace(
        candidates=(first_band, second_band),
        settings=SimpleNamespace(
            metric="vpp",
            vpp=SimpleNamespace(method="frequency_rms_error"),
        ),
    )
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
    # 频域 Vpp 实际是复误差 RMS，纵轴和列表都必须明确使用 Vrms。
    assert payload["metric_axis_label"] == "频域误差改善 (Vrms)"
    assert payload["candidates"][0].endswith("0 Vrms")


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


# 用户只提供 PAM4 与 M，控制器必须保留调制并从脉冲长度推导 Np。
def test_controller_preserves_pam4_and_derives_np_from_pulse_length() -> None:
    """16 点脉冲和 M=4 应自动形成 Np=4 的 VirtualEyeSettings。"""

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
        samples_per_ui=4,
        vpp_method=None,
        pattern_source=None,
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=None,
        post_cursor_ui=None,
        band_width_hz=250.0e6,
        frequency_settings=frequency_settings,
        auto_frequency_bands=False,
        version=1,
    )

    # 只调用请求到纯算法设置的转换层。
    settings = _build_attribution_settings(
        request,
        reference_pulse,
        dut_pulse,
    )

    # 眼图设置必须存在。
    assert settings.eye is not None
    # 最终调制仍是用户选择的 PAM4，不是默认 NRZ。
    assert settings.eye.modulation == "pam4"
    # Np 来自真实脉冲样点数 16 除以 M=4，不再依赖页面重复输入。
    assert settings.eye.pulse_length_ui == 4
    # 一个页面频宽同时定义相邻核心中心间距。
    assert settings.frequency_step_hz == 250.0e6
    # 满权核心宽度必须与用户输入完全一致。
    assert settings.requested_window_hz == 250.0e6


def test_automatic_scan_preserves_a_manually_confirmed_phase_fit_band() -> None:
    """自动扫描补偿频带时，后台不能改写主窗口已经确认的相位拟合范围。"""

    sample_rate_hz = 1.0e9
    samples = 64
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    impulse = np.zeros(samples, dtype=np.float64)
    impulse[0] = 1.0
    reference_pulse = TimeSeries(time_s, impulse, sample_rate_hz)
    dut_pulse = TimeSeries(time_s, impulse.copy(), sample_rate_hz)
    manual_phase_low_hz = 123.0e6
    manual_phase_high_hz = 234.0e6
    request = InfluenceRequest(
        reference_pulse_path=Path("reference.csv"),
        dut_pulse_path=Path("dut.csv"),
        metric="eye_height",
        modulation="nrz",
        samples_per_ui=4,
        vpp_method=None,
        pattern_source=None,
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=None,
        post_cursor_ui=None,
        band_width_hz=50.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0,
            phase_fit_low_hz=manual_phase_low_hz,
            phase_fit_high_hz=manual_phase_high_hz,
            detrend_phase=True,
            analysis_points=257,
        ),
        auto_frequency_bands=True,
        auto_phase_fit_band=False,
        version=1,
    )

    settings = _build_attribution_settings(request, reference_pulse, dut_pulse)

    assert settings.scan_high_hz > settings.scan_low_hz
    assert settings.phase_fit_low_hz == manual_phase_low_hz
    assert settings.phase_fit_high_hz == manual_phase_high_hz


@pytest.mark.parametrize(
    ("phase_low_hz", "phase_high_hz"),
    (
        (450.0e6, 490.0e6),
        (490.0e6, 499.0e6),
    ),
)
def test_automatic_scan_rejects_manual_phase_band_outside_evaluation_domain(
    phase_low_hz: float,
    phase_high_hz: float,
) -> None:
    """部分或完全越过自动扫描域时必须明示，不能静默裁剪或替换手工值。"""

    sample_rate_hz = 1.0e9
    samples = 64
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    impulse = np.zeros(samples, dtype=np.float64)
    impulse[0] = 1.0
    pulse = TimeSeries(time_s, impulse, sample_rate_hz)
    request = InfluenceRequest(
        reference_pulse_path=Path("reference.csv"),
        dut_pulse_path=Path("dut.csv"),
        metric="eye_height",
        modulation="nrz",
        samples_per_ui=4,
        vpp_method=None,
        pattern_source=None,
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=None,
        post_cursor_ui=None,
        band_width_hz=50.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0,
            phase_fit_low_hz=phase_low_hz,
            phase_fit_high_hz=phase_high_hz,
            detrend_phase=True,
            analysis_points=257,
        ),
        auto_frequency_bands=True,
        auto_phase_fit_band=False,
        version=1,
    )

    with pytest.raises(ValueError, match="手工相位拟合频带.*自动扫描范围"):
        _build_attribution_settings(request, pulse, pulse)


# 自动 Np 只接受两份等长且可被 M 整除的完整拟合脉冲。
def test_controller_rejects_ambiguous_automatic_np_inputs() -> None:
    """不等长或非整数 UI 长度都必须在构造眼图设置时明确失败。"""

    # 8 GSa/s 只用于构造合法时间轴，Np 推导只读取样点数和 M。
    sample_rate_hz = 8.0e9

    # 局部工厂生成指定点数的单通道脉冲，避免测试依赖生产加载器。
    def pulse(samples: int) -> TimeSeries:
        # 每份时间轴都从零开始并严格均匀采样。
        time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
        # 中心附近放置一个非零主抽头，保持脉冲物理含义。
        values = np.zeros(samples, dtype=np.float64)
        # 整数中点在奇偶长度下都位于数组有效范围内。
        values[samples // 2] = 1.0
        # 返回真实 TimeSeries，让控制器读取公开 samples 属性。
        return TimeSeries(time_s, values, sample_rate_hz)

    # 有效手动频带隔离 Np 推导错误，不触发自动频带建议。
    frequency_settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=1.0e9,
        phase_fit_low_hz=0.0,
        phase_fit_high_hz=1.0e9,
        detrend_phase=False,
    )
    # 页面请求只提供用户仍需选择的 PAM4 和 M=4。
    request = InfluenceRequest(
        reference_pulse_path=Path("reference.csv"),
        dut_pulse_path=Path("dut.csv"),
        metric="eye_width",
        modulation="pam4",
        samples_per_ui=4,
        vpp_method=None,
        pattern_source=None,
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=None,
        post_cursor_ui=None,
        band_width_hz=100.0e6,
        frequency_settings=frequency_settings,
        auto_frequency_bands=False,
        version=1,
    )

    # 参考 16 点、DUT 20 点会得到两个不同 Np，必须拒绝而不是裁剪。
    with np.testing.assert_raises_regex(ValueError, "必须等长"):
        # 通过正式设置构造边界观察用户最终会收到的领域错误。
        _build_attribution_settings(request, pulse(16), pulse(20))
    # 两份 18 点脉冲虽等长，但 M=4 时包含 4.5 UI，仍不能构造眼图。
    with np.testing.assert_raises_regex(ValueError, "不能被 M=4 整除"):
        # 禁止向上取整、向下取整或静默补零形成假的 Np。
        _build_attribution_settings(request, pulse(18), pulse(18))
    # 8 点脉冲配 M=8 虽能整除，但派生 Np=1 没有足够稳态边界。
    with np.testing.assert_raises_regex(ValueError, "Np 必须至少为 2"):
        # 最低 Np 继续由内部 VirtualEyeSettings 合同统一裁决。
        _build_attribution_settings(
            replace(request, samples_per_ui=8),
            pulse(8),
            pulse(8),
        )

    # Vpp 不构造眼图，17 点记录且没有 M 时不能误触发整除校验。
    vpp_request = InfluenceRequest(
        reference_pulse_path=Path("reference.csv"),
        dut_pulse_path=Path("dut.csv"),
        metric="vpp",
        modulation=None,
        samples_per_ui=1,
        vpp_method="lfp",
        pattern_source="builtin_prbs13q_gray",
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        band_width_hz=100.0e6,
        frequency_settings=frequency_settings,
        auto_frequency_bands=False,
        version=2,
    )
    # 奇数样点数是明确反例，任何无条件 samples%M 路径都会在这里失败。
    vpp_settings = _build_attribution_settings(
        vpp_request,
        pulse(17),
        pulse(17),
    )
    # Vpp 设置不携带内部 Np/M 眼图配置。
    assert vpp_settings.eye is None

    # 自动取模只接受真正整数且至少为三的 M，拒绝 bool、浮点伪整数和低分辨率。
    invalid_m_cases = (
        (True, "M 必须是整数"),
        (4.0, "M 必须是整数"),
        (2, "M 必须至少为 3"),
    )
    # 每个分区都通过同一冻结请求替换唯一目标字段。
    for invalid_m, expected_error in invalid_m_cases:
        # 错误必须发生在取模前，不能泄漏 Python 除零或类型异常。
        with np.testing.assert_raises_regex(ValueError, expected_error):
            # 16 点等长脉冲隔离 M 类型和下限，不引入其他失败原因。
            _build_attribution_settings(
                replace(request, samples_per_ui=invalid_m),
                pulse(16),
                pulse(16),
            )


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
    # 三种指标覆盖 Vpp 不需要眼参数及两个眼指标需要调制和 M 的分支。
    metric_inputs = (
        ("vpp", None, 1),
        ("eye_height", "nrz", 4),
        ("eye_width", "pam4", 4),
    )
    # 零、负数和两个方向的非有限值都不能进入候选生成器。
    invalid_widths_hz = (0.0, -100.0e6, np.nan, np.inf, -np.inf)

    # 逐指标验证同一个公开请求合同。
    for metric, modulation, samples_per_ui in metric_inputs:
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
                    samples_per_ui=samples_per_ui,
                    vpp_method="lfp" if metric == "vpp" else None,
                    pattern_source=(
                        "builtin_prbs13q_gray" if metric == "vpp" else None
                    ),
                    pattern_path=None,
                    pattern_value_kind=None,
                    pre_cursor_ui=0 if metric == "vpp" else None,
                    post_cursor_ui=0 if metric == "vpp" else None,
                    band_width_hz=band_width_hz,
                    frequency_settings=frequency_settings,
                    auto_frequency_bands=False,
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
    # 五千万点目标即使只扫少量频段，保守峰值内存也超过 8 GiB。
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
            target_samples=50_000_000,
            other_input_samples=50_000_000,
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


def test_eye_width_workload_counts_forty_one_crossing_slices() -> None:
    """眼宽的水平切片成本不能只按一次 FFT 卷积长度估算。"""

    common = dict(
        scan_low_hz=0.0,
        scan_high_hz=2.8e9,
        eye=VirtualEyeSettings(
            modulation="pam4",
            pulse_length_ui=10,
            samples_per_ui=32,
            symbol_count=400,
        ),
        detrend_phase=False,
    )
    width_settings = AttributionSettings(metric="eye_width", **common)
    height_settings = AttributionSettings(metric="eye_height", **common)

    _, _, width_notice = _estimate_workload(
        width_settings,
        physical_resolution_hz=100.0e6,
        target_samples=320,
        other_input_samples=640,
    )
    _, _, height_notice = _estimate_workload(
        height_settings,
        physical_resolution_hz=100.0e6,
        target_samples=320,
        other_input_samples=640,
    )

    assert "较长时间" in width_notice
    assert height_notice == ""


def test_vpp_workload_estimate_envelopes_measured_rms_peak_and_lfp_ifft() -> None:
    """Vpp 周期缓存应按实测校准，并为 LFP 的候选 IFFT 追加余量。"""

    base_vpp = VppAnalysisSettings(
        method="frequency_rms_error",
        pattern_source="builtin_prbs13q_gray",
        samples_per_ui=32,
        pre_cursor_ui=8,
        post_cursor_ui=24,
        pattern_path=None,
        file_value_kind="symbol_codes",
    )
    rms_settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        vpp=base_vpp,
        detrend_phase=False,
    )
    lfp_settings = replace(
        rms_settings,
        vpp=replace(base_vpp, method="lfp"),
    )
    period_samples = 8191 * 32

    rms_bytes = _estimate_influence_peak_memory_bytes(
        rms_settings,
        target_samples=period_samples,
        other_input_samples=2048,
    )
    lfp_bytes = _estimate_influence_peak_memory_bytes(
        lfp_settings,
        target_samples=period_samples,
        other_input_samples=2048,
    )

    # 独立子进程该配置新增 RSS 峰值为 117,489,664 B（约 448 B/周期点）。
    assert rms_bytes >= 117_489_664
    assert lfp_bytes > rms_bytes


def test_vpp_workload_gate_uses_period_model_not_legacy_192_bytes_per_point() -> None:
    """超过 8 GiB 的大周期 Vpp 应在 prepare_vpp_analysis 前被拒绝。"""

    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        vpp=VppAnalysisSettings(
            method="frequency_rms_error",
            pattern_source="builtin_prbs13q_gray",
            samples_per_ui=32,
            pre_cursor_ui=8,
            post_cursor_ui=24,
            pattern_path=None,
            file_value_kind="symbol_codes",
        ),
        detrend_phase=False,
    )

    # 旧 192 B/点估算会低估大周期；新 RMS 模型约 10.7 GiB，必须提前停止。
    with np.testing.assert_raises_regex(ValueError, "峰值内存"):
        _estimate_workload(
            settings,
            physical_resolution_hz=100.0e6,
            target_samples=20_000_000,
            other_input_samples=2048,
        )


def test_vpp_workload_uses_dynamic_budget_before_rfft(
    monkeypatch,
) -> None:
    """低可用内存必须让 Vpp 周期 FFT 之前的纯估算器拒绝任务。"""

    settings = AttributionSettings(
        metric="vpp",
        scan_low_hz=0.0,
        scan_high_hz=1.0e9,
        vpp=VppAnalysisSettings(
            method="frequency_rms_error",
            pattern_source="builtin_prbs13q_gray",
            samples_per_ui=32,
            pre_cursor_ui=8,
            post_cursor_ui=24,
            pattern_path=None,
            file_value_kind="symbol_codes",
        ),
        detrend_phase=False,
    )
    monkeypatch.setattr(
        influence_controller_module,
        "system_available_memory_bytes",
        lambda: 600 * 1024**2,
        raising=False,
    )

    with np.testing.assert_raises_regex(ValueError, "动态安全预算"):
        _estimate_workload(
            settings,
            physical_resolution_hz=100.0e6,
            target_samples=8191 * 32,
            other_input_samples=2048,
        )


def test_vpp_thread_dynamic_budget_blocks_prepare_and_rfft(
    tmp_path,
    monkeypatch,
) -> None:
    """真实后台编排必须在 prepare_vpp_analysis 入口之前返回预算错误。"""

    sample_rate_hz = 10.0e9
    samples = 64
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    pulse_values = np.exp(-0.5 * ((np.arange(samples) - 32.0) / 2.0) ** 2)
    reference_path = tmp_path / "reference.csv"
    dut_path = tmp_path / "dut.csv"
    np.savetxt(reference_path, np.column_stack((time_s, pulse_values)), delimiter=",")
    np.savetxt(dut_path, np.column_stack((time_s, 0.9 * pulse_values)), delimiter=",")
    request = InfluenceRequest(
        reference_pulse_path=reference_path,
        dut_pulse_path=dut_path,
        metric="vpp",
        modulation=None,
        samples_per_ui=32,
        vpp_method="frequency_rms_error",
        pattern_source="builtin_prbs13q_gray",
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=0,
        post_cursor_ui=0,
        band_width_hz=200.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0e9,
            phase_fit_low_hz=0.0,
            phase_fit_high_hz=1.0e9,
            detrend_phase=False,
            analysis_points=257,
        ),
        auto_frequency_bands=False,
        version=7,
    )
    monkeypatch.setattr(
        influence_controller_module,
        "system_available_memory_bytes",
        lambda: 600 * 1024**2,
    )
    prepare_called = False

    def forbidden_prepare(*_args, **_kwargs):
        nonlocal prepare_called
        prepare_called = True
        raise AssertionError("dynamic budget must reject before Vpp rFFT preparation")

    monkeypatch.setattr(
        influence_controller_module,
        "prepare_frequency_attribution",
        forbidden_prepare,
    )
    failures: list[tuple[str, int]] = []
    thread = InfluenceAnalysisThread(request)
    thread.failed.connect(lambda detail, version: failures.append((detail, version)))

    thread.run()

    assert prepare_called is False
    assert failures and "动态安全预算" in failures[0][0]
    assert failures[0][1] == 7


def test_vpp_thread_rejects_pmax_window_before_loading_external_pattern(
    tmp_path,
    monkeypatch,
) -> None:
    """生产编排必须先验证两份脉冲窗口，再读取用户码型文本。"""

    sample_rate_hz = 10.0e9
    samples = 64
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    pulse_values = np.zeros(samples, dtype=np.float64)
    pulse_values[1] = 1.0
    reference_path = tmp_path / "reference_boundary.csv"
    dut_path = tmp_path / "dut_boundary.csv"
    pattern_path = tmp_path / "pattern_boundary.csv"
    np.savetxt(reference_path, np.column_stack((time_s, pulse_values)), delimiter=",")
    np.savetxt(dut_path, np.column_stack((time_s, pulse_values)), delimiter=",")
    pattern_path.write_text("0\n1\n", encoding="utf-8")
    request = InfluenceRequest(
        reference_pulse_path=reference_path,
        dut_pulse_path=dut_path,
        metric="vpp",
        modulation=None,
        samples_per_ui=4,
        vpp_method="lfp",
        pattern_source="file",
        pattern_path=pattern_path,
        pattern_value_kind="symbol_codes",
        pre_cursor_ui=1,
        post_cursor_ui=0,
        band_width_hz=200.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0e9,
            phase_fit_low_hz=0.0,
            phase_fit_high_hz=1.0e9,
            detrend_phase=False,
            analysis_points=257,
        ),
        auto_frequency_bands=False,
        version=8,
    )

    def forbidden_pattern_load(*_args, **_kwargs):
        raise AssertionError("invalid pmax window must fail before pattern load")

    monkeypatch.setattr(
        influence_controller_module,
        "load_pattern_levels",
        forbidden_pattern_load,
    )
    failures: list[tuple[str, int]] = []
    thread = InfluenceAnalysisThread(request)
    thread.failed.connect(lambda detail, version: failures.append((detail, version)))

    thread.run()

    assert failures and "窗口越界" in failures[0][0]
    assert failures[0][1] == 8


def test_external_pattern_vpp_budget_runs_through_loader_callback_before_parse(
    tmp_path,
    monkeypatch,
) -> None:
    """外部码型应先用同描述符统计的 symbol 数完成周期 FFT 内存门禁。"""

    sample_rate_hz = 10.0e9
    samples = 64
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    pulse_values = np.exp(-0.5 * ((np.arange(samples) - 32.0) / 2.0) ** 2)
    reference_path = tmp_path / "reference_external_budget.csv"
    dut_path = tmp_path / "dut_external_budget.csv"
    pattern_path = tmp_path / "pattern_external_budget.csv"
    np.savetxt(reference_path, np.column_stack((time_s, pulse_values)), delimiter=",")
    np.savetxt(dut_path, np.column_stack((time_s, 0.9 * pulse_values)), delimiter=",")
    pattern_path.write_text("0\n1\n", encoding="utf-8")
    request = InfluenceRequest(
        reference_pulse_path=reference_path,
        dut_pulse_path=dut_path,
        metric="vpp",
        modulation=None,
        samples_per_ui=32,
        vpp_method="frequency_rms_error",
        pattern_source="file",
        pattern_path=pattern_path,
        pattern_value_kind="symbol_codes",
        pre_cursor_ui=0,
        post_cursor_ui=0,
        band_width_hz=200.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0e9,
            phase_fit_low_hz=0.0,
            phase_fit_high_hz=1.0e9,
            detrend_phase=False,
            analysis_points=257,
        ),
        auto_frequency_bands=False,
        version=9,
    )
    monkeypatch.setattr(
        influence_controller_module,
        "system_available_memory_bytes",
        lambda: 600 * 1024**2,
    )
    callback_invoked = False

    def guarded_pattern_load(
        _settings,
        *,
        symbol_count_preflight=None,
        cancelled=None,
    ):
        nonlocal callback_invoked
        assert symbol_count_preflight is not None
        assert cancelled is not None
        callback_invoked = True
        symbol_count_preflight(8191)
        raise AssertionError("dynamic Vpp budget must reject before pattern parse")

    monkeypatch.setattr(
        influence_controller_module,
        "load_pattern_levels",
        guarded_pattern_load,
    )
    failures: list[tuple[str, int]] = []
    thread = InfluenceAnalysisThread(request)
    thread.failed.connect(lambda detail, version: failures.append((detail, version)))

    thread.run()

    assert callback_invoked is True
    assert failures and "动态安全预算" in failures[0][0]
    assert failures[0][1] == 9


def test_influence_thread_reports_loader_cancellation_separately(
    tmp_path,
    monkeypatch,
) -> None:
    """Loader cancellation is expected control flow, not a failed analysis."""

    request = InfluenceRequest(
        reference_pulse_path=tmp_path / "reference.csv",
        dut_pulse_path=tmp_path / "dut.csv",
        metric="eye_height",
        modulation="PAM4",
        samples_per_ui=4,
        vpp_method=None,
        pattern_source=None,
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=None,
        post_cursor_ui=None,
        band_width_hz=200.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0e9,
            phase_fit_low_hz=0.0,
            phase_fit_high_hz=1.0e9,
            detrend_phase=False,
            analysis_points=257,
        ),
        auto_frequency_bands=False,
        version=23,
    )

    def cancelled_loader(*_args, **_kwargs):
        raise OperationCancelledError("影响频段分析已取消")

    monkeypatch.setattr(
        influence_controller_module,
        "load_csv_timeseries",
        cancelled_loader,
    )
    cancellations: list[int] = []
    failures: list[tuple[str, int]] = []
    thread = InfluenceAnalysisThread(request)
    thread.cancelled.connect(cancellations.append)
    thread.failed.connect(lambda detail, version: failures.append((detail, version)))

    thread.run()

    assert cancellations == [23]
    assert failures == []


def test_influence_thread_passes_cancellation_into_automatic_band_suggestion(
    tmp_path,
    monkeypatch,
) -> None:
    """自动频带的脉冲 FFT 必须能直接看到关窗取消请求。"""

    sample_rate_hz = 8.0e9
    samples = 64
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    pulse = np.exp(-0.5 * ((np.arange(samples) - 32.0) / 3.0) ** 2)
    reference_path = tmp_path / "reference.csv"
    dut_path = tmp_path / "dut.csv"
    np.savetxt(reference_path, np.column_stack((time_s, pulse)), delimiter=",")
    np.savetxt(dut_path, np.column_stack((time_s, 0.9 * pulse)), delimiter=",")
    request = InfluenceRequest(
        reference_pulse_path=reference_path,
        dut_pulse_path=dut_path,
        metric="eye_height",
        modulation="nrz",
        samples_per_ui=4,
        vpp_method=None,
        pattern_source=None,
        pattern_path=None,
        pattern_value_kind=None,
        pre_cursor_ui=None,
        post_cursor_ui=None,
        band_width_hz=200.0e6,
        frequency_settings=CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0,
            phase_fit_low_hz=0.0,
            phase_fit_high_hz=1.0,
            detrend_phase=False,
            analysis_points=257,
        ),
        auto_frequency_bands=True,
        version=24,
        auto_phase_fit_band=True,
    )

    def cancelled_suggestion(*_args, cancelled=None, **_kwargs):
        assert cancelled is not None
        raise OperationCancelledError("影响频段分析已取消")

    monkeypatch.setattr(
        influence_controller_module,
        "suggest_frequency_settings",
        cancelled_suggestion,
    )
    cancellations: list[int] = []
    failures: list[tuple[str, int]] = []
    thread = InfluenceAnalysisThread(request)
    thread.cancelled.connect(cancellations.append)
    thread.failed.connect(lambda detail, version: failures.append((detail, version)))

    thread.run()

    assert cancellations == [24]
    assert failures == []
