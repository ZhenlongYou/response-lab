"""主要影响频段页签所用 Vpp 指标的独立行为测试。"""

# 延迟类型标注求值，使测试辅助函数保持与 Python 3.11 兼容。
from __future__ import annotations

# NumPy 构造可手算分位数位置的确定性波形。
import numpy as np

# Pytest 提供数值容差、参数化和异常契约断言。
import pytest

# 测试只通过公开 Vpp 接口观察行为，不调用模块私有分块函数。
from response_lab.attribution_metrics import (
    compare_vpp,
    compare_waveform_vpp,
    measure_waveform_vpp,
)

# TimeSeries 为每个测试输入绑定独立采样率和物理时间轴。
from response_lab.models import TimeSeries


# 辅助函数统一构造合法单通道记录，避免测试重复输入样板。
def _series(values: np.ndarray, sample_rate_hz: float) -> TimeSeries:
    """把一维幅度样本包装成带物理时间轴的单通道记录。"""

    # 时间列采用半开采样网格，使 N 个样本对应 N / Fs 的有效采集窗口。
    time_s = np.arange(values.size, dtype=np.float64) / sample_rate_hz
    # TimeSeries 统一使用“样本数 × 通道数”的二维幅度布局。
    return TimeSeries(time_s, values[:, None], sample_rate_hz)


# 该闭式用例固定 Q99.9%-Q0.1% 的数值定义和幅度缩放性质。
def test_equal_duration_records_use_q999_minus_q001() -> None:
    """相同有效时长的两条记录都应各自作为一个完整比较块。"""

    # 递增整数序列允许直接手算 NumPy 线性分位数的位置与插值结果。
    reference_values = np.arange(1000, dtype=np.float64)
    # 两倍幅度用于确认 DUT 指标独立计算，而不是复用参考结果。
    dut_values = 2.0 * reference_values

    # 两条记录具有相同点数与采样率，因此公共窗口就是整条记录。
    comparison = compare_vpp(
        _series(reference_values, 1.0e9),
        _series(dut_values, 1.0e9),
    )

    # 对 0..999，Q0.1%=0.999、Q99.9%=998.001，差值应为 997.002。
    assert comparison.reference.value == pytest.approx(997.002)
    # 幅度整体乘二后，稳健 Vpp 也应严格乘二。
    assert comparison.dut.value == pytest.approx(1994.004)
    # 相同时长时每条记录只产生一个块，不能被无意义地再次分段。
    assert comparison.reference.block_values == pytest.approx((997.002,))
    # DUT 块证据也必须保留为唯一一块，不能只让最终标量正确。
    assert comparison.dut.block_values == pytest.approx((1994.004,))
    # 参考极值诊断使用同一整块的 max-min，因此 0..999 的极差应为 999。
    assert comparison.reference_extrema_vpp == pytest.approx(999.0)
    # DUT 幅度整体乘二后，极值诊断也应乘二，但不能替代稳健 Vpp 排名值。
    assert comparison.dut_extrema_vpp == pytest.approx(1998.0)
    # 块级极值证据保存在参考估计中，便于诊断孤立毛刺是否放大极差。
    assert comparison.reference.extrema_block_values == pytest.approx((999.0,))


# 该用例用巨大残缺尾部专门杀死“整条记录直接算 Vpp”的错误实现。
def test_longer_record_uses_complete_common_duration_blocks_and_median() -> None:
    """较长记录应按较短记录时长分块，并丢弃不足一块的尾部。"""

    # 1000 点、1 kSa/s 构成精确 1 秒的较短有效窗口。
    reference_values = np.arange(1000, dtype=np.float64)
    # 每个 500 点块采用不同斜率，使五个块的 Vpp 可独立手算并排序。
    complete_blocks = [
        scale * np.arange(500, dtype=np.float64)
        for scale in (1.0, 2.0, 10.0, 4.0, 5.0)
    ]
    # 余下 123 点故意放置巨大摆幅，用于杀死“把不完整尾块也参与统计”的错误。
    incomplete_tail = np.linspace(-1.0e9, 1.0e9, 123, dtype=np.float64)
    # DUT 总时长超过 5 秒，且采样率与参考不同，因此不能按样本下标配对。
    dut_values = np.concatenate((*complete_blocks, incomplete_tail))

    # 公共窗口应取参考记录的 1 秒；DUT 每个完整块对应 500 个样本。
    comparison = compare_vpp(
        _series(reference_values, 1000.0),
        _series(dut_values, 500.0),
    )

    # 对 0..499，基础分位差为 498.002；五个块再分别乘以各自斜率。
    expected_blocks = tuple(498.002 * scale for scale in (1.0, 2.0, 10.0, 4.0, 5.0))
    # 五个块排序后的中值对应斜率 4，不能对整条 DUT 直接计算一次 Vpp。
    assert comparison.dut.value == pytest.approx(4.0 * 498.002)
    # 暴露块级值便于报告和测试确认分块边界没有漂移。
    assert comparison.dut.block_values == pytest.approx(expected_blocks)
    # 1 秒乘以 DUT 的 500 Sa/s，得到每块 500 点。
    assert comparison.dut.block_samples == 500
    # 只有五个完整块进入中位数统计。
    assert comparison.dut.used_samples == 2500
    # 不足 1 秒的 123 点尾部必须明确记录为丢弃样本。
    assert comparison.dut.discarded_samples == 123
    # 五个完整块的极差分别为 499 乘各自斜率，残缺尾部的巨大摆幅不得进入诊断。
    expected_extrema_blocks = tuple(
        499.0 * scale for scale in (1.0, 2.0, 10.0, 4.0, 5.0)
    )
    # 块级极值证据应沿用与稳健 Vpp 完全相同的完整块边界。
    assert comparison.dut.extrema_block_values == pytest.approx(expected_extrema_blocks)
    # 五个完整块的极值 Vpp 中位数对应斜率 4，而不是包含巨大尾部的整记录极差。
    assert comparison.dut_extrema_vpp == pytest.approx(4.0 * 499.0)
    # 公共窗口使用物理秒而非较短记录的样本数。
    assert comparison.comparison_window_s == pytest.approx(1.0)


# 对调长短记录所在侧，防止分块策略只在 DUT 分支生效。
def test_longer_reference_uses_the_same_block_policy_as_dut() -> None:
    """较长记录位于参考侧时也必须执行相同的分块中位数规则。"""

    # 32 点、32 Sa/s 的 DUT 提供精确 1 秒公共比较窗口。
    short_dut = _series(np.arange(32, dtype=np.float64), 32.0)
    # 三个 16 点块具有不同斜率，参考侧采样率 16 Sa/s，因此每块也是 1 秒。
    reference_blocks = tuple(
        scale * np.arange(16, dtype=np.float64)
        for scale in (1.0, 9.0, 3.0)
    )
    # 拼接后的参考记录总长 3 秒，明确长于 DUT。
    long_reference = _series(np.concatenate(reference_blocks), 16.0)

    # 交换长短记录所在侧，确认实现没有把分块规则硬编码为仅处理 DUT。
    comparison = compare_vpp(long_reference, short_dut)

    # 对 0..15，固定分位差为 14.97；三个块的中值对应斜率 3。
    assert comparison.reference.value == pytest.approx(3.0 * 14.97)
    # 参考侧应保留三个块级证据，而短 DUT 仍只保留自身一块。
    assert len(comparison.reference.block_values) == 3
    # 短 DUT 自身仍必须作为单块，不能跟随参考的三块结构。
    assert len(comparison.dut.block_values) == 1


# 该用例锁定主扫描器依赖的函数名称和扁平字段。
def test_public_measurement_and_comparison_api_expose_scanner_fields() -> None:
    """主扫描器应能直接取得单条标量和比较结果的扁平字段。"""

    # 两个 100 点块分别具有 98.802 和 296.406 的稳健 Vpp。
    first_block = np.arange(100, dtype=np.float64)
    # 第二块幅度扩大三倍，使块中位数可通过两个值的算术中点手算。
    second_block = 3.0 * first_block
    # 200 Sa/s 下每个 100 点块的物理时长为 0.5 秒。
    dut = _series(np.concatenate((first_block, second_block)), 200.0)
    # 参考记录正好覆盖 0.5 秒，负责确定公共比较窗口。
    reference = _series(2.0 * first_block, 200.0)

    # 显式块时长入口应返回两个块 Vpp 的中位数，而不是整条记录极差。
    measured = measure_waveform_vpp(dut, block_duration_s=0.5)
    # 100 点线性序列的固定分位差为 98.802，两个缩放块中值对应 2 倍。
    assert measured == pytest.approx(2.0 * 98.802)

    # 扫描器使用的比较入口应自动采用较短参考记录的 0.5 秒窗口。
    comparison = compare_waveform_vpp(reference, dut)
    # 扁平字段避免扫描器依赖内部块证据模型。
    assert comparison.reference_vpp == pytest.approx(2.0 * 98.802)
    # DUT 扁平字段应等于其块中位数，而不是某个单独块。
    assert comparison.dut_vpp == pytest.approx(2.0 * 98.802)
    # 公共时长必须以秒暴露，扫描器无需重新推导样本点数。
    assert comparison.common_duration_s == pytest.approx(0.5)


# 多通道参数化用例锁定两侧完全对称的失败边界。
@pytest.mark.parametrize("multichannel_side", ["reference", "dut"])
def test_compare_vpp_rejects_multichannel_records(multichannel_side: str) -> None:
    """任一输入包含多个通道时都必须拒绝，不能静默选择第一个通道。"""

    # 单通道记录作为另一侧的合法对照输入。
    single_channel = _series(np.arange(32, dtype=np.float64), 1.0e9)
    # 两列采用不同幅度，确保静默取第 0 列会隐藏真实的通道歧义。
    multichannel_values = np.column_stack(
        (
            np.arange(32, dtype=np.float64),
            100.0 * np.arange(32, dtype=np.float64),
        )
    )
    # 多通道记录仍使用合法、等间隔的物理时间轴。
    multichannel = TimeSeries(
        np.arange(32, dtype=np.float64) / 1.0e9,
        multichannel_values,
        1.0e9,
    )
    # 参数化交换两侧，验证校验不偏向参考或 DUT。
    reference = multichannel if multichannel_side == "reference" else single_channel
    # DUT 侧使用与参考侧互补的输入。
    dut = multichannel if multichannel_side == "dut" else single_channel

    # 公共 API 应给出领域错误，而不是继续计算某个未授权通道。
    with pytest.raises(ValueError, match="单通道"):
        # 任一侧的通道歧义都必须在计算分位数前被拒绝。
        compare_vpp(reference, dut)


# 类型参数化用例防止内部属性异常泄漏到公共调用方。
@pytest.mark.parametrize("invalid_side", ["reference", "dut"])
def test_compare_vpp_rejects_non_timeseries_inputs(invalid_side: str) -> None:
    """公开入口应把输入类型错误报告在接口边界。"""

    # 合法一侧继续使用单通道 TimeSeries，隔离本测试关注的类型错误。
    valid_series = _series(np.arange(32, dtype=np.float64), 1.0e9)
    # 普通数组不携带采样率和物理时间轴，不能被当作可比较记录。
    invalid_series = np.arange(32, dtype=np.float64)
    # 参数化交换两侧，确认参考和 DUT 的类型检查完全对称。
    reference = invalid_series if invalid_side == "reference" else valid_series
    # DUT 输入与参考输入保持互补，只让一侧在一次调用中非法。
    dut = invalid_series if invalid_side == "dut" else valid_series

    # 清晰的 TypeError 比内部缺少 channels 属性更能指导调用方修正输入。
    with pytest.raises(TypeError, match="TimeSeries"):
        # 非 TimeSeries 输入应得到稳定的接口类型错误。
        compare_vpp(reference, dut)  # type: ignore[arg-type]
