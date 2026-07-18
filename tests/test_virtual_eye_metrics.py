"""轻量虚拟眼测量的独立数值合同测试。"""

# 新人导向逐语句注释会打断导入块并含较长中文，仅关闭对应排版告警。
# ruff: noqa: E501, I001

# Codex说明(自动生成)： 从 __future__ 导入 annotations，启用较新的类型标注行为，减少运行期导入或前向引用问题。
from __future__ import annotations

# Codex说明(自动生成)： 导入 numpy as np，执行数组、向量化和数值仿真计算。
import numpy as np
# Codex说明(自动生成)： 导入 pytest，提供本文件后续流程需要的库能力。
import pytest

# Codex说明(自动生成)： 从 response_lab.virtual_eye_metrics 导入 _select_innermost_crossings, _select_innermost_crossings_many, measure_virtual_eye_openings，提供本文件后续流程需要的库能力。
from response_lab.virtual_eye_metrics import (
    _select_innermost_crossings,
    _select_innermost_crossings_many,
    measure_virtual_eye_openings,
)


# Codex说明(自动生成)： 定义函数 _v_eye_trace，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _v_eye_trace(x_ui: np.ndarray, center: float, edge: float) -> np.ndarray:
    """生成中心 rail 与两侧 rail 之间的分段线性 V 形轨迹。"""

    # Codex说明(自动生成)： 返回 center + (edge - center) * np.minimum(np.abs(x_ui) / 0....，让调用方取得本函数的处理结果。
    return center + (edge - center) * np.minimum(np.abs(x_ui) / 0.5, 1.0)


# Codex说明(自动生成)： 定义函数 test_nrz_height_and_width_follow_adjacent_rail_contract，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_nrz_height_and_width_follow_adjacent_rail_contract() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.linspace(-0.5, 0.5, 101)
    # Codex说明(自动生成)： 计算并保存 lower，供后续语句继续读取或更新。
    lower = _v_eye_trace(x_ui, -1.0, 1.0)
    # Codex说明(自动生成)： 计算并保存 upper，供后续语句继续读取或更新。
    upper = _v_eye_trace(x_ui, 1.0, -1.0)
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.repeat(np.vstack((lower, upper)), 20, axis=0)
    # Codex说明(自动生成)： 计算并保存 labels，供后续语句继续读取或更新。
    labels = np.repeat(np.array([0, 1]), 20)

    # Codex说明(自动生成)： 计算并保存 result，供后续语句继续读取或更新。
    result = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
        opening_probability=0.0,
    )

    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert result.eye_heights_v == pytest.approx((2.0,))
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert result.eye_widths_ui == pytest.approx((0.5,))


# Codex说明(自动生成)： 定义函数 test_pam4_returns_three_openings_in_low_to_high_order，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_pam4_returns_three_openings_in_low_to_high_order() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.linspace(-0.5, 0.5, 201)
    # Codex说明(自动生成)： 计算并保存 levels，供后续语句继续读取或更新。
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    # 将每个中心电平向两侧镜像，保证三个眼的水平切片都有左右 crossing。
    base = np.vstack([_v_eye_trace(x_ui, level, -level) for level in levels])
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.repeat(base, 30, axis=0)
    # Codex说明(自动生成)： 计算并保存 labels，供后续语句继续读取或更新。
    labels = np.repeat(np.arange(4), 30)

    # Codex说明(自动生成)： 计算并保存 result，供后续语句继续读取或更新。
    result = measure_virtual_eye_openings(traces, x_ui, labels, levels)

    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert result.eye_heights_v == pytest.approx((2.0, 2.0, 2.0))
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert len(result.eye_widths_ui) == 3
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert result.eye_widths_ui[0] == pytest.approx(result.eye_widths_ui[2])
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert all(0.0 < width <= 1.0 for width in result.eye_widths_ui)


# Codex说明(自动生成)： 定义函数 test_eye_height_preserves_negative_overlap，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_eye_height_preserves_negative_overlap() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.array([-0.5, 0.0, 0.5])
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.array(
        [
            [-1.0, 0.4, -1.0],
            [-1.0, 0.5, -1.0],
            [1.0, -0.5, 1.0],
            [1.0, -0.4, 1.0],
        ]
    )
    # Codex说明(自动生成)： 计算并保存 labels，供后续语句继续读取或更新。
    labels = np.array([0, 0, 1, 1])

    # Codex说明(自动生成)： 计算并保存 result，供后续语句继续读取或更新。
    result = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
        opening_probability=0.0,
    )

    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert result.eye_heights_v == pytest.approx((-1.0,))
    # 中心 rail 的中位数顺序已经反转，眼宽应为不可测而不是伪造零。
    assert np.isnan(result.eye_widths_ui[0])


# Codex说明(自动生成)： 定义函数 test_missing_crossings_are_unavailable_instead_of_zero_width，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_missing_crossings_are_unavailable_instead_of_zero_width() -> None:
    """常量 rail 没有水平 crossing，不能冒充已经测得的闭眼。"""

    # 横轴满足固定中心和两侧窗口合同。
    x_ui = np.linspace(-1.0, 1.0, 9)
    # 上下轨各自保持常量，所以任意内侧水平切片都没有 crossing。
    traces = np.repeat(
        np.vstack((np.full(9, -1.0), np.full(9, 1.0))),
        20,
        axis=0,
    )
    # 两个已知发送电平各有二十条轨迹。
    labels = np.repeat(np.array([0, 1]), 20)

    # 眼高仍可在 0 UI 测量，但眼宽缺少边界事件。
    result = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
    )

    # 固定中心上下轨相距 2 V。
    assert result.eye_heights_v == pytest.approx((2.0,))
    # 与本地眼图库 unavailable 语义一致，缺少 crossing 返回 NaN。
    assert np.isnan(result.eye_widths_ui[0])


# Codex说明(自动生成)： 定义函数 test_fixed_center_axis_must_contain_zero_sample，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_fixed_center_axis_must_contain_zero_sample() -> None:
    """公共内核不得把离 0 最近的半采样点静默当成固定眼中心。"""

    # 偶数点对称网格跨过 0，却故意不包含精确的 0 UI 样点。
    x_ui = np.array([-1.0, -0.5, 0.5, 1.0])
    # 两条有限轨迹仅用于进入横轴合同校验。
    traces = np.array([[-1.0] * 4, [1.0] * 4])
    # 每条轨迹对应一个已知 NRZ 电平。
    labels = np.array([0, 1])

    # 没有 0 UI 时必须要求调用方先建立正确眼图网格。
    with pytest.raises(ValueError, match="明确包含固定测量中心"):
        # 不允许用 argmin 悄悄选择 -0.5 UI。
        measure_virtual_eye_openings(
            traces,
            x_ui,
            labels,
            np.array([-1.0, 1.0]),
        )


# Codex说明(自动生成)： 定义函数 test_multiple_crossings_select_the_hit_closest_to_zero，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_multiple_crossings_select_the_hit_closest_to_zero() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.linspace(-1.0, 1.0, 9)
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.array(
        [
            [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
        ]
    )

    # Codex说明(自动生成)： 计算并保存 (left, right)，供后续语句继续读取或更新。
    left, right = _select_innermost_crossings(traces, x_ui, 0.0)

    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert left == pytest.approx(np.array([-0.375, -0.375]))
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert right == pytest.approx(np.array([0.375, 0.375]))


# Codex说明(自动生成)： 定义函数 test_probability_zero_keeps_minority_worst_case_crossing，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_probability_zero_keeps_minority_worst_case_crossing() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.linspace(-0.5, 0.5, 101)
    # Codex说明(自动生成)： 计算并保存 normal_lower，供后续语句继续读取或更新。
    normal_lower = _v_eye_trace(x_ui, -1.0, 1.0)
    # Codex说明(自动生成)： 计算并保存 normal_upper，供后续语句继续读取或更新。
    normal_upper = _v_eye_trace(x_ui, 1.0, -1.0)
    # 少数轨迹更靠近中心，p=0 时必须由它决定确定性边界，不能被多数投票抹掉。
    narrow_lower = _v_eye_trace(x_ui / 0.6, -1.0, 1.0)
    # Codex说明(自动生成)： 计算并保存 narrow_upper，供后续语句继续读取或更新。
    narrow_upper = _v_eye_trace(x_ui / 0.6, 1.0, -1.0)
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.vstack(
        (
            np.repeat(normal_lower[None, :], 49, axis=0),
            narrow_lower,
            np.repeat(normal_upper[None, :], 49, axis=0),
            narrow_upper,
        )
    )
    # Codex说明(自动生成)： 计算并保存 labels，供后续语句继续读取或更新。
    labels = np.array([0] * 50 + [1] * 50)

    # Codex说明(自动生成)： 计算并保存 deterministic，供后续语句继续读取或更新。
    deterministic = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
        opening_probability=0.0,
    )
    # Codex说明(自动生成)： 计算并保存 percentile，供后续语句继续读取或更新。
    percentile = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
        opening_probability=0.05,
    )

    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert deterministic.eye_widths_ui == pytest.approx((0.3,))
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert percentile.eye_widths_ui[0] > deterministic.eye_widths_ui[0]


# Codex说明(自动生成)： 定义函数 test_positive_probability_matches_local_minority_jitter_oracle，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_positive_probability_matches_local_minority_jitter_oracle() -> None:
    """正分位眼宽应复现本地眼图库的少数确定性抖动反例。"""

    # 细网格把 tanh 边沿的线性插值误差压到本地库测试使用的容差以内。
    x_ui = np.linspace(-1.0, 1.0, 801)
    # 九成左 crossing 位于 -0.5 UI，一成少数模式侵入到 -0.3 UI。
    left_crossings = np.concatenate((np.full(90, -0.5), np.full(10, -0.3)))
    # 上轨轨迹在各自左位置上升，并统一在 +0.5 UI 下降。
    upper_traces = np.asarray(
        [
            0.5 * np.tanh((x_ui - left) / 0.01)
            - 0.5 * np.tanh((x_ui - 0.5) / 0.01)
            - 0.5
            for left in left_crossings
        ]
    )
    # 低轨常量只提供已知发送标签与中心 rail，不制造额外 crossing。
    lower_traces = np.full_like(upper_traces, -0.5)
    # crossing 算法读取全部折叠轨迹，顺序不改变分位统计。
    traces = np.vstack((lower_traces, upper_traces))
    # 前一百条属于低轨，后一百条属于高轨。
    labels = np.repeat(np.array([0, 1]), 100)

    # p=5% 应保留占 10% 的 -0.3 UI 少数抖动模式。
    result = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-0.5, 0.5]),
        opening_probability=0.05,
    )

    # 本地 eye_diagram_engine 的权威反例给出约 0.8 UI，而非多数模式的 1 UI。
    assert result.eye_widths_ui == pytest.approx((0.8,), abs=0.03)


# Codex说明(自动生成)： 定义函数 test_eye_width_matches_local_library_without_one_ui_clipping，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_eye_width_matches_local_library_without_one_ui_clipping() -> None:
    """有限轨迹分位数略超 1 UI 时不得被归因层静默改写。"""

    # 构造宽约 1.3 UI 的中心平台，模拟 crossing 分布落在正负 0.65 UI 附近。
    x_ui = np.linspace(-1.0, 1.0, 401)
    # 0.1 UI 的线性肩部让 crossing 位置可由插值精确恢复。
    gate = np.clip((0.65 - np.abs(x_ui)) / 0.1, 0.0, 1.0)
    # 低轨与高轨共享相同的宽平台，只在电压符号上镜像。
    traces = np.repeat(np.vstack((-gate, gate)), 20, axis=0)
    # 每个已知发送电平各二十条轨迹，满足 crossing 最小事件数。
    labels = np.repeat(np.array([0, 1]), 20)

    # 使用默认 1% 经验边界，与本地眼图库正式测量口径相同。
    result = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
    )

    # 本地眼图库不把经验结果强制裁到 1 UI；锁定这一复用语义。
    assert result.eye_widths_ui == pytest.approx((1.3045,), abs=1.0e-12)


# Codex说明(自动生成)： 定义函数 test_height_only_mode_skips_horizontal_crossing_scan，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_height_only_mode_skips_horizontal_crossing_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量眼高排名不应支付 41 条水平切片的眼宽开销。"""

    # 两条 V 形轨道提供可手算眼高，同时具备可测眼宽以防测试输入退化。
    x_ui = np.linspace(-0.5, 0.5, 101)
    # 每个 rail 重复二十次，满足经验边界的最小事件数要求。
    traces = np.repeat(
        np.vstack(
            (
                _v_eye_trace(x_ui, -1.0, 1.0),
                _v_eye_trace(x_ui, 1.0, -1.0),
            )
        ),
        20,
        axis=0,
    )
    # 前二十条属于低轨，后二十条属于高轨。
    labels = np.repeat(np.array([0, 1]), 20)

    # 若实现错误进入水平扫描，替身会立即让测试失败。
    def forbidden_width(*args: object, **kwargs: object) -> float:
        """标记纯眼高模式中任何不应发生的眼宽调用。"""

        # 抛错比只统计次数更直接地证明昂贵路径没有执行。
        raise AssertionError("height-only mode must not measure eye width")

    # 只替换当前模块的单眼宽函数，不影响中心分位眼高计算。
    monkeypatch.setattr(
        "response_lab.virtual_eye_metrics._measure_one_eye_width",
        forbidden_width,
    )
    # 显式关闭眼宽，模拟影响频段页的批量眼高候选扫描。
    result = measure_virtual_eye_openings(
        traces,
        x_ui,
        labels,
        np.array([-1.0, 1.0]),
        measure_width=False,
    )

    # 固定中心上下轨相距二，眼高仍必须完整计算。
    assert result.eye_heights_v == pytest.approx((2.0,))
    # 形状保持一只眼，但 NaN 明确表示该维度为性能而未测量。
    assert np.isnan(result.eye_widths_ui[0])


# Codex说明(自动生成)： 定义函数 test_vectorized_crossing_selector_matches_manual_row_oracle，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_vectorized_crossing_selector_matches_manual_row_oracle() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.array([-1.0, -0.7, -0.3, -0.1, 0.0, 0.2, 0.55, 1.0])
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.array(
        [
            [1.0, -1.0, 1.0, -1.0, -0.2, 1.0, -1.0, 1.0],
            [-2.0, -1.0, 0.2, 0.3, 0.1, -1.0, 1.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        ]
    )

    # Codex说明(自动生成)： 计算并保存 (actual_left, actual_right)，供后续语句继续读取或更新。
    actual_left, actual_right = _select_innermost_crossings(traces, x_ui, 0.0)
    # Codex说明(自动生成)： 计算并保存 (expected_left, expected_right)，供后续语句继续读取或更新。
    expected_left, expected_right = _manual_crossing_oracle(traces, x_ui, 0.0)

    # Codex说明(自动生成)： 调用 np.testing.assert_allclose 检查测试期望，确认实际结果符合预期。
    np.testing.assert_allclose(actual_left, expected_left, equal_nan=True)
    # Codex说明(自动生成)： 调用 np.testing.assert_allclose 检查测试期望，确认实际结果符合预期。
    np.testing.assert_allclose(actual_right, expected_right, equal_nan=True)


# Codex说明(自动生成)： 定义函数 test_batched_threshold_crossings_preserve_shuffled_threshold_order，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_batched_threshold_crossings_preserve_shuffled_threshold_order() -> None:
    """41 条切片跨两个内存批次后仍应逐阈值匹配独立逐行 oracle。"""

    # M=32 对应真实虚拟眼的 65 点横轴。
    x_ui = np.linspace(-1.0, 1.0, 65)
    # 2048 条轨迹使批次上限计算为 30，41 阈值会真实拆成 30+11 两批。
    generator = np.random.default_rng(20260718)
    # 随机连续轨迹几乎不会精确命中阈值，专门核对普通线性 crossing。
    traces = generator.normal(size=(2048, x_ui.size)).cumsum(axis=1)
    # 打乱阈值顺序，能杀死按数值排序后忘记还原行号的批处理错误。
    thresholds = generator.permutation(np.linspace(-1.5, 1.5, 41))

    # 正式批量实现按受控内存分成两个三维 NumPy 批次。
    actual_left, actual_right = _select_innermost_crossings_many(
        traces,
        x_ui,
        thresholds,
    )
    # 独立逐行 oracle 每次只处理一个阈值，不共享批量实现的拼接逻辑。
    expected_pairs = [
        _manual_crossing_oracle(traces, x_ui, float(threshold))
        for threshold in thresholds
    ]
    # 左侧按原始打乱顺序堆叠成 41×2048。
    expected_left = np.vstack([pair[0] for pair in expected_pairs])
    # 右侧使用同一阈值顺序独立堆叠。
    expected_right = np.vstack([pair[1] for pair in expected_pairs])

    # NaN 位置和亚采样 crossing 数值都必须逐元素一致。
    np.testing.assert_allclose(actual_left, expected_left, equal_nan=True)
    # 第二批不能被遗漏、倒序或拼接到错误阈值。
    np.testing.assert_allclose(actual_right, expected_right, equal_nan=True)


# Codex说明(自动生成)： 定义函数 test_exact_zero_is_retained_as_a_boundary_hit，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_exact_zero_is_retained_as_a_boundary_hit() -> None:
    # Codex说明(自动生成)： 计算并保存 x_ui，供后续语句继续读取或更新。
    x_ui = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    # Codex说明(自动生成)： 计算并保存 traces，供后续语句继续读取或更新。
    traces = np.array(
        [
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0, -1.0],
        ]
    )

    # Codex说明(自动生成)： 计算并保存 (left, right)，供后续语句继续读取或更新。
    left, right = _select_innermost_crossings(traces, x_ui, 0.0)

    # exact-zero 即使没有严格异号也是真实的水平切片命中，中心命中会把眼宽闭合到零。
    np.testing.assert_array_equal(left, np.array([0.0, 0.0]))
    # Codex说明(自动生成)： 调用 np.testing.assert_array_equal 检查测试期望，确认实际结果符合预期。
    np.testing.assert_array_equal(right, np.array([0.0, 0.0]))


# Codex说明(自动生成)： 定义函数 _manual_crossing_oracle，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _manual_crossing_oracle(
    traces: np.ndarray,
    x_ui: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """逐行复现眼图库 1% side window 与 signbit crossing 规则。"""

    # Codex说明(自动生成)： 声明并保存 left_values，同时保留类型信息方便维护和静态检查。
    left_values: list[float] = []
    # Codex说明(自动生成)： 声明并保存 right_values，同时保留类型信息方便维护和静态检查。
    right_values: list[float] = []
    # Codex说明(自动生成)： 计算并保存 span，供后续语句继续读取或更新。
    span = float(x_ui[-1] - x_ui[0])
    # Codex说明(自动生成)： 计算并保存 side_bounds，供后续语句继续读取或更新。
    side_bounds = {
        "left": (float(x_ui[0] + 0.01 * span), 0.0),
        "right": (0.0, float(x_ui[-1] - 0.01 * span)),
    }
    # Codex说明(自动生成)： 遍历 traces 中的 trace，逐项执行循环体逻辑。
    for trace in traces:
        # Codex说明(自动生成)： 计算并保存 shifted，供后续语句继续读取或更新。
        shifted = trace - threshold
        # Codex说明(自动生成)： 声明并保存 selected，同时保留类型信息方便维护和静态检查。
        selected: dict[str, float] = {}
        # Codex说明(自动生成)： 遍历 side_bounds.items() 中的 (side, (low_x, high_x))，逐项执行循环体逻辑。
        for side, (low_x, high_x) in side_bounds.items():
            # Codex说明(自动生成)： 计算并保存 indices，供后续语句继续读取或更新。
            indices = np.flatnonzero((x_ui >= low_x) & (x_ui <= high_x))
            # Codex说明(自动生成)： 声明并保存 positions，同时保留类型信息方便维护和静态检查。
            positions: list[float] = []
            # Codex说明(自动生成)： 检查条件 indices.size < 2，根据结果选择后续执行路径。
            if indices.size < 2:
                # Codex说明(自动生成)： 跳过本轮剩余逻辑，直接进入下一轮循环判断。
                continue
            # Codex说明(自动生成)： 计算并保存 start，供后续语句继续读取或更新。
            start = int(indices[0])
            # Codex说明(自动生成)： 计算并保存 stop，供后续语句继续读取或更新。
            stop = int(indices[-1])
            # Codex说明(自动生成)： 计算并保存 signs，供后续语句继续读取或更新。
            signs = np.signbit(shifted[start : stop + 1])
            # Codex说明(自动生成)： 遍历 start + np.flatnonzero(signs[:-1] != signs[1:]) 中的 index，逐项执行循环体逻辑。
            for index in start + np.flatnonzero(signs[:-1] != signs[1:]):
                # Codex说明(自动生成)： 计算并保存 y0，供后续语句继续读取或更新。
                y0 = float(shifted[index])
                # Codex说明(自动生成)： 计算并保存 y1，供后续语句继续读取或更新。
                y1 = float(shifted[index + 1])
                # Codex说明(自动生成)： 计算并保存 fraction，供后续语句继续读取或更新。
                fraction = -y0 / (y1 - y0)
                # Codex说明(自动生成)： 调用 positions.append 更新列表或集合，把当前步骤产生的数据加入结果。
                positions.append(float(x_ui[index] + fraction * (x_ui[index + 1] - x_ui[index])))
            # Codex说明(自动生成)： 检查条件 positions，根据结果选择后续执行路径。
            if positions:
                # Codex说明(自动生成)： 计算并保存 selected[side]，供后续语句继续读取或更新。
                selected[side] = max(positions) if side == "left" else min(positions)
        # Codex说明(自动生成)： 调用 left_values.append 更新列表或集合，把当前步骤产生的数据加入结果。
        left_values.append(selected.get("left", np.nan))
        # Codex说明(自动生成)： 调用 right_values.append 更新列表或集合，把当前步骤产生的数据加入结果。
        right_values.append(selected.get("right", np.nan))
    # Codex说明(自动生成)： 返回 (np.asarray(left_values), np.asarray(right_values))，让调用方取得本函数的处理结果。
    return np.asarray(left_values), np.asarray(right_values)
