"""“影响频段”独立页面的可见交互与展示契约。"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本测试仅关闭对应格式告警。
# ruff: noqa: I001

# 延迟解析类型标注，让 pytest 的 Path fixture 注解不影响测试模块加载。
from __future__ import annotations

# 离屏后端让测试在无桌面会话的 CI 和 PyCharm 测试进程中稳定构造 Qt 控件。
import os
# Path 用于核对信号载荷保留了明确的文件路径类型。
from pathlib import Path

# 若外部尚未指定 Qt 平台，则选择不会弹出真实窗口的离屏实现。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# NumPy 提供确定性的曲线和眼图轨迹，并用独立数组断言真实绘制内容。
import numpy as np
# PyQtGraph 类型用于核对真实轨迹 PlotDataItem、0 UI 中心线和鼠标模式。
import pyqtgraph as pg
# pytest 用于断言非法结果被拒绝，且不破坏上一份有效展示。
import pytest
# QPoint 用页面坐标核对窄窗口中上下绘图区没有重叠。
from PySide6.QtCore import QPoint
# QLabel 查询用于核对三幅图的可见标题和冗余文案边界。
from PySide6.QtWidgets import QLabel

# 项目统一的 QApplication 工厂复用真实字体和应用级配置。
from response_lab.app import _qt_application
# 被测页面通过公开控件状态表达指标切换后的用户界面。
from response_lab.influence_ui import InfluenceBandPage


# 从一幅真实眼图的单个 PlotDataItem 还原 NaN 分隔的多条轨迹。
def _drawn_eye_traces(
    plot: pg.PlotWidget,
    *,
    samples_per_trace: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回画布中的时间矩阵和幅值矩阵，不依赖输入字典。"""

    # 每幅眼图严格只用一个 PlotDataItem 叠加全部轨迹。
    curves = plot.listDataItems()
    # 多于一条表示实现退回了每轨迹一个图形对象。
    assert len(curves) == 1
    # 直接读取 PlotDataItem 提交时的原始 x/y，避免 getData 为显示删掉末尾 NaN。
    x_values = np.asarray(curves[0].xData, dtype=np.float64)
    # yData 同样保留最后一条轨迹的行末分隔位。
    y_values = np.asarray(curves[0].yData, dtype=np.float64)
    # 每条轨迹比实际样点多一个 NaN 分隔位。
    separated_width = samples_per_trace + 1
    # 扁平数组必须能整齐还原为轨迹行。
    assert x_values.size % separated_width == 0
    # 重塑后每行就是一条轨迹加一个分隔点。
    separated_x = x_values.reshape(-1, separated_width)
    # y 使用同样的形状。
    separated_y = y_values.reshape(-1, separated_width)
    # 每行末必须真正是 NaN，否则相邻符号会被一条伪线连起。
    assert np.all(np.isnan(separated_x[:, -1]))
    # y 分隔位同样必须断开。
    assert np.all(np.isnan(separated_y[:, -1]))
    # 剔除分隔列后返回可与预期轨迹逐点比较的数组。
    return separated_x[:, :-1], separated_y[:, :-1]


# Vpp 与眼图指标需要不同输入，切换指标时不能同时展示两套冗余参数。
def test_metric_selection_switches_visible_inputs() -> None:
    """页面应严格提供约定选项，并只展示当前指标需要的输入。"""

    # 构造真实 QApplication，保证组合框信号与显隐状态按 Qt 路径执行。
    application = _qt_application()
    # 创建独立页面，不依赖主窗口即可验证公开交互契约。
    page = InfluenceBandPage()

    # 指标文案和顺序直接来自用户确认规格，不能添加解释性后缀。
    assert [page.metric_combo.itemText(index) for index in range(page.metric_combo.count())] == [
        "Vpp",
        "眼高",
        "眼宽",
    ]
    # 调制格式同样只保留面向业务的两个名称。
    assert [
        page.modulation_combo.itemText(index)
        for index in range(page.modulation_combo.count())
    ] == ["NRZ", "PAM4"]
    # 频段宽度对三个指标都生效，页面默认使用 100 MHz。
    assert page.band_width_spin.value() == pytest.approx(100.0)
    # 输入框直接展示 MHz，避免用户把页面值误解为 Hz。
    assert page.band_width_spin.suffix() == " MHz"
    # crossing 端点保护要求 M 至少为 3，页面不能提交必然不可测的 M=2。
    assert page.m_spin.minimum() == 3
    # Np 已由拟合脉冲样点数和 M 自动推导，页面不再显示重复输入。
    assert "Np" not in [label.text() for label in page.findChildren(QLabel)]
    # 默认 Vpp 模式需要两份原始数据路径。
    assert not page.vpp_paths_panel.isHidden()
    # Vpp 不依赖 M 或调制格式，整组眼参数必须隐藏。
    assert page.eye_parameters_panel.isHidden()
    # Vpp 模式不展示三幅眼图，避免空图占据页面空间。
    assert page.eye_plots_panel.isHidden()
    # 两个文件按钮必须有不同的读屏名称。
    assert page.reference_data_row.choose_button.accessibleName() == "选择参考数据文件"
    # DUT 按钮不与参考按钮重复名称。
    assert page.dut_data_row.choose_button.accessibleName() == "选择DUT数据文件"
    # 四幅图在读屏树中也必须可独立识别。
    assert [
        plot.accessibleName()
        for plot in (
            page.reference_plot,
            page.before_plot,
            page.after_plot,
            page.impact_plot,
        )
    ] == ["参考眼图", "补偿前眼图", "补偿后眼图", "候选影响曲线"]

    # 切换到眼高，通过真实组合框信号触发页面状态更新。
    page.metric_combo.setCurrentText("眼高")
    # 处理排队的布局事件，使测试读取最终显隐状态。
    application.processEvents()

    # 眼图指标不再显示 Vpp 专属的两份数据路径。
    assert page.vpp_paths_panel.isHidden()
    # 调制与 M 在眼图模式下必须可见。
    assert not page.eye_parameters_panel.isHidden()
    # 频段宽度是公共扫描参数，切换指标后仍须可见。
    assert not page.band_width_spin.isHidden()
    # 三幅对比图也应随眼图指标一并出现。
    assert not page.eye_plots_panel.isHidden()

    # 释放页面拥有的图形对象，避免影响后续 Qt 测试。
    page.close()
    # 冲刷关闭事件，保证测试在完整套件中可重复。
    application.processEvents()


# 请求载荷应只包含当前可见输入，防止隐藏字段的旧值污染后台分析。
def test_analysis_request_contains_only_active_metric_inputs(tmp_path: Path) -> None:
    """眼图请求忽略数据路径，Vpp 请求忽略调制和 M。"""

    # 构造真实 Qt 应用，使按钮点击通过生产信号槽路径发出请求。
    application = _qt_application()
    # 页面信号是本测试观察的公开边界。
    page = InfluenceBandPage()
    # 两份路径先写入页面，用来证明眼图模式不会误带隐藏的 Vpp 输入。
    reference_path = tmp_path / "reference.csv"
    # DUT 路径与参考路径保持不同，避免错误复用同一路径仍能通过。
    dut_path = tmp_path / "dut.csv"
    # 通过公开 API 设置路径，不依赖文件对话框内部实现。
    page.set_vpp_paths(reference_path, dut_path)
    # 选择眼宽覆盖第三个指标分支。
    page.metric_combo.setCurrentText("眼宽")
    # PAM4 覆盖非默认调制选项。
    page.modulation_combo.setCurrentText("PAM4")
    # 频段宽度使用带小数的 MHz，杀死遗漏单位换算或整数截断的实现。
    page.band_width_spin.setValue(125.5)
    # 每 UI 样点数设置为 32。
    page.m_spin.setValue(32)
    # 列表保存每次信号发出的独立参数快照。
    captured: list[dict[str, object]] = []
    # 连接公开信号，不调用页面私有槽函数。
    page.analysis_requested.connect(captured.append)

    # 真实点击主按钮，触发第一份眼宽请求。
    page.start_button.click()
    # 处理可能排队的信号和控件事件。
    application.processEvents()

    # 眼宽请求只携带调制和 M，同时把隐藏的数据路径明确置空。
    assert captured == [
        {
            "metric": "eye_width",
            "band_width_hz": 125_500_000.0,
            "modulation": "pam4",
            "m": 32,
            "reference_data_path": None,
            "dut_data_path": None,
        }
    ]
    # 后续控件变化不能回写已经发出的第一份字典快照。
    page.m_spin.setValue(40)
    # 已发送请求仍应保留点击瞬间的 M=32。
    assert captured[0]["m"] == 32

    # 切回 Vpp，验证另一套有效字段边界。
    page.metric_combo.setCurrentText("Vpp")
    # 真实点击发出第二份 Vpp 请求。
    page.start_button.click()
    # 冲刷 Qt 事件后读取第二份载荷。
    application.processEvents()

    # Vpp 请求保留两条 Path，并把隐藏的眼参数明确置空。
    assert captured[1] == {
        "metric": "vpp",
        "band_width_hz": 125_500_000.0,
        "modulation": None,
        "m": None,
        "reference_data_path": reference_path,
        "dut_data_path": dut_path,
    }

    # 关闭页面释放 Qt 图形资源。
    page.close()
    # 完成销毁事件，保持测试隔离。
    application.processEvents()


# 展示 API 必须把独立结果真实画入曲线、轨迹眼图和候选列表，而非只缓存字典。
def test_render_result_draws_shared_axis_eye_traces_and_candidate_scores() -> None:
    """三幅眼图应保留角色轨迹、共轴和 0 UI 中心线。"""

    # 创建真实 QApplication，使 PyQtGraph 完成 ViewBox 和 PlotDataItem 初始化。
    application = _qt_application()
    # 独立页面不依赖主窗口即可渲染后台计算结果。
    page = InfluenceBandPage()
    # 眼高模式显示三幅对比图。
    page.metric_combo.setCurrentText("眼高")
    # 给页面足够尺寸，确保三幅 ViewBox 能完成相同范围设置。
    page.resize(1200, 760)
    # 显示页面触发真实布局和绘图场景创建。
    page.show()
    # 处理首次布局事件后再提交结果。
    application.processEvents()

    # 三个频点用于独立验证 Hz 到 GHz 的显示换算。
    frequency_hz = np.array([1.0e9, 1.1e9, 1.2e9], dtype=np.float64)
    # 三种模式使用互不相同的数列，顺序交换会被 yData 断言发现。
    scores = {
        "magnitude": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "phase": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        "both": np.array([7.0, 8.0, 9.0], dtype=np.float64),
    }
    # M=2 对应附件中的 2*M+1=5 点眼图时窗。
    time_ui = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    # 参考轨迹整体位于 10 V 区域，可明确识别角色串位。
    reference_traces = np.arange(10, dtype=np.float64).reshape(2, 5) + 10.0
    # 补偿前轨迹使用 20 V 偏置。
    before_traces = np.arange(10, dtype=np.float64).reshape(2, 5) + 20.0
    # 补偿后轨迹使用 30 V 偏置。
    after_traces = np.arange(10, dtype=np.float64).reshape(2, 5) + 30.0
    # 轻量字典直接携带轨迹，不再分箱成像素强度。
    result = {
        "frequency_hz": frequency_hz,
        "scores": scores,
        "candidates": ["1.0–1.1 GHz · 幅度", "1.1–1.2 GHz · 幅相"],
        "eyes": {
            "time_ui": time_ui,
            "amplitude_range_v": (0.0, 40.0),
            "reference": {"traces_v": reference_traces},
            "before": {"traces_v": before_traces},
            "after": {"traces_v": after_traces},
        },
    }

    # 通过公开 API 渲染一份完整眼高结果。
    page.render_result(result)
    # 处理 PlotDataItem、中心线和 ViewBox 范围更新事件。
    application.processEvents()

    # 影响曲线必须严格以面向用户的幅度、相位、幅相顺序出现。
    impact_curves = page.impact_plot.listDataItems()
    # 曲线名称同时驱动图例，不能使用内部英文键代替。
    assert [curve.name() for curve in impact_curves] == ["幅度", "相位", "幅相"]
    # 每条曲线的横坐标都应从 Hz 显示为 GHz。
    for curve in impact_curves:
        # 读取真实 PlotDataItem 数据，而不是再次读取输入字典。
        x_values, _y_values = curve.getData()
        # 1.0e9 Hz 必须显示为 1.0 GHz。
        np.testing.assert_allclose(x_values, np.array([1.0, 1.1, 1.2]))
    # 三条 yData 分别对应三个互异输入数列。
    np.testing.assert_allclose(impact_curves[0].getData()[1], scores["magnitude"])
    # 相位曲线不能误用幅度或幅相得分。
    np.testing.assert_allclose(impact_curves[1].getData()[1], scores["phase"])
    # 幅相曲线必须保留联合模式得分。
    np.testing.assert_allclose(impact_curves[2].getData()[1], scores["both"])
    # 候选列表应按后台排序原样展示两项。
    assert [
        page.candidate_list.item(index).text()
        for index in range(page.candidate_list.count())
    ] == ["1.0–1.1 GHz · 幅度", "1.1–1.2 GHz · 幅相"]

    # 从三幅真实图层还原轨迹，验证参考/补偿前/补偿后没有串位。
    expected_traces = (reference_traces, before_traces, after_traces)
    # 画布顺序与页面三列角色一致。
    for plot, expected in zip(
        (page.reference_plot, page.before_plot, page.after_plot),
        expected_traces,
        strict=True,
    ):
        # 单一 PlotDataItem 中的 x 轨应对每条轨迹复用公共时间。
        drawn_time, drawn_traces = _drawn_eye_traces(
            plot,
            samples_per_trace=time_ui.size,
        )
        # 两条轨迹都使用 -1 到 +1 UI。
        np.testing.assert_array_equal(drawn_time, np.tile(time_ui, (2, 1)))
        # 真实 yData 必须逐点等于该角色输入。
        np.testing.assert_array_equal(drawn_traces, expected)
        # 眼图轨迹不得继承通用曲线的自动峰值降采样，否则 NaN 分段会被压成伪竖线。
        trace_item = plot.listDataItems()[0]
        # 原始 2 UI 轨迹按一倍采样完整提交。
        assert trace_item.opts["downsample"] == 1
        # 禁止图层随视窗宽度再次自动降采样。
        assert trace_item.opts["autoDownsample"] is False
        # 禁止视窗裁剪在 NaN 两侧重组多段轨迹。
        assert trace_item.opts["clipToView"] is False
        # 每幅图都有一条 0 UI 主光标参考线。
        center_lines = [
            item
            for item in plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine)
        ]
        # 不多画也不漏画中心线。
        assert len(center_lines) == 1
        # 中心线真正位于 0 UI。
        assert center_lines[0].value() == 0.0

    # 三幅图的最终 ViewBox 范围必须一致，禁止各自自动缩放产生假改善。
    view_ranges = [
        np.asarray(plot.viewRange(), dtype=np.float64)
        for plot in (page.reference_plot, page.before_plot, page.after_plot)
    ]
    # x 轴严格固定在 -1 到 +1 UI。
    np.testing.assert_allclose(view_ranges[0][0], np.array([-1.0, 1.0]))
    # y 轴使用后台给出的共同范围。
    np.testing.assert_allclose(view_ranges[0][1], np.array([0.0, 40.0]))
    # 补偿前坐标与参考坐标一致。
    np.testing.assert_allclose(view_ranges[1], view_ranges[0])
    # 补偿后坐标同样与参考坐标一致。
    np.testing.assert_allclose(view_ranges[2], view_ranges[0])

    # 真实拖动或缩放参考图后，另两幅图也必须继续共用坐标。
    page.reference_plot.setXRange(-0.5, 0.5, padding=0.0)
    # y 轴同时改成与初始范围不同的区间，覆盖两个链接方向。
    page.reference_plot.setYRange(5.0, 35.0, padding=0.0)
    # ViewBox 链接通过 Qt 信号传递，先处理排队事件。
    application.processEvents()
    # 从真实 ViewBox 重新读取交互后范围。
    linked_ranges = [
        np.asarray(plot.viewRange(), dtype=np.float64)
        for plot in (page.reference_plot, page.before_plot, page.after_plot)
    ]
    # 补偿前图在交互后仍与参考图严格一致。
    np.testing.assert_allclose(linked_ranges[1], linked_ranges[0])
    # 补偿后图也不能保留过期的独立范围。
    np.testing.assert_allclose(linked_ranges[2], linked_ranges[0])

    # 三列可见标题必须严格保持用户确认的三个短名称。
    eye_titles = [
        label.text() for label in page.findChildren(QLabel, "eyePlotTitle")
    ]
    # 标题不添加技术后缀或括号说明。
    assert eye_titles == ["参考", "补偿前", "补偿后"]
    # 汇总全部可见标签，检查界面没有重复展示技术边界词。
    visible_text = "\n".join(
        label.text() for label in page.findChildren(QLabel) if not label.isHidden()
    )
    # 界面业务文案不能出现用户明确要求移除的词。
    assert "虚拟" not in visible_text
    # 英文缩写也不应作为可见冗余标注出现。
    assert "ISI" not in visible_text

    # 公开清空 API 应移除当前曲线、眼图轨迹、中心线和候选列表。
    page.clear_result()
    # 影响曲线全部被清空。
    assert page.impact_plot.listDataItems() == []
    # 候选列表不保留过期推荐。
    assert page.candidate_list.count() == 0
    # 三幅眼图场景中都不应残留轨迹或 0 UI 标线。
    for plot in (page.reference_plot, page.before_plot, page.after_plot):
        # PlotDataItem 必须已全部移除。
        assert plot.listDataItems() == []
        # InfiniteLine 同样不得残留。
        assert not any(
            isinstance(item, pg.InfiniteLine) for item in plot.getPlotItem().items
        )

    # 关闭页面释放 ViewBox 和图形场景。
    page.close()
    # 处理 Qt 销毁事件，防止影响其他 GUI 测试。
    application.processEvents()


# NaN 断点与真实零值必须在实际曲线中保持可区分，同时不能破坏视窗范围。
def test_render_result_draws_invalid_scores_as_finite_connected_gaps() -> None:
    """页面应接受 NaN、拒绝 Inf，并用断线与诊断表达不可解析候选。"""

    # 构造真实 Qt 页面以核对 PlotDataItem 的连接策略和 ViewBox 范围。
    application = _qt_application()
    # 默认 Vpp 页面足以展示影响曲线，不需要构造三幅眼图。
    page = InfluenceBandPage()
    # 三个频率中心包含一个中间不可解析点，便于观察真正断线。
    frequency_hz = np.array([1.0e9, 1.1e9, 1.2e9], dtype=np.float64)
    # 幅度曲线首点是真实零，中间点是无效 NaN，末点重新有效。
    magnitude_scores = np.array([0.0, np.nan, 2.0], dtype=np.float64)
    # 其余两模式保持全有限，证明一个断点不会清空整张图。
    scores = {
        "magnitude": magnitude_scores,
        "phase": np.array([-1.0, 0.5, 1.0], dtype=np.float64),
        "both": np.array([-0.5, 1.0, 3.0], dtype=np.float64),
    }
    # 显式掩码与 NaN 位置一致，真实零位置仍为 True。
    valid_masks = {
        "magnitude": np.array([True, False, True]),
        "phase": np.array([True, True, True]),
        "both": np.array([True, True, True]),
    }
    # 页面协议同时携带无效数量、短诊断和只含有效项的候选列表。
    result = {
        "frequency_hz": frequency_hz,
        "scores": scores,
        "valid_masks": valid_masks,
        "invalid_count": 1,
        "diagnostic": "1 个候选不可解析，曲线以断点表示",
        "candidates": ["1.0–1.1 GHz · 幅度"],
    }

    # 渲染包含 NaN 的完整结果。
    page.render_result(result)
    # 处理 PlotItem 和 ViewBox 更新事件。
    application.processEvents()

    # 三种模式仍各有一条真实曲线。
    curves = page.impact_plot.listDataItems()
    # 稳定顺序仍是幅度、相位、幅相。
    assert [curve.name() for curve in curves] == ["幅度", "相位", "幅相"]
    # 第一个有效零点没有被改成 NaN 或删除。
    assert curves[0].getData()[1][0] == 0.0
    # 中间不可解析候选在真实 yData 中保留 NaN。
    assert np.isnan(curves[0].getData()[1][1])
    # PlotCurveItem 只连接连续有限点，跨 NaN 位置形成断线。
    assert curves[0].curve.opts["connect"] == "finite"
    # 自动选择后的两个轴范围必须全部有限。
    view_range = np.asarray(page.impact_plot.viewRange(), dtype=np.float64)
    # NaN 不得让视窗边界变成 NaN 或 Inf。
    assert np.all(np.isfinite(view_range))
    # 有效最小值 -1 仍位于纵轴范围内。
    assert view_range[1, 0] <= -1.0
    # 有效最大值 3 同样没有被 NaN 自动范围遗漏。
    assert view_range[1, 1] >= 3.0
    # 独立诊断标签明确当前曲线存在不可解析候选。
    assert page.diagnostic_label.text() == "1 个候选不可解析，曲线以断点表示"
    # 有诊断时标签实际可见。
    assert not page.diagnostic_label.isHidden()
    # 候选列表只展示调用方提供的有效项。
    assert page.candidate_list.count() == 1

    # 构造包含 Inf 的新结果以验证事务渲染边界。
    invalid_result = dict(result)
    # 复制得分映射，避免修改已显示的基线字典。
    invalid_result["scores"] = dict(scores)
    # Inf 不是“不可解析断点”的合法表示，必须拒绝整份结果。
    invalid_result["scores"]["both"] = np.array([-0.5, np.inf, 3.0])
    # 错误输入应在清空旧曲线前返回。
    with pytest.raises(ValueError):
        # 通过公开 API 提交非法无穷值。
        page.render_result(invalid_result)
    # 旧幅度曲线中的 NaN 断点仍然保留。
    assert np.isnan(page.impact_plot.listDataItems()[0].getData()[1][1])
    # 旧诊断同样没有被半份新结果覆盖。
    assert page.diagnostic_label.text() == "1 个候选不可解析，曲线以断点表示"

    # 关闭页面释放绘图场景。
    page.close()
    # 冲刷 Qt 销毁事件，保持测试隔离。
    application.processEvents()


# 构造一份可重用的完整眼图结果，供失败原子性和状态失效测试使用。
def _complete_eye_result() -> dict[str, object]:
    """返回三图共时轴、三模式等长的有效展示字典。"""

    # 三个频点便于在失败后核对旧曲线是否完整保留。
    frequency_hz = np.array([1.0e9, 1.1e9, 1.2e9], dtype=np.float64)
    # M=2 时共时轴含 5 个对称递增点。
    time_ui = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    # 三幅眼图分别使用不同数值的两条轨迹。
    eyes = {
        "time_ui": time_ui,
        "amplitude_range_v": (-4.0, 4.0),
        "reference": {"traces_v": np.full((2, 5), 1.0, dtype=np.float64)},
        "before": {"traces_v": np.full((2, 5), 2.0, dtype=np.float64)},
        "after": {"traces_v": np.full((2, 5), 3.0, dtype=np.float64)},
    }
    # 返回值只使用页面轻量协议，不实例化任何扫描模型。
    return {
        "frequency_hz": frequency_hz,
        "scores": {
            "magnitude": np.array([1.0, 2.0, 3.0]),
            "phase": np.array([4.0, 5.0, 6.0]),
            "both": np.array([7.0, 8.0, 9.0]),
        },
        "candidates": ["1.0–1.1 GHz · 幅度", "1.1–1.2 GHz · 幅相"],
        "eyes": eyes,
    }


# render_result 应先完整验证新数据，任一错误都不能留下半份新结果。
def test_render_result_is_transactional_and_rejects_invalid_eye_traces() -> None:
    """非法曲线、候选或眼图应保留上一份有效结果。"""

    # 构造真实 Qt 页面并先提交一份可见的基线结果。
    application = _qt_application()
    # 独立页面足以验证展示原子性。
    page = InfluenceBandPage()
    # 切到眼高使三幅轨迹图处于当前结果范围。
    page.metric_combo.setCurrentText("眼高")
    # 有效结果用作所有反例的不可破坏快照。
    valid_result = _complete_eye_result()
    # 真实渲染基线。
    page.render_result(valid_result)

    # 第二种模式长度错误会在曲线验证阶段失败。
    bad_score = dict(valid_result)
    # 复制 scores 映射，避免修改基线字典。
    bad_score["scores"] = dict(valid_result["scores"])
    # 相位得分故意少一点。
    bad_score["scores"]["phase"] = np.array([8.0, 9.0])
    # 单个字符串不能被拆成候选字符。
    bad_candidates = dict(valid_result)
    # 使用明确非法类型覆盖候选列表。
    bad_candidates["candidates"] = "1.0 GHz"
    # 轨迹列数不同会破坏公共 2 UI 时轴，页面必须拒绝。
    bad_shape = dict(valid_result)
    # 层层复制眼图映射，保留其他角色不变。
    bad_shape["eyes"] = dict(valid_result["eyes"])
    # 只把补偿前轨迹改成每条 4 点，与公共 5 点时轴不同。
    bad_shape["eyes"]["before"] = {
        "traces_v": np.ones((2, 4), dtype=np.float64),
    }
    # 不是 -1 到 +1 UI 的时间范围同样无法按附件算法解释。
    bad_range = dict(valid_result)
    # 单独复制眼图层。
    bad_range["eyes"] = dict(valid_result["eyes"])
    # 公共时间轴故意改成 0–2 UI。
    bad_range["eyes"]["time_ui"] = np.linspace(0.0, 2.0, 5)
    # 候选文字也必须遵守用户确认的简洁文案边界。
    bad_copy = dict(valid_result)
    # 显示边界遇到禁用术语时应拒绝整份新结果。
    bad_copy["candidates"] = ["虚拟 ISI 候选"]
    # 结果摘要也不能重新带回已经约定删除的解释性后缀。
    bad_summary = dict(valid_result)
    # 摘要使用另一个禁用词，独立覆盖摘要入口。
    bad_summary["summary"] = "眼高（工程近似）"

    # 六个失败阶段都应循同一原子性契约。
    for invalid_result in (
        bad_score,
        bad_candidates,
        bad_shape,
        bad_range,
        bad_copy,
        bad_summary,
    ):
        # 错误应传回调用方，而不是画出部分数据。
        with pytest.raises(ValueError):
            # 所有反例都通过公开展示 API 提交。
            page.render_result(invalid_result)
        # 三条基线曲线仍完整存在。
        assert [curve.name() for curve in page.impact_plot.listDataItems()] == [
            "幅度",
            "相位",
            "幅相",
        ]
        # 旧频率轴仍是 1.0–1.2 GHz，没有半份新曲线。
        np.testing.assert_allclose(
            page.impact_plot.listDataItems()[0].getData()[0],
            np.array([1.0, 1.1, 1.2]),
        )
        # 两项旧候选保持不变。
        assert page.candidate_list.count() == 2
        # 异常后参考画布仍只有一个轨迹 PlotDataItem。
        reference_time, reference_traces = _drawn_eye_traces(
            page.reference_plot,
            samples_per_trace=5,
        )
        # 真实时轴仍是旧快照的 -1 到 +1 UI。
        np.testing.assert_array_equal(
            reference_time,
            np.tile(np.linspace(-1.0, 1.0, 5), (2, 1)),
        )
        # 真实轨迹数据保留旧快照的全 1 矩阵。
        np.testing.assert_array_equal(reference_traces, np.ones((2, 5)))

    # 关闭页面释放图形场景。
    page.close()
    # 处理销毁事件。
    application.processEvents()


# 请求参数变化必须使旧结果失效，候选点选则使用独立索引信号。
def test_request_change_candidate_selection_and_busy_state_are_public(
    tmp_path: Path,
) -> None:
    """参数、路径和候选交互应通过稳定公开 API 与主窗通信。"""

    # 真实 Qt 信号用于验证主窗可以安全做请求版本管理。
    application = _qt_application()
    # 页面自身拥有全部信号和状态。
    page = InfluenceBandPage()
    # 列表记录每次请求条件变化。
    request_changes: list[bool] = []
    # 公开无载荷信号只需要告知主窗递增版本。
    page.request_changed.connect(lambda: request_changes.append(True))
    # 候选选择只发出后台结果顺序索引。
    selected_rows: list[int] = []
    # 连接公开选择信号。
    page.candidate_selected.connect(selected_rows.append)

    # 先在眼高模式渲染一份有效结果。
    page.metric_combo.setCurrentText("眼高")
    # 指标变化事件不属于本次计数基线。
    request_changes.clear()
    # 渲染完整基线。
    page.render_result(_complete_eye_result())
    # 改变公共频段宽度会使当前快照不再对应新条件。
    page.band_width_spin.setValue(page.band_width_spin.value() + 10.0)
    # 处理控件信号。
    application.processEvents()
    # 主窗必须收到一次请求变化通知。
    assert request_changes == [True]
    # 过期曲线必须立即清空。
    assert page.impact_plot.listDataItems() == []
    # 过期候选也不得保留。
    assert page.candidate_list.count() == 0

    # 重新渲染后点选第二个候选。
    page.render_result(_complete_eye_result())
    # 通过真实列表交互路径选择行。
    page.candidate_list.setCurrentRow(1)
    # 处理 currentRowChanged 信号。
    application.processEvents()
    # 主窗得到唯一的结果索引 1。
    assert selected_rows == [1]

    # 路径变化同样是可导致 Vpp 结果失效的请求条件。
    request_changes.clear()
    # 通过公开辅助一次设置两份不同路径。
    page.set_vpp_paths(tmp_path / "reference.csv", tmp_path / "dut.csv")
    # 每个真正变化的路径各发出一次失效信号。
    assert request_changes == [True, True]
    # 第一次路径变化已经清空旧影响曲线。
    assert page.impact_plot.listDataItems() == []
    # 候选列表同样不保留旧行。
    assert page.candidate_list.count() == 0

    # 忙状态禁止重复提交和参数修改。
    page.set_busy(True)
    # 主操作按钮忙时不可点击。
    assert not page.start_button.isEnabled()
    # 指标下拉同样锁定，避免在途任务语义变化。
    assert not page.metric_combo.isEnabled()
    # 公共频段宽度在扫描期间不可修改。
    assert not page.band_width_spin.isEnabled()
    # 忙状态使用简短动作文案。
    assert page.start_button.text() == "分析中…"
    # 恢复后所有参数和主按钮重新可用。
    page.set_busy(False)
    # 主按钮回到初始动作。
    assert page.start_button.isEnabled()
    # 可见文字恢复为用户确认的名称。
    assert page.start_button.text() == "开始分析"
    # 分析结束后公共频段宽度恢复可编辑。
    assert page.band_width_spin.isEnabled()

    # 公开绘图列表返回稳定顺序，主窗不需要取猜内部属性。
    assert page.plots() == (
        page.impact_plot,
        page.vpp_waveform_plot,
        page.reference_plot,
        page.before_plot,
        page.after_plot,
    )
    # 切到框选缩放模式。
    page.set_mouse_mode("zoom")
    # 五幅图都必须使用 PyQtGraph 的 RectMode。
    assert all(
        plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
        for plot in page.plots()
    )
    # 再切回平移模式。
    page.set_mouse_mode("pan")
    # 所有图使用 PanMode。
    assert all(
        plot.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode
        for plot in page.plots()
    )
    # 未知模式不能静默退回某个默认交互。
    with pytest.raises(ValueError):
        # 通过公开 API 提交非法键。
        page.set_mouse_mode("unknown")
    # 重置所有页内图表，空图与已清空图都应安全返回。
    page.reset_view()

    # 关闭页面并处理销毁事件。
    page.close()
    # 冲刷 Qt 事件队列。
    application.processEvents()


# Vpp 模式需要用一幅同轴图对比原始、补偿前和补偿后波形。
def test_vpp_mode_renders_independent_time_axis_waveforms() -> None:
    """Vpp 波形可不同长且不同采样率，眼图模式则隐藏该区。"""

    # 创建默认 Vpp 页面。
    application = _qt_application()
    # 页面初始指标就是 Vpp。
    page = InfluenceBandPage()
    # Vpp 模式必须显示波形对比区。
    assert not page.vpp_waveform_panel.isHidden()
    # 参考、DUT 和补偿后记录故意使用不同长度。
    waveforms = {
        "reference": {
            "time_s": np.array([0.0, 1.0e-9, 2.0e-9]),
            "values": np.array([0.0, 1.0, 0.0]),
        },
        "before": {
            "time_s": np.array([0.0, 0.5e-9, 1.0e-9, 1.5e-9]),
            "values": np.array([0.0, 0.5, -0.5, 0.0]),
        },
        "after": {
            "time_s": np.array([0.0, 0.5e-9, 1.0e-9, 1.5e-9]),
            "values": np.array([0.0, 0.8, -0.8, 0.0]),
        },
    }
    # 在基础影响曲线协议上加入独立波形映射。
    result = _complete_eye_result()
    # Vpp 结果不需要三幅眼图。
    result.pop("eyes")
    # 波形映射保留各自时间轴。
    result["waveforms"] = waveforms
    # 通过公开 API 绘制 Vpp 结果。
    page.render_result(result)
    # 读取真实 PlotDataItem。
    curves = page.vpp_waveform_plot.listDataItems()
    # 参考是可选第三条曲线，当提供时使用稳定顺序。
    assert [curve.name() for curve in curves] == ["参考", "补偿前", "补偿后"]
    # 页面只把秒换成 ns 显示，不对不同采样网格重采样。
    np.testing.assert_allclose(curves[0].getData()[0], np.array([0.0, 1.0, 2.0]))
    # DUT 波形仍保留 0.5 ns 间隔。
    np.testing.assert_allclose(curves[1].getData()[0], np.array([0.0, 0.5, 1.0, 1.5]))
    # 补偿后 y 数据来自独立数组。
    np.testing.assert_allclose(curves[2].getData()[1], waveforms["after"]["values"])

    # 切到眼宽后 Vpp 波形区必须隐藏。
    page.metric_combo.setCurrentText("眼宽")
    # 处理显隐和失效信号。
    application.processEvents()
    # 眼图指标不同时显示原始波形。
    assert page.vpp_waveform_panel.isHidden()

    # 关闭页面释放曲线。
    page.close()
    # 冲刷销毁事件。
    application.processEvents()


# 候选点选后只更新大型详情图，扫描总览和当前行必须保持。
def test_render_selection_preserves_scan_overview_and_selected_row() -> None:
    """局部详情刷新不应清空影响曲线或递归重置候选索引。"""

    # 构造真实 Qt 页面。
    application = _qt_application()
    # 选择眼高以验证三幅详情图更新。
    page = InfluenceBandPage()
    # 切换到眼图模式。
    page.metric_combo.setCurrentText("眼高")
    # 先渲染完整扫描总览。
    page.render_result(_complete_eye_result())
    # 当前用户选中第二行。
    page.candidate_list.setCurrentRow(1)
    # 保存三条影响曲线的真实 y 数据。
    old_scores = [curve.getData()[1].copy() for curve in page.impact_plot.listDataItems()]

    # 新详情使用不同幅值轨迹，可明确证明 PlotDataItem 已被替换。
    detail_eyes = {
        "time_ui": np.linspace(-1.0, 1.0, 5),
        "amplitude_range_v": (0.0, 7.0),
        "reference": {"traces_v": np.full((2, 5), 4.0)},
        "before": {"traces_v": np.full((2, 5), 5.0)},
        "after": {"traces_v": np.full((2, 5), 6.0)},
    }
    # 通过专用公开 API 只提交当前候选详情。
    page.render_selection({"eyes": detail_eyes, "summary": "1.1–1.2 GHz · 幅相"})
    # 处理图层和标签更新事件。
    application.processEvents()

    # 三条扫描影响曲线仍全部存在。
    assert len(page.impact_plot.listDataItems()) == 3
    # 逐条曲线保持刷新前的数值。
    for curve, old_score in zip(page.impact_plot.listDataItems(), old_scores, strict=True):
        # 读取真实 PlotDataItem 数据。
        np.testing.assert_array_equal(curve.getData()[1], old_score)
    # 候选列表仍有两行。
    assert page.candidate_list.count() == 2
    # 当前行号仍为 1，不会因重建列表变成 -1。
    assert page.candidate_list.currentRow() == 1
    # 当前候选摘要已显示。
    assert page.selection_summary.text() == "1.1–1.2 GHz · 幅相"
    # 从真实图层还原补偿后轨迹。
    after_time, after_traces = _drawn_eye_traces(
        page.after_plot,
        samples_per_trace=5,
    )
    # 详情刷新后两条轨迹仍共用 -1 到 +1 UI 时轴。
    np.testing.assert_array_equal(
        after_time,
        np.tile(np.linspace(-1.0, 1.0, 5), (2, 1)),
    )
    # 轨迹内容确实来自新候选的全 6 矩阵。
    np.testing.assert_array_equal(after_traces, np.full((2, 5), 6.0))
    # 旧中心线已被清空，新详情只有一条 0 UI 线。
    assert sum(
        isinstance(item, pg.InfiniteLine)
        for item in page.after_plot.getPlotItem().items
    ) == 1

    # 关闭页面释放图形场景。
    page.close()
    # 冲刷销毁事件。
    application.processEvents()


# 宽页签应把全部分析参数压到同一行，把第二行高度还给眼图。
def test_eye_controls_share_one_row_at_wide_tab_size() -> None:
    """宽页中指标、调制、频段宽度和 M 使用同一条参数基线。"""

    # 使用接近用户截图的宽页面，避免紧凑布局回退影响本项验收。
    application = _qt_application()
    # 创建真实页面并切换到会显示全部眼图参数的眼高模式。
    page = InfluenceBandPage()
    # 眼高状态同时显示指标、调制、M 和频段宽度。
    page.metric_combo.setCurrentText("眼高")
    # 固定宽度保证五个参数和按钮有足够空间排成一行。
    page.setFixedSize(1200, 760)
    # 显示页面后 Qt 才会计算字段容器的最终几何位置。
    page.show()
    # 冲刷布局事件，取得用户实际看到的位置。
    application.processEvents()

    # 所有输入控件映射到同一页面坐标系，局部字段坐标不会制造假相等。
    control_top_positions = [
        control.mapTo(page, QPoint(0, 0)).y()
        for control in (
            page.metric_combo,
            page.band_width_spin,
            page.modulation_combo,
            page.m_spin,
        )
    ]
    # 一行布局允许组合框与数字框字体度量带来至多五像素的基线差异。
    assert max(control_top_positions) - min(control_top_positions) <= 5
    # 眼图区应在首行控件下方紧接出现，不再包含第二排参数的高度。
    eye_top = page.eye_plots_panel.mapTo(page, QPoint(0, 0)).y()
    # 相对控件高度的断言适应不同 DPI，同时会拒绝旧双行布局。
    assert eye_top - min(control_top_positions) <= 2 * page.metric_combo.height()

    # 关闭页面释放离屏图形资源。
    page.close()
    # 冲刷关闭事件，避免影响后续 Qt 测试。
    application.processEvents()


# 同一页面可能随主窗口拖动反复跨过断点，控件必须安全搬移而不丢值。
def test_eye_controls_reflow_across_widths_and_metric_changes() -> None:
    """宽窄切换和指标显隐后，调制与 M 保值且能回到同一行。"""

    # 创建真实页面，通过 resizeEvent 驱动生产响应式路径。
    application = _qt_application()
    # 初始进入眼宽，使调制和 M 两个条件均处于生效状态。
    page = InfluenceBandPage()
    # 非默认 PAM4 能检测控件被重建或替换后偷偷恢复默认值。
    page.metric_combo.setCurrentText("眼宽")
    # 保存非默认调制选择。
    page.modulation_combo.setCurrentText("PAM4")
    # M=37 同样用于检测动态搬移后数值是否丢失。
    page.m_spin.setValue(37)
    # 900 px 应使用单行参数结构。
    page.resize(900, 600)
    # 显示后尺寸事件和布局事件才会真实发生。
    page.show()
    # 冲刷首次宽布局。
    application.processEvents()

    # 宽页中调制框与指标框位于同一行。
    wide_metric_y = page.metric_combo.mapTo(page, QPoint(0, 0)).y()
    # 读取调制框的页面坐标用于成对比较。
    wide_modulation_y = page.modulation_combo.mapTo(page, QPoint(0, 0)).y()
    # 不同控件字体度量允许五像素误差。
    assert abs(wide_metric_y - wide_modulation_y) <= 5

    # 断点下方 639 px 应回退两行，而不是把输入框压到重叠。
    page.resize(639, 600)
    # 提交断点下方的布局搬移。
    application.processEvents()
    # 调制框必须真实位于指标框下方。
    assert page.modulation_combo.mapTo(page, QPoint(0, 0)).y() > wide_metric_y
    # 搬移 QWidget 不能重建组合框或丢失用户选择。
    assert page.modulation_combo.currentText() == "PAM4"
    # M 的非默认值也必须保留。
    assert page.m_spin.value() == 37

    # 恰好 640 px 必须进入单行布局，锁定规格中的包含边界。
    page.resize(640, 600)
    # 提交断点边界的反向搬移。
    application.processEvents()
    # 读取四个可见参数的页面顶边。
    boundary_positions = [
        control.mapTo(page, QPoint(0, 0)).y()
        for control in (
            page.metric_combo,
            page.band_width_spin,
            page.modulation_combo,
            page.m_spin,
        )
    ]
    # 若实现把断点误改成 700，640 px 会错误保持两行并在此失败。
    assert max(boundary_positions) - min(boundary_positions) <= 5

    # 再缩到 500 px，覆盖第二行状态下的指标显隐往返。
    page.resize(500, 600)
    # 提交更窄布局。
    application.processEvents()

    # Vpp 会隐藏整组眼参数，覆盖窄布局下的显隐切换。
    page.metric_combo.setCurrentText("Vpp")
    # Qt 处理隐藏事件后眼参数不能占据空白第二行。
    application.processEvents()
    # 整组眼参数在 Vpp 下不可见。
    assert page.eye_parameters_panel.isHidden()
    # 再切回眼高，确认同一个控件组仍能恢复。
    page.metric_combo.setCurrentText("眼高")
    # 提交重新显示事件。
    application.processEvents()
    # 调制和 M 的值不因指标往返而变化。
    assert (page.modulation_combo.currentText(), page.m_spin.value()) == ("PAM4", 37)

    # 重新放宽页面，覆盖窄到宽的反向搬移。
    page.resize(900, 600)
    # 提交最终单行布局。
    application.processEvents()
    # 调制框应再次回到与指标框相同的参数行。
    restored_positions = [
        control.mapTo(page, QPoint(0, 0)).y()
        for control in (page.metric_combo, page.modulation_combo, page.m_spin)
    ]
    # 反复搬移后仍维持单行基线，没有重复加入布局或残留空行。
    assert max(restored_positions) - min(restored_positions) <= 5

    # 关闭页面释放图形对象。
    page.close()
    # 冲刷关闭事件，保持 Qt 测试隔离。
    application.processEvents()


# 实际主窗中页签可用区较窄，必须验证紧凑尺寸而非只看大图。
def test_eye_mode_remains_usable_at_compact_tab_size() -> None:
    """350×464 页签中参数可读，三图与下方结果不重叠。"""

    # 构造与真实主窗最小尺寸下接近的页签可用区。
    application = _qt_application()
    # 页面被强制置于容器实际分配的紧凑尺寸。
    page = InfluenceBandPage()
    # 眼高会显示最多的顶部控件和三幅图。
    page.metric_combo.setCurrentText("眼高")
    # 固定尺寸杀死布局通过自行扩大顶层窗口的假绿。
    page.setFixedSize(350, 464)
    # 显示后 Qt 才会分配最终子控件几何尺寸。
    page.show()
    # 冲刷布局事件。
    application.processEvents()

    # 调制下拉不应被压成几乎无法操作的宽度。
    assert page.modulation_combo.width() >= 64
    # M 数字输入需要保留至少 64 px。
    assert page.m_spin.width() >= 64
    # 两个眼参数映射到共同父区后按调制、M 的阅读顺序紧凑排列。
    eye_parameter_positions = [
        control.mapTo(page.eye_parameters_panel, QPoint(0, 0)).x()
        for control in (page.modulation_combo, page.m_spin)
    ]
    # 使用共同坐标系避免字段容器内的局部 x=0 掩盖真实顺序。
    assert eye_parameter_positions == sorted(eye_parameter_positions)
    # 两个字段不能重叠到同一横向位置。
    assert len(set(eye_parameter_positions)) == 2
    # 最大宽度限制避免宽窗口把同组参数拉成互不相关的孤岛。
    assert page.m_spin.maximumWidth() <= 80
    # 紧凑宽度下三幅眼图改为纵向，每幅都使用接近完整视口的可读宽度。
    for plot in (page.reference_plot, page.before_plot, page.after_plot):
        # 260 px 以上可避免 UI 轴八个相位刻度互相覆盖。
        assert plot.width() >= 260
    # 候选列表改到影响曲线下方，也应获得完整窄页宽度。
    assert page.candidate_list.width() >= 260
    # 内容总高超过最小窗口时必须提供纵向滚动，而不是继续压缩图表。
    assert page.content_scroll.verticalScrollBar().maximum() > 0
    # 眼图区的底边必须位于影响曲线顶边之上。
    eye_bottom = page.eye_plots_panel.mapTo(page, QPoint(0, page.eye_plots_panel.height())).y()
    # 影响曲线顶部映射到页面坐标。
    impact_top = page.impact_plot.mapTo(page, QPoint(0, 0)).y()
    # 上下两区不可重叠或反序。
    assert eye_bottom <= impact_top
    # 三幅图的底部都必须留在眼图父区内。
    for plot in (page.reference_plot, page.before_plot, page.after_plot):
        # 将每幅图的底部映射到父区坐标。
        plot_bottom = plot.mapTo(page.eye_plots_panel, QPoint(0, plot.height())).y()
        # 绘图子控件不得越过父区底边。
        assert plot_bottom <= page.eye_plots_panel.height()

    # 关闭页面释放离屏图形资源。
    page.close()
    # 冲刷关闭事件。
    application.processEvents()
