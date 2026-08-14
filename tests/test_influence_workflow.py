"""影响频段页签的真实文件、后台线程与主窗口集成验收。"""

# 延迟解析测试辅助函数标注，保持 offscreen Qt 测试在 Python 3.11 稳定导入。
from __future__ import annotations

# 无显示器的 CI 使用 Qt offscreen 后端。
import os

# 轮询 Qt 事件时使用单调时钟控制超时。
import time

# 测试波形、拟合脉冲与绘图数据都使用 float64 NumPy 数组。
import numpy as np

# PyQtGraph 类型用于确认三幅眼图真正含有 0 UI 中心线。
import pyqtgraph as pg

# pytest 提供临时目录与 monkeypatch 夹具。
import pytest

# FIR 设计构造一个物理真值已知的 200–300 MHz 幅度缺口。
from scipy import signal

# 先锁定 offscreen，再导入任何 Qt 组件。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Qt 枚举为测试点击提供明确的左键类型，QPoint 用于主窗口内的真实几何比较。
from PySide6.QtCore import QPoint, Qt

# QtTest 用真实点击触发页面信号，而不是绕过 UI 调内部函数。
from PySide6.QtTest import QTest

# 错误对话框在测试中被收集，避免算法回归时永久阻塞 CI。
from PySide6.QtWidgets import QFileDialog, QMessageBox

# 实际 QApplication 初始化与演示补偿结果复用正式入口。
from response_lab.app import (
    _qt_application,
    build_demo_influence_run,
    build_demo_run,
)

# 主窗口是本文件要验收的集成边界。
from response_lab.ui import ResponseLabWindow


# 默认主窗口的中央页签比独立页面窄，必须在真实容器中验证单行参数。
def test_default_main_window_keeps_eye_controls_on_one_row() -> None:
    """1400×860 主窗口中影响页应把四个参数放在同一行。"""

    # 创建正式应用和主窗口，覆盖左右侧栏参与后的真实页签宽度。
    application = _qt_application()
    # 主窗口使用用户实际启动的完整结构，而不是独立页面替身。
    window = ResponseLabWindow()
    # 固定项目默认上限尺寸，复现约 756 px 的中央页签。
    window.resize(1400, 860)
    # 眼高显示调制和 M，使参数行处于最大可见状态。
    window.influence_page.metric_combo.setCurrentText("眼高")
    # 切换到影响频段页，隐藏页不会获得可靠最终几何。
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    # 显示窗口触发真实 splitter、侧栏和页签尺寸分配。
    window.show()
    # 冲刷全部布局事件。
    application.processEvents()

    # 保存页面短名，后续几何全部映射到同一个真实页签坐标系。
    page = window.influence_page
    # 该断言锁定测试确实覆盖窄于旧 760 px 断点的主窗口场景。
    assert page.width() <= 760
    # 四个用户输入的顶边应位于同一行。
    top_positions = [
        control.mapTo(page, QPoint(0, 0)).y()
        for control in (
            page.metric_combo,
            page.band_width_spin,
            page.modulation_combo,
            page.m_spin,
        )
    ]
    # Qt 控件字体度量允许五像素以内的视觉基线差异。
    assert max(top_positions) - min(top_positions) <= 5
    # 眼图区紧随这一行出现，证明旧第二排高度已经返还给绘图区域。
    eye_top = page.eye_plots_panel.mapTo(page, QPoint(0, 0)).y()
    # 相对控件高度的阈值适应不同 DPI，同时会拒绝旧两行结构。
    assert eye_top - min(top_positions) <= 2 * page.metric_combo.height()

    # 当前无后台任务，可直接关闭正式窗口。
    window.close()
    # 冲刷关闭事件，释放 QThread 与图形场景资源。
    application.processEvents()


# 写出已知 200–300 MHz 纯幅度缺口的两份 CSV，作为真实文件入口谕示。
def _write_known_band_pulses(
    tmp_path,
    *,
    pulse_length_ui: int = 50,
    samples_per_ui: int = 8,
    keysight_header: bool = False,
) -> tuple[object, object]:
    """写出拥有 200–300 MHz 纯幅度缺口的等长拟合脉冲。"""

    # 2 GSa/s 下的 400 点记录具有 5 MHz 物理分辨率。
    sample_rate_hz = 2.0e9
    # 总点数严格由调用方的 Np*M 得到，默认仍是 50*8=400 点。
    samples = pulse_length_ui * samples_per_ui
    # CSV 时间轴使加载器能独立反推采样率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 线性相位带通核的完整支撑就是已知真值频段。
    bandpass = signal.firwin(
        201,
        [200.0e6, 300.0e6],
        pass_zero=False,
        fs=sample_rate_hz,
        window=("kaiser", 8.0),
    )
    # 参考脉冲放在 FIR 群时延位置，不引入额外相位差。
    reference = np.zeros(samples, dtype=np.float64)
    # 201 抽头对称 FIR 的群时延是 100 点。
    reference[100] = 1.0
    # DUT 仅在已知带内降低 40% 幅度。
    dut = reference.copy()
    # 浅缺口不跨过响应零点，因此局部补偿可解。
    dut[:201] -= 0.4 * bandpass
    # 两份 CSV 都使用时间+电压两列；指定时覆盖正式 Keysight v2 表头路径。
    reference_path = tmp_path / "influence_reference_pulse.csv"
    # DUT 路径与参考路径分开，还原用户的两设备工作流。
    dut_path = tmp_path / "influence_dut_pulse.csv"
    # 保留高精度时间轴，避免 CSV 舍入改变采样率。
    if keysight_header:
        for path, values, source_name in (
            (reference_path, reference, "Reference Pulse"),
            (dut_path, dut, "DUT Pulse"),
        ):
            header = "\n".join(
                (
                    "File Format, WaveformXYValues",
                    "Format Version, 2",
                    "Instrument, D9300A",
                    f"Points, {samples}",
                    f"Source Name, {source_name}",
                    "X Units, Second",
                    "Y Units, Volt",
                    "Data,",
                    "double, double",
                )
            )
            with path.open("w", encoding="utf-8", newline="") as stream:
                stream.write(header + "\n")
                np.savetxt(stream, np.column_stack((time_s, values)), delimiter=",")
    else:
        np.savetxt(reference_path, np.column_stack((time_s, reference)), delimiter=",")
        # DUT 使用完全相同的时间轴合同。
        np.savetxt(dut_path, np.column_stack((time_s, dut)), delimiter=",")
    # 返回 Path-like 对象供主窗口路径卡片使用。
    return reference_path, dut_path


# 关闭自动建议并固定扫描频带，使集成测试的候选集合完全可重放。
def _configure_manual_scan(window: ResponseLabWindow) -> None:
    """把主窗口频带设为包含四个 100 MHz 候选的手动范围。"""

    # 手动模式使测试真值不受自动 -20 dB 带宽建议变化影响。
    window.auto_frequency_bands.setChecked(False)
    # 界面默认频率单位是 GHz。
    window.band_low.setValue(0.1)
    # 0.5 GHz 上限低于 2 GSa/s 的 Nyquist。
    window.band_high.setValue(0.5)
    # 相位去斜也使用同一可解频带。
    window.phase_low.setValue(0.1)
    # 显式上限避免手动模式中的空相位拟合范围。
    window.phase_high.setValue(0.5)


# 在处理 Qt 事件的同时等待后台扫描或候选回放，超时后给出明确失败证据。
def _wait_for_influence(
    window: ResponseLabWindow,
    application: object,
    *,
    selected_row: int | None = None,
    errors: list[str] | None = None,
    timeout_s: float = 10.0,
) -> None:
    """处理 Qt 事件，直到扫描或候选回放完整收尾。"""

    # 单调截止时间不受系统时钟校时影响。
    deadline = time.monotonic() + timeout_s
    # 后台 QThread 结束前持续让主线程消费队列信号。
    while time.monotonic() < deadline:
        # 处理 succeeded/finished 队列连接。
        application.processEvents()
        # 扫描必须同时得到完整 run 且清空 worker。
        completed = window._worker is None and window._influence_run is not None  # noqa: SLF001
        # 候选回放测试还要等待目标行号真正提交。
        if completed and (
            selected_row is None or window._influence_selected_row == selected_row  # noqa: SLF001
        ):
            # 已到达一致的静态界面。
            return
        # 后台已失败且 worker 收尾时，立即报出原始文字而不等到超时。
        if errors and window._worker is None:  # noqa: SLF001 - 检查真实线程收尾状态
            # 列表包含 warning/critical 的完整错误文字。
            raise AssertionError("影响频段分析失败：" + " | ".join(errors))
        # Python sleep 释放 GIL；真实应用的 C++ 事件循环也会等待而不饿死后台算法线程。
        time.sleep(0.02)
    # 超时表示线程、版本门禁或页面渲染中至少一层失效。
    raise AssertionError("等待影响频段分析完成超时")


# 把模态错误对话框收集成文字，避免 CI 被窗口阻塞并保留真实失败原因。
def _capture_dialog_errors(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把本应阻塞的错误对话框转成可断言的文字列表。"""

    # 失败文字按实际发生顺序保留。
    errors: list[str] = []
    # 替换静态 critical 调用，让 CI 在失败时仍能返回。
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )
    # 参数校验使用 warning，同样收集而不打开模态窗口。
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: errors.append(str(message)),
    )
    # 调用方在等待后检查列表必须为空。
    return errors


# 影响频段后台不能绕过主比较/补偿已收紧的普通 CSV 单通道合同。
def test_influence_entry_rejects_wide_headerless_fitted_pulse(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """影响频段真实按钮路径应拒绝无表头宽表，而非静默取前两列。"""

    reference_path, dut_path = _write_known_band_pulses(tmp_path)
    reference_table = np.loadtxt(reference_path, delimiter=",", ndmin=2)
    np.savetxt(
        reference_path,
        np.column_stack((reference_table, np.full(reference_table.shape[0], 99.0))),
        delimiter=",",
    )
    application = _qt_application()
    window = ResponseLabWindow()
    errors = _capture_dialog_errors(monkeypatch)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    _configure_manual_scan(window)
    window.show()
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    application.processEvents()

    QTest.mouseClick(
        window.influence_page.start_button,
        Qt.MouseButton.LeftButton,
    )
    # 先消费按钮信号，保证 worker 已创建或快速失败信号已投递。
    application.processEvents()
    deadline = time.monotonic() + 10.0
    while window._worker is not None and time.monotonic() < deadline:  # noqa: SLF001
        application.processEvents()
        QTest.qWait(20)
    run = window._influence_run  # noqa: SLF001
    window.close()
    application.processEvents()

    assert window._worker is None  # noqa: SLF001
    assert run is None
    assert len(errors) == 1
    assert "无表头 CSV" in errors[0]
    assert "恰好 2 列" in errors[0]


def test_failed_reanalysis_does_not_leave_old_influence_result_visible(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新分析失败时不能继续显示上一批候选、曲线或工作区。"""

    reference_path, dut_path = _write_known_band_pulses(tmp_path)
    reference_table = np.loadtxt(reference_path, delimiter=",", ndmin=2)
    np.savetxt(
        reference_path,
        np.column_stack((reference_table, np.ones(reference_table.shape[0]))),
        delimiter=",",
    )
    application = _qt_application()
    window = ResponseLabWindow()
    errors = _capture_dialog_errors(monkeypatch)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    _configure_manual_scan(window)
    window._influence_run = object()  # type: ignore[assignment]  # noqa: SLF001
    window.influence_page.candidate_list.addItem("旧候选")
    window.influence_page.diagnostic_label.setText("旧诊断")
    window.influence_page.diagnostic_label.show()
    window.influence_page.impact_plot.plot([0.0, 1.0], [1.0, 0.0])
    assert window.influence_page.impact_plot.listDataItems()

    window.influence_page.start_button.click()
    deadline = time.monotonic() + 10.0
    while window._worker is not None and time.monotonic() < deadline:  # noqa: SLF001
        application.processEvents()
        QTest.qWait(20)

    assert errors and "恰好 2 列" in errors[0]
    assert window._influence_run is None  # noqa: SLF001
    assert window.influence_page.candidate_list.count() == 0
    assert not window.influence_page.impact_plot.listDataItems()
    assert window.influence_page.diagnostic_label.text() == ""
    assert window.influence_page.diagnostic_label.isHidden()
    window.close()
    application.processEvents()


def test_same_path_fitted_pulse_overwrite_uses_current_content_without_reconfirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """路径不变但源字节变化时，下一次分析应直接读取当前内容。"""

    reference_path, dut_path = _write_known_band_pulses(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    errors = _capture_dialog_errors(monkeypatch)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    _configure_manual_scan(window)
    window.influence_page.m_spin.setValue(8)
    # Seed an already rendered result: same-path rewrites do not emit a path-change
    # signal, so starting a replacement task must clear this stale workspace.
    window._influence_run = object()  # type: ignore[assignment]  # noqa: SLF001

    table = np.loadtxt(reference_path, delimiter=",", ndmin=2)
    table[:, 1] *= 0.75
    np.savetxt(reference_path, table, delimiter=",")
    window.show()
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    application.processEvents()

    QTest.mouseClick(window.influence_page.start_button, Qt.MouseButton.LeftButton)
    assert window._influence_run is None  # noqa: SLF001
    application.processEvents()
    _wait_for_influence(window, application, errors=errors)

    assert errors == []
    assert window._influence_run is not None  # noqa: SLF001
    np.testing.assert_allclose(
        window._influence_run.workspace.reference_pulse.values[:, 0],  # noqa: SLF001
        table[:, 1],
    )
    window.close()
    application.processEvents()


# 端到端覆盖拟合脉冲文件、后台扫描、三眼图和候选点选的完整主窗口路径。
def test_eye_height_scan_runs_from_files_and_candidate_click_only_updates_detail(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """眼高从文件扫描到三眼图，候选点选不重置总览。"""

    # 准备真值已知的两份拟合脉冲。
    reference_path, dut_path = _write_known_band_pulses(tmp_path)
    # 初始化真实 Qt 应用和主窗口。
    application = _qt_application()
    # 主窗口包含完整左右参数面板与第六页签。
    window = ResponseLabWindow()
    # 新页签使用用户确认的简洁名称，不在标签中附加工程近似术语。
    assert window.visual_tabs.tabText(window.influence_tab_index) == "影响频段"
    # 收集意外后台错误，不让模态对话框卡住测试。
    errors = _capture_dialog_errors(monkeypatch)
    # 两份拟合脉冲继续使用现有左栏输入。
    window.reference_card.set_path(reference_path)
    # DUT 脉冲单独设置。
    window.dut_card.set_path(dut_path)
    # 频带冻结为 100–500 MHz。
    _configure_manual_scan(window)
    # 下拉框一次只选中眼高指标。
    window.influence_page.metric_combo.setCurrentIndex(
        window.influence_page.metric_combo.findData("eye_height")
    )
    # 用户只输入每 UI 8 点，Np=50 将由 400 点脉冲自动推导。
    window.influence_page.m_spin.setValue(8)
    # 非默认 200 MHz 贯穿页面、主窗口请求和后台候选生成，防止只改了显示控件。
    window.influence_page.band_width_spin.setValue(200.0)
    # 显示窗口使 Qt 的真实点击路径生效。
    window.show()
    # 切换到第六页签，确保开始按钮是当前可见控件。
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    # 先处理布局与控件状态更新。
    application.processEvents()
    # 点击正式“开始分析”按钮触发后台线程。
    QTest.mouseClick(
        window.influence_page.start_button,
        Qt.MouseButton.LeftButton,
    )
    # 立即处理按钮信号，让参数错误在进入等待循环前就可见。
    application.processEvents()
    # 启动边界不应产生任何 warning/critical。
    assert errors == []
    # 任务要么正在运行，要么已经快速完成并提交 run。
    assert window._worker is not None or window._influence_run is not None  # noqa: SLF001
    # 等待扫描、默认候选回放和三幅轨迹眼图全部完成。
    _wait_for_influence(window, application, errors=errors)
    # 真实文件的 400 个样点除以 M=8，后台必须自动得到 Np=50。
    assert window._influence_run.workspace.settings.eye.pulse_length_ui == 50  # noqa: SLF001

    # 正常路径不应出现错误对话框。
    assert errors == []
    # 已知纯幅度缺口应被识别为幅度模式。
    recommendation = window._influence_run.result.recommendation  # noqa: SLF001
    # 页面设置必须真实进入归因核心，而不是后台仍静默使用 100 MHz 默认值。
    assert window._influence_run.workspace.settings.frequency_step_hz == 200.0e6  # noqa: SLF001
    # 满权核心宽度与规则步进使用同一个用户值。
    assert window._influence_run.workspace.settings.requested_window_hz == 200.0e6  # noqa: SLF001
    # 100–500 MHz 扫描范围应恰好生成两个 200 MHz 核心。
    assert len(window._influence_run.workspace.candidates) == 2  # noqa: SLF001
    # 合成数据必须生成保守推荐。
    assert recommendation is not None
    # 模式判定不应被相位或联合标签抢占。
    assert recommendation.mode == "magnitude"
    # 推荐带与 200–300 MHz 真值必须相交。
    assert recommendation.band.high_hz > 200.0e6
    # 下边界同样必须低于真值上限。
    assert recommendation.band.low_hz < 300.0e6
    # 幅度、相位、幅相三条影响曲线同时展示。
    assert len(window.influence_page.impact_plot.listDataItems()) == 3
    # 默认候选已构造三角色共时轴轨迹。
    comparison = window._influence_run.eye_comparison  # noqa: SLF001
    # 眼指标的有效候选必须保存比较数据。
    assert comparison is not None
    # 页面三列与领域三角色使用互异轨迹数组，能发现串位。
    role_pairs = (
        (window.influence_page.reference_plot, comparison.reference_traces_v),
        (window.influence_page.before_plot, comparison.before_traces_v),
        (window.influence_page.after_plot, comparison.after_traces_v),
    )
    # 从真实 PlotDataItem 还原每个角色的 NaN 分隔轨迹。
    for plot, expected_traces in role_pairs:
        # 每幅图严格只有一条聚合后的 PlotDataItem。
        curves = plot.listDataItems()
        # 禁止回退到每条轨迹一个 Qt 对象。
        assert len(curves) == 1
        # 使用原始 xData/yData，不读取视窗自动降采样结果。
        raw_x = np.asarray(curves[0].xData, dtype=np.float64)
        # yData 同样是页面真正提交的扁平数组。
        raw_y = np.asarray(curves[0].yData, dtype=np.float64)
        # 每条轨迹后有一个 NaN 分隔点。
        separated_width = comparison.time_ui.size + 1
        # 还原为“轨迹数 × (2*M+2)”矩阵。
        separated_x = raw_x.reshape(-1, separated_width)
        # y 矩阵必须和领域轨迹行数对应。
        separated_y = raw_y.reshape(-1, separated_width)
        # 行末 x/y 都以 NaN 断开，相邻符号不会被伪线相连。
        assert np.all(np.isnan(separated_x[:, -1]))
        # y 分隔位同样存在。
        assert np.all(np.isnan(separated_y[:, -1]))
        # 每条轨迹的时间列完整等于核心的 -1 到 +1 UI 轴。
        np.testing.assert_array_equal(
            separated_x[:, :-1],
            np.tile(comparison.time_ui, (expected_traces.shape[0], 1)),
        )
        # 页面真实幅值必须逐点等于对应角色，不允许参考/前/后串位。
        np.testing.assert_array_equal(separated_y[:, :-1], expected_traces)
        # 每幅图只有一条 0 UI 中心线。
        center_lines = [
            item
            for item in plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine)
        ]
        # 中心线数量和位置都必须精确。
        assert len(center_lines) == 1 and center_lines[0].value() == 0.0
        # 视窗横轴固定为 -1 到 +1 UI。
        np.testing.assert_allclose(plot.viewRange()[0], np.array([-1.0, 1.0]))
        # 视窗纵轴统一使用核心生成的共同幅值范围。
        np.testing.assert_allclose(plot.viewRange()[1], comparison.amplitude_range_v)
    # 结果摘要直接给出三份同口径标量，不要求用户从轨迹图反推。
    assert all(
        label in window.influence_page.selection_summary.text()
        for label in ("参考", "补偿前", "补偿后")
    )
    # 眼高按参考主光标归一化，摘要不得误标成绝对电压 V。
    assert " V" not in window.influence_page.selection_summary.text()

    # 保留影响曲线数值，验证候选切换不重算总览。
    impact_before = [
        np.asarray(curve.getData()[1]).copy()
        for curve in window.influence_page.impact_plot.listDataItems()
    ]
    # 展示列表应包含至少一个非默认候选。
    assert window.influence_page.candidate_list.count() > 1
    # 改选第二行会启动轻量候选回放线程。
    window.influence_page.candidate_list.setCurrentRow(1)
    # 等待行号与详情图一致。
    _wait_for_influence(window, application, selected_row=1, errors=errors)
    # 候选切换后三条总览曲线应逐点保持。
    for expected, curve in zip(
        impact_before,
        window.influence_page.impact_plot.listDataItems(),
        strict=True,
    ):
        # 只允许下方详情变化。
        np.testing.assert_allclose(curve.getData()[1], expected)
    # 候选回放同样不应弹出错误。
    assert errors == []
    # 关闭前已没有活动线程，可直接释放窗口。
    window.close()
    # 处理 deferred delete 与 close 事件。
    application.processEvents()


# 单独走一次眼宽后台链路，防止 crossing 内核正确但真实 QThread 工作流仍超时。
def test_eye_width_scan_completes_and_draws_measured_virtual_eyes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """眼宽扫描应在真实文件和后台线程路径内完成并画出三幅眼图。"""

    # 复用具有已知带内幅度缺口的等长拟合脉冲文件。
    reference_path, dut_path = _write_known_band_pulses(
        tmp_path,
        pulse_length_ui=20,
        samples_per_ui=32,
    )
    # 初始化正式 Qt 应用，保证测试覆盖真实主线程事件队列。
    application = _qt_application()
    # 主窗口负责把页面请求、后台 worker 和领域结果串起来。
    window = ResponseLabWindow()
    # 模态错误改为文字列表，失败时测试能直接报告原因。
    errors = _capture_dialog_errors(monkeypatch)
    # 参考拟合脉冲放入现有左栏输入卡片。
    window.reference_card.set_path(reference_path)
    # DUT 拟合脉冲独立设置，避免意外复用同一文件。
    window.dut_card.set_path(dut_path)
    # 锁定 100–500 MHz 手动扫描域，排除自动建议差异。
    _configure_manual_scan(window)
    # 本用例只选择眼宽，不能退回眼高的快速路径。
    window.influence_page.metric_combo.setCurrentIndex(
        window.influence_page.metric_combo.findData("eye_width")
    )
    # 显式选择 PAM4，确保真实后台测量覆盖三只眼而不是默认 NRZ 单眼。
    window.influence_page.modulation_combo.setCurrentText("PAM4")
    # M=32 压测常用高分辨率，Np=20 从 640 点文件自动得到。
    window.influence_page.m_spin.setValue(32)
    # 两个 200 MHz 核心足以覆盖候选生成，又让测试保持快速。
    window.influence_page.band_width_spin.setValue(200.0)
    # 显示窗口后 QtTest 的鼠标点击才走真实可见控件路径。
    window.show()
    # 切到影响频段页，避免隐藏页控件状态掩盖布局问题。
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    # 提交当前布局和下拉框联动状态。
    application.processEvents()
    # 点击正式按钮启动眼宽准备、扫描和默认候选回放。
    QTest.mouseClick(
        window.influence_page.start_button,
        Qt.MouseButton.LeftButton,
    )
    # 消费启动信号后再进入统一等待循环。
    application.processEvents()
    # Windows CI 的无头绘图和 crossing 内核明显慢于本机；这里验证有限时间内
    # 完成真实工作流，不把 10 秒本机性能预算误当成功能合同。
    _wait_for_influence(window, application, errors=errors, timeout_s=30.0)

    # 正常眼宽路径不应触发参数或后台错误对话框。
    assert errors == []
    # 领域工作区必须保留用户实际选择的指标，不能被界面映射成眼高。
    assert window._influence_run.workspace.settings.metric == "eye_width"  # noqa: SLF001
    # 下拉框必须原样映射到核心 PAM4 设置。
    assert window._influence_run.workspace.settings.eye.modulation == "pam4"  # noqa: SLF001
    # 后台必须从 640 点脉冲和 M=32 推导出 Np=20。
    assert window._influence_run.workspace.settings.eye.pulse_length_ui == 20  # noqa: SLF001
    # 参考 PAM4 三只眼都应得到有限 crossing 眼宽。
    assert len(window._influence_run.workspace.reference_eye.eye_widths_ui) == 3  # noqa: SLF001
    # 三个参考眼宽不能含未测量或 crossing 不足的 NaN。
    assert all(  # noqa: SLF001
        np.isfinite(width)
        for width in window._influence_run.workspace.reference_eye.eye_widths_ui
    )
    # DUT 补偿前三只眼同样必须具有有限眼宽。
    assert all(  # noqa: SLF001
        np.isfinite(width)
        for width in window._influence_run.workspace.before_eye.eye_widths_ui
    )
    # 默认候选回放应生成参考、补偿前、补偿后三角色比较。
    comparison = window._influence_run.eye_comparison  # noqa: SLF001
    # 有效眼宽候选必须提供真实轨迹而不是空占位图。
    assert comparison is not None
    # 三幅图各自只用一个 NaN 分隔的聚合曲线，和眼高路径保持同一渲染合同。
    assert all(
        len(plot.listDataItems()) == 1
        for plot in (
            window.influence_page.reference_plot,
            window.influence_page.before_plot,
            window.influence_page.after_plot,
        )
    )
    # 眼宽摘要使用 UI 单位，让用户无需从图像像素反推标量。
    assert " UI" in window.influence_page.selection_summary.text()
    # 参考和补偿前限制眼宽都必须来自完整 crossing 测量而不是未计算 NaN。
    assert np.isfinite(window._influence_run.workspace.reference_metric)  # noqa: SLF001
    # DUT 基线同样应提供有限限制眼宽。
    assert np.isfinite(window._influence_run.workspace.before_metric)  # noqa: SLF001
    # 等待函数返回时 worker 已完整收尾，可安全关闭窗口。
    window.close()
    # 处理 deferred delete，避免 Qt 对象泄漏到后续用例。
    application.processEvents()


# 验证内置 PRBS13Q、pmax 拖尾窗和 LFP Vpp 能走通真实 GUI 后台路径。
def test_lfp_vpp_scan_uses_builtin_pattern_and_draws_steady_state_models(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LFP Vpp should scan the shared periodic pattern-pulse model end to end."""

    # 两份拟合脉冲包含已知的 200–300 MHz 纯幅度缺口。
    reference_pulse_path, dut_pulse_path = _write_known_band_pulses(
        tmp_path,
        keysight_header=True,
    )

    application = _qt_application()
    window = ResponseLabWindow()
    errors = _capture_dialog_errors(monkeypatch)
    window.reference_card.set_path(reference_pulse_path)
    window.dut_card.set_path(dut_pulse_path)
    _configure_manual_scan(window)
    window.influence_page.metric_combo.setCurrentIndex(
        window.influence_page.metric_combo.findData("vpp")
    )
    # 内置 PRBS13Q 的 8191-symbol 周期按 M=8 上采样。
    window.influence_page.vpp_pattern_source_combo.setCurrentIndex(
        window.influence_page.vpp_pattern_source_combo.findData(
            "builtin_prbs13q_gray"
        )
    )
    window.influence_page.vpp_method_combo.setCurrentIndex(
        window.influence_page.vpp_method_combo.findData("lfp")
    )
    window.influence_page.m_spin.setValue(8)
    # 峰前后各保留 8 UI，既覆盖已知 FIR 拖尾又位于 400 点记录边界内。
    window.influence_page.pre_cursor_ui_spin.setValue(8)
    window.influence_page.post_cursor_ui_spin.setValue(8)
    window.show()
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    application.processEvents()
    QTest.mouseClick(
        window.influence_page.start_button,
        Qt.MouseButton.LeftButton,
    )
    application.processEvents()
    assert errors == []
    assert window._worker is not None or window._influence_run is not None  # noqa: SLF001
    _wait_for_influence(window, application, errors=errors)

    assert errors == []
    recommendation = window._influence_run.result.recommendation  # noqa: SLF001
    assert recommendation is not None
    assert recommendation.mode == "magnitude"
    assert recommendation.band.low_hz < 250.0e6 < recommendation.band.high_hz
    waveform_curves = window.influence_page.vpp_waveform_plot.listDataItems()
    assert [curve.name() for curve in waveform_curves] == ["参考", "补偿前", "补偿后"]
    expected_period_samples = 8191 * 8
    assert all(len(curve.xData) == expected_period_samples for curve in waveform_curves)
    workspace = window._influence_run.workspace  # noqa: SLF001
    assert (
        workspace.reference_pulse.source_metadata["container"] == "keysight_infiniium_waveform_xy"
    )
    assert workspace.dut_pulse.source_metadata["keysight_format_version"] == 2
    assert workspace.vpp_cache is not None
    np.testing.assert_allclose(
        waveform_curves[0].yData,
        workspace.vpp_cache.reference_model.waveform_v,
    )
    np.testing.assert_allclose(
        waveform_curves[1].yData,
        workspace.vpp_cache.dut_model.waveform_v,
    )
    selected_evaluation = window._influence_run.selected_evaluation  # noqa: SLF001
    assert selected_evaluation is not None
    np.testing.assert_allclose(
        waveform_curves[2].yData,
        selected_evaluation.corrected_values[:, 0],
    )
    assert all(
        label in window.influence_page.selection_summary.text()
        for label in ("参考", "补偿前", "补偿后")
    )
    assert " V" in window.influence_page.selection_summary.text()
    window.close()
    application.processEvents()


def test_frequency_rms_scan_loads_external_symbol_pattern_and_labels_vrms(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file-pattern RMS path must remain spectral and visibly use Vrms units."""

    reference_pulse_path, dut_pulse_path = _write_known_band_pulses(
        tmp_path,
        keysight_header=True,
    )
    pattern_path = tmp_path / "ideal_pattern_codes.csv"
    # 非平凡 64-symbol PAM4 周期覆盖四个 Gray symbol code，且不是产品发生器输出。
    pattern_codes = np.resize(np.array([1, 0, 3, 1, 2, 3, 0, 2]), 64)
    np.savetxt(pattern_path, pattern_codes, fmt="%d")

    application = _qt_application()
    window = ResponseLabWindow()
    errors = _capture_dialog_errors(monkeypatch)
    window.reference_card.set_path(reference_pulse_path)
    window.dut_card.set_path(dut_pulse_path)
    _configure_manual_scan(window)
    page = window.influence_page
    page.metric_combo.setCurrentIndex(page.metric_combo.findData("vpp"))
    page.vpp_method_combo.setCurrentIndex(
        page.vpp_method_combo.findData("frequency_rms_error")
    )
    page.vpp_pattern_source_combo.setCurrentIndex(
        page.vpp_pattern_source_combo.findData("file")
    )
    page.ideal_pattern_row.set_path(pattern_path)
    page.m_spin.setValue(8)
    page.pre_cursor_ui_spin.setValue(8)
    page.post_cursor_ui_spin.setValue(8)
    window.show()
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    application.processEvents()
    QTest.mouseClick(page.start_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert errors == []
    _wait_for_influence(window, application, errors=errors)

    run = window._influence_run  # noqa: SLF001
    assert run is not None
    assert (
        run.workspace.reference_pulse.source_metadata["container"]
        == "keysight_infiniium_waveform_xy"
    )
    assert run.workspace.dut_pulse.source_metadata["keysight_format_version"] == 2
    assert run.workspace.vpp_cache is not None
    assert run.workspace.vpp_cache.settings.method == "frequency_rms_error"
    assert run.workspace.vpp_cache.pattern_levels.size == 64
    assert run.result.reference_metric == 0.0
    assert run.result.before_metric > 0.0
    assert " Vrms" in page.selection_summary.text()
    assert "参考误差 0 Vrms" in page.selection_summary.text()
    assert "补偿前误差" in page.selection_summary.text()
    # 候选曲线纵轴和列表单位也必须保持 RMS 误差口径。
    assert page.impact_plot.getAxis("left").labelText == "频域误差改善 (Vrms)"
    assert page.candidate_list.item(0).text().endswith(" Vrms")
    recommendation = run.result.recommendation
    assert recommendation is not None
    assert recommendation.mode == "magnitude"
    assert recommendation.band.low_hz < 250.0e6 < recommendation.band.high_hz
    assert all(len(curve.xData) == 64 * 8 for curve in page.vpp_waveform_plot.listDataItems())
    window.close()
    application.processEvents()


# 锁定影响页独立版本状态，避免修改 M 意外破坏既有补偿导出资格。
def test_influence_only_parameter_change_does_not_invalidate_existing_export() -> None:
    """仅修改影响频段页参数不使已有补偿导出过期。"""

    # 主窗口先展示一份完整、可导出的既有补偿结果。
    application = _qt_application()
    # 创建窗口后直接用正式 present_run 入口提交结果。
    window = ResponseLabWindow()
    # 内置演示 run 包含输出波形和完整补偿设置。
    completed_run = build_demo_run()
    # 正式 worker 成功槽同时提交结果版本并打开导出。
    window._analysis_succeeded(  # noqa: SLF001 - 模拟后台任务完整成功信号
        completed_run,
        window._parameter_version,  # noqa: SLF001 - 冻结当前有效版本
    )
    # 保留原补偿参数版本作为隔离证据。
    compensation_version = window._parameter_version  # noqa: SLF001
    # present_run 将输出页与导出按钮设为有效。
    assert window.export_button.isEnabled()
    # 共享标题此时属于既有补偿流程。
    compensation_header = window.header_state.text()
    # 切到眼高使演示影响结果与页面可见区域一致。
    window.influence_page.metric_combo.setCurrentIndex(
        window.influence_page.metric_combo.findData("eye_height")
    )
    # 用正式影响成功槽提交一份完整扫描结果。
    window._influence_succeeded(  # noqa: SLF001 - 验证两个状态源的隔离
        build_demo_influence_run(),
        window._influence_version,  # noqa: SLF001 - 使用当前影响页版本
    )
    # 影响结果只写页面摘要和状态栏，不覆盖仍有效的补偿标题。
    assert window.header_state.text() == compensation_header
    # 仅修改第六页的 M，不触碰全局补偿设置。
    window.influence_page.m_spin.setValue(window.influence_page.m_spin.value() + 1)
    # 原补偿结果仍然可导出。
    assert window.export_button.isEnabled()
    # 全局补偿版本未被第六页污染。
    assert window._parameter_version == compensation_version  # noqa: SLF001
    # 影响页清空后，共享标题仍准确反映原补偿结果有效。
    assert window.header_state.text() == compensation_header
    # 窗口没有活动 worker，可直接关闭。
    window.close()
    # 处理关闭事件。
    application.processEvents()


def test_export_cannot_replace_running_influence_worker_and_restores_after_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """影响任务占用唯一 worker 时，导出入口必须 fail-closed 且保留原引用。"""

    application = _qt_application()
    window = ResponseLabWindow()
    completed_run = build_demo_run()
    window._analysis_succeeded(  # noqa: SLF001 - establish a valid export preview
        completed_run,
        window._parameter_version,  # noqa: SLF001
    )
    assert window.export_button.isEnabled()

    class RunningInfluenceWorker:
        running = True
        deleted = False

        def isRunning(self) -> bool:  # noqa: N802 - Qt-compatible test double
            return self.running

        def deleteLater(self) -> None:  # noqa: N802 - Qt-compatible test double
            self.deleted = True

    worker = RunningInfluenceWorker()
    window._worker = worker  # type: ignore[assignment]  # noqa: SLF001
    window._active_action = "influence"  # noqa: SLF001
    window.export_button.setEnabled(False)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: pytest.fail("busy export must not open a dialog"),
    )

    window._export()  # noqa: SLF001 - exercise the public button slot guard

    assert window._worker is worker  # noqa: SLF001
    assert window._active_action == "influence"  # noqa: SLF001
    assert "完成后" in window.statusBar().currentMessage()

    worker.running = False
    window._worker_finished()  # noqa: SLF001 - simulate the original signal sender

    assert worker.deleted
    assert window._worker is None  # noqa: SLF001
    assert window.export_button.isEnabled()
    window.close()
    application.processEvents()


# 候选回放失败后列表高亮必须回到最后一次成功行，不能与仍显示的旧详情错位。
def test_failed_candidate_replay_restores_last_successful_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回放失败保留旧详情时同步恢复候选列表行。"""

    # 创建真实页面和 Qt 列表，覆盖 blockSignals 与 currentRow 的实际行为。
    application = _qt_application()
    # 主窗口持有候选回放失败槽和最后成功行状态。
    window = ResponseLabWindow()
    # 收集错误弹窗，避免模态窗口阻塞测试。
    errors = _capture_dialog_errors(monkeypatch)
    # 两行分别模拟已经成功显示的候选和刚刚点击但回放失败的候选。
    window.influence_page.candidate_list.addItems(["已提交", "待回放"])
    # 设置失败行时阻塞信号，避免没有工作区的测试触发无关后台入口。
    previous = window.influence_page.candidate_list.blockSignals(True)
    # 当前高亮先位于待回放行，还原真实失败瞬间。
    window.influence_page.candidate_list.setCurrentRow(1)
    # 恢复列表原信号状态。
    window.influence_page.candidate_list.blockSignals(previous)
    # 第零行是最后一次已经成功提交到详情图的候选。
    window._influence_selected_row = 0  # noqa: SLF001 - 验证失败恢复合同

    # 直接投递当前版本的后台失败信号。
    window._influence_selection_failed(  # noqa: SLF001 - Qt 槽验收
        "候选数值不可解析",
        window._influence_version,  # noqa: SLF001 - 当前影响页版本
    )

    # 列表必须恢复到旧详情对应的成功行。
    assert window.influence_page.candidate_list.currentRow() == 0
    # 后台原始原因仍通过错误边界展示。
    assert errors == ["候选数值不可解析"]
    # 状态栏明确说明旧结果被保留，而不是暗示失败行已经生效。
    assert "已保留上一候选结果" in window.statusBar().currentMessage()
    # 无活动线程，可以直接关闭窗口。
    window.close()
    # 处理关闭和延迟销毁事件。
    application.processEvents()


# 普通比较或数据补偿占用唯一 worker 时，候选列表不能切到尚未回放的新行。
def test_candidate_selection_restores_committed_row_while_other_worker_runs() -> None:
    """跨任务互斥期间保持候选高亮与已提交详情一致。"""

    # 创建真实候选列表和信号连接。
    application = _qt_application()
    # 主窗口的 currentRowChanged 已连接候选回放入口。
    window = ResponseLabWindow()
    # 两行模拟已有详情和用户试图选择的新候选。
    window.influence_page.candidate_list.addItems(["已提交", "未回放"])
    # 第零行是最后成功提交的详情。
    window._influence_selected_row = 0  # noqa: SLF001 - 状态一致性验收
    # 初始行设置时阻塞信号，避免没有工作区时进入候选入口。
    previous = window.influence_page.candidate_list.blockSignals(True)
    # 页面当前高亮与详情一致。
    window.influence_page.candidate_list.setCurrentRow(0)
    # 恢复真实信号连接。
    window.influence_page.candidate_list.blockSignals(previous)

    # 最小替身只表达普通比较/补偿线程仍在运行。
    class RunningWorker:
        """提供主窗口互斥检查所需的运行状态。"""

        # Qt 风格方法名与真实 QThread 保持一致。
        @staticmethod
        def isRunning() -> bool:  # noqa: N802 - 模拟 Qt API
            # 当前普通任务仍占用唯一 worker。
            return True

    # 注入运行中 worker，隔离本测试关注的候选恢复分支。
    window._worker = RunningWorker()  # type: ignore[assignment]  # noqa: SLF001
    # 用户动作先把列表高亮改到第二行并发出 currentRowChanged。
    window.influence_page.candidate_list.setCurrentRow(1)

    # 入口必须立即恢复到旧详情对应的第零行，不能留下视觉错位。
    assert window.influence_page.candidate_list.currentRow() == 0
    # 清除替身，避免 closeEvent 把它当作仍运行的真实任务。
    window._worker = None  # noqa: SLF001 - 测试清理
    # 窗口现在可以正常关闭。
    window.close()
    # 处理关闭和延迟销毁事件。
    application.processEvents()
