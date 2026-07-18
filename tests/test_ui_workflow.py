"""真实 Qt 窗口中的无表头 CSV 与手工采样率 BIN 工作流。"""

from __future__ import annotations

import os
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

from response_lab.app import _qt_application, build_demo_run
from response_lab.dsp import run_compensation
from response_lab.models import CompensationSettings, PulseComparison, TimeSeries
from response_lab.reporting import bundle_paths
from response_lab.ui import AnalysisThread, ResponseLabWindow


def _write_demo_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 1.0e9
    pulse_time_s = np.arange(1024, dtype=np.float64) / sample_rate_hz
    index = np.arange(1024, dtype=np.float64)
    reference = np.exp(-0.5 * ((index - 240.0) / 2.0) ** 2)
    dut = 0.8 * reference
    signal_time_s = np.arange(4096, dtype=np.float64) / sample_rate_hz
    signal = np.sin(2.0 * np.pi * 100.0e6 * signal_time_s)

    reference_path = tmp_path / "reference.csv"
    dut_path = tmp_path / "dut.csv"
    target_csv_path = tmp_path / "target.csv"
    target_bin_path = tmp_path / "target.bin"
    np.savetxt(
        reference_path,
        np.column_stack((pulse_time_s, reference)),
        delimiter=",",
    )
    np.savetxt(dut_path, np.column_stack((pulse_time_s, dut)), delimiter=",")
    np.savetxt(
        target_csv_path,
        np.column_stack((signal_time_s, signal)),
        delimiter=",",
    )
    signal.astype("<f4").tofile(target_bin_path)
    return reference_path, dut_path, target_csv_path, target_bin_path


def _write_high_rate_pulses(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 3.4e12
    samples = 12_800
    index = np.arange(samples, dtype=np.float64)
    time_s = index / sample_rate_hz
    reference = np.exp(-0.5 * ((index - 200.0) / 25.0) ** 2)
    dut = 0.82 * np.exp(-0.5 * ((index - 202.0) / 27.0) ** 2)
    reference_path = tmp_path / "high_rate_reference.csv"
    dut_path = tmp_path / "high_rate_dut.csv"
    np.savetxt(reference_path, np.column_stack((time_s, reference)), delimiter=",")
    np.savetxt(dut_path, np.column_stack((time_s, dut)), delimiter=",")
    return reference_path, dut_path


def _wait_for_analysis(window: ResponseLabWindow, application, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        application.processEvents()
        if window._worker is None and window._run is not None:  # noqa: SLF001 - GUI smoke-test
            return
        QTest.qWait(20)
    raise AssertionError("等待 GUI 分析完成超时")


def _wait_for_result(window: ResponseLabWindow, application, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        application.processEvents()
        if window._worker is None and window._result is not None:  # noqa: SLF001
            return
        QTest.qWait(20)
    raise AssertionError("等待 GUI 任务完成超时")


def _signed_delay_run(delay_samples: int):
    sample_rate_hz = 1.0e9
    pulse_samples = 1024
    index = np.arange(pulse_samples, dtype=np.float64)
    base = np.exp(-0.5 * ((index - 240.0) / 2.0) ** 2)
    shifted = np.zeros_like(base)
    shifted[3:] = base[:-3]
    if delay_samples > 0:
        reference_values, dut_values = base, shifted
    else:
        reference_values, dut_values = shifted, base
    pulse_time_s = index / sample_rate_hz
    reference = TimeSeries(pulse_time_s, reference_values[:, None], sample_rate_hz)
    dut = TimeSeries(pulse_time_s, dut_values[:, None], sample_rate_hz)
    target_time_s = np.arange(4096, dtype=np.float64) / sample_rate_hz
    target = TimeSeries(target_time_s, np.zeros((4096, 1)), sample_rate_hz)
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=5.0e6,
        band_high_hz=350.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        detrend_phase=True,
        analysis_points=4097,
    )
    return run_compensation(reference, dut, target, settings)


def test_fitted_pulses_can_be_compared_without_compensation_data(tmp_path) -> None:
    reference_path, dut_path, target_path, _ = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)

    window._start_comparison()  # noqa: SLF001 - exercise the comparison action
    # 普通比较占用唯一 worker 时，影响页候选和开始按钮也必须锁定。
    assert not window.influence_page.candidate_list.isEnabled()
    assert not window.influence_page.start_button.isEnabled()
    _wait_for_result(window, application)

    # worker 统一收尾后恢复影响页操作。
    assert window.influence_page.candidate_list.isEnabled()
    assert window.influence_page.start_button.isEnabled()

    assert window.target_card.path is None
    assert window._run is None  # noqa: SLF001 - comparison must not synthesize a run
    assert isinstance(window._result, PulseComparison)  # noqa: SLF001
    window.present_comparison(window._result)  # direct call must not hide a Qt slot exception
    assert len(window.pulse_plots[0].listDataItems()) == 2
    assert window.header_state.text() == "比较有效"
    assert not window.export_button.isEnabled()
    assert not window.visual_tabs.isTabEnabled(4)

    window.target_card.set_path(target_path)

    assert window.header_state.text() == "比较有效"
    assert window._result_version == window._parameter_version  # noqa: SLF001
    assert not window.export_button.isEnabled()
    window.close()
    application.processEvents()


def test_default_auto_frequency_bands_allow_high_rate_pulse_comparison(tmp_path) -> None:
    reference_path, dut_path = _write_high_rate_pulses(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()

    assert window.auto_frequency_bands.isChecked()
    assert not window.band_low.isEnabled()
    assert window.band_low.text() == "分析后自动"
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)

    window._start_comparison()  # noqa: SLF001
    _wait_for_result(window, application)

    assert isinstance(window._result, PulseComparison)  # noqa: SLF001
    settings = window._result.analysis.settings  # noqa: SLF001
    assert 40.0e9 < settings.band_high_hz < 100.0e9
    assert settings.phase_fit_low_hz < settings.phase_fit_high_hz
    assert "可比较上限" not in window.metric_label.text()
    assert "分析频带" in window.metric_label.text()
    assert "分析/候选补偿频带" in window.band_legend_label.text()
    assert window.band_high.value() == pytest.approx(settings.band_high_hz / 1.0e9)
    window.close()
    application.processEvents()


def test_manual_phase_detrend_band_survives_automatic_compensation_band(tmp_path) -> None:
    reference_path, dut_path = _write_high_rate_pulses(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()

    assert window.auto_frequency_bands.isChecked()
    assert window.phase_low.isEnabled()
    assert window.phase_high.isEnabled()
    window.phase_low.setValue(5.0)
    window.phase_high.setValue(35.0)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)

    window._start_comparison()  # noqa: SLF001
    _wait_for_result(window, application)

    assert isinstance(window._result, PulseComparison)  # noqa: SLF001
    settings = window._result.analysis.settings  # noqa: SLF001
    assert settings.phase_fit_low_hz == pytest.approx(5.0e9)
    assert settings.phase_fit_high_hz == pytest.approx(35.0e9)
    assert window.phase_low.value() == pytest.approx(5.0)
    assert window.phase_high.value() == pytest.approx(35.0)
    window.close()
    application.processEvents()


def test_manual_phase_band_after_first_suggestion_is_not_overwritten(tmp_path) -> None:
    reference_path, dut_path, _, _ = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)

    window._start_comparison()  # noqa: SLF001
    _wait_for_result(window, application)
    assert window.phase_low.value() > 0.0
    assert not window._phase_band_is_manual  # noqa: SLF001
    assert window._phase_band_initialized  # noqa: SLF001
    first_settings = window._result.analysis.settings  # noqa: SLF001
    next_settings = window._current_settings()  # noqa: SLF001
    assert next_settings.phase_fit_low_hz == pytest.approx(
        first_settings.phase_fit_low_hz, abs=1.0e-3
    )
    assert next_settings.phase_fit_high_hz == pytest.approx(
        first_settings.phase_fit_high_hz, abs=1.0e-3
    )

    window.phase_low.setValue(0.03)
    window.phase_high.setValue(0.20)
    window._start_comparison()  # noqa: SLF001
    _wait_for_result(window, application)

    settings = window._result.analysis.settings  # noqa: SLF001
    assert settings.phase_fit_low_hz == pytest.approx(30.0e6)
    assert settings.phase_fit_high_hz == pytest.approx(200.0e6)
    assert window.phase_low.value() == pytest.approx(0.03)
    assert window.phase_high.value() == pytest.approx(0.20)
    window.close()
    application.processEvents()


def test_detrend_checkbox_marks_result_stale_and_controls_effective_setting() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.present_run(build_demo_run())
    window.show()
    application.processEvents()
    version_before = window._parameter_version  # noqa: SLF001

    assert window.detrend_phase_checkbox.isVisible()
    assert window.detrend_phase_checkbox.isChecked()
    assert window.detrend_phase_checkbox.text() == "去除线性相位"
    window.detrend_phase_checkbox.setChecked(False)

    assert window._parameter_version == version_before + 1  # noqa: SLF001
    assert window.header_state.text() == "预览已过期"
    assert window._current_settings().detrend_phase is False  # noqa: SLF001
    window.close()
    application.processEvents()


@pytest.mark.parametrize(
    ("delay_samples", "expected_text"),
    [(3, "+3 ns（DUT 较晚）"), (-3, "-3 ns（DUT 较早）")],
)
def test_ui_reports_signed_three_nanosecond_relative_delay(
    delay_samples: int,
    expected_text: str,
) -> None:
    application = _qt_application()
    window = ResponseLabWindow()

    window.present_run(_signed_delay_run(delay_samples))

    assert expected_text in window.metric_label.text()
    assert "已去除线性相位" in window.metric_label.text()
    assert window.detrend_phase_checkbox.isChecked()
    window.close()
    application.processEvents()


def test_phase_display_and_legend_follow_disabled_detrend_setting() -> None:
    application = _qt_application()
    original = build_demo_run()
    settings = replace(original.analysis.settings, detrend_phase=False)
    run = run_compensation(
        original.reference_pulse,
        original.dut_pulse,
        original.input_signal,
        settings,
    )
    window = ResponseLabWindow()

    window.present_run(run)

    assert not window.detrend_phase_checkbox.isChecked()
    assert "已保留线性相位" in window.metric_label.text()
    assert "补偿相位保留线性趋势" in window.band_legend_label.text()
    assert [curve.name() for curve in window.response_plots[1].listDataItems()] == [
        "参考",
        "待补偿",
    ]
    assert [
        curve.name() for curve in window.difference_plots[1].listDataItems()
    ] == ["相位差（未去斜，实际补偿）"]
    window.close()
    application.processEvents()


def test_plot_toolbar_exposes_zoom_pan_and_reset_modes() -> None:
    application = _qt_application()
    window = ResponseLabWindow()

    assert window.zoom_button.text() == ""
    assert window.pan_button.text() == ""
    assert window.reset_button.text() == ""
    assert not window.zoom_button.icon().isNull()
    assert not window.pan_button.icon().isNull()
    assert not window.reset_button.icon().isNull()
    assert window.zoom_button.toolTip().startswith("左键拖出矩形区域")
    assert window.pan_button.toolTip().startswith("按住左键拖动画布")
    assert window.reset_button.toolTip().startswith("恢复当前数据")
    assert window.pan_button.isChecked()
    assert all(
        plot.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode
        for plot in window._all_plots()  # noqa: SLF001
    )

    QTest.mouseClick(window.zoom_button, Qt.MouseButton.LeftButton)
    assert window.zoom_button.isChecked()
    assert all(
        plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
        for plot in window._all_plots()  # noqa: SLF001
    )

    QTest.mouseClick(window.pan_button, Qt.MouseButton.LeftButton)
    assert window.pan_button.isChecked()
    assert all(
        plot.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode
        for plot in window._all_plots()  # noqa: SLF001
    )

    pulse_plot = window.pulse_plots[0]
    pulse_plot.plot([0.0, 1.0], [0.0, 1.0])
    pulse_plot.autoRange()
    pulse_plot.setXRange(0.4, 0.6, padding=0.0)
    QTest.mouseClick(window.reset_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    restored_low, restored_high = pulse_plot.viewRange()[0]
    assert restored_low < 0.1
    assert restored_high > 0.9
    window.close()
    application.processEvents()


def test_phase_plots_use_equivalent_centered_cycles_and_label_fit_boundaries() -> None:
    application = _qt_application()
    original = build_demo_run()
    shifted_analysis = replace(
        original.analysis,
        reference_phase_rad=original.analysis.reference_phase_rad + 2.0 * np.pi,
        phase_difference_rad=original.analysis.phase_difference_rad + 2.0 * np.pi,
    )
    run = replace(original, analysis=shifted_analysis)
    window = ResponseLabWindow()

    window.present_run(original)
    original_phase_curves = [
        np.asarray(curve.getData()[1]).copy()
        for curve in window.response_plots[1].listDataItems()
    ]
    window.present_run(run)

    shifted_phase_curves = [
        np.asarray(curve.getData()[1])
        for curve in window.response_plots[1].listDataItems()
    ]
    for original_curve, shifted_curve in zip(
        original_phase_curves,
        shifted_phase_curves,
        strict=True,
    ):
        np.testing.assert_allclose(shifted_curve, original_curve, atol=1.0e-10)
    continuous_phase_deg = window._center_phase_islands(  # noqa: SLF001
        np.radians(np.array([160.0, 170.0, 180.0, 190.0, 200.0])),
        np.ones(5, dtype=bool),
        np.ones(5, dtype=np.float64),
    )
    np.testing.assert_allclose(np.diff(continuous_phase_deg), 10.0, atol=1.0e-12)
    _, displayed_difference_deg = window.difference_plots[1].listDataItems()[0].getData()
    finite_difference = np.asarray(displayed_difference_deg)[
        np.isfinite(displayed_difference_deg)
    ]
    assert np.max(np.abs(finite_difference)) <= 360.0
    phase_items = window.difference_plots[1].getPlotItem().items
    assert sum(isinstance(item, pg.LinearRegionItem) for item in phase_items) == 1
    assert sum(isinstance(item, pg.InfiniteLine) for item in phase_items) == 2
    assert "+2.5 ns（DUT 较晚）" in window.metric_label.text()
    assert "相位拟合频带" in window.metric_label.text()
    assert [
        curve.name() for curve in window.difference_plots[1].listDataItems()
    ] == ["相位差（去斜前）", "实际补偿相位（去斜后）"]
    assert window.band_legend_label.text() == (
        "蓝色阴影：实际补偿频带　橙色虚线：线性相位拟合频带边界　"
        "补偿相位已去线性趋势"
    )
    window.close()
    application.processEvents()


def test_response_phase_plot_removes_each_pulse_linear_phase_for_comparison() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.present_run(build_demo_run())

    reference_curve, dut_curve = window.response_plots[1].listDataItems()
    frequency_ghz, reference_phase_deg = reference_curve.getData()
    _, dut_phase_deg = dut_curve.getData()
    comparison = (
        np.isfinite(reference_phase_deg)
        & np.isfinite(dut_phase_deg)
        & (np.asarray(frequency_ghz) >= 0.02)
        & (np.asarray(frequency_ghz) <= 0.25)
    )

    assert np.count_nonzero(comparison) > 100
    np.testing.assert_allclose(
        np.asarray(reference_phase_deg)[comparison],
        np.asarray(dut_phase_deg)[comparison],
        atol=1.0e-8,
    )
    window.close()
    application.processEvents()


def test_magnitude_mode_does_not_label_raw_response_phase_as_detrended() -> None:
    application = _qt_application()
    original = build_demo_run()
    magnitude_analysis = replace(
        original.analysis,
        settings=replace(original.analysis.settings, mode="magnitude"),
    )
    window = ResponseLabWindow()
    window.present_run(replace(original, analysis=magnitude_analysis))

    assert [curve.name() for curve in window.response_plots[1].listDataItems()] == [
        "参考",
        "待补偿",
    ]
    assert "去斜" not in window.response_plots[1].getAxis("left").labelText
    window.close()
    application.processEvents()


def test_reset_restores_loaded_output_preview_range() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.present_run(build_demo_run())
    waveform_plot = window.output_plots[0]
    difference_plot = window.difference_plots[0]
    compensation_phase_plot = window.compensator_plots[1]
    expected_waveform_x = waveform_plot.viewRange()[0]
    expected_difference_y = difference_plot.viewRange()[1]
    expected_compensation_phase_y = compensation_phase_plot.viewRange()[1]
    waveform_plot.setXRange(2.0, 3.0, padding=0.0)
    difference_plot.setYRange(2.85, 2.86, padding=0.0)
    compensation_phase_plot.setYRange(-0.01, 0.01, padding=0.0)

    QTest.mouseClick(window.reset_button, Qt.MouseButton.LeftButton)
    application.processEvents()

    np.testing.assert_allclose(
        waveform_plot.viewRange()[0], expected_waveform_x, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        difference_plot.viewRange()[1], expected_difference_y, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        compensation_phase_plot.viewRange()[1],
        expected_compensation_phase_y,
        rtol=0.0,
        atol=1e-9,
    )
    window.close()
    application.processEvents()


def test_initial_magnitude_view_fits_visible_band_and_reset_stays_fixed() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.present_run(build_demo_run())
    window.show()
    application.processEvents()
    magnitude_plot = window.response_plots[0]
    x_low, x_high = magnitude_plot.viewRange()[0]
    visible_values = []
    for curve in magnitude_plot.listDataItems():
        x, y = curve.getData()
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(y) & (x >= x_low) & (x <= x_high)
        visible_values.extend(y[mask])
    visible_values = np.asarray(visible_values)
    data_low = float(np.min(visible_values))
    data_high = float(np.max(visible_values))
    y_low, y_high = magnitude_plot.viewRange()[1]

    assert y_low <= data_low
    assert y_high >= data_high
    assert y_high - y_low <= max(8.0, 2.0 * (data_high - data_low))
    initial_recommended = np.asarray(magnitude_plot.viewRange(), dtype=np.float64)

    magnitude_plot.setXRange(0.12, 0.14, padding=0.0)
    magnitude_plot.setYRange(-40.0, -30.0, padding=0.0)
    QTest.mouseClick(window.reset_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    restored = np.asarray(magnitude_plot.viewRange(), dtype=np.float64)
    np.testing.assert_allclose(restored, initial_recommended, atol=1.0e-12)
    for index in range(window.visual_tabs.count()):
        window.visual_tabs.setCurrentIndex(index)
        application.processEvents()
        QTest.qWait(10)

    np.testing.assert_allclose(magnitude_plot.viewRange(), restored, atol=1.0e-12)
    assert magnitude_plot.getViewBox().state["autoRange"] == [False, False]
    window.close()
    application.processEvents()


def test_frequency_unit_switch_preserves_low_physical_frequencies() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.auto_frequency_bands.setChecked(False)
    window.frequency_unit_combo.setCurrentText("Hz")
    window.band_low.setValue(1.0)
    window.band_high.setValue(10.0)
    window.phase_low.setValue(2.0)
    window.phase_high.setValue(8.0)
    assert all(
        spin.suffix() == " Hz"
        for spin in (window.band_low, window.band_high, window.phase_low, window.phase_high)
    )
    before = window._current_settings()  # noqa: SLF001
    version_before = window._parameter_version  # noqa: SLF001

    window.frequency_unit_combo.setCurrentText("GHz")

    after = window._current_settings()  # noqa: SLF001
    assert after.band_low_hz == pytest.approx(before.band_low_hz, abs=1.0e-12)
    assert after.band_high_hz == pytest.approx(before.band_high_hz, abs=1.0e-12)
    assert all(
        spin.suffix() == " GHz"
        for spin in (window.band_low, window.band_high, window.phase_low, window.phase_high)
    )
    assert window._parameter_version == version_before  # noqa: SLF001
    window.close()
    application.processEvents()


def test_manual_mode_without_analysis_requires_explicit_frequency_values() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.frequency_unit_combo.setCurrentText("MHz")

    window.auto_frequency_bands.setChecked(False)

    for spin in (window.band_low, window.band_high, window.phase_low, window.phase_high):
        assert spin.value() == 0.0
        assert spin.text() == "请输入"
    with pytest.raises(ValueError, match="补偿频带"):
        window._current_settings()  # noqa: SLF001
    window.close()
    application.processEvents()


def test_ui_uses_concise_single_concept_labels() -> None:
    application = _qt_application()
    window = ResponseLabWindow()

    tab_labels = [
        window.visual_tabs.tabText(index)
        for index in range(window.visual_tabs.count())
    ]
    assert tab_labels == [
        "拟合脉冲",
        "频率响应",
        "频响差异比较",
        "频响补偿",
        "输出预览",
        "影响频段",
    ]
    assert all("/" not in label for label in tab_labels)

    visible_text = "\n".join(label.text() for label in window.findChildren(QLabel))
    # 分析摘要已从常驻界面移除，避免与图表和设置中的信息重复。
    assert "相对时延" not in visible_text
    assert "无表头 CSV" not in visible_text
    assert "第 1 列时间" not in visible_text
    assert "从时间列推导" not in visible_text
    assert "CSV 时间单位" not in visible_text
    assert "PCHIP" not in visible_text
    assert not hasattr(window, "time_unit_combo")
    assert not hasattr(window, "advanced_panel")
    assert not hasattr(window, "max_boost_db")
    assert not hasattr(window, "max_cut_db")
    assert not hasattr(window, "fir_length")
    assert not hasattr(window, "floor_db")
    assert "频响分析与补偿" in visible_text
    assert "仅显示" not in visible_text
    assert "不写入" not in visible_text
    assert "已移除" not in visible_text
    assert "未执行" not in visible_text
    assert window.windowTitle() == "ResponseLab · 频响分析与补偿"
    assert window.auto_frequency_bands.isChecked()
    assert window.band_low.text() == "分析后自动"
    assert window.band_high.text() == "分析后自动"
    assert window.phase_low.text() == "首次分析自动建议"
    assert window.phase_high.text() == "首次分析自动建议"
    assert window.phase_low.isEnabled()
    assert window.phase_high.isEnabled()
    assert window.detrend_phase_checkbox.isEnabled()
    assert window.detrend_phase_checkbox.isChecked()

    window.close()
    application.processEvents()


def test_band_change_updates_physical_compensation_range() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.auto_frequency_bands.setChecked(False)

    window.band_low.setValue(1.0)
    window.band_high.setValue(30.0)
    window.phase_low.setValue(2.0)
    window.phase_high.setValue(20.0)

    settings = window._current_settings()  # noqa: SLF001
    assert settings.band_low_hz == pytest.approx(1.0e9)
    assert settings.band_high_hz == pytest.approx(30.0e9)
    window.close()
    application.processEvents()


def test_frequency_plots_focus_on_analysis_band(tmp_path) -> None:
    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    window.auto_frequency_bands.setChecked(False)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_csv_path)
    window.band_low.setValue(0.01)
    window.band_high.setValue(0.10)
    window.phase_low.setValue(0.02)
    window.phase_high.setValue(0.08)

    window._start_analysis()  # noqa: SLF001 - exercise the primary action
    _wait_for_analysis(window, application)

    assert window._run is not None  # noqa: SLF001
    assert window._run.warnings == ()  # noqa: SLF001
    assert not window.result_warning.isVisible()

    frequency_plots = [
        *window.response_plots,
        *window.difference_plots,
        *window.compensator_plots[:2],
        window.output_plots[1],
    ]
    for plot in frequency_plots:
        x_low, x_high = plot.viewRange()[0]
        assert x_low <= 0.0
        assert 0.10 < x_high < 0.20

    window.close()
    application.processEvents()


def test_window_runs_headerless_csv_then_manual_rate_bin_workflow(tmp_path) -> None:
    reference_path, dut_path, target_csv_path, target_bin_path = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    application.processEvents()
    window.auto_frequency_bands.setChecked(False)
    window.band_low.setValue(0.01)
    window.band_high.setValue(0.30)
    window.phase_low.setValue(0.02)
    window.phase_high.setValue(0.25)

    analyze_position = window.analyze_button.mapTo(window, QPoint(0, 0))
    assert analyze_position.y() + window.analyze_button.height() <= window.height()
    before = window._current_settings()  # noqa: SLF001
    window.frequency_unit_combo.setCurrentText("MHz")
    after = window._current_settings()  # noqa: SLF001
    assert window.band_high.value() == pytest.approx(300.0)
    assert after.band_high_hz == pytest.approx(before.band_high_hz)
    window.frequency_unit_combo.setCurrentText("GHz")

    window.mode_combo.setCurrentIndex(1)
    assert not window.phase_low.isEnabled()
    window.mode_combo.setCurrentIndex(2)
    assert window.phase_low.isEnabled()
    window.mode_combo.setCurrentIndex(0)

    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_csv_path)
    assert not window.bin_group.isVisible()
    window._start_analysis()  # noqa: SLF001 - exercise the same slot as the primary button
    _wait_for_analysis(window, application)

    assert window._run is not None  # noqa: SLF001
    assert window._run.input_signal.source_format == "csv"  # noqa: SLF001
    assert window._run.input_signal.sample_rate_hz == pytest.approx(1.0e9)  # noqa: SLF001
    assert window.export_button.isEnabled()

    window.target_card.set_path(target_bin_path)
    assert window.bin_sample_rate.value() == 0.0
    window.bin_sample_rate.setValue(1.0e9)
    assert window.bin_group.isVisible()
    analyze_position = window.analyze_button.mapTo(window, QPoint(0, 0))
    assert analyze_position.y() + window.analyze_button.height() <= window.height()
    assert not window.export_button.isEnabled()
    window._start_analysis()  # noqa: SLF001
    _wait_for_analysis(window, application)

    assert window._run is not None  # noqa: SLF001
    assert window._run.input_signal.source_format == "bin"  # noqa: SLF001
    assert window._run.input_signal.sample_rate_hz == pytest.approx(1.0e9)  # noqa: SLF001
    assert window.export_button.isEnabled()
    window.close()
    application.processEvents()


def test_bin_import_keeps_advanced_parsing_defaults_collapsed() -> None:
    """普通 BIN 工作流只要求采样率，解析细节按需展开。"""

    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    application.processEvents()

    # 用户选择 BIN 后应直接看到唯一必填的物理参数，而不是一整组解析器内部默认值。
    window.target_card.set_path("/tmp/target.bin")
    application.processEvents()

    assert window.bin_group.isVisible()
    assert window.bin_sample_rate.isVisible()
    assert not window.bin_advanced_toggle.isChecked()
    assert not window.bin_advanced_fields.isVisible()

    # 多通道、整数编码或带文件头的数据仍可显式展开全部解析参数。
    window.bin_advanced_toggle.setChecked(True)
    application.processEvents()

    assert window.bin_advanced_fields.isVisible()
    assert window.bin_dtype.isVisible()
    assert window.bin_value_offset.isVisible()
    window.close()
    application.processEvents()


def test_closing_during_analysis_waits_for_worker(tmp_path, monkeypatch) -> None:
    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path)
    original_run = AnalysisThread.run

    def delayed_run(worker):
        time.sleep(0.2)
        original_run(worker)

    monkeypatch.setattr(AnalysisThread, "run", delayed_run)
    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_csv_path)
    window._start_analysis()  # noqa: SLF001

    window.close()
    application.processEvents()
    assert window.isVisible()
    assert window._close_when_finished is True  # noqa: SLF001
    deadline = time.monotonic() + 10.0
    while window.isVisible() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(20)

    assert not window.isVisible()
    assert window._worker is None  # noqa: SLF001


def test_ui_exports_bundle_then_invalidates_preview_when_source_changes(
    tmp_path,
    monkeypatch,
) -> None:
    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path / "inputs")
    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_csv_path)
    window._start_analysis()  # noqa: SLF001
    _wait_for_analysis(window, application)

    first_output = tmp_path / "first.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(first_output), "CSV (*.csv)"),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    window._export()  # noqa: SLF001
    assert all(path.is_file() for path in bundle_paths(first_output).as_tuple())

    target_csv_path.write_bytes(target_csv_path.read_bytes() + b"\n")
    second_output = tmp_path / "second.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(second_output), "CSV (*.csv)"),
    )
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )
    window._export()  # noqa: SLF001

    assert errors and "源文件" in errors[0]
    assert window.header_state.text() == "源文件已变化"
    assert not window.export_button.isEnabled()
    assert not any(path.exists() for path in bundle_paths(second_output).as_tuple())
    window.close()
    application.processEvents()
