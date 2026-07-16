"""真实 Qt 窗口中的无表头 CSV 与手工采样率 BIN 工作流。"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

from response_lab.app import _qt_application
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


def _wait_for_analysis(window: ResponseLabWindow, application, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        application.processEvents()
        if window._worker is None and window._run is not None:  # noqa: SLF001 - GUI smoke-test
            return
        QTest.qWait(20)
    raise AssertionError("等待 GUI 分析完成超时")


def test_frequency_unit_switch_preserves_low_physical_frequencies() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.frequency_unit_combo.setCurrentText("Hz")
    window.band_low.setValue(1.0)
    window.band_high.setValue(10.0)
    window.phase_low.setValue(2.0)
    window.phase_high.setValue(8.0)
    before = window._current_settings()  # noqa: SLF001
    version_before = window._parameter_version  # noqa: SLF001

    window.frequency_unit_combo.setCurrentText("GHz")

    after = window._current_settings()  # noqa: SLF001
    assert after.band_low_hz == pytest.approx(before.band_low_hz, abs=1.0e-12)
    assert after.band_high_hz == pytest.approx(before.band_high_hz, abs=1.0e-12)
    assert window._parameter_version == version_before  # noqa: SLF001
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
    ]
    assert all("/" not in label for label in tab_labels)

    visible_text = "\n".join(label.text() for label in window.findChildren(QLabel))
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
    assert window.band_low.text() == "0.01"
    assert window.band_high.text() == "0.3"

    window.close()
    application.processEvents()


def test_band_change_updates_physical_compensation_range() -> None:
    application = _qt_application()
    window = ResponseLabWindow()

    window.band_low.setValue(1.0)
    window.band_high.setValue(30.0)

    settings = window._current_settings()  # noqa: SLF001
    assert settings.band_low_hz == pytest.approx(1.0e9)
    assert settings.band_high_hz == pytest.approx(30.0e9)
    window.close()
    application.processEvents()


def test_frequency_plots_focus_on_analysis_band(tmp_path) -> None:
    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
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
