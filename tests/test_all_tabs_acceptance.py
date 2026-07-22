"""六个页签的真实按钮工作流与独立闭式数值验收。"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from response_lab.app import _qt_application
from response_lab.ui import ResponseLabWindow


def _write_keysight_xy(
    path: Path,
    time_s: np.ndarray,
    values_v: np.ndarray,
    *,
    source_name: str,
) -> None:
    """写出独立构造的 Keysight WaveformXYValues v2 两列表。"""

    header = "\n".join(
        (
            "File Format, WaveformXYValues",
            "Format Version, 2",
            "Instrument, D9300A",
            f"Points, {time_s.size}",
            f"Source Name, {source_name}",
            "X Units, Second",
            "Y Units, Volt",
            "Data,",
            "double, double",
        )
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(header + "\n")
        np.savetxt(stream, np.column_stack((time_s, values_v)), delimiter=",")


def _wait_for_run(
    window: ResponseLabWindow,
    application,
    *,
    timeout_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        application.processEvents()
        if window._worker is None and window._run is not None:  # noqa: SLF001
            return
        QTest.qWait(20)
    raise AssertionError("等待六页签补偿工作流完成超时")


def _curve_xy(plot, index: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = plot.listDataItems()[index].getData()
    return np.asarray(x_values), np.asarray(y_values)


def test_keysight_csv_compensation_populates_all_tabs_with_closed_form_values(
    tmp_path: Path,
) -> None:
    """0.5 倍 DUT 应产生 +6.0206 dB 补偿，并让输出严格变为输入的两倍。"""

    sample_rate_hz = 1.0e9
    # 奇数记录没有独立 Nyquist bin；全带 2 倍补偿因此可由时域闭式结果直接验算。
    samples = 511
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    reference = np.zeros(samples, dtype=np.float64)
    dut = np.zeros(samples, dtype=np.float64)
    reference[samples // 2] = 1.0
    dut[samples // 2] = 0.5
    target = np.random.default_rng(20260722).normal(size=samples)

    reference_path = tmp_path / "reference_keysight.csv"
    dut_path = tmp_path / "dut_keysight.csv"
    target_path = tmp_path / "target_keysight.csv"
    _write_keysight_xy(reference_path, time_s, reference, source_name="Reference")
    _write_keysight_xy(dut_path, time_s, dut, source_name="DUT")
    _write_keysight_xy(target_path, time_s, target, source_name="Channel 1")

    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    application.processEvents()
    try:
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("magnitude"))
        window.auto_frequency_bands.setChecked(False)
        window.frequency_unit_combo.setCurrentText("GHz")
        window.band_low.setValue(0.0)
        window.band_high.setValue(0.5)
        window.reference_card.set_path(reference_path)
        window.dut_card.set_path(dut_path)
        window.target_card.set_path(target_path)

        QTest.mouseClick(window.compensate_button, Qt.MouseButton.LeftButton)
        _wait_for_run(window, application)

        assert window.header_state.text() == "预览有效"
        assert window.export_button.isEnabled()
        assert window._run is not None  # noqa: SLF001
        assert window._run.warnings == ()  # noqa: SLF001
        assert {
            series.source_metadata["container"]
            for series in (
                window._run.reference_pulse,  # noqa: SLF001
                window._run.dut_pulse,  # noqa: SLF001
                window._run.input_signal,  # noqa: SLF001
            )
        } == {"keysight_infiniium_waveform_xy"}

        expected_tabs = [
            "拟合脉冲",
            "频率响应",
            "频响差异比较",
            "频响补偿",
            "输出预览",
            "影响频段",
        ]
        assert [
            window.visual_tabs.tabText(index)
            for index in range(window.visual_tabs.count())
        ] == expected_tabs
        assert all(
            window.visual_tabs.isTabEnabled(index)
            for index in range(window.visual_tabs.count())
        )
        tab_bar = window.visual_tabs.tabBar()
        for index in range(window.visual_tabs.count()):
            QTest.mouseClick(
                tab_bar,
                Qt.MouseButton.LeftButton,
                pos=tab_bar.tabRect(index).center(),
            )
            application.processEvents()
            assert window.visual_tabs.currentIndex() == index

        pulse_curves = window.pulse_plots[0].listDataItems()
        assert [curve.name() for curve in pulse_curves] == [
            "参考拟合脉冲",
            "待补偿拟合脉冲",
        ]
        pulse_time_ns, displayed_reference = _curve_xy(window.pulse_plots[0], 0)
        dut_time_ns, displayed_dut = _curve_xy(window.pulse_plots[0], 1)
        np.testing.assert_allclose(pulse_time_ns, np.arange(samples), atol=0.0)
        np.testing.assert_allclose(dut_time_ns, np.arange(samples), atol=0.0)
        np.testing.assert_allclose(displayed_reference, reference, atol=0.0)
        np.testing.assert_allclose(displayed_dut, dut, atol=0.0)
        assert window.pulse_plots[0].getAxis("bottom").labelText == "时间"
        assert window.pulse_plots[0].getAxis("bottom").labelUnits == "ns"
        assert window.pulse_plots[0].getAxis("left").labelText == "幅值"

        gain_db = 20.0 * np.log10(2.0)
        reference_db = 20.0 * np.log10(1.0 / sample_rate_hz)
        response_reference_x, response_reference_y = _curve_xy(
            window.response_plots[0], 0
        )
        response_dut_x, response_dut_y = _curve_xy(window.response_plots[0], 1)
        assert [
            curve.name() for curve in window.response_plots[0].listDataItems()
        ] == ["参考", "待补偿"]
        np.testing.assert_allclose(response_reference_x, response_dut_x, atol=0.0)
        # 相位连续段插值会把公共网格的最外侧端点标为不可信，绘图主动断开这些点。
        assert 0.0 <= response_reference_x[0] < 1.0e-3
        assert 0.499 < response_reference_x[-1] <= 0.5
        # pyqtgraph 的峰值下采样会复制每个可视桶的左右端点，因此横轴允许相邻重复。
        assert np.all(np.diff(response_reference_x) >= 0.0)
        assert np.unique(response_reference_x).size > 1000
        np.testing.assert_allclose(response_reference_y, reference_db, atol=1.0e-10)
        np.testing.assert_allclose(response_dut_y, reference_db - gain_db, atol=1.0e-10)
        _, reference_phase_deg = _curve_xy(window.response_plots[1], 0)
        _, dut_phase_deg = _curve_xy(window.response_plots[1], 1)
        np.testing.assert_allclose(reference_phase_deg, dut_phase_deg, atol=1.0e-10)
        assert window.response_plots[0].getAxis("left").labelText == "幅度"
        assert window.response_plots[0].getAxis("left").labelUnits == "dB"
        assert window.response_plots[0].getAxis("bottom").labelText == "频率"
        assert window.response_plots[0].getAxis("bottom").labelUnits == "GHz"
        assert window.response_plots[1].getAxis("left").labelUnits == "°"

        difference_frequency_ghz, difference_db = _curve_xy(
            window.difference_plots[0], 0
        )
        difference_phase_frequency_ghz, difference_phase_deg = _curve_xy(
            window.difference_plots[1], 0
        )
        assert [
            curve.name() for curve in window.difference_plots[0].listDataItems()
        ] == ["参考 - 待补偿"]
        assert [
            curve.name() for curve in window.difference_plots[1].listDataItems()
        ] == ["相位差（仅诊断）"]
        np.testing.assert_allclose(
            difference_frequency_ghz, response_reference_x, atol=0.0
        )
        np.testing.assert_allclose(
            difference_phase_frequency_ghz, response_reference_x, atol=0.0
        )
        np.testing.assert_allclose(difference_db, gain_db, atol=1.0e-10)
        np.testing.assert_allclose(difference_phase_deg, 0.0, atol=1.0e-10)
        assert window.difference_plots[0].getAxis("left").labelText == "幅度差"
        assert window.difference_plots[1].getAxis("left").labelText == "相位差"
        assert window.difference_plots[0].getAxis("bottom").labelText == "频率"
        assert window.difference_plots[1].getAxis("bottom").labelText == "频率"
        assert window.difference_plots[0].getAxis("bottom").labelUnits == "GHz"
        assert window.difference_plots[1].getAxis("bottom").labelUnits == "GHz"
        assert window.difference_plots[0].getAxis("left").labelUnits == "dB"
        assert window.difference_plots[1].getAxis("left").labelUnits == "°"

        correction_frequency_ghz, correction_db = _curve_xy(
            window.compensator_plots[0], 0
        )
        correction_phase_frequency_ghz, correction_phase_deg = _curve_xy(
            window.compensator_plots[1], 0
        )
        assert [
            curve.name() for curve in window.compensator_plots[0].listDataItems()
        ] == ["补偿幅度"]
        assert [
            curve.name() for curve in window.compensator_plots[1].listDataItems()
        ] == ["补偿相位"]
        assert 0.0 <= correction_frequency_ghz[0] < 1.0e-3
        assert 0.499 < correction_frequency_ghz[-1] <= 0.5
        np.testing.assert_allclose(
            correction_phase_frequency_ghz, correction_frequency_ghz, atol=0.0
        )
        # 文本时间轴反推 Fs 时会有末位舍入，公共网格最末端可刚好落到频带外；
        # 除这一边界显示点外，整条补偿曲线必须是闭式 +6.0206 dB。
        assert np.count_nonzero(np.isclose(correction_db, gain_db, atol=1.0e-10)) >= (
            correction_db.size - 1
        )
        assert np.all(
            np.isclose(correction_db, gain_db, atol=1.0e-10)
            | np.isclose(correction_db, 0.0, atol=1.0e-10)
        )
        finite_correction_phase = np.isfinite(correction_phase_deg)
        assert np.count_nonzero(~finite_correction_phase) <= 2
        np.testing.assert_allclose(
            correction_phase_deg[finite_correction_phase], 0.0, atol=1.0e-10
        )
        assert window.compensator_plots[0].getAxis("left").labelText == "补偿幅度"
        assert window.compensator_plots[1].getAxis("left").labelText == "补偿相位"
        assert window.compensator_plots[0].getAxis("bottom").labelText == "频率"
        assert window.compensator_plots[1].getAxis("bottom").labelText == "频率"
        assert window.compensator_plots[0].getAxis("bottom").labelUnits == "GHz"
        assert window.compensator_plots[1].getAxis("bottom").labelUnits == "GHz"
        assert window.compensator_plots[0].getAxis("left").labelUnits == "dB"
        assert window.compensator_plots[1].getAxis("left").labelUnits == "°"

        output_time_ns, before_values = _curve_xy(window.output_plots[0], 0)
        output_after_time_ns, after_values = _curve_xy(window.output_plots[0], 1)
        assert [curve.name() for curve in window.output_plots[0].listDataItems()] == [
            "补偿前",
            "补偿后",
        ]
        np.testing.assert_allclose(output_time_ns, np.arange(samples), atol=0.0)
        np.testing.assert_allclose(output_after_time_ns, output_time_ns, atol=0.0)
        np.testing.assert_allclose(before_values, target, atol=0.0)
        np.testing.assert_allclose(after_values, 2.0 * target, rtol=1.0e-12, atol=1.0e-12)

        spectrum_frequency_ghz, before_spectrum_db = _curve_xy(
            window.output_plots[1], 0
        )
        after_spectrum_frequency_ghz, after_spectrum_db = _curve_xy(
            window.output_plots[1], 1
        )
        assert [curve.name() for curve in window.output_plots[1].listDataItems()] == [
            "补偿前",
            "补偿后",
        ]
        expected_frequency_ghz = np.fft.rfftfreq(
            samples, d=1.0 / sample_rate_hz
        ) / 1.0e9
        expected_before_spectrum_db = 20.0 * np.log10(
            np.abs(np.fft.rfft(target)) / np.max(np.abs(np.fft.rfft(2.0 * target)))
        )
        np.testing.assert_allclose(
            spectrum_frequency_ghz, expected_frequency_ghz, atol=0.0
        )
        np.testing.assert_allclose(
            after_spectrum_frequency_ghz, expected_frequency_ghz, atol=0.0
        )
        np.testing.assert_allclose(
            before_spectrum_db, expected_before_spectrum_db, atol=1.0e-10
        )
        np.testing.assert_allclose(
            after_spectrum_db,
            expected_before_spectrum_db + gain_db,
            atol=1.0e-10,
        )
        assert window.output_plots[1].getAxis("left").labelText == "相对 DFT 幅度"
        assert window.output_plots[0].getAxis("bottom").labelText == "时间"
        assert window.output_plots[0].getAxis("left").labelText == "幅值"
        assert window.output_plots[1].getAxis("bottom").labelText == "频率"
        assert window.output_plots[0].getAxis("bottom").labelUnits == "ns"
        assert window.output_plots[1].getAxis("bottom").labelUnits == "GHz"
        assert window.output_plots[1].getAxis("left").labelUnits == "dB"

        assert window.influence_page.start_button.isEnabled()
        assert window.influence_page.metric_combo.count() == 3
    finally:
        window.close()
        application.processEvents()


@pytest.mark.parametrize(
    ("mode", "expected_output_scale", "expected_correction_db"),
    [
        ("phase", -1.0, 0.0),
        ("both", -2.0, 20.0 * np.log10(2.0)),
    ],
)
def test_phase_and_combined_modes_run_through_the_real_gui_button(
    tmp_path: Path,
    mode: str,
    expected_output_scale: float,
    expected_correction_db: float,
) -> None:
    """负半幅 DUT 给出闭式 -1（仅相位）或 -2（幅相）输出。"""

    sample_rate_hz = 1.0e9
    samples = 511
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    reference = np.zeros(samples, dtype=np.float64)
    dut = np.zeros(samples, dtype=np.float64)
    reference[samples // 2] = 1.0
    dut[samples // 2] = -0.5
    target = np.random.default_rng(20260723).normal(size=samples)
    reference_path = tmp_path / f"reference_{mode}.csv"
    dut_path = tmp_path / f"dut_{mode}.csv"
    target_path = tmp_path / f"target_{mode}.csv"
    _write_keysight_xy(reference_path, time_s, reference, source_name="Reference")
    _write_keysight_xy(dut_path, time_s, dut, source_name="DUT")
    _write_keysight_xy(target_path, time_s, target, source_name="Channel 1")

    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    application.processEvents()
    try:
        window.mode_combo.setCurrentIndex(window.mode_combo.findData(mode))
        window.auto_frequency_bands.setChecked(False)
        window.frequency_unit_combo.setCurrentText("GHz")
        window.band_low.setValue(0.0)
        window.band_high.setValue(0.5)
        window.phase_low.setValue(0.05)
        window.phase_high.setValue(0.45)
        window.reference_card.set_path(reference_path)
        window.dut_card.set_path(dut_path)
        window.target_card.set_path(target_path)

        QTest.mouseClick(window.compensate_button, Qt.MouseButton.LeftButton)
        _wait_for_run(window, application)

        assert window._run is not None  # noqa: SLF001
        assert window._run.analysis.settings.mode == mode  # noqa: SLF001
        _, displayed_output = _curve_xy(window.output_plots[0], 1)
        np.testing.assert_allclose(
            displayed_output,
            expected_output_scale * target,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        _, correction_db = _curve_xy(window.compensator_plots[0], 0)
        assert np.count_nonzero(
            np.isclose(correction_db, expected_correction_db, atol=1.0e-10)
        ) >= (correction_db.size - 1)
        _, correction_phase_deg = _curve_xy(window.compensator_plots[1], 0)
        finite_phase = correction_phase_deg[np.isfinite(correction_phase_deg)]
        assert finite_phase.size >= correction_phase_deg.size - 2
        np.testing.assert_allclose(np.abs(finite_phase), 180.0, atol=1.0e-9)
    finally:
        window.close()
        application.processEvents()


def test_missing_files_fail_before_worker_and_leave_all_actions_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实按钮遇到缺失文件时必须原子失败，不能把六页签留在忙碌态。"""

    application = _qt_application()
    window = ResponseLabWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )
    window.reference_card.set_path(tmp_path / "missing-reference.csv")
    window.dut_card.set_path(tmp_path / "missing-dut.csv")
    window.target_card.set_path(tmp_path / "missing-target.csv")

    QTest.mouseClick(window.compensate_button, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert errors and "以下文件无法读取" in errors[0]
    assert window._worker is None  # noqa: SLF001
    assert window.compare_button.isEnabled()
    assert window.compensate_button.isEnabled()
    assert window.influence_page.start_button.isEnabled()
    assert not window.progress.isVisible()
    window.close()
    application.processEvents()
