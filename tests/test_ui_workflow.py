"""真实 Qt 窗口中的通用/Keysight CSV 与自描述 Keysight BIN 工作流。"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, Qt, QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

import response_lab.dsp as dsp_module
import response_lab.io as io_module
import response_lab.ui as ui_module
from response_lab.app import _qt_application, build_demo_run
from response_lab.dsp import run_compensation
from response_lab.io import save_bin_timeseries
from response_lab.models import (
    CompensationRun,
    CompensationSettings,
    PulseComparison,
    TimeSeries,
)
from response_lab.reporting import (
    BundleCleanupWarning,
    BundleRollbackError,
    bundle_paths,
)
from response_lab.ui import AnalysisThread, ResponseLabWindow


def test_vpp_summary_displays_fs_rs_m_and_ui_duration() -> None:
    """The visible summary must make the user's M-to-baud interpretation auditable."""

    cache = SimpleNamespace(
        sample_rate_hz=8.0e9,
        symbol_rate_hz=2.0e9,
        ui_duration_s=0.5e-9,
        settings=SimpleNamespace(samples_per_ui=4, method="frequency_rms_error"),
    )
    run = SimpleNamespace(
        workspace=SimpleNamespace(
            settings=SimpleNamespace(
                metric="vpp",
                vpp=SimpleNamespace(method="frequency_rms_error"),
            ),
            vpp_cache=cache,
        ),
        result=SimpleNamespace(reference_metric=0.0, before_metric=0.125),
    )

    summary = ResponseLabWindow._influence_metric_summary(run, 0.01)

    assert "Fs 8 GSa/s" in summary
    assert "Rs 2 GBd" in summary
    assert "M 4 samples/UI" in summary
    assert "UI 500 ps" in summary
    assert "补偿后误差 0.01 Vrms" in summary


@pytest.mark.parametrize("metric", ["eye_height", "eye_width"])
def test_eye_summary_displays_fs_rs_m_and_ui_duration(metric: str) -> None:
    """眼高和眼宽同样必须显示用户 M 对应的实际码率与 UI 时长。"""

    run = SimpleNamespace(
        workspace=SimpleNamespace(
            settings=SimpleNamespace(
                metric=metric,
                vpp=None,
                eye=SimpleNamespace(samples_per_ui=4),
            ),
            reference_pulse=SimpleNamespace(sample_rate_hz=8.0e9),
            vpp_cache=None,
        ),
        result=SimpleNamespace(reference_metric=0.25, before_metric=0.125),
    )

    summary = ResponseLabWindow._influence_metric_summary(run, 0.2)

    assert "Fs 8 GSa/s" in summary
    assert "Rs 2 GBd" in summary
    assert "M 4 samples/UI" in summary
    assert "UI 500 ps" in summary
    assert "补偿后 0.2" in summary


def test_multiseed_instability_is_the_visible_no_recommendation_reason(
    monkeypatch,
) -> None:
    """多种子不稳定必须直接显示，不能被较早的物理分辨率告警遮住。"""

    application = _qt_application()
    window = ResponseLabWindow()
    captured: dict[str, object] = {}
    monkeypatch.setattr(ui_module, "influence_curve_payload", lambda _run: {})
    monkeypatch.setattr(
        window.influence_page,
        "render_result",
        lambda payload: captured.update(payload),
    )
    run = SimpleNamespace(
        workspace=SimpleNamespace(
            settings=SimpleNamespace(
                metric="eye_height",
                vpp=None,
                eye=SimpleNamespace(samples_per_ui=4),
            ),
            reference_pulse=SimpleNamespace(sample_rate_hz=8.0e9),
            vpp_cache=None,
        ),
        result=SimpleNamespace(
            status="no_recommendation",
            recommendation=None,
            reference_metric=0.25,
            before_metric=0.125,
            warnings=(
                "请求频宽小于物理分辨率，已扩大候选窗",
                "眼图推荐未通过 3 个确定性符号种子的稳定性复核，已保守取消推荐",
            ),
        ),
        displayed_candidates=(),
        selected_evaluation=None,
        eye_comparison=None,
    )

    window._influence_succeeded(run, window._influence_version)  # noqa: SLF001

    assert "3 个确定性符号种子" in str(captured["summary"])
    assert "物理分辨率" in str(captured["summary"])
    assert "稳定性" in window.statusBar().currentMessage()
    assert "物理分辨率" in window.statusBar().currentMessage()

    captured.clear()
    recommendation = SimpleNamespace(
        mode="magnitude",
        band=SimpleNamespace(low_hz=100.0e6, high_hz=200.0e6),
    )
    passed_run = SimpleNamespace(
        workspace=run.workspace,
        result=SimpleNamespace(
            status="ok",
            recommendation=recommendation,
            reference_metric=0.25,
            before_metric=0.125,
            warnings=(
                "请求频宽小于物理分辨率，已扩大候选窗",
                "眼图推荐已通过 3 个确定性符号种子的稳定性复核",
            ),
        ),
        displayed_candidates=(),
        selected_evaluation=None,
        eye_comparison=None,
    )

    window._influence_succeeded(  # noqa: SLF001
        passed_run,
        window._influence_version,  # noqa: SLF001
    )

    assert "物理分辨率" in str(captured["summary"])
    assert "已通过 3 个确定性符号种子" in str(captured["summary"])
    assert "物理分辨率" in window.statusBar().currentMessage()
    assert "已通过 3 个确定性符号种子" in window.statusBar().currentMessage()
    window.close()
    application.processEvents()


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
    save_bin_timeseries(target_bin_path, signal_time_s, signal)
    return reference_path, dut_path, target_csv_path, target_bin_path


def _rewrite_as_keysight_waveform_xy(path, source_name: str) -> None:
    """Wrap an independent numeric fixture in the documented v2 instrument header."""

    table = np.loadtxt(path, delimiter=",", ndmin=2)
    header = "\n".join(
        (
            "File Format, WaveformXYValues",
            "Format Version, 2",
            "Instrument, D9300A",
            f"Points, {table.shape[0]}",
            f"Source Name, {source_name}",
            "X Units, Second",
            "Y Units, Volt",
            "Data,",
            "double, double",
        )
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(header + "\n")
        np.savetxt(stream, table, delimiter=",")


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


def _write_medium_rate_pulses(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 2.0e9
    samples = 2048
    index = np.arange(samples, dtype=np.float64)
    time_s = index / sample_rate_hz
    reference = np.exp(-0.5 * ((index - 480.0) / 4.0) ** 2)
    dut = 0.8 * reference
    reference_path = tmp_path / "medium_rate_reference.csv"
    dut_path = tmp_path / "medium_rate_dut.csv"
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


def _wait_for_task_completion(
    window: ResponseLabWindow,
    application,
    timeout_s: float = 10.0,
) -> None:
    """等待当前后台任务收尾，不把上一次留存结果误判为本次成功。"""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        application.processEvents()
        if window._worker is None:  # noqa: SLF001 - observe the real GUI task boundary
            return
        QTest.qWait(20)
    raise AssertionError("等待 GUI 后台任务收尾超时")


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

    QTest.mouseClick(window.compare_button, Qt.MouseButton.LeftButton)
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


def test_auto_phase_band_is_resuggested_when_switching_high_to_low_rate_pulses(
    tmp_path,
    monkeypatch,
) -> None:
    """同一窗口更换整套拟合脉冲时，自动相位频带必须按新 Fs 重算。"""

    high_reference, high_dut = _write_high_rate_pulses(tmp_path / "high")
    low_reference, low_dut, _, _ = _write_demo_inputs(tmp_path / "low")
    application = _qt_application()
    window = ResponseLabWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )

    window.reference_card.set_path(high_reference)
    window.dut_card.set_path(high_dut)
    window._start_comparison()  # noqa: SLF001 - exercise the real GUI request path
    _wait_for_task_completion(window, application)
    high_result = window._result  # noqa: SLF001
    assert isinstance(high_result, PulseComparison)
    assert high_result.analysis.settings.phase_fit_high_hz > 1.0e9

    window.reference_card.set_path(low_reference)
    window.dut_card.set_path(low_dut)
    window._start_comparison()  # noqa: SLF001 - same live window, new fitted-pulse set
    _wait_for_task_completion(window, application)
    low_result = window._result  # noqa: SLF001
    window.close()
    application.processEvents()

    assert errors == []
    assert isinstance(low_result, PulseComparison)
    assert low_result is not high_result
    assert low_result.analysis.settings.phase_fit_high_hz < 500.0e6


def test_fitted_pulse_change_keeps_visible_m_immediately_usable(tmp_path) -> None:
    """切换文件后继续使用当前可见 M，不要求额外确认动作。"""

    application = _qt_application()
    window = ResponseLabWindow()
    captured: list[dict[str, object]] = []
    window.influence_page.analysis_requested.disconnect(
        window._start_influence_analysis  # noqa: SLF001 - isolate page emission
    )
    window.influence_page.analysis_requested.connect(captured.append)
    reference_path, dut_path, _, _ = _write_demo_inputs(tmp_path / "initial")
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.influence_page.m_spin.setValue(33)
    window.influence_page.start_button.click()
    application.processEvents()
    assert [request["m"] for request in captured] == [33]

    new_reference = tmp_path / "new-reference.csv"
    new_reference.write_text("0,0\n", encoding="utf-8")
    window.reference_card.set_path(new_reference)
    window.influence_page.start_button.click()
    application.processEvents()

    window.close()
    application.processEvents()
    assert [request["m"] for request in captured] == [33, 33]
    assert not hasattr(window.influence_page, "m_confirmation_label")


def test_m_change_does_not_start_a_confirmation_worker(tmp_path) -> None:
    """修改 M 只更新当前参数，不启动隐藏的确认任务。"""

    reference_path, dut_path, _, _ = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.influence_page.m_spin.setValue(33)
    application.processEvents()

    assert window._worker is None  # noqa: SLF001
    assert window.influence_page.current_request()["m"] == 33
    window.close()
    application.processEvents()


def test_changing_inputs_interrupts_obsolete_analysis_worker(monkeypatch) -> None:
    """普通比较/补偿过期后应协作停止旧任务，而不是让大文件白跑到底。"""

    application = _qt_application()
    window = ResponseLabWindow()
    worker = AnalysisThread.__new__(AnalysisThread)
    QThread.__init__(worker)
    interruption_requests: list[bool] = []
    monkeypatch.setattr(
        worker,
        "requestInterruption",
        lambda: interruption_requests.append(True),
    )
    window._worker = worker  # noqa: SLF001
    window._active_action = "compensate"  # noqa: SLF001

    window._mark_stale()  # noqa: SLF001

    assert interruption_requests == [True]
    window._worker = None  # noqa: SLF001
    worker.deleteLater()
    window.close()
    application.processEvents()


def test_auto_phase_band_is_resuggested_when_switching_low_to_high_rate_pulses(
    tmp_path,
    monkeypatch,
) -> None:
    """自动相位频带重算不能只修复高 Fs 到低 Fs 的越界方向。"""

    low_reference, low_dut, _, _ = _write_demo_inputs(tmp_path / "low")
    high_reference, high_dut = _write_high_rate_pulses(tmp_path / "high")
    application = _qt_application()
    window = ResponseLabWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )

    window.reference_card.set_path(low_reference)
    window.dut_card.set_path(low_dut)
    window._start_comparison()  # noqa: SLF001 - exercise the real GUI request path
    _wait_for_task_completion(window, application)
    low_result = window._result  # noqa: SLF001
    assert isinstance(low_result, PulseComparison)
    assert low_result.analysis.settings.phase_fit_high_hz < 500.0e6

    window.reference_card.set_path(high_reference)
    window.dut_card.set_path(high_dut)
    window._start_comparison()  # noqa: SLF001 - same live window, new fitted-pulse set
    _wait_for_task_completion(window, application)
    high_result = window._result  # noqa: SLF001
    window.close()
    application.processEvents()

    assert errors == []
    assert isinstance(high_result, PulseComparison)
    assert high_result is not low_result
    assert high_result.analysis.settings.phase_fit_high_hz > 1.0e9


def test_auto_phase_band_resuggestion_is_used_by_compensation_path(
    tmp_path,
    monkeypatch,
) -> None:
    """普通数据补偿与拟合脉冲比较应共享同一个文件切换修复。"""

    high_reference, high_dut = _write_high_rate_pulses(tmp_path / "high")
    low_reference, low_dut, low_target, _ = _write_demo_inputs(tmp_path / "low")
    application = _qt_application()
    window = ResponseLabWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )

    window.reference_card.set_path(high_reference)
    window.dut_card.set_path(high_dut)
    window._start_comparison()  # noqa: SLF001 - establish a high-Fs automatic band
    _wait_for_task_completion(window, application)
    assert isinstance(window._result, PulseComparison)  # noqa: SLF001

    window.reference_card.set_path(low_reference)
    window.dut_card.set_path(low_dut)
    window.target_card.set_path(low_target)
    window._start_compensation()  # noqa: SLF001 - exercise real load/compensate path
    _wait_for_task_completion(window, application)
    low_run = window._run  # noqa: SLF001
    window.close()
    application.processEvents()

    assert errors == []
    assert isinstance(low_run, CompensationRun)
    assert low_run.analysis.settings.phase_fit_high_hz < 500.0e6


def test_auto_bin_analysis_reuses_one_prepared_spectrum_per_pulse(
    tmp_path,
    monkeypatch,
) -> None:
    """BIN 自动频带的 preflight、建议和正式分析不能重复做六次脉冲 FFT。"""

    reference_path, dut_path, _, target_bin_path = _write_demo_inputs(tmp_path)
    request = ui_module.AnalysisRequest(
        reference_path=reference_path,
        dut_path=dut_path,
        target_path=target_bin_path,
        settings=CompensationSettings(
            mode="both",
            band_low_hz=10.0e6,
            band_high_hz=300.0e6,
            phase_fit_low_hz=20.0e6,
            phase_fit_high_hz=250.0e6,
        ),
        version=11,
        action="compensate",
        auto_frequency_bands=True,
        auto_phase_fit_band=True,
    )
    worker = AnalysisThread(request)
    real_pulse_spectrum = dsp_module._pulse_spectrum
    real_suggest = ui_module.suggest_frequency_settings
    calls: list[Path] = []
    suggestion_calls = 0

    def counting_pulse_spectrum(pulse, settings, **kwargs):
        calls.append(pulse.source_path)
        return real_pulse_spectrum(pulse, settings, **kwargs)

    def counting_suggest(*args, **kwargs):
        nonlocal suggestion_calls
        suggestion_calls += 1
        return real_suggest(*args, **kwargs)

    results: list[object] = []
    failures: list[str] = []
    monkeypatch.setattr(dsp_module, "_pulse_spectrum", counting_pulse_spectrum)
    monkeypatch.setattr(ui_module, "suggest_frequency_settings", counting_suggest)
    worker.succeeded.connect(lambda result, _version: results.append(result))
    worker.failed.connect(lambda message, _version: failures.append(message))

    worker.run()

    assert failures == []
    assert len(results) == 1
    assert suggestion_calls == 1
    assert calls == [reference_path, dut_path]


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


def test_manual_phase_band_is_frozen_for_automatic_influence_scan(
    tmp_path,
    monkeypatch,
) -> None:
    """影响频段请求也必须区分自动扫描范围与已经手工确认的相位拟合带。"""

    reference_path, dut_path = _write_high_rate_pulses(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    monkeypatch.setattr(ui_module.InfluenceAnalysisThread, "start", lambda _self: None)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.phase_low.setValue(5.0)
    window.phase_high.setValue(35.0)
    window.influence_page.start_button.click()
    application.processEvents()

    worker = window._worker  # noqa: SLF001 - verify the frozen production request
    assert isinstance(worker, ui_module.InfluenceAnalysisThread)
    assert worker.request.auto_frequency_bands is True
    assert worker.request.auto_phase_fit_band is False
    assert worker.request.frequency_settings.phase_fit_low_hz == pytest.approx(5.0e9)
    assert worker.request.frequency_settings.phase_fit_high_hz == pytest.approx(35.0e9)
    window.close()
    application.processEvents()


def test_influence_keeps_confirmed_phase_band_after_main_mode_becomes_magnitude(
    tmp_path,
    monkeypatch,
) -> None:
    """影响分析包含相位分支，不能沿用主“仅幅度”请求的 0–1 Hz 占位值。"""

    reference_path, dut_path = _write_high_rate_pulses(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    monkeypatch.setattr(ui_module.InfluenceAnalysisThread, "start", lambda _self: None)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.phase_low.setValue(5.0)
    window.phase_high.setValue(35.0)
    window.mode_combo.setCurrentIndex(window.mode_combo.findData("magnitude"))
    window.influence_page.start_button.click()
    application.processEvents()

    worker = window._worker  # noqa: SLF001 - inspect the exact frozen GUI request
    assert isinstance(worker, ui_module.InfluenceAnalysisThread)
    assert worker.request.frequency_settings.mode == "magnitude"
    assert worker.request.auto_phase_fit_band is False
    assert worker.request.frequency_settings.phase_fit_low_hz == pytest.approx(5.0e9)
    assert worker.request.frequency_settings.phase_fit_high_hz == pytest.approx(35.0e9)
    window.close()
    application.processEvents()


def test_manual_phase_band_survives_switching_fitted_pulse_files(tmp_path) -> None:
    """用户明确输入的相位拟合带应跨文件保留，不得被自动重算。"""

    low_reference, low_dut, _, _ = _write_demo_inputs(tmp_path / "low")
    medium_reference, medium_dut = _write_medium_rate_pulses(tmp_path / "medium")
    application = _qt_application()
    window = ResponseLabWindow()
    window.phase_low.setValue(0.02)
    window.phase_high.setValue(0.12)

    window.reference_card.set_path(low_reference)
    window.dut_card.set_path(low_dut)
    window._start_comparison()  # noqa: SLF001 - exercise the real GUI request path
    _wait_for_task_completion(window, application)
    low_result = window._result  # noqa: SLF001
    assert isinstance(low_result, PulseComparison)
    assert low_result.analysis.settings.phase_fit_low_hz == pytest.approx(20.0e6)
    assert low_result.analysis.settings.phase_fit_high_hz == pytest.approx(120.0e6)

    window.reference_card.set_path(medium_reference)
    window.dut_card.set_path(medium_dut)
    assert window.phase_low.value() == pytest.approx(0.02)
    assert window.phase_high.value() == pytest.approx(0.12)
    window._start_comparison()  # noqa: SLF001 - same live window, new fitted-pulse set
    _wait_for_task_completion(window, application)
    medium_result = window._result  # noqa: SLF001
    window.close()
    application.processEvents()

    assert isinstance(medium_result, PulseComparison)
    assert medium_result is not low_result
    assert medium_result.analysis.settings.phase_fit_low_hz == pytest.approx(20.0e6)
    assert medium_result.analysis.settings.phase_fit_high_hz == pytest.approx(120.0e6)


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
    assert "自动选择：精确整段" in window.statusBar().currentMessage()
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


def test_ui_exposes_gain_limit_and_edge_transition_as_explicit_settings() -> None:
    application = _qt_application()
    window = ResponseLabWindow()

    assert window.limit_gain_checkbox.isChecked()
    assert window.maximum_gain_db.value() == pytest.approx(20.0)
    assert window.edge_transition_percent.value() == pytest.approx(10.0)
    defaults = window._current_settings()  # noqa: SLF001
    assert defaults.maximum_gain_db == pytest.approx(20.0)
    assert defaults.edge_transition_fraction == pytest.approx(0.10)

    window.limit_gain_checkbox.setChecked(False)
    assert not window.maximum_gain_db.isEnabled()
    unlimited = window._current_settings()  # noqa: SLF001
    assert unlimited.maximum_gain_db is None

    window.limit_gain_checkbox.setChecked(True)
    window.maximum_gain_db.setValue(12.0)
    window.edge_transition_percent.setValue(15.0)
    customized = window._current_settings()  # noqa: SLF001
    assert customized.maximum_gain_db == pytest.approx(12.0)
    assert customized.edge_transition_fraction == pytest.approx(0.15)
    window.close()
    application.processEvents()


def test_ui_uses_concise_single_concept_labels() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    application.processEvents()

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
    # 数据输入栏只保留文件角色与状态，不常驻展示CSV格式说明。
    assert "普通无表头 CSV" not in visible_text
    assert "第 1 列时间（s）" not in visible_text
    assert "第 2 列电压（V）" not in visible_text
    assert "仅接受两列" not in visible_text
    assert all(
        label.accessibleName() != "普通 CSV 输入格式"
        for label in window.findChildren(QLabel)
    )
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


def test_window_runs_headerless_csv_then_automatic_keysight_bin_workflow(
    tmp_path,
    monkeypatch,
) -> None:
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
    window._start_analysis()  # noqa: SLF001 - exercise the same slot as the primary button
    _wait_for_analysis(window, application)

    assert window._run is not None  # noqa: SLF001
    assert window._run.input_signal.source_format == "csv"  # noqa: SLF001
    assert window._run.input_signal.sample_rate_hz == pytest.approx(1.0e9)  # noqa: SLF001
    assert window.export_button.isEnabled()

    window.target_card.set_path(target_bin_path)
    # 包装真实加载器，核对界面不再给 BIN 独设一个与 CSV 不一致的固定点数门限。
    real_bin_loader = ui_module.load_bin_timeseries
    observed_sample_budgets: list[int | None] = []

    def recording_bin_loader(path, **kwargs):
        observed_sample_budgets.append(kwargs.get("max_samples"))
        return real_bin_loader(path, **kwargs)

    monkeypatch.setattr(ui_module, "load_bin_timeseries", recording_bin_loader)
    analyze_position = window.analyze_button.mapTo(window, QPoint(0, 0))
    assert analyze_position.y() + window.analyze_button.height() <= window.height()
    assert not window.export_button.isEnabled()
    window._start_analysis()  # noqa: SLF001
    _wait_for_analysis(window, application)

    assert window._run is not None  # noqa: SLF001
    assert window._run.input_signal.source_format == "bin"  # noqa: SLF001
    assert window._run.input_signal.sample_rate_hz == pytest.approx(1.0e9)  # noqa: SLF001
    assert observed_sample_budgets == [None]
    assert window.export_button.isEnabled()
    window.close()
    application.processEvents()


@pytest.mark.parametrize("wide_role", ["reference", "target"])
def test_window_rejects_wide_headerless_csv_instead_of_ignoring_extra_channel(
    tmp_path,
    monkeypatch,
    wide_role,
) -> None:
    """GUI 的固定单通道合同不得静默丢弃无表头 CSV 的额外列。"""

    reference_path, dut_path, target_path, _ = _write_demo_inputs(tmp_path)
    wide_path = reference_path if wide_role == "reference" else target_path
    wide_table = np.loadtxt(wide_path, delimiter=",", ndmin=2)
    np.savetxt(
        wide_path,
        np.column_stack((wide_table, np.full(wide_table.shape[0], 99.0))),
        delimiter=",",
    )
    application = _qt_application()
    window = ResponseLabWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    if wide_role == "target":
        window.target_card.set_path(target_path)

    if wide_role == "reference":
        window._start_comparison()  # noqa: SLF001 - real comparison thread path
    else:
        window._start_compensation()  # noqa: SLF001 - real compensation thread path
    _wait_for_task_completion(window, application)
    result = window._result  # noqa: SLF001
    window.close()
    application.processEvents()

    assert result is None
    assert len(errors) == 1
    assert "无表头 CSV" in errors[0]
    assert "恰好 2 列" in errors[0]


def test_bin_import_uses_self_describing_metadata_without_manual_controls() -> None:
    """Keysight BIN should expose only the automatic-metadata contract."""

    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    application.processEvents()

    # 选择 BIN 后只展示自动解析说明，不再要求用户猜采样率或字节布局。
    window.target_card.set_path("/tmp/target.bin")
    application.processEvents()

    assert not hasattr(window, "bin_sample_rate")
    assert not hasattr(window, "bin_advanced_toggle")
    window.close()
    application.processEvents()


def test_gui_rejects_bin_on_full_dsp_budget_before_target_payload_read(
    tmp_path,
    monkeypatch,
) -> None:
    """BIN 头部几何必须先通过完整补偿预算，之后才能哈希或映射 payload。"""

    reference_path, dut_path, _, target_bin_path = _write_demo_inputs(tmp_path)
    application = _qt_application()
    window = ResponseLabWindow()
    window.auto_frequency_bands.setChecked(False)
    window.band_low.setValue(0.01)
    window.band_high.setValue(0.30)
    window.phase_low.setValue(0.02)
    window.phase_high.setValue(0.25)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_bin_path)
    observed_plans: list[dict[str, object]] = []
    bin_payload_touched: list[bool] = []
    errors: list[str] = []
    real_snapshot = io_module._snapshot_open_file  # noqa: SLF001

    def reject_full_pipeline(**kwargs):
        observed_plans.append(dict(kwargs))
        raise MemoryError("完整补偿内存预检拒绝：测试预算不足")

    def observe_snapshot(handle, *args, **kwargs):
        if os.path.samefile(handle.name, target_bin_path):
            bin_payload_touched.append(True)
        return real_snapshot(handle, *args, **kwargs)

    monkeypatch.setattr(ui_module, "preflight_compensation_shape", reject_full_pipeline)
    monkeypatch.setattr(io_module, "_snapshot_open_file", observe_snapshot)
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )

    window._start_compensation()  # noqa: SLF001 - real GUI request and worker path
    _wait_for_task_completion(window, application)

    assert len(observed_plans) == 1, errors
    assert observed_plans[0]["target_samples"] == 4096
    assert observed_plans[0]["target_channels"] == 1
    assert observed_plans[0]["anticipated_input_resident_bytes"] == 4096 * 12
    assert bin_payload_touched == []
    assert errors and "完整补偿内存预检拒绝" in errors[0]
    assert window._result is None  # noqa: SLF001
    assert not window.export_button.isEnabled()
    window.close()
    application.processEvents()


def test_window_uses_keysight_xy_csv_for_both_pulses_and_target(tmp_path) -> None:
    reference_path, dut_path, target_path, _ = _write_demo_inputs(tmp_path)
    _rewrite_as_keysight_waveform_xy(reference_path, "Reference Pulse")
    _rewrite_as_keysight_waveform_xy(dut_path, "DUT Pulse")
    _rewrite_as_keysight_waveform_xy(target_path, "Channel 1")

    application = _qt_application()
    window = ResponseLabWindow()
    window.auto_frequency_bands.setChecked(False)
    window.band_low.setValue(0.01)
    window.band_high.setValue(0.30)
    window.phase_low.setValue(0.02)
    window.phase_high.setValue(0.25)
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_path)
    window.show()
    application.processEvents()

    QTest.mouseClick(window.compensate_button, Qt.MouseButton.LeftButton)
    _wait_for_analysis(window, application)

    assert window._run is not None  # noqa: SLF001
    for series in (
        window._run.reference_pulse,  # noqa: SLF001
        window._run.dut_pulse,  # noqa: SLF001
        window._run.input_signal,  # noqa: SLF001
    ):
        assert series.source_metadata["container"] == "keysight_infiniium_waveform_xy"
        assert series.source_metadata["sample_rate_source"] == "keysight_xy_time_column"
    assert window._run.input_signal.sample_rate_hz == pytest.approx(1.0e9)  # noqa: SLF001
    window.close()
    application.processEvents()


def test_closing_during_analysis_cooperatively_interrupts_worker(tmp_path, monkeypatch) -> None:
    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path)
    interruption_observed: list[bool] = []

    def delayed_run(worker):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if worker.isInterruptionRequested():
                interruption_observed.append(True)
                return
            time.sleep(0.01)
        interruption_observed.append(False)

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
    assert interruption_observed == [True]


def test_analysis_error_after_close_request_does_not_open_modal(monkeypatch) -> None:
    """关闭等待期间的真实错误只能静默收尾，不能用模态框阻断自动关闭。"""

    application = _qt_application()
    window = ResponseLabWindow()
    modals: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: modals.append(str(message)),
    )
    window._close_when_finished = True  # noqa: SLF001

    window._analysis_failed("late loader failure", window._parameter_version)  # noqa: SLF001

    assert modals == []
    assert "窗口" in window.statusBar().currentMessage()
    window._close_when_finished = False  # noqa: SLF001
    window.close()
    application.processEvents()


def test_late_success_after_close_request_skips_all_result_rendering(monkeypatch) -> None:
    """取消检查后的窄竞态不能让即将关闭的窗口再渲染大图或眼图。"""

    application = _qt_application()
    window = ResponseLabWindow()
    window._close_when_finished = True  # noqa: SLF001

    def forbidden_render(*_args, **_kwargs):
        raise AssertionError("closing window must discard late success payload")

    monkeypatch.setattr(window, "present_run", forbidden_render)
    monkeypatch.setattr(ui_module, "influence_curve_payload", forbidden_render)
    run = build_demo_run()

    window._analysis_succeeded(run, window._parameter_version)  # noqa: SLF001
    window._influence_succeeded(  # noqa: SLF001
        SimpleNamespace(),
        window._influence_version,  # noqa: SLF001
    )
    window._influence_selection_succeeded(  # noqa: SLF001
        SimpleNamespace(),
        window._influence_version,  # noqa: SLF001
    )

    assert window._run is None  # noqa: SLF001
    assert window._influence_run is None  # noqa: SLF001
    window._close_when_finished = False  # noqa: SLF001
    window.close()
    application.processEvents()


def test_analysis_cancellation_stops_before_loading_next_input(tmp_path, monkeypatch) -> None:
    """参考脉冲加载后收到关闭请求时，不应继续读取 DUT 或目标文件。"""

    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path)
    request = ui_module.AnalysisRequest(
        reference_path=reference_path,
        dut_path=dut_path,
        target_path=target_csv_path,
        settings=CompensationSettings(
            mode="magnitude",
            band_low_hz=10.0e6,
            band_high_hz=300.0e6,
        ),
        version=7,
        action="compensate",
        auto_frequency_bands=False,
    )
    worker = AnalysisThread(request)
    real_loader = ui_module.load_csv_timeseries
    loaded_paths: list[object] = []
    cancel_requested = False

    def cancelling_loader(path, **kwargs):
        nonlocal cancel_requested
        loaded_paths.append(path)
        result = real_loader(path, **kwargs)
        cancel_requested = True
        return result

    cancelled_versions: list[int] = []
    successes: list[object] = []
    failures: list[str] = []
    monkeypatch.setattr(ui_module, "load_csv_timeseries", cancelling_loader)
    monkeypatch.setattr(worker, "isInterruptionRequested", lambda: cancel_requested)
    worker.cancelled.connect(cancelled_versions.append)
    worker.succeeded.connect(lambda result, _version: successes.append(result))
    worker.failed.connect(lambda message, _version: failures.append(message))

    worker.run()

    assert loaded_paths == [reference_path]
    assert cancelled_versions == [7]
    assert successes == []
    assert failures == []


def test_export_runs_off_the_gui_thread(tmp_path, monkeypatch) -> None:
    """大文件整包导出不能阻塞 Qt 主事件循环。"""

    reference_path, dut_path, target_csv_path, _ = _write_demo_inputs(tmp_path / "inputs")
    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    window.reference_card.set_path(reference_path)
    window.dut_card.set_path(dut_path)
    window.target_card.set_path(target_csv_path)
    window._start_analysis()  # noqa: SLF001
    _wait_for_analysis(window, application)

    output = tmp_path / "async.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "CSV (*.csv)"),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    observed_gui_thread: list[bool] = []

    def recording_export(run, output_path, **_kwargs):
        assert run is window._run  # noqa: SLF001
        observed_gui_thread.append(QThread.currentThread() is application.thread())
        return bundle_paths(output_path)

    monkeypatch.setattr(ui_module, "export_run_bundle", recording_export)

    QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton)
    _wait_for_task_completion(window, application)

    assert observed_gui_thread == [False]
    assert window.export_button.isEnabled()
    assert window.header_state.text() == "导出完成"
    assert window.header_state.property("tone") == "success"
    window.close()
    application.processEvents()


def test_ui_export_preserves_destination_changed_after_overwrite_confirmation(
    tmp_path,
    monkeypatch,
) -> None:
    """确认覆盖后的外部改写必须被 CAS 拒绝，不能覆盖后来者的新字节。"""

    application = _qt_application()
    window = ResponseLabWindow()
    run = build_demo_run()
    window._run = run  # noqa: SLF001 - install a valid current preview
    window._result = run  # noqa: SLF001
    window._result_version = window._parameter_version  # noqa: SLF001
    window.export_button.setEnabled(True)
    output = tmp_path / "concurrent.csv"
    output.write_bytes(b"approved-old")
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "CSV (*.csv)"),
    )

    def replace_after_snapshot(*_args, **_kwargs):
        output.write_bytes(b"external-new")
        return QMessageBox.StandardButton.Yes

    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", replace_after_snapshot)
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )

    window._export()  # noqa: SLF001 - exercise the real overwrite-confirmation path
    _wait_for_task_completion(window, application)

    assert output.read_bytes() == b"external-new"
    assert errors and "目标" in errors[0] and "变化" in errors[0]
    assert window.header_state.text() == "导出失败"
    window.close()
    application.processEvents()


def test_closing_during_export_cancels_worker_without_modal(tmp_path, monkeypatch) -> None:
    """关窗必须协作取消后台导出并静默收尾，不能留下输出或弹框阻塞。"""

    application = _qt_application()
    window = ResponseLabWindow()
    run = build_demo_run()
    window._run = run  # noqa: SLF001
    window._result = run  # noqa: SLF001
    window._result_version = window._parameter_version  # noqa: SLF001
    window.export_button.setEnabled(True)
    window.show()
    output = tmp_path / "cancelled.csv"
    started = threading.Event()
    modals: list[str] = []
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "CSV (*.csv)"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: modals.append(str(message)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, message: modals.append(str(message)),
    )

    def cancellable_export(*_args, cancelled=None, **_kwargs):
        started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                raise ui_module.OperationCancelledError("导出已取消")
            time.sleep(0.005)
        raise AssertionError("export worker did not receive close cancellation")

    monkeypatch.setattr(ui_module, "export_run_bundle", cancellable_export)
    window._export()  # noqa: SLF001
    deadline = time.monotonic() + 2.0
    while not started.is_set() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert started.is_set()

    window.close()
    deadline = time.monotonic() + 5.0
    while window.isVisible() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)

    assert not window.isVisible()
    assert window._worker is None  # noqa: SLF001
    assert modals == []
    assert not any(path.exists() for path in bundle_paths(output).as_tuple())


def test_export_failure_does_not_promise_rollback_succeeded(monkeypatch) -> None:
    """回滚本身失败时，界面不能声称旧文件一定未被替换。"""

    application = _qt_application()
    window = ResponseLabWindow()
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(str(message)),
    )

    window._close_when_finished = True  # noqa: SLF001
    window._export_rollback_incomplete(  # noqa: SLF001
        "导出提交失败，且回滚未完全成功：permission denied",
        window._parameter_version,  # noqa: SLF001
    )

    status = window.statusBar().currentMessage()
    assert "原有文件未被替换" not in status
    assert "不完整" in status
    assert errors and "回滚未完全成功" in errors[0]
    assert window._close_when_finished is False  # noqa: SLF001
    assert window._result_version == -1  # noqa: SLF001
    assert window.header_state.text() == "导出批次不完整"
    assert window.header_state.property("tone") == "error"
    window.close()
    application.processEvents()


def test_export_thread_routes_incomplete_rollback_to_dedicated_signal(
    tmp_path,
    monkeypatch,
) -> None:
    """回滚不完整必须走独立错误信号，不能退化成普通导出失败。"""

    application = _qt_application()
    run = build_demo_run()

    def broken_export(*_args, **_kwargs):
        raise BundleRollbackError("rollback broken")

    monkeypatch.setattr(ui_module, "export_run_bundle", broken_export)
    worker = ui_module.ExportThread(run, tmp_path / "result.csv", 9)
    routed: list[tuple[str, int]] = []
    ordinary: list[tuple[str, int]] = []
    worker.rollback_incomplete.connect(
        lambda message, version: routed.append((message, version))
    )
    worker.failed.connect(lambda message, version: ordinary.append((message, version)))

    worker.run()

    assert routed == [("rollback broken", 9)]
    assert ordinary == []
    worker.deleteLater()
    application.processEvents()


def test_export_cleanup_warning_is_visible_after_success(
    tmp_path,
    monkeypatch,
) -> None:
    """窗口版必须显示清理残留路径，不能只向不可见 stderr 发 warning。"""

    application = _qt_application()
    run = build_demo_run()
    paths = bundle_paths(tmp_path / "result.csv")

    def exported_with_cleanup_warning(*_args, **_kwargs):
        import warnings

        warnings.warn(
            "旧输出备份清理未完成，请检查：recovery-path",
            BundleCleanupWarning,
            stacklevel=2,
        )
        return paths

    monkeypatch.setattr(ui_module, "export_run_bundle", exported_with_cleanup_warning)
    worker = ui_module.ExportThread(run, paths.output, 7)
    outcomes: list[object] = []
    worker.succeeded.connect(lambda outcome, _version: outcomes.append(outcome))

    worker.run()

    assert len(outcomes) == 1
    assert outcomes[0].paths == paths
    assert "recovery-path" in outcomes[0].cleanup_warning
    visible_warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: visible_warnings.append(str(message)),
    )
    window = ResponseLabWindow()
    window._export_succeeded(outcomes[0], 7)  # noqa: SLF001
    assert window.header_state.text() == "导出完成，需检查"
    assert visible_warnings and "recovery-path" in visible_warnings[0]
    window.close()
    application.processEvents()
    worker.deleteLater()
    application.processEvents()


@pytest.mark.parametrize(
    ("failure", "expected_context"),
    [
        (OSError("primary export failure"), "primary export failure"),
        (ui_module.OperationCancelledError("导出已取消"), "导出已取消"),
    ],
)
def test_export_cleanup_warning_is_visible_after_failure_or_cancellation(
    tmp_path,
    monkeypatch,
    failure,
    expected_context,
) -> None:
    """失败/取消不能吞掉需要人工处理的清理残留路径。"""

    application = _qt_application()
    run = build_demo_run()

    def failed_with_cleanup_warning(*_args, **_kwargs):
        import warnings

        warnings.warn(
            "staging 清理未完成，请检查：recovery-path",
            BundleCleanupWarning,
            stacklevel=2,
        )
        raise failure

    monkeypatch.setattr(ui_module, "export_run_bundle", failed_with_cleanup_warning)
    worker = ui_module.ExportThread(run, tmp_path / "result.csv", 7)
    cleanup_failures: list[tuple[str, int]] = []
    ordinary_failures: list[tuple[str, int]] = []
    cancellations: list[int] = []
    worker.cleanup_incomplete.connect(
        lambda message, version: cleanup_failures.append((message, version))
    )
    worker.failed.connect(
        lambda message, version: ordinary_failures.append((message, version))
    )
    worker.cancelled.connect(cancellations.append)

    worker.run()

    assert len(cleanup_failures) == 1
    assert cleanup_failures[0][1] == 7
    assert expected_context in cleanup_failures[0][0]
    assert "recovery-path" in cleanup_failures[0][0]
    assert ordinary_failures == []
    assert cancellations == []

    visible_warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: visible_warnings.append(str(message)),
    )
    window = ResponseLabWindow()
    window._close_when_finished = True  # noqa: SLF001
    window._export_cleanup_incomplete(  # noqa: SLF001
        cleanup_failures[0][0],
        cleanup_failures[0][1],
    )
    assert window._close_when_finished is False  # noqa: SLF001
    assert window.header_state.text() == "导出未完成，需检查"
    assert visible_warnings and "recovery-path" in visible_warnings[0]
    window.close()
    application.processEvents()
    worker.deleteLater()
    application.processEvents()


def test_stale_preview_export_success_does_not_overwrite_warning_state(
    tmp_path,
    monkeypatch,
) -> None:
    """旧预览导出完成后，header 仍必须明确当前参数已让预览过期。"""

    application = _qt_application()
    window = ResponseLabWindow()
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    paths = bundle_paths(tmp_path / "stale.csv")
    window._parameter_version = 2  # noqa: SLF001

    window._export_succeeded(paths, 1)  # noqa: SLF001

    assert window.header_state.text() == "当前预览已过期"
    assert window.header_state.property("tone") == "warning"
    assert "旧预览已导出" in window.statusBar().currentMessage()
    window.close()
    application.processEvents()


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
    QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton)
    _wait_for_task_completion(window, application)
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
    QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton)
    _wait_for_task_completion(window, application)

    assert errors and "源文件" in errors[0]
    assert window.header_state.text() == "源文件已变化"
    assert not window.export_button.isEnabled()
    assert not any(path.exists() for path in bundle_paths(second_output).as_tuple())
    window.close()
    application.processEvents()
