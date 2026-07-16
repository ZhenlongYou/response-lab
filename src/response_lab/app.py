"""ResponseLab 的命令行路由、无窗口自检与 Qt 启动。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from .dsp import run_compensation
from .models import CompensationRun, CompensationSettings, TimeSeries


def build_demo_run() -> CompensationRun:
    """生成不依赖文件的确定性演示运行，供自检、截图和首次界面预览使用。"""

    sample_rate_hz = 2.0e9
    pulse_samples = 2048
    index = np.arange(pulse_samples, dtype=np.float64)
    reference_values = np.exp(-0.5 * ((index - 420.0) / 3.0) ** 2)
    dut_values = np.zeros_like(reference_values)
    dut_values[5:] = 0.72 * reference_values[:-5]
    pulse_time_s = index / sample_rate_hz
    reference = TimeSeries(pulse_time_s, reference_values[:, None], sample_rate_hz)
    dut = TimeSeries(pulse_time_s, dut_values[:, None], sample_rate_hz)

    signal_samples = 16384
    signal_time_s = np.arange(signal_samples, dtype=np.float64) / sample_rate_hz
    input_values = (
        0.55 * np.sin(2.0 * np.pi * 120.0e6 * signal_time_s)
        + 0.22 * np.sin(2.0 * np.pi * 240.0e6 * signal_time_s + 0.4)
    )
    input_signal = TimeSeries(signal_time_s, input_values[:, None], sample_rate_hz)
    settings = CompensationSettings(
        mode="both",
        band_low_hz=10.0e6,
        band_high_hz=300.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        remove_relative_delay=True,
        analysis_points=8193,
    )
    return run_compensation(reference, dut, input_signal, settings)


def run_self_test() -> int:
    """不导入 Qt 的算法入口自检，适合 PyCharm、CI 与无显示终端。"""

    run = build_demo_run()
    if run.output_values.shape != run.input_signal.values.shape:
        raise RuntimeError("自检失败：补偿输出形状改变")
    if not np.all(np.isfinite(run.output_values)):
        raise RuntimeError("自检失败：补偿输出包含 NaN/Inf")
    if not run.analysis.settings.remove_relative_delay:
        raise RuntimeError("自检失败：默认相对时延未移除")
    input_rms = float(np.sqrt(np.mean(run.input_signal.values**2)))
    output_rms = float(np.sqrt(np.mean(run.output_values**2)))
    print(
        "ResponseLab self-test: PASS\n"
        f"  pulse sample rate: {run.reference_pulse.sample_rate_hz:.6g} Hz\n"
        f"  estimated DUT delay: {run.analysis.estimated_dut_delay_s:.6g} s (not applied)\n"
        "  application: FFT multiply IFFT\n"
        f"  input/output RMS: {input_rms:.6g} / {output_rms:.6g}"
    )
    return 0


def _qt_application():
    """延迟导入 GUI 依赖，使 ``--self-test`` 在无 Qt 环境也能给出算法结果。"""

    try:
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError(
            "缺少 PySide6。请在项目目录执行：python3 -m venv .venv && "
            ".venv/bin/python -m pip install -e '.[dev]'"
        ) from exc
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("ResponseLab")
    application.setOrganizationName("RinysProject")
    system_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    if sys.platform == "darwin" and system_font.family() == "Sans Serif":
        # Qt 的 macOS offscreen 后端有时返回一个不存在的通用别名，导致启动告警。
        system_font = QFont("Helvetica Neue", system_font.pointSize())
    application.setFont(system_font)
    return application


def run_gui_smoke_test(render_path: Path | None = None) -> int:
    """构造真实主窗口并可选渲染截图，不进入永久事件循环。"""

    from PySide6.QtTest import QTest

    from .ui import ResponseLabWindow

    application = _qt_application()
    window = ResponseLabWindow()
    window.resize(1440, 900)
    window.present_run(build_demo_run(), source_label="内置演示数据")
    window.show()
    application.processEvents()
    QTest.qWait(300)
    application.processEvents()
    if window.visual_tabs.count() != 5:
        raise RuntimeError("GUI 自检失败：可视化页面数量不是 5")
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(render_path)):
            raise RuntimeError(f"GUI 截图保存失败：{render_path}")
        print(f"ResponseLab UI render: {render_path}")
    window.close()
    application.processEvents()
    print("ResponseLab GUI smoke-test: PASS")
    return 0


def run_gui() -> int:
    """启动正常交互式桌面窗口。"""

    from .ui import ResponseLabWindow

    application = _qt_application()
    window = ResponseLabWindow()
    window.show()
    return int(application.exec())


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResponseLab 频响分析与补偿")
    parser.add_argument("--self-test", action="store_true", help="运行无窗口算法自检")
    parser.add_argument("--gui-smoke-test", action="store_true", help="构造并关闭真实 Qt 主窗口")
    parser.add_argument("--render-ui", type=Path, help="把内置演示界面渲染为 PNG")
    options = parser.parse_args(arguments)
    if options.self_test:
        return run_self_test()
    if options.gui_smoke_test or options.render_ui is not None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return run_gui_smoke_test(options.render_ui)
    return run_gui()
