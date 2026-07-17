"""频率响应页使用面向用户的简洁幅相纵轴标题。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from response_lab.app import _qt_application, build_demo_run
from response_lab.ui import ResponseLabWindow


def test_frequency_response_axes_use_simple_amplitude_and_phase_labels() -> None:
    application = _qt_application()
    window = ResponseLabWindow()

    window.present_run(build_demo_run())

    assert window.response_plots[0].getAxis("left").labelText == "幅度"
    assert window.response_plots[1].getAxis("left").labelText == "相位"
    window.close()
    application.processEvents()
