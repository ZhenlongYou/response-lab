"""频率响应页使用面向用户的简洁幅相纵轴标题。"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from response_lab.app import _qt_application, build_demo_run
from response_lab.dsp import analyze_responses
from response_lab.models import TimeSeries
from response_lab.ui import (
    ResponseLabWindow,
    _output_spectrum_preview_slice,
    _output_waveform_preview_slice,
)


def test_thirty_million_output_preview_is_bounded_before_plot_or_fft() -> None:
    waveform_slice = _output_waveform_preview_slice(30_000_000)
    waveform_points = len(range(*waveform_slice.indices(30_000_000)))
    spectrum_slice = _output_spectrum_preview_slice(30_000_000)

    assert waveform_points <= 200_000
    assert spectrum_slice.stop - spectrum_slice.start == 1_048_576
    assert spectrum_slice.start > 0
    assert spectrum_slice.stop < 30_000_000


def test_output_focus_uses_full_record_bounds_for_large_record() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    observed: dict[str, object] = {}

    class TimeAxisProbe:
        def __getitem__(self, key):
            observed["key"] = key
            return np.array([0.0, 14.9999995e-3])

    fake_run = SimpleNamespace(
        input_signal=SimpleNamespace(
            samples=30_000_000,
            time_s=TimeAxisProbe(),
        )
    )

    window._focus_output_preview(fake_run)

    np.testing.assert_array_equal(observed["key"], [0, -1])
    window.close()
    application.processEvents()


def test_frequency_response_axes_use_simple_amplitude_and_phase_labels() -> None:
    application = _qt_application()
    window = ResponseLabWindow()
    run = build_demo_run()

    window.present_run(run)

    assert window.response_plots[0].getAxis("left").labelText == "幅度"
    assert window.response_plots[1].getAxis("left").labelText == "相位"
    reference_curve, dut_curve = window.response_plots[0].listDataItems()
    reference_magnitude_db = reference_curve.yData
    dut_magnitude_db = dut_curve.yData
    np.testing.assert_allclose(
        reference_magnitude_db,
        np.where(
            run.analysis.reliable_mask,
            run.analysis.reference_magnitude_db,
            np.nan,
        ),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        dut_magnitude_db,
        np.where(
            run.analysis.reliable_mask,
            run.analysis.dut_magnitude_db,
            np.nan,
        ),
        equal_nan=True,
    )
    window.close()
    application.processEvents()


def test_frequency_response_magnitude_preserves_input_scale() -> None:
    run = build_demo_run()
    scale = 0.1
    reference = run.reference_pulse
    dut = run.dut_pulse
    scaled = analyze_responses(
        TimeSeries(
            reference.time_s,
            scale * reference.values,
            reference.sample_rate_hz,
        ),
        TimeSeries(
            dut.time_s,
            scale * dut.values,
            dut.sample_rate_hz,
        ),
        run.analysis.settings,
    )
    reliable = (
        run.analysis.reliable_mask
        & scaled.reliable_mask
        & (
            run.analysis.reference_magnitude_db
            >= np.max(run.analysis.reference_magnitude_db) - 120.0
        )
        & (
            run.analysis.dut_magnitude_db
            >= np.max(run.analysis.dut_magnitude_db) - 120.0
        )
    )
    expected_shift_db = 20.0 * np.log10(scale)

    np.testing.assert_allclose(
        scaled.reference_magnitude_db[reliable],
        run.analysis.reference_magnitude_db[reliable] + expected_shift_db,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        scaled.dut_magnitude_db[reliable],
        run.analysis.dut_magnitude_db[reliable] + expected_shift_db,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        scaled.magnitude_difference_db[reliable],
        run.analysis.magnitude_difference_db[reliable],
        atol=1.0e-10,
    )
