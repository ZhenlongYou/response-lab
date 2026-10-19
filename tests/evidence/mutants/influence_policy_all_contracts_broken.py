"""定向突变：复现旧 100 ppm、固定镜像边界和未限制正向增益。"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

BoundaryMode = Literal["zero", "reflect"]
FloatArray = NDArray[np.float64]
PULSE_SAMPLE_RATE_TOLERANCE_PPM = 100.0


def validate_cross_pulse_sample_rates(
    reference_rate_hz: float,
    dut_rate_hz: float,
    *,
    subject: str,
) -> None:
    reference_rate = float(reference_rate_hz)
    dut_rate = float(dut_rate_hz)
    difference_hz = abs(dut_rate - reference_rate)
    allowed_difference_hz = reference_rate * PULSE_SAMPLE_RATE_TOLERANCE_PPM * 1.0e-6
    comparison_roundoff_hz = 8.0 * np.spacing(max(reference_rate, dut_rate))
    if difference_hz <= allowed_difference_hz + comparison_roundoff_hz:
        return
    mismatch_ppm = difference_hz / reference_rate * 1.0e6
    raise ValueError(
        f"{subject}采样率差异 {mismatch_ppm:.3f} ppm，"
        f"超过允许的 {PULSE_SAMPLE_RATE_TOLERANCE_PPM:g} ppm"
    )


def pad_finite_record(
    values: FloatArray,
    *,
    padding: int,
    boundary_mode: BoundaryMode,
) -> FloatArray:
    del boundary_mode
    return np.pad(values, ((padding, padding), (0, 0)), mode="reflect")


def limit_log_magnitude_gain(
    log_magnitude_ratio: FloatArray,
    maximum_gain_db: float | None,
) -> FloatArray:
    del maximum_gain_db
    return log_magnitude_ratio
