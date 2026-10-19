"""影响频段与跨脉冲分析共用的可审计安全合同。"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

BoundaryMode = Literal["zero", "reflect"]
FloatArray = NDArray[np.float64]

# 用户确认眼图与 Vpp 可接受两份拟合脉冲最多 0.1% 的采样率差异。
PULSE_SAMPLE_RATE_TOLERANCE_PPM = 1000.0


def validate_cross_pulse_sample_rates(
    reference_rate_hz: float,
    dut_rate_hz: float,
    *,
    subject: str,
) -> None:
    """允许用户确认的 1000 ppm 工程差异，超限拒绝且不静默重采样。"""

    reference_rate = float(reference_rate_hz)
    dut_rate = float(dut_rate_hz)
    difference_hz = abs(dut_rate - reference_rate)
    allowed_difference_hz = (
        reference_rate * PULSE_SAMPLE_RATE_TOLERANCE_PPM * 1.0e-6
    )
    # 恰好位于门限时只补偿输入浮点量级的八个 ULP，不吞掉真实工程 ppm 差异。
    comparison_roundoff_hz = 8.0 * np.spacing(max(reference_rate, dut_rate))
    if difference_hz <= allowed_difference_hz + comparison_roundoff_hz:
        return
    mismatch_ppm = difference_hz / reference_rate * 1.0e6
    raise ValueError(
        f"{subject}采样率差异 {mismatch_ppm:.3f} ppm，"
        f"超过允许的 {PULSE_SAMPLE_RATE_TOLERANCE_PPM:g} ppm；"
        "工具不会静默重采样"
    )


def pad_finite_record(
    values: FloatArray,
    *,
    padding: int,
    boundary_mode: BoundaryMode,
) -> FloatArray:
    """按主补偿同一合同延拓有限记录，默认零状态且可显式选择镜像。"""

    if boundary_mode == "zero":
        mode = "constant"
    elif boundary_mode == "reflect":
        mode = "reflect"
    else:
        raise ValueError("边界模式必须是 zero 或 reflect")
    return np.pad(values, ((padding, padding), (0, 0)), mode=mode)


def limit_log_magnitude_gain(
    log_magnitude_ratio: FloatArray,
    maximum_gain_db: float | None,
) -> FloatArray:
    """在候选权重应用前按电压 dB 上限裁剪正向对数增益。"""

    if maximum_gain_db is None:
        return log_magnitude_ratio
    if (
        isinstance(maximum_gain_db, (bool, np.bool_))
        or not np.isfinite(maximum_gain_db)
        or maximum_gain_db < 0.0
    ):
        raise ValueError("最大补偿增益必须是非负有限 dB，或设为 None")
    maximum_log_gain = maximum_gain_db * np.log(10.0) / 20.0
    return np.minimum(log_magnitude_ratio, maximum_log_gain)
