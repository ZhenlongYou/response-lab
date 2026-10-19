"""同一探针验证影响频段安全合同，并为定向突变生成可判定 RED。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from response_lab.influence_policy import (  # noqa: E402
    limit_log_magnitude_gain,
    pad_finite_record,
    validate_cross_pulse_sample_rates,
)

CASE_ROOT = PROJECT_ROOT / "tests" / "evidence" / "inputs"
TEST_ID = "RL-INFLUENCE-CONTRACTS"


def _load_case(name: str) -> dict[str, object]:
    """读取身份绑定的单一输入分区。"""

    return json.loads((CASE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def _run_contract() -> None:
    """覆盖名义、边界、非法、对抗、真实和已知失败六类输入。"""

    reference_rate_hz = float(_load_case("realistic")["reference_rate_hz"])
    for partition in ("nominal", "boundary"):
        mismatch_ppm = float(_load_case(partition)["mismatch_ppm"])
        validate_cross_pulse_sample_rates(
            reference_rate_hz,
            reference_rate_hz * (1.0 + mismatch_ppm * 1.0e-6),
            subject="证据探针",
        )

    invalid_ppm = float(_load_case("invalid")["mismatch_ppm"])
    try:
        validate_cross_pulse_sample_rates(
            reference_rate_hz,
            reference_rate_hz * (1.0 + invalid_ppm * 1.0e-6),
            subject="证据探针",
        )
    except ValueError as error:
        if "2000.000 ppm" not in str(error) or "1000 ppm" not in str(error):
            raise AssertionError("超限错误没有报告实际 ppm 与 1000 ppm 门限") from error
    else:
        raise AssertionError("2000 ppm 未被拒绝")

    known_failure = _load_case("known_failure")
    values = np.asarray(known_failure["values"], dtype=np.float64)[:, np.newaxis]
    padding = int(known_failure["padding"])
    actual_padded = pad_finite_record(values, padding=padding, boundary_mode="zero")
    expected_padded = np.pad(values, ((padding, padding), (0, 0)), mode="constant")
    np.testing.assert_array_equal(actual_padded, expected_padded)

    realistic = _load_case("realistic")
    requested_gain_db = float(realistic["requested_gain_db"])
    maximum_gain_db = float(realistic["maximum_gain_db"])
    requested_log_gain = np.asarray(
        [requested_gain_db * np.log(10.0) / 20.0],
        dtype=np.float64,
    )
    limited = limit_log_magnitude_gain(requested_log_gain, maximum_gain_db)
    actual_voltage_gain = float(np.exp(limited[0]))
    expected_voltage_gain = 10.0 ** (maximum_gain_db / 20.0)
    if not np.isclose(actual_voltage_gain, expected_voltage_gain, rtol=0.0, atol=1.0e-12):
        raise AssertionError("40 dB 请求未按 20 dB 限制为 10 倍电压增益")

    adversarial_gain_db = float(_load_case("adversarial")["maximum_gain_db"])
    try:
        limit_log_magnitude_gain(requested_log_gain, adversarial_gain_db)
    except ValueError:
        pass
    else:
        raise AssertionError("负最大增益未被拒绝")


def main() -> int:
    """把语义失败转换为固定签名，避免把导入失败误当目标 RED。"""

    try:
        _run_contract()
    except (AssertionError, ValueError) as error:
        print(f"{TEST_ID} failure-signature: {error}")
        return 7
    print(f"{TEST_ID} GREEN: 500/1000/2000 ppm, zero boundary, 20 dB cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
