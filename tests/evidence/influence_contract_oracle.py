"""不导入 ResponseLab 的独立闭式谕示。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = PROJECT_ROOT / "tests" / "evidence" / "inputs"


def _load_case(name: str) -> dict[str, object]:
    return json.loads((CASE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def main() -> int:
    """用相对误差定义、显式零填充和 dB 闭式认证合同。"""

    reference_rate_hz = float(_load_case("realistic")["reference_rate_hz"])
    accepted_ppm = [
        float(_load_case(partition)["mismatch_ppm"])
        for partition in ("nominal", "boundary")
    ]
    rejected_ppm = float(_load_case("invalid")["mismatch_ppm"])
    assert all(
        abs(reference_rate_hz * (1.0 + ppm * 1.0e-6) - reference_rate_hz)
        <= reference_rate_hz * 1000.0e-6 + 8.0 * np.spacing(reference_rate_hz)
        for ppm in accepted_ppm
    )
    assert rejected_ppm > 1000.0

    known_failure = _load_case("known_failure")
    values = np.asarray(known_failure["values"], dtype=np.float64)
    padding = int(known_failure["padding"])
    expected = np.concatenate((np.zeros(padding), values, np.zeros(padding)))
    assert expected[0] == 0.0 and expected[padding] == values[0]

    realistic = _load_case("realistic")
    maximum_gain_db = float(realistic["maximum_gain_db"])
    assert 10.0 ** (maximum_gain_db / 20.0) == 10.0
    print("ORACLE_OK: relative ppm, explicit finite zeros, and voltage-dB identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
