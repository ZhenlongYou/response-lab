"""在身份绑定快照中重放影响页真实按钮路径。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


def main() -> int:
    """执行 1000 ppm、零边界与增益限制的真实 Qt 点击回归。"""

    exit_code = int(
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_influence_workflow.py::test_exact_1000_ppm_pulses_run_from_real_influence_button",
                "tests/test_influence_workflow.py::test_zero_boundary_repair_runs_from_real_influence_button",
                "tests/test_influence_workflow.py::test_gain_limit_repair_runs_from_real_influence_button",
            ]
        )
    )
    if exit_code != 0:
        return exit_code
    print("REAL_PATH_OK: 4 influence-button cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
