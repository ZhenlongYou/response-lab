"""在身份绑定快照中执行影响合同的完整受影响测试分区。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


def main() -> int:
    """覆盖算法、Vpp、控制器、按钮集成和 UI 失效合同。"""

    exit_code = int(
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_attribution.py",
                "tests/test_vpp_analysis.py",
                "tests/test_influence_controller.py",
                "tests/test_influence_workflow.py",
                "tests/test_ui_workflow.py",
            ]
        )
    )
    if exit_code != 0:
        return exit_code
    print("SUITE_OK: influence contract partitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
