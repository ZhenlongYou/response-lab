"""真实调用大 BIN 验证脚本，防止入口层再次掩盖算法已完成后的失败。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPT = PROJECT_ROOT / "examples" / "validate_large_bin_streaming.py"


def test_large_bin_validation_cli_reports_streaming_pass() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_SCRIPT),
            "--samples",
            "10000",
            "--fft-samples",
            "4096",
            "--strategy",
            "streaming",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60.0,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["measurement"]["status"] == "PASS"
    assert report["measurement"]["application_metadata"]["strategy"] == "streaming"
    assert report["measurement"]["output_dtype"] == "float32"
    assert report["invocation"]["worker_stderr"] == ""
