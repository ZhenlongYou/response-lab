"""真实调用大 BIN 验证脚本，防止入口层再次掩盖算法已完成后的失败。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from examples import validate_large_bin_streaming as validation_module
from response_lab.keysight_bin import write_keysight_bin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPT = PROJECT_ROOT / "examples" / "validate_large_bin_streaming.py"


def _legacy_windows_console_environment() -> dict[str, str]:
    """Exercise the CLI through a common non-UTF-8 Windows console codec."""

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "cp1252"
    return environment


def test_validation_modules_import_without_posix_resource_module() -> None:
    """Windows 测试收集不能依赖仅 POSIX 提供的 ``resource``。"""

    probe = """
import builtins

original_import = builtins.__import__

def import_without_resource(name, *args, **kwargs):
    if name == "resource":
        raise ModuleNotFoundError("No module named 'resource'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_resource
import examples.validate_large_bin_streaming
import examples.validate_vpp_keysight_pipeline
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr


def test_large_bin_validation_cli_reports_streaming_pass() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_SCRIPT),
            "--samples",
            "1000000",
            "--fft-samples",
            "1048576",
            "--strategy",
            "streaming",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        env=_legacy_windows_console_environment(),
        text=True,
        timeout=60.0,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["measurement"]["status"] == "PASS"
    assert report["measurement"]["application_metadata"]["strategy"] == "streaming"
    assert report["measurement"]["output_dtype"] == "float32"
    assert report["measurement"]["maximum_allowed_error"] == (
        validation_module.closed_form_absolute_tolerance_v(np.float32)
    )
    assert report["measurement"]["acceptance_checks"][
        "observed_compensation_peak_is_informative"
    ]
    assert report["measurement"]["acceptance_checks"][
        "observed_compensation_peak_is_enveloped"
    ]
    assert all(report["acceptance_checks"].values())
    assert report["invocation"]["worker_stderr"] == ""


def test_large_bin_validation_cli_preserves_worker_failure_stderr() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_SCRIPT),
            "--samples",
            "10000",
            "--fft-samples",
            "4",
            "--strategy",
            "streaming",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        env=_legacy_windows_console_environment(),
        text=True,
        timeout=60.0,
    )

    assert completed.returncode != 0
    report = json.loads(completed.stdout)
    assert report["status"] == "FAIL"
    assert report["invocation"]["worker_returncode"] != 0
    assert report["invocation"]["worker_stderr"]
    assert "分块 FFT 点数必须是至少 512 的整数" in report["invocation"][
        "worker_stderr"
    ]
    assert "UnicodeEncodeError" not in completed.stderr


def test_worker_marks_equal_rss_high_water_as_inconclusive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bin_path = tmp_path / "small.bin"
    write_keysight_bin(
        bin_path,
        np.linspace(-1.0, 1.0, 10_000, dtype=np.float32),
        2.0e9,
    )
    rss_samples = iter((100_000_000, 100_000_000, 100_000_000))
    monkeypatch.setattr(
        validation_module,
        "_peak_rss_bytes",
        lambda: next(rss_samples),
    )

    measurement = validation_module._worker(bin_path, 4096, "streaming")

    assert measurement["status"] == "INCONCLUSIVE"
    assert not measurement["acceptance_checks"][
        "observed_compensation_peak_is_informative"
    ]
    assert not measurement["acceptance_checks"][
        "observed_compensation_peak_is_enveloped"
    ]
