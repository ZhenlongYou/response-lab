"""Generate a real Keysight BIN and measure full-band streaming compensation.

The parent process creates the fixture.  A fresh child process performs load and
compensation so ``ru_maxrss`` has a meaningful pre-load baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from response_lab.dsp import (
    _compensation_memory_estimate_from_shape,
    _streaming_memory_estimate_from_shape,
    run_compensation,
)
from response_lab.io import load_bin_timeseries
from response_lab.keysight_bin import write_keysight_bin
from response_lab.memory_budget import current_memory_budget
from response_lab.models import CompensationSettings, TimeSeries

CLOSED_FORM_EXPECTED_PEAK_V = 2.0
CLOSED_FORM_FFT_ROUNDTRIP_EPSILON_FACTOR = 64.0


def closed_form_absolute_tolerance_v(dtype: np.dtype | type[np.floating]) -> float:
    """Return the 2*x fixture's absolute FFT round-trip acceptance bound in volts."""

    return float(
        CLOSED_FORM_FFT_ROUNDTRIP_EPSILON_FACTOR
        * np.finfo(np.dtype(dtype)).eps
        * CLOSED_FORM_EXPECTED_PEAK_V
    )


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pulse(scale: float, sample_rate_hz: float) -> TimeSeries:
    values = np.zeros(64, dtype=np.float64)
    values[16] = scale
    return TimeSeries.from_uniform_samples(
        values=values,
        sample_rate_hz=sample_rate_hz,
        time_origin_s=0.0,
        time_increment_s=1.0 / sample_rate_hz,
    )


def _worker(
    bin_path: Path,
    fft_samples: int,
    application_strategy: str,
) -> dict[str, object]:
    requested_sample_rate_hz = 2.0e9
    baseline_rss = _peak_rss_bytes()
    budget_before = current_memory_budget()
    started = time.perf_counter()
    target = load_bin_timeseries(bin_path)
    load_seconds = time.perf_counter() - started
    after_load_rss = _peak_rss_bytes()
    budget_after_load = current_memory_budget()

    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=0.5 * requested_sample_rate_hz,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        edge_transition_fraction=0.0,
        maximum_gain_db=None,
        analysis_points=1025,
        application_strategy=application_strategy,
        streaming_fft_samples=fft_samples,
    )
    estimate_arguments = {
        "target_samples": target.samples,
        "target_channels": target.channels,
        "sample_rate_hz": target.sample_rate_hz,
        "reference_samples": 64,
        "dut_samples": 64,
        "settings": settings,
    }
    exact_estimate = _compensation_memory_estimate_from_shape(**estimate_arguments)
    streaming_estimate = _streaming_memory_estimate_from_shape(**estimate_arguments)

    started = time.perf_counter()
    run = run_compensation(
        _pulse(1.0, requested_sample_rate_hz),
        _pulse(0.5, requested_sample_rate_hz),
        target,
        settings,
    )
    compensation_seconds = time.perf_counter() - started
    after_run_rss = _peak_rss_bytes()

    maximum_error = 0.0
    validation_chunk = 1_048_576
    for start in range(0, target.samples, validation_chunk):
        stop = min(target.samples, start + validation_chunk)
        actual = np.asarray(run.output_values[start:stop, 0], dtype=np.float64)
        expected = 2.0 * np.asarray(target.values[start:stop, 0], dtype=np.float64)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(actual - expected))),
        )

    maximum_allowed_error = closed_form_absolute_tolerance_v(
        run.output_values.dtype
    )
    budget_bytes = budget_after_load.budget_bytes
    if application_strategy == "auto":
        expected_strategy = (
            "exact"
            if exact_estimate.estimated_peak_bytes <= budget_bytes
            else "streaming"
        )
    else:
        expected_strategy = application_strategy
    selected_estimated_peak_bytes = (
        exact_estimate.estimated_peak_bytes
        if expected_strategy == "exact"
        else streaming_estimate.estimated_peak_bytes
    )
    observed_compensation_peak_delta_bytes = max(
        0,
        after_run_rss - after_load_rss,
    )
    rss_observation_is_informative = after_run_rss > after_load_rss
    checks = {
        "selected_expected_strategy": bool(
            run.application_metadata["strategy"] == expected_strategy
        ),
        "streaming_output_is_float32": bool(
            expected_strategy != "streaming"
            or run.output_values.dtype == np.dtype(np.float32)
        ),
        "exact_output_is_float64": bool(
            expected_strategy != "exact"
            or run.output_values.dtype == np.dtype(np.float64)
        ),
        "closed_form_error_within_contract": bool(
            maximum_error <= maximum_allowed_error
        ),
        "selected_estimate_within_budget": bool(
            selected_estimated_peak_bytes <= budget_bytes
        ),
        "observed_compensation_peak_is_informative": bool(
            rss_observation_is_informative
        ),
        "observed_compensation_peak_is_enveloped": bool(
            rss_observation_is_informative
            and observed_compensation_peak_delta_bytes
            <= selected_estimated_peak_bytes
        ),
    }
    if not rss_observation_is_informative:
        status = "INCONCLUSIVE"
    else:
        status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "acceptance_checks": checks,
        "maximum_allowed_error": float(maximum_allowed_error),
        "samples": target.samples,
        "input_dtype": str(target.values.dtype),
        "output_dtype": str(run.output_values.dtype),
        "recovered_sample_rate_hz": target.sample_rate_hz,
        "application_method": run.application_method,
        "application_metadata": dict(run.application_metadata),
        "exact_memory_estimate": asdict(exact_estimate),
        "streaming_memory_estimate": asdict(streaming_estimate),
        "load_wall_seconds": load_seconds,
        "compensation_wall_seconds": compensation_seconds,
        "rss_baseline_bytes": baseline_rss,
        "rss_after_load_bytes": after_load_rss,
        "rss_after_run_bytes": after_run_rss,
        "load_peak_delta_bytes": max(0, after_load_rss - baseline_rss),
        "post_load_compensation_peak_delta_bytes": max(
            0,
            after_run_rss - after_load_rss,
        ),
        "total_peak_delta_bytes": max(0, after_run_rss - baseline_rss),
        "budget_before": asdict(budget_before),
        "budget_after_load": asdict(budget_after_load),
        "maximum_absolute_error_against_2x": maximum_error,
        "warnings": list(run.warnings),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证大 Keysight BIN 的全频分块补偿、误差和峰值 RSS",
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--fft-samples", type=int, default=1_048_576)
    parser.add_argument(
        "--strategy",
        choices=("auto", "exact", "streaming"),
        default="streaming",
        help="默认强制 streaming；30M 自动路由验证请显式选择 auto",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--worker-bin", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.samples < 8:
        raise ValueError("--samples 必须至少为 8")
    if arguments.worker_bin is not None:
        measurement = _worker(
            arguments.worker_bin,
            arguments.fft_samples,
            arguments.strategy,
        )
        print(
            json.dumps(
                measurement,
                ensure_ascii=False,
            )
        )
        if measurement["status"] == "PASS":
            return 0
        return 2 if measurement["status"] == "INCONCLUSIVE" else 1

    with tempfile.TemporaryDirectory(prefix="responselab-large-bin-") as folder:
        bin_path = Path(folder) / "large.bin"
        fixture_values = np.linspace(
            -1.0,
            1.0,
            arguments.samples,
            dtype=np.float32,
        )
        started = time.perf_counter()
        write_keysight_bin(
            bin_path,
            fixture_values,
            2.0e9,
            label="large-bin",
        )
        write_seconds = time.perf_counter() - started
        del fixture_values

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-bin",
            str(bin_path),
            "--fft-samples",
            str(arguments.fft_samples),
            "--strategy",
            arguments.strategy,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        try:
            measurement = json.loads(completed.stdout)
        except json.JSONDecodeError:
            measurement = {
                "status": "FAIL",
                "error": "worker 未产生可解析 JSON",
            }
        project_root = Path(__file__).resolve().parents[1]

        def git_output(*git_arguments: str) -> str:
            result = subprocess.run(
                ["git", *git_arguments],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() if result.returncode == 0 else "unavailable"

        source_diff_result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
        )
        source_diff = source_diff_result.stdout
        git_head = git_output("rev-parse", "--verify", "HEAD")
        git_status = git_output("status", "--short")
        untracked_runtime_listing = git_output(
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "examples/validate_large_bin_streaming.py",
        )
        if untracked_runtime_listing == "unavailable":
            untracked_runtime_hashes: dict[str, str] = {}
        else:
            untracked_runtime_hashes = {
                relative_path: _sha256_file(project_root / relative_path)
                for relative_path in untracked_runtime_listing.splitlines()
                if relative_path
            }
        provenance_checks = {
            "valid_git_head": bool(
                len(git_head) == 40
                and all(character in "0123456789abcdef" for character in git_head)
            ),
            "tracked_diff_captured": bool(source_diff_result.returncode == 0),
            "untracked_runtime_files_captured": bool(
                untracked_runtime_listing != "unavailable"
            ),
        }
        worker_status = measurement.get("status")
        if worker_status == "INCONCLUSIVE":
            report_status = "INCONCLUSIVE"
        elif (
            completed.returncode == 0
            and worker_status == "PASS"
            and all(provenance_checks.values())
        ):
            report_status = "PASS"
        else:
            report_status = "FAIL"
        report = {
            "schema": "response-lab-large-bin-streaming-validation/v1",
            "status": report_status,
            "acceptance_checks": {
                "worker_passed": bool(
                    completed.returncode == 0 and worker_status == "PASS"
                ),
                **provenance_checks,
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "source": {
                "git_head": git_head,
                "git_status_porcelain": git_status,
                "git_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "untracked_runtime_file_sha256": untracked_runtime_hashes,
            },
            "invocation": {
                "samples": arguments.samples,
                "fft_samples": arguments.fft_samples,
                "strategy": arguments.strategy,
                "worker_command": command,
                "worker_returncode": completed.returncode,
                "worker_stdout": completed.stdout,
                "worker_stderr": completed.stderr,
            },
            "fixture": {
                "container": "Keysight Infiniium AG10 BIN",
                "samples": arguments.samples,
                "payload_dtype": "float32",
                "file_size_bytes": bin_path.stat().st_size,
                "sha256": _sha256_file(bin_path),
                "write_wall_seconds": write_seconds,
                "closed_form_output": "2*x",
                "reference_pulse": {
                    "samples": 64,
                    "nonzero_index": 16,
                    "amplitude": 1.0,
                },
                "dut_pulse": {
                    "samples": 64,
                    "nonzero_index": 16,
                    "amplitude": 0.5,
                },
            },
            "measurement": measurement,
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output_json is not None:
        destination = arguments.output_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "PASS":
        print(
            "验证失败；完整 worker stdout/stderr 已写入上方 JSON。",
            file=sys.stderr,
        )
        return 2 if report["status"] == "INCONCLUSIVE" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
