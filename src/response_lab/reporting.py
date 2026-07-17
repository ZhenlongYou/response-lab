"""ResponseLab reproducible evidence and rollback-capable bundle export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .io import save_bin_float32, save_csv_timeseries
from .models import CompensationRun, TimeSeries


@dataclass(frozen=True)
class BundlePaths:
    """The three final paths derived from one user-selected output path."""

    output: Path
    response_csv: Path
    manifest: Path

    def as_tuple(self) -> tuple[Path, Path, Path]:
        return self.output, self.response_csv, self.manifest


class SourceVerificationError(RuntimeError):
    """Raised when a source no longer matches the bytes loaded for analysis."""


def sha256_file(path: str | Path) -> str:
    """Stream a file into SHA-256 without loading a large BIN into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Hash an array together with its explicit dtype and shape."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def bundle_paths(output_path: str | Path) -> BundlePaths:
    """Derive final paths so callers can preview and check all collisions."""

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".csv", ".bin"}:
        raise ValueError("主输出文件扩展名必须是 .csv 或 .bin")
    return BundlePaths(
        output=output,
        response_csv=output.with_name(f"{output.stem}_response.csv"),
        manifest=output.with_name(f"{output.name}.response-lab.json"),
    )


def _atomic_savetxt(
    destination: Path,
    table: np.ndarray,
    *,
    header: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            np.savetxt(stream, table, delimiter=",", header=header, comments="")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return destination


def export_response_csv(path: str | Path, run: CompensationRun) -> Path:
    """Atomically export the complete plotted response table with units in headers."""

    analysis = run.analysis
    correction_magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(analysis.correction_ideal), np.finfo(np.float64).tiny)
    )
    table = np.column_stack(
        [
            analysis.frequency_hz,
            analysis.reference_magnitude_db,
            analysis.dut_magnitude_db,
            analysis.magnitude_difference_db,
            np.degrees(analysis.phase_difference_rad),
            np.degrees(analysis.phase_trend_rad),
            np.degrees(analysis.phase_after_optional_detrend_rad),
            correction_magnitude_db,
            np.degrees(np.angle(analysis.correction_ideal)),
            analysis.reliable_mask.astype(np.int8),
        ]
    )
    header = (
        "frequency_hz,reference_magnitude_db,dut_magnitude_db,magnitude_difference_db,"
        "phase_difference_deg,fitted_linear_phase_trend_deg,"
        "phase_after_optional_detrend_deg,"
        "correction_magnitude_db,correction_phase_deg,reliable"
    )
    return _atomic_savetxt(Path(path), table, header=header)


def _snapshot_evidence(series: TimeSeries) -> tuple[int, str] | None:
    metadata = series.source_metadata
    size = metadata.get("source_size_bytes")
    digest = metadata.get("source_sha256")
    if size is None and digest is None:
        return None
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or int(size) < 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError("输入来源快照字段无效，请重新加载源文件")
    return int(size), digest


def _source_manifest(series: TimeSeries) -> dict[str, Any]:
    source_path = series.source_path
    result: dict[str, Any] = {
        "format": series.source_format,
        "sample_rate_hz": series.sample_rate_hz,
        "samples": series.samples,
        "channels": series.channels,
        "time_unit": series.time_unit,
        "time_scale_to_s": series.time_scale_to_s,
        "value_columns": list(series.value_columns),
        "source_metadata": dict(series.source_metadata),
    }
    if source_path is not None:
        result["path"] = str(Path(source_path).resolve())
        evidence = _snapshot_evidence(series)
        if evidence is not None:
            result.update({"size_bytes": evidence[0], "sha256": evidence[1]})
    return result


def verify_source_files_unchanged(run: CompensationRun) -> None:
    """Reject export if any file source differs from its load-time snapshot."""

    labelled_series = (
        ("参考拟合脉冲", run.reference_pulse),
        ("待补偿拟合脉冲", run.dut_pulse),
        ("待补偿信号", run.input_signal),
    )
    verified: dict[Path, tuple[int, str]] = {}
    for label, series in labelled_series:
        if series.source_path is None:
            continue
        source = Path(series.source_path).resolve()
        expected = _snapshot_evidence(series)
        if expected is None:
            raise SourceVerificationError(f"{label}缺少加载时来源快照，请重新加载")
        if source in verified:
            if verified[source] != expected:
                raise SourceVerificationError(f"{label}的来源快照与同路径输入不一致")
            continue
        try:
            size = source.stat().st_size
        except FileNotFoundError as error:
            raise SourceVerificationError(f"{label}源文件已删除或移动：{source}") from error
        except OSError as error:
            raise SourceVerificationError(f"无法验证{label}源文件：{source}") from error
        if size != expected[0]:
            raise SourceVerificationError(f"{label}源文件大小已变化，请重新加载后分析")
        try:
            digest = sha256_file(source)
        except OSError as error:
            raise SourceVerificationError(f"无法验证{label}源文件：{source}") from error
        if digest != expected[1]:
            raise SourceVerificationError(f"{label}源文件内容已变化，请重新加载后分析")
        verified[source] = expected


def build_manifest(
    run: CompensationRun,
    output_path: str | Path,
    *,
    output_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize effective inputs, settings, frequency application, and output evidence.

    ``output_path`` is always the final user-facing path. During bundle export,
    ``output_evidence_path`` points at the fully written staged output so the
    recorded size and digest describe the bytes that will be committed.
    """

    output = np.asarray(run.output_values, dtype=np.float64)
    resolved_output = Path(output_path).resolve()
    evidence_path = (
        Path(output_evidence_path).resolve()
        if output_evidence_path is not None
        else resolved_output
    )
    output_file = (
        {
            "size_bytes": evidence_path.stat().st_size,
            "sha256": sha256_file(evidence_path),
        }
        if evidence_path.is_file()
        else {}
    )
    extended_samples = 3 * run.input_signal.samples - 2
    settings_manifest = asdict(run.analysis.settings)
    return {
        "schema": "response-lab-manifest/v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "software": {"name": "ResponseLab", "version": __version__},
        "inputs": {
            "reference_pulse": _source_manifest(run.reference_pulse),
            "dut_pulse": _source_manifest(run.dut_pulse),
            "target_signal": _source_manifest(run.input_signal),
        },
        "settings": settings_manifest,
        "analysis": {
            "phase_detrend_slope_rad_per_hz": (
                run.analysis.phase_detrend_slope_rad_per_hz
            ),
            "estimated_relative_delay_s": run.analysis.estimated_relative_delay_s,
            "relative_delay_sign_convention": "positive_means_dut_later_than_reference",
            "reliable_points": int(np.count_nonzero(run.analysis.reliable_mask)),
            "frequency_points": int(run.analysis.frequency_hz.size),
        },
        "application": {
            "method": "reflect_extend_rfft_multiply_irfft_crop",
            "sample_rate_hz": run.input_signal.sample_rate_hz,
            "original_samples": run.input_signal.samples,
            "extended_samples": extended_samples,
            "frequency_bins": extended_samples // 2 + 1,
            "frequency_axis_sha256": sha256_array(
                np.fft.rfftfreq(
                    extended_samples,
                    d=1.0 / run.input_signal.sample_rate_hz,
                )
            ),
        },
        "output": {
            "path": str(resolved_output),
            "shape": list(output.shape),
            "values_sha256": sha256_array(output),
            "minimum": float(np.min(output)),
            "maximum": float(np.max(output)),
            "rms": float(np.sqrt(np.mean(output**2))),
            **output_file,
        },
        "warnings": list(run.warnings),
    }


def write_manifest_atomic(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically replace one manifest after fully writing and syncing its temp file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return destination


def _same_existing_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _check_bundle_collisions(
    run: CompensationRun,
    paths: BundlePaths,
    *,
    allow_overwrite: bool,
) -> None:
    finals = paths.as_tuple()
    if len(set(finals)) != len(finals):
        raise ValueError("主输出、响应表和 manifest 的最终路径必须互不相同")
    sources = tuple(
        Path(series.source_path).resolve()
        for series in (run.reference_pulse, run.dut_pulse, run.input_signal)
        if series.source_path is not None
    )
    for final in finals:
        if any(_same_existing_file(final, source) for source in sources):
            raise ValueError(f"导出路径不能覆盖任何输入源文件：{final}")
        if final.exists() and not final.is_file():
            raise IsADirectoryError(f"导出路径不是普通文件：{final}")
        if not allow_overwrite and os.path.lexists(final):
            raise FileExistsError(f"导出文件已存在：{final}")


def _write_primary_output(path: Path, run: CompensationRun) -> None:
    if path.suffix.lower() == ".bin":
        save_bin_float32(path, run.output_values)
    else:
        save_csv_timeseries(
            path,
            run.input_signal.time_s,
            run.output_values,
            time_scale_to_s=run.input_signal.time_scale_to_s,
        )


def _commit_replace(source: Path, destination: Path) -> None:
    """Small indirection used only by the rollback-capable commit phase."""

    os.replace(source, destination)


def _commit_bundle_with_rollback(staged: BundlePaths, final: BundlePaths) -> None:
    staged_files = staged.as_tuple()
    final_files = final.as_tuple()
    staging_directory = staged.output.parent
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for index, destination in enumerate(final_files):
            if os.path.lexists(destination):
                backup = staging_directory / f".backup-{index}-{destination.name}"
                _commit_replace(destination, backup)
                backups[destination] = backup
        for staged_file, destination in zip(staged_files, final_files, strict=True):
            _commit_replace(staged_file, destination)
            committed.append(destination)
    except Exception as commit_error:
        rollback_errors: list[Exception] = []
        for destination in reversed(committed):
            try:
                destination.unlink(missing_ok=True)
            except OSError as error:
                rollback_errors.append(error)
        for destination, backup in backups.items():
            try:
                if os.path.lexists(backup):
                    _commit_replace(backup, destination)
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise RuntimeError(f"导出提交失败，且回滚未完全成功：{details}") from commit_error
        raise


def export_run_bundle(
    run: CompensationRun,
    output_path: str | Path,
    *,
    allow_overwrite: bool = True,
) -> BundlePaths:
    """Export all three artifacts using a rollback-capable transaction.

    This is intentionally described as a rollback transaction, not as a
    power-loss-atomic multi-file operation. All files are generated in a staging
    directory on the destination filesystem; existing files are restored if a
    generation or commit step raises an exception.
    """

    if not isinstance(allow_overwrite, bool):
        raise ValueError("allow_overwrite 必须是布尔值")
    paths = bundle_paths(output_path)
    verify_source_files_unchanged(run)
    _check_bundle_collisions(run, paths, allow_overwrite=allow_overwrite)
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{paths.output.name}.response-lab-staging-",
            dir=paths.output.parent,
        )
    )
    staged = BundlePaths(
        output=staging_directory / paths.output.name,
        response_csv=staging_directory / paths.response_csv.name,
        manifest=staging_directory / paths.manifest.name,
    )
    try:
        _write_primary_output(staged.output, run)
        export_response_csv(staged.response_csv, run)
        manifest = build_manifest(
            run,
            paths.output,
            output_evidence_path=staged.output,
        )
        write_manifest_atomic(staged.manifest, manifest)
        verify_source_files_unchanged(run)
        _commit_bundle_with_rollback(staged, paths)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)
    return paths
