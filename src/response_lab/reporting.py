"""ResponseLab reproducible evidence and rollback-capable bundle export."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
import unicodedata
import warnings
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .cancellation import CancellationCheck, raise_if_cancelled
from .io import save_bin_timeseries, save_csv_timeseries
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


class BundleRollbackError(RuntimeError):
    """Raised when a bundle commit fails and old files cannot all be restored."""


class BundleBusyError(RuntimeError):
    """Raised when another ResponseLab process is exporting an overlapping bundle."""


class DestinationChangedError(RuntimeError):
    """Raised when output paths change after overwrite approval or staging begins."""


class BundleCleanupWarning(RuntimeWarning):
    """Warn that a foreign or unverifiable path was preserved during cleanup."""


@dataclass(frozen=True)
class DestinationFingerprint:
    """One output path's metadata identity plus an optional worker-side digest."""

    exists: bool
    device: int = 0
    inode: int = 0
    size_bytes: int = 0
    modified_time_ns: int = 0
    changed_time_ns: int = 0
    sha256: str = ""


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int


def _identity_from_stat(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(device=int(metadata.st_dev), inode=int(metadata.st_ino))


def _directory_identity(path: Path) -> _FileIdentity:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"导出父目录不是实际目录：{path}")
    return _identity_from_stat(metadata)


def _verify_directory_identity(path: Path, expected: _FileIdentity) -> None:
    try:
        current = _directory_identity(path)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise DestinationChangedError(f"导出父目录在批准后被移动或替换：{path}") from error
    if current != expected:
        raise DestinationChangedError(f"导出父目录在批准后被移动或替换：{path}")


def sha256_file(
    path: str | Path,
    *,
    cancelled: CancellationCheck | None = None,
) -> str:
    """Stream a file into SHA-256 without loading a large BIN into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            raise_if_cancelled(cancelled, message="导出已取消")
            digest.update(block)
    raise_if_cancelled(cancelled, message="导出已取消")
    return digest.hexdigest()


def sha256_array(
    values: np.ndarray,
    *,
    cancelled: CancellationCheck | None = None,
) -> str:
    """Hash dtype、shape 和连续字节；大数组按块送入摘要，不创建 ``tobytes`` 副本。"""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    byte_view = memoryview(array).cast("B")
    chunk_bytes = 4 * 1024 * 1024
    for start in range(0, byte_view.nbytes, chunk_bytes):
        raise_if_cancelled(cancelled, message="导出已取消")
        digest.update(byte_view[start : start + chunk_bytes])
    raise_if_cancelled(cancelled, message="导出已取消")
    return digest.hexdigest()


def _bounded_output_statistics(
    values: np.ndarray,
    *,
    cancelled: CancellationCheck | None = None,
) -> tuple[float, float, float]:
    """以有界 float64 临时块计算 min/max/RMS，避免整条 float32 输出翻倍。"""

    flat = np.asarray(values).reshape(-1)
    minimum = float("inf")
    maximum = float("-inf")
    square_sum = np.longdouble(0.0)
    chunk_elements = 1_048_576
    for start in range(0, flat.size, chunk_elements):
        raise_if_cancelled(cancelled, message="导出已取消")
        chunk = np.asarray(flat[start : start + chunk_elements], dtype=np.float64)
        minimum = min(minimum, float(np.min(chunk)))
        maximum = max(maximum, float(np.max(chunk)))
        square_sum += np.sum(np.square(chunk), dtype=np.longdouble)
    rms = float(np.sqrt(square_sum / np.longdouble(flat.size)))
    return minimum, maximum, rms


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


def _destination_fingerprint(
    path: Path,
    *,
    include_digest: bool = False,
    cancelled: CancellationCheck | None = None,
) -> DestinationFingerprint:
    """Capture one final path without following symlinks or accepting directories."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return DestinationFingerprint(exists=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise IsADirectoryError(f"导出路径不是普通文件：{path}")
    digest = sha256_file(path, cancelled=cancelled) if include_digest else ""
    try:
        verified_metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise DestinationChangedError(f"读取导出目标快照时文件发生变化：{path}") from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if not stat.S_ISREG(verified_metadata.st_mode) or any(
        getattr(metadata, field) != getattr(verified_metadata, field) for field in stable_fields
    ):
        raise DestinationChangedError(f"读取导出目标快照时文件发生变化：{path}")
    return DestinationFingerprint(
        exists=True,
        device=int(verified_metadata.st_dev),
        inode=int(verified_metadata.st_ino),
        size_bytes=int(verified_metadata.st_size),
        modified_time_ns=int(verified_metadata.st_mtime_ns),
        changed_time_ns=int(verified_metadata.st_ctime_ns),
        sha256=digest,
    )


def snapshot_bundle_destinations(
    paths: BundlePaths,
    *,
    include_digest: bool = False,
    cancelled: CancellationCheck | None = None,
) -> tuple[DestinationFingerprint, DestinationFingerprint, DestinationFingerprint]:
    """Freeze final identities; SHA-256 is opt-in for background commit baselines."""

    return tuple(  # type: ignore[return-value]
        _destination_fingerprint(
            path,
            include_digest=include_digest,
            cancelled=cancelled,
        )
        for path in paths.as_tuple()
    )


def _verify_destination_fingerprints(
    paths: BundlePaths,
    expected: tuple[
        DestinationFingerprint,
        DestinationFingerprint,
        DestinationFingerprint,
    ],
    *,
    allow_overwrite: bool,
    cancelled: CancellationCheck | None = None,
) -> None:
    """Compare current destinations with an approval or digest commit baseline."""

    include_digest = any(
        fingerprint.exists and bool(fingerprint.sha256) for fingerprint in expected
    )
    current = snapshot_bundle_destinations(
        paths,
        include_digest=include_digest,
        cancelled=cancelled,
    )
    if not allow_overwrite:
        for path, fingerprint in zip(paths.as_tuple(), current, strict=True):
            if fingerprint.exists:
                raise FileExistsError(f"导出文件已存在：{path}")
    if current != expected:
        raise DestinationChangedError(
            "导出目标在确认后发生变化；为避免覆盖其他程序的新文件，本次未提交"
        )


def _controlled_lock_directory() -> Path:
    lock_directory = Path(tempfile.gettempdir()) / "response-lab-export-locks"
    with suppress(FileExistsError):
        lock_directory.mkdir(mode=0o700)
    metadata = os.lstat(lock_directory)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("ResponseLab 导出锁目录不是受控目录")
    if os.name != "nt" and (
        metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise PermissionError("ResponseLab 导出锁目录权限不安全")
    return lock_directory


def _lock_path_for_material(material: bytes) -> Path:
    digest = hashlib.sha256(material).hexdigest()
    lock_directory = _controlled_lock_directory()
    return lock_directory / f"{digest}.lock"


def _lock_file_path(destination: Path) -> Path:
    """Return a conservative path lock key across case/Unicode-insensitive volumes."""

    resolved = os.path.realpath(os.fspath(destination))
    normalized = unicodedata.normalize("NFC", os.path.normpath(resolved)).casefold()
    return _lock_path_for_material(b"path\0" + os.fsencode(normalized))


def _lock_file_paths(destination: Path) -> set[Path]:
    """Lock both the canonical spelling and any currently existing inode alias."""

    lock_paths = {_lock_file_path(destination)}
    try:
        metadata = os.lstat(destination)
    except FileNotFoundError:
        return lock_paths
    if stat.S_ISREG(metadata.st_mode):
        identity = f"inode\0{int(metadata.st_dev)}\0{int(metadata.st_ino)}".encode("ascii")
        lock_paths.add(_lock_path_for_material(identity))
    return lock_paths


def _open_lock_path(lock_path: Path) -> object:
    """Open a regular lock file without following an attacker-controlled symlink."""

    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("ResponseLab 导出锁不是普通文件")
        return os.fdopen(descriptor, "r+b")
    except Exception:
        os.close(descriptor)
        raise


def _acquire_lock(handle: object) -> None:
    """Acquire one non-blocking advisory lock on POSIX or Windows."""

    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_lock(handle: object) -> None:
    """Release one advisory lock before closing its file handle."""

    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_error_is_contention(error: OSError) -> bool:
    contention_errnos = {errno.EACCES, errno.EAGAIN}
    if hasattr(errno, "EWOULDBLOCK"):
        contention_errnos.add(errno.EWOULDBLOCK)
    if error.errno in contention_errnos:
        return True
    return os.name == "nt" and getattr(error, "winerror", None) in {
        32,  # ERROR_SHARING_VIOLATION
        33,  # ERROR_LOCK_VIOLATION
        36,  # ERROR_SHARING_BUFFER_EXCEEDED
    }


@contextmanager
def _bundle_destination_locks(paths: BundlePaths):
    """Serialize every ResponseLab bundle that shares any one final path."""

    handles: list[object] = []
    try:
        lock_paths = {
            lock_path
            for destination in paths.as_tuple()
            for lock_path in _lock_file_paths(destination)
        }
        for lock_path in sorted(lock_paths, key=lambda path: str(path)):
            handle = _open_lock_path(lock_path)
            try:
                _acquire_lock(handle)
            except OSError as error:
                handle.close()
                if _lock_error_is_contention(error):
                    raise BundleBusyError(
                        "另一个 ResponseLab 任务正在导出相同文件；请等待其完成后重试"
                    ) from error
                raise
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            with suppress(OSError):
                _release_lock(handle)
            handle.close()


def export_response_csv(
    path: str | Path,
    run: CompensationRun,
    *,
    cancelled: CancellationCheck | None = None,
) -> Path:
    """Atomically stream the plotted response table with bounded memory."""

    raise_if_cancelled(cancelled, message="导出已取消")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "frequency_hz,reference_magnitude_db,dut_magnitude_db,magnitude_difference_db,"
        "phase_difference_deg,fitted_linear_phase_trend_deg,"
        "phase_after_optional_detrend_deg,"
        "correction_magnitude_db,correction_phase_deg,reliable"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    analysis = run.analysis
    row_count = int(analysis.frequency_hz.size)
    # NumPy 的文本格式化在单块内不可抢占；较小行块把关窗/取消延迟控制在短时间内，
    # 同时把十列 float64 临时矩阵限制在约 320 KiB。
    chunk_rows = 4_096
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(header + "\n")
            raise_if_cancelled(cancelled, message="导出已取消")
            for start in range(0, row_count, chunk_rows):
                stop = min(start + chunk_rows, row_count)
                correction = analysis.correction_ideal[start:stop]
                table = np.column_stack(
                    [
                        analysis.frequency_hz[start:stop],
                        analysis.reference_magnitude_db[start:stop],
                        analysis.dut_magnitude_db[start:stop],
                        analysis.magnitude_difference_db[start:stop],
                        np.degrees(analysis.phase_difference_rad[start:stop]),
                        np.degrees(analysis.phase_trend_rad[start:stop]),
                        np.degrees(analysis.phase_after_optional_detrend_rad[start:stop]),
                        20.0
                        * np.log10(
                            np.maximum(
                                np.abs(correction),
                                np.finfo(np.float64).tiny,
                            )
                        ),
                        np.degrees(np.angle(correction)),
                        analysis.reliable_mask[start:stop].astype(np.int8),
                    ]
                )
                np.savetxt(stream, table, delimiter=",")
                raise_if_cancelled(cancelled, message="导出已取消")
            stream.flush()
            os.fsync(stream.fileno())
        raise_if_cancelled(cancelled, message="导出已取消")
        os.replace(temporary_name, destination)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return destination


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


def verify_source_files_unchanged(
    run: CompensationRun,
    *,
    cancelled: CancellationCheck | None = None,
) -> None:
    """Reject export if any file source differs from its load-time snapshot."""

    labelled_series = (
        ("参考拟合脉冲", run.reference_pulse),
        ("待补偿拟合脉冲", run.dut_pulse),
        ("待补偿信号", run.input_signal),
    )
    verified: dict[Path, tuple[int, str]] = {}
    for label, series in labelled_series:
        raise_if_cancelled(cancelled, message="导出已取消")
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
            digest = sha256_file(source, cancelled=cancelled)
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
    cancelled: CancellationCheck | None = None,
) -> dict[str, Any]:
    """Serialize effective inputs, settings, frequency application, and output evidence.

    ``output_path`` is always the final user-facing path. During bundle export,
    ``output_evidence_path`` points at the fully written staged output so the
    recorded size and digest describe the bytes that will be committed.
    """

    output = np.asarray(run.output_values)
    resolved_output = Path(output_path).resolve()
    evidence_path = (
        Path(output_evidence_path).resolve()
        if output_evidence_path is not None
        else resolved_output
    )
    output_file = (
        {
            "size_bytes": evidence_path.stat().st_size,
            "sha256": sha256_file(evidence_path, cancelled=cancelled),
        }
        if evidence_path.is_file()
        else {}
    )
    settings_manifest = asdict(run.analysis.settings)
    application_metadata = dict(run.application_metadata)
    application = {
        "method": run.application_method,
        "sample_rate_hz": run.input_signal.sample_rate_hz,
        **application_metadata,
        "boundary_mode": run.analysis.settings.boundary_mode,
    }
    if not application_metadata:
        extended_samples = 3 * run.input_signal.samples - 2
        application.update(
            {
                "original_samples": run.input_signal.samples,
                "extended_samples": extended_samples,
                "frequency_bins": extended_samples // 2 + 1,
                "output_dtype": str(output.dtype),
            }
        )
    frequency_axis_samples = int(
        application.get("extended_samples", application.get("fft_samples", 0))
    )
    if frequency_axis_samples <= 0:
        raise ValueError("应用元数据缺少有效的 FFT 点数")
    application["frequency_axis_sha256"] = sha256_array(
        np.fft.rfftfreq(
            frequency_axis_samples,
            d=1.0 / run.input_signal.sample_rate_hz,
        ),
        cancelled=cancelled,
    )
    output_minimum, output_maximum, output_rms = _bounded_output_statistics(
        output,
        cancelled=cancelled,
    )
    return {
        "schema": "response-lab-manifest/v4",
        "created_utc": datetime.now(UTC).isoformat(),
        "software": {"name": "ResponseLab", "version": __version__},
        "inputs": {
            "reference_pulse": _source_manifest(run.reference_pulse),
            "dut_pulse": _source_manifest(run.dut_pulse),
            "target_signal": _source_manifest(run.input_signal),
        },
        "settings": settings_manifest,
        "analysis": {
            "response_magnitude_db_definition": (
                "20*log10(abs(dt_s*rfft(h)))_interpolated_on_common_frequency_grid"
            ),
            "response_magnitude_scale": "raw_input_scale",
            "phase_detrend_slope_rad_per_hz": (run.analysis.phase_detrend_slope_rad_per_hz),
            "estimated_relative_delay_s": run.analysis.estimated_relative_delay_s,
            "relative_delay_sign_convention": "positive_means_dut_later_than_reference",
            "reliable_points": int(np.count_nonzero(run.analysis.reliable_mask)),
            "frequency_points": int(run.analysis.frequency_hz.size),
        },
        "application": application,
        "output": {
            "path": str(resolved_output),
            "shape": list(output.shape),
            "values_sha256": sha256_array(output, cancelled=cancelled),
            "minimum": output_minimum,
            "maximum": output_maximum,
            "rms": output_rms,
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
        fingerprint = _destination_fingerprint(final)
        if not allow_overwrite and fingerprint.exists:
            raise FileExistsError(f"导出文件已存在：{final}")


def _write_primary_output(
    path: Path,
    run: CompensationRun,
    *,
    cancelled: CancellationCheck | None = None,
) -> None:
    if path.suffix.lower() == ".bin":
        save_bin_timeseries(
            path,
            run.input_signal.time_s,
            run.output_values,
            cancelled=cancelled,
        )
    else:
        save_csv_timeseries(
            path,
            run.input_signal.time_s,
            run.output_values,
            time_scale_to_s=run.input_signal.time_scale_to_s,
            cancelled=cancelled,
        )


def _raise_native_rename_error(destination: Path) -> None:
    error_number = ctypes.get_errno() or errno.EIO
    raise OSError(error_number, os.strerror(error_number), destination)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing a destination on supported kernels.

    There is no portable POSIX emulation for this contract. Unsupported kernels or
    filesystems therefore fail closed instead of using a check-then-replace window.
    """

    if os.name == "nt":
        # Python maps this to a non-replacing Windows rename operation.
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            renamex_np = library.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "当前 macOS 不支持原子无覆盖重命名",
                destination,
            ) from error
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        if renamex_np(source_bytes, destination_bytes, 0x00000004) != 0:
            _raise_native_rename_error(destination)
        return
    if sys.platform.startswith("linux"):
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "当前 Linux C 库不支持原子无覆盖重命名",
                destination,
            ) from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        if renameat2(-100, source_bytes, -100, destination_bytes, 0x00000001) != 0:
            _raise_native_rename_error(destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "当前平台不支持原子无覆盖重命名，已安全停止提交",
        destination,
    )


def _rename_no_replace_at(
    source: Path,
    destination: Path,
    *,
    source_parent_descriptor: int | None,
    destination_parent_descriptor: int | None,
) -> None:
    """Descriptor-relative no-replace rename where POSIX exposes the primitive."""

    if source_parent_descriptor is None or destination_parent_descriptor is None:
        _rename_no_replace(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source.name)
    destination_bytes = os.fsencode(destination.name)
    if sys.platform == "darwin":
        try:
            renameatx_np = library.renameatx_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "当前 macOS 不支持目录绑定的原子无覆盖重命名",
                destination,
            ) from error
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        if (
            renameatx_np(
                source_parent_descriptor,
                source_bytes,
                destination_parent_descriptor,
                destination_bytes,
                0x00000004,
            )
            != 0
        ):
            _raise_native_rename_error(destination)
        return
    if sys.platform.startswith("linux"):
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "当前 Linux C 库不支持目录绑定的原子无覆盖重命名",
                destination,
            ) from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        if (
            renameat2(
                source_parent_descriptor,
                source_bytes,
                destination_parent_descriptor,
                destination_bytes,
                0x00000001,
            )
            != 0
        ):
            _raise_native_rename_error(destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "当前平台不支持目录绑定的原子无覆盖重命名，已安全停止提交",
        destination,
    )


@contextmanager
def _pinned_directory(path: Path, expected: _FileIdentity):
    """Keep a directory bound for descriptor-relative POSIX commit operations.

    Windows does not expose Python ``dir_fd`` operations.  It nevertheless keeps
    the existing identity checks and path-based no-replace primitive, which fail
    closed on a changed path.  POSIX publication never resolves either parent by
    its mutable pathname after this descriptor has been opened.
    """

    descriptor: int | None = None
    windows_handle: int | None = None
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if _identity_from_stat(metadata) != expected or not stat.S_ISDIR(metadata.st_mode):
            os.close(descriptor)
            raise DestinationChangedError(f"目录在绑定时发生变化：{path}")
    else:
        _verify_directory_identity(path, expected)
        from ctypes import wintypes

        create_file = ctypes.windll.kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0x80000000,  # GENERIC_READ
            0x00000001 | 0x00000002,  # SHARE_READ | SHARE_WRITE; never SHARE_DELETE
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "无法锁定导出目录，已安全停止", path)
        windows_handle = int(handle)
        try:
            _verify_directory_identity(path, expected)
        except Exception:
            ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(windows_handle))
            raise
    try:
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if windows_handle is not None:
            ctypes.windll.kernel32.CloseHandle(windows_handle)


def _destination_fingerprint_at(
    parent_descriptor: int,
    path: Path,
    *,
    include_digest: bool,
) -> DestinationFingerprint:
    """Snapshot a regular child through a pinned POSIX parent descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IsADirectoryError(f"导出路径不是普通文件：{path}")
        digest = ""
        if include_digest:
            hasher = hashlib.sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while block := os.read(descriptor, 1024 * 1024):
                hasher.update(block)
            digest = hasher.hexdigest()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise DestinationChangedError(f"读取目录绑定快照时文件发生变化：{path}")
    return DestinationFingerprint(
        exists=True,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size_bytes=int(after.st_size),
        modified_time_ns=int(after.st_mtime_ns),
        changed_time_ns=int(after.st_ctime_ns),
        sha256=digest,
    )


def _hardlink_is_unavailable(error: OSError, *, windows: bool | None = None) -> bool:
    """Classify only errors for which an atomic rename fallback is appropriate."""

    if windows is None:
        windows = os.name == "nt"
    unsupported_errnos = {
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
        errno.EPERM,
        errno.EXDEV,
        errno.ENOSYS,
    }
    if error.errno in unsupported_errnos:
        return True
    if not windows:
        return False
    windows_unsupported = {
        1,  # ERROR_INVALID_FUNCTION
        50,  # ERROR_NOT_SUPPORTED
        87,  # ERROR_INVALID_PARAMETER
        1314,  # ERROR_PRIVILEGE_NOT_HELD
    }
    return getattr(error, "winerror", None) in windows_unsupported or error.errno in {
        errno.EACCES,
        errno.EINVAL,
    }


def _commit_replace(source: Path, destination: Path) -> None:
    """Compatibility indirection for an atomic, non-replacing transaction move."""

    _rename_no_replace(source, destination)


def _commit_staged_file(
    source: Path,
    destination: Path,
    expected_source: DestinationFingerprint | None = None,
    *,
    source_parent_descriptor: int | None = None,
    destination_parent_descriptor: int | None = None,
    expected_source_parent: _FileIdentity | None = None,
    expected_destination_parent: _FileIdentity | None = None,
) -> None:
    """Atomically publish one staged regular file without clobbering a late arrival."""

    include_digest = expected_source is not None and bool(expected_source.sha256)
    current_source = (
        _destination_fingerprint_at(
            source_parent_descriptor,
            source,
            include_digest=include_digest,
        )
        if source_parent_descriptor is not None
        else _destination_fingerprint(source, include_digest=include_digest)
    )
    if expected_source is not None and not _same_destination_after_rename(
        expected_source,
        current_source,
    ):
        raise DestinationChangedError("staged 文件或其父目录在提交前发生变化；本次未发布该路径")
    try:
        if source_parent_descriptor is not None and destination_parent_descriptor is not None:
            os.link(
                source.name,
                destination.name,
                src_dir_fd=source_parent_descriptor,
                dst_dir_fd=destination_parent_descriptor,
                follow_symlinks=False,
            )
        else:
            os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        raise
    except OSError as error:
        if not _hardlink_is_unavailable(error):
            raise
        _rename_no_replace_at(
            source,
            destination,
            source_parent_descriptor=source_parent_descriptor,
            destination_parent_descriptor=destination_parent_descriptor,
        )
    # A hard link is already the complete publish point.  Never unlink ``source``
    # by its mutable pathname here: another process could replace that name in
    # the link→unlink window.  The bound staging cleanup owns its later removal.
    try:
        if expected_source_parent is not None:
            _verify_directory_identity(source.parent, expected_source_parent)
        if expected_destination_parent is not None:
            _verify_directory_identity(destination.parent, expected_destination_parent)
    except DestinationChangedError:
        # Publication happened in the pinned original parent, not a replacement
        # pathname.  Remove only the inode just published through that same bound
        # directory before reporting the takeover.
        if destination_parent_descriptor is not None and expected_source is not None:
            published = _destination_fingerprint_at(
                destination_parent_descriptor,
                destination,
                include_digest=bool(expected_source.sha256),
            )
            if _same_destination_after_rename(expected_source, published):
                os.unlink(destination.name, dir_fd=destination_parent_descriptor)
        raise


def _reserve_backup_path(destination: Path) -> Path:
    """Reserve a unique sibling name, then leave it absent for atomic replacement."""

    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.response-lab-backup-",
        dir=destination.parent,
    )
    os.close(descriptor)
    os.unlink(name)
    return Path(name)


def _preserve_path_at_recovery_name(path: Path, *, purpose: str) -> Path:
    """Atomically move the current path to a unique sibling recovery name."""

    recovery = _reserve_backup_path(path.with_name(f"{path.name}.response-lab-{purpose}"))
    _commit_replace(path, recovery)
    return recovery


def _restore_quarantined_path(quarantine: Path, original: Path) -> Path:
    """Restore a quarantined foreign path, or leave it at a named recovery path."""

    try:
        _commit_replace(quarantine, original)
    except OSError:
        # Another owner may already have claimed ``original``.  The quarantined
        # bytes remain available at this explicit path for manual recovery.
        return quarantine
    return original


def _remove_owned_regular_file(
    path: Path,
    expected: DestinationFingerprint,
    *,
    mismatch_message: str,
) -> None:
    """Remove only bytes atomically detached from ``path`` and proven to be ours.

    Checking an inode and then unlinking the original name has an unavoidable
    cross-platform check/use window. Move the name to an unpredictable sibling
    first, inspect what was actually moved, and restore/preserve it if ownership
    does not match. This protects ordinary concurrent application instances; it
    is not a sandbox against another process running as the same OS account and
    deliberately stealing unpredictable quarantine names between the last check
    and unlink.
    """

    if not os.path.lexists(path):
        return
    try:
        quarantine = _preserve_path_at_recovery_name(path, purpose="quarantine")
    except FileNotFoundError:
        return
    try:
        moved = _destination_fingerprint(quarantine, include_digest=bool(expected.sha256))
    except Exception as error:
        recovery = _restore_quarantined_path(quarantine, path)
        raise DestinationChangedError(f"{mismatch_message}；已保留于：{recovery}") from error
    if not _same_destination_after_rename(expected, moved):
        recovery = _restore_quarantined_path(quarantine, path)
        raise DestinationChangedError(f"{mismatch_message}；已保留于：{recovery}")
    # Re-resolve once more after potentially slow hashing.  If the quarantine
    # name changed, preserving both names is safer than unlinking an unknown file.
    current = _destination_fingerprint(
        quarantine,
        include_digest=bool(expected.sha256),
    )
    if not _same_destination_after_rename(expected, current):
        raise DestinationChangedError(f"{mismatch_message}；quarantine 路径已换主")
    try:
        os.unlink(quarantine)
    except OSError as error:
        warnings.warn(
            f"本次导出拥有的临时文件清理失败，已保留：{quarantine}（{error}）",
            BundleCleanupWarning,
            stacklevel=2,
        )


def _restore_owned_backup(
    backup: Path,
    destination: Path,
    expected: DestinationFingerprint,
) -> None:
    """Move a backup back without a pre-check/use race and verify moved bytes."""

    _commit_replace(backup, destination)
    try:
        moved = _destination_fingerprint(destination, include_digest=True)
    except Exception as error:
        try:
            recovery = _preserve_path_at_recovery_name(
                destination,
                purpose="rollback-recovery",
            )
        except OSError:
            recovery = destination
        raise DestinationChangedError(
            f"回滚后的文件无法核验；未删除该路径，请人工恢复：{recovery}"
        ) from error
    if _same_destination_after_rename(expected, moved):
        return
    try:
        recovery = _preserve_path_at_recovery_name(
            destination,
            purpose="rollback-recovery",
        )
    except OSError:
        recovery = destination
    raise DestinationChangedError(
        f"回滚备份在恢复瞬间已由其他程序替换；疑似外部文件已保留于：{recovery}"
    )


def _same_destination_after_rename(
    expected: DestinationFingerprint,
    moved: DestinationFingerprint,
) -> bool:
    """Compare stable inode/content metadata; rename itself may update ctime."""

    return (
        expected.exists
        and moved.exists
        and expected.device == moved.device
        and expected.inode == moved.inode
        and expected.size_bytes == moved.size_bytes
        and expected.modified_time_ns == moved.modified_time_ns
        and expected.sha256 == moved.sha256
    )


def _fsync_directory(directory: Path) -> bool:
    """Best-effort directory fsync; unsupported filesystems must still export safely."""

    if os.name == "nt":
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return False
    return True


def _directory_has_identity(path: Path, expected: _FileIdentity) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and int(metadata.st_dev) == expected.device
        and int(metadata.st_ino) == expected.inode
    )


def _verify_staging_directory_identity(path: Path, expected: _FileIdentity) -> None:
    try:
        _verify_directory_identity(path, expected)
    except DestinationChangedError as error:
        raise DestinationChangedError(
            f"staging 目录在生成或提交期间被移动或替换：{path}"
        ) from error


def _cleanup_staging_directory(
    staging_directory: Path,
    expected_identity: _FileIdentity,
) -> None:
    """Conservatively clean an atomically detached staging directory.

    随机 quarantine 名和重复身份校验保护普通并发实例；同一系统账户若在最后一次
    核验后故意抢占该随机名，属于操作系统权限边界之外的对抗场景。
    """

    if not os.path.lexists(staging_directory):
        warnings.warn(
            "staging 路径在清理前已消失或被移动；"
            f"原路径可能仍有残留，请人工检查：{staging_directory}",
            BundleCleanupWarning,
            stacklevel=2,
        )
        return
    try:
        quarantine = _preserve_path_at_recovery_name(
            staging_directory,
            purpose="staging-cleanup",
        )
    except FileNotFoundError:
        warnings.warn(
            "staging 路径在清理前已消失或被移动；"
            f"原路径可能仍有残留，请人工检查：{staging_directory}",
            BundleCleanupWarning,
            stacklevel=2,
        )
        return
    except OSError as error:
        warnings.warn(
            f"staging 目录无法安全隔离，未执行递归删除：{staging_directory}（{error}）",
            BundleCleanupWarning,
            stacklevel=2,
        )
        return
    if not _directory_has_identity(quarantine, expected_identity):
        recovery = _restore_quarantined_path(quarantine, staging_directory)
        warnings.warn(
            f"staging 路径身份已变化，未递归删除疑似外部目录；请人工检查：{recovery}",
            BundleCleanupWarning,
            stacklevel=2,
        )
        return
    if not _directory_has_identity(quarantine, expected_identity):
        warnings.warn(
            f"staging quarantine 在校验后身份已变化，未执行任何删除；请人工检查：{quarantine}",
            BundleCleanupWarning,
            stacklevel=2,
        )
        return
    try:
        children = tuple(quarantine.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in children):
            warnings.warn(
                f"staging 目录包含非普通文件，已保留而不递归删除：{quarantine}",
                BundleCleanupWarning,
                stacklevel=2,
            )
            return
        for child in children:
            fingerprint = _destination_fingerprint(child, include_digest=True)
            _remove_owned_regular_file(
                child,
                fingerprint,
                mismatch_message="staging 临时文件在清理时身份已变化",
            )
        if not _directory_has_identity(quarantine, expected_identity):
            warnings.warn(
                "staging quarantine 在叶文件清理后身份已变化，未删除目录；"
                f"请人工检查：{quarantine}",
                BundleCleanupWarning,
                stacklevel=2,
            )
            return
        quarantine.rmdir()
    except OSError as error:
        warnings.warn(
            f"staging 目录清理失败，请人工检查：{quarantine}（{error}）",
            BundleCleanupWarning,
            stacklevel=2,
        )


def _commit_bundle_with_rollback(
    staged: BundlePaths,
    final: BundlePaths,
    expected: tuple[
        DestinationFingerprint,
        DestinationFingerprint,
        DestinationFingerprint,
    ],
    expected_parent_identity: _FileIdentity,
    *,
    staged_parent_descriptor: int | None = None,
    final_parent_descriptor: int | None = None,
    expected_staged_parent_identity: _FileIdentity | None = None,
) -> None:
    staged_files = staged.as_tuple()
    final_files = final.as_tuple()
    _verify_directory_identity(final.output.parent, expected_parent_identity)
    backups: dict[Path, Path] = {}
    for destination, fingerprint in zip(final_files, expected, strict=True):
        if fingerprint.exists:
            _verify_directory_identity(final.output.parent, expected_parent_identity)
            backups[destination] = _reserve_backup_path(destination)
    staged_fingerprints = tuple(
        _destination_fingerprint_at(
            staged_parent_descriptor,
            path,
            include_digest=True,
        )
        if staged_parent_descriptor is not None
        else _destination_fingerprint(path, include_digest=True)
        for path in staged_files
    )
    backup_fingerprints: dict[Path, DestinationFingerprint] = {}
    committed: list[tuple[Path, DestinationFingerprint]] = []
    try:
        for destination, fingerprint in zip(final_files, expected, strict=True):
            if not fingerprint.exists:
                continue
            _verify_directory_identity(final.output.parent, expected_parent_identity)
            backup = backups[destination]
            if _destination_fingerprint(destination, include_digest=True) != fingerprint:
                raise DestinationChangedError("导出目标在最终提交瞬间发生变化；本次未覆盖该文件")
            _commit_replace(destination, backup)
            # Record ownership immediately after the atomic move.  Any exception
            # while hashing/verifying the backup must still enter rollback.
            backup_fingerprints[destination] = fingerprint
            moved_fingerprint = _destination_fingerprint(backup, include_digest=True)
            if not _same_destination_after_rename(fingerprint, moved_fingerprint):
                try:
                    _commit_replace(backup, destination)
                except OSError as restore_error:
                    raise BundleRollbackError(
                        "导出目标在提交瞬间发生变化，且无法自动恢复；"
                        f"请从人工恢复路径取回：{backup}"
                    ) from restore_error
                raise DestinationChangedError("导出目标在最终提交瞬间发生变化；本次未覆盖该文件")
        for staged_file, destination, staged_fingerprint in zip(
            staged_files,
            final_files,
            staged_fingerprints,
            strict=True,
        ):
            _verify_directory_identity(final.output.parent, expected_parent_identity)
            _commit_staged_file(
                staged_file,
                destination,
                staged_fingerprint,
                source_parent_descriptor=staged_parent_descriptor,
                destination_parent_descriptor=final_parent_descriptor,
                expected_source_parent=expected_staged_parent_identity,
                expected_destination_parent=expected_parent_identity,
            )
            # Append before post-commit verification: once the atomic publish
            # returned, every subsequent failure must include this path in rollback.
            committed.append((destination, staged_fingerprint))
            committed_fingerprint = (
                _destination_fingerprint_at(
                    final_parent_descriptor,
                    destination,
                    include_digest=True,
                )
                if final_parent_descriptor is not None
                else _destination_fingerprint(
                    destination,
                    include_digest=True,
                )
            )
            if not _same_destination_after_rename(
                staged_fingerprint,
                committed_fingerprint,
            ):
                raise DestinationChangedError("新输出在提交瞬间发生变化；将进入安全回滚")
        _fsync_directory(final.output.parent)
    except Exception as commit_error:
        rollback_errors: list[Exception] = []
        for destination, committed_fingerprint in reversed(committed):
            try:
                _remove_owned_regular_file(
                    destination,
                    committed_fingerprint,
                    mismatch_message=("回滚时目标已由其他程序替换，未删除疑似外部文件"),
                )
            except (OSError, DestinationChangedError) as error:
                rollback_errors.append(error)
        # Iterate every reserved backup, not only those whose post-move hash
        # completed.  A move can succeed immediately before verification raises.
        for destination, backup in backups.items():
            fingerprint = next(
                expected_fingerprint
                for final_path, expected_fingerprint in zip(
                    final_files,
                    expected,
                    strict=True,
                )
                if final_path == destination
            )
            backup = backups[destination]
            try:
                if os.path.lexists(backup):
                    _verify_directory_identity(
                        final.output.parent,
                        expected_parent_identity,
                    )
                    _restore_owned_backup(backup, destination, fingerprint)
                else:
                    # A backup may disappear between its successful move and
                    # rollback.  Silence is only safe when the original bytes are
                    # already back at the final path.
                    try:
                        current = _destination_fingerprint(
                            destination,
                            include_digest=True,
                        )
                    except (OSError, DestinationChangedError) as error:
                        raise DestinationChangedError(
                            f"旧输出备份在回滚前消失，且最终路径无法核验；请人工检查：{destination}"
                        ) from error
                    if not _same_destination_after_rename(fingerprint, current):
                        raise DestinationChangedError(
                            "旧输出备份在回滚前被移动，原字节不在最终路径；"
                            f"请人工检查目标目录：{final.output.parent}"
                        )
            except (OSError, DestinationChangedError) as error:
                rollback_errors.append(error)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            recovery_paths = ", ".join(
                str(path) for path in backups.values() if os.path.lexists(path)
            )
            recovery_text = recovery_paths or "无可用自动备份，请检查目标目录"
            raise BundleRollbackError(
                f"导出提交失败，且回滚未完全成功：{details}；人工恢复路径：{recovery_text}"
            ) from commit_error
        raise
    else:
        for destination, fingerprint in backup_fingerprints.items():
            backup = backups[destination]
            if os.path.lexists(backup):
                try:
                    _remove_owned_regular_file(
                        backup,
                        fingerprint,
                        mismatch_message=("旧输出备份路径身份已变化，未删除疑似外部文件"),
                    )
                except (OSError, DestinationChangedError) as error:
                    warnings.warn(
                        f"旧输出备份清理未完成，已保留可疑路径：{backup}（{error}）",
                        BundleCleanupWarning,
                        stacklevel=2,
                    )
        _fsync_directory(final.output.parent)


def export_run_bundle(
    run: CompensationRun,
    output_path: str | Path,
    *,
    allow_overwrite: bool = True,
    cancelled: CancellationCheck | None = None,
    expected_destination_fingerprints: tuple[
        DestinationFingerprint,
        DestinationFingerprint,
        DestinationFingerprint,
    ]
    | None = None,
) -> BundlePaths:
    """Export all three artifacts using a rollback-capable transaction.

    This is intentionally described as a rollback transaction, not as a
    power-loss-atomic multi-file operation. All files are generated in a staging
    directory on the destination filesystem; existing files are restored if a
    generation or commit step raises an exception.
    """

    if not isinstance(allow_overwrite, bool):
        raise ValueError("allow_overwrite 必须是布尔值")
    raise_if_cancelled(cancelled, message="导出已取消")
    paths = bundle_paths(output_path)
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _directory_identity(paths.output.parent)
    with _bundle_destination_locks(paths):
        _verify_directory_identity(paths.output.parent, parent_identity)
        verify_source_files_unchanged(run, cancelled=cancelled)
        _check_bundle_collisions(run, paths, allow_overwrite=allow_overwrite)
        if expected_destination_fingerprints is not None:
            _verify_destination_fingerprints(
                paths,
                expected_destination_fingerprints,
                allow_overwrite=allow_overwrite,
                cancelled=cancelled,
            )
        # GUI overwrite approval uses metadata only. The worker establishes the
        # digest-bearing commit baseline here, under the ResponseLab lock, so hashing
        # a large old output never freezes the GUI thread.
        destination_fingerprints = snapshot_bundle_destinations(
            paths,
            include_digest=True,
            cancelled=cancelled,
        )
        if expected_destination_fingerprints is not None:
            expected_metadata = tuple(
                DestinationFingerprint(
                    exists=fingerprint.exists,
                    device=fingerprint.device,
                    inode=fingerprint.inode,
                    size_bytes=fingerprint.size_bytes,
                    modified_time_ns=fingerprint.modified_time_ns,
                    changed_time_ns=fingerprint.changed_time_ns,
                )
                for fingerprint in expected_destination_fingerprints
            )
            baseline_metadata = tuple(
                DestinationFingerprint(
                    exists=fingerprint.exists,
                    device=fingerprint.device,
                    inode=fingerprint.inode,
                    size_bytes=fingerprint.size_bytes,
                    modified_time_ns=fingerprint.modified_time_ns,
                    changed_time_ns=fingerprint.changed_time_ns,
                )
                for fingerprint in destination_fingerprints
            )
            if baseline_metadata != expected_metadata:
                raise DestinationChangedError(
                    "导出目标在确认后发生变化；为避免覆盖其他程序的新文件，本次未提交"
                )
        _verify_directory_identity(paths.output.parent, parent_identity)
        with _pinned_directory(paths.output.parent, parent_identity) as final_parent_fd:
            staging_directory = Path(
                tempfile.mkdtemp(
                    prefix=f".{paths.output.name}.response-lab-staging-",
                    dir=paths.output.parent,
                )
            )
            staging_identity = _directory_identity(staging_directory)
            staged = BundlePaths(
                output=staging_directory / paths.output.name,
                response_csv=staging_directory / paths.response_csv.name,
                manifest=staging_directory / paths.manifest.name,
            )
            try:
                with _pinned_directory(
                    staging_directory,
                    staging_identity,
                ) as staged_parent_fd:
                    raise_if_cancelled(cancelled, message="导出已取消")
                    _verify_staging_directory_identity(staging_directory, staging_identity)
                    _write_primary_output(staged.output, run, cancelled=cancelled)
                    _verify_staging_directory_identity(staging_directory, staging_identity)
                    raise_if_cancelled(cancelled, message="导出已取消")
                    export_response_csv(
                        staged.response_csv,
                        run,
                        cancelled=cancelled,
                    )
                    _verify_staging_directory_identity(staging_directory, staging_identity)
                    raise_if_cancelled(cancelled, message="导出已取消")
                    manifest = build_manifest(
                        run,
                        paths.output,
                        output_evidence_path=staged.output,
                        cancelled=cancelled,
                    )
                    _verify_staging_directory_identity(staging_directory, staging_identity)
                    write_manifest_atomic(staged.manifest, manifest)
                    _verify_staging_directory_identity(staging_directory, staging_identity)
                    raise_if_cancelled(cancelled, message="导出已取消")
                    verify_source_files_unchanged(run, cancelled=cancelled)
                    raise_if_cancelled(cancelled, message="导出已取消")
                    _verify_destination_fingerprints(
                        paths,
                        destination_fingerprints,
                        allow_overwrite=allow_overwrite,
                        cancelled=cancelled,
                    )
                    _verify_directory_identity(paths.output.parent, parent_identity)
                    _commit_bundle_with_rollback(
                        staged,
                        paths,
                        destination_fingerprints,
                        parent_identity,
                        staged_parent_descriptor=staged_parent_fd,
                        final_parent_descriptor=final_parent_fd,
                        expected_staged_parent_identity=staging_identity,
                    )
            finally:
                _cleanup_staging_directory(staging_directory, staging_identity)
    return paths
