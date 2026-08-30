"""领域不变量、单文件原子写与整包回滚事务的回归测试。"""

from __future__ import annotations

import errno
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from response_lab import io as io_module
from response_lab import reporting as reporting_module
from response_lab.app import build_demo_run
from response_lab.cancellation import OperationCancelledError
from response_lab.dsp import run_compensation
from response_lab.io import (
    load_bin_timeseries,
    load_csv_timeseries,
    save_bin_timeseries,
    save_csv_timeseries,
)
from response_lab.models import CompensationRun, CompensationSettings, TimeSeries
from response_lab.reporting import (
    BundleBusyError,
    BundleRollbackError,
    SourceVerificationError,
    build_manifest,
    bundle_paths,
    export_response_csv,
    export_run_bundle,
    sha256_file,
    verify_source_files_unchanged,
    write_manifest_atomic,
)

_POSIX_OPEN_PATH_REPLACEMENT_ONLY = pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "该竞态需要重命名或删除仍打开的文件/目录；Windows 的无 SHARE_DELETE 锁会在"
        "注入前直接阻止操作，正常 Windows 导出与锁路径由其余跨平台测试覆盖"
    ),
)


def _file_backed_demo_run(tmp_path, *, reference_name="reference.csv"):
    run = build_demo_run()
    reference_path = save_csv_timeseries(
        tmp_path / reference_name,
        run.reference_pulse.time_s,
        run.reference_pulse.values,
    )
    dut_path = save_csv_timeseries(
        tmp_path / "dut.csv",
        run.dut_pulse.time_s,
        run.dut_pulse.values,
    )
    signal_path = save_csv_timeseries(
        tmp_path / "signal.csv",
        run.input_signal.time_s,
        run.input_signal.values,
    )
    return replace(
        run,
        reference_pulse=load_csv_timeseries(reference_path),
        dut_pulse=load_csv_timeseries(dut_path),
        input_signal=load_csv_timeseries(signal_path),
    )


def _signed_delay_run(delay_samples: int):
    sample_rate_hz = 1.0e9
    index = np.arange(1024, dtype=np.float64)
    base = np.exp(-0.5 * ((index - 240.0) / 2.0) ** 2)
    shifted = np.zeros_like(base)
    shifted[3:] = base[:-3]
    if delay_samples > 0:
        reference_values, dut_values = base, shifted
    else:
        reference_values, dut_values = shifted, base
    pulse_time_s = index / sample_rate_hz
    target_time_s = np.arange(2048, dtype=np.float64) / sample_rate_hz
    settings = CompensationSettings(
        mode="phase",
        band_low_hz=5.0e6,
        band_high_hz=350.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        detrend_phase=False,
        analysis_points=4097,
    )
    return run_compensation(
        TimeSeries(pulse_time_s, reference_values[:, None], sample_rate_hz),
        TimeSeries(pulse_time_s, dut_values[:, None], sample_rate_hz),
        TimeSeries(target_time_s, np.zeros((2048, 1)), sample_rate_hz),
        settings,
    )


def test_timeseries_rejects_nonuniform_axis_and_mismatched_sample_rate() -> None:
    uniform_time_s = np.arange(8, dtype=np.float64) * 1.0e-9
    values = np.arange(8, dtype=np.float64)

    with pytest.raises(ValueError, match="采样率.*时间列"):
        TimeSeries(uniform_time_s, values, 0.9e9)

    nonuniform_time_s = uniform_time_s.copy()
    nonuniform_time_s[5:] += 0.01e-9
    with pytest.raises(ValueError, match="等间隔"):
        TimeSeries(nonuniform_time_s, values, 1.0e9)


def test_csv_export_failure_preserves_existing_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "protected.csv"
    destination.write_text("old-result\n", encoding="utf-8")

    def broken_savetxt(stream, *_args, **_kwargs):
        stream.write("partial-result\n")
        raise OSError("模拟写盘失败")

    monkeypatch.setattr(io_module.np, "savetxt", broken_savetxt)
    with pytest.raises(OSError, match="写盘失败"):
        save_csv_timeseries(destination, np.arange(8), np.arange(8))

    assert destination.read_text(encoding="utf-8") == "old-result\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_bin_export_replace_failure_preserves_existing_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "protected.bin"
    destination.write_bytes(b"old-result")

    def broken_replace(*_args, **_kwargs):
        raise OSError("模拟原子替换失败")

    monkeypatch.setattr(io_module.os, "replace", broken_replace)
    with pytest.raises(OSError, match="原子替换失败"):
        save_bin_timeseries(
            destination,
            np.arange(8, dtype=np.float64),
            np.arange(8, dtype=np.float64),
        )

    assert destination.read_bytes() == b"old-result"
    assert list(tmp_path.iterdir()) == [destination]


def test_manifest_and_response_report_include_file_evidence(tmp_path) -> None:
    run = build_demo_run()
    output_path = save_csv_timeseries(
        tmp_path / "compensated.csv",
        run.input_signal.time_s,
        run.output_values,
    )

    manifest = build_manifest(run, output_path)
    manifest_path = write_manifest_atomic(tmp_path / "run.json", manifest)
    response_path = export_response_csv(tmp_path / "response.csv", run)

    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert parsed["schema"] == "response-lab-manifest/v4"
    assert parsed["analysis"]["phase_detrend_slope_rad_per_hz"] == pytest.approx(
        2.0 * np.pi * 2.5e-9
    )
    assert parsed["analysis"]["estimated_relative_delay_s"] == pytest.approx(2.5e-9)
    assert (
        parsed["analysis"]["relative_delay_sign_convention"]
        == "positive_means_dut_later_than_reference"
    )
    assert (
        parsed["analysis"]["response_magnitude_db_definition"]
        == "20*log10(abs(dt_s*rfft(h)))_interpolated_on_common_frequency_grid"
    )
    assert parsed["analysis"]["response_magnitude_scale"] == "raw_input_scale"
    assert parsed["output"]["sha256"] == sha256_file(output_path)
    assert parsed["output"]["size_bytes"] == output_path.stat().st_size
    assert parsed["settings"]["detrend_phase"] is True
    assert "remove_relative_delay" not in parsed["settings"]
    assert parsed["application"]["method"] == "zero_extend_czt_pulse_ratio_rfft_multiply_irfft_crop"
    assert parsed["application"]["boundary_mode"] == "zero"
    assert parsed["application"]["extended_samples"] == 3 * run.input_signal.samples - 2
    assert "fir" not in parsed

    header = response_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("frequency_hz,reference_magnitude_db,dut_magnitude_db")
    assert "fitted_linear_phase_trend_deg,phase_after_optional_detrend_deg" in header
    assert "phase_linear_fit_deg" not in header
    response = np.loadtxt(response_path, delimiter=",", skiprows=1)
    assert response.shape[1] == 10
    np.testing.assert_allclose(response[:, 1], run.analysis.reference_magnitude_db)
    np.testing.assert_allclose(response[:, 2], run.analysis.dut_magnitude_db)


def test_reflect_run_without_application_metadata_derives_consistent_manifest(tmp_path) -> None:
    """省略应用细节时，运行模型和 manifest 都必须从设置派生 reflect 合同。"""

    base = build_demo_run()
    reflect_settings = replace(base.analysis.settings, boundary_mode="reflect")
    reflect_analysis = replace(base.analysis, settings=reflect_settings)
    direct_run = replace(
        base,
        analysis=reflect_analysis,
        application_method="",
        application_metadata={},
    )
    owned_run = CompensationRun.from_owned_output(
        reference_pulse=base.reference_pulse,
        dut_pulse=base.dut_pulse,
        input_signal=base.input_signal,
        output_values=base.output_values.copy(),
        analysis=reflect_analysis,
    )

    for index, run in enumerate((direct_run, owned_run)):
        manifest = build_manifest(run, tmp_path / f"reflect-{index}.csv")
        assert run.application_method == (
            "reflect_extend_czt_pulse_ratio_rfft_multiply_irfft_crop"
        )
        assert manifest["settings"]["boundary_mode"] == "reflect"
        assert manifest["application"]["boundary_mode"] == "reflect"
        assert manifest["application"]["method"] == run.application_method


@pytest.mark.parametrize("delay_samples", [3, -3])
def test_csv_and_json_preserve_signed_three_nanosecond_delay_contract(
    tmp_path,
    delay_samples: int,
) -> None:
    run = _signed_delay_run(delay_samples)
    response_path = export_response_csv(tmp_path / f"response-{delay_samples}.csv", run)
    parsed = json.loads(json.dumps(build_manifest(run, tmp_path / f"output-{delay_samples}.csv")))

    expected_delay_s = delay_samples / 1.0e9
    assert parsed["analysis"]["estimated_relative_delay_s"] == pytest.approx(
        expected_delay_s,
        abs=0.05e-9,
    )
    assert (
        parsed["analysis"]["relative_delay_sign_convention"]
        == "positive_means_dut_later_than_reference"
    )
    response = np.loadtxt(response_path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(
        response[:, 5],
        np.degrees(run.analysis.phase_trend_rad),
        atol=1.0e-12,
    )


def test_domain_result_arrays_are_copied_and_read_only() -> None:
    run = build_demo_run()
    arrays = (
        run.analysis.frequency_hz,
        run.analysis.correction_ideal,
        run.output_values,
    )

    assert all(array.flags.writeable is False for array in arrays)
    for array in arrays:
        with pytest.raises(ValueError, match="read-only|只读"):
            array.flat[0] = 0.0


def test_response_analysis_rejects_mismatched_array_shape() -> None:
    run = build_demo_run()

    with pytest.raises(ValueError, match="等长"):
        replace(
            run.analysis,
            dut_magnitude_db=run.analysis.dut_magnitude_db[:-1],
        )


def test_source_verification_detects_same_size_change_and_deletion(tmp_path) -> None:
    changed_run = _file_backed_demo_run(tmp_path / "changed")
    changed_path = changed_run.reference_pulse.source_path
    assert changed_path is not None
    original = changed_path.read_bytes()
    changed_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(SourceVerificationError, match="内容已变化"):
        verify_source_files_unchanged(changed_run)

    deleted_run = _file_backed_demo_run(tmp_path / "deleted")
    deleted_path = deleted_run.dut_pulse.source_path
    assert deleted_path is not None
    deleted_path.unlink()
    with pytest.raises(SourceVerificationError, match="已删除或移动"):
        verify_source_files_unchanged(deleted_run)


def test_manifest_uses_load_time_source_snapshot_not_current_file(tmp_path) -> None:
    run = _file_backed_demo_run(tmp_path)
    source = run.reference_pulse.source_path
    assert source is not None
    loaded_size = run.reference_pulse.source_metadata["source_size_bytes"]
    loaded_digest = run.reference_pulse.source_metadata["source_sha256"]
    source.write_bytes(b"x" * int(loaded_size))

    manifest = build_manifest(run, tmp_path / "not-written.csv")
    evidence = manifest["inputs"]["reference_pulse"]

    assert evidence["size_bytes"] == loaded_size
    assert evidence["sha256"] == loaded_digest
    assert evidence["sha256"] != sha256_file(source)


def test_manifest_can_hash_staged_output_while_recording_final_path(tmp_path) -> None:
    run = build_demo_run()
    final_path = tmp_path / "final.csv"
    final_path.write_text("old\n", encoding="utf-8")
    staged_path = tmp_path / "staged.csv"
    staged_path.write_text("new-output\n", encoding="utf-8")

    manifest = build_manifest(
        run,
        final_path,
        output_evidence_path=staged_path,
    )

    assert manifest["output"]["path"] == str(final_path.resolve())
    assert manifest["output"]["sha256"] == sha256_file(staged_path)
    assert manifest["output"]["size_bytes"] == staged_path.stat().st_size


def test_response_csv_failure_preserves_existing_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "protected_response.csv"
    destination.write_text("old-response\n", encoding="utf-8")

    def broken_savetxt(stream, *_args, **_kwargs):
        stream.write("partial-response\n")
        raise OSError("模拟响应表写盘失败")

    monkeypatch.setattr(reporting_module.np, "savetxt", broken_savetxt)
    with pytest.raises(OSError, match="响应表写盘失败"):
        export_response_csv(destination, build_demo_run())

    assert destination.read_text(encoding="utf-8") == "old-response\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_bundle_paths_are_predictable_and_validate_suffix(tmp_path) -> None:
    paths = bundle_paths(tmp_path / "capture.csv")

    assert paths.output == (tmp_path / "capture.csv").resolve()
    assert paths.response_csv.name == "capture_response.csv"
    assert paths.manifest.name == "capture.csv.response-lab.json"
    with pytest.raises(ValueError, match="扩展名"):
        bundle_paths(tmp_path / "capture.dat")


def test_gui_destination_snapshot_is_metadata_only_but_worker_baseline_hashes(
    tmp_path,
    monkeypatch,
) -> None:
    """批准快照不阻塞 GUI；worker 基线仍能识别 stat 盲区内的等长改写。"""

    paths = bundle_paths(tmp_path / "existing.csv")
    paths.output.write_bytes(b"AAAA")
    real_sha256 = reporting_module.sha256_file

    def forbidden_gui_hash(*_args, **_kwargs):
        raise AssertionError("quick approval snapshot must not hash file contents")

    monkeypatch.setattr(reporting_module, "sha256_file", forbidden_gui_hash)
    quick = reporting_module.snapshot_bundle_destinations(paths)
    assert quick[0].sha256 == ""

    monkeypatch.setattr(reporting_module, "sha256_file", real_sha256)
    worker_baseline = reporting_module.snapshot_bundle_destinations(
        paths,
        include_digest=True,
    )
    paths.output.write_bytes(b"BBBB")
    rewritten = reporting_module.snapshot_bundle_destinations(
        paths,
        include_digest=True,
    )
    stat_blind_rewrite = replace(
        rewritten[0],
        device=worker_baseline[0].device,
        inode=worker_baseline[0].inode,
        size_bytes=worker_baseline[0].size_bytes,
        modified_time_ns=worker_baseline[0].modified_time_ns,
        changed_time_ns=worker_baseline[0].changed_time_ns,
    )

    assert stat_blind_rewrite.sha256 != worker_baseline[0].sha256
    assert stat_blind_rewrite != worker_baseline[0]
    with pytest.raises(reporting_module.DestinationChangedError):
        reporting_module._verify_destination_fingerprints(
            paths,
            worker_baseline,
            allow_overwrite=True,
        )


def test_bundle_export_writes_three_consistent_artifacts_without_staging_residue(
    tmp_path,
) -> None:
    run = _file_backed_demo_run(tmp_path / "inputs")

    paths = export_run_bundle(run, tmp_path / "result.csv")

    assert all(path.is_file() for path in paths.as_tuple())
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["output"]["path"] == str(paths.output)
    assert manifest["output"]["sha256"] == sha256_file(paths.output)
    assert manifest["output"]["size_bytes"] == paths.output.stat().st_size
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


def test_bundle_export_writes_self_describing_bin_with_sidecars(tmp_path) -> None:
    run = build_demo_run()

    paths = export_run_bundle(run, tmp_path / "result.bin")

    assert paths.output.read_bytes()[:2] == b"AG"
    decoded = load_bin_timeseries(paths.output)
    np.testing.assert_allclose(
        decoded.values[:, 0],
        run.output_values[:, 0],
        rtol=2.0e-7,
        atol=1.0e-7,
    )
    assert decoded.sample_rate_hz == pytest.approx(run.input_signal.sample_rate_hz)
    assert paths.output.stat().st_size == 164 + run.output_values.shape[0] * 4
    assert paths.response_csv.is_file()
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["output"]["sha256"] == sha256_file(paths.output)


def test_bundle_export_blocks_changed_source_before_writing(tmp_path) -> None:
    run = _file_backed_demo_run(tmp_path / "inputs")
    source = run.input_signal.source_path
    assert source is not None
    source.write_bytes(source.read_bytes() + b"\n")
    paths = bundle_paths(tmp_path / "result.csv")

    with pytest.raises(SourceVerificationError, match="大小已变化"):
        export_run_bundle(run, paths.output)

    assert not any(path.exists() for path in paths.as_tuple())
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


def test_bundle_export_never_overwrites_a_source_through_sidecar_name(tmp_path) -> None:
    run = _file_backed_demo_run(
        tmp_path,
        reference_name="result_response.csv",
    )
    source = run.reference_pulse.source_path
    assert source is not None
    before = source.read_bytes()

    with pytest.raises(ValueError, match="不能覆盖任何输入源"):
        export_run_bundle(run, tmp_path / "result.csv")

    assert source.read_bytes() == before
    assert not (tmp_path / "result.csv").exists()


def test_bundle_export_respects_sidecar_conflict_when_overwrite_disabled(tmp_path) -> None:
    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "result.csv")
    paths.response_csv.write_text("keep-me\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="已存在"):
        export_run_bundle(run, paths.output, allow_overwrite=False)

    assert paths.response_csv.read_text(encoding="utf-8") == "keep-me\n"
    assert not paths.output.exists()
    assert not paths.manifest.exists()


def test_bundle_generation_failure_preserves_all_existing_files(
    tmp_path,
    monkeypatch,
) -> None:
    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "result.csv")
    old_contents = {
        paths.output: b"old-output",
        paths.response_csv: b"old-response",
        paths.manifest: b"old-manifest",
    }
    for path, content in old_contents.items():
        path.write_bytes(content)

    def fail_response(*_args, **_kwargs):
        raise OSError("模拟整包生成失败")

    monkeypatch.setattr(reporting_module, "export_response_csv", fail_response)
    with pytest.raises(OSError, match="整包生成失败"):
        export_run_bundle(run, paths.output)

    assert {path: path.read_bytes() for path in old_contents} == old_contents
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


def test_no_overwrite_rechecks_late_concurrent_file_before_commit(
    tmp_path,
    monkeypatch,
) -> None:
    """allow_overwrite=False 的合同必须覆盖生成结束到提交之间的并发创建。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "late-create.csv")
    real_write = reporting_module._write_primary_output

    def write_then_create_concurrent_file(path, current_run, **kwargs):
        real_write(path, current_run, **kwargs)
        paths.output.write_bytes(b"concurrent-owner")

    monkeypatch.setattr(
        reporting_module,
        "_write_primary_output",
        write_then_create_concurrent_file,
    )

    with pytest.raises(FileExistsError, match="导出文件已存在"):
        export_run_bundle(run, paths.output, allow_overwrite=False)

    assert paths.output.read_bytes() == b"concurrent-owner"
    assert not paths.response_csv.exists()
    assert not paths.manifest.exists()


def test_commit_uses_atomic_no_clobber_against_last_instant_regular_file(
    tmp_path,
    monkeypatch,
) -> None:
    """最终身份复核返回后出现的普通文件也不能被提交阶段覆盖。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "commit-race.csv")
    real_commit = reporting_module._commit_bundle_with_rollback

    def inject_file_at_commit(
        staged,
        final,
        expected,
        expected_parent_identity,
        **commit_kwargs,
    ):
        paths.output.write_bytes(b"last-instant-owner")
        return real_commit(
            staged,
            final,
            expected,
            expected_parent_identity,
            **commit_kwargs,
        )

    monkeypatch.setattr(
        reporting_module,
        "_commit_bundle_with_rollback",
        inject_file_at_commit,
    )

    with pytest.raises(FileExistsError):
        export_run_bundle(run, paths.output)

    assert paths.output.read_bytes() == b"last-instant-owner"
    assert not paths.response_csv.exists()
    assert not paths.manifest.exists()


def test_export_falls_back_safely_when_destination_filesystem_has_no_hardlinks(
    tmp_path,
    monkeypatch,
) -> None:
    """exFAT/SMB 等无 hardlink 文件系统仍应以原子 no-replace 完成导出。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "no-hardlink.csv")

    def unsupported_link(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    monkeypatch.setattr(reporting_module.os, "link", unsupported_link)

    returned = export_run_bundle(run, paths.output)

    assert returned == paths
    assert all(path.is_file() for path in paths.as_tuple())


def test_native_no_replace_primitive_preserves_existing_destination(tmp_path) -> None:
    """平台原语本身必须原子拒绝已存在目标，而非覆盖后再检测。"""

    source = tmp_path / "staged.tmp"
    destination = tmp_path / "owned.csv"
    source.write_bytes(b"new-response-lab")
    destination.write_bytes(b"external-owner")

    with pytest.raises(FileExistsError):
        reporting_module._rename_no_replace(source, destination)

    assert source.read_bytes() == b"new-response-lab"
    assert destination.read_bytes() == b"external-owner"


def test_no_hardlink_fallback_never_overwrites_owner_arriving_at_native_commit(
    tmp_path,
    monkeypatch,
) -> None:
    """无 hardlink 路径必须把最终竞争交给原子 no-replace 原语。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "native-race.csv")
    real_rename_no_replace_at = reporting_module._rename_no_replace_at

    def unsupported_link(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    def late_owner_wins(source, destination, **rename_kwargs):
        if Path(destination) == paths.output:
            Path(destination).write_bytes(b"late-native-owner")
            raise FileExistsError(errno.EEXIST, "destination exists", destination)
        return real_rename_no_replace_at(
            source,
            destination,
            **rename_kwargs,
        )

    monkeypatch.setattr(reporting_module.os, "link", unsupported_link)
    monkeypatch.setattr(reporting_module, "_rename_no_replace_at", late_owner_wins)

    with pytest.raises(FileExistsError):
        export_run_bundle(run, paths.output)

    assert paths.output.read_bytes() == b"late-native-owner"
    assert not paths.response_csv.exists()
    assert not paths.manifest.exists()


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_rollback_never_unlinks_foreign_replacement_of_committed_output(
    tmp_path,
    monkeypatch,
) -> None:
    """已发布路径在后续失败前换主时，回滚只能删除原发布 inode。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "rollback-owner-race.csv")
    real_staged_commit = reporting_module._commit_staged_file
    owned_handles = []

    def replace_first_then_fail_second(
        source,
        destination,
        expected_source=None,
        **commit_kwargs,
    ):
        if destination == paths.response_csv:
            raise OSError("模拟第二份 staged 文件提交失败")
        real_staged_commit(source, destination, expected_source, **commit_kwargs)
        if destination == paths.output:
            # 保持原 inode 仍被打开，确保随后外部文件不会复用该 inode。
            owned_handles.append(destination.open("rb"))
            destination.unlink()
            destination.write_bytes(b"foreign-late-owner")

    monkeypatch.setattr(
        reporting_module,
        "_commit_staged_file",
        replace_first_then_fail_second,
    )

    try:
        with pytest.raises(BundleRollbackError, match="回滚未完全成功"):
            export_run_bundle(run, paths.output)
    finally:
        for handle in owned_handles:
            handle.close()

    assert paths.output.read_bytes() == b"foreign-late-owner"
    assert not paths.response_csv.exists()
    assert not paths.manifest.exists()


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_commit_rejects_parent_directory_replaced_by_symlink(
    tmp_path,
    monkeypatch,
) -> None:
    """批准的父目录换成 symlink 后，提交不得跟随到 victim。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    approved_parent = tmp_path / "approved"
    approved_parent.mkdir()
    paths = bundle_paths(approved_parent / "result.csv")
    moved_parent = tmp_path / "approved-moved"
    victim = tmp_path / "victim"
    victim.mkdir()
    real_staged_commit = reporting_module._commit_staged_file
    injected = False
    foreign_staged_files: list[Path] = []

    def swap_parent_after_check(
        source,
        destination,
        expected_source=None,
        **commit_kwargs,
    ):
        nonlocal injected
        if not injected:
            injected = True
            relative_source = Path(source).relative_to(approved_parent)
            approved_parent.rename(moved_parent)
            approved_parent.symlink_to(victim, target_is_directory=True)
            foreign_source = victim / relative_source
            foreign_source.parent.mkdir(parents=True)
            foreign_source.write_bytes(b"foreign-staged-owner")
            foreign_staged_files.append(foreign_source)
        return real_staged_commit(
            source,
            destination,
            expected_source,
            **commit_kwargs,
        )

    monkeypatch.setattr(
        reporting_module,
        "_commit_staged_file",
        swap_parent_after_check,
    )

    try:
        with (
            pytest.warns(reporting_module.BundleCleanupWarning, match="staging"),
            pytest.raises(
                reporting_module.DestinationChangedError,
                match="父目录|staged",
            ),
        ):
            export_run_bundle(run, paths.output)
        assert foreign_staged_files[0].read_bytes() == b"foreign-staged-owner"
        assert not (victim / paths.output.name).exists()
    finally:
        if approved_parent.is_symlink():
            approved_parent.unlink()
        if moved_parent.exists():
            moved_parent.rename(approved_parent)


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_staging_cleanup_preserves_foreign_same_name_directory(
    tmp_path,
    monkeypatch,
) -> None:
    """finally 只能递归删除本次 mkdtemp 的 inode，不能删除同名后来者。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "staging-race.csv")
    moved_staging: list[Path] = []
    foreign_staging: list[Path] = []

    def replace_staging_then_fail(staged, *_args, **_kwargs):
        original = staged.output.parent
        moved = original.with_name(f"{original.name}-moved-owned")
        original.rename(moved)
        original.mkdir()
        (original / "valuable.txt").write_bytes(b"foreign-staging-owner")
        moved_staging.append(moved)
        foreign_staging.append(original)
        raise OSError("模拟提交前 staging 路径换主")

    monkeypatch.setattr(
        reporting_module,
        "_commit_bundle_with_rollback",
        replace_staging_then_fail,
    )

    with (
        pytest.warns(reporting_module.BundleCleanupWarning, match="疑似外部目录"),
        pytest.raises(OSError, match="staging 路径换主"),
    ):
        export_run_bundle(run, paths.output)

    assert foreign_staging[0].is_dir()
    assert (foreign_staging[0] / "valuable.txt").read_bytes() == (b"foreign-staging-owner")
    assert moved_staging[0].is_dir()


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_success_cleanup_preserves_foreign_replacement_of_backup(
    tmp_path,
    monkeypatch,
) -> None:
    """提交成功后的旧备份清理也必须核对记录的 inode。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "backup-cleanup-race.csv")
    for path in paths.as_tuple():
        path.write_bytes(b"old-owner")
    real_fsync_directory = reporting_module._fsync_directory
    injected = False
    old_handles = []
    foreign_backups: list[Path] = []

    def replace_backup_before_cleanup(directory):
        nonlocal injected
        if not injected:
            candidates = tuple(Path(directory).glob(".*.response-lab-backup-*"))
            if candidates:
                injected = True
                backup = candidates[0]
                old_handles.append(backup.open("rb"))
                backup.unlink()
                backup.write_bytes(b"foreign-backup-owner")
                foreign_backups.append(backup)
        return real_fsync_directory(directory)

    monkeypatch.setattr(
        reporting_module,
        "_fsync_directory",
        replace_backup_before_cleanup,
    )

    try:
        with pytest.warns(reporting_module.BundleCleanupWarning, match="备份路径身份已变化"):
            returned = export_run_bundle(run, paths.output)
    finally:
        for handle in old_handles:
            handle.close()

    assert returned == paths
    surviving_foreign = tuple(
        path
        for path in tmp_path.iterdir()
        if path.is_file() and path.read_bytes() == b"foreign-backup-owner"
    )
    assert surviving_foreign
    assert all(path.is_file() for path in paths.as_tuple())


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_rollback_preserves_foreign_replacement_of_old_backup(
    tmp_path,
    monkeypatch,
) -> None:
    """rollback 也不能把换主后的 backup 当作旧输出恢复或删除。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "backup-rollback-race.csv")
    for path in paths.as_tuple():
        path.write_bytes(b"old-owner")
    real_staged_commit = reporting_module._commit_staged_file
    old_handles = []
    foreign_backups: list[Path] = []

    def replace_backup_then_fail(
        source,
        destination,
        expected_source=None,
        **commit_kwargs,
    ):
        if destination == paths.response_csv:
            backup = next(tmp_path.glob(".*.response-lab-backup-*"))
            old_handles.append(backup.open("rb"))
            backup.unlink()
            backup.write_bytes(b"foreign-backup-owner")
            foreign_backups.append(backup)
            raise OSError("模拟 staged 提交失败")
        real_staged_commit(source, destination, expected_source, **commit_kwargs)

    monkeypatch.setattr(
        reporting_module,
        "_commit_staged_file",
        replace_backup_then_fail,
    )

    try:
        with pytest.raises(BundleRollbackError, match="回滚未完全成功"):
            export_run_bundle(run, paths.output)
    finally:
        for handle in old_handles:
            handle.close()

    surviving_foreign = tuple(
        path
        for path in tmp_path.iterdir()
        if path.is_file() and path.read_bytes() == b"foreign-backup-owner"
    )
    assert surviving_foreign


def test_late_directory_collision_is_never_moved_into_staging_cleanup(
    tmp_path,
    monkeypatch,
) -> None:
    """提交前出现的目录必须原地保留，绝不能作为 backup 被递归删除。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "directory-race.csv")
    real_write = reporting_module._write_primary_output

    def write_then_create_directory(path, current_run, **kwargs):
        real_write(path, current_run, **kwargs)
        paths.response_csv.mkdir()
        (paths.response_csv / "valuable.txt").write_bytes(b"must-survive")

    monkeypatch.setattr(
        reporting_module,
        "_write_primary_output",
        write_then_create_directory,
    )

    with pytest.raises(IsADirectoryError, match="不是普通文件"):
        export_run_bundle(run, paths.output)

    assert (paths.response_csv / "valuable.txt").read_bytes() == b"must-survive"
    assert not paths.output.exists()
    assert not paths.manifest.exists()


def test_incomplete_rollback_preserves_old_bytes_in_recovery_backup(
    tmp_path,
    monkeypatch,
) -> None:
    """恢复旧文件失败时必须留下可人工恢复的副本，不能随 staging 一起删除。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "rollback-recovery.csv")
    old_contents = {
        paths.output: b"old-output",
        paths.response_csv: b"old-response",
        paths.manifest: b"old-manifest",
    }
    for path, content in old_contents.items():
        path.write_bytes(content)
    real_replace = reporting_module._commit_replace
    real_staged_commit = reporting_module._commit_staged_file
    staged_commit_failed = False

    def fail_primary_restore(source, destination):
        nonlocal staged_commit_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            staged_commit_failed
            and destination_path == paths.output
            and "response-lab-backup" in source_path.name
        ):
            raise OSError("模拟旧主输出恢复失败")
        real_replace(source_path, destination_path)

    def fail_second_staged_commit(
        source,
        destination,
        expected_source=None,
        **commit_kwargs,
    ):
        nonlocal staged_commit_failed
        if destination == paths.response_csv:
            staged_commit_failed = True
            raise OSError("模拟第二份 staged 文件提交失败")
        real_staged_commit(source, destination, expected_source, **commit_kwargs)

    monkeypatch.setattr(
        reporting_module,
        "_commit_replace",
        fail_primary_restore,
    )
    monkeypatch.setattr(
        reporting_module,
        "_commit_staged_file",
        fail_second_staged_commit,
    )

    with pytest.raises(BundleRollbackError, match="人工恢复"):
        export_run_bundle(run, paths.output)

    recovery_files = tuple(tmp_path.glob(".*.response-lab-backup-*"))
    assert any(path.read_bytes() == b"old-output" for path in recovery_files)
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


def test_backup_disappearing_before_rollback_is_reported_and_old_bytes_survive(
    tmp_path,
    monkeypatch,
) -> None:
    """已移动的 backup 不能在回滚时静默跳过并留下缺失 final。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "missing-backup.csv")
    for path in paths.as_tuple():
        path.write_bytes(b"old-owner")
    real_staged_commit = reporting_module._commit_staged_file
    moved_old = tmp_path / "externally-moved-old-output"
    injected = False

    def move_backup_then_fail(
        source,
        destination,
        expected_source=None,
        **commit_kwargs,
    ):
        nonlocal injected
        if not injected:
            injected = True
            backup = next(
                path
                for path in tmp_path.glob(".missing-backup.csv.response-lab-backup-*")
                if path.is_file()
            )
            backup.rename(moved_old)
            raise OSError("模拟 backup 在回滚前被外部移动")
        return real_staged_commit(
            source,
            destination,
            expected_source,
            **commit_kwargs,
        )

    monkeypatch.setattr(
        reporting_module,
        "_commit_staged_file",
        move_backup_then_fail,
    )

    with pytest.raises(BundleRollbackError, match="回滚未完全成功"):
        export_run_bundle(run, paths.output)

    assert moved_old.read_bytes() == b"old-owner"
    assert not paths.output.exists()
    assert paths.response_csv.read_bytes() == b"old-owner"
    assert paths.manifest.read_bytes() == b"old-owner"


def test_backup_snapshot_failure_after_move_still_restores_complete_old_batch(
    tmp_path,
    monkeypatch,
) -> None:
    """final→backup 已成功后，即使摘要失败也必须把全部旧文件恢复。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "backup-hash-error.csv")
    old_contents = {
        paths.output: b"old-output",
        paths.response_csv: b"old-response",
        paths.manifest: b"old-manifest",
    }
    for path, content in old_contents.items():
        path.write_bytes(content)
    real_fingerprint = reporting_module._destination_fingerprint
    injected = False

    def fail_first_backup_digest(path, *, include_digest=False, cancelled=None):
        nonlocal injected
        if include_digest and "response-lab-backup" in Path(path).name and not injected:
            injected = True
            raise OSError("模拟 backup 摘要失败")
        return real_fingerprint(
            Path(path),
            include_digest=include_digest,
            cancelled=cancelled,
        )

    monkeypatch.setattr(
        reporting_module,
        "_destination_fingerprint",
        fail_first_backup_digest,
    )

    with pytest.raises(OSError, match="backup 摘要失败"):
        export_run_bundle(run, paths.output)

    assert injected
    assert {path: path.read_bytes() for path in old_contents} == old_contents
    assert not tuple(tmp_path.glob(".*.response-lab-backup-*"))


def test_hardlink_publish_cleanup_failure_is_success_not_partial_batch(
    tmp_path,
    monkeypatch,
) -> None:
    """hardlink 已发布后 staged unlink 失败只能告警，不能漏记 committed final。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "link-cleanup.csv")
    real_unlink = reporting_module.os.unlink
    injected = False

    def fail_first_quarantine_unlink(path, *args, **kwargs):
        nonlocal injected
        candidate = Path(path)
        if "response-lab-quarantine" in candidate.name and not injected:
            injected = True
            raise OSError("模拟 staged hardlink 名称清理失败")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(reporting_module.os, "unlink", fail_first_quarantine_unlink)

    with pytest.warns(reporting_module.BundleCleanupWarning, match="清理失败|清理未完成"):
        returned = export_run_bundle(run, paths.output)

    assert returned == paths
    assert injected
    assert all(path.is_file() for path in paths.as_tuple())


def test_overlapping_response_lab_exports_are_serialized_by_all_final_paths(
    tmp_path,
) -> None:
    """共享任一最终路径的两个整包导出不能同时进入生成或提交。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "locked.csv")
    paths.output.parent.mkdir(parents=True, exist_ok=True)

    with (
        reporting_module._bundle_destination_locks(paths),
        pytest.raises(BundleBusyError, match="另一个 ResponseLab"),
    ):
        export_run_bundle(run, paths.output)

    assert not tuple(tmp_path.glob("*.lock"))


def test_export_locks_casefold_equivalent_destination_spellings(tmp_path) -> None:
    """大小写不敏感卷的等价拼写必须无条件落到同一锁键。"""

    upper = bundle_paths(tmp_path / "CaseSensitiveSpelling.csv")
    lower = bundle_paths(tmp_path / "casesensitivespelling.csv")

    assert reporting_module._lock_file_path(upper.output) == (
        reporting_module._lock_file_path(lower.output)
    )
    with (
        reporting_module._bundle_destination_locks(upper),
        pytest.raises(BundleBusyError, match="另一个 ResponseLab"),
        reporting_module._bundle_destination_locks(lower),
    ):
        pass


def test_export_locks_existing_hardlink_aliases_by_file_identity(tmp_path) -> None:
    """两个不同路径指向同一现存 inode 时也必须共享身份锁。"""

    first = bundle_paths(tmp_path / "first.csv")
    second = bundle_paths(tmp_path / "second.csv")
    first.output.write_bytes(b"shared-old-output")
    os.link(first.output, second.output)

    with (
        reporting_module._bundle_destination_locks(first),
        pytest.raises(BundleBusyError, match="另一个 ResponseLab"),
        reporting_module._bundle_destination_locks(second),
    ):
        pass


def test_export_lock_resource_failure_is_not_misreported_as_contention(
    tmp_path,
    monkeypatch,
) -> None:
    """ENOLCK 等锁设施故障必须保留真实错误，不能提示“另一个任务”。"""

    paths = bundle_paths(tmp_path / "lock-resource.csv")

    def no_lock_resources(_handle):
        raise OSError(errno.ENOLCK, "lock table exhausted")

    monkeypatch.setattr(reporting_module, "_acquire_lock", no_lock_resources)

    with (
        pytest.raises(OSError) as captured,
        reporting_module._bundle_destination_locks(paths),
    ):
        pass

    assert captured.value.errno == errno.ENOLCK
    assert not isinstance(captured.value, BundleBusyError)


def test_windows_hardlink_error_classifier_handles_winerror_variants() -> None:
    """FAT/SMB/权限型 Windows hardlink 失败应进入安全 rename fallback。"""

    unsupported = OSError(errno.EINVAL, "unsupported hardlink")
    unsupported.winerror = 50
    collision = OSError(errno.EEXIST, "destination exists")
    collision.winerror = 183

    assert reporting_module._hardlink_is_unavailable(unsupported, windows=True)
    assert not reporting_module._hardlink_is_unavailable(collision, windows=True)


def test_export_lock_directory_symlink_is_rejected_without_touching_target(
    tmp_path,
    monkeypatch,
) -> None:
    """锁目录若被替换成 symlink，不能跟随它在用户文件旁创建或写入锁。"""

    temp_root = tmp_path / "temp-root"
    temp_root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (temp_root / "response-lab-export-locks").symlink_to(victim, target_is_directory=True)
    monkeypatch.setattr(reporting_module.tempfile, "gettempdir", lambda: str(temp_root))
    paths = bundle_paths(tmp_path / "result.csv")

    with (
        pytest.raises(RuntimeError, match="锁目录"),
        reporting_module._bundle_destination_locks(paths),
    ):
        pass

    assert tuple(victim.iterdir()) == ()


def test_bundle_commit_failure_restores_all_existing_files(tmp_path, monkeypatch) -> None:
    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "result.csv")
    old_contents = {
        paths.output: b"old-output",
        paths.response_csv: b"old-response",
        paths.manifest: b"old-manifest",
    }
    for path, content in old_contents.items():
        path.write_bytes(content)

    real_staged_commit = reporting_module._commit_staged_file
    failure_injected = False

    def fail_second_staged_commit(
        source,
        destination,
        expected_source=None,
        **commit_kwargs,
    ):
        nonlocal failure_injected
        is_new_response_commit = destination == paths.response_csv
        if is_new_response_commit and not failure_injected:
            failure_injected = True
            raise OSError("模拟整包提交失败")
        real_staged_commit(source, destination, expected_source, **commit_kwargs)

    monkeypatch.setattr(
        reporting_module,
        "_commit_staged_file",
        fail_second_staged_commit,
    )
    with pytest.raises(OSError, match="整包提交失败"):
        export_run_bundle(run, paths.output)

    assert failure_injected is True
    assert {path: path.read_bytes() for path in old_contents} == old_contents
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


def test_csv_export_cancellation_removes_temp_and_preserves_destination(
    tmp_path,
    monkeypatch,
) -> None:
    """分块写出收到取消时不得替换已有文件或遗留临时文件。"""

    destination = tmp_path / "large.csv"
    destination.write_bytes(b"existing-output")
    samples = io_module._EXPORT_CHUNK_SAMPLES + 8  # noqa: SLF001
    time_s = np.arange(samples, dtype=np.float64) * 1.0e-9
    values = np.linspace(-1.0, 1.0, samples, dtype=np.float64)
    real_mkstemp = io_module.tempfile.mkstemp
    staged_descriptors: list[int] = []

    def capture_staging_descriptor(*args, **kwargs):
        descriptor, temporary_name = real_mkstemp(*args, **kwargs)
        staged_descriptors.append(descriptor)
        return descriptor, temporary_name

    monkeypatch.setattr(io_module.tempfile, "mkstemp", capture_staging_descriptor)

    def cancelled_after_staging_starts() -> bool:
        return bool(list(tmp_path.glob(".large.csv.*.tmp")))

    with pytest.raises(OperationCancelledError, match="导出已取消"):
        save_csv_timeseries(
            destination,
            time_s,
            values,
            cancelled=cancelled_after_staging_starts,
        )

    assert len(staged_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(staged_descriptors[0])
    assert destination.read_bytes() == b"existing-output"
    assert not list(tmp_path.glob(".large.csv.*.tmp"))


def test_response_csv_export_cancels_between_chunks_and_preserves_destination(
    tmp_path,
    monkeypatch,
) -> None:
    """响应诊断表也必须可协作取消，不能一次性拼接并写完整个大矩阵。"""

    run = build_demo_run()
    rows = 12_289
    frequency_hz = np.linspace(0.0, 1.0e9, rows)
    zeros = np.zeros(rows, dtype=np.float64)
    analysis = replace(
        run.analysis,
        frequency_hz=frequency_hz,
        reference_magnitude_db=zeros,
        dut_magnitude_db=zeros,
        reference_phase_rad=zeros,
        dut_phase_rad=zeros,
        magnitude_difference_db=zeros,
        phase_difference_rad=zeros,
        phase_trend_rad=zeros,
        phase_after_optional_detrend_rad=zeros,
        reliable_mask=np.ones(rows, dtype=np.bool_),
        correction_ideal=np.ones(rows, dtype=np.complex128),
    )
    run = replace(run, analysis=analysis)
    destination = tmp_path / "response.csv"
    destination.write_bytes(b"existing-response")
    cancellation_checks = 0
    written_chunks = 0
    written_row_counts: list[int] = []
    real_savetxt = reporting_module.np.savetxt

    def counting_savetxt(*args, **kwargs):
        nonlocal written_chunks
        written_chunks += 1
        written_row_counts.append(int(np.asarray(args[1]).shape[0]))
        return real_savetxt(*args, **kwargs)

    monkeypatch.setattr(reporting_module.np, "savetxt", counting_savetxt)

    def cancel_after_first_chunk() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks >= 3

    with pytest.raises(OperationCancelledError, match="导出已取消"):
        export_response_csv(
            destination,
            run,
            cancelled=cancel_after_first_chunk,
        )

    assert cancellation_checks >= 3
    assert written_chunks == 1
    assert written_row_counts == [4_096]
    assert written_row_counts[0] < rows
    assert destination.read_bytes() == b"existing-response"
    assert not list(tmp_path.glob(".response.csv.*.tmp"))


def test_bin_export_cancels_during_time_validation_before_writer(
    tmp_path,
    monkeypatch,
) -> None:
    """超长 BIN 的首轮时间轴扫描也必须可取消，且不得触碰 writer 或旧文件。"""

    destination = tmp_path / "large.bin"
    destination.write_bytes(b"existing-output")
    samples = 2 * io_module._EXPORT_CHUNK_SAMPLES + 8  # noqa: SLF001
    time_s = np.arange(samples, dtype=np.float64) * 1.0e-9
    values = np.linspace(-1.0, 1.0, samples, dtype=np.float64)
    checks = 0

    def cancel_on_second_validation_chunk() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError("cancelled BIN validation must not enter writer")

    monkeypatch.setattr(io_module, "write_keysight_bin", forbidden_writer)

    with pytest.raises(OperationCancelledError, match="导出已取消"):
        save_bin_timeseries(
            destination,
            time_s,
            values,
            cancelled=cancel_on_second_validation_chunk,
        )

    assert destination.read_bytes() == b"existing-output"
    assert not list(tmp_path.glob(".large.bin.*.tmp"))


def test_bundle_cancellation_before_commit_preserves_existing_batch(
    tmp_path,
    monkeypatch,
) -> None:
    """三份 staged 文件完成后取消，仍必须保留整批旧文件。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "result.csv")
    old_contents = {
        paths.output: b"old-output",
        paths.response_csv: b"old-response",
        paths.manifest: b"old-manifest",
    }
    for path, content in old_contents.items():
        path.write_bytes(content)

    armed = False
    real_write_manifest = reporting_module.write_manifest_atomic

    def arm_after_manifest(path, manifest):
        nonlocal armed
        result = real_write_manifest(path, manifest)
        armed = True
        return result

    monkeypatch.setattr(reporting_module, "write_manifest_atomic", arm_after_manifest)

    with pytest.raises(OperationCancelledError, match="导出已取消"):
        export_run_bundle(
            run,
            paths.output,
            allow_overwrite=True,
            cancelled=lambda: armed,
        )

    assert {path: path.read_bytes() for path in old_contents} == old_contents
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


def test_cancel_during_source_hash_is_not_reported_as_source_change(tmp_path) -> None:
    """用户取消与源文件校验失败必须保持不同的结果类型。"""

    run = _file_backed_demo_run(tmp_path / "inputs")

    with pytest.raises(OperationCancelledError, match="导出已取消"):
        verify_source_files_unchanged(run, cancelled=lambda: True)


def test_cancellation_after_bundle_commit_point_still_reports_success(
    tmp_path,
    monkeypatch,
) -> None:
    """最终提交已完成后到达的取消不能把已落盘批次误报成取消。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "result.csv")
    committed = False
    real_commit = reporting_module._commit_bundle_with_rollback

    def commit_then_cancel(
        staged,
        final,
        expected,
        expected_parent_identity,
        **commit_kwargs,
    ):
        nonlocal committed
        real_commit(
            staged,
            final,
            expected,
            expected_parent_identity,
            **commit_kwargs,
        )
        committed = True

    monkeypatch.setattr(reporting_module, "_commit_bundle_with_rollback", commit_then_cancel)

    returned = export_run_bundle(run, paths.output, cancelled=lambda: committed)

    assert returned == paths
    assert all(path.is_file() for path in paths.as_tuple())
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_generation_never_writes_or_publishes_replacement_staging_directory(
    tmp_path,
    monkeypatch,
) -> None:
    """生成首个文件后 staging 换主时，后续写入与提交都必须安全停止。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    paths = bundle_paths(tmp_path / "generation-takeover.csv")
    real_write_primary = reporting_module._write_primary_output
    owned_staging: list[Path] = []
    foreign_staging: list[Path] = []

    def write_then_replace_staging(path, current_run, **kwargs):
        real_write_primary(path, current_run, **kwargs)
        original = Path(path).parent
        moved = original.with_name(f"{original.name}-owned-moved")
        original.rename(moved)
        original.mkdir()
        for name in (
            paths.output.name,
            paths.response_csv.name,
            paths.manifest.name,
        ):
            (original / name).write_bytes(f"foreign:{name}".encode())
        owned_staging.append(moved)
        foreign_staging.append(original)

    monkeypatch.setattr(
        reporting_module,
        "_write_primary_output",
        write_then_replace_staging,
    )

    with (
        pytest.warns(reporting_module.BundleCleanupWarning, match="staging"),
        pytest.raises(reporting_module.DestinationChangedError, match="staging"),
    ):
        export_run_bundle(run, paths.output)

    assert {path.name: path.read_bytes() for path in foreign_staging[0].iterdir()} == {
        name: f"foreign:{name}".encode()
        for name in (paths.output.name, paths.response_csv.name, paths.manifest.name)
    }
    assert owned_staging[0].is_dir()
    assert not any(path.exists() for path in paths.as_tuple())


@_POSIX_OPEN_PATH_REPLACEMENT_ONLY
def test_publish_uses_pinned_parents_when_path_names_are_replaced(
    tmp_path,
    monkeypatch,
) -> None:
    """指纹完成后父目录变成 symlink，发布不得读取或写入替代目录。"""

    run = _file_backed_demo_run(tmp_path / "inputs")
    approved_parent = tmp_path / "approved"
    approved_parent.mkdir()
    paths = bundle_paths(approved_parent / "result.csv")
    moved_parent = tmp_path / "approved-owned-moved"
    victim = tmp_path / "victim"
    victim.mkdir()
    real_link = reporting_module.os.link
    injected = False
    foreign_sources: list[Path] = []

    def replace_parents_then_link(source, destination, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            staging = next(
                path
                for path in approved_parent.iterdir()
                if path.is_dir() and "response-lab-staging" in path.name
            )
            source_path = staging / Path(source).name
            relative_source = source_path.relative_to(approved_parent)
            approved_parent.rename(moved_parent)
            approved_parent.symlink_to(victim, target_is_directory=True)
            foreign_source = victim / relative_source
            foreign_source.parent.mkdir(parents=True)
            foreign_source.write_bytes(b"foreign-staged-source")
            foreign_sources.append(foreign_source)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(reporting_module.os, "link", replace_parents_then_link)

    try:
        with pytest.raises(
            reporting_module.DestinationChangedError,
            match="父目录|staged",
        ):
            export_run_bundle(run, paths.output)
        assert foreign_sources[0].read_bytes() == b"foreign-staged-source"
        assert not (victim / paths.output.name).exists()
        assert not (victim / paths.response_csv.name).exists()
        assert not (victim / paths.manifest.name).exists()
    finally:
        if approved_parent.is_symlink():
            approved_parent.unlink()
        if moved_parent.exists():
            moved_parent.rename(approved_parent)


def test_hardlink_publish_never_unlinks_replacement_staged_source(
    tmp_path,
    monkeypatch,
) -> None:
    """hardlink 提交后 source 同名换主时，外部文件必须保持原路径与字节。"""

    staging = tmp_path / "staging"
    staging.mkdir()
    source = staging / "source.csv"
    source.write_bytes(b"response-lab-owned")
    destination = tmp_path / "published.csv"
    expected = reporting_module._destination_fingerprint(source, include_digest=True)
    real_link = reporting_module.os.link
    injected = False

    def link_then_replace_source(link_source, link_destination, *args, **kwargs):
        nonlocal injected
        result = real_link(link_source, link_destination, *args, **kwargs)
        source.unlink()
        source.write_bytes(b"foreign-staged-source")
        injected = True
        return result

    monkeypatch.setattr(reporting_module.os, "link", link_then_replace_source)

    reporting_module._commit_staged_file(source, destination, expected)

    assert injected
    assert source.read_bytes() == b"foreign-staged-source"
    assert destination.read_bytes() == b"response-lab-owned"


def test_owned_file_cleanup_detects_replacement_before_final_check(
    tmp_path,
    monkeypatch,
) -> None:
    """最终核验前发生的 quarantine 换主必须停止清理并保留双方字节。"""

    owned = tmp_path / "owned.csv"
    owned.write_bytes(b"response-lab-owned")
    expected = reporting_module._destination_fingerprint(owned, include_digest=True)
    real_fingerprint = reporting_module._destination_fingerprint
    moved_owned: list[Path] = []
    foreign_quarantine: list[Path] = []

    def replace_quarantine_after_snapshot(path, *, include_digest=False, cancelled=None):
        fingerprint = real_fingerprint(
            path,
            include_digest=include_digest,
            cancelled=cancelled,
        )
        current = Path(path)
        if "response-lab-quarantine" in current.name and not moved_owned:
            moved = current.with_name(f"{current.name}-owned-moved")
            current.rename(moved)
            current.write_bytes(b"foreign-quarantine-owner")
            moved_owned.append(moved)
            foreign_quarantine.append(current)
        return fingerprint

    monkeypatch.setattr(
        reporting_module,
        "_destination_fingerprint",
        replace_quarantine_after_snapshot,
    )

    with pytest.raises(reporting_module.DestinationChangedError, match="换主|变化"):
        reporting_module._remove_owned_regular_file(
            owned,
            expected,
            mismatch_message="quarantine 路径换主",
        )

    assert foreign_quarantine[0].read_bytes() == b"foreign-quarantine-owner"
    assert moved_owned[0].read_bytes() == b"response-lab-owned"


def test_staging_cleanup_detects_replacement_before_enumeration(
    tmp_path,
    monkeypatch,
) -> None:
    """枚举前发生的 quarantine 目录换主必须停止递归清理。"""

    staging = tmp_path / ".result.response-lab-staging-owned"
    staging.mkdir()
    (staging / "owned.txt").write_bytes(b"owned-staging")
    expected = reporting_module._directory_identity(staging)
    real_has_identity = reporting_module._directory_has_identity
    moved_owned: list[Path] = []
    foreign_quarantine: list[Path] = []

    def replace_after_identity(path, identity):
        matches = real_has_identity(path, identity)
        current = Path(path)
        if matches and "staging-cleanup" in current.name and not moved_owned:
            moved = current.with_name(f"{current.name}-owned-moved")
            current.rename(moved)
            current.mkdir()
            (current / "valuable.txt").write_bytes(b"foreign-directory-owner")
            moved_owned.append(moved)
            foreign_quarantine.append(current)
        return matches

    monkeypatch.setattr(
        reporting_module,
        "_directory_has_identity",
        replace_after_identity,
    )

    with pytest.warns(reporting_module.BundleCleanupWarning, match="身份已变化"):
        reporting_module._cleanup_staging_directory(staging, expected)

    assert (foreign_quarantine[0] / "valuable.txt").read_bytes() == (b"foreign-directory-owner")
    assert (moved_owned[0] / "owned.txt").read_bytes() == b"owned-staging"
