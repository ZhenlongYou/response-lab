"""领域不变量、单文件原子写与整包回滚事务的回归测试。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from response_lab import io as io_module
from response_lab import reporting as reporting_module
from response_lab.app import build_demo_run
from response_lab.dsp import run_compensation
from response_lab.io import (
    load_csv_timeseries,
    save_bin_float32,
    save_csv_timeseries,
)
from response_lab.models import BinConfig, CompensationSettings, TimeSeries
from response_lab.reporting import (
    SourceVerificationError,
    build_manifest,
    bundle_paths,
    export_response_csv,
    export_run_bundle,
    sha256_file,
    verify_source_files_unchanged,
    write_manifest_atomic,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [("offset_bytes", 1.5), ("channels", True), ("channel_index", 0.25)],
)
def test_bin_config_rejects_non_integer_layout_fields(field, value) -> None:
    arguments = {"sample_rate_hz": 1.0e9, field: value}

    with pytest.raises(ValueError, match="必须是整数"):
        BinConfig(**arguments)


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
        save_bin_float32(destination, np.arange(8, dtype=np.float64))

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
    assert parsed["schema"] == "response-lab-manifest/v3"
    assert parsed["analysis"]["phase_detrend_slope_rad_per_hz"] == pytest.approx(
        2.0 * np.pi * 2.5e-9
    )
    assert parsed["analysis"]["estimated_relative_delay_s"] == pytest.approx(2.5e-9)
    assert (
        parsed["analysis"]["relative_delay_sign_convention"]
        == "positive_means_dut_later_than_reference"
    )
    assert parsed["output"]["sha256"] == sha256_file(output_path)
    assert parsed["output"]["size_bytes"] == output_path.stat().st_size
    assert parsed["settings"]["detrend_phase"] is True
    assert "remove_relative_delay" not in parsed["settings"]
    assert (
        parsed["application"]["method"]
        == "reflect_extend_czt_pulse_ratio_rfft_multiply_irfft_crop"
    )
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


@pytest.mark.parametrize("delay_samples", [3, -3])
def test_csv_and_json_preserve_signed_three_nanosecond_delay_contract(
    tmp_path,
    delay_samples: int,
) -> None:
    run = _signed_delay_run(delay_samples)
    response_path = export_response_csv(tmp_path / f"response-{delay_samples}.csv", run)
    parsed = json.loads(
        json.dumps(build_manifest(run, tmp_path / f"output-{delay_samples}.csv"))
    )

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


def test_bundle_export_writes_little_endian_float32_bin_with_sidecars(tmp_path) -> None:
    run = build_demo_run()

    paths = export_run_bundle(run, tmp_path / "result.bin")

    decoded = np.fromfile(paths.output, dtype="<f4")
    np.testing.assert_allclose(
        decoded,
        run.output_values[:, 0],
        rtol=2.0e-7,
        atol=1.0e-7,
    )
    assert paths.output.stat().st_size == run.output_values.shape[0] * 4
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

    real_commit_replace = reporting_module._commit_replace
    failure_injected = False

    def fail_second_staged_commit(source, destination):
        nonlocal failure_injected
        is_new_response_commit = (
            source.name == paths.response_csv.name
            and destination == paths.response_csv
            and "response-lab-staging" in source.parent.name
        )
        if is_new_response_commit and not failure_injected:
            failure_injected = True
            raise OSError("模拟整包提交失败")
        real_commit_replace(source, destination)

    monkeypatch.setattr(reporting_module, "_commit_replace", fail_second_staged_commit)
    with pytest.raises(OSError, match="整包提交失败"):
        export_run_bundle(run, paths.output)

    assert failure_injected is True
    assert {path: path.read_bytes() for path in old_contents} == old_contents
    assert not list(tmp_path.glob(".*.response-lab-staging-*"))
