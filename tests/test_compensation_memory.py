"""直接频域补偿的共享内存预检回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import response_lab.dsp as dsp_module
from response_lab.dsp import (
    CompensationMemoryPlan,
    ResponseAnalysisMemoryEstimate,
    _compensation_memory_estimate_from_shape,
    _parse_macos_vm_stat,
    _safe_compensation_memory_budget_bytes,
    _streaming_memory_estimate_from_shape,
    compare_pulses,
    preflight_compensation_shape,
    preflight_response_analysis_shape,
    run_compensation,
    suggest_frequency_settings,
)
from response_lab.models import CompensationSettings, TimeSeries


def _settings(*, band_low_hz: float, band_high_hz: float) -> CompensationSettings:
    return CompensationSettings(
        mode="magnitude",
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        analysis_points=4097,
    )


def _series(source_format: str) -> TimeSeries:
    sample_rate_hz = 2.0e9
    time_s = np.arange(32, dtype=np.float64) / sample_rate_hz
    return TimeSeries(
        time_s,
        np.ones(32, dtype=np.float64),
        sample_rate_hz,
        source_path=Path(f"target.{source_format}"),
        source_format=source_format,
    )


def test_memory_estimate_envelopes_two_independent_rss_measurements() -> None:
    """估算包住优化后实测，同时不得继续误报旧实现的巨大峰值。"""

    # 独立 macOS arm64 子进程（Python 3.12 / NumPy 2.5 / SciPy 1.18）：
    # N=1,000,000、50–200 MHz 时新增峰值 157,581,312 B。
    narrow = _compensation_memory_estimate_from_shape(
        target_samples=1_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=_settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
    )
    assert narrow.active_band_bins >= 225_000
    assert 157_581_312 <= narrow.estimated_peak_bytes < 768 * 1024**2

    # N=500,000、0–Nyquist 的独立进程峰值为 160,022,528 B。
    full_band = _compensation_memory_estimate_from_shape(
        target_samples=500_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=_settings(band_low_hz=0.0, band_high_hz=1.0e9),
    )
    assert full_band.active_band_bins >= 750_000
    assert 160_022_528 <= full_band.estimated_peak_bytes < 512 * 1024**2
    assert full_band.estimated_bytes_per_target_sample > narrow.estimated_bytes_per_target_sample


def test_exact_memory_estimate_envelopes_real_two_million_sample_fft_workspace() -> None:
    """真实 AG10 几何的 fresh-process RSS 必须被准入估算包住。"""

    settings = CompensationSettings(
        mode="both",
        band_low_hz=3.0339e9,
        band_high_hz=144.112e9,
        phase_fit_low_hz=12.1357e9,
        phase_fit_high_hz=136.527e9,
        analysis_points=16385,
    )
    estimate = _compensation_memory_estimate_from_shape(
        target_samples=2_000_000,
        target_channels=1,
        sample_rate_hz=3.4e12,
        reference_samples=12_800,
        dut_samples=12_800,
        settings=settings,
    )

    # Documents 中约 8 MiB AG10 夹具（SHA-256 0b9e6e...0c2b）在两次全新
    # macOS arm64 进程中的补偿新增高水位分别为 771,768,320 和 751,517,696 B。
    assert estimate.active_band_bins == 248_964
    assert estimate.estimated_peak_bytes >= 771_768_320


def test_large_analysis_grid_estimates_envelope_fresh_process_measurements() -> None:
    """公开 API 的百万点分析网格不能绕过 exact/streaming 内存门禁。"""

    common = dict(
        mode="magnitude",
        band_low_hz=0.0,
        band_high_hz=500.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        analysis_points=1_000_001,
        streaming_fft_samples=1024,
    )
    shape = dict(
        target_samples=1024,
        target_channels=1,
        sample_rate_hz=1.0e9,
        reference_samples=64,
        dut_samples=64,
    )
    exact = _compensation_memory_estimate_from_shape(
        **shape,
        settings=CompensationSettings(**common, application_strategy="exact"),
    )
    streaming = _streaming_memory_estimate_from_shape(
        **shape,
        settings=CompensationSettings(**common, application_strategy="streaming"),
    )

    # 两个独立 fresh process 的补偿阶段新增峰值分别为 418,480,128 B
    # 与 432,521,216 B；旧 96 B/点模型只估到约 194 MiB。
    assert exact.estimated_peak_bytes >= 418_480_128
    assert streaming.estimated_peak_bytes >= 432_521_216


def test_exact_memory_estimate_rejects_32_mib_geometry_on_five_gib_available() -> None:
    """临界大文件不能因遗漏原生 FFT 工作区而被 auto 误选为 exact。"""

    settings = CompensationSettings(
        mode="both",
        band_low_hz=3.0339e9,
        band_high_hz=144.112e9,
        phase_fit_low_hz=12.1357e9,
        phase_fit_high_hz=136.527e9,
        analysis_points=16385,
    )
    estimate = _compensation_memory_estimate_from_shape(
        target_samples=8_388_544,
        target_channels=1,
        sample_rate_hz=3.4e12,
        reference_samples=12_800,
        dut_samples=12_800,
        settings=settings,
    )
    budget = _safe_compensation_memory_budget_bytes(5 * 1024**3)

    assert budget == int(2.5 * 1024**3)
    assert estimate.estimated_peak_bytes > budget


def test_memory_estimate_counts_extra_channels_and_long_pulse_czt() -> None:
    """多通道 FFT 和长脉冲 CZT 都必须提高预算。"""

    settings = _settings(band_low_hz=50.0e6, band_high_hz=200.0e6)
    baseline = _compensation_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    multi_channel = _compensation_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=4,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    long_pulse = _compensation_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=300_000,
        dut_samples=300_000,
        settings=settings,
    )

    assert multi_channel.estimated_peak_bytes > baseline.estimated_peak_bytes
    assert long_pulse.czt_working_samples > baseline.czt_working_samples
    assert long_pulse.estimated_peak_bytes > baseline.estimated_peak_bytes


def test_thirty_million_full_band_switches_from_unsafe_exact_to_bounded_streaming() -> None:
    """30M 分块估算须包住归档峰值并落在 8 GiB 门禁内。"""

    settings = _settings(band_low_hz=0.0, band_high_hz=1.0e9)
    exact = _compensation_memory_estimate_from_shape(
        target_samples=30_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    streaming = _streaming_memory_estimate_from_shape(
        target_samples=30_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    multi_channel_streaming = _streaming_memory_estimate_from_shape(
        target_samples=30_000_000,
        target_channels=4,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )

    assert exact.estimated_peak_bytes > 8 * 1024**3
    assert streaming.estimated_peak_bytes < 8 * 1024**3
    evidence_folder = Path(__file__).resolve().parents[1] / "docs"
    calibration_paths = (
        evidence_folder / "30M_BIN分块补偿估算反例_2026-07-23.json",
    )
    observations = [
        json.loads(path.read_text(encoding="utf-8"))["measurement"]
        for path in calibration_paths
    ]
    # 带 clean HEAD、命令和验收字段的旧估算曾被同一夹具/算法的 fresh-worker
    # 高水位反例击穿；当前模型必须按 M 线性计入 FFT 后端余量并包住原始记录。
    for observation in observations:
        assert observation["streaming_memory_estimate"]["estimated_peak_bytes"] < (
            observation["post_load_compensation_peak_delta_bytes"]
        )
    assert streaming.estimated_peak_bytes >= max(
        observation["post_load_compensation_peak_delta_bytes"]
        for observation in observations
    )
    assert streaming.fft_samples == settings.streaming_fft_samples
    assert streaming.estimated_peak_bytes >= 30_000_000 * np.dtype(np.float32).itemsize
    assert multi_channel_streaming.estimated_peak_bytes > (
        streaming.estimated_peak_bytes
        + 3
        * settings.streaming_fft_samples
        * dsp_module._STREAMING_BACKEND_RESERVE_BYTES_PER_FFT_SAMPLE
    )


def test_streaming_estimate_structurally_counts_two_x_grid_refinement_audit() -> None:
    """N_FFT→2N_FFT 网格门禁的数组、CZT 与后端工作区必须进入准入估算。"""

    settings = CompensationSettings(
        mode="both",
        band_low_hz=0.0,
        band_high_hz=1.0e9,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=200.0e6,
        streaming_fft_samples=1024,
    )
    estimate = _streaming_memory_estimate_from_shape(
        target_samples=100_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )

    assert estimate.refinement_fft_samples == 2 * estimate.fft_samples
    assert estimate.refinement_rfft_bins == estimate.refinement_fft_samples // 2 + 1
    assert estimate.refinement_active_band_bins >= 2 * estimate.active_band_bins - 1
    assert estimate.refinement_czt_working_samples > estimate.czt_working_samples
    visible_refinement_arrays = (
        estimate.fft_samples * np.dtype(np.float64).itemsize
        + estimate.refinement_fft_samples * np.dtype(np.float64).itemsize
        + estimate.refinement_rfft_bins * np.dtype(np.complex128).itemsize
        + estimate.refinement_active_band_bins
        * np.dtype(np.complex128).itemsize
    )
    refined_czt_and_backend = (
        estimate.refinement_czt_working_samples
        * dsp_module._COMPENSATION_CZT_BYTES_PER_WORKING_SAMPLE
        + estimate.refinement_fft_samples
        * dsp_module._STREAMING_BACKEND_RESERVE_BYTES_PER_FFT_SAMPLE
    )
    assert estimate.refinement_audit_bytes >= (
        visible_refinement_arrays + refined_czt_and_backend
    )
    assert estimate.estimated_peak_bytes >= (
        100_000 * np.dtype(np.float32).itemsize
        + estimate.refinement_audit_bytes
        + dsp_module._COMPENSATION_FIXED_OVERHEAD_BYTES
    )


def test_dynamic_budget_keeps_half_available_and_512_mib_headroom() -> None:
    """动态预算同时受 8 GiB 上限、50% 比例和系统余量约束。"""

    gib = 1024**3
    assert _safe_compensation_memory_budget_bytes(16 * gib) == 8 * gib
    assert _safe_compensation_memory_budget_bytes(8 * gib) == 4 * gib
    assert _safe_compensation_memory_budget_bytes(2 * gib) == 1 * gib
    assert _safe_compensation_memory_budget_bytes(768 * 1024**2) == 256 * 1024**2


def test_macos_available_memory_parser_counts_only_distinct_reclaimable_pages() -> None:
    """inactive 已含 purgeable，解析器不得因重复相加而抬高安全预算。"""

    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             999.
Pages inactive:                           200.
Pages speculative:                         50.
Pages purgeable:                           500.
"""

    assert _parse_macos_vm_stat(output) == (100 + 200 + 50) * 16384


def test_public_response_analysis_preflight_rejects_before_any_pulse_fft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """比较与自动建议的公共门禁必须早于首个脉冲 FFT。"""

    pulse = _series("memory")
    settings = _settings(band_low_hz=50.0e6, band_high_hz=200.0e6)
    monkeypatch.setattr(
        dsp_module,
        "_system_available_memory_bytes",
        lambda: 96 * 1024**2,
    )

    def forbidden_fft(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("response-analysis preflight must run before pulse FFT")

    monkeypatch.setattr(dsp_module, "_pulse_spectrum", forbidden_fft)

    with pytest.raises(MemoryError, match="响应分析内存预检"):
        compare_pulses(pulse, pulse, settings)
    with pytest.raises(MemoryError, match="响应分析内存预检"):
        suggest_frequency_settings(pulse, pulse, settings)


def test_response_analysis_shape_preflight_returns_auditable_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """公开形状门禁应报告 FFT 几何、预算和保守峰值。"""

    monkeypatch.setattr(
        dsp_module,
        "_system_available_memory_bytes",
        lambda: 2 * 1024**3,
    )
    estimate = preflight_response_analysis_shape(
        reference_samples=1024,
        dut_samples=2048,
        settings=_settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
    )

    assert isinstance(estimate, ResponseAnalysisMemoryEstimate)
    assert estimate.reference_samples == 1024
    assert estimate.dut_samples == 2048
    assert estimate.pulse_fft_samples >= 4096
    assert estimate.analysis_points == 4097
    assert 0 < estimate.estimated_peak_bytes <= estimate.budget_bytes


def test_shape_preflight_reserves_not_yet_loaded_input_before_auto_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BIN 载荷尚未映射时，门禁须从同一内存快照预留其常驻量。"""

    mib = 1024**2
    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=50.0e6,
        band_high_hz=200.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=100.0e6,
        analysis_points=4097,
        application_strategy="auto",
        streaming_fft_samples=1024,
    )
    shape = dict(
        target_samples=1_000_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=settings,
    )
    monkeypatch.setattr(
        dsp_module,
        "_system_available_memory_bytes",
        lambda: 1200 * mib,
    )

    before_load = preflight_compensation_shape(**shape)
    with_payload_reserved = preflight_compensation_shape(
        **shape,
        anticipated_input_resident_bytes=400 * mib,
    )

    assert before_load.strategy == "exact"
    assert before_load.budget_bytes == 600 * mib
    assert with_payload_reserved.strategy == "streaming"
    assert with_payload_reserved.available_memory_bytes == 1200 * mib
    assert with_payload_reserved.anticipated_input_resident_bytes == 400 * mib
    assert with_payload_reserved.budget_bytes == 288 * mib
    assert with_payload_reserved.exact_estimate is not None
    assert with_payload_reserved.streaming_estimate is not None
    assert (
        with_payload_reserved.exact_estimate.estimated_peak_bytes
        > with_payload_reserved.budget_bytes
        >= with_payload_reserved.streaming_estimate.estimated_peak_bytes
    )


def test_shape_preflight_subtracts_resident_bytes_from_unknown_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """探测失败时也必须预留待载入输入，不能重新获得完整回退预算。"""

    mib = 1024**2
    monkeypatch.setattr(dsp_module, "_system_available_memory_bytes", lambda: None)
    plan = preflight_compensation_shape(
        target_samples=100_000,
        target_channels=1,
        sample_rate_hz=2.0e9,
        reference_samples=1024,
        dut_samples=1024,
        settings=CompensationSettings(
            mode="magnitude",
            band_low_hz=50.0e6,
            band_high_hz=200.0e6,
            phase_fit_low_hz=20.0e6,
            phase_fit_high_hz=100.0e6,
            analysis_points=4097,
            application_strategy="auto",
            streaming_fft_samples=1024,
        ),
        anticipated_input_resident_bytes=700 * mib,
    )

    assert plan.available_memory_bytes is None
    assert plan.budget_bytes == 68 * mib
    assert plan.strategy == "streaming"


@pytest.mark.parametrize("resident_bytes", [-1, True, 1.5])
def test_shape_preflight_rejects_invalid_anticipated_resident_bytes(
    resident_bytes: object,
) -> None:
    """预载荷预算只接受明确的非负整数字节数，避免静默截断。"""

    with pytest.raises(ValueError, match="预计输入常驻内存.*非负整数"):
        preflight_compensation_shape(
            target_samples=32,
            target_channels=1,
            sample_rate_hz=2.0e9,
            reference_samples=32,
            dut_samples=32,
            settings=_settings(band_low_hz=1.0e6, band_high_hz=10.0e6),
            anticipated_input_resident_bytes=resident_bytes,  # type: ignore[arg-type]
        )


def test_run_compensation_uses_public_shape_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实际运行与载入前检查必须共用同一个策略选择入口。"""

    base_pulse = _series("memory")
    pulse_values = np.zeros(base_pulse.samples, dtype=np.float64)
    pulse_values[0] = 1.0
    pulse = TimeSeries(
        base_pulse.time_s,
        pulse_values,
        base_pulse.sample_rate_hz,
        source_path=base_pulse.source_path,
        source_format=base_pulse.source_format,
    )
    target = _series("bin")
    observed: list[dict[str, object]] = []
    real_preflight = preflight_compensation_shape

    def tracked_preflight(**kwargs: object) -> CompensationMemoryPlan:
        observed.append(dict(kwargs))
        return real_preflight(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dsp_module, "preflight_compensation_shape", tracked_preflight)

    run_compensation(
        pulse,
        pulse,
        target,
        _settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
    )

    assert len(observed) == 1
    assert observed[0]["target_samples"] == target.samples
    assert observed[0]["target_channels"] == target.channels
    assert observed[0]["sample_rate_hz"] == target.sample_rate_hz
    assert observed[0]["reference_samples"] == pulse.samples
    assert observed[0]["dut_samples"] == pulse.samples
    assert observed[0]["anticipated_input_resident_bytes"] == 0


@pytest.mark.parametrize("source_format", ["csv", "bin"])
def test_run_compensation_rejects_csv_and_bin_before_analysis_allocation(
    monkeypatch: pytest.MonkeyPatch,
    source_format: str,
) -> None:
    """共享入口应在 compare/CZT/FFT 之前对两种文件来源执行同一门禁。"""

    pulse = _series("memory")
    target = _series(source_format)
    monkeypatch.setattr(dsp_module, "_system_available_memory_bytes", lambda: 96 * 1024**2)

    def forbidden_compare(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("memory preflight must run before compare_pulses")

    monkeypatch.setattr(dsp_module, "compare_pulses", forbidden_compare)

    with pytest.raises(MemoryError, match="补偿内存预检.*CSV/BIN 共用"):
        run_compensation(
            pulse,
            pulse,
            target,
            _settings(band_low_hz=50.0e6, band_high_hz=200.0e6),
        )


def test_run_adopts_owned_application_output_without_a_second_full_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FFT 已返回自有紧凑数组时，运行模型应校验后直接接管并设为只读。"""

    pulse = _series("memory")
    target = _series("bin")
    owned_output = np.arange(32, dtype=np.float64).reshape(32, 1).copy()

    def return_owned_output(*_args: object, **_kwargs: object) -> np.ndarray:
        return owned_output

    monkeypatch.setattr(dsp_module, "apply_frequency_correction", return_owned_output)

    run = run_compensation(
        pulse,
        pulse,
        target,
        _settings(band_low_hz=1.0e6, band_high_hz=10.0e6),
    )

    assert np.shares_memory(run.output_values, owned_output)
    assert not run.output_values.flags.writeable
