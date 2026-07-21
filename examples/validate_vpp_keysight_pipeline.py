"""生成并验证 ResponseLab 的 Vpp、CSV 与 Keysight AG10 全链路数据。

运行示例（输出目录必须尚不存在）：

    python3 examples/validate_vpp_keysight_pipeline.py \
        --output-dir /tmp/responselab-vpp-validation

AG10 写入器只使用 Python ``struct`` 和公开格式字段，不调用生产
``write_keysight_bin``，因此它可以作为加载器的独立格式 oracle。大文件采用稀疏
零 payload：逻辑长度真实、格式完整，但不会为了元数据与内存测试占满磁盘。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

# 允许从项目根目录或直接在 PyCharm 中运行本文件，无需额外设置 PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from response_lab.dsp import run_compensation  # noqa: E402
from response_lab.io import (  # noqa: E402
    load_bin_timeseries,
    load_csv_timeseries,
    save_bin_timeseries,
)
from response_lab.keysight_bin import (  # noqa: E402
    inspect_keysight_bin,
    load_keysight_waveform,
)
from response_lab.models import CompensationSettings  # noqa: E402
from response_lab.vpp_analysis import (  # noqa: E402
    VppAnalysisSettings,
    measure_candidate,
    prepare_vpp_analysis,
)

# 以下默认值集中在文件顶部，便于不熟悉命令行的用户直接在 PyCharm 中调整。
DEFAULT_MODERATE_SAMPLES = 262_144
DEFAULT_LARGE_SAMPLES = 33_554_432
SAMPLE_RATE_HZ = 8.0e9
TARGET_X_ORIGIN_S = -8.0e-9
PULSE_SAMPLES = 2_048
PULSE_PEAK_INDEX = 768
SAMPLES_PER_UI = 8
CURSOR_UI = 8
PATTERN_SYMBOLS = 257
PATTERN_SEED = 0xA5C31E27
# CSV 时间戳在远离零点时会受 float64 相减舍入影响；1e-10 相对容差在
# 8 GHz 下仅为 0.8 Hz，同时仍由逐点时间轴误差检查约束累计相位误差。
SAMPLE_RATE_COMPARISON_RTOL = 1.0e-10

# Keysight Infiniium AG10 的固定头布局：12 B 文件头、140 B 波形头、12 B 数据头。
_FILE_HEADER = struct.Struct("<2s2sii")
_WAVEFORM_HEADER = struct.Struct("<iiiiifdddii16s16s24s16sdI")
_DATA_HEADER = struct.Struct("<ihhi")
_INT32_MAX = 2_147_483_647
_WRITE_CHUNK_SAMPLES = 1_048_576

T = TypeVar("T")


def _fixed_ascii(text: str, size: int) -> bytes:
    """把 ASCII 元数据补零到 Keysight 固定字段宽度，拒绝静默截断。"""

    encoded = text.encode("ascii")
    if len(encoded) > size:
        raise ValueError(f"Keysight 文本字段超过 {size} 字节：{text!r}")
    return encoded.ljust(size, b"\0")


def _ag10_headers(
    points: int,
    *,
    sample_rate_hz: float,
    x_origin_s: float,
    label: str,
) -> tuple[bytes, int]:
    """独立构造一个 Normal/float32/Seconds/Volts 波形的 AG10 头。"""

    payload_bytes = points * 4
    total_bytes = _FILE_HEADER.size + _WAVEFORM_HEADER.size + _DATA_HEADER.size + payload_bytes
    if points < 1 or payload_bytes > _INT32_MAX or total_bytes > _INT32_MAX:
        raise ValueError("AG10 点数或文件长度超过有符号 int32 格式上限")
    x_increment_s = 1.0 / sample_rate_hz
    file_header = _FILE_HEADER.pack(b"AG", b"10", total_bytes, 1)
    waveform_header = _WAVEFORM_HEADER.pack(
        _WAVEFORM_HEADER.size,
        1,
        1,
        points,
        0,
        points * x_increment_s,
        x_origin_s,
        x_increment_s,
        x_origin_s,
        2,
        1,
        _fixed_ascii("22 JUL 2026", 16),
        _fixed_ascii("12:00:00", 16),
        _fixed_ascii("RESP-LAB-VALIDATION", 24),
        _fixed_ascii(label, 16),
        0.0,
        0,
    )
    data_header = _DATA_HEADER.pack(_DATA_HEADER.size, 1, 4, payload_bytes)
    return file_header + waveform_header + data_header, total_bytes


def _write_dense_ag10(
    path: Path,
    values_v: np.ndarray,
    *,
    sample_rate_hz: float,
    x_origin_s: float,
    label: str,
) -> Path:
    """用独立 struct 头和分块 little-endian float32 payload 写普通 AG10。"""

    values = np.asarray(values_v)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("AG10 oracle 只接受一维有限电压数组")
    headers, total_bytes = _ag10_headers(
        int(values.size),
        sample_rate_hz=sample_rate_hz,
        x_origin_s=x_origin_s,
        label=label,
    )
    with path.open("wb") as stream:
        stream.write(headers)
        for start in range(0, values.size, _WRITE_CHUNK_SAMPLES):
            encoded = np.ascontiguousarray(
                values[start : start + _WRITE_CHUNK_SAMPLES],
                dtype="<f4",
            )
            stream.write(memoryview(encoded).cast("B"))
        stream.flush()
        os.fsync(stream.fileno())
    if path.stat().st_size != total_bytes:
        raise RuntimeError("独立 AG10 writer 的实际文件长度与头部声明不一致")
    return path


def _write_sparse_zero_ag10(
    path: Path,
    points: int,
    *,
    sample_rate_hz: float,
    x_origin_s: float,
) -> Path:
    """写格式完整的稀疏零波形，用于低内存的大文件元数据与 mmap 验证。"""

    headers, total_bytes = _ag10_headers(
        points,
        sample_rate_hz=sample_rate_hz,
        x_origin_s=x_origin_s,
        label="Stress Zero",
    )
    with path.open("wb") as stream:
        stream.write(headers)
        stream.seek(total_bytes - 1)
        stream.write(b"\0")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _write_two_column_csv(path: Path, time_s: np.ndarray, values_v: np.ndarray) -> Path:
    """分块写无表头 time(s),voltage(V) CSV，避免大矩阵临时副本。"""

    if time_s.shape != values_v.shape:
        raise ValueError("CSV 时间和值必须等长")
    with path.open("w", encoding="utf-8", newline="") as stream:
        for start in range(0, time_s.size, _WRITE_CHUNK_SAMPLES):
            stop = min(start + _WRITE_CHUNK_SAMPLES, time_s.size)
            block = np.column_stack((time_s[start:stop], values_v[start:stop]))
            np.savetxt(stream, block, delimiter=",", fmt=("%.17g", "%.17g"))
    return path


def _write_pattern(path: Path) -> tuple[Path, np.ndarray]:
    """生成固定 seed 的工程 PAM4 symbol-code 文件；它不冒充标准 PRBS。"""

    state = PATTERN_SEED
    codes = np.empty(PATTERN_SYMBOLS, dtype=np.uint8)
    for index in range(PATTERN_SYMBOLS):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        codes[index] = state & 0x3
    np.savetxt(path, codes, fmt="%d")
    return path, codes


def _peak_rss_bytes() -> int:
    """返回进程峰值 RSS；macOS 单位为 byte，其余常见 Unix 为 KiB。"""

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _measure(action: Callable[[], T]) -> tuple[T, dict[str, float | int]]:
    """记录一次操作的墙钟时间与进程峰值 RSS 增量。"""

    gc.collect()
    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    result = action()
    elapsed_s = time.perf_counter() - started
    rss_after = _peak_rss_bytes()
    return result, {
        "elapsed_s": elapsed_s,
        "peak_rss_before_bytes": rss_before,
        "peak_rss_after_bytes": rss_after,
        "peak_rss_delta_bytes": max(0, rss_after - rss_before),
    }


def _sha256(path: Path) -> str:
    """分块计算普通验证文件哈希；不会用于逻辑很大的稀疏 stress 文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    """Convert NumPy scalar evidence without silently stringifying unknown objects."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不支持写入 JSON 的验证结果类型：{type(value).__name__}")


def _tone_amplitudes(
    values_v: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """用复指数投影测量确定性整周期音调的峰值幅度，单位 V。"""

    amplitudes = np.empty(frequencies_hz.size, dtype=np.float64)
    for index, frequency_hz in enumerate(frequencies_hz):
        basis = np.exp(2j * np.pi * frequency_hz * time_s)
        amplitudes[index] = 2.0 * abs(np.vdot(basis, values_v)) / values_v.size
    return amplitudes


def _generate_physical_fixture(samples: int) -> dict[str, np.ndarray]:
    """构造已知五抽头通道、拟合脉冲和闭式多音目标波形。"""

    pulse_index = np.arange(PULSE_SAMPLES, dtype=np.float64)
    pulse_time_s = (pulse_index - PULSE_PEAK_INDEX) / SAMPLE_RATE_HZ
    reference_pulse_v = np.exp(
        -0.5 * ((pulse_index - PULSE_PEAK_INDEX) / 1.75) ** 2
    )
    channel_taps = np.array([0.08, 0.20, 0.44, 0.20, 0.08], dtype=np.float64)
    dut_pulse_v = np.convolve(reference_pulse_v, channel_taps, mode="same")

    sample_index = np.arange(samples, dtype=np.float64)
    target_time_s = TARGET_X_ORIGIN_S + sample_index / SAMPLE_RATE_HZ
    nominal_frequency_hz = np.array([0.20e9, 0.70e9, 1.15e9, 1.45e9])
    tone_bins = np.rint(nominal_frequency_hz * samples / SAMPLE_RATE_HZ).astype(np.int64)
    frequency_hz = tone_bins * SAMPLE_RATE_HZ / samples
    ideal_amplitude_v = np.array([0.28, 0.42, 0.30, 0.20])
    phase_rad = np.array([0.10, 0.30, -0.60, 0.90])
    normalized_angular_frequency = 2.0 * np.pi * frequency_hz / SAMPLE_RATE_HZ
    channel_gain = (
        0.44
        + 0.40 * np.cos(normalized_angular_frequency)
        + 0.16 * np.cos(2.0 * normalized_angular_frequency)
    )
    components = np.array(
        [
            amplitude_v * np.cos(2.0 * np.pi * tone_hz * target_time_s + tone_phase)
            for amplitude_v, tone_hz, tone_phase in zip(
                ideal_amplitude_v,
                frequency_hz,
                phase_rad,
                strict=True,
            )
        ]
    )
    target_v = np.sum(channel_gain[:, None] * components, axis=0)
    expected_compensated_v = target_v + np.sum(
        (1.0 - channel_gain[1:])[:, None] * components[1:],
        axis=0,
    )
    return {
        "pulse_time_s": pulse_time_s,
        "reference_pulse_v": reference_pulse_v,
        "dut_pulse_v": dut_pulse_v,
        "target_time_s": target_time_s,
        "target_v": target_v.astype(np.float32).astype(np.float64),
        "expected_compensated_v": expected_compensated_v,
        "frequency_hz": frequency_hz,
        "ideal_amplitude_v": ideal_amplitude_v,
        "channel_gain": channel_gain,
        "channel_taps": channel_taps,
    }


def _vpp_matrix(
    reference_pulse: Any,
    dut_pulse: Any,
    pattern_path: Path,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """运行 LFP/RMS × 内置 PRBS13Q/外部码型的四种真实数值路径。"""

    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for source in ("builtin_prbs13q_gray", "file"):
        source_results: dict[str, dict[str, float | int]] = {}
        for method in ("lfp", "frequency_rms_error"):
            settings = VppAnalysisSettings(
                method=method,
                pattern_source=source,
                samples_per_ui=SAMPLES_PER_UI,
                pre_cursor_ui=CURSOR_UI,
                post_cursor_ui=CURSOR_UI,
                pattern_path=None if source == "builtin_prbs13q_gray" else pattern_path,
                file_value_kind="symbol_codes",
            )
            cache, timing = _measure(
                lambda settings=settings: prepare_vpp_analysis(
                    reference_pulse,
                    dut_pulse,
                    settings,
                )
            )
            # 独立使用已知五抽头通道的闭式频响，验证候选指标确实回到参考模型。
            angular_frequency = 2.0 * np.pi * cache.frequency_hz / cache.sample_rate_hz
            channel_response = (
                0.44
                + 0.40 * np.cos(angular_frequency)
                + 0.16 * np.cos(2.0 * angular_frequency)
            )
            measurement, correction_timing = _measure(
                lambda cache=cache, channel_response=channel_response: measure_candidate(
                    cache,
                    1.0 / channel_response,
                )
            )
            before_gap_v = abs(cache.dut_metric_v - cache.reference_metric_v)
            after_gap_v = abs(measurement.value_v - cache.reference_metric_v)
            source_results[method] = {
                "period_samples": cache.period_samples,
                "sample_rate_hz": cache.sample_rate_hz,
                "symbol_rate_hz": cache.symbol_rate_hz,
                "ui_duration_s": cache.ui_duration_s,
                "reference_metric_v": cache.reference_metric_v,
                "dut_metric_v": cache.dut_metric_v,
                "corrected_metric_v": measurement.value_v,
                "before_reference_gap_v": before_gap_v,
                "after_reference_gap_v": after_gap_v,
                "gap_improvement_factor": before_gap_v / max(
                    after_gap_v,
                    np.finfo(np.float64).tiny,
                ),
                "correction_elapsed_s": correction_timing["elapsed_s"],
                **timing,
            }
        results[source] = source_results
    return results


def _git_snapshot() -> dict[str, Any]:
    """记录当前提交和脏文件列表；Git 不可用时保留明确的空状态。"""

    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_entries": None}
    return {"commit": commit, "dirty_entries": dirty}


def _markdown_report(report: dict[str, Any]) -> str:
    """把机器可读 JSON 的关键结果压缩为便于人工复核的 Markdown。"""

    validation = report["validation"]
    performance = report["performance"]
    lines = [
        "# ResponseLab Vpp / Keysight 验证报告",
        "",
        f"- 总体结果：**{'PASS' if report['passed'] else 'FAIL'}**",
        f"- CSV/BIN 最大幅值差：{validation['csv_bin_max_abs_value_difference_v']:.6g} V",
        f"- CSV/BIN 最大时间差：{validation['csv_bin_max_abs_time_difference_s']:.6g} s",
        f"- CSV/BIN 采样率差：{validation['csv_bin_sample_rate_difference_hz']:.6g} Hz "
        f"（相对 {validation['csv_bin_sample_rate_relative_difference']:.3g}）",
        f"- 补偿后内区 RMS 误差改善倍数：{validation['interior_rms_improvement_factor']:.3f}×",
        f"- 带外控制音变化：{validation['outside_control_tone_change_v']:.6g} V",
        f"- 大 BIN 逻辑大小：{report['large_bin']['logical_size_bytes'] / 2**20:.1f} MiB",
        f"- 大 BIN 实际占盘：{report['large_bin']['allocated_size_bytes'] / 2**20:.1f} MiB",
        f"- 大 BIN mmap：{report['large_bin']['low_level_is_memmap']}",
        f"- 高层安全拒绝：{report['large_bin']['high_level_rejected_before_payload']}",
        "",
        "## 主要耗时",
        "",
    ]
    for name, timing in performance.items():
        lines.append(
            f"- {name}: {timing['elapsed_s']:.4f} s，峰值 RSS 增量 "
            f"{timing['peak_rss_delta_bytes'] / 2**20:.1f} MiB"
        )
    lines.extend(
        [
            "",
            "> RSS 为进程峰值统计；若前序步骤已经达到更高峰值，后续增量可能为 0。",
            "",
        ]
    )
    return "\n".join(lines)


def run_validation(
    output_dir: Path,
    *,
    moderate_samples: int,
    large_samples: int,
) -> dict[str, Any]:
    """生成全部数据、运行真实加载/补偿/Vpp 路径并写 JSON/Markdown 报告。"""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录必须尚不存在，避免覆盖已有数据：{output_dir}")
    if moderate_samples < 8_192:
        raise ValueError("moderate-samples 必须至少为 8192")
    if large_samples <= moderate_samples:
        raise ValueError("large-samples 必须严格大于 moderate-samples")
    _ag10_headers(
        large_samples,
        sample_rate_hz=SAMPLE_RATE_HZ,
        x_origin_s=TARGET_X_ORIGIN_S,
        label="Stress Zero",
    )
    output_dir.mkdir(parents=True)

    fixture, generation_timing = _measure(lambda: _generate_physical_fixture(moderate_samples))
    reference_path = output_dir / "reference_fitted_pulse.csv"
    dut_path = output_dir / "dut_fitted_pulse.csv"
    pattern_path = output_dir / "user_ideal_pattern_codes.csv"
    target_csv_path = output_dir / "target_signal.csv"
    target_bin_path = output_dir / "target_signal_keysight_ag10.bin"
    expected_path = output_dir / "expected_compensated_target.csv"
    output_path = output_dir / "compensated_output.csv"
    output_bin_path = output_dir / "compensated_output_keysight_ag10.bin"
    large_bin_path = output_dir / "large_sparse_keysight_ag10.bin"

    _, pulse_write_timing = _measure(
        lambda: (
            _write_two_column_csv(
                reference_path,
                fixture["pulse_time_s"],
                fixture["reference_pulse_v"],
            ),
            _write_two_column_csv(
                dut_path,
                fixture["pulse_time_s"],
                fixture["dut_pulse_v"],
            ),
        )
    )
    (_, pattern_codes), pattern_write_timing = _measure(lambda: _write_pattern(pattern_path))
    _, target_csv_write_timing = _measure(
        lambda: _write_two_column_csv(
            target_csv_path,
            fixture["target_time_s"],
            fixture["target_v"],
        )
    )
    _, target_bin_write_timing = _measure(
        lambda: _write_dense_ag10(
            target_bin_path,
            fixture["target_v"],
            sample_rate_hz=SAMPLE_RATE_HZ,
            x_origin_s=TARGET_X_ORIGIN_S,
            label="Target DUT",
        )
    )

    reference_pulse = load_csv_timeseries(reference_path, time_unit="s")
    dut_pulse = load_csv_timeseries(dut_path, time_unit="s")
    target_csv, csv_load_timing = _measure(
        lambda: load_csv_timeseries(target_csv_path, time_unit="s")
    )
    target_bin, bin_load_timing = _measure(lambda: load_bin_timeseries(target_bin_path))
    value_difference_v = float(np.max(np.abs(target_csv.values - target_bin.values)))
    time_difference_s = float(np.max(np.abs(target_csv.time_s - target_bin.time_s)))
    sample_rate_difference_hz = float(
        abs(target_csv.sample_rate_hz - target_bin.sample_rate_hz)
    )
    sample_rate_relative_difference = float(
        sample_rate_difference_hz / target_bin.sample_rate_hz
    )

    settings = CompensationSettings(
        mode="magnitude",
        band_low_hz=0.40e9,
        band_high_hz=1.60e9,
        phase_fit_low_hz=0.10e9,
        phase_fit_high_hz=0.30e9,
        detrend_phase=True,
        analysis_points=8_193,
    )
    csv_run, csv_compensation_timing = _measure(
        lambda: run_compensation(reference_pulse, dut_pulse, target_csv, settings)
    )
    bin_run, bin_compensation_timing = _measure(
        lambda: run_compensation(reference_pulse, dut_pulse, target_bin, settings)
    )
    csv_output_v = csv_run.output_values[:, 0]
    bin_output_v = bin_run.output_values[:, 0]
    output_difference_v = float(np.max(np.abs(csv_output_v - bin_output_v)))

    guard_samples = max(128, moderate_samples // 50)
    interior = slice(guard_samples, -guard_samples)
    before_error_vrms = float(
        np.sqrt(
            np.mean(
                np.square(
                    fixture["target_v"][interior]
                    - fixture["expected_compensated_v"][interior]
                )
            )
        )
    )
    after_error_vrms = float(
        np.sqrt(
            np.mean(
                np.square(
                    csv_output_v[interior]
                    - fixture["expected_compensated_v"][interior]
                )
            )
        )
    )
    improvement_factor = before_error_vrms / after_error_vrms
    input_tone_v = _tone_amplitudes(
        fixture["target_v"],
        fixture["target_time_s"],
        fixture["frequency_hz"],
    )
    output_tone_v = _tone_amplitudes(
        csv_output_v,
        fixture["target_time_s"],
        fixture["frequency_hz"],
    )
    outside_control_change_v = float(abs(output_tone_v[0] - input_tone_v[0]))
    maximum_in_band_tone_error_v = float(
        np.max(np.abs(output_tone_v[1:] - fixture["ideal_amplitude_v"][1:]))
    )
    _write_two_column_csv(
        expected_path,
        fixture["target_time_s"],
        fixture["expected_compensated_v"],
    )
    _write_two_column_csv(output_path, fixture["target_time_s"], csv_output_v)
    _, output_bin_write_timing = _measure(
        lambda: save_bin_timeseries(
            output_bin_path,
            fixture["target_time_s"],
            csv_output_v,
            label="Compensated",
        )
    )
    output_bin_reloaded = load_bin_timeseries(output_bin_path)
    output_bin_roundtrip_difference_v = float(
        np.max(np.abs(output_bin_reloaded.values[:, 0] - csv_output_v))
    )

    vpp_results = _vpp_matrix(reference_pulse, dut_pulse, pattern_path)

    _, large_write_timing = _measure(
        lambda: _write_sparse_zero_ag10(
            large_bin_path,
            large_samples,
            sample_rate_hz=SAMPLE_RATE_HZ,
            x_origin_s=TARGET_X_ORIGIN_S,
        )
    )
    large_info, large_inspect_timing = _measure(lambda: inspect_keysight_bin(large_bin_path))
    large_waveform, large_mmap_timing = _measure(
        lambda: load_keysight_waveform(large_bin_path)
    )
    low_level_is_memmap = isinstance(large_waveform.values, np.memmap)
    sparse_edge_values_are_zero = bool(
        large_waveform.values[0] == 0.0 and large_waveform.values[-1] == 0.0
    )
    del large_waveform
    gc.collect()

    rejection_message = ""
    rejection_started = time.perf_counter()
    rejection_rss_before = _peak_rss_bytes()
    try:
        # 不传固定点数门限；应由与 CSV/DSP 共用的动态内存预算在 payload 前拒绝。
        load_bin_timeseries(large_bin_path)
    except MemoryError as error:
        rejection_message = str(error)
    rejection_timing = {
        "elapsed_s": time.perf_counter() - rejection_started,
        "peak_rss_before_bytes": rejection_rss_before,
        "peak_rss_after_bytes": _peak_rss_bytes(),
        "peak_rss_delta_bytes": max(0, _peak_rss_bytes() - rejection_rss_before),
    }
    high_level_rejected = bool(rejection_message)

    logical_size_bytes = large_bin_path.stat().st_size
    allocated_size_bytes = getattr(large_bin_path.stat(), "st_blocks", 0) * 512
    time_tolerance_s = 64.0 * np.finfo(np.float64).eps * max(
        abs(float(target_csv.time_s[-1])),
        1.0 / SAMPLE_RATE_HZ,
    )
    checks = {
        "csv_bin_sample_count_equal": target_csv.samples == target_bin.samples,
        "csv_bin_sample_rate_equal": bool(
            np.isclose(
                target_csv.sample_rate_hz,
                target_bin.sample_rate_hz,
                rtol=SAMPLE_RATE_COMPARISON_RTOL,
                atol=0.0,
            )
        ),
        "csv_bin_values_equal": value_difference_v <= 1.0e-12,
        "csv_bin_time_axis_equal": time_difference_s <= time_tolerance_s,
        "csv_bin_compensation_outputs_equal": output_difference_v <= 1.0e-12,
        "visible_rms_improvement": improvement_factor >= 50.0,
        "in_band_tones_recovered": maximum_in_band_tone_error_v <= 1.0e-3,
        "outside_control_tone_unchanged": outside_control_change_v <= 5.0e-4,
        "large_header_points_match": large_info.waveforms[0].points == large_samples,
        "large_low_level_load_is_memmap": low_level_is_memmap,
        "large_sparse_edge_values_are_zero": sparse_edge_values_are_zero,
        "large_high_level_load_rejected": high_level_rejected,
        "compensated_bin_roundtrip": output_bin_roundtrip_difference_v <= 1.0e-6,
        "external_pattern_has_all_four_codes": np.unique(pattern_codes).size == 4,
    }
    for source_results in vpp_results.values():
        checks[f"{len(checks)}_lfp_is_finite"] = bool(
            np.isfinite(source_results["lfp"]["dut_metric_v"])
        )
        checks[f"{len(checks)}_rms_is_positive"] = bool(
            source_results["frequency_rms_error"]["dut_metric_v"] > 0.0
        )
        checks[f"{len(checks)}_lfp_candidate_recovers_reference"] = bool(
            source_results["lfp"]["after_reference_gap_v"] <= 1.0e-10
        )
        checks[f"{len(checks)}_rms_candidate_recovers_reference"] = bool(
            source_results["frequency_rms_error"]["after_reference_gap_v"]
            <= 1.0e-10
        )

    artifact_paths = [
        reference_path,
        dut_path,
        pattern_path,
        target_csv_path,
        target_bin_path,
        expected_path,
        output_path,
        output_bin_path,
    ]
    performance = {
        "fixture_generation": generation_timing,
        "pulse_csv_write": pulse_write_timing,
        "pattern_write": pattern_write_timing,
        "target_csv_write": target_csv_write_timing,
        "target_bin_write": target_bin_write_timing,
        "compensated_bin_write": output_bin_write_timing,
        "target_csv_load": csv_load_timing,
        "target_bin_load": bin_load_timing,
        "csv_compensation": csv_compensation_timing,
        "bin_compensation": bin_compensation_timing,
        "large_sparse_write": large_write_timing,
        "large_header_inspect": large_inspect_timing,
        "large_low_level_memmap": large_mmap_timing,
        "large_high_level_rejection": rejection_timing,
    }
    report: dict[str, Any] = {
        "passed": all(checks.values()),
        "configuration": {
            "moderate_samples": moderate_samples,
            "large_samples": large_samples,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "target_x_origin_s": TARGET_X_ORIGIN_S,
            "samples_per_ui": SAMPLES_PER_UI,
            "cursor_ui": CURSOR_UI,
            "external_pattern_symbols": PATTERN_SYMBOLS,
            "external_pattern_seed_hex": hex(PATTERN_SEED),
            "compensation_band_hz": [settings.band_low_hz, settings.band_high_hz],
        },
        "physical_model": {
            "channel_taps": fixture["channel_taps"].tolist(),
            "tone_frequency_hz": fixture["frequency_hz"].tolist(),
            "ideal_tone_amplitude_v": fixture["ideal_amplitude_v"].tolist(),
            "channel_gain_at_tones": fixture["channel_gain"].tolist(),
        },
        "validation": {
            "csv_bin_max_abs_value_difference_v": value_difference_v,
            "csv_bin_max_abs_time_difference_s": time_difference_s,
            "csv_sample_rate_hz": target_csv.sample_rate_hz,
            "bin_sample_rate_hz": target_bin.sample_rate_hz,
            "csv_bin_sample_rate_difference_hz": sample_rate_difference_hz,
            "csv_bin_sample_rate_relative_difference": (
                sample_rate_relative_difference
            ),
            "sample_rate_comparison_rtol": SAMPLE_RATE_COMPARISON_RTOL,
            "csv_bin_compensation_max_abs_difference_v": output_difference_v,
            "compensated_bin_roundtrip_max_abs_difference_v": (
                output_bin_roundtrip_difference_v
            ),
            "guard_samples_each_side": guard_samples,
            "before_interior_error_vrms": before_error_vrms,
            "after_interior_error_vrms": after_error_vrms,
            "interior_rms_improvement_factor": improvement_factor,
            "input_tone_amplitude_v": input_tone_v.tolist(),
            "output_tone_amplitude_v": output_tone_v.tolist(),
            "maximum_in_band_tone_error_v": maximum_in_band_tone_error_v,
            "outside_control_tone_change_v": outside_control_change_v,
            "checks": checks,
        },
        "vpp": vpp_results,
        "large_bin": {
            "path": str(large_bin_path),
            "points": large_info.waveforms[0].points,
            "sample_rate_hz": large_info.waveforms[0].sample_rate_hz,
            "x_origin_s": large_info.waveforms[0].x_origin_s,
            "logical_size_bytes": logical_size_bytes,
            "allocated_size_bytes": allocated_size_bytes,
            "low_level_is_memmap": low_level_is_memmap,
            "high_level_rejected_before_payload": high_level_rejected,
            "high_level_rejection_message": rejection_message,
        },
        "performance": performance,
        "artifacts": {
            path.name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "git": _git_snapshot(),
            "rss_note": (
                "ru_maxrss 是进程峰值；前序步骤已达到更高峰值时，后续增量可能为 0"
            ),
        },
    }
    json_path = output_dir / "validation_report.json"
    markdown_path = output_dir / "validation_report.md"
    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"验证失败，详见报告；失败项：{failed}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    """定义面向终端与 PyCharm Run Configuration 的显式参数。"""

    parser = argparse.ArgumentParser(
        description="生成 ResponseLab Vpp/Keysight 数据并运行全链路验证。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="必须尚不存在的输出目录；脚本不会覆盖已有目录。",
    )
    parser.add_argument(
        "--moderate-samples",
        type=int,
        default=DEFAULT_MODERATE_SAMPLES,
        help=f"执行 CSV/BIN 双路径补偿的样点数（默认 {DEFAULT_MODERATE_SAMPLES}）。",
    )
    parser.add_argument(
        "--large-samples",
        type=int,
        default=DEFAULT_LARGE_SAMPLES,
        help=f"稀疏大 BIN 的逻辑样点数（默认 {DEFAULT_LARGE_SAMPLES}）。",
    )
    return parser


def main() -> int:
    """解析参数、执行验证，并以退出码表达完整通过或异常失败。"""

    arguments = _build_parser().parse_args()
    report = run_validation(
        arguments.output_dir,
        moderate_samples=arguments.moderate_samples,
        large_samples=arguments.large_samples,
    )
    print(f"PASS：验证报告已写入 {arguments.output_dir.resolve()}")
    print(
        "补偿内区 RMS 误差改善："
        f"{report['validation']['interior_rms_improvement_factor']:.1f}×"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
