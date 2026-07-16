"""生成可直接拖入 ResponseLab 的无表头 CSV 和原始 BIN 演示数据。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "demo_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate_hz = 2.0e9
    pulse_samples = 2048
    pulse_time_s = np.arange(pulse_samples, dtype=np.float64) / sample_rate_hz
    index = np.arange(pulse_samples, dtype=np.float64)
    reference = np.exp(-0.5 * ((index - 420.0) / 3.0) ** 2)
    dut = np.zeros_like(reference)
    dut[5:] = 0.72 * reference[:-5]

    signal_samples = 16384
    signal_time_s = np.arange(signal_samples, dtype=np.float64) / sample_rate_hz
    signal = (
        0.55 * np.sin(2.0 * np.pi * 120.0e6 * signal_time_s)
        + 0.22 * np.sin(2.0 * np.pi * 240.0e6 * signal_time_s + 0.4)
    )

    np.savetxt(
        output_dir / "reference_pulse.csv",
        np.column_stack((pulse_time_s, reference)),
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(
        output_dir / "dut_pulse.csv",
        np.column_stack((pulse_time_s, dut)),
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(
        output_dir / "target_signal.csv",
        np.column_stack((signal_time_s, signal)),
        delimiter=",",
        fmt="%.17g",
    )
    signal.astype("<f4").tofile(output_dir / "target_signal_float32_le.bin")
    print(f"已生成演示数据：{output_dir}")
    print("BIN 解析参数：2e9 Hz / float32 / little / 1 通道 / interleaved")


if __name__ == "__main__":
    main()
