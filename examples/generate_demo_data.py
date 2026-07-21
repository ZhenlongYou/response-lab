"""生成可直接导入 ResponseLab 的无表头 CSV 和 Keysight AG10 BIN 演示数据。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from response_lab.io import save_bin_timeseries


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
    save_bin_timeseries(
        output_dir / "target_signal_keysight_ag10.bin",
        signal_time_s,
        signal,
        label="ResponseLab demo",
    )
    print(f"已生成演示数据：{output_dir}")
    print("BIN：Keysight Infiniium AG10，采样率与时间原点由文件自动读取")


if __name__ == "__main__":
    main()
