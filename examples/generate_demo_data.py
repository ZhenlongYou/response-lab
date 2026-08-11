"""生成 Np=400、Nb=4 的 ResponseLab 大文件导入示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 让 PyCharm 直接运行本脚本时也能加载本项目的 src 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from response_lab.keysight_bin import write_keysight_bin  # noqa: E402
from response_lab.vpp_analysis import generate_prbs13q_gray_symbols  # noqa: E402

# 示例遵循 M=32、Np=400 UI、Nb=4 UI；Fs=M*Rs=3.4 TSa/s。
M = 32
NP_UI = 400
NB_UI = 4
BAUD_RATE_HZ = 106.25e9
SAMPLE_RATE_HZ = M * BAUD_RATE_HZ
PULSE_SAMPLES = M * NP_UI
PEAK_INDEX = M * NB_UI
# BIN 为 float32 payload，约 32 MiB；CSV 因文本编码每行更长，使用较少点数。
BIN_COMPENSATION_SAMPLES = 8_388_544
# 786700 行文本 CSV 实测约 32 MiB，与 float32 BIN 的大小相当。
CSV_TARGET_SAMPLES = 786_700
RANDOM_SEED = 20260722

# 大文件演示使用接近重合的零相位三抽头通道；在 1–60 GHz 内，
# 理论反向补偿需求约为 0.02–0.35 dB，用于温和流程演示。
DEMO_MAIN_TAP = 0.985
DEMO_SYMMETRIC_1_UI_TAP = 0.009
DEMO_SYMMETRIC_2_UI_TAP = -0.003


README_TEXT = """# ResponseLab 大文件示例

固定拟合脉冲参数：`M=32`、`Np=400 UI`、`Nb=4 UI`、`Fs=3.4 TSa/s`、`Rs=106.25 GBd`。

- `01_参考拟合脉冲...`：填“参考拟合脉冲”。
- `02_待补偿拟合脉冲...`：填“待补偿拟合脉冲”。
- `03_待补偿原始信号_约32MiB.csv` 或 `03_...可补偿...bin`：填“待补偿信号”，二选一。
- `04_Vpp理想码型...`：只在“加载理想码型”时使用，文件值类型选“Gray 符号码 0–3”。

参考与 DUT 是接近重合的同类脉冲：主光标 `0.985`，在前后各有第一游标
`+0.009`、第二游标 `-0.003`。前后游标严格对称，因此相对频响是正实数，
相位差为 `0°`；在 1–60 GHz 内，对应的理论补偿需求约 `0.02–0.35 dB`。

CSV 与 BIN 都是约 32 MiB 的可补偿输入；两者的点数因文本与二进制编码不同，
但都使用相同的确定性 PAM4+ISI 通道模型；记录长度不同，因此末尾 64 个样点的
零延拓边界可能不同。

这里的 PAM4 目标只通过相对通道 `T = H_dut / H_ref`，而不再额外卷积参考
脉冲。因为补偿响应是 `H_ref / H_dut = 1 / T`，参考脉冲部分在这个相对模型中
相消。为让示例相位重合，`T` 是零相位的对称模型，并非原始因果示波器通道；
本示例用于验证补偿方向和大文件流程，不是示波器实测波形。
"""


def _build_pulses() -> tuple[np.ndarray, np.ndarray]:
    """构造主峰位于 Nb*M 的参考/DUT 拟合脉冲，单位为每单位符号的 V。"""

    # 参考主峰在第 4 UI 后，完整记录恰好覆盖 400 UI。
    index = np.arange(PULSE_SAMPLES, dtype=np.float64)
    reference = np.exp(-0.5 * ((index - PEAK_INDEX) / 7.0) ** 2)
    # 前后游标严格对称，使 H_dut/H_ref 为正实数且没有额外相位差。
    # 直接在时域叠加，避免 IFFT 的极小数值噪声填满原本为零的长记录尾部。
    dut = DEMO_MAIN_TAP * reference
    dut[M:] += DEMO_SYMMETRIC_1_UI_TAP * reference[:-M]
    dut[:-M] += DEMO_SYMMETRIC_1_UI_TAP * reference[M:]
    dut[2 * M :] += DEMO_SYMMETRIC_2_UI_TAP * reference[: -2 * M]
    dut[: -2 * M] += DEMO_SYMMETRIC_2_UI_TAP * reference[2 * M :]
    # 两条脉冲共用参考主峰的幅度基准，避免重新归一化掩盖通道损耗。
    return reference, dut


def _build_target(samples: int) -> np.ndarray:
    """生成确定性 PAM4 过 ISI 通道后的待补偿波形，单位 V。"""

    # 先生成按 UI 对齐的 PAM4 电平，使 M=32 在时域中可直接观察。
    rng = np.random.default_rng(RANDOM_SEED)
    levels = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float64)
    symbols = rng.choice(levels, size=(samples + M - 1) // M + 2)
    ideal = np.repeat(symbols, M)[:samples]
    # 目标采用同一零相位相对通道；前后样本等权参与，记录边缘按零延拓处理。
    target = DEMO_MAIN_TAP * ideal
    target[M:] += DEMO_SYMMETRIC_1_UI_TAP * ideal[:-M]
    target[:-M] += DEMO_SYMMETRIC_1_UI_TAP * ideal[M:]
    target[2 * M :] += DEMO_SYMMETRIC_2_UI_TAP * ideal[: -2 * M]
    target[: -2 * M] += DEMO_SYMMETRIC_2_UI_TAP * ideal[2 * M :]
    return target


def _write_csv(path: Path, samples: int) -> None:
    """分块写两列 time(s),voltage(V) CSV，避免构造大二维临时表。"""

    values = _build_target(samples)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for start in range(0, samples, 65_536):
            stop = min(start + 65_536, samples)
            time_s = np.arange(start, stop, dtype=np.float64) / SAMPLE_RATE_HZ
            block = np.column_stack((time_s, values[start:stop]))
            np.savetxt(stream, block, delimiter=",", fmt="%.17g")


def main(output_dir: Path) -> None:
    """生成所有角色清晰的脉冲、码型和约 32 MiB CSV/BIN 输入文件。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    reference, dut = _build_pulses()
    pulse_time_s = (
        np.arange(PULSE_SAMPLES, dtype=np.float64) - PEAK_INDEX
    ) / SAMPLE_RATE_HZ
    np.savetxt(
        output_dir / "01_参考拟合脉冲_M32_Np400_Nb4.csv",
        np.column_stack((pulse_time_s, reference)),
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(
        output_dir / "02_待补偿拟合脉冲_M32_Np400_Nb4.csv",
        np.column_stack((pulse_time_s, dut)),
        delimiter=",",
        fmt="%.17g",
    )
    _write_csv(output_dir / "03_待补偿原始信号_约32MiB.csv", CSV_TARGET_SAMPLES)
    bin_values = _build_target(BIN_COMPENSATION_SAMPLES)
    write_keysight_bin(
        output_dir / "03_待补偿原始信号_可补偿_约32MiB_Keysight_AG10.bin",
        bin_values,
        SAMPLE_RATE_HZ,
        label="DUT Target",
    )
    np.savetxt(
        output_dir / "04_Vpp理想码型_PRBS13Q_Gray_8191_符号码.csv",
        generate_prbs13q_gray_symbols(),
        fmt="%d",
    )
    (output_dir / "README_导入顺序.md").write_text(README_TEXT, encoding="utf-8")
    print(f"输出目录：{output_dir}")
    print(f"M={M}, Np={NP_UI}, Nb={NB_UI}, Fs={SAMPLE_RATE_HZ / 1e12:.4f} TSa/s")


if __name__ == "__main__":
    # 默认输出到示例目录，也允许在 PyCharm 的参数中改成任意目录。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples" / "ResponseLab_大文件示例_温和差异",
    )
    main(parser.parse_args().output_dir)
