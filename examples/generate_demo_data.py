"""生成 Np=400、Nb=4 的 ResponseLab 大文件导入示例。"""

# Codex说明(自动生成)： 从 __future__ 导入 annotations，启用较新的类型标注行为，减少运行期导入或前向引用问题。
from __future__ import annotations

# Codex说明(自动生成)： 导入 argparse，解析命令行参数，支持用户从终端覆盖默认配置。
import argparse
# Codex说明(自动生成)： 导入 sys，访问解释器路径、退出码和标准错误输出。
import sys
# Codex说明(自动生成)： 从 pathlib 导入 Path，用 Path 对象处理跨平台文件路径。
from pathlib import Path

# Codex说明(自动生成)： 导入 numpy as np，执行数组、向量化和数值仿真计算。
import numpy as np
# Codex说明(自动生成)： 从 scipy 导入 signal，提供本文件后续流程需要的库能力。
from scipy import signal

# 让 PyCharm 直接运行本脚本时也能加载本项目的 src 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Codex说明(自动生成)： 计算并保存 SRC_DIR，供后续语句继续读取或更新。
SRC_DIR = PROJECT_ROOT / "src"
# Codex说明(自动生成)： 检查条件 str(SRC_DIR) not in sys.path，根据结果选择后续执行路径。
if str(SRC_DIR) not in sys.path:
    # Codex说明(自动生成)： 调用 sys.path.insert 更新列表或集合，把当前步骤产生的数据加入结果。
    sys.path.insert(0, str(SRC_DIR))

# Codex说明(自动生成)： 从 response_lab.keysight_bin 导入 write_keysight_bin，提供本文件后续流程需要的库能力。
from response_lab.keysight_bin import write_keysight_bin
# Codex说明(自动生成)： 从 response_lab.vpp_analysis 导入 generate_prbs13q_gray_symbols，提供本文件后续流程需要的库能力。
from response_lab.vpp_analysis import generate_prbs13q_gray_symbols

# 示例遵循 M=32、Np=400 UI、Nb=4 UI；Fs=M*Rs=3.4 TSa/s。
M = 32
# Codex说明(自动生成)： 计算并保存 NP_UI，供后续语句继续读取或更新。
NP_UI = 400
# Codex说明(自动生成)： 计算并保存 NB_UI，供后续语句继续读取或更新。
NB_UI = 4
# Codex说明(自动生成)： 计算并保存 BAUD_RATE_HZ，供后续语句继续读取或更新。
BAUD_RATE_HZ = 106.25e9
# Codex说明(自动生成)： 计算并保存 SAMPLE_RATE_HZ，供后续语句继续读取或更新。
SAMPLE_RATE_HZ = M * BAUD_RATE_HZ
# Codex说明(自动生成)： 计算并保存 PULSE_SAMPLES，供后续语句继续读取或更新。
PULSE_SAMPLES = M * NP_UI
# Codex说明(自动生成)： 计算并保存 PEAK_INDEX，供后续语句继续读取或更新。
PEAK_INDEX = M * NB_UI
# BIN 为 float32 payload，约 32 MiB；CSV 因文本编码每行更长，使用较少点数以同样约 32 MiB。
# 这个 BIN 是可补偿输入，不再另造一个只能读取、不能用于补偿的压力文件。
BIN_COMPENSATION_SAMPLES = 8_388_544
# Codex说明(自动生成)： 计算并保存 CSV_TARGET_SAMPLES，供后续语句继续读取或更新。
# 786700 行文本 CSV 实测约 32 MiB，与 float32 BIN 的大小相当。
CSV_TARGET_SAMPLES = 786_700
# Codex说明(自动生成)： 计算并保存 RANDOM_SEED，供后续语句继续读取或更新。
RANDOM_SEED = 20260722

# 大文件演示使用温和的三抽头通道：主光标只低 8%，仍保留一阶、二阶后游标。
# 在 1–60 GHz 内，其理论反向补偿需求约为 0.27–1.67 dB，便于观察而不过度夸张。
DEMO_MAIN_TAP = 0.92
DEMO_POSTCURSOR_1_TAP = 0.07
DEMO_POSTCURSOR_2_TAP = -0.025


# Codex说明(自动生成)： 计算并保存 README_TEXT，供后续语句继续读取或更新。
README_TEXT = """# ResponseLab 大文件示例

固定拟合脉冲参数：`M=32`、`Np=400 UI`、`Nb=4 UI`、`Fs=3.4 TSa/s`、`Rs=106.25 GBd`。

- `01_参考拟合脉冲...`：填“参考拟合脉冲”。
- `02_待补偿拟合脉冲...`：填“待补偿拟合脉冲”。
- `03_待补偿原始信号_约32MiB.csv` 或 `03_...可补偿...bin`：填“待补偿信号”，二选一。
- `04_Vpp理想码型...`：只在“加载理想码型”时使用，文件值类型选“Gray 符号码 0–3”。

参考与 DUT 是同一类脉冲的温和三抽头差异：主光标 `0.92`、第一后游标
`+0.07`、第二后游标 `-0.025`。在 1–60 GHz 内，对应的理论补偿需求约
`0.27–1.67 dB`。

CSV 与 BIN 都是约 32 MiB 的可补偿输入；两者的点数因文本与二进制编码不同，
但都使用相同的确定性 PAM4+ISI 通道模型。

这里的 PAM4 目标只通过相对通道 `T = H_dut / H_ref`，而不再额外卷积参考
脉冲。因为补偿响应是 `H_ref / H_dut = 1 / T`，参考脉冲部分在这个相对模型中
相消；本示例用于验证补偿方向和大文件流程，不是示波器实测波形。
"""


# Codex说明(自动生成)： 定义函数 _build_pulses，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _build_pulses() -> tuple[np.ndarray, np.ndarray]:
    """构造主峰位于 Nb*M 的参考/DUT 拟合脉冲，单位为每单位符号的 V。"""

    # 参考主峰在第 4 UI 后，完整记录恰好覆盖 400 UI。
    index = np.arange(PULSE_SAMPLES, dtype=np.float64)
    # Codex说明(自动生成)： 计算并保存 reference，供后续语句继续读取或更新。
    reference = np.exp(-0.5 * ((index - PEAK_INDEX) / 7.0) ** 2)
    # DUT 保留主抽头并增加温和的两个后游标，确保补偿可见但不夸大通道失真。
    dut = DEMO_MAIN_TAP * reference
    # Codex说明(自动生成)： 基于旧值更新 dut[M:]，累积当前循环或处理步骤的结果。
    dut[M:] += DEMO_POSTCURSOR_1_TAP * reference[:-M]
    # Codex说明(自动生成)： 基于旧值更新 dut[2 * M:]，累积当前循环或处理步骤的结果。
    dut[2 * M :] += DEMO_POSTCURSOR_2_TAP * reference[: -2 * M]
    # 两条脉冲共用参考主峰的幅度基准，避免重新归一化掩盖通道损耗。
    return reference, dut


# Codex说明(自动生成)： 定义函数 _build_target，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _build_target(samples: int) -> np.ndarray:
    """生成确定性 PAM4 过 ISI 通道后的待补偿波形，单位 V。"""

    # 先生成按 UI 对齐的 PAM4 电平，使 M=32 在时域中可直接观察。
    rng = np.random.default_rng(RANDOM_SEED)
    # Codex说明(自动生成)： 计算并保存 levels，供后续语句继续读取或更新。
    levels = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float64)
    # Codex说明(自动生成)： 计算并保存 symbols，供后续语句继续读取或更新。
    symbols = rng.choice(levels, size=(samples + M - 1) // M + 2)
    # Codex说明(自动生成)： 计算并保存 ideal，供后续语句继续读取或更新。
    ideal = np.repeat(symbols, M)[:samples]
    # 两个 UI 延迟抽头制造稳定、可复现的 ISI；它与 DUT 脉冲的后游标方向一致。
    taps = np.zeros(2 * M + 1, dtype=np.float64)
    # Codex说明(自动生成)： 计算并保存 taps[0]，供后续语句继续读取或更新。
    taps[0] = DEMO_MAIN_TAP
    # Codex说明(自动生成)： 计算并保存 taps[M]，供后续语句继续读取或更新。
    taps[M] = DEMO_POSTCURSOR_1_TAP
    # Codex说明(自动生成)： 计算并保存 taps[2 * M]，供后续语句继续读取或更新。
    taps[2 * M] = DEMO_POSTCURSOR_2_TAP
    # Codex说明(自动生成)： 返回 np.asarray(signal.lfilter(taps, [1.0], ideal), dtype=np...，让调用方取得本函数的处理结果。
    return np.asarray(signal.lfilter(taps, [1.0], ideal), dtype=np.float64)


# Codex说明(自动生成)： 定义函数 _write_csv，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _write_csv(path: Path, samples: int) -> None:
    """分块写两列 time(s),voltage(V) CSV，避免构造大二维临时表。"""

    # Codex说明(自动生成)： 计算并保存 values，供后续语句继续读取或更新。
    values = _build_target(samples)
    # Codex说明(自动生成)： 进入上下文 path.open('w', encoding='utf-8', newline='')，确保文件、资源或临时状态按作用域正确释放。
    with path.open("w", encoding="utf-8", newline="") as stream:
        # Codex说明(自动生成)： 遍历 range(0, samples, 65536) 中的 start，逐项执行循环体逻辑。
        for start in range(0, samples, 65_536):
            # Codex说明(自动生成)： 计算并保存 stop，供后续语句继续读取或更新。
            stop = min(start + 65_536, samples)
            # Codex说明(自动生成)： 计算并保存 time_s，供后续语句继续读取或更新。
            time_s = np.arange(start, stop, dtype=np.float64) / SAMPLE_RATE_HZ
            # Codex说明(自动生成)： 调用 np.savetxt，执行当前流程需要的具体操作或副作用。
            np.savetxt(stream, np.column_stack((time_s, values[start:stop])), delimiter=",", fmt="%.17g")


# Codex说明(自动生成)： 定义函数 main，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def main(output_dir: Path) -> None:
    """生成所有角色清晰的脉冲、码型和约 32 MiB CSV/BIN 输入文件。"""

    # Codex说明(自动生成)： 调用 output_dir.mkdir，执行当前流程需要的具体操作或副作用。
    output_dir.mkdir(parents=True, exist_ok=True)
    # Codex说明(自动生成)： 计算并保存 (reference, dut)，供后续语句继续读取或更新。
    reference, dut = _build_pulses()
    # Codex说明(自动生成)： 计算并保存 pulse_time_s，供后续语句继续读取或更新。
    pulse_time_s = (np.arange(PULSE_SAMPLES, dtype=np.float64) - PEAK_INDEX) / SAMPLE_RATE_HZ
    # Codex说明(自动生成)： 调用 np.savetxt，执行当前流程需要的具体操作或副作用。
    np.savetxt(output_dir / "01_参考拟合脉冲_M32_Np400_Nb4.csv", np.column_stack((pulse_time_s, reference)), delimiter=",", fmt="%.17g")
    # Codex说明(自动生成)： 调用 np.savetxt，执行当前流程需要的具体操作或副作用。
    np.savetxt(output_dir / "02_待补偿拟合脉冲_M32_Np400_Nb4.csv", np.column_stack((pulse_time_s, dut)), delimiter=",", fmt="%.17g")
    # Codex说明(自动生成)： 调用 _write_csv，执行当前流程需要的具体操作或副作用。
    _write_csv(output_dir / "03_待补偿原始信号_约32MiB.csv", CSV_TARGET_SAMPLES)
    # Codex说明(自动生成)： 计算并保存 bin_values，供后续语句继续读取或更新。
    bin_values = _build_target(BIN_COMPENSATION_SAMPLES)
    # Codex说明(自动生成)： 调用 write_keysight_bin，执行当前流程需要的具体操作或副作用。
    write_keysight_bin(
        output_dir / "03_待补偿原始信号_可补偿_约32MiB_Keysight_AG10.bin",
        bin_values,
        SAMPLE_RATE_HZ,
        label="DUT Target",
    )
    # Codex说明(自动生成)： 调用 np.savetxt，执行当前流程需要的具体操作或副作用。
    np.savetxt(output_dir / "04_Vpp理想码型_PRBS13Q_Gray_8191_符号码.csv", generate_prbs13q_gray_symbols(), fmt="%d")
    # Codex说明(自动生成)： 调用 output_dir / 'README_导入顺序.md'.write_text 写出文件或数据，保存当前处理结果。
    (output_dir / "README_导入顺序.md").write_text(README_TEXT, encoding="utf-8")
    # Codex说明(自动生成)： 输出面向用户的运行信息，帮助确认当前脚本进度或结果路径。
    print(f"输出目录：{output_dir}")
    # Codex说明(自动生成)： 输出面向用户的运行信息，帮助确认当前脚本进度或结果路径。
    print(f"M={M}, Np={NP_UI}, Nb={NB_UI}, Fs={SAMPLE_RATE_HZ / 1e12:.4f} TSa/s")


# Codex说明(自动生成)： 检查条件 __name__ == '__main__'，根据结果选择后续执行路径。
if __name__ == "__main__":
    # 默认输出到用户当前数据目录，也允许在 PyCharm 的参数中改成任意空目录。
    parser = argparse.ArgumentParser()
    # Codex说明(自动生成)： 调用 parser.add_argument 注册命令行参数，让用户可以从终端配置运行选项。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples" / "ResponseLab_大文件示例_温和差异",
    )
    # Codex说明(自动生成)： 调用 main，执行当前流程需要的具体操作或副作用。
    main(parser.parse_args().output_dir)
