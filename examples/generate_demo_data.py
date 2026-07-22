"""生成按角色清晰命名、可直接导入 ResponseLab 的示例数据。"""

# Codex说明(自动生成)： 从 __future__ 导入 annotations，启用较新的类型标注行为，减少运行期导入或前向引用问题。
from __future__ import annotations

# sys 仅用于在未执行 editable install 时让本示例直接引用项目 src 包。
import sys
# Codex说明(自动生成)： 从 pathlib 导入 Path，用 Path 对象处理跨平台文件路径。
from pathlib import Path

# Codex说明(自动生成)： 导入 numpy as np，执行数组、向量化和数值仿真计算。
import numpy as np

# 直接在 PyCharm 运行本文件时，把项目 src 根目录放到导入搜索路径首位。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Codex说明(自动生成)： 计算并保存 SRC_DIR，供后续语句继续读取或更新。
SRC_DIR = PROJECT_ROOT / "src"
# 仅在 src 尚未可见时插入一次，避免重复路径影响 PyCharm 的模块解析顺序。
if str(SRC_DIR) not in sys.path:
    # Codex说明(自动生成)： 调用 sys.path.insert 更新列表或集合，把当前步骤产生的数据加入结果。
    sys.path.insert(0, str(SRC_DIR))

# 导出 AG10 前复用生产写入器，以保留自描述的采样率和时间原点。
from response_lab.io import save_bin_timeseries
# 导出用户可加载的码型时复用 Vpp 内置 PRBS13Q Gray 的同一发生器。
from response_lab.vpp_analysis import generate_prbs13q_gray_symbols


# 说明文件把 UI 字段、数据角色和文件名固定对应，防止把两条脉冲互换。
README_TEXT = """# ResponseLab 示例文件：导入顺序与角色

本目录中的文件按数字顺序命名；数字是**导入角色**，不是算法处理顺序。

| 文件 | 在 ResponseLab 中选择到哪里 | 是否必需 |
| --- | --- | --- |
| `01_参考拟合脉冲_ideal_reference_pulse.csv` | 主页面左侧“参考拟合脉冲” | 是 |
| `02_待补偿拟合脉冲_dut_pulse.csv` | 主页面左侧“待补偿拟合脉冲” | 是 |
| `03_待补偿原始信号_target.csv` | 主页面“待补偿信号” | 仅数据补偿时需要 |
| `03_待补偿原始信号_target_keysight_ag10.bin` | 同上；与 CSV 是同一条信号，二选一 | 仅数据补偿时需要 |
| `04_Vpp理想码型_PRBS13Q_Gray_8191_符号码.csv` | 影响频段页：选择“加载用户理想码型”后填写路径 | 可选 |

## 推荐操作

1. 先载入 `01` 与 `02`，点击“分析拟合脉冲”。参考脉冲是理想目标，DUT 脉冲是存在信道拖尾、需要被补偿的一方。
2. 若要补偿一段真实数据，再载入 `03` 的 CSV 或 BIN（两者内容等价，只选一个）。
3. 在影响频段页选择 Vpp：
   - **LFP 峰峰值**：比较补偿前后 DUT 周期 Vpp 与参考周期 Vpp 的差距；
   - **频域 RMS 误差**：比较补偿前后 DUT 复频谱与参考复频谱的去 DC 误差，单位 Vrms。
4. 可直接选择内置“PRBS13Q Gray（8191）”；若要验证用户加载模式，则选择 `04`，文件值类型选择“Gray 符号码 0–3”。

注意：`01`/`02` 是**拟合脉冲**，不是待补偿原始波形；`03` 才是应用补偿响应的原始信号。
"""


# Codex说明(自动生成)： 定义函数 main，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def main() -> None:
    # 新目录名强调其用途；重复运行只更新同名示例，不会影响用户自己的采集文件。
    output_dir = Path(__file__).resolve().parent / "ResponseLab_角色清晰导入示例"
    # 创建示例目录，已存在时保持其余用户文件不变。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 示例采用 2 GSa/s 的均匀时间轴；所有 CSV 时间列单位均为 s。
    sample_rate_hz = 2.0e9
    # 拟合脉冲长度足够展示主光标和一段可见拖尾。
    pulse_samples = 2048
    # 脉冲时间轴从 0 s 开始，采样间隔为 1/Fs。
    pulse_time_s = np.arange(pulse_samples, dtype=np.float64) / sample_rate_hz
    # 样点索引用于构造以主光标为中心的高斯型参考脉冲。
    index = np.arange(pulse_samples, dtype=np.float64)
    # 参考脉冲代表理想目标响应，主抽头幅度约为 1 V。
    reference = np.exp(-0.5 * ((index - 420.0) / 3.0) ** 2)
    # DUT 加入 5-sample 延迟和 0.72 倍衰减，代表待补偿通道的简化响应。
    dut = np.zeros_like(reference)
    # Codex说明(自动生成)： 计算并保存 dut[5:]，供后续语句继续读取或更新。
    dut[5:] = 0.72 * reference[:-5]

    # 目标原始信号长度与拟合脉冲独立；它用于“数据补偿”页，而非拟合比较页。
    signal_samples = 16384
    # 原始信号沿用同一采样率，便于补偿频点与示例脉冲对齐。
    signal_time_s = np.arange(signal_samples, dtype=np.float64) / sample_rate_hz
    # 两个不同频率的正弦成分使补偿前后频带效果容易观察。
    signal = (
        0.55 * np.sin(2.0 * np.pi * 120.0e6 * signal_time_s)
        + 0.22 * np.sin(2.0 * np.pi * 240.0e6 * signal_time_s + 0.4)
    )

    # 01 明确标为参考拟合脉冲，应填入主页面的“参考拟合脉冲”字段。
    np.savetxt(
        output_dir / "01_参考拟合脉冲_ideal_reference_pulse.csv",
        np.column_stack((pulse_time_s, reference)),
        delimiter=",",
        fmt="%.17g",
    )
    # 02 明确标为待补偿 DUT 拟合脉冲，应填入主页面的“待补偿拟合脉冲”字段。
    np.savetxt(
        output_dir / "02_待补偿拟合脉冲_dut_pulse.csv",
        np.column_stack((pulse_time_s, dut)),
        delimiter=",",
        fmt="%.17g",
    )
    # 03 CSV 是待补偿原始信号，和下方 AG10 BIN 的电压样点完全相同。
    np.savetxt(
        output_dir / "03_待补偿原始信号_target.csv",
        np.column_stack((signal_time_s, signal)),
        delimiter=",",
        fmt="%.17g",
    )
    # 03 BIN 让用户验证 Keysight 自描述 BIN 导入；它和 CSV 在 UI 中二选一。
    save_bin_timeseries(
        output_dir / "03_待补偿原始信号_target_keysight_ag10.bin",
        signal_time_s,
        signal,
        label="DUT Target",
    )
    # 04 写出与内置模式一致的 8191 个 Gray 符号码，供“加载用户理想码型”模式验证。
    np.savetxt(
        output_dir / "04_Vpp理想码型_PRBS13Q_Gray_8191_符号码.csv",
        generate_prbs13q_gray_symbols(),
        fmt="%d",
    )
    # README 让示例目录脱离聊天上下文后仍能安全使用。
    (output_dir / "README_导入顺序.md").write_text(README_TEXT, encoding="utf-8")
    # 控制台只给出目录与关键合同，便于从 PyCharm Run 窗口复制路径。
    print(f"已生成角色清晰的 ResponseLab 示例：{output_dir}")
    # Codex说明(自动生成)： 输出面向用户的运行信息，帮助确认当前脚本进度或结果路径。
    print("03 的 CSV/BIN 是同一待补偿信号，二选一；04 是可选的 Vpp 用户码型。")


# Codex说明(自动生成)： 检查条件 __name__ == '__main__'，根据结果选择后续执行路径。
if __name__ == "__main__":
    # 直接运行脚本时生成或更新该固定示例目录。
    main()
