"""指标归因、轻量眼图与频段扫描的核心数学。

本模块中的眼高和眼宽只表达拟合脉冲在固定理想符号序列下的 LTI/ISI 工程近似。
它不包含 CDR、抖动、噪声、BER 外推或示波器私有测量算法。
"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本文件仅关闭对应格式告警。
# ruff: noqa: I001

# 延迟解析类型标注，使前后定义的数据类可以互相引用且不增加运行期导入耦合。
from __future__ import annotations

# Callable 为后台扫描提供进度和取消回调，不依赖 Qt。
from collections.abc import Callable
# dataclass 使物理设置和结果保持不可变，避免界面意外修改后台结果。
from dataclasses import dataclass, replace
# Literal 把用户可见的 NRZ/PAM4 选项限制在明确集合内。
from typing import Literal

# NumPy 承担固定种子符号、2 UI 轨迹提取和条件包络计算。
import numpy as np
# NDArray 为数值数组保留可审查的维度与类型契约。
from numpy.typing import NDArray
# SciPy FFT 允许准备阶段缓存固定符号冲激的频谱。
from scipy import fft as scipy_fft

# 归因准备、候选 IFFT 与眼图卷积共用后台任务的协作取消合同。
from .cancellation import (
    CancellationCheck,
    OperationCancelledError,
    raise_if_cancelled,
)
# 直接频响求值与分段相位工具复用已验证的补偿数学，避免另写一套 FFT 约定。
from .dsp import (
    _anchor_phase_islands,
    _pulse_response_on_uniform_frequencies,
    _segmented_unwrap,
    fit_linear_phase_slope,
)
# CompensationSettings 只作为直接 DTFT 求值的窗参数载体；TimeSeries 统一时轴合同。
from .models import (
    CompensationSettings,
    TimeSeries,
    validate_cross_pulse_sample_rates,
)
# Vpp 的理想码型、pmax 窗口和周期稳态模型由专用数值模块统一定义。
from .vpp_analysis import (
    VppAnalysisCache,
    VppAnalysisSettings,
    measure_candidate,
    prepare_vpp_analysis,
)
# 轻量眼图测量复用本地眼图库的分位数与水平 crossing 口径，但不引入 CDR 或聚类依赖。
from .virtual_eye_metrics import measure_virtual_eye_openings

# 虚拟眼调制格式只支持用户已确认的 NRZ 和 PAM4。
Modulation = Literal["nrz", "pam4"]
# 归因指标一次只选择一个，避免把不同量纲强行合并成单个分数。
AttributionMetric = Literal["vpp", "eye_height", "eye_width"]
# 三个模式分别回答幅度、相位及二者共同补偿后的恢复程度。
AttributionMode = Literal["magnitude", "phase", "both"]
# 数值结果统一使用 float64，与 ResponseLab 其他 DSP 路径一致。
FloatArray = NDArray[np.float64]
# 复补偿数组统一使用 complex128，避免幅相模式在中间步骤降精度。
ComplexArray = NDArray[np.complex128]

# 候选中心步进默认 100 MHz；界面可覆盖，内部始终使用 Hz。
DEFAULT_FREQUENCY_STEP_HZ = 100.0e6
# 满权核心宽度默认 100 MHz；界面可覆盖并与中心步进保持一致。
DEFAULT_WINDOW_WIDTH_HZ = 100.0e6
# 每侧半余弦肩宽默认为核心宽度的 0.5 倍。
DEFAULT_BAND_TAPER_ALPHA = 0.5
# 轨迹叠加图最多保留附件示例约定的 600 条确定性轨迹，避免界面被数千条线拖慢。
MAX_EYE_PLOT_TRACES = 600
# 主种子扫描后，用另外两个确定性种子复核全部频段及全部三种模式。
EYE_ROBUSTNESS_SEED_COUNT = 3
EYE_ROBUSTNESS_EXTRA_EVALUATIONS_PER_BAND = (
    (EYE_ROBUSTNESS_SEED_COUNT - 1) * 3
)

# FrequencyBand 把候选满权核心的 Hz 边界和绘图中心绑定，防止显示位置脱离真实补偿窗。
@dataclass(frozen=True)
class FrequencyBand:
    """一个以 Hz 表示的可扫描物理频段。"""

    # low_hz 和 high_hz 是用户可见且权重恒为一的核心边界。
    low_hz: float
    # high_hz 必须大于 low_hz，二者均不超出用户扫描范围。
    high_hz: float
    # center_hz 保留本次用户频宽对应的候选中心网格，便于画影响曲线。
    center_hz: float

    # 在候选进入频率权重计算前验证边界，避免负频率、空频段或中心越界流入扫描。
    def __post_init__(self) -> None:
        """拒绝倒置或非有限的物理频段。"""

        # 三个物理字段都必须能参与确定性浮点比较。
        numeric = (self.low_hz, self.high_hz, self.center_hz)
        # NaN/Inf 会破坏频带掩码，因此在进入 FFT 前终止。
        if not all(np.isfinite(value) for value in numeric):
            # 错误信息使用界面术语，方便直接转交给用户。
            raise ValueError("候选频段必须是有限 Hz 数值")
        # 频段必须有正宽度且不能进入负频率。
        if not 0.0 <= self.low_hz < self.high_hz:
            # 单边真实频谱不接受负频率或零宽度。
            raise ValueError("候选频段必须满足 0 <= low < high")
        # 中心必须落在自身边界内，避免图上点位与实际补偿范围不一致。
        if not self.low_hz <= self.center_hz <= self.high_hz:
            # 此校验能捕获 Hz/GHz 换算只改显示未改 DSP 的错误。
            raise ValueError("候选中心必须位于频段内")

# AttributionSettings 汇总指标、候选窗几何、相位去趋势和眼图参数，形成一次扫描的不可变合同。
@dataclass(frozen=True)
class AttributionSettings:
    """一次影响频段扫描的物理设置。"""

    # metric 对应界面一次只选一个的 Vpp、眼高或眼宽。
    metric: AttributionMetric
    # scan_low_hz 是候选频段的最低边界，内部不使用 GHz。
    scan_low_hz: float
    # scan_high_hz 是候选频段最高边界，不能超过公共 Nyquist。
    scan_high_hz: float
    # eye 在眼指标下提供派生 Np、用户 M 和 NRZ/PAM4；Vpp 模式允许为空。
    eye: VirtualEyeSettings | None = None
    # vpp 在新 Vpp 模式下冻结码型、M 和 pmax 拖尾窗；旧原始波形 API 允许为空。
    vpp: VppAnalysisSettings | None = None
    # frequency_step_hz 默认 100 MHz，界面可按本次分析覆盖。
    frequency_step_hz: float = DEFAULT_FREQUENCY_STEP_HZ
    # requested_window_hz 是用户期望的满权核心宽度，默认 100 MHz。
    requested_window_hz: float = DEFAULT_WINDOW_WIDTH_HZ
    # taper_alpha 描述每侧余弦肩宽相对核心宽度的比例，默认 0.5。
    taper_alpha: float = DEFAULT_BAND_TAPER_ALPHA
    # detrend_phase 打开时剔除整体线性时延，不把设备时移归因到局部频段。
    detrend_phase: bool = True
    # phase_fit_low_hz 为空时沿用扫描下限。
    phase_fit_low_hz: float | None = None
    # phase_fit_high_hz 为空时沿用扫描上限。
    phase_fit_high_hz: float | None = None
    # mode_materiality_fraction 只用于幅度/相位/幅相标签的简化，不用于压掉频段推荐。
    mode_materiality_fraction: float = 0.01
    # 推荐至少要恢复基线差距的 1%，避免把单种子微小随机波动当作主要影响频段。
    recommendation_materiality_fraction: float = 0.01

    # 统一校验所有扫描物理量及互斥条件，让后台线程在分配 FFT 缓存前就能明确失败。
    def __post_init__(self) -> None:
        """验证指标、频带和固定扫描几何。"""

        # 未知指标没有可解释的参考误差定义。
        if self.metric not in {"vpp", "eye_height", "eye_width"}:
            # 错误仅列出公共 API 允许的三个内部键。
            raise ValueError("指标必须是 vpp、eye_height 或 eye_width")
        # 扫描上下限必须是有限的单边物理频率。
        if not all(np.isfinite(value) for value in (self.scan_low_hz, self.scan_high_hz)):
            # 在候选生成前拒绝 NaN/Inf。
            raise ValueError("扫描频带必须是有限值")
        # 扫描范围必须有正宽度且从非负频率开始。
        if not 0.0 <= self.scan_low_hz < self.scan_high_hz:
            # 真实时域 RFFT 只处理非负单边频率。
            raise ValueError("扫描频带必须满足 0 <= low < high")
        # 步进与请求核心宽度共同决定候选数量，必须是正有限量。
        scan_geometry = (self.frequency_step_hz, self.requested_window_hz)
        # 任一非正值都会导致空候选或无限循环。
        if not all(np.isfinite(value) and value > 0.0 for value in scan_geometry):
            # 字段错误保持可由界面原样显示。
            raise ValueError("频率步进和核心宽度必须是正的有限值")
        # 肩宽比例限制在闭区间 0 到 1，避免肩部无限扩张。
        if not np.isfinite(self.taper_alpha) or not 0.0 <= self.taper_alpha <= 1.0:
            # alpha=0 允许矩形核心用于测试，界面默认始终为 0.5。
            raise ValueError("频段余弦肩宽比例必须位于 0 到 1")
        # 眼高和眼宽必须有明确的内部 Np、用户 M 与调制配置。
        if self.metric in {"eye_height", "eye_width"} and self.eye is None:
            # 控制器先完成 Np 推导，本纯算法层不再猜测缺失设置。
            raise ValueError("眼图指标必须提供 Np、M 和调制设置")
        # Vpp 模型设置不能泄漏到眼图指标，避免隐藏控件继续改变眼结果。
        if self.metric != "vpp" and self.vpp is not None:
            raise ValueError("Vpp 模型设置只能用于 vpp 指标")
        # 两个相位拟合边界必须同时省略或同时填写。
        if (self.phase_fit_low_hz is None) != (self.phase_fit_high_hz is None):
            # 半套边界会使整体时延拟合范围含糊。
            raise ValueError("相位拟合上下限必须同时填写或同时省略")
        # 显式相位拟合范围需要位于非负频率且保持正宽度。
        if self.phase_fit_low_hz is not None and not (
            np.isfinite(self.phase_fit_low_hz)
            and np.isfinite(self.phase_fit_high_hz)
            and 0.0 <= self.phase_fit_low_hz < self.phase_fit_high_hz
        ):
            # 类型收窄由前置 None 配对校验保证，这里只检查物理范围。
            raise ValueError("相位拟合频带必须满足 0 <= low < high")
        # 模式简化比例必须位于 0–1，默认 1% 表示次要分量需产生可见增益才标联合。
        if (
            not np.isfinite(self.mode_materiality_fraction)
            or not 0.0 <= self.mode_materiality_fraction <= 1.0
        ):
            # 该参数不影响“是否推荐”，只影响并列附近选择简单模式。
            raise ValueError("模式贡献阈值必须位于 0 到 1")
        if (
            not np.isfinite(self.recommendation_materiality_fraction)
            or not 0.0 <= self.recommendation_materiality_fraction <= 1.0
        ):
            raise ValueError("推荐显著性阈值必须位于 0 到 1")

# 生成通用 Tukey 频带权重，保留独立窗函数测试与兼容路径所需的标准平滑定义。
def tukey_band_weights(
    frequency_hz: FloatArray,
    *,
    low_hz: float,
    high_hz: float,
    alpha: float = DEFAULT_BAND_TAPER_ALPHA,
) -> FloatArray:
    """在任意均匀频率轴上构造物理边界明确的 Tukey 频带权重。"""

    # 复制为 float64 一维数组，避免调用者修改返回值时影响输入。
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    # 频率轴必须是一维有限数组；空轴没有可应用的频点。
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
        # 扫描器会把此错误转成明确的不可解析状态。
        raise ValueError("频率轴必须是一维有限非空数组")
    # 物理边界与 alpha 使用和设置模型相同的定义域。
    if not np.isfinite(low_hz) or not np.isfinite(high_hz) or not 0.0 <= low_hz < high_hz:
        # 不允许交换边界后静默继续，因为那可能掩盖单位错误。
        raise ValueError("平滑频带必须满足 0 <= low < high")
    # alpha 超出标准 Tukey 定义域时拒绝，而不是裁剪。
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        # 明确提示有效闭区间。
        raise ValueError("Tukey alpha 必须位于 0 到 1")
    # 默认全零表示频带外完全不施加该候选补偿。
    weights = np.zeros(frequencies.shape, dtype=np.float64)
    # 支撑掩码包含两端点，端点权重随后由余弦边沿降到零。
    inside = (frequencies >= low_hz) & (frequencies <= high_hz)
    # 频带在当前 DFT 网格没有频点时直接返回全零供上层判无效。
    if not np.any(inside):
        # 设置只读，防止缓存权重被候选循环原地修改。
        weights.setflags(write=False)
        # 全零权重保留与输入相同的形状。
        return weights
    # alpha=0 是显式矩形窗，用于边界测试或用户未来扩展。
    if alpha == 0.0:
        # 矩形窗在完整闭区间内权重为一。
        weights[inside] = 1.0
    # 正 alpha 使用两侧各 alpha*width/2 的半余弦过渡。
    else:
        # 总物理宽度只从 Hz 边界计算，不依赖频点数量。
        width_hz = high_hz - low_hz
        # 每侧过渡宽度遵循标准 Tukey 定义。
        transition_hz = 0.5 * alpha * width_hz
        # 中间平台先赋一，再分别覆盖左右过渡边沿。
        weights[inside] = 1.0
        # 左边沿包含 low 与 low+transition 两个端点。
        rising = inside & (frequencies <= low_hz + transition_hz)
        # 半余弦从边界零平滑上升到平台一。
        weights[rising] = 0.5 * (
            1.0
            - np.cos(np.pi * (frequencies[rising] - low_hz) / transition_hz)
        )
        # 右边沿与左边沿镜像，避免相位或幅度硬截断。
        falling = inside & (frequencies >= high_hz - transition_hz)
        # 半余弦从平台一下降到 high 边界零。
        weights[falling] = 0.5 * (
            1.0
            - np.cos(np.pi * (high_hz - frequencies[falling]) / transition_hz)
        )
    # 返回只读数组，候选扫描不得原地叠加多个频段权重。
    weights.setflags(write=False)
    # 调用者可复用同一频率轴生成幅度、相位和幅相三种补偿。
    return weights

# 为每个候选构造满权核心和余弦肩部，使相邻核心在公共边界不留下扫描盲区。
def cosine_core_band_weights(
    frequency_hz: FloatArray,
    *,
    core_low_hz: float,
    core_high_hz: float,
    shoulder_hz: float,
    domain_low_hz: float | None = None,
    domain_high_hz: float | None = None,
) -> FloatArray:
    """构造“满权核心 + 两侧半余弦肩部”的局部补偿权重。

    候选列表中的频段始终表示用户要求的满权核心。肩部只负责平滑连接，避免相邻
    核心在 Tukey 零端点处留下低敏感度扫描缝。
    """

    # 频率轴复制为 float64 视图，函数不会修改调用者数据。
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    # 绘制和补偿都要求一维、非空且有限的物理频率轴。
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(
        np.isfinite(frequencies)
    ):
        # 在构造掩码前拒绝坏轴，避免 NaN 被静默当作频带外。
        raise ValueError("频率轴必须是一维有限非空数组")
    # 可见核心必须位于单边非负频率并具有正宽度。
    if not (
        np.isfinite(core_low_hz)
        and np.isfinite(core_high_hz)
        and 0.0 <= core_low_hz < core_high_hz
    ):
        # 核心边界同时用于候选文字，不能静默交换。
        raise ValueError("满权核心必须满足 0 <= low < high")
    # 肩宽允许为零以表达显式矩形核心，其余情况必须为正有限值。
    if not np.isfinite(shoulder_hz) or shoulder_hz < 0.0:
        # 负肩宽会反转余弦方向，必须在进入除法前拒绝。
        raise ValueError("余弦肩宽必须是非负有限值")
    # 有效域上下限必须成对出现，避免只裁一侧造成含糊支撑。
    if (domain_low_hz is None) != (domain_high_hz is None):
        # 调用方应同时给出扫描上下限或同时省略。
        raise ValueError("权重有效域上下限必须同时填写或同时省略")
    # 默认整个输入频率轴都可使用。
    domain = np.ones(frequencies.shape, dtype=np.bool_)
    # 显式扫描域把第一个和最后一个肩部裁到可解析范围。
    if domain_low_hz is not None and domain_high_hz is not None:
        # 扫描域必须完整包含可见核心。
        if not (
            np.isfinite(domain_low_hz)
            and np.isfinite(domain_high_hz)
            and 0.0 <= domain_low_hz <= core_low_hz
            and core_high_hz <= domain_high_hz
        ):
            # 若核心越域，候选文字与真实应用范围会不一致。
            raise ValueError("权重有效域必须完整包含满权核心")
        # 域外频点保持单位补偿，不读取未准备的频响比。
        domain = (frequencies >= domain_low_hz) & (
            frequencies <= domain_high_hz
        )
    # 默认全零表示核心和肩部以外完全不施加该候选。
    weights = np.zeros(frequencies.shape, dtype=np.float64)
    # 核心包含两端点且始终满权，保证相邻核心公共边界不会漏扫。
    core = (
        domain
        & (frequencies >= core_low_hz)
        & (frequencies <= core_high_hz)
    )
    # 满权核心先赋一。
    weights[core] = 1.0
    # 零肩宽就是显式矩形核心，不进入余弦除法。
    if shoulder_hz > 0.0:
        # 左肩从 core_low-shoulder 的零平滑升到核心左边界的一。
        left = (
            domain
            & (frequencies >= core_low_hz - shoulder_hz)
            & (frequencies < core_low_hz)
        )
        # 半余弦在肩部中点恰好为 0.5。
        weights[left] = 0.5 * (
            1.0
            - np.cos(
                np.pi
                * (frequencies[left] - core_low_hz + shoulder_hz)
                / shoulder_hz
            )
        )
        # 右肩从核心右边界的一平滑降到 high+shoulder 的零。
        right = (
            domain
            & (frequencies > core_high_hz)
            & (frequencies <= core_high_hz + shoulder_hz)
        )
        # 右侧公式与左肩严格镜像。
        weights[right] = 0.5 * (
            1.0
            + np.cos(
                np.pi * (frequencies[right] - core_high_hz) / shoulder_hz
            )
        )
    # 只读保护避免三种模式共享权重时发生原地污染。
    weights.setflags(write=False)
    # 返回的 low/high 仍由 FrequencyBand 表示核心，肩部不混入候选标签。
    return weights

# 按固定中心步进铺设候选核心，并用有限输入记录的 1/T 决定最小可解释宽度。
def candidate_frequency_bands(
    settings: AttributionSettings,
    *,
    physical_resolution_hz: float,
) -> tuple[tuple[FrequencyBand, ...], float, tuple[str, ...]]:
    """按用户中心步进生成不夸大输入物理分辨率的候选频段。"""

    # 物理分辨率取自有限脉冲和目标记录 1/T，而非零填充后的显示网格。
    if not np.isfinite(physical_resolution_hz) or physical_resolution_hz <= 0.0:
        # 无有效分辨率时不能声称定位到任何频段。
        raise ValueError("物理频率分辨率必须是正的有限值")
    # 有效核心宽度至少覆盖一份脉冲可独立解析的频宽。
    minimum_width_hz = max(settings.requested_window_hz, physical_resolution_hz)
    # 向上取整到用户步进的整数倍，保持候选边界易读。
    width_steps = int(np.ceil(minimum_width_hz / settings.frequency_step_hz))
    # 浮点乘法恢复最终可见 Hz 宽度。
    effective_width_hz = width_steps * settings.frequency_step_hz
    # 扫描跨度不足一个有效核心时不能生成可解释候选。
    scan_span_hz = settings.scan_high_hz - settings.scan_low_hz
    # 明确失败优于把窗口静默裁到比物理分辨率更窄。
    if effective_width_hz > scan_span_hz * (1.0 + 64.0 * np.finfo(np.float64).eps):
        # 用户可通过扩大扫描范围解决此问题。
        raise ValueError("扫描范围小于当前物理分辨率要求的有效核心宽度")
    # 第一个中心保证完整满权核心从 scan_low 起步。
    first_center_hz = settings.scan_low_hz + 0.5 * effective_width_hz
    # 最后一个规则中心保证完整核心不越过 scan_high。
    last_center_hz = settings.scan_high_hz - 0.5 * effective_width_hz
    # 微小容差避免十进制 MHz 步进累加在末端少一个候选。
    center_tolerance_hz = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(first_center_hz),
        abs(last_center_hz),
    )
    # 用整数候选数而非 arange 浮点终止条件生成确定网格。
    candidate_count = int(
        np.floor(
            (last_center_hz - first_center_hz + center_tolerance_hz)
            / settings.frequency_step_hz
        )
    ) + 1
    # 每个规则中心都保留完整有效核心宽度。
    bands = [
        FrequencyBand(
            low_hz=first_center_hz
            + index * settings.frequency_step_hz
            - 0.5 * effective_width_hz,
            high_hz=first_center_hz
            + index * settings.frequency_step_hz
            + 0.5 * effective_width_hz,
            center_hz=first_center_hz + index * settings.frequency_step_hz,
        )
        for index in range(candidate_count)
    ]
    # 规则步进可能在 scan_high 前留下不足一个步进的尾部。
    tail_gap_hz = settings.scan_high_hz - bands[-1].high_hz
    # 只要尾部大于浮点容差，就追加一个锚定到 scan_high 的完整核心。
    if tail_gap_hz > center_tolerance_hz:
        # 尾部核心允许与倒数第二个核心重叠，但绝不缩窄用户看到的频段。
        tail_low_hz = settings.scan_high_hz - effective_width_hz
        # 中心由完整核心两端的算术平均得到。
        tail_center_hz = 0.5 * (tail_low_hz + settings.scan_high_hz)
        # 追加后扫描高端的每个频率都至少落入一个满权核心。
        bands.append(
            FrequencyBand(
                low_hz=tail_low_hz,
                high_hz=settings.scan_high_hz,
                center_hz=tail_center_hz,
            )
        )
    # 默认没有告警；只有用户请求宽度超出物理能力时追加说明。
    warnings: tuple[str, ...] = ()
    # 使用严格大于加机器容差，避免等于请求宽度时误报。
    if effective_width_hz > settings.requested_window_hz * (
        1.0 + 64.0 * np.finfo(np.float64).eps
    ):
        # 告警同时给出请求值、物理分辨率和最终核心宽度，便于审查。
        warnings = (
            "输入记录的物理频率分辨率不足请求的 "
            f"{settings.requested_window_hz / 1.0e6:g} MHz；"
            f"候选中心仍按 {settings.frequency_step_hz / 1.0e6:g} MHz 步进，"
            f"有效核心宽度扩大为 {effective_width_hz / 1.0e6:g} MHz。",
        )
    # 返回不可变候选、最终核心宽度和可直接展示的告警。
    return tuple(bands), float(effective_width_hz), warnings

# 在复对数域把选中频带的幅度比、相位差或二者组合成可直接乘到 RFFT 的补偿量。
def compose_frequency_correction(
    log_magnitude_ratio: FloatArray,
    phase_ratio_rad: FloatArray,
    weights: FloatArray,
    *,
    mode: AttributionMode,
) -> ComplexArray:
    """在复对数域按平滑权重组合幅度、相位或幅相补偿。"""

    # 三个输入先统一为可审查的一维数组，不在函数内修改调用者缓存。
    log_ratio = np.asarray(log_magnitude_ratio, dtype=np.float64)
    # 连续相位必须由准备阶段先展开并去斜，本函数不对包裹相位猜分支。
    phase_ratio = np.asarray(phase_ratio_rad, dtype=np.float64)
    # 权重限定每个频点施加多少复对数差异。
    band_weights = np.asarray(weights, dtype=np.float64)
    # 模式只接受界面对应的三个明确选项。
    if mode not in {"magnitude", "phase", "both"}:
        # 未知模式不能默认为联合补偿，否则会混淆归因结论。
        raise ValueError("归因模式必须是 magnitude、phase 或 both")
    # 三个数组必须同为一维且逐频点对齐。
    if (
        log_ratio.ndim != 1
        or phase_ratio.ndim != 1
        or band_weights.ndim != 1
        or log_ratio.shape != phase_ratio.shape
        or log_ratio.shape != band_weights.shape
    ):
        # 形状错误在指数运算前终止，避免 NumPy 广播生成错误二维矩阵。
        raise ValueError("幅度、相位和频段权重必须是同长一维数组")
    # 权重必须是标准 0–1 插值，不能过冲或包含非有限值。
    if not np.all(np.isfinite(band_weights)) or np.any(
        (band_weights < 0.0) | (band_weights > 1.0)
    ):
        # 不做静默裁剪，保留窗函数错误的可检测性。
        raise ValueError("频段权重必须是 0 到 1 的有限值")
    # 只要求实际施加补偿的频点具有对应模式分量，频带外缓存可用 NaN 标记。
    active = band_weights > 0.0
    # 负无穷表示 Href 数值为零且 Hdut 可逆，对纯幅度是合法的零补偿。
    valid_log_ratio = np.isfinite(log_ratio) | np.isneginf(log_ratio)
    # 幅度模式只拒绝 NaN/+Inf，不因合法参考零点或孤立相位误判无效。
    invalid_magnitude = active & ~valid_log_ratio
    # 相位模式只依赖连续相位，不要求幅度分量进入指数。
    invalid_phase = active & ~np.isfinite(phase_ratio)
    # 联合模式要求两个分量都有限，单模式仅检查自身分量。
    if (
        (mode in {"magnitude", "both"} and np.any(invalid_magnitude))
        or (mode in {"phase", "both"} and np.any(invalid_phase))
    ):
        # 上层可捕获此错误并在候选表给出原因。
        raise ValueError("候选频段内存在无法解析的幅度或相位")
    # 默认复指数为零，对应频带外单位补偿。
    exponent = np.zeros(log_ratio.shape, dtype=np.complex128)
    # 幅度与联合模式在对数幅度域插值，半权重自然得到几何均值。
    if mode in {"magnitude", "both"}:
        # 只写活跃点，避免零乘 NaN；负无穷会经 exp 明确映射为零。
        with np.errstate(invalid="ignore"):
            # 正权重乘负无穷仍为负无穷，符合零幅度比的数学极限。
            exponent[active] += band_weights[active] * log_ratio[active]
    # 相位与联合模式按连续相位乘权重，避免 ±pi 边界制造伪跳变。
    if mode in {"phase", "both"}:
        # 复指数的虚部就是需要施加的相位旋转。
        exponent[active] += 1j * band_weights[active] * phase_ratio[active]
    # 复指数一次得到电压幅度比和相位旋转，不使用功率比定义。
    with np.errstate(over="ignore", invalid="ignore"):
        # NumPy exp 保持 complex128 频点数组。
        correction = np.exp(exponent)
    # 极大比值溢出时不能继续 IFFT，以免输出出现看似随机的有限裁剪值。
    if not np.all(np.isfinite(correction)):
        # 提醒调用者缩小频带或避开接近零的 DUT 响应。
        raise ValueError("候选频段响应比超出浮点数值范围")
    # 返回只读副本，幅度、相位、幅相三条曲线之间不能共享原地修改。
    correction.setflags(write=False)
    # 该结果可直接与目标 RFFT 逐频点相乘。
    return correction

# VirtualEyeSettings 锁定调制电平、派生 Np、M 和固定符号源，使所有候选成对比较。
@dataclass(frozen=True)
class VirtualEyeSettings:
    """固定理想符号源和拟合脉冲时序的轻量眼图设置。"""

    # modulation 决定电平数；NRZ 对应 PAM2，PAM4 使用峰值归一化的四个附件电平。
    modulation: Modulation
    # pulse_length_ui 是由完整脉冲样点数除以 M 得到的 Np，用于长度校验和稳态裁剪。
    pulse_length_ui: int
    # samples_per_ui 是 M，用于从采样率推导 UI/波特率并定义 2*M+1 轨迹长度。
    samples_per_ui: int
    # symbol_count 控制条件分布的样本量，不改变拟合脉冲本身。
    symbol_count: int = 4096
    # random_seed 使每个候选频段都在完全相同的符号序列上成对比较。
    random_seed: int = 20260718
    # rail_quantile 是一侧经验边界概率；默认 1% 与本地眼图库正式测量口径一致。
    rail_quantile: float = 0.01

    # 在卷积前校验 UI 几何和统计样本量，避免用不足的轨道样本生成看似稳定的眼指标。
    def __post_init__(self) -> None:
        """在进入卷积前拒绝无法解释的调制和 UI 参数。"""

        # 未知调制会让电平间距和眼数失去定义，因此显式拒绝。
        if self.modulation not in {"nrz", "pam4"}:
            # 错误文字直接告诉界面可用的两个选项。
            raise ValueError("调制格式必须是 NRZ 或 PAM4")
        # bool 是 int 的子类，需要单独排除以防 True 被当作 1 UI。
        integer_values = {
            "Np": self.pulse_length_ui,
            "M": self.samples_per_ui,
            "符号数": self.symbol_count,
            "随机种子": self.random_seed,
        }
        # 逐项校验让界面能报出具体无效字段，而不是模糊的数组错误。
        for label, value in integer_values.items():
            # Np、M 和符号数必须是真正整数；种子允许为零但仍不允许 bool。
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                # 类型错误在进入 NumPy 前终止，避免隐式截断小数。
                raise ValueError(f"{label} 必须是整数")
        # 至少 2 UI 才能同时包含主脉冲与稳态裁剪边界。
        if self.pulse_length_ui < 2:
            # Np 过小时不能把拟合脉冲解释为多 UI 响应。
            raise ValueError("Np 必须至少为 2")
        # M=2 的理想矩形眼 crossing 会全部落入两端 1% 保护区，无法可靠测量眼宽。
        if self.samples_per_ui < 3:
            # 至少三点/UI 才能同时保留中心、内侧边界和端点保护。
            raise ValueError("M 必须至少为 3")
        # 符号序列要在首尾各丢弃 Np 后仍留有充足条件样本。
        if self.symbol_count <= 2 * self.pulse_length_ui + 32:
            # 过短序列无法覆盖各电平的代表性确定性 ISI 组合。
            raise ValueError("符号数必须大于 2*Np+32")
        # 零表示使用实际轨迹包络；正分位仍必须低于中位数。
        if not 0.0 <= self.rail_quantile < 0.5:
            # 越界分位会交换上下轨或产生空定义。
            raise ValueError("眼图轨道分位数必须位于 0（含）和 0.5 之间")

# VirtualEyeResult 同时保存限制眼指标及 2 UI 轨迹，让扫描标量和点选绘图共享同一测量口径。
@dataclass(frozen=True)
class VirtualEyeResult:
    """一份拟合脉冲的指标与可绘制 2 UI 轨迹。"""

    # eye_heights_v 按 NRZ 一眼或 PAM4 下/中/上三眼保存有符号垂直间隙；负值表示轨道重叠。
    eye_heights_v: tuple[float, ...]
    # eye_widths_ui 保存各相邻轨道 41 条水平切片中的最大内侧经验开口；NaN 表示不可测。
    eye_widths_ui: tuple[float, ...]
    # sampling_phase_ui 固定为主光标所在的 0 UI，不再从一 UI 密度图重新寻优。
    sampling_phase_ui: float
    # baud_rate_hz 由 TimeSeries 采样率除以 M 推导，不从额外手工字段猜测。
    baud_rate_hz: float
    # plot_time_ui 是从 -1 UI 到 +1 UI、总长 2*M+1 的共同轨迹横轴。
    plot_time_ui: FloatArray
    # plot_traces_v 每行是一条围绕符号主光标提取的轨迹，最多保留 600 行。
    plot_traces_v: FloatArray
    # plot_trace_indices 是这些轨迹在稳态符号行中的位置，供三幅对比图严格复用。
    plot_trace_indices: NDArray[np.int64]


# _VirtualEyeCache 复用固定符号、稳态索引和冲激频谱，避免每个候选重复构造眼图激励。
@dataclass(frozen=True)
class _VirtualEyeCache:
    """一次影响频段工作区共享的固定符号与卷积缓存。"""

    # settings 锁定派生 Np、M、调制和随机种子，避免缓存被错配复用。
    settings: VirtualEyeSettings
    # levels 在参考、补偿前和所有候选中保持同一峰值归一化电平集。
    levels: FloatArray
    # symbols 是固定种子产生的完整符号序列，每个工作区只构造一次。
    symbols: FloatArray
    # stable_symbol_indices 锁定排除卷积首尾瞬态后的符号范围。
    stable_symbol_indices: NDArray[np.int64]
    # stable_symbols 预先取出稳态区条件电平，候选不再重复高级索引。
    stable_symbols: FloatArray
    # stable_symbol_labels 把已知发送电平映射为 0..K-1，测量时无需重新聚类轨迹。
    stable_symbol_labels: NDArray[np.int64]
    # impulse_spectrum 是上采样符号冲激的固定 RFFT，只需与各脉冲频谱相乘。
    impulse_spectrum: ComplexArray
    # fft_length 选择真实线性卷积的快速长度，不引入循环回卷。
    fft_length: int
    # convolution_samples 是裁去 FFT 补零后应保留的完整线性卷积长度。
    convolution_samples: int
    # plot_time_ui 只与 M 有关，三幅眼图共享 -1 UI 到 +1 UI 的只读横轴。
    plot_time_ui: FloatArray
    # empty_traces 供扫描标量路径复用，明确表示本次没有保留任何绘图轨迹。
    empty_traces: FloatArray
    # empty_trace_indices 与空轨迹成对复用，扫描结果不携带绘图行号。
    empty_trace_indices: NDArray[np.int64]

# 把用户选择的 NRZ/PAM4 映射为附件约定的固定对称电平。
def _modulation_levels(modulation: Modulation) -> FloatArray:
    """返回峰值为一的 NRZ 或 PAM4 固定电平。"""

    # NRZ 就是两电平 PAM2，其相邻距离为 2。
    if modulation == "nrz":
        # 显式 float64 避免后续卷积从整数隐式转型。
        return np.array([-1.0, 1.0], dtype=np.float64)
    # PAM4 严格使用附件中的 -1、-1/3、+1/3、+1 四个电平。
    return np.array(
        [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
        dtype=np.float64,
    )

# 核对拟合脉冲是否满足派生 Np*M 单通道合同，禁止静默裁剪或补零改变 UI 几何。
def _validate_virtual_eye_pulse(
    pulse: TimeSeries,
    settings: VirtualEyeSettings,
) -> None:
    """校验单份拟合脉冲是否符合轻量眼图合同。"""

    # Np*M 来自控制器的整除推导，不允许后续候选静默截剪或补零。
    expected_samples = settings.pulse_length_ui * settings.samples_per_ui
    # 长度不符时直接告知期望和实际样点数，便于检查拟合脉冲与 M。
    if pulse.samples != expected_samples:
        # 拒绝后不会得到一张看似合理但 UI 错位的眼图。
        raise ValueError(
            f"拟合脉冲样点数 {pulse.samples} 不等于 Np*M={expected_samples}"
        )
    # 归因页签第一版只对单电压通道生成眼图，避免隐式混合多通道。
    if pulse.channels != 1:
        # 多通道需要用户显式选择通道，本接口不自行猜测。
        raise ValueError("眼图指标当前只支持单通道拟合脉冲")

# 为一次工作区预计算固定符号激励、稳态条件分组和 RFFT，供参考、DUT 与全部候选复用。
def _prepare_virtual_eye_cache(
    settings: VirtualEyeSettings,
    *,
    cancelled: CancellationCheck | None = None,
) -> _VirtualEyeCache:
    """为一次工作区构造且只构造一次固定眼图激励。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 电平数组同时决定条件分组数和眼的数量。
    levels = _modulation_levels(settings.modulation)
    # 固定种子保证参考、DUT 和所有候选补偿使用同一符号序列。
    generator = np.random.default_rng(settings.random_seed)
    # 等概率选择对称电平，不引入协议特定 PRBS 或编码偏置。
    symbols = np.asarray(
        generator.choice(levels, size=settings.symbol_count, replace=True),
        dtype=np.float64,
    )
    # 在每 UI 起点放置一个符号冲激，其余 M-1 个样点保持为零。
    impulses = np.zeros(
        settings.symbol_count * settings.samples_per_ui,
        dtype=np.float64,
    )
    # 符号值在线性卷积后按拟合脉冲的绝对电压比例映射到波形。
    impulses[:: settings.samples_per_ui] = symbols
    # 脉冲样点数由 Np*M 合同唯一确定。
    pulse_samples = settings.pulse_length_ui * settings.samples_per_ui
    # 线性卷积长度为两输入长度之和减一。
    convolution_samples = impulses.size + pulse_samples - 1
    # real=True 选择适合实数 RFFT 的快速长度，与 fftconvolve 的补零策略一致。
    fft_length = int(scipy_fft.next_fast_len(convolution_samples, real=True))
    # 固定符号冲激只做一次 RFFT，各脉冲只需补做自己的频谱。
    impulse_spectrum = np.asarray(
        scipy_fft.rfft(impulses, n=fft_length),
        dtype=np.complex128,
    )
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 首尾各舍弃 Np 个符号，保证条件分布不包含卷积启动和收尾瞬态。
    stable_symbol_indices = np.arange(
        settings.pulse_length_ui,
        settings.symbol_count - settings.pulse_length_ui,
        dtype=np.int64,
    )
    # 稳态电平只从固定符号序列取出一次。
    stable_symbols = np.asarray(symbols[stable_symbol_indices], dtype=np.float64)
    # 已知发送电平可直接映射成整数标签，不需要像通用眼图库那样重新聚类中心样点。
    stable_symbol_labels = np.asarray(
        np.searchsorted(levels, stable_symbols),
        dtype=np.int64,
    )
    # 确定性包络只需轨道非空；正分位至少需要约 1/q 个条件样本才有尾部意义。
    minimum_level_samples = (
        1
        if settings.rail_quantile == 0.0
        else int(np.ceil(1.0 / settings.rail_quantile))
    )
    # 逐电平计数保留 NRZ 与 PAM4 的同一统计可信度门槛。
    level_sample_counts = np.asarray(
        [np.count_nonzero(stable_symbols == level) for level in levels],
        dtype=np.int64,
    )
    # 任一电平不足都会影响与它相邻的眼，因此不能只跳过那一条轨道。
    if np.any(level_sample_counts < minimum_level_samples):
        # 用户可通过增加符号数解决；错误中同时给出当前最小计数便于判断规模。
        raise ValueError(
            "每个调制电平至少需要 "
            f"{minimum_level_samples} 个稳态符号，当前最少 "
            f"{int(np.min(level_sample_counts))} 个；请增加符号数"
        )
    # 附件眼图横轴固定从 -1 UI 到 +1 UI，并包含两端共 2*M+1 点。
    plot_time_ui = (
        np.arange(
            -settings.samples_per_ui,
            settings.samples_per_ui + 1,
            dtype=np.float64,
        )
        / settings.samples_per_ui
    )
    # 扫描标量路径共享零行二维数组，不保留任何候选绘图轨迹。
    empty_traces = np.empty(
        (0, 2 * settings.samples_per_ui + 1),
        dtype=np.float64,
    )
    empty_trace_indices = np.empty(0, dtype=np.int64)
    # 所有缓存数组均设为只读，候选不能意外改变后续比较的激励。
    for cached_array in (
        levels,
        symbols,
        stable_symbol_indices,
        stable_symbols,
        stable_symbol_labels,
        impulse_spectrum,
        plot_time_ui,
        empty_traces,
        empty_trace_indices,
    ):
        # NumPy 只读标志保护同一工作区内的共享数组。
        cached_array.setflags(write=False)
    # 返回不可变缓存，供准备、扫描和点选路径共用。
    return _VirtualEyeCache(
        settings=settings,
        levels=levels,
        symbols=symbols,
        stable_symbol_indices=stable_symbol_indices,
        stable_symbols=stable_symbols,
        stable_symbol_labels=stable_symbol_labels,
        impulse_spectrum=impulse_spectrum,
        fft_length=fft_length,
        convolution_samples=convolution_samples,
        plot_time_ui=plot_time_ui,
        empty_traces=empty_traces,
        empty_trace_indices=empty_trace_indices,
    )


def _representative_eye_trace_indices(
    traces: FloatArray,
    labels: NDArray[np.int64],
    maximum_traces: int = MAX_EYE_PLOT_TRACES,
) -> NDArray[np.int64]:
    """按发送电平和 0 UI 幅度分位确定性选择绘图轨迹。"""

    trace_count = int(traces.shape[0])
    if trace_count <= maximum_traces:
        return np.arange(trace_count, dtype=np.int64)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    exact_quotas = maximum_traces * label_counts.astype(np.float64) / trace_count
    quotas = np.floor(exact_quotas).astype(np.int64)
    remaining = maximum_traces - int(np.sum(quotas))
    if remaining > 0:
        fractional_order = np.argsort(-(exact_quotas - quotas), kind="stable")
        quotas[fractional_order[:remaining]] += 1

    center_index = traces.shape[1] // 2
    selected: list[NDArray[np.int64]] = []
    for label, quota in zip(unique_labels, quotas, strict=True):
        label_indices = np.flatnonzero(labels == label)
        center_values = traces[label_indices, center_index]
        tie_tolerance = max(1.0, float(np.max(np.abs(center_values)))) * 1.0e-12
        stable_center_ranks = np.rint(center_values / tie_tolerance)
        ordered = label_indices[
            np.argsort(stable_center_ranks, kind="stable")
        ]
        ranks = np.rint(
            np.linspace(0, ordered.size - 1, int(quota), dtype=np.float64)
        ).astype(np.int64)
        selected.append(ordered[ranks])
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)

# 在共享激励上卷积一份脉冲，并围绕冻结原点计算可公平比较的眼高、眼宽。
def _build_virtual_eye_from_cache(
    pulse: TimeSeries,
    cache: _VirtualEyeCache,
    *,
    sampling_phase_index: int | None = None,
    main_index: int | None = None,
    amplitude_normalizer_v: float | None = None,
    include_plot: bool = True,
    measure_width: bool = True,
    plot_trace_indices: NDArray[np.int64] | None = None,
    cancelled: CancellationCheck | None = None,
) -> VirtualEyeResult:
    """使用共享固定激励构造 2 UI 轨迹并计算眼指标。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 每份脉冲仍独立校验长度和通道，缓存不放宽公共合同。
    _validate_virtual_eye_pulse(pulse, cache.settings)
    # 未传入原点时由当前脉冲的最大绝对样点选择，作为该脉冲的主抽头。
    if main_index is None:
        # standalone 调用以自身最高绝对峰值建立 0 UI 原点。
        selected_main_index = int(np.argmax(np.abs(pulse.values[:, 0])))
    # 候选评估传入 DUT 基线原点，禁止补偿后重新对齐。
    else:
        # bool 不是有意义的样点索引，同时要求原点位于脉冲记录内。
        if isinstance(main_index, (bool, np.bool_)) or not 0 <= int(
            main_index
        ) < pulse.samples:
            # 错误不静默裁剪，避免一个越界原点改变眼图稳态区。
            raise ValueError("主脉冲原点必须位于拟合脉冲样点范围内")
        # 显式转换同时兼容 Python 与 NumPy 整数。
        selected_main_index = int(main_index)
    # standalone 以自身主光标归一化；设备比较会显式传入参考主光标作为共同基准。
    if amplitude_normalizer_v is None:
        # 保留主光标符号，负向脉冲归一化后仍得到正主光标而不是只除绝对值。
        selected_normalizer_v = float(
            pulse.values[selected_main_index, 0]
        )
    # 工作区传入参考主光标幅度，使 DUT 相对增益不会被各自归一化抹掉。
    else:
        # 显式转换兼容 Python 与 NumPy 标量。
        selected_normalizer_v = float(amplitude_normalizer_v)
    # 零或非有限主光标无法定义稳定的幅度归一化基准。
    if not np.isfinite(selected_normalizer_v) or selected_normalizer_v == 0.0:
        # 明确失败优于生成 Inf/NaN 眼图轨迹。
        raise ValueError("眼图主光标幅度必须是有限非零值")
    # 拟合脉冲先按共同主光标基准归一化，再与同一符号冲激列卷积。
    normalized_pulse = np.asarray(
        pulse.values[:, 0] / selected_normalizer_v,
        dtype=np.float64,
    )
    # 每份脉冲只计算自身 RFFT，固定冲激频谱直接从工作区复用。
    pulse_spectrum = scipy_fft.rfft(
        normalized_pulse,
        n=cache.fft_length,
    )
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 频域相乘后指定原 FFT 长度还原实数线性卷积。
    padded_waveform = scipy_fft.irfft(
        cache.impulse_spectrum * pulse_spectrum,
        n=cache.fft_length,
    )
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 只保留真实线性卷积区，丢弃快速 FFT 长度的尾部补零。
    waveform = np.asarray(
        padded_waveform[: cache.convolution_samples],
        dtype=np.float64,
    )
    # 第一条轨迹中心是首个稳态符号输出位置，即符号冲激位置加固定主光标延时。
    first_center = (
        int(cache.stable_symbol_indices[0]) * cache.settings.samples_per_ui
        + selected_main_index
    )
    # 最后一条轨迹中心使用同一冻结原点，不让候选按新峰值移动。
    last_center = (
        int(cache.stable_symbol_indices[-1]) * cache.settings.samples_per_ui
        + selected_main_index
    )
    # 每条轨迹从中心前一 UI 开始。
    trace_start = first_center - cache.settings.samples_per_ui
    # 半开区间终点越过最后中心后一 UI 一个样点，从而包含 +1 UI 端点。
    trace_stop = last_center + cache.settings.samples_per_ui + 1
    # 正常稳态裁剪应保证完整窗口仍位于线性卷积内；防御检查避免负索引静默回卷。
    if trace_start < 0 or trace_stop > waveform.size:
        # 若未来修改稳态裁剪，此错误会直接暴露附件轨迹窗口不完整。
        raise RuntimeError("稳态符号不足以提取完整的 -1 UI 到 +1 UI 轨迹")
    # 先取覆盖全部稳态中心的连续波形片段，避免为每条轨迹构造大型整数索引。
    trace_region = waveform[trace_start:trace_stop]
    # 滑窗视图产生所有可能的 2*M+1 点窗口，本身不复制波形。
    all_windows = np.lib.stride_tricks.sliding_window_view(
        trace_region,
        2 * cache.settings.samples_per_ui + 1,
    )
    # 每隔 M 点取一个符号中心，得到与 stable_symbols 一一对应的 2 UI 轨迹。
    traces = np.asarray(
        all_windows[:: cache.settings.samples_per_ui],
        dtype=np.float64,
    )
    # 轨迹行数偏离稳态符号数表示窗口端点或步进计算发生了回归。
    if traces.shape != (
        cache.stable_symbol_indices.size,
        2 * cache.settings.samples_per_ui + 1,
    ):
        # 不允许用截断轨迹继续计算条件分位数。
        raise RuntimeError("眼图轨迹数量或长度与稳态符号几何不一致")
    # 新轨迹口径只允许主光标 0 UI 采样；旧调用传非零相位必须明确失败而非静默忽略。
    if sampling_phase_index is not None and (
        isinstance(sampling_phase_index, (bool, np.bool_))
        or int(sampling_phase_index) != 0
    ):
        # 用户附件没有重新寻优相位的步骤，主光标就是唯一测量中心。
        raise ValueError("2 UI 轨迹眼图的采样相位固定为 0 UI")
    # 折叠后的轨迹交给独立轻量测量内核；它不再执行 CDR、聚类或最佳中心寻优。
    openings = measure_virtual_eye_openings(
        traces,
        cache.plot_time_ui,
        cache.stable_symbol_labels,
        cache.levels,
        opening_probability=cache.settings.rail_quantile,
        measure_width=measure_width,
    )
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 眼高在固定 0 UI 读取相邻 rail 内侧经验边界之差，负值保留为闭眼证据。
    eye_heights_v = openings.eye_heights_v
    # 眼宽使用 41 条固定电压水平切片的最大内侧 crossing 开口，单位为 UI。
    eye_widths_ui = openings.eye_widths_ui
    # 完整评估保留附件上限内的真实轨迹；扫描只要标量时复用零行数组。
    if include_plot:
        # 横轴完全由缓存共享，参考、补偿前和点选候选严格对齐。
        plot_time_ui = cache.plot_time_ui
        # 按发送电平和中心幅度分位覆盖完整记录，避免“前 600 条”遗漏后段尾部事件。
        if plot_trace_indices is None:
            plot_indices = _representative_eye_trace_indices(
                traces,
                cache.stable_symbol_labels,
            )
        else:
            plot_indices = np.asarray(plot_trace_indices, dtype=np.int64)
            if (
                plot_indices.ndim != 1
                or plot_indices.size > MAX_EYE_PLOT_TRACES
                or np.any(plot_indices < 0)
                or np.any(plot_indices >= traces.shape[0])
                or np.unique(plot_indices).size != plot_indices.size
            ):
                raise ValueError("共享眼图轨迹索引必须是一维、唯一且位于稳态轨迹范围内")
        stored_plot_indices = np.array(plot_indices, dtype=np.int64, copy=True)
        stored_plot_indices.setflags(write=False)
        plot_traces_v = np.array(
            traces[stored_plot_indices],
            dtype=np.float64,
            copy=True,
        )
        # 二维轨迹设为只读，确保候选切换时基线不会被图层修改。
        plot_traces_v.setflags(write=False)
    # 扫描时不复制或保留候选轨迹，仅共享很小的固定时间轴。
    else:
        # 时间轴仍表明指标使用的是 -1 UI 到 +1 UI 的统一几何。
        plot_time_ui = cache.plot_time_ui
        # 零行数组明确证明扫描结果没有持有任何候选轨迹。
        plot_traces_v = cache.empty_traces
        stored_plot_indices = cache.empty_trace_indices
    # 返回的指标契约与公共 build_virtual_eye 保持一致。
    return VirtualEyeResult(
        eye_heights_v=eye_heights_v,
        eye_widths_ui=eye_widths_ui,
        sampling_phase_ui=0.0,
        baud_rate_hz=pulse.sample_rate_hz
        / float(cache.settings.samples_per_ui),
        plot_time_ui=plot_time_ui,
        plot_traces_v=plot_traces_v,
        plot_trace_indices=stored_plot_indices,
    )

# 提供独立的轻量眼公共入口，适合不经过完整频段扫描时直接测量一份拟合脉冲。
def build_virtual_eye(
    pulse: TimeSeries,
    settings: VirtualEyeSettings,
    *,
    sampling_phase_index: int | None = None,
    cancelled: CancellationCheck | None = None,
) -> VirtualEyeResult:
    """用固定符号源对拟合脉冲做线性卷积并计算虚拟眼经验开口。"""

    # 公共单次接口先保持原有的脉冲校验顺序。
    _validate_virtual_eye_pulse(pulse, settings)
    # 单独调用仍构造一份局部缓存，不要求调用者管理内部状态。
    cache = _prepare_virtual_eye_cache(settings, cancelled=cancelled)
    # 默认始终返回完整绘图数据，保持现有公共行为。
    return _build_virtual_eye_from_cache(
        pulse,
        cache,
        sampling_phase_index=sampling_phase_index,
        include_plot=True,
        cancelled=cancelled,
    )

# BandAttribution 只保存一个频段和一种补偿模式的标量证据，便于全扫描低内存排名。
@dataclass(frozen=True)
class BandAttribution:
    """一个候选频段、一个补偿模式的标量归因结果。"""

    # band 保存真实应用的有效窗边界，而非只保存显示中心。
    band: FrequencyBand
    # mode 区分幅度、相位及幅相共同作用。
    mode: AttributionMode
    # metric_after 是该局部补偿后的 Vpp、限制眼高或限制眼宽。
    metric_after: float
    # improvement 是补偿前参考误差减去补偿后参考误差，可为负数。
    improvement: float
    # recovery_ratio 以补偿前参考差距为分母，不裁剪到 0–1。
    recovery_ratio: float
    # valid=False 表示当前频带落在不可解析谱零点或没有实际 DFT 频点。
    valid: bool
    # invalid_reason 给界面候选表提供可读原因；有效候选为空字符串。
    invalid_reason: str = ""

# BandEvaluation 在标量证据之外按需携带补偿波形和眼图，供默认候选或用户点选重放。
@dataclass(frozen=True)
class BandEvaluation:
    """候选标量结果及按需绘图所需的补偿后波形。"""

    # attribution 可直接进入排名表和影响曲线。
    attribution: BandAttribution
    # corrected_values 仅为单次点选或当前评估保留，不在全扫描中批量缓存。
    corrected_values: FloatArray | None
    # eye_after 在眼指标下保存冻结 DUT 原点后的 2 UI 轨迹眼结果。
    eye_after: VirtualEyeResult | None = None

# FrequencyAttributionResult 汇总全频闭环、局部贡献曲线和保守推荐，不缓存所有候选波形。
@dataclass(frozen=True)
class FrequencyAttributionResult:
    """完整扫描结果，不包含每个候选的大型波形数组。"""

    # reference_metric 是参考设备的目标标量。
    reference_metric: float
    # before_metric 是 DUT 补偿前标量。
    before_metric: float
    # full_band_results 用来验证幅度、相位或幅相在全扫描带是否真能缩小差距。
    full_band_results: tuple[BandAttribution, ...]
    # candidates 依次按频段中心和模式保存，便于绘制三条贡献曲线。
    candidates: tuple[BandAttribution, ...]
    # recommendation 为空表示没有超过数值容差的有效正改善局部频段。
    recommendation: BandAttribution | None
    # effective_frequency_resolution_hz 来自有限脉冲 1/T，不是零填充网格。
    effective_frequency_resolution_hz: float
    # effective_window_width_hz 是分辨率不足时扩大后的满权核心宽度。
    effective_window_width_hz: float
    # status 用简短内部键区分成功、无差距、无推荐或取消。
    status: Literal["ok", "no_difference", "no_recommendation", "cancelled"]
    # warnings 保留物理分辨率降级等需要展示但不使扫描失败的边界。
    warnings: tuple[str, ...]

# EyeComparisonData 约束三幅轨迹眼使用同一横轴与纵轴范围，避免自动缩放制造视觉误判。
@dataclass(frozen=True)
class EyeComparisonData:
    """参考、补偿前、补偿后三幅共轴的 2 UI 眼图轨迹。"""

    # time_ui 是三幅图完全相同的 -1 UI 到 +1 UI、2*M+1 点横轴。
    time_ui: FloatArray
    # reference_traces_v 保存参考设备最多 600 条二维轨迹。
    reference_traces_v: FloatArray
    # before_traces_v 保存 DUT 补偿前的同一符号位置轨迹。
    before_traces_v: FloatArray
    # after_traces_v 保存当前候选补偿后的同一符号位置轨迹。
    after_traces_v: FloatArray
    # amplitude_range_v 是三组轨迹共同决定且留有显示余量的纵轴范围。
    amplitude_range_v: tuple[float, float]
    # sampling_phase_ui 固定为三幅图共同的 0 UI 主光标。
    sampling_phase_ui: float
    # reference/before/after 保留指标值，界面无需从像素反推眼高眼宽。
    reference: VirtualEyeResult
    # 补偿前眼用于候选切换时保持基线不变。
    before: VirtualEyeResult
    # 补偿后眼对应当前选中候选。
    after: VirtualEyeResult

# PreparedAttribution 固化频响比、目标频谱、候选窗及基线指标，供后台扫描重复只读使用。
@dataclass(frozen=True)
class PreparedAttribution:
    """一次扫描共享的频响、目标频谱与基线指标缓存。"""

    # 两份脉冲决定需要从 DUT 修到参考的复频响比。
    reference_pulse: TimeSeries
    # DUT 拟合脉冲与参考必须同长度，采样率差异不得超过公共容差。
    dut_pulse: TimeSeries
    # target_signal 在眼指标下就是 DUT 脉冲，在 Vpp 下是 DUT 原始波形。
    target_signal: TimeSeries
    # reference_waveform 只在 Vpp 模式下存在，用于定义目标标量。
    reference_waveform: TimeSeries | None
    # settings 保存指标、扫描边界和眼图几何。
    settings: AttributionSettings
    # eye_cache 在眼指标下只构造一次固定符号激励；Vpp 模式为空。
    eye_cache: _VirtualEyeCache | None
    # vpp_cache 在新 Vpp 模式下保存同一码型的参考/DUT 周期频谱与指标。
    vpp_cache: VppAnalysisCache | None
    # frequency_hz 是镜像延拓目标记录的真实 RFFT 频率轴。
    frequency_hz: FloatArray
    # base_spectrum 只计算一次，候选循环只乘不同复补偿。
    base_spectrum: ComplexArray
    # log_magnitude_ratio 保存 log(|Href|/|Hdut|)。
    log_magnitude_ratio: FloatArray
    # phase_ratio_rad 保存去整体时延后的连续相位差。
    phase_ratio_rad: FloatArray
    # magnitude_valid_mask 标记两份脉冲幅度都能形成有限比值的频点。
    magnitude_valid_mask: NDArray[np.bool_]
    # phase_valid_mask 标记连续相位也能可靠展开和锚定的频点。
    phase_valid_mask: NDArray[np.bool_]
    # padding 是镜像延拓后裁回原记录所需的左偏移。
    padding: int
    # original_samples 用于从 IFFT 中准确取回原长度。
    original_samples: int
    # candidates 是已经结合物理分辨率生成的候选频段。
    candidates: tuple[FrequencyBand, ...]
    # physical_resolution_hz 为 max(Fs/N)，不采用延拓或零填充后的细网格。
    physical_resolution_hz: float
    # effective_window_width_hz 与候选边界一一对应。
    effective_window_width_hz: float
    # reference_metric 和 before_metric 在所有候选间固定。
    reference_metric: float
    # DUT 补偿前指标只计算一次。
    before_metric: float
    # reference_eye 只在眼模式存在，并把自身主光标定义为 0 UI。
    reference_eye: VirtualEyeResult | None
    # before_eye 使用 DUT 固定主光标原点和参考公共幅度基准。
    before_eye: VirtualEyeResult | None
    # sampling_phase_index 在 2 UI 轨迹口径中固定为主光标相位零。
    sampling_phase_index: int | None
    # reference_eye_main_index 是参考脉冲自身选定的固定样点原点。
    reference_eye_main_index: int | None
    # dut_eye_main_index 在 DUT 基线选择一次，所有补偿候选必须复用。
    dut_eye_main_index: int | None
    # eye_amplitude_normalizer_v 在眼模式保存参考主光标，三份设备轨迹共同除以它。
    eye_amplitude_normalizer_v: float | None
    # warnings 汇总物理分辨率和模型边界提示。
    warnings: tuple[str, ...]

# 用专用异常区分用户主动取消与算法输入失败，避免后台把取消显示成错误。
class AttributionCancelled(RuntimeError):
    """后台用户主动取消扫描时使用的轻量控制流异常。"""

# 将 NRZ 一只眼或 PAM4 三只眼压成最差开口，确保排名不会被较好眼平均掩盖。
def _limiting_eye_metric(
    eye: VirtualEyeResult,
    metric: AttributionMetric,
) -> float:
    """把 NRZ 一眼或 PAM4 三眼压缩为最差眼的单次比较标量。"""

    # 眼高取所有眼的最小值，避免 PAM4 中一只闭眼被平均值掩盖。
    if metric == "eye_height":
        # tuple 至少包含 NRZ 一眼或 PAM4 三眼，由构建函数保证非空。
        return float(min(eye.eye_heights_v))
    # 眼宽同样取限制眼，并保持单位为 UI。
    if metric == "eye_width":
        # NaN 表示某只眼缺少足够 crossing，不能被当成数值零进入排名。
        if not all(np.isfinite(width) for width in eye.eye_widths_ui):
            # 基线会明确失败，单个候选则由评估层降级为不可解析断点。
            raise ValueError("眼宽不可测：至少一只眼缺少足够的左右 crossing")
        # 所有候选使用同一 M 和冻结相位，有限宽度可直接比较。
        return float(min(eye.eye_widths_ui))
    # Vpp 不应通过眼图路径测量。
    raise ValueError("Vpp 指标不能从眼图结果计算")

# 把外部标量或数组复制成有限只读 float64，保护参考与候选缓存不被图层原地修改。
def _readonly_float(values: object) -> FloatArray:
    """复制为有限 float64 只读数组。"""

    # 候选波形和 2 UI 轨迹都需要与后台缓存隔离。
    array = np.array(values, dtype=np.float64, copy=True)
    # NaN/Inf 会使指标和 Qt 色阶不可解释。
    if not np.all(np.isfinite(array)):
        # 直接拒绝而不把坏点替换为零。
        raise ValueError("归因数组包含 NaN 或 Inf")
    # 只读保护候选切换时的参考与补偿前基线。
    array.setflags(write=False)
    # 保留调用者原有维度。
    return array

# 把频域缓存复制成有限只读 complex128，阻止候选补偿污染后续模式的基础频谱。
def _readonly_complex(values: object) -> ComplexArray:
    """复制为有限 complex128 只读数组。"""

    # 频谱允许二维通道轴，故不强制具体维数。
    array = np.array(values, dtype=np.complex128, copy=True)
    # 实部或虚部非有限都会污染 IFFT。
    if not np.all(np.isfinite(array)):
        # 频响比溢出应在候选层明确失败。
        raise ValueError("归因复频谱包含 NaN 或 Inf")
    # 后台候选循环只读取基础频谱。
    array.setflags(write=False)
    # 返回独立缓存。
    return array

# 按共同记录时长规则测量参考与 DUT 原始波形的稳健 Vpp，允许两份数据长度和采样率不同。
def _measure_vpp_baseline(
    reference_waveform: TimeSeries,
    dut_waveform: TimeSeries,
) -> tuple[float, float]:
    """延迟导入 Vpp 模块并取得不同采样率、不同长度记录的可比标量。"""

    # 局部导入避免单独使用眼图算法时强制加载 Vpp 辅助实现。
    from .attribution_metrics import compare_waveform_vpp

    # 比较器按较短记录时长分块，不做逐点对齐或重采样。
    comparison = compare_waveform_vpp(reference_waveform, dut_waveform)
    # 两个标量已使用同一时间尺度规则，可直接构造参考差距。
    return float(comparison.reference_vpp), float(comparison.dut_vpp)

# 用准备阶段锁定的分块时长测量补偿后 DUT Vpp，使候选值与基线保持相同统计口径。
def _measure_corrected_vpp(
    corrected_values: FloatArray,
    workspace: PreparedAttribution,
) -> float:
    """按准备阶段锁定的共同记录时长测量补偿后 DUT Vpp。"""

    # 局部导入保持 Vpp 实现与频响扫描器职责分离。
    from .attribution_metrics import measure_waveform_vpp

    # 参考和 DUT 原始波形在 Vpp 模式均由 prepare 强制提供。
    if workspace.reference_waveform is None:
        # 内部状态不完整时拒绝返回伪标量。
        raise RuntimeError("Vpp 工作区缺少参考原始波形")
    # 较短记录的有效时长按样点覆盖 N/Fs 定义，和分块样点数完全一致。
    common_duration_s = min(
        workspace.reference_waveform.samples / workspace.reference_waveform.sample_rate_hz,
        workspace.target_signal.samples / workspace.target_signal.sample_rate_hz,
    )
    # 补偿后数组沿用 DUT 时间轴和元数据，避免额外重采样。
    corrected_series = TimeSeries(
        workspace.target_signal.time_s,
        corrected_values,
        workspace.target_signal.sample_rate_hz,
        source_format="memory",
    )
    # 与补偿前完全相同的块时长保证候选排名是成对比较。
    return float(
        measure_waveform_vpp(
            corrected_series,
            block_duration_s=common_duration_s,
        )
    )

# 在目标 RFFT 网格上计算参考/DUT 的复频响比，并分离可解释的幅度、连续相位及有效掩码。
def _prepare_response_ratio(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    frequency_hz: FloatArray,
    settings: AttributionSettings,
    *,
    model_peak_delay_s: float = 0.0,
    cancelled: CancellationCheck | None = None,
) -> tuple[
    FloatArray,
    FloatArray,
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """在目标 DFT 网格一次求出连续复频响比及可信掩码。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 只计算扫描范围内频点，频带外最终保持单位补偿。
    scan_mask = (
        (frequency_hz >= settings.scan_low_hz)
        & (frequency_hz <= settings.scan_high_hz)
    )
    # 目标记录过短导致扫描范围没有任何 DFT 点时明确失败。
    if not np.any(scan_mask):
        # 用户可加长目标记录或扩大扫描范围。
        raise ValueError("目标记录频率分辨率不足，扫描范围内没有 DFT 频点")
    # 选中频点仍保持 RFFT 的均匀递增间隔，可直接交给 CZT 路径。
    scan_frequency_hz = np.asarray(frequency_hz[scan_mask], dtype=np.float64)
    # 相位拟合范围为空时沿用完整扫描范围。
    phase_fit_low_hz = (
        settings.scan_low_hz
        if settings.phase_fit_low_hz is None
        else settings.phase_fit_low_hz
    )
    # 上限同理沿用扫描上限。
    phase_fit_high_hz = (
        settings.scan_high_hz
        if settings.phase_fit_high_hz is None
        else settings.phase_fit_high_hz
    )
    # CompensationSettings 在此只承载脉冲窗参数；候选平滑由满权核心和余弦肩部负责。
    response_settings = CompensationSettings(
        mode="both",
        band_low_hz=settings.scan_low_hz,
        band_high_hz=settings.scan_high_hz,
        phase_fit_low_hz=phase_fit_low_hz,
        phase_fit_high_hz=phase_fit_high_hz,
        detrend_phase=settings.detrend_phase,
        taper_alpha=0.0,
        analysis_points=257,
    )
    # 脉冲绝对值积分给直接 DTFT 数值门限一个保守峰值上界。
    reference_peak = float(
        np.sum(np.abs(reference_pulse.values[:, 0]), dtype=np.longdouble)
        / reference_pulse.sample_rate_hz
    )
    # DUT 使用自身绝对值积分，避免两设备整体增益差影响零点判定。
    dut_peak = float(
        np.sum(np.abs(dut_pulse.values[:, 0]), dtype=np.longdouble)
        / dut_pulse.sample_rate_hz
    )
    # 参考响应在目标精确 DFT 频点求值，不插值显示网格。
    reference_response, reference_magnitude, reference_valid = (
        _pulse_response_on_uniform_frequencies(
            reference_pulse,
            scan_frequency_hz,
            response_settings,
            reference_peak=reference_peak,
            cancelled=cancelled,
        )
    )
    # DUT 响应使用完全相同的物理频率轴。
    dut_response, dut_magnitude, dut_valid = _pulse_response_on_uniform_frequencies(
        dut_pulse,
        scan_frequency_hz,
        response_settings,
        reference_peak=dut_peak,
        cancelled=cancelled,
    )
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 幅度比只有 DUT 分母为数值零点时不可逆；参考零点对应合法的零比。
    local_magnitude_valid = np.asarray(dut_valid, dtype=np.bool_)
    # 先分配 NaN，使频带外和谱零点不会因零乘权重渗入指数。
    local_log_ratio = np.full(scan_frequency_hz.shape, np.nan, dtype=np.float64)
    # 两侧都可解析的普通频点使用电压幅度自然对数。
    regular_magnitude = reference_valid & dut_valid
    # 普通频点的半权重自然得到几何均值。
    local_log_ratio[regular_magnitude] = np.log(
        reference_magnitude[regular_magnitude]
    ) - np.log(
        dut_magnitude[regular_magnitude]
    )
    # 参考数值零点而 DUT 非零时，log(0/Hdut) 的数学值为负无穷。
    reference_zero = (~reference_valid) & dut_valid
    # 显式负无穷让纯幅度 compose 生成精确零，而不是伪造任意小正数。
    local_log_ratio[reference_zero] = -np.inf
    # 数组索引时移的包裹相位先由复乘得到，符号固定为 Href/Hdut。
    wrapped_phase = np.angle(reference_response * np.conj(dut_response))
    # CSV 时间轴起点差属于坐标原点而非设备频响，需要显式抵消。
    delta_t0_s = float(dut_pulse.time_s[0] - reference_pulse.time_s[0])
    # 加回起点差后得到设备本身的连续相位差候选。
    wrapped_phase += 2.0 * np.pi * scan_frequency_hz * delta_t0_s
    # 周期 Vpp 模型把两条脉冲各自的 pmax 定义为 lag=0；频响比必须移除同一峰值时移。
    if not np.isfinite(model_peak_delay_s):
        raise ValueError("Vpp 模型峰值时延必须是有限秒数")
    wrapped_phase -= 2.0 * np.pi * scan_frequency_hz * model_peak_delay_s
    # 归一化置信度仅用于展开锚点和线性拟合加权，不做人为工程门限。
    tiny = np.finfo(np.float64).tiny
    # 各自归一化后取较弱一方，避免深衰减点主导相位斜率。
    confidence = np.minimum(
        reference_magnitude / max(float(np.max(reference_magnitude)), tiny),
        dut_magnitude / max(float(np.max(dut_magnitude)), tiny),
    )
    # 相位仍要求参考和 DUT 两侧都可解析，不能给参考零点编造相位。
    local_phase_seed = reference_valid & dut_valid
    # 每个连续可信岛独立展开，不跨谱零点传播相位圈数。
    unwrapped_phase = _segmented_unwrap(
        wrapped_phase,
        local_phase_seed,
        confidence,
    )
    # 展开后仍为 NaN 的孤立单点不支持平滑相位补偿。
    local_phase_valid = local_phase_seed & np.isfinite(unwrapped_phase)
    # 默认斜率为零；关闭去斜时保留完整设备相位差。
    phase_slope_rad_per_hz = 0.0
    # 去整体时延时仅在用户拟合带和可信点上估计共同斜率。
    if settings.detrend_phase:
        # 拟合掩码保留至少三个物理频点的明确约束。
        fit_mask = (
            local_phase_valid
            & (scan_frequency_hz >= phase_fit_low_hz)
            & (scan_frequency_hz <= phase_fit_high_hz)
        )
        # 相位点不足时不能可靠区分整体时延与局部相位差。
        if int(np.count_nonzero(fit_mask)) < 3:
            # 提示用户扩大拟合或扫描范围。
            raise ValueError("相位去斜频带内没有至少三个可信频点")
        # 复用岛感知的无截距加权斜率拟合，与现有补偿页口径一致。
        phase_slope_rad_per_hz = fit_linear_phase_slope(
            scan_frequency_hz,
            unwrapped_phase,
            confidence**2,
            fit_mask,
        )
    # 从连续相位中减去整体线性时延；关闭时斜率为零。
    detrended_phase = unwrapped_phase - phase_slope_rad_per_hz * scan_frequency_hz
    # 每个可信岛减整数个 2π，保留复响应同时给平滑权重连续分支。
    local_phase_ratio = _anchor_phase_islands(
        detrended_phase,
        local_phase_valid,
        confidence**2,
    )
    # 锚定失败点从可信集合剔除。
    local_phase_valid &= np.isfinite(local_phase_ratio)
    # 完整目标频率轴的幅度缓存默认 NaN。
    log_ratio = np.full(frequency_hz.shape, np.nan, dtype=np.float64)
    # 完整相位缓存同样只在扫描可信点有值。
    phase_ratio = np.full(frequency_hz.shape, np.nan, dtype=np.float64)
    # 完整幅度可信掩码的频带外保持 False。
    magnitude_valid = np.zeros(frequency_hz.shape, dtype=np.bool_)
    # 相位可信掩码单独保存，不能让相位孤点阻断纯幅度扫描。
    phase_valid = np.zeros(frequency_hz.shape, dtype=np.bool_)
    # 将局部计算结果映射回目标 RFFT 轴。
    log_ratio[scan_mask] = local_log_ratio
    # 连续去斜相位写回相同位置。
    phase_ratio[scan_mask] = local_phase_ratio
    # 幅度可信状态同步映射。
    magnitude_valid[scan_mask] = local_magnitude_valid
    # 相位可信状态同步映射。
    phase_valid[scan_mask] = local_phase_valid
    # 三份缓存都设为只读，候选循环不能修改准备结果。
    log_ratio.setflags(write=False)
    # 相位缓存同样只读。
    phase_ratio.setflags(write=False)
    # 幅度掩码锁定，防止候选循环原地改写。
    magnitude_valid.setflags(write=False)
    # 相位掩码同样锁定并保持与幅度分离。
    phase_valid.setflags(write=False)
    # 返回一次计算、所有候选共享的复对数分量与模式专属掩码。
    return log_ratio, phase_ratio, magnitude_valid, phase_valid

# 一次性准备脉冲响应比、目标频谱、候选核心和参考基线，建立后续扫描的只读工作区。
def prepare_frequency_attribution(
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
    settings: AttributionSettings,
    *,
    reference_waveform: TimeSeries | None = None,
    dut_waveform: TimeSeries | None = None,
    prepared_vpp_pattern_levels: object | None = None,
    cancelled: CancellationCheck | None = None,
) -> PreparedAttribution:
    """校验输入并预计算频响、目标频谱、候选几何和基线指标。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    if settings.metric != "vpp" and prepared_vpp_pattern_levels is not None:
        raise ValueError("预加载理想码型只能用于 Vpp 指标")
    # 首版归因只处理单通道拟合脉冲，避免自动混合通道。
    if reference_pulse.channels != 1 or dut_pulse.channels != 1:
        # 后续如需多通道应在界面增加明确通道选择。
        raise ValueError("影响频段当前只支持单通道拟合脉冲")
    # 用户确认两份拟合脉冲等长；不允许在算法内静默补零对齐。
    if reference_pulse.samples != dut_pulse.samples:
        # 错误说明区别于原始 Vpp 波形可以不同长度。
        raise ValueError("两份拟合脉冲必须等长")
    # 跨文件只容忍导出舍入级差异；不在此重采样，避免改变频响相位。
    validate_cross_pulse_sample_rates(
        reference_pulse.sample_rate_hz,
        dut_pulse.sample_rate_hz,
        subject="两份拟合脉冲的",
    )
    # 眼指标直接补偿 DUT 拟合脉冲并生成轻量眼图。
    if settings.metric in {"eye_height", "eye_width"}:
        # 类型验证已保证 eye 非空，此判断使静态和运行时都明确。
        if settings.eye is None:
            # 防御性错误不应在正常构造的设置中触发。
            raise RuntimeError("眼图设置缺失")
        # Vpp 原始波形参数在眼模式不参与计算。
        target_signal = dut_pulse
        # 参考脉冲以自身最大绝对峰值选定主抽头。
        reference_eye_main_index = int(
            np.argmax(np.abs(reference_pulse.values[:, 0]))
        )
        # DUT 基线同样只选择一次最高峰；全部补偿候选复用该冻结原点。
        dut_eye_main_index = int(np.argmax(np.abs(dut_pulse.values[:, 0])))
        # 参考主光标的带符号幅度是三份实际设备脉冲共同的归一化基准。
        eye_amplitude_normalizer_v = float(
            reference_pulse.values[reference_eye_main_index, 0]
        )
        # 全零或非有限参考脉冲不能提供可解释的共同幅度标尺。
        if (
            not np.isfinite(eye_amplitude_normalizer_v)
            or eye_amplitude_normalizer_v == 0.0
        ):
            # 在构造符号卷积前停止，避免后续轨迹充满 NaN/Inf。
            raise ValueError("参考拟合脉冲的主光标幅度必须是有限非零值")
        # 固定种子符号、稳态索引和冲激 FFT 每个工作区只构造一次。
        eye_cache = _prepare_virtual_eye_cache(settings.eye, cancelled=cancelled)
        # 附件轨迹把主光标直接定义为 0 UI，因此冻结相位索引恒为零。
        sampling_phase_index = 0
        # 先由 DUT 补偿前眼按电平和中心幅度分位选择一次代表性符号位置。
        before_eye = _build_virtual_eye_from_cache(
            dut_pulse,
            eye_cache,
            sampling_phase_index=sampling_phase_index,
            main_index=dut_eye_main_index,
            amplitude_normalizer_v=eye_amplitude_normalizer_v,
            include_plot=True,
            measure_width=settings.metric == "eye_width",
            cancelled=cancelled,
        )
        # 参考眼严格复用同一组符号位置，三联图的视觉差异不再混入抽样差异。
        reference_eye = _build_virtual_eye_from_cache(
            reference_pulse,
            eye_cache,
            main_index=reference_eye_main_index,
            amplitude_normalizer_v=eye_amplitude_normalizer_v,
            include_plot=True,
            measure_width=settings.metric == "eye_width",
            plot_trace_indices=before_eye.plot_trace_indices,
            cancelled=cancelled,
        )
        # 参考标量取限制眼。
        reference_metric = _limiting_eye_metric(reference_eye, settings.metric)
        # DUT 基线同样取限制眼。
        before_metric = _limiting_eye_metric(before_eye, settings.metric)
        # 眼模式不需要参考原始波形。
        stored_reference_waveform = None
        # 眼模式不构造稳态 Vpp 模型。
        vpp_cache = None
    # Vpp 使用稳态码型模型；仅为旧公共 API 保留原始波形兼容路径。
    else:
        # Vpp 模式没有眼图缓存和采样相位。
        reference_eye = None
        # 补偿前眼同样为空。
        before_eye = None
        # Vpp 只测量原始波形标量，不构造固定眼图激励。
        eye_cache = None
        # 采样相位只属于眼图指标。
        sampling_phase_index = None
        # Vpp 不生成参考眼，因此没有眼图主脉冲原点。
        reference_eye_main_index = None
        # DUT 原始波形的 Vpp 补偿不使用眼图折叠原点。
        dut_eye_main_index = None
        # Vpp 不构造眼图，因此没有共同主光标幅度标尺。
        eye_amplitude_normalizer_v = None
        # 新合同从两份完整拟合脉冲与同一理想码型构造稳态周期。
        if settings.vpp is not None:
            vpp_cache = prepare_vpp_analysis(
                reference_pulse,
                dut_pulse,
                settings.vpp,
                prepared_pattern_levels=prepared_vpp_pattern_levels,
                cancelled=cancelled,
            )
            model_time_s = np.arange(
                vpp_cache.period_samples,
                dtype=np.float64,
            ) / vpp_cache.sample_rate_hz
            stored_reference_waveform = TimeSeries(
                model_time_s,
                vpp_cache.reference_model.waveform_v,
                vpp_cache.sample_rate_hz,
                source_format="memory",
                source_metadata={"model": "periodic_pattern_pulse"},
            )
            target_signal = TimeSeries(
                model_time_s,
                vpp_cache.dut_model.waveform_v,
                vpp_cache.sample_rate_hz,
                source_format="memory",
                source_metadata={"model": "periodic_pattern_pulse"},
            )
            reference_metric = vpp_cache.reference_metric_v
            before_metric = vpp_cache.dut_metric_v
        # 旧调用仍可传两条真实波形，便于既有批处理在迁移期间保持结果。
        else:
            if reference_waveform is None or dut_waveform is None:
                raise ValueError("Vpp 指标必须提供 Vpp 模型设置")
            if reference_waveform.channels != 1 or dut_waveform.channels != 1:
                raise ValueError("Vpp 指标当前只支持单通道原始波形")
            target_signal = dut_waveform
            reference_metric, before_metric = _measure_vpp_baseline(
                reference_waveform,
                dut_waveform,
            )
            stored_reference_waveform = reference_waveform
            vpp_cache = None
    # 扫描上限先受两脉冲和实际被补偿的 DUT 目标数据 Nyquist 约束。
    nyquist_limits_hz = [
        reference_pulse.nyquist_hz,
        dut_pulse.nyquist_hz,
        target_signal.nyquist_hz,
    ]
    # Vpp 的参考标量同样不能为其 Nyquist 以上的候选提供因果证据。
    if stored_reference_waveform is not None:
        # 手动频带与自动频带统一采用两份原始波形的较低 Nyquist。
        nyquist_limits_hz.append(stored_reference_waveform.nyquist_hz)
    # 共同上限取全部实际参与指标或频响计算的数据源最小值。
    common_nyquist_hz = min(nyquist_limits_hz)
    # 浮点容差只允许端点舍入，不允许真正越过 Nyquist。
    nyquist_tolerance_hz = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        common_nyquist_hz,
    )
    # 超限候选无法在真实时域 RFFT 中应用。
    if settings.scan_high_hz > common_nyquist_hz + nyquist_tolerance_hz:
        # 错误同时给出公共上限，界面可帮助用户修正。
        raise ValueError(
            f"扫描上限超过脉冲与目标数据的公共 Nyquist {common_nyquist_hz:g} Hz"
        )
    # 独立物理分辨率同时受两份拟合脉冲和实际被补偿目标的有限时长约束。
    physical_resolution_hz = max(
        reference_pulse.sample_rate_hz / reference_pulse.samples,
        dut_pulse.sample_rate_hz / dut_pulse.samples,
        target_signal.sample_rate_hz / target_signal.samples,
    )
    # 候选生成同时返回分辨率不足时的实际核心宽度与告警。
    candidates, effective_window_width_hz, warnings = candidate_frequency_bands(
        settings,
        physical_resolution_hz=physical_resolution_hz,
    )
    original_samples = target_signal.samples
    # 稳态码型已经定义了严格周期边界，必须直接使用其圆周频谱。
    if vpp_cache is not None:
        padding = 0
        frequency_hz = np.asarray(vpp_cache.frequency_hz, dtype=np.float64)
        base_spectrum = np.asarray(vpp_cache.dut_model.spectrum_v)[:, None]
    # 眼图和旧原始波形路径继续使用镜像延拓，抑制有限记录首尾回卷。
    else:
        padding = original_samples - 1
        extended_values = np.pad(
            np.asarray(target_signal.values, dtype=np.float64),
            ((padding, padding), (0, 0)),
            mode="reflect",
        )
        frequency_hz = np.fft.rfftfreq(
            extended_values.shape[0],
            d=1.0 / target_signal.sample_rate_hz,
        )
        base_spectrum = np.fft.rfft(extended_values, axis=0)
        raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 周期模型分别以自身 pmax 为 lag=0，频响补偿也必须采用同一相位基准。
    model_peak_delay_s = 0.0
    if vpp_cache is not None:
        model_peak_delay_s = float(
            dut_pulse.time_s[vpp_cache.dut_model.peak_index]
            - reference_pulse.time_s[vpp_cache.reference_model.peak_index]
        )
    # 两脉冲在目标 DFT 网格的幅度与连续相位差同样只计算一次。
    log_ratio, phase_ratio, magnitude_valid, phase_valid = _prepare_response_ratio(
        reference_pulse,
        dut_pulse,
        frequency_hz,
        settings,
        model_peak_delay_s=model_peak_delay_s,
        cancelled=cancelled,
    )
    # 频率轴复制为只读，避免绘图排序破坏频谱逐点对应。
    readonly_frequency = _readonly_float(frequency_hz)
    # 基础频谱复制只读供所有候选共享。
    readonly_spectrum = _readonly_complex(base_spectrum)
    # 返回完整工作区；大型候选输出按点选即时生成，不在此复制多份。
    return PreparedAttribution(
        reference_pulse=reference_pulse,
        dut_pulse=dut_pulse,
        target_signal=target_signal,
        reference_waveform=stored_reference_waveform,
        settings=settings,
        eye_cache=eye_cache,
        vpp_cache=vpp_cache,
        frequency_hz=readonly_frequency,
        base_spectrum=readonly_spectrum,
        log_magnitude_ratio=log_ratio,
        phase_ratio_rad=phase_ratio,
        magnitude_valid_mask=magnitude_valid,
        phase_valid_mask=phase_valid,
        padding=padding,
        original_samples=original_samples,
        candidates=candidates,
        physical_resolution_hz=float(physical_resolution_hz),
        effective_window_width_hz=effective_window_width_hz,
        reference_metric=float(reference_metric),
        before_metric=float(before_metric),
        reference_eye=reference_eye,
        before_eye=before_eye,
        sampling_phase_index=sampling_phase_index,
        reference_eye_main_index=reference_eye_main_index,
        dut_eye_main_index=dut_eye_main_index,
        eye_amplitude_normalizer_v=eye_amplitude_normalizer_v,
        warnings=warnings,
    )

# 按指标数值尺度生成只覆盖浮点舍入的容差，避免把真实微小恢复误判成零差异。
def _metric_tolerance(*values: float) -> float:
    """构造仅覆盖浮点舍入误差、不吞掉真实小差异的绝对容差。"""

    # 固定 64 eps 与规格中的模型差距边界一致。
    scale = max(1.0, *(abs(float(value)) for value in values if np.isfinite(value)))
    # 不使用任意百分比门限，保留相对 1e-9 量级的真实改善。
    return float(64.0 * np.finfo(np.float64).eps * scale)

# 把不可解析谱点或数值失败封装成带原因的无效候选，使单点失败不会终止整条扫描曲线。
def _invalid_evaluation(
    band: FrequencyBand,
    mode: AttributionMode,
    reason: str,
) -> BandEvaluation:
    """以统一结构返回一个不可解析候选。"""

    # NaN 明确表示没有指标值，不能被候选排序当成零改善。
    attribution = BandAttribution(
        band=band,
        mode=mode,
        metric_after=float("nan"),
        improvement=float("nan"),
        recovery_ratio=float("nan"),
        valid=False,
        invalid_reason=reason,
    )
    # 无效候选没有补偿后波形或眼图。
    return BandEvaluation(
        attribution=attribution,
        corrected_values=None,
        eye_after=None,
    )

# 强制实值波形 RFFT 的 DC 与 Nyquist 端点为实数，维持可逆的共轭对称合同。
def _project_real_rfft_endpoints(
    correction: ComplexArray,
    *,
    extended_samples: int,
) -> ComplexArray:
    """只投影数值噪声，并拒绝实值 RFFT 端点无法表达的真实复相位。"""

    # compose 返回只读缓存，因此复制后才可安全投影端点。
    projected = np.array(correction, dtype=np.complex128, copy=True)

    # 内部函数同时处理 DC 和偶数长度记录的 Nyquist 端点。
    def project(index: int, label: str) -> None:
        """把近实复数投影到带符号实轴。"""

        # 当前端点的模长决定相对可表示性容差。
        value = projected[index]
        # sqrt(eps) 与原补偿路径的直接响应复核容差一致量级。
        tolerance = max(abs(value), np.finfo(np.float64).tiny) * np.sqrt(
            np.finfo(np.float64).eps
        )
        # 真实非零虚部无法由共轭对称的实值时域记录表达。
        if abs(value.imag) > tolerance:
            # 让候选标无效而不是静默丢掉相位。
            raise ValueError(f"{label} 频点需要非实补偿，实值波形无法表示")
        # 实部符号保留 0 或 pi 相位，模长保持不变。
        projected[index] = np.copysign(abs(value), value.real or 1.0) + 0.0j

    # DC 始终是实值 RFFT 端点。
    project(0, "DC")
    # 只有偶数延拓长度才包含独立 Nyquist 实端点。
    if extended_samples % 2 == 0:
        # 最后一个 RFFT 频点对应 Nyquist。
        project(-1, "Nyquist")
    # 端点处理后再次锁定数组。
    projected.setflags(write=False)
    # 返回可直接用于频谱相乘的补偿。
    return projected

# 将候选复补偿乘到缓存目标频谱并 IFFT 裁回原记录，避免为每个模式重复准备信号。
def _apply_cached_correction(
    workspace: PreparedAttribution,
    correction: ComplexArray,
    *,
    cancelled: CancellationCheck | None = None,
) -> FloatArray:
    """在准备好的目标频谱上应用一次候选补偿并裁回原记录。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 基础频谱的最后一轴是通道，补偿沿频率轴广播。
    filtered_spectrum = workspace.base_spectrum * correction[:, None]
    # 延拓长度可由 RFFT 点数和奇偶信息直接从缓存裁剪参数恢复。
    extended_samples = workspace.original_samples + 2 * workspace.padding
    # 指定 n 避免 irfft 对奇数长度做错误的默认推断。
    filtered_extended = np.fft.irfft(
        filtered_spectrum,
        n=extended_samples,
        axis=0,
    )
    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 只取镜像延拓中央的原始 N 点。
    corrected = filtered_extended[
        workspace.padding : workspace.padding + workspace.original_samples
    ]
    # 复制只读并验证有限值，供候选点选和指标计算共享。
    return _readonly_float(corrected)

# 按当前指标测量候选补偿波形：Vpp 沿用固定分块，眼指标沿用 DUT 冻结原点。
def _measure_candidate_metric(
    workspace: PreparedAttribution,
    corrected_values: FloatArray,
    *,
    include_plot: bool,
    cancelled: CancellationCheck | None = None,
) -> tuple[float, VirtualEyeResult | None]:
    """用准备阶段冻结的度量口径计算一个候选补偿后指标。"""

    # Vpp 使用 DUT 原始波形和共同时间块，不构造眼图。
    if workspace.settings.metric == "vpp":
        # 新 LFP 模型对确定性稳态周期使用真实 max-min，不用采集波形分位数。
        if workspace.vpp_cache is not None:
            values = np.asarray(corrected_values[:, 0], dtype=np.float64)
            if workspace.vpp_cache.settings.method == "lfp":
                return float(np.max(values) - np.min(values)), None
            # 频域 RMS 误差的时域等价式仅用于保留波形的回放路径。
            reference = workspace.vpp_cache.reference_model.waveform_v
            error = values - reference
            error -= float(np.mean(error))
            return float(np.sqrt(np.mean(np.square(error)))), None
        # Vpp 返回标量和空眼图。
        return _measure_corrected_vpp(corrected_values, workspace), None
    # 眼模式在 prepare 已验证 eye、主光标零相位和共同幅度基准均存在。
    if (
        workspace.settings.eye is None
        or workspace.eye_cache is None
        or workspace.sampling_phase_index is None
        or workspace.dut_eye_main_index is None
        or workspace.eye_amplitude_normalizer_v is None
    ):
        # 工作区状态不完整表示内部编程错误。
        raise RuntimeError("眼图工作区缺少缓存、冻结原点或共同幅度基准")
    # 补偿后脉冲沿用 DUT 时间轴，使采样率仍由时间列推导。
    corrected_pulse = TimeSeries(
        workspace.dut_pulse.time_s,
        corrected_values,
        workspace.dut_pulse.sample_rate_hz,
        source_format="memory",
    )
    # 所有候选严格使用 DUT 基线原点和主光标 0 UI 测量口径；眼图绘制只需轨迹，
    # 因此纯眼高无论批量扫描还是点选候选都跳过无关的 41 条眼宽水平切片。
    eye_after = _build_virtual_eye_from_cache(
        corrected_pulse,
        workspace.eye_cache,
        sampling_phase_index=workspace.sampling_phase_index,
        main_index=workspace.dut_eye_main_index,
        amplitude_normalizer_v=workspace.eye_amplitude_normalizer_v,
        include_plot=include_plot,
        measure_width=workspace.settings.metric == "eye_width",
        plot_trace_indices=(
            workspace.before_eye.plot_trace_indices
            if include_plot and workspace.before_eye is not None
            else None
        ),
        cancelled=cancelled,
    )
    # 限制眼标量用于排名，完整结果用于当前候选绘图。
    metric_after = _limiting_eye_metric(eye_after, workspace.settings.metric)
    # 返回标量与眼图成对结果。
    return metric_after, eye_after

# 把候选显示核心扩展为相同的半余弦肩部，并裁在用户扫描域内供三种模式共用。
def _candidate_band_weights(
    workspace: PreparedAttribution,
    band: FrequencyBand,
) -> FloatArray:
    """把可见满权核心转换为扫描域内的平滑余弦权重。"""

    # alpha=0.5 表示每侧肩宽为当前有效核心宽度的一半。
    shoulder_hz = (
        workspace.settings.taper_alpha * (band.high_hz - band.low_hz)
    )
    # 肩部裁到准备阶段的扫描域，避免读取域外 NaN 频响缓存。
    return cosine_core_band_weights(
        workspace.frequency_hz,
        core_low_hz=band.low_hz,
        core_high_hz=band.high_hz,
        shoulder_hz=shoulder_hz,
        domain_low_hz=workspace.settings.scan_low_hz,
        domain_high_hz=workspace.settings.scan_high_hz,
    )

# 在预构造权重上完成单频段单模式反事实，计算指标改善和无裁剪恢复比例。
def _evaluate_attribution_band_with_weights(
    workspace: PreparedAttribution,
    band: FrequencyBand,
    mode: AttributionMode,
    weights: FloatArray,
    *,
    retain_outputs: bool,
    cancelled: CancellationCheck | None = None,
) -> BandEvaluation:
    """使用已构造的频带权重评估单个模式。"""

    raise_if_cancelled(cancelled, message="影响频段分析已取消")
    # 频段必须完整位于准备时锁定的扫描范围内。
    boundary_tolerance_hz = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        workspace.settings.scan_high_hz,
    )
    # 越界频段不能利用未预计算的频响缓存。
    if (
        band.low_hz < workspace.settings.scan_low_hz - boundary_tolerance_hz
        or band.high_hz > workspace.settings.scan_high_hz + boundary_tolerance_hz
    ):
        # 这是调用错误而不是单个候选的谱不可解析状态。
        raise ValueError("候选频段超出准备时的扫描范围")
    # 端点权重可能为零，至少需要一个真正非零的 DFT 频点。
    active = weights > 0.0
    # 没有实际频点意味着目标记录不足以应用该候选。
    if not np.any(active):
        # 返回候选级无效状态而不中断整次扫描。
        return _invalid_evaluation(
            band,
            mode,
            "目标记录在该频段没有可应用的 DFT 频点",
        )
    # 纯幅度只要求幅度比有效。
    if mode == "magnitude":
        # 模式专属掩码避免相位孤点误伤幅度结论。
        required_valid = workspace.magnitude_valid_mask
    # 纯相位只要求连续相位有效。
    elif mode == "phase":
        # 相位掩码已经包含幅度可解析与展开成功两层条件。
        required_valid = workspace.phase_valid_mask
    # 幅相共同模式要求两者同时有效。
    elif mode == "both":
        # 逐点交集保证两个复对数分量都可用。
        required_valid = workspace.magnitude_valid_mask & workspace.phase_valid_mask
    # 未知模式在接口边界明确失败。
    else:
        # 不把拼写错误当成联合模式。
        raise ValueError("归因模式必须是 magnitude、phase 或 both")
    # 任一实际加权频点不可解析时保守跳过整个候选。
    if np.any(active & ~required_valid):
        # 不用插值跨越谱零点制造补偿。
        return _invalid_evaluation(
            band,
            mode,
            "候选频段内存在无法解析的脉冲频响",
        )
    # 在连续复对数域组合当前模式的平滑局部补偿。
    try:
        # 三模式共享同一幅相缓存和同一权重，保证正交比较公平。
        correction = compose_frequency_correction(
            workspace.log_magnitude_ratio,
            workspace.phase_ratio_rad,
            weights,
            mode=mode,
        )
        # 实值波形的 DC/Nyquist 端点必须保持共轭对称可表示性。
        correction = _project_real_rfft_endpoints(
            correction,
            extended_samples=workspace.original_samples + 2 * workspace.padding,
        )
        # 稳态 Vpp 模型直接复用其周期频谱；RMS 扫描全程不执行候选 IFFT。
        if workspace.vpp_cache is not None:
            measurement = measure_candidate(
                workspace.vpp_cache,
                correction,
                cancelled=cancelled,
            )
            metric_after = measurement.value_v
            eye_after = None
            if retain_outputs:
                candidate_waveform = measurement.waveform_v
                if candidate_waveform is None:
                    candidate_waveform = scipy_fft.irfft(
                        measurement.corrected_spectrum_v,
                        n=workspace.vpp_cache.period_samples,
                    )
                corrected_values = _readonly_float(candidate_waveform[:, None])
            else:
                corrected_values = None
        else:
            # 有限记录和眼图路径沿用镜像延拓后的 IFFT 与原指标口径。
            corrected_values = _apply_cached_correction(
                workspace,
                correction,
                cancelled=cancelled,
            )
            metric_after, eye_after = _measure_candidate_metric(
                workspace,
                corrected_values,
                include_plot=retain_outputs,
                cancelled=cancelled,
            )
    except OperationCancelledError:
        raise
    # 数值不可逆点和端点不可表示只使当前候选无效。
    except ValueError as error:
        # 错误原因保留给候选表，不终止其他频段和模式。
        return _invalid_evaluation(band, mode, str(error))
    # 补偿前与参考的绝对差距是可恢复总量。
    before_gap = abs(workspace.before_metric - workspace.reference_metric)
    # 当前候选后的残余参考差距允许过补偿。
    after_gap = abs(metric_after - workspace.reference_metric)
    # 正分表示更接近参考，负分如实表示该频段使指标更差。
    improvement = before_gap - after_gap
    # 数值容差仅用于零差距分母处理，不裁剪真实小改善。
    tolerance = _metric_tolerance(
        workspace.reference_metric,
        workspace.before_metric,
        metric_after,
    )
    # 有可解析基线差距时计算无裁剪恢复比例。
    recovery_ratio = improvement / before_gap if before_gap > tolerance else 0.0
    # 形成可排名候选标量。
    attribution = BandAttribution(
        band=band,
        mode=mode,
        metric_after=float(metric_after),
        improvement=float(improvement),
        recovery_ratio=float(recovery_ratio),
        valid=True,
        invalid_reason="",
    )
    # 默认或点选评估保留波形；扫描路径在标量算完后立即释放它。
    stored_corrected_values = corrected_values if retain_outputs else None
    # 完整评估保留眼图；扫描不保留仅标量的临时结果。
    stored_eye_after = eye_after if retain_outputs else None
    # 返回轻量或完整评估，两者共享完全相同的指标计算。
    return BandEvaluation(
        attribution=attribution,
        corrected_values=stored_corrected_values,
        eye_after=stored_eye_after,
    )

# 公共点选入口为指定核心生成权重并保留补偿波形，供界面重画当前候选结果。
def evaluate_attribution_band(
    workspace: PreparedAttribution,
    band: FrequencyBand,
    mode: AttributionMode,
    *,
    cancelled: CancellationCheck | None = None,
) -> BandEvaluation:
    """只补偿指定频段并计算向参考指标恢复了多少。"""

    # 公共单次评估把列表中的满权核心扩展成不留扫描缝的余弦肩部。
    weights = _candidate_band_weights(workspace, band)
    # 默认保留补偿后波形和完整眼图，保持点选与现有公共行为。
    return _evaluate_attribution_band_with_weights(
        workspace,
        band,
        mode,
        weights,
        retain_outputs=True,
        cancelled=cancelled,
    )

# 直接按每个局部反事实的指标改善锁定最佳频段，再只在该频段内简化幅相标签。
def _select_recommendation(
    candidates: tuple[BandAttribution, ...] | list[BandAttribution],
    *,
    baseline_gap: float,
    baseline_tolerance: float,
    mode_materiality_fraction: float,
) -> BandAttribution | None:
    """先按局部直接回放锁定最佳频段，再只在该频段内简化模式。"""

    # 局部候选自身已经完成时域回放；有效且显著缩小指标差距即可进入排名。
    eligible = [
        result
        for result in candidates
        if result.valid
        and result.improvement > baseline_tolerance
    ]
    # 没有局部正改善时保守返回空，不从负改善或机器舍入中挑选频道。
    if not eligible:
        # 调用方会保留曲线并标记 no_recommendation。
        return None
    # 原始最大改善先唯一锁定主要频段；稳定 max 在精确同分时保留扫描顺序。
    raw_best = max(eligible, key=lambda result: result.improvement)
    # 1% 容差只回答“同一频段是否需要标成幅相”，绝不跨频段。
    mode_tolerance = max(
        baseline_tolerance,
        mode_materiality_fraction * abs(baseline_gap),
    )
    # 只有原始最佳是联合模式时才有必要尝试简化标签。
    if raw_best.mode != "both":
        # 单模式已经是该频段与全局的真实最大值，直接保留。
        return raw_best
    # 同一频段内寻找接近联合改善的幅度或相位单模式。
    simple_candidates = [
        result
        for result in eligible
        if result.band == raw_best.band
        and result.mode in {"magnitude", "phase"}
        and raw_best.improvement - result.improvement <= mode_tolerance
    ]
    # 没有足够接近的单模式时，联合标签具有实质贡献。
    if not simple_candidates:
        # 保留原始最佳联合候选。
        return raw_best
    # 幅度与相位同为简单模式，选择其中改善更大的一个而非固定偏向幅度。
    return max(simple_candidates, key=lambda result: result.improvement)


def _workspace_with_eye_seed(
    workspace: PreparedAttribution,
    random_seed: int,
    *,
    cancelled: CancellationCheck | None = None,
) -> PreparedAttribution:
    """复用频响缓存，只替换固定符号激励和该种子的眼图基线。"""

    eye_settings = workspace.settings.eye
    if eye_settings is None:
        raise ValueError("只有眼图工作区可以执行多种子复核")
    if (
        workspace.reference_eye_main_index is None
        or workspace.dut_eye_main_index is None
        or workspace.eye_amplitude_normalizer_v is None
    ):
        raise RuntimeError("眼图工作区缺少冻结主光标或公共幅度基准")
    seeded_eye_settings = replace(eye_settings, random_seed=int(random_seed))
    seeded_settings = replace(workspace.settings, eye=seeded_eye_settings)
    seeded_cache = _prepare_virtual_eye_cache(
        seeded_eye_settings,
        cancelled=cancelled,
    )
    measure_width = workspace.settings.metric == "eye_width"
    reference_eye = _build_virtual_eye_from_cache(
        workspace.reference_pulse,
        seeded_cache,
        main_index=workspace.reference_eye_main_index,
        amplitude_normalizer_v=workspace.eye_amplitude_normalizer_v,
        include_plot=False,
        measure_width=measure_width,
        cancelled=cancelled,
    )
    before_eye = _build_virtual_eye_from_cache(
        workspace.dut_pulse,
        seeded_cache,
        sampling_phase_index=workspace.sampling_phase_index,
        main_index=workspace.dut_eye_main_index,
        amplitude_normalizer_v=workspace.eye_amplitude_normalizer_v,
        include_plot=False,
        measure_width=measure_width,
        cancelled=cancelled,
    )
    return replace(
        workspace,
        settings=seeded_settings,
        eye_cache=seeded_cache,
        reference_metric=_limiting_eye_metric(reference_eye, workspace.settings.metric),
        before_metric=_limiting_eye_metric(before_eye, workspace.settings.metric),
        reference_eye=reference_eye,
        before_eye=before_eye,
    )


def _verify_eye_recommendation_robustness(
    workspace: PreparedAttribution,
    recommendation: BandAttribution,
    *,
    recommendation_tolerance: float,
    completed: int,
    total_evaluations: int,
    progress: Callable[[int, int], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[bool | None, int]:
    """在两个额外符号种子上完整复扫所有频段和模式。"""

    modes: tuple[AttributionMode, ...] = ("magnitude", "phase", "both")
    assert workspace.settings.eye is not None
    for offset in range(1, EYE_ROBUSTNESS_SEED_COUNT):
        if cancelled is not None and cancelled():
            return None, completed
        seeded_workspace = _workspace_with_eye_seed(
            workspace,
            workspace.settings.eye.random_seed + offset,
            cancelled=cancelled,
        )
        seeded_results: list[BandAttribution] = []
        for band in seeded_workspace.candidates:
            weights = _candidate_band_weights(seeded_workspace, band)
            for mode in modes:
                if cancelled is not None and cancelled():
                    return None, completed
                evaluation = _evaluate_attribution_band_with_weights(
                    seeded_workspace,
                    band,
                    mode,
                    weights,
                    retain_outputs=False,
                    cancelled=cancelled,
                )
                seeded_results.append(evaluation.attribution)
                completed += 1
                if progress is not None:
                    progress(completed, total_evaluations)
        seeded_best = _select_recommendation(
            seeded_results,
            baseline_gap=abs(
                seeded_workspace.before_metric - seeded_workspace.reference_metric
            ),
            baseline_tolerance=max(
                recommendation_tolerance,
                seeded_workspace.settings.recommendation_materiality_fraction
                * abs(
                    seeded_workspace.before_metric
                    - seeded_workspace.reference_metric
                ),
            ),
            mode_materiality_fraction=(
                seeded_workspace.settings.mode_materiality_fraction
            ),
        )
        if (
            seeded_best is None
            or abs(
                seeded_best.band.center_hz - recommendation.band.center_hz
            )
            > workspace.effective_window_width_hz
            + workspace.physical_resolution_hz * 1.0e-9
            or seeded_best.mode != recommendation.mode
        ):
            return False, completed
    return True, completed

# 扫描全部候选核心的幅度、相位和联合反事实，并把全频结果仅保留为诊断证据。
def scan_frequency_attribution(
    workspace: PreparedAttribution,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FrequencyAttributionResult:
    """扫描所有用户频宽中心和三种模式，返回一个保守推荐。"""

    # 模式顺序固定为幅度、相位、幅相，便于曲线颜色和并列结果稳定。
    modes: tuple[AttributionMode, ...] = ("magnitude", "phase", "both")
    # 全频闭环占三次评估，每个候选再占三次；眼图推荐在两个额外种子上
    # 完整复扫所有候选，避免主种子前三名之外的频段跃升却被漏掉。
    total_evaluations = len(modes) * (1 + len(workspace.candidates))
    if workspace.settings.metric in {"eye_height", "eye_width"}:
        total_evaluations += (
            len(workspace.candidates)
            * EYE_ROBUSTNESS_EXTRA_EVALUATIONS_PER_BAND
        )
    # 已完成计数从零开始，并只在一次评估结束后递增。
    completed = 0
    # 基线差距容差只覆盖机器舍入，不吞掉真实小差异。
    baseline_tolerance = _metric_tolerance(
        workspace.reference_metric,
        workspace.before_metric,
    )
    baseline_gap = abs(workspace.before_metric - workspace.reference_metric)
    recommendation_tolerance = max(
        baseline_tolerance,
        workspace.settings.recommendation_materiality_fraction * baseline_gap,
    )
    # 没有可解析基线差距时，任何 IFFT 反事实都不可能得到有意义的推荐。
    if baseline_gap <= baseline_tolerance:
        # 早退不构造全频或局部权重，更不会进入补偿 IFFT。
        return FrequencyAttributionResult(
            reference_metric=workspace.reference_metric,
            before_metric=workspace.before_metric,
            full_band_results=(),
            candidates=(),
            recommendation=None,
            effective_frequency_resolution_hz=workspace.physical_resolution_hz,
            effective_window_width_hz=workspace.effective_window_width_hz,
            status="no_difference",
            warnings=workspace.warnings,
        )
    # 扫描范围整体作为全频闭环候选，中心仅用于结构一致。
    full_band = FrequencyBand(
        low_hz=workspace.settings.scan_low_hz,
        high_hz=workspace.settings.scan_high_hz,
        center_hz=0.5
        * (workspace.settings.scan_low_hz + workspace.settings.scan_high_hz),
    )
    # 全频带三种模式的核心几何相同，平滑权重只构造一次。
    full_band_weights = _candidate_band_weights(workspace, full_band)
    # 全频结果先计算并作为诊断保存，但不能替代任一局部反事实的直接证据。
    full_results: list[BandAttribution] = []
    # 逐模式执行相同的全频窗口。
    for mode in modes:
        # 用户取消在每次可能较重的眼图卷积前检查。
        if cancelled is not None and cancelled():
            # 取消结果不伪造推荐，并保留已经得到的全频证据。
            return FrequencyAttributionResult(
                reference_metric=workspace.reference_metric,
                before_metric=workspace.before_metric,
                full_band_results=tuple(full_results),
                candidates=(),
                recommendation=None,
                effective_frequency_resolution_hz=workspace.physical_resolution_hz,
                effective_window_width_hz=workspace.effective_window_width_hz,
                status="cancelled",
                warnings=workspace.warnings,
            )
        # 扫描只需标量证据，不保留补偿波形或大型眼图轨迹。
        try:
            evaluation = _evaluate_attribution_band_with_weights(
                workspace,
                full_band,
                mode,
                full_band_weights,
                retain_outputs=False,
                cancelled=cancelled,
            )
        except OperationCancelledError:
            return FrequencyAttributionResult(
                reference_metric=workspace.reference_metric,
                before_metric=workspace.before_metric,
                full_band_results=tuple(full_results),
                candidates=(),
                recommendation=None,
                effective_frequency_resolution_hz=workspace.physical_resolution_hz,
                effective_window_width_hz=workspace.effective_window_width_hz,
                status="cancelled",
                warnings=workspace.warnings,
            )
        # 追加全频标量证据。
        full_results.append(evaluation.attribution)
        # 完成一次评估后推进进度。
        completed += 1
        # Qt 线程可通过回调安全发射整数进度，不让核心依赖 Qt。
        if progress is not None:
            # 回调接收已完成数和总数，界面自行换算百分比。
            progress(completed, total_evaluations)
    # 局部候选只保存标量摘要，避免候选数乘波形长度造成内存膨胀。
    candidate_results: list[BandAttribution] = []
    # 中心频段按低到高顺序扫描。
    for band in workspace.candidates:
        # 同一满权核心的幅度、相位和幅相模式使用完全相同的余弦肩部。
        band_weights = _candidate_band_weights(workspace, band)
        # 每个中心都公平比较三种正交模式。
        for mode in modes:
            # 候选之间检查取消，使 UI 可在一个卷积完成后响应。
            if cancelled is not None and cancelled():
                # 返回已完成的部分曲线但不产生不完整推荐。
                return FrequencyAttributionResult(
                    reference_metric=workspace.reference_metric,
                    before_metric=workspace.before_metric,
                    full_band_results=tuple(full_results),
                    candidates=tuple(candidate_results),
                    recommendation=None,
                    effective_frequency_resolution_hz=workspace.physical_resolution_hz,
                    effective_window_width_hz=workspace.effective_window_width_hz,
                    status="cancelled",
                    warnings=workspace.warnings,
                )
            # 只补当前频段并计算一个模式的恢复分数。
            try:
                evaluation = _evaluate_attribution_band_with_weights(
                    workspace,
                    band,
                    mode,
                    band_weights,
                    retain_outputs=False,
                    cancelled=cancelled,
                )
            except OperationCancelledError:
                return FrequencyAttributionResult(
                    reference_metric=workspace.reference_metric,
                    before_metric=workspace.before_metric,
                    full_band_results=tuple(full_results),
                    candidates=tuple(candidate_results),
                    recommendation=None,
                    effective_frequency_resolution_hz=workspace.physical_resolution_hz,
                    effective_window_width_hz=workspace.effective_window_width_hz,
                    status="cancelled",
                    warnings=workspace.warnings,
                )
            # 释放大型输出前先保存轻量摘要。
            candidate_results.append(evaluation.attribution)
            # 推进完成计数。
            completed += 1
            # 有进度回调时在每个模式后通知。
            if progress is not None:
                # 不在核心线程操作任何 Qt 控件。
                progress(completed, total_evaluations)
    # 排名直接使用局部时域回放证据；全频结果只帮助用户诊断整体模型表现。
    best_candidate = _select_recommendation(
        candidate_results,
        baseline_gap=baseline_gap,
        baseline_tolerance=recommendation_tolerance,
        mode_materiality_fraction=workspace.settings.mode_materiality_fraction,
    )
    result_warnings = list(workspace.warnings)
    if (
        best_candidate is not None
        and workspace.settings.metric in {"eye_height", "eye_width"}
    ):
        try:
            robust, completed = _verify_eye_recommendation_robustness(
                workspace,
                best_candidate,
                recommendation_tolerance=recommendation_tolerance,
                completed=completed,
                total_evaluations=total_evaluations,
                progress=progress,
                cancelled=cancelled,
            )
        except OperationCancelledError:
            robust = None
        if robust is None:
            return FrequencyAttributionResult(
                reference_metric=workspace.reference_metric,
                before_metric=workspace.before_metric,
                full_band_results=tuple(full_results),
                candidates=tuple(candidate_results),
                recommendation=None,
                effective_frequency_resolution_hz=workspace.physical_resolution_hz,
                effective_window_width_hz=workspace.effective_window_width_hz,
                status="cancelled",
                warnings=tuple(result_warnings),
            )
        if robust:
            result_warnings.append("眼图推荐已通过 3 个确定性符号种子的稳定性复核")
        else:
            result_warnings.append(
                "眼图推荐未通过 3 个确定性符号种子的稳定性复核，已保守取消推荐"
            )
            best_candidate = None
    # 有效且显著为正的局部改善可形成保守推荐。
    if best_candidate is not None:
        # 推荐保留其幅度/相位/幅相模式，直接回答用户第二个问题。
        recommendation = best_candidate
        # 成功状态只表示找到大致主要影响频段，不声称因果唯一。
        status: Literal["ok", "no_difference", "no_recommendation", "cancelled"] = "ok"
    # 否则保守返回无推荐，不从数值噪声中强挑第一名。
    else:
        # 没有局部正改善时仍保留完整曲线和全频诊断。
        recommendation = None
        # 局部充分性证据不足。
        status = "no_recommendation"
    # 返回完整标量结果与物理分辨率证据。
    return FrequencyAttributionResult(
        reference_metric=workspace.reference_metric,
        before_metric=workspace.before_metric,
        full_band_results=tuple(full_results),
        candidates=tuple(candidate_results),
        recommendation=recommendation,
        effective_frequency_resolution_hz=workspace.physical_resolution_hz,
        effective_window_width_hz=workspace.effective_window_width_hz,
        status=status,
        warnings=tuple(result_warnings),
    )

# 汇总参考、补偿前和补偿后的 2 UI 轨迹与共同纵轴范围，供眼图三联图直接叠线比较。
def build_eye_comparison(
    workspace: PreparedAttribution,
    evaluation: BandEvaluation,
) -> EyeComparisonData:
    """为当前候选构建参考、补偿前、补偿后三幅共轴轨迹眼。"""

    # 只有眼图指标具有三份轻量眼结果。
    if workspace.settings.metric not in {"eye_height", "eye_width"}:
        # Vpp 页面应改画补偿前后原始波形。
        raise ValueError("Vpp 指标不生成眼图")
    # 工作区基线眼与候选补偿后眼必须完整存在。
    if (
        workspace.reference_eye is None
        or workspace.before_eye is None
        or evaluation.eye_after is None
    ):
        # 无效候选不能生成一张伪造的补偿后图。
        raise ValueError("当前候选没有可绘制的眼图结果")
    # 三组眼图以固定顺序汇总纵轴边界。
    eyes = (
        workspace.reference_eye,
        workspace.before_eye,
        evaluation.eye_after,
    )
    # 每组完整点选眼必须含至少一条轨迹，否则无法定义共同纵轴。
    if any(eye.plot_traces_v.size == 0 for eye in eyes):
        # 扫描标量结果必须经点选重算 include_plot=True 后才能绘制。
        raise ValueError("当前候选没有保留可绘制的 2 UI 轨迹")
    # 公共最小电压来自三组真实轨迹而非单图自动缩放。
    minimum_v = min(float(np.min(eye.plot_traces_v)) for eye in eyes)
    # 公共最大电压同理覆盖三组轨迹。
    maximum_v = max(float(np.max(eye.plot_traces_v)) for eye in eyes)
    # 常值轨迹需要人为加入对称跨度，避免纵轴上下限相等。
    if maximum_v <= minimum_v:
        # 幅度尺度至少为一，避免零信号得到次正规小数边界。
        padding_v = 0.5 * max(1.0, abs(minimum_v))
    # 正常轨迹在上下各留 5% 余量，避免外轨贴边。
    else:
        # 公共余量只从三图总跨度计算。
        padding_v = 0.05 * (maximum_v - minimum_v)
    # 三组轨迹必须共享逐点相同的 -1 UI 到 +1 UI 横轴。
    if not all(
        np.array_equal(workspace.reference_eye.plot_time_ui, eye.plot_time_ui)
        for eye in eyes[1:]
    ):
        # 不允许界面把不同 M 或不同端点定义的轨迹画在同一坐标上。
        raise ValueError("三组眼图轨迹的时间轴不一致")
    # 共同显示范围把真实最值和 5% 余量一起交给界面，三幅图不再分别自动缩放。
    amplitude_range_v = (
        float(minimum_v - padding_v),
        float(maximum_v + padding_v),
    )
    # 返回原始轨迹和指标，界面只负责逐行画线而不再生成密度图。
    return EyeComparisonData(
        time_ui=workspace.reference_eye.plot_time_ui,
        reference_traces_v=workspace.reference_eye.plot_traces_v,
        before_traces_v=workspace.before_eye.plot_traces_v,
        after_traces_v=evaluation.eye_after.plot_traces_v,
        amplitude_range_v=amplitude_range_v,
        sampling_phase_ui=workspace.reference_eye.sampling_phase_ui,
        reference=workspace.reference_eye,
        before=workspace.before_eye,
        after=evaluation.eye_after,
    )
