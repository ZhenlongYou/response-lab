"""“影响频段”页签的后台请求、文件加载与展示适配。

Qt 页面只收集轻量参数；本模块在后台线程加载数据、调用归因引擎，并保留候选点选时
需要复用的工作区。它不修改现有频响补偿结果或导出状态。
"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本文件仅关闭对应格式告警。
# ruff: noqa: I001
# 延迟解析类型标注，避免 Qt 线程数据类中的前向类型在导入阶段求值。
from __future__ import annotations

# dataclass 把一次扫描请求和完成结果冻结为可在线程间传递的对象。
from dataclasses import dataclass
# Path 保留文件路径语义并用于判断 CSV/BIN 格式。
from pathlib import Path
# Literal 限定页面内部指标和调制键，避免依赖中文显示文字。
from typing import Literal

# NumPy 用于结果曲线分组和波形展示副本。
import numpy as np
# QThread 与 Signal 让耗时 FFT、卷积和文件读取不阻塞主窗口。
from PySide6.QtCore import QThread, Signal

# 归因公共接口负责准备缓存、扫描、候选回放和三眼轨迹提取。
from .attribution import (
    AttributionSettings,
    BandAttribution,
    BandEvaluation,
    EyeComparisonData,
    EYE_ROBUSTNESS_EXTRA_EVALUATIONS_PER_BAND,
    FrequencyAttributionResult,
    PreparedAttribution,
    VirtualEyeSettings,
    build_eye_comparison,
    candidate_frequency_bands,
    evaluate_attribution_band,
    prepare_frequency_attribution,
    scan_frequency_attribution,
)
# 自动频带建议沿用现有脉冲比较入口，不在页签中重复发明带宽判据。
from .dsp import suggest_frequency_settings
# CSV/BIN 解析继续由项目统一 I/O 层完成。
from .io import load_csv_timeseries
# 影响频段与主补偿、CSV/BIN 加载器共用同一个动态系统内存预算。
from .memory_budget import safe_memory_budget_bytes, system_available_memory_bytes
# CompensationSettings 保存现有右栏的频带设置。
from .models import CompensationSettings, TimeSeries
from .vpp_analysis import (
    VppAnalysisSettings,
    load_pattern_levels,
    validate_vpp_pulse_windows,
)

# 页面指标内部键与归因引擎保持一致。
InfluenceMetric = Literal["vpp", "eye_height", "eye_width"]
# 页面调制只在眼指标下存在。
InfluenceModulation = Literal["nrz", "pam4"]

# 候选核心数量设硬上限，防止错误单位把一次扫描膨胀到数万次 IFFT。
_MAX_CANDIDATE_BANDS = 2_000
# 旧原始波形与眼图前游的目标 FFT 仍按延拓、频谱和临时 IFFT 估算 192 字节/点。
_ESTIMATED_BYTES_PER_TARGET_SAMPLE = 192
# 参考模型与两份拟合脉冲按时间轴+单通道数组的保守 24 字节/点计入。
_ESTIMATED_BYTES_PER_INPUT_SAMPLE = 24
# 每个眼图卷积样点计入冲激、绘图横轴、复频谱、IFFT 输出和数值库临时区。
_ESTIMATED_BYTES_PER_EYE_SAMPLE = 64
# Vpp 周期模型的独立 RSS 实测约为 448 B/周期点；RMS 取 576 B/点保留约 28%
# 余量，LFP 再计入候选完整周期 IFFT/波形副本而使用 640 B/点。
_ESTIMATED_BYTES_PER_VPP_RMS_SAMPLE = 576
_ESTIMATED_BYTES_PER_VPP_LFP_SAMPLE = 640
# 目标样点乘评估次数超过五千万时给用户明确长任务提示。
_LONG_WORK_UNITS = 50_000_000

# 冻结整次分析输入，保证主线程改动控件后不会篡改后台线程正在使用的参数。
@dataclass(frozen=True)
class InfluenceRequest:
    """主窗口在点击“开始分析”时冻结的一次完整请求。"""

    # reference_pulse_path 与 dut_pulse_path 来自主窗口左栏两份拟合脉冲。
    reference_pulse_path: Path
    # DUT 拟合脉冲路径与参考必须都为 CSV。
    dut_pulse_path: Path
    # metric 对应 Vpp、眼高或眼宽中的一个。
    metric: InfluenceMetric
    # modulation 在 Vpp 下为空，在眼指标下为 NRZ/PAM4。
    modulation: InfluenceModulation | None
    # samples_per_ui 是用户输入的 M，Vpp 模型与眼图共同使用。
    samples_per_ui: int | None
    # Vpp 分析方法在眼指标下为空。
    vpp_method: Literal["lfp", "frequency_rms_error"] | None
    # 理想码型来源在眼指标下为空。
    pattern_source: Literal["builtin_prbs13q_gray", "file"] | None
    # 外部码型路径只在 file 来源下存在。
    pattern_path: Path | None
    # 外部码型数值语义不由算法猜测。
    pattern_value_kind: Literal["symbol_codes", "amplitude_values"] | None
    # pmax 前后窗口用 UI 表示，再由 M 精确换算为样点。
    pre_cursor_ui: int | None
    post_cursor_ui: int | None
    # band_width_hz 同时定义候选满权核心宽度和相邻核心中心间距。
    band_width_hz: float
    # frequency_settings 复用右栏当前补偿频带、相位去斜和单位换算结果。
    frequency_settings: CompensationSettings
    # auto_frequency_bands 表示后台需要先根据两脉冲建议扫描范围。
    auto_frequency_bands: bool
    # version 与影响页自己的版本号绑定，不使原补偿导出过期。
    version: int

    # 请求在进入后台线程前完成轻量领域校验，避免无效频宽触发空候选或巨大循环。
    def __post_init__(self) -> None:
        """拒绝非正、非有限或布尔型的物理频段宽度。"""

        # bool 是 int 的子类，但不能解释为 1 Hz 的用户频段宽度。
        if isinstance(self.band_width_hz, (bool, np.bool_)) or not isinstance(
            self.band_width_hz,
            (int, float, np.integer, np.floating),
        ):
            # 类型错误与数值越界统一指向同一个页面控件。
            raise ValueError("频段宽度必须是正的有限 Hz 数值")
        # 零、负数、NaN 和 Inf 都不能参与候选数量与窗函数计算。
        if not np.isfinite(self.band_width_hz) or self.band_width_hz <= 0.0:
            # 在任何文件加载或 FFT 分配前终止。
            raise ValueError("频段宽度必须是正的有限 Hz 数值")
        # Vpp 请求在文件加载前先冻结完整码型模型设置。
        if self.metric == "vpp":
            if (
                self.samples_per_ui is None
                or self.vpp_method is None
                or self.pattern_source is None
                or self.pre_cursor_ui is None
                or self.post_cursor_ui is None
            ):
                raise ValueError("Vpp 指标必须提供方法、码型、M 和 pmax 前后窗口")
            VppAnalysisSettings(
                method=self.vpp_method,
                pattern_source=self.pattern_source,
                samples_per_ui=self.samples_per_ui,
                pre_cursor_ui=self.pre_cursor_ui,
                post_cursor_ui=self.post_cursor_ui,
                pattern_path=self.pattern_path,
                file_value_kind=self.pattern_value_kind or "symbol_codes",
            )

# 冻结扫描产物与缓存引用，使候选列表行号能稳定回放同一次分析工作区。
@dataclass(frozen=True)
class InfluenceRun:
    """一次完整扫描及默认选中候选的可复用状态。"""

    # workspace 缓存频响和目标频谱，点选候选时无需重新加载或求 DTFT。
    workspace: PreparedAttribution
    # result 保存全部候选标量和保守推荐。
    result: FrequencyAttributionResult
    # displayed_candidates 是候选列表从行号到领域结果的稳定映射。
    displayed_candidates: tuple[BandAttribution, ...]
    # selected_evaluation 是推荐或列表首项的补偿后波形。
    selected_evaluation: BandEvaluation | None
    # eye_comparison 只在眼指标且候选有效时存在。
    eye_comparison: EyeComparisonData | None
    # version 用于主窗口拒绝参数变化前完成的旧线程结果。
    version: int

# 冻结单个候选的回放结果，供 Qt 通过版本号拒绝过期的眼图或波形更新。
@dataclass(frozen=True)
class InfluenceSelection:
    """点选一个已有候选后的补偿波形与可选眼图。"""

    # candidate 是当前列表行对应的标量结果。
    candidate: BandAttribution
    # evaluation 保存实际重放得到的补偿后波形。
    evaluation: BandEvaluation
    # eye_comparison 仅眼模式存在。
    eye_comparison: EyeComparisonData | None
    # version 保证切换参数后旧点选结果不会覆盖新页面。
    version: int

# 将页面选择转换为纯算法参数，集中处理 Nyquist、相位拟合带和眼图符号设置。
def _build_attribution_settings(
    request: InfluenceRequest,
    reference_pulse: TimeSeries,
    dut_pulse: TimeSeries,
) -> AttributionSettings:
    """把页面参数与右栏频带合并为纯算法设置。"""

    # 自动频带开启时复用现有建议器计算真实扫描范围。
    if request.auto_frequency_bands:
        # 归因始终比较三种模式，因此建议器以 both 构造相位拟合带。
        automatic_seed = CompensationSettings(
            mode="both",
            band_low_hz=0.0,
            band_high_hz=1.0,
            phase_fit_low_hz=0.0,
            phase_fit_high_hz=1.0,
            detrend_phase=request.frequency_settings.detrend_phase,
            taper_alpha=0.0,
            analysis_points=request.frequency_settings.analysis_points,
        )
        # 建议器只读脉冲并限制到目标 Nyquist。
        frequency_settings = suggest_frequency_settings(
            reference_pulse,
            dut_pulse,
            automatic_seed,
            suggest_phase_fit_band=True,
        )
    # 手动频带直接使用主窗口已完成 Hz 换算的设置。
    else:
        # 不修改用户当前补偿页的对象。
        frequency_settings = request.frequency_settings
    # 相位拟合范围无效或与扫描带不相交时退回完整扫描范围。
    phase_low_hz = frequency_settings.phase_fit_low_hz
    # 复制上限便于成对修正。
    phase_high_hz = frequency_settings.phase_fit_high_hz
    # magnitude 主设置允许相位上下限相等，归因三模式不能沿用该空范围。
    if not (
        0.0 <= phase_low_hz < phase_high_hz
        and phase_low_hz < frequency_settings.band_high_hz
        and phase_high_hz > frequency_settings.band_low_hz
    ):
        # 完整扫描范围足以拟合并剔除整体线性时延。
        phase_low_hz = frequency_settings.band_low_hz
        # 上限同样使用扫描上界。
        phase_high_hz = frequency_settings.band_high_hz
    # 眼指标构造固定符号设置；Vpp 不创建该对象。
    eye_settings = None
    # 用户选择眼高或眼宽时只需提供调制与 M，Np 从脉冲长度推导。
    if request.metric in {"eye_height", "eye_width"}:
        # 防御性校验避免 None 在取模和除法中泄漏 TypeError。
        if request.modulation is None or request.samples_per_ui is None:
            # 页面请求不完整时给出领域错误。
            raise ValueError("眼图指标必须提供调制和 M")
        # bool 虽属于 int，但不能解释为每 UI 一个采样点。
        if isinstance(request.samples_per_ui, (bool, np.bool_)) or not isinstance(
            request.samples_per_ui,
            (int, np.integer),
        ):
            # 自动推导 Np 前先锁定整数 M，避免隐式截断页面协议。
            raise ValueError("M 必须是整数")
        # M 至少为三，与眼宽 crossing 的端点保护合同一致。
        if request.samples_per_ui < 3:
            # 在取模前给出直接可操作的输入错误。
            raise ValueError("M 必须至少为 3")
        # 两份脉冲必须以同一个真实样点数推导出唯一 Np。
        if reference_pulse.samples != dut_pulse.samples:
            # 不允许分别推导后再裁剪或补零对齐。
            raise ValueError("两份拟合脉冲必须等长，无法自动计算 Np")
        # 完整脉冲长度必须恰好由整数个 UI 组成。
        if reference_pulse.samples % request.samples_per_ui != 0:
            # 错误同时给出样点数和 M，用户可直接修正唯一保留的输入。
            raise ValueError(
                f"拟合脉冲样点数 {reference_pulse.samples} 不能被 "
                f"M={request.samples_per_ui} 整除"
            )
        # Np 是完整脉冲包含的 UI 数，不再由页面重复输入。
        pulse_length_ui = reference_pulse.samples // request.samples_per_ui
        # 稳态样本至少保留约 1024 个符号，同时限制扫描成本。
        symbol_count = max(2048, 2 * pulse_length_ui + 1024)
        # 固定种子保证参考、DUT 和全部候选成对比较。
        eye_settings = VirtualEyeSettings(
            modulation=request.modulation,
            pulse_length_ui=pulse_length_ui,
            samples_per_ui=request.samples_per_ui,
            symbol_count=symbol_count,
            random_seed=20260718,
        )
    # Vpp 使用同一份显式模型设置；内置码型的 file_value_kind 取稳定占位值。
    vpp_settings = None
    if request.metric == "vpp":
        if (
            request.samples_per_ui is None
            or request.vpp_method is None
            or request.pattern_source is None
            or request.pre_cursor_ui is None
            or request.post_cursor_ui is None
        ):
            raise ValueError("Vpp 指标必须提供方法、码型、M 和 pmax 前后窗口")
        vpp_settings = VppAnalysisSettings(
            method=request.vpp_method,
            pattern_source=request.pattern_source,
            samples_per_ui=request.samples_per_ui,
            pre_cursor_ui=request.pre_cursor_ui,
            post_cursor_ui=request.post_cursor_ui,
            pattern_path=request.pattern_path,
            file_value_kind=request.pattern_value_kind or "symbol_codes",
        )
    # 返回归因设置；用户频宽同时控制步进和满权核心，alpha 继续使用核心默认常量。
    return AttributionSettings(
        metric=request.metric,
        scan_low_hz=frequency_settings.band_low_hz,
        scan_high_hz=frequency_settings.band_high_hz,
        eye=eye_settings,
        vpp=vpp_settings,
        frequency_step_hz=request.band_width_hz,
        requested_window_hz=request.band_width_hz,
        detrend_phase=frequency_settings.detrend_phase,
        phase_fit_low_hz=phase_low_hz,
        phase_fit_high_hz=phase_high_hz,
    )

def _estimate_influence_peak_memory_bytes(
    settings: AttributionSettings,
    *,
    target_samples: int,
    other_input_samples: int,
) -> int:
    """按指标实际缓存形状估算峰值，不把 Vpp 套入旧原始波形系数。"""

    input_bytes = int(other_input_samples) * _ESTIMATED_BYTES_PER_INPUT_SAMPLE
    if settings.vpp is not None:
        # Vpp 候选补偿、频率轴和误差谱都覆盖完整周期 RFFT；缩窄某个候选频带只会
        # 改变非零权重数，不会缩短这些数组，因此不能按 active bins 折减预算。
        bytes_per_period_sample = (
            _ESTIMATED_BYTES_PER_VPP_LFP_SAMPLE
            if settings.vpp.method == "lfp"
            else _ESTIMATED_BYTES_PER_VPP_RMS_SAMPLE
        )
        return int(target_samples) * bytes_per_period_sample + input_bytes

    estimated = (
        int(target_samples) * _ESTIMATED_BYTES_PER_TARGET_SAMPLE + input_bytes
    )
    if settings.eye is not None:
        eye_convolution_samples = (
            settings.eye.symbol_count * settings.eye.samples_per_ui
            + int(target_samples)
            - 1
        )
        estimated += eye_convolution_samples * _ESTIMATED_BYTES_PER_EYE_SAMPLE
    return int(estimated)


# 在申请 FFT 工作区前估算候选数量、评估次数与内存，提前拦截危险任务。
def _estimate_workload(
    settings: AttributionSettings,
    *,
    physical_resolution_hz: float,
    target_samples: int,
    other_input_samples: int,
) -> tuple[int, int, str]:
    """在分配镜像频谱前估算候选数、峰值内存与长任务提示。"""

    # 使用与核心完全相同的候选生成器，尾部锚定和物理分辨率不会发生口径漂移。
    candidates, _effective_width_hz, _warnings = candidate_frequency_bands(
        settings,
        physical_resolution_hz=physical_resolution_hz,
    )
    # 候选数不包含三次全频闭环。
    candidate_count = len(candidates)
    # 过多候选通常来自 Hz/GHz 单位错误或不合理的超宽扫描范围。
    if candidate_count > _MAX_CANDIDATE_BANDS:
        # 给出实际值与上限，用户可直接缩小频带。
        raise ValueError(
            f"候选频段数量 {candidate_count} 超过上限 "
            f"{_MAX_CANDIDATE_BANDS}，请缩小扫描频带"
        )
    # 三种模式分别执行全频一次和每个局部核心一次。
    total_evaluations = 3 * (1 + candidate_count)
    if settings.metric in {"eye_height", "eye_width"}:
        total_evaluations += (
            candidate_count * EYE_ROBUSTNESS_EXTRA_EVALUATIONS_PER_BAND
        )
    # 按 Vpp/LFP、Vpp/RMS、眼图或旧波形路径选择经实测校准的不同缓存模型。
    estimated_peak_bytes = _estimate_influence_peak_memory_bytes(
        settings,
        target_samples=target_samples,
        other_input_samples=other_input_samples,
    )
    # Vpp 的每次评估规模就是目标原始记录；眼指标还要卷积固定符号激励。
    evaluation_samples = int(target_samples)
    # 眼图缓存长度由 symbol_count*M 决定，可能远大于 Np*M 拟合脉冲本身。
    if settings.eye is not None:
        # 线性卷积长度同时覆盖固定符号冲激和当前拟合脉冲。
        eye_convolution_samples = (
            settings.eye.symbol_count * settings.eye.samples_per_ui
            + int(target_samples)
            - 1
        )
        # 长任务提示也必须按真实眼图卷积长度估算，而不是只看短拟合脉冲。
        evaluation_samples = eye_convolution_samples
    # 与主补偿和加载器共用当前系统可用内存快照，不能只依赖固定 1.5 GB 上限。
    available_memory_bytes = system_available_memory_bytes()
    memory_budget_bytes = safe_memory_budget_bytes(available_memory_bytes)
    # 超动态预算时在 prepare_vpp_analysis 的 rFFT 或眼图卷积之前停止。
    if estimated_peak_bytes > memory_budget_bytes:
        available_text = (
            f"，系统当前可用约 {available_memory_bytes / (1024.0**2):.0f} MiB"
            if available_memory_bytes is not None
            else "，系统可用内存不可探测，使用 768 MiB 回退预算"
        )
        raise ValueError(
            "预计峰值内存约 "
            f"{estimated_peak_bytes / (1024.0**2):.0f} MiB，超过动态安全预算 "
            f"{memory_budget_bytes / (1024.0**2):.0f} MiB{available_text}；"
            "请缩短输入或使用更低的每 UI 采样点数数据"
        )
    # IFFT 工作量用每次真实处理样点数乘评估次数形成可比较的确定性代理。
    work_units = evaluation_samples * total_evaluations
    if settings.metric == "eye_width" and settings.eye is not None:
        stable_traces = max(
            settings.eye.symbol_count - 2 * settings.eye.pulse_length_ui,
            0,
        )
        eye_count = 1 if settings.eye.modulation == "nrz" else 3
        crossing_work_units = (
            stable_traces
            * (2 * settings.eye.samples_per_ui + 1)
            * 41
            * eye_count
            * total_evaluations
        )
        work_units += crossing_work_units
    # 普通短任务不增加状态栏文字。
    notice = ""
    # 长任务只提示并允许后台继续，不静默下采样或改变用户所设频宽语义。
    if work_units > _LONG_WORK_UNITS:
        # 说明候选和评估数量，用户可据此决定缩窄频带或等待。
        notice = (
            f"将扫描 {candidate_count} 个频段、执行 {total_evaluations} 次评估；"
            "长记录可能需要较长时间，可修改参数安全取消"
        )
    # 返回候选数、总评估数和可选提示供线程进度使用。
    return candidate_count, total_evaluations, notice

# 建立候选列表的稳定展示顺序，并将保守推荐固定在首行便于默认回放。
def _ordered_candidates(result: FrequencyAttributionResult) -> tuple[BandAttribution, ...]:
    """把推荐置顶，其余有效候选按改善量降序，限制列表为可操作规模。"""

    # 无效候选保留在影响曲线之外的诊断中，不进入可点击回放列表。
    valid_candidates = [candidate for candidate in result.candidates if candidate.valid]
    # 按改善量从高到低排列；Python 稳定排序保留同分时的频率/模式顺序。
    valid_candidates.sort(key=lambda candidate: candidate.improvement, reverse=True)
    # 推荐可能因模式简化规则不是原始最大值，需要显式放到第一行。
    ordered: list[BandAttribution] = []
    # 有推荐时先加入。
    if result.recommendation is not None:
        # 第一行始终代表页面摘要中的推荐。
        ordered.append(result.recommendation)
    # 逐项追加尚未出现的候选。
    for candidate in valid_candidates:
        # 数据类值相等可直接用于去重。
        if candidate not in ordered:
            # 保留当前排序。
            ordered.append(candidate)
        # 最多展示 120 行，避免数千候选让列表难以操作。
        if len(ordered) >= 120:
            # 已覆盖最显著候选后停止。
            break
    # 返回不可变映射供行号点选。
    return tuple(ordered)

# 只为首个可展示候选生成大型波形或眼图数据，避免扫描结果落地时重复回放。
def _evaluate_default_candidate(
    workspace: PreparedAttribution,
    displayed_candidates: tuple[BandAttribution, ...],
) -> tuple[BandEvaluation | None, EyeComparisonData | None]:
    """为列表第一项生成默认波形或眼图。"""

    # 无推荐且没有有效候选时只展示影响曲线和失败状态。
    if not displayed_candidates:
        # 两种大型展示数据都为空。
        return None, None
    # 第一项已由排序器确保是推荐或最大改善候选。
    candidate = displayed_candidates[0]
    # 复用准备缓存重放一次该候选。
    evaluation = evaluate_attribution_band(
        workspace,
        candidate.band,
        candidate.mode,
    )
    # 无效重放不生成眼图。
    if not evaluation.attribution.valid:
        # 标量扫描与回放不一致时保守返回空展示。
        return evaluation, None
    # Vpp 页面只需要参考、补偿前后的稳态码型模型波形。
    if workspace.settings.metric == "vpp":
        # 不构造眼图轨迹。
        return evaluation, None
    # 眼页面使用同一 UI 时间窗构造三组可直接叠加的轨迹。
    eye_comparison = build_eye_comparison(workspace, evaluation)
    # 返回补偿后波形和三角色轨迹。
    return evaluation, eye_comparison

# 将完整扫描隔离到 QThread，避免文件读取、FFT 和候选 IFFT 阻塞界面事件循环。
class InfluenceAnalysisThread(QThread):
    """后台加载全部输入并执行一次完整影响频段扫描。"""

    # 成功信号携带 InfluenceRun 与请求版本。
    succeeded = Signal(object, int)
    # 失败信号只发送可操作文字和版本，不把异常对象跨线程传给 Qt。
    failed = Signal(str, int)
    # 进度信号发送已完成与总候选评估数。
    progressed = Signal(int, int)
    # 工作量提示携带请求版本，主窗口不会显示过期任务文字。
    noticed = Signal(str, int)

    # 在线程启动前仅保存冻结请求，实际文件 I/O 统一留在后台 run 阶段。
    def __init__(self, request: InfluenceRequest) -> None:
        """保存冻结请求，线程启动前不读取文件。"""

        # 初始化 QThread 生命周期。
        super().__init__()
        # 请求由数据类冻结，可安全从主线程交给后台读取。
        self.request = request

    # 分阶段执行加载、成本门限、扫描和默认回放，并在安全边界响应取消请求。
    def run(self) -> None:
        """执行文件加载、准备、扫描和默认候选回放。"""

        # GUI 边界捕获全部异常并转成简短失败信号。
        try:
            # 拟合脉冲固定按 CSV 时间列读取，采样率从时间轴推导。
            reference_pulse = load_csv_timeseries(
                self.request.reference_pulse_path,
                time_unit="s",
                time_column=0,
                value_columns=(1,),
            )
            # 文件加载阶段之间响应窗口关闭或参数变化发出的安全中断。
            if self.isInterruptionRequested():
                # 取消不继续读取第二份脉冲。
                raise RuntimeError("影响频段分析已取消")
            # DUT 拟合脉冲使用相同加载合同。
            dut_pulse = load_csv_timeseries(
                self.request.dut_pulse_path,
                time_unit="s",
                time_column=0,
                value_columns=(1,),
            )
            # 两份脉冲完成后再次检查中断。
            if self.isInterruptionRequested():
                # 不再加载可能很大的原始 Vpp 文件。
                raise RuntimeError("影响频段分析已取消")
            # 合并自动/手动频带和页面指标参数。
            attribution_settings = _build_attribution_settings(
                self.request,
                reference_pulse,
                dut_pulse,
            )
            # Vpp 的周期样点数可在 FFT 前由码型长度和 M 精确预估。
            target_samples = dut_pulse.samples
            prepared_vpp_pattern_levels = None
            # 其余输入样点只用于保守内存预算，不参与工作量乘法。
            other_input_samples = reference_pulse.samples + dut_pulse.samples
            workload_result: tuple[int, int, str] | None = None
            if attribution_settings.vpp is not None:
                # pmax 窗口错误必须先于外部码型文件扫描或 NumPy 文本解析返回。
                validate_vpp_pulse_windows(
                    reference_pulse,
                    dut_pulse,
                    attribution_settings.vpp,
                )
                if self.isInterruptionRequested():
                    raise RuntimeError("影响频段分析已取消")

                def preflight_vpp_symbol_count(symbol_count: int) -> None:
                    """在码型数组解析前用同描述符统计值完成周期 FFT 工作量门禁。"""

                    nonlocal target_samples, workload_result
                    target_samples = int(
                        symbol_count * attribution_settings.vpp.samples_per_ui
                    )
                    if target_samples < 8:
                        raise ValueError("理想码型周期按 M 上采样后至少需要 8 个样点")
                    workload_result = _estimate_workload(
                        attribution_settings,
                        physical_resolution_hz=max(
                            reference_pulse.sample_rate_hz / reference_pulse.samples,
                            dut_pulse.sample_rate_hz / dut_pulse.samples,
                            dut_pulse.sample_rate_hz / target_samples,
                        ),
                        target_samples=target_samples,
                        other_input_samples=other_input_samples,
                    )

                # 外部文件只打开一次；同一描述符先计数并门禁，再解析为冻结电平数组。
                prepared_vpp_pattern_levels = load_pattern_levels(
                    attribution_settings.vpp,
                    symbol_count_preflight=preflight_vpp_symbol_count,
                )
                if workload_result is None:
                    raise RuntimeError("理想码型周期工作量预检未执行")
                parsed_target_samples = int(
                    prepared_vpp_pattern_levels.size
                    * attribution_settings.vpp.samples_per_ui
                )
                if parsed_target_samples != target_samples:
                    raise ValueError("理想码型解析点数与同描述符预检不一致")
            else:
                # 非 Vpp 路径没有码型回调，直接按目标记录长度完成原有工作量估算。
                workload_result = _estimate_workload(
                    attribution_settings,
                    physical_resolution_hz=max(
                        reference_pulse.sample_rate_hz / reference_pulse.samples,
                        dut_pulse.sample_rate_hz / dut_pulse.samples,
                        dut_pulse.sample_rate_hz / target_samples,
                    ),
                    target_samples=target_samples,
                    other_input_samples=other_input_samples,
                )
            # 在分配镜像频谱前应用候选数和峰值内存门禁。
            _candidate_count, total_evaluations, notice = workload_result
            # 长任务提示在 prepare 前发送，用户无需等到首个 IFFT 才看到成本。
            if notice:
                # 版本随文字一起发送，防止旧线程覆盖当前状态栏。
                self.noticed.emit(notice, self.request.version)
            # 提前建立确定进度范围，零表示尚未完成任何反事实。
            self.progressed.emit(0, total_evaluations)
            # 成本估算后若收到取消则不再分配工作区。
            if self.isInterruptionRequested():
                # 保持取消状态，不返回半份结果。
                raise RuntimeError("影响频段分析已取消")
            # 一次准备缓存目标 DFT、复频响比、基线指标和候选几何。
            workspace = prepare_frequency_attribution(
                reference_pulse,
                dut_pulse,
                attribution_settings,
                prepared_vpp_pattern_levels=prepared_vpp_pattern_levels,
            )
            # prepare 包含一次较大的 FFT；完成后立即给关闭请求一次退出机会。
            if self.isInterruptionRequested():
                # 不进入三模式扫描。
                raise RuntimeError("影响频段分析已取消")
            # 核心回调把评估进度转发到主线程。
            def report_progress(completed: int, total: int) -> None:
                """从后台安全发射整数进度。"""

                # Qt 自动以队列连接跨线程传递信号。
                self.progressed.emit(completed, total)

            # 扫描支持线程中断请求，在每个候选之间停止。
            result = scan_frequency_attribution(
                workspace,
                progress=report_progress,
                cancelled=self.isInterruptionRequested,
            )
            # 主动取消不产生可误读的部分推荐。
            if result.status == "cancelled":
                # 使用明确文字交给主窗口状态栏。
                raise RuntimeError("影响频段分析已取消")
            # 列表顺序同时保留推荐置顶和候选点选映射。
            displayed_candidates = _ordered_candidates(result)
            # 扫描结束到默认大图回放之间也响应关闭请求。
            if self.isInterruptionRequested():
                # 不为即将丢弃的页面生成波形或眼图轨迹。
                raise RuntimeError("影响频段分析已取消")
            # 默认回放第一候选，眼模式同时生成三组叠加轨迹。
            selected_evaluation, eye_comparison = _evaluate_default_candidate(
                workspace,
                displayed_candidates,
            )
            # 默认候选可能包含一次 IFFT 和三份轨迹提取，完成后再对称检查取消。
            if self.isInterruptionRequested():
                # 不构造也不发射已过期的完整运行状态。
                raise RuntimeError("影响频段分析已取消")
            # 冻结完整运行状态供主窗口保存和候选切换。
            run = InfluenceRun(
                workspace=workspace,
                result=result,
                displayed_candidates=displayed_candidates,
                selected_evaluation=selected_evaluation,
                eye_comparison=eye_comparison,
                version=self.request.version,
            )
            # 成功信号只在完整默认展示准备好后发出。
            self.succeeded.emit(run, self.request.version)
        # 所有 I/O、参数和算法异常都由主窗口统一显示。
        except Exception as error:  # GUI boundary: convert failure to text.
            # 异常类型通常由具体文字已表达，不输出冗长 traceback 到弹窗。
            message = f"{type(error).__name__}: {error}"
            # 失败信号附带版本，避免旧任务覆盖新参数状态。
            self.failed.emit(message, self.request.version)

# 用独立线程重放用户点选的一个候选，复用已有频谱缓存而不重跑完整扫描。
class InfluenceSelectionThread(QThread):
    """后台重放一个已扫描候选，避免点选时阻塞 Qt 主线程。"""

    # 成功信号携带点选结果与影响页版本。
    succeeded = Signal(object, int)
    # 失败信号沿用分析线程的文字+版本接口。
    failed = Signal(str, int)

    # 保存只读工作区、候选和页面版本，为晚到结果提供完整的过期判断依据。
    def __init__(
        self,
        workspace: PreparedAttribution,
        candidate: BandAttribution,
        version: int,
    ) -> None:
        """保存只读工作区与候选，不重复加载文件。"""

        # 初始化 QThread。
        super().__init__()
        # 工作区中的大型数组均为只读，可安全跨线程读取。
        self.workspace = workspace
        # 候选数据类不可变。
        self.candidate = candidate
        # 版本用于拒绝参数变化后的旧回放。
        self.version = version

    # 执行一次局部补偿 IFFT，并仅在眼指标下追加共时窗轨迹提取。
    def run(self) -> None:
        """重放局部补偿并构造可选眼图。"""

        # 捕获候选谱零点或眼图轨迹异常。
        try:
            # 用户快速修改参数或关闭窗口时，候选回放可在 IFFT 前停止。
            if self.isInterruptionRequested():
                # 不产生过期详情。
                raise RuntimeError("影响频段分析已取消")
            # 使用已有频响与目标频谱缓存计算补偿后波形。
            evaluation = evaluate_attribution_band(
                self.workspace,
                self.candidate.band,
                self.candidate.mode,
            )
            # 单次 IFFT 无法安全强停，但结束后会在构造轨迹前再次响应中断。
            if self.isInterruptionRequested():
                # 丢弃过期补偿波形。
                raise RuntimeError("影响频段分析已取消")
            # 扫描时有效而回放时无效属于需要展示的明确错误。
            if not evaluation.attribution.valid:
                # 把领域原因送回主线程。
                raise ValueError(evaluation.attribution.invalid_reason)
            # Vpp 不生成眼图。
            eye_comparison = None
            # 眼指标为当前候选构造三图共同时窗轨迹。
            if self.workspace.settings.metric in {"eye_height", "eye_width"}:
                # 复用核心展示数据生成器。
                eye_comparison = build_eye_comparison(self.workspace, evaluation)
            # 轨迹构造完成后最后检查一次，避免晚到结果覆盖新选择。
            if self.isInterruptionRequested():
                # 不发送成功信号。
                raise RuntimeError("影响频段分析已取消")
            # 包装点选结果。
            selection = InfluenceSelection(
                candidate=self.candidate,
                evaluation=evaluation,
                eye_comparison=eye_comparison,
                version=self.version,
            )
            # 成功结果发回主线程。
            self.succeeded.emit(selection, self.version)
        # GUI 边界把异常转成简短错误文字。
        except Exception as error:  # GUI boundary: convert failure to text.
            # 保留异常类型帮助定位输入还是数值问题。
            message = f"{type(error).__name__}: {error}"
            # 发送失败和版本。
            self.failed.emit(message, self.version)

# 将领域候选转换为含 GHz 频段、补偿类型和改善量的简短可见文本。
def candidate_label(
    candidate: BandAttribution,
    *,
    recommended: bool = False,
    unit_suffix: str = "",
) -> str:
    """把候选格式化为简洁列表文字。"""

    # 三个内部模式映射为用户确认的短标签。
    mode_labels = {
        "magnitude": "幅度",
        "phase": "相位",
        "both": "幅相",
    }
    # 推荐第一行增加前缀，其他行不重复“候选”字样。
    prefix = "推荐 · " if recommended else ""
    # GHz 只在展示层换算，领域对象继续保存 Hz。
    return (
        f"{prefix}{candidate.band.low_hz / 1.0e9:.3f}–"
        f"{candidate.band.high_hz / 1.0e9:.3f} GHz · "
        f"{mode_labels[candidate.mode]} · 改善 {candidate.improvement:.4g}{unit_suffix}"
    )


# 根据冻结的指标合同返回曲线纵轴和候选数值单位，避免把 Vrms 标成 Vpp。
def _metric_display_contract(workspace: object) -> tuple[str, str]:
    """返回当前工作区的改善量纵轴文字和候选单位后缀。"""

    # 轻量展示测试或旧调用方可能没有 settings，此时保留通用无量纲文字。
    settings = getattr(workspace, "settings", None)
    # 没有真实领域设置就不猜测指标类型。
    if settings is None:
        return "改善量", ""
    # Vpp 的频域方法是复频谱 AC 误差，物理单位必须明确为 Vrms。
    if settings.metric == "vpp":
        vpp_settings = getattr(settings, "vpp", None)
        if vpp_settings is not None and vpp_settings.method == "frequency_rms_error":
            return "频域误差改善 (Vrms)", " Vrms"
        # LFP 比较完整周期 max-min，结果单位是电压 V。
        return "LFP Vpp 差距改善 (V)", " V"
    # 眼宽按 UI 报告；眼高已由公共幅度基准归一化，因此不附电压单位。
    if settings.metric == "eye_width":
        return "眼宽改善 (UI)", " UI"
    # 归一化眼高是无量纲比值。
    return "归一化眼高改善", ""

# 把不规则候选对象整理为三条等长曲线、显式有效掩码和可点击标签协议。
def influence_curve_payload(run: InfluenceRun) -> dict[str, object]:
    """把领域扫描结果转换为页面影响曲线和候选列表协议。"""

    # 候选中心来自工作区物理网格，单位保持 Hz。
    frequency_hz = np.array(
        [band.center_hz for band in run.workspace.candidates],
        dtype=np.float64,
    )
    # 为每个物理中心建立稳定数组下标，扫描结果不依赖候选元组的排列方式。
    center_indexes = {
        float(center_hz): index for index, center_hz in enumerate(frequency_hz)
    }
    # 三模式得分先填 NaN；NaN 表示不可解析，不能与真实零改善混为一谈。
    scores = {
        mode: np.full(frequency_hz.shape, np.nan, dtype=np.float64)
        for mode in ("magnitude", "phase", "both")
    }
    # 独立布尔掩码使展示层无需从数值大小猜测候选是否有效。
    valid_masks = {
        mode: np.zeros(frequency_hz.shape, dtype=np.bool_)
        for mode in ("magnitude", "phase", "both")
    }
    # 逐项把领域结果放回对应中心和模式；无效项继续保留 NaN 与 False。
    for candidate in run.result.candidates:
        # 取得该候选中心在统一频率轴上的位置。
        center_index = center_indexes.get(float(candidate.band.center_hz))
        # 不属于工作区候选网格的结果不能错误覆盖任一真实频点。
        if center_index is None:
            # 缺失位置最终仍由全 False 掩码计入不可解析数量。
            continue
        # 只有领域有效且改善量有限时才形成真实曲线点。
        if candidate.valid and np.isfinite(candidate.improvement):
            # 有效零改善会按原值 0.0 保存，不会被视为缺失。
            scores[candidate.mode][center_index] = candidate.improvement
            # 同位置掩码同步置真，构成可验证的成对协议。
            valid_masks[candidate.mode][center_index] = True
    # 三个模式所有 False 位置都是无法绘制的模式-频段候选。
    invalid_count = int(
        sum(np.count_nonzero(~mask) for mask in valid_masks.values())
    )
    # 简短诊断只说明证据边界，不把失败位置伪装成零影响。
    diagnostic = (
        f"{invalid_count} 个候选不可解析，曲线以断点表示"
        if invalid_count > 0
        else ""
    )
    # 指标显示合同同时供纵轴和候选列表使用，确保数值单位不会彼此矛盾。
    metric_axis_label, unit_suffix = _metric_display_contract(run.workspace)
    # 第一行是否为推荐由领域对象值比较确定。
    labels = [
        candidate_label(
            candidate,
            recommended=(
                run.result.recommendation is not None
                and candidate == run.result.recommendation
            ),
            unit_suffix=unit_suffix,
        )
        for candidate in run.displayed_candidates
    ]
    # 返回页面可直接验证并绘制的轻量映射。
    return {
        "frequency_hz": frequency_hz,
        "scores": scores,
        "valid_masks": valid_masks,
        "invalid_count": invalid_count,
        "diagnostic": diagnostic,
        "candidates": labels,
        "metric_axis_label": metric_axis_label,
    }

# 将三份眼图轨迹绑定到完全相同的 UI/电压坐标，保证角色不会串位。
def eye_payload(comparison: EyeComparisonData) -> dict[str, object]:
    """把三组共时窗轨迹转成页面角色映射。"""

    # 公共横轴是以主光标为中心的 -1 到 +1 UI，长度严格为 2*M+1。
    time_ui = comparison.time_ui
    # 共同纵轴范围由三组轨迹一次确定，不允许单图自动缩放伪造改善。
    amplitude_range_v = comparison.amplitude_range_v
    # 角色键与 InfluenceBandPage 的展示顺序严格一致。
    return {
        "time_ui": time_ui,
        "amplitude_range_v": amplitude_range_v,
        "reference": {"traces_v": comparison.reference_traces_v},
        "before": {"traces_v": comparison.before_traces_v},
        "after": {"traces_v": comparison.after_traces_v},
    }

# 暴露参考、补偿前和补偿后的模型时间轴；旧兼容路径仍允许独立采样网格。
def waveform_payload(
    workspace: PreparedAttribution,
    evaluation: BandEvaluation,
) -> dict[str, object]:
    """为 Vpp 页面准备参考、补偿前和补偿后三条模型波形。"""

    # Vpp 工作区必须保存参考模型数据。
    if workspace.reference_waveform is None or evaluation.corrected_values is None:
        # 不完整状态不应渲染占位曲线。
        raise ValueError("Vpp 候选缺少参考或补偿后波形")
    # 三条记录允许使用不同时间轴；页面分别绘制真实秒值。
    return {
        "reference": {
            "time_s": workspace.reference_waveform.time_s,
            "values": workspace.reference_waveform.values[:, 0],
        },
        "before": {
            "time_s": workspace.target_signal.time_s,
            "values": workspace.target_signal.values[:, 0],
        },
        "after": {
            "time_s": workspace.target_signal.time_s,
            "values": evaluation.corrected_values[:, 0],
        },
    }

# 明确本模块供主窗口使用的线程、数据协议和展示适配接口，隐藏内部成本估算细节。
__all__ = [
    "InfluenceAnalysisThread",
    "InfluenceRequest",
    "InfluenceRun",
    "InfluenceSelection",
    "InfluenceSelectionThread",
    "candidate_label",
    "eye_payload",
    "influence_curve_payload",
    "waveform_payload",
]
