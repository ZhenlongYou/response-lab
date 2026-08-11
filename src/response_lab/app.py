"""ResponseLab 的命令行路由、无窗口自检与 Qt 启动。"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本文件仅关闭对应格式告警。
# ruff: noqa: I001

# 延迟解析类型标注，避免仅用于注解的类型在模块导入阶段产生额外依赖。
from __future__ import annotations

# argparse 负责把无窗口自检、GUI 烟雾测试和正常启动分成明确的命令行路由。
import argparse
# dataclass 为不依赖 Qt 的演示归因结果提供结构化只读载体。
from dataclasses import dataclass
# os 只用于设置 Qt 平台后端，确保无显示环境也能构造并截图真实窗口。
import os
# sys 提供平台名称、Qt 启动参数以及应用进程的生命周期上下文。
import sys
# Path 让截图输出路径同时兼容 macOS、Windows 与 Linux 的路径规则。
from pathlib import Path

# NumPy 用于生成确定性演示脉冲、频谱缺口和自检统计量。
import numpy as np

# 影响频段自检走与真实页签相同的准备、扫描、回放和 2 UI 轨迹路径。
from .attribution import (
    AttributionSettings,
    BandAttribution,
    BandEvaluation,
    EyeComparisonData,
    FrequencyAttributionResult,
    PreparedAttribution,
    VirtualEyeSettings,
    build_eye_comparison,
    evaluate_attribution_band,
    prepare_frequency_attribution,
    scan_frequency_attribution,
)
# 正常补偿演示调用产品实际 DSP 入口，避免自检另写一套简化算法。
from .dsp import run_compensation
# 模型类型固定演示数据的时间单位为秒、采样率单位为 Hz，并约束返回结构。
from .models import CompensationRun, CompensationSettings, TimeSeries


class GuiDependencyError(RuntimeError):
    """GUI dependency is missing or its native binary cannot be loaded."""


# 自检结果只组合纯算法对象，不引用含 QThread 的页面控制器，使 --self-test 无需 Qt。
@dataclass(frozen=True)
class _DemoInfluenceRun:
    """已知频段演示的纯算法运行结果。"""

    # workspace 保存扫描与点选共同使用的只读频谱缓存。
    workspace: PreparedAttribution
    # result 保存全频诊断、局部候选和推荐。
    result: FrequencyAttributionResult
    # 演示只展示推荐候选，但仍沿用页面的稳定候选元组协议。
    displayed_candidates: tuple[BandAttribution, ...]
    # selected_evaluation 保存推荐候选的完整回放。
    selected_evaluation: BandEvaluation
    # eye_comparison 保存三组 2 UI 轨迹和共同显示范围。
    eye_comparison: EyeComparisonData
    # version 与真实页面协议同形，自检固定为零。
    version: int


# 构造旧有补偿链路的确定性数据，供无文件自检和主窗口首次预览复用。
def build_demo_run() -> CompensationRun:
    """生成不依赖文件的确定性演示运行，供自检、截图和首次界面预览使用。"""

    # 2 GSa/s 对应 0.5 ns 采样间隔，覆盖演示输入的 120 MHz 与 240 MHz 分量。
    sample_rate_hz = 2.0e9
    # 2048 个拟合脉冲样点兼顾频率分辨率与自检启动速度。
    pulse_samples = 2048
    # index 是无量纲样点序号，后续除以采样率才转换成秒。
    index = np.arange(pulse_samples, dtype=np.float64)
    # 参考脉冲以第 420 点为中心、标准差为 3 个样点，峰值归一化为 1。
    reference_values = np.exp(-0.5 * ((index - 420.0) / 3.0) ** 2)
    # 先建立与参考等长的零数组，再注入可手工核对的增益和时延差异。
    dut_values = np.zeros_like(reference_values)
    # DUT 相对参考衰减到 72% 并延迟 5 点，即在 2 GSa/s 下延迟 2.5 ns。
    dut_values[5:] = 0.72 * reference_values[:-5]
    # TimeSeries 的横轴使用秒；采样率由相邻点间隔可反向验证。
    pulse_time_s = index / sample_rate_hz
    # 单列数组表示一条参考通道，避免把样点维误当成通道维。
    reference = TimeSeries(pulse_time_s, reference_values[:, None], sample_rate_hz)
    # DUT 与参考共用同一时间轴，演示差异只来自上面注入的脉冲响应。
    dut = TimeSeries(pulse_time_s, dut_values[:, None], sample_rate_hz)

    # 16384 点约覆盖 8.192 us，使两个演示正弦都包含足够多周期供 RMS 自检。
    signal_samples = 16384
    # 输入波形时间轴仍以秒表示，并与拟合脉冲保持相同 2 GSa/s。
    signal_time_s = np.arange(signal_samples, dtype=np.float64) / sample_rate_hz
    # 两个幅值按输入波形电压单位解释；0.4 rad 固定相位保证演示可重复。
    input_values = (
        0.55 * np.sin(2.0 * np.pi * 120.0e6 * signal_time_s)
        + 0.22 * np.sin(2.0 * np.pi * 240.0e6 * signal_time_s + 0.4)
    )
    # 把一维演示电压扩成单通道 TimeSeries，交给真实补偿入口处理。
    input_signal = TimeSeries(signal_time_s, input_values[:, None], sample_rate_hz)
    # 设置同时补幅度和相位，并把分析频带限制在演示信号有意义的 10–300 MHz。
    settings = CompensationSettings(
        mode="both",
        band_low_hz=10.0e6,
        band_high_hz=300.0e6,
        phase_fit_low_hz=20.0e6,
        phase_fit_high_hz=250.0e6,
        detrend_phase=True,
        analysis_points=8193,
    )
    # 返回实际 DSP 运行结果，使自检覆盖频响估计、频域相乘和时域重建整条链路。
    return run_compensation(reference, dut, input_signal, settings)


# 构造带已知频段真值的归因运行，供算法 oracle 与第六页签截图共同复用。
def build_demo_influence_run() -> _DemoInfluenceRun:
    """构造已知 200–300 MHz 纯幅度缺口，供算法和真实页签共同自检。"""

    # 8 GSa/s 和 2048 点脉冲给出约 3.91 MHz 物理分辨率，足以验证 100 MHz 候选。
    sample_rate_hz = 8.0e9
    # Np=64、M=32 对应附件示例的高分辨率轨迹，同时保持 2048 点自检规模。
    pulse_length_ui = 64
    # 每 UI 三十二点决定 250 MBd，并让每条 2 UI 轨迹包含 65 个样点。
    samples_per_ui = 32
    # 总点数严格满足 Np*M。
    samples = pulse_length_ui * samples_per_ui
    # 时间轴用于 TimeSeries 反算采样率。
    time_s = np.arange(samples, dtype=np.float64) / sample_rate_hz
    # 以记录中心为零点建立 UI 横轴，脉冲形状与用户附件的主光标、前后游标思路一致。
    pulse_center = samples // 2
    # UI 横轴只用于生成演示拟合脉冲，产品算法仍从 CSV 时间轴推导真实采样率。
    pulse_time_ui = (
        np.arange(samples, dtype=np.float64) - float(pulse_center)
    ) / samples_per_ui
    # 主脉冲采用带高斯包络的 sinc，既有清晰主光标也保留可见的确定性 ISI。
    main = np.sinc(pulse_time_ui / 0.72) * np.exp(
        -(pulse_time_ui / 2.2) ** 2
    )
    # 轻微前游标用于让叠加轨迹呈现真实的非对称形状。
    precursor = -0.08 * np.exp(-((pulse_time_ui + 0.75) / 0.24) ** 2)
    # 第一后游标构成主要的确定性尾迹。
    postcursor = 0.12 * np.exp(-((pulse_time_ui - 1.0) / 0.33) ** 2)
    # 较远后游标使多符号卷积路径不能退化成单脉冲平移测试。
    postcursor_2 = -0.05 * np.exp(-((pulse_time_ui - 2.0) / 0.45) ** 2)
    # 合成附件风格的实际拟合脉冲。
    reference_values = main + precursor + postcursor + postcursor_2
    # 参考主光标归一化为一；DUT 后续沿用同一电压尺度。
    reference_values /= reference_values[np.argmax(np.abs(reference_values))]
    # RFFT 频率网格与拟合脉冲时间轴严格对应。
    frequency_hz = np.fft.rfftfreq(samples, d=1.0 / sample_rate_hz)
    # 缺口权重在 200–300 MHz 满幅，两侧各用 50 MHz 余弦肩部平滑连接。
    defect_weight = np.zeros_like(frequency_hz)
    # 用户看到的核心内保持满权，构成已知真值频段。
    defect_weight[(frequency_hz >= 200.0e6) & (frequency_hz <= 300.0e6)] = 1.0
    # 左肩从 150 MHz 的零权连续上升到 200 MHz 的满权。
    left_shoulder = (frequency_hz >= 150.0e6) & (frequency_hz < 200.0e6)
    # 半余弦避免硬切频带在脉冲中制造额外强振铃。
    defect_weight[left_shoulder] = 0.5 * (
        1.0
        - np.cos(
            np.pi * (frequency_hz[left_shoulder] - 150.0e6) / 50.0e6
        )
    )
    # 右肩从 300 MHz 的满权连续下降到 350 MHz 的零权。
    right_shoulder = (frequency_hz > 300.0e6) & (frequency_hz <= 350.0e6)
    # 与左肩镜像的半余弦保持响应实数、非负且相位不变。
    defect_weight[right_shoulder] = 0.5 * (
        1.0
        + np.cos(
            np.pi * (frequency_hz[right_shoulder] - 300.0e6) / 50.0e6
        )
    )
    # 40% 衰减不穿过零点，因此真值只包含幅度差异。
    magnitude_response = 1.0 - 0.4 * defect_weight
    # 频域只乘实数正响应，DUT 与参考不存在额外相位差异。
    dut_values = np.fft.irfft(
        np.fft.rfft(reference_values) * magnitude_response,
        n=samples,
    )
    # 构造参考拟合脉冲。
    reference_pulse = TimeSeries(
        time_s,
        reference_values[:, None],
        sample_rate_hz,
    )
    # 构造 DUT 拟合脉冲。
    dut_pulse = TimeSeries(time_s, dut_values[:, None], sample_rate_hz)
    # 扫描 100–500 MHz，四个 100 MHz 候选包含已知真值窗。
    settings = AttributionSettings(
        metric="eye_height",
        scan_low_hz=100.0e6,
        scan_high_hz=500.0e6,
        eye=VirtualEyeSettings(
            modulation="nrz",
            pulse_length_ui=pulse_length_ui,
            samples_per_ui=samples_per_ui,
            symbol_count=1200,
            random_seed=20260718,
        ),
    )
    # 准备一次频响和固定符号缓存。
    workspace = prepare_frequency_attribution(
        reference_pulse,
        dut_pulse,
        settings,
    )
    # 扫描三模式及全部候选。
    result = scan_frequency_attribution(workspace)
    # 自检 oracle 必须产生一个候选才能继续生成真实展示数据。
    if result.recommendation is None:
        # 此失败说明算法已无法定位已知合成缺口。
        raise RuntimeError("影响频段自检失败：已知幅度缺口没有推荐")
    # 用推荐候选重放补偿后拟合脉冲。
    evaluation = evaluate_attribution_band(
        workspace,
        result.recommendation.band,
        result.recommendation.mode,
    )
    # 三幅眼图使用共同坐标，并叠加相同符号序列得到的 2 UI 轨迹。
    comparison = build_eye_comparison(workspace, evaluation)
    # 纯算法载体与真实后台成功结果具有相同字段，但不会为了自检导入 Qt。
    return _DemoInfluenceRun(
        workspace=workspace,
        result=result,
        displayed_candidates=(result.recommendation,),
        selected_evaluation=evaluation,
        eye_comparison=comparison,
        version=0,
    )


# 执行完全不依赖 Qt 的算法自检，并用进程退出码向 PyCharm 或 CI 报告结果。
def run_self_test() -> int:
    """不导入 Qt 的算法入口自检，适合 PyCharm、CI 与无显示终端。"""

    # 先运行旧有补偿演示，确认新增页签没有破坏基础补偿路径。
    run = build_demo_run()
    # 输出必须保持输入的样点数和通道数，否则导出波形将无法对应原时间轴。
    if run.output_values.shape != run.input_signal.values.shape:
        # 形状变化表示补偿链路错误地裁剪、扩展或转置了用户波形。
        raise RuntimeError("自检失败：补偿输出形状改变")
    # 所有输出样点都必须有限，防止极小频响除法把 NaN/Inf 传播到导出文件。
    if not np.all(np.isfinite(run.output_values)):
        # 非有限值说明数值保护失效，不能把该结果宣称为可用补偿波形。
        raise RuntimeError("自检失败：补偿输出包含 NaN/Inf")
    # 演示合同要求先去除线性相位斜率，避免纯时延被误判为相位畸变。
    if not run.analysis.settings.detrend_phase:
        # 该失败通常意味着默认配置或设置传递发生回归。
        raise RuntimeError("自检失败：默认相位未去斜")
    # 输入 RMS 使用原波形单位；若输入按 V 读取，则这里的单位也是 V RMS。
    input_rms = float(np.sqrt(np.mean(run.input_signal.values**2)))
    # 输出 RMS 与输入同单位，摘要中的比值可快速发现异常增益爆炸或全零输出。
    output_rms = float(np.sqrt(np.mean(run.output_values**2)))
    # 额外执行已知频段归因 oracle，避免自检只覆盖 shape/finite。
    influence = build_demo_influence_run()
    # 已知合成缺口必须识别为纯幅度模式。
    if (
        influence.result.recommendation is None
        or influence.result.recommendation.mode != "magnitude"
    ):
        # 模式交换或相位符号错误会在此直接失败。
        raise RuntimeError("自检失败：已知纯幅度缺口的模式识别错误")
    # 推荐窗必须与 200–300 MHz 注入真值相交。
    if not (
        influence.result.recommendation.band.high_hz > 200.0e6
        and influence.result.recommendation.band.low_hz < 300.0e6
    ):
        # Hz/GHz 或候选边界错误会触发此物理 oracle。
        raise RuntimeError("自检失败：主要影响频段未覆盖已知真值")
    # 打印可人工核对的采样率、算法路径、已知真值定位和 RMS，而非只给一个 PASS。
    print(
        "ResponseLab self-test: PASS\n"
        f"  pulse sample rate: {run.reference_pulse.sample_rate_hz:.6g} Hz\n"
        "  phase detrend: linear phase removed before phase comparison\n"
        "  application: exact-bin CZT pulse ratio + FFT multiply + IFFT\n"
        "  attribution: known 200–300 MHz magnitude defect localized\n"
        f"  input/output RMS: {input_rms:.6g} / {output_rms:.6g}"
    )
    # 返回 0 表示全部物理与数值 oracle 通过，供 shell、PyCharm 和 CI 判定成功。
    return 0


# 延迟创建或复用唯一 QApplication，隔离 Qt 依赖并维持正确对象生命周期。
def _qt_application():
    """延迟导入 GUI 依赖，使 ``--self-test`` 在无 Qt 环境也能给出算法结果。"""

    # 仅 GUI 路由才加载 PySide6，因此纯算法自检在未安装 Qt 时仍可运行到明确结果。
    try:
        # pyqtgraph 是主窗口的直接 GUI 依赖；在导入本地 UI 前单独验证，避免误包内部错误。
        __import__("pyqtgraph")
        # 字体类型用于选择系统默认字体，并修复 macOS offscreen 的无效通用别名。
        from PySide6.QtGui import QFont, QFontDatabase
        # QApplication 是所有 Qt 控件的唯一应用级所有者，必须先于主窗口存在。
        from PySide6.QtWidgets import QApplication
    # 依赖缺失时把底层 ImportError 转换成用户可直接执行的安装指令。
    except (ImportError, OSError) as exc:
        # 保留原异常链，便于开发者区分未安装 PySide6 与其二进制依赖加载失败。
        raise GuiDependencyError(
            "缺少或无法加载 PySide6 GUI 依赖（含 pyqtgraph）"
        ) from exc
    # 复用测试环境已有实例；新建时只传程序名，避免本工具的 CLI 参数被 Qt 误解析。
    application = QApplication.instance() or QApplication(sys.argv[:1])
    # 应用名用于窗口系统、日志以及 Qt 设置命名空间识别本程序。
    application.setApplicationName("ResponseLab")
    # 组织名与项目归属保持一致，避免 Qt 持久设置与其他应用发生键冲突。
    application.setOrganizationName("RinysProject")
    # 优先采用操作系统通用界面字体，使中文和数字在不同平台保持可读。
    system_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    # 只在 macOS 返回无效 Sans Serif 别名时替换，正常系统字体不受影响。
    if sys.platform == "darwin" and system_font.family() == "Sans Serif":
        # Qt 的 macOS offscreen 后端有时返回一个不存在的通用别名，导致启动告警。
        system_font = QFont("Helvetica Neue", system_font.pointSize())
    # 在创建主窗口前设置全局字体，让所有后续控件继承同一字体度量。
    application.setFont(system_font)
    # 调用方持有该引用直到事件循环或烟雾测试结束，防止 Qt 应用对象被提前回收。
    return application


# 构造并检查真实主窗口；可选截图但不进入永久事件循环，适合 CI/offscreen 验证。
def run_gui_smoke_test(render_path: Path | None = None) -> int:
    """构造真实主窗口并可选渲染截图，不进入永久事件循环。"""

    # 先在专门边界验证 GUI 依赖；后续本地模块错误应保留真实 traceback。
    application = _qt_application()
    try:
        # QTest 提供短暂事件等待，让布局、绘图和延迟重绘在截图前真正完成。
        from PySide6.QtTest import QTest
    except (ImportError, OSError) as exc:
        raise GuiDependencyError("缺少或无法加载 PySide6 GUI 测试依赖") from exc

    # 延迟导入主窗口，保证仅运行 --self-test 时不会间接加载任何 Qt 控件。
    from .ui import ResponseLabWindow
    # 展示适配器依赖 Qt 线程类型，只在真实 GUI 烟雾路径中延迟加载。
    from .influence_controller import eye_payload, influence_curve_payload

    # application 在整个函数内保持强引用，确保所有控件都拥有有效 Qt 父对象。
    # 创建产品实际主窗口，而不是为烟雾测试另造无法代表用户路径的测试壳。
    window = ResponseLabWindow()
    # 1440×900 像素覆盖常用桌面视口，也给三幅眼图留出可判读空间。
    window.resize(1440, 900)
    # 注入确定性补偿结果，让原有五个页签也经过真实数据展示路径。
    window.present_run(build_demo_run(), source_label="内置演示数据")
    # 第六页签使用真实归因引擎结果，而不是只验证空页签数量。
    influence = build_demo_influence_run()
    # 曲线和候选列表通过正式展示适配器生成。
    influence_view = influence_curve_payload(influence)
    # 自检构造函数保证眼图比较存在。
    if influence.eye_comparison is None:
        # 缺少三图表示默认候选回放链路已损坏。
        raise RuntimeError("GUI 自检失败：影响频段没有眼图结果")
    # 将三幅共坐标轨迹加入页面协议。
    influence_view["eyes"] = eye_payload(influence.eye_comparison)
    # 截图同时展示当前候选与三组眼高数值，覆盖真实页签的结果摘要区域。
    recommendation = influence.result.recommendation
    # 构造函数已经确认推荐存在；断言让静态类型和后续格式化都保持明确。
    if recommendation is None:
        # 该分支只在自检构造合同被破坏时触发。
        raise RuntimeError("GUI 自检失败：影响频段没有推荐候选")
    # 界面只使用用户需要的业务名称，不增加“虚拟”或内部 ISI 术语。
    influence_view["summary"] = (
        f"推荐 {recommendation.band.low_hz / 1.0e9:.3f}–"
        f"{recommendation.band.high_hz / 1.0e9:.3f} GHz · 幅度\n"
        f"参考 {influence.result.reference_metric:.4g} · "
        f"补偿前 {influence.result.before_metric:.4g} · "
        f"补偿后 {recommendation.metric_after:.4g}"
    )
    # 切换为眼高以显示参考、补偿前和补偿后三幅图。
    window.influence_page.metric_combo.setCurrentIndex(1)
    # M=32 与 8 GSa/s 演示时间轴共同推导 250 MBd，Np=64 从 2048 点自动得到。
    window.influence_page.m_spin.setValue(32)
    # 渲染真实影响曲线和眼图。
    window.influence_page.render_result(influence_view)
    # 截图和烟雾测试聚焦新增页签。
    window.visual_tabs.setCurrentIndex(window.influence_tab_index)
    # show 触发布局和绘图对象创建；offscreen 后端不会真的弹出桌面窗口。
    window.show()
    # 先处理一轮排队事件，使尺寸变更和页签切换传递到子控件。
    application.processEvents()
    # 等待 300 ms 给 pyqtgraph 多轨迹和字体栅格化留出稳定时间。
    QTest.qWait(300)
    # 再处理等待期间产生的重绘事件，保证随后检查和截图看到最终帧。
    application.processEvents()
    # 六个页签是新增“影响频段”页面已接入主窗口的最小结构合同。
    if window.visual_tabs.count() != 6:
        # 数量错误说明页面未注册或既有页签被误删，GUI 集成不能视为通过。
        raise RuntimeError("GUI 自检失败：可视化页面数量不是 6")
    # 三个物理模式必须各有一条真实影响曲线。
    if len(window.influence_page.impact_plot.listDataItems()) != 3:
        # 只创建空 PlotWidget 不能通过烟雾测试。
        raise RuntimeError("GUI 自检失败：影响曲线数量不是 3")
    # 三幅眼图各自必须包含覆盖 -1 UI 到 +1 UI 的多轨迹数据，而不是空坐标框。
    for plot in (
        window.influence_page.reference_plot,
        window.influence_page.before_plot,
        window.influence_page.after_plot,
    ):
        # 使用原始 xData/yData；getData 会按当前视窗降采样并可能改写 NaN 分隔形态。
        trace_data = [
            (item.xData, item.yData)
            for item in plot.listDataItems()
        ]
        # 至少一项必须横跨完整 2 UI 且含多条被 NaN 分开的轨迹。
        has_eye_traces = any(
            x_values is not None
            and y_values is not None
            and np.asarray(x_values).size > 65
            and np.nanmin(np.asarray(x_values, dtype=np.float64)) <= -1.0
            and np.nanmax(np.asarray(x_values, dtype=np.float64)) >= 1.0
            and np.any(np.isnan(np.asarray(y_values, dtype=np.float64)))
            for x_values, y_values in trace_data
        )
        # 只有坐标轴或零 UI 参考线不能冒充附件要求的轨迹叠加眼图。
        if not has_eye_traces:
            # 缺少任意角色都破坏补偿前后对比。
            raise RuntimeError("GUI 自检失败：三幅眼图缺少 2 UI 叠加轨迹")
    # 只有 --render-ui 提供目标路径时才产生文件，普通烟雾测试不留下构建产物。
    if render_path is not None:
        # 递归创建目标目录，同时允许用户重复渲染到已有目录。
        render_path.parent.mkdir(parents=True, exist_ok=True)
        # grab 获取窗口当前帧；save 返回 False 时必须报告失败而不能假装已有截图。
        if not window.grab().save(str(render_path)):
            # 路径权限、格式插件或磁盘问题都会通过带目标路径的异常显式暴露。
            raise RuntimeError(f"GUI 截图保存失败：{render_path}")
        # 输出绝对或调用方给定路径，方便 CI 和用户直接定位渲染结果。
        print(f"ResponseLab UI render: {render_path}")
    # 主动关闭窗口，验证 closeEvent 的线程清理路径并释放顶层 Qt 资源。
    window.close()
    # 处理 close 产生的延迟删除事件，避免后续测试继承尚未销毁的窗口状态。
    application.processEvents()
    # 仅在结构、曲线、三图和可选截图全部通过后打印成功标记。
    print("ResponseLab GUI smoke-test: PASS")
    # 返回 0 让命令行调用者确认真实 Qt 窗口完成了完整烟雾测试。
    return 0


# 启动用户正常使用的交互式主窗口，并把 Qt 事件循环退出码原样交给操作系统。
def run_gui() -> int:
    """启动正常交互式桌面窗口。"""

    # application 必须在 window 前创建并持续存活到 exec 返回。
    application = _qt_application()
    # 依赖预检后再加载本地 UI；内部 ImportError/OSError 不伪装成依赖安装问题。
    from .ui import ResponseLabWindow
    # 顶层窗口保存在局部强引用中，防止进入事件循环前被 Python 回收。
    window = ResponseLabWindow()
    # 显示窗口后由 Qt 事件循环持续处理用户输入、后台结果和重绘。
    window.show()
    # exec 阻塞到用户退出，并把 Qt 退出码转换成标准 Python int 返回。
    return int(application.exec())


# 解析 CLI 并按优先级路由到算法自检、offscreen GUI 验证或正常交互界面。
def main(arguments: list[str] | None = None) -> int:
    # parser 的说明文字同时服务终端 --help 与 PyCharm 参数配置提示。
    parser = argparse.ArgumentParser(description="ResponseLab 频响分析与补偿")
    # --self-test 明确选择不加载 Qt 的算法路径，适合无显示 CI 和依赖排查。
    parser.add_argument("--self-test", action="store_true", help="运行无窗口算法自检")
    # --gui-smoke-test 创建后关闭真实窗口，用于验证 Qt 集成而不等待人工退出。
    parser.add_argument("--gui-smoke-test", action="store_true", help="构造并关闭真实 Qt 主窗口")
    # --render-ui 接收 PNG 路径，并复用烟雾测试生成可人工审阅的界面快照。
    parser.add_argument("--render-ui", type=Path, help="把内置演示界面渲染为 PNG")
    # arguments 允许测试传入独立参数列表；None 时 argparse 自动读取真实 sys.argv。
    options = parser.parse_args(arguments)
    # 算法自检优先级最高，确保即使误带 GUI 参数也不会加载 Qt 或创建窗口。
    if options.self_test:
        # 原样返回自检退出码，让最外层 main.py 或 CI 正确感知失败。
        return run_self_test()
    # 烟雾测试与截图都需要短生命周期窗口，因此共享同一 offscreen 路由。
    if options.gui_smoke_test or options.render_ui is not None:
        # 仅在用户未指定时选择 offscreen，保留其显式配置 xcb、cocoa 等后端的权利。
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        # render_ui 为 None 时只验证窗口；非 None 时还把最终帧写到指定 PNG。
        return run_gui_smoke_test(options.render_ui)
    # 没有诊断参数时进入日常交互 GUI，这是 PyCharm 直接运行的默认行为。
    return run_gui()
