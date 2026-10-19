"""ResponseLab 的 Codex 风格 PySide6 桌面界面。

界面只负责收集明确参数、调度后台任务和展示结果。所有解析与 DSP 都通过公开模块完成，
因此 GUI 不会拥有一套无法测试的“隐藏算法”。
"""

# 逐语句中文维护注释会自然打断导入块并超过英文字符宽度，文件级忽略仅覆盖这两类格式告警。
# ruff: noqa: E501, I001
from __future__ import annotations

# Mapping 验证影响页通过 Qt 信号发送的轻量参数快照。
from collections.abc import Mapping

# 系统环境变量提供显式的减少动态效果开关，避免需要运动敏感的用户被迫观看动画。
import os

# 系统命令只在首次构造真实 macOS 窗口时读取辅助功能偏好，不参与绘制热路径。
import subprocess

# 平台判断用于仅在 macOS 调用系统偏好读取工具。
import sys

# 把后台异常压缩为可操作的错误文字，再通过 Qt 信号安全送回主线程。
import traceback
import warnings
from dataclasses import dataclass, replace

# 单次缓存避免测试或多窗口重复启动系统命令。
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

# pyqtgraph 承担大数组下采样、平移、框选缩放和工程曲线渲染。
import pyqtgraph as pg
from scipy.fft import rfft

# QtCore 提供后台线程、信号槽、定时关闭、二维坐标和布局方向等 GUI 基础能力。
from PySide6.QtCore import QEvent, QPointF, QSize, Qt, QThread, QTimer, Signal

# QtGui 同时承担拖放事件、矢量绘制和品牌图标加载。
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QEnterEvent,
    QGuiApplication,
    QIcon,
    QLinearGradient,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
)

# QtWidgets 组成三栏桌面工作台、参数表单、反馈状态和导出对话框。
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# UI 只调用经过测试的 DSP 入口，不在按钮回调里复制频响或相位拟合算法。
from .dsp import (
    compare_pulses,
    fit_linear_phase_slope,
    preflight_compensation_shape,
    prepare_response_spectra,
    run_compensation,
    suggest_frequency_settings,
)
from .cancellation import OperationCancelledError, raise_if_cancelled

# 项目 I/O 层统一解析 CSV 与 Keysight 自描述 BIN，并返回真实时间轴。
from .io import (
    BinPayloadLayout,
    load_bin_timeseries,
    load_csv_timeseries,
)

# 影响频段控制器在独立后台线程中加载数据、扫描候选并回放点选结果。
from .influence_controller import (
    InfluenceAnalysisThread,
    InfluenceRequest,
    InfluenceRun,
    InfluenceSelection,
    InfluenceSelectionThread,
    eye_payload,
    influence_curve_payload,
    waveform_payload,
)

# 新页签保持纯展示职责，不在主窗口中复制眼图轨迹和候选列表代码。
from .influence_ui import InfluenceBandPage

# 模型类型明确区分“只比较脉冲”和“已对目标数据完成补偿”两类结果。
from .models import CompensationRun, CompensationSettings, PulseComparison

# 报告层在导出前验证源文件，并原子生成数据、频响诊断和参数清单。
from .reporting import (
    BundleCleanupWarning,
    BundlePaths,
    BundleRollbackError,
    DestinationFingerprint,
    SourceVerificationError,
    bundle_paths,
    export_run_bundle,
    snapshot_bundle_destinations,
)
from .ui_components import CompactDoubleSpinBox


@dataclass(frozen=True)
class ExportOutcome:
    """A committed bundle plus any cleanup detail the GUI must surface."""

    paths: BundlePaths
    cleanup_warning: str = ""


# 绘图工具使用随包分发的矢量图标，避免依赖操作系统主题导致不同电脑显示不一致。
ICON_DIRECTORY = Path(__file__).with_name("assets")

# 深色仪器工作台采用偏蓝黑表面，既降低长时间观察疲劳，也让曲线颜色保持主角地位。
BACKGROUND = "#080C12"
SURFACE = "#0E141D"
# 主分区底部使用更深的同色相表面，建立纵向层级但不抬亮用户确认的近黑色调。
SURFACE_LOW = "#0B1119"
SURFACE_SUBTLE = "#111925"
# 输入卡底部只降低一个明度台阶，使卡片层次来自表面而不是宽阴影。
SURFACE_SUBTLE_LOW = "#101721"
SURFACE_RAISED = "#172130"
SURFACE_HOVER = "#1C293A"
BORDER = "#263448"
BORDER_STRONG = "#344760"
TEXT = "#F4F7FB"
TEXT_MUTED = "#9EACC0"
TEXT_FAINT = "#718198"
# 蓝、琥珀、青绿分别承担主操作、DUT 对比与成功结果，避免仅靠明暗区分曲线。
ACCENT = "#5B8FF9"
ACCENT_BRIGHT = "#78A6FF"
REFERENCE = "#72A7FF"
DUT = "#F2B763"
RESULT = "#45D6B4"
WARNING = "#F4C768"
ERROR = "#FF7A86"

TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9, "ps": 1e-12}
FREQUENCY_FACTORS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

_MAX_OUTPUT_WAVEFORM_PREVIEW_SAMPLES = 40_960
_MAX_OUTPUT_SPECTRUM_PREVIEW_SAMPLES = 1_048_576


def _output_waveform_preview_slice(samples: int) -> slice:
    """返回记录开头的连续有界窗口，供时域波形快速预览。"""

    return slice(0, min(samples, _MAX_OUTPUT_WAVEFORM_PREVIEW_SAMPLES))


def _output_spectrum_preview_slice(samples: int) -> slice:
    """频谱使用中间连续窗口，避免下采样后混叠并限制 FFT 工作区。"""

    window_samples = min(samples, _MAX_OUTPUT_SPECTRUM_PREVIEW_SAMPLES)
    start = (samples - window_samples) // 2
    return slice(start, start + window_samples)


# 统一解析显式环境开关与 macOS 辅助功能设置，保持界面本身无额外常驻控件。
@lru_cache(maxsize=1)
def _prefers_reduced_motion() -> bool:
    """返回当前进程或 macOS 是否请求减少动态效果。"""

    # 环境变量允许 PyCharm、终端和自动化明确关闭脉冲轨迹动画。
    environment_value = os.environ.get("RESPONSELAB_REDUCE_MOTION", "").strip().lower()
    # 常见真值都视为显式请求，避免只接受数字 1 带来使用歧义。
    if environment_value in {"1", "true", "yes", "on"}:
        # 显式请求优先于平台探测，且不再启动外部进程。
        return True
    # 非 macOS 平台没有 defaults 工具，默认保留动画并继续支持上方环境开关。
    if sys.platform != "darwin":
        # 返回 False 表示未检测到减少动态效果请求。
        return False
    # macOS 的 reduceMotion 偏好由 universalAccess 域维护。
    try:
        # 使用绝对路径避免 PyCharm 与终端 PATH 不一致；200 ms 超时防止启动被系统服务阻塞。
        result = subprocess.run(
            ["/usr/bin/defaults", "read", "com.apple.universalAccess", "reduceMotion"],
            capture_output=True,
            text=True,
            timeout=0.2,
            check=False,
        )
    # 缺少命令、系统拒绝读取或超时时安全回退到默认动画行为。
    except (OSError, subprocess.SubprocessError):
        # 辅助偏好探测失败不能阻止主窗口启动。
        return False
    # defaults 成功且返回常见真值时，视为系统已经启用减少动态效果。
    return result.returncode == 0 and result.stdout.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class AnalysisRequest:
    reference_path: Path
    dut_path: Path
    target_path: Path | None
    settings: CompensationSettings
    version: int
    action: Literal["compare", "compensate"]
    auto_frequency_bands: bool = False
    auto_phase_fit_band: bool = False


class FileCard(QFrame):
    """带明确角色、可拖放路径和解析状态的单个输入卡片。"""

    path_selected = Signal(str)

    def __init__(
        self,
        step: str,
        title: str,
        file_filter: str,
    ) -> None:
        super().__init__()
        # 卡片自身提供可访问名称，使读屏用户先听到输入角色，再进入路径和按钮。
        self.file_filter = file_filter
        self.setObjectName("fileCard")
        self.setAccessibleName(f"第 {step} 步，{title}")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 13)
        layout.setSpacing(9)

        # 数字徽标、标题和用途说明组成稳定的步骤层级，避免把编号混进标题字符串。
        title_row = QHBoxLayout()
        title_row.setSpacing(9)
        step_label = QLabel(step)
        step_label.setObjectName("stepBadge")
        step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_row.addWidget(step_label)
        title_row.addWidget(title_label, 1)
        # 路径输入保持只读，点击按钮和拖放两条路径都具有明确文本提示。
        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("选择或拖入文件")
        self.path_edit.setAccessibleName(f"{title}文件路径")
        browse_button = QPushButton("选择")
        browse_button.setObjectName("secondaryButton")
        browse_button.setAccessibleName(f"选择{title}文件")
        browse_button.setMinimumSize(64, 38)
        browse_button.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_button)

        # 状态始终保留可读文字，不依赖青色或琥珀色单独表达成功与警告。
        self.status_label = QLabel("尚未选择")
        self.status_label.setObjectName("statusMuted")
        self.status_label.setWordWrap(True)
        layout.addLayout(title_row)
        layout.addLayout(path_row)
        layout.addWidget(self.status_label)

    @property
    def path(self) -> Path | None:
        text = self.path_edit.text().strip()
        return Path(text) if text else None

    def set_path(self, path: str | Path) -> None:
        path_text = str(Path(path))
        self.path_edit.setText(path_text)
        self.path_edit.setToolTip(path_text)
        self.status_label.setText("已选择，等待分析")
        self.status_label.setObjectName("statusReady")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.path_selected.emit(str(path))

    def set_summary(self, summary: str, *, warning: bool = False) -> None:
        self.status_label.setText(summary)
        self.status_label.setObjectName("statusWarning" if warning else "statusReady")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", self.file_filter)
        if path:
            self.set_path(path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls() and event.mimeData().urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.set_path(urls[0].toLocalFile())
            event.acceptProposedAction()


# 左下连续脉冲轨迹把原来的空白转化为任务状态反馈，但不增加标题、说明或额外操作。
class ResponseField(QWidget):
    """用连续拟合脉冲轨迹表达输入、鼠标悬停和后台处理状态。"""

    # 轨迹始终沿用用户确认的蓝色主色；业务结果仍由状态栏文字提供完整语义。
    _TONE_COLORS = {
        "neutral": ACCENT,
        # 运行态保留基础蓝色，把更亮一级留给移动扫光；否则两者同色时，
        # 小尺寸透明画布上的动画只改变少量 alpha 像素，用户几乎看不见。
        "active": ACCENT,
        "success": ACCENT,
        "warning": ACCENT,
        "error": ACCENT,
    }

    # 初始化只保存轻量交互状态，波形路径按控件真实尺寸确定性生成。
    def __init__(self, parent: QWidget | None = None) -> None:
        # QWidget 作为透明绘制表面，不引入卡片、边框或文字层级。
        super().__init__(parent)
        # 对象名供视觉回归、辅助功能和后续主题扩展稳定定位。
        self.setObjectName("responseField")
        # 读屏用户能够知道该区域代表拟合脉冲与任务状态，而不依赖曲线颜色。
        self.setAccessibleName("拟合脉冲轨迹状态")
        # 持续接收鼠标位置，曲线只在光标附近提亮而不引入点击操作。
        self.setMouseTracking(True)
        # 不启用自动背景填充，让曲线直接叠加在父侧栏表面而不形成矩形色块。
        self.setAutoFillBackground(False)
        # 初始时尚未选择输入，轨迹使用最低亮度的中性状态。
        self._input_count = 0
        # tone 与主窗口状态语义保持一致，成功、警告和失败均可独立表达。
        self._tone: Literal["neutral", "active", "success", "warning", "error"] = "neutral"
        # work_phase 控制处理态高光从轨迹左端扫到右端的位置。
        self._work_phase = 0.0
        # hover_intensity 保存局部高光的当前淡入程度，0 为隐藏、1 为完全显示。
        self._hover_intensity = 0.0
        # hover_target 在鼠标进入时变为 1，离开后回到 0 并平滑淡出。
        self._hover_target = 0.0
        # hover_position 使用 0–1 归一化横坐标，避免窗口缩放后高光错位。
        self._hover_position = 0.50
        # hover_position_target 保存鼠标最新位置，当前高光会以轻微延迟追随。
        self._hover_position_target = 0.50
        # 离屏渲染默认静止；真实界面可通过环境变量关闭动态效果。
        self._motion_enabled = (
            QGuiApplication.platformName() != "offscreen" and not _prefers_reduced_motion()
        )
        # 30 FPS 兼顾局部高光跟随的连贯性和小画布的低刷新开销。
        self._timer = QTimer(self)
        # 粗粒度定时器允许系统合并唤醒，减少后台分析期间的无意义 CPU 占用。
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        # 每 33 ms 更新一次局部高光或运行扫光，不触发中央图表与主布局刷新。
        self._timer.setInterval(33)
        # 每次超时只推进高光插值和工作相位，不修改业务数据。
        self._timer.timeout.connect(self._advance)
        # 初始辅助描述明确输入数量和静止状态。
        self._update_accessibility()

    # 建议高度仅在有剩余空间时生效，640 px 最小窗口仍允许该区域收缩而不挤压输入卡。
    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        # 220×148 与当前左栏比例匹配，并保证脉冲峰和两侧基线完整可见。
        return QSize(220, 148)

    # 暴露只读状态给自动化和测试，不让外部代码直接操纵内部定时器。
    @property
    def animation_running(self) -> bool:
        # Qt 定时器活动即代表当前视觉仍在持续更新。
        return self._timer.isActive()

    # 公开当前基础亮度，兼容既有自动化并让不同语义状态保持稳定层级。
    @property
    def work_intensity(self) -> float:
        # active 与 success 保持清晰，等待或警告状态则降低装饰性轨迹权重。
        return 0.82 if self._tone == "active" else (0.72 if self._tone == "success" else 0.52)

    # 输入数量只接受 0–3，分别控制轨迹亮度而不改变界面文案。
    def set_input_count(self, count: int) -> None:
        # 钳位避免外部错误计数让绘制索引越界。
        self._input_count = max(0, min(3, int(count)))
        # 输入变化后同步辅助描述，保证非视觉用户获得等价信息。
        self._update_accessibility()
        # 立即重绘输入完成度对应的亮度，不等待任何动画帧。
        self.update()

    # 公开运动开关便于辅助设置、自动化和未来偏好页复用。
    def set_motion_enabled(self, enabled: bool) -> None:
        # 保存用户选择，运行状态下会据此决定是否启动定时器。
        self._motion_enabled = bool(enabled)
        # 关闭动态效果时把局部高光直接落到目标状态，不留下半完成动画。
        if not self._motion_enabled:
            # 高光淡入程度立即同步，减少动态效果仍保留清晰的静态鼠标反馈。
            self._hover_intensity = self._hover_target
            # 跟随位置同样立即同步，避免关闭动画后出现位置跳变。
            self._hover_position = self._hover_position_target
        # 统一经过状态同步逻辑，避免开关后留下无法停止的定时器。
        self._sync_animation()
        # 开关变化后立即重绘静态替代帧。
        self.update()

    # 主窗口只传递已有语义状态，脉冲轨迹不自行推断业务成功或失败。
    def set_tone(
        self,
        tone: Literal["neutral", "active", "success", "warning", "error"],
    ) -> None:
        # 拒绝未知字符串，防止无对应颜色的状态悄悄退回错误视觉。
        if tone not in self._TONE_COLORS:
            # 明确异常让调用方在开发阶段发现不一致的状态词。
            raise ValueError(f"不支持的脉冲轨迹状态：{tone}")
        # 首次进入运行态时从左端开始，让扫光具有明确且可预测的方向。
        if tone == "active" and self._tone != "active":
            # 相位归零只发生在任务启动，不会在每次状态刷新时打断扫光节奏。
            self._work_phase = 0.0
        # 保存状态后更新颜色、辅助描述和运动策略。
        self._tone = tone
        # 只有 active 且允许运动时才启动定时器。
        self._sync_animation()
        # 状态文字同步到辅助树，颜色不作为唯一信息来源。
        self._update_accessibility()
        # 静态状态也需要立即刷新对应的语义色。
        self.update()

    # 包络只定义用户确认的视觉轮廓：窄主峰、一次下探和无波纹的低位长拖尾。
    @staticmethod
    def _trace_envelope(x_value: float) -> float:
        # 左侧逻辑门在 18% 附近快速抬升，形成比旧粒子方案更窄的主峰。
        rising_edge = 1.0 / (1.0 + np.exp(-(x_value - 0.18) / 0.018))
        # 右侧逻辑门在 30% 附近快速回落，峰宽保持在画布约 12%。
        falling_edge = 1.0 / (1.0 + np.exp((x_value - 0.30) / 0.018))
        # 两个平滑门相乘形成无尖角的窄平台主峰。
        main_pulse = rising_edge * falling_edge
        # 36% 附近的一次负向高斯下探复现用户参考图中的主峰后回落。
        undershoot = -0.12 * np.exp(-(((x_value - 0.36) / 0.047) ** 2))
        # 拖尾门在下探结束后缓慢开启，不产生突兀折点。
        tail_gate = 1.0 / (1.0 + np.exp(-(x_value - 0.45) / 0.050))
        # 单调指数衰减形成低位长拖尾，不叠加任何正弦波动。
        tail = 0.052 * tail_gate * np.exp(-max(0.0, x_value - 0.62) / 0.75)
        # 最终包络允许短暂负值以绘制下探，但上限限制主峰不越出画布。
        return min(1.0, float(main_pulse + undershoot + tail))

    # 每一帧同时推进鼠标高光跟随和运行态扫光相位。
    def _advance(self) -> None:
        # 180–220 ms 左右的指数缓出让高光淡入淡出自然且不拖沓。
        self._hover_intensity += (self._hover_target - self._hover_intensity) * 0.24
        # 位置跟随略慢于透明度，形成轻微磁性延迟而不扭曲脉冲本身。
        self._hover_position += (self._hover_position_target - self._hover_position) * 0.20
        # 接近目标时直接吸附，避免定时器因无限小尾差永久运行。
        if abs(self._hover_target - self._hover_intensity) < 0.006:
            # 精确落到目标透明度后，静止鼠标可以可靠停止定时器。
            self._hover_intensity = self._hover_target
        # 位置同样在亚百分比误差内吸附，避免隐藏高光仍触发周期更新。
        if abs(self._hover_position_target - self._hover_position) < 0.004:
            # 精确同步当前与目标横坐标。
            self._hover_position = self._hover_position_target
        # 只有后台任务 active 时才推进从左到右的扫光相位。
        if self._tone == "active":
            # 每帧 0.009 对应约 3.7 秒一轮，状态可感知但不会像加载器一样急促。
            self._work_phase = (self._work_phase + 0.009) % 1.0
        # 仅请求本控件重绘，后台 DSP 和中央图表不受影响。
        self.update()
        # 高光已经稳定且任务不再运行时，停止局部动画定时器。
        self._sync_animation()

    # 将当前语义状态转换为是否需要持续刷新。
    def _sync_animation(self) -> None:
        # 未完成的淡入淡出或横向跟随任一存在时才需要持续刷新。
        has_hover_motion = (
            abs(self._hover_target - self._hover_intensity) >= 0.006
            or abs(self._hover_position_target - self._hover_position) >= 0.004
        )
        # 减少动态效果关闭所有曲线运动，底部状态栏仍保留文字和进度条反馈。
        should_run = self._motion_enabled and (self._tone == "active" or has_hover_motion)
        # 需要运动但定时器尚未启动时才调用 start，避免重复重置相位。
        if should_run and not self._timer.isActive():
            # 从当前相位继续流动，切换设置时不会出现突兀跳变。
            self._timer.start()
        # 非运行状态或关闭运动时立即停止周期唤醒。
        elif not should_run and self._timer.isActive():
            # 停止后保留当前静态帧，内容不会突然消失。
            self._timer.stop()

    # 鼠标进入区域后开始平滑显示局部高光，事件本身不改变任何业务状态。
    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802 - Qt API
        # 鼠标进入时把局部高光目标提升到最大值。
        self._hover_target = 1.0
        # 运动允许时通过定时器淡入，否则直接显示静态高光。
        if self._motion_enabled:
            # 启动或保持局部定时器完成淡入。
            self._sync_animation()
        # 减少动态效果下直接同步透明度，不产生过渡。
        else:
            # 静态高光仍提供等价鼠标反馈。
            self._hover_intensity = 1.0
            # 立即请求一次重绘。
            self.update()
        # 交还 Qt 默认进入处理，维持标准悬停语义。
        super().enterEvent(event)

    # 鼠标划过时更新局部高光中心，波形几何保持完全不变。
    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        # 把逻辑像素横坐标归一化到 0–1，窗口缩放后仍可稳定跟随。
        self._hover_position_target = max(
            0.0, min(1.0, event.position().x() / max(1.0, float(self.width())))
        )
        # 鼠标仍在区域内时持续维持最大高光目标。
        self._hover_target = 1.0
        # 未请求减少动态效果时通过定时器形成轻微跟随延迟。
        if self._motion_enabled:
            # 确保静态业务状态也能因悬停启动局部定时器。
            self._sync_animation()
        # 减少动态效果下立即跳到鼠标位置，不产生跟随动画。
        else:
            # 当前高光位置直接同步到目标坐标。
            self._hover_position = self._hover_position_target
            # 高光立即显示。
            self._hover_intensity = 1.0
        # 每次真实鼠标移动至少重绘一帧，已收敛的高光也能跟随新位置。
        self.update()
        # 保留 QWidget 的标准鼠标移动分发行为。
        super().mouseMoveEvent(event)

    # 鼠标离开后局部高光淡出，基础拟合脉冲轨迹保持不变。
    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt API
        # 目标归零后，允许运动时继续运行到透明度完全回落。
        self._hover_target = 0.0
        # 正常动效通过统一状态逻辑决定是否继续定时器。
        if self._motion_enabled:
            # 当前仍有可见高光时维持定时器完成淡出。
            self._sync_animation()
        # 减少动态效果下直接隐藏高光。
        else:
            # 静态替代不保留半透明残影。
            self._hover_intensity = 0.0
            # 立即重绘基础轨迹。
            self.update()
        # 调用父类以保持 Qt 标准离开事件处理。
        super().leaveEvent(event)

    # 辅助描述提供脉冲轨迹无法用颜色直接表达的输入数量与运行状态。
    def _update_accessibility(self) -> None:
        # 中文状态与主窗口已有状态词一致，避免辅助树出现另一套术语。
        tone_text = {
            "neutral": "等待输入",
            "active": "正在处理",
            "success": "结果有效",
            "warning": "结果需更新",
            "error": "处理失败",
        }[self._tone]
        # 描述同时包含输入完成度和任务状态，非视觉用户无需读取颜色或动画。
        self.setAccessibleDescription(f"输入 {self._input_count}/3 · {tone_text}")

    # 在既有轨迹上绘制局部高光，不修改任何采样点或波形几何。
    @staticmethod
    def _draw_highlight(
        painter: QPainter,
        points: list[QPointF],
        center: float,
        strength: float,
        width: float,
    ) -> None:
        # 相邻采样点逐段绘制，允许高光沿真实曲线弯折而不是覆盖一个矩形渐变。
        for point_index in range(len(points) - 1):
            # 每段中心横坐标与生成轨迹时的 3%–97% 范围一致。
            segment_position = 0.03 + 0.94 * (point_index + 0.5) / 120.0
            # 横向距离决定局部高光衰减，波形纵向位置不会参与形变。
            distance = abs(segment_position - center)
            # 超出局部半径的线段保持基础亮度，避免整条轨迹同时闪烁。
            if distance > width:
                # 继续检查下一段轨迹。
                continue
            # 平方缓出让高光中心清晰、边缘柔和且没有硬截断感。
            local_strength = (1.0 - distance / width) ** 2
            # 高光透明度同时受淡入程度与局部距离控制。
            highlight_alpha = int(245 * strength * local_strength)
            # 高光沿用原蓝色的更亮层级，不引入新的装饰色。
            highlight_color = QColor(ACCENT_BRIGHT)
            # 设置逐段透明度形成连续光斑。
            highlight_color.setAlpha(max(0, min(235, highlight_alpha)))
            # 3 px 左右的圆头高光只在局部出现，静态脉冲仍保持 1.45 px 细线。
            painter.setPen(
                QPen(
                    highlight_color,
                    2.6 + 0.6 * strength,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            # 绘制当前高光线段。
            painter.drawLine(points[point_index], points[point_index + 1])

    # 绘制函数只在本地小画布上渲染矢量曲线，避免大图缓存和外部动画依赖。
    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        # 尺寸过小时直接跳过，最小窗口不会为了装饰强行占据布局空间。
        if self.width() < 24 or self.height() < 24:
            # 没有足够像素时保持完全透明是最稳妥的退化方式。
            return
        # QPainter 在当前 QWidget 上进行抗锯齿矢量绘制。
        painter = QPainter(self)
        # 曲线启用抗锯齿，避免高 DPI 屏幕出现阶梯边缘。
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # 左右各留 9 px，保证圆头线与光晕不会被控件边缘切掉。
        draw_width = max(1.0, float(self.width() - 18))
        # 上下共留 14 px，为主峰光晕和负向下探提供安全空间。
        draw_height = max(1.0, float(self.height() - 14))
        # 横向绘制起点与左右边距一致。
        left = 9.0
        # 纵向绘制起点保留 7 px 光晕空间。
        top = 7.0
        # 88% 高度作为低位基线，使拖尾落在用户参考图要求的下部区域。
        baseline = top + 0.88 * draw_height

        # points 供基础轨迹、鼠标高光和运行扫光共用，避免三套波形位置漂移。
        points: list[QPointF] = []
        # QPainterPath 绘制连续抗锯齿曲线而不是离散粒子。
        trace_path = QPainterPath()
        # 121 个横向样本在 220 px 侧栏中足以形成平滑轮廓且绘制开销很低。
        for point_index in range(121):
            # 归一化横坐标覆盖 3%–97%，完整保留左右基线和长拖尾。
            x_normalized = 0.03 + 0.94 * point_index / 120.0
            # 当前控件宽度将归一化横坐标转换为逻辑像素。
            x_position = left + x_normalized * draw_width
            # 0.66 峰高在窄峰清晰度与底部下探空间之间取得平衡。
            y_normalized = 0.88 - 0.66 * self._trace_envelope(x_normalized)
            # 归一化纵坐标转换为逻辑像素，较小值位于画布上方。
            y_position = top + y_normalized * draw_height
            # QPointF 保留高 DPI 下的亚像素精度。
            point = QPointF(x_position, y_position)
            # 保存采样点供后续局部高光沿同一路径绘制。
            points.append(point)
            # 首个样本只移动路径起点，不连接到默认原点。
            if point_index == 0:
                # 设置连续曲线的起点。
                trace_path.moveTo(point)
            # 后续样本依次连接为连续轨迹。
            else:
                # 直线段在高密度采样和抗锯齿下形成平滑曲线。
                trace_path.lineTo(point)

        # 面积填充从真实曲线闭合到低位基线，提供很轻的能量层次。
        fill_path = QPainterPath(trace_path)
        # 右端闭合到基线下方 3 px，避免尾端留下发亮竖边。
        fill_path.lineTo(points[-1].x(), baseline + 3.0)
        # 左端同样闭合到基线下方。
        fill_path.lineTo(points[0].x(), baseline + 3.0)
        # 闭合路径后才允许渐变填充。
        fill_path.closeSubpath()

        # 输入越完整，基础轨迹越清晰；语义状态仍保留稳定上限。
        visibility = min(0.90, self.work_intensity + self._input_count * 0.055)
        # 当前语义色只改变蓝色明度，不改变用户确认的色相体系。
        tone_color = QColor(self._TONE_COLORS[self._tone])
        # 垂直渐变在峰顶最明显，向低位基线完全透明。
        fill_gradient = QLinearGradient(0.0, top + draw_height * 0.18, 0.0, baseline + 4.0)
        # 峰区只使用低透明蓝色，避免形成新的色块卡片。
        fill_peak_color = QColor(tone_color)
        # 25×visibility 控制能量晕染强度。
        fill_peak_color.setAlpha(int(25 * visibility))
        # 顶部渐变起点应用最明显但仍克制的填充。
        fill_gradient.setColorAt(0.0, fill_peak_color)
        # 中段进一步降低透明度。
        fill_middle_color = QColor(tone_color)
        # 中段仅保留约 10 alpha 的能量感。
        fill_middle_color.setAlpha(int(10 * visibility))
        # 55% 位置应用中段颜色。
        fill_gradient.setColorAt(0.55, fill_middle_color)
        # 基线位置必须完全透明，避免拖尾下方出现横向色带。
        fill_end_color = QColor(tone_color)
        # alpha=0 让填充自然消失。
        fill_end_color.setAlpha(0)
        # 渐变末端应用透明色。
        fill_gradient.setColorAt(1.0, fill_end_color)
        # 绘制轻量面积填充。
        painter.fillPath(fill_path, fill_gradient)

        # 第一遍使用低透明宽线形成收敛光晕，不绘制粒子或离散光球。
        halo_color = QColor(tone_color)
        # 光晕透明度随输入和状态可见度变化。
        halo_color.setAlpha(int(22 * visibility))
        # 5.5 px 光晕仍限制在轨迹附近。
        painter.setPen(
            QPen(
                halo_color,
                5.5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        # 绘制基础光晕路径。
        painter.drawPath(trace_path)

        # 第二遍使用横向明度渐变突出主峰，同时让低位长拖尾保持克制。
        line_gradient = QLinearGradient(left, 0.0, left + draw_width, 0.0)

        # 小工具函数创建同色相、不同透明度的渐变节点。
        def line_color(alpha: int) -> QColor:
            # 从当前语义色复制，避免手写另一套 RGB。
            color = QColor(tone_color)
            # visibility 同步控制所有基础线段的视觉权重。
            color.setAlpha(int(alpha * visibility))
            # 返回可交给 QLinearGradient 的独立颜色对象。
            return color

        # 左侧短基线保持低亮度。
        line_gradient.setColorAt(0.0, line_color(58))
        # 主峰上升沿逐步提亮。
        line_gradient.setColorAt(0.17, line_color(165))
        # 主峰顶部获得最高亮度。
        line_gradient.setColorAt(0.27, line_color(220))
        # 下探后迅速回到较低视觉权重。
        line_gradient.setColorAt(0.42, line_color(118))
        # 长拖尾仍保持可辨认但不抢占中心图表注意力。
        line_gradient.setColorAt(1.0, line_color(64))
        # 1.45 px 圆头核心线形成干净、连续的仪器轨迹。
        painter.setPen(
            QPen(
                line_gradient,
                1.45,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        # 绘制核心连续曲线。
        painter.drawPath(trace_path)

        # 鼠标高光只在当前淡入程度大于零时绘制，静态帧没有额外开销。
        if self._hover_intensity > 0.0:
            # 16% 画布宽度形成柔和局部提亮，不会让整条轨迹一起闪烁。
            self._draw_highlight(
                painter,
                points,
                self._hover_position,
                self._hover_intensity * 0.82,
                0.16,
            )
        # active 状态额外绘制从左到右移动的工作高光。
        if self._tone == "active":
            # 3%–97% 映射保证高光完整扫过可见轨迹而不进入空白边距。
            scan_position = 0.03 + 0.94 * self._work_phase
            # 运行高光略窄于鼠标高光，强调方向和进度感。
            self._draw_highlight(
                painter,
                points,
                scan_position,
                0.96,
                0.13,
            )


class AnalysisThread(QThread):
    """在后台解析文件并运行频响分析，防止大记录冻结界面。"""

    succeeded = Signal(object, int)
    failed = Signal(str, int)
    cancelled = Signal(int)

    def __init__(self, request: AnalysisRequest) -> None:
        super().__init__()
        self.request = request

    def run(self) -> None:
        try:
            self._raise_if_interrupted()
            reference = load_csv_timeseries(
                self.request.reference_path,
                time_unit="s",
                time_column=0,
                value_columns=(1,),
                expected_columns=2,
                cancelled=self.isInterruptionRequested,
            )
            self._raise_if_interrupted()
            dut = load_csv_timeseries(
                self.request.dut_path,
                time_unit="s",
                time_column=0,
                value_columns=(1,),
                expected_columns=2,
                cancelled=self.isInterruptionRequested,
            )
            self._raise_if_interrupted()
            settings = self.request.settings
            prepared_spectra = prepare_response_spectra(
                reference,
                dut,
                settings,
                cancelled=self.isInterruptionRequested,
            )
            self._raise_if_interrupted()
            target = None
            automatic_settings_ready = False
            if self.request.action == "compensate":
                if self.request.target_path is None:
                    raise ValueError("数据补偿需要选择待补偿信号")
                if self.request.target_path.suffix.lower() == ".bin":

                    def preflight_bin_payload(layout: BinPayloadLayout) -> None:
                        """Reject unsafe full-DSP geometry before BIN payload access."""

                        nonlocal automatic_settings_ready, settings
                        self._raise_if_interrupted()
                        if self.request.auto_frequency_bands and not automatic_settings_ready:
                            settings = suggest_frequency_settings(
                                reference,
                                dut,
                                settings,
                                maximum_frequency_hz=0.5 * layout.sample_rate_hz,
                                suggest_phase_fit_band=(self.request.auto_phase_fit_band),
                                prepared_spectra=prepared_spectra,
                                cancelled=self.isInterruptionRequested,
                            )
                            automatic_settings_ready = True
                        preflight_compensation_shape(
                            target_samples=layout.samples,
                            target_channels=layout.channels,
                            sample_rate_hz=layout.sample_rate_hz,
                            reference_samples=reference.samples,
                            dut_samples=dut.samples,
                            settings=settings,
                            anticipated_input_resident_bytes=(layout.estimated_resident_bytes),
                        )
                        self._raise_if_interrupted()

                    target = load_bin_timeseries(
                        self.request.target_path,
                        payload_preflight=preflight_bin_payload,
                        cancelled=self.isInterruptionRequested,
                    )
                else:
                    target = load_csv_timeseries(
                        self.request.target_path,
                        time_unit="s",
                        time_column=0,
                        value_columns=(1,),
                        expected_columns=2,
                        cancelled=self.isInterruptionRequested,
                    )
                self._raise_if_interrupted()
            if self.request.auto_frequency_bands and not automatic_settings_ready:
                settings = suggest_frequency_settings(
                    reference,
                    dut,
                    settings,
                    maximum_frequency_hz=(target.nyquist_hz if target is not None else None),
                    suggest_phase_fit_band=self.request.auto_phase_fit_band,
                    prepared_spectra=prepared_spectra,
                    cancelled=self.isInterruptionRequested,
                )
                automatic_settings_ready = True
                self._raise_if_interrupted()
            if self.request.action == "compare":
                result = compare_pulses(
                    reference,
                    dut,
                    settings,
                    prepared_spectra=prepared_spectra,
                    cancelled=self.isInterruptionRequested,
                )
            else:
                assert target is not None
                result = run_compensation(
                    reference,
                    dut,
                    target,
                    settings,
                    prepared_spectra=prepared_spectra,
                    cancelled=self.isInterruptionRequested,
                )
            self._raise_if_interrupted()
            self.succeeded.emit(result, self.request.version)
        except OperationCancelledError:
            self.cancelled.emit(self.request.version)
        except Exception as exc:  # GUI boundary: convert full failure to actionable text.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail, self.request.version)

    def _raise_if_interrupted(self) -> None:
        """Stop at a safe stage boundary when the window requests shutdown."""

        raise_if_cancelled(
            self.isInterruptionRequested,
            message="分析已取消",
        )


class ExportThread(QThread):
    """Write a rollback-capable bundle without blocking the Qt event loop."""

    succeeded = Signal(object, int)
    failed = Signal(str, int)
    source_invalid = Signal(str, int)
    rollback_incomplete = Signal(str, int)
    cleanup_incomplete = Signal(str, int)
    cancelled = Signal(int)

    def __init__(
        self,
        run: CompensationRun,
        output_path: Path,
        version: int,
        destination_fingerprints: tuple[
            DestinationFingerprint,
            DestinationFingerprint,
            DestinationFingerprint,
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self.run_data = run
        self.output_path = output_path
        self.version = version
        self.destination_fingerprints = destination_fingerprints

    def run(self) -> None:
        caught: list[warnings.WarningMessage] = []

        def cleanup_detail() -> str:
            return "\n".join(
                str(item.message)
                for item in caught
                if issubclass(item.category, BundleCleanupWarning)
            )

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", BundleCleanupWarning)
                paths = export_run_bundle(
                    self.run_data,
                    self.output_path,
                    allow_overwrite=True,
                    cancelled=self.isInterruptionRequested,
                    expected_destination_fingerprints=self.destination_fingerprints,
                )
            cleanup_warning = cleanup_detail()
            # export_run_bundle 返回意味着不可取消的最终提交已完整结束；此后即使
            # 关窗标志刚到，也必须报告成功，不能把已落盘批次误报成取消。
            self.succeeded.emit(
                ExportOutcome(paths=paths, cleanup_warning=cleanup_warning),
                self.version,
            )
        except OperationCancelledError as exc:
            cleanup_warning = cleanup_detail()
            if cleanup_warning:
                self.cleanup_incomplete.emit(
                    f"{exc}\n\n清理残留：\n{cleanup_warning}",
                    self.version,
                )
            else:
                self.cancelled.emit(self.version)
        except SourceVerificationError as exc:
            cleanup_warning = cleanup_detail()
            if cleanup_warning:
                self.cleanup_incomplete.emit(
                    f"{exc}\n\n清理残留：\n{cleanup_warning}",
                    self.version,
                )
            else:
                self.source_invalid.emit(str(exc), self.version)
        except BundleRollbackError as exc:
            cleanup_warning = cleanup_detail()
            detail = str(exc)
            if cleanup_warning:
                detail += f"\n\n清理残留：\n{cleanup_warning}"
            self.rollback_incomplete.emit(detail, self.version)
        except Exception as exc:  # GUI boundary: keep worker failures actionable.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            cleanup_warning = cleanup_detail()
            if cleanup_warning:
                self.cleanup_incomplete.emit(
                    f"{detail}\n\n清理残留：\n{cleanup_warning}",
                    self.version,
                )
            else:
                self.failed.emit(detail, self.version)


def _plot_widget() -> pg.PlotWidget:
    # 图表使用比面板更深的画布，以建立“控制层 / 数据层”的清晰表面层级。
    plot = pg.PlotWidget(background=BACKGROUND)
    plot.setMinimumHeight(220)
    plot.showGrid(x=True, y=True, alpha=0.11)
    plot.setMouseEnabled(x=True, y=True)
    plot.setClipToView(True)
    plot.setDownsampling(auto=True, mode="peak")
    # 初始空图不创建图例；第一条命名曲线加入时再按需创建，避免出现无内容的小方框。
    for axis_name in ("left", "bottom"):
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(BORDER_STRONG))
        axis.setTextPen(pg.mkPen(TEXT_MUTED))
        axis.enableAutoSIPrefix(False)
    return plot


def _plot_page(count: int) -> tuple[QWidget, list[pg.PlotWidget]]:
    page = QWidget()
    layout = QVBoxLayout(page)
    # 页签下方单独保留 12 px 呼吸空间，避免图框紧贴选中下划线；其余方向维持紧凑 8 px。
    layout.setContentsMargins(8, 12, 8, 8)
    layout.setSpacing(8)
    plots = [_plot_widget() for _ in range(count)]
    for plot in plots:
        layout.addWidget(plot, 1)
    return page, plots


class ResponseLabWindow(QMainWindow):
    """ResponseLab 三栏主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        # 较低的最小尺寸允许三栏在 13 英寸屏幕内完整显示，详细设置仍可纵向滚动。
        self.setWindowTitle("ResponseLab · 频响分析与补偿")
        self.setMinimumSize(960, 640)
        self._result: PulseComparison | CompensationRun | None = None
        self._run: CompensationRun | None = None
        self._worker: (
            AnalysisThread
            | ExportThread
            | InfluenceAnalysisThread
            | InfluenceSelectionThread
            | None
        ) = None
        self._active_action: (
            Literal[
                "compare",
                "compensate",
                "export",
                "influence",
                "influence_candidate",
            ]
            | None
        ) = None
        # 影响页使用独立版本，改变其参数不会使现有数据补偿导出失效。
        self._influence_version = 0
        # 完整影响扫描结果供候选点选复用，不混入原补偿 self._result。
        self._influence_run: InfluenceRun | None = None
        # 当前候选行用于避免 render 后默认选中又触发重复后台计算。
        self._influence_selected_row = -1
        self._parameter_version = 0
        self._result_version = -1
        self._close_when_finished = False
        self._last_frequency_unit = "GHz"
        self._phase_band_is_manual = False
        self._phase_band_initialized = False
        self._building = True
        self._build_ui()
        # 首次窗口尺寸依据当前屏幕可用区域计算，避免固定 1440 像素把右栏推到屏幕外。
        self._fit_initial_window_to_screen()
        self._connect_stale_signals()
        self._building = False
        self.statusBar().setAccessibleName("运行状态")
        self.statusBar().showMessage("就绪 · 比较只需两份拟合脉冲；数据补偿时再选择第三份信号")

    @staticmethod
    def _preferred_initial_size(available_size: QSize) -> QSize:
        """在屏幕边缘预留 16 px 安全间距，并限制超宽屏上的默认窗口尺寸。"""

        # 宽度在 960–1400 px 内自适应，既容纳三栏又避免首次启动超出可用桌面。
        width = max(960, min(1400, available_size.width() - 32))
        # 高度在 640–860 px 内自适应，给 macOS 菜单栏、Dock 和窗口边框留出空间。
        height = max(640, min(860, available_size.height() - 32))
        # 返回 Qt 尺寸对象，供实际窗口和纯布局回归测试共同使用。
        return QSize(width, height)

    def _fit_initial_window_to_screen(self) -> None:
        """按窗口所在屏幕的可用区域设置并居中首次启动尺寸。"""

        # 优先使用窗口当前屏幕；构造早期没有关联屏幕时退回主屏幕。
        screen = self.screen() or QGuiApplication.primaryScreen()
        # 极少数无显示环境没有屏幕对象，此时使用保守尺寸保证窗口仍可构造。
        if screen is None:
            self.resize(1280, 800)
            return
        # availableGeometry 已排除菜单栏和 Dock，比物理屏幕尺寸更符合可见窗口范围。
        available = screen.availableGeometry()
        # 计算当前屏幕对应的自适应初始尺寸。
        target_size = self._preferred_initial_size(available.size())
        # 应用尺寸后再计算居中位置，确保左右两栏都处于可见区域。
        self.resize(target_size)
        # 水平方向在可用桌面内居中，并保留几何原点以支持副屏坐标。
        x_position = available.x() + max(0, (available.width() - self.width()) // 2)
        # 垂直方向同样使用可用区域居中，不覆盖系统菜单栏。
        y_position = available.y() + max(0, (available.height() - self.height()) // 2)
        # 移动只影响首次打开位置，用户之后仍可自由调整窗口和分栏宽度。
        self.move(x_position, y_position)

    def _build_ui(self) -> None:
        # 全局样式集中应用，避免单个控件出现与设计系统不一致的临时颜色。
        self.setStyleSheet(self._stylesheet())
        root = QWidget()
        # 根容器使用独立对象名承载近黑背景，保证面板间隙不会露出系统默认底色。
        root.setObjectName("workspaceRoot")
        root_layout = QVBoxLayout(root)
        # 四周统一留 6 px，形成用户确认的紧凑悬浮轮廓而不浪费工程画布。
        root_layout.setContentsMargins(6, 6, 6, 6)
        # 顶栏与三栏工作区使用同一 6 px 间隔，保持外部空间节奏一致。
        root_layout.setSpacing(6)
        root_layout.addWidget(self._build_header())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("workspaceSplitter")
        splitter.setChildrenCollapsible(False)
        # 分隔手柄同时充当两栏之间的 6 px 暗色间隙，并继续支持用户拖动调整宽度。
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_visual_workspace())
        splitter.addWidget(self._build_inspector())
        splitter.setSizes([250, 690, 340])
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.statusBar().setSizeGripEnabled(True)
        # 状态栏右侧使用不确定进度条：持续滑动代表后台线程仍在处理，不伪造无法计算的百分比。
        self.progress = QProgressBar()
        self.progress.setObjectName("statusProgress")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(168)
        self.progress.setAccessibleName("后台处理进度")
        self.progress.setAccessibleDescription("滑块持续移动表示比较或补偿任务仍在执行")
        self.statusBar().addPermanentWidget(self.progress)
        self.progress.hide()

    def _build_header(self) -> QWidget:
        # 顶栏只承载品牌、处理环境和全局状态，避免与图表操作抢占同一层级。
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 9, 20, 9)
        layout.setSpacing(12)

        brand_mark = QLabel("RL")
        brand_mark.setObjectName("brandMark")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(38, 38)
        brand_mark.setAccessibleName("ResponseLab 标识")

        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        title = QLabel("ResponseLab")
        title.setObjectName("appTitle")
        subtitle = QLabel("频响分析与补偿")
        subtitle.setObjectName("helperText")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)

        layout.addWidget(brand_mark)
        layout.addLayout(title_column)
        layout.addStretch(1)

        # 状态对象保留给自动化与辅助功能，但不再占据顶栏；可见反馈统一放在底部状态栏。
        self.header_state = QLabel("等待输入", header)
        self.header_state.setObjectName("statePill")
        self.header_state.setProperty("tone", "neutral")
        self.header_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_state.setAccessibleName("当前任务状态")
        self.header_state.hide()
        return header

    def _build_left_panel(self) -> QWidget:
        # 左栏固定为输入流程，用户从上到下即可判断比较和补偿分别需要哪些文件。
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(236)
        panel.setMaximumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 18)
        layout.setSpacing(10)
        section = QLabel("数据输入")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)
        self.reference_card = FileCard(
            "01",
            "参考拟合脉冲",
            "CSV (*.csv);;所有文件 (*)",
        )
        self.dut_card = FileCard(
            "02",
            "待补偿拟合脉冲",
            "CSV (*.csv);;所有文件 (*)",
        )
        self.target_card = FileCard(
            "03",
            "待补偿信号",
            "信号 (*.csv *.bin);;所有文件 (*)",
        )
        self.target_card.path_selected.connect(self._target_path_changed)
        layout.addWidget(self.reference_card)
        layout.addWidget(self.dut_card)
        layout.addWidget(self.target_card)
        # 左下脉冲轨迹使用输入卡之后原有的弹性空白，不改变三张卡片的尺寸和间距。
        self.response_field = ResponseField(panel)
        # 三张卡片共用一个只读计数刷新入口，让轨迹亮度与实际已选文件数量保持一致。
        for card in (self.reference_card, self.dut_card, self.target_card):
            # 文件选择完成后即时增强轨迹亮度，不等待后台分析开始。
            card.path_selected.connect(self._update_response_field_inputs)
        # stretch=1 让脉冲轨迹吸收全部剩余高度；窗口较矮时它可自然收缩到零。
        layout.addWidget(self.response_field, 1)
        return panel

    def _build_visual_workspace(self) -> QWidget:
        # 中央区域让页签直接贴近顶部，并把绘图工具收进同一行的右侧角落。
        workspace = QFrame()
        workspace.setObjectName("workspace")
        # 380 px 下限让 960 px 最小窗口仍能排开三栏；更宽窗口继续由分隔器把额外空间交给画布。
        workspace.setMinimumWidth(380)
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(14, 8, 14, 12)
        layout.setSpacing(8)
        # 旧自动化仍可读取分析摘要，但该兼容对象不加入布局，因此不会再显示常驻文案。
        self.metric_label = QLabel("", workspace)
        # 辅助名称说明隐藏对象的用途，避免它在自动化树中成为无语义节点。
        self.metric_label.setAccessibleName("分析摘要兼容状态")
        # 明确隐藏兼容对象，实际读屏摘要由页签控件的辅助描述承担。
        self.metric_label.hide()

        # 采用科学绘图软件熟悉的框选、四向平移和主页图形，减少工具栏文字占用。
        self.zoom_button = QPushButton()
        self.pan_button = QPushButton()
        self.reset_button = QPushButton()
        # 三枚 SVG 图标与 Matplotlib 工具栏的视觉语义一致，但保持本项目自己的线条风格。
        self.zoom_button.setIcon(QIcon(str(ICON_DIRECTORY / "zoom-area.svg")))
        self.pan_button.setIcon(QIcon(str(ICON_DIRECTORY / "pan.svg")))
        self.reset_button.setIcon(QIcon(str(ICON_DIRECTORY / "home.svg")))
        for button in (self.zoom_button, self.pan_button, self.reset_button):
            button.setIconSize(QSize(20, 20))
            button.setFixedWidth(40)
        self.plot_mode_group = QButtonGroup(self)
        self.plot_mode_group.setExclusive(True)
        for button in (self.zoom_button, self.pan_button):
            button.setObjectName("toolButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            self.plot_mode_group.addButton(button)
        self.pan_button.setChecked(True)
        self.reset_button.setObjectName("toolButton")
        self.reset_button.setMinimumHeight(36)
        self.zoom_button.setToolTip("左键拖出矩形区域进行放大")
        self.pan_button.setToolTip("按住左键拖动画布；滚轮可继续缩放")
        self.reset_button.setToolTip("恢复当前数据的推荐显示范围")
        self.zoom_button.setAccessibleName("矩形框选放大图表")
        self.pan_button.setAccessibleName("平移图表")
        self.reset_button.setAccessibleName("恢复图表推荐范围")
        self.zoom_button.clicked.connect(lambda: self._set_plot_mouse_mode("zoom"))
        self.pan_button.clicked.connect(lambda: self._set_plot_mouse_mode("pan"))
        self.reset_button.clicked.connect(self._reset_plots)

        tool_group = QFrame()
        tool_group.setObjectName("segmentedControl")
        tool_layout = QHBoxLayout(tool_group)
        tool_layout.setContentsMargins(4, 4, 4, 4)
        tool_layout.setSpacing(3)
        tool_layout.addWidget(self.zoom_button)
        tool_layout.addWidget(self.pan_button)
        tool_layout.addWidget(self.reset_button)

        self.band_legend_label = QLabel(
            "蓝色阴影：分析/候选补偿频带　橙色虚线：线性相位拟合频带边界"
        )
        self.band_legend_label.setObjectName("helperText")
        self.band_legend_label.setWordWrap(True)
        # 频带说明保留在对象中供测试与辅助读取，不再作为常驻文案占用画布空间。
        self.band_legend_label.hide()
        self.visual_tabs = QTabWidget()
        self.visual_tabs.setDocumentMode(True)
        # 禁用系统样式绘制的亮色页签基线，避免未被页签覆盖的窄缝在深色主题中闪白。
        self.visual_tabs.tabBar().setDrawBase(False)
        # 页签是中央区域的主要导航，辅助名称保留给键盘和读屏用户。
        self.visual_tabs.setAccessibleName("分析结果页签")
        # 初始说明不占据可见空间，分析完成后会更新为当前频带与相位处理摘要。
        self.visual_tabs.setAccessibleDescription("尚未分析")
        # 绘图工具作为页签栏右上角控件，与页签共享第一行并释放纵向空间。
        self.visual_tabs.setCornerWidget(tool_group, Qt.Corner.TopRightCorner)
        pulse_page, self.pulse_plots = _plot_page(1)
        response_page, self.response_plots = _plot_page(2)
        difference_page, self.difference_plots = _plot_page(2)
        compensator_page, self.compensator_plots = _plot_page(2)
        output_page, self.output_plots = _plot_page(2)
        # 影响频段页独立管理自身曲线、眼图和 Vpp 波形，不加入旧页面清空列表。
        self.influence_page = InfluenceBandPage()
        self.visual_tabs.addTab(pulse_page, "拟合脉冲")
        self.visual_tabs.addTab(response_page, "频率响应")
        self.visual_tabs.addTab(difference_page, "频响差异比较")
        self.visual_tabs.addTab(compensator_page, "频响补偿")
        self.visual_tabs.addTab(output_page, "输出预览")
        # 新功能追加在末尾，保持已有五个页签的顺序和索引完全不变。
        self.influence_tab_index = self.visual_tabs.addTab(
            self.influence_page,
            "影响频段",
        )
        # 页面按钮只发轻量请求，主窗口负责唯一后台 worker 生命周期。
        self.influence_page.analysis_requested.connect(self._start_influence_analysis)
        # 页面专属参数变化只让影响结果过期。
        self.influence_page.request_changed.connect(self._mark_influence_stale)
        # 候选列表行号映射到扫描结果中的不可变 BandAttribution。
        self.influence_page.candidate_selected.connect(self._start_influence_selection)
        self._set_plot_mouse_mode("pan")
        layout.addWidget(self.visual_tabs, 1)
        self.result_warning = QLabel()
        self.result_warning.setObjectName("warningNote")
        self.result_warning.setWordWrap(True)
        self.result_warning.hide()
        layout.addWidget(self.result_warning)
        return workspace

    def _build_inspector(self) -> QWidget:
        # 右栏只呈现会改变分析结果的设置与动作，并通过滚动保持底部主操作常驻。
        panel = QFrame()
        panel.setObjectName("inspectorPanel")
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(400)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 16, 14, 20)
        layout.setSpacing(10)
        title = QLabel("分析设置")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addSpacing(4)

        compensation_group = QGroupBox("补偿设置")
        # 补偿参数同样使用上下表单，长中文标签和自动建议文字不会再被右边缘裁切。
        compensation_form = QFormLayout(compensation_group)
        # 所有行固定换行，保证在窗口缩放或高 DPI 字体下仍具有确定布局。
        compensation_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        # 输入控件横向填满可用区域，避免依赖内容文字计算过大的尺寸提示。
        compensation_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # 标签从左侧对齐，与右栏标题和复选框建立统一扫描线。
        compensation_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        # 8 px 垂直间距兼顾紧凑度与不同字段之间的分组感。
        compensation_form.setVerticalSpacing(8)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("幅相", "both")
        self.mode_combo.addItem("仅幅频", "magnitude")
        self.mode_combo.addItem("仅相频", "phase")
        self.frequency_unit_combo = QComboBox()
        self.frequency_unit_combo.addItems(list(FREQUENCY_FACTORS))
        self.frequency_unit_combo.setCurrentText("GHz")
        # 模式和单位是首要选择，优先放在自动化开关之前。
        compensation_form.addRow("补偿模式", self.mode_combo)
        # 频率单位与所有频带数值共享，紧邻模式便于一次确认。
        compensation_form.addRow("频率单位", self.frequency_unit_combo)
        self.auto_frequency_bands = QCheckBox("根据拟合脉冲自动设置补偿频带")
        self.auto_frequency_bands.setChecked(True)
        self.auto_frequency_bands.setToolTip(
            "按各自峰值归一化，选择共同 -20 dB 最长连续谱宽候选；输入需已去基线"
        )
        # 复选框作为整行内容，不与其他字段共享横向空间。
        compensation_form.addRow(self.auto_frequency_bands)
        self.detrend_phase_checkbox = QCheckBox("去除线性相位")
        self.detrend_phase_checkbox.setChecked(True)
        self.detrend_phase_checkbox.setToolTip(
            "打开：报告相对时延，但不用该线性相位平移目标数据。"
            "关闭：仍拟合并报告时延，同时在补偿相位中保留该线性项。"
        )
        # 相位处理开关同样独占一行，完整文字在高 DPI 下仍可显示。
        compensation_form.addRow(self.detrend_phase_checkbox)
        self.limit_gain_checkbox = QCheckBox("限制最大补偿增益")
        self.limit_gain_checkbox.setChecked(True)
        self.limit_gain_checkbox.setToolTip(
            "默认限制为 20 dB，避免深陷波或整体幅度过小导致噪声和输出爆炸；"
            "关闭后将按原始响应比应用，并在参数记录中保留该选择。"
        )
        compensation_form.addRow(self.limit_gain_checkbox)
        self.maximum_gain_db = CompactDoubleSpinBox()
        self.maximum_gain_db.setRange(0.0, 200.0)
        self.maximum_gain_db.setDecimals(1)
        self.maximum_gain_db.setValue(20.0)
        self.maximum_gain_db.setSuffix(" dB")
        self.maximum_gain_db.setKeyboardTracking(False)
        compensation_form.addRow("最大补偿增益", self.maximum_gain_db)
        self.edge_transition_percent = CompactDoubleSpinBox()
        self.edge_transition_percent.setRange(0.0, 50.0)
        self.edge_transition_percent.setDecimals(1)
        self.edge_transition_percent.setValue(10.0)
        self.edge_transition_percent.setSuffix(" %")
        self.edge_transition_percent.setKeyboardTracking(False)
        self.edge_transition_percent.setToolTip(
            "在补偿频带两侧用 raised-cosine 从单位响应平滑过渡；百分比按补偿带宽的每一侧计算。"
        )
        compensation_form.addRow("边缘过渡（每侧）", self.edge_transition_percent)
        self.band_low = self._frequency_spin(0.0)
        self.band_high = self._frequency_spin(0.0)
        self.phase_low = self._frequency_spin(0.0)
        self.phase_high = self._frequency_spin(0.0)
        for spin in (self.band_low, self.band_high):
            spin.setSpecialValueText("分析后自动")
        for spin in (self.phase_low, self.phase_high):
            spin.setSpecialValueText("首次分析自动建议")
        rows = [
            ("补偿起点", self.band_low),
            ("补偿终点", self.band_high),
            ("相位拟合频带起点", self.phase_low),
            ("相位拟合频带终点", self.phase_high),
        ]
        # 四个频带字段依次加入上下表单，完整标签不会再推挤输入框。
        for label, widget in rows:
            # addRow 保留字段语义关系，并让控件使用表单全部可用宽度。
            compensation_form.addRow(label, widget)
        layout.addWidget(compensation_group)

        layout.addStretch(1)
        scroll.setWidget(content)
        panel_layout.addWidget(scroll, 1)

        action_bar = QFrame()
        action_bar.setObjectName("inspectorActions")
        action_layout = QVBoxLayout(action_bar)
        action_layout.setContentsMargins(14, 10, 14, 12)
        action_layout.setSpacing(8)
        self.compare_button = QPushButton("拟合脉冲比较")
        self.compare_button.setObjectName("secondaryButton")
        self.compare_button.setMinimumHeight(44)
        self.compare_button.setAccessibleName("比较参考和待补偿拟合脉冲")
        self.compare_button.clicked.connect(self._start_comparison)
        self.compensate_button = QPushButton("数据补偿")
        self.compensate_button.setObjectName("primaryButton")
        self.compensate_button.setMinimumHeight(48)
        self.compensate_button.setAccessibleName("对目标信号执行数据补偿")
        self.compensate_button.clicked.connect(self._start_compensation)
        # 保留旧属性，避免外部自动化脚本在一次版本升级中失效。
        self.analyze_button = self.compensate_button
        self.export_button = QPushButton("导出补偿结果")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.setMinimumHeight(44)
        self.export_button.setAccessibleName("导出补偿结果和诊断文件")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        action_layout.addWidget(self.compare_button)
        action_layout.addWidget(self.compensate_button)
        action_layout.addWidget(self.export_button)
        panel_layout.addWidget(action_bar)
        return panel

    @staticmethod
    def _frequency_spin(value: float) -> CompactDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(0.0, 1.0e15)
        # GHz 显示下仍保留到 1 mHz，切换单位不会把低频设置静默量化为 0。
        spin.setDecimals(12)
        spin.setValue(value)
        spin.setSuffix(" GHz")
        spin.setKeyboardTracking(False)
        # QDoubleSpinBox 默认把范围上限的完整十进制字符串计入最小宽度；Windows
        # 字体度量较宽时会因此撑破右栏。允许表单按视口压缩，编辑器仍可通过
        # 光标水平滚动访问任意长的用户值。
        spin.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        return spin

    def _set_header_state(
        self,
        text: str,
        tone: Literal["neutral", "active", "success", "warning", "error"],
    ) -> None:
        """同步更新状态文字、语义色和辅助功能描述。"""

        # 可见文字保留完整状态含义，语义色只用于加快扫视，不能成为唯一提示。
        self.header_state.setText(text)
        self.header_state.setProperty("tone", tone)
        self.header_state.setAccessibleDescription(f"ResponseLab 当前状态：{text}")
        # Qt 动态属性变化后需要重新抛光，才能立即命中 tone 属性选择器。
        self.header_state.style().unpolish(self.header_state)
        self.header_state.style().polish(self.header_state)
        # 左下脉冲轨迹复用同一语义状态，运行时扫光、结果态静止，避免出现两套状态机。
        self.response_field.set_tone(tone)

    # 根据三张输入卡的真实路径刷新轨迹亮度，不复制文件解析或有效性判断。
    def _update_response_field_inputs(self, *_args: object) -> None:
        # 只统计已选择路径的卡片；文件存在性仍由启动任务时的统一校验负责。
        selected_count = sum(
            card.path is not None for card in (self.reference_card, self.dut_card, self.target_card)
        )
        # 将 0–3 的输入完成度交给纯视觉组件，并同步其辅助描述。
        self.response_field.set_input_count(selected_count)

    def _connect_stale_signals(self) -> None:
        for card in (self.reference_card, self.dut_card):
            # 拟合脉冲决定自动相位频带；更换文件时先按自动/手动语义处理旧建议。
            card.path_selected.connect(self._fitted_pulse_path_changed)
            # 两份拟合脉冲同样决定影响频段缓存，但不与导出版本共用状态。
            card.path_selected.connect(self._mark_influence_stale)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.auto_frequency_bands.toggled.connect(self._automatic_frequency_bands_changed)
        # 自动/手动扫描范围会改变候选频段，因此影响页也需失效。
        self.auto_frequency_bands.toggled.connect(self._mark_influence_stale)
        self.detrend_phase_checkbox.toggled.connect(self._mark_stale)
        # 相位去斜开关改变相位归因，但不应禁用已有补偿导出之外的额外状态。
        self.detrend_phase_checkbox.toggled.connect(self._mark_influence_stale)
        self.limit_gain_checkbox.toggled.connect(self._gain_limit_toggled)
        self.limit_gain_checkbox.toggled.connect(self._mark_stale)
        self.limit_gain_checkbox.toggled.connect(self._mark_influence_stale)
        self.maximum_gain_db.valueChanged.connect(self._mark_stale)
        self.maximum_gain_db.valueChanged.connect(self._mark_influence_stale)
        self.edge_transition_percent.valueChanged.connect(self._mark_stale)
        self.edge_transition_percent.valueChanged.connect(self._mark_influence_stale)
        self.frequency_unit_combo.currentTextChanged.connect(self._frequency_unit_changed)
        self.band_low.valueChanged.connect(self._band_edges_changed)
        # 手动扫描下限变化使影响候选失效。
        self.band_low.valueChanged.connect(self._mark_influence_stale)
        self.band_high.valueChanged.connect(self._band_edges_changed)
        # 手动扫描上限变化同样使影响候选失效。
        self.band_high.valueChanged.connect(self._mark_influence_stale)
        for spin in (self.phase_low, self.phase_high):
            spin.valueChanged.connect(self._phase_band_changed)
            # 相位拟合带变化会改变局部相位归因结果。
            spin.valueChanged.connect(self._mark_influence_stale)
        self._mode_changed(self.mode_combo.currentIndex())
        self._automatic_frequency_bands_changed(True)

    def _gain_limit_toggled(self, checked: bool) -> None:
        """只在含幅度补偿且用户启用上限时开放 dB 数值。"""

        magnitude_enabled = str(self.mode_combo.currentData()) != "phase"
        self.maximum_gain_db.setEnabled(magnitude_enabled and checked)

    def _band_edges_changed(self, *_args: object) -> None:
        self._mark_stale()

    def _mode_changed(self, _index: int) -> None:
        mode = str(self.mode_combo.currentData())
        phase_enabled = mode != "magnitude"
        self.detrend_phase_checkbox.setEnabled(phase_enabled)
        self.phase_low.setEnabled(phase_enabled)
        self.phase_high.setEnabled(phase_enabled)
        magnitude_enabled = mode != "phase"
        self.limit_gain_checkbox.setEnabled(magnitude_enabled)
        self.maximum_gain_db.setEnabled(magnitude_enabled and self.limit_gain_checkbox.isChecked())
        self._mark_stale()

    def _automatic_frequency_bands_changed(self, automatic: bool) -> None:
        for spin in (self.band_low, self.band_high):
            spin.setSpecialValueText("分析后自动" if automatic else "请输入")
        self.band_low.setEnabled(not automatic)
        self.band_high.setEnabled(not automatic)
        if self._phase_band_initialized:
            phase_special = "0"
        elif automatic:
            phase_special = "首次分析自动建议"
        else:
            phase_special = "请输入"
        for spin in (self.phase_low, self.phase_high):
            spin.setSpecialValueText(phase_special)
        phase_enabled = str(self.mode_combo.currentData()) != "magnitude"
        self.detrend_phase_checkbox.setEnabled(phase_enabled)
        self.phase_low.setEnabled(phase_enabled)
        self.phase_high.setEnabled(phase_enabled)
        self._mark_stale()

    def _phase_band_changed(self, *_args: object) -> None:
        if not self._building:
            self._phase_band_is_manual = True
            self._phase_band_initialized = True
            for spin in (self.phase_low, self.phase_high):
                spin.setSpecialValueText("0")
        self._mark_stale()

    def _frequency_unit_changed(self, new_unit: str) -> None:
        old_unit = self._last_frequency_unit
        if new_unit == old_unit or new_unit not in FREQUENCY_FACTORS:
            return
        spins = (
            self.band_low,
            self.band_high,
            self.phase_low,
            self.phase_high,
        )
        physical_before_hz = [spin.value() * FREQUENCY_FACTORS[old_unit] for spin in spins]
        conversion = FREQUENCY_FACTORS[old_unit] / FREQUENCY_FACTORS[new_unit]
        for spin in spins:
            previous = spin.blockSignals(True)
            spin.setValue(spin.value() * conversion)
            spin.setSuffix(f" {new_unit}")
            spin.blockSignals(previous)
        self._last_frequency_unit = new_unit
        physical_after_hz = [spin.value() * FREQUENCY_FACTORS[new_unit] for spin in spins]
        quantized = any(
            not np.isclose(before, after, rtol=1.0e-12, atol=1.0e-12)
            for before, after in zip(physical_before_hz, physical_after_hz, strict=True)
        )
        if quantized:
            self._mark_stale()
        if self._result is not None:
            self._populate_plots(self._result)

    def _target_path_changed(self, path: str) -> None:
        self._mark_compensation_input_stale()

    def _fitted_pulse_path_changed(self, *_args: object) -> None:
        """更换拟合脉冲后废弃旧的自动相位频带，但保留用户手动值。"""

        # 用户手动输入是明确配置，跨文件切换时不擅自改写。
        if not self._phase_band_is_manual:
            # 旧建议只对上一组 Fs/脉冲有效；下一次请求必须重新调用频带建议器。
            self._phase_band_initialized = False
            # 清掉可见旧值，避免分析期间误显示上一组脉冲的物理频带。
            for spin in (self.phase_low, self.phase_high):
                previous = spin.blockSignals(True)
                spin.setValue(0.0)
                spin.setSpecialValueText(
                    "首次分析自动建议" if self.auto_frequency_bands.isChecked() else "请输入"
                )
                spin.blockSignals(previous)
        # 文件变化无论是否保留手动频带，都必须让已有结果失效。
        self._mark_stale()

    def _mark_compensation_input_stale(self, *_args: object) -> None:
        """只让真正依赖第三份数据的任务或结果失效。"""

        if self._building or self._active_action == "compare":
            return
        if self._active_action == "compensate" or isinstance(self._result, CompensationRun):
            self._mark_stale()

    def _mark_stale(self, *_args: object) -> None:
        if self._building:
            return
        self._parameter_version += 1
        # 正在运行的普通比较/补偿已绑定旧版本；让加载器、FFT 或分块 DSP 在下一个
        # 安全边界停止，避免用户换文件或参数后仍必须等待整份旧任务完成。
        if isinstance(self._worker, AnalysisThread):
            self._worker.requestInterruption()
        if self._result is not None:
            self.export_button.setEnabled(False)
            self._set_header_state("预览已过期", "warning")
            self.statusBar().showMessage("参数或输入已变化，请重新分析后再导出")

    # 影响页参数变化只递增自己的版本，既有补偿预览和导出状态保持有效。
    def _mark_influence_stale(self, *_args: object) -> None:
        """只让影响频段结果过期，不修改现有补偿预览与导出资格。"""

        # 构造期控件初值会发信号，但尚不存在需要失效的用户结果。
        if self._building:
            # 保持初始版本为零。
            return
        # 独立版本递增，使已在后台运行的旧影响任务完成后被拒绝。
        self._influence_version += 1
        # 清除可点选工作区，避免旧候选按新 M 回放。
        self._influence_run = None
        # 当前行恢复为未选择状态。
        self._influence_selected_row = -1
        # 页面参数变化时移除旧曲线和图像；原补偿页完全不受影响。
        self.influence_page.clear_result()
        # 正在运行的影响任务在下一个候选边界响应中断，节省无用计算。
        if isinstance(
            self._worker,
            (InfluenceAnalysisThread, InfluenceSelectionThread),
        ):
            # QThread 中断请求由扫描回调轮询，不做危险强制终止。
            self._worker.requestInterruption()
        # 只有用户正在查看影响页时才用状态栏提示，不覆盖其他页的主要反馈。
        if self.visual_tabs.currentIndex() == self.influence_tab_index:
            # 提示重新开始影响分析，同时不改变导出按钮状态。
            self.statusBar().showMessage("影响频段参数或输入已变化，请重新分析")

    def _current_settings(self) -> CompensationSettings:
        # 模式决定补偿响应是否使用幅度、相位或两者，频率数值随后统一换算为 Hz。
        mode = str(self.mode_combo.currentData())
        # UI 可显示 Hz/kHz/MHz/GHz；DSP 合同始终使用 Hz，避免单位混入算法层。
        factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
        maximum_gain_db = (
            self.maximum_gain_db.value()
            if mode != "phase" and self.limit_gain_checkbox.isChecked()
            else None
        )
        edge_transition_fraction = self.edge_transition_percent.value() / 100.0
        # 自动频带用 0–1 Hz 占位通过模型校验，后台会在计算前替换为真实公共可信频带。
        if self.auto_frequency_bands.isChecked():
            # 用户确认过相位拟合频带后继续沿用；首次分析则交给频带建议器给出初值。
            use_initialized_phase_band = mode != "magnitude" and self._phase_band_initialized
            return CompensationSettings(
                mode=mode,
                band_low_hz=0.0,
                band_high_hz=1.0,
                phase_fit_low_hz=(
                    self.phase_low.value() * factor if use_initialized_phase_band else 0.0
                ),
                phase_fit_high_hz=(
                    self.phase_high.value() * factor if use_initialized_phase_band else 1.0
                ),
                detrend_phase=self.detrend_phase_checkbox.isChecked(),
                maximum_gain_db=maximum_gain_db,
                edge_transition_fraction=edge_transition_fraction,
                analysis_points=16385,
            )
        # 手动模式把显示单位换回 Hz，所有边界检查由 CompensationSettings 集中完成。
        band_low_hz = self.band_low.value() * factor
        band_high_hz = self.band_high.value() * factor
        return CompensationSettings(
            mode=mode,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            phase_fit_low_hz=self.phase_low.value() * factor,
            phase_fit_high_hz=self.phase_high.value() * factor,
            detrend_phase=self.detrend_phase_checkbox.isChecked(),
            maximum_gain_db=maximum_gain_db,
            edge_transition_fraction=edge_transition_fraction,
            analysis_points=16385,
        )

    # 把当前页面快照冻结成后台请求，并在启动前完成路径、格式和 worker 占用校验。
    def _start_influence_analysis(self, payload: object) -> None:
        """校验影响页快照并占用主窗口唯一后台 worker。"""

        # 已有比较、补偿或候选回放任务运行时不并发启动第二个线程。
        if self._worker is not None:
            # 单 worker 约束同时保护关闭流程和状态栏语义。
            return
        # 页面合同要求轻量映射；非法信号载荷在 GUI 边界明确提示。
        if not isinstance(payload, Mapping):
            # 不让属性异常泄漏到 Qt 事件循环。
            QMessageBox.warning(self, "参数无效", "影响频段请求必须是参数映射")
            # 停止本次启动。
            return
        # 两份拟合脉冲继续复用主窗口左栏输入。
        reference_path = self.reference_card.path
        # DUT 拟合脉冲同样来自左栏第二张卡。
        dut_path = self.dut_card.path
        # 缺少任一脉冲无法建立 Href/Hdut。
        if reference_path is None or dut_path is None:
            # 明确区分拟合脉冲与 Vpp 原始数据。
            QMessageBox.warning(self, "输入不完整", "请先选择两份拟合脉冲。")
            # 不创建线程。
            return
        # 指标内部键不依赖中文下拉文字。
        metric = str(payload.get("metric", ""))
        # 只允许页面确认的三个互斥指标。
        if metric not in {"vpp", "eye_height", "eye_width"}:
            # 拼写或协议错误不默认成 Vpp。
            QMessageBox.warning(self, "参数无效", "请选择 Vpp、眼高或眼宽。")
            # 停止启动。
            return
        # 只有外部码型来源需要额外文件；内置 PRBS13Q 不携带旧路径状态。
        pattern_source_value = payload.get("pattern_source")
        pattern_source = None if pattern_source_value is None else str(pattern_source_value)
        pattern_path_value = payload.get("pattern_path")
        pattern_path = None if pattern_path_value is None else Path(pattern_path_value)
        if metric == "vpp" and pattern_source == "file" and pattern_path is None:
            QMessageBox.warning(self, "输入不完整", "请选择理想码型文件。")
            return
        # 汇总当前任务真正依赖的全部文件。
        required_paths = [reference_path, dut_path]
        # 外部理想码型与两份拟合脉冲共同决定本次 Vpp 模型。
        if pattern_path is not None:
            required_paths.append(pattern_path)
        # 启动前只做存在性检查，大文件读取留在后台。
        missing_paths = [
            str(path) for path in required_paths if path is not None and not path.is_file()
        ]
        # 任一文件不存在时集中列出。
        if missing_paths:
            # 用户可以直接修正具体路径。
            QMessageBox.critical(
                self,
                "文件不存在",
                "以下文件无法读取：\n" + "\n".join(missing_paths),
            )
            # 不创建后台线程。
            return
        # 页面眼参数在 Vpp 下为空。
        modulation_value = payload.get("modulation")
        # 内部键统一转成字符串或 None。
        modulation = None if modulation_value is None else str(modulation_value)
        # M 同时用于眼图与 Vpp 稳态码型的上采样网格。
        samples_per_ui = payload.get("m")
        try:
            current_m = int(samples_per_ui)
        except (TypeError, ValueError, OverflowError):
            QMessageBox.warning(self, "参数无效", "M 必须是整数")
            return
        # 页面统一用 Hz 传递本次候选核心宽度和相邻中心步进。
        band_width_hz_value = payload.get("band_width_hz")
        # 构造设置和请求可能因空手动频带或 BIN 参数失败。
        try:
            # 缺失字段通常表示旧页面协议，不能静默退回固定 100 MHz。
            if band_width_hz_value is None:
                # 错误直接指向新增公共输入。
                raise ValueError("请设置频段宽度")
            # InfluenceRequest 继续负责正数、有限值和 bool 等领域校验。
            band_width_hz = float(band_width_hz_value)
            # 当前右栏设置已经完成显示单位到 Hz 的换算。
            frequency_settings = self._current_settings()
            # 影响分析固定比较三种模式，即使主页面当前选“仅相位”，幅度候选仍必须
            # 遵守用户保留的增益上限；不能沿用主补偿为纯相位返回的 None。
            influence_maximum_gain_db = (
                self.maximum_gain_db.value()
                if self.limit_gain_checkbox.isChecked()
                else None
            )
            frequency_settings = replace(
                frequency_settings,
                maximum_gain_db=influence_maximum_gain_db,
            )
            # 影响分析始终比较幅度、相位和幅相三支；主模式切到“仅幅度”时，
            # _current_settings 会用 0–1 Hz 占位，但不能因此丢掉此前确认的可见相位带。
            frequency_factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
            visible_phase_low_hz = self.phase_low.value() * frequency_factor
            visible_phase_high_hz = self.phase_high.value() * frequency_factor
            has_confirmed_phase_band = (
                self._phase_band_initialized and 0.0 <= visible_phase_low_hz < visible_phase_high_hz
            )
            if has_confirmed_phase_band:
                frequency_settings = replace(
                    frequency_settings,
                    phase_fit_low_hz=visible_phase_low_hz,
                    phase_fit_high_hz=visible_phase_high_hz,
                )
            # 冻结完整请求；后台不会读取正在变化的 Qt 控件。
            request = InfluenceRequest(
                reference_pulse_path=reference_path,
                dut_pulse_path=dut_path,
                metric=metric,
                modulation=modulation,
                samples_per_ui=(None if samples_per_ui is None else current_m),
                vpp_method=(
                    None if payload.get("vpp_method") is None else str(payload.get("vpp_method"))
                ),
                pattern_source=pattern_source,
                pattern_path=pattern_path,
                pattern_value_kind=(
                    None
                    if payload.get("pattern_value_kind") is None
                    else str(payload.get("pattern_value_kind"))
                ),
                pre_cursor_ui=(
                    None
                    if payload.get("pre_cursor_ui") is None
                    else int(payload.get("pre_cursor_ui"))
                ),
                post_cursor_ui=(
                    None
                    if payload.get("post_cursor_ui") is None
                    else int(payload.get("post_cursor_ui"))
                ),
                band_width_hz=band_width_hz,
                frequency_settings=frequency_settings,
                auto_frequency_bands=self.auto_frequency_bands.isChecked(),
                version=self._influence_version,
                auto_phase_fit_band=(
                    self.auto_frequency_bands.isChecked() and not has_confirmed_phase_band
                ),
            )
        # 参数模型给出的 ValueError 可直接展示给用户。
        except (TypeError, ValueError) as error:
            # 不让无效数字进入后台。
            QMessageBox.warning(self, "参数无效", str(error))
            # 停止启动。
            return
        # 同一路径文件可在外部被改写而不触发路径控件信号。新任务一旦通过启动校验，
        # 旧候选、曲线和工作区必须立即失效，不能在新加载失败时继续冒充当前结果。
        self._influence_run = None
        self._influence_selected_row = -1
        self.influence_page.clear_result()
        # 页面按钮和候选列表进入忙碌态，参数本身保持可读。
        self.influence_page.set_busy(True)
        # 比较按钮禁用，维持唯一 worker。
        self.compare_button.setEnabled(False)
        # 数据补偿按钮同样禁用。
        self.compensate_button.setEnabled(False)
        # 影响任务不修改旧补偿资格，但唯一 worker 正被占用；暂时禁用导出，
        # 收尾后再按既有补偿版本恢复，避免覆盖仍运行的 QThread 引用。
        self.export_button.setEnabled(False)
        self.progress.setRange(0, 0)
        # 显示不定进度，收到候选总数后再切成确定进度。
        self.progress.show()
        # 记录当前动作供统一收尾和安全关闭使用。
        self._active_action = "influence"
        # 状态栏给出当前步骤。
        self.statusBar().showMessage("正在扫描主要影响频段…")
        # 创建唯一影响扫描线程。
        self._worker = InfluenceAnalysisThread(request)
        # 成功结果走独立版本门禁和页面适配。
        self._worker.succeeded.connect(self._influence_succeeded)
        # 失败不调用旧补偿结果处理器。
        self._worker.failed.connect(self._influence_failed)
        self._worker.cancelled.connect(self._influence_cancelled)
        # 候选级进度更新共享状态栏进度条。
        self._worker.progressed.connect(self._influence_progressed)
        # 长记录工作量提示在大型频谱分配前显示。
        self._worker.noticed.connect(self._influence_noticed)
        # 所有路径共用同一个 worker 收尾函数。
        self._worker.finished.connect(self._worker_finished)
        # 启动后台读取和扫描。
        self._worker.start()

    # 将候选评估计数映射到共用进度条，同时防止过期线程写入当前任务状态。
    def _influence_progressed(self, completed: int, total: int) -> None:
        """把核心候选计数映射到状态栏确定进度。"""

        # 总数至少为一，防止 QProgressBar 退回不定模式。
        safe_total = max(1, int(total))
        # 进度范围按真实候选评估数设置。
        self.progress.setRange(0, safe_total)
        # 已完成数钳位到合法范围，避免晚到信号越界。
        self.progress.setValue(max(0, min(int(completed), safe_total)))

    # 长任务提示只接受当前影响页版本，避免旧线程文字覆盖新请求。
    def _influence_noticed(self, message: str, version: int) -> None:
        """仅显示当前影响请求的工作量提示。"""

        # 参数变化前的旧线程不得覆盖当前页面状态。
        if version != self._influence_version:
            # 忽略过期提示。
            return
        # 非空提示来自后台的候选数和记录长度估算。
        if message:
            # 状态栏保持单行，不弹出阻塞式确认框。
            self.statusBar().showMessage(message)

    # 用统一口径格式化参考、补偿前和补偿后三个指标，供扫描与点选摘要复用。
    @staticmethod
    def _influence_metric_summary(
        run: InfluenceRun,
        metric_after: float | None,
    ) -> str:
        """格式化参考、补偿前和可选补偿后的同口径指标。"""

        # LFP 使用 Vpp 电压；频域方法明确标为 Vrms，不能伪装成等效 Vpp。
        if (
            run.workspace.settings.metric == "vpp"
            and run.workspace.settings.vpp is not None
            and run.workspace.settings.vpp.method == "frequency_rms_error"
        ):
            unit_suffix = " Vrms"
            metric_labels = ("参考误差", "补偿前误差", "补偿后误差")
        else:
            unit_suffix = {
                "vpp": " V",
                "eye_height": "",
                "eye_width": " UI",
            }[run.workspace.settings.metric]
            # LFP/眼图三份数值本身就是同一指标，不额外重复长算法名称。
            metric_labels = ("参考", "补偿前", "补偿后")
        # 参考与补偿前是整次扫描固定的成对基线。
        parts = [
            f"{metric_labels[0]} {run.result.reference_metric:.4g}{unit_suffix}",
            f"{metric_labels[1]} {run.result.before_metric:.4g}{unit_suffix}",
        ]
        # 只有有效候选回放才追加补偿后指标。
        if metric_after is not None and np.isfinite(metric_after):
            # 当前候选的标量与扫描排名使用完全相同的度量口径。
            parts.append(f"{metric_labels[2]} {metric_after:.4g}{unit_suffix}")
        # 依赖 M 的指标必须同时显示 Fs、Rs、M 和 UI 时长，避免错误 M 生成貌似合理的结果。
        if run.workspace.vpp_cache is not None:
            cache = run.workspace.vpp_cache
            rate_context = (
                f"Fs {cache.sample_rate_hz / 1.0e9:.4g} GSa/s · "
                f"Rs {cache.symbol_rate_hz / 1.0e9:.4g} GBd · "
                f"M {cache.settings.samples_per_ui} samples/UI · "
                f"UI {cache.ui_duration_s * 1.0e12:.4g} ps"
            )
            return rate_context + "\n" + " · ".join(parts)
        if run.workspace.settings.eye is not None:
            sample_rate_hz = run.workspace.reference_pulse.sample_rate_hz
            samples_per_ui = run.workspace.settings.eye.samples_per_ui
            symbol_rate_hz = sample_rate_hz / samples_per_ui
            ui_duration_s = samples_per_ui / sample_rate_hz
            rate_context = (
                f"Fs {sample_rate_hz / 1.0e9:.4g} GSa/s · "
                f"Rs {symbol_rate_hz / 1.0e9:.4g} GBd · "
                f"M {samples_per_ui} samples/UI · "
                f"UI {ui_duration_s * 1.0e12:.4g} ps"
            )
            return rate_context + "\n" + " · ".join(parts)
        # 中点分隔符保持一行可扫读，页面在窄窗口可自动换行。
        return " · ".join(parts)

    # 仅在版本匹配且默认候选数据完整时，一次性提交曲线、候选和眼图/波形。
    def _influence_succeeded(self, run: InfluenceRun, version: int) -> None:
        """把完整扫描结果事务式提交到第六页签。"""

        if self._close_when_finished:
            self.statusBar().showMessage("影响分析已结束，窗口即将关闭")
            return
        # 参数变化前完成的旧任务不能覆盖当前空白页。
        if version != self._influence_version:
            # 只给状态栏提示，现有补偿结果继续有效。
            self.statusBar().showMessage("旧影响分析已结束，但参数已变化；结果未采用")
            # 不保存旧工作区。
            return
        # 保存工作区和候选映射供点击回放。
        self._influence_run = run
        # 页面渲染会默认选中首行，先设行号避免触发重复回放。
        self._influence_selected_row = 0 if run.displayed_candidates else -1
        # 先准备三模式曲线和候选列表。
        view = influence_curve_payload(run)
        # 眼模式附加三幅共轴的 2 UI 轨迹叠加图。
        if run.eye_comparison is not None:
            # 角色映射由控制器统一生成。
            view["eyes"] = eye_payload(run.eye_comparison)
        # Vpp 模式附加参考、补偿前和补偿后三条稳态码型模型波形。
        if run.workspace.settings.metric == "vpp" and run.selected_evaluation is not None:
            # 波形各自保留真实独立时间轴。
            view["waveforms"] = waveform_payload(
                run.workspace,
                run.selected_evaluation,
            )
        # 根据保守状态生成简短摘要，不在 UI 标题重复算法限定词。
        if run.result.status == "ok" and run.result.recommendation is not None:
            # 推荐同时包含频段和幅度/相位/幅相模式。
            recommendation = run.result.recommendation
            # 模式显示使用用户确认的短标签。
            mode_label = {
                "magnitude": "幅度",
                "phase": "相位",
                "both": "幅相",
            }[recommendation.mode]
            # 摘要频率换算为 GHz，只发生在展示层。
            summary = (
                f"推荐 {recommendation.band.low_hz / 1.0e9:.3f}–"
                f"{recommendation.band.high_hz / 1.0e9:.3f} GHz · {mode_label}"
            )
        # 指标本来就无差距时不显示伪频段。
        elif run.result.status == "no_difference":
            # 页面摘要清楚说明没有可归因差距。
            summary = "参考与 DUT 指标没有可解析差距"
        # 全频模型不闭环或局部改善不足时明确无推荐。
        else:
            # 不强行选择最大数值噪声候选；具体原因由结构化结果中的全部告警补充。
            summary = "当前频响模型未找到可推荐频段"
        # 默认候选的补偿后指标来自同一次正式回放。
        selected_metric = (
            run.selected_evaluation.attribution.metric_after
            if run.selected_evaluation is not None and run.selected_evaluation.attribution.valid
            else None
        )
        # 页面摘要先给推荐，再给三份同口径标量，避免用户从图片反推数值。
        visible_summary = summary + "\n" + self._influence_metric_summary(run, selected_metric)
        # 所有算法告警都持久显示在页面中；状态栏不是完整诊断的唯一载体。
        if run.result.warnings:
            remaining_page_warnings = tuple(
                warning for warning in dict.fromkeys(run.result.warnings) if warning not in summary
            )
            if remaining_page_warnings:
                visible_summary += "\n" + " · ".join(remaining_page_warnings)
        view["summary"] = visible_summary
        # 完整映射先验证后一次提交，异常时保留旧完整结果。
        self.influence_page.render_result(view)
        # 若存在候选，页面列表默认聚焦第一行。
        if run.displayed_candidates:
            # 程序化默认选择暂时阻塞信号，避免重复启动一次候选回放。
            previous = self.influence_page.candidate_list.blockSignals(True)
            # 第一行已由控制器保证是推荐或最大改善候选。
            self.influence_page.candidate_list.setCurrentRow(0)
            # 恢复候选列表原信号状态。
            self.influence_page.candidate_list.blockSignals(previous)
        # 状态栏也保留全部去重告警，避免一个告警修复反而遮住另一个。
        status_message = summary
        if run.result.warnings:
            remaining_warnings = tuple(
                warning
                for warning in dict.fromkeys(run.result.warnings)
                if warning not in status_message
            )
            if remaining_warnings:
                status_message += " · " + " · ".join(remaining_warnings)
        # 更新状态栏。
        self.statusBar().showMessage(status_message)

    # 当前任务失败时恢复影响页交互，但不清除用户已有的正常补偿结果。
    def _influence_failed(self, message: str, version: int) -> None:
        """显示当前影响任务失败，同时保护旧补偿导出状态。"""

        # 窗口已请求任务结束后关闭时不弹出模态对话框阻挡关闭流程。
        if self._close_when_finished:
            # 失败或取消均交给统一 finished 槽完成关闭。
            self.statusBar().showMessage("影响分析已安全停止，窗口即将关闭")
            # 不再显示错误弹窗。
            return
        # 旧版本失败只说明任务结束，不弹出与当前参数无关的对话框。
        if version != self._influence_version:
            # 当前页仍等待用户按新参数重新开始。
            self.statusBar().showMessage("旧影响分析已停止；请按当前参数重新分析")
            # 不更改全局补偿结果。
            return
        # 状态栏给出恢复动作。
        self.statusBar().showMessage("影响分析失败 · 请检查输入与 M")
        # 弹窗展示后台领域错误。
        QMessageBox.critical(self, "无法完成影响分析", message)

    def _influence_cancelled(self, version: int) -> None:
        """Handle an expected influence stop without presenting a failure modal."""

        if self._close_when_finished:
            self.statusBar().showMessage("影响分析已取消，窗口正在关闭")
            return
        if version != self._influence_version:
            self.statusBar().showMessage("旧影响任务已安全停止；请按当前参数重新分析")
            return
        if self._active_action == "influence_candidate" and self._influence_run is not None:
            previous = self.influence_page.candidate_list.blockSignals(True)
            self.influence_page.candidate_list.setCurrentRow(self._influence_selected_row)
            self.influence_page.candidate_list.blockSignals(previous)
        self.statusBar().showMessage("影响分析已安全取消")

    # 候选回放失败时恢复上一条已提交选择，避免列表高亮与详情图指向不同频段。
    def _influence_selection_failed(self, message: str, version: int) -> None:
        """恢复上一候选行，并保留最后一次成功提交的详情。"""

        # 关闭流程和过期版本沿用普通影响任务的静默保护边界。
        if self._close_when_finished or version != self._influence_version:
            # 共用处理器负责关闭提示或过期任务提示。
            self._influence_failed(message, version)
            # 不操作可能已被新版本清空的候选列表。
            return
        # 阻塞 currentRowChanged，防止恢复行号再次启动后台回放。
        previous = self.influence_page.candidate_list.blockSignals(True)
        # -1 表示此前没有成功候选，否则恢复最后一次成功提交的行。
        self.influence_page.candidate_list.setCurrentRow(self._influence_selected_row)
        # 恢复调用前的信号状态。
        self.influence_page.candidate_list.blockSignals(previous)
        # 状态栏明确说明旧详情仍然有效，避免用户把失败行误认为已经应用。
        self.statusBar().showMessage("频段回放失败 · 已保留上一候选结果")
        # 弹窗保留后台给出的具体原因。
        QMessageBox.critical(self, "无法更新所选频段", message)

    # 点选列表行只重放已有工作区候选，不重新加载文件或重扫全部频段。
    def _start_influence_selection(self, row: int) -> None:
        """在后台重放用户点选的已有候选。"""

        # 清空列表会发出 -1，直接忽略。
        if row < 0:
            # 没有候选无需回放。
            return
        # 默认选中或重复点击当前行不重复做 FFT/眼图卷积。
        if row == self._influence_selected_row:
            # 保持当前详情图。
            return
        # 运行中的其他任务占用唯一 worker 时不启动点选回放。
        if self._worker is not None:
            # 阻塞信号恢复最后成功行，避免程序化选择或竞态造成高亮与详情错位。
            previous = self.influence_page.candidate_list.blockSignals(True)
            # -1 表示尚无已提交详情，其余值对应当前可见详情。
            self.influence_page.candidate_list.setCurrentRow(self._influence_selected_row)
            # 恢复列表原信号状态。
            self.influence_page.candidate_list.blockSignals(previous)
            # 当前 worker 完成前不排队新候选。
            return
        # 必须有与当前版本匹配的完整扫描结果。
        run = self._influence_run
        # 行号必须位于候选映射范围内。
        if run is None or not 0 <= row < len(run.displayed_candidates):
            # 忽略晚到或越界行号。
            return
        # 取得不可变候选。
        candidate = run.displayed_candidates[row]
        # 页面进入忙碌态以避免用户连续排队多个回放。
        self.influence_page.set_busy(True)
        # 旧比较按钮也遵守单 worker 约束。
        self.compare_button.setEnabled(False)
        # 数据补偿按钮禁用到回放结束。
        self.compensate_button.setEnabled(False)
        # 候选回放同样占用唯一 worker；保留资格但暂时禁止启动导出线程。
        self.export_button.setEnabled(False)
        # 点选回放使用短暂不定进度。
        self.progress.setRange(0, 0)
        # 显示进度条。
        self.progress.show()
        # 当前动作记录为候选回放。
        self._active_action = "influence_candidate"
        # 状态栏提示当前操作。
        self.statusBar().showMessage("正在更新所选频段的补偿后结果…")
        # 使用原工作区缓存创建候选线程。
        self._worker = InfluenceSelectionThread(
            run.workspace,
            candidate,
            self._influence_version,
        )
        # 成功只更新详情图，不动影响曲线和列表。
        self._worker.succeeded.connect(self._influence_selection_succeeded)
        # 回放失败要恢复上一条已提交行，保持列表与详情一致。
        self._worker.failed.connect(self._influence_selection_failed)
        self._worker.cancelled.connect(self._influence_cancelled)
        # 共用统一收尾。
        self._worker.finished.connect(self._worker_finished)
        # 启动回放。
        self._worker.start()

    # 候选回放完成后只替换详情图和三值摘要，总览曲线与参考基线保持不变。
    def _influence_selection_succeeded(
        self,
        selection: InfluenceSelection,
        version: int,
    ) -> None:
        """只替换补偿后波形/眼图和当前候选摘要。"""

        if self._close_when_finished:
            self.statusBar().showMessage("候选计算已结束，窗口即将关闭")
            return
        # 参数变化后的旧点选结果不再有效。
        if version != self._influence_version or self._influence_run is None:
            # 不覆盖当前页面。
            return
        # 详情映射不包含频率曲线和候选列表。
        detail: dict[str, object] = {}
        # 眼模式更新三图，其中参考和补偿前数组由同一工作区保持不变。
        if selection.eye_comparison is not None:
            # 页面使用同一协议更新图像。
            detail["eyes"] = eye_payload(selection.eye_comparison)
        # Vpp 模式更新三条稳态码型模型波形。
        if selection.evaluation.corrected_values is not None and (
            self._influence_run.workspace.settings.metric == "vpp"
        ):
            # 参考和补偿前保持工作区基线，只有补偿后随候选变化。
            detail["waveforms"] = waveform_payload(
                self._influence_run.workspace,
                selection.evaluation,
            )
        # 当前行可通过数据类值在稳定候选元组中定位。
        self._influence_selected_row = self._influence_run.displayed_candidates.index(
            selection.candidate
        )
        # 模式短标签用于摘要。
        mode_label = {
            "magnitude": "幅度",
            "phase": "相位",
            "both": "幅相",
        }[selection.candidate.mode]
        # 摘要只包含当前频段、模式和改善量。
        detail["summary"] = (
            f"{selection.candidate.band.low_hz / 1.0e9:.3f}–"
            f"{selection.candidate.band.high_hz / 1.0e9:.3f} GHz · "
            f"{mode_label} · 改善 {selection.candidate.improvement:.4g}"
            + "\n"
            + self._influence_metric_summary(
                self._influence_run,
                selection.evaluation.attribution.metric_after,
            )
        )
        # 页面只更新详情区域，不重置列表选择。
        self.influence_page.render_selection(detail)
        # 状态栏同步当前候选。
        self.statusBar().showMessage(str(detail["summary"]))

    def _start_comparison(self) -> None:
        self._start_task("compare")

    def _start_compensation(self) -> None:
        self._start_task("compensate")

    def _start_analysis(self) -> None:
        """兼容旧的自动化入口；等价于点击“数据补偿”。"""

        self._start_compensation()

    def _start_task(self, action: Literal["compare", "compensate"]) -> None:
        # finished 信号收尾前引用仍表示任务占用，不能只依赖 isRunning()。
        if self._worker is not None:
            return
        reference_path = self.reference_card.path
        dut_path = self.dut_card.path
        target_path = self.target_card.path if action == "compensate" else None
        required_paths = [reference_path, dut_path]
        if action == "compensate":
            required_paths.append(target_path)
        if any(path is None for path in required_paths):
            requirement = (
                "请先选择参考拟合脉冲和待补偿拟合脉冲。"
                if action == "compare"
                else "请先选择两份拟合脉冲和待补偿信号。"
            )
            QMessageBox.warning(self, "输入不完整", requirement)
            return
        assert reference_path is not None and dut_path is not None
        missing = [str(path) for path in required_paths if path is not None and not path.is_file()]
        if missing:
            QMessageBox.critical(self, "文件不存在", "以下文件无法读取：\n" + "\n".join(missing))
            return
        try:
            request = AnalysisRequest(
                reference_path=reference_path,
                dut_path=dut_path,
                target_path=target_path,
                settings=self._current_settings(),
                version=self._parameter_version,
                action=action,
                auto_frequency_bands=self.auto_frequency_bands.isChecked(),
                auto_phase_fit_band=(
                    self.auto_frequency_bands.isChecked()
                    and str(self.mode_combo.currentData()) != "magnitude"
                    and not self._phase_band_initialized
                ),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self.compare_button.setEnabled(False)
        self.compensate_button.setEnabled(False)
        # 唯一后台 worker 被普通比较/补偿占用时，影响页也进入只读忙碌态。
        self.influence_page.set_busy(True)
        self.export_button.setEnabled(False)
        self.progress.show()
        self._active_action = action
        # 活动态使用蓝色语义边框，并继续保留“比较中/补偿中”文字供读屏读取。
        self._set_header_state("比较中" if action == "compare" else "补偿中", "active")
        self.statusBar().showMessage(
            "正在比较两份拟合脉冲…" if action == "compare" else "正在解析输入并执行数据补偿…"
        )
        self._worker = AnalysisThread(request)
        self._worker.succeeded.connect(self._analysis_succeeded)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.cancelled.connect(self._analysis_cancelled)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _analysis_succeeded(self, result: PulseComparison | CompensationRun, version: int) -> None:
        if self._close_when_finished:
            self.statusBar().showMessage("分析已结束，窗口即将关闭")
            return
        if version != self._parameter_version:
            self._set_header_state("预览已过期", "warning")
            self.statusBar().showMessage("旧分析任务已结束，但参数已变化；结果未用于导出")
            return
        self._result_version = version
        if isinstance(result, CompensationRun):
            self.present_run(result)
            self.export_button.setEnabled(True)
        else:
            self.present_comparison(result)
            self.export_button.setEnabled(False)

    def _analysis_failed(self, message: str, version: int) -> None:
        if self._close_when_finished:
            self.statusBar().showMessage("分析已停止，窗口即将关闭")
            return
        if version == self._parameter_version:
            self._set_header_state("分析失败", "error")
            self.statusBar().showMessage("分析失败 · 请按提示修正输入或参数")
            QMessageBox.critical(self, "无法完成分析", message)
        else:
            self._set_header_state("参数已变化", "warning")
            self.statusBar().showMessage("旧分析任务失败后已结束；请按当前参数重新分析")

    def _analysis_cancelled(self, version: int) -> None:
        """Handle cooperative cancellation without presenting it as an error."""

        if self._close_when_finished:
            self.statusBar().showMessage("分析已取消，窗口正在关闭")
        elif version == self._parameter_version:
            self._set_header_state("分析已取消", "warning")
            self.statusBar().showMessage("分析已安全取消")

    def _worker_finished(self) -> None:
        self.progress.hide()
        # 下次普通比较任务继续使用不定进度，避免遗留候选计数范围。
        self.progress.setRange(0, 0)
        self.compare_button.setEnabled(True)
        self.compensate_button.setEnabled(True)
        if self._active_action in {
            "export",
            "influence",
            "influence_candidate",
        }:
            self.export_button.setEnabled(
                self._run is not None and self._result_version == self._parameter_version
            )
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._active_action = None
        # 任意唯一后台 worker 结束后都恢复影响页操作。
        self.influence_page.set_busy(False)
        if self._close_when_finished:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        # finished 信号收尾前引用仍表示窗口正在处理该任务。
        if self._worker is not None:
            # 所有后台任务都在安全阶段或分块边界轮询；绝不使用 terminate 强杀线程。
            self._worker.requestInterruption()
            self._close_when_finished = True
            self._set_header_state("完成后关闭", "active")
            self.statusBar().showMessage("任务正在安全停止，完成当前步骤后窗口将自动关闭")
            event.ignore()
            return
        event.accept()

    def present_run(self, run: CompensationRun, source_label: str | None = None) -> None:
        """把一次已完成运行绑定到全部卡片、指标与五个绘图页面。"""

        self._present_result(run, source_label=source_label)

    def present_comparison(
        self, comparison: PulseComparison, source_label: str | None = None
    ) -> None:
        """展示只依赖两份拟合脉冲的比较，不伪造待补偿数据。"""

        self._present_result(comparison, source_label=source_label)

    def _present_result(
        self,
        result: PulseComparison | CompensationRun,
        *,
        source_label: str | None = None,
    ) -> None:
        self._result = result
        self._run = result if isinstance(result, CompensationRun) else None
        is_compensation = isinstance(result, CompensationRun)
        self._set_header_state("预览有效" if is_compensation else "比较有效", "success")
        reference_rate = self._format_frequency(result.reference_pulse.sample_rate_hz)
        dut_rate = self._format_frequency(result.dut_pulse.sample_rate_hz)
        self.reference_card.set_summary(f"{result.reference_pulse.samples:,} 点 · {reference_rate}")
        self.dut_card.set_summary(f"{result.dut_pulse.samples:,} 点 · {dut_rate}")
        if is_compensation:
            target_rate = self._format_frequency(result.input_signal.sample_rate_hz)
            self.target_card.set_summary(f"{result.input_signal.samples:,} 点 · {target_rate}")
        self._show_effective_frequency_settings(result.analysis.settings)
        settings = result.analysis.settings
        analysis_range = (
            f"{self._format_frequency(settings.band_low_hz)}–"
            f"{self._format_frequency(settings.band_high_hz)}"
        )
        if result.analysis.settings.mode == "magnitude":
            metric_text = f"分析频带 {analysis_range}"
        else:
            phase_range = (
                f"{self._format_frequency(settings.phase_fit_low_hz)}–"
                f"{self._format_frequency(settings.phase_fit_high_hz)}"
            )
            delay_text = self._format_signed_delay(result.analysis.estimated_relative_delay_s)
            detrend_text = "已去除线性相位" if settings.detrend_phase else "已保留线性相位"
            metric_text = (
                f"分析频带 {analysis_range} · 相位拟合频带 {phase_range} · "
                f"相对时延 {delay_text} · {detrend_text}"
            )
        gain_limit_text = (
            f"最大增益 {settings.maximum_gain_db:g} dB"
            if settings.maximum_gain_db is not None
            else "最大增益不限制"
        )
        metric_text += (
            f" · {gain_limit_text} · 边缘过渡 {100.0 * settings.edge_transition_fraction:g}%/侧"
        )
        # 保留旧自动化读取接口，同时不把摘要控件重新放回可见布局。
        self.metric_label.setText(metric_text)
        # 详细摘要通过页签辅助描述提供给读屏和新版自动化检查。
        self.visual_tabs.setAccessibleDescription(metric_text)
        self.result_warning.setText(" · ".join(result.warnings))
        self.result_warning.setVisible(bool(result.warnings))
        blue_description = "实际补偿频带" if is_compensation else "分析/候选补偿频带"
        legend_text = f"蓝色阴影：{blue_description}"
        if result.analysis.settings.mode != "magnitude":
            phase_application = (
                "补偿相位已去线性趋势" if settings.detrend_phase else "补偿相位保留线性趋势"
            )
            legend_text += f"　橙色虚线：线性相位拟合频带边界　{phase_application}"
        self.band_legend_label.setText(legend_text)
        self.visual_tabs.setTabEnabled(4, is_compensation)
        self._populate_plots(result)
        label = source_label or ("文件补偿" if is_compensation else "拟合脉冲比较")
        if is_compensation:
            strategy_label = {
                "exact": "精确整段",
                "streaming": "有限边界分块",
            }.get(str(result.application_metadata.get("strategy", "")))
            if strategy_label and settings.application_strategy == "auto":
                suffix = f"频响补偿已应用（自动选择：{strategy_label}）"
            elif strategy_label:
                suffix = f"频响补偿已应用（{strategy_label}）"
            else:
                suffix = "频响补偿已应用"
        else:
            suffix = "未读取或改写待补偿数据"
        self.statusBar().showMessage(f"{label}完成 · {suffix}")

    def _show_effective_frequency_settings(self, settings: CompensationSettings) -> None:
        factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
        previous = self.detrend_phase_checkbox.blockSignals(True)
        self.detrend_phase_checkbox.setChecked(settings.detrend_phase)
        self.detrend_phase_checkbox.blockSignals(previous)
        previous = self.limit_gain_checkbox.blockSignals(True)
        self.limit_gain_checkbox.setChecked(settings.maximum_gain_db is not None)
        self.limit_gain_checkbox.blockSignals(previous)
        if settings.maximum_gain_db is not None:
            previous = self.maximum_gain_db.blockSignals(True)
            self.maximum_gain_db.setValue(settings.maximum_gain_db)
            self.maximum_gain_db.blockSignals(previous)
        previous = self.edge_transition_percent.blockSignals(True)
        self.edge_transition_percent.setValue(100.0 * settings.edge_transition_fraction)
        self.edge_transition_percent.blockSignals(previous)
        self._gain_limit_toggled(self.limit_gain_checkbox.isChecked())
        pairs = [
            (self.band_low, settings.band_low_hz / factor),
            (self.band_high, settings.band_high_hz / factor),
        ]
        initialize_phase_band = settings.mode != "magnitude" and not self._phase_band_initialized
        if initialize_phase_band:
            pairs.extend(
                [
                    (self.phase_low, settings.phase_fit_low_hz / factor),
                    (self.phase_high, settings.phase_fit_high_hz / factor),
                ]
            )
        for spin, value in pairs:
            previous = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(previous)
        if initialize_phase_band:
            self._phase_band_initialized = True
            for spin in (self.phase_low, self.phase_high):
                spin.setSpecialValueText("0")

    @staticmethod
    def _format_frequency(value_hz: float) -> str:
        for unit, factor in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
            if abs(value_hz) >= factor:
                return f"{value_hz / factor:.4g} {unit}"
        return f"{value_hz:.4g} Hz"

    @staticmethod
    def _format_signed_delay(value_s: float) -> str:
        """以带符号的工程单位显示相对时延。"""

        absolute_s = abs(value_s)
        # 从大到小选择能让绝对值至少为 1 的工程单位，避免显示大量前导零。
        for candidate_unit, candidate_factor in (
            ("s", 1.0),
            ("ms", 1.0e-3),
            ("µs", 1.0e-6),
            ("ns", 1.0e-9),
        ):
            if absolute_s >= candidate_factor:
                unit, factor = candidate_unit, candidate_factor
                break
        else:
            unit, factor = "ps", 1.0e-12
        if value_s > 0.0:
            sign = "+"
            relation = "DUT 较晚"
        elif value_s < 0.0:
            sign = "-"
            relation = "DUT 较早"
        else:
            sign = ""
            relation = "无相对偏移"
        return f"{sign}{absolute_s / factor:.4g} {unit}（{relation}）"

    @staticmethod
    def _time_display(time_s: np.ndarray) -> tuple[np.ndarray, str]:
        span = float(np.max(time_s) - np.min(time_s)) if time_s.size else 0.0
        if span < 1e-9:
            return time_s / 1e-12, "ps"
        if span < 1e-6:
            return time_s / 1e-9, "ns"
        if span < 1e-3:
            return time_s / 1e-6, "µs"
        if span < 1.0:
            return time_s / 1e-3, "ms"
        return time_s, "s"

    def _frequency_display(self, frequency_hz: np.ndarray) -> tuple[np.ndarray, str]:
        unit = self.frequency_unit_combo.currentText()
        return frequency_hz / FREQUENCY_FACTORS[unit], unit

    @staticmethod
    def _plot_curve(
        plot: pg.PlotWidget,
        x: np.ndarray,
        y: np.ndarray,
        *,
        name: str,
        color: str,
        dashed: bool = False,
        width: float = 1.8,
    ) -> None:
        # 首条曲线出现时才创建图例，初始空图因此不会显示无内容边框。
        legend = plot.getPlotItem().legend
        # 没有现成图例时创建并应用与深色画布一致的样式。
        if legend is None:
            # 图例靠近数据且使用高对比文字，用户不必跨越图表寻找系列含义。
            legend = plot.addLegend(offset=(12, 10), labelTextColor=TEXT)
            # 半透明深色底使图例在曲线上方仍保持清晰，同时不过度遮挡数据。
            legend.setBrush(pg.mkBrush(14, 20, 29, 218))
            # 细边框与工作台分隔线一致，避免形成截图中的突兀空框。
            legend.setPen(pg.mkPen(BORDER))
        # 清空图表时图例会暂时隐藏；新曲线加入前重新显示它。
        legend.show()
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        plot.plot(x, y, name=name, pen=pg.mkPen(color, width=width, style=style))

    @staticmethod
    def _add_band_region(
        plot: pg.PlotWidget,
        low: float,
        high: float,
        *,
        color: str,
        alpha: int = 22,
    ) -> None:
        fill = pg.mkColor(color)
        fill.setAlpha(alpha)
        outline = pg.mkColor(color)
        outline.setAlpha(min(110, alpha * 4))
        region = pg.LinearRegionItem(
            values=(low, high),
            movable=False,
            brush=pg.mkBrush(fill),
            pen=pg.mkPen(outline, width=1.0),
        )
        region.setZValue(-10)
        plot.addItem(region)

    @staticmethod
    def _add_band_boundaries(
        plot: pg.PlotWidget,
        low: float,
        high: float,
        *,
        color: str,
    ) -> None:
        pen = pg.mkPen(color, width=1.2, style=Qt.PenStyle.DashLine)
        for position, description in (
            (low, "线性相位拟合频带起点"),
            (high, "线性相位拟合频带终点"),
        ):
            boundary = pg.InfiniteLine(
                pos=position,
                angle=90,
                movable=False,
                pen=pen,
            )
            boundary.setToolTip(description)
            boundary.setZValue(-5)
            plot.addItem(boundary)

    @staticmethod
    def _center_phase_islands(
        phase_rad: np.ndarray,
        reliable_mask: np.ndarray,
        anchor_score: np.ndarray,
    ) -> np.ndarray:
        """每个连续可信岛统一减去整数圈，避免逐点 wrap 制造假跳变。"""

        phase = np.asarray(phase_rad, dtype=np.float64)
        reliable = np.asarray(reliable_mask, dtype=bool)
        score = np.asarray(anchor_score, dtype=np.float64)
        valid = reliable & np.isfinite(phase) & np.isfinite(score)
        output = np.full(phase.shape, np.nan, dtype=np.float64)
        padded = np.r_[False, valid, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in changes.reshape(-1, 2):
            segment = phase[start:stop].copy()
            segment_score = score[start:stop]
            maximum = float(np.max(segment_score))
            candidates = np.flatnonzero(np.isclose(segment_score, maximum, rtol=1.0e-12, atol=0.0))
            center = 0.5 * (segment.size - 1)
            anchor = int(candidates[np.argmin(np.abs(candidates - center))])
            segment -= 2.0 * np.pi * np.round(segment[anchor] / (2.0 * np.pi))
            output[start:stop] = np.degrees(segment)
        return output

    @staticmethod
    def _fit_visible_curves_y_range(
        plot: pg.PlotWidget,
        minimum_span: float,
    ) -> None:
        """按当前可见 x 范围内的全部曲线设置确定性的 y 范围。"""

        # 只统计当前 x 视窗内的有限样本，避免视窗外尖峰把局部曲线压成一条直线。
        x_low, x_high = plot.viewRange()[0]
        visible_values: list[np.ndarray] = []
        for curve in plot.listDataItems():
            x_values, y_values = curve.getData()
            x = np.asarray(x_values, dtype=np.float64)
            y = np.asarray(y_values, dtype=np.float64)
            visible = np.isfinite(x) & np.isfinite(y) & (x >= x_low) & (x <= x_high)
            if np.any(visible):
                visible_values.append(y[visible])
        if not visible_values:
            return
        finite = np.concatenate(visible_values)
        low = float(np.min(finite))
        high = float(np.max(finite))
        # 各图的最小跨度防止近乎常数的曲线产生零高度视窗，8% 留白便于读数。
        span = max(high - low, minimum_span)
        center = 0.5 * (low + high)
        margin = 0.08 * span
        plot.setYRange(
            center - 0.5 * span - margin,
            center + 0.5 * span + margin,
            padding=0.0,
        )

    def _populate_plots(self, run: PulseComparison | CompensationRun) -> None:
        for plot in self._all_plots():
            plot.clear()
            # 如果该图曾有图例，先隐藏空壳；后续真正添加曲线时再由 _plot_curve 显示。
            legend = plot.getPlotItem().legend
            # 仅对已经创建过图例的图执行隐藏，初始空图保持完全没有图例对象。
            if legend is not None:
                # 隐藏空图例可避免比较模式下无输出曲线的页面再次出现小方框。
                legend.hide()
        # analysis 已包含同一频率网格上的幅相、可信掩码和补偿响应，UI 不再重算主算法。
        analysis = run.analysis
        # 横轴按用户所选单位显示，但频带边界仍从 Hz 物理量等比例换算。
        frequency, frequency_unit = self._frequency_display(analysis.frequency_hz)
        frequency_factor = FREQUENCY_FACTORS[frequency_unit]
        band_low = analysis.settings.band_low_hz / frequency_factor
        band_high = analysis.settings.band_high_hz / frequency_factor
        phase_low = analysis.settings.phase_fit_low_hz / frequency_factor
        phase_high = analysis.settings.phase_fit_high_hz / frequency_factor
        # 低能量或相位不可信频点显示为断线，不能用连线伪装成可靠测量。
        reliable = analysis.reliable_mask

        ref_time, pulse_time_unit = self._time_display(run.reference_pulse.time_s)
        dut_time = run.dut_pulse.time_s / TIME_FACTORS[pulse_time_unit]
        pulse_plot = self.pulse_plots[0]
        self._plot_curve(
            pulse_plot,
            ref_time,
            run.reference_pulse.values[:, 0],
            name="参考拟合脉冲",
            color=REFERENCE,
        )
        self._plot_curve(
            pulse_plot,
            dut_time,
            run.dut_pulse.values[:, 0],
            name="待补偿拟合脉冲",
            color=DUT,
            dashed=True,
        )
        pulse_plot.setLabel("bottom", "时间", units=pulse_time_unit)
        pulse_plot.setLabel("left", "幅值")

        magnitude_plot, phase_plot = self.response_plots
        reference_magnitude_display = np.where(
            reliable,
            analysis.reference_magnitude_db,
            np.nan,
        )
        dut_magnitude_display = np.where(
            reliable,
            analysis.dut_magnitude_db,
            np.nan,
        )
        self._plot_curve(
            magnitude_plot,
            frequency,
            reference_magnitude_display,
            name="参考",
            color=REFERENCE,
        )
        self._plot_curve(
            magnitude_plot,
            frequency,
            dut_magnitude_display,
            name="待补偿",
            color=DUT,
            dashed=True,
        )
        self._add_band_region(
            magnitude_plot,
            band_low,
            band_high,
            color=ACCENT,
        )
        magnitude_plot.setLabel("bottom", "频率", units=frequency_unit)
        magnitude_plot.setLabel("left", "幅度", units="dB")
        # 相位拟合只使用“可信掩码 ∩ 用户拟合频带”，单位保持 Hz。
        phase_observation_mask = (
            reliable
            & (analysis.frequency_hz >= analysis.settings.phase_fit_low_hz)
            & (analysis.frequency_hz <= analysis.settings.phase_fit_high_hz)
        )
        reference_phase_trend = np.zeros_like(analysis.frequency_hz)
        dut_phase_trend = np.zeros_like(analysis.frequency_hz)
        # 仅在相位参与补偿且用户打开去斜时，图中参考与 DUT 扣除各自线性趋势。
        phase_is_detrended = (
            analysis.settings.mode != "magnitude" and analysis.settings.detrend_phase
        )
        if phase_is_detrended and np.count_nonzero(phase_observation_mask) >= 3:
            reference_peak_db = float(np.max(analysis.reference_magnitude_db))
            dut_peak_db = float(np.max(analysis.dut_magnitude_db))
            # 两条频谱归一化功率的逐点较小值作为联合权重，弱谱一侧会主动降低影响。
            joint_weights = np.minimum(
                np.exp((analysis.reference_magnitude_db - reference_peak_db) * np.log(10.0) / 10.0),
                np.exp((analysis.dut_magnitude_db - dut_peak_db) * np.log(10.0) / 10.0),
            )
            try:
                # 共享拟合器按连续可信岛估计 rad/Hz 斜率，与 DSP 报告时延的口径一致。
                reference_slope = fit_linear_phase_slope(
                    analysis.frequency_hz,
                    analysis.reference_phase_rad,
                    joint_weights,
                    phase_observation_mask,
                )
                dut_slope = fit_linear_phase_slope(
                    analysis.frequency_hz,
                    analysis.dut_phase_rad,
                    joint_weights,
                    phase_observation_mask,
                )
            # 可视化拟合失败时保留原始相位；DSP 结果本身已经完成独立校验，不在这里伪造趋势。
            except ValueError:
                pass
            else:
                reference_phase_trend = reference_slope * analysis.frequency_hz
                dut_phase_trend = dut_slope * analysis.frequency_hz
        self._plot_curve(
            phase_plot,
            frequency,
            np.where(
                reliable,
                self._center_phase_islands(
                    analysis.reference_phase_rad - reference_phase_trend,
                    reliable,
                    analysis.reference_magnitude_db,
                ),
                np.nan,
            ),
            name="参考（去斜）" if phase_is_detrended else "参考",
            color=REFERENCE,
        )
        if analysis.settings.mode == "magnitude":
            phase_difference_name = "相位差（仅诊断）"
        elif analysis.settings.detrend_phase:
            phase_difference_name = "相位差（去斜前）"
        else:
            phase_difference_name = "相位差（未去斜，实际补偿）"
        self._plot_curve(
            phase_plot,
            frequency,
            np.where(
                reliable,
                self._center_phase_islands(
                    analysis.dut_phase_rad - dut_phase_trend,
                    reliable,
                    analysis.dut_magnitude_db,
                ),
                np.nan,
            ),
            name="待补偿（去斜）" if phase_is_detrended else "待补偿",
            color=DUT,
            dashed=True,
        )
        phase_plot.setLabel("bottom", "频率", units=frequency_unit)
        phase_plot.setLabel("left", "相位", units="°")

        difference_magnitude, difference_phase = self.difference_plots
        self._plot_curve(
            difference_magnitude,
            frequency,
            np.where(reliable, analysis.magnitude_difference_db, np.nan),
            name="参考 - 待补偿",
            color=RESULT,
        )
        self._add_band_region(
            difference_magnitude,
            band_low,
            band_high,
            color=ACCENT,
        )
        difference_magnitude.setLabel("bottom", "频率", units=frequency_unit)
        difference_magnitude.setLabel("left", "幅度差", units="dB")
        phase_difference_display = analysis.phase_after_optional_detrend_rad.copy()
        if analysis.settings.detrend_phase:
            phase_difference_display = phase_difference_display + analysis.phase_trend_rad
        self._plot_curve(
            difference_phase,
            frequency,
            np.degrees(phase_difference_display),
            name=phase_difference_name,
            color=DUT,
        )
        if analysis.settings.mode != "magnitude" and analysis.settings.detrend_phase:
            self._plot_curve(
                difference_phase,
                frequency,
                np.degrees(analysis.phase_after_optional_detrend_rad),
                name="实际补偿相位（去斜后）",
                color=RESULT,
            )
        self._add_band_region(
            difference_phase,
            band_low,
            band_high,
            color=ACCENT,
        )
        if analysis.settings.mode != "magnitude":
            self._add_band_boundaries(
                difference_phase,
                phase_low,
                phase_high,
                color=WARNING,
            )
        difference_phase.setLabel("bottom", "频率", units=frequency_unit)
        difference_phase.setLabel("left", "相位差", units="°")

        compensation_magnitude, compensation_phase = self.compensator_plots
        # 把理想复数补偿响应换算为 dB；1e-300 只保护 log10，不改变可见有效频带。
        ideal_db = 20.0 * np.log10(np.maximum(np.abs(analysis.correction_ideal), 1e-300))
        self._plot_curve(
            compensation_magnitude,
            frequency,
            ideal_db,
            name="补偿幅度",
            color=ACCENT,
        )
        self._add_band_region(
            compensation_magnitude,
            band_low,
            band_high,
            color=ACCENT,
        )
        compensation_magnitude.setLabel("bottom", "频率", units=frequency_unit)
        compensation_magnitude.setLabel("left", "补偿幅度", units="dB")
        ideal_phase_display = np.where(
            (analysis.frequency_hz >= analysis.settings.band_low_hz)
            & (analysis.frequency_hz <= analysis.settings.band_high_hz),
            np.degrees(np.angle(analysis.correction_ideal)),
            np.nan,
        )
        self._plot_curve(
            compensation_phase,
            frequency,
            ideal_phase_display,
            name="补偿相位",
            color=ACCENT,
        )
        self._add_band_region(
            compensation_phase,
            band_low,
            band_high,
            color=ACCENT,
        )
        compensation_phase.setLabel("bottom", "频率", units=frequency_unit)
        compensation_phase.setLabel("left", "补偿相位", units="°")

        waveform_plot, spectrum_plot = self.output_plots
        if isinstance(run, CompensationRun):
            waveform_preview = _output_waveform_preview_slice(run.input_signal.samples)
            output_time, output_time_unit = self._time_display(
                run.input_signal.time_s[waveform_preview]
            )
            self._plot_curve(
                waveform_plot,
                output_time,
                run.input_signal.values[waveform_preview, 0],
                name="补偿前",
                color=DUT,
                dashed=True,
            )
            self._plot_curve(
                waveform_plot,
                output_time,
                run.output_values[waveform_preview, 0],
                name="补偿后",
                color=RESULT,
            )
            waveform_plot.setLabel("bottom", "时间", units=output_time_unit)
            waveform_plot.setLabel("left", "幅值")
            # 大记录只对中间连续窗口做单边 FFT；连续窗口不引入抽取混叠，且把界面
            # 预览工作区限制在约百万点。完整输出本身及导出数据不被裁剪。
            spectrum_preview = _output_spectrum_preview_slice(run.input_signal.samples)
            spectrum_samples = spectrum_preview.stop - spectrum_preview.start
            input_spectrum = rfft(run.input_signal.values[spectrum_preview, 0])
            output_spectrum = rfft(run.output_values[spectrum_preview, 0])
            # rfftfreq 根据目标信号采样率生成 Hz 横轴，随后再切换到界面选择的显示单位。
            signal_frequency_hz = np.fft.rfftfreq(
                spectrum_samples,
                d=1.0 / run.input_signal.sample_rate_hz,
            )
            signal_frequency, signal_unit = self._frequency_display(signal_frequency_hz)
            # 两条曲线共用峰值基准，既保留补偿增益差，又把数学零点限制为可读的 -160 dB。
            spectrum_peak = max(
                float(np.max(np.abs(input_spectrum))),
                float(np.max(np.abs(output_spectrum))),
                np.finfo(np.float64).tiny,
            )
            spectrum_floor = spectrum_peak * 1.0e-8
            input_spectrum_db = 20.0 * np.log10(
                np.maximum(np.abs(input_spectrum), spectrum_floor) / spectrum_peak
            )
            output_spectrum_db = 20.0 * np.log10(
                np.maximum(np.abs(output_spectrum), spectrum_floor) / spectrum_peak
            )
            self._plot_curve(
                spectrum_plot,
                signal_frequency,
                input_spectrum_db,
                name="补偿前",
                color=DUT,
                dashed=True,
            )
            self._plot_curve(
                spectrum_plot,
                signal_frequency,
                output_spectrum_db,
                name="补偿后",
                color=RESULT,
            )
            spectrum_plot.setLabel("bottom", "频率", units=signal_unit)
            spectrum_plot.setLabel("left", "相对 DFT 幅度", units="dB")
        self._reset_plots()

    def _all_plots(self) -> list[pg.PlotWidget]:
        return [
            *self.pulse_plots,
            *self.response_plots,
            *self.difference_plots,
            *self.compensator_plots,
            *self.output_plots,
        ]

    def _set_plot_mouse_mode(self, mode: Literal["zoom", "pan"]) -> None:
        mouse_mode = pg.ViewBox.RectMode if mode == "zoom" else pg.ViewBox.PanMode
        self.zoom_button.setChecked(mode == "zoom")
        self.pan_button.setChecked(mode == "pan")
        for plot in self._all_plots():
            plot.getViewBox().setMouseMode(mouse_mode)
        # 影响频段页签自己管理四幅图，但仍跟随主工具栏的缩放/平移模式。
        self.influence_page.set_mouse_mode(mode)

    def _frequency_plots(self) -> list[pg.PlotWidget]:
        return [
            *self.response_plots,
            *self.difference_plots,
            *self.compensator_plots[:2],
            self.output_plots[1],
        ]

    def _focus_frequency_plots(self, run: PulseComparison | CompensationRun) -> None:
        # 默认视窗至少覆盖补偿带；相位模式还要完整包含拟合带，避免关键边界落在屏幕外。
        settings = run.analysis.settings
        relevant_high_hz = settings.band_high_hz
        if settings.mode != "magnitude":
            relevant_high_hz = max(relevant_high_hz, settings.phase_fit_high_hz)
        # 右侧增加 8% 物理频宽留白，并用分析网格 Nyquist 截断，防止展示不存在的频率。
        margin_hz = max(
            0.08 * (settings.band_high_hz - settings.band_low_hz),
            relevant_high_hz * 0.08,
        )
        view_high_hz = min(
            float(run.analysis.frequency_hz[-1]),
            relevant_high_hz + margin_hz,
        )
        frequency_factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
        view_high = view_high_hz / frequency_factor
        for plot in self._frequency_plots():
            plot.setXRange(0.0, view_high, padding=0.02)

    def _focus_output_preview(self, run: CompensationRun) -> None:
        # 波形页默认展示开头连续 40960 点；这一窗口可直接反映局部形状，且不会让
        # 数千万点记录拖慢 GUI。只换算两个标量，不为大记录分配完整时间轴副本。
        preview_end = min(
            run.input_signal.samples - 1,
            _MAX_OUTPUT_WAVEFORM_PREVIEW_SAMPLES - 1,
        )
        output_time, _ = self._time_display(run.input_signal.time_s[[0, preview_end]])
        self.output_plots[0].setXRange(
            output_time[0],
            output_time[1],
            padding=0.02,
        )

    def _apply_recommended_y_spans(self, run: PulseComparison | CompensationRun) -> None:
        del run  # 曲线已经包含所有显示变换，直接按实际绘制数据定范围。
        # 幅度、相位和差值使用不同最小跨度，保证近似平坦结果仍保留有意义的刻度。
        recommended = (
            (self.response_plots[0], 6.0),
            (self.response_plots[1], 20.0),
            (self.difference_plots[0], 1.0),
            (self.difference_plots[1], 20.0),
            (self.compensator_plots[0], 1.0),
            (self.compensator_plots[1], 2.0),
        )
        for plot, minimum_span in recommended:
            self._fit_visible_curves_y_range(plot, minimum_span)

    def _reset_plots(self) -> None:
        for plot in self._all_plots():
            plot.enableAutoRange()
            plot.autoRange()
            plot.disableAutoRange()
        if self._result is not None:
            self._focus_frequency_plots(self._result)
            self._apply_recommended_y_spans(self._result)
        if isinstance(self._result, CompensationRun):
            self._focus_output_preview(self._result)
        # 第六页签按自身当前数据恢复视图，不依赖补偿页的 _result。
        self.influence_page.reset_view()

    def _export(self) -> None:
        # 所有后台工作共享同一个受控 worker 引用。即使按钮状态因平台事件
        # 或程序化调用滞后，也不能用导出线程覆盖仍运行的分析/回放线程。
        if self._worker is not None:
            self.statusBar().showMessage("当前任务完成后才能导出补偿结果")
            return
        if self._run is None or self._result_version != self._parameter_version:
            QMessageBox.warning(self, "预览已过期", "请先按当前参数重新分析。")
            return
        is_bin = self._run.input_signal.source_format == "bin"
        suffix = ".bin" if is_bin else ".csv"
        file_filter = "BIN (*.bin)" if is_bin else "CSV (*.csv)"
        source = self._run.input_signal.source_path
        suggested = (
            source.with_name(f"{source.stem}_compensated{suffix}")
            if source is not None
            else Path(f"compensated{suffix}")
        )
        destination, _ = QFileDialog.getSaveFileName(
            self, "导出补偿结果", str(suggested), file_filter
        )
        if not destination:
            return
        try:
            paths = bundle_paths(destination)
            destination_fingerprints = snapshot_bundle_destinations(paths)
            existing = [
                path
                for path, fingerprint in zip(
                    paths.as_tuple(),
                    destination_fingerprints,
                    strict=True,
                )
                if fingerprint.exists
            ]
            if existing:
                names = "\n".join(f"• {path}" for path in existing)
                answer = QMessageBox.question(
                    self,
                    "确认覆盖导出文件",
                    f"以下文件已存在，将作为同一导出批次一起替换：\n\n{names}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.compare_button.setEnabled(False)
        self.compensate_button.setEnabled(False)
        self.influence_page.set_busy(True)
        self.export_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self._active_action = "export"
        self._set_header_state("导出中", "active")
        self.statusBar().showMessage("正在后台生成补偿结果与诊断文件…")
        self._worker = ExportThread(
            self._run,
            paths.output,
            self._result_version,
            destination_fingerprints,
        )
        self._worker.succeeded.connect(self._export_succeeded)
        self._worker.source_invalid.connect(self._export_source_invalid)
        self._worker.rollback_incomplete.connect(self._export_rollback_incomplete)
        self._worker.cleanup_incomplete.connect(self._export_cleanup_incomplete)
        self._worker.failed.connect(self._export_failed)
        self._worker.cancelled.connect(self._export_cancelled)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _export_succeeded(self, outcome: object, version: int) -> None:
        """Present one fully committed background export bundle."""

        if isinstance(outcome, ExportOutcome):
            paths = outcome.paths
            cleanup_warning = outcome.cleanup_warning
        else:
            paths = outcome
            cleanup_warning = ""
        if not all(hasattr(paths, field) for field in ("output", "response_csv", "manifest")):
            self._export_failed("导出线程返回了无效路径结果", self._result_version)
            return
        if cleanup_warning:
            self._set_header_state("导出完成，需检查", "warning")
            self.statusBar().showMessage("输出已提交 · 部分临时文件未清理，请查看详情")
        elif version == self._parameter_version:
            self._set_header_state("导出完成", "success")
            self.statusBar().showMessage(f"已导出：{paths.output}")
        else:
            self._set_header_state("当前预览已过期", "warning")
            self.statusBar().showMessage(f"旧预览已导出：{paths.output} · 当前参数需要重新分析")
        if cleanup_warning:
            self._close_when_finished = False
            QMessageBox.warning(
                self,
                "导出完成，但需要检查",
                "三个输出文件已完整提交，但部分临时文件或旧备份未能安全清理：\n\n"
                + cleanup_warning,
            )
        elif not self._close_when_finished:
            QMessageBox.information(
                self,
                "导出完成",
                f"补偿结果：{paths.output}\n"
                f"响应诊断：{paths.response_csv}\n"
                f"参数记录：{paths.manifest}",
            )

    def _export_source_invalid(self, message: str, _version: int) -> None:
        """Invalidate a preview whose source bytes changed before export."""

        self._result_version = -1
        self.export_button.setEnabled(False)
        self._set_header_state("源文件已变化", "error")
        self.statusBar().showMessage("输入文件已变化 · 请重新分析后再导出")
        if not self._close_when_finished:
            QMessageBox.critical(self, "需要重新分析", message)

    def _export_failed(self, message: str, version: int) -> None:
        """Report a failure while keeping a still-current preview reusable."""

        if version == self._parameter_version:
            self._set_header_state("导出失败", "error")
        self.statusBar().showMessage("导出失败 · 请检查错误详情与目标目录")
        if not self._close_when_finished:
            QMessageBox.critical(self, "导出失败", message)

    def _export_rollback_incomplete(self, message: str, version: int) -> None:
        """Warn explicitly when the destination batch may be partially committed."""

        del version
        # 磁盘恢复风险与预览版本无关；即使用户刚请求关窗，也必须保持窗口并明确告警。
        self._close_when_finished = False
        self._result_version = -1
        self.export_button.setEnabled(False)
        self._set_header_state("导出批次不完整", "error")
        self.statusBar().showMessage("导出失败 · 目标目录可能包含不完整批次，请按错误详情检查")
        QMessageBox.critical(self, "导出批次可能不完整", message)

    def _export_cleanup_incomplete(self, message: str, version: int) -> None:
        """Keep cleanup/recovery details visible even when export did not finish."""

        del version
        # 残留路径属于磁盘状态，不随预览版本变化；关窗请求也不能把它静默吞掉。
        self._close_when_finished = False
        self._result_version = -1
        self.export_button.setEnabled(False)
        self._set_header_state("导出未完成，需检查", "error")
        self.statusBar().showMessage("导出未完成 · 存在需要人工检查的临时或恢复文件")
        QMessageBox.warning(self, "导出未完成，需要检查", message)

    def _export_cancelled(self, _version: int) -> None:
        """Report cooperative cancellation without invalidating the preview."""

        if self._close_when_finished:
            self.statusBar().showMessage("导出已取消，窗口正在关闭")
        else:
            self._set_header_state("导出已取消", "warning")
            self.statusBar().showMessage("导出已安全取消，原有文件保持不变")

    @staticmethod
    def _stylesheet() -> str:
        """返回与工程图表配套的深色仪器工作台样式。"""

        # 所有颜色都引用上方语义常量，避免同一状态在不同控件上出现随机色值。
        return f"""
        QMainWindow {{
            background: {BACKGROUND};
        }}
        QWidget#workspaceRoot {{
            background: {BACKGROUND};
        }}
        QWidget {{
            color: {TEXT};
            font-size: 13px;
        }}
        QLabel {{
            background: transparent;
        }}
        QFrame#header {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {SURFACE}, stop: 0.55 {SURFACE_SUBTLE}, stop: 1 {SURFACE}
            );
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
        QFrame#sidePanel {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 {SURFACE}, stop: 0.68 {SURFACE}, stop: 1 {SURFACE_LOW}
            );
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
        QFrame#workspace {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 {SURFACE}, stop: 0.68 {SURFACE}, stop: 1 {SURFACE_LOW}
            );
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
        QFrame#inspectorPanel {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 {SURFACE}, stop: 0.68 {SURFACE}, stop: 1 {SURFACE_LOW}
            );
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
        QScrollArea#inspectorScroll {{
            background: {SURFACE}; border: none;
        }}
        QScrollArea#inspectorScroll > QWidget > QWidget {{ background: {SURFACE}; }}
        QFrame#inspectorActions {{
            background: {SURFACE};
            border-top: 1px solid {BORDER};
            border-bottom-left-radius: 9px;
            border-bottom-right-radius: 9px;
        }}
        QLabel#brandMark {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 {ACCENT_BRIGHT}, stop: 1 #6D5DFB
            );
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 12px;
            font-size: 15px;
            font-weight: 750;
        }}
        QLabel#eyebrow {{
            color: {TEXT_FAINT};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.2px;
        }}
        QLabel#appTitle {{ font-size: 20px; font-weight: 720; color: {TEXT}; }}
        QLabel#sectionTitle {{ font-size: 16px; font-weight: 680; color: {TEXT}; }}
        QLabel#cardTitle {{ font-size: 14px; font-weight: 600; color: {TEXT}; }}
        QLabel#helperText, QLabel#statusMuted {{ color: {TEXT_MUTED}; font-size: 12px; }}
        QLabel#statusReady {{ color: {RESULT}; font-size: 12px; font-weight: 600; }}
        QLabel#statusWarning {{ color: {WARNING}; font-size: 12px; font-weight: 600; }}
        QLabel#stepBadge {{
            background: rgba(91, 143, 249, 0.14);
            color: {ACCENT_BRIGHT};
            border: 1px solid rgba(91, 143, 249, 0.36);
            border-radius: 8px;
            min-width: 28px;
            min-height: 22px;
            max-width: 28px;
            font-size: 11px;
            font-weight: 750;
        }}
        QLabel#contextPill {{
            background: {SURFACE_RAISED};
            color: {TEXT_MUTED};
            border: 1px solid {BORDER};
            border-radius: 13px;
            padding: 5px 11px;
            min-width: 82px;
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel#statePill {{
            background: {SURFACE_RAISED};
            color: {TEXT};
            border: 1px solid {BORDER_STRONG};
            border-radius: 13px;
            padding: 5px 12px;
            min-width: 86px;
            font-size: 11px;
            font-weight: 700;
        }}
        QLabel#statePill[tone="active"] {{
            background: rgba(91, 143, 249, 0.14);
            color: {ACCENT_BRIGHT};
            border-color: rgba(91, 143, 249, 0.58);
        }}
        QLabel#statePill[tone="success"] {{
            background: rgba(69, 214, 180, 0.11);
            color: {RESULT};
            border-color: rgba(69, 214, 180, 0.42);
        }}
        QLabel#statePill[tone="warning"] {{
            background: rgba(244, 199, 104, 0.10);
            color: {WARNING};
            border-color: rgba(244, 199, 104, 0.42);
        }}
        QLabel#statePill[tone="error"] {{
            background: rgba(255, 122, 134, 0.10);
            color: {ERROR};
            border-color: rgba(255, 122, 134, 0.44);
        }}
        QLabel#infoNote {{
            background: {SURFACE_RAISED}; color: {TEXT_MUTED}; border: 1px solid {BORDER};
            border-radius: 8px; padding: 10px;
        }}
        QLabel#successNote {{
            background: rgba(66, 211, 181, 0.10); color: {RESULT};
            border: 1px solid rgba(66, 211, 181, 0.38); border-radius: 7px; padding: 8px;
        }}
        QLabel#warningNote {{
            background: rgba(244, 199, 104, 0.08); color: {WARNING};
            border: 1px solid rgba(244, 199, 104, 0.32); border-radius: 8px;
            padding: 8px 10px;
        }}
        QFrame#fileCard {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 {SURFACE_SUBTLE}, stop: 1 {SURFACE_SUBTLE_LOW}
            );
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
        QFrame#fileCard:hover {{
            background: {SURFACE_RAISED};
            border-color: {BORDER_STRONG};
        }}
        QFrame#segmentedControl {{
            background: {SURFACE_SUBTLE};
            border: 1px solid {BORDER};
            border-radius: 10px;
        }}
        QWidget#responseField {{
            background: transparent;
            border: none;
        }}
        QGroupBox {{
            background: {SURFACE_SUBTLE};
            border: 1px solid {BORDER};
            border-radius: 11px;
            margin-top: 13px;
            padding: 15px 10px 11px 10px;
            font-weight: 650;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 11px;
            padding: 0 6px;
            color: {TEXT};
        }}
        QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
            background: {SURFACE_RAISED};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 8px;
            min-height: 36px;
            padding: 0 9px;
            selection-background-color: {ACCENT};
        }}
        QLineEdit[readOnly="true"] {{ color: {TEXT_MUTED}; }}
        QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
            border-color: {BORDER_STRONG};
        }}
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
            border: 1px solid {ACCENT_BRIGHT};
        }}
        QCheckBox {{
            color: {TEXT_MUTED};
            spacing: 8px;
            padding: 6px 0;
            min-height: 24px;
        }}
        QCheckBox:hover {{ color: {TEXT}; }}
        QCheckBox:focus {{ color: {ACCENT_BRIGHT}; }}
        QComboBox::drop-down {{ border: none; width: 24px; }}
        QPushButton {{
            border-radius: 8px;
            padding: 7px 12px;
            font-weight: 650;
        }}
        QPushButton:focus {{ border: 1px solid {ACCENT_BRIGHT}; }}
        QPushButton#primaryButton {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {ACCENT}, stop: 1 #6D7DFB
            );
            color: white;
            border: 1px solid {ACCENT_BRIGHT};
        }}
        QPushButton#primaryButton:hover {{ background: {ACCENT_BRIGHT}; }}
        QPushButton#primaryButton:pressed {{ background: #4C7FDF; }}
        QPushButton#secondaryButton {{
            background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
        }}
        QPushButton#secondaryButton:hover {{
            border-color: {BORDER_STRONG}; background: {SURFACE_HOVER};
        }}
        QPushButton#secondaryButton:checked {{
            border-color: {ACCENT}; background: rgba(91, 143, 249, 0.18); color: {TEXT};
        }}
        QPushButton#toolButton {{
            background: transparent;
            color: {TEXT_MUTED};
            border: 1px solid transparent;
            border-radius: 7px;
            padding: 5px;
        }}
        QPushButton#toolButton:hover {{
            background: {SURFACE_HOVER};
            color: {TEXT};
        }}
        QPushButton#toolButton:checked {{
            background: rgba(91, 143, 249, 0.18);
            color: {ACCENT_BRIGHT};
            border-color: rgba(91, 143, 249, 0.42);
        }}
        QPushButton:disabled {{
            color: #66758A; background: #121923; border-color: #202B3A;
        }}
        QTabWidget {{ background: transparent; }}
        QTabWidget::tab-bar {{ alignment: left; }}
        QTabWidget::pane {{
            border: none;
            background: {SURFACE};
            top: 0px;
        }}
        QTabBar {{
            background: transparent;
            border: none;
            border-bottom: 1px solid {BORDER};
        }}
        QTabBar::tab {{
            background: {SURFACE};
            color: {TEXT_MUTED};
            min-height: 22px;
            padding: 9px 14px;
            border: none;
            border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {TEXT};
            background: {SURFACE_SUBTLE};
            border-bottom: 2px solid {ACCENT_BRIGHT};
        }}
        QTabBar::tab:hover {{ color: {TEXT}; background: {SURFACE_RAISED}; }}
        QTabBar::tab:disabled {{ color: #56657A; }}
        QTabBar::tear {{ image: none; background: {SURFACE}; border: none; }}
        QTabBar::scroller {{ width: 42px; background: {SURFACE}; }}
        QTabBar QToolButton {{
            background: rgba(17, 25, 37, 0.94);
            border: 1px solid {BORDER};
            border-radius: 0px;
            padding: 2px;
        }}
        QTabBar QToolButton:hover {{
            background: {SURFACE_HOVER};
            border-color: {BORDER_STRONG};
        }}
        QTabBar QToolButton::left-arrow {{
            image: url("{(ICON_DIRECTORY / "chevron-left.svg").as_posix()}");
        }}
        QTabBar QToolButton::right-arrow {{
            image: url("{(ICON_DIRECTORY / "chevron-right.svg").as_posix()}");
        }}
        QProgressBar {{
            background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 5px;
            min-height: 8px; max-height: 8px; text-align: center;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {ACCENT}, stop: 1 {RESULT}
            );
            border-radius: 4px;
        }}
        QStatusBar {{
            background: {SURFACE};
            color: {TEXT_MUTED};
            border: 1px solid {BORDER};
            border-radius: 8px;
            margin: 0px 6px 6px 6px;
            min-height: 22px;
        }}
        QStatusBar::item {{ border: none; }}
        QSplitter#workspaceSplitter::handle {{ background: {BACKGROUND}; }}
        QSplitter#workspaceSplitter::handle:hover {{ background: {ACCENT}; }}
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 3px 2px;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER_STRONG};
            border-radius: 4px;
            min-height: 32px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
        QToolTip {{
            background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
            padding: 6px;
        }}
        """
