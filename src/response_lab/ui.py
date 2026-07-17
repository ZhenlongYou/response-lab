"""ResponseLab 的 Codex 风格 PySide6 桌面界面。

界面只负责收集明确参数、调度后台任务和展示结果。所有解析与 DSP 都通过公开模块完成，
因此 GUI 不会拥有一套无法测试的“隐藏算法”。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .dsp import (
    compare_pulses,
    fit_linear_phase_slope,
    run_compensation,
    suggest_frequency_settings,
)
from .io import load_bin_timeseries, load_csv_timeseries
from .models import BinConfig, CompensationRun, CompensationSettings, PulseComparison
from .reporting import (
    SourceVerificationError,
    bundle_paths,
    export_run_bundle,
)

BACKGROUND = "#0D0F12"
SURFACE = "#15181D"
SURFACE_RAISED = "#1B1F25"
BORDER = "#2B3038"
TEXT = "#F2F4F7"
TEXT_MUTED = "#AAB1BC"
ACCENT = "#5B8CFF"
REFERENCE = "#6EA8FE"
DUT = "#F0B35A"
RESULT = "#42D3B5"
WARNING = "#FFCA5C"
ERROR = "#FF6B6B"

TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9, "ps": 1e-12}
FREQUENCY_FACTORS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


@dataclass(frozen=True)
class AnalysisRequest:
    reference_path: Path
    dut_path: Path
    target_path: Path | None
    bin_config: BinConfig
    settings: CompensationSettings
    version: int
    action: Literal["compare", "compensate"]
    auto_frequency_bands: bool = False
    auto_phase_fit_band: bool = False


class FileCard(QFrame):
    """带明确角色、可拖放路径和解析状态的单个输入卡片。"""

    path_selected = Signal(str)

    def __init__(self, title: str, file_filter: str) -> None:
        super().__init__()
        self.file_filter = file_filter
        self.setObjectName("fileCard")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("选择或拖入文件")
        self.path_edit.setAccessibleName(f"{title}文件路径")
        browse_button = QPushButton("选择")
        browse_button.setObjectName("secondaryButton")
        browse_button.setMinimumHeight(36)
        browse_button.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_button)
        self.status_label = QLabel("尚未选择")
        self.status_label.setObjectName("statusMuted")
        self.status_label.setWordWrap(True)
        layout.addWidget(title_label)
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


class AnalysisThread(QThread):
    """在后台解析文件并运行频响分析，防止大记录冻结界面。"""

    succeeded = Signal(object, int)
    failed = Signal(str, int)

    def __init__(self, request: AnalysisRequest) -> None:
        super().__init__()
        self.request = request

    def run(self) -> None:
        try:
            reference = load_csv_timeseries(
                self.request.reference_path,
                time_unit="s",
                time_column=0,
                value_columns=(1,),
            )
            dut = load_csv_timeseries(
                self.request.dut_path,
                time_unit="s",
                time_column=0,
                value_columns=(1,),
            )
            target = None
            if self.request.action == "compensate":
                if self.request.target_path is None:
                    raise ValueError("数据补偿需要选择待补偿信号")
                if self.request.target_path.suffix.lower() == ".bin":
                    target = load_bin_timeseries(
                        self.request.target_path,
                        self.request.bin_config,
                    )
                else:
                    target = load_csv_timeseries(
                        self.request.target_path,
                        time_unit="s",
                        time_column=0,
                        value_columns=(1,),
                    )
            settings = self.request.settings
            if self.request.auto_frequency_bands:
                settings = suggest_frequency_settings(
                    reference,
                    dut,
                    settings,
                    maximum_frequency_hz=(target.nyquist_hz if target is not None else None),
                    suggest_phase_fit_band=self.request.auto_phase_fit_band,
                )
            if self.request.action == "compare":
                result = compare_pulses(reference, dut, settings)
            else:
                assert target is not None
                result = run_compensation(reference, dut, target, settings)
            self.succeeded.emit(result, self.request.version)
        except Exception as exc:  # GUI boundary: convert full failure to actionable text.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail, self.request.version)


def _plot_widget() -> pg.PlotWidget:
    plot = pg.PlotWidget(background=SURFACE)
    plot.showGrid(x=True, y=True, alpha=0.16)
    plot.setMouseEnabled(x=True, y=True)
    plot.setClipToView(True)
    plot.setDownsampling(auto=True, mode="peak")
    plot.addLegend(offset=(10, 8), labelTextColor=TEXT_MUTED)
    for axis_name in ("left", "bottom"):
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(BORDER))
        axis.setTextPen(pg.mkPen(TEXT_MUTED))
        axis.enableAutoSIPrefix(False)
    return plot


def _plot_page(count: int) -> tuple[QWidget, list[pg.PlotWidget]]:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    plots = [_plot_widget() for _ in range(count)]
    for plot in plots:
        layout.addWidget(plot, 1)
    return page, plots


class CompactDoubleSpinBox(QDoubleSpinBox):
    """保留输入精度，同时去掉界面上没有信息量的尾随零。"""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        text = super().textFromValue(value)
        decimal_point = self.locale().decimalPoint()
        if decimal_point in text:
            text = text.rstrip("0").rstrip(decimal_point)
        return text


class ResponseLabWindow(QMainWindow):
    """ResponseLab 三栏主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ResponseLab · 频响分析与补偿")
        self.setMinimumSize(1120, 720)
        self.resize(1440, 900)
        self._result: PulseComparison | CompensationRun | None = None
        self._run: CompensationRun | None = None
        self._worker: AnalysisThread | None = None
        self._active_action: Literal["compare", "compensate"] | None = None
        self._parameter_version = 0
        self._result_version = -1
        self._close_when_finished = False
        self._last_frequency_unit = "GHz"
        self._phase_band_is_manual = False
        self._phase_band_initialized = False
        self._building = True
        self._build_ui()
        self._connect_stale_signals()
        self._building = False
        self.statusBar().showMessage("就绪 · 比较只需两份拟合脉冲；数据补偿时再选择第三份信号")

    def _build_ui(self) -> None:
        self.setStyleSheet(self._stylesheet())
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_header())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_visual_workspace())
        splitter.addWidget(self._build_inspector())
        splitter.setSizes([300, 800, 340])
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.statusBar().setSizeGripEnabled(True)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(72)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 12, 22, 12)
        title_column = QVBoxLayout()
        title = QLabel("ResponseLab")
        title.setObjectName("appTitle")
        subtitle = QLabel("频响分析与补偿")
        subtitle.setObjectName("helperText")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        layout.addLayout(title_column)
        layout.addStretch(1)
        self.header_state = QLabel("等待输入")
        self.header_state.setObjectName("statePill")
        self.header_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.header_state)
        return header

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(12)
        section = QLabel("数据与步骤")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)
        self.reference_card = FileCard(
            "01  参考拟合脉冲",
            "CSV (*.csv);;所有文件 (*)",
        )
        self.dut_card = FileCard(
            "02  待补偿拟合脉冲",
            "CSV (*.csv);;所有文件 (*)",
        )
        self.target_card = FileCard(
            "03  待补偿信号",
            "信号 (*.csv *.bin);;所有文件 (*)",
        )
        self.target_card.path_selected.connect(self._target_path_changed)
        layout.addWidget(self.reference_card)
        layout.addWidget(self.dut_card)
        layout.addWidget(self.target_card)
        layout.addStretch(1)
        return panel

    def _build_visual_workspace(self) -> QWidget:
        workspace = QFrame()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        toolbar = QHBoxLayout()
        heading = QLabel("分析工作区")
        heading.setObjectName("sectionTitle")
        self.metric_label = QLabel("分析频带 — · 去斜频带 —")
        self.metric_label.setObjectName("helperText")
        self.metric_label.setToolTip(
            "分析频带是当前实际使用的补偿范围；去斜频带由用户控制，工具在该频带内"
            "拟合相位差的线性项并从相位差中扣除，仅用于比较残余相位。"
        )
        self.zoom_button = QPushButton("放大")
        self.pan_button = QPushButton("拖动")
        self.reset_button = QPushButton("恢复")
        self.plot_mode_group = QButtonGroup(self)
        self.plot_mode_group.setExclusive(True)
        for button in (self.zoom_button, self.pan_button):
            button.setObjectName("secondaryButton")
            button.setCheckable(True)
            self.plot_mode_group.addButton(button)
        self.pan_button.setChecked(True)
        self.reset_button.setObjectName("secondaryButton")
        self.zoom_button.setToolTip("左键拖出矩形区域进行放大")
        self.pan_button.setToolTip("按住左键拖动画布；滚轮可继续缩放")
        self.reset_button.setToolTip("恢复当前数据的推荐显示范围")
        self.zoom_button.clicked.connect(lambda: self._set_plot_mouse_mode("zoom"))
        self.pan_button.clicked.connect(lambda: self._set_plot_mouse_mode("pan"))
        self.reset_button.clicked.connect(self._reset_plots)
        toolbar.addWidget(heading)
        toolbar.addStretch(1)
        toolbar.addWidget(self.metric_label)
        toolbar.addWidget(self.zoom_button)
        toolbar.addWidget(self.pan_button)
        toolbar.addWidget(self.reset_button)
        layout.addLayout(toolbar)
        self.band_legend_label = QLabel(
            "蓝色阴影：分析/候选补偿频带　橙色虚线：去斜频带边界"
        )
        self.band_legend_label.setObjectName("helperText")
        self.band_legend_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.band_legend_label)

        self.visual_tabs = QTabWidget()
        pulse_page, self.pulse_plots = _plot_page(1)
        response_page, self.response_plots = _plot_page(2)
        difference_page, self.difference_plots = _plot_page(2)
        compensator_page, self.compensator_plots = _plot_page(2)
        output_page, self.output_plots = _plot_page(2)
        self.visual_tabs.addTab(pulse_page, "拟合脉冲")
        self.visual_tabs.addTab(response_page, "频率响应")
        self.visual_tabs.addTab(difference_page, "频响差异比较")
        self.visual_tabs.addTab(compensator_page, "频响补偿")
        self.visual_tabs.addTab(output_page, "输出预览")
        self._set_plot_mouse_mode("pan")
        layout.addWidget(self.visual_tabs, 1)
        self.result_warning = QLabel()
        self.result_warning.setObjectName("warningNote")
        self.result_warning.setWordWrap(True)
        self.result_warning.hide()
        layout.addWidget(self.result_warning)
        return workspace

    def _build_inspector(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("inspectorPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 16, 14, 18)
        layout.setSpacing(12)
        title = QLabel("参数检查器")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.bin_group = QGroupBox("BIN 导入设置")
        bin_layout = QGridLayout(self.bin_group)
        self.bin_sample_rate = QDoubleSpinBox()
        self.bin_sample_rate.setRange(0.0, 1.0e15)
        self.bin_sample_rate.setDecimals(0)
        self.bin_sample_rate.setValue(0.0)
        self.bin_sample_rate.setSpecialValueText("请输入")
        self.bin_dtype = QComboBox()
        self.bin_dtype.addItems(["float32", "float64", "int16", "int32"])
        self.bin_byte_order = QComboBox()
        self.bin_byte_order.addItems(["little", "big"])
        self.bin_channels = QSpinBox()
        self.bin_channels.setRange(1, 128)
        self.bin_channel_index = QSpinBox()
        self.bin_channel_index.setRange(0, 0)
        self.bin_layout = QComboBox()
        self.bin_layout.addItems(["interleaved", "planar"])
        self.bin_offset_bytes = QSpinBox()
        self.bin_offset_bytes.setRange(0, 2_147_483_647)
        self.bin_scale = QDoubleSpinBox()
        self.bin_scale.setRange(-1.0e12, 1.0e12)
        self.bin_scale.setDecimals(9)
        self.bin_scale.setValue(1.0)
        self.bin_value_offset = QDoubleSpinBox()
        self.bin_value_offset.setRange(-1.0e12, 1.0e12)
        self.bin_value_offset.setDecimals(9)
        bin_layout.addWidget(QLabel("采样率 (Hz)"), 0, 0)
        bin_layout.addWidget(self.bin_sample_rate, 0, 1)
        bin_layout.addWidget(QLabel("数据类型"), 1, 0)
        bin_layout.addWidget(self.bin_dtype, 1, 1)
        bin_layout.addWidget(QLabel("字节序"), 2, 0)
        bin_layout.addWidget(self.bin_byte_order, 2, 1)
        bin_layout.addWidget(QLabel("通道数"), 3, 0)
        bin_layout.addWidget(self.bin_channels, 3, 1)
        bin_layout.addWidget(QLabel("目标通道 (0 起)"), 4, 0)
        bin_layout.addWidget(self.bin_channel_index, 4, 1)
        bin_layout.addWidget(QLabel("排列"), 5, 0)
        bin_layout.addWidget(self.bin_layout, 5, 1)
        bin_layout.addWidget(QLabel("文件头偏移 (字节)"), 6, 0)
        bin_layout.addWidget(self.bin_offset_bytes, 6, 1)
        bin_layout.addWidget(QLabel("幅值缩放"), 7, 0)
        bin_layout.addWidget(self.bin_scale, 7, 1)
        bin_layout.addWidget(QLabel("幅值偏置"), 8, 0)
        bin_layout.addWidget(self.bin_value_offset, 8, 1)
        layout.addWidget(self.bin_group)

        compensation_group = QGroupBox("补偿设置")
        compensation_layout = QGridLayout(compensation_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("幅频 + 相频", "both")
        self.mode_combo.addItem("仅幅频", "magnitude")
        self.mode_combo.addItem("仅相频", "phase")
        self.frequency_unit_combo = QComboBox()
        self.frequency_unit_combo.addItems(list(FREQUENCY_FACTORS))
        self.frequency_unit_combo.setCurrentText("GHz")
        compensation_layout.addWidget(QLabel("补偿模式"), 0, 0)
        compensation_layout.addWidget(self.mode_combo, 0, 1)
        compensation_layout.addWidget(QLabel("频率单位"), 1, 0)
        compensation_layout.addWidget(self.frequency_unit_combo, 1, 1)
        self.auto_frequency_bands = QCheckBox("根据拟合脉冲自动设置补偿频带")
        self.auto_frequency_bands.setChecked(True)
        self.auto_frequency_bands.setToolTip(
            "按各自峰值归一化，选择共同 -20 dB 最长连续谱宽候选；输入需已去基线"
        )
        compensation_layout.addWidget(self.auto_frequency_bands, 2, 0, 1, 2)
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
            ("去斜频带起点", self.phase_low),
            ("去斜频带终点", self.phase_high),
        ]
        for row, (label, widget) in enumerate(rows, start=3):
            compensation_layout.addWidget(QLabel(label), row, 0)
            compensation_layout.addWidget(widget, row, 1)
        auto_hint = QLabel(
            "自动模式只调整补偿频带。去斜频带首次可自动给出初值；手动输入后保持不变。"
        )
        auto_hint.setObjectName("helperText")
        auto_hint.setWordWrap(True)
        compensation_layout.addWidget(auto_hint, 7, 0, 1, 2)
        layout.addWidget(compensation_group)

        layout.addStretch(1)
        scroll.setWidget(content)
        panel_layout.addWidget(scroll, 1)

        action_bar = QFrame()
        action_bar.setObjectName("inspectorActions")
        action_layout = QVBoxLayout(action_bar)
        action_layout.setContentsMargins(14, 10, 14, 12)
        action_layout.setSpacing(8)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.compare_button = QPushButton("拟合脉冲比较")
        self.compare_button.setObjectName("secondaryButton")
        self.compare_button.setMinimumHeight(42)
        self.compare_button.clicked.connect(self._start_comparison)
        self.compensate_button = QPushButton("数据补偿")
        self.compensate_button.setObjectName("primaryButton")
        self.compensate_button.setMinimumHeight(46)
        self.compensate_button.clicked.connect(self._start_compensation)
        # 保留旧属性，避免外部自动化脚本在一次版本升级中失效。
        self.analyze_button = self.compensate_button
        self.export_button = QPushButton("导出补偿结果")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.setMinimumHeight(42)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        action_layout.addWidget(self.progress)
        action_layout.addWidget(self.compare_button)
        action_layout.addWidget(self.compensate_button)
        action_layout.addWidget(self.export_button)
        panel_layout.addWidget(action_bar)
        self.bin_group.setVisible(False)
        return panel

    @staticmethod
    def _frequency_spin(value: float) -> QDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(0.0, 1.0e15)
        # GHz 显示下仍保留到 1 mHz，切换单位不会把低频设置静默量化为 0。
        spin.setDecimals(12)
        spin.setValue(value)
        spin.setSuffix(" GHz")
        spin.setKeyboardTracking(False)
        return spin

    def _connect_stale_signals(self) -> None:
        for card in (self.reference_card, self.dut_card):
            card.path_selected.connect(self._mark_stale)
        for combo in (self.bin_dtype, self.bin_byte_order, self.bin_layout):
            combo.currentIndexChanged.connect(self._mark_compensation_input_stale)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.auto_frequency_bands.toggled.connect(self._automatic_frequency_bands_changed)
        self.frequency_unit_combo.currentTextChanged.connect(self._frequency_unit_changed)
        self.band_low.valueChanged.connect(self._band_edges_changed)
        self.band_high.valueChanged.connect(self._band_edges_changed)
        for spin in (
            self.bin_sample_rate,
            self.bin_channels,
            self.bin_channel_index,
            self.bin_offset_bytes,
            self.bin_scale,
            self.bin_value_offset,
        ):
            spin.valueChanged.connect(self._mark_compensation_input_stale)
        for spin in (self.phase_low, self.phase_high):
            spin.valueChanged.connect(self._phase_band_changed)
        self.bin_channels.valueChanged.connect(
            lambda count: self.bin_channel_index.setMaximum(max(0, count - 1))
        )
        self._mode_changed(self.mode_combo.currentIndex())
        self._automatic_frequency_bands_changed(True)

    def _band_edges_changed(self, *_args: object) -> None:
        self._mark_stale()

    def _mode_changed(self, _index: int) -> None:
        mode = str(self.mode_combo.currentData())
        phase_enabled = mode != "magnitude"
        self.phase_low.setEnabled(phase_enabled)
        self.phase_high.setEnabled(phase_enabled)
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
        physical_before_hz = [
            spin.value() * FREQUENCY_FACTORS[old_unit] for spin in spins
        ]
        conversion = FREQUENCY_FACTORS[old_unit] / FREQUENCY_FACTORS[new_unit]
        for spin in spins:
            previous = spin.blockSignals(True)
            spin.setValue(spin.value() * conversion)
            spin.setSuffix(f" {new_unit}")
            spin.blockSignals(previous)
        self._last_frequency_unit = new_unit
        physical_after_hz = [
            spin.value() * FREQUENCY_FACTORS[new_unit] for spin in spins
        ]
        quantized = any(
            not np.isclose(before, after, rtol=1.0e-12, atol=1.0e-12)
            for before, after in zip(physical_before_hz, physical_after_hz, strict=True)
        )
        if quantized:
            self._mark_stale()
        if self._result is not None:
            self._populate_plots(self._result)

    def _target_path_changed(self, path: str) -> None:
        self.bin_group.setVisible(Path(path).suffix.lower() == ".bin")
        self._mark_compensation_input_stale()

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
        if self._result is not None:
            self.export_button.setEnabled(False)
            self.header_state.setText("预览已过期")
            self.statusBar().showMessage("参数或输入已变化，请重新分析后再导出")

    def _current_settings(self) -> CompensationSettings:
        mode = str(self.mode_combo.currentData())
        factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
        if self.auto_frequency_bands.isChecked():
            use_initialized_phase_band = (
                mode != "magnitude" and self._phase_band_initialized
            )
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
                detrend_phase=True,
                analysis_points=16385,
            )
        band_low_hz = self.band_low.value() * factor
        band_high_hz = self.band_high.value() * factor
        return CompensationSettings(
            mode=mode,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            phase_fit_low_hz=self.phase_low.value() * factor,
            phase_fit_high_hz=self.phase_high.value() * factor,
            detrend_phase=True,
            analysis_points=16385,
        )

    def _bin_config(self) -> BinConfig:
        return BinConfig(
            sample_rate_hz=self.bin_sample_rate.value(),
            dtype=self.bin_dtype.currentText(),
            byte_order=self.bin_byte_order.currentText(),
            offset_bytes=self.bin_offset_bytes.value(),
            channels=self.bin_channels.value(),
            channel_index=self.bin_channel_index.value(),
            layout=self.bin_layout.currentText(),
            scale=self.bin_scale.value(),
            value_offset=self.bin_value_offset.value(),
        )

    def _start_comparison(self) -> None:
        self._start_task("compare")

    def _start_compensation(self) -> None:
        self._start_task("compensate")

    def _start_analysis(self) -> None:
        """兼容旧的自动化入口；等价于点击“数据补偿”。"""

        self._start_compensation()

    def _start_task(self, action: Literal["compare", "compensate"]) -> None:
        if self._worker is not None and self._worker.isRunning():
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
            is_bin = target_path is not None and target_path.suffix.lower() == ".bin"
            request = AnalysisRequest(
                reference_path=reference_path,
                dut_path=dut_path,
                target_path=target_path,
                bin_config=(
                    self._bin_config() if is_bin else BinConfig(sample_rate_hz=1.0)
                ),
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
        self.export_button.setEnabled(False)
        self.progress.show()
        self._active_action = action
        self.header_state.setText("比较中" if action == "compare" else "补偿中")
        self.statusBar().showMessage(
            "正在比较两份拟合脉冲…"
            if action == "compare"
            else "正在解析输入并执行数据补偿…"
        )
        self._worker = AnalysisThread(request)
        self._worker.succeeded.connect(self._analysis_succeeded)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _analysis_succeeded(
        self, result: PulseComparison | CompensationRun, version: int
    ) -> None:
        if version != self._parameter_version:
            self.header_state.setText("预览已过期")
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
        if version == self._parameter_version:
            self.header_state.setText("分析失败")
            self.statusBar().showMessage("分析失败 · 请按提示修正输入或参数")
            QMessageBox.critical(self, "无法完成分析", message)
        else:
            self.header_state.setText("参数已变化")
            self.statusBar().showMessage("旧分析任务失败后已结束；请按当前参数重新分析")

    def _worker_finished(self) -> None:
        self.progress.hide()
        self.compare_button.setEnabled(True)
        self.compensate_button.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._active_action = None
        if self._close_when_finished:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._worker is not None and self._worker.isRunning():
            self._close_when_finished = True
            self.header_state.setText("完成后关闭")
            self.statusBar().showMessage("分析正在安全收尾，完成后窗口将自动关闭")
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
        self.header_state.setText("预览有效" if is_compensation else "比较有效")
        reference_rate = self._format_frequency(result.reference_pulse.sample_rate_hz)
        dut_rate = self._format_frequency(result.dut_pulse.sample_rate_hz)
        self.reference_card.set_summary(
            f"{result.reference_pulse.samples:,} 点 · {reference_rate}"
        )
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
            metric_text = f"分析频带 {analysis_range} · 去斜频带 {phase_range}"
        self.metric_label.setText(metric_text)
        self.result_warning.setText(" · ".join(result.warnings))
        self.result_warning.setVisible(bool(result.warnings))
        blue_description = "实际补偿频带" if is_compensation else "分析/候选补偿频带"
        legend_text = f"蓝色阴影：{blue_description}"
        if result.analysis.settings.mode != "magnitude":
            legend_text += "　橙色虚线：去斜频带边界"
        self.band_legend_label.setText(legend_text)
        self.visual_tabs.setTabEnabled(4, is_compensation)
        self._populate_plots(result)
        label = source_label or ("文件补偿" if is_compensation else "拟合脉冲比较")
        suffix = "频响补偿已应用" if is_compensation else "未读取或改写待补偿数据"
        self.statusBar().showMessage(f"{label}完成 · {suffix}")

    def _show_effective_frequency_settings(self, settings: CompensationSettings) -> None:
        factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
        pairs = [
            (self.band_low, settings.band_low_hz / factor),
            (self.band_high, settings.band_high_hz / factor),
        ]
        initialize_phase_band = (
            settings.mode != "magnitude" and not self._phase_band_initialized
        )
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
            (low, "去斜频带起点"),
            (high, "去斜频带终点"),
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
            candidates = np.flatnonzero(
                np.isclose(segment_score, maximum, rtol=1.0e-12, atol=0.0)
            )
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
        analysis = run.analysis
        frequency, frequency_unit = self._frequency_display(analysis.frequency_hz)
        frequency_factor = FREQUENCY_FACTORS[frequency_unit]
        band_low = analysis.settings.band_low_hz / frequency_factor
        band_high = analysis.settings.band_high_hz / frequency_factor
        phase_low = analysis.settings.phase_fit_low_hz / frequency_factor
        phase_high = analysis.settings.phase_fit_high_hz / frequency_factor
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
        common_peak_db = max(
            float(np.max(analysis.reference_magnitude_db)),
            float(np.max(analysis.dut_magnitude_db)),
        )
        reference_magnitude_display = np.where(
            reliable,
            analysis.reference_magnitude_db - common_peak_db,
            np.nan,
        )
        dut_magnitude_display = np.where(
            reliable,
            analysis.dut_magnitude_db - common_peak_db,
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
        phase_observation_mask = (
            reliable
            & (analysis.frequency_hz >= analysis.settings.phase_fit_low_hz)
            & (analysis.frequency_hz <= analysis.settings.phase_fit_high_hz)
        )
        reference_phase_trend = np.zeros_like(analysis.frequency_hz)
        dut_phase_trend = np.zeros_like(analysis.frequency_hz)
        if (
            analysis.settings.mode != "magnitude"
            and np.count_nonzero(phase_observation_mask) >= 3
        ):
            reference_peak_db = float(np.max(analysis.reference_magnitude_db))
            dut_peak_db = float(np.max(analysis.dut_magnitude_db))
            joint_weights = np.minimum(
                np.exp(
                    (analysis.reference_magnitude_db - reference_peak_db)
                    * np.log(10.0)
                    / 10.0
                ),
                np.exp(
                    (analysis.dut_magnitude_db - dut_peak_db)
                    * np.log(10.0)
                    / 10.0
                ),
            )
            try:
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
            except ValueError:
                pass
            else:
                reference_phase_trend = reference_slope * analysis.frequency_hz
                dut_phase_trend = dut_slope * analysis.frequency_hz
        phase_is_detrended = analysis.settings.mode != "magnitude"
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
            name="相位差（去斜前）",
            color=DUT,
        )
        if analysis.settings.mode != "magnitude":
            self._plot_curve(
                difference_phase,
                frequency,
                np.degrees(analysis.phase_after_optional_detrend_rad),
                name="相位差（去斜后）",
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
            output_time, output_time_unit = self._time_display(run.input_signal.time_s)
            self._plot_curve(
                waveform_plot,
                output_time,
                run.input_signal.values[:, 0],
                name="补偿前",
                color=DUT,
                dashed=True,
            )
            self._plot_curve(
                waveform_plot,
                output_time,
                run.output_values[:, 0],
                name="补偿后",
                color=RESULT,
            )
            waveform_plot.setLabel("bottom", "时间", units=output_time_unit)
            waveform_plot.setLabel("left", "幅值")
            input_spectrum = np.fft.rfft(run.input_signal.values[:, 0])
            output_spectrum = np.fft.rfft(run.output_values[:, 0])
            signal_frequency_hz = np.fft.rfftfreq(
                run.input_signal.samples, d=1.0 / run.input_signal.sample_rate_hz
            )
            signal_frequency, signal_unit = self._frequency_display(signal_frequency_hz)
            floor = np.finfo(np.float64).tiny
            self._plot_curve(
                spectrum_plot,
                signal_frequency,
                20.0 * np.log10(np.maximum(np.abs(input_spectrum), floor)),
                name="补偿前",
                color=DUT,
                dashed=True,
            )
            self._plot_curve(
                spectrum_plot,
                signal_frequency,
                20.0 * np.log10(np.maximum(np.abs(output_spectrum), floor)),
                name="补偿后",
                color=RESULT,
            )
            spectrum_plot.setLabel("bottom", "频率", units=signal_unit)
            spectrum_plot.setLabel("left", "原始 DFT 幅度", units="dB")
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

    def _frequency_plots(self) -> list[pg.PlotWidget]:
        return [
            *self.response_plots,
            *self.difference_plots,
            *self.compensator_plots[:2],
            self.output_plots[1],
        ]

    def _focus_frequency_plots(self, run: PulseComparison | CompensationRun) -> None:
        settings = run.analysis.settings
        relevant_high_hz = settings.band_high_hz
        if settings.mode != "magnitude":
            relevant_high_hz = max(relevant_high_hz, settings.phase_fit_high_hz)
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
        output_time, _ = self._time_display(run.input_signal.time_s)
        preview_end = min(run.input_signal.samples - 1, 511)
        self.output_plots[0].setXRange(
            output_time[0],
            output_time[preview_end],
            padding=0.02,
        )

    def _apply_recommended_y_spans(self, run: PulseComparison | CompensationRun) -> None:
        del run  # 曲线已经包含所有显示变换，直接按实际绘制数据定范围。
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

    def _export(self) -> None:
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
            existing = [
                path for path in paths.as_tuple() if path.exists() or path.is_symlink()
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
            paths = export_run_bundle(self._run, paths.output, allow_overwrite=True)
        except SourceVerificationError as exc:
            self._result_version = -1
            self.export_button.setEnabled(False)
            self.header_state.setText("源文件已变化")
            self.statusBar().showMessage("输入文件已变化 · 请重新分析后再导出")
            QMessageBox.critical(self, "需要重新分析", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出：{paths.output}")
        QMessageBox.information(
            self,
            "导出完成",
            f"补偿结果：{paths.output}\n"
            f"响应诊断：{paths.response_csv}\n"
            f"参数记录：{paths.manifest}",
        )

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        QMainWindow, QWidget {{
            background: {BACKGROUND}; color: {TEXT}; font-size: 13px;
        }}
        QFrame#header {{
            background: {SURFACE}; border-bottom: 1px solid {BORDER};
        }}
        QFrame#sidePanel {{
            background: {SURFACE}; border-right: 1px solid {BORDER};
        }}
        QFrame#workspace {{ background: {BACKGROUND}; }}
        QFrame#inspectorPanel {{
            background: {SURFACE}; border-left: 1px solid {BORDER};
        }}
        QScrollArea#inspectorScroll {{
            background: {SURFACE}; border: none;
        }}
        QScrollArea#inspectorScroll > QWidget > QWidget {{ background: {SURFACE}; }}
        QFrame#inspectorActions {{
            background: {SURFACE}; border-top: 1px solid {BORDER};
        }}
        QLabel#appTitle {{ font-size: 22px; font-weight: 700; color: {TEXT}; }}
        QLabel#sectionTitle {{ font-size: 16px; font-weight: 650; color: {TEXT}; }}
        QLabel#cardTitle {{ font-size: 14px; font-weight: 600; color: {TEXT}; }}
        QLabel#helperText, QLabel#statusMuted {{ color: {TEXT_MUTED}; font-size: 12px; }}
        QLabel#statusReady {{ color: {RESULT}; font-size: 12px; }}
        QLabel#statusWarning {{ color: {WARNING}; font-size: 12px; }}
        QLabel#statePill {{
            background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
            border-radius: 13px; padding: 5px 12px; min-width: 88px;
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
            background: rgba(255, 202, 92, 0.08); color: {WARNING};
            border: 1px solid rgba(255, 202, 92, 0.32); border-radius: 7px;
            padding: 7px 10px;
        }}
        QFrame#fileCard {{
            background: {BACKGROUND}; border: 1px solid {BORDER}; border-radius: 10px;
        }}
        QFrame#fileCard:hover {{ border-color: #46505E; }}
        QGroupBox {{
            background: {BACKGROUND}; border: 1px solid {BORDER}; border-radius: 10px;
            margin-top: 12px; padding: 14px 10px 10px 10px; font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 12px; padding: 0 5px; color: {TEXT};
        }}
        QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
            background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
            border-radius: 7px; min-height: 34px; padding: 0 8px;
            selection-background-color: {ACCENT};
        }}
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
            border: 2px solid {ACCENT};
        }}
        QCheckBox {{ color: {TEXT_MUTED}; spacing: 8px; padding: 5px 0; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; }}
        QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid #76A0FF; }}
        QComboBox::drop-down {{ border: none; width: 24px; }}
        QPushButton {{ border-radius: 7px; padding: 7px 12px; font-weight: 600; }}
        QPushButton#primaryButton {{
            background: {ACCENT}; color: white; border: 1px solid #76A0FF;
        }}
        QPushButton#primaryButton:hover {{ background: #6A98FF; }}
        QPushButton#primaryButton:pressed {{ background: #4E7FEF; }}
        QPushButton#secondaryButton {{
            background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
        }}
        QPushButton#secondaryButton:hover {{
            border-color: {ACCENT}; background: #202631;
        }}
        QPushButton#secondaryButton:checked {{
            border-color: {ACCENT}; background: #263A66; color: {TEXT};
        }}
        QPushButton:disabled {{
            color: #68707C; background: #171A1F; border-color: #252A31;
        }}
        QTabWidget::pane {{
            border: 1px solid {BORDER}; border-radius: 9px;
            background: {SURFACE}; top: -1px;
        }}
        QTabBar::tab {{
            background: transparent; color: {TEXT_MUTED}; padding: 10px 14px;
            border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
        QTabBar::tab:hover {{ color: {TEXT}; background: {SURFACE_RAISED}; }}
        QProgressBar {{
            background: {SURFACE_RAISED}; border: 1px solid {BORDER}; border-radius: 5px;
            min-height: 8px; max-height: 8px; text-align: center;
        }}
        QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
        QStatusBar {{
            background: {SURFACE}; color: {TEXT_MUTED}; border-top: 1px solid {BORDER};
        }}
        QSplitter::handle {{ background: {BORDER}; width: 1px; }}
        QToolTip {{
            background: {SURFACE_RAISED}; color: {TEXT}; border: 1px solid {BORDER};
            padding: 5px;
        }}
        """
