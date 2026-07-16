"""ResponseLab 的 Codex 风格 PySide6 桌面界面。

界面只负责收集明确参数、调度后台任务和展示结果。所有解析与 DSP 都通过公开模块完成，
因此 GUI 不会拥有一套无法测试的“隐藏算法”。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
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

from .dsp import run_compensation
from .io import load_bin_timeseries, load_csv_timeseries
from .models import BinConfig, CompensationRun, CompensationSettings
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
    target_path: Path
    bin_config: BinConfig
    settings: CompensationSettings
    version: int


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
            if self.request.target_path.suffix.lower() == ".bin":
                target = load_bin_timeseries(self.request.target_path, self.request.bin_config)
            else:
                target = load_csv_timeseries(
                    self.request.target_path,
                    time_unit="s",
                    time_column=0,
                    value_columns=(1,),
                )
            result = run_compensation(reference, dut, target, self.request.settings)
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
        self._run: CompensationRun | None = None
        self._worker: AnalysisThread | None = None
        self._parameter_version = 0
        self._result_version = -1
        self._close_when_finished = False
        self._last_frequency_unit = "GHz"
        self._building = True
        self._build_ui()
        self._connect_stale_signals()
        self._building = False
        self.statusBar().showMessage("就绪 · 请选择三份输入文件")

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
        self.metric_label = QLabel("公共频率上限 — · 估计时延 —")
        self.metric_label.setObjectName("helperText")
        reset_button = QPushButton("复位缩放")
        reset_button.setObjectName("secondaryButton")
        reset_button.clicked.connect(self._reset_plots)
        toolbar.addWidget(heading)
        toolbar.addStretch(1)
        toolbar.addWidget(self.metric_label)
        toolbar.addWidget(reset_button)
        layout.addLayout(toolbar)

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
        self.band_low = self._frequency_spin(0.01)
        self.band_high = self._frequency_spin(0.30)
        self.phase_low = self._frequency_spin(0.02)
        self.phase_high = self._frequency_spin(0.25)
        rows = [
            ("补偿起点", self.band_low),
            ("补偿终点", self.band_high),
            ("去斜观察起点", self.phase_low),
            ("去斜观察终点", self.phase_high),
        ]
        for row, (label, widget) in enumerate(rows, start=2):
            compensation_layout.addWidget(QLabel(label), row, 0)
            compensation_layout.addWidget(widget, row, 1)
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
        self.analyze_button = QPushButton("分析并预览")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setMinimumHeight(46)
        self.analyze_button.clicked.connect(self._start_analysis)
        self.export_button = QPushButton("导出补偿结果")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.setMinimumHeight(42)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        action_layout.addWidget(self.progress)
        action_layout.addWidget(self.analyze_button)
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
        spin.setKeyboardTracking(False)
        return spin

    def _connect_stale_signals(self) -> None:
        for card in (self.reference_card, self.dut_card, self.target_card):
            card.path_selected.connect(self._mark_stale)
        for combo in (self.bin_dtype, self.bin_byte_order, self.bin_layout):
            combo.currentIndexChanged.connect(self._mark_stale)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
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
            self.phase_low,
            self.phase_high,
        ):
            spin.valueChanged.connect(self._mark_stale)
        self.bin_channels.valueChanged.connect(
            lambda count: self.bin_channel_index.setMaximum(max(0, count - 1))
        )
        self._mode_changed(self.mode_combo.currentIndex())

    def _band_edges_changed(self, *_args: object) -> None:
        self._mark_stale()

    def _mode_changed(self, _index: int) -> None:
        mode = str(self.mode_combo.currentData())
        phase_enabled = mode != "magnitude"
        self.phase_low.setEnabled(phase_enabled)
        self.phase_high.setEnabled(phase_enabled)
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
        if self._run is not None:
            self._populate_plots(self._run)

    def _target_path_changed(self, path: str) -> None:
        self.bin_group.setVisible(Path(path).suffix.lower() == ".bin")

    def _mark_stale(self, *_args: object) -> None:
        if self._building:
            return
        self._parameter_version += 1
        if self._run is not None:
            self.export_button.setEnabled(False)
            self.header_state.setText("预览已过期")
            self.statusBar().showMessage("参数或输入已变化，请重新分析后再导出")

    def _current_settings(self) -> CompensationSettings:
        factor = FREQUENCY_FACTORS[self.frequency_unit_combo.currentText()]
        band_low_hz = self.band_low.value() * factor
        band_high_hz = self.band_high.value() * factor
        return CompensationSettings(
            mode=str(self.mode_combo.currentData()),
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            phase_fit_low_hz=self.phase_low.value() * factor,
            phase_fit_high_hz=self.phase_high.value() * factor,
            remove_relative_delay=True,
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

    def _start_analysis(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        paths = (self.reference_card.path, self.dut_card.path, self.target_card.path)
        if any(path is None for path in paths):
            QMessageBox.warning(self, "输入不完整", "请先选择参考脉冲、待补偿脉冲和待补偿信号。")
            return
        assert paths[0] is not None and paths[1] is not None and paths[2] is not None
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            QMessageBox.critical(self, "文件不存在", "以下文件无法读取：\n" + "\n".join(missing))
            return
        try:
            is_bin = paths[2].suffix.lower() == ".bin"
            request = AnalysisRequest(
                reference_path=paths[0],
                dut_path=paths[1],
                target_path=paths[2],
                bin_config=(
                    self._bin_config() if is_bin else BinConfig(sample_rate_hz=1.0)
                ),
                settings=self._current_settings(),
                version=self._parameter_version,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self.analyze_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.show()
        self.header_state.setText("分析中")
        self.statusBar().showMessage("正在解析输入并计算频响补偿…")
        self._worker = AnalysisThread(request)
        self._worker.succeeded.connect(self._analysis_succeeded)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _analysis_succeeded(self, result: CompensationRun, version: int) -> None:
        if version != self._parameter_version:
            self.header_state.setText("预览已过期")
            self.statusBar().showMessage("旧分析任务已结束，但参数已变化；结果未用于导出")
            return
        self._result_version = version
        self.present_run(result)
        self.export_button.setEnabled(True)

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
        self.analyze_button.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
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

        self._run = run
        self.header_state.setText("预览有效")
        reference_rate = self._format_frequency(run.reference_pulse.sample_rate_hz)
        dut_rate = self._format_frequency(run.dut_pulse.sample_rate_hz)
        target_rate = self._format_frequency(run.input_signal.sample_rate_hz)
        self.reference_card.set_summary(
            f"{run.reference_pulse.samples:,} 点 · {reference_rate}"
        )
        self.dut_card.set_summary(f"{run.dut_pulse.samples:,} 点 · {dut_rate}")
        self.target_card.set_summary(f"{run.input_signal.samples:,} 点 · {target_rate}")
        common_limit = run.analysis.frequency_hz[-1]
        delay_ps = run.analysis.estimated_dut_delay_s / 1e-12
        common_limit_text = self._format_frequency(common_limit)
        if run.analysis.settings.mode == "magnitude":
            metric_text = f"公共上限 {common_limit_text}"
        else:
            metric_text = f"公共上限 {common_limit_text} · 相对时延 {delay_ps:.3f} ps"
        self.metric_label.setText(metric_text)
        self.result_warning.setText(" · ".join(run.warnings))
        self.result_warning.setVisible(bool(run.warnings))
        self._populate_plots(run)
        label = source_label or "文件分析"
        self.statusBar().showMessage(f"{label}完成 · 频响补偿已应用")

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
    def _set_minimum_y_span(
        plot: pg.PlotWidget,
        values: np.ndarray,
        minimum_span: float,
    ) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high - low < minimum_span:
            center = 0.5 * (low + high)
            plot.setYRange(
                center - 0.5 * minimum_span,
                center + 0.5 * minimum_span,
                padding=0.05,
            )

    def _populate_plots(self, run: CompensationRun) -> None:
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
        magnitude_plot.setLabel("left", "相对公共峰值幅度", units="dB")
        phase_observation_mask = (
            reliable
            & (analysis.frequency_hz >= analysis.settings.phase_fit_low_hz)
            & (analysis.frequency_hz <= analysis.settings.phase_fit_high_hz)
        )
        common_phase_trend = np.zeros_like(analysis.frequency_hz)
        if np.count_nonzero(phase_observation_mask) >= 2:
            fit_frequency = analysis.frequency_hz[phase_observation_mask]
            fit_phase = analysis.reference_phase_rad[phase_observation_mask]
            frequency_center = float(np.mean(fit_frequency))
            phase_center = float(np.mean(fit_phase))
            denominator = float(np.sum((fit_frequency - frequency_center) ** 2))
            if denominator > 0.0:
                slope = float(
                    np.sum(
                        (fit_frequency - frequency_center) * (fit_phase - phase_center)
                    )
                    / denominator
                )
                intercept = phase_center - slope * frequency_center
                common_phase_trend = slope * analysis.frequency_hz + intercept
        self._plot_curve(
            phase_plot,
            frequency,
            np.where(
                reliable,
                np.degrees(analysis.reference_phase_rad - common_phase_trend),
                np.nan,
            ),
            name="参考",
            color=REFERENCE,
        )
        self._plot_curve(
            phase_plot,
            frequency,
            np.where(
                reliable,
                np.degrees(analysis.dut_phase_rad - common_phase_trend),
                np.nan,
            ),
            name="待补偿",
            color=DUT,
            dashed=True,
        )
        phase_plot.setLabel("bottom", "频率", units=frequency_unit)
        phase_plot.setLabel("left", "去共同线性趋势相位", units="°")

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
        self._plot_curve(
            difference_phase,
            frequency,
            np.degrees(analysis.phase_difference_rad),
            name="原始相位差",
            color=DUT,
        )
        if analysis.settings.mode != "magnitude":
            self._plot_curve(
                difference_phase,
                frequency,
                np.where(
                    phase_observation_mask,
                    np.degrees(analysis.phase_trend_rad),
                    np.nan,
                ),
                name="线性趋势",
                color=WARNING,
                dashed=True,
            )
            self._plot_curve(
                difference_phase,
                frequency,
                np.degrees(analysis.delay_removed_phase_rad),
                name="实际补偿相位（去时延）",
                color=RESULT,
            )
        self._add_band_region(
            difference_phase,
            band_low,
            band_high,
            color=ACCENT,
        )
        if analysis.settings.mode != "magnitude":
            self._add_band_region(
                difference_phase,
                phase_low,
                phase_high,
                color=WARNING,
                alpha=16,
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

        output_time, output_time_unit = self._time_display(run.input_signal.time_s)
        waveform_plot, spectrum_plot = self.output_plots
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
        self._set_minimum_y_span(
            difference_magnitude,
            np.where(reliable, analysis.magnitude_difference_db, np.nan),
            1.0,
        )
        self._set_minimum_y_span(
            compensation_phase,
            ideal_phase_display,
            2.0,
        )
        preview_start = 0
        preview_end = min(run.input_signal.samples - 1, 511)
        waveform_plot.setXRange(
            output_time[preview_start],
            output_time[preview_end],
            padding=0.02,
        )

    def _all_plots(self) -> list[pg.PlotWidget]:
        return [
            *self.pulse_plots,
            *self.response_plots,
            *self.difference_plots,
            *self.compensator_plots,
            *self.output_plots,
        ]

    def _frequency_plots(self) -> list[pg.PlotWidget]:
        return [
            *self.response_plots,
            *self.difference_plots,
            *self.compensator_plots[:2],
            self.output_plots[1],
        ]

    def _focus_frequency_plots(self, run: CompensationRun) -> None:
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

    def _reset_plots(self) -> None:
        for plot in self._all_plots():
            plot.enableAutoRange()
        if self._run is not None:
            self._focus_frequency_plots(self._run)

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
