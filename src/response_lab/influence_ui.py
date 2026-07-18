"""“影响频段”页签的独立 PySide6 展示组件。

本模块只负责收集页面参数、发出分析请求和展示已计算结果，不导入扫描算法，
因此主窗口可以在后台线程完成计算后再把轻量展示数据交给页面。
"""

# 逐项中文导入说明会打断 Ruff 的自动排序分组，本文件仅关闭对应格式告警。
# ruff: noqa: I001

# 延迟解析类型标注，使页面可以引用 Qt 控件和映射类型而不增加运行期导入耦合。
from __future__ import annotations

# Mapping 允许展示 API 接收普通字典或只读映射，而不绑定扫描算法类。
from collections.abc import Mapping, Sequence
# Path 让文件选择结果保持跨平台路径语义，而不是在页面中拼接字符串。
from pathlib import Path

# NumPy 负责把后台结果转换为有限的一维曲线和二维眼图轨迹。
import numpy as np
# PyQtGraph 提供与现有 ResponseLab 一致的深色工程曲线和可缩放坐标轴。
import pyqtgraph as pg
# Signal 是页面与主窗口之间的轻量请求边界，QSignalBlocker 保证单位换算不会伪装成参数修改。
from PySide6.QtCore import QSignalBlocker, Qt, Signal
# QResizeEvent 让页面按真实页签宽度切换横向或纵向对比布局。
from PySide6.QtGui import QResizeEvent
# Qt 控件组成顶部参数、Vpp 文件入口、三幅眼图和候选结果区域。
from PySide6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# 页面沿用主工作台的近黑画布，独立文件中保留局部常量以避免与 ui.py 循环导入。
_BACKGROUND = "#080C12"
# 次级表面承载参数条和文件入口，不改变现有主窗口的批准配色。
_SURFACE = "#111925"
# 抬升表面用于输入框和列表，使控件从近黑背景中保持可辨识。
_SURFACE_RAISED = "#172130"
# 低对比边框只用于分区，不制造发光或宽阴影。
_BORDER = "#263448"
# 更强边框用于悬停反馈和坐标轴线。
_BORDER_STRONG = "#344760"
# 主文字在深色背景上保持足够对比。
_TEXT = "#F4F7FB"
# 次级文字用于轴刻度和字段名，避免与数据曲线争夺视觉层级。
_TEXT_MUTED = "#9EACC0"
# 蓝、橙、青分别表示幅度、相位和幅相结果。
_REFERENCE = "#72A7FF"
# 橙色对应相位影响曲线，与现有 DUT 对比色保持一致。
_DUT = "#F2B763"
# 青色对应幅相联合影响曲线和补偿后角色。
_RESULT = "#45D6B4"
# 一幅眼图最多叠加 600 条轨迹，与后台抽样上限一致。
_MAX_EYE_TRACES = 600
# 主光标位置使用低亮度蓝灰虚线，不与三个数据角色竞争。
_EYE_CENTER_LINE = "#52647C"
# 影响页只在显示层使用工程单位，所有后台请求仍统一传递 Hz。
_FREQUENCY_FACTORS = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}
# 保留旧页面 0.1 MHz 的物理下限，切换单位不会静默放宽算法输入域。
_BAND_WIDTH_MIN_HZ = 100_000.0
# 旧页面 1,000,000 MHz 上限等价于 1 THz。
_BAND_WIDTH_MAX_HZ = 1.0e12
# 各单位使用等价的 10 MHz 步进，键盘或步进按钮的物理增量保持一致。
_BAND_WIDTH_STEP_HZ = 10.0e6


# QDoubleSpinBox 默认固定补齐小数零；紧凑格式避免单位拆分后数值框显得冗长。
class _CompactDoubleSpinBox(QDoubleSpinBox):
    """保留高精度数值，同时隐藏无意义的末尾零。"""

    # Qt 绘制和编辑都会使用此公开格式入口，数值解析仍由基类负责。
    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        # 先按控件精度格式化，再只删除小数末尾的零和孤立小数点。
        return f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")


# 单个数据路径行把角色、只读路径和文件按钮组织为一个可复用控件。
class _DataPathRow(QWidget):
    """显示一份 Vpp 数据路径，并允许用户从文件对话框选择。"""

    # 路径真正变化时通知页面使旧分析结果失效。
    path_changed = Signal()

    # 构造时固定角色名称，后续读取路径只通过公开 path 属性完成。
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        # 初始化 QWidget 父类，使本行能安全加入任意 Qt 布局。
        super().__init__(parent)
        # 保存角色名称，用于文件对话框和辅助功能文字。
        self._title = title
        # 横向布局让字段名、路径和按钮保持一行，减少页面纵向占用。
        layout = QHBoxLayout(self)
        # 子控件自身不再增加外边距，由父级 Vpp 面板统一控制留白。
        layout.setContentsMargins(0, 0, 0, 0)
        # 8 px 间距保持与现有输入卡一致的紧凑节奏。
        layout.setSpacing(8)
        # 角色标签只显示用户确认的简洁业务名称。
        title_label = QLabel(title)
        # 固定最小宽度让两行路径输入在视觉上对齐。
        title_label.setMinimumWidth(64)
        # 只读路径避免用户键入无法验证的半截路径。
        self.path_edit = QLineEdit()
        # 明确禁止直接编辑，文件对话框仍可完整更新路径。
        self.path_edit.setReadOnly(True)
        # 空路径使用简短占位文字，不添加格式说明。
        self.path_edit.setPlaceholderText("选择文件")
        # 读屏名称包含该行角色，避免两个只读输入无法区分。
        self.path_edit.setAccessibleName(f"{title}路径")
        # 文件按钮沿用项目现有的“选择”动作词。
        self.choose_button = QPushButton("选择")
        # 对象名复用主界面的次级按钮样式语义。
        self.choose_button.setObjectName("secondaryButton")
        # 读屏名称包含数据角色，两个“选择”按钮不再混淆。
        self.choose_button.setAccessibleName(f"选择{title}文件")
        # 点击后打开本行自己的数据文件选择器。
        self.choose_button.clicked.connect(self._choose_file)
        # 标签不伸展，路径输入吸收主要横向空间。
        layout.addWidget(title_label)
        # 路径字段使用 stretch=1，窄窗口优先压缩文字而不是按钮。
        layout.addWidget(self.path_edit, 1)
        # 选择按钮保持可发现的固定操作区域。
        layout.addWidget(self.choose_button)

    # path 属性把空文本转换为 None，让请求对象能明确表达缺少输入。
    @property
    def path(self) -> Path | None:
        # 去掉路径两端空白，防止只包含空格的文本被当作文件。
        text = self.path_edit.text().strip()
        # 有内容时返回 Path，否则返回 None 交给上层统一验证。
        return Path(text) if text else None

    # 测试、主窗口恢复会话和文件对话框都通过同一入口设置路径。
    def set_path(self, path: str | Path) -> None:
        # Path 规范化分隔符后再转回可显示文字。
        path_text = str(Path(path))
        # 相同路径不重复发射失效信号。
        if path_text == self.path_edit.text():
            # 已有工具提示和文本无需再写入。
            return
        # 更新只读输入框中的完整路径。
        self.path_edit.setText(path_text)
        # 工具提示保留完整路径，字段被压缩时仍可查看。
        self.path_edit.setToolTip(path_text)
        # 页面收到信号后清理与旧路径对应的结果。
        self.path_changed.emit()

    # 文件选择器只负责路径，不在 GUI 主线程中读取大数据。
    def _choose_file(self) -> None:
        # CSV 和 BIN 都是现有工具支持的原始数据入口。
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            f"选择{self._title}",
            "",
            "数据文件 (*.csv *.bin);;所有文件 (*)",
        )
        # 用户取消时保持原路径不变，避免清空已经有效的选择。
        if path:
            # 选中路径后通过统一入口更新文本和工具提示。
            self.set_path(path)


# 创建统一深色 PlotWidget，保证新页签与已有频响页面具有相同交互质感。
def _plot_widget(*, minimum_height: int = 120) -> pg.PlotWidget:
    # 近黑背景把眼图轨迹和影响曲线作为主要视觉对象。
    plot = pg.PlotWidget(background=_BACKGROUND)
    # 页面在最小窗口仍保留可读高度，外层分隔器可以继续分配额外空间。
    plot.setMinimumHeight(minimum_height)
    # 低透明网格帮助读取 UI 和频率位置，不遮挡主要数据。
    plot.showGrid(x=True, y=True, alpha=0.11)
    # 平移和缩放保持 PyQtGraph 原生交互，主窗口也可统一修改鼠标模式。
    plot.setMouseEnabled(x=True, y=True)
    # 视窗外数据不参与绘制，长扫描结果不会拖慢交互。
    plot.setClipToView(True)
    # 大数组自动按峰值方式降采样，保留窄频段的局部极值。
    plot.setDownsampling(auto=True, mode="peak")
    # 左轴和底轴使用同一低亮度主题，避免局部出现系统默认白色坐标轴。
    for axis_name in ("left", "bottom"):
        # 取得一条真实坐标轴，随后设置轴线和文字颜色。
        axis = plot.getAxis(axis_name)
        # 轴线使用比分区边框稍强的蓝灰色。
        axis.setPen(pg.mkPen(_BORDER_STRONG))
        # 刻度文字使用次级颜色，不与曲线图例混淆。
        axis.setTextPen(pg.mkPen(_TEXT_MUTED))
        # 页面自行选择 GHz 或 UI，关闭自动 SI 前缀避免重复单位。
        axis.enableAutoSIPrefix(False)
    # 返回完成主题配置的绘图控件。
    return plot


# 将标签放在控件上方，使三个眼参数在窄页中仍能紧凑排列且保持清晰分组。
def _parameter_field(
    title: str,
    control: QWidget,
    *,
    minimum_width: int,
    maximum_width: int,
) -> QWidget:
    """创建不横向拉伸的参数字段，并统一标签与输入控件间距。"""

    # 独立容器让布局只在字段之间分配间距，不把标签和相邻控件混成一组。
    field = QWidget()
    # 稳定名称便于局部样式保持透明背景。
    field.setObjectName("influenceParameterField")
    # 标签和输入垂直排列可减少同一字段的横向占用。
    layout = QVBoxLayout(field)
    # 字段外部间距由参数行统一管理，内部不重复增加留白。
    layout.setContentsMargins(0, 0, 0, 0)
    # 3 px 只分隔字段标题与输入，不制造松散的纵向空白。
    layout.setSpacing(3)
    # 使用用户确认的简短参数名，不在界面增加算法说明。
    label = QLabel(title)
    # 稳定名称让字段标题可以统一使用紧凑字号。
    label.setObjectName("influenceFieldLabel")
    # 最小宽度保证窄页仍能读取当前值和操作步进按钮。
    control.setMinimumWidth(minimum_width)
    # 最大宽度防止宽页把输入框拉成与参数值无关的大块区域。
    control.setMaximumWidth(maximum_width)
    # 标题位于输入上方，阅读顺序与字段顺序一致。
    layout.addWidget(label)
    # 输入控件使用字段自己的有限宽度。
    layout.addWidget(control)
    # 返回完整字段供公共参数行或眼参数行复用。
    return field


# 独立页面通过一个信号把当前控件快照交给主窗口的后台调度层。
class InfluenceBandPage(QWidget):
    """收集影响频段参数，并展示候选曲线与补偿前后对比图。"""

    # 信号载荷是轻量字典，页面不依赖尚未固定的扫描结果模型。
    analysis_requested = Signal(object)
    # 任一有效请求参数变化时，主窗可用该信号使后台版本失效。
    request_changed = Signal()
    # 候选索引对应扫描结果原顺序，不在页面内持有算法对象。
    candidate_selected = Signal(int)

    # 构造完整页面骨架，但不自动启动任何分析或文件读取。
    def __init__(self, parent: QWidget | None = None) -> None:
        # 初始化 QWidget，使页面可以直接加入主窗口 QTabWidget。
        super().__init__(parent)
        # 稳定对象名方便主窗口、自动化和样式表定位新页签。
        self.setObjectName("influenceBandPage")
        # 页面级样式只作用于本页，不改动主窗口已有样式表。
        self.setStyleSheet(self._stylesheet())
        # 外层只承载滚动视窗，使主窗口最小高度下仍能访问全部三眼图和候选。
        page_layout = QVBoxLayout(self)
        # 滚动视窗自己贴合页签边缘，内容区域再负责业务留白。
        page_layout.setContentsMargins(0, 0, 0, 0)
        # 单一子控件无需额外布局间距。
        page_layout.setSpacing(0)
        # 内容宽度跟随页签，窄页只出现纵向滚动而不横向截断文字。
        self.content_scroll = QScrollArea()
        # 去除系统默认边框，保持与现有中央画布一致。
        self.content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # 页面内容自动使用视口宽度，响应式布局负责重新排列控件。
        self.content_scroll.setWidgetResizable(True)
        # 横向滚动会隐藏字段与候选关系，因此始终关闭。
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # 纵向滚动只在紧凑窗口内容确实放不下时出现。
        self.content_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # 透明视口不改变现有近黑工作台背景。
        self.content_scroll.viewport().setAutoFillBackground(False)
        # 独立内容控件承载原有四个业务区域。
        content = QWidget()
        # 稳定名称供局部样式表消除滚动区默认底色。
        content.setObjectName("influenceContent")
        # 把内容交给滚动视窗管理尺寸与滚动范围。
        self.content_scroll.setWidget(content)
        # 滚动视窗填满整个页签。
        page_layout.addWidget(self.content_scroll)
        # 根业务布局按“参数、条件输入、眼图、候选结果”从上到下排列。
        root_layout = QVBoxLayout(content)
        # 与现有绘图页保持 8 px 左右边距和 12 px 顶部呼吸空间。
        root_layout.setContentsMargins(8, 12, 8, 8)
        # 各区域使用 8 px 间距，保持紧凑工程工作台比例。
        root_layout.setSpacing(8)

        # 顶部参数条始终可见，指标变化只控制其内部眼参数组。
        controls_panel = QFrame()
        # 对象名让局部样式表提供深色卡片表面。
        controls_panel.setObjectName("influenceControls")
        # 响应式布局在宽页使用单行，在窄页保留可操作的两行回退。
        self.controls_layout = QVBoxLayout(controls_panel)
        # 参数条内部使用 10 px 水平留白，避免控件贴住边框。
        self.controls_layout.setContentsMargins(10, 8, 10, 8)
        # 仅在窄页回退成两行时使用 7 px 紧凑间隔。
        self.controls_layout.setSpacing(7)
        # 公共扫描参数、眼参数和主操作在宽页共同使用这一行。
        self.primary_controls_layout = QHBoxLayout()
        # 首行子布局不再增加外边距。
        self.primary_controls_layout.setContentsMargins(0, 0, 0, 0)
        # 字段组之间使用 10 px，显著大于字段内部的 3 px 间距。
        self.primary_controls_layout.setSpacing(10)
        # 指标下拉框按用户确认顺序提供三种互斥选择。
        self.metric_combo = QComboBox()
        # Vpp 数据值供调度层识别，同时可见文字严格保持简洁。
        self.metric_combo.addItem("Vpp", "vpp")
        # 眼高使用稳定英文数据值，避免业务逻辑依赖中文显示文字。
        self.metric_combo.addItem("眼高", "eye_height")
        # 眼宽同样使用独立数据值。
        self.metric_combo.addItem("眼宽", "eye_width")
        # 为自动化和读屏提供明确字段语义。
        self.metric_combo.setAccessibleName("分析指标")
        # 指标作为紧凑字段加入首行，不随页签宽度无限拉伸。
        self.primary_controls_layout.addWidget(
            _parameter_field(
                "指标",
                self.metric_combo,
                minimum_width=72,
                maximum_width=104,
            ),
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        # 频段宽度对 Vpp、眼高和眼宽三种指标都生效。
        self.band_width_spin = _CompactDoubleSpinBox()
        # 九位小数足以在 GHz 显示下保持 1 Hz 分辨率，紧凑格式会隐藏末尾零。
        self.band_width_spin.setDecimals(9)
        # 初始范围以默认 MHz 表示，单位切换时会按同一物理上下限重设。
        self.band_width_spin.setRange(
            _BAND_WIDTH_MIN_HZ / _FREQUENCY_FACTORS["MHz"],
            _BAND_WIDTH_MAX_HZ / _FREQUENCY_FACTORS["MHz"],
        )
        # 默认步进等价于 10 MHz，切换单位后仍保持同一物理增量。
        self.band_width_spin.setSingleStep(
            _BAND_WIDTH_STEP_HZ / _FREQUENCY_FACTORS["MHz"]
        )
        # 默认保持原有 100 MHz 扫描核心宽度。
        self.band_width_spin.setValue(100.0)
        # 物理 Hz 值是单位切换的唯一数据源，避免较大单位显示舍入后累计误差。
        self._band_width_hz = 100.0e6
        # 读屏名称包含当前默认单位语义，切换后会同步更新。
        self.band_width_spin.setAccessibleName("频段宽度 MHz")
        # 独立单位下拉框让用户按数据量级选择 Hz、kHz、MHz 或 GHz。
        self.band_width_unit_combo = QComboBox()
        # 单位顺序与主补偿设置一致，避免两个页面形成不同心智模型。
        self.band_width_unit_combo.addItems(list(_FREQUENCY_FACTORS))
        # 默认仍用 MHz，保持现有 100 MHz 参数和用户习惯。
        self.band_width_unit_combo.setCurrentText("MHz")
        # 保存上一次显示单位，切换时先从旧单位还原真实 Hz 值。
        self._band_width_unit = "MHz"
        # 读屏明确说明此下拉只控制频段宽度的显示与输入单位。
        self.band_width_unit_combo.setAccessibleName("频段宽度单位")
        # 单位框保持紧凑宽度，不让两个到三个字符占据过多参数栏空间。
        self.band_width_unit_combo.setMinimumWidth(64)
        # 最大宽度限制单位控件在宽页中被布局拉伸。
        self.band_width_unit_combo.setMaximumWidth(72)
        # 数值和单位共同组成一个业务字段，标题只出现一次。
        band_width_control = QWidget()
        # 稳定对象名允许局部样式保持组合容器透明。
        band_width_control.setObjectName("bandWidthControl")
        # 横向布局让单位紧贴数值右侧，同时保留清晰点击边界。
        band_width_layout = QHBoxLayout(band_width_control)
        # 容器不增加额外外边距，与其他参数字段对齐。
        band_width_layout.setContentsMargins(0, 0, 0, 0)
        # 6 px 间距足以区分两个输入，又不会扩大分区间隔。
        band_width_layout.setSpacing(6)
        # 数值框吸收字段中的主要宽度。
        band_width_layout.addWidget(self.band_width_spin, 1)
        # 单位框只使用自身紧凑宽度。
        band_width_layout.addWidget(self.band_width_unit_combo)
        # 公共频段字段紧随指标，保持从“测什么”到“扫多宽”的阅读顺序。
        self.primary_controls_layout.addWidget(
            _parameter_field(
                "频段宽度",
                band_width_control,
                minimum_width=176,
                maximum_width=208,
            ),
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        # 主按钮使用已确认的动作名称。
        self.start_button = QPushButton("开始分析")
        # 主操作对象名使用本页局部主按钮样式。
        self.start_button.setObjectName("primaryButton")
        # 最小高度保持鼠标和键盘操作的可发现性。
        self.start_button.setMinimumHeight(40)
        # 限制按钮宽度，避免宽页把参数区视觉重心推得过远。
        self.start_button.setMaximumWidth(112)
        # 点击只发出参数快照，耗时分析由主窗口线程负责。
        self.start_button.clicked.connect(self._emit_analysis_request)
        # 主操作与字段输入底部对齐，首行视觉基线保持稳定。
        self.primary_controls_layout.addWidget(
            self.start_button,
            0,
            Qt.AlignmentFlag.AlignBottom,
        )

        # 眼参数作为一个整体显隐，Vpp 模式不会留下孤立标签。
        self.eye_parameters_panel = QWidget()
        # 眼参数内部使用横向字段组，按调制和 M 的业务顺序紧凑排列。
        eye_parameters_layout = QHBoxLayout(self.eye_parameters_panel)
        # 子面板不增加额外边距，和外层控件共享同一基线。
        eye_parameters_layout.setContentsMargins(0, 0, 0, 0)
        # 10 px 字段间距与首行一致，同时大于标签到输入的内部间距。
        eye_parameters_layout.setSpacing(10)
        # 调制下拉只包含用户确认的 NRZ 和 PAM4。
        self.modulation_combo = QComboBox()
        # NRZ 可见文字与内部值分别服务用户和算法。
        self.modulation_combo.addItem("NRZ", "nrz")
        # PAM4 保持行业通用写法，不添加括号说明。
        self.modulation_combo.addItem("PAM4", "pam4")
        # 读屏名称明确这是调制格式而非补偿模式。
        self.modulation_combo.setAccessibleName("调制格式")
        # 调制是眼图参数组的第一项。
        eye_parameters_layout.addWidget(
            _parameter_field(
                "调制",
                self.modulation_combo,
                minimum_width=72,
                maximum_width=88,
            ),
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        # M 表示每 UI 样点数，采样率仍由拟合脉冲时间轴计算。
        # M 至少为三，避免端点保护裁掉 M=2 的全部理想 crossing。
        self.m_spin = QSpinBox()
        # 与核心合同一致，至少每 UI 三点才能给眼宽提供有效内侧线段。
        self.m_spin.setRange(3, 1_000_000)
        # 当前演示数据常用 M=32。
        self.m_spin.setValue(32)
        # 读屏名称保持简洁参数标识。
        self.m_spin.setAccessibleName("M")
        # M 紧随调制，先确定虚拟眼的 UI 采样网格。
        eye_parameters_layout.addWidget(
            _parameter_field(
                "M",
                self.m_spin,
                minimum_width=64,
                maximum_width=80,
            ),
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        # 剩余空间全部留在参数组右侧，两个字段不会被平均拉散。
        eye_parameters_layout.addStretch(1)
        # 初始先按窄页安全结构加入，首次尺寸事件会在宽页移到首行。
        self.controls_layout.addLayout(self.primary_controls_layout)
        # 调制和 M 在窄页使用第二行，避免输入框被强行压窄。
        self.controls_layout.addWidget(self.eye_parameters_panel)
        # 公共字段和眼参数之后保留弹性空间，把主操作稳定推到右侧。
        self.primary_controls_layout.insertStretch(2, 1)
        # 参数条作为页面第一层加入根布局。
        root_layout.addWidget(controls_panel)

        # Vpp 专属区域包含参考数据与 DUT 数据两份路径。
        self.vpp_paths_panel = QFrame()
        # 对象名应用与参数条一致的深色卡片表面。
        self.vpp_paths_panel.setObjectName("vppPathsPanel")
        # 两份路径纵向排列，长文件名不会挤压成不可读的窄字段。
        vpp_layout = QVBoxLayout(self.vpp_paths_panel)
        # 文件区域保留 10 px 水平和 8 px 垂直边距。
        vpp_layout.setContentsMargins(10, 8, 10, 8)
        # 两行之间保持 7 px 间距。
        vpp_layout.setSpacing(7)
        # 参考数据行用于计算目标 Vpp 标量。
        self.reference_data_row = _DataPathRow("参考数据")
        # DUT 数据行用于应用候选补偿并计算补偿前后 Vpp。
        self.dut_data_row = _DataPathRow("DUT数据")
        # 参考路径放在第一行，符合比较阅读顺序。
        vpp_layout.addWidget(self.reference_data_row)
        # DUT 路径紧随其后。
        vpp_layout.addWidget(self.dut_data_row)
        # Vpp 文件区域加入根布局。
        root_layout.addWidget(self.vpp_paths_panel)

        # Vpp 专属波形区在同一坐标上展示参考、补偿前和补偿后。
        self.vpp_waveform_panel = QFrame()
        # 稳定对象名便于主窗和自动化定位波形区。
        self.vpp_waveform_panel.setObjectName("vppWaveformPanel")
        # 单图占满整个面板，不额外增加标题占用高度。
        vpp_waveform_layout = QVBoxLayout(self.vpp_waveform_panel)
        # 绘图外边距交由页面根布局统一控制。
        vpp_waveform_layout.setContentsMargins(0, 0, 0, 0)
        # 创建紧凑深色波形图。
        self.vpp_waveform_plot = _plot_widget(minimum_height=110)
        # 输入协议使用秒，页面统一换成 ns 显示。
        self.vpp_waveform_plot.setLabel("bottom", "时间", units="ns")
        # 纵轴保留原波形幅值单位语义。
        self.vpp_waveform_plot.setLabel("left", "幅值")
        # 读屏名称明确该图是 Vpp 原始波形对比。
        self.vpp_waveform_plot.setAccessibleName("Vpp 波形对比")
        # 波形图填满面板。
        vpp_waveform_layout.addWidget(self.vpp_waveform_plot)
        # Vpp 模式时波形图获得主要纵向空间。
        root_layout.addWidget(self.vpp_waveform_panel, 2)

        # 三幅眼图区域只在眼高或眼宽模式显示。
        self.eye_plots_panel = QFrame()
        # 对象名使测试和主窗口能稳定定位整个对比区域。
        self.eye_plots_panel.setObjectName("eyePlotsPanel")
        # 三列横向并排，让用户直接比较参考、补偿前和补偿后。
        self.eye_plots_layout = QHBoxLayout(self.eye_plots_panel)
        # 绘图区域不再增加额外边距，充分利用中央画布。
        self.eye_plots_layout.setContentsMargins(0, 0, 0, 0)
        # 三幅图之间使用 8 px 间距，与其他分区一致。
        self.eye_plots_layout.setSpacing(8)
        # 第一列标题和图表严格使用“参考”。
        reference_column, self.reference_plot = self._eye_plot_column("参考")
        # 第二列严格使用“补偿前”。
        before_column, self.before_plot = self._eye_plot_column("补偿前")
        # 第三列严格使用“补偿后”。
        after_column, self.after_plot = self._eye_plot_column("补偿后")
        # 补偿前图的水平轴跟随参考图，交互缩放后仍可直接比较。
        self.before_plot.setXLink(self.reference_plot)
        # 补偿前图的幅值轴也使用同一参考视窗。
        self.before_plot.setYLink(self.reference_plot)
        # 补偿后图使用相同的时间轴链接。
        self.after_plot.setXLink(self.reference_plot)
        # 补偿后图的幅值轴同样跟随参考图。
        self.after_plot.setYLink(self.reference_plot)
        # 三幅图的读屏名称与可见标题一一对应。
        self.reference_plot.setAccessibleName("参考眼图")
        # 补偿前图与参考图可被读屏独立识别。
        self.before_plot.setAccessibleName("补偿前眼图")
        # 补偿后图同样有独立读屏名称。
        self.after_plot.setAccessibleName("补偿后眼图")
        # 三列平均分配可用宽度，避免某一幅图因标题长度获得更多空间。
        self.eye_plots_layout.addWidget(reference_column, 1)
        # 补偿前列使用同等伸展权重，保证与参考图可直接比较。
        self.eye_plots_layout.addWidget(before_column, 1)
        # 补偿后列同样使用等宽布局，不通过尺寸变化暗示改善。
        self.eye_plots_layout.addWidget(after_column, 1)
        # 眼图区域加入根布局并获得主要纵向伸展空间。
        root_layout.addWidget(self.eye_plots_panel, 2)

        # 下部结果分隔器同时承载候选影响曲线和可点击候选列表。
        self.result_splitter = QSplitter(Qt.Orientation.Horizontal)
        # 稳定对象名支持主窗口保存或测试检查分栏状态。
        self.result_splitter.setObjectName("influenceResultSplitter")
        # 影响曲线使用统一深色绘图工厂。
        self.impact_plot = _plot_widget(minimum_height=110)
        # 横轴固定显示 GHz，实际输入仍以 Hz 传入 render_result。
        self.impact_plot.setLabel("bottom", "频率", units="GHz")
        # 纵轴使用通用“改善量”，不绑定尚未固定的算法量纲。
        self.impact_plot.setLabel("left", "改善量")
        # 读屏名称与候选列表区分。
        self.impact_plot.setAccessibleName("候选影响曲线")
        # 右侧容器同时承载当前候选摘要和候选列表。
        candidate_panel = QWidget()
        # 纵向布局让摘要位于列表上方。
        candidate_layout = QVBoxLayout(candidate_panel)
        # 分栏自身已提供间距，右侧容器不再增外边距。
        candidate_layout.setContentsMargins(0, 0, 0, 0)
        # 摘要与列表之间保持 6 px。
        candidate_layout.setSpacing(6)
        # 摘要由主窗格式化当前频段值，页面不解析算法模型。
        self.selection_summary = QLabel()
        # 长摘要在右侧分栏内换行。
        self.selection_summary.setWordWrap(True)
        # 空摘要不占据列表空间。
        self.selection_summary.hide()
        # 诊断只在存在不可解析候选时显示，不与当前候选摘要混用。
        self.diagnostic_label = QLabel()
        # 长诊断在窄候选栏中自动换行。
        self.diagnostic_label.setWordWrap(True)
        # 读屏名称明确该文字描述曲线断点而非推荐结果。
        self.diagnostic_label.setAccessibleName("候选解析状态")
        # 默认没有诊断时不占据列表高度。
        self.diagnostic_label.hide()
        # 候选列表独立显示推荐频段或诊断模式名称。
        self.candidate_list = QListWidget()
        # 对象名供主窗口连接候选点击信号。
        self.candidate_list.setObjectName("candidateList")
        # 空列表提示保持简短，不制造尚未计算的默认推荐。
        self.candidate_list.setAccessibleName("候选频段")
        # 候选文字在窄布局使用整栏宽度和工具提示，不出现难以操作的横向滚动条。
        self.candidate_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # 摘要先加入右侧容器。
        candidate_layout.addWidget(self.selection_summary)
        # 解析诊断位于摘要下方、候选列表上方。
        candidate_layout.addWidget(self.diagnostic_label)
        # 候选列表占据右侧剩余高度。
        candidate_layout.addWidget(self.candidate_list, 1)
        # 曲线区域占据下部分隔器主要宽度。
        self.result_splitter.addWidget(self.impact_plot)
        # 列表作为右侧紧凑选择区。
        self.result_splitter.addWidget(candidate_panel)
        # 默认约四分之三空间分配给曲线。
        self.result_splitter.setSizes([720, 240])
        # 下部结果区域加入根布局。
        root_layout.addWidget(self.result_splitter, 1)

        # 指标变化统一经过一个显隐函数，避免 Vpp 与眼图模式状态漂移。
        self.metric_combo.currentIndexChanged.connect(self._update_metric_visibility)
        # 指标改变后清空旧结果并通知主窗递增请求版本。
        self.metric_combo.currentIndexChanged.connect(self._invalidate_request)
        # 用户修改显示数值时先更新物理 Hz，再让旧结果失效。
        self.band_width_spin.valueChanged.connect(self._band_width_value_changed)
        # 单位变化只做等值显示换算，不让物理参数和旧分析结果失效。
        self.band_width_unit_combo.currentTextChanged.connect(
            self._band_width_unit_changed
        )
        # 调制格式是眼图请求的有效条件。
        self.modulation_combo.currentIndexChanged.connect(self._invalidate_request)
        # M 改变后取样网格随之变化。
        self.m_spin.valueChanged.connect(self._invalidate_request)
        # 参考原始数据路径更新使 Vpp 结果失效。
        self.reference_data_row.path_changed.connect(self._invalidate_request)
        # DUT 路径也是 Vpp 请求的有效条件。
        self.dut_data_row.path_changed.connect(self._invalidate_request)
        # 列表行变化通过过滤槽只发出非负索引。
        self.candidate_list.currentRowChanged.connect(self._emit_candidate_selected)
        # 构造完成后立即应用默认 Vpp 状态。
        self._update_metric_visibility()

    # 页签宽度变化时重新排列三眼图和结果分栏，避免最小窗口把刻度压在一起。
    def resizeEvent(self, event: QResizeEvent) -> None:
        """宽页横向对比，窄页纵向堆叠并交给滚动视窗访问全部内容。"""

        # 先让 QWidget 更新自身和滚动视口几何。
        super().resizeEvent(event)
        # 使用事件中的真实宽度决定布局，而不是依赖启动时的默认尺寸。
        self._apply_compact_layout(event.size().width())

    # 640 px 以下每幅眼图若仍三列并排就不足 200 px，改为单列可读宽度。
    def _apply_compact_layout(self, width: int) -> None:
        """按页签宽度切换对比区域方向，并保留同一组图表对象和坐标链接。"""

        # 640 px 以上足够容纳四个字段和主按钮，直接取消第二排参数。
        stack_parameters = int(width) < 640
        # 查找眼参数当前是否已经位于公共参数行中。
        eye_parameters_in_primary_row = (
            self.primary_controls_layout.indexOf(self.eye_parameters_panel) >= 0
        )
        # 窄页把眼参数移回第二行，保持每个输入框的最低可操作宽度。
        if stack_parameters and eye_parameters_in_primary_row:
            # 先从首行移除，控件对象及其已输入数值都保持不变。
            self.primary_controls_layout.removeWidget(self.eye_parameters_panel)
            # 外层纵向布局把眼参数作为紧凑第二行重新接入。
            self.controls_layout.addWidget(self.eye_parameters_panel)
        # 宽页把整组眼参数插入频段宽度之后、弹性空间之前。
        elif not stack_parameters and not eye_parameters_in_primary_row:
            # 从第二行移除后，外层布局会自动收回对应高度和行间距。
            self.controls_layout.removeWidget(self.eye_parameters_panel)
            # 索引二位于指标、频段宽度之后，并让字段顶部使用同一基线。
            self.primary_controls_layout.insertWidget(
                2,
                self.eye_parameters_panel,
                0,
                Qt.AlignmentFlag.AlignTop,
            )

        # 640 px 阈值让三列宽布局中的每幅图至少接近 200 px。
        compact = int(width) < 640
        # 窄页把参考、补偿前、补偿后三图从上到下排列，每幅使用完整视口宽度。
        eye_direction = (
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        # 方向未变化时 setDirection 是幂等操作，不重建任何轨迹曲线。
        self.eye_plots_layout.setDirection(eye_direction)
        # 下方影响曲线和候选列表在窄页同样改为上下排列，文字不再挤成窄列。
        splitter_orientation = (
            Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
        )
        # QSplitter 保留两个子控件状态，只改变分隔条方向。
        self.result_splitter.setOrientation(splitter_orientation)
        # 为两种方向提供可操作的初始比例，用户仍可拖动分隔条。
        self.result_splitter.setSizes([260, 180] if compact else [720, 240])

    # 生成一列标题和 PlotWidget，三幅图共享完全相同的结构。
    @staticmethod
    def _eye_plot_column(title: str) -> tuple[QWidget, pg.PlotWidget]:
        # 列容器把标题与其下方图表绑定在同一布局中。
        column = QWidget()
        # 纵向布局让标题始终位于图表上方。
        layout = QVBoxLayout(column)
        # 子列不增加额外边距，三列间距由父布局统一提供。
        layout.setContentsMargins(0, 0, 0, 0)
        # 标题与图表之间保留 5 px 间距。
        layout.setSpacing(5)
        # 标题文字严格使用调用方给定的业务名称。
        title_label = QLabel(title)
        # 对象名便于一次查找三幅图标题并验证顺序。
        title_label.setObjectName("eyePlotTitle")
        # 标题居中，形成清晰的三列对比关系。
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 创建可缩放的深色眼图画布。
        plot = _plot_widget(minimum_height=100)
        # 横轴用 UI 表示相对符号相位。
        plot.setLabel("bottom", "时间", units="UI")
        # 纵轴保留通用幅值名称，实际电压或归一化由结果数据决定。
        plot.setLabel("left", "幅值")
        # 标题先加入布局。
        layout.addWidget(title_label)
        # 图表吸收本列剩余纵向空间。
        layout.addWidget(plot, 1)
        # 返回列容器和公开 PlotWidget 供结果渲染与测试使用。
        return column, plot

    # 数值框是真实用户输入入口，变化后立即刷新独立 Hz 数据源。
    def _band_width_value_changed(self, value: float) -> None:
        """按当前工程单位保存新的物理频段宽度。"""

        # 使用已确认的当前单位换算；组合框非可编辑，不会产生未知单位。
        self._band_width_hz = value * _FREQUENCY_FACTORS[self._band_width_unit]
        # 物理宽度变化会改变候选边界，因此清空旧结果并通知主窗口。
        self._invalidate_request()

    # 单位切换只改变显示数值，物理频段宽度、候选结果和后台合同保持不变。
    def _band_width_unit_changed(self, new_unit: str) -> None:
        """把当前频段宽度等值换算到新工程单位。"""

        # 防御未知文字和重复信号，避免用缺失因子换算或重复缩放。
        if new_unit not in _FREQUENCY_FACTORS or new_unit == self._band_width_unit:
            # 没有有效单位变化时保持现状。
            return
        # 后续读取统一以新单位解释数值框。
        self._band_width_unit = new_unit
        # 换算期间阻止 valueChanged，纯显示变化不应让现有分析结果过期。
        blocker = QSignalBlocker(self.band_width_spin)
        # 新单位仍使用旧页面等价的 100 kHz 至 1 THz 物理范围。
        self.band_width_spin.setRange(
            _BAND_WIDTH_MIN_HZ / _FREQUENCY_FACTORS[new_unit],
            _BAND_WIDTH_MAX_HZ / _FREQUENCY_FACTORS[new_unit],
        )
        # 步进也按新单位换算，键盘增减始终相当于 10 MHz。
        self.band_width_spin.setSingleStep(
            _BAND_WIDTH_STEP_HZ / _FREQUENCY_FACTORS[new_unit]
        )
        # 最后写入等价显示值，九位精度可在 GHz 下保留到 1 Hz。
        self.band_width_spin.setValue(
            self._band_width_hz / _FREQUENCY_FACTORS[new_unit]
        )
        # 显式释放信号阻塞器，后续真实用户输入继续发出失效信号。
        del blocker
        # 读屏名称同步当前单位，不要求用户同时读取旁边的组合框。
        self.band_width_spin.setAccessibleName(f"频段宽度 {new_unit}")

    # 返回当前参数快照，后台算法可以独立验证频段宽度、路径与 M。
    def current_request(self) -> dict[str, object]:
        # 当前指标决定哪一组控件是真正生效的输入。
        is_vpp = self.metric_combo.currentData() == "vpp"
        # 字典字段使用稳定英文键；隐藏字段明确置空，避免旧值污染后台任务。
        return {
            "metric": self.metric_combo.currentData(),
            "band_width_hz": self._band_width_hz,
            "modulation": None if is_vpp else self.modulation_combo.currentData(),
            "m": None if is_vpp else self.m_spin.value(),
            "reference_data_path": self.reference_data_row.path if is_vpp else None,
            "dut_data_path": self.dut_data_row.path if is_vpp else None,
        }

    # 主窗口或会话恢复可直接设置两份 Vpp 路径，不需要模拟文件对话框。
    def set_vpp_paths(self, reference_path: str | Path, dut_path: str | Path) -> None:
        # 通过路径行公开入口设置参考数据。
        self.reference_data_row.set_path(reference_path)
        # 通过同一入口设置 DUT 数据。
        self.dut_data_row.set_path(dut_path)

    # 把后台计算得到的轻量字典渲染为曲线、候选列表和可选三幅眼图。
    def render_result(self, result: Mapping[str, object]) -> None:
        """展示一次影响频段结果；输入只需满足本页定义的轻量映射协议。"""

        # 显式拒绝非映射对象，避免属性错误泄漏到 Qt 事件循环。
        if not isinstance(result, Mapping):
            # 调用方能直接看到页面 API 的输入类型要求。
            raise TypeError("影响频段展示结果必须是映射")
        # 先复制并验证全部新数据，任一错误都不破坏上一份有效结果。
        frequency_hz = np.array(result.get("frequency_hz"), dtype=np.float64, copy=True)
        # 影响曲线要求一维、非空且全部有限，防止 PyQtGraph 静默断线。
        if (
            frequency_hz.ndim != 1
            or frequency_hz.size == 0
            or not np.all(np.isfinite(frequency_hz))
        ):
            # 错误信息直接指出页面展示合同，而不猜测算法原因。
            raise ValueError("frequency_hz 必须是一维非空有限数组")
        # scores 应按 magnitude、phase、both 三个稳定键提供改善量。
        scores = result.get("scores")
        # 非映射 scores 无法可靠匹配三种模式。
        if not isinstance(scores, Mapping):
            # 明确提示缺少三模式映射。
            raise ValueError("scores 必须提供幅度、相位和幅相三种模式")
        # 三模式的键、可见名称和颜色固定成同一顺序。
        mode_styles = (
            ("magnitude", "幅度", _REFERENCE),
            ("phase", "相位", _DUT),
            ("both", "幅相", _RESULT),
        )
        # 可选有效掩码明确区分 NaN 断点和真实零分。
        valid_masks = result.get("valid_masks")
        # 提供掩码时必须按三种模式命名。
        if valid_masks is not None and not isinstance(valid_masks, Mapping):
            # 非映射无法与模式得分逐项配对。
            raise ValueError("valid_masks 必须提供幅度、相位和幅相三种模式")
        # 已验证得分保存为页面内部快照。
        prepared_scores: list[tuple[str, str, np.ndarray]] = []
        # 每种模式的最终掩码用于核对诊断计数。
        prepared_masks: list[np.ndarray] = []
        # 每次循环完整验证一种物理补偿模式。
        for mode_key, visible_name, color in mode_styles:
            # 复制成 float64 一维数组，后台修改原数组不会回写界面。
            values = np.array(scores.get(mode_key), dtype=np.float64, copy=True)
            # 每种得分必须与频率轴等长；NaN 是断点，Inf 仍属于非法数值。
            if values.shape != frequency_hz.shape or np.any(np.isinf(values)):
                # 具体模式名帮助调用方定位错误数据。
                raise ValueError(
                    f"{mode_key} 得分必须与 frequency_hz 等长且不得包含 Inf"
                )
            # 未提供显式掩码时从有限位置推导，兼容页面的轻量调用协议。
            if valid_masks is None:
                # 有限零值会得到 True，NaN 则得到 False。
                mode_mask = np.isfinite(values)
            # 后台提供掩码时严格验证其数据类型与有限值位置，拒绝把缺失点伪装成零改善。
            else:
                # 保留原始 dtype，避免整数 0/1 被静默接受为布尔证据。
                raw_mask = np.asarray(valid_masks.get(mode_key))
                # 掩码必须与频率轴同形且确实使用布尔类型。
                if raw_mask.shape != frequency_hz.shape or raw_mask.dtype.kind != "b":
                    # 错误模式名帮助定位后台协议问题。
                    raise ValueError(
                        f"valid_masks.{mode_key} 必须是与 frequency_hz 等长的布尔数组"
                    )
                # 复制掩码，避免后台修改已显示状态。
                mode_mask = np.array(raw_mask, dtype=np.bool_, copy=True)
                # True 必须对应有限分数，False 必须对应 NaN 断点。
                if not np.array_equal(mode_mask, np.isfinite(values)):
                    # 拒绝掩码与数值互相矛盾的整份新结果。
                    raise ValueError(f"valid_masks.{mode_key} 与得分有效位置不一致")
            # 验证成功后才加入中间列表。
            prepared_scores.append((visible_name, color, values))
            # 同步保留该模式掩码供无效计数使用。
            prepared_masks.append(mode_mask)

        # 页面从已验证掩码计算不可解析的模式-频段点数。
        derived_invalid_count = int(
            sum(np.count_nonzero(~mask) for mask in prepared_masks)
        )
        # 后台可显式携带计数，缺失时页面直接使用推导值。
        raw_invalid_count = result.get("invalid_count", derived_invalid_count)
        # 布尔值虽然是 Python 整数子类，但不能作为候选数量。
        if isinstance(raw_invalid_count, (bool, np.bool_)) or not isinstance(
            raw_invalid_count,
            (int, np.integer),
        ):
            # 非整数诊断无法和实际断点数量核对。
            raise ValueError("invalid_count 必须是非负整数")
        # 转成普通整数，避免 Qt 属性持有 NumPy 标量。
        invalid_count = int(raw_invalid_count)
        # 显式计数必须和曲线中的 NaN 断点完全一致。
        if invalid_count < 0 or invalid_count != derived_invalid_count:
            # 不展示可能误导用户的诊断数量。
            raise ValueError("invalid_count 与曲线中的不可解析位置数量不一致")
        # 诊断文字先经过简洁文案边界验证。
        raw_diagnostic = result.get("diagnostic", "")
        # None 表示调用方未提供诊断。
        diagnostic = "" if raw_diagnostic is None else str(raw_diagnostic)
        # 存在断点但未给文字时由页面生成最短可读说明。
        if invalid_count > 0 and not diagnostic:
            # 默认文字明确断点不是零改善。
            diagnostic = f"{invalid_count} 个候选不可解析，曲线以断点表示"
        # 诊断同样不能绕过用户确认的文案边界。
        self._validate_visible_text(diagnostic, "解析诊断")

        # 候选列表允许任意字符串序列，具体频段格式由后台或主窗口决定。
        candidates = result.get("candidates", ())
        # 单个字符串不是候选序列，避免它被错误拆成逐字符列表。
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            # 页面要求明确的字符串序列。
            raise ValueError("candidates 必须是字符串序列")
        # 先格式化全部候选，避免中途错误时列表只显示一部分。
        prepared_candidates: list[str] = []
        # 逐项验证并保留后台给出的排序。
        for candidate in candidates:
            # 所有候选统一转成字符串。
            candidate_text = str(candidate)
            # 展示边界拒绝用户明确不希望出现的冗余术语。
            self._validate_visible_text(candidate_text, "候选文字")
            # 验证后收集为不依赖输入序列的快照。
            prepared_candidates.append(candidate_text)

        # Vpp 结果可以不包含 eyes；眼图结果则一次提供完整三幅图。
        eyes = result.get("eyes")
        # 眼图中间对象在页面变更前完成所有时轴和轨迹验证。
        prepared_eyes = None
        # 提供 eyes 时必须是按角色命名的映射。
        if eyes is not None and not isinstance(eyes, Mapping):
            # 拒绝无法识别三图角色的序列。
            raise ValueError("eyes 必须是包含参考、补偿前和补偿后的映射")
        # 三幅图全部验证后才返回绘图快照。
        if isinstance(eyes, Mapping):
            # 调用无副作用的眼图准备函数。
            prepared_eyes = self._prepare_eyes(eyes)

        # Vpp 波形允许参考、补偿前和补偿后使用各自时间轴。
        waveforms = result.get("waveforms")
        # 非空波形必须是角色到数据的映射。
        if waveforms is not None and not isinstance(waveforms, Mapping):
            # 明确拒绝不能保留独立时间轴的位置序列。
            raise ValueError("waveforms 必须是按角色命名的映射")
        # 在清空旧图前完整准备波形。
        prepared_waveforms = (
            self._prepare_waveforms(waveforms)
            if isinstance(waveforms, Mapping)
            else None
        )
        # 当前候选摘要是主窗预格式化的可选短文字。
        summary = self._prepare_summary(result.get("summary", ""))

        # 所有数据都通过验证后，才原子地替换当前展示。
        self.clear_result()
        # Hz 除以 1e9 只改变显示单位，不修改后台物理数据。
        frequency_ghz = frequency_hz / 1.0e9
        # 逐条画入已验证的影响曲线。
        self._draw_impact_curves(frequency_ghz, prepared_scores)
        # 候选列表按后台排序一次性提交。
        self.candidate_list.addItems(prepared_candidates)
        # 每行工具提示保留完整候选文字，系统字体较大或文本省略时仍可查看。
        for row in range(self.candidate_list.count()):
            # 列表项必然来自刚加入的字符串，逐行绑定自己的完整文本。
            item = self.candidate_list.item(row)
            # Qt 返回的真实列表项设置工具提示，不复制额外业务状态。
            item.setToolTip(item.text())
        # 绘制可选眼图，Vpp 结果通常为 None。
        if prepared_eyes is not None:
            # 三图使用已验证的同一时轴和幅值范围。
            self._draw_prepared_eyes(prepared_eyes)
        # 绘制可选 Vpp 波形。
        if prepared_waveforms is not None:
            # 各曲线保留独立时间轴。
            self._draw_waveforms(prepared_waveforms)
        # 摘要为空时保持隐藏。
        self._show_summary(summary)
        # 不可解析候选使用独立短诊断，不占用当前候选摘要。
        self._show_diagnostic(diagnostic)

    # 点选候选后只更新详情图和摘要，不重置影响曲线或列表行号。
    def render_selection(self, detail: Mapping[str, object]) -> None:
        """更新当前候选的眼图或 Vpp 波形，保留扫描总览。"""

        # 独立详情 API 同样只接收轻量映射。
        if not isinstance(detail, Mapping):
            # 类型错误在修改界面前返回。
            raise TypeError("候选详情必须是映射")
        # 眼图映射可选。
        eyes = detail.get("eyes")
        # 非空眼图必须使用角色映射。
        if eyes is not None and not isinstance(eyes, Mapping):
            # 拒绝不可识别的详情结构。
            raise ValueError("eyes 必须是包含三幅眼图的映射")
        # 先准备眼图，失败时保留旧候选详情。
        prepared_eyes = self._prepare_eyes(eyes) if isinstance(eyes, Mapping) else None
        # Vpp 详情也使用可选映射。
        waveforms = detail.get("waveforms")
        # 波形必须按角色命名。
        if waveforms is not None and not isinstance(waveforms, Mapping):
            # 非映射无法表达每条波形的独立时间轴。
            raise ValueError("waveforms 必须是按角色命名的映射")
        # 先准备所有波形。
        prepared_waveforms = (
            self._prepare_waveforms(waveforms)
            if isinstance(waveforms, Mapping)
            else None
        )
        # 摘要在更新图层前完成文案验证。
        summary = self._prepare_summary(detail.get("summary", ""))
        # 只清空三幅眼图、Vpp 波形和摘要。
        self._clear_detail()
        # 绘制新眼图详情。
        if prepared_eyes is not None:
            # 三图继续共轴，并保留稳定的三角色颜色。
            self._draw_prepared_eyes(prepared_eyes)
        # 绘制新 Vpp 波形详情。
        if prepared_waveforms is not None:
            # 参考波形仍是可选角色。
            self._draw_waveforms(prepared_waveforms)
        # 更新当前候选摘要。
        self._show_summary(summary)

    # 清空全部展示对象，但保留用户输入与当前指标选择。
    def clear_result(self) -> None:
        """移除过期曲线、眼图轨迹和候选项。"""

        # 清空影响曲线和其图例条目。
        self.impact_plot.clear()
        # 取得可能已经创建的图例对象。
        legend = self.impact_plot.getPlotItem().legend
        # 空图隐藏图例边框，避免出现无内容的小方框。
        if legend is not None:
            # 隐藏但保留对象，下次渲染可以安全复用。
            legend.hide()
        # 候选详情图和摘要通过独立助手统一清理。
        self._clear_detail()
        # 完整结果清空时诊断也不应继续描述过期曲线。
        self.diagnostic_label.clear()
        # 空诊断不占据右栏空间。
        self.diagnostic_label.hide()
        # 候选列表清空后不再允许用户点击过期频段。
        self.candidate_list.clear()

    # 无副作用地验证三幅眼图共用完全相同的 2 UI 时间轴和幅值范围。
    def _prepare_eyes(
        self,
        eyes: Mapping[str, object],
    ) -> tuple[
        list[tuple[pg.PlotWidget, str, np.ndarray, np.ndarray]],
        tuple[float, float],
    ]:
        # 公共时间轴只解析一次，三角色不能分别提供不同相位网格。
        time_ui = np.array(eyes.get("time_ui"), dtype=np.float64, copy=True)
        # 2*M+1 必须是奇数，中心点对应 0 UI，两端固定为 -1/+1 UI。
        if (
            time_ui.ndim != 1
            or time_ui.size < 5
            or time_ui.size % 2 != 1
            or not np.all(np.isfinite(time_ui))
            or np.any(np.diff(time_ui) <= 0.0)
            or not np.isclose(time_ui[0], -1.0)
            or not np.isclose(time_ui[-1], 1.0)
            or not np.isclose(time_ui[time_ui.size // 2], 0.0)
            or not np.allclose(time_ui, -time_ui[::-1])
        ):
            # 严格时窗能发现主光标偏置、少一个右端点或时间轴翻转。
            raise ValueError("eyes.time_ui 必须是 -1 到 +1 UI 的 2*M+1 对称递增数组")
        # 三图共用的幅值范围防止独立自动缩放伪造眼张开程度差异。
        amplitude_values = np.asarray(
            eyes.get("amplitude_range_v"),
            dtype=np.float64,
        )
        # 范围必须是两个严格递增的有限端点。
        if (
            amplitude_values.shape != (2,)
            or not np.all(np.isfinite(amplitude_values))
            or amplitude_values[0] >= amplitude_values[1]
        ):
            # 无效范围不能交给 ViewBox 自行猜测。
            raise ValueError("eyes.amplitude_range_v 必须是严格递增的两个有限值")
        # 转成普通 float 元组，避免 Qt 状态持有 NumPy 标量。
        amplitude_range_v = (
            float(amplitude_values[0]),
            float(amplitude_values[1]),
        )
        # 三个键按页面从左到右的顺序绑定到对应 PlotWidget 和角色色彩。
        eye_targets = (
            ("reference", self.reference_plot, _REFERENCE),
            ("before", self.before_plot, _DUT),
            ("after", self.after_plot, _RESULT),
        )
        # 每个元组保存画布、角色颜色、公共时间轴和该角色的轨迹快照。
        prepared: list[tuple[pg.PlotWidget, str, np.ndarray, np.ndarray]] = []
        # 一次循环解析并验证三份眼图轨迹数据。
        for role, plot, color in eye_targets:
            # 每个角色必须提供一个轻量映射。
            eye = eyes.get(role)
            # 缺少任意角色都会破坏三图对比，应立即拒绝。
            if not isinstance(eye, Mapping):
                # 错误信息包含缺失的稳定英文角色键。
                raise ValueError(f"eyes.{role} 必须是映射")
            # 轨迹复制为 float64，渲染后不受调用方原数组修改影响。
            traces_v = np.array(eye.get("traces_v"), dtype=np.float64, copy=True)
            # 每行是一条 2 UI 轨迹；页面最多接受 600 条，防止绘图负载失控。
            if (
                traces_v.ndim != 2
                or traces_v.shape[0] < 1
                or traces_v.shape[0] > _MAX_EYE_TRACES
                or traces_v.shape[1] != time_ui.size
                or not np.all(np.isfinite(traces_v))
            ):
                # 错误信息同时说明轨迹数上限和每条的列数合同。
                raise ValueError(
                    f"eyes.{role}.traces_v 必须是 1–{_MAX_EYE_TRACES} 行、"
                    "与 time_ui 等列的二维有限数组"
                )
            # 保存一份已验证数据，角色顺序不再依赖输入字典的插入顺序。
            prepared.append((plot, color, time_ui, traces_v))
        # 返回纯数据快照，到此仍未改变任何图层。
        return prepared, amplitude_range_v

    # 把已验证的三幅眼图一次性画入场景。
    def _draw_prepared_eyes(
        self,
        prepared_eyes: tuple[
            list[tuple[pg.PlotWidget, str, np.ndarray, np.ndarray]],
            tuple[float, float],
        ],
    ) -> None:
        # 解包三角色轨迹与共同幅值范围。
        prepared, amplitude_range_v = prepared_eyes
        # 逐幅把所有轨迹拼成一条带 NaN 分隔的 PlotDataItem，避免为 600 条线创建 600 个图形对象。
        for plot, color_name, time_ui, traces_v in prepared:
            # 每条轨迹后多一列 NaN，PyQtGraph 会在该处断线而不连到下一个符号。
            separated_x = np.empty(
                (traces_v.shape[0], time_ui.size + 1),
                dtype=np.float64,
            )
            # 每一行复用同一条 -1 到 +1 UI 时间轴。
            separated_x[:, :-1] = time_ui
            # 行末的 NaN 强制两条眼图轨迹不相连。
            separated_x[:, -1] = np.nan
            # 幅值使用相同形状的分隔数组。
            separated_y = np.empty_like(separated_x)
            # 保留后台给出的每条轨迹数值和角色顺序。
            separated_y[:, :-1] = traces_v
            # y 数据同样以 NaN 结束每条轨迹。
            separated_y[:, -1] = np.nan
            # 角色颜色保持参考蓝、补偿前橙、补偿后青的稳定语义。
            trace_color = pg.mkColor(color_name)
            # 低不透明度还原附件中多轨迹叠加的视觉，重合越多的通道自然越亮。
            trace_color.setAlpha(52)
            # 每幅眼图只创建这一个 PlotDataItem，线宽保持细而可读。
            trace_item = plot.plot(
                separated_x.ravel(),
                separated_y.ravel(),
                pen=pg.mkPen(trace_color, width=0.65),
                connect="finite",
            )
            # 通用绘图工厂的峰值降采样不了解 NaN 轨迹边界，会把多段线压成伪竖线。
            trace_item.setDownsampling(ds=1, auto=False, method="peak")
            # 视窗裁剪也可能在分隔符两侧重组数据；2 UI 轨迹规模受 600 条上限保护。
            trace_item.setClipToView(False)
            # 0 UI 是主光标中心，独立虚线让三图采样相位可直接对照。
            center_line = pg.InfiniteLine(
                pos=0.0,
                angle=90,
                pen=pg.mkPen(
                    _EYE_CENTER_LINE,
                    width=1.0,
                    style=Qt.PenStyle.DashLine,
                ),
            )
            # 中心线只是坐标参考，不出现在曲线图例中。
            plot.addItem(center_line)
            # x 轴固定为附件算法的 -1 到 +1 UI 窗口，不增加自动 padding。
            plot.setXRange(-1.0, 1.0, padding=0.0)
            # y 轴三图使用同一范围。
            plot.setYRange(
                amplitude_range_v[0],
                amplitude_range_v[1],
                padding=0.0,
            )
            # 固定当前范围，后续图像刷新不会触发独立自动缩放。
            plot.disableAutoRange()

    # 无副作用地验证 Vpp 波形，允许每个角色使用自己的采样网格。
    def _prepare_waveforms(
        self,
        waveforms: Mapping[str, object],
    ) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
        # 参考是可选第三条曲线，补偿前和补偿后为必需。
        role_styles = (
            ("reference", "参考", _REFERENCE, False),
            ("before", "补偿前", _DUT, True),
            ("after", "补偿后", _RESULT, True),
        )
        # 中间列表保存显示名、颜色、ns 时间轴和幅值。
        prepared: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        # 逐角色完整验证。
        for role, visible_name, color, required in role_styles:
            # 取得当前角色的轻量映射。
            waveform = waveforms.get(role)
            # 可选参考未提供时直接跳过。
            if waveform is None and not required:
                # 补偿前后仍会继续验证。
                continue
            # 必需角色缺失或非映射都无法绘图。
            if not isinstance(waveform, Mapping):
                # 错误中保留稳定英文角色键。
                raise ValueError(f"waveforms.{role} 必须是映射")
            # 复制秒时间轴。
            time_s = np.array(waveform.get("time_s"), dtype=np.float64, copy=True)
            # 复制单通道幅值。
            values = np.array(waveform.get("values"), dtype=np.float64, copy=True)
            # 两个数组必须一维、等长、非空且全部有限。
            if (
                time_s.ndim != 1
                or values.ndim != 1
                or time_s.size == 0
                or time_s.shape != values.shape
                or not np.all(np.isfinite(time_s))
                or not np.all(np.isfinite(values))
                or (time_s.size > 1 and np.any(np.diff(time_s) <= 0.0))
            ):
                # 严格递增时间轴保留原采样率物理意义。
                raise ValueError(f"waveforms.{role} 必须提供等长有限数组和递增时间轴")
            # 页面只把秒换成 ns 显示，不重采样。
            time_ns = time_s * 1.0e9
            # 收集已验证曲线。
            prepared.append((visible_name, color, time_ns, values))
        # 返回与输入映射解耦的数组快照。
        return prepared

    # 绘制三种影响模式曲线。
    def _draw_impact_curves(
        self,
        frequency_ghz: np.ndarray,
        scores: list[tuple[str, str, np.ndarray]],
    ) -> None:
        # 空页面首次渲染时创建图例，之后复用。
        legend = self._show_plot_legend(self.impact_plot)
        # 逐模式加入真实 PlotDataItem。
        for visible_name, color, values in scores:
            # 可见名称直接驱动图例。
            self.impact_plot.plot(
                frequency_ghz,
                values,
                name=visible_name,
                pen=pg.mkPen(color, width=1.8),
                connect="finite",
            )
        # 汇总全部有限得分，NaN 断点不会污染纵轴范围。
        finite_score_parts = [values[np.isfinite(values)] for _name, _color, values in scores]
        # 至少一个有效点时按真实得分确定范围。
        if any(part.size > 0 for part in finite_score_parts):
            # 拼接非空数组后计算有限最小值和最大值。
            finite_scores = np.concatenate(
                [part for part in finite_score_parts if part.size > 0]
            )
            # 纵轴下界来自所有模式的有效点。
            y_low = float(np.min(finite_scores))
            # 纵轴上界同样忽略全部 NaN 位置。
            y_high = float(np.max(finite_scores))
        # 三种模式都没有有效得分时仍给坐标轴一个可交互范围，并保留曲线断点证据。
        else:
            # 全部不可解析时使用中性有限范围，图中只保留断线证据。
            y_low, y_high = -1.0, 1.0
        # 常量曲线需要人工展开少量范围，真实零仍清晰位于中心。
        if y_low == y_high:
            # 绝对值很小时使用 1 的尺度，避免浮点下溢成零跨度。
            y_margin = max(abs(y_low), 1.0) * 0.05
            # 对称扩展纵轴下界。
            y_low -= y_margin
            # 对称扩展纵轴上界。
            y_high += y_margin
        # 横轴全部已验证有限，直接取得候选中心范围。
        x_low = float(np.min(frequency_ghz))
        # 横轴上界覆盖最后一个候选中心。
        x_high = float(np.max(frequency_ghz))
        # 单候选也需要有限可交互宽度。
        if x_low == x_high:
            # GHz 轴至少展开 0.1 GHz，避免 ViewBox 零宽。
            x_margin = max(abs(x_low) * 0.05, 0.05)
            # 左右对称扩展唯一候选中心。
            x_low -= x_margin
            # 右侧使用相同边距。
            x_high += x_margin
        # 显式设置有限 x/y 范围，避免图形库把 NaN 纳入自动边界。
        self.impact_plot.setXRange(x_low, x_high, padding=0.02)
        # 纵轴使用少量边距保留曲线可读性。
        self.impact_plot.setYRange(y_low, y_high, padding=0.05)
        # 用户后续缩放时保持手动视窗。
        self.impact_plot.disableAutoRange()
        # 显式引用让类型和意图清晰，图例已由曲线名称自动更新。
        _ = legend

    # 绘制保留独立时间轴的 Vpp 波形。
    def _draw_waveforms(
        self,
        waveforms: list[tuple[str, str, np.ndarray, np.ndarray]],
    ) -> None:
        # 波形图例与影响曲线图例独立管理。
        self._show_plot_legend(self.vpp_waveform_plot)
        # 逐条波形使用自己的 ns 时间轴。
        for visible_name, color, time_ns, values in waveforms:
            # 不重采样、不截短任何输入记录。
            self.vpp_waveform_plot.plot(
                time_ns,
                values,
                name=visible_name,
                pen=pg.mkPen(color, width=1.5),
            )
        # 一次自动覆盖全部时间和幅值。
        self.vpp_waveform_plot.enableAutoRange()
        # 立即应用范围。
        self.vpp_waveform_plot.autoRange()
        # 后续交互缩放不被自动范围抢回。
        self.vpp_waveform_plot.disableAutoRange()

    # 图例使用统一深色样式，但每幅图保留自己的 LegendItem。
    @staticmethod
    def _show_plot_legend(plot: pg.PlotWidget) -> object:
        # 取得已有图例。
        legend = plot.getPlotItem().legend
        # 首次绘制时才创建图例。
        if legend is None:
            # 图例放在左上角并使用高对比文字。
            legend = plot.addLegend(offset=(10, 8), labelTextColor=_TEXT)
            # 半透明近黑底不过度遮挡波形。
            legend.setBrush(pg.mkBrush(14, 20, 29, 218))
            # 细边框与页面分区一致。
            legend.setPen(pg.mkPen(_BORDER))
        # 下次渲染重新显示被清空助手隐藏的图例。
        legend.show()
        # 返回图例便于调用方保持强引用语义。
        return legend

    # 摘要输入只作文字复制和展示边界验证。
    def _prepare_summary(self, summary: object) -> str:
        # None 与缺失摘要都视为空文字。
        text = "" if summary is None else str(summary)
        # 摘要不能绕过页面简洁文案合同。
        self._validate_visible_text(text, "候选摘要")
        # 返回不依赖输入对象的字符串。
        return text

    # 所有后台可控可见文字共用同一禁词边界。
    @staticmethod
    def _validate_visible_text(text: str, field_name: str) -> None:
        # 中文词或不分大小写的英文缩写都不应出现在页面。
        if "虚拟" in text or "工程近似" in text or "ISI" in text.upper():
            # 拒绝整份新展示，不擅自删词造成语义残缺。
            raise ValueError(f"{field_name}包含不允许的冗余标注")

    # 只清理当前候选的大型图层和摘要。
    def _clear_detail(self) -> None:
        # 三幅图逐一清空轨迹 PlotDataItem 和 0 UI 中心线。
        for plot in (self.reference_plot, self.before_plot, self.after_plot):
            # PlotWidget.clear 会从真实场景移除全部当前绘图项。
            plot.clear()
        # Vpp 波形图独立清空。
        self.vpp_waveform_plot.clear()
        # 取得可能已创建的波形图例。
        waveform_legend = self.vpp_waveform_plot.getPlotItem().legend
        # 空图隐藏图例边框。
        if waveform_legend is not None:
            # 保留图例对象供下次候选复用。
            waveform_legend.hide()
        # 清空摘要文字。
        self.selection_summary.clear()
        # 空摘要不占据右侧列表空间。
        self.selection_summary.hide()

    # 有内容时显示当前候选摘要。
    def _show_summary(self, summary: str) -> None:
        # 设置已验证文字。
        self.selection_summary.setText(summary)
        # 只有非空文字才显示标签。
        self.selection_summary.setVisible(bool(summary))

    # 有不可解析候选时显示独立诊断，不覆盖当前候选摘要。
    def _show_diagnostic(self, diagnostic: str) -> None:
        # 写入已经通过文案校验的短文字。
        self.diagnostic_label.setText(diagnostic)
        # 空文字时彻底隐藏，避免正常扫描多出冗余标注。
        self.diagnostic_label.setVisible(bool(diagnostic))

    # 点击按钮时只发送不可耗时的当前参数字典。
    def _emit_analysis_request(self) -> None:
        # 立即构造快照，后续用户改动不会修改已经发出的请求。
        request = dict(self.current_request())
        # 发出对象信号，由主窗口决定线程、校验和状态管理。
        self.analysis_requested.emit(request)

    # 任一请求条件变化时清空过期结果并递增主窗版本。
    def _invalidate_request(self, *_args: object) -> None:
        # 界面不允许旧曲线继续冒充新参数的结果。
        self.clear_result()
        # 主窗可取消后台任务或忽略旧版本回调。
        self.request_changed.emit()

    # 列表清空会产生 -1，只把真实候选行发给主窗。
    def _emit_candidate_selected(self, row: int) -> None:
        # 非负索引才能对应扫描结果中的候选。
        if row >= 0:
            # 主窗按自己的工作区缓存重新计算候选详情。
            self.candidate_selected.emit(row)

    # 统一暴露本页拥有的四类绘图，不让主窗取猜属性名。
    def plots(self) -> tuple[pg.PlotWidget, ...]:
        """返回影响曲线、Vpp 波形与三幅眼图的稳定顺序。"""

        # 影响曲线和 Vpp 波形先于三幅对比眼图。
        return (
            self.impact_plot,
            self.vpp_waveform_plot,
            self.reference_plot,
            self.before_plot,
            self.after_plot,
        )

    # 主窗工具栏可以一次切换本页所有图的鼠标语义。
    def set_mouse_mode(self, mode: str) -> None:
        """把所有页内绘图切换为框选缩放或平移。"""

        # 只接受与主窗工具栏一致的两个稳定键。
        if mode not in {"zoom", "pan"}:
            # 拒绝未知键，避免静默退回错误交互模式。
            raise ValueError("mode 必须是 zoom 或 pan")
        # zoom 对应 PyQtGraph 矩形框选，pan 对应拖动平移。
        mouse_mode = pg.ViewBox.RectMode if mode == "zoom" else pg.ViewBox.PanMode
        # 每幅图独立设置，三幅眼图的坐标链接不受影响。
        for plot in self.plots():
            # 直接修改 ViewBox 的真实鼠标模式。
            plot.getViewBox().setMouseMode(mouse_mode)

    # 主窗“重置视图”按钮可将本页回到全数据范围。
    def reset_view(self) -> None:
        """对页内全部绘图执行一次自动范围并随后保持视窗。"""

        # 四类绘图由页面自己管理，不加入旧页面的清空列表。
        for plot in self.plots():
            # 暂时允许 ViewBox 根据当前图层计算范围。
            plot.enableAutoRange()
            # 立即应用自动范围，不等待下一帧。
            plot.autoRange()
            # 重置完成后恢复手动交互视窗。
            plot.disableAutoRange()

    # 主窗可在后台扫描期间锁定请求条件。
    def set_busy(self, busy: bool) -> None:
        """设置分析处理状态，保留已绘图内容供用户查看。"""

        # 布尔化防止 NumPy bool 等对象被 Qt 属性意外保留。
        is_busy = bool(busy)
        # 忙时禁止重复提交。
        self.start_button.setEnabled(not is_busy)
        # 指标在任务中保持锁定。
        self.metric_combo.setEnabled(not is_busy)
        # 公共频段宽度在任务中保持锁定。
        self.band_width_spin.setEnabled(not is_busy)
        # 单位与数值共同定义一个输入，任务中必须同步锁定。
        self.band_width_unit_combo.setEnabled(not is_busy)
        # 调制格式同样锁定。
        self.modulation_combo.setEnabled(not is_busy)
        # M 在任务中不可修改。
        self.m_spin.setEnabled(not is_busy)
        # Vpp 文件选择区整体锁定。
        self.vpp_paths_panel.setEnabled(not is_busy)
        # 扫描期间不点选旧候选，避免与更新缓存竞争。
        self.candidate_list.setEnabled(not is_busy)
        # 按钮文字给出简短可见进度。
        self.start_button.setText("分析中…" if is_busy else "开始分析")

    # 指标切换只影响专属输入区域，候选影响结果区始终保留。
    def _update_metric_visibility(self, *_args: object) -> None:
        # Vpp 是唯一需要两份原始数据路径的指标。
        is_vpp = self.metric_combo.currentData() == "vpp"
        # Vpp 模式显示路径，眼图模式隐藏路径。
        self.vpp_paths_panel.setVisible(is_vpp)
        # Vpp 模式显示原始波形对比区。
        self.vpp_waveform_panel.setVisible(is_vpp)
        # 调制和 M 只在眼图模式出现。
        self.eye_parameters_panel.setVisible(not is_vpp)
        # 三幅眼图同样只服务眼高与眼宽。
        self.eye_plots_panel.setVisible(not is_vpp)

    # 页面局部样式复用现有深色工作台的控件语义。
    @staticmethod
    def _stylesheet() -> str:
        # 返回只作用于 InfluenceBandPage 子树的 Qt 样式表。
        return f"""
        QWidget#influenceBandPage, QWidget#influenceContent {{
            background: transparent;
            color: {_TEXT};
        }}
        QScrollArea {{ background: transparent; border: none; }}
        QFrame#influenceControls, QFrame#vppPathsPanel {{
            background: {_SURFACE};
            border: 1px solid {_BORDER};
            border-radius: 10px;
        }}
        QLabel {{ background: transparent; color: {_TEXT_MUTED}; font-size: 12px; }}
        QLabel#eyePlotTitle {{ color: {_TEXT}; font-size: 13px; font-weight: 650; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {_SURFACE_RAISED};
            color: {_TEXT};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            min-height: 34px;
            padding: 0 8px;
        }}
        QLineEdit[readOnly="true"] {{ color: {_TEXT_MUTED}; }}
        QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {_BORDER_STRONG};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {_REFERENCE};
        }}
        QComboBox QAbstractItemView {{
            background: {_SURFACE_RAISED};
            color: {_TEXT};
            border: 1px solid {_BORDER_STRONG};
            selection-background-color: {_REFERENCE};
            selection-color: white;
            outline: 0;
            padding: 3px;
        }}
        QPushButton {{ border-radius: 8px; padding: 7px 12px; font-weight: 650; }}
        QPushButton#primaryButton {{
            background: {_REFERENCE};
            color: white;
            border: 1px solid #78A6FF;
        }}
        QPushButton#primaryButton:hover {{ background: #78A6FF; }}
        QPushButton#secondaryButton {{
            background: {_SURFACE_RAISED};
            color: {_TEXT};
            border: 1px solid {_BORDER};
        }}
        QPushButton#secondaryButton:hover {{ border-color: {_BORDER_STRONG}; }}
        QListWidget {{
            background: {_BACKGROUND};
            color: {_TEXT};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            padding: 4px;
        }}
        QListWidget::item {{ padding: 7px 8px; border-radius: 6px; }}
        QListWidget::item:selected {{ background: rgba(91, 143, 249, 0.22); }}
        QSplitter::handle {{ background: {_BORDER}; width: 6px; }}
        """
