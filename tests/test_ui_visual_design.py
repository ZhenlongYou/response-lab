"""ResponseLab 仪器工作台视觉层级与状态语义回归测试。"""

# 逐语句中文维护注释会打断测试导入块并超过英文字符宽度，文件级忽略仅覆盖格式告警。
# ruff: noqa: E501, I001
from __future__ import annotations

# Codex说明(自动生成)： 导入 os，提供本文件后续流程需要的库能力。
import os

# 在无显示的 CI 与 PyCharm 测试进程中使用 Qt 离屏后端，避免弹出真实窗口。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Codex说明(自动生成)： 从 PySide6.QtCore 导入 QSize, Qt，提供本文件后续流程需要的库能力。
from PySide6.QtCore import QSize, Qt
# Codex说明(自动生成)： 从 PySide6.QtWidgets 导入 QFrame, QLabel, QScrollArea, QSplitter 等名称，提供本文件后续流程需要的库能力。
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QSplitter, QToolButton

# Codex说明(自动生成)： 从 response_lab.app 导入 _qt_application, build_demo_run，提供本文件后续流程需要的库能力。
from response_lab.app import _qt_application, build_demo_run
# 导入主窗口与用户批准的语义色，防止悬浮改版意外抬亮整个黑色主题。
from response_lab.ui import (
    BACKGROUND,
    BORDER,
    ICON_DIRECTORY,
    SURFACE,
    SURFACE_SUBTLE,
    ResponseLabWindow,
)


# Codex说明(自动生成)： 定义函数 test_compact_floating_panels_preserve_approved_palette，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_compact_floating_panels_preserve_approved_palette() -> None:
    """悬浮分区必须紧凑，并严格沿用用户确认的近黑色板。"""

    # 构造并显示真实窗口，让 QSplitter 完成三个面板的最终几何分配。
    application = _qt_application()
    # 使用效果图对应的 1248×768 尺寸，直接核对桌面首屏的可见间距。
    window = ResponseLabWindow()
    # 固定为用户确认稿尺寸，避免离屏虚拟屏幕改变自适应结果。
    window.resize(1248, 768)
    # 显示窗口后 Qt 才会应用布局边距、分隔手柄和页签内容边距。
    window.show()
    # 冲刷布局事件，保证后续几何断言读取的是最终值。
    application.processEvents()

    # 根布局四周统一保留 6 px，使顶栏和主面板呈现紧凑悬浮边界。
    root_layout = window.centralWidget().layout()
    # 主窗口必须继续使用一个可检查的根布局承载顶栏与三栏工作区。
    assert root_layout is not None
    # 读取四个方向的边距，防止仅左右悬浮而上下仍贴住窗口边缘。
    root_margins = root_layout.contentsMargins()
    # 四周严格使用效果图确认的紧凑 6 px，而不是宽松仪表盘留白。
    assert (
        root_margins.left(),
        root_margins.top(),
        root_margins.right(),
        root_margins.bottom(),
    ) == (6, 6, 6, 6)
    # 顶栏与三栏之间同样只留 6 px，保持高密度工程工作台比例。
    assert root_layout.spacing() == 6

    # QSplitter 的手柄宽度就是左右面板之间的可见暗色间隔。
    splitter = window.findChild(QSplitter, "workspaceSplitter")
    # 分隔器必须存在，否则无法通过拖动继续调整三栏宽度。
    assert splitter is not None
    # 6 px 既形成悬浮分区，也不会像上一版效果图那样浪费画布宽度。
    assert splitter.handleWidth() == 6
    # 依次取得三个分区，核对真实屏幕坐标而非只检查配置常量。
    side_panel = window.findChild(QFrame, "sidePanel")
    # 中央绘图工作区必须仍是独立面板。
    workspace = window.findChild(QFrame, "workspace")
    # 右侧设置区必须仍是独立面板。
    inspector = window.findChild(QFrame, "inspectorPanel")
    # 三个面板缺一不可，后续间隔断言才有意义。
    assert side_panel is not None and workspace is not None and inspector is not None
    # 计算左栏与中央栏之间不含控件本身的空白像素数。
    left_gap = workspace.geometry().left() - side_panel.geometry().right() - 1
    # 计算中央栏与右栏之间的空白像素数。
    right_gap = inspector.geometry().left() - workspace.geometry().right() - 1
    # 两处分区间隔都必须保持紧凑一致。
    assert (left_gap, right_gap) == (6, 6)

    # 页签与绘图区域使用独立的 12 px 顶部呼吸空间，不影响外部分区密度。
    first_plot_layout = window.visual_tabs.widget(0).layout()
    # 绘图页必须保留布局对象来承载一到两幅 PyQtGraph 图。
    assert first_plot_layout is not None
    # 顶部距离明显大于原来的 8 px，避免画布贴住页签下划线。
    assert first_plot_layout.contentsMargins().top() == 12

    # 用户确认稿以近黑背景和低亮度面板为基准，悬浮结构不得改变这些色值。
    assert BACKGROUND == "#080C12"
    # 主面板只比背景抬高一级，不能变成偏亮的蓝灰色。
    assert SURFACE == "#0E141D"
    # 卡片继续使用原来的深蓝黑表面。
    assert SURFACE_SUBTLE == "#111925"
    # 边框保持低对比蓝灰色，只负责分区而不产生发光效果。
    assert BORDER == "#263448"

    # 主动关闭窗口并处理销毁事件，防止 Qt 对象影响下一项测试。
    window.close()
    # 冲刷关闭事件，让测试在单独运行和完整套件中都可重复。
    application.processEvents()


# Codex说明(自动生成)： 定义函数 test_instrument_workspace_has_accessible_visual_hierarchy，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_instrument_workspace_has_accessible_visual_hierarchy() -> None:
    """关键层级、触控尺寸和辅助名称不能在后续样式调整中退化。"""

    # 构造真实 Qt 窗口，检查最终控件树而不是只匹配样式字符串。
    application = _qt_application()
    # Codex说明(自动生成)： 计算并保存 window，供后续语句继续读取或更新。
    window = ResponseLabWindow()

    # 品牌标识仍在顶栏，中央区不再创建占用纵向空间的摘要卡片。
    assert window.findChild(QLabel, "brandMark") is not None
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.findChild(QFrame, "analysisSummary") is None
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.findChild(QFrame, "segmentedControl") is not None
    # 绘图图标嵌入页签栏右上角，让页签成为中央区域第一行。
    assert window.visual_tabs.cornerWidget(Qt.Corner.TopRightCorner) is window.findChild(
        QFrame, "segmentedControl"
    )
    # 三个步骤徽标必须与三个输入卡一一对应，帮助用户按顺序扫描。
    assert len(window.findChildren(QLabel, "stepBadge")) == 3
    # 主要交互保持舒适点击高度，并为键盘或读屏用户提供明确名称。
    assert window.compare_button.minimumHeight() >= 44
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.compensate_button.minimumHeight() >= 44
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.export_button.minimumHeight() >= 44
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.zoom_button.minimumHeight() >= 36
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.compensate_button.accessibleName() == "对目标信号执行数据补偿"
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.header_state.accessibleName() == "当前任务状态"
    # 不确定进度条固定在底部状态栏；0 到 0 范围由 Qt 绘制持续移动的忙碌动画。
    assert window.progress.parentWidget() is window.statusBar()
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.progress.minimum() == 0
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.progress.maximum() == 0
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert not window.progress.isTextVisible()
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.progress.isHidden()
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.progress.accessibleName() == "后台处理进度"
    # 顶栏不再显示重复的环境和结果状态，状态对象只服务自动化与辅助读取。
    assert window.header_state.isHidden()
    # Codex说明(自动生成)： 计算并保存 label_text，供后续语句继续读取或更新。
    label_text = "\n".join(label.text() for label in window.findChildren(QLabel))
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert "本地离线分析" not in label_text
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert "SIGNAL ANALYSIS STUDIO" not in label_text
    # 用户指出的两个冗余文案不能再出现在可见控件树中。
    assert "分析工作区" not in label_text
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert not any(text.startswith("分析频带 ") for text in label_text.splitlines())
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.metric_label.isHidden()

    # 以较窄桌面宽度显示窗口，右侧表单必须完全落在滚动视口内而不横向裁切。
    window.resize(1024, 720)
    # 显示后 Qt 才会完成 QFormLayout、滚动区域和高 DPI 尺寸提示计算。
    window.show()
    # 处理布局事件后再读取几何范围，避免测试构造阶段的零尺寸假结果。
    application.processEvents()
    # 必须在真实 1024×720 画布上通过，禁止最小宽度意外增大后由 Qt 静默放大窗口造成假绿。
    assert window.size() == QSize(1024, 720)
    # 通过对象名取得真实右栏滚动区，测试最终控件树而不是实现变量。
    inspector_scroll = window.findChild(QScrollArea, "inspectorScroll")
    # 右栏结构必须存在，否则无法判断字段是否被裁切。
    assert inspector_scroll is not None
    # 横向范围为零表示全部内容均可在当前栏宽内显示。
    assert inspector_scroll.horizontalScrollBar().maximum() == 0
    # 最长自动建议字段的右边缘也必须位于可见视口以内。
    phase_high_right = window.phase_high.mapTo(
        inspector_scroll.viewport(), window.phase_high.rect().bottomRight()
    ).x()
    # 留 1 px 给控件边框，防止视觉边缘与裁切边界重合。
    assert phase_high_right < inspector_scroll.viewport().width()

    # 窄窗口会启用页签左右滚动按钮，必须继续显示深色主题下可辨识的图标。
    tab_scroll_buttons = window.visual_tabs.tabBar().findChildren(QToolButton)
    # Qt 固定创建一左一右两个内部按钮，二者在当前宽度下都应可见。
    assert len(tab_scroll_buttons) == 2 and all(button.isVisible() for button in tab_scroll_buttons)
    # 深色页签必须关闭系统基线绘制，否则未被页签覆盖的窄缝会重新出现亮色横线。
    assert not window.visual_tabs.tabBar().drawBase()
    # 三枚绘图工具按钮必须实际载入图标，不能因资源遗漏退化为空白按钮。
    assert not window.zoom_button.icon().isNull()
    # 平移按钮同样需要可用图标，维持科学绘图工具栏的形象语义。
    assert not window.pan_button.icon().isNull()
    # 恢复按钮图标负责替代冗长文字，也必须随安装包正确加载。
    assert not window.reset_button.icon().isNull()
    # 图标文件随应用打包，避免安装后回退成突兀的系统白色箭头区域。
    assert (ICON_DIRECTORY / "chevron-left.svg").is_file()
    # 右向箭头同样必须存在，保证两个方向的页签都可导航。
    assert (ICON_DIRECTORY / "chevron-right.svg").is_file()
    # 框选放大图标必须与界面代码一起交付。
    assert (ICON_DIRECTORY / "zoom-area.svg").is_file()
    # 平移图标必须与界面代码一起交付。
    assert (ICON_DIRECTORY / "pan.svg").is_file()
    # 恢复视图图标必须与界面代码一起交付。
    assert (ICON_DIRECTORY / "home.svg").is_file()

    # 最小支持宽度也必须保持三栏分离，避免用户拖窄窗口后中央与右栏重叠。
    window.resize(960, 640)
    # 刷新分隔器几何后再核对最窄状态。
    application.processEvents()
    # 窗口应准确保持声明的最小尺寸。
    assert window.size() == QSize(960, 640)
    # 取得最窄状态下的中央画布面板。
    compact_workspace = window.findChild(QFrame, "workspace")
    # 取得最窄状态下的右侧设置面板。
    compact_inspector = window.findChild(QFrame, "inspectorPanel")
    # 两个目标面板必须存在，后续几何断言才有有效对象。
    assert compact_workspace is not None and compact_inspector is not None
    # 三栏之间仍由 6 px 分隔手柄隔开，不得出现任何像素重叠。
    assert compact_inspector.geometry().left() - compact_workspace.geometry().right() - 1 == 6

    # 初始空图不应创建图例；用户截图中的小方框正是无条目图例的边框。
    assert window.pulse_plots[0].getPlotItem().legend is None

    # 1280×800 屏幕应保留 16 px 四周间距，不再使用固定 1440 px 宽度。
    preferred_size = window._preferred_initial_size(QSize(1280, 800))  # noqa: SLF001
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert preferred_size == QSize(1248, 768)

    # 主动关闭并处理延迟事件，避免 Qt 对象泄漏到下一项测试。
    window.close()
    # Codex说明(自动生成)： 调用 application.processEvents，执行当前流程需要的具体操作或副作用。
    application.processEvents()


# Codex说明(自动生成)： 定义函数 test_hidden_result_state_tracks_export_validity，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def test_hidden_result_state_tracks_export_validity() -> None:
    """隐藏状态继续同步结果有效性，保证既有自动化与导出保护不失效。"""

    # 内置演示运行能稳定触发成功状态，不依赖外部文件和随机数据。
    application = _qt_application()
    # Codex说明(自动生成)： 计算并保存 window，供后续语句继续读取或更新。
    window = ResponseLabWindow()
    # Codex说明(自动生成)： 调用 window.present_run，执行当前流程需要的具体操作或副作用。
    window.present_run(build_demo_run(), source_label="视觉回归")

    # 成功状态仍保留可读文字，同时提供样式选择器使用的语义属性。
    assert window.header_state.text() == "预览有效"
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.header_state.property("tone") == "success"
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert "预览有效" in window.header_state.accessibleDescription()
    # 频带摘要只作为页签辅助描述保留，不再生成常驻可见文本。
    assert "分析频带" in window.visual_tabs.accessibleDescription()
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.metric_label.isHidden()
    # 有命名曲线后才创建图例，并显示参考与待补偿两条曲线。
    pulse_legend = window.pulse_plots[0].getPlotItem().legend
    # 结果图必须拥有可见图例，不能因消除空框而丢失系列说明。
    assert pulse_legend is not None
    # 两条拟合脉冲应分别生成一个图例条目。
    assert len(pulse_legend.items) == 2

    # 改动补偿模式会让预览过期，文字和警告语义必须同步更新。
    window.mode_combo.setCurrentIndex(1)
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.header_state.text() == "预览已过期"
    # Codex说明(自动生成)： 断言关键前提必须成立，帮助测试或调试时尽早暴露异常状态。
    assert window.header_state.property("tone") == "warning"

    # 主动关闭并冲刷事件队列，保证测试进程可重复运行。
    window.close()
    # Codex说明(自动生成)： 调用 application.processEvents，执行当前流程需要的具体操作或副作用。
    application.processEvents()
