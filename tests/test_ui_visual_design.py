"""ResponseLab 仪器工作台视觉层级与状态语义回归测试。"""

# 逐语句中文维护注释会打断测试导入块并超过英文字符宽度，文件级忽略仅覆盖格式告警。
# ruff: noqa: E501, I001
from __future__ import annotations

# Codex说明(自动生成)： 导入 os，提供本文件后续流程需要的库能力。
import os

# 在无显示的 CI 与 PyCharm 测试进程中使用 Qt 离屏后端，避免弹出真实窗口。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Codex说明(自动生成)： 从 PySide6.QtCore 导入 QEvent, QPointF, QSize, Qt，提供本文件后续流程需要的库能力。
from PySide6.QtCore import QEvent, QPointF, QSize, Qt
# QImage 类型标注让像素差与亮度能量的独立视觉判据保持清晰，QMouseEvent 则跨 DPI 投递真实移动事件。
from PySide6.QtGui import QImage, QMouseEvent
# QTest 发送真实鼠标移动并推进 Qt 事件循环，验证粒子散开和运行呼吸动效。
from PySide6.QtTest import QTest
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
    ResponseField,
    ResponseLabWindow,
)


# 逐像素比较两帧，避免只验证定时器或内部状态而没有验证真实绘制结果。
def _pixel_difference_count(first: QImage, second: QImage) -> int:
    # 两帧必须来自同一个粒子控件尺寸，否则几何差异会污染动效判据。
    assert first.size() == second.size()
    # 统计 RGB 总差超过 18 的像素，排除抗锯齿的单级舍入噪声。
    return sum(
        1
        for y_position in range(first.height())
        for x_position in range(first.width())
        if sum(
            abs(first.pixelColor(x_position, y_position).getRgb()[channel] - second.pixelColor(x_position, y_position).getRgb()[channel])
            for channel in range(3)
        )
        > 18
    )


# 蓝色能量直接来自渲染像素，用于证明 active 明暗呼吸实际进入 paintEvent。
def _blue_energy(image: QImage) -> int:
    # 对全画布蓝通道求和；背景固定，因此两帧差值只来自粒子明暗。
    return sum(
        image.pixelColor(x_position, y_position).blue()
        for y_position in range(image.height())
        for x_position in range(image.width())
    )


# 左下粒子簇必须形成拟合脉冲，并用鼠标散开与明暗呼吸表达交互和运行状态。
def test_response_field_renders_pulse_cluster_and_tracks_interaction_state() -> None:
    """粒子簇应形成左偏脉冲峰，并在悬停或后台运行时产生目的明确的动画。"""

    # 使用真实 Qt 布局核对视觉组件的最终位置，避免只测试构造参数产生假绿。
    application = _qt_application()
    # 1248×768 是用户确认稿的目标首屏尺寸，左栏在该尺寸下拥有明确的剩余空间。
    window = ResponseLabWindow()
    # 固定目标尺寸后显示窗口，让布局引擎分配第三张卡片下面的弹性区域。
    window.resize(1248, 768)
    # 真实显示并处理事件，确保几何值和可见状态已经稳定。
    window.show()
    # 冲刷布局事件后再读取粒子簇尺寸。
    application.processEvents()

    # 通过公开组件类型和对象名定位左下视觉区域，防止无关占位 QWidget 冒充实现。
    response_field = window.findChild(ResponseField, "responseField")
    # 用户要求的左下区域必须实际存在。
    assert response_field is not None
    # 粒子簇必须位于第三张输入卡下方，而不是覆盖文件选择控件。
    assert response_field.geometry().top() > window.target_card.geometry().bottom()
    # 目标桌面尺寸下至少保留 96 px 高度，保证粒子曲线不是一条无法辨认的细线。
    assert response_field.height() >= 96
    # 区域不添加标题或说明文字，继续遵守界面简洁约束。
    assert response_field.findChildren(QLabel) == []

    # 粒子区域必须接收鼠标事件，悬停散开不能被透明事件属性直接禁用。
    assert not response_field.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # 两份拟合脉冲已就绪时，辅助描述必须提供同样的非颜色状态信息。
    response_field.set_input_count(2)
    # 状态可由读屏和自动化读取，不能只依靠粒子亮度传达。
    assert "2/3" in response_field.accessibleDescription()
    # 在静止成功态抓取真实 QWidget 像素，避免运动帧改变脉冲轮廓的独立几何检查。
    response_field.set_tone("success")
    # 冲刷重绘事件后再读取粒子像素。
    application.processEvents()
    # 抓取真实 QWidget 像素，防止只有状态机而 paintEvent 没有绘出任何粒子簇。
    rendered_field = response_field.grab().toImage()
    # 收集明显偏蓝或偏青的粒子像素；近黑背景不会达到这一色差与亮度。
    accent_pixels = [
        (x_position, y_position)
        for y_position in range(rendered_field.height())
        for x_position in range(rendered_field.width())
        if (
            (color := rendered_field.pixelColor(x_position, y_position)).blue() >= 70
            and color.blue() >= color.red() + 18
            and color.green() >= color.red() + 8
        )
    ]
    # 至少二十个强调像素证明脉冲簇与多个粒子真实可见，而非单个偶然亮点。
    assert len(accent_pixels) >= 20
    # 参考图要求主脉冲位于左侧约四分之一位置，而不是机械居中。
    peak_top = min(
        y_position
        for x_position, y_position in accent_pixels
        if rendered_field.width() * 0.12 <= x_position <= rendered_field.width() * 0.42
    )
    # 右半段只允许低幅起伏和轻微尾部抬升，不能出现第二个高峰。
    right_tail_top = min(
        y_position
        for x_position, y_position in accent_pixels
        if rendered_field.width() * 0.55 <= x_position <= rendered_field.width() * 0.92
    )
    # 左侧主峰至少高出右侧尾部四分之一画布，形成参考图的不对称脉冲构图。
    assert peak_top <= right_tail_top - rendered_field.height() * 0.25
    # 最高粒子的横坐标必须落在左侧脉冲区，直接防止后续改回居中高斯峰。
    highest_particle_x = min(accent_pixels, key=lambda position: position[1])[0]
    # 允许顶部平台内少量位置变化，但峰值不得越过 42% 宽度。
    assert rendered_field.width() * 0.12 <= highest_particle_x <= rendered_field.width() * 0.42
    # 收集右侧尾部像素，直接保护用户要求的纤细尾迹。
    tail_pixels = [
        (x_position, y_position)
        for x_position, y_position in accent_pixels
        if rendered_field.width() * 0.55 <= x_position <= rendered_field.width() * 0.92
    ]
    # 尾部纵向厚度不得超过画布高度的 10%，避免多层粒子再次堆成粗带。
    assert max(y for _x, y in tail_pixels) - min(y for _x, y in tail_pixels) <= rendered_field.height() * 0.10
    # 峰顶区域取最高点以下 12% 画布，统计真正聚集在顶部平台附近的粒子像素。
    peak_cap_pixels = [
        (x_position, y_position)
        for x_position, y_position in accent_pixels
        if rendered_field.width() * 0.12 <= x_position <= rendered_field.width() * 0.42
        and y_position <= peak_top + rendered_field.height() * 0.12
    ]
    # 峰顶虽比长尾更窄，像素密度仍应达到尾部的三分之一，防止顶部重新变稀。
    assert len(peak_cap_pixels) * 3 >= len(tail_pixels)

    # 测试显式打开运动，验证鼠标进入会启动粒子散开动画。
    response_field.set_motion_enabled(True)
    # 把鼠标移动到脉冲峰附近，走真实 Qt hover 路径而不是直接修改内部状态。
    pointer_position = QPointF(
        response_field.width() * 0.28,
        response_field.height() * 0.50,
    )
    # 直接向控件投递逻辑坐标事件，避免离屏后端在高 DPI 下把全局光标坐标重复缩放。
    application.sendEvent(
        response_field,
        QMouseEvent(
            QEvent.Type.MouseMove,
            pointer_position,
            pointer_position,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    # 等待散开插值收敛；收敛后定时器应停止但粒子维持散开位置。
    QTest.qWait(1100)
    # 冲刷鼠标和定时器事件后再抓取悬停帧。
    application.processEvents()
    # 抓取真实散开帧，不能只读取内部 scatter 数值。
    scattered_field = response_field.grab().toImage()
    # 至少二十个像素发生明显变化，证明多个粒子真实离开了拟合脉冲位置。
    assert _pixel_difference_count(rendered_field, scattered_field) >= 20
    # 鼠标静止且散开完成后不应继续 30 FPS 空转。
    assert not response_field.animation_running

    # 把鼠标移到中央图表区域，触发粒子从散开状态回聚到拟合脉冲。
    application.sendEvent(response_field, QEvent(QEvent.Type.Leave))
    # 回聚使用同一缓出插值，等待有限时间后应完全收敛。
    QTest.qWait(1100)
    # 处理 leave 与回聚定时器事件。
    application.processEvents()
    # 离开后定时器必须停止，防止后台永久周期唤醒。
    assert not response_field.animation_running
    # 回聚后的真实像素应与静态基线一致，证明粒子没有停在半散开位置。
    regrouped_field = response_field.grab().toImage()
    # 同一确定性构图允许严格像素一致，任何残余位移都会被发现。
    assert _pixel_difference_count(rendered_field, regrouped_field) == 0

    # 运行态从暗相位开始，之后应逐渐变亮以表达比较或补偿仍在工作。
    response_field.set_tone("active")
    # 立即抓取暗相位，作为 paintEvent 明暗变化的像素基线。
    dark_field = response_field.grab().toImage()
    # 保存暗相位强度作为时间变化的基准。
    dark_intensity = response_field.work_intensity
    # 等待若干帧但不完成一个完整周期，亮度应处于上升阶段。
    QTest.qWait(350)
    # 冲刷定时器触发的局部重绘和强度更新。
    application.processEvents()
    # 明暗呼吸必须产生可观测亮度变化，不能只是固定的 active 颜色。
    assert response_field.work_intensity > dark_intensity
    # 抓取上升阶段的亮帧，验证生产绘制实际使用了工作强度。
    bright_field = response_field.grab().toImage()
    # 背景不变时蓝色能量增加，证明粒子簇肉眼可见地由暗变亮。
    assert _blue_energy(bright_field) > _blue_energy(dark_field)

    # 开启减少动态效果后，即使任务仍是 active，也必须停止周期刷新。
    response_field.set_motion_enabled(False)
    # 静止替代保留运行态颜色和光点，但不再持续更新相位。
    assert not response_field.animation_running
    # 成功态应回到静态粒子簇，后台工作呼吸不再继续。
    response_field.set_tone("success")
    # 静止结果同时满足低干扰和减少运动需求。
    assert not response_field.animation_running

    # 关闭窗口并处理销毁事件，避免定时器或 QWidget 泄漏到后续用例。
    window.close()
    # 冲刷关闭事件，让本测试在完整套件中保持可重复。
    application.processEvents()


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
