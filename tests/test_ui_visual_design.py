"""ResponseLab 仪器工作台视觉层级与状态语义回归测试。"""

# 逐语句中文维护注释会打断测试导入块并超过英文字符宽度，文件级忽略仅覆盖格式告警。
# ruff: noqa: E501, I001
from __future__ import annotations

import os

# subprocess.TimeoutExpired 用于验证 macOS 辅助偏好读取超时后的安全回退。
import subprocess

# SimpleNamespace 构造不依赖真实系统命令的 defaults 返回值。
from types import SimpleNamespace

# 在无显示的 CI 与 PyCharm 测试进程中使用 Qt 离屏后端，避免弹出真实窗口。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, QSize, Qt

# QImage 类型标注让像素差与亮度能量的独立视觉判据保持清晰，QMouseEvent 则跨 DPI 投递真实移动事件。
from PySide6.QtGui import QImage, QMouseEvent

# QTest 推进 Qt 事件循环，验证局部鼠标高光和运行扫光动效。
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
)

from response_lab.app import _qt_application, build_demo_run

# 模块级导入允许验证减少动态效果探测、缓存与构造器接线。
import response_lab.ui as ui_module

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


# 补偿模式使用简短显示名，但内部枚举值必须保持不变以兼容算法与旧报告。
def test_compensation_mode_uses_compact_amplitude_phase_label() -> None:
    """幅相联合补偿应显示为“幅相”，同时继续向算法传递 both。"""

    # 先取得共享 QApplication，保证该测试可独立运行而不依赖其他测试顺序。
    application = _qt_application()
    # 构造真实主窗口，验证最终下拉框而不是重复一份文案常量。
    window = ResponseLabWindow()
    # 默认联合补偿项使用用户确认的紧凑名称。
    assert window.mode_combo.itemText(0) == "幅相"
    # 显示文案变化不能破坏 DSP 使用的稳定模式键。
    assert window.mode_combo.itemData(0) == "both"
    # 两个单项模式保持原有术语，避免一次小改动扩大为整套重命名。
    assert [window.mode_combo.itemText(index) for index in range(1, 3)] == [
        "仅幅频",
        "仅相频",
    ]
    # 主动关闭窗口，避免测试间残留顶层 Qt 控件。
    window.close()
    # 处理关闭事件，确保共享 QApplication 中没有遗留待处理控件。
    application.processEvents()


# 逐像素比较两帧，避免只验证定时器或内部状态而没有验证真实绘制结果。
def _pixel_difference_count(first: QImage, second: QImage) -> int:
    # 两帧必须来自同一个轨迹控件尺寸，否则几何差异会污染动效判据。
    assert first.size() == second.size()
    # 统计 RGB 总差超过 18 的像素，排除抗锯齿的单级舍入噪声。
    return sum(
        1
        for y_position in range(first.height())
        for x_position in range(first.width())
        if sum(
            abs(
                first.pixelColor(x_position, y_position).getRgb()[channel]
                - second.pixelColor(x_position, y_position).getRgb()[channel]
            )
            for channel in range(3)
        )
        > 18
    )


def _wait_for_response_field_to_settle(
    response_field: ResponseField,
    application: object,
    *,
    timeout_ms: int = 2_000,
) -> None:
    """等待有限悬停动效停止，不假设各平台恰好调度相同数量的帧。"""

    elapsed_ms = 0
    while response_field.animation_running and elapsed_ms < timeout_ms:
        QTest.qWait(25)
        application.processEvents()
        elapsed_ms += 25
    assert not response_field.animation_running


# 找出真正偏蓝的轨迹像素，排除近黑背景和低透明面积填充。
def _accent_pixels(image: QImage) -> list[tuple[int, int]]:
    # 阈值只依赖最终像素，不调用生产端的脉冲包络或绘制路径。
    return [
        (x_position, y_position)
        for y_position in range(image.height())
        for x_position in range(image.width())
        if (
            (color := image.pixelColor(x_position, y_position)).blue() >= 60
            and color.blue() >= color.red() + 14
            and color.green() >= color.red() + 6
        )
    ]


# 将每个可见横坐标的轨迹像素压缩为独立中心位置，便于检查连续性与波形轮廓。
def _trace_centers(accent_pixels: list[tuple[int, int]]) -> dict[int, float]:
    # 按横坐标收集纵向像素，面积填充不会通过上方强调色阈值。
    columns: dict[int, list[int]] = {}
    # 每个强调像素都归入自己的物理像素列，高 DPI 下仍使用同一坐标系。
    for x_position, y_position in accent_pixels:
        # 同列像素共同描述抗锯齿轨迹的纵向厚度。
        columns.setdefault(x_position, []).append(y_position)
    # 使用上下边界中点抵消线宽和光晕，而不复用生产端公式。
    return {
        x_position: (min(y_positions) + max(y_positions)) / 2.0
        for x_position, y_positions in columns.items()
    }


# 左下轨迹必须形成用户确认的窄峰、低位平滑长拖尾，并提供局部鼠标高光和运行扫光。
def test_response_field_renders_smooth_trace_and_tracks_interaction_state() -> None:
    """连续脉冲轨迹应保持确认轮廓，并用局部高光表达悬停和后台运行。"""

    # 使用真实 Qt 布局核对视觉组件的最终位置，避免只测试构造参数产生假绿。
    application = _qt_application()
    # 1248×768 是用户确认稿的目标首屏尺寸，左栏在该尺寸下拥有明确的剩余空间。
    window = ResponseLabWindow()
    # 固定目标尺寸后显示窗口，让布局引擎分配第三张卡片下面的弹性区域。
    window.resize(1248, 768)
    # 真实显示并处理事件，确保几何值和可见状态已经稳定。
    window.show()
    # 冲刷布局事件后再读取轨迹区域尺寸。
    application.processEvents()

    # 通过公开组件类型和对象名定位左下视觉区域，防止无关占位 QWidget 冒充实现。
    response_field = window.findChild(ResponseField, "responseField")
    # 用户要求的左下区域必须实际存在。
    assert response_field is not None
    # 脉冲轨迹必须位于第三张输入卡下方，而不是覆盖文件选择控件。
    assert response_field.geometry().top() > window.target_card.geometry().bottom()
    # 目标桌面尺寸下至少保留 96 px 高度，保证脉冲轮廓可以辨认。
    assert response_field.height() >= 96
    # 区域不添加标题或说明文字，继续遵守界面简洁约束。
    assert response_field.findChildren(QLabel) == []

    # 轨迹区域必须接收鼠标事件，局部高光不能被透明事件属性直接禁用。
    assert not response_field.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # 两份拟合脉冲已就绪时，辅助描述必须提供同样的非颜色状态信息。
    response_field.set_input_count(2)
    # 状态可由读屏和自动化读取，不能只依靠曲线亮度传达。
    assert "2/3" in response_field.accessibleDescription()
    # 在静止成功态抓取真实 QWidget 像素，避免运行扫光影响独立几何检查。
    response_field.set_tone("success")
    # 冲刷重绘事件后再读取轨迹像素。
    application.processEvents()
    # 抓取真实 QWidget 像素，防止只有状态机而 paintEvent 没有绘出连续轨迹。
    rendered_field = response_field.grab().toImage()
    # 强调色像素由最终渲染独立提取，不依赖生产端的包络采样点。
    accent_pixels = _accent_pixels(rendered_field)
    # 至少覆盖画布 78% 的物理像素列，证明它是一条连续线而不是离散粒子。
    trace_centers = _trace_centers(accent_pixels)
    # 连续性是用户从粒子方案切换到曲线方案的首要可见差异。
    assert len(trace_centers) >= rendered_field.width() * 0.78
    # 对物理像素列排序后检查最大间隔，防止总覆盖率足够但中间存在明显断线。
    ordered_trace_columns = sorted(trace_centers)
    # 连续矢量线在 1× 与 2× DPI 下均不允许跳过任何中间物理像素列。
    assert (
        max(
            ordered_trace_columns[index + 1] - ordered_trace_columns[index]
            for index in range(len(ordered_trace_columns) - 1)
        )
        <= 1
    )
    # 主峰最高点必须位于左侧 12%–32%，保护用户确认的左偏窄峰构图。
    peak_x, peak_top = min(trace_centers.items(), key=lambda position: position[1])
    # 峰位允许小范围抗锯齿偏移，但不能回到居中布局。
    assert rendered_field.width() * 0.12 <= peak_x <= rendered_field.width() * 0.32
    # 右侧 55%–92% 是用户要求的低位长拖尾，用其中位高度抵抗单像素光晕。
    tail_centers = [
        y_position
        for x_position, y_position in trace_centers.items()
        if rendered_field.width() * 0.55 <= x_position <= rendered_field.width() * 0.92
    ]
    # 连续长拖尾必须在目标区间的大多数像素列可见。
    assert len(tail_centers) >= rendered_field.width() * 0.34
    # 中位数用排序后的中央样本手算，保持测试不调用 NumPy 或生产工具。
    sorted_tail_centers = sorted(tail_centers)
    # 取中间值作为拖尾的独立纵向基准。
    tail_center = sorted_tail_centers[len(sorted_tail_centers) // 2]
    # 拖尾需位于画布下部，防止再次被抬高到视觉中心。
    assert tail_center >= rendered_field.height() * 0.78
    # 主峰半高宽不得超过画布 18%，直接保护“主峰窄一点”的用户要求。
    half_height = peak_top + (tail_center - peak_top) * 0.50
    # 只在主峰左侧区域寻找半高宽，排除后方低位拖尾。
    peak_columns = [
        x_position
        for x_position, y_position in trace_centers.items()
        if x_position <= rendered_field.width() * 0.42 and y_position <= half_height
    ]
    # 宽度采用最终像素坐标，在 1× 与 2× DPI 下保持同一比例。
    assert max(peak_columns) - min(peak_columns) <= rendered_field.width() * 0.18
    # 主峰后应先出现一次平滑下探，再回到较低的长拖尾。
    undershoot_centers = [
        y_position
        for x_position, y_position in trace_centers.items()
        if rendered_field.width() * 0.31 <= x_position <= rendered_field.width() * 0.44
    ]
    # 下探最低点至少比拖尾低 4% 画布高度，形成用户参考图中的明确回落。
    assert max(undershoot_centers) >= tail_center + rendered_field.height() * 0.04
    # 将下探区分成六段，检查只存在一次由下降转为回升的方向变化。
    undershoot_chunks: list[float] = []
    # 分段均值抑制抗锯齿量化，不掩盖人为加入的多次明显波纹。
    for chunk_index in range(6):
        # 下探区从 31% 到 44%，每段宽度固定为总区间的六分之一。
        chunk_left = rendered_field.width() * (0.31 + chunk_index * (0.13 / 6.0))
        # 当前段右边界与下一段左边界重合。
        chunk_right = rendered_field.width() * (0.31 + (chunk_index + 1) * (0.13 / 6.0))
        # 收集当前下探分段的真实轨迹中心。
        chunk_values = [
            y_position
            for x_position, y_position in trace_centers.items()
            if chunk_left <= x_position < chunk_right
        ]
        # 连续下探不允许任何分段完全缺失。
        assert chunk_values
        # 用分段平均位置描述宽缓趋势。
        undershoot_chunks.append(sum(chunk_values) / len(chunk_values))
    # 超过半个物理像素才计为真实方向，排除抗锯齿上下取整。
    undershoot_directions = [
        1 if difference > 0.5 else -1
        for difference in (
            undershoot_chunks[index + 1] - undershoot_chunks[index]
            for index in range(len(undershoot_chunks) - 1)
        )
        if abs(difference) > 0.5
    ]
    # 一次下探最多只有一次方向反转，多次反转意味着重新出现波纹。
    assert (
        sum(
            undershoot_directions[index] != undershoot_directions[index - 1]
            for index in range(1, len(undershoot_directions))
        )
        <= 1
    )
    # 把拖尾分成八个等宽区间，检查只有一次宽缓转折而没有细碎波动。
    tail_chunks: list[float] = []
    # 每个区间独立汇总中心位置，避免单像素抗锯齿造成虚假转折。
    for chunk_index in range(8):
        # 右侧长拖尾覆盖 50%–94% 画布宽度。
        chunk_left = rendered_field.width() * (0.50 + chunk_index * 0.055)
        # 相邻区间无缝衔接。
        chunk_right = rendered_field.width() * (0.50 + (chunk_index + 1) * 0.055)
        # 收集当前区间实际可见的轨迹中心。
        chunk_values = [
            y_position
            for x_position, y_position in trace_centers.items()
            if chunk_left <= x_position < chunk_right
        ]
        # 连续轨迹不允许出现空区间。
        assert chunk_values
        # 区间均值平滑掉抗锯齿的上下半像素偏差。
        tail_chunks.append(sum(chunk_values) / len(chunk_values))
    # 忽略不足半个物理像素的量化抖动，只统计真实上升或下降方向。
    tail_directions = [
        1 if difference > 0.5 else -1
        for difference in (
            tail_chunks[index + 1] - tail_chunks[index] for index in range(len(tail_chunks) - 1)
        )
        if abs(difference) > 0.5
    ]
    # 平滑拖尾最多允许一次由回升转为缓降的宽缓方向变化。
    assert (
        sum(
            tail_directions[index] != tail_directions[index - 1]
            for index in range(1, len(tail_directions))
        )
        <= 1
    )

    # 测试显式打开运动，验证鼠标进入会产生局部高光而不扭曲波形。
    response_field.set_motion_enabled(True)
    # 把鼠标移动到长拖尾附近，走真实 Qt 事件路径而不是修改内部状态。
    pointer_position = QPointF(
        response_field.width() * 0.68,
        response_field.height() * 0.84,
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
    # 按状态等待动效完成；Windows 粗粒度定时器不保证 700 ms 内恰好送达
    # 与 macOS 相同数量的帧，但最终必须在有限时间内停止空转。
    _wait_for_response_field_to_settle(response_field, application)
    # 抓取真实高光帧，不能只读取内部 hover 数值。
    highlighted_field = response_field.grab().toImage()
    # 至少十二个像素发生明显变化，证明曲线局部确实获得可见高光。
    assert _pixel_difference_count(rendered_field, highlighted_field) >= 12
    # 统计变化像素的横坐标，确保互动只影响光标附近而不是整条曲线闪烁。
    changed_x_positions = [
        x_position
        for y_position in range(rendered_field.height())
        for x_position in range(rendered_field.width())
        if sum(
            abs(
                rendered_field.pixelColor(x_position, y_position).getRgb()[channel]
                - highlighted_field.pixelColor(x_position, y_position).getRgb()[channel]
            )
            for channel in range(3)
        )
        > 18
    ]
    # 局部高光范围限制在鼠标左右各 18% 画布内，避免重新形成花哨的全局动画。
    assert min(changed_x_positions) >= rendered_field.width() * 0.50
    # 高光右边界同样保持局部性。
    assert max(changed_x_positions) <= rendered_field.width() * 0.86
    # 鼠标静止且淡入完成后不应继续 30 FPS 空转（等待助手已验证）。

    # 鼠标离开后局部高光淡出并恢复静态脉冲轨迹。
    application.sendEvent(response_field, QEvent(QEvent.Type.Leave))
    # 淡出使用同一缓出插值，并在有限时间内完全收敛。
    _wait_for_response_field_to_settle(response_field, application)
    # 淡出后的真实像素应与静态基线一致，证明局部高光没有残留。
    restored_field = response_field.grab().toImage()
    # 同一确定性构图允许严格像素一致，任何残余位移都会被发现。
    assert _pixel_difference_count(rendered_field, restored_field) == 0

    # 运行态启动沿轨迹移动的高光，表达比较或补偿仍在工作。
    response_field.set_tone("active")
    # 立即抓取第一处扫光位置作为真实像素基线。
    first_active_field = response_field.grab().toImage()
    # 等待高光沿轨迹移动一段明显距离，但不完成整轮扫描。
    QTest.qWait(500)
    # 冲刷定时器触发的局部重绘和扫描位置更新。
    application.processEvents()
    # 抓取第二处扫光位置，验证 paintEvent 实际使用了变化相位。
    second_active_field = response_field.grab().toImage()
    # 两个运行帧必须具有可观测像素差，不能只是固定 active 颜色。
    assert _pixel_difference_count(first_active_field, second_active_field) >= 12
    # active 扫光需要持续更新，直到真实后台工作完成。
    assert response_field.animation_running

    # 开启减少动态效果后，即使任务仍是 active，也必须停止周期刷新。
    response_field.set_motion_enabled(False)
    # 静止替代保留运行态曲线，但不再持续更新扫光相位。
    assert not response_field.animation_running
    # 成功态应回到静态脉冲轨迹，后台扫光不再继续。
    response_field.set_tone("success")
    # 静止结果同时满足低干扰和减少运动需求。
    assert not response_field.animation_running

    # 关闭窗口并处理销毁事件，避免定时器或 QWidget 泄漏到后续用例。
    window.close()
    # 冲刷关闭事件，让本测试在完整套件中保持可重复。
    application.processEvents()


# 自动减少动态效果必须覆盖环境变量、macOS defaults 与真实 ResponseField 构造器接线。
def test_reduced_motion_detection_and_response_field_wiring(monkeypatch) -> None:
    """显式开关或 macOS 辅助设置应让新轨迹组件使用静止替代效果。"""

    # 平台固定为 macOS，确保后续分支不会因 CI 系统类型被提前跳过。
    monkeypatch.setattr(ui_module.sys, "platform", "darwin")
    # 显式环境真值应拥有最高优先级。
    monkeypatch.setenv("RESPONSELAB_REDUCE_MOTION", "yes")

    # 若显式环境值生效，系统 defaults 命令不应被调用。
    def fail_if_called(*_args, **_kwargs):
        # 任何调用都说明环境优先级接线失效。
        raise AssertionError("显式减少动态效果不应继续读取 defaults")

    # 用失败桩保护环境变量短路行为。
    monkeypatch.setattr(ui_module.subprocess, "run", fail_if_called)
    # 清除上一窗口可能留下的单次缓存。
    ui_module._prefers_reduced_motion.cache_clear()
    # 常见真值 yes 必须被识别。
    assert ui_module._prefers_reduced_motion()

    # 移除显式开关后验证真实 macOS defaults 成功返回路径。
    monkeypatch.delenv("RESPONSELAB_REDUCE_MOTION", raising=False)
    # 保存调用参数，独立检查绝对命令路径和超时限制。
    defaults_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    # 伪造 defaults 返回 1，避免测试依赖当前电脑实际辅助设置。
    def defaults_enabled(*args, **kwargs):
        # 记录完整调用供下方断言检查。
        defaults_calls.append((args, kwargs))
        # 返回与 subprocess.CompletedProcess 相同的必要字段。
        return SimpleNamespace(returncode=0, stdout="1\n")

    # 替换系统调用后清除环境分支的缓存结果。
    monkeypatch.setattr(ui_module.subprocess, "run", defaults_enabled)
    # 下一次调用必须重新读取伪造的系统偏好。
    ui_module._prefers_reduced_motion.cache_clear()
    # defaults 的真值输出必须启用减少动态效果。
    assert ui_module._prefers_reduced_motion()
    # 系统命令只应调用一次，证明单次缓存生效。
    assert len(defaults_calls) == 1
    # 第一个位置参数必须是绝对 defaults 命令与目标偏好键。
    assert defaults_calls[0][0][0] == [
        "/usr/bin/defaults",
        "read",
        "com.apple.universalAccess",
        "reduceMotion",
    ]
    # 200 ms 超时保护窗口启动不被系统服务阻塞。
    assert defaults_calls[0][1]["timeout"] == 0.2

    # 超时属于可恢复系统探测失败，应安全回退到默认动画行为。
    def defaults_timeout(*_args, **_kwargs):
        # 构造与 subprocess.run 一致的超时异常类型。
        raise subprocess.TimeoutExpired("/usr/bin/defaults", timeout=0.2)

    # 注入超时并清除上一成功结果。
    monkeypatch.setattr(ui_module.subprocess, "run", defaults_timeout)
    # 强制重新执行探测分支。
    ui_module._prefers_reduced_motion.cache_clear()
    # 探测失败不能错误地永久关闭所有正常动画。
    assert not ui_module._prefers_reduced_motion()

    # 最后验证构造器确实使用探测结果，而不是只有独立函数正确。
    application = _qt_application()
    # 模拟真实 Cocoa 后端，绕开离屏测试默认关闭动画的保护分支。
    monkeypatch.setattr(
        ui_module.QGuiApplication,
        "platformName",
        staticmethod(lambda: "cocoa"),
    )
    # 构造器探测固定返回 True，直接验证接线关系。
    monkeypatch.setattr(ui_module, "_prefers_reduced_motion", lambda: True)
    # 新建真实 QWidget 组件以触发构造器逻辑。
    response_field = ResponseField()
    # active 通常会启动持续扫光。
    response_field.set_tone("active")
    # 构造器接线正确时，减少动态效果会阻止定时器启动。
    assert not response_field.animation_running
    # 安排组件销毁，避免 QObject 泄漏到后续用例。
    response_field.deleteLater()
    # 冲刷 deleteLater 事件。
    application.processEvents()


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


def test_instrument_workspace_has_accessible_visual_hierarchy() -> None:
    """关键层级、触控尺寸和辅助名称不能在后续样式调整中退化。"""

    # 构造真实 Qt 窗口，检查最终控件树而不是只匹配样式字符串。
    application = _qt_application()
    window = ResponseLabWindow()

    # 品牌标识仍在顶栏，中央区不再创建占用纵向空间的摘要卡片。
    assert window.findChild(QLabel, "brandMark") is not None
    assert window.findChild(QFrame, "analysisSummary") is None
    assert window.findChild(QFrame, "segmentedControl") is not None
    # 绘图图标嵌入页签栏右上角，让页签成为中央区域第一行。
    assert window.visual_tabs.cornerWidget(Qt.Corner.TopRightCorner) is window.findChild(
        QFrame, "segmentedControl"
    )
    # 三个步骤徽标必须与三个输入卡一一对应，帮助用户按顺序扫描。
    assert len(window.findChildren(QLabel, "stepBadge")) == 3
    # 主要交互保持舒适点击高度，并为键盘或读屏用户提供明确名称。
    assert window.compare_button.minimumHeight() >= 44
    assert window.compensate_button.minimumHeight() >= 44
    assert window.export_button.minimumHeight() >= 44
    assert window.zoom_button.minimumHeight() >= 36
    assert window.compensate_button.accessibleName() == "对目标信号执行数据补偿"
    assert window.header_state.accessibleName() == "当前任务状态"
    # 不确定进度条固定在底部状态栏；0 到 0 范围由 Qt 绘制持续移动的忙碌动画。
    assert window.progress.parentWidget() is window.statusBar()
    assert window.progress.minimum() == 0
    assert window.progress.maximum() == 0
    assert not window.progress.isTextVisible()
    assert window.progress.isHidden()
    assert window.progress.accessibleName() == "后台处理进度"
    # 顶栏不再显示重复的环境和结果状态，状态对象只服务自动化与辅助读取。
    assert window.header_state.isHidden()
    label_text = "\n".join(label.text() for label in window.findChildren(QLabel))
    assert "本地离线分析" not in label_text
    assert "SIGNAL ANALYSIS STUDIO" not in label_text
    # 用户指出的两个冗余文案不能再出现在可见控件树中。
    assert "分析工作区" not in label_text
    assert not any(text.startswith("分析频带 ") for text in label_text.splitlines())
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
    # 频率框不能再用理论范围上限的超长字符串撑大右栏；实际编辑仍可水平滚动。
    for spin in (window.band_low, window.band_high, window.phase_low, window.phase_high):
        assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
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
    assert preferred_size == QSize(1248, 768)

    # 主动关闭并处理延迟事件，避免 Qt 对象泄漏到下一项测试。
    window.close()
    application.processEvents()


def test_hidden_result_state_tracks_export_validity() -> None:
    """隐藏状态继续同步结果有效性，保证既有自动化与导出保护不失效。"""

    # 内置演示运行能稳定触发成功状态，不依赖外部文件和随机数据。
    application = _qt_application()
    window = ResponseLabWindow()
    window.present_run(build_demo_run(), source_label="视觉回归")

    # 成功状态仍保留可读文字，同时提供样式选择器使用的语义属性。
    assert window.header_state.text() == "预览有效"
    assert window.header_state.property("tone") == "success"
    assert "预览有效" in window.header_state.accessibleDescription()
    # 频带摘要只作为页签辅助描述保留，不再生成常驻可见文本。
    assert "分析频带" in window.visual_tabs.accessibleDescription()
    assert window.metric_label.isHidden()
    # 有命名曲线后才创建图例，并显示参考与待补偿两条曲线。
    pulse_legend = window.pulse_plots[0].getPlotItem().legend
    # 结果图必须拥有可见图例，不能因消除空框而丢失系列说明。
    assert pulse_legend is not None
    # 两条拟合脉冲应分别生成一个图例条目。
    assert len(pulse_legend.items) == 2

    # 改动补偿模式会让预览过期，文字和警告语义必须同步更新。
    window.mode_combo.setCurrentIndex(1)
    assert window.header_state.text() == "预览已过期"
    assert window.header_state.property("tone") == "warning"

    # 主动关闭并冲刷事件队列，保证测试进程可重复运行。
    window.close()
    application.processEvents()
