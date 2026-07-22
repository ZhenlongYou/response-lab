"""从已经折叠好的虚拟眼轨迹中测量轻量眼高和眼宽。

这里不负责 CDR、符号恢复或眼图折叠。调用方已经知道每条轨迹的中心符号标签，
因此可以直接复用眼图库的“相邻电平分位数 + 水平切片 crossing”测量口径，同时
保持 ResponseLab 的纯 NumPy、确定性依赖边界。
"""

# 新人导向逐语句注释会打断导入块并含较长中文，仅关闭对应排版告警。
# ruff: noqa: E501, I001

# Codex说明(自动生成)： 从 __future__ 导入 annotations，启用较新的类型标注行为，减少运行期导入或前向引用问题。
from __future__ import annotations

# Codex说明(自动生成)： 从 dataclasses 导入 dataclass，声明轻量数据结构并减少样板初始化代码。
from dataclasses import dataclass

# Codex说明(自动生成)： 导入 numpy as np，执行数组、向量化和数值仿真计算。
import numpy as np
# Codex说明(自动生成)： 从 numpy.typing 导入 NDArray，执行数组、向量化和数值仿真计算。
from numpy.typing import NDArray

# Codex说明(自动生成)： 计算并保存 FloatArray，供后续语句继续读取或更新。
FloatArray = NDArray[np.float64]


# Codex说明(自动生成)： 定义 VirtualEyeOpenings 类，把相关数据结构、校验规则或操作方法组织在一起。
@dataclass(frozen=True)
class VirtualEyeOpenings:
    """每个相邻发送电平之间的眼开口，顺序由低眼到高眼。"""

    # Codex说明(自动生成)： 声明并保存 eye_heights_v，同时保留类型信息方便维护和静态检查。
    eye_heights_v: tuple[float, ...]
    # Codex说明(自动生成)： 声明并保存 eye_widths_ui，同时保留类型信息方便维护和静态检查。
    eye_widths_ui: tuple[float, ...]


# Codex说明(自动生成)： 定义函数 measure_virtual_eye_openings，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def measure_virtual_eye_openings(
    traces: FloatArray,
    x_ui: FloatArray,
    labels: NDArray[np.integer],
    nominal_levels: FloatArray,
    *,
    opening_probability: float = 0.01,
    measure_width: bool = True,
) -> VirtualEyeOpenings:
    """测量固定中心处的眼高和最大水平开口。

    ``nominal_levels`` 只定义标签顺序和眼数；眼宽切片幅值来自 0 UI 各标签轨迹的
    实测中位数。``opening_probability`` 是有限轨迹数据库的一侧经验分位数，不是
    BER/SER 外推，样本支撑由调用方保证。设为 0 时使用确定性的最坏轨迹边界。批量
    只扫眼高时可关闭 ``measure_width``，避免重复执行 41 条水平切片。
    """

    # Codex说明(自动生成)： 计算并保存 (trace_array, x_array, label_array, level_array, probab...，供后续语句继续读取或更新。
    trace_array, x_array, label_array, level_array, probability = _validate_inputs(
        traces,
        x_ui,
        labels,
        nominal_levels,
        opening_probability,
    )
    # 输入合同要求横轴显式包含 0 UI，因此固定中心不会被静默量化到邻近样点。
    center_index = int(np.flatnonzero(x_array == 0.0)[0])
    # Codex说明(自动生成)： 计算并保存 center_values，供后续语句继续读取或更新。
    center_values = trace_array[:, center_index]

    # 标签来自虚拟眼的已知发送符号，不能在每个采样列重新聚类；否则闭眼也可能
    # 被重新分成上下两组并制造出虚假的正开口。
    rail_samples = [center_values[label_array == index] for index in range(level_array.size)]
    # 各已知发送轨在固定 0 UI 的中位数只用于确定眼宽的水平切片范围。
    rail_centers = np.asarray([np.median(values) for values in rail_samples], dtype=float)

    # 眼高输出始终计算，因为固定中心分位数相对 crossing 扫描开销很小。
    heights: list[float] = []
    # 眼宽关闭时仍按眼数返回 NaN，使结果形状稳定且明确表示“未计算”。
    widths: list[float] = []
    # 一次循环对应一对相邻发送电平，即 NRZ 一只眼或 PAM4 三只眼之一。
    for lower_index in range(level_array.size - 1):
        # 较低发送电平的中心样本用于取靠眼内侧的上分位边界。
        lower_samples = rail_samples[lower_index]
        # 较高发送电平的中心样本用于取靠眼内侧的下分位边界。
        upper_samples = rail_samples[lower_index + 1]

        # 眼高保留符号：负值明确表示两个经验轨迹包络已经重叠。
        lower_boundary = float(np.quantile(lower_samples, 1.0 - probability))
        # Codex说明(自动生成)： 计算并保存 upper_boundary，供后续语句继续读取或更新。
        upper_boundary = float(np.quantile(upper_samples, probability))
        # 上轨内侧边界减下轨内侧边界得到有符号眼高。
        heights.append(upper_boundary - lower_boundary)

        # 纯眼高批量扫描不需要昂贵的 41 阈值 crossing；NaN 不会进入眼高排名。
        if not measure_width:
            # 每只跳过的眼仍占一个位置，保持 NRZ/PAM4 输出维度不变。
            widths.append(float("nan"))
            # 直接进入下一对相邻电平，不计算 rail 中心或 crossing。
            continue
        # 低轨中位数是水平阈值扫描的下边界参考。
        lower_center = float(rail_centers[lower_index])
        # 高轨中位数是水平阈值扫描的上边界参考。
        upper_center = float(rail_centers[lower_index + 1])
        # rail 中位数顺序反转说明中心眼已退化，无法定义正水平切片区间。
        if upper_center <= lower_center:
            # 中心轨道次序反转时没有合法水平扫描区间，应报告不可测而非伪造零宽。
            widths.append(float("nan"))
            # 其余 PAM4 眼仍可独立测量，不因一只眼关闭而整体终止。
            continue
        # 正常眼在两轨中位数之间执行本地眼图库的最大水平切片测量。
        widths.append(
            _measure_one_eye_width(
                trace_array,
                x_array,
                lower_center,
                upper_center,
                probability,
            )
        )

    # 不可变元组防止候选排名后被绘图层原地改写。
    return VirtualEyeOpenings(tuple(heights), tuple(widths))


# Codex说明(自动生成)： 定义函数 _measure_one_eye_width，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _measure_one_eye_width(
    traces: FloatArray,
    x_ui: FloatArray,
    lower_center: float,
    upper_center: float,
    opening_probability: float,
) -> float:
    """在相邻 rail 的 5%--95% 高度切片中寻找最大经验眼宽。"""

    # Codex说明(自动生成)： 计算并保存 level_gap，供后续语句继续读取或更新。
    level_gap = upper_center - lower_center
    # Codex说明(自动生成)： 计算并保存 thresholds，供后续语句继续读取或更新。
    thresholds = np.linspace(
        lower_center + 0.05 * level_gap,
        upper_center - 0.05 * level_gap,
        41,
    )
    # 分位边界至少要有约 1/p 个真实 crossing，才能让尾部概率具有一个事件的
    # 基本分辨率；同时保留五事件和总轨迹 1% 的最低支撑。p=0 的确定性包络只需
    # 最坏事件，因此沿用原来的五事件门槛。
    quantile_support = (
        1
        if opening_probability == 0.0
        else int(np.ceil(1.0 / opening_probability))
    )
    minimum_crossing_count = max(
        5,
        quantile_support,
        int(np.ceil(0.01 * traces.shape[0])),
    )
    # 41 条切片按受控批次同时求 crossing，减少 QThread 与主线程争用 Python GIL。
    left_crossings, right_crossings = _select_innermost_crossings_many(
        traces,
        x_ui,
        thresholds,
    )
    # 每个阈值独立统计左侧有效事件数。
    left_counts = np.count_nonzero(np.isfinite(left_crossings), axis=1)
    # 右侧计数不要求与左侧来自同一条轨迹。
    right_counts = np.count_nonzero(np.isfinite(right_crossings), axis=1)
    # 只有左右两侧均有足够经验支撑的水平切片才可参与最大眼宽选择。
    supported = (left_counts >= minimum_crossing_count) & (
        right_counts >= minimum_crossing_count
    )
    # 所有切片都缺少 crossing 时，眼宽没有可解释的正开口。
    if not np.any(supported):
        # 本地眼图库把 crossing 证据不足标为 unavailable；NaN 保留这一语义。
        return float("nan")
    # 左侧越靠右越保守；一次沿轨迹轴求分位数，避免 41 次 Python 循环。
    left_boundaries = np.nanquantile(
        left_crossings[supported],
        1.0 - opening_probability,
        axis=1,
    )
    # 右侧越靠左越保守；p=0 自然退化为每行最小值。
    right_boundaries = np.nanquantile(
        right_crossings[supported],
        opening_probability,
        axis=1,
    )
    # 每个受支持水平切片的内侧开口由右边界减左边界得到。
    candidate_widths = right_boundaries - left_boundaries
    # 与本地眼图库保持一致：只禁止负开口，不把有限轨迹分位数人为截到 1 UI。
    return float(max(0.0, np.max(candidate_widths)))


# Codex说明(自动生成)： 定义函数 _select_innermost_crossings，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _select_innermost_crossings(
    traces: FloatArray,
    x_ui: FloatArray,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    """向量化返回每条轨迹在零点左右最靠内的 crossing。

    无 crossing 的位置返回 NaN。等于阈值的样点直接作为 crossing；只有两个非零
    端点严格异号时才在线段内插值，因而不会漏掉 exact-zero，也不会把一段零平台
    附近的同一 crossing 错算到相反方向。
    """

    # 单阈值公共测试入口复用正式批量实现，防止两套 crossing 规则漂移。
    left, right = _select_innermost_crossings_many(
        traces,
        x_ui,
        np.asarray([threshold], dtype=float),
    )
    # 批量结果第一维只有当前一个阈值，去掉该维后保持旧的一维接口。
    return left[0], right[0]


# Codex说明(自动生成)： 定义函数 _select_innermost_crossings_many，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _select_innermost_crossings_many(
    traces: FloatArray,
    x_ui: FloatArray,
    thresholds: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """分批向量化返回多个水平切片的逐轨迹左右内 crossing。"""

    # 阈值数组始终使用 float64 一维语义，41 条正式切片也走同一路径。
    threshold_array = np.asarray(thresholds, dtype=float)
    # 空阈值没有可测切片，返回形状稳定的零行矩阵。
    if threshold_array.ndim != 1 or threshold_array.size == 0:
        # 内部只传有限非空数组；显式错误能防止未来误用悄悄生成空眼宽。
        raise ValueError("眼宽阈值必须是一维有限非空数组")
    # NaN 或 Inf 阈值无法形成真实电压水平切片。
    if not np.all(np.isfinite(threshold_array)):
        # 在分配三维批次缓冲前拒绝坏阈值。
        raise ValueError("眼宽阈值必须是一维有限非空数组")
    # 与 eye_diagram_engine 一样裁掉显示窗两端各 1%，防止折叠边界的重复样点
    # 被误认成中心眼边界；本轻量内核的眼中心合同固定为 x=0。
    span = float(x_ui[-1] - x_ui[0])
    # 左侧窗口从 -1 UI 端点内缩 1%，终止在固定眼中心 0 UI。
    left_limit = float(x_ui[0] + 0.01 * span)
    # 右侧窗口从 0 UI 开始，并在 +1 UI 端点前内缩 1%。
    right_limit = float(x_ui[-1] - 0.01 * span)
    # 先取得左窗口的连续样点索引，少于两点时无法定义线性 crossing。
    left_indices = np.flatnonzero((x_ui >= left_limit) & (x_ui <= 0.0))
    # 右窗口独立取索引，使两侧都只处理约半条轨迹并降低扫频内存带宽。
    right_indices = np.flatnonzero((x_ui >= 0.0) & (x_ui <= right_limit))
    # 输入验证保证 x 轴覆盖零点；防御检查仍避免极低 M 时索引空窗。
    if left_indices.size < 2 or right_indices.size < 2:
        # 没有完整左右线段时，每个阈值、每条轨迹都明确返回无 crossing。
        empty = np.full(
            (threshold_array.size, traces.shape[0]),
            np.nan,
            dtype=float,
        )
        # 两个独立副本防止调用方原地修改一侧时污染另一侧。
        return empty, empty.copy()
    # 单个批次最多约四百万个“阈值×轨迹×样点”元素，限制临时缓冲峰值。
    per_threshold_elements = max(1, traces.shape[0] * traces.shape[1])
    # 小 M 可一次处理全部 41 条；大 M 自动拆批，避免峰值内存随 UI 分辨率失控。
    batch_size = max(
        1,
        min(threshold_array.size, 4_000_000 // per_threshold_elements),
    )
    # 左、右输出按批次积累，最终只保留约 2*T*N 个 crossing 浮点数。
    left_batches: list[FloatArray] = []
    # 右侧单独积累，左右的大型三维临时数组不会跨批次保留。
    right_batches: list[FloatArray] = []
    # 每个批次同时处理一组固定判决电压，显著减少后台线程的 Python 重入次数。
    for batch_start in range(0, threshold_array.size, batch_size):
        # 半开终点把最后一个不足整批的阈值也完整纳入。
        batch_stop = min(batch_start + batch_size, threshold_array.size)
        # 当前批次阈值扩展成 B×1×1，与 N×L 轨迹广播相减。
        batch_thresholds = threshold_array[batch_start:batch_stop]
        # 三维平移数组只在本批次存活，后续可安全规范精确零的符号位。
        shifted = np.asarray(traces, dtype=float)[np.newaxis, :, :] - batch_thresholds[
            :, np.newaxis, np.newaxis
        ]
        # 精确命中阈值的样点统一使用正零；相切命中仍由 exact-zero 分支保留。
        shifted[shifted == 0.0] = 0.0
        # 左侧为每个“阈值×轨迹”选择最靠近 0 UI 的最大 crossing。
        left_batches.append(
            _select_side_crossings_many(
                shifted,
                x_ui,
                start=int(left_indices[0]),
                stop=int(left_indices[-1]),
                choose_maximum=True,
            )
        )
        # 右侧镜像选择最靠近 0 UI 的最小 crossing。
        right_batches.append(
            _select_side_crossings_many(
                shifted,
                x_ui,
                start=int(right_indices[0]),
                stop=int(right_indices[-1]),
                choose_maximum=False,
            )
        )
    # 按原阈值顺序拼接各批左 crossing，形状为 T×N。
    left = np.concatenate(left_batches, axis=0)
    # 右 crossing 使用同一批次边界，逐行与左侧同阈值对齐。
    right = np.concatenate(right_batches, axis=0)
    # 返回每个水平切片的逐轨迹左右边界；无命中元素保留 NaN。
    return left, right


# Codex说明(自动生成)： 定义函数 _select_side_crossings_many，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _select_side_crossings_many(
    shifted_traces: NDArray[np.float64],
    x_ui: FloatArray,
    *,
    start: int,
    stop: int,
    choose_maximum: bool,
) -> FloatArray:
    """在一侧窗口内选择每个阈值、每条轨迹最靠近眼中心的 crossing。"""

    # 只切一侧窗口，避免为完整 2 UI 轨迹分配更多三维临时数组。
    side_values = shifted_traces[:, :, start : stop + 1]
    # 相邻样点符号位变化与本地眼图库的普通 crossing 判据一致。
    sign_changes = np.signbit(side_values[:, :, :-1]) != np.signbit(
        side_values[:, :, 1:]
    )
    # 每个 crossing 的左端幅值用于求线性插值比例。
    y0 = side_values[:, :, :-1]
    # 右端幅值与左端共同确定阈值穿越位置。
    y1 = side_values[:, :, 1:]
    # 非 crossing 线段的输出保持零，后续掩码会把它们替换成无效候选。
    fractions = np.divide(
        -y0,
        y1 - y0,
        out=np.zeros_like(y0),
        where=sign_changes,
    )
    # 小数位置保留样点之间的亚采样 crossing，不量化到 1/M UI。
    interpolated = x_ui[start:stop][np.newaxis, np.newaxis, :] + fractions * np.diff(
        x_ui[start : stop + 1]
    )[np.newaxis, np.newaxis, :]
    # 精确等于阈值的样点即使只是相切也应作为水平切片命中。
    exact_hits = side_values == 0.0
    # 最大选择用于左侧；无命中默认负无穷，便于一次按行归约。
    if choose_maximum:
        # 普通异号 crossing 中取最靠右的位置。
        interpolated_choice = np.max(
            np.where(sign_changes, interpolated, -np.inf),
            axis=2,
        )
        # exact-zero 命中同样取最靠右的样点位置。
        exact_choice = np.max(
            np.where(
                exact_hits,
                x_ui[start : stop + 1][np.newaxis, np.newaxis, :],
                -np.inf,
            ),
            axis=2,
        )
        # 两类候选再取最大，得到左侧最靠近 0 UI 的唯一边界。
        selected = np.maximum(interpolated_choice, exact_choice)
        # 负无穷表示两类候选都不存在，转换成上层可过滤的 NaN。
        selected[selected == -np.inf] = np.nan
    # 最小选择用于右侧；逻辑与左侧镜像。
    else:
        # 普通异号 crossing 中取最靠左的位置。
        interpolated_choice = np.min(
            np.where(sign_changes, interpolated, np.inf),
            axis=2,
        )
        # exact-zero 命中同样取最靠左的样点位置。
        exact_choice = np.min(
            np.where(
                exact_hits,
                x_ui[start : stop + 1][np.newaxis, np.newaxis, :],
                np.inf,
            ),
            axis=2,
        )
        # 两类候选再取最小，得到右侧最靠近 0 UI 的唯一边界。
        selected = np.minimum(interpolated_choice, exact_choice)
        # 正无穷表示两类候选都不存在，转换成上层可过滤的 NaN。
        selected[selected == np.inf] = np.nan
    # 结果保持 B×N float64 且不额外复制，供上层一次统计 41 条切片分位数。
    return selected.astype(float, copy=False)


# Codex说明(自动生成)： 定义函数 _validate_inputs，把一段可复用的业务步骤、计算过程或入口逻辑封装起来。
def _validate_inputs(
    traces: FloatArray,
    x_ui: FloatArray,
    labels: NDArray[np.integer],
    nominal_levels: FloatArray,
    opening_probability: float,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64], FloatArray, float]:
    """集中验证数组形状、有限性及发送电平合同。"""

    # Codex说明(自动生成)： 计算并保存 trace_array，供后续语句继续读取或更新。
    trace_array = np.asarray(traces, dtype=float)
    # Codex说明(自动生成)： 计算并保存 x_array，供后续语句继续读取或更新。
    x_array = np.asarray(x_ui, dtype=float)
    # Codex说明(自动生成)： 计算并保存 raw_labels，供后续语句继续读取或更新。
    raw_labels = np.asarray(labels)
    # Codex说明(自动生成)： 计算并保存 level_array，供后续语句继续读取或更新。
    level_array = np.asarray(nominal_levels, dtype=float)

    # Codex说明(自动生成)： 检查条件 trace_array.ndim != 2 or trace_array.shape[0] == 0 or t...，根据结果选择后续执行路径。
    if trace_array.ndim != 2 or trace_array.shape[0] == 0 or trace_array.shape[1] < 3:
        # Codex说明(自动生成)： 抛出 ValueError('traces 必须是至少含一条、每条至少三个样点的二维数组')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("traces 必须是至少含一条、每条至少三个样点的二维数组")
    # Codex说明(自动生成)： 检查条件 not np.all(np.isfinite(trace_array))，根据结果选择后续执行路径。
    if not np.all(np.isfinite(trace_array)):
        # Codex说明(自动生成)： 抛出 ValueError('traces 必须全部为有限值')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("traces 必须全部为有限值")
    # Codex说明(自动生成)： 检查条件 x_array.ndim != 1 or x_array.size != trace_array.shape[1]，根据结果选择后续执行路径。
    if x_array.ndim != 1 or x_array.size != trace_array.shape[1]:
        # Codex说明(自动生成)： 抛出 ValueError('x_ui 必须是一维数组且长度与每条轨迹相同')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("x_ui 必须是一维数组且长度与每条轨迹相同")
    # Codex说明(自动生成)： 检查条件 not np.all(np.isfinite(x_array)) or not np.all(np.diff(...，根据结果选择后续执行路径。
    if not np.all(np.isfinite(x_array)) or not np.all(np.diff(x_array) > 0.0):
        # Codex说明(自动生成)： 抛出 ValueError('x_ui 必须有限且严格递增')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("x_ui 必须有限且严格递增")
    # Codex说明(自动生成)： 检查条件 not x_array[0] < 0.0 < x_array[-1]，根据结果选择后续执行路径。
    if not x_array[0] < 0.0 < x_array[-1]:
        # Codex说明(自动生成)： 抛出 ValueError('x_ui 必须同时覆盖 x=0 左右两侧')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("x_ui 必须同时覆盖 x=0 左右两侧")
    # Codex说明(自动生成)： 检查条件 not np.any(x_array == 0.0)，根据结果选择后续执行路径。
    if not np.any(x_array == 0.0):
        # Codex说明(自动生成)： 抛出 ValueError('x_ui 必须明确包含固定测量中心 x=0')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("x_ui 必须明确包含固定测量中心 x=0")

    # Codex说明(自动生成)： 检查条件 raw_labels.ndim != 1 or raw_labels.size != trace_array....，根据结果选择后续执行路径。
    if raw_labels.ndim != 1 or raw_labels.size != trace_array.shape[0]:
        # Codex说明(自动生成)： 抛出 ValueError('labels 必须是一维数组且每条轨迹恰有一个标签')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("labels 必须是一维数组且每条轨迹恰有一个标签")
    # Codex说明(自动生成)： 检查条件 raw_labels.dtype.kind not in 'iu'，根据结果选择后续执行路径。
    if raw_labels.dtype.kind not in "iu":
        # Codex说明(自动生成)： 抛出 ValueError('labels 必须使用整数电平编号')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("labels 必须使用整数电平编号")
    # Codex说明(自动生成)： 计算并保存 label_array，供后续语句继续读取或更新。
    label_array = raw_labels.astype(np.int64, copy=False)

    # Codex说明(自动生成)： 检查条件 level_array.ndim != 1 or level_array.size < 2，根据结果选择后续执行路径。
    if level_array.ndim != 1 or level_array.size < 2:
        # Codex说明(自动生成)： 抛出 ValueError('nominal_levels 必须至少包含两个有序电平')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("nominal_levels 必须至少包含两个有序电平")
    # Codex说明(自动生成)： 检查条件 not np.all(np.isfinite(level_array)) or not np.all(np.d...，根据结果选择后续执行路径。
    if not np.all(np.isfinite(level_array)) or not np.all(np.diff(level_array) > 0.0):
        # Codex说明(自动生成)： 抛出 ValueError('nominal_levels 必须有限且严格递增')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("nominal_levels 必须有限且严格递增")
    # Codex说明(自动生成)： 检查条件 np.any(label_array < 0) or np.any(label_array >= level_...，根据结果选择后续执行路径。
    if np.any(label_array < 0) or np.any(label_array >= level_array.size):
        # Codex说明(自动生成)： 抛出 ValueError('labels 必须位于 0 到 K-1')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("labels 必须位于 0 到 K-1")
    # Codex说明(自动生成)： 检查条件 np.unique(label_array).size != level_array.size，根据结果选择后续执行路径。
    if np.unique(label_array).size != level_array.size:
        # Codex说明(自动生成)： 抛出 ValueError('labels 必须至少包含每个 nominal level 的一条轨迹')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("labels 必须至少包含每个 nominal level 的一条轨迹")

    # Codex说明(自动生成)： 检查条件 isinstance(opening_probability, (bool, np.bool_))，根据结果选择后续执行路径。
    if isinstance(opening_probability, (bool, np.bool_)):
        # Codex说明(自动生成)： 抛出 ValueError('opening_probability 必须位于 [0, 0.5)')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("opening_probability 必须位于 [0, 0.5)")
    # Codex说明(自动生成)： 开始执行可能失败的代码块，并把异常、收尾或兜底逻辑交给后续分支处理。
    try:
        # Codex说明(自动生成)： 计算并保存 probability，供后续语句继续读取或更新。
        probability = float(opening_probability)
    # Codex说明(自动生成)： 捕获 (TypeError, ValueError)，执行对应的恢复、记录或重新报错逻辑。
    except (TypeError, ValueError) as exc:
        # Codex说明(自动生成)： 抛出 ValueError('opening_probability 必须位于 [0, 0.5)')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("opening_probability 必须位于 [0, 0.5)") from exc
    # Codex说明(自动生成)： 检查条件 not np.isfinite(probability) or not 0.0 <= probability ...，根据结果选择后续执行路径。
    if not np.isfinite(probability) or not 0.0 <= probability < 0.5:
        # Codex说明(自动生成)： 抛出 ValueError('opening_probability 必须位于 [0, 0.5)')，明确提示输入、状态或处理流程无法继续。
        raise ValueError("opening_probability 必须位于 [0, 0.5)")

    # Codex说明(自动生成)： 返回 (trace_array, x_array, label_array, level_array, probab...，让调用方取得本函数的处理结果。
    return trace_array, x_array, label_array, level_array, probability


# Codex说明(自动生成)： 计算并保存 __all__，供后续语句继续读取或更新。
__all__ = ["VirtualEyeOpenings", "measure_virtual_eye_openings"]
