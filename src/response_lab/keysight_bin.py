"""Keysight Infiniium AG BIN 文件的严格扫描、加载与导出。"""

from __future__ import annotations

# math 负责验证采样间隔、时间原点和采样率均为有限物理量。
import math

# operator.index 接受真正的整数索引，同时拒绝会被 int 静默截断的浮点数。
import operator

# os 提供原子替换、落盘同步和临时文件描述符封装。
import os

# struct 用显式 little-endian 格式解码 Keysight 固定宽度字段。
import struct

# tempfile 在目标目录中建立同文件系统临时文件，保证 os.replace 原子提交。
import tempfile

# nullcontext 让公共扫描和加载复用同一已打开文件；suppress 只用于失败清理。
from contextlib import nullcontext, suppress

# dataclass 用不可变描述符保存文件头信息，调用方不会意外改变 payload 边界。
from dataclasses import dataclass

# Path 为公开接口统一字符串与路径对象，并保留源文件身份。
from pathlib import Path

# NumPy 类型只用于后续公开加载结果，inspect 阶段不会创建 payload 数组。
import numpy as np
from numpy.typing import NDArray

# 官方 File Header 固定为 cookie、版本、总长度和波形数量，共 12 字节。
_FILE_HEADER = struct.Struct("<2s2sii")
# 官方 Waveform Header 的已知基准字段共 140 字节，Header Size 可声明尾部扩展。
_WAVEFORM_HEADER = struct.Struct("<iiiiifdddii16s16s24s16sdI")
# 每个 waveform buffer 的已知数据头共 12 字节，之后紧跟 Buffer Size 字节。
_DATA_HEADER = struct.Struct("<ihhi")
# 当前模块只把 type 1 的 little-endian float32 buffer 当作模拟电压序列。
_NORMAL_FLOAT_BUFFER_TYPE = 1
# Normal 与 Average 是本模块允许加载的两种时域模拟波形类型。
_LOADABLE_WAVEFORM_TYPES = frozenset({1, 3})
# Keysight 单位枚举中 2 表示秒，1 表示伏特。
_SECONDS_UNIT = 2
_VOLTS_UNIT = 1
# Keysight 头字段由有符号 int32 承载点数和字节长度，writer 不允许溢出。
_INT32_MAX = 2_147_483_647
# 每次最多转换约 4 MiB float32 payload，控制大型导出的临时峰值内存。
_WRITE_CHUNK_POINTS = 1_048_576


class KeysightBinError(ValueError):
    """表示文件不满足受支持的 Keysight AG BIN 合同。"""


@dataclass(frozen=True)
class KeysightBufferInfo:
    """一个 waveform buffer 的头部元数据和磁盘 payload 边界。"""

    # header_size_bytes 包含 12 字节基准头和可能存在的扩展字段。
    header_size_bytes: int
    # buffer_type 使用 Keysight 官方枚举，例如 1=normal float32。
    buffer_type: int
    # bytes_per_point 是 payload 中单个数据点的存储字节数。
    bytes_per_point: int
    # buffer_size_bytes 是紧跟数据头的完整 payload 长度。
    buffer_size_bytes: int
    # data_offset 是 payload 相对于文件起点的绝对字节偏移。
    data_offset: int

    @property
    def point_count(self) -> int | None:
        """在字节数可整除时返回 buffer 点数，否则保留不可解释状态。"""

        # 非正每点字节数不能参与除法，也不能被误标为零点 buffer。
        if self.bytes_per_point <= 0:
            return None
        # 残余字节说明 payload 不能由完整数据点组成。
        if self.buffer_size_bytes % self.bytes_per_point:
            return None
        # 整除结果只来自头部整数，不会触碰 payload。
        return self.buffer_size_bytes // self.bytes_per_point


@dataclass(frozen=True)
class KeysightWaveformInfo:
    """一个 waveform 的可展示元数据与全部 buffer 描述符。"""

    index: int
    header_size_bytes: int
    waveform_type: int
    points: int
    count: int
    x_display_range: float
    x_display_origin: float
    x_increment_s: float
    x_origin_s: float
    x_units: int
    y_units: int
    date: str
    time: str
    frame: str
    label: str
    time_tag: float
    segment_index: int
    buffers: tuple[KeysightBufferInfo, ...]

    @property
    def normal_float32_buffer(self) -> KeysightBufferInfo | None:
        """返回唯一、尺寸与 Points 一致的 normal float32 buffer。"""

        # 只筛选官方 type 1、每点四字节且点数可以完整解释的候选。
        candidates = tuple(
            buffer
            for buffer in self.buffers
            if buffer.buffer_type == _NORMAL_FLOAT_BUFFER_TYPE
            and buffer.bytes_per_point == 4
            and buffer.point_count == self.points
        )
        # 多个 normal buffer 仍然具有歧义，不能由加载器静默选择。
        if len(candidates) != 1:
            return None
        # 唯一候选的 offset 和 size 可以供 load 阶段直接映射。
        return candidates[0]

    @property
    def unsupported_reason(self) -> str | None:
        """返回不能作为均匀时域伏特波形加载的首个明确原因。"""

        # 仅 Normal/Average 具有本工具需要的单值时域语义。
        if self.waveform_type not in _LOADABLE_WAVEFORM_TYPES:
            return f"不支持的 Waveform Type：{self.waveform_type}"
        # 频域分析必须由秒单位的 X Increment 推导采样率。
        if self.x_units != _SECONDS_UNIT:
            return f"X Units 必须为 Second(2)，实际为 {self.x_units}"
        # 当前 Vpp/补偿链路只接受伏特幅值，不猜测其他单位换算。
        if self.y_units != _VOLTS_UNIT:
            return f"Y Units 必须为 Volt(1)，实际为 {self.y_units}"
        # 零、负数、NaN 和 Inf 都不能形成有效采样率。
        if not math.isfinite(self.x_increment_s) or self.x_increment_s <= 0.0:
            return "X Increment 必须是正的有限秒数"
        # 极小正间隔的倒数仍可能溢出为 Inf，不能进入频域算法。
        if self.sample_rate_hz is None:
            return "X Increment 推导的采样率必须是正的有限 Hz 数值"
        # 时间原点必须能安全生成有限的隐式时间轴。
        if not math.isfinite(self.x_origin_s):
            return "X Origin 必须是有限秒数"
        # 本模块有意拒绝 Peak Detect 的双 buffer 和其他复合布局。
        if len(self.buffers) != 1:
            return f"每个可加载 waveform 必须只有 1 个 buffer，实际为 {len(self.buffers)}"
        # 唯一 buffer 必须同时满足 type、宽度、整除和 Points 一致性。
        if self.normal_float32_buffer is None:
            return "唯一 buffer 必须是与 Points 一致的 normal float32 数据"
        # 所有支持条件闭合后，调用方可以安全加载 payload。
        return None

    @property
    def is_loadable(self) -> bool:
        """说明当前 waveform 是否满足严格模拟时域加载合同。"""

        # 无拒绝原因即表示全部公开格式条件已经通过。
        return self.unsupported_reason is None

    @property
    def sample_rate_hz(self) -> float | None:
        """从有效 X Increment 推导采样率，非法间隔返回 None。"""

        # 非秒 X 轴的 increment 可能表示 Hz/bin 等量，倒数不能标成 samples/s。
        if self.x_units != _SECONDS_UNIT:
            return None
        # inspect 也会展示不受支持 waveform，因此只对正有限时间间隔计算倒数。
        if not math.isfinite(self.x_increment_s) or self.x_increment_s <= 0.0:
            return None
        # 秒/点的倒数是点/秒；极小正间隔仍可能让倒数溢出。
        sample_rate_hz = 1.0 / self.x_increment_s
        if not math.isfinite(sample_rate_hz):
            return None
        # 只有正有限倒数才可以对外标记为 Hz。
        return sample_rate_hz


@dataclass(frozen=True)
class KeysightBinInfo:
    """一次纯头部扫描得到的文件级索引。"""

    path: Path
    version: str
    declared_file_size: int
    actual_file_size: int
    waveforms: tuple[KeysightWaveformInfo, ...]


@dataclass(frozen=True)
class KeysightWaveform:
    """一个已选择 waveform 的电压值和隐式均匀时间元数据。"""

    path: Path
    waveform_index: int
    values: NDArray[np.float32]
    sample_rate_hz: float
    x_increment_s: float
    x_origin_s: float
    label: str
    segment_index: int


def _read_exact(handle: object, size: int, *, field: str) -> bytes:
    """读取固定字节数，截断时给出当前格式字段名称。"""

    # 二进制文件对象提供 read；独立封装使所有短读都采用同一失败语义。
    data = handle.read(size)
    # 少一个字节也会使后续字段错位，因此必须立即拒绝。
    if len(data) != size:
        raise KeysightBinError(f"Keysight BIN 在 {field} 处截断")
    # 返回精确长度字节供 Struct 解包。
    return data


def _decode_fixed_text(raw: bytes, *, field: str) -> str:
    """解码 NUL 补齐的 Keysight 文本字段。"""

    # 第一个 NUL 后属于固定数组填充，不应出现在用户可见标签中。
    payload = raw.split(b"\0", 1)[0]
    try:
        # 官方 Python 示例使用 UTF-8；ASCII 元数据是其严格子集。
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        # 不以替换字符掩盖损坏 label 或 frame。
        raise KeysightBinError(f"Keysight BIN 的 {field} 不是有效 UTF-8") from error


def _encode_fixed_text(text: str, *, size: int, field: str) -> bytes:
    """把公开字符串编码为不截断的 Keysight 定长 UTF-8 字段。"""

    # NUL 会让官方 C 字符数组提前结束，禁止 label 中混入隐藏后缀。
    if "\0" in text:
        raise KeysightBinError(f"{field} 不能包含 NUL 字符")
    # 官方 Python 示例按 UTF-8 解码，因此 writer 使用相同编码。
    encoded = text.encode("utf-8")
    # 固定数组没有额外长度字段，溢出必须报错而不是截断多字节字符。
    if len(encoded) > size:
        raise KeysightBinError(f"{field} 的 UTF-8 编码不能超过 {size} 字节")
    # NUL 补齐到精确字段宽度，保证后续 double/int 偏移稳定。
    return encoded.ljust(size, b"\0")


def _skip_extension(handle: object, *, base_size: int, declared_size: int, field: str) -> None:
    """按 Header Size 跳过未知扩展字段，同时拒绝缩短的旧头。"""

    # 小于已知基准会让固定字段互相重叠，不能继续猜测布局。
    if declared_size < base_size:
        raise KeysightBinError(f"{field} Header Size 小于 {base_size} 字节")
    # seek 只移动到扩展字段末尾，不把可能很大的扩展头载入内存。
    handle.seek(declared_size - base_size, 1)


def _inspect_keysight_bin_from_open_file(
    source_path: Path,
    opened_file: object,
) -> KeysightBinInfo:
    """在调用方持有的同一个只读文件上完成严格头部扫描。"""

    # 保留原有单一缩进扫描体，同时明确不在内部关闭调用方拥有的文件。
    with nullcontext(opened_file) as handle:
        # 实际文件长度是 Infiniium 2026 file_size=0 情况下的唯一边界真值。
        handle.seek(0, 2)
        actual_file_size = handle.tell()
        # 回到开头读取唯一 File Header。
        handle.seek(0)
        file_header = _read_exact(handle, _FILE_HEADER.size, field="File Header")
        # 显式小端解包，禁止随运行平台改变字节序。
        cookie, raw_version, declared_file_size, waveform_count = _FILE_HEADER.unpack(file_header)
        # AG cookie 是区分 Keysight 容器与裸二进制的强制签名。
        if cookie != b"AG":
            raise KeysightBinError("不是 Keysight AG BIN：Cookie 必须为 AG")
        # 本实现的固定字段布局只来自官方 AG10 文档，未知版本不得套用相同偏移。
        if raw_version != b"10":
            raise KeysightBinError("不支持的 Keysight BIN Version：仅支持 AG10")
        # 当前格式允许历史真实长度和 Infiniium 2026 的零占位两种写法。
        if declared_file_size not in {0, actual_file_size}:
            raise KeysightBinError(
                "Keysight BIN 的 File Size 与实际文件长度不一致"
            )
        # 负数或零个 waveform 都不能形成可选择的时域记录。
        if waveform_count <= 0:
            raise KeysightBinError("Keysight BIN 的 Number of Waveforms 必须为正数")
        # 每个 waveform 至少需要一个基准波形头和一个基准数据头；先拦截恶意巨大计数。
        minimum_waveform_bytes = _WAVEFORM_HEADER.size + _DATA_HEADER.size
        if waveform_count > (actual_file_size - _FILE_HEADER.size) // minimum_waveform_bytes:
            raise KeysightBinError("Keysight BIN 的 Number of Waveforms 超出文件边界")
        # 已验证的两字节版本按官方脚本 UTF-8 规则解码，供调用方展示。
        version = _decode_fixed_text(raw_version, field="Version")
        # 按文件顺序收集每个 waveform 描述符，不触碰 payload 内容。
        waveforms: list[KeysightWaveformInfo] = []
        for waveform_index in range(waveform_count):
            # 先读取 140 字节已知字段，Header Size 决定其后是否还有扩展。
            raw_waveform_header = _read_exact(
                handle,
                _WAVEFORM_HEADER.size,
                field=f"Waveform Header {waveform_index}",
            )
            # 单次解包保持字段偏移与官方表格一致。
            (
                waveform_header_size,
                waveform_type,
                buffer_count,
                points,
                count,
                x_display_range,
                x_display_origin,
                x_increment,
                x_origin,
                x_units,
                y_units,
                raw_date,
                raw_time,
                raw_frame,
                raw_label,
                time_tag,
                segment_index,
            ) = _WAVEFORM_HEADER.unpack(raw_waveform_header)
            # 负点数或零点无法与任何 payload 建立可信长度关系。
            if points <= 0:
                raise KeysightBinError(f"Waveform {waveform_index} 的 Points 必须为正数")
            # 至少一个 buffer 才能承载 waveform 数据。
            if buffer_count <= 0:
                raise KeysightBinError(
                    f"Waveform {waveform_index} 的 Number of Buffers 必须为正数"
                )
            # 按声明大小跳过未来版本添加在基准字段之后的扩展头。
            _skip_extension(
                handle,
                base_size=_WAVEFORM_HEADER.size,
                declared_size=waveform_header_size,
                field=f"Waveform {waveform_index}",
            )
            # 每个循环读取一个数据头并仅通过 seek 越过 payload。
            buffers: list[KeysightBufferInfo] = []
            for buffer_index in range(buffer_count):
                # 读取当前 buffer 的 12 字节已知头部。
                raw_data_header = _read_exact(
                    handle,
                    _DATA_HEADER.size,
                    field=f"Waveform {waveform_index} Buffer {buffer_index} Header",
                )
                # 解包类型、点宽和 payload 总字节数。
                data_header_size, buffer_type, bytes_per_point, buffer_size = (
                    _DATA_HEADER.unpack(raw_data_header)
                )
                # 负 payload 长度会让 seek 倒退并重读已解析区域，必须拒绝。
                if buffer_size < 0:
                    raise KeysightBinError(
                        f"Waveform {waveform_index} Buffer {buffer_index} Size 不能为负数"
                    )
                # 尊重扩展 Data Header，使 data_offset 指向真实 payload 首字节。
                _skip_extension(
                    handle,
                    base_size=_DATA_HEADER.size,
                    declared_size=data_header_size,
                    field=f"Waveform {waveform_index} Buffer {buffer_index}",
                )
                # Header Size 跳转完成后当前位置就是 payload 的绝对偏移。
                data_offset = handle.tell()
                # payload 末尾不能越过操作系统报告的实际文件长度。
                if data_offset + buffer_size > actual_file_size:
                    raise KeysightBinError(
                        f"Waveform {waveform_index} Buffer {buffer_index} payload 截断"
                    )
                # 保存纯元数据描述符，inspect 不调用 read(buffer_size)。
                buffers.append(
                    KeysightBufferInfo(
                        header_size_bytes=data_header_size,
                        buffer_type=buffer_type,
                        bytes_per_point=bytes_per_point,
                        buffer_size_bytes=buffer_size,
                        data_offset=data_offset,
                    )
                )
                # 仅移动文件指针到下一个 buffer 或 waveform 的头部。
                handle.seek(buffer_size, 1)
            # 文本字段在扫描阶段完成严格解码，避免加载后才暴露损坏元数据。
            waveforms.append(
                KeysightWaveformInfo(
                    index=waveform_index,
                    header_size_bytes=waveform_header_size,
                    waveform_type=waveform_type,
                    points=points,
                    count=count,
                    x_display_range=x_display_range,
                    x_display_origin=x_display_origin,
                    x_increment_s=x_increment,
                    x_origin_s=x_origin,
                    x_units=x_units,
                    y_units=y_units,
                    date=_decode_fixed_text(raw_date, field="Date"),
                    time=_decode_fixed_text(raw_time, field="Time"),
                    frame=_decode_fixed_text(raw_frame, field="Frame"),
                    label=_decode_fixed_text(raw_label, field="Waveform Label"),
                    time_tag=time_tag,
                    segment_index=segment_index,
                    buffers=tuple(buffers),
                )
            )
        # 所有声明对象结束后必须恰好到达 EOF，尾随未知字节不被静默接受。
        if handle.tell() != actual_file_size:
            raise KeysightBinError("Keysight BIN 在最后一个 waveform 后包含未解析字节")
    # 返回冻结索引；payload 始终留在磁盘上。
    return KeysightBinInfo(
        path=source_path,
        version=version,
        declared_file_size=declared_file_size,
        actual_file_size=actual_file_size,
        waveforms=tuple(waveforms),
    )


def inspect_keysight_bin(path: str | Path) -> KeysightBinInfo:
    """扫描 Keysight AG BIN 的所有头部，不读取任何 waveform payload。"""

    # resolve 固定报告中的源文件身份，但不会改变或创建文件。
    source_path = Path(path).expanduser().resolve()
    # 公共 inspect 自己持有文件；内部扫描不会触碰 waveform payload 内容。
    with source_path.open("rb") as opened_file:
        return _inspect_keysight_bin_from_open_file(source_path, opened_file)


def _load_keysight_waveform_from_open_file(
    source_path: Path,
    opened_file: object,
    waveform_index: int = 0,
) -> KeysightWaveform:
    """从已经打开且即将扫描的同一个文件映射所选 waveform。"""

    try:
        # Python int 与 NumPy 整数都可以作为稳定的 waveform 序号。
        selected_index = operator.index(waveform_index)
    except TypeError as error:
        # 不把 0.9 之类的值静默截断成第零个 waveform。
        raise KeysightBinError("waveform_index 必须是整数") from error
    # bool 虽是 int 子类，但不能表达用户明确选择的 waveform 序号。
    if isinstance(waveform_index, (bool, np.bool_)):
        raise KeysightBinError("waveform_index 必须是整数且不能为布尔值")
    # 扫描先验证全部 header 和文件边界，随后才允许映射 payload。
    info = _inspect_keysight_bin_from_open_file(source_path, opened_file)
    # 负索引会把损坏选择伪装成 Python 的倒序语义，因此显式拒绝。
    if not 0 <= selected_index < len(info.waveforms):
        raise KeysightBinError(
            f"waveform_index {selected_index} 超出 0..{len(info.waveforms) - 1}"
        )
    # 严格使用调用方选择的 header，不在多 waveform 文件中自动寻找“第一个可用项”。
    waveform_info = info.waveforms[selected_index]
    # inspect 保留不支持对象用于 UI 展示，load 在触碰 payload 前 fail-closed。
    if waveform_info.unsupported_reason is not None:
        raise KeysightBinError(
            f"Waveform {selected_index} 无法加载：{waveform_info.unsupported_reason}"
        )
    # is_loadable 已保证该描述符唯一存在，此断言防止未来属性实现发生合同漂移。
    buffer_info = waveform_info.normal_float32_buffer
    if buffer_info is None:
        raise KeysightBinError(f"Waveform {selected_index} 缺少 normal float32 buffer")
    # 只读 memmap 复用已扫描文件描述符，路径被同长度替换也不会混合新旧内容。
    values = np.memmap(
        opened_file,
        dtype="<f4",
        mode="r",
        offset=buffer_info.data_offset,
        shape=(waveform_info.points,),
    )
    # 有效加载条件已经证明 XIncrement 可求倒数，防御分支仍避免 Optional 泄漏。
    sample_rate_hz = waveform_info.sample_rate_hz
    if sample_rate_hz is None:
        raise KeysightBinError(f"Waveform {selected_index} 无法从 X Increment 推导采样率")
    # 返回原始 float32 电压和隐式时间轴元数据，不创建全长 time_s 数组。
    return KeysightWaveform(
        path=info.path,
        waveform_index=selected_index,
        values=values,
        sample_rate_hz=sample_rate_hz,
        x_increment_s=waveform_info.x_increment_s,
        x_origin_s=waveform_info.x_origin_s,
        label=waveform_info.label,
        segment_index=waveform_info.segment_index,
    )


def load_keysight_waveform(
    path: str | Path,
    waveform_index: int = 0,
) -> KeysightWaveform:
    """加载一个明确索引的受支持模拟 waveform。"""

    # 从扫描开始到 mmap 建立完成始终持有同一个文件描述符，关闭路径替换竞态。
    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as opened_file:
        return _load_keysight_waveform_from_open_file(
            source_path,
            opened_file,
            waveform_index,
        )


def write_keysight_bin(
    path: str | Path,
    values: object,
    sample_rate_hz: float,
    x_origin_s: float = 0.0,
    label: str = "ResponseLab",
) -> Path:
    """写出一个标准单 waveform Keysight AG10 BIN。"""

    # 复数电压不能直接存入 Keysight normal float32 buffer。
    if np.iscomplexobj(values):
        raise KeysightBinError("Keysight BIN 导出不支持复数电压")
    try:
        # 对 NumPy 数组和 memmap 保留零拷贝视图；普通序列只在此转换一次。
        source_values = np.asarray(values)
    except (TypeError, ValueError) as error:
        # 把对象数组或不可数值序列转换失败统一成公开格式错误。
        raise KeysightBinError("Keysight BIN 导出值必须是一维实数序列") from error
    # Keysight normal waveform 只有一条样本轴，二维通道不能静默展平。
    if source_values.ndim != 1 or source_values.size == 0:
        raise KeysightBinError("Keysight BIN 导出值必须是一维非空序列")
    # bool、文本、对象和日期等 dtype 不具有明确的伏特数值语义。
    if source_values.dtype.kind not in "iuf":
        raise KeysightBinError("Keysight BIN 导出值必须是整数或浮点实数")
    # 点数和四字节 payload 都必须能写入官方有符号 int32 字段。
    points = int(source_values.size)
    payload_size = points * 4
    if points > _INT32_MAX or payload_size > _INT32_MAX:
        raise KeysightBinError("Keysight BIN 点数或 payload 超过 int32 格式上限")
    # bool 采样率不能被解释为 1 Hz；其余数值统一转成 Python float。
    if isinstance(sample_rate_hz, (bool, np.bool_)):
        raise KeysightBinError("sample_rate_hz 必须是正的有限 Hz 数值")
    try:
        # 公开采样率单位固定为 Hz。
        numeric_sample_rate_hz = float(sample_rate_hz)
    except (TypeError, ValueError, OverflowError) as error:
        raise KeysightBinError("sample_rate_hz 必须是正的有限 Hz 数值") from error
    # 零、负数、NaN 和 Inf 都不能生成合法 X Increment。
    if not math.isfinite(numeric_sample_rate_hz) or numeric_sample_rate_hz <= 0.0:
        raise KeysightBinError("sample_rate_hz 必须是正的有限 Hz 数值")
    # bool 原点没有物理时间含义；合法输入统一使用秒。
    if isinstance(x_origin_s, (bool, np.bool_)):
        raise KeysightBinError("x_origin_s 必须是有限秒数")
    try:
        # 时间原点保留 float64 精度写入 Waveform Header。
        numeric_x_origin_s = float(x_origin_s)
    except (TypeError, ValueError, OverflowError) as error:
        raise KeysightBinError("x_origin_s 必须是有限秒数") from error
    # 非有限原点会使每个隐式时间样本都失去物理意义。
    if not math.isfinite(numeric_x_origin_s):
        raise KeysightBinError("x_origin_s 必须是有限秒数")
    # label 必须由调用方明确提供字符串，不能依赖任意对象的 __str__。
    if not isinstance(label, str):
        raise KeysightBinError("label 必须是字符串")
    # 16 字节标签按 UTF-8/NUL 合同编码，超长时 fail-closed。
    encoded_label = _encode_fixed_text(label, size=16, field="label")
    # X Increment 是秒/点，直接由 Fs 的倒数写入自描述头。
    x_increment_s = 1.0 / numeric_sample_rate_hz
    # 极端上溢或下溢后不能再声称文件包含有效采样率。
    if not math.isfinite(x_increment_s) or x_increment_s <= 0.0:
        raise KeysightBinError("sample_rate_hz 无法表示为正的 float64 X Increment")
    # 显示范围使用 N 个均匀采样间隔，并受 Waveform Header float32 字段限制。
    x_display_range = points * x_increment_s
    if not math.isfinite(x_display_range) or abs(x_display_range) > np.finfo(np.float32).max:
        raise KeysightBinError("采样时长超出 float32 X Display Range")
    # 完整文件为三层固定头加 float32 payload，writer 写入真实长度而非零占位。
    file_size = _FILE_HEADER.size + _WAVEFORM_HEADER.size + _DATA_HEADER.size + payload_size
    if file_size > _INT32_MAX:
        raise KeysightBinError("Keysight BIN 文件长度超过 int32 格式上限")
    # File Header 固定 AG10、真实文件长度和一个 waveform。
    file_header = _FILE_HEADER.pack(b"AG", b"10", file_size, 1)
    # Waveform Header 写入 Normal、Seconds、Volts 和调用方的时间轴元数据。
    waveform_header = _WAVEFORM_HEADER.pack(
        _WAVEFORM_HEADER.size,
        1,
        1,
        points,
        0,
        x_display_range,
        numeric_x_origin_s,
        x_increment_s,
        numeric_x_origin_s,
        _SECONDS_UNIT,
        _VOLTS_UNIT,
        _encode_fixed_text("01 JAN 1970", size=16, field="Date"),
        _encode_fixed_text("00:00:00", size=16, field="Time"),
        _encode_fixed_text("ResponseLab", size=24, field="Frame"),
        encoded_label,
        0.0,
        0,
    )
    # 唯一 Data Header 声明 normal float32、四字节点宽和精确 payload 长度。
    data_header = _DATA_HEADER.pack(
        _DATA_HEADER.size,
        _NORMAL_FLOAT_BUFFER_TYPE,
        4,
        payload_size,
    )
    # 先逐块验证所有输入，保证数值错误不会留下部分写入的临时产物。
    float32_limit = np.finfo(np.float32).max
    for start in range(0, points, _WRITE_CHUNK_POINTS):
        # 每个切片最多约 4 MiB，验证内存不会随完整捕获长度线性增加临时副本。
        chunk = source_values[start : start + _WRITE_CHUNK_POINTS]
        # NaN/Inf 不能成为后续 LFP/RMS 的静默污染源。
        if not np.all(np.isfinite(chunk)):
            raise KeysightBinError("Keysight BIN 导出值包含 NaN 或 Inf")
        # 超出 float32 的有限实数会在量化时溢出，必须在写文件前拒绝。
        if np.any(np.abs(chunk) > float32_limit):
            raise KeysightBinError("Keysight BIN 导出值超出 float32 有限范围")
    # 目标父目录按现有导出习惯自动创建。
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件让最终 os.replace 保持单文件原子性。
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        # fdopen 接管描述符，并在上下文退出时可靠关闭。
        with os.fdopen(descriptor, "wb") as stream:
            # 三层头部在 payload 前一次写入，offset 与独立官方布局一致。
            stream.write(file_header)
            stream.write(waveform_header)
            stream.write(data_header)
            for start in range(0, points, _WRITE_CHUNK_POINTS):
                # 每个输出块单独量化为连续 little-endian float32。
                encoded_chunk = np.ascontiguousarray(
                    source_values[start : start + _WRITE_CHUNK_POINTS],
                    dtype="<f4",
                )
                # memoryview 直接写数组缓冲区，避免额外创建整块 bytes 副本。
                stream.write(memoryview(encoded_chunk).cast("B"))
            # flush 与 fsync 在重命名前把用户数据提交给文件系统。
            stream.flush()
            os.fsync(stream.fileno())
        # 完整临时文件通过验证和落盘后才替换最终路径。
        os.replace(temporary_name, output_path)
    except Exception:
        # 失败时删除仅由本函数创建的临时文件，不触碰既有目标文件。
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        # 保留原始异常类型和上下文给调用方诊断。
        raise
    # 返回实际写入的绝对路径，便于报告和后续独立解析。
    return output_path


# 明确公开面，避免调用方依赖内部 Struct 或格式常量。
__all__ = [
    "KeysightBinError",
    "KeysightBinInfo",
    "KeysightBufferInfo",
    "KeysightWaveform",
    "KeysightWaveformInfo",
    "inspect_keysight_bin",
    "load_keysight_waveform",
    "write_keysight_bin",
]
