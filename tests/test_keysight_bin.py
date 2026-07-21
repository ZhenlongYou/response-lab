"""Keysight Infiniium BIN 公共接口的独立格式夹具测试。"""

from __future__ import annotations

# struct 只在测试侧按官方字段布局构造字节，避免复用生产解析器形成循环证明。
import os
import struct
from pathlib import Path

# NumPy 提供具有明确 little-endian float32 编码的已知电压样本。
import numpy as np
import pytest

# 测试只通过三个公开入口观察文件扫描、波形加载和导出行为。
from response_lab.keysight_bin import (
    inspect_keysight_bin,
    load_keysight_waveform,
    write_keysight_bin,
)

# 三个 Struct 在测试侧独立声明官方 12/140/12 字节布局。
_FILE_HEADER = struct.Struct("<2s2sii")
_WAVE_HEADER = struct.Struct("<iiiiifdddii16s16s24s16sdI")
_DATA_HEADER = struct.Struct("<ihhi")


def _fixed_text(text: str, size: int) -> bytes:
    """把测试元数据编码成官方定长、NUL 补齐的 UTF-8 字段。"""

    # 夹具文字均为 ASCII 子集，UTF-8 编码可以直接对应文件中的单字节字符。
    encoded = text.encode("utf-8")
    # 测试生成器自身先拒绝溢出，避免无意截断掩盖生产代码问题。
    if len(encoded) > size:
        raise ValueError("测试文本超过 Keysight 定长字段")
    # 右侧 NUL 补齐模拟示波器写入的 C 字符数组。
    return encoded.ljust(size, b"\0")


def _normal_waveform_bytes(
    values: np.ndarray,
    *,
    x_increment_s: float = 25.0e-12,
    x_origin_s: float = -175.0e-12,
    label: str = "Channel 1",
    waveform_type: int = 1,
    x_units: int = 2,
    y_units: int = 1,
) -> bytes:
    """独立构造一个标准模拟 waveform header、data header 与 payload。"""

    # 官方模拟数据 payload 是 little-endian 32-bit float 电压序列。
    encoded_values = np.asarray(values, dtype="<f4")
    # 波形头声明真实点数、秒单位、伏特单位和隐式均匀时间轴。
    waveform_header = _WAVE_HEADER.pack(
        _WAVE_HEADER.size,
        waveform_type,
        1,
        int(encoded_values.size),
        0,
        float(encoded_values.size * x_increment_s),
        x_origin_s,
        x_increment_s,
        x_origin_s,
        x_units,
        y_units,
        _fixed_text("22 JUL 2026", 16),
        _fixed_text("12:00:00", 16),
        _fixed_text("D9300A:TEST", 24),
        _fixed_text(label, 16),
        0.0,
        0,
    )
    # 单一 normal buffer 的字节数必须等于点数乘以四字节。
    payload = encoded_values.tobytes()
    # Data Header 独立声明 normal float、每点四字节和 payload 总长度。
    data_header = _DATA_HEADER.pack(_DATA_HEADER.size, 1, 4, len(payload))
    # waveform 是波形头、buffer 头和 payload 的顺序拼接。
    return waveform_header + data_header + payload


def _peak_detect_waveform_bytes(minimum: np.ndarray, maximum: np.ndarray) -> bytes:
    """独立构造带 minimum/maximum 双 buffer 的 Peak Detect waveform。"""

    # Peak Detect 的两条包络必须等长，才能共同对应 header 中的 Points。
    minimum_values = np.asarray(minimum, dtype="<f4")
    maximum_values = np.asarray(maximum, dtype="<f4")
    if minimum_values.shape != maximum_values.shape:
        raise ValueError("Peak Detect 测试包络必须等长")
    # waveform type=2 且 buffer_count=2，明确区别于单值时域记录。
    waveform_header = _WAVE_HEADER.pack(
        _WAVE_HEADER.size,
        2,
        2,
        int(minimum_values.size),
        0,
        float(minimum_values.size * 25.0e-12),
        0.0,
        25.0e-12,
        0.0,
        2,
        1,
        _fixed_text("22 JUL 2026", 16),
        _fixed_text("12:00:00", 16),
        _fixed_text("D9300A:TEST", 24),
        _fixed_text("Peak Detect", 16),
        0.0,
        0,
    )
    # 官方 type 2 表示 maximum float32，type 3 表示 minimum float32。
    maximum_payload = maximum_values.tobytes()
    minimum_payload = minimum_values.tobytes()
    maximum_buffer = (
        _DATA_HEADER.pack(_DATA_HEADER.size, 2, 4, len(maximum_payload))
        + maximum_payload
    )
    minimum_buffer = (
        _DATA_HEADER.pack(_DATA_HEADER.size, 3, 4, len(minimum_payload))
        + minimum_payload
    )
    # 两个 buffer 顺序存放，inspect 必须逐一跨过 payload 后继续解析。
    return waveform_header + maximum_buffer + minimum_buffer


def _normal_waveform_with_header_extensions(values: np.ndarray) -> tuple[bytes, int]:
    """在两个 Header Size 管理的尾部加入独立扩展字节。"""

    # 从标准夹具取得已知基准字段和 payload，再由测试侧重新拼装扩展布局。
    standard = _normal_waveform_bytes(values)
    waveform_header = bytearray(standard[: _WAVE_HEADER.size])
    data_header_start = _WAVE_HEADER.size
    data_header_end = data_header_start + _DATA_HEADER.size
    data_header = bytearray(standard[data_header_start:data_header_end])
    payload = standard[data_header_end:]
    # 非零且不同长度的扩展可以发现只硬编码 140/12 偏移的解析器。
    waveform_extension = b"WAVE-EXT"
    data_extension = b"DATA-EXTENSION"
    # 两个 size 字段分别覆盖自己的基准头与扩展，不包含 payload。
    struct.pack_into("<i", waveform_header, 0, _WAVE_HEADER.size + len(waveform_extension))
    struct.pack_into("<i", data_header, 0, _DATA_HEADER.size + len(data_extension))
    # 新 payload offset 由 File Header、两种扩展头和两个基准头共同确定。
    expected_payload_offset = (
        _FILE_HEADER.size
        + _WAVE_HEADER.size
        + len(waveform_extension)
        + _DATA_HEADER.size
        + len(data_extension)
    )
    # 返回完整 waveform 和独立手算 offset 供公共接口断言。
    return (
        bytes(waveform_header)
        + waveform_extension
        + bytes(data_header)
        + data_extension
        + payload,
        expected_payload_offset,
    )


def _write_bin(
    path: Path,
    waveforms: tuple[bytes, ...],
    *,
    cookie: bytes = b"AG",
    declared_size: int | None = None,
) -> Path:
    """写出测试专用 File Header，并允许构造 size=0 或损坏 cookie。"""

    # 先计算完整文件长度，正例的声明大小不依赖生产 writer。
    actual_size = _FILE_HEADER.size + sum(len(waveform) for waveform in waveforms)
    # None 表示按官方传统格式写入真实总长度，显式 0 用于 Infiniium 2026 变体。
    file_size = actual_size if declared_size is None else declared_size
    # File Header 只包含 cookie、两字节版本、文件大小和波形数量。
    header = _FILE_HEADER.pack(cookie, b"10", file_size, len(waveforms))
    # 测试文件由完全独立的二进制拼装逻辑生成。
    path.write_bytes(header + b"".join(waveforms))
    # 返回路径让测试保持 Arrange 阶段紧凑。
    return path


def _independently_parse_export(path: Path) -> dict[str, object]:
    """只用测试侧 Struct 解码 writer 产物，不调用生产 inspect/load。"""

    # 小型导出夹具可以整体读取，独立 oracle 不复用生产文件扫描逻辑。
    raw = path.read_bytes()
    # File Header 从绝对偏移零开始，先验证 writer 的外层容器字段。
    cookie, version, declared_size, waveform_count = _FILE_HEADER.unpack_from(raw, 0)
    # 唯一 Waveform Header 紧跟 12 字节 File Header。
    waveform_fields = _WAVE_HEADER.unpack_from(raw, _FILE_HEADER.size)
    waveform_header_size = waveform_fields[0]
    # Data Header 的位置由 writer 声明的 Waveform Header Size 决定。
    data_header_offset = _FILE_HEADER.size + waveform_header_size
    data_header_size, buffer_type, bytes_per_point, buffer_size = _DATA_HEADER.unpack_from(
        raw,
        data_header_offset,
    )
    # payload 位于完整 Data Header 之后，长度严格取自 Buffer Size。
    payload_offset = data_header_offset + data_header_size
    payload = raw[payload_offset : payload_offset + buffer_size]
    # 独立按 little-endian float32 解码电压，不调用 response_lab 任何辅助函数。
    values = np.frombuffer(payload, dtype="<f4").copy()
    # 字典只返回公开合同相关字段，测试不依赖生产 dataclass 形状。
    return {
        "raw_size": len(raw),
        "cookie": cookie,
        "version": version,
        "declared_size": declared_size,
        "waveform_count": waveform_count,
        "waveform_fields": waveform_fields,
        "buffer_type": buffer_type,
        "bytes_per_point": bytes_per_point,
        "buffer_size": buffer_size,
        "payload_end": payload_offset + buffer_size,
        "values": values,
    }


def test_inspect_reads_single_normal_waveform_without_loading_payload(tmp_path: Path) -> None:
    """扫描应从头部得到波形、buffer 偏移和物理时间元数据。"""

    # 16 个手算样本足以验证点数和 payload 边界，不依赖生产导出器。
    expected_values = np.linspace(-0.75, 0.75, 16, dtype=np.float32)
    # 独立夹具把 25 ps 间隔和负 XOrigin 写入官方 AG10 容器。
    path = _write_bin(tmp_path / "single.bin", (_normal_waveform_bytes(expected_values),))

    # 公共扫描入口只应读取头部并返回结构化索引。
    info = inspect_keysight_bin(path)

    # File Header 中的版本和波形数必须原样保留。
    assert info.version == "10"
    assert len(info.waveforms) == 1
    # 第一个 payload 从 12 + 140 + 12 = 164 字节处开始。
    assert info.waveforms[0].normal_float32_buffer is not None
    assert info.waveforms[0].normal_float32_buffer.data_offset == 164
    # 时间间隔和原点来自 Waveform Header，而不是用户手填参数。
    assert info.waveforms[0].x_increment_s == pytest.approx(25.0e-12)
    assert info.waveforms[0].x_origin_s == pytest.approx(-175.0e-12)
    # 标准 Normal/Seconds/Volts/单 normal buffer 组合应标记为可加载。
    assert info.waveforms[0].is_loadable is True


def test_inspect_matches_official_absolute_waveform_field_offsets(tmp_path: Path) -> None:
    """绝对偏移 golden 不复用复合格式串，防止生产与夹具共同错位。"""

    # payload 使用不对称值，同时约束端序、起点和幅值不发生隐式缩放。
    expected_values = np.array([-0.75, 0.125, 1.5], dtype="<f4")
    payload = expected_values.tobytes()
    # 140 字节 Waveform Header 按官方字段绝对偏移逐项写入。
    waveform_header = bytearray(140)
    struct.pack_into("<i", waveform_header, 0, 140)
    struct.pack_into("<i", waveform_header, 4, 1)
    struct.pack_into("<i", waveform_header, 8, 1)
    struct.pack_into("<i", waveform_header, 12, expected_values.size)
    struct.pack_into("<i", waveform_header, 16, 0)
    struct.pack_into("<f", waveform_header, 20, expected_values.size * 12.5e-12)
    struct.pack_into("<d", waveform_header, 24, -25.0e-12)
    struct.pack_into("<d", waveform_header, 32, 12.5e-12)
    struct.pack_into("<d", waveform_header, 40, -25.0e-12)
    struct.pack_into("<i", waveform_header, 48, 2)
    struct.pack_into("<i", waveform_header, 52, 1)
    waveform_header[56:72] = _fixed_text("22 JUL 2026", 16)
    waveform_header[72:88] = _fixed_text("12:34:56", 16)
    waveform_header[88:112] = _fixed_text("ABSOLUTE-OFFSET", 24)
    waveform_header[112:128] = _fixed_text("Golden", 16)
    struct.pack_into("<d", waveform_header, 128, 3.25)
    struct.pack_into("<I", waveform_header, 136, 7)
    # 12 字节 Data Header 也逐字段按 0/4/6/8 偏移写入。
    data_header = bytearray(12)
    struct.pack_into("<i", data_header, 0, 12)
    struct.pack_into("<h", data_header, 4, 1)
    struct.pack_into("<h", data_header, 6, 4)
    struct.pack_into("<i", data_header, 8, len(payload))
    # File Header 逐字段拼装，完全绕过测试侧的三个复合 Struct 常量。
    file_size = 12 + len(waveform_header) + len(data_header) + len(payload)
    file_header = bytearray(12)
    file_header[0:2] = b"AG"
    file_header[2:4] = b"10"
    struct.pack_into("<i", file_header, 4, file_size)
    struct.pack_into("<i", file_header, 8, 1)
    path = tmp_path / "absolute-offset-golden.bin"
    path.write_bytes(file_header + waveform_header + data_header + payload)

    # 关键字段必须由各自绝对偏移得到，并能加载同一 payload。
    info = inspect_keysight_bin(path)
    waveform = info.waveforms[0]
    assert waveform.x_increment_s == pytest.approx(12.5e-12)
    assert waveform.x_origin_s == pytest.approx(-25.0e-12)
    assert waveform.label == "Golden"
    assert waveform.segment_index == 7
    assert waveform.buffers[0].data_offset == 164
    np.testing.assert_array_equal(load_keysight_waveform(path).values, expected_values)


def test_load_derives_sample_rate_origin_and_float32_values(tmp_path: Path) -> None:
    """加载应使用 XIncrement/XOrigin，并精确读取 little-endian 电压 payload。"""

    # 不对称电压值可以同时发现错误偏移、错误端序和静默缩放。
    expected_values = np.array([-1.25, -0.5, 0.125, 0.75, 1.5], dtype=np.float32)
    # 20 ps 对应手算 50 GSa/s，原点故意不设为零。
    waveform_bytes = _normal_waveform_bytes(
        expected_values,
        x_increment_s=20.0e-12,
        x_origin_s=-40.0e-12,
        label="DUT",
    )
    # 单波形文件允许使用公开接口的默认索引 0。
    path = _write_bin(tmp_path / "load.bin", (waveform_bytes,))

    # 加载入口应先重用严格头部索引，再只读取选中的 payload。
    waveform = load_keysight_waveform(path)

    # float32 保留仪器原始精度并避免无必要的两倍内存占用。
    assert waveform.values.dtype == np.dtype("float32")
    assert isinstance(waveform.values, np.memmap)
    assert waveform.values.mode == "r"
    np.testing.assert_array_equal(waveform.values, expected_values)
    # Fs 是 20 ps/点的倒数，不能来自任何手工参数。
    assert waveform.sample_rate_hz == pytest.approx(50.0e9)
    assert waveform.x_increment_s == pytest.approx(20.0e-12)
    assert waveform.x_origin_s == pytest.approx(-40.0e-12)
    # 标签和索引来自同一个被选择的 waveform header。
    assert waveform.label == "DUT"
    assert waveform.waveform_index == 0


def test_load_reuses_scanned_file_when_path_is_replaced_with_same_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """扫描后同长度替换路径时，不得把旧 header 与新 payload 混合。"""

    # 两个文件布局和长度完全相同，只让 payload 值明显不同。
    original_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    replacement_values = np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32)
    path = _write_bin(
        tmp_path / "race.bin",
        (_normal_waveform_bytes(original_values),),
    )
    replacement = _write_bin(
        tmp_path / "replacement.bin",
        (_normal_waveform_bytes(replacement_values),),
    )
    # 在生产代码即将建立 mmap 时替换目录项，稳定复现扫描与映射之间的竞态窗口。
    original_memmap = np.memmap

    def replace_path_before_mapping(file: object, *args: object, **kwargs: object) -> np.memmap:
        os.replace(replacement, path)
        return original_memmap(file, *args, **kwargs)

    monkeypatch.setattr(np, "memmap", replace_path_before_mapping)

    # 正确实现映射已扫描的打开文件；错误实现会静默返回 replacement_values。
    waveform = load_keysight_waveform(path)
    np.testing.assert_array_equal(waveform.values, original_values)


def test_file_size_zero_is_accepted_for_infiniium_2026(tmp_path: Path) -> None:
    """Infiniium 2026 的零 File Size 应以实际文件长度作为安全边界。"""

    # 明确写入零而不是测试生成器默认的真实长度。
    path = _write_bin(
        tmp_path / "size-zero.bin",
        (_normal_waveform_bytes(np.arange(8, dtype=np.float32)),),
        declared_size=0,
    )

    # 扫描器应保留零声明，同时仍记录操作系统报告的真实长度。
    info = inspect_keysight_bin(path)

    assert info.declared_file_size == 0
    assert info.actual_file_size == path.stat().st_size
    # 零占位不能阻止后续严格加载已验证的 payload。
    np.testing.assert_array_equal(
        load_keysight_waveform(path).values,
        np.arange(8, dtype=np.float32),
    )


def test_inspect_rejects_non_keysight_cookie(tmp_path: Path) -> None:
    """随机裸 BIN 即使长度可解析，也不能绕过 AG 文件签名。"""

    # 使用完整合法 waveform，仅破坏 cookie，确保失败原因锁定在格式识别边界。
    path = _write_bin(
        tmp_path / "wrong-cookie.bin",
        (_normal_waveform_bytes(np.arange(8, dtype=np.float32)),),
        cookie=b"XX",
    )

    # 错误文件必须 fail-closed，不能回退成旧的无头 float32 解析器。
    with pytest.raises(ValueError, match="Cookie.*AG"):
        inspect_keysight_bin(path)


def test_inspect_rejects_unknown_ag_version(tmp_path: Path) -> None:
    """AG cookie 之后的未知版本不能套用 AG10 固定偏移继续解析。"""

    # 先构造完整 AG10 正例，再只替换 File Header 的两字节版本字段。
    path = _write_bin(
        tmp_path / "unknown-version.bin",
        (_normal_waveform_bytes(np.arange(8, dtype=np.float32)),),
    )
    raw = bytearray(path.read_bytes())
    raw[2:4] = b"99"
    path.write_bytes(raw)

    # 未知布局必须 fail-closed，不能仅把版本作为展示文本保留下来。
    with pytest.raises(ValueError, match="Version.*10"):
        inspect_keysight_bin(path)


def test_inspect_does_not_auto_detect_big_endian_headers(tmp_path: Path) -> None:
    """AG cookie 相同也不能让 big-endian 数字字段绕过固定小端合同。"""

    # payload 部分保持标准，只把 File Header 数字字段故意编码为 big-endian。
    waveform_bytes = _normal_waveform_bytes(np.arange(8, dtype=np.float32))
    actual_size = _FILE_HEADER.size + len(waveform_bytes)
    big_endian_header = struct.pack(">2s2sii", b"AG", b"10", actual_size, 1)
    path = tmp_path / "big-endian.bin"
    path.write_bytes(big_endian_header + waveform_bytes)

    # 解析器必须按 little-endian 读取并拒绝边界，不应尝试启发式翻转字节序。
    with pytest.raises(ValueError, match="File Size.*不一致"):
        inspect_keysight_bin(path)


def test_inspect_rejects_truncated_payload(tmp_path: Path) -> None:
    """Buffer Size 声明的最后一个字节缺失时必须在 inspect 阶段失败。"""

    # file_size=0 避免先由总长度不一致遮蔽真正要验证的 payload 边界检查。
    path = _write_bin(
        tmp_path / "truncated.bin",
        (_normal_waveform_bytes(np.arange(8, dtype=np.float32)),),
        declared_size=0,
    )
    # 只删除 payload 的最后一个字节，所有 header 仍然完整可解包。
    original_bytes = path.read_bytes()
    path.write_bytes(original_bytes[:-1])

    # 扫描不能把短 payload 当作七点或忽略残缺尾部。
    with pytest.raises(ValueError, match="payload.*截断"):
        inspect_keysight_bin(path)


def test_inspect_rejects_nonzero_declared_size_mismatch(tmp_path: Path) -> None:
    """正 File Size 必须与实际文件长度完全一致，只有零值属于兼容例外。"""

    # 123 是显式错误的非零长度，不能获得 Infiniium 2026 零占位豁免。
    path = _write_bin(
        tmp_path / "wrong-size.bin",
        (_normal_waveform_bytes(np.arange(8, dtype=np.float32)),),
        declared_size=123,
    )

    # 扫描在读取 waveform 前即拒绝不可信的全文件边界。
    with pytest.raises(ValueError, match="File Size.*不一致"):
        inspect_keysight_bin(path)


@pytest.mark.parametrize(
    ("x_units", "y_units", "expected_message"),
    [
        # X Units=6 表示 Hz，不能由 X Increment 解释为秒/点。
        (6, 1, "X Units.*Second"),
        # Y Units=5 表示 dB，不能未经物理换算冒充伏特。
        (2, 5, "Y Units.*Volt"),
    ],
)
def test_load_rejects_non_time_or_non_voltage_units(
    tmp_path: Path,
    x_units: int,
    y_units: int,
    expected_message: str,
) -> None:
    """inspect 可展示不支持记录，但 load 不得猜测 Hz/dB 等单位。"""

    # 非法单位在结构上仍可扫描，却不是本工具允许加载的时域伏特输入。
    path = _write_bin(
        tmp_path / "wrong-units.bin",
        (
            _normal_waveform_bytes(
                np.arange(8, dtype=np.float32),
                x_units=x_units,
                y_units=y_units,
            ),
        ),
    )

    # 元数据扫描仍应允许 UI 告知用户文件里实际包含什么。
    info = inspect_keysight_bin(path)
    assert info.waveforms[0].is_loadable is False
    # 真正加载在映射 payload 前拒绝，并指向当前参数化的错误单位。
    with pytest.raises(ValueError, match=expected_message):
        load_keysight_waveform(path)


def test_inspect_does_not_report_sample_rate_for_non_seconds_x_axis(tmp_path: Path) -> None:
    """只有 X Units=Second 时，X Increment 的倒数才能标记为 Hz 采样率。"""

    # 频率轴仍可有正 increment，但其倒数不具有 samples/s 的物理意义。
    path = _write_bin(
        tmp_path / "frequency-axis.bin",
        (
            _normal_waveform_bytes(
                np.arange(8, dtype=np.float32),
                x_increment_s=1.0e6,
                x_units=6,
            ),
        ),
    )

    # inspect 保留原始 increment，同时以 None 表示采样率不可从该单位推导。
    waveform = inspect_keysight_bin(path).waveforms[0]
    assert waveform.x_increment_s == pytest.approx(1.0e6)
    assert waveform.sample_rate_hz is None


def test_load_rejects_x_increment_whose_reciprocal_overflows(tmp_path: Path) -> None:
    """正 subnormal 秒间隔的倒数若为 Inf，也不能形成合法采样率。"""

    # float64 最小正数本身有限且大于零，但 1/x 会超过 float64 有限范围。
    smallest_positive = np.nextafter(0.0, 1.0)
    path = _write_bin(
        tmp_path / "overflowing-sample-rate.bin",
        (
            _normal_waveform_bytes(
                np.arange(4, dtype=np.float32),
                x_increment_s=smallest_positive,
            ),
        ),
    )

    # inspect 不应把 Inf 标成 Hz；load 必须在映射 payload 前给出明确错误。
    waveform = inspect_keysight_bin(path).waveforms[0]
    assert waveform.sample_rate_hz is None
    assert waveform.is_loadable is False
    with pytest.raises(ValueError, match="采样率.*有限"):
        load_keysight_waveform(path)


def test_multi_waveform_inspect_and_explicit_index_load_are_independent(tmp_path: Path) -> None:
    """每个 waveform 应保留自己的 label、Fs、origin 和 payload 选择。"""

    # 第一条参考记录使用 25 ps 间隔和较小电压，第二条 DUT 使用 10 ps 间隔。
    reference_values = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    dut_values = np.array([3.0, -2.0, 1.0, -4.0, 5.0], dtype=np.float32)
    # 两个 waveform 顺序写入同一个 File Header 管理的容器。
    path = _write_bin(
        tmp_path / "multi.bin",
        (
            _normal_waveform_bytes(
                reference_values,
                x_increment_s=25.0e-12,
                x_origin_s=-100.0e-12,
                label="Reference",
            ),
            _normal_waveform_bytes(
                dut_values,
                x_increment_s=10.0e-12,
                x_origin_s=30.0e-12,
                label="DUT",
                waveform_type=3,
            ),
        ),
    )

    # inspect 必须扫描全部 waveform，而不是遇到第一个 payload 后把余下内容当作通道交织。
    info = inspect_keysight_bin(path)
    assert [waveform.label for waveform in info.waveforms] == ["Reference", "DUT"]
    assert [waveform.sample_rate_hz for waveform in info.waveforms] == pytest.approx(
        [40.0e9, 100.0e9]
    )

    # 显式索引 1 只能加载第二个 Average waveform 的独立 payload 和时间元数据。
    selected = load_keysight_waveform(path, waveform_index=1)
    np.testing.assert_array_equal(selected.values, dut_values)
    assert selected.label == "DUT"
    assert selected.sample_rate_hz == pytest.approx(100.0e9)
    assert selected.x_origin_s == pytest.approx(30.0e-12)


@pytest.mark.parametrize(
    ("waveform_index", "expected_message"),
    [
        # 负数不能借用 Python 序列的倒序含义。
        (-1, "超出"),
        # 单 waveform 文件的索引 1 已越过文件声明范围。
        (1, "超出"),
        # 浮点数即使数值为整数，也不能被静默截断。
        (0.0, "必须是整数"),
        # bool 是 int 子类，但不表示明确的 waveform 选择。
        (True, "不能为布尔值"),
    ],
)
def test_load_rejects_implicit_or_out_of_range_waveform_indexes(
    tmp_path: Path,
    waveform_index: object,
    expected_message: str,
) -> None:
    """load 只接受文件范围内的显式整数 waveform 索引。"""

    # 单 waveform 夹具让每个非法索引都具有唯一、可手算的有效范围 0..0。
    path = _write_bin(
        tmp_path / "strict-index.bin",
        (_normal_waveform_bytes(np.arange(4, dtype=np.float32)),),
    )

    # 类型错误和范围错误都必须在映射 payload 前关闭。
    with pytest.raises(ValueError, match=expected_message):
        load_keysight_waveform(path, waveform_index=waveform_index)  # type: ignore[arg-type]


def test_peak_detect_buffers_are_inspected_but_rejected_for_waveform_load(tmp_path: Path) -> None:
    """Peak Detect 的最大/最小包络可见，但不能伪装成单一原始波形。"""

    # 两条四点包络使用不同值，防止解析器错误地把它们拼成八点 normal 数据。
    path = _write_bin(
        tmp_path / "peak.bin",
        (
            _peak_detect_waveform_bytes(
                np.array([-1.0, -0.8, -0.6, -0.4], dtype=np.float32),
                np.array([0.4, 0.6, 0.8, 1.0], dtype=np.float32),
            ),
        ),
    )

    # inspect 应保留两个 buffer 的官方类型和独立 offset。
    info = inspect_keysight_bin(path)
    assert [buffer.buffer_type for buffer in info.waveforms[0].buffers] == [2, 3]
    assert info.waveforms[0].normal_float32_buffer is None
    assert info.waveforms[0].is_loadable is False
    # LFP/RMS 需要单值采样序列，因此 load 必须按 waveform type 明确拒绝。
    with pytest.raises(ValueError, match="Waveform Type.*2"):
        load_keysight_waveform(path)


def test_load_rejects_ambiguous_multiple_normal_buffers(tmp_path: Path) -> None:
    """Normal waveform 内出现两个 type-1 buffer 时不得自动挑选或拼接。"""

    # 从单 buffer 正例复制头部，再由测试侧把 buffer_count 改成二。
    values = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    waveform = bytearray(_normal_waveform_bytes(values))
    struct.pack_into("<i", waveform, 8, 2)
    # 第二个完整 normal buffer 使用不同电压，确保任意自动选择都具有可见歧义。
    second_values = np.array([10.0, 20.0, 30.0, 40.0], dtype="<f4")
    second_payload = second_values.tobytes()
    waveform.extend(
        _DATA_HEADER.pack(_DATA_HEADER.size, 1, 4, len(second_payload))
        + second_payload
    )
    path = _write_bin(tmp_path / "two-normal-buffers.bin", (bytes(waveform),))

    # inspect 保留两个描述符，但不能声称其中某一个是唯一可加载数据。
    info = inspect_keysight_bin(path)
    assert len(info.waveforms[0].buffers) == 2
    assert info.waveforms[0].normal_float32_buffer is None
    # load 在读取任何一个 payload 前按“必须单 buffer”合同失败。
    with pytest.raises(ValueError, match="必须只有 1 个 buffer"):
        load_keysight_waveform(path)


def test_declared_header_extensions_shift_payload_without_being_loaded(tmp_path: Path) -> None:
    """Waveform/Data Header Size 扩展必须改变 offset，不能被当作电压。"""

    # 已知电压和测试侧手算 offset 共同独立约束解析结果。
    expected_values = np.array([-2.0, -0.25, 0.5, 3.0], dtype=np.float32)
    waveform_bytes, expected_payload_offset = _normal_waveform_with_header_extensions(
        expected_values
    )
    path = _write_bin(tmp_path / "extensions.bin", (waveform_bytes,))

    # inspect 应按两个 Header Size 跳过扩展字节并记录新的 payload 首地址。
    info = inspect_keysight_bin(path)
    assert info.waveforms[0].normal_float32_buffer is not None
    assert (
        info.waveforms[0].normal_float32_buffer.data_offset
        == expected_payload_offset
    )
    # load 读取的仍应是原电压，而不是扩展头的 ASCII 字节。
    np.testing.assert_array_equal(load_keysight_waveform(path).values, expected_values)


@pytest.mark.parametrize(
    ("relative_offset", "short_size", "expected_message"),
    [
        # Waveform Header 不能短于官方 140 字节基准字段。
        (0, _WAVE_HEADER.size - 1, "Waveform 0 Header Size.*140"),
        # Data Header 不能短于官方 12 字节基准字段。
        (_WAVE_HEADER.size, _DATA_HEADER.size - 1, "Buffer 0 Header Size.*12"),
    ],
)
def test_inspect_rejects_headers_shorter_than_official_base_layout(
    tmp_path: Path,
    relative_offset: int,
    short_size: int,
    expected_message: str,
) -> None:
    """Header Size 只能扩展，不能缩短并重叠已知字段。"""

    # 先生成标准文件，再只修改目标 Header Size 的四字节整数。
    path = _write_bin(
        tmp_path / f"short-header-{relative_offset}.bin",
        (_normal_waveform_bytes(np.arange(8, dtype=np.float32)),),
    )
    raw = bytearray(path.read_bytes())
    # relative_offset 相对于第一个 Waveform Header，File Header 固定 12 字节。
    struct.pack_into("<i", raw, _FILE_HEADER.size + relative_offset, short_size)
    path.write_bytes(raw)

    # 解析器必须拒绝缩短的头，不能把剩余已知字段解释成扩展或 payload。
    with pytest.raises(ValueError, match=expected_message):
        inspect_keysight_bin(path)


def test_write_keysight_bin_is_verified_by_independent_struct_parser(tmp_path: Path) -> None:
    """导出应生成自描述 AG10 容器，而不是旧的无头 float32 文件。"""

    # 非整数和正负混合值可以发现静默缩放、偏置或错误 payload 起点。
    source_values = np.array([-1.125, -0.25, 0.0, 0.625, 1.75], dtype=np.float64)
    # 80 GSa/s 与 -125 ps 原点都能用独立闭式关系核对。
    output = write_keysight_bin(
        tmp_path / "nested" / "export.bin",
        source_values,
        sample_rate_hz=80.0e9,
        x_origin_s=-125.0e-12,
        label="Compensated",
    )

    # 独立测试解析器完全绕开生产 inspect/load，避免 writer 与 reader 同错。
    parsed = _independently_parse_export(output)
    waveform_fields = parsed["waveform_fields"]

    # 外层签名、版本、实际大小和唯一 waveform 数必须符合官方容器。
    assert parsed["cookie"] == b"AG"
    assert parsed["version"] == b"10"
    assert parsed["declared_size"] == parsed["raw_size"]
    assert parsed["waveform_count"] == 1
    # Waveform Header 必须声明 Normal、点数、秒、伏特、XIncrement 和 XOrigin。
    assert waveform_fields[0] == _WAVE_HEADER.size
    assert waveform_fields[1] == 1
    assert waveform_fields[2] == 1
    assert waveform_fields[3] == source_values.size
    assert waveform_fields[7] == pytest.approx(1.0 / 80.0e9)
    assert waveform_fields[8] == pytest.approx(-125.0e-12)
    assert waveform_fields[9:11] == (2, 1)
    assert waveform_fields[14].rstrip(b"\0") == b"Compensated"
    # 唯一 buffer 必须是每点四字节的 normal float，并恰好结束于 EOF。
    assert parsed["buffer_type"] == 1
    assert parsed["bytes_per_point"] == 4
    assert parsed["buffer_size"] == source_values.size * 4
    assert parsed["payload_end"] == parsed["raw_size"]
    # 电压只发生规范要求的 float32 量化，不得出现额外缩放或偏置。
    np.testing.assert_array_equal(parsed["values"], source_values.astype(np.float32))
