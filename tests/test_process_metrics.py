"""跨平台进程峰值内存采样的回归测试。"""

from __future__ import annotations

import ctypes
import sys
from types import ModuleType, SimpleNamespace

import pytest

import response_lab.process_metrics as process_metrics


def test_peak_rss_uses_windows_peak_working_set(monkeypatch) -> None:
    """Windows 必须走原生 PeakWorkingSetSize，不能导入 POSIX resource。"""

    monkeypatch.setattr(process_metrics.sys, "platform", "win32")
    monkeypatch.setattr(
        process_metrics,
        "_windows_peak_working_set_bytes",
        lambda: 123_456_789,
    )

    assert process_metrics.peak_rss_bytes() == 123_456_789
    assert process_metrics.peak_rss_backend() == "windows-peak-working-set"


@pytest.mark.parametrize(
    ("platform_name", "raw_peak", "expected_bytes"),
    [
        ("darwin", 12_345_678, 12_345_678),
        ("linux", 12_345, 12_345 * 1024),
    ],
)
def test_posix_peak_rss_preserves_platform_units(
    monkeypatch,
    platform_name: str,
    raw_peak: int,
    expected_bytes: int,
) -> None:
    """macOS 的 ru_maxrss 是 byte，Linux 的同名字段是 KiB。"""

    fake_resource = ModuleType("resource")
    fake_resource.RUSAGE_SELF = object()
    fake_resource.getrusage = lambda _who: SimpleNamespace(ru_maxrss=raw_peak)
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(process_metrics.sys, "platform", platform_name)

    assert process_metrics.peak_rss_bytes() == expected_bytes
    assert process_metrics.peak_rss_backend() == "posix-ru-maxrss"


def test_windows_peak_rss_reads_peak_working_set_from_psapi(monkeypatch) -> None:
    """用假 DLL 验证 ctypes 签名最终读取的是 PeakWorkingSetSize。"""

    expected_peak_bytes = 987_654_321

    class FakeFunction:
        def __init__(self, callback):
            self._callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._callback(*args)

    def get_process_memory_info(handle, counters_pointer, structure_size):
        assert handle == 0x1234
        counters = counters_pointer._obj
        assert structure_size == ctypes.sizeof(counters)
        counters.PeakWorkingSetSize = expected_peak_bytes
        return 1

    kernel32 = SimpleNamespace(GetCurrentProcess=FakeFunction(lambda: 0x1234))
    psapi = SimpleNamespace(GetProcessMemoryInfo=FakeFunction(get_process_memory_info))

    def load_dll(name: str, *, use_last_error: bool):
        assert use_last_error
        return kernel32 if name == "kernel32" else psapi

    monkeypatch.setattr(ctypes, "WinDLL", load_dll, raising=False)

    assert process_metrics._windows_peak_working_set_bytes() == expected_peak_bytes
