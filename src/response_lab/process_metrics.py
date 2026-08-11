"""Cross-platform process peak-memory measurements used by validation tools."""

from __future__ import annotations

import ctypes
import sys


def _windows_peak_working_set_bytes() -> int:
    """Return this process' peak resident working set through the Windows API."""

    class ProcessMemoryCounters(ctypes.Structure):
        """Match the public ``PROCESS_MEMORY_COUNTERS`` layout from Psapi.h."""

        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        load_dll = ctypes.WinDLL
    except AttributeError as error:  # pragma: no cover - defensive non-Windows misuse
        raise RuntimeError("Windows 峰值 RSS 采样只能在 Windows 上执行") from error

    kernel32 = load_dll("kernel32", use_last_error=True)
    psapi = load_dll("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_uint32,
    ]
    get_process_memory_info.restype = ctypes.c_int

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    succeeded = get_process_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        error_code = ctypes.get_last_error()
        raise OSError(error_code, ctypes.FormatError(error_code))
    peak_bytes = int(counters.PeakWorkingSetSize)
    if peak_bytes <= 0:
        raise RuntimeError("Windows GetProcessMemoryInfo 返回了无效的峰值工作集")
    return peak_bytes


def _resource_peak_rss_bytes() -> int:
    """Normalize POSIX ``ru_maxrss`` units to bytes."""

    try:
        import resource
    except ModuleNotFoundError as error:  # pragma: no cover - unsupported platform
        raise RuntimeError("当前平台不支持进程峰值 RSS 采样") from error

    raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_bytes = raw_peak if sys.platform == "darwin" else raw_peak * 1024
    if peak_bytes <= 0:
        raise RuntimeError("resource.getrusage 返回了无效的峰值 RSS")
    return peak_bytes


def peak_rss_bytes() -> int:
    """Return this process' high-water resident memory in bytes."""

    if sys.platform == "win32":
        return _windows_peak_working_set_bytes()
    return _resource_peak_rss_bytes()


def peak_rss_backend() -> str:
    """Describe the operating-system counter used by :func:`peak_rss_bytes`."""

    if sys.platform == "win32":
        return "windows-peak-working-set"
    return "posix-ru-maxrss"
