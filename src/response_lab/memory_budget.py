"""Cross-platform memory budget shared by loaders and numerical workflows."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
from dataclasses import dataclass

# 用户确认允许更大的工作区；可用内存比例和系统余量仍会在运行时继续收紧该上限。
MAX_WORKSPACE_BYTES = 8 * 1024**3
AVAILABLE_MEMORY_FRACTION = 0.5
REQUIRED_HEADROOM_BYTES = 512 * 1024**2
FALLBACK_BUDGET_BYTES = 768 * 1024**2


@dataclass(frozen=True)
class MemoryBudget:
    """One snapshot of system availability and the derived safe allocation budget."""

    available_bytes: int | None
    budget_bytes: int


def parse_macos_vm_stat(output: str) -> int | None:
    """Parse immediately reclaimable pages from ``vm_stat`` without double counting."""

    page_size_match = re.search(r"page size of\s+(\d+)\s+bytes", output)
    if page_size_match is None:
        return None
    page_size = int(page_size_match.group(1))
    pages_by_label: dict[str, int] = {}
    for label, value in re.findall(
        r"Pages (free|inactive|speculative):\s+(\d+)\.",
        output,
    ):
        pages_by_label[label] = int(value)
    if not pages_by_label:
        return None
    # purgeable pages are generally a subset of inactive pages, so do not add them twice.
    return page_size * sum(pages_by_label.values())


def system_available_memory_bytes() -> int | None:
    """Return currently available physical memory using OS-native read-only interfaces."""

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    elif sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            available = parse_macos_vm_stat(completed.stdout)
            if available is not None:
                return available
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    elif sys.platform == "win32":
        try:
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError, ValueError):
            pass

    try:
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if available_pages > 0 and page_size > 0:
            return available_pages * page_size
    except (AttributeError, OSError, ValueError):
        return None
    return None


def safe_memory_budget_bytes(available_memory_bytes: int | None) -> int:
    """Apply the shared absolute cap, availability fraction, and OS headroom rules."""

    if available_memory_bytes is None:
        return FALLBACK_BUDGET_BYTES
    available = max(int(available_memory_bytes), 0)
    return min(
        MAX_WORKSPACE_BYTES,
        int(available * AVAILABLE_MEMORY_FRACTION),
        max(available - REQUIRED_HEADROOM_BYTES, 0),
    )


def current_memory_budget() -> MemoryBudget:
    """Capture availability once so a check reports the exact budget it used."""

    available = system_available_memory_bytes()
    return MemoryBudget(
        available_bytes=available,
        budget_bytes=safe_memory_budget_bytes(available),
    )


__all__ = [
    "AVAILABLE_MEMORY_FRACTION",
    "FALLBACK_BUDGET_BYTES",
    "MAX_WORKSPACE_BYTES",
    "MemoryBudget",
    "REQUIRED_HEADROOM_BYTES",
    "current_memory_budget",
    "parse_macos_vm_stat",
    "safe_memory_budget_bytes",
    "system_available_memory_bytes",
]
