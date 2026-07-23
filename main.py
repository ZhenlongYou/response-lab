"""ResponseLab 的 PyCharm 与终端统一入口。

这个文件只负责环境引导和入口转发。真正的文件解析、补偿算法与界面代码均位于
``src/response_lab``，因此从 PyCharm 点击 Run 与在终端执行 ``python3 main.py``
会走完全相同的代码路径。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# === 用户可直接编辑的启动参数 ===
# 保持 True 时，系统 Python 或 PyCharm 解释器会自动切换到项目自己的 .venv。
AUTO_USE_PROJECT_VENV = True
# 可以在这里加入固定启动参数，例如 ["--self-test"]；正常使用请保持为空。
DEFAULT_ARGS: list[str] = []


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
VENV_ROOT = PROJECT_ROOT / ".venv"


def _inside_project_venv() -> bool:
    """判断当前解释器的环境根目录是否就是项目 ``.venv``。

    这里比较 ``sys.prefix``，不比较 Python 可执行文件路径；macOS venv 经常使用
    Framework Python 符号链接，比较可执行文件会产生假阴性。
    """

    return Path(sys.prefix).resolve() == VENV_ROOT.resolve()


def _project_venv_python(platform_name: str | None = None) -> Path:
    """Return the local venv interpreter path for Unix-like systems or Windows."""

    effective_platform = sys.platform if platform_name is None else platform_name
    if effective_platform == "win32":
        return VENV_ROOT / "Scripts" / "python.exe"
    return VENV_ROOT / "bin" / "python"


def _dependency_install_instructions(platform_name: str | None = None) -> str:
    """Return copyable setup commands matching the current operating system."""

    effective_platform = sys.platform if platform_name is None else platform_name
    if effective_platform == "win32":
        return (
            "  py -3 -m venv .venv\n"
            '  .venv\\Scripts\\python.exe -m pip install -e ".[dev]"\n'
            "  .venv\\Scripts\\python.exe main.py"
        )
    return (
        "  python3 -m venv .venv\n"
        "  .venv/bin/python -m pip install -e '.[dev]'\n"
        "  python3 main.py"
    )


def _reexec_into_project_venv() -> None:
    """必要时用项目 venv 重新启动当前命令，不修改系统或 Conda 环境。"""

    if not AUTO_USE_PROJECT_VENV or _inside_project_venv():
        return
    venv_python = _project_venv_python()
    if not venv_python.exists():
        return
    if os.environ.get("RESPONSELAB_VENV_REEXEC") == "1":
        raise RuntimeError("项目虚拟环境重复重启失败，请删除 .venv 后重新安装")
    environment = os.environ.copy()
    environment["RESPONSELAB_VENV_REEXEC"] = "1"
    arguments = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execve(str(venv_python), arguments, environment)


def _preflight_core_dependencies() -> None:
    """Load third-party numerical modules at a narrow, user-actionable boundary."""

    # Import the submodules used by the application, not only top-level scipy: native extension
    # failures can otherwise surface later while importing local ResponseLab modules.
    for module_name in ("numpy", "scipy.fft", "scipy.interpolate", "scipy.signal"):
        importlib.import_module(module_name)


def main() -> int:
    """准备导入路径并进入应用入口。"""

    _reexec_into_project_venv()
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    try:
        _preflight_core_dependencies()
    except (ImportError, OSError) as exc:
        missing = getattr(exc, "name", None) or "Python 依赖"
        print(
            f"ResponseLab 缺少 {missing}。首次使用请在项目目录依次执行：\n"
            f"{_dependency_install_instructions()}\n"
            "安装完成后重新运行最后一条命令。",
            file=sys.stderr,
        )
        return 2

    # Dependency preflight has passed. Import local application code outside the catch so an
    # internal module regression retains its original exception and traceback.
    from response_lab.app import GuiDependencyError
    from response_lab.app import main as application_main

    try:
        return application_main([*DEFAULT_ARGS, *sys.argv[1:]])
    except GuiDependencyError as exc:
        print(
            f"ResponseLab {exc}。请在项目目录依次执行：\n"
            f"{_dependency_install_instructions()}\n"
            "安装完成后重新运行最后一条命令。",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
