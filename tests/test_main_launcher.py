"""项目根目录启动器的跨平台行为测试。"""

from __future__ import annotations

import builtins
import os
import sys

import pytest

import main as launcher
import response_lab.app as app_module


def test_project_venv_python_uses_windows_scripts_directory() -> None:
    r"""Windows 自动切换应定位 ``.venv\Scripts\python.exe``。"""

    assert launcher._project_venv_python("win32") == (
        launcher.VENV_ROOT / "Scripts" / "python.exe"
    )


def test_dependency_instructions_use_windows_commands() -> None:
    """Windows 缺依赖提示不能要求用户执行 Unix ``.venv/bin`` 路径。"""

    instructions = launcher._dependency_install_instructions("win32")

    assert "py -3 -m venv .venv" in instructions
    assert "py -3.11" not in instructions
    assert '.venv\\Scripts\\python.exe -m pip install -e ".[dev]"' in instructions
    assert ".venv\\Scripts\\python.exe main.py" in instructions
    assert "  python main.py" not in instructions
    assert ".venv/bin/python" not in instructions


def test_reexec_uses_existing_windows_venv_interpreter(tmp_path, monkeypatch) -> None:
    """真实重启参数必须指向 Windows venv，并保留用户原始参数。"""

    venv_root = tmp_path / ".venv"
    venv_python = venv_root / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    observed: dict[str, object] = {}

    def record_execve(path: str, arguments: list[str], environment: dict[str, str]) -> None:
        observed.update(path=path, arguments=arguments, environment=environment)

    monkeypatch.setattr(launcher, "VENV_ROOT", venv_root)
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher, "_inside_project_venv", lambda: False)
    monkeypatch.setattr(launcher.os, "execve", record_execve)
    monkeypatch.setattr(launcher.sys, "argv", ["main.py", "--self-test"])
    monkeypatch.delenv("RESPONSELAB_VENV_REEXEC", raising=False)

    launcher._reexec_into_project_venv()

    assert observed["path"] == str(venv_python)
    assert observed["arguments"] == [
        str(venv_python),
        str(launcher.PROJECT_ROOT / "main.py"),
        "--self-test",
    ]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["RESPONSELAB_VENV_REEXEC"] == "1"
    # 启动器只在子进程环境里设置防循环标记，不污染父进程。
    assert "RESPONSELAB_VENV_REEXEC" not in os.environ


def test_windows_build_accepts_newer_x64_python_and_rejects_arm64() -> None:
    """批处理应接受 3.11+，同时把 x64 与其他 64 位架构明确区分。"""

    script = (launcher.PROJECT_ROOT / "build_window.bat").read_text(encoding="utf-8")

    assert 'set "BOOTSTRAP_PYTHON=py -3"' in script
    assert 'set "BOOTSTRAP_PYTHON=python"' in script
    assert "py -3.11" not in script
    assert "sys.version_info >= (3, 11)" in script
    assert script.count("struct.calcsize('P') == 8") >= 2
    assert script.count("sysconfig.get_platform().lower() == 'win-amd64'") >= 2
    existing_venv_guard = 'if exist ".venv\\Scripts\\python.exe" goto :validate_venv'
    assert existing_venv_guard in script
    assert script.index(existing_venv_guard) < script.index("where python")
    assert script.index('set "BOOTSTRAP_PYTHON=python"') < script.index(
        'set "BOOTSTRAP_PYTHON=py -3"'
    )
    venv_assignment = script.index('set "PYTHON=%CD%\\.venv\\Scripts\\python.exe"')
    assert script.index("struct.calcsize('P') == 8", venv_assignment) > venv_assignment


def test_windows_ci_exercises_bootstrap_with_the_selected_interpreter() -> None:
    """CI 应从空目录运行 batch，并由 PATH 中的矩阵解释器创建项目 venv。"""

    workflow = (
        launcher.PROJECT_ROOT / ".github" / "workflows" / "responselab-windows.yml"
    ).read_text(encoding="utf-8")

    assert 'python-version: ["3.11", "3.13"]' in workflow
    assert "run: python -m venv .venv" not in workflow
    assert "run: build_window.bat" in workflow
    assert "codex_projects/frequency_response_compensator" not in workflow
    assert 'architecture: "x86"' in workflow
    assert "must use Windows x64 Python" in workflow
    rejection_message = workflow.index("must use Windows x64 Python")
    assert workflow.index("exit 0", rejection_message) > rejection_message


def test_windows_build_runs_the_packaged_exe_before_declaring_success() -> None:
    """公司分发前必须执行打包后的入口，不能只检查 EXE 文件存在。"""

    script = (launcher.PROJECT_ROOT / "build_window.bat").read_text(encoding="utf-8")

    exe_guard = 'if not exist "dist\\ResponseLab\\ResponseLab.exe"'
    packaged_self_test = '"dist\\ResponseLab\\ResponseLab.exe" --self-test'
    packaged_gui_smoke = '"dist\\ResponseLab\\ResponseLab.exe" --gui-smoke-test'
    success_message = "SUCCESS: dist\\ResponseLab\\ResponseLab.exe"

    assert exe_guard in script
    assert packaged_self_test in script
    assert packaged_gui_smoke in script
    assert script.index(exe_guard) < script.index(packaged_self_test)
    assert script.index(packaged_self_test) < script.index(packaged_gui_smoke)
    assert script.index(packaged_gui_smoke) < script.index(success_message)
    self_test_failure_guard = script.index(
        "if errorlevel 1 goto :fail",
        script.index(packaged_self_test),
    )
    gui_smoke_failure_guard = script.index(
        "if errorlevel 1 goto :fail",
        script.index(packaged_gui_smoke),
    )
    assert self_test_failure_guard < script.index(packaged_gui_smoke)
    assert gui_smoke_failure_guard < script.index(success_message)


def test_windows_handoff_targets_the_standalone_repository() -> None:
    """拆仓后的 Windows 手册必须检出并在独立仓根目录构建。"""

    handoff = (launcher.PROJECT_ROOT / "docs" / "WINDOWS_EXE_BUILD_HANDOFF.md").read_text(
        encoding="utf-8"
    )
    readme = (launcher.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://github.com/ZhenlongYou/response-lab.git" in handoff
    assert "cd response-lab" in handoff
    assert "ZhenlongYou/codex" not in handoff
    assert "codex\\codex_projects\\frequency_response_compensator" not in handoff
    assert "<仓库目录>\\codex_projects\\frequency_response_compensator" not in readme


def test_delayed_gui_dependency_error_uses_windows_instructions(
    monkeypatch,
    capsys,
) -> None:
    """PySide6 延迟导入失败必须返回可操作的 Windows 提示，而不是泄漏 traceback。"""

    error_type = getattr(app_module, "GuiDependencyError", RuntimeError)

    def fail_gui(_arguments):
        raise error_type("无法加载 PySide6 GUI 依赖")

    monkeypatch.setattr(app_module, "main", fail_gui)
    monkeypatch.setattr(launcher, "_reexec_into_project_venv", lambda: None)
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.sys, "argv", ["main.py"])

    result = launcher.main()
    captured = capsys.readouterr()

    assert result == 2
    assert "无法加载 PySide6 GUI 依赖" in captured.err
    assert "py -3 -m venv .venv" in captured.err
    assert ".venv\\Scripts\\python.exe" in captured.err
    assert ".venv/bin/python" not in captured.err


def test_qt_binary_load_failure_becomes_a_typed_dependency_error(monkeypatch) -> None:
    """Windows DLL/Qt 二进制加载错误也应进入启动器可处理的依赖错误类型。"""

    real_import = builtins.__import__

    def fail_pyside_gui_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PySide6.QtGui":
            raise OSError("simulated Qt DLL load failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_pyside_gui_import)

    with pytest.raises(app_module.GuiDependencyError, match="PySide6 GUI"):
        app_module._qt_application()


@pytest.mark.parametrize(
    ("arguments", "blocked_import"),
    (
        ([], "PySide6.QtGui"),
        (["--gui-smoke-test"], "PySide6.QtTest"),
    ),
)
def test_real_gui_routes_wrap_dependency_failure_before_traceback(
    monkeypatch,
    capsys,
    arguments: list[str],
    blocked_import: str,
) -> None:
    """正常窗口和 smoke 路由最早的 GUI import 都必须转换为可处理错误。"""

    class BlockingFinder:
        def find_spec(self, fullname, _path, _target=None):
            if fullname == blocked_import:
                raise OSError(f"simulated failure importing {fullname}")
            if not arguments and fullname == "response_lab.ui":
                raise AssertionError("local UI imported before GUI dependency preflight")
            return None

    monkeypatch.delitem(sys.modules, blocked_import, raising=False)
    if not arguments:
        monkeypatch.delitem(sys.modules, "response_lab.ui", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockingFinder(), *sys.meta_path])
    monkeypatch.setattr(launcher, "_reexec_into_project_venv", lambda: None)
    monkeypatch.setattr(launcher.sys, "argv", ["main.py", *arguments])

    result = launcher.main()
    captured = capsys.readouterr()

    assert result == 2
    assert "GUI" in captured.err
    assert "依赖" in captured.err
    assert "Traceback" not in captured.err


def test_local_ui_import_error_is_not_mislabeled_as_a_dependency(
    monkeypatch,
) -> None:
    """依赖预检通过后的本地模块错误必须保留异常链，不能提示用户盲目重装。"""

    class BlockingFinder:
        def find_spec(self, fullname, _path, _target=None):
            if fullname == "response_lab.ui":
                raise OSError("simulated local UI module failure")
            return None

    monkeypatch.delitem(sys.modules, "response_lab.ui", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockingFinder(), *sys.meta_path])
    monkeypatch.setattr(app_module, "_qt_application", lambda: object())
    monkeypatch.setattr(launcher, "_reexec_into_project_venv", lambda: None)
    monkeypatch.setattr(launcher.sys, "argv", ["main.py"])

    with pytest.raises(OSError, match="local UI module failure"):
        launcher.main()


def test_initial_binary_dependency_import_failure_is_actionable(
    monkeypatch,
    capsys,
) -> None:
    """NumPy/SciPy 等初始二进制导入失败也不能绕过平台化安装提示。"""

    real_import_module = launcher.importlib.import_module

    def fail_scipy_binary_import(name: str, package: str | None = None):
        if name == "scipy.fft":
            raise OSError("simulated scipy binary dependency failure")
        return real_import_module(name, package)

    monkeypatch.setattr(launcher.importlib, "import_module", fail_scipy_binary_import)
    monkeypatch.setattr(launcher, "_reexec_into_project_venv", lambda: None)
    monkeypatch.setattr(launcher.sys, "argv", ["main.py"])

    result = launcher.main()
    captured = capsys.readouterr()

    assert result == 2
    assert "Python 依赖" in captured.err
    assert "Traceback" not in captured.err


def test_initial_local_app_import_error_is_not_mislabeled_as_a_dependency(
    monkeypatch,
) -> None:
    """数值依赖通过后，本地 app 导入错误应保留真实异常链。"""

    class BlockingFinder:
        def find_spec(self, fullname, _path, _target=None):
            if fullname == "response_lab.app":
                raise OSError("simulated local app module failure")
            return None

    monkeypatch.delitem(sys.modules, "response_lab.app", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockingFinder(), *sys.meta_path])
    monkeypatch.setattr(launcher, "_reexec_into_project_venv", lambda: None)
    monkeypatch.setattr(launcher.sys, "argv", ["main.py"])

    with pytest.raises(OSError, match="local app module failure"):
        launcher.main()
