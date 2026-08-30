# ResponseLab Handoff

## 当前任务

- task_id: `responselab-samplerate-windows-20260831`
- 目标：把影响频段的跨波形采样率兼容门限放宽到 `100 ppm`，审计同类严格比较，并让 GitHub Windows x64 工作流真实生成 onedir EXE。
- 权威仓库：`/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator`；GitHub `ZhenlongYou/response-lab`。
- 写入分支：`project/response-lab`；合入目标：`main`。
- 起点：`caccf5e241e3877c19487d6c81a59698419ad8ba`。
- status: `awaiting_windows_runner`

## 已完成

- 眼图/影响频段和 Vpp 窗口入口改用同一 `100 ppm` 跨脉冲采样率合同；`50 ppm` 通过，`200 ppm` 报告实际偏差并拒绝，不执行重采样。
- 审计其余严格比较：单文件时间轴均匀性、分析结果内部恒等式和候选并列判定不属于跨文件兼容门限，保持原值。
- 文档已同步说明“容忍导出舍入，不重采样，超限拒绝”。
- 修复 Windows UTF-8 读取、BIN 改写错误语义，以及取消发生在 `fdopen` 前时临时文件描述符泄漏。
- 7 个依赖 POSIX“重命名/删除仍打开路径”的攻击注入测试在 Windows 精确跳过；macOS 仍实际执行。Windows 无 `SHARE_DELETE` 的锁保护由正常导出、锁冲突和 Windows 分类测试继续覆盖。

## 当前证据

- 采样率 RED：旧实现的 4 个公共入口回归全部失败；GREEN：`4 passed`。
- 描述符 RED：取消导出后 `os.fstat(fd)` 仍成功；GREEN：描述符已关闭，聚焦测试通过。
- 完整 macOS 测试：`512 passed, 1 warning in 87.23s`；warning 为故意模拟 staging 身份变化的安全回归。
- `ruff check .`、`git diff --check`、`compileall` 均通过。
- `python3 main.py --self-test` 与 offscreen `--gui-smoke-test` 均 PASS。

## 下一步

1. 提交并推送 `project/response-lab`，等待 Windows Python 3.11/3.13 构建与 x86 拒绝任务。
2. 若 runner 失败，按真实日志继续修复；若通过，核对两份完整 onedir artifact 和 `ResponseLab.exe`。
3. 更新本文件，独立复核最终 OID，合入 `main` 并运行交付门禁。

## 未完成边界

- 当前 Windows runner 与 EXE artifact 仍为 `NOT_RUN`；本地 macOS 绿测不能替代它。
- Windows 干净机器人工启动和真实 Keysight AG10 文件回读不在 GitHub runner 能力内，最终仍需标记 `NOT_RUN` 或人工验收。
