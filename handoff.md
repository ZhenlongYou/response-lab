# ResponseLab Handoff

## 当前任务

- task_id: `responselab-company-distribution-20260831`
- 目标：完成公司内部分发前的缺陷审查、真实大文件工作流、Windows x64 EXE 和可追溯交付验证。
- 权威仓库：`/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator`；GitHub `ZhenlongYou/response-lab`。
- 写入分支：`project/response-lab`；合入目标：`main`。
- 续作起点：`37ea4c90a18058188d1e6387470f422eedfc9ec2`。
- status: `verification_in_progress`

## 已经完成

- 上一轮的 `100 ppm` 跨脉冲采样率兼容、Windows UTF-8/文件描述符修复均保留；仓库现已公开，GitHub Windows runner 可以正常启动。
- 发现并修复打包门禁缺口：`build_window.bat` 生成 EXE 后必须实际运行 `ResponseLab.exe --self-test` 和 `ResponseLab.exe --gui-smoke-test`，不再只检查文件存在。
- 发现并修复真实大 BIN 导出误拒：有限时间原点叠加 `5 ps` 间隔时，导出改为由完整记录跨度恢复标称间隔，并继续逐块拒绝超过 `1 ppm` 合同的局部时间抖动。
- 附件 `/Users/mac/Downloads/lxi_temp_1787818700601_807/826_fig1-1.bin`：SHA-256 `4551d8b8e3906f4b0822d513a930df66d9f89d45bc691687bcb54383f1adb113`，40,000,164 B，10,000,000 点，200 GSa/s，AG10 单通道 float32。
- 真实 offscreen GUI 按钮工作流完成：约 6.7 s，1052 次事件循环心跳，无错误，自动选择有限边界分块，结果页恢复为“预览有效”；进程最大 RSS 约 639 MiB。
- 同一附件完成补偿、BIN 导出和重新导入：导出约 0.38 s；输出 10,000,000 点、200 GSa/s、原始时间原点，10001 点抽查误差为 0，全部有限。
- 最终代码修改后的本地证据：`516 passed, 1 warning in 86.17s`；Ruff、compileall、`git diff --check`、`pip check`、`.venv`/系统入口 self-test 与 offscreen GUI smoke test 均通过。warning 为故意模拟 staging 路径身份变化的安全回归。
- GitHub Windows 基线 run `33324664446` 绑定续作起点；Python 3.13 x64 已通过并生成约 114 MiB artifact，x86 拒绝任务通过；Python 3.11 x64 仍在运行。该基线尚不包含本轮两项修复。

## 当前状态或阻塞

- 本轮代码尚未提交和推送，最终 Windows 3.11/3.13 打包后 EXE 自检尚未运行。
- Windows 干净办公电脑上的可见窗口启动、真实人工导入导出和断网复测仍为 `NOT_RUN`；GitHub offscreen runner 不能替代这一项。
- 多代理独立 reviewer 未获用户明确授权，本轮只能执行单代理五轴审查；按交付编排规则，在 reviewer 门禁缺失时不得合入 `main`。

## 下一步计划

1. 审查当前 diff，提交并推送 `project/response-lab`。
2. 触发该提交的 GitHub Windows 3.11/3.13 x64 构建，确认源码门禁、打包后 EXE 两个入口、x86 拒绝和两份 artifact 全部通过。
3. 下载 artifact，核对完整 onedir 内容、大小、提交身份和 SHA-256，写入公司分发验收记录。
4. 若用户授权独立 reviewer，完成 reviewer attestation、合入 `main` 和最终交付门禁；否则保留在持久项目分支并明确报告 main 未集成。
5. 在一台干净 Windows 10/11 x64 办公电脑按 `docs/WINDOWS_EXE_BUILD_HANDOFF.md` 第 4.4 节做人工验收后，再作为正式内部版本流通。

## 不要再踩的坑

- 不得把“PyInstaller 成功且 EXE 存在”当作打包入口可运行；必须执行打包后的 self-test 和 GUI smoke test。
- 不得用第一对 float64 时间戳直接决定 AG10 `XIncrement`；大时间原点会把舍入误差写成错误采样率，并可能误拒真实等间隔数据。
- GitHub runner 绿色不等于干净办公电脑人工验收，也不代表兼容所有 Keysight BIN；当前合同只覆盖文档声明的 Infiniium AG10 子集。
- 不得只复制 `ResponseLab.exe`；Qt、SciPy 和插件依赖要求交付完整 `ResponseLab` onedir 文件夹。
- 建议继续使用 `software-verification`、`test-effectiveness-gate`、`python-gui-venv`、`github-code-handoff` 和 `delivery-acceptance-gate`。
