# ResponseLab Handoff

## 当前任务

- task_id: `responselab-html-manual-upload-20260831`
- 目标：把已验收的 ResponseLab 中文 HTML 使用说明书、六页签截图和验收记录上传到 GitHub。
- 权威仓库：`/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator`；GitHub `ZhenlongYou/response-lab`。
- 交付路径：`docs/user_manual/`。
- Windows 受控试用预发布仍为 `v0.1.0-rc.1`，终端人工验收状态仍为 `manual_review`。

## 已经完成

- 已按当前界面生成并验收中文 HTML 使用说明书，覆盖六个页签、全部按钮、设置、输入、导出、错误处理和结果边界。
- 说明书、六张截图、验收记录及下载说明已经整理到 `docs/user_manual/`；说明书基于仓库快照 `a78e1ecfed5f3643f9e5ff5032cc282567716d2d`。
- 上一轮的 `100 ppm` 跨脉冲采样率兼容、Windows UTF-8/文件描述符修复均保留；仓库现已公开，GitHub Windows runner 可以正常启动。
- 发现并修复打包门禁缺口：`build_window.bat` 生成 EXE 后必须实际运行 `ResponseLab.exe --self-test` 和 `ResponseLab.exe --gui-smoke-test`，不再只检查文件存在。
- 发现并修复真实大 BIN 导出误拒：有限时间原点叠加 `5 ps` 间隔时，导出改为由完整记录跨度恢复标称间隔，并继续逐块拒绝超过 `1 ppm` 合同的局部时间抖动。
- 附件 `/Users/mac/Downloads/lxi_temp_1787818700601_807/826_fig1-1.bin`：SHA-256 `4551d8b8e3906f4b0822d513a930df66d9f89d45bc691687bcb54383f1adb113`，40,000,164 B，10,000,000 点，200 GSa/s，AG10 单通道 float32。
- 真实 offscreen GUI 按钮工作流完成：约 6.7 s，1052 次事件循环心跳，无错误，自动选择有限边界分块，结果页恢复为“预览有效”；进程最大 RSS 约 639 MiB。
- 同一附件完成补偿、BIN 导出和重新导入：导出约 0.38 s；输出 10,000,000 点、200 GSa/s、原始时间原点，10001 点抽查误差为 0，全部有限。
- 发布前的新鲜本地证据：`516 passed, 1 warning in 84.85s`；Ruff、compileall、`git diff --check`、`pip check`、`.venv`/系统入口 self-test 与 offscreen GUI smoke test 均通过。warning 为故意模拟 staging 路径身份变化的安全回归。
- 同一真实 1000 万点 AG10 再次完成身份补偿、BIN 导出与重读：自动选择 `streaming`，补偿约 0.64 s、导出约 0.06 s、峰值 RSS 约 769 MiB；10001 点身份误差不超过 `3.73e-8 V`，导出重读抽查误差为 0。
- `main` 已快进到 `77cc78735c95e56d49ac25f77383d698ce6e3c84`；GitHub Windows run `33345275464` 的 Python 3.11/3.13 x64 均为 `509 passed, 7 skipped`，源码与打包后 EXE self-test、GUI smoke test 及 x86 拒绝任务全部通过。
- GitHub 预发布 `v0.1.0-rc.1` 已发布并回下载验证。ZIP SHA-256 为 `aa936da027318f4623e6456f205f30ec9a2e41aa64f8a2c6225eaa5fd970a840`，包含 2244 个文件；内部 `ResponseLab.exe` SHA-256 为 `df88cdbbab710a0a02bf7fba24f988816243a277b71ccfa2752cdac644924419`。
- 按当前任务规则执行了单代理五轴审查，没有发现 Critical 或 Required 问题；用户未明确要求独立 reviewer，因此独立 reviewer 为 `NOT_RUN`。

## 当前状态或阻塞

- 中文 HTML 说明书、六张截图、下载说明和验收记录已经完成本地验收，随本次文档提交进入 `project/response-lab` 与 `main`。
- 本次文档交付的完整回归为 `516 passed, 1 warning`，GUI 冒烟测试和 HTML 结构/资产检查通过；warning 为既有的 staging 路径身份变化安全回归。
- Windows 干净办公电脑上的可见窗口启动、真实人工导入导出、断网、SmartScreen、杀毒软件和公司白名单复测仍为 `NOT_RUN`；GitHub offscreen runner 不能替代这一项。
- 当前没有公司代码签名，因此只作为 `v0.1.0-rc.1` 受控试用预发布，不升级为正式稳定版。

## 下一步计划

1. 若 ResponseLab 后续新增或改名控件，按最新 `main` 重新生成 `docs/user_manual/` 的界面截图并更新说明书版本快照。
2. 从 GitHub `v0.1.0-rc.1` 下载 ZIP，在一台干净 Windows 10/11 x64 办公电脑按 `docs/公司内部分发验收记录_2026-08-31.md` 做人工验收。
3. 人工检查全部通过后，把测试有效性记录从 `manual_review` 更新为正式内部发布 `PASS`，再决定是否发布 `v0.1.0` 稳定版。
4. 如需广泛分发，完成公司代码签名或应用白名单流程，并对签名后发生变化的最终字节重新做入口和哈希验收。

## 不要再踩的坑

- 不得把“PyInstaller 成功且 EXE 存在”当作打包入口可运行；必须执行打包后的 self-test 和 GUI smoke test。
- 不得用第一对 float64 时间戳直接决定 AG10 `XIncrement`；大时间原点会把舍入误差写成错误采样率，并可能误拒真实等间隔数据。
- GitHub runner 绿色不等于干净办公电脑人工验收，也不代表兼容所有 Keysight BIN；当前合同只覆盖文档声明的 Infiniium AG10 子集。
- 不得只复制 `ResponseLab.exe`；Qt、SciPy 和插件依赖要求交付完整 `ResponseLab` onedir 文件夹。
- 不得把公开的 `v0.1.0-rc.1` 预发布误称为已完成公司终端验收的正式稳定版。
- 不得只下载 `docs/user_manual/` 中的 HTML；必须保留同目录下的截图资产目录。
- 建议继续使用 `software-verification`、`test-effectiveness-gate`、`python-gui-venv`、`github-code-handoff` 和 `delivery-acceptance-gate`。
