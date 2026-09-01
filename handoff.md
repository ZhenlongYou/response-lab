# ResponseLab Handoff

## 当前任务

- task_id: `responselab-influence-contracts-20260902`
- 目标：修复影响频段的有限记录边界和最大增益合同，把眼图/Vpp 跨脉冲采样率兼容门限按用户确认放宽到 `1000 ppm`。
- 权威仓库：`/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator`；GitHub `ZhenlongYou/response-lab`。
- 交付分支：`project/response-lab`；改动范围为 `src/response_lab/`、`tests/`、`README.md`、`docs/` 和本文件。
- Windows 受控试用预发布仍为 `v0.1.0-rc.1`，终端人工验收状态仍为 `manual_review`。

## 已经完成

- 影响频段眼高/眼宽有限记录不再固定镜像延拓，已与主补偿的 `boundary_mode` 一致；默认记录外补零，周期 Vpp 仍保持周期边界。
- 幅度与幅相候选现执行右栏最大补偿增益；右栏增益开关、增益值和边缘过渡变化会使旧影响结果失效。即使主补偿选择“仅相位”，影响页仍比较三支并保留增益上限。
- 影响频段余弦肩宽已由固定 50% 改为复用右栏“边缘过渡”比例，默认每侧为满权核心的 10%。
- 两份拟合脉冲的公共兼容门限已由 `100 ppm` 放宽到 `1000 ppm`（`0.1%`）：500 ppm 和数学上恰好 1000 ppm 接受，2000 ppm 拒绝；门限内保留两条原始时间轴，不重采样，并在影响结果警告中显示实际 ppm。
- 新增同探针 RED/GREEN、独立闭式谕示、六类输入分区、定向突变和权威 escaped-defect ledger；两项审计缺陷均通过真实“开始分析”按钮路径复测。
- 本轮最后一次产品代码修改后的新鲜本地证据为 `526 passed, 1 warning in 85.05s`、Ruff PASS、self-test PASS、GUI smoke-test PASS 和 `git diff --check` PASS；warning 是既有 staging 身份保护测试。
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

- 本地实现和针对性按钮回归已通过，正在生成绑定提交快照的 executable v2 测试有效性 receipt，并准备任务内提交和推送。
- 用户要求的 `1000 ppm` 是相对参考脉冲采样率的闭区间门限；超过门限仍明确拒绝，工具不会通过静默重采样掩盖物理网格差异。
- 中文 HTML 说明书、六张截图、下载说明和验收记录已经完成本地验收，随本次文档提交进入 `project/response-lab` 与 `main`。
- 本次文档交付的完整回归为 `516 passed, 1 warning`，GUI 冒烟测试和 HTML 结构/资产检查通过；warning 为既有的 staging 路径身份变化安全回归。
- GitHub `main` 与 `project/response-lab` 已通过 GitHub API 核对为同一交付提交；根据当前任务规则，独立 reviewer 未由用户请求，状态为 `NOT_RUN`，已完成非委派审查并保留该验收边界。
- Windows 干净办公电脑上的可见窗口启动、真实人工导入导出、断网、SmartScreen、杀毒软件和公司白名单复测仍为 `NOT_RUN`；GitHub offscreen runner 不能替代这一项。
- 当前没有公司代码签名，因此只作为 `v0.1.0-rc.1` 受控试用预发布，不升级为正式稳定版。

## 下一步计划

1. 在最终代码快照上刷新完整测试、Ruff、self-test、GUI smoke-test 和 executable v2 证据，再提交并推送 `project/response-lab`。
2. 若需要把本轮改动并入 `main` 或重新发布 Windows EXE，应另开集成/发布任务，并重新构建和验证最终 Windows 字节。
3. 从 GitHub `v0.1.0-rc.1` 下载 ZIP，在一台干净 Windows 10/11 x64 办公电脑按 `docs/公司内部分发验收记录_2026-08-31.md` 做人工验收。
4. 人工检查全部通过后，再决定是否发布 `v0.1.0` 稳定版；如需广泛分发，还要完成公司代码签名或应用白名单流程。

## 不要再踩的坑

- 不得把 `1000 ppm` 解释成自动重采样；它只放宽“继续分析”的兼容门限，并保留可见告警和原始采样率。
- 影响页固定比较幅度、相位和幅相三支，不能因为主补偿模式为“仅相位”而旁路最大增益。
- 有限记录眼图与周期 Vpp 的边界物理含义不同：前者使用显式 zero/reflect，后者必须保持周期边界。
- 不得把“PyInstaller 成功且 EXE 存在”当作打包入口可运行；必须执行打包后的 self-test 和 GUI smoke test。
- 不得用第一对 float64 时间戳直接决定 AG10 `XIncrement`；大时间原点会把舍入误差写成错误采样率，并可能误拒真实等间隔数据。
- GitHub runner 绿色不等于干净办公电脑人工验收，也不代表兼容所有 Keysight BIN；当前合同只覆盖文档声明的 Infiniium AG10 子集。
- 不得只复制 `ResponseLab.exe`；Qt、SciPy 和插件依赖要求交付完整 `ResponseLab` onedir 文件夹。
- 不得把公开的 `v0.1.0-rc.1` 预发布误称为已完成公司终端验收的正式稳定版。
- 不得只下载 `docs/user_manual/` 中的 HTML；必须保留同目录下的截图资产目录。
- 建议继续使用 `software-verification`、`test-effectiveness-gate`、`python-gui-venv`、`github-code-handoff` 和 `delivery-acceptance-gate`。
