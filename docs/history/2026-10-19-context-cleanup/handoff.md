# ResponseLab Handoff

## 当前任务

- task_id: `responselab-github-windows-artifact-verification-20260902`
- 目标：记录并交付精确项目分支 OID 的 GitHub Windows x64 onedir 构建产物与可追溯自动化验收证据。
- 权威仓库：`/Users/mac/PycharmProjects/RinysProject/codex_projects/frequency_response_compensator`；GitHub `ZhenlongYou/response-lab`。
- 交付分支：`project/response-lab`；改动范围为 `src/response_lab/`、`tests/`、`README.md`、`docs/` 和本文件。
- recorded_commit: `0ce0eb34fb6de9778b3c312995e0cf287dd7b608`
- status: `ready`
- 已公开版本仍为受控试用预发布 `v0.1.0-rc.1`；本轮精确 OID 候选尚未发布，终端人工验收状态仍为 `manual_review`。

## 已经完成

- 已从 `project/response-lab` 手动触发 GitHub Actions run `33536003559`；事件为 `workflow_dispatch`，`headSha` 精确等于 `0ce0eb34fb6de9778b3c312995e0cf287dd7b608`，避免把 PR 临时合并提交当成交付源码。
- Windows Python 3.11/3.13 x64 均为 `519 passed, 7 skipped`；源码与打包后 `ResponseLab.exe` 的 self-test、GUI smoke-test 全部 PASS，Windows x86 虚拟环境拒绝门禁也 PASS。
- 两个 GitHub onedir 产物均已下载回验；服务端上传摘要与本地下载 ZIP 的 SHA-256 完全一致，压缩完整性通过，内部 EXE 均识别为 `PE32+ GUI x86-64`。
- Python 3.11 被选为唯一内部候选：`/Users/mac/Documents/频响补偿工具/deliverables/ResponseLab-windows-x64-py3.11-0ce0eb34fb6de9778b3c312995e0cf287dd7b608.zip`，119469972 bytes，SHA-256 `c814afee3e496d443e5917f56251c5fc871efb0bc3b0bc0b794a96e5960efba4`，包含 2244 个文件；内部 EXE SHA-256 为 `a2df60f7df61bc69384e0af1a89fc8263ae744e5cbda10baeb81860a7c6934a9`。
- Python 3.13 兼容性产物包含 2249 个文件，下载 ZIP SHA-256 `cf6e7ada668b22775401d4e2a1a20e3f3d771e669b45626c34f13d5465ce385a`；验证后未作为第二候选保留，避免交付歧义。
- 首次由 PR 事件触发的 run `33535400132` 构建了 GitHub 合成 merge OID `be7f68fee44e742027b08596b17eabca32e226e0`。虽然其 tree 与分支 OID 完全相同，仍不作为最终候选证据；后续必须优先使用精确分支 `workflow_dispatch`。
- 已把 `docs/WINDOWS_EXE_BUILD_HANDOFF.md` 升级为 Windows 打包 Agent 执行合同；它明确 `.md` 负责步骤/证据/停止条件，根目录 `build_window.bat` 是唯一发布构建入口。
- 合同新增精确 commit、干净检出、不可跳过门禁、同一字节验收、完整 onedir ZIP、哈希与依赖记录，并禁止用“EXE 存在”或 CI 绿色替代真实工作流。
- 合同已纳入本轮 `1000 ppm` 闭区间、眼图有限边界、最大增益与边缘过渡回归；列出的重点测试经 `pytest --collect-only` 确认为 6 项。
- 干净机器矩阵改为使用独立哈希的验收输入，不把示例事后追加进候选 ZIP；同时覆盖中文/空格路径、CSV/AG10、BIN 导出重读、眼图/Vpp、断网和终端安全策略。
- 合同编写阶段的本地核对为 `20 passed in 2.91s`、self-test PASS、Ruff PASS 和 `git diff --check` PASS；本轮 Windows x64 实际结果以上述精确 OID GitHub run 为准。
- 影响频段眼高/眼宽有限记录不再固定镜像延拓，已与主补偿的 `boundary_mode` 一致；默认记录外补零，周期 Vpp 仍保持周期边界。
- 幅度与幅相候选现执行右栏最大补偿增益；右栏增益开关、增益值和边缘过渡变化会使旧影响结果失效。即使主补偿选择“仅相位”，影响页仍比较三支并保留增益上限。
- 影响频段余弦肩宽已由固定 50% 改为复用右栏“边缘过渡”比例，默认每侧为满权核心的 10%。
- 两份拟合脉冲的公共兼容门限已由 `100 ppm` 放宽到 `1000 ppm`（`0.1%`）：500 ppm 和数学上恰好 1000 ppm 接受，2000 ppm 拒绝；门限内保留两条原始时间轴，不重采样，并在影响结果警告中显示实际 ppm。
- 新增同探针 RED/GREEN、独立闭式谕示、六类输入分区、定向突变和权威 escaped-defect ledger；两项审计缺陷均通过真实“开始分析”按钮路径复测。绑定代码提交 `52e3aa6aa78db3834b3e7ddfebad2c67421a5fe3` 的 executable v2 门禁返回 `EXECUTED_EVIDENCE_PASS`，范围限定为本轮影响频段合同。
- 本轮最后一次产品代码修改后的新鲜本地证据为 `526 passed, 1 warning in 85.05s`、Ruff PASS、self-test PASS、GUI smoke-test PASS 和 `git diff --check` PASS；warning 是既有 staging 身份保护测试。
- 已按当前界面生成并验收中文 HTML 使用说明书，覆盖六个页签、全部按钮、设置、输入、导出、错误处理和结果边界。
- 说明书、六张截图、验收记录及下载说明已经整理到 `docs/user_manual/`；说明书基于仓库快照 `a78e1ecfed5f3643f9e5ff5032cc282567716d2d`。
- 上一轮的 Windows UTF-8/文件描述符修复均保留；原 `100 ppm` 合同已由本任务的 `1000 ppm` 合同取代。仓库现已公开，GitHub Windows runner 可以正常启动。
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

- Agent 打包合同内容提交为 `3a26d86b7373284dbb76098104da7d53655c4011`；`project/response-lab` 是本任务约定交付分支。
- GitHub Windows x64 自动构建、完整回归、打包后 EXE self-test/GUI smoke-test、产物上传与下载摘要回验均为 `PASS`，绑定 run `33536003559` 和源码 OID `0ce0eb34fb6de9778b3c312995e0cf287dd7b608`。
- 干净 Windows 办公电脑上的可见窗口、真实人工导入导出、断网、SmartScreen、杀毒软件和公司白名单仍为 `NOT_RUN`；GitHub runner 不能替代这一项。
- 用户要求的 `1000 ppm` 是相对参考脉冲采样率的闭区间门限；超过门限仍明确拒绝，工具不会通过静默重采样掩盖物理网格差异。
- 当前 `project/response-lab` 包含本轮前置算法修复，`main` 尚未包含；独立 reviewer 未由用户请求，状态为 `NOT_RUN`。
- 当前候选没有公司代码签名且尚未发布；已公开的 `v0.1.0-rc.1` 继续保持受控试用预发布，不升级为正式稳定版。

## 下一步计划

1. 把上述 Python 3.11 候选 ZIP 原样复制到一台干净 Windows 10/11 x64 办公电脑，按 `docs/WINDOWS_EXE_BUILD_HANDOFF.md` 完成可见窗口、真实文件、断网和终端安全策略验收。
2. 若人工矩阵全部通过，再决定是否把本轮代码和合同并入 `main`、发布新的受控预发布；必须发布同一 SHA-256 字节，签名或重压缩后要重新验收。

## 不要再踩的坑

- `.bat` 不是完整交接文档，`.md` 也不能代替执行脚本；未来 Agent 必须同时遵守 `docs/WINDOWS_EXE_BUILD_HANDOFF.md` 和 `build_window.bat`。
- PR 事件中的 `github.sha` 是合成 merge OID，即使 tree 与分支相同也不能冒充指定源码提交；发布候选应从精确分支执行 `workflow_dispatch` 并核对 `headSha`。
- 没有完整 commit OID、干净检出、同一 ZIP 哈希和干净机器结果时，不得声称“完美打包”或“没有功能错误”。
- 示例输入是独立验收材料，不属于产品 onedir；不能在 ZIP 哈希生成后再把示例或 DLL 塞进包内。
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
